"""Use reviewed tooth polygons to make clean crops with the inference transform."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from app.schemas import Tooth
from app.services.impaction import crop_and_geometry, GEOMETRY_VERSION, ANGLES
from training.prepare import check_splits

def prepare(manifest, annotations, images, output):
    check_splits(manifest)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in manifest:
        record = annotations[row['image']]
        teeth = [Tooth(**t) for t in record['teeth']]
        source = (images/row['image']).resolve()
        if not source.is_relative_to(images.resolve()):
            raise ValueError('Image path escapes approved directory.')
        with Image.open(source) as image:
            rgb = np.array(image.convert('RGB'))
        for tooth in teeth:
            if tooth.fdi not in ('18', '28', '38', '48'):
                continue
            labels = record.get('labels', {}).get(tooth.fdi, {})
            impacted, angle = labels.get('impacted'), labels.get('angulation')
            if impacted is not None and type(impacted) is not bool:
                raise ValueError('Impacted label must be true, false or null.')
            if angle is not None and angle not in ANGLES:
                raise ValueError('Unknown angulation label.')
            crop, origin, geometry = crop_and_geometry(rgb, tooth, teeth)
            name = hashlib.sha256((row['image']+tooth.fdi).encode()).hexdigest()[:20]+'.png'
            Image.fromarray(crop).save(output/name)
            rows.append({**row, 'fdi': tooth.fdi, 'crop': name, 'origin': origin, 'geometry': geometry.tolist(),
                         'geometry_version': GEOMETRY_VERSION, 'impacted': impacted, 'angulation': angle if impacted else None})
    (output/'rows.json').write_text(json.dumps(rows, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('manifest', 'annotations', 'images', 'output'): parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    prepare(json.loads(args.manifest.read_text()), json.loads(args.annotations.read_text()), args.images, args.output)
