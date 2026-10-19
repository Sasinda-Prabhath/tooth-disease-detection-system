"""Create clean tooth-context crops for expert six-point annotation; never guess labels."""
import argparse
import json
from pathlib import Path

from PIL import Image

from dental_fdi.analysis.bone_pose import BONE_SCHEMA, canonical_bone_crop
from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.data.dataset_checks import check_dataset
from dental_fdi.paths import DATASET_DIR, PROJECT_ROOT


def prepare(source, output):
    errors=check_dataset(source)
    if errors:
        raise ValueError('\n'.join(errors))
    if any(p.is_file() for directory in ('images','labels') for p in (output/directory).rglob('*')):
        raise ValueError('Output images already exist; choose a new output folder to preserve annotations')
    records=[]
    for split in ('train','val','test'):
        (output/'images'/split).mkdir(parents=True,exist_ok=True)
        (output/'labels'/split).mkdir(parents=True,exist_ok=True)
        for path in sorted((source/'images'/split).glob('*')):
            if path.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
                continue
            with Image.open(path) as original:
                if original.getexif().get(274,1)!=1:
                    raise ValueError('Normalize source EXIF orientation and annotations first')
                image=original.convert('RGB')
            label=source/'labels'/split/(path.stem+'.txt')
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                c,x,y,w,h=map(float,line.split());fdi=FDI_CODES[int(c)]
                box=[(x-w/2)*image.width,(y-h/2)*image.height,(x+w/2)*image.width,(y+h/2)*image.height]
                crop,bounds,mirrored=canonical_bone_crop(image,[max(0,box[0]),max(0,box[1]),min(image.width,box[2]),min(image.height,box[3])],fdi)
                name=path.stem+'_FDI'+fdi+'.png'
                if (output/'images'/split/name).exists():
                    raise ValueError(f'Duplicate crop filename: {name}')
                crop.save(output/'images'/split/name)
                records.append({'image':name,'source_image':path.name,'split':split,'fdi':fdi,
                                'original_size':list(image.size),'crop_bounds':bounds,'tooth_box':box,
                                'canonical_mirrored':mirrored})
    (output/'crop_manifest.json').write_text(json.dumps({'schema':BONE_SCHEMA,'records':records,
        'status':'Unannotated clean crops. Expert keypoints must be supplied before training.'},indent=2))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DATASET_DIR)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'periodontal_dataset')
    args=parser.parse_args()
    records=prepare(args.source,args.output)
    print(f'Created {len(records)} clean context crops in {args.output}. No landmark labels were generated. Annotate them with a dentist before training.')


if __name__=='__main__':
    main()
