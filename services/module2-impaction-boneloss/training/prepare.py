"""Create immutable patient splits BEFORE annotation; convert CVAT image polygons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from app.schemas import FDI

def split_patients(rows, seed=20261001):
    if not rows or any(not r.get('patient_id') or not r.get('image') for r in rows):
        raise ValueError('Every image requires a pseudonymous patient_id.')
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image entries are not allowed.')
    patients = sorted({r['patient_id'] for r in rows})
    if len(patients) < 3:
        raise ValueError('At least three patients are needed for disjoint partitions.')
    random.Random(seed).shuffle(patients)
    ntest, nval = max(1, round(len(patients)*.15)), max(1, round(len(patients)*.15))
    assignment = {p: 'test' if i < ntest else 'val' if i < ntest+nval else 'train' for i, p in enumerate(patients)}
    return [dict(row, split=assignment[row['patient_id']]) for row in rows]

def check_splits(rows):
    assignments, images = {}, set()
    for row in rows:
        patient, split = row['patient_id'], row['split']
        if not patient or split not in ('train', 'val', 'test') or assignments.setdefault(patient, split) != split:
            raise ValueError('Patient leakage or invalid partition.')
        if row['image'] in images:
            raise ValueError('Duplicate image across manifest.')
        images.add(row['image'])
    if set(assignments.values()) != {'train', 'val', 'test'}:
        raise ValueError('All three patient-separated partitions are required.')

def cvat_to_yolo(xml, manifest, image_root, output):
    import shutil
    from PIL import Image
    check_splits(manifest)
    lookup = {r['image']: r for r in manifest}
    output.mkdir(parents=True, exist_ok=False)
    seen, hashes = set(), {}
    for image in ET.parse(xml).getroot().findall('image'):
        name = image.attrib['name']
        if name not in lookup or name in seen:
            raise ValueError('CVAT image missing from manifest or duplicated.')
        seen.add(name)
        row = lookup[name]
        source = (image_root/name).resolve()
        if not source.is_relative_to(image_root.resolve()):
            raise ValueError('Image path escapes approved data directory.')
        with Image.open(source) as im:
            width, height = im.size
            digest = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate pixel data detected; deduplicate before splitting.')
        hashes[digest] = row['split']
        if (width, height) != (int(image.attrib['width']), int(image.attrib['height'])):
            raise ValueError('CVAT and processed image dimensions differ.')
        lines, fdis = [], set()
        for polygon in image.findall('polygon'):
            if polygon.attrib['label'] != 'tooth':
                continue
            attributes = {a.attrib['name']: a.text for a in polygon.findall('attribute')}
            fdi = attributes.get('fdi_number')
            if fdi not in FDI or fdi in fdis:
                raise ValueError('Invalid or duplicate FDI annotation.')
            fdis.add(fdi)
            points = [tuple(map(float, point.split(','))) for point in polygon.attrib['points'].split(';')]
            if len(points) < 3 or any(not (0 <= x < width and 0 <= y < height) for x, y in points):
                raise ValueError('Invalid tooth polygon.')
            lines.append(str(FDI.index(fdi))+' '+' '.join(f'{v:.8f}' for x, y in points for v in (x/width, y/height)))
        stem = hashlib.sha256(name.encode()).hexdigest()[:20]
        for kind in ('images', 'labels'):
            (output/kind/row['split']).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output/'images'/row['split']/(stem+source.suffix))
        (output/'labels'/row['split']/(stem+'.txt')).write_text('\n'.join(lines))
    if seen != set(lookup):
        raise ValueError('CVAT export does not cover the frozen manifest.')
    # JSON is a YAML subset, read directly by Ultralytics.
    (output/'dataset.yaml').write_text(json.dumps({'path': str(output.resolve()), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': dict(enumerate(FDI))}, indent=2))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))

def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    split = subs.add_parser('split'); split.add_argument('csv', type=Path); split.add_argument('output', type=Path)
    convert = subs.add_parser('cvat'); convert.add_argument('xml', type=Path); convert.add_argument('manifest', type=Path); convert.add_argument('images', type=Path); convert.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'split':
        rows = list(csv.DictReader(args.csv.open(encoding='utf-8-sig')))
        with args.output.open('x', encoding='utf-8') as file:
            json.dump(split_patients(rows), file, indent=2)
    else:
        cvat_to_yolo(args.xml, json.loads(args.manifest.read_text()), args.images, args.output)

if __name__ == '__main__':
    main()
