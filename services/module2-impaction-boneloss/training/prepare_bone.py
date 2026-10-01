"""Export reviewed dense ROI masks, with explicit patient-level nnU-Net split."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

def prepare(rows_path, masks, output):
    rows = json.loads(rows_path.read_text())
    assignments = {}
    output.mkdir(parents=True, exist_ok=False)
    ids = {'train': [], 'val': [], 'test': []}
    for index, row in enumerate(rows):
        patient, split = row['patient_id'], row['split']
        if split not in ids or assignments.setdefault(patient, split) != split:
            raise ValueError('Patient leakage or invalid split.')
        case = f'roi_{index:06d}'
        ids[split].append(case)
        image_path = (rows_path.parent/row['crop']).resolve()
        mask_path = (masks/row['crop']).resolve()
        if not image_path.is_relative_to(rows_path.parent.resolve()) or not mask_path.is_relative_to(masks.resolve()):
            raise ValueError('ROI path escapes dataset.')
        with Image.open(image_path) as image:
            import cv2
            gray = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
        with Image.open(mask_path) as image:
            mask = np.array(image)
        if mask.shape != gray.shape or not np.isin(mask, [0, 1, 2]).all():
            raise ValueError('Dense mask must match crop dimensions and use 0/1/2 classes.')
        suffix = 'Ts' if split == 'test' else 'Tr'
        for name in ('images'+suffix, 'labels'+suffix): (output/name).mkdir(exist_ok=True)
        Image.fromarray(gray).save(output/('images'+suffix)/(case+'_0000.png'))
        Image.fromarray(mask.astype(np.uint8)).save(output/('labels'+suffix)/(case+'.png'))
    if any(not v for v in ids.values()): raise ValueError('All patient-separated partitions are required.')
    (output/'dataset.json').write_text(json.dumps({'channel_names': {'0': 'grayscale'}, 'labels': {'background': 0, 'CEJ': 1, 'crest': 2},
        'numTraining': len(ids['train'])+len(ids['val']), 'file_ending': '.png', 'overwrite_image_reader_writer': 'NaturalImage2DIO'}, indent=2))
    (output/'splits_final.json').write_text(json.dumps([{'train': ids['train'], 'val': ids['val']}], indent=2))
    (output/'heldout_ids.json').write_text(json.dumps(ids['test']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('rows', 'masks', 'output'): parser.add_argument(name, type=Path)
    args = parser.parse_args(); prepare(args.rows, args.masks, args.output)
