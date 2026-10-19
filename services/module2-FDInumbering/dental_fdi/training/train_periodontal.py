"""Train a YOLO26 six-landmark bone-level candidate after patient-split checks."""
import argparse
import csv
import json
from pathlib import Path
import yaml
from dental_fdi.paths import PROJECT_ROOT
from dental_fdi.analysis.bone_pose import BONE_SCHEMA, check_bone_dataset, write_bone_schema


def check_patient_manifest(root, manifest):
    errors=[]
    if not manifest.is_file():
        return [f'Missing {manifest}. Supply image,patient_id,split columns using pseudonymous patient IDs.']
    with manifest.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if not {'image','patient_id','split'}.issubset(reader.fieldnames or []):
            return ['Patient manifest requires image,patient_id,split columns']
        rows=list(reader)
    patients={}; seen=set()
    for row in rows:
        image=row['image'].strip();patient=row['patient_id'].strip();split=row['split'].strip()
        if not patient or not image or Path(image).name!=image or split not in ('train','val','test'):
            errors.append(f'Invalid patient-manifest row: {image}, {split}');continue
        key=(split,image)
        if key in seen:errors.append(f'Duplicate manifest entry: {key}')
        seen.add(key)
        if not (root/'images'/split/image).is_file():errors.append(f'Manifest image missing: {split}/{image}')
        patients.setdefault(patient,set()).add(split)
    for patient,splits in patients.items():
        if len(splits)>1:errors.append(f'Patient {patient} occurs in multiple splits: {sorted(splits)}')
    for split in ('train','val','test'):
        for image in (root/'images'/split).glob('*'):
            if image.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'} and (split,image.name) not in seen:
                errors.append(f'No patient ID for {split}/{image.name}')
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=PROJECT_ROOT/'configs/periodontal_pose.yaml')
    parser.add_argument('--patients',type=Path)
    parser.add_argument('--weights',default='yolo26s-pose.pt')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--imgsz',type=int,default=512)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--name',default='periodontal_landmarks')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(args.data.read_text(encoding='utf-8-sig'))
    root=Path(config.get('path','.'))
    if not root.is_absolute():root=(args.data.resolve().parent/root).resolve()
    names=config.get('names')
    if isinstance(names,dict):names=[names[k] for k in sorted(names)]
    errors=[]
    if names!=BONE_SCHEMA['names'] or config.get('kpt_shape')!=[6,3]:
        errors.append('Expected tooth_region class and kpt_shape [6,3]')
    if any(config.get(split)!=f'images/{split}' for split in ('train','val','test')):
        errors.append('Config must specify images/train, images/val, images/test')
    errors+=check_bone_dataset(root,splits=('train','val','test'))
    errors+=check_patient_manifest(root,args.patients or root/'patients.csv')
    if errors:
        print('Bone dataset is NOT ready:\n'+'\n'.join(errors));return 2
    print('Structural and declared patient-split checks passed. Expert landmark quality still requires review.')
    if args.check_only:return 0
    from ultralytics import YOLO
    model=YOLO(args.weights)
    if model.task!='pose' or 'yolo26' not in str(model.model.yaml.get('yaml_file','')).lower():
        raise ValueError('Use YOLO26 pose weights; detection/human-pose predictions are not dental landmarks until trained.')
    config['path']=str(root)
    resolved=PROJECT_ROOT/'runs/configs/_periodontal_yolo26.yaml'
    resolved.parent.mkdir(parents=True,exist_ok=True)
    resolved.write_text(yaml.safe_dump(config,sort_keys=False))
    model.train(data=str(resolved),epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,
        device=args.device,workers=0,project=str(PROJECT_ROOT/'runs/pose'),name=args.name,
        exist_ok=False,fliplr=0,flipud=0,mosaic=0,mixup=0,copy_paste=0,
        degrees=3,translate=.02,scale=.08,hsv_h=0,hsv_s=0,hsv_v=.1,seed=42,deterministic=True)
    best=Path(model.trainer.best)
    write_bone_schema(best.parent)
    (best.parent/'bone_training.json').write_text(json.dumps({
        'architecture':'YOLO26-pose','protocol':'yolo26_six_landmark_axial_v1',
        'paper_adaptation':'Landmarks, not the three U-Net segmentations in Lee et al.',
        'data':str(root),'patient_manifest':str(args.patients or root/'patients.csv'),
        'clinical_validation':'Not established','test_evaluation':'Not performed during training'},indent=2))
    print(f'Candidate: {best}. Evaluate on the untouched test set before enabling model proposals.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
