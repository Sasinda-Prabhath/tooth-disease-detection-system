"""Fine-tune detection or two-landmark molar pose models without replacing deployed weights."""
from dental_fdi.paths import PROJECT_ROOT
import argparse
import json
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO
from dental_fdi.data.dataset_checks import check_dataset
from dental_fdi.analysis.pose_pipeline import POSE_NAMES, check_pose_dataset, write_axis_schema
from dental_fdi.analysis.bone_pose import BONE_SCHEMA, check_bone_dataset, write_bone_schema

ROOT = PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['detect', 'pose', 'periodontal'], default='detect')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--weights')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--name', default='fdi_finetune')
    parser.add_argument('--freeze', type=int, default=0)
    parser.add_argument('--lr0', type=float, default=0.0001)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config_name = {'detect':'data.yaml','pose':'pose_data.yaml','periodontal':'periodontal_pose.yaml'}[args.task]
    data_file = (args.data or ROOT / 'configs' / config_name).resolve()
    config = yaml.safe_load(data_file.read_text(encoding='utf-8-sig'))
    data_root = Path(config.get('path', '.'))
    if not data_root.is_absolute():
        data_root = (data_file.parent / data_root).resolve()
    if config.get('train') != 'images/train' or config.get('val') != 'images/val':
        raise ValueError('Use images/train and images/val directories in the dataset configuration.')
    names = config['names']
    names = list(names.values()) if isinstance(names, dict) else names
    expected = ([f'{q}{t}' for q in range(1,5) for t in range(1,9)] if args.task == 'detect'
                else BONE_SCHEMA['names'] if args.task == 'periodontal' else POSE_NAMES)
    if list(map(str, names)) != expected:
        raise ValueError('Dataset class ordering does not match the required FDI schema.')
    if args.task == 'pose' and config.get('kpt_shape') != [2,3]:
        raise ValueError('Pose training requires kpt_shape: [2, 3], root then crown.')
    if args.task == 'periodontal' and config.get('kpt_shape') != [6,3]:
        raise ValueError('Periodontal training requires six CEJ/crest/apex landmarks, kpt_shape [6,3].')
    errors = (check_dataset(data_root) if args.task == 'detect' else
              check_bone_dataset(data_root) if args.task == 'periodontal' else check_pose_dataset(data_root))
    if errors:
        raise ValueError('Dataset is not ready:\n' + '\n'.join(errors))
    print('Structural checks passed. Expert label review and patient-disjoint splits are still required.')
    if args.check_only:
        return
    default_weights = str(ROOT / 'runs/detect/train-2/weights/best.pt') if args.task == 'detect' else 'yolo26s-pose.pt'
    model = YOLO(args.weights or default_weights)
    model_task = 'detect' if args.task == 'detect' else 'pose'
    if model.task != model_task:
        raise ValueError(f'Expected {model_task} weights; got {model.task}.')
    config['path'] = str(data_root)
    resolved = ROOT / 'runs/configs' / f'_{args.task}_resolved_data.yaml'
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    # Gentle whole-OPG augmentation preserves FDI side and global anatomy.
    model.train(data=str(resolved), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0,
                project=str(ROOT / 'runs' / model_task), name=args.name,
                exist_ok=False, optimizer='AdamW', lr0=args.lr0, lrf=0.1,
                freeze=args.freeze, patience=25, seed=42, deterministic=True,
                warmup_bias_lr=0.0, warmup_epochs=1.0,
                fliplr=0, flipud=0, mosaic=0, mixup=0, copy_paste=0,
                degrees=3, translate=0.02, scale=0.08,
                hsv_h=0, hsv_s=0, hsv_v=0.1, plots=True)
    best = Path(model.trainer.best)
    if args.task == 'pose':
        write_axis_schema(best.parent)
    if args.task == 'periodontal':
        write_bone_schema(best.parent)
    summary = {'best': str(best), 'task': args.task, 'data': str(data_root),
               'training_arguments': vars(args), 'deployed': False,
               'note': 'Candidate only. Compare held-out localization and angle errors before deployment.'}
    (best.parent.parent / 'candidate_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(f'Candidate saved: {best}. Existing deployed weights were not changed.')

if __name__ == '__main__':
    main()
