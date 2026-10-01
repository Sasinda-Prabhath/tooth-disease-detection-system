import argparse
import json
from pathlib import Path
from training.prepare import check_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='0')
    parser.add_argument('--evaluate', type=Path, help='Evaluate supplied best.pt on locked test patients; no training.')
    args = parser.parse_args()
    check_splits(json.loads((args.dataset.parent/'manifest.json').read_text()))
    from ultralytics import YOLO
    import mlflow
    with mlflow.start_run():
        mlflow.log_params({'architecture': 'yolo26l-seg', 'epochs': args.epochs, 'seed': 20261001})
        model = YOLO(str(args.evaluate) if args.evaluate else 'yolo26l-seg.pt')
        if not args.evaluate:
            model.train(data=str(args.dataset), epochs=args.epochs, imgsz=1024, seed=20261001,
                        deterministic=True, device=args.device, fliplr=0.0, flipud=0.0)
            model = YOLO(str(model.trainer.best))
        metrics = model.val(data=str(args.dataset), split='test' if args.evaluate else 'val', device=args.device)
        mlflow.log_metrics({'map50': float(metrics.box.map50), 'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr)})
        mlflow.log_artifact(str(args.dataset.parent/'manifest.json'))

if __name__ == '__main__':
    main()
