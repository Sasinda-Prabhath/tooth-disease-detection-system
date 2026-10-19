"""Compare original and fine-tuned detector on identical local validation settings."""
from dental_fdi.paths import PROJECT_ROOT
import argparse
import json
from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT=PROJECT_ROOT

def evaluate(weights, data, name):
    model=YOLO(str(weights))
    metrics=model.val(data=str(data), imgsz=1024, batch=1, device='cpu', workers=0,
                      conf=.001, iou=.7, plots=False, project=str(ROOT/'runs/evaluation'), name=name)
    return {'weights':str(weights), 'precision':float(metrics.box.mp),
            'recall':float(metrics.box.mr), 'mAP50':float(metrics.box.map50),
            'mAP50_95':float(metrics.box.map),
            'per_class_mAP50_95':{str(model.names[i]):float(v) for i,v in enumerate(metrics.box.maps)}}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'runs/detect/train-2/weights/best.pt')
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/'configs/data.yaml').read_text())
    config['path']=str(ROOT/'dataset')
    data=ROOT/'runs/configs/_evaluation_data.yaml'
    data.parent.mkdir(parents=True,exist_ok=True)
    data.write_text(yaml.safe_dump(config))
    baseline=evaluate(args.baseline,data,'baseline')
    candidate=evaluate(args.candidate,data,'candidate')
    result={'baseline':baseline,'candidate':candidate,
            'candidate_improves_mAP50_95':candidate['mAP50_95']>baseline['mAP50_95'],
            'validation_images':len(list((ROOT/'dataset/images/val').glob('*'))),
            'limitation':'Same small development validation set, not independent clinical validation. No angle ground truth available.',
            'automatically_deployed':False}
    (ROOT/'reports/evaluation').mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/evaluation/detector_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
