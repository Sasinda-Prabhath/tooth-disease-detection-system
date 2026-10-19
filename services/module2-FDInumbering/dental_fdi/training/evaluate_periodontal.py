"""Evaluate a trained YOLO26 bone model on annotated canonical test crops."""
import argparse
import json
import math
from pathlib import Path
from PIL import Image
from dental_fdi.analysis.bone_pose import validate_bone_model, check_bone_dataset
from dental_fdi.analysis.periodontal_measurement import measure_periodontal_sites
from dental_fdi.training.train_periodontal import check_patient_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True,type=Path)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=Path('reports/evaluation/periodontal_yolo26.json'))
    args=parser.parse_args()
    errors=check_bone_dataset(args.dataset,('train','val','test'))+check_patient_manifest(args.dataset,args.dataset/'patients.csv')
    if errors:raise ValueError('\n'.join(errors))
    from ultralytics import YOLO
    model=YOLO(str(args.weights));validate_bone_model(model,args.weights)
    records=[];landmark_errors=[];ratio_errors=[]
    for path in sorted((args.dataset/'images/test').glob('*')):
        if path.suffix.lower() not in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}:continue
        image=Image.open(path).convert('RGB')
        values=list(map(float,(args.dataset/'labels/test'/(path.stem+'.txt')).read_text().split()))
        truth=[[values[i]*image.width,values[i+1]*image.height] if values[i+2]==2 else None for i in range(5,23,3)]
        row={'image':path.name,'status':'Unavailable','landmark_errors_normalized':[]}
        result=model.predict(image,conf=.25,imgsz=512,verbose=False)[0]
        if result.boxes is not None and result.keypoints is not None and len(result.boxes):
            index=max(range(len(result.boxes)),key=lambda i:float(result.boxes[i].conf[0]))
            raw=result.keypoints.data[index].cpu().tolist()
            if len(raw)==6:
                predicted=[[x,y] if score>=.5 and all(map(math.isfinite,(x,y,score))) and 0<=x<image.width and 0<=y<image.height else None for x,y,score in raw]
                for gt,pred in zip(truth,predicted):
                    if gt is not None and pred is not None:
                        error=math.dist(gt,pred)/math.hypot(*image.size)
                        landmark_errors.append(error);row['landmark_errors_normalized'].append(error)
                if all(p is not None for p in truth+predicted):
                    try:
                        reference=measure_periodontal_sites(truth,*image.size)
                        estimate=measure_periodontal_sites(predicted,*image.size)
                        errors=[abs(estimate['sites'][s]['radiographic_bone_level_percent']-reference['sites'][s]['radiographic_bone_level_percent']) for s in ('mesial','distal')]
                        ratio_errors+=errors
                        row.update(status='Measured',ratio_errors_percentage_points=errors)
                    except ValueError as exc:row['reason']=str(exc)
        records.append(row)
    measured=sum(r['status']=='Measured' for r in records)
    report={'test_crops':len(records),'measured_crops':measured,'unavailable_crops':len(records)-measured,
            'mean_landmark_error_fraction_of_crop_diagonal':sum(landmark_errors)/len(landmark_errors) if landmark_errors else None,
            'bone_ratio_MAE_percentage_points':sum(ratio_errors)/len(ratio_errors) if ratio_errors else None,
            'note':'Errors cover available outputs; report coverage alongside error. No clinical validation implied.',
            'records':records}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
