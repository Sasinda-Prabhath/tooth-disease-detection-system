from ultralytics import YOLO
from PIL import Image
from dental_fdi.analysis.dental_analysis import select_detections
from dental_fdi.analysis.image_angulation import estimate_image_angles
import json
from dental_fdi.paths import DATASET_DIR, DEFAULT_DETECTOR
im=Image.open(DATASET_DIR / 'images/val/train_1500.png').convert('RGB')
m=YOLO(str(DEFAULT_DETECTOR))
r=m.predict(im,conf=.35,imgsz=1024,verbose=False)[0]
rows=[{'pred_fdi':str(r.names[int(b.cls[0])]),'conf':float(b.conf[0]),'xyxy':b.xyxy[0].tolist()} for b in r.boxes]
boxes=select_detections(rows,*im.size)
out,axes=estimate_image_angles(im,boxes)
print(json.dumps([{k:v for k,v in x.items() if k!='points'} for x in out],indent=2))
