"""Six-landmark periodontal model contract and crop-coordinate conversion."""
import hashlib
import json
import math
from pathlib import Path
from PIL import ImageOps
from dental_fdi.analysis.dental_analysis import FDI_CODES

BONE_SCHEMA = {'version': 1, 'names': ['tooth_region'], 'kpt_shape': [6, 3],
               'keypoints': ['mesial_cej','mesial_crest','mesial_apex','distal_cej','distal_crest','distal_apex'],
               'input': 'clean_tooth_context_crop', 'padding_fraction': .25,
               'orientation': 'mesial_on_image_left; mirror quadrants 1 and 4'}


def context_crop(image, box, padding=.25):
    x1,y1,x2,y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError('Tooth box must be inside the clean image')
    px,py = padding*(x2-x1),padding*(y2-y1)
    bounds = [max(0,int(x1-px)),max(0,int(y1-py)),min(image.width,math.ceil(x2+px)),min(image.height,math.ceil(y2+py))]
    return image.crop(bounds).convert('RGB'), bounds


def canonical_bone_crop(image, box, fdi):
    if str(fdi) not in FDI_CODES:
        raise ValueError('A reviewed permanent FDI code is required for bone crop orientation')
    crop,bounds=context_crop(image,box)
    mirrored=str(fdi)[0] in ('1','4')
    return ImageOps.mirror(crop) if mirrored else crop,bounds,mirrored


def point_to_crop(point, bounds, mirrored):
    x,y=point[0]-bounds[0],point[1]-bounds[1]
    return [(bounds[2]-bounds[0]-1)-x if mirrored else x,y]


def point_from_crop(point, bounds, mirrored):
    x,y=point
    if mirrored:
        x=(bounds[2]-bounds[0]-1)-x
    return [x+bounds[0],y+bounds[1]]


def check_bone_dataset(root, splits=('train', 'val')):
    root=Path(root); errors=[]; hashes={}
    for split in splits:
        images=[p for p in (root/'images'/split).glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}]
        hashes[split]=set()
        if not images:
            errors.append(f'{split}: no expert-annotated periodontal crops')
        for image in images:
            hashes[split].add(hashlib.sha256(image.read_bytes()).hexdigest())
            label=root/'labels'/split/(image.stem+'.txt')
            try:
                lines=[line for line in label.read_text().splitlines() if line.strip()]
                if len(lines)!=1:
                    raise ValueError('one tooth_region row per crop is required')
                values=list(map(float,lines[0].split()))
                if len(values)!=23 or not all(map(math.isfinite,values)):
                    raise ValueError('expected class + box + six x/y/visibility triples (23 values)')
                c,x,y,w,h=values[:5]
                if c!=0 or not (0<w<=1 and 0<h<=1 and -1e-6<=x-w/2<=x+w/2<=1+1e-6 and -1e-6<=y-h/2<=y+h/2<=1+1e-6):
                    raise ValueError('invalid tooth_region box')
                visible=0
                for i in range(5,23,3):
                    kx,ky,v=values[i:i+3]
                    if v not in (0,1,2) or not (0<=kx<=1 and 0<=ky<=1):
                        raise ValueError('invalid landmark coordinate or visibility')
                    if v!=0:
                        visible+=1
                        if not (x-w/2-1e-6<=kx<=x+w/2+1e-6 and y-h/2-1e-6<=ky<=y+h/2+1e-6):
                            raise ValueError('visible landmark outside region box')
                if not visible:
                    raise ValueError('crop has no assessable landmarks')
            except (OSError,ValueError) as exc:
                errors.append(f'{label}: {exc}')
    import itertools
    for left,right in itertools.combinations(splits,2):
        if hashes[left] & hashes[right]:
            errors.append(f'Duplicate periodontal crops across {left}/{right}')
    return errors


def write_bone_schema(weights_dir):
    (Path(weights_dir)/'bone_schema.json').write_text(json.dumps(BONE_SCHEMA,indent=2))


def validate_bone_model(model, path):
    schema=Path(path).parent/'bone_schema.json'
    if (model.task!='pose' or [str(model.names[i]) for i in sorted(model.names)]!=BONE_SCHEMA['names']
            or list(model.model.model[-1].kpt_shape)!=[6,3]
            or not schema.is_file() or json.loads(schema.read_text())!=BONE_SCHEMA):
        raise ValueError('Expected trained six-landmark tooth_region pose weights and bone_schema.json')


def predict_bone_points(model, image, box, fdi, threshold=.5):
    crop,bounds,mirrored=canonical_bone_crop(image,box,fdi)
    result=model.predict(crop,conf=.25,imgsz=512,verbose=False)[0]
    if result.boxes is None or result.keypoints is None or len(result.boxes)==0:
        return [], []
    index=max(range(len(result.boxes)),key=lambda i:float(result.boxes[i].conf[0]))
    values=result.keypoints.data[index].cpu().tolist()
    if len(values)!=6:
        raise ValueError('Periodontal model did not return six keypoints')
    points=[]; confidence=[]
    for x,y,score in values:
        valid=all(map(math.isfinite,(x,y,score))) and score>=threshold and 0<=x<=crop.width-1 and 0<=y<=crop.height-1
        points.append(point_from_crop([x,y],bounds,mirrored) if valid else None)
        confidence.append(score if math.isfinite(score) else None)
    return points,confidence
