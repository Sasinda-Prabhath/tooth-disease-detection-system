"""Export clean pixels and annotations separately; never train on painted overlays."""
import hashlib
import io
import json
import zipfile

from dental_fdi.analysis.bone_pose import BONE_SCHEMA, canonical_bone_crop, point_to_crop
from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.analysis.tooth_inventory import tooth_inventory


def export_fdi_example(image, boxes, confirmed_missing):
    inventory=tooth_inventory(boxes,confirmed_missing)
    digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    name='opg_'+digest[:16]
    lines=[]
    for box in boxes:
        x1,y1,x2,y2=box['xyxy']
        if not (0<=x1<x2<=image.width and 0<=y1<y2<=image.height):
            raise ValueError('Training boxes must lie inside the clean image')
        lines.append(f"{FDI_CODES.index(box['fdi'])} {(x1+x2)/2/image.width:.8f} {(y1+y2)/2/image.height:.8f} {(x2-x1)/image.width:.8f} {(y2-y1)/image.height:.8f}")
    status={'schema':'fdi_presence_annotation_v1','status':{
        row['FDI']: 'missing' if row['Status']=='Confirmed missing' else 'present' if row['Status']=='Detected' else 'uncertain'
        for row in inventory}}
    pixels=io.BytesIO();image.save(pixels,format='PNG')
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('images/'+name+'.png',pixels.getvalue())
        archive.writestr('labels/'+name+'.txt','\n'.join(lines)+'\n')
        archive.writestr('presence/'+name+'.json',json.dumps(status,indent=2))
        archive.writestr('review.json',json.dumps({'source_image_sha256':digest,'image_size':list(image.size),
            'boxes':boxes,'training_status':'User-reviewed; expert verification and patient split still required',
            'pixel_content':'Clean, oriented radiograph; no boxes or lines burned in'},indent=2))
    return output.getvalue()


def export_bone_example(image, box, points, source_id):
    from dental_fdi.analysis.manual_geometry import measure_bone_sites
    measure_bone_sites(points,*image.size)
    crop,bounds,mirrored=canonical_bone_crop(image,box['xyxy'],box['fdi'])
    values=['0','0.5','0.5','1','1']
    for x,y in points:
        lx,ly=point_to_crop([x,y],bounds,mirrored)
        px,py=lx/crop.width,ly/crop.height
        if not (0<=px<=1 and 0<=py<=1):
            raise ValueError('A reviewed landmark is outside the training crop; enlarge the corrected tooth box')
        values.extend([f'{px:.8f}',f'{py:.8f}','2'])
    name='opg_'+source_id[:16]+'_FDI'+box['fdi']
    pixels=io.BytesIO();crop.save(pixels,format='PNG');output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('images/'+name+'.png',pixels.getvalue())
        archive.writestr('labels/'+name+'.txt',' '.join(values)+'\n')
        archive.writestr('review.json',json.dumps({'source_image_sha256':source_id,'fdi':box['fdi'],
            'crop_bounds':bounds,'canonical_mirrored':mirrored,'schema':BONE_SCHEMA,'points_in_original_image':points,
            'split_requirement':'All crops from the source patient must stay in the same split'},indent=2))
    return output.getvalue()
