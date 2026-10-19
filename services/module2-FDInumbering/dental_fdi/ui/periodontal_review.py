"""Bone-level geometry from explicitly reviewed landmarks, never box heuristics."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import ImageDraw

from dental_fdi.analysis.bone_pose import BONE_SCHEMA, context_crop, predict_bone_points, validate_bone_model
from dental_fdi.analysis.manual_geometry import measure_bone_sites
from dental_fdi.analysis.bone_predictions import predict_bone_inventory
from dental_fdi.paths import RUNS_DIR
from dental_fdi.ui.molar_review import axis_picker
from dental_fdi.reporting.training_examples import export_bone_example


@st.cache_resource
def load_bone_model(path, modified_ns):
    from ultralytics import YOLO
    model=YOLO(path); validate_bone_model(model,path); return model


def periodontal_review(image, boxes, image_id):
    st.subheader('Periodontal bone-level measurements — reviewed landmarks')
    st.caption('Uses the clean X-ray and corrected tooth boxes. No assumed apex, CEJ, millimetre scale, '
               'furcation grade or patient disease stage is generated. The definition below needs clinician agreement.')
    path=Path(os.environ.get('FDI_BONE_MODEL_PATH',str(RUNS_DIR/'pose/periodontal_landmarks/weights/best.pt')))
    model=None
    if path.is_file():
        try:
            model=load_bone_model(str(path),path.stat().st_mtime_ns)
            st.caption('Six-landmark model available. Predictions remain unvalidated proposals until reviewed.')
        except Exception as exc:
            st.error(f'Bone landmark model unavailable: {exc}')
    else:
        st.info('No trained bone-landmark model is installed. Manual measurement is available; automatic bone prediction is unavailable.')
    if not boxes:
        st.info('Detect or manually draw a visible tooth box first. No bone measurement was made.'); return
    signature=hashlib.sha256(json.dumps(boxes,sort_keys=True).encode()).hexdigest()[:16]
    identity=image_id+signature
    states=st.session_state.setdefault('bone_review_'+identity,{})
    if model is not None:
        model_signature=str(path.resolve())+str(path.stat().st_mtime_ns)
        cache_key='bone_predictions_'+identity+hashlib.sha256(model_signature.encode()).hexdigest()[:12]
        if cache_key not in st.session_state:
            with st.spinner('Predicting CEJ, bone crest and root apex for every reviewed tooth...'):
                st.session_state[cache_key]=predict_bone_inventory(model,image,boxes)
        for fdi,proposal in st.session_state[cache_key].items():
            saved=states.get(fdi)
            if saved is None or (saved.get('source','').startswith('Model') and saved.get('model_signature')!=model_signature):
                states[fdi]={**proposal,'model_signature':model_signature}
    selected=st.selectbox('Tooth for bone-level measurement',[b['fdi'] for b in boxes],key='bone_fdi_'+identity)
    box=next(b for b in boxes if b['fdi']==selected)
    state=states.setdefault(selected,{'points':[],'confirmed':False,'revision':0,'source':'Manual landmarks'})
    if st.button('Propose six bone landmarks',disabled=model is None,key='bone_propose_'+identity):
        points,confidence=predict_bone_points(model,image,box['xyxy'],selected)
        state.update(points=points,confidence=confidence,confirmed=False,revision=state['revision']+1,source='Model estimate; requires review')
        state.pop('error',None)
    if st.button('Clear bone landmarks for this tooth',key='bone_clear_'+identity):
        state.update(points=[],confirmed=False,revision=state['revision']+1,source='Manual landmarks')
        state.pop('error',None)
    st.caption('Click mesial CEJ, mesial crest, mesial apex, distal CEJ, distal crest, distal apex. '
               'Mesial means toward the midline. Leave obscured landmarks unmeasured. Use the dropdown to correct a point.')
    preview,bounds=context_crop(image,box['xyxy'])
    buffer=io.BytesIO();preview.save(buffer,format='PNG')
    component_id=identity+selected+str(state['revision'])
    local=[[p[0]-bounds[0],p[1]-bounds[1]] if p else None for p in state['points']]
    value=axis_picker(image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),
        points=local,identity=component_id,key='bone_picker_'+component_id,
        labels=[key.replace('_',' ') for key in BONE_SCHEMA['keypoints']],
        colors=['#ff4b78','#00d2ff','#38ef7d']*2,
        segments=[[0,1,'#ff4b78'],[0,2,'#38ef7d'],[3,4,'#ff4b78'],[3,5,'#38ef7d']],default=None)
    if value and value.get('identity')==component_id:
        points=[[p[0]+bounds[0],p[1]+bounds[1]] if p else None for p in value['points']]
        if points!=state['points']:
            state.update(points=points,confirmed=False,source='User-edited landmarks')
            state.pop('error',None)
    result=None
    try:
        if len(state['points'])==6 and all(p is not None for p in state['points']):
            result=measure_bone_sites(state['points'],*image.size)
            st.write(f"Draft CEJ/crest/apex ratio: mesial **{result['sites']['mesial']['radiographic_bone_level_percent']:.1f}%**, distal **{result['sites']['distal']['radiographic_bone_level_percent']:.1f}%**")
    except ValueError as exc:
        st.error(str(exc))
    point_signature=hashlib.sha256(json.dumps(state['points']).encode()).hexdigest()[:12]
    state['confirmed']=st.checkbox('I reviewed these six anatomical landmarks and the measurement definition',
        value=state['confirmed'],disabled=result is None,key='bone_confirm_'+component_id+point_signature)
    if state['confirmed']:
        try:
            example=export_bone_example(image,box,state['points'],image_id)
            st.download_button('Download this reviewed bone training example',example,
                'bone_training_'+selected+'.zip','application/zip',on_click='ignore')
        except ValueError as exc:
            st.error(str(exc))
    st.caption('Definition: 100 x CEJ-to-crest distance / CEJ-to-apex distance per side, in 2D image pixels. '
               'No physiological offset is applied. Model estimates require review; these ratios do not establish a clinical stage.')
    rows=[]
    measurements={}
    for box in boxes:
        fdi=box['fdi']; saved=states.get(fdi,{})
        measurement=None
        status=saved.get('error','Unavailable: landmarks not supplied')
        points=saved.get('points',[])
        if len(points)==6 and all(p is not None for p in points):
            try:
                measurement=measure_bone_sites(points,*image.size)
                status='Reviewed' if saved.get('confirmed') else 'Unreviewed estimate'
            except ValueError as exc:
                status=f'Unavailable: {exc}'
        row={'FDI':fdi,'Status':status,'Source':saved.get('source','Unavailable'),
             'Mesial (%)':None,'Distal (%)':None,'Maximum (%)':None,
             'Minimum landmark confidence':None}
        if measurement is not None:
            measurements[fdi]=measurement
            row.update({'Mesial (%)':measurement['sites']['mesial']['radiographic_bone_level_percent'],
                        'Distal (%)':measurement['sites']['distal']['radiographic_bone_level_percent'],
                        'Maximum (%)':measurement['maximum_percent']})
            scores=saved.get('confidence',[])
            if saved.get('source','').startswith('Model') and scores and all(c is not None for c in scores):
                row['Minimum landmark confidence']=min(scores)
        rows.append(row)
    cards=st.columns(4)
    cards[0].metric('Teeth with measurements',len(measurements))
    cards[1].metric('Reviewed measurements',sum(r['Status']=='Reviewed' for r in rows))
    cards[2].metric('Unavailable',len(boxes)-len(measurements))
    maximum=max((m['maximum_percent'] for m in measurements.values()),default=None)
    cards[3].metric('Maximum bone-level ratio',f'{maximum:.1f}%' if maximum is not None else 'Unavailable')
    view,controls=st.columns([3,1])
    with controls:
        st.write('Overlay controls')
        show_cej=st.checkbox('CEJ landmarks (pink)',True,key='bone_show_cej_'+identity)
        show_crest=st.checkbox('Bone crest (cyan)',True,key='bone_show_crest_'+identity)
        show_apex=st.checkbox('Root apex / root reference (green)',True,key='bone_show_apex_'+identity)
        show_ratio=st.checkbox('FDI and percentage labels',True,key='bone_show_ratio_'+identity)
        reviewed_only=st.checkbox('Show reviewed measurements only',False,key='bone_reviewed_only_'+identity)
    overlay=image.convert('RGB').copy();draw=ImageDraw.Draw(overlay)
    radius=max(2,round(image.width/500));line_width=max(2,round(image.width/700))
    for fdi,measurement in measurements.items():
        saved=states[fdi]
        if reviewed_only and not saved.get('confirmed'):
            continue
        for offset in (0,3):
            cej,crest,apex=map(tuple,saved['points'][offset:offset+3])
            if show_apex:
                draw.line([cej,apex],fill='#38ef7d',width=line_width)
            if show_crest:
                draw.line([cej,crest],fill='#00d2ff',width=line_width+1)
            for point,color,enabled in [(cej,'#ff4b78',show_cej),(crest,'#00d2ff',show_crest),(apex,'#38ef7d',show_apex)]:
                if enabled:
                    x,y=point;draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
        if show_ratio:
            marker='' if saved.get('confirmed') else ' [unreviewed]'
            draw.text(tuple(saved['points'][0]),f"{fdi}: {measurement['maximum_percent']:.1f}%{marker}",fill='#ff4b78')
    with view:
        st.image(overlay,caption='Bone-level measurements on the clean X-ray; absent results are unavailable, not zero.',width='stretch')
        out=io.BytesIO();overlay.save(out,format='PNG')
        st.download_button('Download bone prediction overlay',out.getvalue(),'bone_prediction.png','image/png',on_click='ignore')
    filter_value=st.selectbox('Bone measurement table',['All teeth','Reviewed','Unreviewed estimates','Unavailable'],key='bone_table_filter_'+identity)
    filtered=[r for r in rows if filter_value=='All teeth' or
              (filter_value=='Reviewed' and r['Status']=='Reviewed') or
              (filter_value=='Unreviewed estimates' and r['Status']=='Unreviewed estimate') or
              (filter_value=='Unavailable' and r['Status'].startswith('Unavailable'))]
    st.dataframe(pd.DataFrame(filtered),hide_index=True,width='stretch')
    st.download_button('Download bone measurements CSV',pd.DataFrame(rows).to_csv(index=False),
                       'bone_measurements.csv','text/csv',on_click='ignore')
    st.download_button('Download bone landmarks and review JSON',json.dumps({'image_sha256':image_id,
        'image_size':list(image.size),'boxes':boxes,'landmark_order':BONE_SCHEMA['keypoints'],
        'landmarks':states,'measurements':rows,'definition':'100 * CEJ-to-crest / CEJ-to-apex; no physiological offset',
        'clinical_stage':'Not assessed','millimetres':None},indent=2),'bone_review.json','application/json',on_click='ignore')
