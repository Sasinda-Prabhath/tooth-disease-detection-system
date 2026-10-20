"""Automatic overlays from an optional landmark model or unsupervised image axes."""
from dental_fdi.paths import PROJECT_ROOT
import io
import math
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from dental_fdi.analysis.image_angulation import estimate_image_angles
from dental_fdi.analysis.pose_pipeline import validate_axis_model, pose_rows, automatic_angles

from dental_fdi.analysis.manual_geometry import line_angle
from dental_fdi.ui.reference_lines import draw_angle_wedge, selected_screen_geometry, manual_summary, draw_coordinate_labels
from dental_fdi.reporting.prediction_exports import render_detection_overlay


def automatic_reference_geometry(axis, width, height, box=None, fdi=None):
    """Center all three guides on the tooth; order yellow endpoints inward first."""
    from dental_fdi.analysis.manual_geometry import validate_points
    validate_points(axis, 2, width, height)
    line_angle([[0,0],[max(2,width-1),0]],axis)
    if box is not None:
        x1,y1,x2,y2=box
        x,y=(x1+x2)/2,(y1+y2)/2
        reach=max(x2-x1,y2-y1)*.85
    else:
        x,y=[(axis[0][i]+axis[1][i])/2 for i in (0,1)]
        reach=math.dist(*axis)*.8
    span=max(12,min(width,height)*.18)
    dx,dy=axis[1][0]-axis[0][0],axis[1][1]-axis[0][1]
    length=math.hypot(dx,dy)
    dx,dy=dx/length,dy/length
    # Image left (FDI quadrants 1/4) starts on the right; image right starts on the left.
    start_right=str(fdi)[0] in '14' if fdi else x<width/2
    if (start_right and dx<0) or (not start_right and dx>0):
        dx,dy=-dx,-dy
    ends=[]
    for sign in (1,-1):
        distance=reach
        for coordinate,delta,bound in zip((x,y),(sign*dx,sign*dy),(width,height)):
            if delta>0:distance=min(distance,(bound-1-coordinate)/delta)
            elif delta<0:distance=min(distance,-coordinate/delta)
        ends.append([x+sign*distance*dx,y+sign*distance*dy])
    points=[[max(0,x-span),y],[min(width-1,x+span),y],
            [x,max(0,y-span)],[x,min(height-1,y+span)]]+ends
    return points,selected_screen_geometry(points,fdi)['horizontal_angle_degrees']


def render_automatic_reference(image, boxes, axis, fdi=None):
    box=next((b['xyxy'] for b in boxes if b['fdi']==fdi),None)
    points,angle=automatic_reference_geometry(axis,*image.size,box=box,fdi=fdi)
    overlay=render_detection_overlay(image,boxes,show_conf=False)
    draw=ImageDraw.Draw(overlay)
    draw_angle_wedge(draw,points,0,image.size,fill='#ffdb35',fdi=fdi)
    stroke=max(3,image.width//500)
    for a,b,color in [(0,1,'#69cf20'),(2,3,'#ef3045'),(4,5,'#ffdb35')]:
        draw.line([tuple(points[a]),tuple(points[b])],fill=color,width=stroke)
    draw_coordinate_labels(draw,points)
    # Arrow identifies the requested start-to-end direction on either side.
    start,end=points[4:6]
    theta=math.atan2(end[1]-start[1],end[0]-start[0])
    size=max(7,image.width/180)
    tips=[(end[0]-size*math.cos(theta+offset),end[1]-size*math.sin(theta+offset))
          for offset in (-.45,.45)]
    draw.line([tips[0],tuple(end),tips[1]],fill='#ffdb35',width=stroke)
    draw.rectangle((0,0,image.width,30),fill='#101820')
    draw.text((8,8),f'Automatic axis to horizontal: {angle:.1f} deg',fill='#ffdb35')
    return overlay,points,angle


from dental_fdi.analysis.manual_geometry import screen_axis_tilt, requested_winter_category


def render_all_wisdom(image, boxes, axes, rows, guide_overrides=None):
    """One source-sized X-ray plus a readable footer with all four outcomes."""
    from PIL import ImageOps
    overlay=render_detection_overlay(image,boxes,show_conf=False)
    results=[]
    by_fdi={r['FDI']:r for r in rows}
    detected={b['fdi'] for b in boxes}
    for fdi in ('18','28','38','48'):
        row=by_fdi.get(fdi,{})
        signed=row.get('Angle (degrees)')
        category=requested_winter_category(signed)
        horizontal=None
        screen_tilt=None
        screen_summary={}
        if fdi in detected and fdi in axes:
            screen_tilt=screen_axis_tilt(axes[fdi]['points'])
            # Use the detected tooth center (and preserve manual adjustments when supplied).
            box=next(b['xyxy'] for b in boxes if b['fdi']==fdi)
            points,horizontal=automatic_reference_geometry(axes[fdi]['points'],*image.size,box,fdi)
            if guide_overrides is not None:
                points=guide_overrides.get(fdi)
                if not points or len(points)!=6 or any(p is None for p in points):
                    points=None
            if points:
                draw=ImageDraw.Draw(overlay)
                draw_angle_wedge(draw,points,0,image.size,fill='#ffdb35',fdi=fdi)
                for a,b,color in [(0,1,'#69cf20'),(2,3,'#ef3045'),(4,5,'#ffdb35')]:
                    draw.line([tuple(points[a]),tuple(points[b])],fill=color,width=max(3,image.width//500))
                draw_coordinate_labels(draw,points)
                horizontal=selected_screen_geometry(points,fdi)['horizontal_angle_degrees']
                screen_summary=manual_summary(fdi,points,*image.size)
        if fdi not in detected:
            signed=None;category='Not detected';horizontal=None
        results.append({'FDI':fdi,'angle_to_second_molar_degrees':signed,
                        'classification':category,'horizontal_angle_degrees':horizontal,
                        'screen_tilt_degrees':screen_tilt,
                        'signed_horizontal_degrees':screen_summary.get('signed_horizontal_degrees'),
                        'screen_category':screen_summary.get('screen_category','Unavailable'),
                        'X_up_pixels':screen_summary.get('X_up_pixels'),
                        'Y_right_pixels':screen_summary.get('Y_right_pixels'),
                        'screen_tilt_convention':'From image-up, root to crown; left negative, right positive',
                        'source':'Provisional axis estimate / manual adjustments; unvalidated'})
    # Remove per-tooth header from the renderer; footer identifies every result explicitly.
    overlay.paste(render_detection_overlay(image,boxes,show_conf=False).crop((0,0,image.width,30)),(0,0))
    font_size=max(12,min(28,image.width//65))
    try:font=ImageFont.truetype('arial.ttf',font_size)
    except OSError:font=ImageFont.load_default()
    line_height=font_size+10
    overlay=ImageOps.expand(overlay,border=(0,0,0,line_height*6+12),fill='#101820')
    draw=ImageDraw.Draw(overlay)
    labels=['Provisional angulation. Screen: left - / right +. Anatomical: mesial + / distal -. Impaction unconfirmed.']
    for r in results:
        signed=r['angle_to_second_molar_degrees']
        value=f'{signed:+.1f} deg' if signed is not None else 'unavailable'
        horizontal=r['horizontal_angle_degrees']
        h=f'{horizontal:.1f} deg' if horizontal is not None else 'unavailable'
        tilt=r['screen_tilt_degrees']
        screen=f'{tilt:+.1f}' if tilt is not None else 'N/A'
        sy=r['signed_horizontal_degrees']
        sy_text=f'{sy:+.1f}' if sy is not None else 'N/A'
        labels.append(f"FDI {r['FDI']}: screen {sy_text} deg {r['screen_category']} | anatomical {value} {r['classification']} | guide {h}")
    labels.append('Ranges (rounded degrees): vertical -10..10; mesial 11..79; horizontal 80..100; distal -79..-11; other outside')
    for i,label in enumerate(labels):draw.text((8,image.height+6+i*line_height),label,font=font,fill='#ffdb35')
    return overlay,results


ROOT = PROJECT_ROOT

@st.cache_resource
def load_axis_model(path, modified_ns):
    model = YOLO(path)
    validate_axis_model(model, path)
    return model

@st.cache_data(max_entries=8, show_spinner=False)
def estimate_pixels(image_bytes, boxes):
    # Geometry revision 2: corrected moment major-axis orientation.
    return estimate_image_angles(Image.open(io.BytesIO(image_bytes)), boxes)

@st.cache_data(max_entries=8, show_spinner=False)
def predict_axes(image_bytes, path, modified_ns, _model):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    result = _model.predict(image, conf=0.35, imgsz=1024, verbose=False)[0]
    axes = pose_rows(result, *image.size)
    return automatic_angles(axes, *image.size), {a['fdi']: a for a in axes}


def automatic_angulation_view(image, boxes, image_id):
    st.subheader("Automatic molar-axis estimates")
    st.caption("Check the proposed axes before using these estimates. Review and correction tools are below.")
    
    path = Path(os.environ.get('FDI_AXIS_MODEL_PATH', str(ROOT / 'runs/pose/molar_axes/weights/best.pt')))
    image_buffer = io.BytesIO()
    image.save(image_buffer, format='PNG')
    rows, axes = None, None

    if path.is_file():
        try:
            model = load_axis_model(str(path), path.stat().st_mtime_ns)
            rows, axes = predict_axes(image_buffer.getvalue(), str(path), path.stat().st_mtime_ns, model)
            st.caption('Automatic root/crown landmark-model estimates. No line drawing required.')
        except Exception as exc:
            st.error(f'Landmark model unavailable: {exc}. Using radiographic image-axis estimates.')

    if rows is None:
        st.info('Automatic image-based estimates: no trained tooth-axis landmark model is installed or usable. Root/crown directions are assumed; results require review.')
        with st.spinner('Calculating 18, 28, 38, 48 longitudinal molar axes & angulations...'):
            rows, axes = estimate_pixels(image_buffer.getvalue(), boxes)

    # Render summary metric cards for 18, 28, 38, 48
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    for idx, r in enumerate(rows):
        fdi = r['FDI']
        angle = r.get('Angle (degrees)', r.get('Estimated angle (degrees)'))
        cat = requested_winter_category(angle)
        imp = r.get('Impaction', 'Not assessed')
        
        with cols[idx]:
            if angle is not None:
                color = "#216647" if cat == "Vertical" else "#825500"
                st.markdown(
                    f"""<div style="background:#ffffff;border:1px solid #dce4e9;border-radius:8px;padding:12px;margin-bottom:10px;">
                    <div style="font-size:18px;font-weight:700;color:#176b70;">Tooth #{fdi}</div>
                    <div style="font-size:22px;font-weight:800;color:{color};">{angle:+.1f}°</div>
                    <div style="font-size:13px;font-weight:600;color:#243746;">{cat}</div>
                    <div style="font-size:12px;color:#536775;">{imp}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="background:#ffffff;border:1px solid #dce4e9;border-radius:8px;padding:12px;margin-bottom:10px;">
                    <div style="font-size:18px;font-weight:700;color:#176b70;">Tooth #{fdi}</div>
                    <div style="font-size:16px;font-weight:700;color:#536775;">Not Available</div>
                    <div style="font-size:12px;color:#536775;">{r.get('Status', 'Not detected')}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

    overlay, combined_results = render_all_wisdom(image, boxes, axes, rows)
    st.image(overlay,width='stretch',caption='All detected wisdom teeth in one image. Green: horizontal guides and wedges; red: vertical guides; yellow: tooth axes.')
    st.download_button('Download all wisdom-tooth measurements (JSON)',
        json.dumps({'image_sha256':image_id,'measurements':combined_results},indent=2),
        'all_wisdom_angles.json','application/json',on_click='ignore')

    # Detailed Table View
    display_rows = []
    for r in rows:
        row_dict = {
            'FDI': r.get('FDI'),
            'Reference Molar': r.get('Reference'),
            'Angulation (°)': f"{r.get('Angle (degrees)'):+.1f}°" if r.get('Angle (degrees)') is not None else "None",
            "Winter's Classification": requested_winter_category(r.get('Angle (degrees)')),
            'Pell & Gregory Classification': r.get('Pell & Gregory', 'N/A'),
            'Impaction Status': r.get('Impaction', 'Not determined from angulation alone'),
            'Measurement type': 'Automatic estimate; unvalidated',
            'Status': r.get('Status', 'N/A'),
        }
        display_rows.append(row_dict)

    with st.expander('Detailed automatic measurements'):
        st.dataframe(pd.DataFrame(display_rows), hide_index=True, width='stretch')

    # Medical Reference & Classification Rules Expander
    with st.expander("Classification ranges and measurement method"):
        st.markdown(
            """
            **Reference Study:** *Prevalence and Pattern of Impacted Mandibular Third Molar: An Institution-Based Retrospective Study*  
            *Hiren Patel, Haren Pandya, Hitesh Dewan, Bijal Bhavsar, Urvi Shah, Hirakben Patel, Kartik Dholakia, Nirmay Patel (2024)*  
            *PMCID: PMC11607221 | PMID: 39618470*

            #### 1. Winter's Classification (Angulation)
            Angulation is measured as the angle between the longitudinal axis of the third molar and the intersected longitudinal axis of the adjacent second molar:
            - **Vertical:** -10° to +10°
            - **Mesioangular:** +11° to +80° (crown tilted mesially toward adjacent molar)
            - **Horizontal:** +81° to +100° (tooth lying horizontally in alveolar bone/ramus)
            - **Distoangular:** -11° to -80° (crown tilted distally away from adjacent molar)
            - **Inverted / Transverse:** > +100° or < -80°

            #### 2. Pell & Gregory Classification (Mandibular 38 & 48)
            - **Occlusal Depth:**
              - **Position A:** Occlusal plane of third molar at or above occlusal plane of second molar.
              - **Position B:** Occlusal plane between occlusal plane and cervical line (neck) of second molar.
              - **Position C:** Occlusal plane below cervical line of second molar.
            - **Ramus Relationship (Space Availability):**
              - **Class I:** Distance between distal second molar and anterior ramus > third molar crown width (sufficient space).
              - **Class II:** Space is less than crown width (reduced space).
              - **Class III:** Complete absence of space / embedded in ramus (no space).
            """
        )

    output = io.BytesIO()
    overlay.save(output, format='PNG')
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button('Download automatic angle overlay', output.getvalue(), 'automatic_molar_angles.png', 'image/png')
    with col_d2:
        st.download_button('Download automatic estimates CSV', pd.DataFrame(display_rows).to_csv(index=False), 'automatic_angles.csv', 'text/csv')
    with col_d3:
        st.download_button('Download automatic axis details', json.dumps({'image_sha256': image_id, 'measurements': rows, 'axes': axes}, indent=2), 'automatic_axes.json', 'application/json')

    return rows
