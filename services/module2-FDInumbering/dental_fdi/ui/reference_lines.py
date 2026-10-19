"""Manual reference guides and tooth-axis angle overlays."""
import base64
import io
import json
import math

import streamlit as st
from PIL import ImageDraw

from dental_fdi.analysis.manual_geometry import line_angle, validate_points
from dental_fdi.ui.molar_review import axis_picker
from dental_fdi.reporting.prediction_exports import render_detection_overlay


def measure_tooth_reference(points, width, height, reference=0):
    validate_points(points[:6], 6, width, height)
    if reference not in (0, 2):
        raise ValueError('Select horizontal or vertical reference')
    return {
        'axis_to_horizontal_degrees': line_angle(points[:2], points[4:6]),
        'axis_to_vertical_degrees': line_angle(points[2:4], points[4:6]),
        'guides_degrees': line_angle(points[:2], points[2:4]),
        'selected_angle_degrees': line_angle(points[reference:reference + 2], points[4:6]),
        'method': 'Acute geometric line angle in image pixels; not a Winter classification',
    }


def draw_angle_wedge(draw, points, reference, size, fill='#69cf20'):
    """Draw the acute sector at the intersection of the infinite selected lines."""
    a, b = points[reference:reference + 2]
    c, d = points[4:6]
    u = (b[0] - a[0], b[1] - a[1])
    v = (d[0] - c[0], d[1] - c[1])
    cross = u[0] * v[1] - u[1] * v[0]
    if abs(cross) < 1e-8 * math.hypot(*u) * math.hypot(*v):
        return False
    t = ((c[0] - a[0]) * v[1] - (c[1] - a[1]) * v[0]) / cross
    center = (a[0] + t * u[0], a[1] + t * u[1])
    if not (0 <= center[0] < size[0] and 0 <= center[1] < size[1]):
        return False
    start = math.atan2(u[1], u[0])
    delta = (math.atan2(v[1], v[0]) - start + math.pi / 2) % math.pi - math.pi / 2
    # Choose the sector above the guide, independent of endpoint click order.
    if math.sin(start + delta / 2) > 0:
        start += math.pi
    radius = max(8, min(size) * .055)
    arc = [(center[0] + radius * math.cos(start + delta * i / 40),
            center[1] + radius * math.sin(start + delta * i / 40)) for i in range(41)]
    draw.polygon([center] + arc, fill=fill)
    draw.line([center] + arc + [center], fill='#202020', width=2)
    return True


def manual_axis_classification(points,fdi,width,height):
    from dental_fdi.analysis.dental_analysis import measure_axes
    from dental_fdi.analysis.manual_geometry import requested_winter_category
    validate_points(points[4:8],4,width,height)
    angle=measure_axes(points[4:8],fdi,width,height)
    return angle,requested_winter_category(angle)


def signed_horizontal_angle(points,width,height):
    """User screen convention: upper ray left negative, right positive; endpoint invariant."""
    validate_points(points[4:6],2,width,height)
    validate_points(points[:2],2,width,height)
    magnitude=line_angle(points[:2],points[4:6])
    lower,upper=sorted(points[4:6],key=lambda p:(-p[1],p[0]))
    dx=upper[0]-lower[0]
    return -magnitude if dx<0 else magnitude


def manual_summary(fdi,points,width,height):
    result={'FDI':fdi,'classification':'Unavailable','angle_to_second_molar_degrees':None,
            'screen_tilt_degrees':None,'horizontal_angle_degrees':None,
            'signed_horizontal_degrees':None,'screen_category':'Unavailable'}
    try:
        validate_points(points[4:6],2,width,height)
        from dental_fdi.analysis.manual_geometry import screen_axis_tilt
        result['screen_tilt_degrees']=screen_axis_tilt(points[4:6])
        validate_points(points[:2],2,width,height)
        result['horizontal_angle_degrees']=line_angle(points[:2],points[4:6])
        from dental_fdi.analysis.manual_geometry import requested_winter_category
        signed=signed_horizontal_angle(points,width,height)
        result.update(signed_horizontal_degrees=signed,screen_category=requested_winter_category(signed))
    except (ValueError,TypeError):pass
    try:
        angle,category=manual_axis_classification(points,fdi,width,height)
        result.update(angle_to_second_molar_degrees=angle,classification=category)
    except (ValueError,TypeError):pass
    return result


def render_manual_measurements(image, boxes, drawings):
    """Render only supplied user lines; never estimate or seed any axes."""
    overlay=render_detection_overlay(image,boxes,show_conf=False)
    draw=ImageDraw.Draw(overlay)
    results=[]
    for fdi,points in drawings.items():
        result={'FDI':fdi,'points':points,'source':'Manual drawing only',
                'classification':'Unavailable: requires reviewed adjacent-molar axis'}
        try:
            result.update(measure_tooth_reference(points,*image.size))
            draw_angle_wedge(draw,points,0,image.size,fill='#ffdb35')
        except (ValueError,TypeError):
            result['selected_angle_degrees']=None
        try:
            signed,category=manual_axis_classification(points,fdi,*image.size)
            result.update(angle_to_second_molar_degrees=signed,classification=category)
        except (ValueError,TypeError):
            pass
        for a,b,color in [(0,1,'#69cf20'),(2,3,'#ef3045'),(4,5,'#ffdb35'),(6,7,'#00d2ff')]:
            if len(points)>b and points[a] is not None and points[b] is not None:
                draw.line([tuple(points[a]),tuple(points[b])],fill=color,width=max(3,image.width//500))
        result.update(manual_summary(fdi,points,*image.size))
        if result['selected_angle_degrees'] is not None:
            x,y=points[4]
            label=f"{fdi}: screen {result['signed_horizontal_degrees']:+.2f} deg | {result['screen_category']} (screen-based)"
            if result.get('angle_to_second_molar_degrees') is not None:
                label+=f" {result['angle_to_second_molar_degrees']:+.1f} deg"
            bounds=draw.textbbox((0,0),label)
            x=max(0,min(x,image.width-(bounds[2]-bounds[0])-8))
            y=max(0,min(y,image.height-20))
            draw.rectangle((x,y,x+bounds[2]+6,y+18),fill='#101820')
            draw.text((x+3,y+3),label,fill='#ffdb35')
        result.update(manual_summary(fdi,points,*image.size))
        results.append(result)
    from PIL import ImageOps, ImageFont
    import textwrap
    detected={b['fdi'] for b in boxes}
    by_fdi={r['FDI']:r for r in results}
    lines=['Manual screen-based categories: upper-left negative / upper-right positive. Not an anatomical diagnosis.']
    for fdi in ('18','28','38','48'):
        r=by_fdi.get(fdi,{})
        category=r.get('screen_category','Not drawn' if fdi in detected else 'Not detected')
        def fmt(value):return f'{value:+.2f} deg' if value is not None else 'unavailable'
        lines.append(f"FDI {fdi}: {category} (screen-based) | signed Y angle: {fmt(r.get('signed_horizontal_degrees'))} | anatomical: {r.get('classification','Unavailable')} {fmt(r.get('angle_to_second_molar_degrees'))}")
    lines.append('Rounded ranges: vertical -10..10; mesial 11..79; horizontal 80..100; distal -79..-11; other outside. Draw both axes root to crown.')
    font_size=max(10,min(24,image.width//85))
    try:font=ImageFont.truetype('arial.ttf',font_size)
    except OSError:font=ImageFont.load_default()
    wrapped=[part for line in lines for part in textwrap.wrap(line,width=max(24,int((image.width-16)/(font_size*.58))))]
    line_height=font_size+7
    overlay=ImageOps.expand(overlay,border=(0,0,0,line_height*len(wrapped)+12),fill='#101820')
    draw=ImageDraw.Draw(overlay)
    for i,line in enumerate(wrapped):draw.text((8,image.height+6+i*line_height),line,font=font,fill='#ffdb35')
    return overlay,results


def combined_manual_export(image,identity,boxes):
    import hashlib
    drawings={}
    for box in boxes:
        fdi=box['fdi']
        if fdi not in ('18','28','38','48'):continue
        signature=hashlib.sha256(json.dumps(box['xyxy']).encode()).hexdigest()[:12]
        key='manual_reference_v2_'+identity+fdi+signature
        drawings[fdi]=st.session_state.get(key,[])
    overlay,results=render_manual_measurements(image,boxes,drawings)
    st.subheader('All manual wisdom-tooth drawings')
    st.image(overlay,width='stretch',caption='All saved manual drawings and measurements. Edit on the canvas above; downloads include the summary below the X-ray.')
    buffer=io.BytesIO();overlay.save(buffer,format='PNG')
    st.download_button('Download all manual wisdom-tooth drawings',buffer.getvalue(),
                       'manual_wisdom_angles.png','image/png',key='manual_all_png_'+identity,on_click='ignore')
    st.download_button('Download manual measurements (JSON)',json.dumps({'image_sha256':identity,'measurements':results},indent=2),
                       'manual_wisdom_angles.json','application/json',key='manual_all_json_'+identity,on_click='ignore')


def reference_line_review(image, identity, boxes=()):
    with st.container():
        st.caption('Drag to draw a straight line. Choose Adjust / move line to edit an endpoint or move a line. '
                   'The result is an acute image angle (0 to 90 degrees). Guides may be tilted. '
                   'Draw the optional cyan second-molar axis to calculate the anatomical tooth-to-tooth angle.')
        detected = [fdi for fdi in ('38','48','18','28') if any(b['fdi']==fdi for b in boxes)]
        if not detected:
            st.info('No wisdom tooth is detected. Add or correct a tooth box in Teeth to enable its angle measurement.')
            return
        selection_key='active_reference_tooth_'+identity
        pending=st.session_state.pop('pending_reference_tooth_'+identity,None)
        if pending in detected:
            st.session_state[selection_key]=pending
        if st.session_state.get(selection_key) not in detected:
            st.session_state[selection_key]=detected[0]
        tooth = st.radio('Choose the wisdom tooth to draw or adjust', detected,
                         horizontal=True, key=selection_key,
                         format_func=lambda fdi: 'Tooth '+fdi)
        st.caption(f'Editing tooth {tooth}. Switch using the tooth buttons or click another wisdom-tooth box. Your drawings are saved separately for each tooth.')
        import hashlib
        box = next(b for b in boxes if b['fdi']==tooth)
        signature = hashlib.sha256(json.dumps(box['xyxy']).encode()).hexdigest()[:12]
        key = 'manual_reference_v2_' + identity + tooth + signature
        st.session_state.setdefault(key, [])
        st.caption('Draw Y (green), X (red), and the wisdom axis (yellow). For anatomical classification, add the neighboring second-molar axis (cyan). Draw tooth axes root to crown.')
        revision = st.session_state.get(key + '_revision', 0)
        component_id = key + str(revision)
        reference_name = 'Horizontal Y (green)'
        reference = 0
        st.caption('Vertical X: up + / down -. Horizontal Y: left - / right +. Yellow shading shows the auxiliary angle to Y. For classification draw both tooth axes from ROOT to CROWN; mesial + / distal - is anatomical.')
        show_boxes = st.checkbox('Show FDI boxes on measurement image', value=True, key=key + '_boxes')
        preview = render_detection_overlay(image, boxes, show_conf=False) if show_boxes else image.copy()
        background=ImageDraw.Draw(preview)
        for other_box in boxes:
            other=other_box['fdi']
            if other not in detected or other==tooth:continue
            other_signature=hashlib.sha256(json.dumps(other_box['xyxy']).encode()).hexdigest()[:12]
            saved=st.session_state.get('manual_reference_v2_'+identity+other+other_signature,[])
            for a,b,color in [(0,1,'#467c32'),(2,3,'#99404a'),(4,5,'#a89536'),(6,7,'#337788')]:
                if len(saved)>b and saved[a] is not None and saved[b] is not None:
                    background.line([tuple(saved[a]),tuple(saved[b])],fill=color,width=2)
        points = st.session_state.setdefault(key, [])
        buffer = io.BytesIO()
        preview.save(buffer, format='PNG')
        live_drawings={}
        for item in boxes:
            if item['fdi'] not in ('18','28','38','48'):continue
            sig=hashlib.sha256(json.dumps(item['xyxy']).encode()).hexdigest()[:12]
            live_drawings[item['fdi']]=st.session_state.get('manual_reference_v2_'+identity+item['fdi']+sig,[])
        value = axis_picker(image='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
            points=points, identity=component_id, key=component_id + '_picker', line_mode=True, active_tooth=tooth, manual_summary=True, manual_drawings=live_drawings,
            line_labels=['Horizontal Y (green)', 'Vertical X (red)', 'Wisdom axis: root to crown (yellow)', 'Second-molar axis: root to crown (cyan)'],
            tooth_boxes=[{'fdi':b['fdi'],'xyxy':b['xyxy']} for b in boxes if b['fdi'] in detected],
            labels=['Y left / start', 'Y right / end', 'X upper / start', 'X lower / end',
                    'Wisdom root', 'Wisdom crown', 'Second-molar root', 'Second-molar crown'],
            colors=['#69cf20'] * 2 + ['#ef3045'] * 2 + ['#ffdb35'] * 2 + ['#00d2ff'] * 2,
            segments=[[0, 1, '#69cf20'], [2, 3, '#ef3045'], [4, 5, '#ffdb35'], [6, 7, '#00d2ff']], default=None)
        if value and value.get('identity') == component_id:
            points = value['points']
            st.session_state[key] = points
            switch=value.get('switch_tooth')
            if switch in detected and switch!=tooth:
                st.session_state['pending_reference_tooth_'+identity]=switch
                st.rerun()
        combined_manual_export(image, identity, boxes)
        if len(points) < 6 or any(p is None for p in points[:6]):
            st.info('Draw the green guide, red guide and yellow tooth axis to calculate the angle. Clear yellow axis only keeps both guides.')
            return
        try:
            result = measure_tooth_reference(points, *image.size, reference)
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            return
        try:
            signed,category=manual_axis_classification(points,tooth,*image.size)
            result.update(angle_to_second_molar_degrees=signed,classification=category)
            st.metric(f'Manual tooth-axis classification: {category}',f'{signed:+.1f} deg')
        except (ValueError,TypeError):
            st.caption('Draw the cyan second-molar axis from root to crown to calculate manual classification.')
        summary=manual_summary(tooth,points,*image.size)
        result.update(summary)
        st.metric(f"Screen-based category: {summary['screen_category']}",f"{summary['signed_horizontal_degrees']:+.2f} deg")
        angle = result['selected_angle_degrees']
        st.metric(f'Tooth {tooth}: axis to {reference_name}', f'{angle:.2f}°')
        st.caption(f"Axis to horizontal: {result['axis_to_horizontal_degrees']:.2f}° | "
                   f"Axis to vertical: {result['axis_to_vertical_degrees']:.2f}° | "
                   f"Between guides: {result['guides_degrees']:.2f}°")
        with st.expander('Selected tooth preview and PNG download'):
            overlay = preview.copy()
            draw = ImageDraw.Draw(overlay)
            wedge_visible = draw_angle_wedge(draw, points, reference, image.size, fill='#ffdb35')
            for a, b, color in [(0, 1, '#69cf20'), (2, 3, '#ef3045'), (4, 5, '#ffdb35'), (6,7,'#00d2ff')]:
                if len(points)<=b or points[a] is None or points[b] is None:continue
                draw.line([tuple(points[a]), tuple(points[b])], fill=color, width=max(3, image.width // 600))
            label = f'Tooth {tooth}: {angle:.2f} deg to {reference_name}'
            draw.rectangle((0, 0, image.width, 28), fill='#101820')
            draw.text((8, 8), label, fill='#ffdb35')
            st.image(overlay, width='stretch', caption='Yellow shading: angle to horizontal Y. Red: vertical X. Yellow line: wisdom axis. Cyan: manually drawn second-molar axis.')
            if not wedge_visible:
                st.caption('The lines are parallel or their intersection is outside the image; the numerical angle remains available.')
            output = io.BytesIO()
            overlay.save(output, format='PNG')
            st.download_button('Download angle overlay', output.getvalue(), f'FDI{tooth}_reference_angle.png',
                               'image/png', key=key + '_png', on_click='ignore')
        report = {'image_sha256': identity, 'image_size': list(image.size), 'fdi': tooth,
                  'reference': reference_name, 'points': points, **result}
        st.download_button('Download angle measurements', json.dumps(report, indent=2),
                           f'FDI{tooth}_reference_angle.json', 'application/json', key=key + '_json', on_click='ignore')
