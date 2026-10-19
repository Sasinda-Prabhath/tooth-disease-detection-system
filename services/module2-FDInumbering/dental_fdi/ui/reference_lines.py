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
    validate_points(points, 6, width, height)
    if reference not in (0, 2):
        raise ValueError('Select horizontal or vertical reference')
    return {
        'axis_to_horizontal_degrees': line_angle(points[:2], points[4:6]),
        'axis_to_vertical_degrees': line_angle(points[2:4], points[4:6]),
        'guides_degrees': line_angle(points[:2], points[2:4]),
        'selected_angle_degrees': line_angle(points[reference:reference + 2], points[4:6]),
        'method': 'Acute geometric line angle in image pixels; not a Winter classification',
    }


def draw_angle_wedge(draw, points, reference, size):
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
    draw.polygon([center] + arc, fill='#69cf20')
    draw.line([center] + arc + [center], fill='#202020', width=2)
    return True


def reference_line_review(image, identity, boxes=()):
    with st.container():
        st.caption('Draw two endpoints for each guide, then two along the tooth axis. '
                   'The result is an acute image angle (0 to 90 degrees). Guides may be tilted. '
                   'Use the separate reviewed molar-axis tool for Winter classification.')
        tooth = st.selectbox('Tooth for reference measurement', ['38', '48', '18', '28'],
                            key='reference_tooth_' + identity)
        key = 'tooth_reference_' + identity + tooth
        reference_name = 'Horizontal (green)'
        reference = 0
        st.caption('Measured region: the green wedge above the horizontal guide, between the guide and tooth axis.')
        show_boxes = st.checkbox('Show FDI boxes on measurement image', value=True, key=key + '_boxes')
        preview = render_detection_overlay(image, boxes, show_conf=False) if show_boxes else image.copy()
        points = st.session_state.setdefault(key, [])
        buffer = io.BytesIO()
        preview.save(buffer, format='PNG')
        value = axis_picker(image='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
            points=points, identity=key, key=key + '_picker',
            labels=['Horizontal start', 'Horizontal end', 'Vertical start', 'Vertical end',
                    'Tooth axis start', 'Tooth axis end'],
            colors=['#69cf20'] * 2 + ['#ef3045'] * 2 + ['#ffdb35'] * 2,
            segments=[[0, 1, '#69cf20'], [2, 3, '#ef3045'], [4, 5, '#ffdb35']], default=None)
        if value and value.get('identity') == key:
            points = value['points']
            st.session_state[key] = points
        if len(points) != 6 or any(p is None for p in points):
            st.info('Place all six endpoints to calculate the angle. Use the landmark dropdown to edit any point.')
            return
        try:
            result = measure_tooth_reference(points, *image.size, reference)
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            return
        angle = result['selected_angle_degrees']
        st.metric(f'Tooth {tooth}: axis to {reference_name}', f'{angle:.2f}°')
        st.caption(f"Axis to horizontal: {result['axis_to_horizontal_degrees']:.2f}° | "
                   f"Axis to vertical: {result['axis_to_vertical_degrees']:.2f}° | "
                   f"Between guides: {result['guides_degrees']:.2f}°")
        overlay = preview.copy()
        draw = ImageDraw.Draw(overlay)
        wedge_visible = draw_angle_wedge(draw, points, reference, image.size)
        for a, b, color in [(0, 1, '#69cf20'), (2, 3, '#ef3045'), (4, 5, '#ffdb35')]:
            draw.line([tuple(points[a]), tuple(points[b])], fill=color, width=max(3, image.width // 600))
        label = f'Tooth {tooth}: {angle:.2f} deg to {reference_name}'
        draw.rectangle((0, 0, image.width, 28), fill='#101820')
        draw.text((8, 8), label, fill='#ffdb35')
        st.image(overlay, width='stretch', caption='Green wedge: measured angle to the horizontal guide. Red: vertical guide. Yellow: tooth axis.')
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
