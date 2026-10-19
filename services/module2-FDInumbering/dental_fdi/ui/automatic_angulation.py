"""Automatic overlays from an optional landmark model or unsupervised image axes."""
from dental_fdi.paths import PROJECT_ROOT
import io
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from dental_fdi.analysis.image_angulation import estimate_image_angles
from dental_fdi.analysis.pose_pipeline import validate_axis_model, pose_rows, automatic_angles

ROOT = PROJECT_ROOT

@st.cache_resource
def load_axis_model(path, modified_ns):
    model = YOLO(path)
    validate_axis_model(model, path)
    return model

@st.cache_data(max_entries=8, show_spinner=False)
def estimate_pixels(image_bytes, boxes):
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
            st.caption('Automatic root/crown landmark-model estimates.')
        except Exception as exc:
            st.error(f'Landmark model unavailable: {exc}. Using radiographic image-axis estimates.')

    if rows is None:
        with st.spinner('Calculating 18, 28, 38, 48 longitudinal molar axes & angulations...'):
            rows, axes = estimate_pixels(image_buffer.getvalue(), boxes)

    # Render summary metric cards for 18, 28, 38, 48
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    for idx, r in enumerate(rows):
        fdi = r['FDI']
        angle = r.get('Angle (degrees)', r.get('Estimated angle (degrees)'))
        cat = r.get('Winter Classification', r.get('Classification', r.get('Provisional category', 'Unavailable')))
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

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font_size = max(18, int(image.width / 75))
    try:
        font = ImageFont.truetype('arialbd.ttf', font_size)
    except OSError:
        font = ImageFont.load_default()

    box_map = {b['fdi']: b for b in boxes}
    for index, row in enumerate(rows):
        fdi = row['FDI']
        box = box_map.get(fdi)
        value = row.get('Angle (degrees)', row.get('Estimated angle (degrees)'))
        category = row.get('Winter Classification', row.get('Classification', ''))
        
        # Bounding box around third molar
        if box:
            draw.rectangle(box['xyxy'], outline='#ffb300', width=max(2, image.width // 600))

        # Draw axis lines: Amber for 3rd molar, Cyan for 2nd molar
        if row.get('points'):
            for offset, color in ((0, '#ffb300'), (2, '#00e5ff')):
                pts = row['points'][offset:offset + 2]
                draw.line([tuple(p) for p in pts], fill=color, width=max(3, image.width // 450))
                for x, y in pts:
                    r = max(4, image.width // 450)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

        # Readable result labels in corners
        if value is not None:
            if category and category not in ('Unavailable', 'Absent / Not detected'):
                label = f"{fdi}: {value:+.1f}° ({category})"
            else:
                label = f"{fdi}: {value:+.1f}°"
        else:
            if box is None:
                label = f"{fdi}: not detected in arch"
            else:
                label = f"{fdi}: angle pending reference"

        text_box = draw.textbbox((0, 0), label, font=font)
        tw = text_box[2]
        th = text_box[3] + 12
        left = fdi[0] in '14'
        upper = fdi[0] in '12'
        x = 14 if left else max(14, image.width - tw - 24)
        y = 14 if upper else max(14, image.height - th - 18)
        draw.rectangle([x - 6, y - 4, x + tw + 10, y + th], fill='#101820', outline='#ffb300', width=1)
        draw.text((x, y), label, fill='#ffb300', font=font)

    st.image(
        overlay,
        width='stretch',
        caption='Amber line: third molar (18, 28, 38, 48) longitudinal axis. Cyan line: adjacent second molar (17, 27, 37, 47) reference axis.'
    )

    # Detailed Table View
    display_rows = []
    for r in rows:
        row_dict = {
            'FDI': r.get('FDI'),
            'Reference Molar': r.get('Reference'),
            'Angulation (°)': f"{r.get('Angle (degrees)'):+.1f}°" if r.get('Angle (degrees)') is not None else "None",
            "Winter's Classification": r.get('Winter Classification', r.get('Classification', 'Unavailable')),
            'Pell & Gregory Classification': r.get('Pell & Gregory', 'N/A'),
            'Impaction Status': r.get('Impaction', 'Not determined'),
            'Status': r.get('Status', 'N/A'),
        }
        display_rows.append(row_dict)

    with st.expander('Detailed automatic measurements'):
        st.dataframe(pd.DataFrame(display_rows), hide_index=True, width='stretch')

    # Medical Reference & Classification Rules Expander
    with st.expander("Medical Literature & Classification Methodology (Patel et al., 2024)"):
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
