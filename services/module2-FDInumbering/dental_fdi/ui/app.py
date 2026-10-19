from dental_fdi.paths import PROJECT_ROOT, UI_ASSETS, DEFAULT_ENSEMBLE
import os
import io
import time
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image, ImageOps
from dental_fdi.analysis.dental_analysis import select_detections
from dental_fdi.ui.molar_review import review_detections, molar_review
from dental_fdi.ui.automatic_angulation import automatic_angulation_view
from dental_fdi.ui.batch_review import batch_review
from dental_fdi.ui.tooth_presence import presence_review
from dental_fdi.ui.reference_lines import reference_line_review
from dental_fdi.ui.periodontal_review import periodontal_review
from dental_fdi.reporting.training_examples import export_fdi_example
from dental_fdi.analysis.tooth_inventory import FDI_ARCHES
from dental_fdi.reporting.prediction_exports import render_detection_overlay
from dental_fdi.inference.detection_inference import infer_detections
from dental_fdi.inference.ensemble import load_prediction_model
from ultralytics import YOLO
from dental_fdi.data.dataset_checks import check_dataset
from dental_fdi.imaging.image_enhancement import enhance_dental_xray, get_enhancement_presets, isolate_detected_teeth


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dental X-ray Review",
    layout="wide",
    initial_sidebar_state="auto"
)

# Shared presentation styles; analysis logic remains independent.
st.markdown("<style>" + (UI_ASSETS / "ui.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------
# FDI Tooth Class Mapping & Anatomical Names
# ---------------------------------------------------------
FDI_NAMES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48"
]

TOOTH_ANATOMY_MAP = {
    "1": "Central Incisor",
    "2": "Lateral Incisor",
    "3": "Canine (Cuspid)",
    "4": "First Premolar (Bicuspid)",
    "5": "Second Premolar (Bicuspid)",
    "6": "First Molar (6-yr Molar)",
    "7": "Second Molar (12-yr Molar)",
    "8": "Third Molar (Wisdom Tooth)"
}

QUADRANT_NAMES = {
    "1": "Upper Right (Maxilla Q1)",
    "2": "Upper Left (Maxilla Q2)",
    "3": "Lower Left (Mandible Q3)",
    "4": "Lower Right (Mandible Q4)"
}

def get_tooth_description(fdi_code):
    if len(fdi_code) != 2:
        return fdi_code
    quad, pos = fdi_code[0], fdi_code[1]
    quad_desc = QUADRANT_NAMES.get(quad, "")
    pos_desc = TOOTH_ANATOMY_MAP.get(pos, "")
    return f"{quad_desc} - {pos_desc}"

def is_upper_fdi(fdi_code):
    return fdi_code.startswith('1') or fdi_code.startswith('2')

# ---------------------------------------------------------
# Model Loading Helper with Cache
# ---------------------------------------------------------
PROJECT_DIR = PROJECT_ROOT
CUSTOM_MODEL_PATH = os.environ.get(
    "FDI_MODEL_PATH", str(PROJECT_DIR / "runs" / "detect" / "train-2" / "weights" / "best.pt")
)

@st.cache_resource
def load_model(weights_path, modified_ns):
    model = load_prediction_model(weights_path)
    if set(map(str, model.names.values())) != set(FDI_NAMES):
        raise ValueError("The selected model does not contain the required 32 FDI tooth classes.")
    return model, True

@st.cache_data(show_spinner=False, max_entries=8)
def predict_rows(image_bytes, weights_path, modified_ns, confidence, iou, resolution, _model, recovery=False):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return infer_detections(_model, image, confidence, iou, resolution, recovery)

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">Dental X-ray Review</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Review panoramic X-rays, identify teeth using FDI numbering, and export your findings.</div>', unsafe_allow_html=True)

model_options = {}
if DEFAULT_ENSEMBLE.is_file():
    model_options['Three-model fusion: YOLO26 + EfficientNet + ResNet50'] = str(DEFAULT_ENSEMBLE)
model_options['YOLO26 detector'] = CUSTOM_MODEL_PATH
for bundle in sorted((PROJECT_DIR / 'runs' / 'hybrid').glob('*/*.fdi.zip')):
    model_options[f"Hybrid: {bundle.parent.name} / {bundle.name.removesuffix('.fdi.zip')}"] = str(bundle)
configured_hybrid = os.environ.get('FDI_HYBRID_PATH')
if configured_hybrid:
    model_options['Configured hybrid model'] = configured_hybrid
model_label = st.sidebar.selectbox('Prediction model', list(model_options),
    index=len(model_options) - 1 if configured_hybrid else
          list(model_options).index('YOLO26 detector') if os.environ.get('FDI_MODEL_PATH') else 0)
CUSTOM_MODEL_PATH = model_options[model_label]

# Check model availability
if not Path(CUSTOM_MODEL_PATH).is_file():
    st.error("Trained FDI model not found. Set FDI_MODEL_PATH to the trained best.pt file and restart the app.")
    st.stop()
try:
    model, is_custom = load_model(CUSTOM_MODEL_PATH, Path(CUSTOM_MODEL_PATH).stat().st_mtime_ns)
except Exception as exc:
    st.error(f"Unable to load the FDI model: {exc}")
    st.stop()

st.warning("Experimental model: tooth detection and FDI numbering have not passed independent validation. Review every prediction.")
if hasattr(model, 'manifest'):
    if model.manifest.get('version') == 2:
        st.caption('All three models active: YOLO26 locates teeth; EfficientNet-B0 and ResNet50 '
                   'both contribute to every final FDI label and score. Scores are uncalibrated.')
    else:
        st.caption('Two-model hybrid: YOLO26 + ' + model.manifest['architecture'] + '. Experimental estimates.')
    comparison_file = Path(CUSTOM_MODEL_PATH).with_suffix('.comparison.json')
    if comparison_file.is_file() and comparison_file.stat().st_mtime_ns >= Path(CUSTOM_MODEL_PATH).stat().st_mtime_ns:
        comparison = json.loads(comparison_file.read_text())
        if not comparison.get('hybrid_improves_f1', False):
            st.info('This model did not improve detector F1 in the saved development evaluation. '
                    'The detector-only option remains available for comparison.')

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Analysis settings")
def inspect_weak_candidates():
    st.session_state["confidence_percent"] = 1


conf_percent = st.sidebar.slider(
    "Minimum detection confidence (%)", min_value=1, max_value=100, value=35,
    step=1, format="%d%%", key="confidence_percent",
)
conf_thresh = conf_percent / 100.0
st.sidebar.caption("Default: 35%. Lower thresholds include weaker candidates and more false positives; confidence is not validated accuracy.")

with st.sidebar.expander("Advanced analysis settings"):
    iou_thresh = st.slider("Overlap threshold", min_value=0.05, max_value=1.00, value=0.50, step=0.05)
    imgsz = st.selectbox("Analysis resolution", [1024, 640, 1280], index=0)

recovery_mode = st.sidebar.checkbox('Search for missed teeth at two resolutions', value=False)
st.sidebar.caption('Runs an additional full-image prediction; slower and may add false positives. No missing tooth numbers are invented.')

st.sidebar.markdown("---")
st.sidebar.header("Image display")
show_boxes = st.sidebar.checkbox("Show tooth outlines", value=True)
show_labels = st.sidebar.checkbox("Show tooth numbers", value=True)
show_conf = st.sidebar.checkbox("Show confidence percentages", value=True)
line_width = st.sidebar.slider("Outline thickness", 1, 5, 3)

# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab1, batch_tab, tab2, tab3 = st.tabs(["X-ray review", "Bulk predictions", "Administration", "FDI guide"])

with batch_tab:
    batch_review(
        lambda data: predict_rows(data, CUSTOM_MODEL_PATH, Path(CUSTOM_MODEL_PATH).stat().st_mtime_ns,
                                  conf_thresh, iou_thresh, imgsz, model, recovery_mode),
        {"path": CUSTOM_MODEL_PATH, "modified_ns": Path(CUSTOM_MODEL_PATH).stat().st_mtime_ns,
         "two_resolution_search": recovery_mode},
        conf_thresh, iou_thresh, imgsz,
        {"show_boxes": show_boxes, "show_labels": show_labels, "show_conf": show_conf, "line_width": line_width},
    )

# ---------------------------------------------------------
# TAB 1: OPG X-Ray Analysis & Detection
# ---------------------------------------------------------
with tab1:
    st.markdown('<div class="workflow"><span>1. Upload an X-ray</span><span>2. Review tooth findings</span><span>3. Download results</span></div>', unsafe_allow_html=True)
    st.subheader("Start a review")
    col_input, col_preset = st.columns([2, 1])

    with col_input:
        uploaded_file = st.file_uploader(
            "Upload a panoramic X-ray",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
        )

    with col_preset:
        st.markdown("**Or select a sample X-ray:**")
        sample_dir = PROJECT_DIR / "samples"
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(sample_dir) else []
        selected_sample = st.selectbox("Sample Images", ["None"] + sample_files)
        st.caption("Bundled sample_opg images are synthetic drawings, not real panoramic X-rays.")

    image_to_process = None

    if uploaded_file is not None:
        image_to_process = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    elif selected_sample != "None":
        image_to_process = Image.open(os.path.join(sample_dir, selected_sample)).convert("RGB")

    if image_to_process is not None:
        mirrored = st.checkbox("Input is mirrored (patient right appears on image right)", value=False)
        if mirrored:
            image_to_process = ImageOps.mirror(image_to_process)
        st.caption("Analysis display: patient right on image left. Verify the radiograph side marker; all coordinates refer to this display.")
        
        # ---------------------------------------------------------
        # Dental Radiograph Preprocessing & Enhancement Controls
        # ---------------------------------------------------------
        with st.expander("Image adjustment", expanded=False):
            st.markdown(
                """
                Adjust contrast and sharpness, then compare the result with the original image.
                These settings change the image used for detection; review the resulting tooth labels.
                """
            )
            col_en1, col_en2, col_en3 = st.columns([1.5, 2, 1.5])
            with col_en1:
                enable_enhancement = st.checkbox(
                    "Adjust image contrast and detail",
                    value=True,
                    help="Applies CLAHE contrast equalisation and multi-scale edge sharpening to boost tooth detection."
                )
            with col_en2:
                preset_options = get_enhancement_presets()
                selected_preset_name = st.selectbox(
                    "Enhancement Preset",
                    list(preset_options.keys()),
                    index=0,
                    disabled=not enable_enhancement
                )
                preset_key = preset_options[selected_preset_name]
            with col_en3:
                compare_mode = st.radio(
                    "Display View",
                    ["Enhanced", "Original", "Side-by-Side"],
                    horizontal=True,
                    disabled=not enable_enhancement
                )

            custom_clip, custom_sharp, custom_gamma, bg_dim = 2.5, 1.3, 1.0, 0.0
            if enable_enhancement and preset_key == "teeth_isolated_black":
                st.info("**Teeth Isolation Mode:** Non-tooth background (cervical spine, skull base, air cavities) is suppressed to solid black, isolating the teeth on a dark background.")
                bg_dim_pct = st.slider("Background Visibility", 0, 30, 0, 5, format="%d%% visible", help="0% = 100% pure black background. 10-20% = subtle shadow of the jaw.")
                bg_dim = bg_dim_pct / 100.0
            elif enable_enhancement and preset_key == "custom":
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    custom_clip = st.slider("CLAHE Contrast Clip Limit", 1.0, 5.0, 2.5, 0.5)
                with sc2:
                    custom_sharp = st.slider("Sharpen Strength", 0.0, 3.0, 1.3, 0.1)
                with sc3:
                    custom_gamma = st.slider("Gamma (Shadow Lift)", 0.6, 1.4, 0.95, 0.05)

        if enable_enhancement:
            enhanced_image = enhance_dental_xray(
                image_to_process,
                preset=preset_key,
                clip_limit=custom_clip,
                sharpen_strength=custom_sharp,
                gamma=custom_gamma,
                background_dim=bg_dim
            )
            detection_input_image = enhanced_image
        else:
            enhanced_image = image_to_process
            detection_input_image = image_to_process

        image_id = hashlib.sha256(str(detection_input_image.size).encode() + detection_input_image.tobytes()).hexdigest()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original and adjusted image")
            if enable_enhancement and compare_mode == "Side-by-Side":
                st.image(
                    [image_to_process, enhanced_image],
                    caption=["Original Raw Radiograph", "Enhanced Radiograph (Fed to AI)"],
                    width="stretch"
                )
            elif enable_enhancement and compare_mode == "Original":
                st.image(image_to_process, caption="Original Raw Radiograph", width="stretch")
            elif enable_enhancement:
                st.image(enhanced_image, caption=f"Enhanced Radiograph ({selected_preset_name})", width="stretch")
            else:
                st.image(image_to_process, caption="Original Raw Radiograph", width="stretch")

            buf_enh = io.BytesIO()
            detection_input_image.save(buf_enh, format="PNG")
            st.download_button(
                "Download Enhanced X-Ray Image",
                buf_enh.getvalue(),
                "enhanced_dental_opg.png",
                "image/png"
            )

        with col2:
            st.subheader("Tooth identification")
            
            with st.spinner("Analyzing your X-ray..."):
                start_time = time.time()
                
                W_img, H_img = detection_input_image.size

                image_buffer = io.BytesIO()
                detection_input_image.save(image_buffer, format="PNG")
                rows = predict_rows(image_buffer.getvalue(), CUSTOM_MODEL_PATH,
                                    Path(CUSTOM_MODEL_PATH).stat().st_mtime_ns,
                                    conf_thresh, iou_thresh, imgsz, model, recovery_mode)
                infer_time = (time.time() - start_time) * 1000
                box_data = select_detections(rows, W_img, H_img, conf_thresh)

        detected_teeth = []
        st.markdown('### Review your X-ray')
        st.caption('Check tooth numbers first, then open angle or bone tools if needed. Downloads are in Reports.')
        teeth_tab, angles_tab, bone_tab, reports_tab = st.tabs([
            '1. Teeth', '2. Angles', '3. Bone levels', '4. Reports & advanced'])
        with teeth_tab:
            summary_panel = st.container()
            review_id = image_id + hashlib.sha256(repr(box_data).encode()).hexdigest()[:16]
            box_data = review_detections(box_data, image_to_process, review_id)
            box_data, confirmed_missing, inventory = presence_review(box_data, image_id)

        with reports_tab:
            contribution_rows = []
            for box in box_data:
                contributions = box.get('model_contributions')
                if not contributions:
                    continue
                contribution_rows.append({
                    'Final FDI': box['fdi'], 'Final score': box['conf'],
                    'YOLO26 FDI': contributions['yolo26']['fdi'],
                    'YOLO26 score': contributions['yolo26']['score'],
                    'EfficientNet FDI': contributions['efficientnet_b0']['fdi'],
                    'EfficientNet score': contributions['efficientnet_b0']['score'],
                    'ResNet50 FDI': contributions['resnet50']['fdi'],
                    'ResNet50 score': contributions['resnet50']['score'],
                })
            if contribution_rows:
                with st.expander('Contributions from all three models'):
                    st.dataframe(pd.DataFrame(contribution_rows), hide_index=True, width='stretch')
                    st.caption('All three contribute to each fused label and score. These scores are not measured accuracy.')

        # Shared renderer keeps single-image and batch overlay exports consistent.
        rendered_image = render_detection_overlay(
            detection_input_image, box_data, show_boxes, show_labels, show_conf, line_width,
            confirmed_missing=confirmed_missing)

        for idx_b, b in enumerate(box_data):
            fdi_code = b.get("fdi", b["pred_fdi"])
            confidence = b["conf"]
            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
            quad = fdi_code[0] if len(fdi_code) == 2 else "1"

            detected_teeth.append({
                "FDI Code": fdi_code,
                "Anatomical Description": get_tooth_description(fdi_code),
                "Confidence": f"{confidence * 100:.1f}%" if confidence is not None else "Manual",
                "Source": b["source"],
                "Conf_Raw": confidence,
                "BBox (xmin, ymin, xmax, ymax)": f"[{x1}, {y1}, {x2}, {y2}]"
            })

        # Display Image in Column 2
        with col2:
            st.image(rendered_image, width="stretch")
            output_buffer = io.BytesIO()
            rendered_image.save(output_buffer, format="PNG")
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button("Download Predictions Overlay", output_buffer.getvalue(), "fdi_predictions.png", "image/png")
            with c_dl2:
                isolated_dt = isolate_detected_teeth(image_to_process, box_data, padding=8, background_dim=0.0)
                buf_iso = io.BytesIO()
                isolated_dt.save(buf_iso, format="PNG")
                st.download_button("Download isolated teeth", buf_iso.getvalue(), "teeth_isolated_black.png", "image/png")

        # Painted overlays are display artifacts. Downstream tasks share clean pixels and reviewed coordinates.
        measurement_image = image_to_process
        measurement_id = hashlib.sha256(str(measurement_image.size).encode() + measurement_image.tobytes()).hexdigest()
        with angles_tab:
            st.subheader('Measure a tooth angle')
            st.caption('Choose a measurement method. Reference lines match the green/red/yellow drawing; molar comparison uses adjacent tooth axes.')
            angle_method = st.radio('Measurement method',
                ['Reference lines', 'Compare molar axes'], horizontal=True, key='angle_method_' + measurement_id)
            if angle_method == 'Reference lines':
                reference_line_review(measurement_image, measurement_id, box_data)
            else:
                automatic_rows = automatic_angulation_view(measurement_image, box_data, measurement_id)
                with st.expander('Review and correct tooth axes'):
                    with st.container(key='molar_measurement_panel'):
                        molar_review(measurement_image, box_data, measurement_id, automatic_rows)

        with bone_tab:
            with st.container(key='bone_measurement_panel'):
                periodontal_review(measurement_image, box_data, measurement_id)

        with reports_tab:
            with st.expander('Export a reviewed training example'):
                training_signature = hashlib.sha256(json.dumps({'boxes':box_data,'missing':sorted(confirmed_missing)},sort_keys=True).encode()).hexdigest()[:16]
                reviewed_for_training = st.checkbox('I reviewed the tooth numbers, boxes and missing teeth',
                    key='training_review_' + review_id + training_signature)
                if reviewed_for_training:
                    st.download_button('Download training example',
                        export_fdi_example(measurement_image, box_data, confirmed_missing),
                        'reviewed_fdi_training_example.zip', 'application/zip', on_click='ignore')

        with summary_panel:
            # Summary Metrics Cards
            detected_codes = set([t["FDI Code"] for t in detected_teeth])

            total_detected = len(detected_codes)
            upper_arch_count = len([c for c in detected_codes if c.startswith(('1', '2'))])
            lower_arch_count = len([c for c in detected_codes if c.startswith(('3', '4'))])
            missing_count = sum(row['Status'] == 'Not detected — review' for row in inventory)

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{total_detected} / 32</div><div class="metric-lbl">Total Teeth Detected</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{upper_arch_count}</div><div class="metric-lbl">Upper Arch (Maxilla)</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{lower_arch_count}</div><div class="metric-lbl">Lower Arch (Mandible)</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#a13333;">{missing_count}</div><div class="metric-lbl">Not Detected — Needs Review</div></div>', unsafe_allow_html=True)
            with m5:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#216647;">{infer_time:.0f} ms</div><div class="metric-lbl">Processing time</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Dental Arch Chart View
            st.subheader("Tooth chart")
            st.caption('Patient right is on image left. Green: detected; orange: confirmed missing; gray: not detected, needs review. Gaps never shift neighboring FDI numbers.')
        
            # Helper to render tooth badges for a quadrant
            def render_quadrant(q_num, codes):
                html = '<div class="quad-box">'
                html += f'<b>{QUADRANT_NAMES[str(q_num)]}</b><br>'
                for code in codes:
                    if code in detected_codes:
                        html += f'<span class="tooth-badge tooth-present"># {code}</span>'
                    elif code in confirmed_missing:
                        html += f'<span class="tooth-badge tooth-confirmed-missing" title="Confirmed missing"># {code}</span>'
                    else:
                        html += f'<span class="tooth-badge tooth-missing" title="Not detected — review"># {code}</span>'
                html += '</div>'
                return html

            q1_cols, q2_cols = st.columns(2)
            with q1_cols:
                st.markdown(render_quadrant(1, FDI_ARCHES['Upper right']), unsafe_allow_html=True)
                st.markdown(render_quadrant(4, FDI_ARCHES['Lower right']), unsafe_allow_html=True)
            with q2_cols:
                st.markdown(render_quadrant(2, FDI_ARCHES['Upper left']), unsafe_allow_html=True)
                st.markdown(render_quadrant(3, FDI_ARCHES['Lower left']), unsafe_allow_html=True)

            st.markdown("---")

        with reports_tab:
            # Detection Data Table
            st.subheader("Tooth findings and export")
            if detected_teeth:
                df = pd.DataFrame(detected_teeth).sort_values("FDI Code")
                st.dataframe(df[["FDI Code", "Anatomical Description", "Confidence", "Source", "BBox (xmin, ymin, xmax, ymax)"]], width="stretch")

                # CSV Download
                csv_buf = io.BytesIO()
                df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="Download Dental Report CSV",
                    data=csv_buf.getvalue(),
                    file_name="dental_fdi_detection_report.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No predictions to report. Lowering confidence may reveal weak predictions, but cannot repair an inadequately trained model.")

    else:
        st.info("Please upload a Dental Panoramic X-Ray image or select a sample image from the options above.")

# ---------------------------------------------------------
# TAB 2: Training Hub
# ---------------------------------------------------------
with tab2:
    st.subheader("Dataset and training administration")
    st.write("Check your FDI dataset before starting model training.")
    with st.expander('Training guide and FDI annotation files'):
        st.caption('Annotate only visible teeth. Missing 16, 24 or 33 never changes the class IDs of neighboring teeth.')
        st.download_button('Download FDI annotation and training guide',
            (PROJECT_ROOT / 'docs/TRAINING_GUIDE.md').read_bytes(),
            'FDI_training_guide.md', 'text/markdown', key='training_guide', on_click='ignore')
        st.download_button('Download YOLO class-to-FDI mapping',
            (PROJECT_ROOT / 'configs/fdi_class_mapping.csv').read_bytes(),
            'fdi_class_mapping.csv', 'text/csv', key='training_mapping', on_click='ignore')
        st.download_button('Download annotation class names',
            (PROJECT_ROOT / 'configs/annotation_classes.txt').read_bytes(),
            'classes.txt', 'text/plain', key='training_class_names', on_click='ignore')
        st.download_button('Download complete dataset and measurement-flow guide',
            (PROJECT_ROOT / 'docs/DATASET_AND_MEASUREMENT_FLOW.md').read_bytes(),
            'dataset_and_measurement_flow.md', 'text/markdown', key='measurement_flow_guide', on_click='ignore')
    if st.button("Check training dataset"):
        problems = check_dataset(PROJECT_DIR / "dataset")
        if problems:
            st.error("Training is blocked until these dataset problems are corrected.")
            st.code("\n".join(problems), language="text")
        else:
            st.success("Structural checks passed. Labels and patient separation still require review.")

    # Check Dataset counts
    train_count = len(os.listdir(PROJECT_ROOT / "dataset/images/train")) if (PROJECT_ROOT / "dataset/images/train").exists() else 0
    val_count = len(os.listdir(PROJECT_ROOT / "dataset/images/val")) if (PROJECT_ROOT / "dataset/images/val").exists() else 0

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Training Set Images", f"{train_count} images")
    with col_d2:
        st.metric("Validation Set Images", f"{val_count} images")
    with col_d3:
        st.metric("FDI Target Classes", "32 teeth classes")

    st.markdown("---")
    st.markdown("### Training configuration")
    st.markdown("""
    - **Detection fine-tuning:** starts from your installed `best.pt`, saving a separate candidate.
    - **Native End-to-End NMS-Free Head:** Eliminates post-processing non-maximum suppression overlap artifacts
    - **Image Resolution:** `1024 x 1024` for detailed root/crown feature extractions
    - **Horizontal Flip (`fliplr`):** set to `0.0` (preserves FDI right/left quadrant orientation)
    - **Validation:** compare candidate and baseline; no training setting guarantees improved accuracy.
    """)

    if st.button("Show training instructions"):
        st.info("Run the following command in the project terminal to start training. Dataset checks must pass first.")
        st.code("python -m dental_fdi.training.train", language="bash")

# ---------------------------------------------------------
# TAB 3: FDI Two-Digit Notation Guide
# ---------------------------------------------------------
with tab3:
    st.subheader("Understanding FDI tooth numbers")
    st.write("""
    The **FDI World Dental Federation notation** is an international standard system for identifying teeth.
    Each tooth is assigned a **two-digit number**:
    - **1st Digit:** Represents the quadrant of the mouth (1 to 4 for permanent teeth, 5 to 8 for deciduous teeth).
    - **2nd Digit:** Represents the position of the tooth relative to the midline (1 to 8).
    """)

    st.markdown("### Quadrant Layout (Patient Perspective)")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        **Upper Right (Quadrant 1)**
        - `18` : 3rd Molar (Wisdom)
        - `17` : 2nd Molar
        - `16` : 1st Molar
        - `15` : 2nd Premolar
        - `14` : 1st Premolar
        - `13` : Canine
        - `12` : Lateral Incisor
        - `11` : Central Incisor
        
        <br>
        
        **Lower Right (Quadrant 4)**
        - `48` : 3rd Molar (Wisdom)
        - `47` : 2nd Molar
        - `46` : 1st Molar
        - `45` : 2nd Premolar
        - `44` : 1st Premolar
        - `43` : Canine
        - `42` : Lateral Incisor
        - `41` : Central Incisor
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("""
        **Upper Left (Quadrant 2)**
        - `21` : Central Incisor
        - `22` : Lateral Incisor
        - `23` : Canine
        - `24` : 1st Premolar
        - `25` : 2nd Premolar
        - `26` : 1st Molar
        - `27` : 2nd Molar
        - `28` : 3rd Molar (Wisdom)
        
        <br>
        
        **Lower Left (Quadrant 3)**
        - `31` : Central Incisor
        - `32` : Lateral Incisor
        - `33` : Canine
        - `34` : 1st Premolar
        - `35` : 2nd Premolar
        - `36` : 1st Molar
        - `37` : 2nd Molar
        - `38` : 3rd Molar (Wisdom)
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader('Periodontal measurements')
    st.info('Bone measurements require reviewed CEJ, crest and apex landmarks. The app does not diagnose clinical periodontitis stage or furcation from an image ratio. See docs/DATASET_AND_MEASUREMENT_FLOW.md for the dataset and training workflow.')
