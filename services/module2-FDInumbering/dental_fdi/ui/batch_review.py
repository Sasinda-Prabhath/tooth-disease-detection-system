"""Streamlit batch upload, prediction and download workflow."""
import hashlib
import json
import base64

import pandas as pd
import streamlit as st

from dental_fdi.imaging.image_enhancement import enhance_dental_xray, get_enhancement_presets
from dental_fdi.reporting.prediction_exports import build_batch_archive, process_batch_item, review_batch_presence
from dental_fdi.analysis.dental_analysis import FDI_CODES


def batch_review(predict, model_identity, confidence, iou, resolution, render_options):
    st.subheader('Bulk X-ray predictions')
    st.write('Upload multiple images, run tooth detection, then download individual overlays or all results as a ZIP.')
    uploads = st.file_uploader('Upload panoramic X-rays',
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'], accept_multiple_files=True, key='batch_uploads')
    mirror = st.checkbox('Mirror all batch images (patient right appears on image right)', key='batch_mirror')
    presets = {'Original (no adjustment)': None,
               **{name: value for name, value in get_enhancement_presets().items() if value != 'custom'}}
    preset_name = st.selectbox('Batch image adjustment', list(presets), key='batch_preset')
    st.caption('Sidebar analysis and display settings apply to every image. Separate batches by orientation when needed. Multi-page TIFF files use the first page.')
    settings = {'schema': 'batch_presence_v1', 'model': model_identity, 'confidence': confidence, 'iou': iou,
                'resolution': resolution, 'mirror': mirror, 'preset': presets[preset_name],
                'display': render_options}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for upload in uploads:
        signature.update(json.dumps([upload.name, upload.size]).encode())
        signature.update(hashlib.sha256(upload.getvalue()).digest())
    identity = signature.hexdigest()
    previous = st.session_state.get('batch_results')
    if previous and previous['identity'] != identity:
        st.session_state.pop('batch_results', None)
        previous = None
    if st.button('Run batch predictions', disabled=not uploads, type='primary'):
        results = []
        progress = st.progress(0, text='Starting batch...')
        preset = presets[preset_name]
        enhance = (lambda image: enhance_dental_xray(image, preset=preset)) if preset else None
        for index, upload in enumerate(uploads, 1):
            progress.progress((index - 1) / len(uploads), text=f'Processing {index}/{len(uploads)}: {upload.name}')
            try:
                result = process_batch_item(upload.name, upload.getvalue(), predict,
                    confidence=confidence, mirror=mirror, enhance=enhance,
                    render_options=render_options, index=index)
            except Exception as exc:
                result = {'input_name': upload.name, 'status': 'Failed', 'error': str(exc), 'detections': []}
            results.append(result)
        progress.progress(1.0, text='Batch complete')
        previous = {'identity': identity, 'results': results,
                    'archive': build_batch_archive(results, settings)}
        st.session_state['batch_results'] = previous
    if not previous:
        return
    results = previous['results']
    available = [i for i, r in enumerate(results) if r.get('overlay') is not None]
    selected = None
    if available:
        selected = st.selectbox('Preview an overlay', available,
            format_func=lambda i: f"{i + 1}. {results[i]['input_name']}", key='batch_preview_' + identity)
        result = results[selected]
        missing = st.multiselect('Confirmed missing teeth for this image (reviewed)', FDI_CODES,
            default=result.get('confirmed_missing', []), key='batch_missing_' + identity + str(selected))
        st.caption('Apply only to the selected image. Unselected, undetected FDI codes remain unresolved; other teeth keep their identities.')
        if sorted(missing) != result.get('confirmed_missing', []):
            result = review_batch_presence(result, missing, render_options)
            results[selected] = result
            previous['archive'] = build_batch_archive(results, settings)
        st.caption('Reviewed missing for this image: ' + (', '.join(result.get('confirmed_missing', [])) or 'None confirmed'))
        with st.expander('Tooth presence status for selected image'):
            st.dataframe(pd.DataFrame(result.get('tooth_status', [])), hide_index=True, width='stretch')
    completed = sum(r['status'] != 'Failed' for r in results)
    st.success(f'{completed} of {len(results)} images processed; {len(results) - completed} failed.')
    st.dataframe(pd.DataFrame([{'Image': r['input_name'], 'Status': r['status'],
        'Teeth detected': len(r['detections']), 'Error': r['error']} for r in results]), hide_index=True)
    st.caption(f"ZIP ready: {completed} overlay(s), summary.csv, predictions.csv, tooth_status.csv and predictions.json ({len(previous['archive']) / 1024 / 1024:.2f} MB).")
    st.download_button('Download all prediction overlays and reports (ZIP)', previous['archive'],
        'dental_batch_predictions.zip', 'application/zip', key='batch_zip_' + identity,
        on_click='ignore')
    with st.expander('ZIP download canceled or not starting?'):
        st.caption('This direct link downloads the same verified archive without using the media endpoint.')
        encoded = base64.b64encode(previous['archive']).decode('ascii')
        st.markdown(
            '<a download="dental_batch_predictions.zip" href="data:application/zip;base64,'
            + encoded + '">Download ZIP directly</a>', unsafe_allow_html=True)
    if selected is not None:
        result = results[selected]
        st.image(result['overlay'], caption=result['input_name'], width='stretch')
        st.download_button('Download this prediction overlay', result['overlay'],
            result['output_name'], 'image/png', key='batch_single_' + identity + str(selected),
            on_click='ignore')
