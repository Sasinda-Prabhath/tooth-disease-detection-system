"""Draw and assign tooth boxes in original-image coordinates."""
import base64
import io
import math
import hashlib
import json

import streamlit as st
import streamlit.components.v1 as components

from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.paths import UI_ASSETS

box_picker = components.declare_component('tooth_box_picker', path=str(UI_ASSETS / 'box_component'))


def put_manual_box(boxes, fdi, xyxy, width, height):
    if fdi not in FDI_CODES or len(xyxy) != 4 or not all(math.isfinite(float(v)) for v in xyxy):
        raise ValueError('Select a permanent FDI code and draw a valid box')
    x1, y1, x2, y2 = xyxy
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) or min(x2-x1, y2-y1) < 2:
        raise ValueError('Draw a box at least two pixels wide and high inside the image')
    manual = {'fdi': fdi, 'pred_fdi': fdi, 'xyxy': list(map(float, xyxy)),
              'conf': None, 'source': 'Manual correction'}
    return sorted([b for b in boxes if b['fdi'] != fdi] + [manual], key=lambda b: b['fdi'])


def draw_boxes(image, boxes, identity):
    updated = boxes
    with st.expander('Draw a tooth box on the X-ray'):
        st.caption('Use this when a tooth is visible but missed or its box needs correction. '
                   'A genuinely missing tooth gets no box. Clear a previous missing-tooth selection before adding that visible tooth. '
                   'Saved boxes are passed to both measurement stages.')
        selected = st.selectbox('FDI code for drawn box', FDI_CODES, key='draw_fdi_' + identity)
        if st.button('Delete selected FDI box', key='remove_drawn_' + identity):
            updated = [b for b in boxes if b['fdi'] != selected]
        buffer = io.BytesIO(); image.save(buffer, format='PNG')
        draft_id = identity + hashlib.sha256(json.dumps(boxes,sort_keys=True).encode()).hexdigest()[:12]
        value = box_picker(image='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                           boxes=[b['xyxy'] for b in boxes], identity=draft_id,
                           key='draw_box_' + draft_id, default=None)
        if st.button('Save drawn box for selected FDI', disabled=not value, key='save_drawn_' + identity):
            try:
                if value.get('identity') != draft_id:
                    raise ValueError('Drawing belongs to a different image')
                updated = put_manual_box(boxes, selected, value['box'], image.width, image.height)
                st.success(f'Saved box for {selected}; neighboring FDI codes stay unchanged.')
            except ValueError as exc:
                st.error(str(exc))
    return updated
