"""Review absence without confusing an absent tooth with detector failure."""
import json

import pandas as pd
import streamlit as st

from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.analysis.tooth_inventory import apply_missing_review, tooth_inventory


def presence_review(boxes, identity):
    st.subheader('Review missing teeth')
    st.caption('Mark a tooth missing only after checking the X-ray. Undetected teeth need review; neighboring numbers stay fixed.')
    missing = st.multiselect('Confirmed missing teeth (reviewed)', FDI_CODES,
                            key='missing_review_' + identity)
    kept, excluded = apply_missing_review(boxes, missing)
    if excluded:
        st.info('Excluded detections for teeth you marked missing: ' + ', '.join(box['fdi'] for box in excluded)
                + '. Check their boxes and neighboring FDI labels in the correction editor.')
    inventory = tooth_inventory(kept, missing)
    undetected = [row['FDI'] for row in inventory if row['Status'] == 'Not detected — review']
    st.write('**Confirmed missing:** ' + (', '.join(sorted(missing)) or 'None confirmed'))
    st.write('**Not detected — needs review:** ' + (', '.join(undetected) or 'None'))
    with st.expander('Presence status for all 32 FDI codes'):
        st.dataframe(pd.DataFrame(inventory), hide_index=True, width='stretch')
    with st.expander('Download tooth presence reports'):
        csv_data = pd.DataFrame(inventory).to_csv(index=False).encode('utf-8-sig')
        st.download_button('Download tooth presence report (CSV)', csv_data, 'tooth_presence.csv',
                           'text/csv', key='presence_csv_' + identity, on_click='ignore')
        st.download_button('Download reviewed tooth inventory (JSON)',
            json.dumps({'schema': 'permanent_fdi_presence_v1', 'image_sha256': identity,
                        'confirmed_missing': sorted(missing), 'teeth': inventory,
                        'excluded_detections': excluded}, indent=2),
            'tooth_presence.json', 'application/json', key='presence_json_' + identity, on_click='ignore')
    return kept, set(missing), inventory
