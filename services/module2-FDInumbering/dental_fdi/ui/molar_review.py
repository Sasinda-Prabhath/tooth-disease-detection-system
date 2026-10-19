"""Streamlit review tools; no angles are inferred from detection rectangles."""
import base64
import hashlib
import io
import json
from pathlib import Path
from dental_fdi.paths import UI_ASSETS

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import ImageDraw
from dental_fdi.analysis.dental_analysis import ARTICLE_URL, FDI_CODES, MOLAR_PAIRS, measure_axes, winter_category
from dental_fdi.ui.box_drawing import draw_boxes

axis_picker = components.declare_component("molar_axis_picker", path=str(UI_ASSETS / "axis_component"))


def review_detections(boxes, image, identity):
    """Persist each edit once; all downstream tasks share this reviewed geometry."""
    key = 'canonical_boxes_' + identity
    current = st.session_state.setdefault(key, boxes)
    drawn = draw_boxes(image, current, identity)
    signature = hashlib.sha256(json.dumps(drawn, sort_keys=True).encode()).hexdigest()[:16]
    edited = review_boxes(drawn, image, identity + signature)
    if edited != current:
        st.session_state[key] = edited
        st.rerun()
    return current


def review_boxes(boxes, image, identity):
    with st.expander("Correct tooth numbers and boxes / add a missed tooth"):
        st.caption("Coordinates use original-image pixels. Delete false detections, edit FDI codes, or add a row. Changes update the overlay and reports; they do not retrain the model.")
        rows = [{"FDI": b["fdi"], **dict(zip(("x1", "y1", "x2", "y2"), b["xyxy"]))} for b in boxes]
        frame = pd.DataFrame(rows, columns=["FDI", "x1", "y1", "x2", "y2"])
        edited = st.data_editor(frame, num_rows="dynamic", hide_index=True, key="boxes_" + identity,
            column_config={"FDI": st.column_config.SelectboxColumn(options=FDI_CODES, required=True),
                           **{c: st.column_config.NumberColumn(min_value=0, required=True) for c in ("x1", "y1", "x2", "y2")}})
        corrected, used = [], set()
        for row in edited.to_dict("records"):
            fdi = str(row["FDI"])
            coords = [row[c] for c in ("x1", "y1", "x2", "y2")]
            if fdi not in FDI_CODES or any(pd.isna(v) for v in coords):
                st.error("Complete each box row or delete it. Showing original predictions until corrected.")
                return boxes
            x1, y1, x2, y2 = map(float, coords)
            if fdi in used or not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                st.error("FDI codes must be unique and each box must have positive size inside the image. Showing original predictions until corrected.")
                return boxes
            used.add(fdi)
            original = next((b for b in boxes if b["fdi"] == fdi and b["xyxy"] == coords), None)
            corrected.append(original or {"fdi": fdi, "pred_fdi": fdi, "xyxy": coords, "conf": None, "source": "Manual correction"})
        return corrected


def molar_review(image, boxes, image_id, automatic_rows=None):
    st.subheader("Third molars: 18, 28, 38 and 48")
    st.markdown(f"Measure each third molar against its adjacent second molar (18/17, 28/27, 38/37, 48/47). [Study and angle definitions]({ARTICLE_URL}).")
    st.caption("Automatic proposals are prefilled where available. Check and correct both longitudinal axes, including which end is root versus crown. Confirm only after anatomical review. Angles use 2D image pixels.")
    box_signature = hashlib.sha256(json.dumps(boxes, sort_keys=True).encode()).hexdigest()[:16]
    identity = image_id + box_signature
    states = st.session_state.setdefault("molar_review_" + identity, {})
    for proposal in automatic_rows or []:
        if proposal.get("points") and proposal["FDI"] not in states:
            states[proposal["FDI"]] = {"points": proposal["points"], "confirmed": False}
    available = [fdi for fdi in MOLAR_PAIRS if any(b['fdi'] == fdi for b in boxes)]
    choices = available + [fdi for fdi in MOLAR_PAIRS if fdi not in available]
    selected = st.selectbox("Third molar to measure", choices, key="molar_select_" + image_id)
    reference = MOLAR_PAIRS[selected]
    state = states.setdefault(selected, {"points": [], "confirmed": False})
    st.write(f"**{selected}: amber axis. {reference}: cyan reference axis.**")
    if len(state["points"]) == 4:
        reverse_third, reverse_second = st.columns(2)
        changed = False
        if reverse_third.button(f"Swap root/crown for {selected}"):
            state["points"][:2] = state["points"][:2][::-1]; changed = True
        if reverse_second.button(f"Swap root/crown for {reference}"):
            state["points"][2:] = state["points"][2:][::-1]; changed = True
        if changed:
            state["confirmed"] = False
            state["revision"] = state.get("revision", 0) + 1
    st.caption("Click in order: 1 third-molar root centre, 2 third-molar crown centre, 3 second-molar root centre, 4 second-molar crown centre. Leave unavailable or absent teeth unmeasured.")
    crop = (0, 0, image.width, image.height)
    pair_boxes = [b for b in boxes if b["fdi"] in (selected, reference)]
    zoom = st.checkbox("Zoom to detected molar pair", value=False, key="zoom_" + identity + selected)
    if zoom and pair_boxes:
        x1 = min(b["xyxy"][0] for b in pair_boxes); y1 = min(b["xyxy"][1] for b in pair_boxes)
        x2 = max(b["xyxy"][2] for b in pair_boxes); y2 = max(b["xyxy"][3] for b in pair_boxes)
        pad = max(x2-x1, y2-y1) * .35
        crop = (max(0, int(x1-pad)), max(0, int(y1-pad)), min(image.width, int(x2+pad)), min(image.height, int(y2+pad)))
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for b in pair_boxes:
        color = "#ffb300" if b["fdi"] == selected else "#00e5ff"
        draw.rectangle(b["xyxy"], outline=color, width=2)
        draw.text(tuple(b["xyxy"][:2]), b["fdi"], fill=color)
    preview = preview.crop(crop)
    buf = io.BytesIO(); preview.save(buf, format="PNG")
    component_id = identity + selected + str(crop) + str(state.get("revision", 0))
    local_points = [[p[0]-crop[0], p[1]-crop[1]] if p else None for p in state["points"]]
    value = axis_picker(image="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                        points=local_points, identity=component_id, key="picker_" + component_id, default=None)
    if value and value.get("identity") == component_id:
        points = [[p[0]+crop[0], p[1]+crop[1]] if p else None for p in value["points"]]
        if points != state["points"]:
            state.update(points=points, confirmed=False)
    points = state["points"]
    point_signature = hashlib.sha256(json.dumps(points).encode()).hexdigest()[:16]
    angle = None
    if len(points) == 4 and all(p is not None for p in points):
        try:
            angle = measure_axes(points, selected, image.width, image.height)
            st.caption(f"Draft angle: {angle:.2f} degrees. Positive = mesial; negative = distal. Confirm landmarks below to include the measurement in the report.")
        except ValueError as exc:
            st.error(str(exc))
    state["confirmed"] = st.checkbox(
        f"I reviewed the identities of {selected}/{reference} and both root-to-crown axes",
        value=state["confirmed"], disabled=angle is None, key="confirm_" + identity + selected + point_signature)
    rows = []
    for fdi, second in MOLAR_PAIRS.items():
        b = next((b for b in boxes if b["fdi"] == fdi), None)
        saved = states.get(fdi, {})
        measured = None
        if saved.get("confirmed"):
            measured = measure_axes(saved["points"], fdi, image.width, image.height)
        rows.append({"FDI": fdi, "Reference": second,
                     "Detection": b["source"] if b else "Not detected (does not prove absence)",
                     "Angle (degrees)": round(measured, 2) if measured is not None else None,
                     "Classification": winter_category(measured, fdi) if measured is not None else "Awaiting reviewed axes",
                     "Measurement source": "User-reviewed landmarks" if measured is not None else "Unavailable"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Study ranges: vertical -10 to 10; mesioangular 11 to 80; horizontal 81 to 100; distoangular -80 to -11 degrees. Fractional gaps and other angles remain unclassified. The study's Winter categories cover 38/48 only; upper-molar angles are descriptive. Angulation alone does not establish impaction.")
    st.download_button("Download third-molar measurements CSV", pd.DataFrame(rows).to_csv(index=False), "third_molar_angles.csv", "text/csv")
    payload = {"image_sha256": image_id, "image_size": list(image.size), "source": ARTICLE_URL,
               "orientation": "patient right displayed on image left", "coordinate_system": "original image pixels, x right, y down",
               "point_order": ["third_root", "third_crown", "second_root", "second_crown"],
               "measurements": rows, "landmarks": states, "boxes": boxes}
    st.download_button("Download landmarks and review JSON", json.dumps(payload, indent=2), "molar_review.json", "application/json")
    overlay = image.copy(); overlay_draw = ImageDraw.Draw(overlay)
    for fdi, saved in states.items():
        if saved.get("confirmed"):
            pts = saved["points"]
            for start, color in ((0, "#ffb300"), (2, "#00e5ff")):
                overlay_draw.line([tuple(pts[start]), tuple(pts[start+1])], fill=color, width=3)
            degrees = measure_axes(pts, fdi, image.width, image.height)
            overlay_draw.text(tuple(pts[1]), f"{fdi}: {degrees:.2f} deg", fill="#ffb300")
    output = io.BytesIO(); overlay.save(output, format="PNG")
    st.download_button("Download reviewed axes image", output.getvalue(), "molar_axes.png", "image/png")
