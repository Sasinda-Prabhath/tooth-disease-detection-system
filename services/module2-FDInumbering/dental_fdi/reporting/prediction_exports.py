"""Shared prediction rendering and in-memory batch exports."""
import csv
import io
import json
import re
import zipfile
import textwrap
from pathlib import PurePosixPath

from PIL import Image, ImageDraw, ImageFont, ImageOps

from dental_fdi.analysis.dental_analysis import select_detections
from dental_fdi.analysis.tooth_inventory import apply_missing_review, tooth_inventory


COLORS = {'1': '#00ff7f', '2': '#00e5ff', '3': '#ff4081', '4': '#ffb300'}


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def render_detection_overlay(image, boxes, show_boxes=True, show_labels=True,
                             show_conf=True, line_width=3, confirmed_missing=()):
    """Render in source-image coordinates, keeping label badges on the canvas."""
    overlay = image.convert('RGB').copy()
    draw = ImageDraw.Draw(overlay)
    size = max(10, min(26, image.width // 45, image.height // 12))
    try:
        font = ImageFont.truetype('arialbd.ttf', size)
    except OSError:
        font = ImageFont.load_default()
    for index, box in enumerate(boxes):
        fdi = box['fdi']
        confidence = box.get('conf')
        x1, y1, x2, y2 = box['xyxy']
        color = COLORS.get(fdi[0], '#00ff00')
        if show_boxes:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        if not show_labels:
            continue
        label = f'#{fdi}'
        if show_conf and confidence is not None:
            label += f' ({confidence:.0%})'
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        width, height = right - left + 8, bottom - top + 8
        x = max(0, min(int(x1), image.width - width))
        preferred_y = y1 - height - (index % 2) * height if fdi[0] in '12' else y2 + (index % 2) * height
        y = max(0, min(int(preferred_y), image.height - height))
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill='#0d1117', outline=color)
        draw.text((x + 4 - left, y + 4 - top), label, font=font, fill='white')
    if confirmed_missing:
        message = 'Confirmed missing: ' + ', '.join(sorted(confirmed_missing))
        lines = textwrap.wrap(message, width=max(12, image.width // max(6, size)))
        height = (size + 4) * len(lines) + 8
        y = max(0, image.height - height)
        draw.rectangle((0, y, image.width, image.height), fill='#0d1117')
        draw.multiline_text((4, y + 4), '\n'.join(lines), font=font, fill='#ffcc80', spacing=4)
    return overlay


def process_batch_item(name, data, predict, *, confidence=.35, mirror=False,
                       enhance=None, render_options=None, index=1):
    """Decode once and use the same oriented pixels for prediction and overlay."""
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    if mirror:
        image = ImageOps.mirror(image)
    if enhance is not None:
        image = enhance(image)
    boxes = select_detections(predict(png_bytes(image)), *image.size, confidence)
    overlay = render_detection_overlay(image, boxes, **(render_options or {}))
    # Prefixes keep duplicate basenames unique; strip directory and unsafe characters.
    basename = PurePosixPath(name.replace('\\', '/')).stem
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', basename).strip('._')[:90] or 'image'
    return {'input_name': name, 'output_name': f'{index:04d}_{stem}_predictions.png',
            'overlay': png_bytes(overlay), 'width': image.width, 'height': image.height,
            'detections': boxes, 'status': 'Complete' if boxes else 'No detections', 'error': '',
            'tooth_status': tooth_inventory(boxes), 'confirmed_missing': [],
            '_original_detections': boxes, '_image': png_bytes(image)}


def review_batch_presence(result, confirmed_missing, render_options=None):
    """Re-render from original pixels so reviewed-away boxes actually disappear."""
    original = result.get('_original_detections', result['detections'])
    kept, excluded = apply_missing_review(original, confirmed_missing)
    with Image.open(io.BytesIO(result['_image'])) as image:
        overlay = render_detection_overlay(image, kept, **(render_options or {}), confirmed_missing=confirmed_missing)
    return dict(result, detections=kept, excluded_detections=excluded,
                confirmed_missing=sorted(confirmed_missing), tooth_status=tooth_inventory(kept, confirmed_missing),
                overlay=png_bytes(overlay))


def build_batch_archive(results, settings):
    """Package successful overlays plus reports for every input, including failures."""
    summary = io.StringIO(newline='')
    writer = csv.writer(summary)
    writer.writerow(['Input image', 'Overlay file', 'Status', 'Detected teeth', 'Error'])
    detections = io.StringIO(newline='')
    teeth = csv.writer(detections)
    teeth.writerow(['Input image', 'Overlay file', 'FDI', 'Confidence', 'xmin', 'ymin', 'xmax', 'ymax',
                    'Prediction method', 'Detector FDI', 'Classifier FDI', 'Classifier confidence', 'Label refined',
                    'EfficientNet FDI', 'EfficientNet score', 'ResNet50 FDI', 'ResNet50 score', 'Fusion class score'])
    metadata = []
    presence = io.StringIO(newline='')
    presence_writer = csv.writer(presence)
    presence_writer.writerow(['Input image', 'FDI', 'Status', 'Evidence', 'Score', 'Box'])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            output = result.get('output_name', '')
            writer.writerow([result['input_name'], output, result['status'],
                             len(result.get('detections', [])), result.get('error', '')])
            if result.get('overlay') is not None:
                archive.writestr(f'overlays/{output}', result['overlay'])
            for box in result.get('detections', []):
                contributions = box.get('model_contributions', {})
                efficientnet = contributions.get('efficientnet_b0', {})
                resnet = contributions.get('resnet50', {})
                teeth.writerow([result['input_name'], output, box['fdi'], box['conf'], *box['xyxy'],
                                box.get('prediction_method', 'YOLO detector'), box.get('detector_fdi', box['fdi']),
                                box.get('classifier_fdi', ''), box.get('classifier_confidence', ''),
                                box.get('label_refined', False),
                                efficientnet.get('fdi', ''), efficientnet.get('score', ''),
                                resnet.get('fdi', ''), resnet.get('score', ''), box.get('fusion_class_score', '')])
            report = {k: v for k, v in result.items() if k != 'overlay' and not k.startswith('_')}
            if result['status'] != 'Failed':
                report['tooth_status'] = result.get('tooth_status', tooth_inventory(result.get('detections', [])))
                for tooth in report['tooth_status']:
                    presence_writer.writerow([result['input_name'], tooth['FDI'], tooth['Status'], tooth['Evidence'], tooth['Score'], tooth['Box']])
            metadata.append(report)
        archive.writestr('summary.csv', summary.getvalue().encode('utf-8-sig'))
        archive.writestr('predictions.csv', detections.getvalue().encode('utf-8-sig'))
        archive.writestr('tooth_status.csv', presence.getvalue().encode('utf-8-sig'))
        archive.writestr('predictions.json', json.dumps({'settings': settings, 'images': metadata}, indent=2))
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise ValueError(f'ZIP integrity check failed: {bad_file}')
    return payload
