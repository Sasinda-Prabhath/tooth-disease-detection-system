"""Detection at one or two resolutions, preserving predicted FDI identities."""


def infer_detections(model, image, confidence, iou, resolution, recovery=False):
    if callable(getattr(type(model), 'predict_rows', None)):
        return model.predict_rows(image, confidence, iou, resolution, recovery)
    rows = []
    sizes = [resolution]
    if recovery:
        sizes.append(1280 if resolution != 1280 else 1024)
    for size in sizes:
        result = model.predict(image, conf=confidence, iou=iou, imgsz=size,
                               agnostic_nms=False, max_det=1000, verbose=False)[0]
        if result.boxes is None:
            continue
        for box in result.boxes:
            rows.append({'pred_fdi': str(result.names[int(box.cls[0].item())]),
                         'conf': float(box.conf[0].item()), 'xyxy': box.xyxy[0].tolist()})
    return rows
