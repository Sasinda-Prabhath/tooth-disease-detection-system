import { useEffect, useMemo, useRef, useState } from 'react';

const severityColors = {
  Mild: '#18875c',
  Moderate: '#c56f14',
  Severe: '#c93636',
};

function percent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function labelsOverlap(first, second) {
  return !(
    first.right <= second.left ||
    first.left >= second.right ||
    first.bottom <= second.top ||
    first.top >= second.bottom
  );
}

function ImageViewer({
  previewUrl,
  result,
  showBoxes,
  onShowBoxesChange,
  selectedDetectionId,
  hoveredDetectionId,
  onSelectDetection,
  onHoverDetection,
  embedded = false,
  showHeader = true,
  overlayMode = 'severity',
}) {
  const imageRef = useRef(null);
  const [renderedSize, setRenderedSize] = useState({ width: 1, height: 1 });
  const activeDetectionId = hoveredDetectionId || selectedDetectionId;
  const detections = result?.detections || [];

  function updateRenderedSize() {
    const image = imageRef.current;
    if (!image) return;
    setRenderedSize({
      width: image.clientWidth || 1,
      height: image.clientHeight || 1,
    });
  }

  useEffect(() => {
    window.addEventListener('resize', updateRenderedSize);
    return () => window.removeEventListener('resize', updateRenderedSize);
  }, []);

  const overlayItems = useMemo(() => {
    if (!result?.gingivitis_detected) return [];

    const imageWidth = renderedSize.width || 1;
    const imageHeight = renderedSize.height || 1;
    const scaleX = imageWidth / (result.image_width || imageWidth);
    const scaleY = imageHeight / (result.image_height || imageHeight);
    const labelHeight = 22;
    const edgePadding = 5;
    const placedLabels = [];

    return detections.map((detection) => {
      const { x1, y1, x2, y2 } = detection.bbox;
      const left = x1 * scaleX;
      const top = y1 * scaleY;
      const width = Math.max(1, (x2 - x1) * scaleX);
      const height = Math.max(1, (y2 - y1) * scaleY);
      const label = `${detection.severity} ${percent(detection.severity_confidence)}`;
      const labelWidth = Math.min(Math.max(label.length * 7 + 14, 58), 112);
      const labelLeft = clamp(left, edgePadding, imageWidth - labelWidth - edgePadding);
      const candidates = [top - labelHeight - 5, top + 4, top + height - labelHeight - 4];
      let labelTop = clamp(candidates[0], edgePadding, imageHeight - labelHeight - edgePadding);

      for (const candidate of candidates) {
        const nextTop = clamp(candidate, edgePadding, imageHeight - labelHeight - edgePadding);
        const rect = {
          left: labelLeft,
          top: nextTop,
          right: labelLeft + labelWidth,
          bottom: nextTop + labelHeight,
        };
        if (!placedLabels.some((placed) => labelsOverlap(rect, placed))) {
          labelTop = nextTop;
          placedLabels.push(rect);
          break;
        }
      }

      return {
        detection,
        label,
        boxStyle: {
          left: `${left}px`,
          top: `${top}px`,
          width: `${width}px`,
          height: `${height}px`,
          '--severity-color':
            overlayMode === 'roi' ? '#2563eb' : severityColors[detection.severity] || '#33515e',
        },
        labelStyle: {
          left: `${labelLeft}px`,
          top: `${labelTop}px`,
          width: `${labelWidth}px`,
          '--severity-color': severityColors[detection.severity] || '#33515e',
        },
      };
    });
  }, [detections, overlayMode, renderedSize, result]);

  const shouldShowBoxes = showBoxes && overlayMode !== 'none';
  const shouldShowLabels = shouldShowBoxes && overlayMode === 'severity';

  return (
    <section className={embedded ? 'image-viewer-card embedded' : 'card image-viewer-card'}>
      {showHeader && (
        <div className="section-title">
          <h2>Detection Image</h2>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={(event) => onShowBoxesChange(event.target.checked)}
            />
            <span>Show Boxes</span>
          </label>
        </div>
      )}

      <div className="image-stage">
        {previewUrl ? (
          <div className="image-wrap">
            <img
              ref={imageRef}
              src={previewUrl}
              alt="Uploaded dental"
              onLoad={updateRenderedSize}
            />
            {shouldShowBoxes &&
              result?.gingivitis_detected &&
              overlayItems.map(({ detection, boxStyle }) => (
                <button
                  key={detection.id}
                  type="button"
                  className={`box-overlay ${overlayMode === 'roi' ? 'roi' : ''} ${
                    activeDetectionId === detection.id ? 'selected' : ''
                  }`}
                  style={boxStyle}
                  onClick={() => onSelectDetection(detection.id)}
                  onMouseEnter={() => onHoverDetection(detection.id)}
                  onMouseLeave={() => onHoverDetection(null)}
                  aria-label={`Region ${detection.id} ${detection.severity}`}
                />
              ))}
            {shouldShowLabels &&
              result?.gingivitis_detected &&
              overlayItems.map(({ detection, label, labelStyle }) => (
                <button
                  key={`${detection.id}-label`}
                  type="button"
                  className={`box-label ${activeDetectionId === detection.id ? 'selected' : ''}`}
                  style={labelStyle}
                  onClick={() => onSelectDetection(detection.id)}
                  onMouseEnter={() => onHoverDetection(detection.id)}
                  onMouseLeave={() => onHoverDetection(null)}
                >
                  {label}
                </button>
              ))}
          </div>
        ) : (
          <div className="empty-state">Upload an image to view detections.</div>
        )}
      </div>
    </section>
  );
}

export { severityColors, percent };
export default ImageViewer;
