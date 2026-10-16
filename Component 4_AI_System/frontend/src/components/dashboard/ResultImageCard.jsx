import { useEffect, useState } from 'react';

import ImageViewer, { severityColors } from '../ImageViewer.jsx';

function percent(value) {
  if (value === null || value === undefined) return '-';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function downloadPredictionImage(previewUrl, result) {
  if (!previewUrl) return;

  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const width = result?.image_width || image.naturalWidth;
    const height = result?.image_height || image.naturalHeight;

    canvas.width = width;
    canvas.height = height;
    context.drawImage(image, 0, 0, width, height);

    if (result?.gingivitis_detected) {
      context.lineWidth = Math.max(8, Math.round(width * 0.0018));
      context.font = `${Math.max(28, Math.round(width * 0.006))}px Arial`;
      context.textBaseline = 'top';

      result.detections.forEach((detection) => {
        const { x1, y1, x2, y2 } = detection.bbox;
        const color = severityColors[detection.severity] || '#2563eb';
        const label = `${detection.severity} ${percent(detection.severity_confidence)}`;

        context.strokeStyle = color;
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const labelWidth = context.measureText(label).width + 28;
        const labelHeight = Math.max(38, Math.round(width * 0.01));
        const labelTop = Math.max(0, y1 - labelHeight - 8);

        context.fillStyle = color;
        context.fillRect(x1, labelTop, labelWidth, labelHeight);
        context.fillStyle = '#ffffff';
        context.fillText(label, x1 + 14, labelTop + 8);
      });
    }

    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'gingivitis-prediction.png';
    link.click();
  };

  image.src = previewUrl;
}

function ResultImageCard({
  previewUrl,
  result,
  showBoxes,
  onShowBoxesChange,
  selectedDetectionId,
  hoveredDetectionId,
  onSelectDetection,
  onHoverDetection,
}) {
  const [activeImageTab, setActiveImageTab] = useState(result ? 'result' : 'original');
  const hasImage = Boolean(previewUrl);
  const hasResult = Boolean(result);
  const hasDetections = Boolean(result?.detections?.length);
  const overlayMode =
    activeImageTab === 'roi' ? 'roi' : activeImageTab === 'result' ? 'severity' : 'none';

  useEffect(() => {
    setActiveImageTab(result ? 'result' : 'original');
  }, [result]);

  function selectTab(nextTab) {
    if (!hasImage) return;
    if ((nextTab === 'roi' || nextTab === 'result') && !hasResult) return;
    setActiveImageTab(nextTab);
  }

  return (
    <section className="dashboard-image-card">
      <div className="result-tabs">
        <button
          type="button"
          className={activeImageTab === 'original' ? 'active' : ''}
          disabled={!hasImage}
          onClick={() => selectTab('original')}
        >
          Original Image
        </button>
        <button
          type="button"
          className={activeImageTab === 'roi' ? 'active' : ''}
          disabled={!hasImage || !hasResult}
          onClick={() => selectTab('roi')}
        >
          Gingival ROI
        </button>
        <button
          type="button"
          className={activeImageTab === 'result' ? 'active' : ''}
          disabled={!hasImage || !hasResult}
          onClick={() => selectTab('result')}
        >
          AI Result
        </button>
      </div>

      <ImageViewer
        previewUrl={previewUrl}
        result={result}
        showBoxes={activeImageTab === 'original' ? false : showBoxes}
        onShowBoxesChange={onShowBoxesChange}
        selectedDetectionId={selectedDetectionId}
        hoveredDetectionId={hoveredDetectionId}
        onSelectDetection={onSelectDetection}
        onHoverDetection={onHoverDetection}
        embedded
        showHeader={false}
        overlayMode={overlayMode}
      />

      {activeImageTab === 'roi' && hasResult && !hasDetections && (
        <div className="tab-empty-message">No gingival regions detected</div>
      )}

      <button
        type="button"
        className="secondary download-prediction-button"
        disabled={!previewUrl}
        onClick={() => downloadPredictionImage(previewUrl, result)}
      >
        Download Prediction Image
      </button>
    </section>
  );
}

export default ResultImageCard;
