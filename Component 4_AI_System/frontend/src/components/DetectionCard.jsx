import { percent, severityColors } from './ImageViewer.jsx';

function DetectionCard({ detection, selected, onSelect, onHover, showBox = false }) {
  return (
    <button
      type="button"
      className={`detection-card ${selected ? 'selected' : ''}`}
      onClick={() => onSelect(detection.id)}
      onMouseEnter={() => onHover(detection.id)}
      onMouseLeave={() => onHover(null)}
    >
      <div className="result-title">
        <strong>Region {String(detection.id).padStart(2, '0')}</strong>
        <span style={{ backgroundColor: severityColors[detection.severity] }}>
          {detection.severity}
        </span>
      </div>
      <dl className="finding-list">
        <div>
          <dt>Severity Confidence</dt>
          <dd>{percent(detection.severity_confidence)}</dd>
        </div>
        <div>
          <dt>YOLO Detection Confidence</dt>
          <dd>{percent(detection.yolo_confidence)}</dd>
        </div>
        {showBox && (
          <div>
            <dt>Bounding Box</dt>
            <dd>
              {detection.bbox.x1}, {detection.bbox.y1}, {detection.bbox.x2}, {detection.bbox.y2}
            </dd>
          </div>
        )}
      </dl>
    </button>
  );
}

export default DetectionCard;
