function decimal(value) {
  return value || value === 0 ? Number(value).toFixed(2) : '-';
}

function ResearchDetails({ result }) {
  return (
    <details className="card collapsible-card">
      <summary>Research / Detection Details</summary>
      {result ? (
        <div className="research-grid">
          <div>
            <span>YOLO Output</span>
            <strong>{result.raw_detection_count}</strong>
          </div>
          <div>
            <span>After Duplicate Filtering</span>
            <strong>{result.filtered_detection_count}</strong>
          </div>
          <div>
            <span>Displayed Regions</span>
            <strong>{result.displayed_detection_count}</strong>
          </div>
          <div>
            <span>Applied Confidence Threshold</span>
            <strong>{decimal(result.confidence_threshold)}</strong>
          </div>
          <div>
            <span>Applied IoU Threshold</span>
            <strong>{decimal(result.iou_threshold)}</strong>
          </div>
          <div>
            <span>Applied Max Detections</span>
            <strong>{result.max_detections}</strong>
          </div>
        </div>
      ) : (
        <div className="empty-state">Detection details will appear after analysis.</div>
      )}
    </details>
  );
}

export default ResearchDetails;
