import DetectionCard from '../components/DetectionCard.jsx';
import ReportActions from '../components/ReportActions.jsx';

function formatDate(value) {
  if (!value) return '-';
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(value);
}

function CombinedReport({ file, imageInfo, previewUrl, result, analysisTimestamp }) {
  const observations = result
    ? result.gingivitis_detected
      ? [
          `${result.displayed_detection_count} gingival region(s) detected`,
          `Most severe detected level: ${result.most_severe_level}`,
          'Tooth discoloration analysis is pending',
        ]
      : ['No confident gingivitis regions detected', 'Tooth discoloration analysis is pending']
    : ['No completed analysis is available yet'];

  return (
    <div className="page-stack report-page">
      <ReportActions
        file={file}
        imageInfo={imageInfo}
        result={result}
        timestamp={analysisTimestamp?.toISOString?.() || null}
      />

      <section className="report-summary-grid">
        <article className="card report-summary-card">
          <span>Gingivitis Analysis</span>
          <strong>{result?.gingivitis_detected ? 'Gingivitis Detected' : result ? 'Healthy Appearance' : 'Pending'}</strong>
          <p>{result ? `${result.displayed_detection_count} displayed region(s)` : 'Run analysis to populate this card.'}</p>
        </article>
        <article className="card report-summary-card">
          <span>Tooth Discoloration Analysis</span>
          <strong>Pending Analysis</strong>
          <p>Awaiting trained Tooth Discoloration model.</p>
        </article>
      </section>

      <section className="card printable-report">
        <div className="report-heading">
          <div>
          <p className="eyebrow">Component 04</p>
            <h2>Analysis Report</h2>
          </div>
          <span>{formatDate(analysisTimestamp)}</span>
        </div>

        <div className="report-section">
          <h3>Image Information</h3>
          <div className="report-grid">
            <div>
              <span>Image filename</span>
              <strong>{file?.name || '-'}</strong>
            </div>
            <div>
              <span>Image dimensions</span>
              <strong>
                {imageInfo?.width && imageInfo?.height
                  ? `${imageInfo.width} x ${imageInfo.height}px`
                  : '-'}
              </strong>
            </div>
            <div>
              <span>Analysis timestamp</span>
              <strong>{formatDate(analysisTimestamp)}</strong>
            </div>
          </div>
          {previewUrl && <img className="report-image" src={previewUrl} alt="Uploaded dental" />}
        </div>

        <div className="report-section">
          <h3>Primary AI Findings</h3>
          {result ? (
            <>
              <div className="report-grid">
                <div>
                  <span>Detected / Not Detected</span>
                  <strong>{result.gingivitis_detected ? 'Detected' : 'Not detected'}</strong>
                </div>
                <div>
                  <span>Region count</span>
                  <strong>{result.displayed_detection_count}</strong>
                </div>
                <div>
                  <span>Most severe level</span>
                  <strong>{result.most_severe_level || 'None'}</strong>
                </div>
              </div>
              {result.detections.length > 0 ? (
                <div className="detection-list report-detections">
                  {result.detections.map((detection) => (
                    <DetectionCard
                      key={detection.id}
                      detection={detection}
                      selected={false}
                      onSelect={() => {}}
                      onHover={() => {}}
                      showBox
                    />
                  ))}
                </div>
              ) : (
                <div className="healthy-empty">
                  <strong>Healthy Appearance</strong>
                  <span>No confident gingivitis regions detected.</span>
                </div>
              )}
            </>
          ) : (
            <div className="empty-state">Run gingivitis analysis to populate this section.</div>
          )}
        </div>

        <div className="report-section">
          <h3>Key Observations</h3>
          <div className="log-list">
            {observations.map((item) => <span key={item}>{item}</span>)}
          </div>
        </div>

        <div className="report-section">
          <h3>AI Analysis Summary</h3>
          <p>
            {result
              ? result.gingivitis_detected
                ? `Gingivitis regions were detected in ${result.displayed_detection_count} area(s). The most severe predicted class was ${result.most_severe_level}. Tooth discoloration analysis is pending.`
                : 'No confident gingivitis regions were detected. Tooth discoloration analysis is pending.'
              : 'No analysis has been completed yet.'}
          </p>
        </div>

        <div className="report-section">
          <h3>Tooth Discoloration Analysis</h3>
          <div className="empty-state">Pending Analysis</div>
        </div>

        <div className="report-section">
          <h3>Overall Screening Summary</h3>
          <p>
            {result
              ? result.gingivitis_detected
                ? result.summary
                : 'No confident gingivitis regions detected.'
              : 'No analysis has been completed yet.'}
          </p>
          <p className="disclaimer">Research prototype - not a clinical diagnosis.</p>
        </div>

        <div className="report-section">
          <h3>Model Information</h3>
          <div className="report-grid">
            <div><span>Detection model</span><strong>YOLO26s</strong></div>
            <div><span>Severity model</span><strong>EfficientNet-B0</strong></div>
            <div><span>Discoloration model</span><strong>Pending</strong></div>
          </div>
        </div>

        <div className="report-section">
          <h3>Applied Detection Settings</h3>
          <div className="report-grid">
            <div><span>Confidence Threshold</span><strong>{result?.confidence_threshold ?? '-'}</strong></div>
            <div><span>IoU Threshold</span><strong>{result?.iou_threshold ?? '-'}</strong></div>
            <div><span>Max Detections</span><strong>{result?.max_detections ?? '-'}</strong></div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default CombinedReport;
