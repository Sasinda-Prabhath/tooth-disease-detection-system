import RegionCard from '../components/RegionCard.jsx';
import ResultImageCard from '../components/dashboard/ResultImageCard.jsx';
import StatCard from '../components/dashboard/StatCard.jsx';
import SummaryCard from '../components/dashboard/SummaryCard.jsx';
import { buildAnalysisExport, downloadJsonReport, downloadPdfReport } from '../services/report.js';

const severityLevels = ['Healthy', 'Mild', 'Moderate', 'Severe'];

function percent(value) {
  if (value === null || value === undefined) return '-';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function countBySeverity(detections, severity) {
  if (severity === 'Healthy') return detections.length === 0 ? 1 : 0;
  return detections.filter((item) => item.severity === severity).length;
}

function formatMetric(value) {
  if (value === null || value === undefined) return 'Not available';
  if (typeof value === 'number') return value.toFixed(2);
  return value;
}

function GingivitisResult({
  file,
  imageInfo,
  previewUrl,
  result,
  analysisTimestamp,
  showBoxes,
  onShowBoxesChange,
  selectedDetectionId,
  hoveredDetectionId,
  onSelectDetection,
  onHoverDetection,
  controls,
  onControlsChange,
  onAnalyze,
  onNavigate,
  isAnalyzing,
  hasFile,
  error,
}) {
  const detections = result?.detections || [];
  const activeDetectionId = hoveredDetectionId || selectedDetectionId;
  const highestYoloDetection = detections.reduce(
    (best, detection) => (!best || detection.yolo_confidence > best.yolo_confidence ? detection : best),
    null,
  );
  const predictedLabel = result
    ? result.gingivitis_detected
      ? result.most_severe_level || 'Detected'
      : 'Healthy'
    : 'Pending';
  const colorMetrics = result?.color_analysis || {};

  function downloadCurrentJson() {
    downloadJsonReport(
      buildAnalysisExport({
        file,
        imageInfo,
        result,
        timestamp: analysisTimestamp?.toISOString?.() || null,
      }),
    );
  }

  function downloadCurrentReport() {
    downloadPdfReport({
      file,
      imageInfo,
      result,
      timestamp: analysisTimestamp?.toISOString?.() || null,
    });
  }

  return (
    <div className="gingivitis-dashboard">
      <div className="gingivitis-page-header card">
        <div>
          <p className="eyebrow">Gingivitis Result</p>
          <h2>Primary Severity Assessment</h2>
          <span>
            YOLO prediction is the primary model result. Color analysis is supporting evidence only.
          </span>
        </div>
        <div className="toolbar-actions">
          <button type="button" className="secondary" disabled={!result} onClick={downloadCurrentReport}>
            Download Report
          </button>
          <button type="button" className="secondary" disabled={!result} onClick={downloadCurrentJson}>
            Download JSON
          </button>
          <button type="button" className="primary blue-primary" onClick={() => onNavigate('upload')}>
            Analyze Another Image
          </button>
        </div>
      </div>

      {error && <div className="alert error">{error}</div>}

      <div className="gingivitis-dashboard-grid">
        <div className="dashboard-main-column">
          <ResultImageCard
            previewUrl={previewUrl}
            result={result}
            showBoxes={showBoxes}
            onShowBoxesChange={onShowBoxesChange}
            selectedDetectionId={selectedDetectionId}
            hoveredDetectionId={hoveredDetectionId}
            onSelectDetection={onSelectDetection}
            onHoverDetection={onHoverDetection}
          />

          <div className="dashboard-stat-grid">
            {severityLevels.map((level) => (
              <StatCard
                key={level}
                label={`${level} count`}
                value={result ? countBySeverity(detections, level) : '-'}
                tone={level.toLowerCase()}
              />
            ))}
          </div>

          <section className="supporting-evidence-card card">
            <div>
              <p className="eyebrow">Supporting Evidence</p>
              <h2>Color Analysis Supporting Evidence</h2>
            </div>
            <div className="evidence-grid">
              <div>
                <span>Average redness</span>
                <strong>{formatMetric(colorMetrics.average_redness)}</strong>
              </div>
              <div>
                <span>Mean saturation</span>
                <strong>{formatMetric(colorMetrics.mean_saturation)}</strong>
              </div>
              <div>
                <span>ROI count</span>
                <strong>{result ? result.displayed_detection_count : '-'}</strong>
              </div>
              <div>
                <span>Support note</span>
                <strong>Used only as secondary evidence</strong>
              </div>
            </div>
          </section>
        </div>

        <aside className="dashboard-summary-column">
          <SummaryCard title="Primary model result">
            <div className={`predicted-label ${predictedLabel.toLowerCase()}`}>
              <span>Predicted label</span>
              <strong>{predictedLabel}</strong>
            </div>
            <p className="model-confidence">
              Highest YOLO confidence: {highestYoloDetection ? percent(highestYoloDetection.yolo_confidence) : '-'}
            </p>
            <div className="severity-chip-row">
              {severityLevels.map((level) => (
                <span key={level} className={predictedLabel === level ? 'active' : ''}>
                  {level}
                </span>
              ))}
            </div>
          </SummaryCard>

          <SummaryCard title="Analysis Summary">
            <div className="summary-metric-list">
              <div>
                <span>Total detections</span>
                <strong>{result?.displayed_detection_count ?? '-'}</strong>
              </div>
              <div>
                <span>Average redness index</span>
                <strong>{formatMetric(colorMetrics.average_redness_index)}</strong>
              </div>
              <div>
                <span>Confidence used</span>
                <strong>{controls.confidenceThreshold.toFixed(2)}</strong>
              </div>
              <div>
                <span>IoU used</span>
                <strong>{controls.iouThreshold.toFixed(2)}</strong>
              </div>
            </div>
            <button
              type="button"
              className="primary dashboard-rerun-button"
              disabled={!hasFile || isAnalyzing}
              onClick={onAnalyze}
            >
              {isAnalyzing ? 'Analyzing...' : 'Re-run Analysis'}
            </button>
          </SummaryCard>

          <SummaryCard title="Region Review">
            {result && detections.length === 0 ? (
              <div className="healthy-empty">
                <strong>Healthy Appearance</strong>
                <span>No confident gingivitis regions detected.</span>
                <small>
                  YOLO output: {result.raw_detection_count} / Displayed regions:{' '}
                  {result.displayed_detection_count}
                </small>
              </div>
            ) : detections.length > 0 ? (
              <div className="detection-list compact-detection-list">
                {detections.map((detection) => (
                  <RegionCard
                    key={detection.id}
                    detection={detection}
                    selected={activeDetectionId === detection.id}
                    onSelect={onSelectDetection}
                    onHover={onHoverDetection}
                  />
                ))}
              </div>
            ) : (
              <div className="empty-state">Region findings will appear after analysis.</div>
            )}
          </SummaryCard>
        </aside>
      </div>
    </div>
  );
}

export default GingivitisResult;
