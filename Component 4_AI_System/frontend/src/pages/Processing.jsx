import PipelineStep from '../components/PipelineStep.jsx';
import ProcessingStepper from '../components/ProcessingStepper.jsx';

const steps = [
  { title: 'YOLO26s', description: 'Gingivitis Region Detection' },
  { title: 'ROI Extraction', description: 'Detected gingival regions are cropped' },
  { title: 'EfficientNet-B0', description: 'Mild / Moderate / Severe Classification' },
  { title: 'Tooth Discoloration', description: 'Reserved for future module' },
];

function stepState(index, isAnalyzing, result) {
  if (result && index < 3) return 'complete';
  if (result && index === 3) return 'pending';
  if (isAnalyzing) return index === 0 ? 'active' : 'idle';
  return 'idle';
}

function Processing({ isAnalyzing, result, error }) {
  const logs = [
    result || isAnalyzing ? ['complete', 'Image validation completed'] : null,
    isAnalyzing ? ['running', 'Backend request sent'] : null,
    result ? ['complete', 'YOLO analysis completed'] : null,
    result ? ['complete', 'Severity classification completed'] : null,
    result ? ['complete', 'Result response received'] : null,
    !result && !isAnalyzing ? ['pending', 'Waiting for image analysis'] : null,
  ].filter(Boolean);

  return (
    <div className="page-stack">
      <ProcessingStepper activeIndex={isAnalyzing ? 2 : result ? 5 : 0} complete={Boolean(result)} />

      <section className="card processing-card figma-processing-card">
        <div className="section-title">
          <h2>{isAnalyzing ? 'Analyzing Your Image...' : 'AI Processing'}</h2>
          {isAnalyzing && <span className="subtle-loader" aria-label="Processing" />}
        </div>
        <div className="ai-core-icon">AI</div>

        <div className="pipeline-grid">
          {steps.map((step, index) => (
            <PipelineStep
              key={step.title}
              step={`Step ${index + 1}`}
              title={step.title}
              description={step.description}
              state={stepState(index, isAnalyzing, result)}
            />
          ))}
        </div>

        {error && <div className="alert error">{error}</div>}
      </section>

      <section className="card">
        <div className="section-title">
          <h2>Detailed Processing Logs</h2>
        </div>
        {logs.length ? (
          <div className="log-list">
            {logs.map(([state, log]) => <span key={log} data-state={state}>{log}</span>)}
          </div>
        ) : (
          <div className="empty-state">Processing events will appear after analysis starts.</div>
        )}
      </section>
    </div>
  );
}

export default Processing;
