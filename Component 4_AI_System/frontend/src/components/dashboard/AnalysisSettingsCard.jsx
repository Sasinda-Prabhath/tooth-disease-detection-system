function decimal(value) {
  return Number(value || 0).toFixed(2);
}

const balancedControls = {
  confidenceThreshold: 0.45,
  iouThreshold: 0.45,
  maxDetections: 6,
};

function AnalysisSettingsCard({
  controls,
  onChange,
  onApply,
  isApplying = false,
  hasFile = false,
  error = '',
  compact = false,
}) {
  function update(next) {
    onChange({ ...controls, ...next });
  }

  function resetToBalanced() {
    onChange(balancedControls);
  }

  return (
    <section className={compact ? 'analysis-settings-card compact' : 'analysis-settings-card'}>
      <div className="sidebar-card-heading">
        <span>Analysis Settings</span>
        <strong>Live Controls</strong>
      </div>

      <label className="sidebar-slider">
        <span>
          Confidence Threshold <strong>{decimal(controls.confidenceThreshold)}</strong>
        </span>
        <input
          type="range"
          min="0.10"
          max="0.90"
          step="0.05"
          value={controls.confidenceThreshold}
          onChange={(event) => update({ confidenceThreshold: Number(event.target.value) })}
        />
        <small>Filters low-confidence gingival region detections.</small>
      </label>

      <label className="sidebar-slider">
        <span>
          IoU NMS Threshold <strong>{decimal(controls.iouThreshold)}</strong>
        </span>
        <input
          type="range"
          min="0.20"
          max="0.80"
          step="0.05"
          value={controls.iouThreshold}
          onChange={(event) => update({ iouThreshold: Number(event.target.value) })}
        />
        <small>Controls overlap suppression for nearby regions.</small>
      </label>

      <label className="sidebar-number-control">
        <span>
          Max Detections <strong>{controls.maxDetections}</strong>
        </span>
        <input
          type="number"
          min="1"
          max="30"
          step="1"
          value={controls.maxDetections}
          onChange={(event) =>
            update({
              maxDetections: Math.min(Math.max(Math.round(Number(event.target.value || 1)), 1), 30),
            })
          }
        />
      </label>

      <button
        type="button"
        className="analysis-apply-button"
        disabled={!hasFile || isApplying}
        onClick={onApply}
      >
        {isApplying ? 'Re-Analyzing...' : 'Apply & Re-Analyze'}
      </button>

      <button
        type="button"
        className="analysis-reset-button"
        disabled={isApplying}
        onClick={resetToBalanced}
      >
        Reset to Balanced
      </button>

      {error && <p className="analysis-settings-error">{error}</p>}
    </section>
  );
}

export default AnalysisSettingsCard;
