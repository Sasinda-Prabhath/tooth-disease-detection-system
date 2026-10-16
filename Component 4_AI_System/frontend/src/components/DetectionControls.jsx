const presets = {
  Sensitive: { confidenceThreshold: 0.35 },
  Balanced: { confidenceThreshold: 0.45, iouThreshold: 0.45, maxDetections: 12 },
  Strict: { confidenceThreshold: 0.55 },
};

function decimal(value) {
  return Number(value).toFixed(2);
}

function DetectionControls({
  controls,
  onChange,
  onAnalyze,
  showBoxes,
  onShowBoxesChange,
  isAnalyzing,
  hasFile,
}) {
  function update(next) {
    onChange({ ...controls, ...next });
  }

  function applyPreset(name) {
    update(presets[name]);
  }

  return (
    <details className="card collapsible-card" open>
      <summary>Advanced Detection Controls</summary>

      <div className="preset-row premium-preset-row">
        {Object.keys(presets).map((name) => (
          <button key={name} type="button" className="secondary preset-chip" onClick={() => applyPreset(name)}>
            {name}
          </button>
        ))}
      </div>

      <label className="control-row">
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
      </label>

      <label className="control-toggle-row">
        <span>Detection Overlay</span>
        <button
          type="button"
          className={showBoxes ? 'toggle-pill active' : 'toggle-pill'}
          onClick={() => onShowBoxesChange(!showBoxes)}
        >
          {showBoxes ? 'Hide Boxes' : 'Show Boxes'}
        </button>
      </label>

      <label className="control-row">
        <span>
          IoU Threshold <strong>{decimal(controls.iouThreshold)}</strong>
        </span>
        <input
          type="range"
          min="0.20"
          max="0.80"
          step="0.05"
          value={controls.iouThreshold}
          onChange={(event) => update({ iouThreshold: Number(event.target.value) })}
        />
      </label>

      <label className="control-row max-detections-row">
        <span>
          Max Detections <strong>{controls.maxDetections}</strong>
        </span>
        <input
          type="range"
          min="1"
          max="30"
          step="1"
          value={controls.maxDetections}
          onChange={(event) => update({ maxDetections: Number(event.target.value) })}
        />
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

      <div className="button-row">
        <button type="button" className="primary" disabled={!hasFile || isAnalyzing} onClick={onAnalyze}>
          {isAnalyzing ? 'Analyzing...' : 'Re-Analyze'}
        </button>
        <button type="button" className="secondary" onClick={() => applyPreset('Balanced')}>
          Reset to Balanced
        </button>
      </div>
    </details>
  );
}

export default DetectionControls;
