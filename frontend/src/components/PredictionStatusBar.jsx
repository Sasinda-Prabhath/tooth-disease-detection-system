import React from 'react';

export default function PredictionStatusBar({ loading, results, error, health }) {
  const stages = results?.workflow ?? [];
  const active = stages.find(stage => stage.status === 'running');
  const complete = stages.find(stage => stage.id === 'report')?.status === 'completed';
  const headline = error ? 'Analysis failed - see error below'
    : loading ? (active?.label ?? 'Waiting for backend updates')
    : complete ? 'Report available'
    : 'Waiting for upload';
  const tone = error ? 'error' : loading ? 'loading'
    : complete ? (results.model_status === 'ready' ? 'success' : 'warn') : 'idle';

  return (
    <div className={`prediction-status tone-${tone}`}>
      <div className="prediction-status-head" role="status">
        <span className={`status-dot ${tone}`} />
        <strong>{headline}</strong>
      </div>
      <ol className="prediction-steps">
        {stages.map(stage => (
          <li key={stage.id} className={stage.status === 'completed' ? 'done' : stage.status === 'running' ? 'active' : 'pending'}>
            {stage.label}: {stage.status}{stage.detail && ` - ${stage.detail}`}
          </li>
        ))}
      </ol>
      <div className="prediction-meta">
        {Object.entries(health?.models ?? {}).map(([name, status]) => <span key={name}>{name}: {status}</span>)}
        {!health && <span>Model availability unknown</span>}
        {results && <span>{results.image_width} ? {results.image_height}px</span>}
      </div>
    </div>
  );
}
