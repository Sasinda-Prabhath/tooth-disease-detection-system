import React from 'react';

const STAGES = [
  { id: 'upload', label: 'Upload X-ray to backend' },
  { id: 'segment', label: 'Tooth instance segmentation (all teeth)' },
  { id: 'detect', label: 'Third molar detection (FDI 18/28/38/48)' },
  { id: 'angulate', label: 'Impaction angulation classification' },
  { id: 'bone', label: 'Alveolar bone loss (CEJ → crest)' },
];

export default function PredictionStatusBar({ loading, results, error, health }) {
  const models = health?.models_loaded || {};
  const allModels = Object.values(models).every(Boolean);
  const segmentCount = results?.all_teeth_segments?.length ?? 0;

  let headline = 'Waiting for upload';
  let tone = 'idle';

  if (loading) {
    headline = 'Backend is analyzing your panoramic X-ray…';
    tone = 'loading';
  } else if (error) {
    headline = 'Analysis failed — see error below';
    tone = 'error';
  } else if (results) {
    headline =
      segmentCount > 0
        ? `Complete — ${segmentCount} teeth segmented (model output)`
        : 'Complete — no tooth masks returned (check tooth_instance_segmenter model)';
    tone = results.inference_mode === 'trained_yolov8' || results.inference_mode === 'ml' ? 'success' : 'warn';
  }

  const activeStageIndex = loading ? 2 : results ? STAGES.length : 0;

  return (
    <div className={`prediction-status tone-${tone}`}>
      <div className="prediction-status-head">
        <span className={`status-dot ${tone}`} />
        <strong>{headline}</strong>
      </div>

      <ol className="prediction-steps">
        {STAGES.map((stage, i) => {
          let stepState = 'pending';
          if (loading && i <= activeStageIndex) stepState = 'active';
          if (results && !error) stepState = 'done';
          if (error && i === 0) stepState = 'done';
          return (
            <li key={stage.id} className={stepState}>
              {stage.label}
            </li>
          );
        })}
      </ol>

      <div className="prediction-meta">
        <span>Models: {allModels ? 'all loaded' : 'missing — train & export required'}</span>
        {results && (
          <>
            <span>{results.image_width} × {results.image_height}px</span>
            <span>{segmentCount} teeth visualized</span>
          </>
        )}
      </div>
    </div>
  );
}
