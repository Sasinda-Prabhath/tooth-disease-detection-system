import React from 'react';
import AngulationBadge from './AngulationBadge';
import BoneLossChart from './BoneLossChart';

export default function ReportView({ results }) {
  if (!results) return null;

  const thirdMolars = results.teeth.filter((t) => t.is_third_molar);
  const displayedTeeth = thirdMolars.length > 0 ? thirdMolars : results.teeth;
  const average = (values) => values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
  const yoloConfidence = average(
    results.teeth
      .map((tooth) => tooth.yolo_confidence)
      .filter((value) => typeof value === 'number'),
  );
  const efficientNetConfidence = average(
    results.teeth
      .map((tooth) => tooth.caries?.confidence)
      .filter((value) => typeof value === 'number'),
  );
  const percent = (value) => value === null ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;

  return (
    <section className="panel report-view">
      <h2>Analysis Report</h2>
      <div className="report-meta">
        <span>Image: {results.image_width} × {results.image_height}px</span>
        <span>Calibration: {results.pixel_spacing_mm} mm/px</span>
      </div>

      <div className="model-status">
        {Object.entries(results.model_status || {}).map(([name, loaded]) => (
          <span key={name} className={loaded ? 'loaded' : 'heuristic'}>
            {name}: {loaded ? 'ML model' : 'Heuristic fallback'}
          </span>
        ))}
      </div>

      <div className="model-confidence-grid">
        <div className="confidence-card yolo-confidence">
          <span className="confidence-label">YOLO26 detection</span>
          <strong>{percent(yoloConfidence)}</strong>
          <small>Bounding-box confidence</small>
        </div>
        <div className="confidence-card efficientnet-confidence">
          <span className="confidence-label">EfficientNet caries</span>
          <strong>{percent(efficientNetConfidence)}</strong>
          <small>Classification confidence</small>
        </div>
      </div>

      {results.teeth.length === 0 && <p className="muted">No detections returned.</p>}

      <div className="tooth-cards">
        {displayedTeeth.map((tooth, index) => (
          <article key={tooth.fdi_number} className="tooth-card">
            <header>
              <h3>{tooth.fdi_number ? `FDI ${tooth.fdi_number}` : `Detection ${index + 1}`}</h3>
              {tooth.is_third_molar && <span className="third-molar-tag">Third Molar</span>}
            </header>
            <div className="prediction-scores">
              <span>YOLO26: {percent(tooth.yolo_confidence)}</span>
              <span>EfficientNet: {tooth.caries ? `${tooth.caries.label} (${percent(tooth.caries.confidence)})` : 'Unavailable'}</span>
              <span>
                Enamel erosion:{' '}
                {tooth.enamel_erosion
                  ? `${tooth.enamel_erosion.label} (${percent(tooth.enamel_erosion.confidence)})`
                  : 'Unavailable'}
              </span>
            </div>
            {tooth.enamel_erosion && (
              <small>
                Affected enamel surface: {tooth.enamel_erosion.affected_surface_pct.toFixed(1)}%
              </small>
            )}
            {tooth.is_third_molar && <AngulationBadge angulation={tooth.angulation} />}
            {tooth.is_third_molar && <BoneLossChart boneLoss={tooth.bone_loss} />}
          </article>
        ))}
      </div>
    </section>
  );
}
