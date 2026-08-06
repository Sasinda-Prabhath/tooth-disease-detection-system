import React from 'react';
import AngulationBadge from './AngulationBadge';
import BoneLossChart from './BoneLossChart';

export default function ReportView({ results }) {
  if (!results) return null;

  const thirdMolars = results.teeth.filter((t) => t.is_third_molar);

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

      {thirdMolars.length === 0 && <p className="muted">No third molars detected.</p>}

      <div className="tooth-cards">
        {thirdMolars.map((tooth) => (
          <article key={tooth.fdi_number} className="tooth-card">
            <header>
              <h3>FDI {tooth.fdi_number}</h3>
              <span className="third-molar-tag">Third Molar</span>
            </header>
            <AngulationBadge angulation={tooth.angulation} />
            <BoneLossChart boneLoss={tooth.bone_loss} />
          </article>
        ))}
      </div>
    </section>
  );
}
