import React from 'react';
import AngulationBadge from './AngulationBadge';
import BoneLossChart from './BoneLossChart';

export default function ReportView({ results }) {
  if (!results) return null;

  const thirdMolars = results.teeth.filter((t) => t.is_third_molar);
  const withBone = results.teeth.filter((t) => t.bone_loss);
  const segmentCount = results.all_teeth_segments?.length ?? 0;
  const allModelsLoaded = Object.values(results.model_status || {}).every(Boolean);

  return (
    <section className="panel report-view">
      <h2>Analysis Report</h2>

      {results.periodontal_stage != null && (
        <div className="stage-banner">
          <span className="stage-label">Periodontal STAGE: {results.periodontal_stage}</span>
          <span className="stage-meta">
            Mean bone loss: {results.mean_bone_loss_mm ?? '—'} mm · {withBone.length} teeth analyzed
          </span>
        </div>
      )}

      <div className="report-meta">
        <span>Image: {results.image_width} × {results.image_height}px</span>
        <span>Calibration: {results.pixel_spacing_mm == null ? 'Unavailable' : `${results.pixel_spacing_mm} mm/px`} ({results.calibration_source || 'unknown'})</span>
        <span>Teeth segmented: {segmentCount}</span>
      </div>

      {(results.warnings || []).map((warning) => <p className="muted" key={warning}>{warning}</p>)}

      <div className="model-status">
        {Object.entries(results.model_status || {}).map(([name, loaded]) => (
          <span key={name} className={loaded ? 'loaded' : 'missing'}>
            {name}: {loaded ? 'loaded' : 'missing'}
          </span>
        ))}
      </div>

      {withBone.length > 0 && (
        <div className="bone-table-wrap">
          <h3>Bone level by tooth</h3>
          <table className="bone-table">
            <thead>
              <tr>
                <th>FDI</th>
                <th>Mean CEJ–crest (mm)</th>
                <th>Maximum bone level (% root length)</th>
                <th>Sites</th>
              </tr>
            </thead>
            <tbody>
              {withBone.map((t) => (
                <tr key={t.fdi_number}>
                  <td>{t.fdi_number}</td>
                  <td>{t.bone_loss.bone_loss_mm == null ? 'Unavailable' : t.bone_loss.bone_loss_mm.toFixed(2)}</td>
                  <td>{t.bone_loss.radiographic_bone_level_percent == null ? 'Not assessable' : `${t.bone_loss.radiographic_bone_level_percent.toFixed(1)}%`}</td>
                  <td>{(t.bone_loss.sites || []).map((site) => <div key={site.site}>{site.site}: {site.assessable ? `${site.radiographic_bone_level_percent.toFixed(1)}%` : `Not assessable — ${site.reason}`}</div>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {thirdMolars.length > 0 && (
        <>
          <h3>Third molar impaction</h3>
          <div className="tooth-cards">
            {thirdMolars.map((tooth) => (
              <article key={tooth.fdi_number} className="tooth-card">
                <header>
                  <h3>FDI {tooth.fdi_number}</h3>
                  <span className="third-molar-tag">YOLO {tooth.detection_confidence != null ? `${(tooth.detection_confidence * 100).toFixed(0)}%` : 'detected'}</span>
                </header>
                <AngulationBadge angulation={tooth.angulation} />
                <BoneLossChart boneLoss={tooth.bone_loss} />
              </article>
            ))}
          </div>
        </>
      )}

      {!allModelsLoaded && (
        <p className="muted">All three trained models must be available for hybrid analysis.</p>
      )}
    </section>
  );
}
