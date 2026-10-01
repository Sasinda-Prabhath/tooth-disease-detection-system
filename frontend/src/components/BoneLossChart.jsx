import React from 'react';

const SEVERITY_COLORS = {
  Normal: '#22c55e',
  Mild: '#eab308',
  Moderate: '#f97316',
  Severe: '#ef4444',
};

export default function BoneLossChart({ boneLoss }) {
  if (!boneLoss) return null;
  if (!boneLoss.assessable) return <p>Bone level not assessable: {boneLoss.reason || 'Landmarks unavailable'}</p>;

  if (boneLoss.sites?.length) return (
    <div className="bone-loss-chart">
      <strong>Radiographic bone level</strong>
      {boneLoss.sites.map((site) => <p key={site.site}>
        {site.site}: {site.assessable
          ? `${site.radiographic_bone_level_percent.toFixed(1)}% of root length${site.cej_crest_distance_mm == null ? '' : ` · ${site.cej_crest_distance_mm.toFixed(2)} mm CEJ–crest`}`
          : `Not assessable (${site.reason})`}
      </p>)}
    </div>
  );

  const maxMm = 8;
  const pct = Math.min(100, (boneLoss.bone_loss_mm / maxMm) * 100);
  const color = SEVERITY_COLORS[boneLoss.severity] || '#64748b';

  return (
    <div className="bone-loss-chart">
      <div className="chart-header">
        <span>Bone Loss</span>
        <strong>{boneLoss.bone_loss_mm} mm</strong>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="severity-row">
        <span>Severity</span>
        <span className="severity-pill" style={{ background: color }}>
          {boneLoss.severity}
        </span>
      </div>
    </div>
  );
}
