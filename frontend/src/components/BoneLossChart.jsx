import React from 'react';

const SEVERITY_COLORS = {
  Normal: '#22c55e',
  Mild: '#eab308',
  Moderate: '#f97316',
  Severe: '#ef4444',
};

export default function BoneLossChart({ boneLoss }) {
  if (!boneLoss) return null;

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
