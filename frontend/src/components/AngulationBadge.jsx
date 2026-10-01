import React from 'react';

const COLORS = {
  Mesioangular: '#2563eb',
  Vertical: '#059669',
  Horizontal: '#d97706',
  Distoangular: '#dc2626',
};

export default function AngulationBadge({ angulation }) {
  if (!angulation) return null;
  const color = COLORS[angulation.label] || '#64748b';

  return (
    <div className="angulation-badge" style={{ borderColor: color }}>
      <span className="badge-label">Angulation</span>
      <strong style={{ color }}>{angulation.label}</strong>
      <span className="confidence">{(angulation.confidence * 100).toFixed(1)}% confidence</span>
      {angulation.is_impacted && <span className="impacted-tag">Impacted</span>}
      {angulation.is_impacted === false && <span>Predicted non-impacted</span>}
      {angulation.is_impacted == null && <span>Impaction uncertain</span>}
      {angulation.impaction_probability != null && <span>Impaction probability: {(angulation.impaction_probability * 100).toFixed(1)}%</span>}
      {angulation.axis_difference_degrees != null && <span>Angle to second molar: {angulation.axis_difference_degrees.toFixed(1)}°</span>}
      {(angulation.review_reasons || []).map((reason) => <small key={reason}>Review: {reason}</small>)}
    </div>
  );
}
