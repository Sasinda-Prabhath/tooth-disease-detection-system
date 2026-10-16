function toneClass(tone) {
  if (tone === 'ok') return 'badge-ok';
  if (tone === 'warn') return 'badge-warn';
  if (tone === 'error') return 'badge-error';
  return 'badge-neutral';
}

function StatusBadge({ label, tone = 'neutral' }) {
  return (
    <span className={`status-badge ${toneClass(tone)}`}>
      <span className="status-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

export default StatusBadge;
