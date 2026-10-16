function StatCard({ label, value, tone = 'neutral' }) {
  return (
    <article className={`dashboard-stat-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

export default StatCard;
