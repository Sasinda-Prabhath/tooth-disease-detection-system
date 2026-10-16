function SummaryCard({ title, children }) {
  return (
    <section className="dashboard-summary-card">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

export default SummaryCard;
