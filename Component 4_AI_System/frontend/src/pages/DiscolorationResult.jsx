function DiscolorationResult({ previewUrl }) {
  return (
    <div className="discoloration-layout">
      <section className="card pending-module image-side-card">
        <div className="section-title">
          <h2>Tooth Discoloration Analysis</h2>
          <span className="pending-badge">Model Integration Pending</span>
        </div>
        {previewUrl ? (
          <img className="preview-image" src={previewUrl} alt="Current uploaded dental" />
        ) : (
          <div className="empty-state">Current uploaded image will appear here.</div>
        )}
      </section>

      <section className="card classification-placeholder">
        <h2>Awaiting trained Tooth Discoloration model</h2>
        <p>No discoloration prediction, confidence, color metric, or visualization is displayed until the model is integrated.</p>
        <div className="future-grid discoloration-grid">
        {[
          'Normal',
          'Extrinsic',
          'Intrinsic',
          'Confidence',
          'Detected Regions',
          'Color Analysis',
        ].map((item) => (
          <article key={item} className="card future-card">
            <span>{item}</span>
            <strong>Pending</strong>
          </article>
        ))}
        </div>
      </section>

      <section className="card discoloration-notes">
        <h2>Future Integration Areas</h2>
        <div className="report-grid">
          <div><span>ROI / Shade map</span><strong>Not available yet</strong></div>
          <div><span>AI Explainability Summary</span><strong>Pending</strong></div>
          <div><span>Analysis Metadata</span><strong>Awaiting model integration</strong></div>
        </div>
      </section>
    </div>
  );
}

export default DiscolorationResult;
