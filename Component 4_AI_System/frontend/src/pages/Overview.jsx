const modelCards = [
  { title: 'Gingivitis Detection', model: 'YOLO26s' },
  { title: 'Severity Classification', model: 'EfficientNet-B0' },
  { title: 'Tooth Discoloration', model: 'Model Integration Pending' },
];

function Overview({ onNavigate, previewUrl }) {
  return (
    <div className="overview-layout">
      <section className="overview-main">
        <div className="overview-hero card figma-hero">
          <div className="hero-copy">
            <p className="eyebrow">Component 04 Overview</p>
            <h2>Gingivitis & Tooth Discoloration Analysis</h2>
            <p>
              AI-based gingivitis severity assessment and tooth discoloration analysis using
              intraoral dental images.
            </p>
            <div className="hero-badge-row">
              <span>Gingivitis Severity</span>
              <span>Tooth Discoloration</span>
              <span>Explainable Analysis</span>
            </div>
            <div className="hero-actions">
              <button type="button" className="primary blue-primary" onClick={() => onNavigate('upload')}>
                Start Analysis
              </button>
              <button type="button" className="secondary premium-secondary" onClick={() => onNavigate('upload')}>
                Upload Image
              </button>
            </div>
          </div>
          <div className="hero-visual">
            {previewUrl ? (
              <img src={previewUrl} alt="Current uploaded dental preview" />
            ) : (
              <div className="dental-illustration">
                <span>RGB</span>
                <strong>Intraoral Image</strong>
              </div>
            )}
          </div>
        </div>

        <section className="legend-grid">
          <div className="card legend-card">
            <h3>Severity Legend</h3>
            {['Healthy', 'Mild', 'Moderate', 'Severe'].map((item) => (
              <span key={item} className={`legend-pill ${item.toLowerCase()}`}>
                {item}
              </span>
            ))}
          </div>
          <div className="card legend-card">
            <h3>Tooth Discoloration Legend</h3>
            {['Normal', 'Extrinsic', 'Intrinsic'].map((item) => (
              <span key={item} className="legend-pill pending">
                {item}
              </span>
            ))}
          </div>
        </section>

        <section className="card">
          <div className="section-title">
            <h2>What This Component Does</h2>
          </div>
          <div className="model-grid compact-feature-grid">
            {[
              ['Gingival Inflammation Detection', 'YOLO26s region detection for visible gingival findings.'],
              ['Severity Classification', 'EfficientNet-B0 classifies displayed regions as mild, moderate, or severe.'],
              ['Tooth Discoloration Analysis', 'Prepared module area for the future trained discoloration model.'],
              ['Explainable / Region-Based AI Analysis', 'Region boxes and confidence values remain inspectable for research review.'],
            ].map(([title, description], index) => (
              <article key={title} className="feature-card">
                <div className="feature-icon">{String(index + 1).padStart(2, '0')}</div>
                <div>
                  <strong>{title}</strong>
                  <p>{description}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="card">
          <div className="section-title">
            <h2>Workflow</h2>
          </div>
          <div className="flow-strip figma-flow">
            {[
              ['Upload Image', 'Select an intraoral RGB photo'],
              ['Quality Check', 'Validate format and dimensions'],
              ['AI Processing', 'Run detection and severity models'],
              ['Analysis', 'Review regions and controls'],
              ['Results & Report', 'Export findings for research'],
            ].map(([step, subtitle], index) => (
                <div key={step} className="flow-node">
                  <span>{String(index + 1).padStart(2, '0')}</span>
                  <strong>{step}</strong>
                  <small>{subtitle}</small>
                </div>
              ))}
          </div>
        </section>
      </section>

      <aside className="overview-side">
        <section className="card quick-card">
          <h2>Quick Actions</h2>
          <button type="button" onClick={() => onNavigate('new')}><span>01</span>Start New Analysis</button>
          <button type="button" onClick={() => onNavigate('upload')}><span>02</span>Upload Intraoral Image</button>
          <button type="button" onClick={() => onNavigate('report')}><span>03</span>View Results / Combined Report</button>
        </section>

        <section className="card quick-card">
          <h2>Supported Input</h2>
          <div className="check-list">
            <span>Intraoral RGB Image</span>
            <span>JPG / JPEG / PNG</span>
          </div>
        </section>

        <section className="card quick-card">
          <h2>Model Stack</h2>
          {modelCards.map((card) => (
            <div key={card.title} className="mini-model-row">
              <span>{card.title}</span>
              <strong>{card.model}</strong>
            </div>
          ))}
        </section>
      </aside>
    </div>
  );
}

export default Overview;
