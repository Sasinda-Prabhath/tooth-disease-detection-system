import React from 'react';
import PredictionStatusBar from './components/PredictionStatusBar';
import AnnotationOverlay from './components/AnnotationOverlay';
import ModuleSelector from './components/ModuleSelector';
import ReportView from './components/ReportView';
import UploadPanel from './components/UploadPanel';
import { useModule2Prediction } from './hooks/useModule2Prediction';

export default function App() {
  const [selectedModule, setSelectedModule] = React.useState('module2');
  const { pixelSpacing, setPixelSpacing, loading, error, results, previewUrl, health, predict } = useModule2Prediction(0.1);
  const apiOnline = health?.status === 'ok' || health?.status === 'degraded';

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="#top" aria-label="Dental AI home">
          <span className="brand-mark" aria-hidden="true">✦</span>
          <span><strong>DENTAL AI</strong><small>Smarter Diagnostics. Better Care.</small></span>
        </a>
        <nav className="main-nav" aria-label="Primary navigation">
          <a href="#top">Home</a><a href="#analysis">Analysis</a><a href="#modules">Modules</a><a href="#about">How It Works</a>
        </nav>
        <span className={`health-pill ${apiOnline ? 'ok' : 'bad'}`}><i /> {apiOnline ? 'System online' : 'System offline'}</span>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-copy">
            <p className="eyebrow">AI-powered dental diagnostics</p>
            <h1 id="page-title">Clearer insights for <span>healthier smiles.</span></h1>
            <p>Upload a panoramic X-ray to identify impacted third molars and assess alveolar bone loss in one clinical-style report.</p>
            <div className="hero-benefits"><span><b>✓</b> Secure image analysis</span><span><b>✓</b> AI-assisted findings</span><span><b>✓</b> Clear visual reports</span></div>
          </div>
          <div className="hero-orbit" aria-hidden="true"><div className="tooth-illustration">✦</div><span className="orbit-dot dot-one" /><span className="orbit-dot dot-two" /><span className="orbit-dot dot-three" /></div>
        </section>

        <section className="module-area" id="modules">
          <div><p className="eyebrow">Disease detection suite</p><h2>Select an analysis module</h2></div>
          <ModuleSelector selected={selectedModule} onSelect={setSelectedModule} />
        </section>

        <section className="layout-grid" id="analysis">
          <div className="left-column">
            <UploadPanel onUpload={predict} loading={loading} pixelSpacing={pixelSpacing} onPixelSpacingChange={setPixelSpacing} />
            {error && <div className="error-banner">{error}</div>}
            <PredictionStatusBar loading={loading} results={results} error={error} health={health} />
            <section className="panel visualization-panel">
              <p className="eyebrow">Clinical visualisation</p><h2>Tooth &amp; bone analysis</h2>
              <p className="muted panel-subtitle">Colored masks show tooth segmentation. Reference lines identify CEJ and alveolar crest landmarks.</p>
              <AnnotationOverlay imageUrl={previewUrl} results={results} imageWidth={results?.image_width} imageHeight={results?.image_height} loading={loading} error={error} />
              <div className="legend clinical-legend"><span><i className="line tooth-outline" /> Tooth outline</span><span><i className="line cej" /> CEJ reference</span><span><i className="line crest" /> Alveolar crest</span>{results?.periodontal_stage != null && <span>STAGE {results.periodontal_stage} shown on X-ray</span>}</div>
            </section>
          </div>
          <div className="right-column"><ReportView results={results} /></div>
        </section>
      </main>
      <footer className="site-footer" id="about"><div className="footer-brand"><span className="brand-mark">✦</span><strong>DENTAL AI</strong></div><p>AI-powered support for clearer dental imaging decisions.</p><span>© 2026 Dental AI. All rights reserved.</span></footer>
    </div>
  );
}
