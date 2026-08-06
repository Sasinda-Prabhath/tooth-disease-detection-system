import React, { useEffect, useState } from 'react';
import { checkHealth, predictModule2 } from './api/client';
import AnnotationOverlay from './components/AnnotationOverlay';
import ModuleSelector from './components/ModuleSelector';
import ReportView from './components/ReportView';
import UploadPanel from './components/UploadPanel';

export default function App() {
  const [selectedModule, setSelectedModule] = useState('module2');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [pixelSpacing, setPixelSpacing] = useState(0.1);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline' }));
  }, []);

  const handleUpload = async (file) => {
    setLoading(true);
    setError('');
    setResults(null);

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));

    try {
      const data = await predictModule2(file, pixelSpacing);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Tooth Disease Detection System</p>
          <h1>Module 2 — Impaction &amp; Bone Loss</h1>
        </div>
        <div className={`health-pill ${health?.status === 'ok' ? 'ok' : 'bad'}`}>
          API: {health?.status === 'ok' ? 'Online' : 'Offline'}
        </div>
      </header>

      <ModuleSelector selected={selectedModule} onSelect={setSelectedModule} />

      <main className="layout-grid">
        <div className="left-column">
          <UploadPanel
            onUpload={handleUpload}
            loading={loading}
            pixelSpacing={pixelSpacing}
            onPixelSpacingChange={setPixelSpacing}
          />
          {error && <div className="error-banner">{error}</div>}
          <section className="panel">
            <h2>X-ray with Annotations</h2>
            <AnnotationOverlay
              imageUrl={previewUrl}
              results={results}
              imageWidth={results?.image_width}
              imageHeight={results?.image_height}
            />
            <div className="legend">
              <span><i className="box third" /> Third molar bbox</span>
              <span><i className="line cej" /> CEJ line</span>
              <span><i className="line crest" /> Alveolar crest</span>
            </div>
          </section>
        </div>

        <div className="right-column">
          <ReportView results={results} />
        </div>
      </main>
    </div>
  );
}
