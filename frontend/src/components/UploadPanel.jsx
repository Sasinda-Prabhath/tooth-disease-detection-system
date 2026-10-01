import React, { useRef } from 'react';

export default function UploadPanel({ onUpload, loading, pixelSpacing, onPixelSpacingChange }) {
  const inputRef = useRef(null);
  const handleFile = (event) => {
    const file = event.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <section className="panel upload-panel">
      <p className="eyebrow">Module 2</p>
      <h2>Upload panoramic X-ray</h2>
      <p className="muted">Impacted third molar and alveolar bone-loss assessment.</p>
      <div className="upload-controls">
        <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/jpg,.dcm,.dicom" onChange={handleFile} disabled={loading} />
        <button className="primary-button" type="button" onClick={() => inputRef.current?.click()} disabled={loading}><span aria-hidden="true">↑</span> {loading ? 'Analyzing...' : 'Choose X-ray image'}</button>
        <span className="upload-help">PNG, JPG, DICOM supported</span>
      </div>
      <label className="spacing-field">Validated calibration (optional, mm/pixel)<input type="number" step="any" min="0.000001" placeholder="Use DICOM when available" value={pixelSpacing} onChange={(event) => onPixelSpacingChange(event.target.value)} /></label>
      <p className="muted">Leave blank unless you have a validated calibration. Without it, the report shows relative bone levels only.</p>
    </section>
  );
}
