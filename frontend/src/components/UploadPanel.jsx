import React, { useRef } from 'react';

export default function UploadPanel({ onUpload, loading, pixelSpacing, onPixelSpacingChange }) {
  const inputRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <section className="panel upload-panel">
      <h2>Upload Panoramic X-ray</h2>
      <p className="muted">
        Module 2 — Impacted Third Molar &amp; Alveolar Bone Loss (Lakshitha R.M.S.K · IT23222618)
      </p>

      <div className="upload-controls">
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg,.dcm,.dicom"
          onChange={handleFile}
          disabled={loading}
        />
        <button type="button" onClick={() => inputRef.current?.click()} disabled={loading}>
          {loading ? 'Analyzing…' : 'Choose X-ray Image'}
        </button>
      </div>

      <label className="spacing-field">
        Pixel spacing (mm/pixel)
        <input
          type="number"
          step="0.01"
          min="0.01"
          value={pixelSpacing}
          onChange={(e) => onPixelSpacingChange(Number(e.target.value))}
        />
      </label>
    </section>
  );
}
