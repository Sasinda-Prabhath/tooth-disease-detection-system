function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '-';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function ImageUpload({
  file,
  previewUrl,
  imageInfo,
  isDragging,
  fileInputRef,
  onBrowse,
  onDrop,
  onDragState,
  onAnalyze,
  onReset,
  isAnalyzing,
  error,
}) {
  const supported = file ? ['image/jpeg', 'image/png'].includes(file.type) : null;

  return (
    <div className="upload-grid">
      <section className="card upload-card numbered-card">
        <div className="numbered-title">
          <span>1</span>
          <div>
            <h2>Upload Intraoral Image</h2>
            <p>Choose a clear RGB intraoral image for Component 04 analysis.</p>
          </div>
        </div>
        <div
          className={`drop-zone ${isDragging ? 'dragging' : ''}`}
          onDragOver={(event) => {
            event.preventDefault();
            onDragState(true);
          }}
          onDragLeave={() => onDragState(false)}
          onDrop={onDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png"
            onChange={(event) => onBrowse(event.target.files?.[0])}
          />
          <div className="upload-icon">↑</div>
          <button type="button" className="primary blue-primary" onClick={() => fileInputRef.current?.click()}>
            Browse Image
          </button>
          <strong>{file ? file.name : 'Drag and drop a dental image'}</strong>
          <span>JPG, JPEG, or PNG</span>
        </div>

        {error && <div className="alert error">{error}</div>}

        <div className="button-row">
          <button type="button" className="primary" onClick={onAnalyze} disabled={!file || isAnalyzing}>
            {isAnalyzing ? 'Analyzing...' : 'Continue to AI Processing'}
          </button>
          <button type="button" className="secondary" onClick={onReset}>
            Replace Image
          </button>
        </div>
      </section>

      <section className="card preview-card numbered-card">
        <div className="section-title">
          <h2>{file ? 'Selected Image' : 'Preview'}</h2>
        </div>
        {previewUrl ? (
          <>
            <img className="preview-image" src={previewUrl} alt="Selected dental upload" />
            <div className="metadata-grid">
              <div>
                <span>Filename</span>
                <strong>{file?.name}</strong>
              </div>
              <div>
                <span>File size</span>
                <strong>{formatBytes(file?.size)}</strong>
              </div>
              <div>
                <span>Dimensions</span>
                <strong>
                  {imageInfo?.width && imageInfo?.height
                    ? `${imageInfo.width} x ${imageInfo.height}px`
                    : 'Reading image...'}
                </strong>
              </div>
              <div>
                <span>Format</span>
                <strong>{file?.type || '-'}</strong>
              </div>
            </div>
          </>
        ) : (
          <div className="empty-state">Image preview will appear here.</div>
        )}
      </section>

      <aside className="upload-side-stack">
        <section className="card numbered-card">
          <div className="numbered-title">
            <span>2</span>
            <div>
              <h2>Analysis Information</h2>
              <p>Runtime values only, no patient identity.</p>
            </div>
          </div>
          <div className="quality-list compact-list">
            <div>
              <span>Component</span>
              <strong>Component 04</strong>
            </div>
            <div>
              <span>Analysis</span>
              <strong>Gingivitis & Tooth Discoloration</strong>
            </div>
            <div>
              <span>Current file</span>
              <strong>{file?.name || '-'}</strong>
            </div>
            <div>
              <span>Ready status</span>
              <strong>{file && supported ? 'Ready' : 'Waiting for image'}</strong>
            </div>
          </div>
        </section>

        <section className="card numbered-card selected-component-card">
          <div className="numbered-title">
            <span>3</span>
            <div>
              <h2>Selected Component</h2>
              <p>Component 04</p>
            </div>
          </div>
          <strong>Gingivitis & Tooth Discoloration Analysis</strong>
        </section>
      </aside>

      <section className="card quality-card numbered-card">
        <div className="numbered-title">
          <span>4</span>
          <div>
            <h2>Image Quality Check</h2>
            <p>Only checks implemented locally are shown.</p>
          </div>
        </div>
        <div className="quality-list">
          <div>
            <span>Format status</span>
            <strong>{supported === null ? '-' : supported ? 'Supported' : 'Unsupported'}</strong>
          </div>
          <div>
            <span>Resolution</span>
            <strong>
              {imageInfo?.width && imageInfo?.height
                ? `${imageInfo.width} x ${imageInfo.height}px`
                : '-'}
            </strong>
          </div>
          <div>
            <span>Minimum resolution</span>
            <strong>
              {imageInfo?.width && imageInfo?.height
                ? imageInfo.width >= 640 && imageInfo.height >= 480
                  ? 'Meets minimum'
                  : 'Below minimum'
                : '-'}
            </strong>
          </div>
          <div>
            <span>File readability</span>
            <strong>{file && supported ? 'Ready for analysis' : '-'}</strong>
          </div>
        </div>
        <div className="guideline-grid">
          <div>
            <h3>Recommended</h3>
            <span>Clear gingival margins</span>
            <span>Even lighting</span>
            <span>Clear tooth surfaces</span>
            <span>Standard RGB photo</span>
          </div>
          <div>
            <h3>Avoid</h3>
            <span>Motion blur</span>
            <span>Excessive reflections</span>
            <span>Obstructed oral structures</span>
            <span>Severe cropping</span>
          </div>
        </div>
      </section>
    </div>
  );
}

export default ImageUpload;
