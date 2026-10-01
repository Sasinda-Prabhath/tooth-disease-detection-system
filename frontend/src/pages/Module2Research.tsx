import React, { useEffect, useRef, useState } from 'react';
import { artifact, checkHealth, runWorkflow, type Prediction, type Health } from '../api/module2';
const tabs = ['Original', 'Processed', 'FDI Numbering', 'Impaction & Angulation', 'Alveolar Bone Loss', 'Final Analysis'];
const workflowLabels = ['Upload radiograph', 'Prepare radiograph', 'FDI tooth numbering', 'Third-molar impaction and angulation', 'Alveolar bone loss', 'Final report'];

export default function Module2Research() {
  const [health, setHealth] = useState<Health | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [spacing, setSpacing] = useState('');
  const [result, setResult] = useState<Prediction | null>(null);
  const [tab, setTab] = useState(0);
  const [images, setImages] = useState<(string | null)[]>([]);
  const [cam, setCam] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const urls = useRef<string[]>([]);
  const generation = useRef(0);
  const request = useRef<AbortController | null>(null);
  const followStages = useRef(true);
  const release = () => { urls.current.forEach(URL.revokeObjectURL); urls.current = []; };
  useEffect(() => {
    const controller = new AbortController();
    checkHealth(controller.signal).then(setHealth).catch(() => {});
    return () => { controller.abort(); request.current?.abort(); generation.current++; release(); };
  }, []);
  async function run(event: React.FormEvent) {
    event.preventDefault(); if (!file) return;
    const version = ++generation.current;
    request.current?.abort();
    const controller = new AbortController(); request.current = controller;
    followStages.current = true;
    release(); setImages([]); setTab(0); setCam(null); setResult(null); setBusy(true); setError('');
    const pending = new Map<string, Promise<void>>();
    function update(response: Prediction) {
      if (generation.current !== version) return;
      setResult(response);
      [response.raw_image_url, response.processed_image_url, response.fdi_preview_url, response.impaction_preview_url, response.bone_loss_preview_url, response.final_image_url].forEach((path, index) => {
        if (!path || pending.has(path)) return;
        pending.set(path, (async () => {
          try {
            const blob = await artifact(path);
            if (generation.current !== version) return;
            const url = URL.createObjectURL(blob); urls.current.push(url);
            setImages(previous => { const next = [...previous]; next[index] = url; return next; });
            if (followStages.current) setTab(previous => Math.max(previous, index));
          } catch {
            if (generation.current === version) setError('One or more stage previews could not be loaded. Findings remain available in the report.');
          }
        })());
      });
    }
    try {
      await runWorkflow(file, spacing, update, controller.signal);
      await Promise.all(pending.values());
      if (generation.current !== version) return;
      const refreshed = await checkHealth().catch(() => null);
      if (generation.current === version) setHealth(refreshed);
    } catch (err) { if (generation.current === version) setError(err instanceof Error ? err.message : 'Analysis failed.'); }
    finally { if (generation.current === version) setBusy(false); }
  }
  async function showCam(path: string) {
    const version = generation.current;
    try {
      const blob = await artifact(path); if (generation.current !== version) return;
      const url = URL.createObjectURL(blob); urls.current.push(url); setCam(url);
    } catch { setError('Explanation expired or unavailable.'); }
  }
  async function downloadReport() {
    if (!result?.final_report_url) return;
    try {
      const url = URL.createObjectURL(await artifact(result.final_report_url)); urls.current.push(url);
      const link = document.createElement('a'); link.href = url; link.download = `${result.case_id}.json`; link.click();
    } catch { setError('Report expired or unavailable.'); }
  }
  return <div className="app-shell">
    <header className="app-header"><a className="brand" href="#analysis"><span className="brand-mark">✦</span><strong>DENTAL OPG RESEARCH</strong></a>
      <span className={`health-pill ${health?.status === 'ok' ? 'ok' : 'bad'}`}><i />{health?.status === 'ok' ? 'Models loaded' : health?.status === 'degraded' ? 'Degraded · models unavailable' : 'Service unavailable'}</span></header>
    <main id="analysis">
      <section className="hero"><div className="hero-copy"><p className="eyebrow">Module 2 · Third-molar research</p><h1>Inspect every <span>analysis stage.</span></h1><p>Review the original OPG, privacy masking, tooth numbering and third-molar findings.</p></div></section>
      <p className="research-warning" role="note">AI-assisted research output only. It is not a clinical diagnosis and requires dentist review.</p>
      <form className="panel" onSubmit={run}><h2>Upload panoramic X-ray</h2>
        <div className="upload-controls"><input aria-label="OPG file" type="file" accept=".png,.jpg,.jpeg,.dcm,.dicom" disabled={busy} onChange={event => { generation.current++; release(); setImages([]); setResult(null); setCam(null); setError(''); setFile(event.target.files?.[0] ?? null); }} />
          <button disabled={!file || busy}>{busy ? 'Processing…' : 'Analyse OPG'}</button></div>
        <label className="spacing-field">Verified pixel spacing (optional, mm/pixel)<input type="number" step="any" min="0.000001" value={spacing} disabled={busy} onChange={event => setSpacing(event.target.value)} /></label>
        <p className="muted">PNG, JPEG or single-frame DICOM · maximum 20 MiB. Leave calibration blank unless verified. DICOM PixelSpacing is read automatically; an entered value overrides it.</p>
      </form>
      {error && <p className="error-banner" role="alert">{error}</p>}
      <section className="panel" aria-label="Analysis workflow" aria-busy={busy}>
        <h2>Radiograph to final report</h2>
        <p className="muted">FDI numbering → impaction and angulation → alveolar bone loss → final report</p>
        <ol className="workflow-steps" aria-live="polite">
          {workflowLabels.map((label, index) => {
            const stage = result?.workflow[index];
            const status = stage?.status ?? 'pending';
            return <li key={label} className={`workflow-step ${status}`}>
              <strong>{label}</strong><span>{status}</span>{stage?.detail && <small>{stage.detail}</small>}
            </li>;
          })}
        </ol>
        {busy && <p role="status">{result?.workflow.find(stage => stage.status === 'running')?.label ?? 'Preparing workflow'}…</p>}
      </section>
      <div className="layout-grid"><section className="panel">
        <div className="stage-tabs" role="tablist" aria-label="OPG analysis stages">{tabs.map((label, index) => <button key={label} id={`tab-${index}`} role="tab" aria-selected={tab === index} aria-controls="stage-preview" onClick={() => { followStages.current = false; setTab(index); setCam(null); }}>{label}</button>)}</div>
        <div id="stage-preview" role="tabpanel" aria-labelledby={`tab-${tab}`} className="stage-preview">
          {images[tab] ? <img src={images[tab]!} alt={`${tabs[tab]} OPG`} /> : <p>{busy ? 'Preparing analysis…' : !result ? 'Upload an OPG to begin.' : 'This stage is unavailable. See status and warnings.'}</p>}
        </div>
        {tab === 0 && result && <p className="muted">Protected original. Available only in this temporary session.</p>}
        {tab === 2 && <p className="muted">Green: contours · Blue: boxes · ?: uncertain FDI. Patient right is assumed to be on image left.</p>}
        {cam && <figure><img className="cam-image" src={cam} alt="Grad-CAM classifier explanation" /><figcaption>Model attention is an explanation aid, not evidence of clinical correctness.</figcaption></figure>}
        {tab === 3 && <p className="muted">Third-molar impaction and angulation findings are shown next to their FDI numbers.</p>}
        {tab === 4 && <p className="muted">Third-molar regions only. CEJ and bone crest landmarks are shown where assessable; distances require verified calibration.</p>}
      </section><aside className="panel report-view"><h2>{busy ? 'Results as they arrive' : 'Final research report'}</h2>
        {!result ? <p className="muted">Missing weights never generate substitute predictions.</p> : <>
          <p>{result.case_id} · {result.model_status}</p>
          <p>Calibration: {result.pixel_spacing_mm ? `${result.pixel_spacing_mm[0]} × ${result.pixel_spacing_mm[1]} mm/pixel (row × column)` : 'Unavailable'}</p>
          {result.final_report_url && result.workflow.find(stage => stage.id === 'report')?.status === 'completed' && <button onClick={downloadReport}>Download final JSON report</button>}
          <ul className="result-warnings">{result.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>
          {result.teeth.length > 0 && <details><summary>All detected teeth ({result.teeth.length})</summary><ul>{result.teeth.map(tooth => <li key={tooth.fdi}>FDI {tooth.fdi}: {(tooth.confidence*100).toFixed(1)}% · {tooth.status} {tooth.warnings.join(' ')}</li>)}</ul></details>}
          <div className="tooth-cards">{result.third_molars.map(tooth => {
            const detection = result.teeth.find(item => item.fdi === tooth.fdi);
            const bone = result.bone_loss_results.find(item => item.fdi === tooth.fdi);
            return <article className="tooth-card" key={tooth.fdi}><h3>FDI {tooth.fdi}</h3>
              <p>Detection: {detection ? `${(detection.confidence*100).toFixed(1)}%` : 'Unavailable'}</p>
              <p>{tooth.impacted === null ? 'Impaction: uncertain / unavailable' : tooth.impacted ? 'Impacted' : 'Not impacted'}{tooth.impaction_confidence != null && ` (${(tooth.impaction_confidence*100).toFixed(1)}%)`}</p>
              <p>Angulation: {tooth.angulation ?? 'Not assessed'}</p>{tooth.reason && <p>{tooth.reason}</p>}
              <p>Bone loss: {bone?.assessable && bone.mean_bone_loss_mm != null ? `${bone.mean_bone_loss_mm.toFixed(2)} mm · assessable` : 'Not assessable'}</p>
              {bone?.sites.map(site => <p key={site.side}>{site.side}: {site.bone_loss_mm == null ? 'Not assessable' : `${site.bone_loss_mm.toFixed(2)} mm`}</p>)}
              <p>{bone?.reason}</p><p>Severity: {bone?.severity ?? 'Unavailable'}</p>
              {tooth.gradcam_url && <button onClick={() => showCam(tooth.gradcam_url!)}>View Grad-CAM</button>}
            </article>;
          })}</div>
          {!busy && !result.third_molars.length && <p>No assessable third-molar results. Missing detections do not establish absence of disease.</p>}
        </>}
        {health && <details><summary>Model availability</summary><ul>{Object.entries(health.models).map(([name, status]) => <li key={name}>{name}: {status}</li>)}</ul></details>}
      </aside></div>
    </main>
  </div>;
}
