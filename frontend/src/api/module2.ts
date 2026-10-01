export interface Health { status: 'ok' | 'degraded'; models: Record<string, string> }
export interface WorkflowStage {
  id: string; label: string;
  status: 'pending' | 'running' | 'completed' | 'partial' | 'blocked' | 'failed';
  detail: string | null; image_url: string | null;
}
export interface Prediction {
  case_id: string; model_status: string; raw_image_url: string | null; processed_image_url: string | null;
  fdi_preview_url: string | null; final_image_url: string | null; final_report_url: string | null;
  impaction_preview_url: string | null; bone_loss_preview_url: string | null; workflow: WorkflowStage[];
  pixel_spacing_mm: [number, number] | null; warnings: string[];
  teeth: { fdi: string; confidence: number; status: string; warnings: string[] }[];
  third_molars: { fdi: string; impacted: boolean | null; impaction_confidence: number | null; angulation: string | null; reason: string | null; gradcam_url: string | null }[];
  bone_loss_results: { fdi: string; assessable: boolean; mean_bone_loss_mm: number | null; severity: string; reason: string | null; sites: { side: string; bone_loss_mm: number | null }[] }[];
}
const base = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002').replace(/\/$/, '');
// Memory-only session capability: no tokens in URLs or persistent browser storage.
const session = Array.from(crypto.getRandomValues(new Uint8Array(32)), value => value.toString(16).padStart(2, '0')).join('');
async function checked(path: string, init?: RequestInit) {
  const response = await fetch(base+path, { ...init, cache: 'no-store', headers: { ...init?.headers, 'X-Case-Session': session } });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === 'string' ? body.detail : `Request failed (${response.status}).`);
  }
  return response;
}
export async function checkHealth(signal?: AbortSignal): Promise<Health> { return (await checked('/module2/health', { signal })).json(); }
export async function predictModule2(file: File, spacing: string): Promise<Prediction> {
  const body = new FormData(); body.append('file', file);
  if (spacing !== '') {
    if (!Number.isFinite(Number(spacing)) || Number(spacing) <= 0) throw new Error('Verified calibration must be finite and positive.');
    body.append('pixel_spacing_mm', spacing);
  }
  return (await checked('/module2/predict', { method: 'POST', body })).json();
}
export async function artifact(path: string): Promise<Blob> {
  if (!/^\/module2\/cases\/case_\d{8}_[a-f0-9]{16}\/[a-z0-9-]+$/.test(path)) throw new Error('Invalid artifact URL.');
  return (await checked(path)).blob();
}

export async function runWorkflow(file: File, spacing: string, onStage: (result: Prediction) => void, signal: AbortSignal): Promise<Prediction> {
  const body = new FormData(); body.append('file', file);
  if (spacing !== '') {
    if (!Number.isFinite(Number(spacing)) || Number(spacing) <= 0) throw new Error('Verified calibration must be finite and positive.');
    body.append('pixel_spacing_mm', spacing);
  }
  const response = await checked('/module2/workflow', { method: 'POST', body, signal });
  if (!response.body) throw new Error('Workflow streaming is unavailable.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: Prediction | null = null;
  function consume(line: string) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === 'error') throw new Error(event.detail);
    if (event.type === 'stage' || event.type === 'result') onStage(event.prediction);
    if (event.type === 'result') result = event.prediction;
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n'); buffer = lines.pop() ?? '';
      lines.forEach(consume);
      if (done) { consume(buffer); break; }
    }
    if (!result) throw new Error('Workflow connection ended before the final report.');
    return result;
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
