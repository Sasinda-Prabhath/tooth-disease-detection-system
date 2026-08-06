const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002';

export async function predictModule2(file, pixelSpacingMm = 0.1) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('pixel_spacing_mm', String(pixelSpacingMm));

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Prediction failed (${response.status})`);
  }

  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('Service unavailable');
  return response.json();
}
