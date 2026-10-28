const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002';
const MODULE_API_BASES = {
  module1: import.meta.env.VITE_MODULE1_API_URL || 'http://localhost:8001',
  module2: import.meta.env.VITE_MODULE2_API_URL || API_BASE,
};
const USE_GATEWAY = import.meta.env.VITE_USE_GATEWAY === 'true';

function modulePath(moduleId, endpoint) {
  if (!USE_GATEWAY) return `/${endpoint}`;
  return `/${moduleId}/${endpoint}`;
}

function moduleBase(moduleId) {
  return USE_GATEWAY ? API_BASE : MODULE_API_BASES[moduleId] || API_BASE;
}

export async function predictModule(moduleId, file, pixelSpacingMm = 0.1) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('pixel_spacing_mm', String(pixelSpacingMm));

  const predictPath = modulePath(moduleId, 'predict');

  const response = await fetch(`${moduleBase(moduleId)}${predictPath}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Prediction failed (${response.status})`);
  }

  return response.json();
}

export async function checkHealth(moduleId) {
  const healthPath = modulePath(moduleId, 'health');
  const response = await fetch(`${moduleBase(moduleId)}${healthPath}`);
  if (!response.ok) throw new Error('Service unavailable');
  return response.json();
}
