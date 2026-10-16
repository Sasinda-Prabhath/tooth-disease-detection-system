const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8004';

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || 'Backend health check failed.');
  }

  return payload;
}

export async function analyzeGingivitis(file, controls) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('confidence_threshold', controls.confidenceThreshold.toString());
  formData.append('iou_threshold', controls.iouThreshold.toString());
  formData.append('max_detections', controls.maxDetections.toString());

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || 'The analysis request failed.');
  }

  return payload;
}

export { API_BASE_URL };
