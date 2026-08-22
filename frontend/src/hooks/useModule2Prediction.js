import { useCallback, useEffect, useRef, useState } from 'react';
import { checkHealth, predictModule2 } from '../api/client';

/**
 * MVVM-style view model: UI state + backend prediction (no client-side mock inference).
 */
export function useModule2Prediction(initialSpacing = 0.1) {
  const [pixelSpacing, setPixelSpacing] = useState(initialSpacing);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const previewRef = useRef('');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', models_loaded: {} }));
  }, []);

  const refreshHealth = useCallback(() => {
    return checkHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', models_loaded: {} }));
  }, []);

  const predict = useCallback(
    async (file) => {
      setLoading(true);
      setError('');
      setResults(null);

      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
      const url = URL.createObjectURL(file);
      previewRef.current = url;
      setPreviewUrl(url);

      try {
        const data = await predictModule2(file, pixelSpacing);
        setResults(data);
        await refreshHealth();
      } catch (err) {
        setError(err.message || 'Prediction failed');
      } finally {
        setLoading(false);
      }
    },
    [pixelSpacing, refreshHealth],
  );

  return {
    pixelSpacing,
    setPixelSpacing,
    loading,
    error,
    results,
    previewUrl,
    health,
    predict,
    refreshHealth,
  };
}
