import { useCallback, useEffect, useRef, useState } from 'react';

import AppHeader from './components/AppHeader.jsx';
import AppSidebar from './components/AppSidebar.jsx';
import CombinedReport from './pages/CombinedReport.jsx';
import DiscolorationResult from './pages/DiscolorationResult.jsx';
import GingivitisResult from './pages/GingivitisResult.jsx';
import Overview from './pages/Overview.jsx';
import Processing from './pages/Processing.jsx';
import Upload from './pages/Upload.jsx';
import { analyzeGingivitis, getHealth } from './services/api.js';

const pageTitles = {
  overview: 'Oral AI Screening System',
  new: 'New Analysis',
  upload: 'Upload & Quality Check',
  processing: 'AI Processing',
  gingivitis: 'Gingivitis Result',
  discoloration: 'Tooth Discoloration Result',
  report: 'Combined Report',
};

const balancedControls = {
  confidenceThreshold: 0.45,
  iouThreshold: 0.45,
  maxDetections: 6,
};

function App() {
  const fileInputRef = useRef(null);
  const analysisRequestIdRef = useRef(0);
  const [activePage, setActivePage] = useState('overview');
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [imageInfo, setImageInfo] = useState(null);
  const [result, setResult] = useState(null);
  const [analysisTimestamp, setAnalysisTimestamp] = useState(null);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [selectedDetectionId, setSelectedDetectionId] = useState(null);
  const [hoveredDetectionId, setHoveredDetectionId] = useState(null);
  const [controls, setControls] = useState(balancedControls);

  const refreshHealth = useCallback(async () => {
    try {
      const payload = await getHealth();
      setHealth(payload);
    } catch (nextError) {
      setHealth({
        status: 'error',
        yolo_model: 'not_loaded',
        efficientnet_model: 'not_loaded',
        device: '',
        error: nextError.message,
      });
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const intervalId = window.setInterval(refreshHealth, 30000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [refreshHealth]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function readImageDimensions(url) {
    const image = new Image();
    image.onload = () => {
      setImageInfo({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    };
    image.src = url;
  }

  function setImageFile(nextFile) {
    setError('');
    setResult(null);
    setAnalysisTimestamp(null);
    setSelectedDetectionId(null);
    setHoveredDetectionId(null);

    if (!nextFile) return;

    if (!['image/jpeg', 'image/png'].includes(nextFile.type)) {
      setFile(null);
      setPreviewUrl('');
      setImageInfo(null);
      setError('Please choose a JPG, JPEG, or PNG image.');
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const nextPreviewUrl = URL.createObjectURL(nextFile);
    setFile(nextFile);
    setPreviewUrl(nextPreviewUrl);
    setImageInfo(null);
    readImageDimensions(nextPreviewUrl);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    setImageFile(event.dataTransfer.files?.[0]);
  }

  function reset() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl('');
    setImageInfo(null);
    setResult(null);
    setAnalysisTimestamp(null);
    setError('');
    setSelectedDetectionId(null);
    setHoveredDetectionId(null);
    setShowBoxes(true);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  async function analyze(options = {}) {
    const { keepCurrentPage = false } = options;

    if (!file) {
      setError('Choose an image before analysis.');
      setActivePage('upload');
      return;
    }

    const requestId = analysisRequestIdRef.current + 1;
    analysisRequestIdRef.current = requestId;
    setError('');
    setResult(null);
    setSelectedDetectionId(null);
    setHoveredDetectionId(null);
    setIsAnalyzing(true);
    if (!keepCurrentPage) setActivePage('processing');

    try {
      const payload = await analyzeGingivitis(file, controls);
      if (requestId !== analysisRequestIdRef.current) return;
      setResult(payload);
      setAnalysisTimestamp(new Date());
      setSelectedDetectionId(null);
      setActivePage('gingivitis');
    } catch (nextError) {
      if (requestId !== analysisRequestIdRef.current) return;
      setError(
        nextError.message ||
          'Could not reach the backend. Check that FastAPI is running on port 8004.',
      );
      if (!keepCurrentPage) setActivePage('processing');
    } finally {
      if (requestId === analysisRequestIdRef.current) {
        setIsAnalyzing(false);
      }
    }
  }

  function applySettingsAndAnalyze() {
    analyze({ keepCurrentPage: activePage === 'gingivitis' });
  }

  function renderPage() {
    if (activePage === 'overview') return <Overview onNavigate={setActivePage} previewUrl={previewUrl} />;
    if (activePage === 'upload' || activePage === 'new') {
      return (
        <Upload
          file={file}
          previewUrl={previewUrl}
          imageInfo={imageInfo}
          isDragging={isDragging}
          fileInputRef={fileInputRef}
          onBrowse={setImageFile}
          onDrop={handleDrop}
          onDragState={setIsDragging}
          onAnalyze={analyze}
          onReset={reset}
          isAnalyzing={isAnalyzing}
          error={error}
        />
      );
    }
    if (activePage === 'processing') {
      return <Processing isAnalyzing={isAnalyzing} result={result} error={error} />;
    }
    if (activePage === 'gingivitis') {
      return (
        <GingivitisResult
          file={file}
          imageInfo={imageInfo}
          previewUrl={previewUrl}
          result={result}
          analysisTimestamp={analysisTimestamp}
          showBoxes={showBoxes}
          onShowBoxesChange={setShowBoxes}
          selectedDetectionId={selectedDetectionId}
          hoveredDetectionId={hoveredDetectionId}
          onSelectDetection={setSelectedDetectionId}
          onHoverDetection={setHoveredDetectionId}
          controls={controls}
          onControlsChange={setControls}
          onAnalyze={analyze}
          onNavigate={setActivePage}
          isAnalyzing={isAnalyzing}
          hasFile={Boolean(file)}
          error={error}
        />
      );
    }
    if (activePage === 'discoloration') return <DiscolorationResult previewUrl={previewUrl} />;
    return (
      <CombinedReport
        file={file}
        imageInfo={imageInfo}
        previewUrl={previewUrl}
        result={result}
        analysisTimestamp={analysisTimestamp}
      />
    );
  }

  return (
    <div className="app-shell">
      <AppSidebar
        activePage={activePage}
        onNavigate={setActivePage}
        health={health}
        controls={controls}
        onControlsChange={setControls}
        onApplySettings={applySettingsAndAnalyze}
        isAnalyzing={isAnalyzing}
        hasFile={Boolean(file)}
        error={error}
        onRefreshHealth={refreshHealth}
      />
      <div className="main-shell">
        <AppHeader title={pageTitles[activePage]} health={health} />
        <main className="page-content">{renderPage()}</main>
        <footer>Research prototype - not a clinical diagnosis.</footer>
      </div>
    </div>
  );
}

export default App;
