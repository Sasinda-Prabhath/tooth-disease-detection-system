import StatusBadge from './StatusBadge.jsx';
import AnalysisSettingsCard from './dashboard/AnalysisSettingsCard.jsx';

const navItems = [
  { id: 'overview', label: 'Component Overview', icon: '01' },
  { id: 'upload', label: 'Upload & Quality Check', icon: '02' },
  { id: 'processing', label: 'AI Processing', icon: '03' },
  { id: 'gingivitis', label: 'Gingivitis Results', icon: '04' },
  { id: 'discoloration', label: 'Tooth Discoloration', icon: '05' },
  { id: 'report', label: 'Combined Report', icon: '06' },
];

function readyText(ready) {
  return ready ? 'Ready' : 'Not ready';
}

function AppSidebar({
  activePage,
  onNavigate,
  health,
  controls,
  onControlsChange,
  onApplySettings,
  isAnalyzing,
  hasFile,
  error,
  onRefreshHealth,
}) {
  const connected = health?.status === 'ok';
  const yoloReady = health?.yolo_model === 'loaded';
  const efficientNetReady = health?.efficientnet_model === 'loaded';

  return (
    <aside className="sidebar figma-sidebar">
      <div className="brand-block">
        <div className="brand-mark">04</div>
        <div>
          <strong>Dental AI Suite</strong>
          <span>Clinical research prototype</span>
        </div>
      </div>

      <div className="sidebar-scroll-area">
        <div className="sidebar-section-label">Analysis Workspace</div>
        <nav className="side-nav" aria-label="Application pages">
          {navItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={activePage === item.id ? 'active' : ''}
              onClick={() => onNavigate(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <section className="sidebar-info-card model-status-card">
          <div className="sidebar-card-heading">
            <span>Model Status</span>
            <strong>Gingivitis Severity Model</strong>
          </div>
          <div className="status-row">
            <span>API Ready</span>
            <strong>{readyText(connected)}</strong>
          </div>
          <div className="status-row">
            <span>Model Ready</span>
            <strong>{readyText(yoloReady && efficientNetReady)}</strong>
          </div>
          <button type="button" className="sidebar-refresh-button" onClick={onRefreshHealth}>
            Refresh Status
          </button>
        </section>

        <AnalysisSettingsCard
          controls={controls}
          onChange={onControlsChange}
          onApply={onApplySettings}
          isApplying={isAnalyzing}
          hasFile={hasFile}
          error={error}
          compact
        />
      </div>

      <div className="sidebar-footer">
        <span className="prototype-label">Research Prototype</span>
        <StatusBadge
          label={connected ? 'Backend Connected' : 'Backend Disconnected'}
          tone={connected ? 'ok' : 'error'}
        />
        <StatusBadge
          label={`YOLO26s ${health?.yolo_model || 'unknown'}`}
          tone={health?.yolo_model === 'loaded' ? 'ok' : 'error'}
        />
        <StatusBadge
          label={`EfficientNet ${health?.efficientnet_model || 'unknown'}`}
          tone={health?.efficientnet_model === 'loaded' ? 'ok' : 'error'}
        />
      </div>
    </aside>
  );
}

export default AppSidebar;
