import StatusBadge from './StatusBadge.jsx';

function statusTone(status) {
  return status === 'loaded' || status === 'ok' ? 'ok' : 'error';
}

function AppHeader({ title, health }) {
  const connected = health?.status === 'ok';

  return (
    <header className="top-header figma-topbar">
      <div className="header-title-block">
        <p className="eyebrow">Component 04</p>
        <h1>{title}</h1>
      </div>
      <label className="top-search" aria-label="Search">
        <span>Search</span>
        <input type="search" placeholder="Search patients, reports, findings" />
      </label>
      <div className="header-actions">
        <StatusBadge
          label={connected ? 'System ready' : 'System offline'}
          tone={connected ? 'ok' : 'error'}
        />
        <button type="button" className="notification-button" aria-label="Notifications">!</button>
        <div className="user-profile">
          <span>RC</span>
          <div>
            <strong>Research Clinician</strong>
            <small>{statusTone(health?.yolo_model) === 'ok' ? 'Model loaded' : 'Model pending'}</small>
          </div>
        </div>
      </div>
    </header>
  );
}

export default AppHeader;
