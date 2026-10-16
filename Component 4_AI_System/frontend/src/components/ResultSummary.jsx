function ResultSummary({ result }) {
  const detected = Boolean(result?.gingivitis_detected);
  const ready = Boolean(result);

  return (
    <section className={`card result-summary-card ${ready && !detected ? 'healthy' : ''}`}>
      <div className="section-title">
        <h2>Result Summary</h2>
      </div>

      {ready ? (
        <>
          <div className="summary-grid">
            <div>
              <span>Gingivitis Detected</span>
              <strong>{detected ? 'Yes' : 'No'}</strong>
            </div>
            <div>
              <span>Detected Regions</span>
              <strong>{result.detection_count}</strong>
            </div>
            <div>
              <span>Most Severe Detected Level</span>
              <strong>{result.most_severe_level || 'None'}</strong>
            </div>
          </div>

          <div className={`screening-status ${detected ? 'detected' : 'clear'}`}>
            <span>Overall Screening Status</span>
            <strong>{detected ? result.status : 'Healthy Appearance'}</strong>
            <p>
              {detected
                ? result.summary
                : result.summary || 'No confident gingivitis regions detected.'}
            </p>
          </div>
        </>
      ) : (
        <div className="empty-state">Analysis summary will appear here.</div>
      )}
    </section>
  );
}

export default ResultSummary;
