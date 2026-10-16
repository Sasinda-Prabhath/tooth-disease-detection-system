import { buildAnalysisExport, downloadJsonReport, downloadPdfReport } from '../services/report.js';

function ReportActions({ file, imageInfo, result, timestamp }) {
  function handleDownloadJson() {
    const exportData = buildAnalysisExport({ file, imageInfo, result, timestamp });
    downloadJsonReport(exportData);
  }

  function handleDownloadPdf() {
    downloadPdfReport({ file, imageInfo, result, timestamp });
  }

  return (
    <div className="report-actions">
      <button type="button" className="primary" onClick={() => window.print()}>
        Print
      </button>
      <button type="button" className="secondary" onClick={handleDownloadPdf}>
        Download PDF
      </button>
      <button type="button" className="secondary" disabled={!result} onClick={handleDownloadJson}>
        Download JSON
      </button>
    </div>
  );
}

export default ReportActions;
