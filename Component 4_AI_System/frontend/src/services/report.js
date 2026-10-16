function safeText(value) {
  return String(value ?? '-').replace(/[()\\]/g, '\\$&');
}

function percent(value) {
  if (value === null || value === undefined) return '-';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function highestYoloConfidence(result) {
  if (!result?.detections?.length) return null;
  return result.detections.reduce(
    (highest, detection) => Math.max(highest, Number(detection.yolo_confidence || 0)),
    0,
  );
}

function buildReportLines({ file, imageInfo, result, timestamp }) {
  const highestConfidence = highestYoloConfidence(result);
  const lines = [
    'Oral AI Screening Report',
    '',
    'Gingivitis Analysis',
    `Analysis timestamp: ${timestamp || '-'}`,
    `Uploaded filename: ${file?.name || '-'}`,
    `Image dimensions: ${imageInfo?.width && imageInfo?.height ? `${imageInfo.width} x ${imageInfo.height}px` : '-'}`,
    `Gingivitis detected: ${result ? (result.gingivitis_detected ? 'Yes' : 'No') : '-'}`,
    `Primary severity result: ${result?.most_severe_level || (result ? 'Healthy' : '-')}`,
    `Total detections: ${result?.displayed_detection_count ?? '-'}`,
    `Highest YOLO confidence: ${highestConfidence === null ? '-' : percent(highestConfidence)}`,
    `Confidence threshold used: ${result?.confidence_threshold ?? '-'}`,
    `IoU threshold used: ${result?.iou_threshold ?? '-'}`,
    `Max detections used: ${result?.max_detections ?? '-'}`,
    '',
    'Region Table',
  ];

  if (result) {
    if (result.detections.length) {
      result.detections.forEach((detection) => {
        lines.push(
          `Region ${detection.id}: ${detection.severity} | YOLO ${percent(detection.yolo_confidence)} | Severity ${percent(detection.severity_confidence)}`,
        );
      });
    } else {
      lines.push('No confident gingivitis regions detected.');
    }
  } else {
    lines.push('No gingivitis analysis completed.');
  }

  lines.push(
    '',
    'Model Information',
    'YOLO26s',
    'EfficientNet-B0',
    '',
    'Tooth Discoloration Analysis: Pending',
    '',
    'Research prototype - not a clinical diagnosis.',
  );
  return lines;
}

function createPdfBlob(lines) {
  const pageWidth = 612;
  const pageHeight = 792;
  const margin = 48;
  const lineHeight = 16;
  const maxLinesPerPage = Math.floor((pageHeight - margin * 2) / lineHeight);
  const pages = [];

  for (let index = 0; index < lines.length; index += maxLinesPerPage) {
    pages.push(lines.slice(index, index + maxLinesPerPage));
  }

  const objects = [];
  const addObject = (body) => {
    objects.push(body);
    return objects.length;
  };

  const fontObject = addObject('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
  const pageObjectIds = [];

  pages.forEach((pageLines) => {
    const textCommands = pageLines
      .map((line, index) => {
        const y = pageHeight - margin - index * lineHeight;
        const size = index === 0 && pageLines[0] === 'Oral AI Screening Report' ? 16 : 10;
        return `BT /F1 ${size} Tf ${margin} ${y} Td (${safeText(line)}) Tj ET`;
      })
      .join('\n');

    const stream = `${textCommands}\n`;
    const contentObject = addObject(`<< /Length ${stream.length} >>\nstream\n${stream}endstream`);
    const pageObject = addObject(
      `<< /Type /Page /Parent 0 0 R /MediaBox [0 0 ${pageWidth} ${pageHeight}] /Resources << /Font << /F1 ${fontObject} 0 R >> >> /Contents ${contentObject} 0 R >>`,
    );
    pageObjectIds.push(pageObject);
  });

  const pagesObject = addObject(
    `<< /Type /Pages /Kids [${pageObjectIds.map((id) => `${id} 0 R`).join(' ')}] /Count ${pageObjectIds.length} >>`,
  );
  const catalogObject = addObject(`<< /Type /Catalog /Pages ${pagesObject} 0 R >>`);

  pageObjectIds.forEach((id) => {
    objects[id - 1] = objects[id - 1].replace('/Parent 0 0 R', `/Parent ${pagesObject} 0 R`);
  });

  let pdf = '%PDF-1.4\n';
  const offsets = [0];
  objects.forEach((body, index) => {
    offsets.push(pdf.length);
    pdf += `${index + 1} 0 obj\n${body}\nendobj\n`;
  });
  const xrefOffset = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  offsets.slice(1).forEach((offset) => {
    pdf += `${String(offset).padStart(10, '0')} 00000 n \n`;
  });
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root ${catalogObject} 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`;

  return new Blob([pdf], { type: 'application/pdf' });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function buildAnalysisExport({ file, imageInfo, result, timestamp }) {
  return {
    analysis_timestamp: timestamp,
    input_filename: file?.name || null,
    image: imageInfo || null,
    models: {
      gingivitis_detection: 'YOLO26s',
      severity_classification: 'EfficientNet-B0',
      tooth_discoloration: 'Model Integration Pending',
    },
    applied_settings: result
      ? {
          confidence_threshold: result.confidence_threshold,
          iou_threshold: result.iou_threshold,
          max_detections: result.max_detections,
        }
      : null,
    gingivitis: result
      ? {
          detected: result.gingivitis_detected,
          displayed_regions: result.displayed_detection_count,
          most_severe_level: result.most_severe_level,
          regions: result.detections.map((detection) => ({
            region_number: detection.id,
            severity: detection.severity,
            yolo_confidence: detection.yolo_confidence,
            severity_confidence: detection.severity_confidence,
          })),
        }
      : null,
    tooth_discoloration: 'Pending Analysis',
    disclaimer: 'Research prototype - not a clinical diagnosis.',
  };
}

export function downloadJsonReport(data, filename = 'oral-ai-analysis.json') {
  downloadBlob(
    new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }),
    filename,
  );
}

export function downloadPdfReport(reportData, filename = 'oral-ai-screening-report.pdf') {
  const lines = buildReportLines(reportData);
  downloadBlob(createPdfBlob(lines), filename);
}
