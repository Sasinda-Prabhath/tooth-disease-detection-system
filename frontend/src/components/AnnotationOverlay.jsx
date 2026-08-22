import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Layer, Rect, Line, Text, Group, Image as KonvaImage } from 'react-konva';

const CLINICAL = {
  toothOutline: '#ef4444',
  cejLine: '#22c55e',
  crestMarker: '#3b82f6',
  stageBg: 'rgba(15, 23, 42, 0.88)',
  stageBorder: '#f8fafc',
};

function computeFit(containerW, containerH, imageW, imageH) {
  const imageAspect = imageW / imageH;
  const containerAspect = containerW / containerH;
  let drawW;
  let drawH;
  let offsetX;
  let offsetY;
  if (imageAspect > containerAspect) {
    drawW = containerW;
    drawH = containerW / imageAspect;
    offsetX = 0;
    offsetY = (containerH - drawH) / 2;
  } else {
    drawH = containerH;
    drawW = containerH * imageAspect;
    offsetX = (containerW - drawW) / 2;
    offsetY = 0;
  }
  return {
    drawW,
    drawH,
    offsetX,
    offsetY,
    scaleX: drawW / imageW,
    scaleY: drawH / imageH,
  };
}

function CrestMarker({ x, y }) {
  const s = 5;
  return (
    <Group>
      <Rect
        x={x - s}
        y={y - s}
        width={s * 2}
        height={s * 2}
        fill={CLINICAL.crestMarker}
        stroke="#93c5fd"
        strokeWidth={1}
      />
    </Group>
  );
}

export default function AnnotationOverlay({
  imageUrl,
  results,
  imageWidth,
  imageHeight,
  loading = false,
  error = '',
}) {
  const [image, setImage] = useState(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 800, height: 520 });

  useEffect(() => {
    if (!imageUrl) {
      setImage(null);
      setNaturalSize({ w: 0, h: 0 });
      return;
    }
    const img = new window.Image();
    img.src = imageUrl;
    img.onload = () => {
      setImage(img);
      setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    };
  }, [imageUrl]);

  useEffect(() => {
    const update = () => {
      if (containerRef.current) {
        setSize({ width: containerRef.current.offsetWidth, height: 520 });
      }
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  const iw = imageWidth || naturalSize.w || 1;
  const ih = imageHeight || naturalSize.h || 1;

  const fit = useMemo(
    () => computeFit(size.width, size.height, iw, ih),
    [size.width, size.height, iw, ih],
  );

  const mapX = (px) => fit.offsetX + px * fit.scaleX;
  const mapY = (py) => fit.offsetY + py * fit.scaleY;

  if (!imageUrl) {
    return (
      <div className="overlay-placeholder" ref={containerRef}>
        Upload a panoramic X-ray for clinical-style tooth segmentation and bone level analysis.
      </div>
    );
  }

  const teeth = (results?.teeth || []).slice().sort((a, b) => a.fdi_number - b.fdi_number);
  const segments = (results?.all_teeth_segments || []).slice().sort((a, b) => a.fdi_number - b.fdi_number);
  const stage = results?.periodontal_stage;

  return (
    <div className="overlay-wrap clinical-overlay" ref={containerRef}>
      <Stage width={size.width} height={size.height}>
        <Layer>
          {image && (
            <KonvaImage
              image={image}
              x={fit.offsetX}
              y={fit.offsetY}
              width={fit.drawW}
              height={fit.drawH}
            />
          )}

          {/* Red tooth outlines (instance segmentation) */}
          {segments.map((seg) => {
            const contour = seg.contour || [];
            if (contour.length >= 3) {
              const points = contour.flatMap(([px, py]) => [mapX(px), mapY(py)]);
              return (
                <Line
                  key={`outline-${seg.fdi_number}`}
                  points={points}
                  closed
                  stroke={CLINICAL.toothOutline}
                  strokeWidth={2}
                  fill="rgba(239, 68, 68, 0.06)"
                />
              );
            }
            const { bbox } = seg;
            return (
              <Rect
                key={`outline-box-${seg.fdi_number}`}
                x={mapX(bbox.x1)}
                y={mapY(bbox.y1)}
                width={(bbox.x2 - bbox.x1) * fit.scaleX}
                height={(bbox.y2 - bbox.y1) * fit.scaleY}
                stroke={CLINICAL.toothOutline}
                strokeWidth={2}
                fill="rgba(239, 68, 68, 0.06)"
              />
            );
          })}

          {/* Green CEJ lines + blue crest markers per tooth */}
          {teeth.map((tooth) => {
            const cej = tooth.bone_loss?.cej_points || [];
            const crest = tooth.bone_loss?.crest_points || [];
            if (cej.length < 2) return null;

            return (
              <Group key={`bone-${tooth.fdi_number}`}>
                <Line
                  points={cej.flatMap(([px, py]) => [mapX(px), mapY(py)])}
                  stroke={CLINICAL.cejLine}
                  strokeWidth={2.5}
                  lineCap="round"
                  lineJoin="round"
                />
                {crest.map(([px, py], i) => (
                  <CrestMarker key={`crest-${tooth.fdi_number}-${i}`} x={mapX(px)} y={mapY(py)} />
                ))}
                {crest.length >= 2 && (
                  <Line
                    points={crest.flatMap(([px, py]) => [mapX(px), mapY(py)])}
                    stroke={CLINICAL.crestMarker}
                    strokeWidth={1.5}
                    dash={[6, 4]}
                    opacity={0.85}
                  />
                )}
              </Group>
            );
          })}

          {stage != null && !loading && (
            <Group x={fit.offsetX + 12} y={fit.offsetY + 12}>
              <Rect width={120} height={36} fill={CLINICAL.stageBg} stroke={CLINICAL.stageBorder} strokeWidth={1} cornerRadius={4} />
              <Text text={`STAGE: ${stage}`} fontSize={16} fontStyle="bold" fill="#f8fafc" padding={10} />
            </Group>
          )}

          {teeth.length > 0 && !loading && (
            <Group x={fit.offsetX + fit.drawW - 52} y={fit.offsetY + fit.drawH - 28}>
              <Text text="CEJ" fontSize={12} fill={CLINICAL.cejLine} fontStyle="bold" />
            </Group>
          )}

          {loading && (
            <>
              <Rect x={0} y={0} width={size.width} height={size.height} fill="rgba(2, 6, 23, 0.55)" />
              <Text
                text="Analyzing teeth and alveolar bone levels…"
                x={size.width / 2 - 170}
                y={size.height / 2 - 8}
                fontSize={16}
                fill="#e2e8f0"
              />
            </>
          )}
        </Layer>
      </Stage>

      {!loading && !error && results && segments.length === 0 && (
        <div className="overlay-hint">
          No tooth segmentation returned. Ensure <code>tooth_instance_segmenter</code> is trained and loaded.
        </div>
      )}
    </div>
  );
}
