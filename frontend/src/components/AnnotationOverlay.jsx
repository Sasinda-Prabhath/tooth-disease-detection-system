import React, { useEffect, useRef, useState } from 'react';
import { Stage, Layer, Rect, Line, Image as KonvaImage } from 'react-konva';

export default function AnnotationOverlay({ imageUrl, results, imageWidth, imageHeight }) {
  const [image, setImage] = useState(null);
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 800, height: 400 });

  useEffect(() => {
    if (!imageUrl) return;
    const img = new window.Image();
    img.src = imageUrl;
    img.onload = () => setImage(img);
  }, [imageUrl]);

  useEffect(() => {
    const update = () => {
      if (containerRef.current) {
        setSize({ width: containerRef.current.offsetWidth, height: 420 });
      }
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  if (!imageUrl) {
    return (
      <div className="overlay-placeholder" ref={containerRef}>
        Upload a panoramic X-ray to see detections and landmarks.
      </div>
    );
  }

  const scaleX = size.width / (imageWidth || 1);
  const scaleY = size.height / (imageHeight || 1);

  return (
    <div className="overlay-container" ref={containerRef}>
      <Stage width={size.width} height={size.height}>
        <Layer>
          {image && <KonvaImage image={image} width={size.width} height={size.height} />}
          {(results?.teeth || []).map((tooth) => {
            const { bbox } = tooth;
            const x = bbox.x1 * scaleX;
            const y = bbox.y1 * scaleY;
            const w = (bbox.x2 - bbox.x1) * scaleX;
            const h = (bbox.y2 - bbox.y1) * scaleY;
            const color = tooth.is_third_molar ? '#38bdf8' : '#94a3b8';

            const cej = tooth.bone_loss?.cej_points || [];
            const crest = tooth.bone_loss?.crest_points || [];
            const offsetX = x;
            const offsetY = y;

            return (
              <React.Fragment key={tooth.fdi_number}>
                <Rect x={x} y={y} width={w} height={h} stroke={color} strokeWidth={2} />
                {cej.length === 2 && (
                  <Line
                    points={cej.flatMap(([px, py]) => [offsetX + px * scaleX, offsetY + py * scaleY])}
                    stroke="#facc15"
                    strokeWidth={2}
                  />
                )}
                {crest.length === 2 && (
                  <Line
                    points={crest.flatMap(([px, py]) => [offsetX + px * scaleX, offsetY + py * scaleY])}
                    stroke="#fb7185"
                    strokeWidth={2}
                  />
                )}
              </React.Fragment>
            );
          })}
        </Layer>
      </Stage>
    </div>
  );
}
