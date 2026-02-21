import { useRef, useEffect, useState } from "react";

/**
 * Renders a [D, F] tensor as a grid of small images. Each row is reshaped
 * to sqrt(F) x sqrt(F) and drawn as a grayscale (or diverging) thumbnail.
 *
 * Used for: encoder_weights [64,784], local_target [64,784],
 * global_target [64,784], som_targets [64,784].
 *
 * For unsigned data (weights): grayscale, 0=black, max=white.
 * For signed data (targets, deltas): diverging, blue=negative, white=zero, red=positive.
 */

interface InspectWeightGridProps {
  data: number[][];
  title: string;
  cols?: number;
  imageSize?: number;
  /** "grayscale" for unsigned data, "diverging" for signed (targets, deltas) */
  colorMode?: "grayscale" | "diverging";
  /** Padding between thumbnails */
  gap?: number;
}

export default function InspectWeightGrid({
  data,
  title,
  cols = 8,
  imageSize = 28,
  colorMode = "grayscale",
  gap = 2,
}: InspectWeightGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    featureIdx: number;
  } | null>(null);

  const D = data.length;
  if (D === 0) {
    return <div style={{ color: "#8b949e", fontSize: 12 }}>{title}: no data</div>;
  }

  const F = data[0].length;
  // Infer image dimensions: assume square images
  const imgH = Math.round(Math.sqrt(F));
  const imgW = Math.round(F / imgH);

  const gridRows = Math.ceil(D / cols);
  const canvasWidth = cols * (imageSize + gap) - gap;
  const canvasHeight = gridRows * (imageSize + gap) - gap;

  // Compute global data range
  let dataMin = Infinity;
  let dataMax = -Infinity;
  for (let d = 0; d < D; d++) {
    for (let f = 0; f < F; f++) {
      const v = data[d][f];
      if (v < dataMin) dataMin = v;
      if (v > dataMax) dataMax = v;
    }
  }
  if (!isFinite(dataMin)) dataMin = 0;
  if (!isFinite(dataMax)) dataMax = 1;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear to background
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    for (let d = 0; d < D; d++) {
      const gridRow = Math.floor(d / cols);
      const gridCol = d % cols;
      const ox = gridCol * (imageSize + gap);
      const oy = gridRow * (imageSize + gap);

      // Create a small ImageData for this thumbnail
      const imgData = ctx.createImageData(imageSize, imageSize);
      const pixels = imgData.data;

      const row = data[d];

      for (let py = 0; py < imageSize; py++) {
        for (let px = 0; px < imageSize; px++) {
          // Map pixel to source index
          const srcY = Math.floor((py / imageSize) * imgH);
          const srcX = Math.floor((px / imageSize) * imgW);
          const srcIdx = srcY * imgW + srcX;
          const value = srcIdx < F ? row[srcIdx] : 0;

          let r: number, g: number, b: number;

          if (colorMode === "diverging") {
            // Blue-White-Red diverging: negative=blue, zero=white, positive=red
            const absMax = Math.max(Math.abs(dataMin), Math.abs(dataMax)) || 1;
            const t = value / absMax; // -1 to 1
            if (t < 0) {
              const s = -t;
              r = Math.round(255 * (1 - s * 0.7));
              g = Math.round(255 * (1 - s * 0.7));
              b = 255;
            } else {
              const s = t;
              r = 255;
              g = Math.round(255 * (1 - s * 0.7));
              b = Math.round(255 * (1 - s * 0.7));
            }
          } else {
            // Grayscale: 0=black, max=white
            const range = dataMax - dataMin || 1;
            const norm = (value - dataMin) / range;
            const v = Math.round(norm * 255);
            r = v;
            g = v;
            b = v;
          }

          const idx = (py * imageSize + px) * 4;
          pixels[idx] = r;
          pixels[idx + 1] = g;
          pixels[idx + 2] = b;
          pixels[idx + 3] = 255;
        }
      }

      ctx.putImageData(imgData, ox, oy);
    }
  }, [data, D, F, cols, imageSize, gap, canvasWidth, canvasHeight, colorMode, dataMin, dataMax, imgH, imgW]);

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvasWidth / rect.width;
    const scaleY = canvasHeight / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    // Which grid cell?
    const cellW = imageSize + gap;
    const cellH = imageSize + gap;
    const gridCol = Math.floor(mx / cellW);
    const gridRow = Math.floor(my / cellH);
    const featureIdx = gridRow * cols + gridCol;

    // Check we're actually over an image, not the gap
    const localX = mx - gridCol * cellW;
    const localY = my - gridRow * cellH;

    if (
      featureIdx >= 0 &&
      featureIdx < D &&
      localX >= 0 && localX < imageSize &&
      localY >= 0 && localY < imageSize
    ) {
      setTooltip({
        x: e.clientX - (containerRef.current?.getBoundingClientRect().left ?? 0),
        y: e.clientY - (containerRef.current?.getBoundingClientRect().top ?? 0),
        featureIdx,
      });
    } else {
      setTooltip(null);
    }
  }

  function handleMouseLeave() {
    setTooltip(null);
  }

  return (
    <div ref={containerRef} style={{ position: "relative", display: "inline-block" }}>
      <div
        style={{
          fontSize: 12,
          fontWeight: 600,
          color: "#e6edf3",
          marginBottom: 4,
        }}
      >
        {title}{" "}
        <span style={{ color: "#8b949e", fontWeight: 400 }}>
          [{D} x {imgH}x{imgW}] range: [{dataMin.toFixed(3)}, {dataMax.toFixed(3)}]
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        style={{
          imageRendering: "pixelated",
          cursor: "crosshair",
          border: "1px solid #30363d",
          maxWidth: "100%",
        }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x + 10,
            top: tooltip.y - 30,
            background: "#161b22",
            border: "1px solid #30363d",
            borderRadius: 4,
            padding: "3px 6px",
            fontSize: 11,
            color: "#e6edf3",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 10,
          }}
        >
          Feature {tooltip.featureIdx}
        </div>
      )}
    </div>
  );
}
