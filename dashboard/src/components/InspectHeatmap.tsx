import { useRef, useEffect, useState } from "react";

/**
 * Canvas-based [M, N] heatmap for rendering matrices like
 * strength [B,D], rank_score [B,D], cosine similarity [D,D], etc.
 *
 * Renders via <canvas> for performance (128x64 = 8192 cells).
 * Hover shows (row, col, value) in a tooltip.
 *
 * Color scales:
 * - "viridis": 0=dark purple, 1=bright yellow (default, for 0-1 data)
 * - "hot": 0=black, 1=white through red/yellow (for magnitude data)
 * - "diverging": blue=negative, white=zero, red=positive (for signed data)
 */

type ColorScale = "viridis" | "hot" | "diverging";

interface InspectHeatmapProps {
  data: number[][];
  title: string;
  xLabel?: string;
  yLabel?: string;
  colorScale?: ColorScale;
  cellSize?: number;
  /** If true, normalize data to [0,1] range before rendering */
  autoNormalize?: boolean;
}

// Viridis-inspired color stops (simplified 8-stop gradient)
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 35, 116],
  [64, 67, 135],
  [52, 94, 141],
  [33, 145, 140],
  [53, 183, 121],
  [143, 215, 68],
  [253, 231, 37],
];

const HOT: [number, number, number][] = [
  [0, 0, 0],
  [87, 0, 0],
  [173, 0, 0],
  [255, 65, 0],
  [255, 160, 0],
  [255, 225, 50],
  [255, 255, 150],
  [255, 255, 255],
];

function interpolateColorStops(
  stops: [number, number, number][],
  t: number
): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (stops.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, stops.length - 1);
  const frac = idx - lo;
  return [
    Math.round(stops[lo][0] + (stops[hi][0] - stops[lo][0]) * frac),
    Math.round(stops[lo][1] + (stops[hi][1] - stops[lo][1]) * frac),
    Math.round(stops[lo][2] + (stops[hi][2] - stops[lo][2]) * frac),
  ];
}

function getColor(
  value: number,
  min: number,
  max: number,
  scale: ColorScale
): [number, number, number] {
  if (scale === "diverging") {
    // Map negative->blue, zero->white, positive->red
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const t = value / absMax; // -1 to 1
    if (t < 0) {
      const s = -t; // 0 to 1
      return [
        Math.round(255 * (1 - s)),
        Math.round(255 * (1 - s)),
        255,
      ];
    } else {
      const s = t;
      return [
        255,
        Math.round(255 * (1 - s)),
        Math.round(255 * (1 - s)),
      ];
    }
  }

  const range = max - min || 1;
  const t = (value - min) / range;
  const stops = scale === "hot" ? HOT : VIRIDIS;
  return interpolateColorStops(stops, t);
}

export default function InspectHeatmap({
  data,
  title,
  xLabel,
  yLabel,
  colorScale = "viridis",
  cellSize,
  autoNormalize = true,
}: InspectHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    row: number;
    col: number;
    value: number;
  } | null>(null);

  const rows = data.length;
  const cols = rows > 0 ? data[0].length : 0;

  // Auto cell size: fit within reasonable bounds
  const cs = cellSize ?? Math.max(2, Math.min(6, Math.floor(600 / Math.max(rows, cols))));
  const canvasWidth = cols * cs;
  const canvasHeight = rows * cs;

  // Compute data range once
  let dataMin = Infinity;
  let dataMax = -Infinity;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = data[r][c];
      if (v < dataMin) dataMin = v;
      if (v > dataMax) dataMax = v;
    }
  }
  if (!isFinite(dataMin)) dataMin = 0;
  if (!isFinite(dataMax)) dataMax = 1;
  const renderMin = autoNormalize ? dataMin : 0;
  const renderMax = autoNormalize ? dataMax : 1;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0 || cols === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(canvasWidth, canvasHeight);
    const pixels = imageData.data;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [cr, cg, cb] = getColor(data[r][c], renderMin, renderMax, colorScale);
        // Fill cell
        for (let py = 0; py < cs; py++) {
          for (let px = 0; px < cs; px++) {
            const idx = ((r * cs + py) * canvasWidth + c * cs + px) * 4;
            pixels[idx] = cr;
            pixels[idx + 1] = cg;
            pixels[idx + 2] = cb;
            pixels[idx + 3] = 255;
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data, rows, cols, cs, canvasWidth, canvasHeight, renderMin, renderMax, colorScale]);

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvasWidth / rect.width;
    const scaleY = canvasHeight / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    const col = Math.floor(mx / cs);
    const row = Math.floor(my / cs);
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      setTooltip({
        x: e.clientX - (containerRef.current?.getBoundingClientRect().left ?? 0),
        y: e.clientY - (containerRef.current?.getBoundingClientRect().top ?? 0),
        row,
        col,
        value: data[row][col],
      });
    } else {
      setTooltip(null);
    }
  }

  function handleMouseLeave() {
    setTooltip(null);
  }

  if (rows === 0 || cols === 0) {
    return <div style={{ color: "#8b949e", fontSize: 12 }}>{title}: no data</div>;
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
          [{rows} x {cols}] range: [{dataMin.toFixed(3)}, {dataMax.toFixed(3)}]
        </span>
      </div>
      {yLabel && (
        <div
          style={{
            position: "absolute",
            left: -18,
            top: canvasHeight / 2 + 20,
            transform: "rotate(-90deg)",
            fontSize: 10,
            color: "#8b949e",
            whiteSpace: "nowrap",
          }}
        >
          {yLabel}
        </div>
      )}
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
      {xLabel && (
        <div style={{ fontSize: 10, color: "#8b949e", textAlign: "center" }}>
          {xLabel}
        </div>
      )}
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
          [{tooltip.row}, {tooltip.col}] = {tooltip.value.toFixed(4)}
        </div>
      )}
    </div>
  );
}
