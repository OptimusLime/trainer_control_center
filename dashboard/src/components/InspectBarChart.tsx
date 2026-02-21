/**
 * InspectBarChart — Canvas-based bar chart for [N] vectors.
 *
 * Renders a horizontal bar chart where each bar represents one element.
 * Works for [B]-shaped (batch) and [D]-shaped (feature) vectors.
 * Canvas-based for performance with 128+ bars.
 *
 * Features:
 * - Hover tooltip showing (index, value)
 * - Configurable color
 * - Auto-scales to max value
 * - Optional log scale for highly skewed distributions
 */
import { useRef, useEffect, useState, useCallback } from 'react';

interface InspectBarChartProps {
  data: number[];
  title: string;
  xLabel?: string;
  yLabel?: string;
  color?: string;
  /** Width of the chart area in pixels */
  width?: number;
  /** Height of the chart area in pixels */
  height?: number;
  /** Use log scale for highly skewed data */
  logScale?: boolean;
}

export default function InspectBarChart({
  data,
  title,
  xLabel,
  yLabel,
  color = '#3fb950',
  width = 512,
  height = 80,
  logScale = false,
}: InspectBarChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  const N = data.length;
  const barWidth = Math.max(1, width / N);
  const canvasWidth = Math.ceil(barWidth * N);

  // Render bars
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvasWidth;
    canvas.height = height;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, canvasWidth, height);

    if (N === 0) return;

    const maxVal = Math.max(...data, 1e-8);
    const minVal = Math.min(...data, 0);

    for (let i = 0; i < N; i++) {
      const val = data[i];
      let barH: number;
      if (logScale) {
        const logMax = Math.log1p(maxVal);
        barH = logMax > 0 ? (Math.log1p(Math.max(0, val)) / logMax) * (height - 2) : 0;
      } else {
        barH = maxVal > 0 ? (Math.max(0, val) / maxVal) * (height - 2) : 0;
      }

      ctx.fillStyle = color;
      ctx.fillRect(
        Math.floor(i * barWidth),
        height - Math.max(1, barH),
        Math.max(1, Math.ceil(barWidth) - (barWidth > 2 ? 1 : 0)),
        Math.max(1, barH),
      );
    }
  }, [data, N, canvasWidth, height, color, logScale]);

  // Hover handler
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvasWidth / rect.width;
      const x = (e.clientX - rect.left) * scaleX;
      const idx = Math.floor(x / barWidth);
      if (idx >= 0 && idx < N) {
        setTooltip({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          text: `[${idx}] = ${data[idx].toFixed(4)}`,
        });
      } else {
        setTooltip(null);
      }
    },
    [data, N, barWidth, canvasWidth],
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  // Summary stats
  const mean = N > 0 ? data.reduce((a, b) => a + b, 0) / N : 0;
  const max = N > 0 ? Math.max(...data) : 0;
  const min = N > 0 ? Math.min(...data) : 0;

  return (
    <div style={{ position: 'relative' }}>
      <div style={{ marginBottom: '4px', display: 'flex', alignItems: 'baseline', gap: '8px' }}>
        <strong style={{ fontSize: '12px' }}>{title}</strong>
        <span style={{ fontSize: '10px', color: '#8b949e' }}>
          N={N} | min={min.toFixed(3)} mean={mean.toFixed(3)} max={max.toFixed(3)}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width: `${width}px`, height: `${height}px`, display: 'block', imageRendering: 'pixelated' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {xLabel && (
        <div style={{ fontSize: '10px', color: '#8b949e', textAlign: 'center', marginTop: '2px' }}>
          {xLabel}
        </div>
      )}
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: tooltip.x + 8,
            top: tooltip.y - 24,
            background: '#1c2128',
            color: '#e6edf3',
            border: '1px solid #30363d',
            borderRadius: '4px',
            padding: '2px 6px',
            fontSize: '11px',
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
            zIndex: 10,
          }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  );
}
