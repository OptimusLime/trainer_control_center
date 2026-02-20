/**
 * BCLDiagnostics — React island for BCL experiment diagnostics.
 *
 * Three tabs:
 * 1. Signal Scatter: grad_magnitude vs som_magnitude per feature, colored by win_rate.
 *    THE core BCL diagnostic — shows whether winners get gradient and losers get SOM.
 * 2. Win Rate Heatmap: [64 x steps] canvas heatmap, rows = features sorted by final
 *    win rate, columns = steps. Shows feature life stories.
 * 3. Dead Diversity: line plot of bottom-20 mean pairwise cosine similarity over steps.
 *    Tests whether SOM causes blob translation (H3) or dispersion (H6).
 *
 * Data comes from GET /eval/bcl/diagnostics?tag=...&mode=scatter|winrate|diversity
 */
import { useEffect, useState, useRef, useCallback } from 'react';
import {
  Chart,
  ScatterController,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  type ChartConfiguration,
} from 'chart.js';
import { fetchJSON } from '../lib/api';
import type {
  BCLScatterResponse,
  BCLWinRateResponse,
  BCLDeadDiversityResponse,
  BCLScatterEntry,
} from '../lib/types';

Chart.register(
  ScatterController, LineController, LineElement, PointElement,
  LinearScale, Title, Tooltip, Legend,
);

type Tab = 'scatter' | 'winrate' | 'diversity';

// --- Scatter chart helpers ---

function winRateColor(wr: number): string {
  if (wr > 0.05) return 'rgba(63, 185, 80, 0.85)';   // green — winner
  if (wr > 0.01) return 'rgba(227, 179, 65, 0.85)';   // yellow — borderline
  return 'rgba(248, 81, 73, 0.85)';                     // red — dead
}

function buildScatterChart(
  canvas: HTMLCanvasElement,
  entry: BCLScatterEntry,
): Chart {
  const D = entry.grad_magnitude.length;
  const data = [];
  const colors: string[] = [];
  for (let i = 0; i < D; i++) {
    data.push({ x: entry.grad_magnitude[i], y: entry.som_magnitude[i] });
    colors.push(winRateColor(entry.win_rate[i]));
  }

  const config: ChartConfiguration<'scatter'> = {
    type: 'scatter',
    data: {
      datasets: [{
        label: `Step ${entry.step}`,
        data,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.85', '1.0')),
        borderWidth: 1,
        pointRadius: 5,
        pointHoverRadius: 7,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'Gradient Signal (winners)',
            color: '#8b949e',
            font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" },
          },
          ticks: { color: '#8b949e', font: { size: 10 } },
          grid: { color: '#21262d' },
          min: 0,
        },
        y: {
          type: 'linear',
          title: {
            display: true,
            text: 'SOM Signal (losers)',
            color: '#8b949e',
            font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" },
          },
          ticks: { color: '#8b949e', font: { size: 10 } },
          grid: { color: '#21262d' },
          min: 0,
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#161b22',
          titleColor: '#f0f6fc',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {
            label(ctx) {
              const idx = ctx.dataIndex;
              const wr = entry.win_rate[idx];
              return `F${idx}: grad=${ctx.parsed.x.toFixed(3)}, som=${ctx.parsed.y.toFixed(3)}, wr=${(wr * 100).toFixed(1)}%`;
            },
          },
        },
      },
    },
  };
  return new Chart(canvas, config);
}

// --- Win Rate Heatmap (canvas 2D) ---

function drawWinRateHeatmap(
  canvas: HTMLCanvasElement,
  entries: { step: number; win_rate: number[] }[],
) {
  if (entries.length === 0) return;

  const D = entries[0].win_rate.length;
  const steps = entries.length;

  // Sort features by final win rate (highest at top)
  const finalWr = entries[entries.length - 1].win_rate;
  const sortedIndices = Array.from({ length: D }, (_, i) => i)
    .sort((a, b) => finalWr[b] - finalWr[a]);

  const cellW = Math.max(2, Math.floor(canvas.parentElement!.clientWidth / steps));
  const cellH = Math.max(3, Math.min(8, Math.floor(400 / D)));
  canvas.width = cellW * steps;
  canvas.height = cellH * D;

  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let col = 0; col < steps; col++) {
    const wr = entries[col].win_rate;
    for (let row = 0; row < D; row++) {
      const featureIdx = sortedIndices[row];
      const val = wr[featureIdx];
      // Color: dark (val=0) to bright green (val=high)
      // Use log scale to emphasize low values
      const intensity = Math.min(1, val * 5); // scale so wr=0.2 saturates
      const r = Math.round(13 + intensity * 50);
      const g = Math.round(17 + intensity * 168);
      const b = Math.round(23 + intensity * 57);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(col * cellW, row * cellH, cellW, cellH);
    }
  }
}

// --- Dead Diversity line chart ---

function buildDiversityChart(
  canvas: HTMLCanvasElement,
  entries: { step: number; mean_similarity: number }[],
): Chart {
  const data = entries.map(e => ({ x: e.step, y: e.mean_similarity }));

  const config: ChartConfiguration<'line'> = {
    type: 'line',
    data: {
      datasets: [{
        label: 'Dead Feature Cosine Similarity',
        data,
        borderColor: '#d2a8ff',
        backgroundColor: '#d2a8ff20',
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5,
        tension: 0.2,
        fill: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'Step',
            color: '#8b949e',
            font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" },
          },
          ticks: { color: '#8b949e', font: { size: 10 } },
          grid: { color: '#21262d' },
        },
        y: {
          type: 'linear',
          title: {
            display: true,
            text: 'Mean Cosine Similarity (bottom 20)',
            color: '#d2a8ff',
            font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" },
          },
          ticks: { color: '#d2a8ff', font: { size: 10 } },
          grid: { color: '#21262d' },
          min: -0.1,
          max: 1.0,
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#161b22',
          titleColor: '#f0f6fc',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {
            label(ctx) {
              const val = ctx.parsed.y;
              const status = val > 0.3 ? 'blob (bad)' : val > 0.1 ? 'moderate' : 'dispersed (good)';
              return `Similarity: ${val.toFixed(4)} (${status})`;
            },
          },
        },
      },
    },
  };
  return new Chart(canvas, config);
}

// --- Component ---

export default function BCLDiagnostics() {
  const [tags, setTags] = useState<string[]>([]);
  const [selectedTag, setSelectedTag] = useState('');
  const [activeTab, setActiveTab] = useState<Tab>('scatter');
  const [loading, setLoading] = useState(false);

  // Scatter state
  const [scatterData, setScatterData] = useState<BCLScatterResponse | null>(null);
  const [scatterStepIdx, setScatterStepIdx] = useState(0);
  const scatterCanvasRef = useRef<HTMLCanvasElement>(null);
  const scatterChartRef = useRef<Chart | null>(null);

  // Win rate heatmap state
  const [winRateData, setWinRateData] = useState<BCLWinRateResponse | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);

  // Diversity state
  const [diversityData, setDiversityData] = useState<BCLDeadDiversityResponse | null>(null);
  const diversityCanvasRef = useRef<HTMLCanvasElement>(null);
  const diversityChartRef = useRef<Chart | null>(null);

  // Fetch available tags on mount
  useEffect(() => {
    fetchJSON<{ tags: string[] }>('/eval/bcl/diagnostics').then(resp => {
      if (resp?.tags) {
        setTags(resp.tags);
        if (resp.tags.length > 0) setSelectedTag(resp.tags[0]);
      }
    });
  }, []);

  // Load data when tag or tab changes
  const loadData = useCallback(async (tag: string, tab: Tab) => {
    if (!tag) return;
    setLoading(true);
    try {
      if (tab === 'scatter') {
        const resp = await fetchJSON<BCLScatterResponse>(
          `/eval/bcl/diagnostics?tag=${encodeURIComponent(tag)}&mode=scatter`,
        );
        if (resp) {
          setScatterData(resp);
          setScatterStepIdx(resp.entries.length - 1); // default to latest
        }
      } else if (tab === 'winrate') {
        const resp = await fetchJSON<BCLWinRateResponse>(
          `/eval/bcl/diagnostics?tag=${encodeURIComponent(tag)}&mode=winrate`,
        );
        if (resp) setWinRateData(resp);
      } else if (tab === 'diversity') {
        const resp = await fetchJSON<BCLDeadDiversityResponse>(
          `/eval/bcl/diagnostics?tag=${encodeURIComponent(tag)}&mode=diversity`,
        );
        if (resp) setDiversityData(resp);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleTagChange = useCallback((tag: string) => {
    setSelectedTag(tag);
    loadData(tag, activeTab);
  }, [activeTab, loadData]);

  const handleTabChange = useCallback((tab: Tab) => {
    setActiveTab(tab);
    loadData(selectedTag, tab);
  }, [selectedTag, loadData]);

  // --- Scatter chart lifecycle ---
  useEffect(() => {
    if (activeTab !== 'scatter') return;
    if (!scatterData || scatterData.entries.length === 0) return;
    if (!scatterCanvasRef.current) return;

    const entry = scatterData.entries[scatterStepIdx];
    if (!entry) return;

    // Destroy previous chart
    if (scatterChartRef.current) {
      scatterChartRef.current.destroy();
      scatterChartRef.current = null;
    }

    scatterChartRef.current = buildScatterChart(scatterCanvasRef.current, entry);
    return () => {
      if (scatterChartRef.current) {
        scatterChartRef.current.destroy();
        scatterChartRef.current = null;
      }
    };
  }, [activeTab, scatterData, scatterStepIdx]);

  // --- Heatmap draw ---
  useEffect(() => {
    if (activeTab !== 'winrate') return;
    if (!winRateData || winRateData.entries.length === 0) return;
    if (!heatmapCanvasRef.current) return;

    drawWinRateHeatmap(heatmapCanvasRef.current, winRateData.entries);
  }, [activeTab, winRateData]);

  // --- Diversity chart lifecycle ---
  useEffect(() => {
    if (activeTab !== 'diversity') return;
    if (!diversityData || diversityData.entries.length === 0) return;
    if (!diversityCanvasRef.current) return;

    if (diversityChartRef.current) {
      diversityChartRef.current.destroy();
      diversityChartRef.current = null;
    }

    diversityChartRef.current = buildDiversityChart(
      diversityCanvasRef.current,
      diversityData.entries,
    );
    return () => {
      if (diversityChartRef.current) {
        diversityChartRef.current.destroy();
        diversityChartRef.current = null;
      }
    };
  }, [activeTab, diversityData]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      scatterChartRef.current?.destroy();
      diversityChartRef.current?.destroy();
    };
  }, []);

  // No BCL data available — hide panel entirely
  if (tags.length === 0) return null;

  const scatterEntries = scatterData?.entries ?? [];
  const currentScatterEntry = scatterEntries[scatterStepIdx];

  // Count feature categories for scatter legend
  let nWinners = 0, nBorderline = 0, nDead = 0;
  if (currentScatterEntry) {
    for (const wr of currentScatterEntry.win_rate) {
      if (wr > 0.05) nWinners++;
      else if (wr > 0.01) nBorderline++;
      else nDead++;
    }
  }

  const tabStyle = (t: Tab) => ({
    padding: '4px 12px',
    fontSize: '0.8rem',
    fontFamily: "'SF Mono','Menlo','Consolas',monospace",
    border: 'none',
    borderBottom: activeTab === t ? '2px solid #58a6ff' : '2px solid transparent',
    background: 'none',
    color: activeTab === t ? '#f0f6fc' : '#8b949e',
    cursor: 'pointer' as const,
  });

  return (
    <div className="panel" id="bcl-diagnostics-panel">
      <div className="panel-header">
        <h3>BCL Diagnostics</h3>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <select
            className="compare-select"
            value={selectedTag}
            onChange={e => handleTagChange(e.target.value)}
          >
            {tags.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <button
            className="btn-action"
            onClick={() => loadData(selectedTag, activeTab)}
            disabled={loading || !selectedTag}
          >
            {loading ? 'Loading...' : 'Load'}
          </button>
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: '4px', borderBottom: '1px solid #21262d', marginBottom: '8px' }}>
        <button style={tabStyle('scatter')} onClick={() => handleTabChange('scatter')}>
          Signal Scatter
        </button>
        <button style={tabStyle('winrate')} onClick={() => handleTabChange('winrate')}>
          Win Rate Heatmap
        </button>
        <button style={tabStyle('diversity')} onClick={() => handleTabChange('diversity')}>
          Dead Diversity
        </button>
      </div>

      {/* Scatter Tab */}
      {activeTab === 'scatter' && (
        <div>
          {scatterEntries.length > 0 ? (
            <>
              {/* Step selector */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <label style={{ fontSize: '0.75rem', color: '#8b949e' }}>Step:</label>
                <input
                  type="range"
                  min={0}
                  max={scatterEntries.length - 1}
                  value={scatterStepIdx}
                  onChange={e => setScatterStepIdx(Number(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ fontSize: '0.75rem', color: '#c9d1d9', minWidth: '60px' }}>
                  {currentScatterEntry?.step?.toLocaleString() ?? '--'}
                </span>
              </div>
              {/* Legend */}
              <div style={{ display: 'flex', gap: '16px', fontSize: '0.75rem', marginBottom: '6px' }}>
                <span>
                  <span style={{ display: 'inline-block', width: 10, height: 10, background: 'rgba(63,185,80,0.85)', borderRadius: '50%', marginRight: 4 }} />
                  Winners ({nWinners})
                </span>
                <span>
                  <span style={{ display: 'inline-block', width: 10, height: 10, background: 'rgba(227,179,65,0.85)', borderRadius: '50%', marginRight: 4 }} />
                  Borderline ({nBorderline})
                </span>
                <span>
                  <span style={{ display: 'inline-block', width: 10, height: 10, background: 'rgba(248,81,73,0.85)', borderRadius: '50%', marginRight: 4 }} />
                  Dead ({nDead})
                </span>
              </div>
              <div className="chart-container">
                <canvas ref={scatterCanvasRef} />
              </div>
            </>
          ) : (
            <div className="empty">
              {loading ? 'Loading scatter data...' : 'No scatter data. Run a BCL recipe first.'}
            </div>
          )}
        </div>
      )}

      {/* Win Rate Heatmap Tab */}
      {activeTab === 'winrate' && (
        <div>
          {winRateData && winRateData.entries.length > 0 ? (
            <>
              <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: '6px' }}>
                Rows = features (sorted by final win rate, top = highest).
                Columns = steps ({winRateData.entries.length} snapshots).
                Bright = high win rate.
              </div>
              <div style={{ overflowX: 'auto' }}>
                <canvas ref={heatmapCanvasRef} style={{ imageRendering: 'pixelated' }} />
              </div>
            </>
          ) : (
            <div className="empty">
              {loading ? 'Loading win rate data...' : 'No win rate data. Run a BCL recipe first.'}
            </div>
          )}
        </div>
      )}

      {/* Dead Diversity Tab */}
      {activeTab === 'diversity' && (
        <div>
          {diversityData && diversityData.entries.length > 0 ? (
            <>
              <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: '6px' }}>
                Mean pairwise cosine similarity of bottom-20 features by win rate.
                Decreasing = SOM dispersing dead features (good, H6).
                Increasing = blob translation (bad, H3).
              </div>
              <div className="chart-container">
                <canvas ref={diversityCanvasRef} />
              </div>
            </>
          ) : (
            <div className="empty">
              {loading ? 'Loading diversity data...' : 'No diversity data. Run a BCL recipe first.'}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
