/**
 * LossChart — Chart.js React island.
 *
 * Subscribes to $lossHistory from the store. No independent fetching.
 * The store handles all data; this component just renders it.
 */
import { useEffect, useRef, useCallback } from 'react';
import { useStore } from '@nanostores/react';
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
  type ChartConfiguration,
} from 'chart.js';
import { CHART_COLORS } from '../lib/theme';
import { $lossHistory } from '../lib/store';
import type { LossEntry } from '../lib/types';

Chart.register(LineController, LineElement, PointElement, LinearScale, Title, Tooltip, Legend, Filler);

/** Chart.js plugin: draws a pill with the last value above the final data point. */
const lastValuePinPlugin = {
  id: 'lastValuePin',
  afterDatasetsDraw(chart: Chart) {
    const { ctx } = chart;
    for (const ds of chart.data.datasets) {
      const data = ds.data as { x: number; y: number }[];
      if (data.length === 0) continue;
      const last = data[data.length - 1];
      const meta = chart.getDatasetMeta(chart.data.datasets.indexOf(ds));
      if (!meta.visible) continue;

      const xScale = chart.scales['x'];
      const yScale = chart.scales['y'];
      if (!xScale || !yScale) continue;

      const px = xScale.getPixelForValue(last.x);
      const py = yScale.getPixelForValue(last.y);

      ctx.save();
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = ds.borderColor as string;
      ctx.fill();

      const label = last.y < 0.001 ? last.y.toExponential(2) : last.y.toFixed(4);
      ctx.font = "bold 11px 'SF Mono','Menlo','Consolas',monospace";
      const textW = ctx.measureText(label).width;
      const pillW = textW + 10;
      const pillH = 18;
      const pillX = px - pillW / 2;
      const pillY = py - pillH - 8;

      ctx.beginPath();
      ctx.roundRect(pillX, pillY, pillW, pillH, 4);
      ctx.fillStyle = '#161b22';
      ctx.fill();
      ctx.strokeStyle = ds.borderColor as string;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = ds.borderColor as string;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, px, pillY + pillH / 2);
      ctx.restore();
    }
  },
};

Chart.register(lastValuePinPlugin);

function buildDatasets(losses: LossEntry[]) {
  const byTask = new Map<string, { x: number; y: number }[]>();
  for (const entry of losses) {
    let arr = byTask.get(entry.task_name);
    if (!arr) {
      arr = [];
      byTask.set(entry.task_name, arr);
    }
    arr.push({ x: entry.step, y: entry.task_loss });
  }

  const datasets: Chart['data']['datasets'] = [];
  let colorIdx = 0;
  for (const [taskName, points] of byTask) {
    const color = CHART_COLORS[colorIdx % CHART_COLORS.length];
    datasets.push({
      label: taskName,
      data: points,
      borderColor: color,
      backgroundColor: color + '20',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.1,
      fill: false,
    });
    colorIdx++;
  }
  return datasets;
}

export default function LossChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const lossHistory = useStore($lossHistory);

  const buildChart = useCallback((canvas: HTMLCanvasElement) => {
    const config: ChartConfiguration<'line'> = {
      type: 'line',
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Step', color: '#8b949e', font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" } },
            ticks: { color: '#8b949e', font: { size: 10 } },
            grid: { color: '#21262d' },
          },
          y: {
            type: 'linear',
            title: { display: true, text: 'Loss', color: '#8b949e', font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" } },
            ticks: { color: '#8b949e', font: { size: 10 } },
            grid: { color: '#21262d' },
          },
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#c9d1d9', font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" }, boxWidth: 12 },
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: '#161b22',
            titleColor: '#f0f6fc',
            bodyColor: '#c9d1d9',
            borderColor: '#30363d',
            borderWidth: 1,
          },
        },
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
      },
    };
    return new Chart(canvas, config);
  }, []);

  // Create chart on mount
  useEffect(() => {
    if (!canvasRef.current) return;
    const chart = buildChart(canvasRef.current);
    chartRef.current = chart;
    return () => { chart.destroy(); };
  }, [buildChart]);

  // Update chart when store data changes
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || lossHistory.length === 0) return;

    chart.data.datasets = buildDatasets(lossHistory);
    chart.update('none');
  }, [lossHistory]);

  return (
    <div className="chart-container">
      <canvas ref={canvasRef} />
    </div>
  );
}
