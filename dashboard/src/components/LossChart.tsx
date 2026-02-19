/**
 * LossChart — Chart.js React island for live loss curves.
 *
 * Polls /jobs/current and /jobs/{id}/loss_history.
 * On fetch failure: keeps existing chart data (no flashing).
 * Appends new points incrementally — doesn't re-render full history.
 */
import { useEffect, useRef, useCallback } from 'react';
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
import { fetchJSON } from '../lib/api';
import { CHART_COLORS } from '../lib/theme';
import type { JobDict, LossEntry } from '../lib/types';

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

      // Draw dot
      ctx.save();
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = ds.borderColor as string;
      ctx.fill();

      // Draw pill label
      const label = last.y < 0.001 ? last.y.toExponential(2) : last.y.toFixed(4);
      ctx.font = "bold 11px 'SF Mono','Menlo','Consolas',monospace";
      const textW = ctx.measureText(label).width;
      const pillW = textW + 10;
      const pillH = 18;
      const pillX = px - pillW / 2;
      const pillY = py - pillH - 8;

      // Pill background
      ctx.beginPath();
      ctx.roundRect(pillX, pillY, pillW, pillH, 4);
      ctx.fillStyle = '#161b22';
      ctx.fill();
      ctx.strokeStyle = ds.borderColor as string;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Pill text
      ctx.fillStyle = ds.borderColor as string;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, px, pillY + pillH / 2);
      ctx.restore();
    }
  },
};

Chart.register(lastValuePinPlugin);

const POLL_INTERVAL = 5000;

export default function LossChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const lastJobIdRef = useRef<string | null>(null);
  const lastStepRef = useRef<number>(0);

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

  useEffect(() => {
    if (!canvasRef.current) return;
    const chart = buildChart(canvasRef.current);
    chartRef.current = chart;

    const interval = setInterval(async () => {
      // 1. Get current job
      const job = await fetchJSON<JobDict | null>('/jobs/current');
      if (!job) {
        // No job running — could be idle or fetch failed
        // If we had a previous job, try to show its final state
        return;
      }

      // 2. If job changed, reset chart
      if (job.id !== lastJobIdRef.current) {
        chart.data.datasets = [];
        chart.update();
        lastJobIdRef.current = job.id;
        lastStepRef.current = 0;
      }

      // 3. Fetch loss history
      const losses = await fetchJSON<LossEntry[]>(`/jobs/${job.id}/loss_history?max_points=500`);
      if (!losses || losses.length === 0) return;

      // 4. Group by task_name
      const byTask = new Map<string, { x: number; y: number }[]>();
      for (const entry of losses) {
        if (entry.step <= lastStepRef.current && chart.data.datasets.length > 0) continue;
        let arr = byTask.get(entry.task_name);
        if (!arr) {
          arr = [];
          byTask.set(entry.task_name, arr);
        }
        arr.push({ x: entry.step, y: entry.task_loss });
      }

      // 5. If we have new data, update datasets
      if (byTask.size === 0 && chart.data.datasets.length > 0) return;

      // On first load or job change, rebuild datasets entirely
      if (chart.data.datasets.length === 0) {
        // Full rebuild from all losses
        const fullByTask = new Map<string, { x: number; y: number }[]>();
        for (const entry of losses) {
          let arr = fullByTask.get(entry.task_name);
          if (!arr) {
            arr = [];
            fullByTask.set(entry.task_name, arr);
          }
          arr.push({ x: entry.step, y: entry.task_loss });
        }

        let colorIdx = 0;
        for (const [taskName, points] of fullByTask) {
          const color = CHART_COLORS[colorIdx % CHART_COLORS.length];
          chart.data.datasets.push({
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
      } else {
        // Append new points to existing datasets
        for (const [taskName, points] of byTask) {
          const ds = chart.data.datasets.find(d => d.label === taskName);
          if (ds) {
            (ds.data as { x: number; y: number }[]).push(...points);
          } else {
            // New task appeared mid-training
            const color = CHART_COLORS[chart.data.datasets.length % CHART_COLORS.length];
            chart.data.datasets.push({
              label: taskName,
              data: points,
              borderColor: color,
              backgroundColor: color + '20',
              borderWidth: 1.5,
              pointRadius: 0,
              tension: 0.1,
              fill: false,
            });
          }
        }
      }

      // Update last step
      const maxStep = losses.reduce((m, e) => Math.max(m, e.step), 0);
      lastStepRef.current = maxStep;

      chart.update('none');
    }, POLL_INTERVAL);

    // Initial fetch
    (async () => {
      const job = await fetchJSON<JobDict | null>('/jobs/current');
      if (!job) return;
      lastJobIdRef.current = job.id;

      const losses = await fetchJSON<LossEntry[]>(`/jobs/${job.id}/loss_history?max_points=500`);
      if (!losses || losses.length === 0) return;

      const fullByTask = new Map<string, { x: number; y: number }[]>();
      for (const entry of losses) {
        let arr = fullByTask.get(entry.task_name);
        if (!arr) {
          arr = [];
          fullByTask.set(entry.task_name, arr);
        }
        arr.push({ x: entry.step, y: entry.task_loss });
      }

      let colorIdx = 0;
      for (const [taskName, points] of fullByTask) {
        const color = CHART_COLORS[colorIdx % CHART_COLORS.length];
        chart.data.datasets.push({
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

      const maxStep = losses.reduce((m, e) => Math.max(m, e.step), 0);
      lastStepRef.current = maxStep;
      chart.update('none');
    })();

    return () => {
      clearInterval(interval);
      chart.destroy();
    };
  }, [buildChart]);

  return (
    <div className="chart-container">
      <canvas ref={canvasRef} />
    </div>
  );
}
