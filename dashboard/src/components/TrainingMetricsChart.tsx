/**
 * TrainingMetricsChart — Chart.js React island for training-time metrics.
 *
 * Shows assignment entropy and gradient CV over training steps.
 * Data is sparse (one point per epoch), extracted from LossEntry.training_metrics.
 *
 * Assignment entropy: 1.0 = uniform feature usage, 0.0 = total collapse.
 * Gradient CV: lower = more uniform gradient distribution.
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
  type ChartConfiguration,
} from 'chart.js';
import { $trainingMetrics } from '../lib/store';
import type { LossEntry } from '../lib/types';

Chart.register(LineController, LineElement, PointElement, LinearScale, Title, Tooltip, Legend);

const ENTROPY_COLOR = '#58a6ff';      // blue
const GRAD_CV_COLOR = '#f0883e';      // orange
const DEAD_COLOR = '#f85149';         // red
const STABILITY_COLOR = '#3fb950';    // green
const COVERAGE_CV_COLOR = '#d2a8ff';  // purple
const GRADUATION_COLOR = '#56d364';   // bright green
const GINI_COLOR = '#e3b341';         // yellow
const REPLACEMENT_COLOR = '#79c0ff';  // light blue
const UNREACHABLE_COLOR = '#ff6b6b'; // coral red (BCL)
const SOM_MAG_COLOR = '#7ee787';     // bright green (BCL)

function buildDatasets(entries: LossEntry[]) {
  const entropyPoints: { x: number; y: number }[] = [];
  const gradCvPoints: { x: number; y: number }[] = [];
  const deadPoints: { x: number; y: number }[] = [];
  const stabilityPoints: { x: number; y: number }[] = [];
  const coverageCvPoints: { x: number; y: number }[] = [];
  const graduationPoints: { x: number; y: number }[] = [];
  const giniPoints: { x: number; y: number }[] = [];
  const replacementPoints: { x: number; y: number }[] = [];
  const unreachablePoints: { x: number; y: number }[] = [];
  const somMagPoints: { x: number; y: number }[] = [];

  for (const e of entries) {
    const m = e.training_metrics;
    if (!m) continue;
    if (m.assignment_entropy != null) {
      entropyPoints.push({ x: e.step, y: m.assignment_entropy });
    }
    if (m.gradient_cv != null) {
      gradCvPoints.push({ x: e.step, y: m.gradient_cv });
    }
    if (m.dead_features != null) {
      deadPoints.push({ x: e.step, y: m.dead_features });
    }
    if (m.neighborhood_stability != null) {
      stabilityPoints.push({ x: e.step, y: m.neighborhood_stability });
    }
    if (m.coverage_cv != null) {
      coverageCvPoints.push({ x: e.step, y: m.coverage_cv });
    }
    if (m.explorer_graduations != null) {
      graduationPoints.push({ x: e.step, y: m.explorer_graduations });
    }
    if (m.gini != null) {
      giniPoints.push({ x: e.step, y: m.gini });
    }
    if (m.replacement_count != null && m.replacement_count > 0) {
      replacementPoints.push({ x: e.step, y: m.replacement_count });
    }
    if (m.unreachable_count != null) {
      unreachablePoints.push({ x: e.step, y: m.unreachable_count });
    }
    if (m.som_magnitude_mean != null) {
      somMagPoints.push({ x: e.step, y: m.som_magnitude_mean });
    }
  }

  const datasets: Chart['data']['datasets'] = [];

  if (entropyPoints.length > 0) {
    datasets.push({
      label: 'Assignment Entropy',
      data: entropyPoints,
      borderColor: ENTROPY_COLOR,
      backgroundColor: ENTROPY_COLOR + '20',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.2,
      fill: false,
      yAxisID: 'y',
    });
  }

  if (stabilityPoints.length > 0) {
    datasets.push({
      label: 'Neighborhood Stability',
      data: stabilityPoints,
      borderColor: STABILITY_COLOR,
      backgroundColor: STABILITY_COLOR + '20',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.2,
      fill: false,
      yAxisID: 'y',  // same 0-1 scale as entropy
    });
  }

  if (gradCvPoints.length > 0) {
    datasets.push({
      label: 'Gradient CV',
      data: gradCvPoints,
      borderColor: GRAD_CV_COLOR,
      backgroundColor: GRAD_CV_COLOR + '20',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.2,
      fill: false,
      yAxisID: 'y2',
    });
  }

  if (deadPoints.length > 0) {
    datasets.push({
      label: 'Dead Features',
      data: deadPoints,
      borderColor: DEAD_COLOR,
      backgroundColor: DEAD_COLOR + '20',
      borderWidth: 1.5,
      pointRadius: 2,
      pointHoverRadius: 4,
      tension: 0.2,
      fill: false,
      yAxisID: 'y2',
    });
  }

  if (coverageCvPoints.length > 0) {
    datasets.push({
      label: 'Coverage CV',
      data: coverageCvPoints,
      borderColor: COVERAGE_CV_COLOR,
      backgroundColor: COVERAGE_CV_COLOR + '20',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.2,
      fill: false,
      yAxisID: 'y2',
      borderDash: [5, 3],
    });
  }

  if (graduationPoints.length > 0) {
    datasets.push({
      label: 'Explorer Graduations',
      data: graduationPoints,
      borderColor: GRADUATION_COLOR,
      backgroundColor: GRADUATION_COLOR + '20',
      borderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.2,
      fill: false,
      yAxisID: 'y2',
      pointStyle: 'triangle',
    });
  }

  if (giniPoints.length > 0) {
    datasets.push({
      label: 'Gini (win inequality)',
      data: giniPoints,
      borderColor: GINI_COLOR,
      backgroundColor: GINI_COLOR + '20',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.2,
      fill: false,
      yAxisID: 'y',  // 0-1 scale
      borderDash: [4, 4],
    });
  }

  if (replacementPoints.length > 0) {
    datasets.push({
      label: 'Replacements (cumulative)',
      data: replacementPoints,
      borderColor: REPLACEMENT_COLOR,
      backgroundColor: REPLACEMENT_COLOR + '20',
      borderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0,
      fill: false,
      yAxisID: 'y2',
      pointStyle: 'rectRot',
    });
  }

  if (unreachablePoints.length > 0) {
    datasets.push({
      label: 'Unreachable Features (BCL)',
      data: unreachablePoints,
      borderColor: UNREACHABLE_COLOR,
      backgroundColor: UNREACHABLE_COLOR + '20',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.2,
      fill: false,
      yAxisID: 'y2',
    });
  }

  if (somMagPoints.length > 0) {
    datasets.push({
      label: 'SOM Magnitude (BCL)',
      data: somMagPoints,
      borderColor: SOM_MAG_COLOR,
      backgroundColor: SOM_MAG_COLOR + '20',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.2,
      fill: false,
      yAxisID: 'y2',
      borderDash: [4, 4],
    });
  }

  return datasets;
}

export default function TrainingMetricsChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const entries = useStore($trainingMetrics);

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
            position: 'left',
            title: { display: true, text: 'Entropy (0-1)', color: ENTROPY_COLOR, font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" } },
            ticks: { color: ENTROPY_COLOR, font: { size: 10 } },
            grid: { color: '#21262d' },
            min: 0,
            max: 1.05,
          },
          y2: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Gradient CV / Dead', color: GRAD_CV_COLOR, font: { size: 11, family: "'SF Mono','Menlo','Consolas',monospace" } },
            ticks: { color: GRAD_CV_COLOR, font: { size: 10 } },
            grid: { drawOnChartArea: false },
            min: 0,
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
            callbacks: {
              label(ctx) {
                const label = ctx.dataset.label ?? '';
                const val = ctx.parsed.y;
                if (label === 'Assignment Entropy') {
                  const status = val > 0.85 ? 'healthy' : val > 0.5 ? 'moderate' : 'collapsing';
                  return `${label}: ${val.toFixed(4)} (${status})`;
                }
                if (label === 'Dead Features' || label === 'Explorer Graduations' || label === 'Replacements (cumulative)' || label === 'Unreachable Features (BCL)') {
                  return `${label}: ${val.toFixed(0)}`;
                }
                if (label === 'Coverage CV') {
                  const status = val < 0.3 ? 'uniform' : val < 0.6 ? 'moderate' : 'uneven';
                  return `${label}: ${val.toFixed(4)} (${status})`;
                }
                if (label.startsWith('Gini')) {
                  const status = val < 0.5 ? 'healthy' : val < 0.7 ? 'concentrated' : 'dominated';
                  return `${label}: ${val.toFixed(4)} (${status})`;
                }
                return `${label}: ${val.toFixed(4)}`;
              },
            },
          },
        },
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
      },
    };
    return new Chart(canvas, config);
  }, []);

  const hasData = entries.length > 0;

  // Create chart on mount (canvas always rendered, just hidden)
  useEffect(() => {
    if (!canvasRef.current) return;
    const chart = buildChart(canvasRef.current);
    chartRef.current = chart;
    return () => { chart.destroy(); chartRef.current = null; };
  }, [buildChart]);

  // Update chart when store data changes
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    if (!hasData) {
      chart.data.datasets = [];
      chart.update('none');
      return;
    }

    chart.data.datasets = buildDatasets(entries);
    chart.update('none');
  }, [entries, hasData]);

  return (
    <div className="panel" id="training-metrics-panel"
         style={{ display: hasData ? 'block' : 'none' }}>
      <h3>Gating Metrics</h3>
      <div className="chart-container">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
