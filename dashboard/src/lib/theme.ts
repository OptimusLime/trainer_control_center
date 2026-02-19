/**
 * Theme constants — ported from old HTMX dashboard.
 *
 * Single source of truth for colors, metric thresholds, chart palette.
 */

// --- Health ---
export const HEALTH_COLORS: Record<string, string> = {
  healthy:  '#7ee787',
  warning:  '#f0883e',
  critical: '#f85149',
};
export const HEALTH_FALLBACK = '#8b949e';

// --- Status ---
export const STATUS_COLORS: Record<string, string> = {
  running:   '#f0883e',
  completed: '#7ee787',
  stopped:   '#8b949e',
  failed:    '#f85149',
};

// --- Chart palette (cycled for multi-task loss lines) ---
export const CHART_COLORS = [
  '#58a6ff', // blue
  '#7ee787', // green
  '#f0883e', // orange
  '#f778ba', // pink
  '#d2a8ff', // purple
  '#ff7b72', // coral
  '#79c0ff', // light blue
  '#a5d6ff', // pale blue
];

// --- Metric thresholds ---
export interface MetricThreshold {
  higherIsBetter: boolean;
  good: number;
  mid: number;
}

export const METRIC_THRESHOLDS: Record<string, MetricThreshold> = {
  accuracy: { higherIsBetter: true,  good: 0.9,  mid: 0.5  },
  psnr:     { higherIsBetter: true,  good: 25.0, mid: 15.0 },
  l1:       { higherIsBetter: false, good: 0.05, mid: 0.2  },
  mae:      { higherIsBetter: false, good: 0.05, mid: 0.2  },
  mse:      { higherIsBetter: false, good: 0.05, mid: 0.2  },
  kl:       { higherIsBetter: false, good: 5.0,  mid: 20.0 },
};

/** Returns 'good', 'mid', 'bad', or '' for unknown metrics. */
export function metricClass(metric: string, value: number): string {
  const t = METRIC_THRESHOLDS[metric];
  if (!t) return '';
  if (t.higherIsBetter) {
    if (value >= t.good) return 'good';
    if (value >= t.mid) return 'mid';
    return 'bad';
  } else {
    if (value <= t.good) return 'good';
    if (value <= t.mid) return 'mid';
    return 'bad';
  }
}

// --- Trend ---
export const TREND_ICONS: Record<string, { icon: string; color: string }> = {
  improving: { icon: '\u25BC', color: '#7ee787' },
  worsening: { icon: '\u25B2', color: '#f85149' },
  flat:      { icon: '\u25AC', color: '#8b949e' },
};
