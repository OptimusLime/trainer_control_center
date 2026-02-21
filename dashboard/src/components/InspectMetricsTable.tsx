/**
 * InspectMetricsTable — sortable 64-row per-feature metrics table.
 *
 * Auto-discovers [D]-shaped vectors from the step response.
 * For [B,D] matrices, summarizes to [D] by averaging across the batch.
 * Sortable by clicking column headers. Dead features highlighted red.
 */
import { useState, useMemo } from 'react';
import type { InspectStepResponse } from '../lib/inspect-types';
import { StepTensorKey } from '../lib/inspect-types';

interface Props {
  stepData: InspectStepResponse;
  modelDim?: number;
}

/** Human-readable short labels for known tensor keys. */
const COLUMN_LABELS: Record<string, string> = {
  [StepTensorKey.WIN_RATE]: 'Win Rate',
  [StepTensorKey.FEATURE_NOVELTY]: 'Novelty',
  [StepTensorKey.GRADIENT_WEIGHT]: 'Grad W',
  [StepTensorKey.SOM_WEIGHT_D]: 'SOM W',
  [StepTensorKey.LOCAL_PULL_SUM]: 'Pull Sum',
  [StepTensorKey.IMAGE_COVERAGE]: 'Img Cov',
  [StepTensorKey.STRENGTH]: 'Strength',
  [StepTensorKey.RANK_SCORE]: 'Rank Score',
  [StepTensorKey.GRAD_MASK]: 'Grad Mask',
  [StepTensorKey.IN_NEIGHBORHOOD]: 'In Nbr',
};

/** Keys to show as columns, in display order. Keys not present in the data are skipped. */
const COLUMN_ORDER: string[] = [
  StepTensorKey.WIN_RATE,
  StepTensorKey.FEATURE_NOVELTY,
  StepTensorKey.GRADIENT_WEIGHT,
  StepTensorKey.SOM_WEIGHT_D,
  StepTensorKey.LOCAL_PULL_SUM,
  StepTensorKey.RANK_SCORE,
  StepTensorKey.STRENGTH,
  StepTensorKey.GRAD_MASK,
  StepTensorKey.IN_NEIGHBORHOOD,
];

/** Keys to hide from the table (images, large matrices, etc.) */
const HIDDEN_KEYS = new Set<string>([
  StepTensorKey.BATCH_IMAGES,
  StepTensorKey.BATCH_LABELS,
  StepTensorKey.ENCODER_WEIGHTS,
  StepTensorKey.ENCODER_WEIGHTS_POST,
  StepTensorKey.LOCAL_TARGET,
  StepTensorKey.GLOBAL_TARGET,
  StepTensorKey.SOM_TARGETS,
  StepTensorKey.SOM_DELTA,
  StepTensorKey.GRAD_MASKED,
  StepTensorKey.NEIGHBORS,
  StepTensorKey.LOCAL_COVERAGE,
  StepTensorKey.LOCAL_NOVELTY,
  StepTensorKey.IMAGE_COVERAGE,
  StepTensorKey.AFFINITY,
  StepTensorKey.IMAGE_NEED,
  StepTensorKey.WEIGHTED_AFFINITY,
  StepTensorKey.RESCUE_PULL,
  'loss',
  '_step',
  '_keys',
]);

type SortDir = 'asc' | 'desc';

/**
 * Given a value from the step response, try to extract a [D]-length array.
 * - If it's already number[] with length == D, return as-is.
 * - If it's number[][] (B x D matrix), return mean across rows.
 * - Otherwise return null.
 */
function toPerFeatureArray(val: unknown, D: number): number[] | null {
  if (!Array.isArray(val) || val.length === 0) return null;
  // [D] vector
  if (typeof val[0] === 'number' && val.length === D) {
    return val as number[];
  }
  // [B, D] matrix — average across batch
  if (Array.isArray(val[0]) && (val[0] as number[]).length === D) {
    const B = val.length;
    const result = new Array<number>(D).fill(0);
    for (let b = 0; b < B; b++) {
      const row = val[b] as number[];
      for (let d = 0; d < D; d++) {
        result[d] += row[d];
      }
    }
    for (let d = 0; d < D; d++) {
      result[d] /= B;
    }
    return result;
  }
  return null;
}

function classifyFeature(winRate: number, novelty: number): 'dead' | 'winner' | 'contender' {
  if (winRate < 0.01) return 'dead';
  if (winRate > 0.05 && novelty > 0.1) return 'winner';
  return 'contender';
}

const STATUS_COLORS: Record<string, string> = {
  dead: '#f85149',
  winner: '#3fb950',
  contender: '#d29922',
};

export default function InspectMetricsTable({ stepData, modelDim = 64 }: Props) {
  const [sortKey, setSortKey] = useState<string>(StepTensorKey.WIN_RATE);
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  // Discover columns: extract [D]-vectors from step data
  const columns = useMemo(() => {
    const found: { key: string; label: string; values: number[] }[] = [];
    const seen = new Set<string>();

    // First pass: ordered columns
    for (const key of COLUMN_ORDER) {
      if (HIDDEN_KEYS.has(key)) continue;
      const val = stepData[key];
      const arr = toPerFeatureArray(val, modelDim);
      if (arr) {
        found.push({ key, label: COLUMN_LABELS[key] ?? key, values: arr });
        seen.add(key);
      }
    }

    // Second pass: any remaining [D]-shaped keys not in the explicit order
    for (const key of (stepData._keys ?? [])) {
      if (seen.has(key) || HIDDEN_KEYS.has(key)) continue;
      const val = stepData[key];
      const arr = toPerFeatureArray(val, modelDim);
      if (arr) {
        found.push({ key, label: COLUMN_LABELS[key] ?? key, values: arr });
        seen.add(key);
      }
    }

    return found;
  }, [stepData, modelDim]);

  // Build row data: one row per feature
  const rows = useMemo(() => {
    const winRateCol = columns.find(c => c.key === StepTensorKey.WIN_RATE);
    const noveltyCol = columns.find(c => c.key === StepTensorKey.FEATURE_NOVELTY);

    return Array.from({ length: modelDim }, (_, i) => {
      const wr = winRateCol?.values[i] ?? 0;
      const nov = noveltyCol?.values[i] ?? 0;
      const status = classifyFeature(wr, nov);
      const vals: Record<string, number> = {};
      for (const col of columns) {
        vals[col.key] = col.values[i];
      }
      return { id: i, status, vals };
    });
  }, [columns, modelDim]);

  // Sort rows
  const sortedRows = useMemo(() => {
    const sorted = [...rows];
    sorted.sort((a, b) => {
      const av = a.vals[sortKey] ?? 0;
      const bv = b.vals[sortKey] ?? 0;
      return sortDir === 'desc' ? bv - av : av - bv;
    });
    return sorted;
  }, [rows, sortKey, sortDir]);

  // Summary stats
  const summaryStats = useMemo(() => {
    const stats: Record<string, { mean: number; min: number; max: number }> = {};
    for (const col of columns) {
      const vals = col.values;
      const sum = vals.reduce((a, b) => a + b, 0);
      stats[col.key] = {
        mean: sum / vals.length,
        min: Math.min(...vals),
        max: Math.max(...vals),
      };
    }
    return stats;
  }, [columns]);

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  // Feature status counts
  const deadCount = rows.filter(r => r.status === 'dead').length;
  const winnerCount = rows.filter(r => r.status === 'winner').length;
  const contenderCount = rows.filter(r => r.status === 'contender').length;

  if (columns.length === 0) return null;

  return (
    <div className="panel" style={{ padding: '12px', marginTop: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '8px' }}>
        <h3 style={{ margin: 0 }}>Per-Feature Metrics ({modelDim} features)</h3>
        <span style={{ fontSize: '12px' }}>
          <span style={{ color: STATUS_COLORS.winner }}>{winnerCount} winners</span>
          {' / '}
          <span style={{ color: STATUS_COLORS.contender }}>{contenderCount} contenders</span>
          {' / '}
          <span style={{ color: STATUS_COLORS.dead }}>{deadCount} dead</span>
        </span>
      </div>
      <div style={{ overflowX: 'auto', maxHeight: '500px', overflowY: 'auto' }}>
        <table className="loss-summary-table" style={{ fontSize: '11px', width: '100%' }}>
          <thead>
            <tr>
              <th
                style={{ cursor: 'pointer', userSelect: 'none', width: '36px' }}
                onClick={() => { setSortKey('_id'); setSortDir('asc'); }}
              >
                ID
              </th>
              <th style={{ width: '48px' }}>Status</th>
              {columns.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{
                    cursor: 'pointer',
                    userSelect: 'none',
                    whiteSpace: 'nowrap',
                    background: sortKey === col.key ? 'rgba(88, 166, 255, 0.1)' : undefined,
                  }}
                >
                  {col.label}
                  {sortKey === col.key && (sortDir === 'desc' ? ' ▼' : ' ▲')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {/* Summary row */}
            <tr style={{ borderTop: '1px solid var(--border, #30363d)', fontWeight: 600, fontSize: '10px' }}>
              <td colSpan={2} style={{ color: 'var(--fg-muted, #8b949e)' }}>mean</td>
              {columns.map(col => (
                <td key={col.key} style={{ color: 'var(--fg-muted, #8b949e)' }}>
                  {summaryStats[col.key]?.mean.toFixed(4)}
                </td>
              ))}
            </tr>
            {/* Data rows */}
            {sortedRows.map(row => (
              <tr key={row.id} style={{ color: STATUS_COLORS[row.status] }}>
                <td>{row.id}</td>
                <td style={{ fontSize: '10px' }}>{row.status}</td>
                {columns.map(col => {
                  const v = row.vals[col.key];
                  return (
                    <td key={col.key} style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {v !== undefined ? v.toFixed(4) : '--'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
