/**
 * InspectPanel — Step-by-step training inspector.
 *
 * Auto-loads bcl-slow on mount. Condition switcher at the top.
 * Step -> see batch images + loss + metrics table + timeline.
 */
import { useStore } from '@nanostores/react';
import {
  $inspectState,
  $inspectCurrentStep,
  $inspectHistory,
  $inspectViewingStep,
  $inspectSetupLoading,
  $inspectStepLoading,
  stepInspector,
  loadInspectStep,
  ensureSession,
  switchCondition,
} from '../lib/inspect-store';
import { StepTensorKey } from '../lib/inspect-types';
import InspectMetricsTable from './InspectMetricsTable';
import InspectHeatmap from './InspectHeatmap';
import InspectWeightGrid from './InspectWeightGrid';
import { useEffect, useState } from 'react';

const CONDITIONS = ['bcl-slow', 'bcl-med', 'bcl-fast'] as const;
const DEFAULT_CONDITION = 'bcl-slow';

export default function InspectPanel() {
  const state = useStore($inspectState);
  const stepData = useStore($inspectCurrentStep);
  const history = useStore($inspectHistory);
  const viewingStep = useStore($inspectViewingStep);
  const setupLoading = useStore($inspectSetupLoading);
  const stepLoading = useStore($inspectStepLoading);
  const [selectedCondition, setSelectedCondition] = useState(DEFAULT_CONDITION);

  // Auto-setup on mount
  useEffect(() => {
    ensureSession(DEFAULT_CONDITION);
  }, []);

  // Keep dropdown in sync with server state
  useEffect(() => {
    if (state.condition) {
      setSelectedCondition(state.condition);
    }
  }, [state.condition]);

  const isActive = state.active;
  const isLoading = setupLoading;
  const batchImage = stepData?.[StepTensorKey.BATCH_IMAGES] as string | undefined;
  const loss = stepData?.[StepTensorKey.LOSS] as number | undefined;
  const currentStep = stepData?._step ?? -1;

  // Alive feature count for the info bar
  const winRate = stepData?.[StepTensorKey.WIN_RATE] as number[] | undefined;

  const handleConditionChange = (newCondition: string) => {
    setSelectedCondition(newCondition);
    switchCondition(newCondition);
  };

  return (
    <div style={{ padding: '16px' }}>
      {/* Header with condition switcher */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
        <h2 style={{ margin: 0 }}>Step Inspector</h2>
        <select
          value={selectedCondition}
          onChange={(e) => handleConditionChange(e.target.value)}
          disabled={isLoading || stepLoading}
          style={{
            background: 'var(--bg-secondary, #161b22)',
            color: 'var(--fg, #e6edf3)',
            border: '1px solid var(--border, #30363d)',
            borderRadius: '6px',
            padding: '4px 8px',
            fontSize: '13px',
          }}
        >
          {CONDITIONS.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        {isLoading && (
          <span className="text-muted" style={{ fontSize: '13px' }}>Setting up...</span>
        )}
        {isActive && !isLoading && (
          <span className="recipe-state" style={{ color: '#3fb950', fontSize: '13px' }}>
            Ready &middot; Step {state.step}
          </span>
        )}
      </div>

      {/* Step controls */}
      {isActive && !isLoading && (
        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
          <button
            className="btn-action btn-train-start"
            onClick={() => stepInspector(1)}
            disabled={stepLoading}
          >
            {stepLoading ? 'Stepping...' : 'Step'}
          </button>
          <button
            className="btn-action"
            onClick={() => stepInspector(10)}
            disabled={stepLoading}
          >
            {stepLoading ? '...' : 'Step x10'}
          </button>
          <button
            className="btn-action"
            onClick={() => stepInspector(50)}
            disabled={stepLoading}
          >
            {stepLoading ? '...' : 'Step x50'}
          </button>
        </div>
      )}

      {/* Step data display */}
      {isActive && stepData && (
        <div>
          {/* Loss + step info bar */}
          <div className="panel" style={{ marginBottom: '12px', padding: '12px' }}>
            <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
              <div>
                <span className="text-muted">Step</span>{' '}
                <strong>{currentStep}</strong>
              </div>
              <div>
                <span className="text-muted">Loss</span>{' '}
                <strong>{loss !== undefined ? loss.toFixed(6) : '--'}</strong>
              </div>
              <div>
                <span className="text-muted">Keys</span>{' '}
                <span>{(stepData._keys as string[])?.length ?? 0} tensors captured</span>
              </div>
              {winRate && (
                <div>
                  <span className="text-muted">Alive features</span>{' '}
                  <strong>{winRate.filter(w => w > 0.01).length}/64</strong>
                </div>
              )}
            </div>
          </div>

          {/* Timeline / history slider */}
          {history.length > 1 && (
            <div className="panel" style={{ marginBottom: '12px', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span className="text-muted">Timeline</span>
                <input
                  type="range"
                  min={0}
                  max={history.length - 1}
                  value={history.findIndex(h => h.step === viewingStep)}
                  onChange={(e) => {
                    const idx = parseInt(e.target.value);
                    if (idx >= 0 && idx < history.length) {
                      loadInspectStep(history[idx].step);
                    }
                  }}
                  style={{ flex: 1 }}
                />
                <span>Step {viewingStep}</span>
              </div>
              {/* Mini loss chart */}
              <div style={{ height: '60px', display: 'flex', alignItems: 'end', gap: '1px', marginTop: '8px' }}>
                {history.map((h, i) => {
                  const maxLoss = Math.max(...history.map(hh => hh.loss ?? 0), 0.001);
                  const barH = ((h.loss ?? 0) / maxLoss) * 56;
                  const isViewing = h.step === viewingStep;
                  return (
                    <div
                      key={h.step}
                      onClick={() => loadInspectStep(h.step)}
                      title={`Step ${h.step}: loss=${(h.loss ?? 0).toFixed(4)}`}
                      style={{
                        flex: 1,
                        maxWidth: '8px',
                        height: `${Math.max(barH, 2)}px`,
                        background: isViewing ? '#58a6ff' : '#3fb950',
                        cursor: 'pointer',
                        borderRadius: '1px',
                        opacity: isViewing ? 1 : 0.6,
                      }}
                    />
                  );
                })}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2px' }}>
                <span className="text-muted" style={{ fontSize: '11px' }}>Step 0</span>
                <span className="text-muted" style={{ fontSize: '11px' }}>
                  Step {history[history.length - 1]?.step}
                </span>
              </div>
            </div>
          )}

          {/* Batch images */}
          {batchImage && (
            <div className="panel" style={{ padding: '12px' }}>
              <h3 style={{ margin: '0 0 8px 0' }}>Batch Images (128 x 28x28)</h3>
              <img
                src={`data:image/png;base64,${batchImage}`}
                alt="Batch grid"
                style={{ width: '100%', imageRendering: 'pixelated' }}
              />
            </div>
          )}

          {/* Per-feature metrics table (auto-discovers [D]-shaped vectors) */}
          <InspectMetricsTable
            stepData={stepData}
            modelDim={state.model_dim ?? 64}
          />

          {/* M-DBG-3: Viz primitives — Heatmaps + Weight Grids */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', marginTop: '16px' }}>
            {/* Strength [B,D] heatmap */}
            {stepData[StepTensorKey.STRENGTH] && Array.isArray(stepData[StepTensorKey.STRENGTH]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectHeatmap
                  data={stepData[StepTensorKey.STRENGTH] as number[][]}
                  title="Strength [B,D]"
                  xLabel="Features (D=64)"
                  yLabel="Batch (B=128)"
                  colorScale="hot"
                />
              </div>
            )}
            {/* Rank Score [B,D] heatmap */}
            {stepData[StepTensorKey.RANK_SCORE] && Array.isArray(stepData[StepTensorKey.RANK_SCORE]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectHeatmap
                  data={stepData[StepTensorKey.RANK_SCORE] as number[][]}
                  title="Rank Score [B,D]"
                  xLabel="Features (D=64)"
                  yLabel="Batch (B=128)"
                  colorScale="viridis"
                />
              </div>
            )}
            {/* Grad Mask [B,D] heatmap */}
            {stepData[StepTensorKey.GRAD_MASK] && Array.isArray(stepData[StepTensorKey.GRAD_MASK]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectHeatmap
                  data={stepData[StepTensorKey.GRAD_MASK] as number[][]}
                  title="Grad Mask [B,D]"
                  xLabel="Features (D=64)"
                  yLabel="Batch (B=128)"
                  colorScale="hot"
                />
              </div>
            )}
            {/* Local Coverage [B,D] heatmap */}
            {stepData[StepTensorKey.LOCAL_COVERAGE] && Array.isArray(stepData[StepTensorKey.LOCAL_COVERAGE]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectHeatmap
                  data={stepData[StepTensorKey.LOCAL_COVERAGE] as number[][]}
                  title="Local Coverage [B,D]"
                  xLabel="Features (D=64)"
                  yLabel="Batch (B=128)"
                  colorScale="viridis"
                />
              </div>
            )}
            {/* Local Novelty [B,D] heatmap */}
            {stepData[StepTensorKey.LOCAL_NOVELTY] && Array.isArray(stepData[StepTensorKey.LOCAL_NOVELTY]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectHeatmap
                  data={stepData[StepTensorKey.LOCAL_NOVELTY] as number[][]}
                  title="Local Novelty [B,D]"
                  xLabel="Features (D=64)"
                  yLabel="Batch (B=128)"
                  colorScale="viridis"
                />
              </div>
            )}
          </div>

          {/* Weight grids: [D, 784] as 28x28 thumbnails */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', marginTop: '16px' }}>
            {/* Encoder Weights */}
            {stepData[StepTensorKey.ENCODER_WEIGHTS] && Array.isArray(stepData[StepTensorKey.ENCODER_WEIGHTS]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectWeightGrid
                  data={stepData[StepTensorKey.ENCODER_WEIGHTS] as number[][]}
                  title="Encoder Weights"
                  colorMode="grayscale"
                />
              </div>
            )}
            {/* SOM Target — local novelty pull target (THE SOM force) */}
            {stepData[StepTensorKey.SOM_TARGETS] && Array.isArray(stepData[StepTensorKey.SOM_TARGETS]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectWeightGrid
                  data={stepData[StepTensorKey.SOM_TARGETS] as number[][]}
                  title="SOM Target (local novelty pull)"
                  colorMode="diverging"
                />
              </div>
            )}
            {/* Grad Masked — Force 1: SGD gradient on encoder weights after BCL mask */}
            {stepData[StepTensorKey.GRAD_MASKED] && Array.isArray(stepData[StepTensorKey.GRAD_MASKED]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectWeightGrid
                  data={stepData[StepTensorKey.GRAD_MASKED] as number[][]}
                  title="Gradient (Force 1 - masked)"
                  colorMode="diverging"
                />
              </div>
            )}
            {/* SOM Delta — actual weight update from SOM */}
            {stepData[StepTensorKey.SOM_DELTA] && Array.isArray(stepData[StepTensorKey.SOM_DELTA]) && (
              <div className="panel" style={{ padding: '12px' }}>
                <InspectWeightGrid
                  data={stepData[StepTensorKey.SOM_DELTA] as number[][]}
                  title="SOM Delta (actual weight update)"
                  colorMode="diverging"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loading state */}
      {!isActive && !isLoading && (
        <div className="panel" style={{ padding: '24px', textAlign: 'center' }}>
          <div className="empty">
            Connecting to server...
          </div>
        </div>
      )}
    </div>
  );
}
