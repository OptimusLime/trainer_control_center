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
      {/* Sticky toolbar: condition switcher + step controls + info */}
      <div style={{
        position: 'sticky',
        top: 0,
        zIndex: 20,
        background: 'var(--bg-primary, #0d1117)',
        borderBottom: '1px solid var(--border, #30363d)',
        padding: '8px 0 8px 0',
        marginBottom: '16px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        flexWrap: 'wrap',
      }}>
        <h2 style={{ margin: 0, fontSize: '16px' }}>Inspector</h2>
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
        {isActive && !isLoading && (
          <>
            <button
              className="btn-action btn-train-start"
              onClick={() => stepInspector(1)}
              disabled={stepLoading}
              style={{ padding: '4px 12px', fontSize: '13px' }}
            >
              {stepLoading ? 'Stepping...' : 'Step'}
            </button>
            <button
              className="btn-action"
              onClick={() => stepInspector(10)}
              disabled={stepLoading}
              style={{ padding: '4px 10px', fontSize: '13px' }}
            >
              {stepLoading ? '...' : 'x10'}
            </button>
            <button
              className="btn-action"
              onClick={() => stepInspector(50)}
              disabled={stepLoading}
              style={{ padding: '4px 10px', fontSize: '13px' }}
            >
              {stepLoading ? '...' : 'x50'}
            </button>
            <span style={{ color: '#8b949e', fontSize: '12px' }}>
              Step <strong style={{ color: '#e6edf3' }}>{currentStep >= 0 ? currentStep : state.step}</strong>
            </span>
            {loss !== undefined && (
              <span style={{ color: '#8b949e', fontSize: '12px' }}>
                Loss <strong style={{ color: '#e6edf3' }}>{loss.toFixed(4)}</strong>
              </span>
            )}
            {winRate && (
              <span style={{ color: '#8b949e', fontSize: '12px' }}>
                Alive <strong style={{ color: '#e6edf3' }}>{winRate.filter(w => w > 0.01).length}/64</strong>
              </span>
            )}
            {/* Direct step number input for history navigation */}
            {history.length > 1 && (
              <>
                <span style={{ color: '#30363d' }}>|</span>
                <span style={{ color: '#8b949e', fontSize: '12px' }}>Go to</span>
                <input
                  type="number"
                  min={0}
                  max={history[history.length - 1]?.step ?? 0}
                  defaultValue={viewingStep}
                  key={viewingStep}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const val = parseInt((e.target as HTMLInputElement).value);
                      if (!isNaN(val)) loadInspectStep(val);
                    }
                  }}
                  onBlur={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) loadInspectStep(val);
                  }}
                  style={{
                    width: '60px',
                    background: 'var(--bg-secondary, #161b22)',
                    color: 'var(--fg, #e6edf3)',
                    border: '1px solid var(--border, #30363d)',
                    borderRadius: '4px',
                    padding: '2px 6px',
                    fontSize: '12px',
                    textAlign: 'center',
                  }}
                />
                <span style={{ color: '#8b949e', fontSize: '11px' }}>
                  / {history[history.length - 1]?.step}
                </span>
              </>
            )}
          </>
        )}
        {isLoading && (
          <span className="text-muted" style={{ fontSize: '13px' }}>Setting up...</span>
        )}
      </div>

      {/* Step data display */}
      {isActive && stepData && (
        <div>
          {/* Mini loss timeline (clickable bars) */}
          {history.length > 1 && (
            <div className="panel" style={{ marginBottom: '12px', padding: '8px 12px' }}>
              <div style={{ height: '40px', display: 'flex', alignItems: 'end', gap: '1px' }}>
                {history.map((h) => {
                  const maxLoss = Math.max(...history.map(hh => hh.loss ?? 0), 0.001);
                  const barH = ((h.loss ?? 0) / maxLoss) * 36;
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
