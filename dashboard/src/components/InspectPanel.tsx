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
import InspectBarChart from './InspectBarChart';
import { useEffect, useState } from 'react';

const CONDITIONS = ['bcl-micro', 'bcl-tiny', 'bcl-slow', 'bcl-med', 'bcl-fast'] as const;
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

          {/* === IMAGE NEIGHBORHOOD RESCUE DIAGNOSTICS ===
           * Five components that determine rescue_target for each dead/losing feature.
           * The question: at which step do the columns of weighted_affinity collapse?
           *
           * 1. affinity [B,D]           — Are weights still pointing different directions?
           * 2. image_coverage [B]        — Are some images actually underserved?
           * 3. image_need [B]            — Which images are screaming for help?
           * 4. weighted_affinity [B,D]   — Does multiplying affinity*need produce diverse pull profiles?
           * 5. rescue_pull [B,D]         — Normalized version. If weighted_affinity cols identical, these identical.
           */}
          <div style={{ marginTop: '24px' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#8b949e' }}>
              Image Neighborhood Rescue — 5 Component Diagnostic
            </h3>
            <p style={{ margin: '0 0 12px 0', fontSize: '11px', color: '#6e7681', lineHeight: 1.4 }}>
              Each column is a feature. If all columns look the same, all features get pulled to the same target.
              The moment columns of weighted_affinity collapse = convergence is locked in.
            </p>

            {/* Row 1: Affinity heatmap + bar charts for image_coverage and image_need */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', marginBottom: '16px' }}>
              {/* 1. Affinity [B,D] — cosine sim of each image to each feature's weights */}
              {stepData[StepTensorKey.AFFINITY] && Array.isArray(stepData[StepTensorKey.AFFINITY]) && (
                <div className="panel" style={{ padding: '12px' }}>
                  <InspectHeatmap
                    data={stepData[StepTensorKey.AFFINITY] as number[][]}
                    title="1. Affinity [B,D] — cos(image, feature weights)"
                    xLabel="Features (D=64)"
                    yLabel="Images (B=128)"
                    colorScale="viridis"
                  />
                  <p style={{ fontSize: '10px', color: '#6e7681', margin: '4px 0 0 0' }}>
                    Each column = one feature's affinity profile. Different columns = weights point different directions.
                  </p>
                </div>
              )}

              {/* 2 + 3: Bar charts side by side */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {/* 2. image_coverage [B] — how many features claim each image */}
                {stepData[StepTensorKey.IMAGE_COVERAGE] && Array.isArray(stepData[StepTensorKey.IMAGE_COVERAGE]) && !Array.isArray((stepData[StepTensorKey.IMAGE_COVERAGE] as unknown[])[0]) && (
                  <div className="panel" style={{ padding: '12px' }}>
                    <InspectBarChart
                      data={stepData[StepTensorKey.IMAGE_COVERAGE] as number[]}
                      title="2. Image Coverage [B] — features claiming each image"
                      xLabel="Images (B=128)"
                      color="#58a6ff"
                      width={400}
                      height={60}
                    />
                    <p style={{ fontSize: '10px', color: '#6e7681', margin: '4px 0 0 0' }}>
                      Flat = every image has same coverage, no novelty signal. Varied = some images underserved.
                    </p>
                  </div>
                )}

                {/* 3. image_need [B] — 1/(coverage+1), which images need help */}
                {stepData[StepTensorKey.IMAGE_NEED] && Array.isArray(stepData[StepTensorKey.IMAGE_NEED]) && !Array.isArray((stepData[StepTensorKey.IMAGE_NEED] as unknown[])[0]) && (
                  <div className="panel" style={{ padding: '12px' }}>
                    <InspectBarChart
                      data={stepData[StepTensorKey.IMAGE_NEED] as number[]}
                      title="3. Image Need [B] — 1/(coverage+1), underservedness"
                      xLabel="Images (B=128)"
                      color="#f0883e"
                      width={400}
                      height={60}
                    />
                    <p style={{ fontSize: '10px', color: '#6e7681', margin: '4px 0 0 0' }}>
                      Should be spiky — a few images with very high need, most with low.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Row 2: Weighted Affinity + Rescue Pull heatmaps side by side */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
              {/* 4. weighted_affinity [B,D] — affinity * image_need — THE diagnostic */}
              {stepData[StepTensorKey.WEIGHTED_AFFINITY] && Array.isArray(stepData[StepTensorKey.WEIGHTED_AFFINITY]) && (
                <div className="panel" style={{ padding: '12px', border: '1px solid #f0883e' }}>
                  <InspectHeatmap
                    data={stepData[StepTensorKey.WEIGHTED_AFFINITY] as number[][]}
                    title="4. Weighted Affinity [B,D] — affinity * image_need"
                    xLabel="Features (D=64)"
                    yLabel="Images (B=128)"
                    colorScale="hot"
                  />
                  <p style={{ fontSize: '10px', color: '#f0883e', margin: '4px 0 0 0', fontWeight: 600 }}>
                    KEY: When all columns look the same, all rescue targets converge. This is THE diagnostic.
                  </p>
                </div>
              )}

              {/* 5. rescue_pull [B,D] — normalized weighted_affinity (top-k sparse) */}
              {stepData[StepTensorKey.RESCUE_PULL] && Array.isArray(stepData[StepTensorKey.RESCUE_PULL]) && (
                <div className="panel" style={{ padding: '12px' }}>
                  <InspectHeatmap
                    data={stepData[StepTensorKey.RESCUE_PULL] as number[][]}
                    title="5. Rescue Pull [B,D] — top-k sparse, normalized"
                    xLabel="Features (D=64)"
                    yLabel="Images (B=128)"
                    colorScale="hot"
                  />
                  <p style={{ fontSize: '10px', color: '#6e7681', margin: '4px 0 0 0' }}>
                    Normalized version. If weighted_affinity columns were identical, rescue_pull columns are identical.
                  </p>
                </div>
              )}
            </div>
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
