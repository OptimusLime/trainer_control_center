/**
 * IecPanel — Interactive Evolutionary CPPN control panel.
 *
 * Layout:
 *   - Top bar: stats + training controls + reset
 *   - Loss chart (SVG line)
 *   - Reconstructions (input/output pairs) with normalize toggle
 *   - Architecture: encoder and decoder shown side by side,
 *     each layer shows resolution labels (e.g. "28->14"),
 *     channel pills with click-to-change activation,
 *     + button to add channel, x to remove,
 *     Add Layer / Remove Layer buttons per side.
 *   - Undo button
 */
import { useStore } from '@nanostores/react';
import {
  $iecState,
  $iecReconstructions,
  $iecSetupLoading,
  $iecStepLoading,
  $iecToast,
  $iecLossHistory,
  $iecNormalize,
  $iecCheckpoints,
  $iecFeatureMaps,
  $iecTasks,
  ensureIecSession,
  stepIec,
  mutateIec,
  undoIec,
  resetIec,
  toggleNormalize,
  saveCheckpoint,
  loadCheckpoint,
  fetchFeatureMaps,
  setSsimWeight,
  setTaskConfig,
  setKernel,
  fetchKernelPresets,
} from '../lib/iec-store';
import { useEffect, useLayoutEffect, useState, useRef, useCallback } from 'react';
import type { IecLayerGenome, IecChannelDescriptor, IecLayerResolution, IecFeatureMaps, IecFeatureLayer, IecTaskConfig } from '../lib/iec-types';
import IecArchGraph from './IecArchGraph';
import type { ChannelSelection } from './IecArchGraph';

/** Identifies which kernel is being edited. */
interface KernelEditTarget {
  side: 'encoder' | 'decoder';
  layerIdx: number;
  channelIdx: number;
  kernelIdx: number;   // which input connection (index into connected inputs)
  inCh: number;        // actual input channel index in the weight tensor
  values: number[][];  // current kernel values (copied for editing)
}

export default function IecPanel() {
  const state = useStore($iecState);
  const reconstructions = useStore($iecReconstructions);
  const setupLoading = useStore($iecSetupLoading);
  const busy = useStore($iecStepLoading);
  const toast = useStore($iecToast);
  const lossHistory = useStore($iecLossHistory);
  const normalize = useStore($iecNormalize);
  const checkpoints = useStore($iecCheckpoints);
  const featureMaps = useStore($iecFeatureMaps);
  const tasks = useStore($iecTasks);
  const [lr, setLr] = useState('0.003');
  const [cpTag, setCpTag] = useState('');
  const [selectedChannel, setSelectedChannel] = useState<ChannelSelection | null>(null);
  const [normalizeActivations, setNormalizeActivations] = useState(true);
  const [kernelEdit, setKernelEdit] = useState<KernelEditTarget | null>(null);
  const [kernelPresets, setKernelPresets] = useState<Record<string, number[][]> | null>(null);

  // useLayoutEffect fires before paint — avoids flash of "No IEC session"
  useLayoutEffect(() => { ensureIecSession(); }, []);

  // Fetch kernel presets once on mount
  useEffect(() => {
    fetchKernelPresets().then(p => { if (p) setKernelPresets(p); });
  }, []);

  if (setupLoading) return <div style={{ color: '#8b949e', padding: 20 }}>Setting up IEC session...</div>;
  if (!state.active) return (
    <div style={{ color: '#8b949e', padding: 20 }}>
      No IEC session active.{' '}
      <button onClick={() => ensureIecSession()} style={{
        background: '#21262d', border: '1px solid #30363d', borderRadius: 4,
        color: '#58a6ff', padding: '4px 12px', fontSize: 12, cursor: 'pointer',
      }}>
        Connect
      </button>
    </div>
  );

  const genome = state.genome;
  const encLayers = genome?.encoder_layers ?? [];
  const decLayers = genome?.decoder_layers ?? [];
  const activations = state.activation_names ?? [];
  const resolutions = state.resolutions;

  return (
    <div style={{ padding: '12px 0', position: 'relative' }}>
      {toast && <Toast message={toast} />}

      {/* Top bar: stats + train + reset — sticky so SGD controls are always accessible */}
      <div style={{
        position: 'sticky', top: 0, zIndex: 20,
        background: '#0d1117', borderBottom: '1px solid #30363d',
        padding: '6px 0', marginBottom: 8,
        display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
      }}>
        <Stat label="Step" value={state.step} />
        <Stat label="Loss" value={state.last_loss != null ? state.last_loss.toFixed(4) : '--'} />
        <Stat label="Latent" value={state.latent_dim} />

        <div style={{ borderLeft: '1px solid #30363d', height: 20, margin: '0 2px' }} />

        <label style={{ fontSize: 12, color: '#8b949e', display: 'flex', alignItems: 'center', gap: 4 }}>
          lr
          <input type="number" value={lr} onChange={e => setLr(e.target.value)}
            step="0.001" min="0.0001" max="1" style={inputStyle} />
        </label>
        <label style={{ fontSize: 12, color: '#8b949e', display: 'flex', alignItems: 'center', gap: 4 }}>
          SSIM
          <input type="number" value={state.ssim_weight} onChange={e => setSsimWeight(parseFloat(e.target.value) || 0)}
            step="0.1" min="0" max="5" style={{ ...inputStyle, width: 50 }} />
        </label>

        <div style={{ borderLeft: '1px solid #30363d', height: 20, margin: '0 2px' }} />

        {/* Structural losses — inline checkboxes */}
        {(() => {
          const sparsity = tasks.find(t => t.name === 'lifetime_sparsity');
          const exclusivity = tasks.find(t => t.name === 'exclusivity');
          return <>
            {sparsity && (
              <label style={{ fontSize: 11, color: sparsity.enabled ? '#e6edf3' : '#484f58', display: 'flex', alignItems: 'center', gap: 3, cursor: 'pointer', userSelect: 'none' }}>
                <input type="checkbox" checked={sparsity.enabled} disabled={busy}
                  onChange={() => setTaskConfig('lifetime_sparsity', { enabled: !sparsity.enabled })}
                  style={{ margin: 0, width: 11, height: 11 }} />
                Sparsity
                {sparsity.enabled && (
                  <input type="number" value={sparsity.weight} disabled={busy}
                    onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v) && v >= 0) setTaskConfig('lifetime_sparsity', { weight: v }); }}
                    step="0.1" min="0" max="10" style={{ ...inputStyle, width: 38, fontSize: 10, padding: '1px 3px' }} />
                )}
              </label>
            )}
            {exclusivity && (
              <label style={{ fontSize: 11, color: exclusivity.enabled ? '#e6edf3' : '#484f58', display: 'flex', alignItems: 'center', gap: 3, cursor: 'pointer', userSelect: 'none' }}>
                <input type="checkbox" checked={exclusivity.enabled} disabled={busy}
                  onChange={() => setTaskConfig('exclusivity', { enabled: !exclusivity.enabled })}
                  style={{ margin: 0, width: 11, height: 11 }} />
                Exclusivity
                {exclusivity.enabled && (
                  <input type="number" value={exclusivity.weight} disabled={busy}
                    onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v) && v >= 0) setTaskConfig('exclusivity', { weight: v }); }}
                    step="0.1" min="0" max="10" style={{ ...inputStyle, width: 38, fontSize: 10, padding: '1px 3px' }} />
                )}
              </label>
            )}
          </>;
        })()}

        {[1, 10, 50, 100, 500].map(n => (
          <button key={n} onClick={() => stepIec(n, parseFloat(lr))} disabled={busy}
            style={busy ? { ...btn, opacity: 0.5 } : btn}>
            x{n}
          </button>
        ))}
        {busy && <span style={{ fontSize: 11, color: '#58a6ff' }}>training...</span>}

        <div style={{ flex: 1 }} />
        <button onClick={() => undoIec()} disabled={busy || state.undo_depth === 0}
          style={state.undo_depth === 0 ? { ...btn, opacity: 0.3 } : { ...btn, borderColor: '#d29922', color: '#d29922' }}>
          Undo ({state.undo_depth})
        </button>
        <button onClick={() => { if (confirm('Reset?')) resetIec(); }}
          style={{ ...btn, borderColor: '#f85149', color: '#f85149' }}>
          Reset
        </button>
      </div>

      {/* Loss chart */}
      {lossHistory.length > 1 && <LossChart losses={lossHistory} />}

      {/* Reconstructions */}
      <div style={{ ...card, marginBottom: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <span style={dimText}>Reconstructions (top=input, bottom=output)</span>
          <label style={{ ...dimText, cursor: 'pointer', userSelect: 'none' }}>
            <input type="checkbox" checked={normalize} onChange={toggleNormalize} style={{ marginRight: 3 }} />
            Normalize
          </label>
        </div>
        {reconstructions && reconstructions.inputs.length > 0 ? (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {reconstructions.inputs.map((inp, i) => (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <img src={`data:image/png;base64,${inp}`} width={56} height={56}
                  style={{ imageRendering: 'pixelated', border: '1px solid #30363d', background: '#000', display: 'block' }} />
                <img src={`data:image/png;base64,${reconstructions.outputs[i]}`} width={56} height={56}
                  style={{ imageRendering: 'pixelated', border: '1px solid #58a6ff', background: '#000', display: 'block' }} />
              </div>
            ))}
          </div>
        ) : (
          <div style={{ color: '#484f58', fontSize: 13 }}>Click a step button to train.</div>
        )}
      </div>

      {/* Architecture Graph (DAG) + Feature Maps inline below */}
      {genome && (
        <div style={{ background: '#0d1117', border: '1px solid #21262d', borderRadius: 4, marginBottom: 8 }}>
          <IecArchGraph
            genome={genome}
            resolutions={resolutions ?? null}
            selected={selectedChannel}
            onSelectChannel={setSelectedChannel}
          />
          {/* Feature maps directly beneath the graph */}
          <div style={{ padding: '4px 10px 8px', borderTop: '1px solid #21262d' }}>
            {featureMaps ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                  <span style={{ ...dimText, fontWeight: 600, fontSize: 10 }}>Feature Maps</span>
                  <span style={{ fontSize: 10, color: '#8b949e', fontFamily: 'monospace' }}>
                    {(state?.loss_fn ?? 'mse').toUpperCase()} <b style={{ color: '#e6edf3' }}>{featureMaps.l1.toFixed(4)}</b>
                    {featureMaps.loss !== featureMaps.l1 && (
                      <> total <b style={{ color: '#e6edf3' }}>{featureMaps.loss.toFixed(4)}</b></>
                    )}
                  </span>
                  <button disabled={busy}
                    onClick={() => fetchFeatureMaps()}
                    style={{ ...btn, fontSize: 9, padding: '1px 5px' }}>
                    Refresh
                  </button>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 9, color: '#8b949e', cursor: 'pointer', userSelect: 'none' }}>
                    <input type="checkbox" checked={normalizeActivations}
                      onChange={e => setNormalizeActivations(e.target.checked)}
                      style={{ margin: 0, width: 10, height: 10 }} />
                    normalize
                  </label>
                </div>
                <FeatureMapDisplay maps={featureMaps} selectedChannel={selectedChannel} normalizeActivations={normalizeActivations}
                  genome={genome}
                  onKernelClick={(side, layerIdx, channelIdx, kernelIdx, inCh, values) => {
                    setKernelEdit({ side, layerIdx, channelIdx, kernelIdx, inCh, values: values.map(r => [...r]) });
                  }} />
              </>
            ) : (
              <div style={{ ...dimText, fontSize: 10 }}>Loading feature maps...</div>
            )}
          </div>
        </div>
      )}

      {/* Kernel Editor — appears when a kernel is clicked in the feature map strip */}
      {kernelEdit && (
        <KernelEditor
          target={kernelEdit}
          presets={kernelPresets}
          busy={busy}
          onApply={async (values, autoFreeze) => {
            await setKernel(kernelEdit.side, kernelEdit.layerIdx, kernelEdit.channelIdx, kernelEdit.inCh, values, autoFreeze);
            setKernelEdit(null);
            await fetchFeatureMaps();
          }}
          onClose={() => setKernelEdit(null)}
          onChange={(values) => setKernelEdit({ ...kernelEdit, values })}
        />
      )}

      {/* Architecture: Encoder + Decoder side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
        <LayerStack
          title="Encoder"
          subtitle="image 28x28 -> bottleneck 3x3"
          layers={encLayers}
          side="encoder"
          activations={activations}
          busy={busy}
          isDecoder={false}
          layerResolutions={resolutions?.encoder ?? null}
        />
        <LayerStack
          title="Decoder"
          subtitle="bottleneck 3x3 -> output 28x28"
          layers={decLayers}
          side="decoder"
          activations={activations}
          busy={busy}
          isDecoder={true}
          layerResolutions={resolutions?.decoder ?? null}
        />
      </div>

      {/* Checkpoints */}
      <div style={{ ...card, marginBottom: 8 }}>
        <div style={{ ...dimText, fontWeight: 600, marginBottom: 6 }}>Checkpoints</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <input type="text" placeholder="tag" value={cpTag} onChange={e => setCpTag(e.target.value)}
            style={{ ...inputStyle, width: 120 }} />
          <button disabled={busy || !cpTag.trim()}
            onClick={() => { saveCheckpoint(cpTag.trim()); setCpTag(''); }}
            style={cpTag.trim() ? { ...btn, borderColor: '#238636', color: '#3fb950' } : { ...btn, opacity: 0.4 }}>
            Save
          </button>
          <div style={{ borderLeft: '1px solid #30363d', height: 20, margin: '0 2px' }} />
          {checkpoints.length > 0 ? checkpoints.map(cp => (
            <button key={cp.id} disabled={busy}
              onClick={() => loadCheckpoint(cp.id)}
              style={{ ...btn, fontSize: 10 }}
              title={`Load: ${cp.tag} (step ${cp.metrics?.iec_step ?? '?'})`}>
              {cp.tag}
            </button>
          )) : (
            <span style={{ ...dimText, fontSize: 10 }}>No checkpoints saved</span>
          )}
        </div>
      </div>

      {/* Feature Maps now shown inline under the architecture graph above */}

      {/* Genome JSON collapse */}
      <details style={{ color: '#8b949e', fontSize: 11 }}>
        <summary style={{ cursor: 'pointer' }}>Genome JSON</summary>
        <pre style={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: 4,
          padding: 8, fontSize: 10, overflow: 'auto', maxHeight: 180, color: '#e6edf3', marginTop: 4 }}>
          {JSON.stringify(genome, null, 2)}
        </pre>
      </details>
    </div>
  );
}

/* -- Layer Stack (encoder or decoder) -- */

function LayerStack({ title, subtitle, layers, side, activations, busy, isDecoder, layerResolutions }: {
  title: string;
  subtitle: string;
  layers: IecLayerGenome[];
  side: 'encoder' | 'decoder';
  activations: string[];
  busy: boolean;
  isDecoder: boolean;
  layerResolutions: IecLayerResolution[] | null;
}) {
  const canRemoveLayer = layers.length > 1;

  return (
    <div style={card}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <div>
          <div style={{ ...dimText, fontWeight: 600, fontSize: 12 }}>{title}</div>
          <div style={{ ...dimText, fontSize: 10, marginTop: 1 }}>{subtitle}</div>
        </div>
        <div style={{ flex: 1 }} />
        {/* Add Layer — pinned top-right so it never moves */}
        <button
          disabled={busy}
          onClick={() => mutateIec('add_layer', {
            side,
            position: isDecoder ? layers.length - 1 : -1,
            activation: 'relu',
            channels: 2,
          })}
          style={{ ...btn, fontSize: 10, padding: '2px 8px', color: '#58a6ff', borderColor: '#1f6feb' }}
        >
          + Layer
        </button>
      </div>

      {layers.map((layer, li) => {
        const isLastDec = isDecoder && li === layers.length - 1;
        const res = layerResolutions?.[li] ?? null;
        return (
          <LayerRow
            key={li}
            layer={layer}
            side={side}
            layerIdx={li}
            activations={activations}
            busy={busy}
            canRemoveChannels={layer.channel_descriptors.length > 1 && !isLastDec}
            isLastDecoder={isLastDec}
            resolution={res}
            canRemoveLayer={canRemoveLayer && !isLastDec}
          />
        );
      })}
    </div>
  );
}

/* -- Single Layer Row -- */

function LayerRow({ layer, side, layerIdx, activations, busy, canRemoveChannels, isLastDecoder, resolution, canRemoveLayer }: {
  layer: IecLayerGenome;
  side: 'encoder' | 'decoder';
  layerIdx: number;
  activations: string[];
  busy: boolean;
  canRemoveChannels: boolean;
  isLastDecoder: boolean;
  resolution: IecLayerResolution | null;
  canRemoveLayer: boolean;
}) {
  const [addAct, setAddAct] = useState('relu');

  const resLabel = resolution
    ? `${resolution.input_res}x${resolution.input_res} -> ${resolution.output_res}x${resolution.output_res}`
    : '';
  const layerLabel = isLastDecoder ? 'Output' : `L${layerIdx}`;

  return (
    <div style={{ marginBottom: 8, borderBottom: '1px solid #21262d', paddingBottom: 6 }}>
      {/* Layer header with resolution */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: '#58a6ff', fontWeight: 600, fontFamily: 'monospace' }}>
          {layerLabel}
        </span>
        {resLabel && (
          <span style={{ fontSize: 10, color: '#8b949e', fontFamily: 'monospace' }}>
            {resLabel}
          </span>
        )}
        <span style={{ fontSize: 10, color: '#484f58' }}>
          {layer.kernel_size}x{layer.kernel_size}
          {layer.stride > 1 ? ` s${layer.stride}` : ''}
          {' '}
          {layer.channel_descriptors.length}ch
        </span>
        {/* Coord toggle — per-layer XY+Gauss injection */}
        <button
          disabled={busy}
          onClick={() => mutateIec('toggle_coords', { side, layer_idx: layerIdx })}
          style={{
            ...pillBtn,
            fontSize: 9,
            padding: '1px 6px',
            color: layer.use_coords ? '#3fb950' : '#484f58',
            borderColor: layer.use_coords ? '#238636' : '#30363d',
            background: layer.use_coords ? '#1a2e1a' : '#21262d',
          }}
          title={layer.use_coords ? 'Coords ON — click to disable XY+Gauss input' : 'Coords OFF — click to enable XY+Gauss input'}
        >
          XY
        </button>
        <div style={{ flex: 1 }} />
        {canRemoveLayer && (
          <button
            disabled={busy}
            onClick={() => mutateIec('remove_layer', { side, layer_idx: layerIdx })}
            style={{ ...pillBtn, fontSize: 9, padding: '1px 5px', color: '#f85149', borderColor: '#f8514966' }}
            title="Remove this layer"
          >
            remove layer
          </button>
        )}
      </div>

      {/* Add channel controls — pinned above pills so they don't move */}
      {!isLastDecoder && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 2, marginBottom: 3 }}>
          <select value={addAct} onChange={e => setAddAct(e.target.value)}
            style={{ ...selectStyle, fontSize: 10, padding: '1px 4px' }}>
            {activations.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
          <button disabled={busy}
            onClick={() => mutateIec('add_channel', { side, layer_idx: layerIdx, activation: addAct })}
            style={{ ...pillBtn, background: '#1f6feb22', borderColor: '#1f6feb', color: '#58a6ff' }}
            title={`Add ${addAct} channel`}>
            + ch
          </button>
        </div>
      )}

      {/* Channel pills */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
        {layer.channel_descriptors.map((ch, ci) => (
          <ChannelPill
            key={ci}
            ch={ch}
            side={side}
            layerIdx={layerIdx}
            channelIdx={ci}
            activations={activations}
            busy={busy}
            canRemove={canRemoveChannels}
          />
        ))}
      </div>
    </div>
  );
}

/* -- Channel Pill -- */

function ChannelPill({ ch, side, layerIdx, channelIdx, activations, busy, canRemove }: {
  ch: IecChannelDescriptor;
  side: 'encoder' | 'decoder';
  layerIdx: number;
  channelIdx: number;
  activations: string[];
  busy: boolean;
  canRemove: boolean;
}) {
  const [open, setOpen] = useState(false);
  const frozen = ch.frozen ?? false;

  return (
    <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', gap: 0 }}>
      {/* Activation name button -- click to open popover */}
      <button onClick={() => setOpen(!open)} disabled={busy}
        style={{
          ...pillBtn,
          borderRadius: (frozen || canRemove) ? '4px 0 0 4px' : 4,
          background: actColor(ch.activation),
          borderColor: frozen ? '#79c0ff' : '#30363d',
        }}>
        {ch.activation}
      </button>

      {/* Freeze toggle */}
      <button disabled={busy}
        onClick={() => mutateIec('toggle_freeze', { side, layer_idx: layerIdx, channel_idx: channelIdx })}
        style={{
          ...pillBtn,
          borderRadius: canRemove ? 0 : '0 4px 4px 0',
          borderLeft: 'none',
          fontSize: 10,
          padding: '2px 4px',
          color: frozen ? '#79c0ff' : '#484f58',
          background: frozen ? '#1a2940' : 'transparent',
          borderColor: frozen ? '#79c0ff' : '#30363d',
        }}
        title={frozen ? 'Frozen — click to unfreeze' : 'Click to freeze (SGD will skip this channel)'}>
        {frozen ? '*' : '.'}
      </button>

      {/* Remove button */}
      {canRemove && (
        <button disabled={busy}
          onClick={() => mutateIec('remove_channel', { side, layer_idx: layerIdx, channel_idx: channelIdx })}
          style={{ ...pillBtn, borderRadius: '0 4px 4px 0', borderLeft: 'none', color: '#f85149', fontSize: 10, padding: '2px 4px' }}
          title="Remove channel">
          x
        </button>
      )}

      {/* Change activation popover */}
      {open && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, zIndex: 10, marginTop: 2,
          background: '#161b22', border: '1px solid #30363d', borderRadius: 4,
          padding: 4, display: 'flex', flexWrap: 'wrap', gap: 2, width: 180,
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
        }}>
          {activations.map(a => (
            <button key={a} disabled={busy}
              onClick={() => {
                if (a !== ch.activation) {
                  mutateIec('change_activation', { side, layer_idx: layerIdx, channel_idx: channelIdx, new_activation: a });
                }
                setOpen(false);
              }}
              style={{
                ...pillBtn,
                background: a === ch.activation ? '#30363d' : actColor(a),
                fontWeight: a === ch.activation ? 700 : 400,
              }}>
              {a}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* -- Loss Line Chart (SVG) -- */

function LossChart({ losses }: { losses: number[] }) {
  const W = 880;
  const H = 100;
  const PL = 48, PR = 8, PT = 14, PB = 4;
  const pW = W - PL - PR, pH = H - PT - PB;

  let pts = losses;
  if (pts.length > pW) {
    const s = Math.ceil(pts.length / pW);
    pts = pts.filter((_, i) => i % s === 0 || i === pts.length - 1);
  }
  const mn = Math.min(...pts), mx = Math.max(...pts), rng = mx - mn || 1;
  const x = (i: number) => PL + (i / Math.max(pts.length - 1, 1)) * pW;
  const y = (v: number) => PT + pH - ((v - mn) / rng) * pH;
  const d = pts.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const ticks = [mx, mn + rng * 0.5, mn];

  return (
    <div style={{ background: '#0d1117', border: '1px solid #21262d', borderRadius: 4, marginBottom: 8, padding: '2px 4px' }}>
      <svg width={W} height={H} style={{ display: 'block' }}>
        {ticks.map((v, i) => (
          <g key={i}>
            <line x1={PL} x2={W - PR} y1={y(v)} y2={y(v)} stroke="#21262d" />
            <text x={PL - 3} y={y(v) + 3} textAnchor="end" fill="#484f58" fontSize={9} fontFamily="monospace">{v.toFixed(3)}</text>
          </g>
        ))}
        <path d={d} fill="none" stroke="#58a6ff" strokeWidth={1.5} />
        <text x={W - PR} y={10} textAnchor="end" fill="#8b949e" fontSize={10} fontFamily="monospace">
          {losses.length} steps &middot; {losses[losses.length - 1].toFixed(4)}
        </text>
      </svg>
    </div>
  );
}

/* -- Feature Map Display -- */

/**
 * Horizontal feature map strip — each layer is a column in a flex row.
 * Columns size to their content so nothing overlaps.
 */
function FeatureMapDisplay({ maps, selectedChannel, normalizeActivations, genome, onKernelClick }: {
  maps: IecFeatureMaps;
  selectedChannel: ChannelSelection | null;
  normalizeActivations: boolean;
  genome: { encoder_layers: IecLayerGenome[]; decoder_layers: IecLayerGenome[] } | null;
  onKernelClick?: (side: 'encoder' | 'decoder', layerIdx: number, channelIdx: number, kernelIdx: number, inCh: number, values: number[][]) => void;
}) {
  // Build ordered slots: Input | E0..En | Latent | D0..Dn(Out)
  const slots: { layer: IecFeatureLayer | null; side: 'encoder' | 'decoder' | null; layerIdx: number; label: string }[] = [];

  slots.push({ layer: null, side: null, layerIdx: -1, label: 'input' });
  for (let i = 0; i < maps.encoder.length; i++) {
    slots.push({ layer: maps.encoder[i], side: 'encoder', layerIdx: i, label: `E${i}` });
  }
  slots.push({ layer: maps.latent ?? null, side: null, layerIdx: -1, label: 'latent' });
  for (let i = 0; i < maps.decoder.length; i++) {
    slots.push({ layer: maps.decoder[i], side: 'decoder', layerIdx: i, label: maps.decoder.length - 1 === i ? 'Out' : `D${i}` });
  }

  return (
    <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
      {slots.map((slot, si) => {
        const highlight = selectedChannel && slot.side === selectedChannel.side && slot.layerIdx === selectedChannel.layerIdx
          ? selectedChannel.channelIdx : null;

        // Get the genome layer for this slot (needed for connection mask → inCh mapping)
        const genomeLayers = slot.side === 'encoder' ? genome?.encoder_layers : slot.side === 'decoder' ? genome?.decoder_layers : null;
        const genomeLayer = genomeLayers && slot.layerIdx >= 0 && slot.layerIdx < genomeLayers.length ? genomeLayers[slot.layerIdx] : null;

        return (
          <div key={si} style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
            minWidth: 0,
          }}>
            {/* Column label */}
            <span style={{ fontSize: 8, color: '#484f58', fontFamily: 'monospace', fontWeight: 600 }}>{slot.label}</span>
            {slot.label === 'input' ? (
              <InputColumnMaps inputImage={maps.input_image} />
            ) : slot.label === 'Out' ? (
              <OutputColumnMaps
                layer={slot.layer}
                reconImage={maps.recon_image}
                errorMap={maps.error_map}
                highlightChannel={highlight}
                normalizeActivations={normalizeActivations}
              />
            ) : slot.layer && slot.side ? (
              <FeatureColumnMaps
                layer={slot.layer}
                highlightChannel={highlight}
                normalizeActivations={normalizeActivations}
                genomeLayer={genomeLayer}
                side={slot.side}
                layerIdx={slot.layerIdx}
                onKernelClick={onKernelClick}
              />
            ) : (
              <span style={{ fontSize: 9, color: '#30363d' }}>--</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Generate coord channel data at given resolution. Deterministic. */
function makeCoordData(size: number): { x: number[][]; y: number[][]; g: number[][] } {
  const x: number[][] = [];
  const y: number[][] = [];
  const g: number[][] = [];
  for (let r = 0; r < size; r++) {
    const xRow: number[] = [];
    const yRow: number[] = [];
    const gRow: number[] = [];
    const yv = -1 + (2 * r) / (size - 1);
    for (let c = 0; c < size; c++) {
      const xv = -1 + (2 * c) / (size - 1);
      xRow.push(xv);
      yRow.push(yv);
      gRow.push(Math.exp(-(xv * xv + yv * yv) / 1.0));
    }
    x.push(xRow);
    y.push(yRow);
    g.push(gRow);
  }
  return { x, y, g };
}

const COORD_28 = makeCoordData(28);

/** Input column: shows the input image + X, Y, Gaussian coord channels. */
function InputColumnMaps({ inputImage }: { inputImage: string }) {
  const sz = 40;
  // MiniHeatmap size for 28x28 at cell=1 → 28px, but let's keep consistent
  const cellSize = Math.max(1, Math.floor(sz / 28));
  const dispW = 28 * cellSize;
  const dispH = 28 * cellSize;

  return (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <img src={`data:image/png;base64,${inputImage}`} width={dispW} height={dispH}
          style={{ imageRendering: 'pixelated', border: '1px solid #30363d', background: '#000', display: 'block' }} />
        <span style={{ fontSize: 7, color: '#484f58', lineHeight: 1 }}>img</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <MiniHeatmap data={COORD_28.x} width={dispW} height={dispH} />
        <span style={{ fontSize: 7, color: '#484f58', lineHeight: 1 }}>X</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <MiniHeatmap data={COORD_28.y} width={dispW} height={dispH} />
        <span style={{ fontSize: 7, color: '#484f58', lineHeight: 1 }}>Y</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <MiniHeatmap data={COORD_28.g} width={dispW} height={dispH} />
        <span style={{ fontSize: 7, color: '#484f58', lineHeight: 1 }}>Gauss</span>
      </div>
    </>
  );
}

/** Output column: reconstruction image + per-pixel error map. */
function OutputColumnMaps({ layer, reconImage, errorMap, highlightChannel, normalizeActivations }: {
  layer: IecFeatureLayer | null;
  reconImage: string;
  errorMap: number[][];
  highlightChannel: number | null;
  normalizeActivations: boolean;
}) {
  const sz = 40;
  const cellSize = Math.max(1, Math.floor(sz / 28));
  const dispW = 28 * cellSize;
  const dispH = 28 * cellSize;

  return (
    <>
      {/* Reconstruction */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <img src={`data:image/png;base64,${reconImage}`} width={dispW} height={dispH}
          style={{ imageRendering: 'pixelated', border: '1px solid #58a6ff', background: '#000', display: 'block' }} />
        <span style={{ fontSize: 7, color: '#58a6ff', lineHeight: 1 }}>recon</span>
      </div>
      {/* Per-pixel error: bright = high error */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <MiniHeatmap data={errorMap} width={dispW} height={dispH} />
        <span style={{ fontSize: 7, color: '#f85149', lineHeight: 1 }}>error</span>
      </div>
      {/* Layer feature maps if present */}
      {layer && layer.channels.map((ch, ci) => (
        <div key={ci} style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          border: highlightChannel === ci ? '2px solid #f0883e' : '2px solid transparent',
          borderRadius: 3, padding: 0,
        }}>
          {ch.grad && <GradHeatmap data={ch.grad} width={dispW} height={dispH} />}
          <span style={{ fontSize: 7, color: '#484f58', lineHeight: 1 }}>grad</span>
        </div>
      ))}
    </>
  );
}

/** A single column of channel heatmaps stacked vertically.
 *  Each channel: activation heatmap + gradient heatmap + kernel weights (clickable). */
function FeatureColumnMaps({ layer, highlightChannel, normalizeActivations, genomeLayer, side, layerIdx, onKernelClick }: {
  layer: IecFeatureLayer;
  highlightChannel: number | null;
  normalizeActivations: boolean;
  genomeLayer: IecLayerGenome | null;
  side: 'encoder' | 'decoder';
  layerIdx: number;
  onKernelClick?: (side: 'encoder' | 'decoder', layerIdx: number, channelIdx: number, kernelIdx: number, inCh: number, values: number[][]) => void;
}) {
  const [H, W] = layer.resolution;
  const cellSize = Math.max(2, Math.min(5, Math.floor(40 / Math.max(H, W))));
  const dispW = W * cellSize;
  const dispH = H * cellSize;
  // Kernel display: 3x3 kernels at a readable size
  const kSize = 14; // px per kernel cell → 42px for a 3x3 kernel

  // Build kernel index → actual in_ch mapping from connection mask
  // kernels only include connected inputs (mask==1), so kernel[ki] maps to
  // the ki-th connected input channel
  const getInCh = (channelIdx: number, kernelIdx: number): number => {
    if (!genomeLayer) return kernelIdx;
    const mask = genomeLayer.connection_mask;
    if (channelIdx >= mask.length) return kernelIdx;
    const row = mask[channelIdx];
    let count = 0;
    for (let i = 0; i < row.length; i++) {
      if (row[i] > 0) {
        if (count === kernelIdx) return i;
        count++;
      }
    }
    return kernelIdx; // fallback
  };

  return (
    <>
      {layer.channels.map((ch, ci) => {
        const frozen = ch.frozen ?? false;
        return (
          <div key={ci} style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center',
            border: highlightChannel === ci ? '2px solid #f0883e' : frozen ? '2px solid #79c0ff' : '2px solid transparent',
            borderRadius: 3, padding: 1, marginBottom: 3,
          }}>
            <MiniHeatmap data={ch.data} width={dispW} height={dispH} normalize={normalizeActivations} />
            {ch.grad && (
              <GradHeatmap data={ch.grad} width={dispW} height={dispH} />
            )}
            {/* Kernels: vertical stack of KxK weight patches, clickable to edit */}
            {ch.kernels && ch.kernels.length > 0 && (
              <div style={{
                display: 'flex', flexDirection: 'column', gap: 1, marginTop: 1,
                border: '1px solid #30363d', borderRadius: 2, padding: 1,
                background: '#0d1117',
              }}>
                {ch.kernels.map((k, ki) => (
                  <div key={ki}
                    onClick={() => onKernelClick?.(side, layerIdx, ci, ki, getInCh(ci, ki), k)}
                    style={{ cursor: onKernelClick ? 'pointer' : 'default' }}
                    title="Click to edit kernel">
                    <KernelHeatmap data={k} cellSize={kSize} />
                  </div>
                ))}
              </div>
            )}
            <span style={{
              fontSize: 7,
              color: frozen ? '#79c0ff' : highlightChannel === ci ? '#f0883e' : '#484f58',
              lineHeight: 1,
            }}>
              {frozen ? `${ch.activation}*` : ch.activation}
            </span>
          </div>
        );
      })}
    </>
  );
}

/** Canvas-based mini heatmap for a 2D array.
 *  normalize=true (default): per-channel min/max → viridis colormap. Loses sign/scale info.
 *  normalize=false: diverging colormap centered at zero. Blue=negative, black=zero, orange=positive.
 *    Scale is symmetric: -absMax to +absMax so zero is always black. */
function MiniHeatmap({ data, width, height, normalize = true }: { data: number[][]; width: number; height: number; normalize?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = data.length;
    const cols = data[0].length;
    canvas.width = width;
    canvas.height = height;
    const cw = width / cols;
    const ch = height / rows;

    if (normalize) {
      // Normalized: per-channel min/max → viridis
      let mn = Infinity, mx = -Infinity;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const v = data[r][c];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      const rng = mx - mn || 1;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const t = (data[r][c] - mn) / rng; // 0-1
          const R = Math.floor(68 + t * 187);
          const G = Math.floor(1 + t * 230);
          const B = Math.floor(84 - t * 47);
          ctx.fillStyle = `rgb(${Math.min(255, R)},${Math.min(255, G)},${Math.max(0, B)})`;
          ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
    } else {
      // Absolute: diverging colormap centered at zero
      // blue (negative) ← black (zero) → orange (positive)
      let absMax = 0;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const a = Math.abs(data[r][c]);
          if (a > absMax) absMax = a;
        }
      if (absMax === 0) absMax = 1;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = data[r][c] / absMax; // -1 to +1
          let R: number, G: number, B: number;
          if (v >= 0) {
            // Positive → orange/warm
            R = Math.floor(v * 255);
            G = Math.floor(v * 160);
            B = Math.floor(v * 30);
          } else {
            // Negative → blue/cool
            const a = -v;
            R = Math.floor(a * 40);
            G = Math.floor(a * 100);
            B = Math.floor(a * 255);
          }
          ctx.fillStyle = `rgb(${R},${G},${B})`;
          ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
    }
  }, [data, width, height, normalize]);

  return (
    <canvas ref={canvasRef} width={width} height={height}
      style={{ border: '1px solid #21262d', borderRadius: 2, display: 'block', imageRendering: 'pixelated' }} />
  );
}

/** Canvas kernel weight heatmap — diverging purple (negative) / black (zero) / green (positive). */
function KernelHeatmap({ data, cellSize }: { data: number[][]; cellSize: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rows = data.length;
  const cols = data[0]?.length ?? 0;
  const width = cols * cellSize;
  const height = rows * cellSize;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0 || cols === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;

    let absMax = 0;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const a = Math.abs(data[r][c]);
        if (a > absMax) absMax = a;
      }
    if (absMax === 0) absMax = 1;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r][c] / absMax; // -1 to +1
        let R: number, G: number, B: number;
        if (v >= 0) {
          // Positive weight → green
          R = Math.floor(v * 30);
          G = Math.floor(v * 200);
          B = Math.floor(v * 30);
        } else {
          // Negative weight → purple
          const a = -v;
          R = Math.floor(a * 160);
          G = Math.floor(a * 20);
          B = Math.floor(a * 200);
        }
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }
  }, [data, cellSize, width, height, rows, cols]);

  return (
    <canvas ref={canvasRef} width={width} height={height}
      style={{ border: '1px solid #30363d', borderRadius: 1, display: 'block', imageRendering: 'pixelated' }} />
  );
}

/** Canvas gradient heatmap — diverging blue (negative) / black (zero) / red (positive). */
function GradHeatmap({ data, width, height }: { data: number[][]; width: number; height: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = data.length;
    const cols = data[0].length;
    canvas.width = width;
    canvas.height = height;
    const cw = width / cols;
    const ch = height / rows;

    // Symmetric normalization around zero
    let absMax = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const a = Math.abs(data[r][c]);
        if (a > absMax) absMax = a;
      }
    }
    if (absMax === 0) absMax = 1;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r][c] / absMax; // -1 to +1
        let R: number, G: number, B: number;
        if (v >= 0) {
          // Positive gradient (loss wants decrease) → red
          R = Math.floor(v * 220);
          G = Math.floor(v * 40);
          B = Math.floor(v * 30);
        } else {
          // Negative gradient (loss wants increase) → blue
          const a = -v;
          R = Math.floor(a * 30);
          G = Math.floor(a * 60);
          B = Math.floor(a * 220);
        }
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
  }, [data, width, height]);

  return (
    <canvas ref={canvasRef} width={width} height={height}
      style={{ border: '1px solid #21262d', borderRadius: 2, display: 'block', imageRendering: 'pixelated', marginTop: 1 }} />
  );
}

/* -- Kernel Editor -- */

/** Inline editor for a single KxK kernel. Shows editable grid + preset buttons.
 *  Appears below the feature maps when a kernel heatmap is clicked. */
function KernelEditor({ target, presets, busy, onApply, onClose, onChange }: {
  target: KernelEditTarget;
  presets: Record<string, number[][]> | null;
  busy: boolean;
  onApply: (values: number[][], autoFreeze: boolean) => void;
  onClose: () => void;
  onChange: (values: number[][]) => void;
}) {
  const [autoFreeze, setAutoFreeze] = useState(true);
  const { side, layerIdx, channelIdx, kernelIdx, values } = target;
  const rows = values.length;
  const cols = values[0]?.length ?? 0;

  const updateCell = (r: number, c: number, val: string) => {
    const v = parseFloat(val);
    if (isNaN(v)) return;
    const newValues = values.map(row => [...row]);
    newValues[r][c] = v;
    onChange(newValues);
  };

  const applyPreset = (preset: number[][]) => {
    // Preset is always 3x3, kernel might be different size — resize if needed
    if (preset.length === rows && preset[0]?.length === cols) {
      onChange(preset.map(r => [...r]));
    } else {
      // Fill center, zero pad
      const newValues = Array.from({ length: rows }, () => Array(cols).fill(0));
      const offR = Math.floor((rows - preset.length) / 2);
      const offC = Math.floor((cols - (preset[0]?.length ?? 0)) / 2);
      for (let r = 0; r < preset.length; r++) {
        for (let c = 0; c < (preset[0]?.length ?? 0); c++) {
          const tr = r + offR, tc = c + offC;
          if (tr >= 0 && tr < rows && tc >= 0 && tc < cols) {
            newValues[tr][tc] = preset[r][c];
          }
        }
      }
      onChange(newValues);
    }
  };

  return (
    <div style={{
      ...card, marginBottom: 8,
      border: '1px solid #58a6ff',
      background: '#0d1117',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: 11, color: '#58a6ff', fontWeight: 600 }}>
          Kernel Editor
        </span>
        <span style={{ fontSize: 10, color: '#8b949e', fontFamily: 'monospace' }}>
          {side} L{layerIdx} ch{channelIdx} in{kernelIdx}
        </span>
        <div style={{ flex: 1 }} />
        <label style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 10, color: '#8b949e', cursor: 'pointer', userSelect: 'none' }}>
          <input type="checkbox" checked={autoFreeze} onChange={e => setAutoFreeze(e.target.checked)}
            style={{ margin: 0, width: 10, height: 10 }} />
          auto-freeze
        </label>
        <button disabled={busy} onClick={() => onApply(values, autoFreeze)}
          style={{ ...btn, fontSize: 10, padding: '2px 8px', color: '#3fb950', borderColor: '#238636' }}>
          Apply
        </button>
        <button onClick={onClose}
          style={{ ...btn, fontSize: 10, padding: '2px 8px', color: '#8b949e' }}>
          Cancel
        </button>
      </div>

      {/* Editable grid */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
        <div style={{ display: 'inline-grid', gridTemplateColumns: `repeat(${cols}, 48px)`, gap: 2 }}>
          {values.map((row, r) =>
            row.map((val, c) => (
              <input
                key={`${r}-${c}`}
                type="number"
                value={val.toFixed(3)}
                onChange={e => updateCell(r, c, e.target.value)}
                step="0.1"
                style={{
                  width: 48, textAlign: 'center',
                  background: val > 0 ? '#0a200a' : val < 0 ? '#200a1a' : '#0d1117',
                  border: '1px solid #30363d', borderRadius: 2,
                  color: '#e6edf3', fontSize: 10, padding: '2px 2px',
                  fontFamily: 'monospace',
                }}
              />
            ))
          )}
        </div>

        {/* Preset buttons */}
        {presets && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span style={{ fontSize: 9, color: '#484f58', marginBottom: 1 }}>Presets</span>
            {Object.entries(presets).map(([name, preset]) => (
              <button key={name}
                onClick={() => applyPreset(preset)}
                style={{ ...btn, fontSize: 9, padding: '2px 6px', textAlign: 'left' }}>
                {name}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* -- Helpers -- */

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <span style={{ fontSize: 12, color: '#8b949e', fontFamily: 'monospace' }}>
      {label} <b style={{ color: '#e6edf3' }}>{value}</b>
    </span>
  );
}

function Toast({ message }: { message: string }) {
  return (
    <div style={{
      position: 'fixed', top: 12, right: 12, zIndex: 1000,
      background: '#30363d', border: '1px solid #f0883e', borderRadius: 4,
      padding: '8px 14px', color: '#f0883e', fontSize: 12,
      boxShadow: '0 4px 12px rgba(0,0,0,0.5)', maxWidth: 400,
    }}>
      {message}
    </div>
  );
}

/** Subtle background tint per activation type. */
function actColor(act: string): string {
  const map: Record<string, string> = {
    identity: '#21262d', relu: '#1a2332', sigmoid: '#2d221a',
    tanh: '#2d1a2d', sin: '#1a2d22', cos: '#1a2d2d',
    gaussian: '#2d2d1a', abs: '#2d1a1a', softplus: '#1a1a2d',
  };
  return map[act] ?? '#21262d';
}

/* -- Styles -- */

const btn: React.CSSProperties = {
  background: '#21262d', border: '1px solid #30363d', borderRadius: 4,
  color: '#e6edf3', padding: '3px 8px', fontSize: 11, cursor: 'pointer',
};
const pillBtn: React.CSSProperties = {
  background: '#21262d', border: '1px solid #30363d', borderRadius: 4,
  color: '#c9d1d9', padding: '2px 7px', fontSize: 11, cursor: 'pointer',
  lineHeight: '1.4',
};
const inputStyle: React.CSSProperties = {
  width: 60, background: '#0d1117', border: '1px solid #30363d',
  borderRadius: 3, color: '#e6edf3', padding: '2px 5px', fontSize: 11,
};
const selectStyle: React.CSSProperties = {
  background: '#0d1117', border: '1px solid #30363d',
  borderRadius: 3, color: '#e6edf3', padding: '2px 4px', fontSize: 11,
};
const card: React.CSSProperties = {
  background: '#161b22', border: '1px solid #30363d', borderRadius: 6,
  padding: '10px 12px',
};
const dimText: React.CSSProperties = { fontSize: 11, color: '#8b949e' };
