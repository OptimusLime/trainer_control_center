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
  ensureIecSession,
  stepIec,
  mutateIec,
  undoIec,
  resetIec,
  toggleNormalize,
} from '../lib/iec-store';
import { useEffect, useState } from 'react';
import type { IecLayerGenome, IecChannelDescriptor, IecLayerResolution } from '../lib/iec-types';

export default function IecPanel() {
  const state = useStore($iecState);
  const reconstructions = useStore($iecReconstructions);
  const setupLoading = useStore($iecSetupLoading);
  const busy = useStore($iecStepLoading);
  const toast = useStore($iecToast);
  const lossHistory = useStore($iecLossHistory);
  const normalize = useStore($iecNormalize);
  const [lr, setLr] = useState('0.01');

  useEffect(() => { ensureIecSession(); }, []);

  if (setupLoading) return <div style={{ color: '#8b949e', padding: 20 }}>Setting up IEC session...</div>;
  if (!state.active) return <div style={{ color: '#8b949e', padding: 20 }}>No IEC session active.</div>;

  const genome = state.genome;
  const encLayers = genome?.encoder_layers ?? [];
  const decLayers = genome?.decoder_layers ?? [];
  const activations = state.activation_names ?? [];
  const resolutions = state.resolutions;

  return (
    <div style={{ padding: '12px 0' }}>
      {toast && <Toast message={toast} />}

      {/* Top bar: stats + train + reset */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
        <Stat label="Step" value={state.step} />
        <Stat label="Loss" value={state.last_loss != null ? state.last_loss.toFixed(4) : '--'} />
        <Stat label="Latent" value={state.latent_dim} />

        <div style={{ borderLeft: '1px solid #30363d', height: 20, margin: '0 2px' }} />

        <label style={{ fontSize: 12, color: '#8b949e', display: 'flex', alignItems: 'center', gap: 4 }}>
          lr
          <input type="number" value={lr} onChange={e => setLr(e.target.value)}
            step="0.001" min="0.0001" max="1" style={inputStyle} />
        </label>
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
      <div style={{ marginBottom: 8 }}>
        <div style={{ ...dimText, fontWeight: 600, fontSize: 12 }}>{title}</div>
        <div style={{ ...dimText, fontSize: 10, marginTop: 1 }}>{subtitle}</div>
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

      {/* Add Layer button */}
      <div style={{ marginTop: 6 }}>
        <button
          disabled={busy}
          onClick={() => mutateIec('add_layer', {
            side,
            position: isDecoder ? layers.length - 1 : -1,
            activation: isDecoder ? 'relu' : 'identity',
            channels: 1,
          })}
          style={{ ...btn, fontSize: 10, padding: '2px 8px', color: '#58a6ff', borderColor: '#1f6feb' }}
        >
          + Add Layer
        </button>
      </div>
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
  const [addAct, setAddAct] = useState('sin');

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

        {/* Add channel */}
        {!isLastDecoder && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 2, marginLeft: 4 }}>
            <select value={addAct} onChange={e => setAddAct(e.target.value)}
              style={{ ...selectStyle, fontSize: 10, padding: '1px 4px' }}>
              {activations.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
            <button disabled={busy}
              onClick={() => mutateIec('add_channel', { side, layer_idx: layerIdx, activation: addAct })}
              style={{ ...pillBtn, background: '#1f6feb22', borderColor: '#1f6feb', color: '#58a6ff' }}
              title={`Add ${addAct} channel`}>
              +
            </button>
          </div>
        )}
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

  return (
    <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', gap: 0 }}>
      {/* Activation name button -- click to open popover */}
      <button onClick={() => setOpen(!open)} disabled={busy}
        style={{ ...pillBtn, borderRadius: canRemove ? '4px 0 0 4px' : 4, background: actColor(ch.activation) }}>
        {ch.activation}
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
    <div style={{ background: '#30363d', border: '1px solid #f0883e', borderRadius: 4,
      padding: '6px 12px', marginBottom: 8, color: '#f0883e', fontSize: 12 }}>
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
