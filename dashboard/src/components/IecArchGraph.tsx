/**
 * IecArchGraph — SVG architecture DAG for the ConvCPPN.
 *
 * Layout: left-to-right columns.
 *   Input(4ch) -> Encoder layers -> Bottleneck -> Decoder layers -> Output(1ch)
 *
 * Channels are circles, colored by activation type.
 * Connections are lines from connection_mask. Disabled = dashed/faded.
 * Coordinate injection nodes (X,Y,G) shown at encoder input and each decoder layer input.
 * Click a channel to select it (calls onSelectChannel).
 */
import { useMemo } from 'react';
import type { IecGenome, IecResolutions } from '../lib/iec-types';

/* ── Activation colors (brighter than pill backgrounds, for node fills) ── */

const ACT_COLORS: Record<string, string> = {
  identity: '#6e7681',
  relu:     '#58a6ff',
  sigmoid:  '#d29922',
  tanh:     '#bc8cff',
  sin:      '#3fb950',
  cos:      '#39d2c0',
  gaussian: '#e3b341',
  abs:      '#f47067',
  softplus: '#6cb6ff',
};

function actFill(act: string): string {
  return ACT_COLORS[act] ?? '#6e7681';
}

/* ── Types ── */

export interface ChannelSelection {
  side: 'encoder' | 'decoder';
  layerIdx: number;
  channelIdx: number;
}

interface ColumnDef {
  kind: 'input' | 'encoder' | 'bottleneck' | 'decoder' | 'output';
  label: string;
  resLabel?: string;
  channels: { activation: string; label: string }[];
  /** Connection mask TO this column from the previous column. [out_ch][in_ch] */
  mask?: number[][];
  /** For coord injection: how many extra coord nodes feed into this column */
  coordCount?: number;
  /** Original layer index and side (for selection callback) */
  side?: 'encoder' | 'decoder';
  layerIdx?: number;
}

/* ── Layout constants ── */

const NODE_R = 10;
const COL_GAP = 100;
const ROW_GAP = 30;
const COORD_R = 6;
const COORD_GAP = 20;
const PAD_X = 30;
const PAD_Y = 30;
const LABEL_H = 16;

/* ── Component ── */

export default function IecArchGraph({
  genome,
  resolutions,
  selected,
  onSelectChannel,
}: {
  genome: IecGenome;
  resolutions: IecResolutions | null;
  selected: ChannelSelection | null;
  onSelectChannel?: (sel: ChannelSelection | null) => void;
}) {
  const columns = useMemo(() => buildColumns(genome, resolutions), [genome, resolutions]);

  // Calculate positions
  // Height needs to fit the tallest column (channels * ROW_GAP + coords * COORD_GAP + gap between)
  const maxTotalHeight = Math.max(...columns.map(c => {
    const nCoord = c.coordCount ?? 0;
    const nMain = c.channels.length;
    return nMain * ROW_GAP + (nCoord > 0 ? nCoord * COORD_GAP + COORD_GAP : 0);
  }), ROW_GAP);
  const svgH = PAD_Y * 2 + LABEL_H + maxTotalHeight + 10;
  const svgW = PAD_X * 2 + (columns.length - 1) * COL_GAP;

  // Column x positions
  const colX = columns.map((_, i) => PAD_X + i * COL_GAP);

  // For each column, compute y positions for channels and coord nodes
  // Coord nodes sit above the main channels
  const colPositions = columns.map((col, ci) => {
    const nCoord = col.coordCount ?? 0;
    const nMain = col.channels.length;
    const colHeight = nMain * ROW_GAP + (nCoord > 0 ? nCoord * COORD_GAP + COORD_GAP : 0);
    const startY = PAD_Y + LABEL_H + (maxTotalHeight - colHeight) / 2;

    const coordY: number[] = [];
    for (let i = 0; i < nCoord; i++) {
      coordY.push(startY + i * COORD_GAP);
    }

    const mainStartY = nCoord > 0 ? coordY[coordY.length - 1] + COORD_GAP + 4 : startY;
    const mainY: number[] = [];
    for (let i = 0; i < nMain; i++) {
      mainY.push(mainStartY + i * ROW_GAP);
    }

    return { coordY, mainY, x: colX[ci] };
  });

  // Build edges between consecutive columns
  const edges: { x1: number; y1: number; x2: number; y2: number; enabled: boolean }[] = [];

  for (let ci = 1; ci < columns.length; ci++) {
    const col = columns[ci];
    const prev = columns[ci - 1];
    const prevPos = colPositions[ci - 1];
    const curPos = colPositions[ci];
    const mask = col.mask;
    if (!mask) continue;

    // Previous column's output nodes = prevPos.mainY
    // Current column's input nodes = curPos.mainY (channels) receiving from prev main + prev coords
    // The mask shape is [cur_out_ch, prev_in_ch]
    // prev_in_ch includes the previous layer's output channels
    // For decoder layers, the mask's last 3 columns are coord channels

    const nPrevMain = prev.channels.length;
    const nCurCoord = col.coordCount ?? 0;

    for (let outCh = 0; outCh < col.channels.length; outCh++) {
      if (outCh >= mask.length) break;
      const row = mask[outCh];
      for (let inCh = 0; inCh < row.length; inCh++) {
        // Determine source position
        let srcX: number, srcY: number;

        if (inCh < nPrevMain) {
          // Connection from previous layer's channel
          srcX = prevPos.x;
          srcY = prevPos.mainY[inCh] ?? prevPos.mainY[prevPos.mainY.length - 1];
        } else {
          // Connection from coord channel — these are the current column's coord nodes
          const coordIdx = inCh - nPrevMain;
          srcX = curPos.x;
          srcY = curPos.coordY[coordIdx] ?? curPos.coordY[curPos.coordY.length - 1] ?? curPos.mainY[0] - 20;
        }

        const dstX = curPos.x;
        const dstY = curPos.mainY[outCh] ?? curPos.mainY[curPos.mainY.length - 1];

        // Don't draw self-column edges for coord nodes as lines (they're shown vertically)
        if (inCh < nPrevMain) {
          edges.push({ x1: srcX + NODE_R, y1: srcY, x2: dstX - NODE_R, y2: dstY, enabled: row[inCh] === 1 });
        } else {
          // Coord edges: draw from coord node down to channel
          edges.push({ x1: srcX, y1: srcY + COORD_R, x2: dstX, y2: dstY - NODE_R, enabled: row[inCh] === 1 });
        }
      }
    }
  }

  const handleClick = (col: ColumnDef, chIdx: number) => {
    if (!onSelectChannel) return;
    if (col.side == null || col.layerIdx == null) return;
    const sel: ChannelSelection = { side: col.side, layerIdx: col.layerIdx, channelIdx: chIdx };
    // Toggle off if already selected
    if (selected && selected.side === sel.side && selected.layerIdx === sel.layerIdx && selected.channelIdx === sel.channelIdx) {
      onSelectChannel(null);
    } else {
      onSelectChannel(sel);
    }
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg width={svgW} height={svgH} style={{ display: 'block', minWidth: svgW }}>
        {/* Edges */}
        {edges.map((e, i) => (
          <line key={i}
            x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
            stroke={e.enabled ? '#30363d' : '#21262d'}
            strokeWidth={e.enabled ? 1.2 : 0.6}
            strokeDasharray={e.enabled ? undefined : '3,3'}
            opacity={e.enabled ? 0.7 : 0.25}
          />
        ))}

        {/* Columns */}
        {columns.map((col, ci) => {
          const pos = colPositions[ci];
          const coordLabels = ['X', 'Y', 'G'];

          return (
            <g key={ci}>
              {/* Column label */}
              <text x={pos.x} y={PAD_Y - 4} textAnchor="middle"
                fill="#8b949e" fontSize={10} fontFamily="monospace" fontWeight={600}>
                {col.label}
              </text>
              {col.resLabel && (
                <text x={pos.x} y={PAD_Y + 8} textAnchor="middle"
                  fill="#484f58" fontSize={8} fontFamily="monospace">
                  {col.resLabel}
                </text>
              )}

              {/* Coord injection nodes (diamonds) */}
              {pos.coordY.map((cy, ki) => (
                <g key={`coord-${ki}`}>
                  <polygon
                    points={`${pos.x},${cy - COORD_R} ${pos.x + COORD_R},${cy} ${pos.x},${cy + COORD_R} ${pos.x - COORD_R},${cy}`}
                    fill="#21262d" stroke="#484f58" strokeWidth={1}
                  />
                  <text x={pos.x} y={cy + 3} textAnchor="middle"
                    fill="#8b949e" fontSize={7} fontFamily="monospace">
                    {coordLabels[ki] ?? '?'}
                  </text>
                </g>
              ))}

              {/* Channel nodes (circles) */}
              {col.channels.map((ch, chIdx) => {
                const cy = pos.mainY[chIdx];
                const isSelected = selected
                  && col.side === selected.side
                  && col.layerIdx === selected.layerIdx
                  && chIdx === selected.channelIdx;
                const clickable = col.side != null && col.layerIdx != null;

                return (
                  <g key={chIdx}
                    style={{ cursor: clickable ? 'pointer' : 'default' }}
                    onClick={() => handleClick(col, chIdx)}>
                    {/* Selection ring */}
                    {isSelected && (
                      <circle cx={pos.x} cy={cy} r={NODE_R + 3}
                        fill="none" stroke="#f0883e" strokeWidth={2} />
                    )}
                    {/* Node circle */}
                    <circle cx={pos.x} cy={cy} r={NODE_R}
                      fill={actFill(ch.activation)} stroke="#e6edf3" strokeWidth={1.2}
                      opacity={0.9}
                    />
                    {/* Channel label */}
                    <text x={pos.x} y={cy + 3} textAnchor="middle"
                      fill="#0d1117" fontSize={8} fontWeight={700} fontFamily="monospace">
                      {ch.label}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 8, padding: '4px 8px', flexWrap: 'wrap' }}>
        {Object.entries(ACT_COLORS).map(([act, color]) => (
          <span key={act} style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: 9, color: '#8b949e' }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: color }} />
            {act}
          </span>
        ))}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: 9, color: '#8b949e' }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, transform: 'rotate(45deg)', background: '#21262d', border: '1px solid #484f58' }} />
          coord
        </span>
      </div>
    </div>
  );
}

/* ── Build column definitions from genome ── */

function buildColumns(genome: IecGenome, resolutions: IecResolutions | null): ColumnDef[] {
  const cols: ColumnDef[] = [];
  const encLayers = genome.encoder_layers;
  const decLayers = genome.decoder_layers;
  const encRes = resolutions?.encoder ?? [];
  const decRes = resolutions?.decoder ?? [];
  const bottleneckRes = resolutions?.bottleneck_res ?? 3;

  // Input column: 1 image + 3 coords = 4 channels
  // Actually, for the graph we show 4 input nodes but no coord diamonds here
  // because the encoder's layer 0 mask has 4 input columns (img + X + Y + G)
  cols.push({
    kind: 'input',
    label: 'Input',
    resLabel: '28x28',
    channels: [
      { activation: 'identity', label: 'img' },
      { activation: 'identity', label: 'X' },
      { activation: 'identity', label: 'Y' },
      { activation: 'identity', label: 'G' },
    ],
  });

  // Encoder layers
  for (let li = 0; li < encLayers.length; li++) {
    const layer = encLayers[li];
    const res = encRes[li];
    cols.push({
      kind: 'encoder',
      label: `E${li}`,
      resLabel: res ? `${res.output_res}x${res.output_res}` : undefined,
      channels: layer.channel_descriptors.map(ch => ({
        activation: ch.activation,
        label: shortAct(ch.activation),
      })),
      mask: layer.connection_mask,
      side: 'encoder',
      layerIdx: li,
    });
  }

  // Latent space (AdaptiveAvgPool2d output)
  const bottleneckCh = encLayers.length > 0
    ? encLayers[encLayers.length - 1].channel_descriptors.length
    : 1;
  cols.push({
    kind: 'bottleneck',
    label: 'Latent',
    resLabel: `${bottleneckRes}x${bottleneckRes}`,
    channels: Array.from({ length: bottleneckCh }, (_, i) => ({
      activation: 'identity',
      label: `${i}`,
    })),
    // 1:1 pass-through mask from last encoder
    mask: Array.from({ length: bottleneckCh }, (_, i) =>
      Array.from({ length: bottleneckCh }, (_, j) => i === j ? 1 : 0)
    ),
  });

  // Decoder layers
  for (let li = 0; li < decLayers.length; li++) {
    const layer = decLayers[li];
    const res = decRes[li];
    // Decoder layers have coord injection: last 3 columns of mask are X, Y, G
    cols.push({
      kind: 'decoder',
      label: li === decLayers.length - 1 ? 'Out' : `D${li}`,
      resLabel: res ? `${res.output_res}x${res.output_res}` : undefined,
      channels: layer.channel_descriptors.map(ch => ({
        activation: ch.activation,
        label: li === decLayers.length - 1 && layer.channel_descriptors.length === 1
          ? 'out'
          : shortAct(ch.activation),
      })),
      mask: layer.connection_mask,
      coordCount: 3, // X, Y, G injected at every decoder layer
      side: 'decoder',
      layerIdx: li,
    });
  }

  return cols;
}

function shortAct(act: string): string {
  const m: Record<string, string> = {
    identity: 'id',
    relu: 'R',
    sigmoid: 'sig',
    tanh: 'th',
    sin: 'sin',
    cos: 'cos',
    gaussian: 'gau',
    abs: 'abs',
    softplus: 'sp',
  };
  return m[act] ?? act.slice(0, 3);
}
