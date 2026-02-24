/**
 * IEC state store — manages Interactive Evolutionary CPPN session.
 *
 * Same pattern as inspect-store: single atom, computed slices, async actions.
 * Not polled — all state comes from explicit user actions.
 */
import { atom, computed } from 'nanostores';
import { fetchJSON, postJSON } from './api';
import type {
  IecState,
  IecSetupResponse,
  IecReconstructions,
  IecStepResponse,
} from './iec-types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

export interface IecStore {
  /** Session state from server */
  state: IecState;
  /** Reconstruction images */
  reconstructions: IecReconstructions | null;
  /** Accumulated loss history (all steps since session start / last reset) */
  lossHistory: number[];
  /** Whether to normalize reconstruction images */
  normalize: boolean;
  /** Loading states */
  setupLoading: boolean;
  stepLoading: boolean;
  /** Toast message */
  toast: string | null;
}

const EMPTY_STATE: IecState = {
  active: false,
  genome: null,
  step: 0,
  last_loss: null,
  latent_dim: 0,
  architecture: '',
  undo_depth: 0,
  activation_names: [],
  resolutions: null,
};

const INITIAL: IecStore = {
  state: EMPTY_STATE,
  reconstructions: null,
  lossHistory: [],
  normalize: false,
  setupLoading: false,
  stepLoading: false,
  toast: null,
};

export const $iec = atom<IecStore>(INITIAL);

// ---------------------------------------------------------------------------
// Computed slices
// ---------------------------------------------------------------------------

export const $iecState = computed($iec, s => s.state);
export const $iecReconstructions = computed($iec, s => s.reconstructions);
export const $iecSetupLoading = computed($iec, s => s.setupLoading);
export const $iecStepLoading = computed($iec, s => s.stepLoading);
export const $iecToast = computed($iec, s => s.toast);
export const $iecLossHistory = computed($iec, s => s.lossHistory);
export const $iecNormalize = computed($iec, s => s.normalize);

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** Toggle normalize flag and re-fetch reconstructions. */
export async function toggleNormalize() {
  const prev = $iec.get();
  const newNorm = !prev.normalize;
  $iec.set({ ...prev, normalize: newNorm });
  // Re-fetch reconstructions with new normalize setting
  await fetchReconstructions();
}

/** Set up a new IEC session. If one already exists (409), fetch its state instead. */
export async function setupIec(genomeDictOverride?: Record<string, unknown>) {
  const prev = $iec.get();
  $iec.set({ ...prev, setupLoading: true, toast: null });

  const body: Record<string, unknown> = {};
  if (genomeDictOverride) body.genome = genomeDictOverride;

  const result = await postJSON<IecSetupResponse>('/iec/setup', body);
  const cur = $iec.get();

  if (result) {
    $iec.set({
      ...cur,
      setupLoading: false,
      state: result.state,
      lossHistory: [],
    });
    await fetchReconstructions();
  } else {
    // Might be 409 (session exists) — try fetching state
    const state = await fetchJSON<IecState>('/iec/state');
    if (state && state.active) {
      $iec.set({ ...cur, setupLoading: false, state });
      await fetchReconstructions();
    } else {
      $iec.set({ ...cur, setupLoading: false, toast: 'Failed to create IEC session' });
    }
  }
}

/** Fetch current session state. */
export async function fetchIecState() {
  const state = await fetchJSON<IecState>('/iec/state');
  if (state) {
    const prev = $iec.get();
    $iec.set({ ...prev, state });
  }
}

/** Fetch reconstruction images. */
export async function fetchReconstructions() {
  const { normalize } = $iec.get();
  const url = normalize ? '/iec/reconstructions?normalize=true' : '/iec/reconstructions';
  const recons = await fetchJSON<IecReconstructions>(url);
  if (recons) {
    const prev = $iec.get();
    $iec.set({ ...prev, reconstructions: recons });
  }
}

/** Train N steps. Returns losses + reconstructions. */
export async function stepIec(n: number = 10, lr?: number) {
  const prev = $iec.get();
  $iec.set({ ...prev, stepLoading: true, toast: null });

  const body: Record<string, unknown> = { n, normalize: prev.normalize };
  if (lr !== undefined) body.lr = lr;

  try {
    const resp = await fetch(`${import.meta.env.DEV ? '/api' : ''}/iec/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000),
    });
    const cur = $iec.get();
    if (!resp.ok) {
      const err = await resp.text();
      $iec.set({ ...cur, stepLoading: false, toast: `Step failed (${resp.status}): ${err}` });
      return;
    }
    const result = await resp.json() as IecStepResponse;
    $iec.set({
      ...cur,
      stepLoading: false,
      state: {
        ...cur.state,
        step: result.step ?? cur.state.step,
        last_loss: result.last_loss ?? cur.state.last_loss,
      },
      reconstructions: result.reconstructions ?? cur.reconstructions,
      lossHistory: [...cur.lossHistory, ...(result.losses ?? [])],
    });
  } catch (e) {
    const cur = $iec.get();
    $iec.set({ ...cur, stepLoading: false, toast: `Step error: ${e}` });
  }
}

/** Response shape from mutate/undo: state fields + optional reconstructions. */
interface IecMutateResponse extends IecState {
  reconstructions?: IecReconstructions;
}

/** Apply a mutation. Returns updated state + reconstructions. */
export async function mutateIec(type: string, params: Record<string, unknown> = {}) {
  const prev = $iec.get();
  $iec.set({ ...prev, stepLoading: true, toast: null });

  const result = await postJSON<IecMutateResponse>('/iec/mutate', { type, ...params });
  const cur = $iec.get();

  if (result) {
    const { reconstructions, ...stateFields } = result;
    $iec.set({
      ...cur,
      stepLoading: false,
      state: stateFields,
      reconstructions: reconstructions ?? cur.reconstructions,
    });
  } else {
    $iec.set({ ...cur, stepLoading: false, toast: 'Mutation failed' });
    setTimeout(() => {
      const c = $iec.get();
      if (c.toast === 'Mutation failed') $iec.set({ ...c, toast: null });
    }, 3000);
  }
}

/** Undo last mutation. Returns updated state + reconstructions. */
export async function undoIec() {
  const prev = $iec.get();
  $iec.set({ ...prev, stepLoading: true, toast: null });

  const result = await postJSON<IecMutateResponse>('/iec/undo');
  const cur = $iec.get();

  if (result) {
    const { reconstructions, ...stateFields } = result;
    $iec.set({
      ...cur,
      stepLoading: false,
      state: stateFields,
      reconstructions: reconstructions ?? cur.reconstructions,
    });
  } else {
    $iec.set({ ...cur, stepLoading: false, toast: 'Undo failed (nothing to undo?)' });
    setTimeout(() => {
      const c = $iec.get();
      if (c.toast?.includes('Undo')) $iec.set({ ...c, toast: null });
    }, 3000);
  }
}

/** Tear down the session and reset all state. */
export async function teardownIec() {
  await postJSON('/iec/teardown');
  $iec.set(INITIAL);
}

/** Reset: teardown + immediate re-setup. */
export async function resetIec() {
  await teardownIec();
  await setupIec();
}

/** Auto-setup: check if session exists, set up if not. */
export async function ensureIecSession() {
  const state = await fetchJSON<IecState>('/iec/state');
  if (state && state.active) {
    const prev = $iec.get();
    $iec.set({ ...prev, state });
    await fetchReconstructions();
    return;
  }
  await setupIec();
}
