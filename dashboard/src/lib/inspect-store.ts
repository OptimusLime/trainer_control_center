/**
 * Inspector state store — separate from the main dashboard store.
 *
 * Not polled. All state comes from explicit user actions (setup, step, load step).
 */
import { atom, computed } from 'nanostores';
import { fetchJSON, postJSON } from './api';
import type {
  InspectState,
  InspectStepResponse,
  InspectStepSummary,
  InspectSetupResponse,
} from './inspect-types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

export interface InspectStore {
  /** Inspector session state from server */
  state: InspectState;
  /** Full tensor data for the currently viewed step */
  currentStepData: InspectStepResponse | null;
  /** History of all step summaries (scalars + small vectors) */
  history: InspectStepSummary[];
  /** Which step is currently being viewed */
  viewingStep: number;
  /** Loading states */
  setupLoading: boolean;
  stepLoading: boolean;
  loadingStep: boolean;
}

const INITIAL: InspectStore = {
  state: { active: false, step: 0, total_steps: 0, condition: null },
  currentStepData: null,
  history: [],
  viewingStep: -1,
  setupLoading: false,
  stepLoading: false,
  loadingStep: false,
};

export const $inspect = atom<InspectStore>(INITIAL);

// ---------------------------------------------------------------------------
// Computed slices
// ---------------------------------------------------------------------------

export const $inspectState = computed($inspect, s => s.state);
export const $inspectCurrentStep = computed($inspect, s => s.currentStepData);
export const $inspectHistory = computed($inspect, s => s.history);
export const $inspectViewingStep = computed($inspect, s => s.viewingStep);
export const $inspectSetupLoading = computed($inspect, s => s.setupLoading);
export const $inspectStepLoading = computed($inspect, s => s.stepLoading);

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** Fetch current inspector state from server. */
export async function refreshInspectState() {
  const state = await fetchJSON<InspectState>('/inspect/state');
  if (state) {
    const prev = $inspect.get();
    $inspect.set({ ...prev, state });
  }
}

/** Set up an inspector session. */
export async function setupInspector(condition: string = 'bcl-med') {
  const prev = $inspect.get();
  $inspect.set({ ...prev, setupLoading: true });

  const result = await postJSON<InspectSetupResponse>('/inspect/setup', { condition });
  const cur = $inspect.get();

  if (result) {
    $inspect.set({
      ...cur,
      setupLoading: false,
      state: {
        active: true,
        step: 0,
        total_steps: 0,
        condition: result.condition,
        model_dim: result.model_dim,
        image_shape: result.image_shape,
      },
      currentStepData: null,
      history: [],
      viewingStep: -1,
    });
  } else {
    $inspect.set({ ...cur, setupLoading: false });
  }
}

/** Run one training step and display the result. */
export async function stepInspector(n: number = 1) {
  const prev = $inspect.get();
  $inspect.set({ ...prev, stepLoading: true });

  const result = await postJSON<InspectStepResponse>('/inspect/step', { n });
  if (!result) {
    const cur = $inspect.get();
    $inspect.set({ ...cur, stepLoading: false });
    return;
  }

  // Fetch updated history
  const history = await fetchJSON<InspectStepSummary[]>('/inspect/history');
  const state = await fetchJSON<InspectState>('/inspect/state');

  const cur = $inspect.get();
  $inspect.set({
    ...cur,
    stepLoading: false,
    currentStepData: result,
    viewingStep: result._step,
    history: history ?? cur.history,
    state: state ?? cur.state,
  });
}

/** Load a past step's full tensor data. */
export async function loadInspectStep(step: number) {
  const prev = $inspect.get();
  $inspect.set({ ...prev, loadingStep: true });

  const result = await fetchJSON<InspectStepResponse>(`/inspect/step/${step}`);
  const cur = $inspect.get();

  if (result) {
    $inspect.set({
      ...cur,
      loadingStep: false,
      currentStepData: result,
      viewingStep: step,
    });
  } else {
    $inspect.set({ ...cur, loadingStep: false });
  }
}

/** Tear down the inspector session. */
export async function teardownInspector() {
  await postJSON('/inspect/teardown');
  $inspect.set(INITIAL);
}

/** Tear down current session and set up a new one with a different condition. */
export async function switchCondition(condition: string) {
  const prev = $inspect.get();
  if (prev.state.active) {
    await postJSON('/inspect/teardown');
  }
  $inspect.set({ ...INITIAL, setupLoading: true });
  await setupInspector(condition);
}

/**
 * Auto-setup: call on mount to ensure a session is running.
 * If already active, just refreshes state. Otherwise sets up with the given condition.
 */
export async function ensureSession(condition: string = 'bcl-slow') {
  const state = await fetchJSON<InspectState>('/inspect/state');
  if (state && state.active) {
    // Already active — load current state + history
    const prev = $inspect.get();
    $inspect.set({ ...prev, state });
    const history = await fetchJSON<InspectStepSummary[]>('/inspect/history');
    if (history && history.length > 0) {
      const cur = $inspect.get();
      $inspect.set({ ...cur, history });
      // Load the latest step's full data
      const latestStep = history[history.length - 1].step;
      await loadInspectStep(latestStep);
    }
    return;
  }
  // No active session — set up
  await setupInspector(condition);
}
