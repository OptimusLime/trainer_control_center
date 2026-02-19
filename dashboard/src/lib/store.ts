/**
 * Dashboard state store — single source of truth.
 *
 * One poll loop fetches all trainer state and writes it to an atom.
 * Components subscribe to computed slices. No independent pollers,
 * no race conditions, no scattered state.
 */
import { atom, computed } from 'nanostores';
import { fetchJSON } from './api';
import { postJSON } from './api';
import type {
  HealthResponse,
  ModelDescribeResponse,
  JobDict,
  LossEntry,
  LossSummaryResponse,
  CurrentCheckpointResponse,
  RecipeJobResponse,
  JobHistorySummary,
  EvalMetrics,
  EvalReconstructionsResponse,
  EvalCheckpointResponse,
  CheckpointTreeResponse,
  CheckpointNode,
} from './types';

// ---------------------------------------------------------------------------
// Core state atom — everything the dashboard needs in one place
// ---------------------------------------------------------------------------

export interface DashboardState {
  connected: boolean;
  health: HealthResponse | null;
  checkpoint: CurrentCheckpointResponse | null;
  model: ModelDescribeResponse | null;
  currentJob: JobDict | null;
  lossSummary: LossSummaryResponse | null;
  lossHistory: LossEntry[];
  recipe: RecipeJobResponse | null;
  jobHistory: JobHistorySummary[];
  // Checkpoint tree (polled)
  checkpointTree: CheckpointNode[];
  checkpointCurrentId: string | null;
  // Eval (action-triggered, not polled)
  evalResults: EvalMetrics | null;
  evalLoading: boolean;
  evalError: string | null;
  reconstructions: EvalReconstructionsResponse | null;
  reconLoading: boolean;
  checkpointComparison: { current: EvalMetrics; compare: EvalCheckpointResponse } | null;
  compareLoading: boolean;
}

const INITIAL: DashboardState = {
  connected: false,
  health: null,
  checkpoint: null,
  model: null,
  currentJob: null,
  lossSummary: null,
  lossHistory: [],
  recipe: null,
  jobHistory: [],
  checkpointTree: [],
  checkpointCurrentId: null,
  evalResults: null,
  evalLoading: false,
  evalError: null,
  reconstructions: null,
  reconLoading: false,
  checkpointComparison: null,
  compareLoading: false,
};

export const $dashboard = atom<DashboardState>(INITIAL);

// ---------------------------------------------------------------------------
// Computed slices — subscribe to just what you need
// ---------------------------------------------------------------------------

export const $connected = computed($dashboard, s => s.connected);
export const $health = computed($dashboard, s => s.health);
export const $checkpoint = computed($dashboard, s => s.checkpoint);
export const $model = computed($dashboard, s => s.model);
export const $currentJob = computed($dashboard, s => s.currentJob);
export const $lossSummary = computed($dashboard, s => s.lossSummary);
export const $lossHistory = computed($dashboard, s => s.lossHistory);
export const $recipe = computed($dashboard, s => s.recipe);
export const $jobHistory = computed($dashboard, s => s.jobHistory);

export const $checkpointTree = computed($dashboard, s => s.checkpointTree);
export const $checkpointCurrentId = computed($dashboard, s => s.checkpointCurrentId);

/** The job ID whose loss data we're displaying. */
export const $activeJobId = computed($dashboard, s => {
  // Running job takes priority
  if (s.currentJob) return s.currentJob.id;
  // Then checkpoint's relevant job
  if (s.checkpoint?.relevant_job_id) return s.checkpoint.relevant_job_id;
  // Then most recent history job
  if (s.jobHistory.length > 0) return s.jobHistory[0].id;
  return null;
});

// ---------------------------------------------------------------------------
// Single poll loop
// ---------------------------------------------------------------------------

const POLL_MS = 5000;
let polling = false;
let timer: ReturnType<typeof setInterval> | null = null;

async function tick(force = false) {
  if (polling && !force) return;
  polling = true;

  try {
    const prev = $dashboard.get();

    // Fetch everything in parallel
    const [health, checkpoint, model, currentJob, recipe, jobHistory, tree] = await Promise.all([
      fetchJSON<HealthResponse>('/health'),
      fetchJSON<CurrentCheckpointResponse>('/checkpoints/current'),
      fetchJSON<ModelDescribeResponse>('/model/describe'),
      fetchJSON<JobDict | null>('/jobs/current'),
      fetchJSON<RecipeJobResponse | null>('/recipes/current'),
      fetchJSON<JobHistorySummary[]>('/jobs/history?limit=10'),
      fetchJSON<CheckpointTreeResponse>('/checkpoints/tree'),
    ]);

    const connected = health !== null;

    // Determine which job to show loss data for
    const activeJobId = currentJob?.id
      ?? checkpoint?.relevant_job_id
      ?? (jobHistory && jobHistory.length > 0 ? jobHistory[0].id : null);

    // Fetch loss data for the active job
    let lossSummary = prev.lossSummary;
    let lossHistory = prev.lossHistory;

    if (activeJobId) {
      const [newSummary, newHistory] = await Promise.all([
        fetchJSON<LossSummaryResponse>(`/jobs/${activeJobId}/loss_summary`),
        fetchJSON<LossEntry[]>(`/jobs/${activeJobId}/loss_history?max_points=500`),
      ]);
      if (newSummary) lossSummary = newSummary;
      if (newHistory) lossHistory = newHistory;
    }

    $dashboard.set({
      ...prev,                  // preserve eval/recon/compare state
      connected,
      health: health ?? prev.health,
      checkpoint: checkpoint ?? prev.checkpoint,
      model: model ?? prev.model,
      currentJob: currentJob,  // null is valid (no running job)
      lossSummary,
      lossHistory,
      recipe: recipe ?? prev.recipe,
      jobHistory: jobHistory ?? prev.jobHistory,
      checkpointTree: tree?.nodes ?? prev.checkpointTree,
      checkpointCurrentId: tree?.current_id ?? prev.checkpointCurrentId,
    });
  } finally {
    polling = false;
  }
}

export function startPolling() {
  if (timer) return;
  tick(); // immediate first fetch
  timer = setInterval(tick, POLL_MS);
}

export function stopPolling() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

/** Force a job's loss data into the store (for history clicks). */
export async function loadJobLoss(jobId: string) {
  const [summary, history] = await Promise.all([
    fetchJSON<LossSummaryResponse>(`/jobs/${jobId}/loss_summary`),
    fetchJSON<LossEntry[]>(`/jobs/${jobId}/loss_history?max_points=500`),
  ]);
  const prev = $dashboard.get();
  $dashboard.set({
    ...prev,
    lossSummary: summary ?? prev.lossSummary,
    lossHistory: history ?? prev.lossHistory,
  });
}

// ---------------------------------------------------------------------------
// Eval slices
// ---------------------------------------------------------------------------

export const $evalResults = computed($dashboard, s => s.evalResults);
export const $evalLoading = computed($dashboard, s => s.evalLoading);
export const $evalError = computed($dashboard, s => s.evalError);
export const $reconstructions = computed($dashboard, s => s.reconstructions);
export const $reconLoading = computed($dashboard, s => s.reconLoading);
export const $checkpointComparison = computed($dashboard, s => s.checkpointComparison);
export const $compareLoading = computed($dashboard, s => s.compareLoading);

// ---------------------------------------------------------------------------
// Eval actions (button-triggered, not polled)
// ---------------------------------------------------------------------------

export async function runEval() {
  const prev = $dashboard.get();
  $dashboard.set({ ...prev, evalLoading: true, evalError: null });

  const result = await postJSON<EvalMetrics>('/eval/run');
  const cur = $dashboard.get();
  if (result) {
    $dashboard.set({ ...cur, evalResults: result, evalLoading: false });
  } else {
    $dashboard.set({ ...cur, evalLoading: false, evalError: 'Eval failed (model busy or no trainer)' });
  }
}

export async function runReconstructions(n: number = 8) {
  const prev = $dashboard.get();
  $dashboard.set({ ...prev, reconLoading: true });

  const result = await postJSON<EvalReconstructionsResponse>('/eval/reconstructions', { n });
  const cur = $dashboard.get();
  $dashboard.set({
    ...cur,
    reconstructions: result ?? cur.reconstructions,
    reconLoading: false,
  });
}

// ---------------------------------------------------------------------------
// Checkpoint actions (button-triggered, refresh tree after)
// ---------------------------------------------------------------------------

export async function saveCheckpoint(tag: string): Promise<boolean> {
  const result = await postJSON<CheckpointNode>('/checkpoints/save', { tag });
  if (result) {
    await tick(true); // refresh tree + current checkpoint
    return true;
  }
  return false;
}

export async function loadCheckpoint(id: string): Promise<boolean> {
  const result = await postJSON<CheckpointNode>('/checkpoints/load', { id });
  if (result) {
    await tick(true); // refresh everything — model changed
    return true;
  }
  return false;
}

export async function forkCheckpoint(id: string, newTag: string): Promise<boolean> {
  const result = await postJSON<CheckpointNode>('/checkpoints/fork', { id, new_tag: newTag });
  if (result) {
    await tick(true); // refresh tree
    return true;
  }
  return false;
}

export async function runCheckpointComparison(checkpointId: string) {
  const prev = $dashboard.get();
  $dashboard.set({ ...prev, compareLoading: true });

  // Run eval on current model first, then eval the checkpoint
  const [currentMetrics, cpResult] = await Promise.all([
    postJSON<EvalMetrics>('/eval/run'),
    postJSON<EvalCheckpointResponse>('/eval/checkpoint', { checkpoint_id: checkpointId }),
  ]);

  const cur = $dashboard.get();
  if (currentMetrics && cpResult) {
    $dashboard.set({
      ...cur,
      checkpointComparison: { current: currentMetrics, compare: cpResult },
      compareLoading: false,
    });
  } else {
    $dashboard.set({ ...cur, compareLoading: false });
  }
}
