/**
 * Dashboard state store — single source of truth.
 *
 * One poll loop fetches all trainer state and writes it to an atom.
 * Components subscribe to computed slices. No independent pollers,
 * no race conditions, no scattered state.
 */
import { atom, computed } from 'nanostores';
import { fetchJSON } from './api';
import type {
  HealthResponse,
  ModelDescribeResponse,
  JobDict,
  LossEntry,
  LossSummaryResponse,
  CurrentCheckpointResponse,
  RecipeJobResponse,
  JobHistorySummary,
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

async function tick() {
  if (polling) return;
  polling = true;

  try {
    const prev = $dashboard.get();

    // Fetch everything in parallel
    const [health, checkpoint, model, currentJob, recipe, jobHistory] = await Promise.all([
      fetchJSON<HealthResponse>('/health'),
      fetchJSON<CurrentCheckpointResponse>('/checkpoints/current'),
      fetchJSON<ModelDescribeResponse>('/model/describe'),
      fetchJSON<JobDict | null>('/jobs/current'),
      fetchJSON<RecipeJobResponse | null>('/recipes/current'),
      fetchJSON<JobHistorySummary[]>('/jobs/history?limit=10'),
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
      connected,
      health: health ?? prev.health,
      checkpoint: checkpoint ?? prev.checkpoint,
      model: model ?? prev.model,
      currentJob: currentJob,  // null is valid (no running job)
      lossSummary,
      lossHistory,
      recipe: recipe ?? prev.recipe,
      jobHistory: jobHistory ?? prev.jobHistory,
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
