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
  TaskDescription,
  RegistryTaskInfo,
  DatasetDescription,
  DatasetSampleResponse,
  GeneratorInfo,
  DeviceResponse,
  RecipeInfo,
  TrainStartParams,
  TraversalsResponse,
  SortByFactorResponse,
  AttentionMapsResponse,
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
  // M-UI-5: Tasks, datasets, training, device
  tasks: TaskDescription[];
  registryTasks: RegistryTaskInfo[];
  datasets: DatasetDescription[];
  datasetSamples: Record<string, string[]>;  // name -> base64[]
  generators: GeneratorInfo[];
  device: DeviceResponse | null;
  recipes: RecipeInfo[];
  // M-UI-6: Latent space visualizations
  traversals: TraversalsResponse | null;
  traversalsLoading: boolean;
  sortByFactor: SortByFactorResponse | null;
  sortByFactorLoading: boolean;
  attentionMaps: AttentionMapsResponse | null;
  attentionMapsLoading: boolean;
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
  tasks: [],
  registryTasks: [],
  datasets: [],
  datasetSamples: {},
  generators: [],
  device: null,
  recipes: [],
  traversals: null,
  traversalsLoading: false,
  sortByFactor: null,
  sortByFactorLoading: false,
  attentionMaps: null,
  attentionMapsLoading: false,
};

export const $dashboard = atom<DashboardState>(INITIAL);

// ---------------------------------------------------------------------------
// Computed slices — subscribe to just what you need
// ---------------------------------------------------------------------------

export const $connected = computed($dashboard, s => s.connected);
export const $health = computed($dashboard, s => s.health);
export const $checkpoint = computed($dashboard, s => s.checkpoint);
export const $model = computed($dashboard, s => s.model);
export const $capabilities = computed($dashboard, s => s.model?.capabilities ?? null);
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
let autoPopulated = false;
let timer: ReturnType<typeof setInterval> | null = null;

async function tick(force = false) {
  if (polling && !force) return;
  polling = true;

  try {
    const prev = $dashboard.get();

    // Fetch everything in parallel
    const [health, checkpoint, model, currentJob, recipe, jobHistory, tree,
           tasks, registryTasks, datasets, generators, device, recipes] = await Promise.all([
      fetchJSON<HealthResponse>('/health'),
      fetchJSON<CurrentCheckpointResponse>('/checkpoints/current'),
      fetchJSON<ModelDescribeResponse>('/model/describe'),
      fetchJSON<JobDict | null>('/jobs/current'),
      fetchJSON<RecipeJobResponse | null>('/recipes/current'),
      fetchJSON<JobHistorySummary[]>('/jobs/history?limit=10'),
      fetchJSON<CheckpointTreeResponse>('/checkpoints/tree'),
      fetchJSON<TaskDescription[]>('/tasks'),
      fetchJSON<RegistryTaskInfo[]>('/registry/tasks'),
      fetchJSON<DatasetDescription[]>('/datasets'),
      fetchJSON<GeneratorInfo[]>('/registry/generators'),
      fetchJSON<DeviceResponse>('/device'),
      fetchJSON<RecipeInfo[]>('/recipes'),
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
      tasks: tasks ?? prev.tasks,
      registryTasks: registryTasks ?? prev.registryTasks,
      datasets: datasets ?? prev.datasets,
      generators: generators ?? prev.generators,
      device: device ?? prev.device,
      recipes: recipes ?? prev.recipes,
    });

    // Auto-populate eval/recon/viz on first load — gated by model capabilities
    const s = $dashboard.get();
    if (s.connected && model && !currentJob && !autoPopulated) {
      autoPopulated = true;  // only try once
      const caps = model.capabilities;
      if (caps?.eval) runEval();
      if (caps?.reconstructions) runReconstructions();
      if (caps?.traversals) fetchTraversals();
      if (caps?.sort_by_factor) fetchSortByFactor();
      if (caps?.attention_maps) fetchAttentionMaps();
    }
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

// M-UI-5 slices
export const $tasks = computed($dashboard, s => s.tasks);
export const $registryTasks = computed($dashboard, s => s.registryTasks);
export const $datasets = computed($dashboard, s => s.datasets);
export const $datasetSamples = computed($dashboard, s => s.datasetSamples);
export const $generators = computed($dashboard, s => s.generators);
export const $device = computed($dashboard, s => s.device);
export const $recipes = computed($dashboard, s => s.recipes);

// M-UI-6 slices
export const $traversals = computed($dashboard, s => s.traversals);
export const $traversalsLoading = computed($dashboard, s => s.traversalsLoading);
export const $sortByFactor = computed($dashboard, s => s.sortByFactor);
export const $sortByFactorLoading = computed($dashboard, s => s.sortByFactorLoading);
export const $attentionMaps = computed($dashboard, s => s.attentionMaps);
export const $attentionMapsLoading = computed($dashboard, s => s.attentionMapsLoading);

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
    // Model changed — wipe ALL model-dependent state
    const prev = $dashboard.get();
    $dashboard.set({
      ...prev,
      lossSummary: null,
      lossHistory: [],
      evalResults: null,
      evalError: null,
      reconstructions: null,
      checkpointComparison: null,
      traversals: null,
      sortByFactor: null,
      attentionMaps: null,
    });
    // Re-fetch polled state
    autoPopulated = false;  // allow auto-populate to fire again for new model
    await tick(true);
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

// ---------------------------------------------------------------------------
// M-UI-5: Task actions
// ---------------------------------------------------------------------------

export async function addTask(params: {
  class_name: string; name: string; dataset_name: string; weight?: number; latent_slice?: string;
}): Promise<boolean> {
  const result = await postJSON<TaskDescription>('/tasks/add', params);
  if (result) { await tick(true); return true; }
  return false;
}

export async function toggleTask(name: string): Promise<boolean> {
  const result = await postJSON<{ name: string; enabled: boolean }>(`/tasks/${encodeURIComponent(name)}/toggle`);
  if (result) { await tick(true); return true; }
  return false;
}

export async function setTaskWeight(name: string, weight: number): Promise<boolean> {
  const result = await postJSON<TaskDescription>(`/tasks/${encodeURIComponent(name)}/set_weight`, { weight });
  if (result) { await tick(true); return true; }
  return false;
}

export async function removeTask(name: string): Promise<boolean> {
  const result = await postJSON<{ removed: string }>(`/tasks/${encodeURIComponent(name)}/remove`);
  if (result) { await tick(true); return true; }
  return false;
}

// ---------------------------------------------------------------------------
// M-UI-5: Dataset actions
// ---------------------------------------------------------------------------

export async function loadBuiltinDataset(name: string, imageSize: number = 64): Promise<boolean> {
  const result = await postJSON<DatasetDescription>('/datasets/load_builtin', { name, image_size: imageSize });
  if (result) { await tick(true); return true; }
  return false;
}

export async function generateDataset(generatorName: string, params: Record<string, unknown>): Promise<boolean> {
  const result = await postJSON<DatasetDescription>('/generators/generate', { generator_name: generatorName, params });
  if (result) { await tick(true); return true; }
  return false;
}

export async function fetchDatasetSamples(name: string, n: number = 8) {
  const result = await fetchJSON<DatasetSampleResponse>(`/datasets/${encodeURIComponent(name)}/sample?n=${n}`);
  if (result) {
    const prev = $dashboard.get();
    $dashboard.set({ ...prev, datasetSamples: { ...prev.datasetSamples, [name]: result.images } });
  }
}

// ---------------------------------------------------------------------------
// M-UI-5: Training actions
// ---------------------------------------------------------------------------

export async function startTraining(params?: TrainStartParams): Promise<boolean> {
  const result = await postJSON<JobDict>('/train/start', params ?? {});
  if (result) { await tick(true); return true; }
  return false;
}

export async function stopTraining(): Promise<boolean> {
  const result = await postJSON<JobDict | { status: string }>('/train/stop');
  if (result) { await tick(true); return true; }
  return false;
}

// ---------------------------------------------------------------------------
// M-UI-5: Device actions
// ---------------------------------------------------------------------------

export async function setDevice(device: string): Promise<boolean> {
  const result = await postJSON<{ previous: string; current: string }>('/device/set', { device });
  if (result) { await tick(true); return true; }
  return false;
}

// ---------------------------------------------------------------------------
// M-UI-5: Recipe actions
// ---------------------------------------------------------------------------

export async function runRecipe(name: string): Promise<boolean> {
  const result = await postJSON<Record<string, unknown>>(`/recipes/${encodeURIComponent(name)}/run`);
  if (result) { await tick(true); return true; }
  return false;
}

export async function stopRecipe(): Promise<boolean> {
  const result = await postJSON<Record<string, unknown>>('/recipes/stop');
  if (result) { await tick(true); return true; }
  return false;
}

// ---------------------------------------------------------------------------
// M-UI-6: Latent space visualization actions
// ---------------------------------------------------------------------------

export async function fetchTraversals() {
  const prev = $dashboard.get();
  $dashboard.set({ ...prev, traversalsLoading: true });
  const result = await fetchJSON<TraversalsResponse>('/eval/traversals');
  const cur = $dashboard.get();
  $dashboard.set({
    ...cur,
    traversals: result ?? cur.traversals,
    traversalsLoading: false,
  });
}

export async function fetchSortByFactor() {
  const prev = $dashboard.get();
  $dashboard.set({ ...prev, sortByFactorLoading: true });
  const result = await fetchJSON<SortByFactorResponse>('/eval/sort_by_factor');
  const cur = $dashboard.get();
  $dashboard.set({
    ...cur,
    sortByFactor: result ?? cur.sortByFactor,
    sortByFactorLoading: false,
  });
}

export async function fetchAttentionMaps() {
  const prev = $dashboard.get();
  $dashboard.set({ ...prev, attentionMapsLoading: true });
  const result = await fetchJSON<AttentionMapsResponse>('/eval/attention_maps');
  const cur = $dashboard.get();
  $dashboard.set({
    ...cur,
    attentionMaps: result ?? cur.attentionMaps,
    attentionMapsLoading: false,
  });
}
