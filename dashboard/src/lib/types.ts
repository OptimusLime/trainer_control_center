/** TypeScript types matching trainer API JSON responses. */

export interface HealthResponse {
  status: 'ok';
  has_model: boolean;
  num_tasks: number;
  num_datasets: number;
  num_recipes: number;
  num_generators: number;
  recipe_running: boolean;
  device: string;
}

export interface ModelCapabilities {
  eval: boolean;
  reconstructions: boolean;
  traversals: boolean;
  sort_by_factor: boolean;
  attention_maps: boolean;
}

export interface ModelDescribeResponse {
  description: string;
  has_decoder: boolean;
  latent_dim: number;
  num_encoder_layers: number;
  num_decoder_layers: number;
  capabilities: ModelCapabilities;
}

export interface JobDict {
  id: string;
  state: 'running' | 'completed' | 'stopped' | 'failed';
  total_steps: number;
  current_step: number;
  task_names: string[];
  n_losses: number;
  started_at: string;
  completed_at: string | null;
  checkpoint_id: string | null;
  error: string | null;
}

export interface TrainingMetrics {
  assignment_entropy?: number;
  gradient_cv?: number;
  dead_features?: number;
  gradient_starved_features?: number;
  win_entropy?: number;
  win_counts?: number[];
  per_feature_grad_norms?: number[];
  neighborhood_stability?: number;
  coverage_cv?: number;
  explorer_graduations?: number;
  gini?: number;
  top5_share?: number;
  replacement_count?: number;
}

export interface LossEntry {
  step: number;
  task_name: string;
  task_type: string;
  task_loss: number;
  health: 'healthy' | 'warning' | 'critical';
  training_metrics?: TrainingMetrics;
}

export interface LossSummaryEntry {
  task_name: string;
  task_type: string;
  mean: number;
  final: number;
  min: number;
  max: number;
  trend: 'improving' | 'worsening' | 'flat';
  health: 'healthy' | 'warning' | 'critical';
  n_steps: number;
}

export type LossSummaryResponse = Record<string, LossSummaryEntry>;

export interface DeviceResponse {
  current: string;
  available: string[];
}

// --- Checkpoints ---

export interface CheckpointNode {
  id: string;
  tag: string;
  parent_id: string | null;
  step: number;
  timestamp: string;
  recipe_name: string | null;
  description: string | null;
  model_config: Record<string, unknown>;
  tasks_snapshot: unknown[];
  metrics: Record<string, unknown>;
}

export interface CurrentCheckpointResponse {
  has_model: boolean;
  checkpoint: CheckpointNode | null;
  checkpoint_id: string | null;
  relevant_job_id: string | null;
}

// --- Recipes ---

export interface RecipeInfo {
  name: string;
  description: string;
}

export interface BranchEntry {
  name: string;
  description: string;
  phase_start: number;
  phase_end?: number;
}

export interface RecipeJobResponse {
  id: string;
  recipe_name: string;
  state: 'running' | 'completed' | 'stopped' | 'failed';
  current_phase: string;
  phases_completed: string[];
  checkpoints_created: string[];
  started_at: string;
  error: string | null;
  current_branch: string | null;
  branch_index: number;
  total_branches: number;
  branches: BranchEntry[];
  branch_results: Record<string, Record<string, Record<string, number>>>;
}

// --- Job History ---

export interface FinalLossEntry {
  loss: number;
  health: 'healthy' | 'warning' | 'critical' | 'unknown';
}

export interface JobHistorySummary {
  id: string;
  state: 'running' | 'completed' | 'stopped' | 'failed';
  total_steps: number;
  current_step: number;
  started_at: string;
  completed_at: string | null;
  final_losses: Record<string, FinalLossEntry>;
  overall_health: 'healthy' | 'warning' | 'critical' | 'unknown';
}

export interface CheckpointTreeResponse {
  nodes: CheckpointNode[];
  current_id: string | null;
}

// --- Tasks ---

export interface TaskDescription {
  name: string;
  class_name: string;
  dataset_name: string;
  weight: number;
  enabled: boolean;
  latent_slice: string | null;
  [key: string]: unknown;  // tasks may include extra fields
}

export interface RegistryTaskInfo {
  class_name: string;
  description: string;
  [key: string]: unknown;
}

// --- Datasets ---

export interface DatasetDescription {
  name: string;
  size: number;
  image_size: number;
  channels: number;
  [key: string]: unknown;
}

export interface DatasetSampleResponse {
  images: string[];  // base64 PNG
}

// --- Generators ---

export interface GeneratorInfo {
  name: string;
  description: string;
  [key: string]: unknown;
}

// --- Training ---

export interface TrainStartParams {
  steps?: number;
  lr?: number;
  probe_lr?: number;
}

// --- Eval (M-UI-6) ---

/** factor_name -> array of rows, each row is array of base64 PNG */
export type TraversalsResponse = Record<string, string[][]>;

/** factor_name -> { lowest: base64[], highest: base64[] } */
export type SortByFactorResponse = Record<string, { lowest: string[]; highest: string[] }>;

/** "originals" + factor_name -> base64 PNG heatmap overlays */
export type AttentionMapsResponse = Record<string, string[]>;

// --- Eval ---

/** task_name -> { metric_name: value } */
export type EvalMetrics = Record<string, Record<string, number>>;

export interface EvalReconstructionsResponse {
  originals: string[];       // raw base64 PNG
  reconstructions: string[]; // raw base64 PNG
}

export interface EvalCheckpointResponse {
  checkpoint_id: string;
  tag: string;
  metrics: EvalMetrics;
}

export interface SiblingCheckpoint {
  id: string;
  tag: string;
  description: string | null;
  eval_results: EvalMetrics;
}

export interface EvalSiblingsResponse {
  siblings: SiblingCheckpoint[];
  current_id: string | null;
}

// --- Features ---

export interface FeatureSibling {
  id: string;
  tag: string;
  description: string | null;
  features: string[];  // base64 PNG per feature
}

export interface FeatureSiblingsResponse {
  siblings: FeatureSibling[];
  current_id: string | null;
  n_features: number;
  image_shape: number[];
}
