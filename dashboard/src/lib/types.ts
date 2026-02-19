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

export interface ModelDescribeResponse {
  description: string;
  has_decoder: boolean;
  latent_dim: number;
  num_encoder_layers: number;
  num_decoder_layers: number;
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

export interface LossEntry {
  step: number;
  task_name: string;
  task_type: string;
  task_loss: number;
  health: 'healthy' | 'warning' | 'critical';
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
