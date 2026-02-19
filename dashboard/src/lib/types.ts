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
