/** TypeScript types for the step inspector API.
 *
 * StepTensorKey must match the Python enum in acc/step_inspector.py exactly.
 */

export enum StepTensorKey {
  BATCH_IMAGES = 'batch_images',
  BATCH_LABELS = 'batch_labels',
  LOSS = 'loss',
  ENCODER_WEIGHTS = 'encoder_weights',
  RANK_SCORE = 'rank_score',
  STRENGTH = 'strength',
  FEATURE_NOVELTY = 'feature_novelty',
  IMAGE_COVERAGE = 'image_coverage',
  WIN_RATE = 'win_rate',
  NEIGHBORS = 'neighbors',
  LOCAL_COVERAGE = 'local_coverage',
  LOCAL_NOVELTY = 'local_novelty',
  IN_NEIGHBORHOOD = 'in_nbr',
  GRADIENT_WEIGHT = 'gradient_weight',
  CONTENDER_WEIGHT = 'contender_weight',
  ATTRACTION_WEIGHT = 'attraction_weight',
  GRAD_MASK = 'grad_mask',
  LOCAL_TARGET = 'local_target',
  GLOBAL_TARGET = 'global_target',
  SOM_TARGETS = 'som_targets',
  SOM_DELTA = 'som_delta',
  GRAD_MASKED = 'grad_masked',
  ENCODER_WEIGHTS_POST = 'encoder_weights_post',
}

/** Full tensor data for a single step (returned by POST /inspect/step and GET /inspect/step/{n}). */
export interface InspectStepResponse {
  _step: number;
  _keys: string[];
  [key: string]: unknown;
}

/** Summary for one step in the history timeline. Contains scalars + [D]-vectors only. */
export interface InspectStepSummary {
  step: number;
  keys: string[];
  loss?: number;
  win_rate?: number[];
  feature_novelty?: number[];
  gradient_weight?: number[];
  contender_weight?: number[];
  attraction_weight?: number[];
  image_coverage?: number[];
}

export interface InspectState {
  active: boolean;
  step: number;
  total_steps: number;
  condition: string | null;
  model_dim?: number;
  image_shape?: number[];
}

export interface InspectSetupResponse {
  status: string;
  condition: string;
  model_dim: number;
  image_shape: number[];
}
