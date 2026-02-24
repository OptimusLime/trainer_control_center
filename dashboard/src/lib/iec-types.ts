/** TypeScript types for the IEC (Interactive Evolutionary CPPN) API.
 *
 * These mirror the Python response shapes from acc/iec.py and trainer_api.py.
 */

export interface IecLayerResolution {
  input_res: number;
  output_res: number;
}

export interface IecResolutions {
  encoder: IecLayerResolution[];
  decoder: IecLayerResolution[];
  bottleneck_res: number;
}

export interface IecState {
  active: boolean;
  genome: IecGenome | null;
  step: number;
  last_loss: number | null;
  latent_dim: number;
  architecture: string;
  undo_depth: number;
  activation_names: string[];
  resolutions: IecResolutions | null;
  ssim_weight: number;
}

export interface IecGenome {
  encoder_layers: IecLayerGenome[];
  decoder_layers: IecLayerGenome[];
  metadata: Record<string, unknown>;
}

export interface IecLayerGenome {
  channel_descriptors: IecChannelDescriptor[];
  connection_mask: number[][];
  kernel_size: number;
  stride: number;
  padding: number;
}

export interface IecChannelDescriptor {
  activation: string;
  is_passthrough?: boolean;
  passthrough_source?: number;
  frozen?: boolean;
}

export interface IecSetupResponse {
  status: string;
  state: IecState;
}

export interface IecReconstructions {
  inputs: string[];   // base64 PNG images
  outputs: string[];  // base64 PNG images
}

export interface IecStepResponse {
  losses: number[];
  step: number;
  last_loss: number;
  reconstructions?: IecReconstructions;
}

export interface IecCheckpoint {
  id: string;
  tag: string;
  parent_id: string | null;
  step: number;
  timestamp: string;
  recipe_name: string | null;
  description: string | null;
  model_config: Record<string, unknown>;
  metrics: Record<string, unknown>;
}

export interface IecFeatureChannel {
  activation: string;
  data: number[][];
  grad?: number[][];      // gradient w.r.t. L1 reconstruction loss
  kernels?: number[][][];  // per-connected-input KxK kernel weights
}

export interface IecFeatureLayer {
  name: string;
  resolution: [number, number];
  channels: IecFeatureChannel[];
}

export interface IecFeatureMaps {
  input_image: string;    // base64 PNG
  recon_image: string;    // base64 PNG — model's reconstruction
  error_map: number[][];  // |recon - input| per pixel, 28x28
  loss: number;           // total loss (L1 + SSIM) for this image
  l1: number;             // L1 component only
  encoder: IecFeatureLayer[];
  latent: IecFeatureLayer | null;
  decoder: IecFeatureLayer[];
}
