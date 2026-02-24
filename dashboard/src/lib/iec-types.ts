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
