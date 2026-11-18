// SPDX-License-Identifier: Apache-2.0

export const FIELD_MODULUS: number = 65537;
export const FORWARD_EXPONENT: number = 5;
export const INVERSE_EXPONENT: number = 52429;

export type ConfigName =
  | '8B-D8'
  | '8B-D16'
  | '9B-D9'
  | '9B-D18'
  | '16B-D16'
  | '16B-D32'
  | '18B-D18'
  | '18B-D36';

export interface BlockConfig {
  name: ConfigName;
  blocks: number;
  dim: number;
  blockWidth: number;
  defaultRounds: number;
}

export const BLOCK_CONFIGS: Record<ConfigName, BlockConfig> = {
  '8B-D8': { name: '8B-D8', blocks: 8, dim: 8, blockWidth: 1, defaultRounds: 8 },
  '8B-D16': { name: '8B-D16', blocks: 8, dim: 16, blockWidth: 2, defaultRounds: 16 },
  '9B-D9': { name: '9B-D9', blocks: 9, dim: 9, blockWidth: 1, defaultRounds: 9 },
  '9B-D18': { name: '9B-D18', blocks: 9, dim: 18, blockWidth: 2, defaultRounds: 18 },
  '16B-D16': { name: '16B-D16', blocks: 16, dim: 16, blockWidth: 1, defaultRounds: 16 },
  '16B-D32': { name: '16B-D32', blocks: 16, dim: 32, blockWidth: 2, defaultRounds: 32 },
  '18B-D18': { name: '18B-D18', blocks: 18, dim: 18, blockWidth: 1, defaultRounds: 18 },
  '18B-D36': { name: '18B-D36', blocks: 18, dim: 36, blockWidth: 2, defaultRounds: 36 },
};

export const CONFIG_NAMES: ConfigName[] = Object.keys(BLOCK_CONFIGS) as ConfigName[];

export type VectorPresetId =
  | 'seeded_random'
  | 'single_spike'
  | 'sequential_ramp'
  | 'boundary_max'
  | 'zero_vector'
  | 'alternating_extremes';

export interface VectorPreset {
  id: VectorPresetId;
  label: string;
  description: string;
}

export const VECTOR_PRESETS: VectorPreset[] = [
  {
    id: 'seeded_random',
    label: 'Seeded Random',
    description: 'Deterministic pseudo-random vector in F_65537 derived from seed.',
  },
  {
    id: 'single_spike',
    label: 'Single Impulse',
    description: 'Unit impulse at index 0 to visualize diffusion across sweeps.',
  },
  {
    id: 'sequential_ramp',
    label: 'Modular Ramp',
    description: 'Stepped linear progression across indices for sweep tracking.',
  },
  {
    id: 'boundary_max',
    label: 'Field Max (65536)',
    description: 'All coordinates set to q - 1 = 65536 boundary value.',
  },
  {
    id: 'alternating_extremes',
    label: 'Alternating 0 / 65536',
    description: 'Alternating field minimum and maximum across adjacent lanes.',
  },
  {
    id: 'zero_vector',
    label: 'Zero Vector',
    description: 'Origin element (0, ..., 0) illustrating affine beta injection.',
  },
];
