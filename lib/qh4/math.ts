// SPDX-License-Identifier: Apache-2.0

import {
  BLOCK_CONFIGS,
  ConfigName,
  FIELD_MODULUS,
  FORWARD_EXPONENT,
  INVERSE_EXPONENT,
  VectorPresetId,
} from './constants';
import {
  QH4InternalStateError,
  validateAlpha,
  validateBeta,
  validateCoefficients,
  validateConfigName,
  validateFieldElement,
  validateRounds,
  validateSeed,
  validateVector,
} from './validation';

export function modPos(v: number, q = FIELD_MODULUS): number {
  const r = v % q;
  return r < 0 ? r + q : r;
}

export function gcd(a: number, b: number): number {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y !== 0) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x;
}

export function isPrime(n: number): boolean {
  if (n < 2) return false;
  if (n % 2 === 0) return n === 2;
  const lim = Math.floor(Math.sqrt(n));
  for (let d = 3; d <= lim; d += 2) {
    if (n % d === 0) return false;
  }
  return true;
}

export function modPow(base: number, exp: number, q = FIELD_MODULUS): number {
  let result = 1;
  let cur = modPos(base, q);
  let e = exp;
  while (e > 0) {
    if ((e & 1) === 1) {
      result = (result * cur) % q;
    }
    cur = (cur * cur) % q;
    e >>>= 1;
  }
  return result;
}

export function modInverse(a: number, q: number = FIELD_MODULUS): number {
  const av = validateAlpha(a, 'a', q);
  let t = 0;
  let newT = 1;
  let r: number = q;
  let newR = av;
  while (newR !== 0) {
    const quotient = Math.floor(r / newR);
    [t, newT] = [newT, t - quotient * newT];
    [r, newR] = [newR, r - quotient * newR];
  }
  if (r > 1) {
    throw new QH4InternalStateError('NOT_INVERTIBLE', `Element ${av} is not invertible mod ${q}.`);
  }
  const inv = t < 0 ? t + q : t;
  if ((av * inv) % q !== 1) {
    throw new QH4InternalStateError('MOD_INV_FAILED', `Modular inverse verification failed for ${av}.`);
  }
  return inv;
}

export function scalarForwardElement(x: number, alpha: number, beta: number, q = FIELD_MODULUS): number {
  const xv = validateFieldElement(x, 'x', q);
  const av = validateAlpha(alpha, 'alpha', q);
  const bv = validateBeta(beta, 'beta', q);
  const x5 = modPow(xv, FORWARD_EXPONENT, q);
  return (av * x5 + bv) % q;
}

export function scalarInverseElement(y: number, alpha: number, beta: number, q = FIELD_MODULUS): number {
  const yv = validateFieldElement(y, 'y', q);
  const av = validateAlpha(alpha, 'alpha', q);
  const bv = validateBeta(beta, 'beta', q);
  const aInv = modInverse(av, q);
  const inner = (aInv * modPos(yv - bv, q)) % q;
  return modPow(inner, INVERSE_EXPONENT, q);
}

export function scalarForwardVector(x: number[], alphas: number[], betas: number[], q = FIELD_MODULUS): number[] {
  const xv = validateVector(x, undefined, 'x', q);
  const { alphas: av, betas: bv } = validateCoefficients(alphas, betas, xv.length, q);
  return xv.map((val, i) => scalarForwardElement(val, av[i], bv[i], q));
}

export function scalarInverseVector(y: number[], alphas: number[], betas: number[], q = FIELD_MODULUS): number[] {
  const yv = validateVector(y, undefined, 'y', q);
  const { alphas: av, betas: bv } = validateCoefficients(alphas, betas, yv.length, q);
  return yv.map((val, i) => scalarInverseElement(val, av[i], bv[i], q));
}

export function mixingForwardRound(x: number[], q = FIELD_MODULUS): number[] {
  const buf = validateVector(x, undefined, 'x', q);
  const n = buf.length;
  for (let i = 0; i < n - 1; i++) {
    buf[i] = (buf[i] + buf[i + 1]) % q;
  }
  for (let i = n - 1; i >= 1; i--) {
    buf[i] = (buf[i] + buf[i - 1]) % q;
  }
  return buf;
}

export function mixingInverseRound(y: number[], q = FIELD_MODULUS): number[] {
  const buf = validateVector(y, undefined, 'y', q);
  const n = buf.length;
  for (let i = 1; i < n; i++) {
    buf[i] = modPos(buf[i] - buf[i - 1], q);
  }
  for (let i = n - 2; i >= 0; i--) {
    buf[i] = modPos(buf[i] - buf[i + 1], q);
  }
  return buf;
}

export function transformForward(
  x: number[],
  alphas: number[],
  betas: number[],
  rounds?: number,
  q = FIELD_MODULUS
): number[] {
  const xv = validateVector(x, undefined, 'x', q);
  const nRounds = validateRounds(rounds ?? xv.length);
  const { alphas: av, betas: bv } = validateCoefficients(alphas, betas, xv.length, q);
  let state = scalarForwardVector(xv, av, bv, q);
  for (let r = 0; r < nRounds; r++) {
    state = mixingForwardRound(state, q);
  }
  return state;
}

export function transformInverse(
  y: number[],
  alphas: number[],
  betas: number[],
  rounds?: number,
  q = FIELD_MODULUS
): number[] {
  const yv = validateVector(y, undefined, 'y', q);
  const nRounds = validateRounds(rounds ?? yv.length);
  const { alphas: av, betas: bv } = validateCoefficients(alphas, betas, yv.length, q);
  let state = [...yv];
  for (let r = 0; r < nRounds; r++) {
    state = mixingInverseRound(state, q);
  }
  return scalarInverseVector(state, av, bv, q);
}

/** Deterministic FNV-1a / SplitMix32 PRNG for cross-browser/node reproducible coefficients & presets. */
function hashStringToUint32(input: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function deriveCoefficients(
  configName: ConfigName,
  seed = 42,
  q = FIELD_MODULUS
): { alphas: number[]; betas: number[] } {
  const cfgName = validateConfigName(configName);
  const s = validateSeed(seed);
  const cfg = BLOCK_CONFIGS[cfgName];
  const alphas: number[] = [];
  const betas: number[] = [];
  for (let i = 0; i < cfg.dim; i++) {
    const blockIdx = Math.floor(i / cfg.blockWidth);
    const laneIdx = i % cfg.blockWidth;
    const hA = hashStringToUint32(`QH4-v0.2.0|A|${cfg.name}|${s}|${blockIdx}|${laneIdx}`);
    const hB = hashStringToUint32(`QH4-v0.2.0|B|${cfg.name}|${s}|${blockIdx}|${laneIdx}`);
    const hA2 = hashStringToUint32(`QH4-v0.2.0|A2|${cfg.name}|${s}|${i}`);
    const hB2 = hashStringToUint32(`QH4-v0.2.0|B2|${cfg.name}|${s}|${i}`);
    const combinedA = (hA ^ (hA2 << 7)) >>> 0;
    const combinedB = (hB ^ (hB2 << 11)) >>> 0;
    alphas.push(1 + (combinedA % (q - 1)));
    betas.push(combinedB % q);
  }
  return { alphas, betas };
}

export function buildPresetVector(configName: ConfigName, preset: VectorPresetId, seed = 42): number[] {
  const cfgName = validateConfigName(configName);
  const s = validateSeed(seed);
  const { dim } = BLOCK_CONFIGS[cfgName];

  switch (preset) {
    case 'zero_vector':
      return new Array(dim).fill(0);
    case 'boundary_max':
      return new Array(dim).fill(FIELD_MODULUS - 1);
    case 'single_spike':
      return Array.from({ length: dim }, (_, i) => (i === 0 ? 1 : 0));
    case 'sequential_ramp':
      return Array.from({ length: dim }, (_, i) => ((i + 1) * 1973) % FIELD_MODULUS);
    case 'alternating_extremes':
      return Array.from({ length: dim }, (_, i) => (i % 2 === 0 ? 0 : FIELD_MODULUS - 1));
    case 'seeded_random':
    default: {
      return Array.from({ length: dim }, (_, i) => {
        const h = hashStringToUint32(`QH4-VEC|${cfgName}|${s}|${i}`);
        return h % FIELD_MODULUS;
      });
    }
  }
}

export interface TransformStage {
  index: number;
  direction: 'forward' | 'inverse';
  kind: 'input' | 'scalar' | 'sweep1' | 'sweep2' | 'output';
  roundNum: number;
  title: string;
  formula: string;
  values: number[];
  changedIndices: number[];
  deltasFromPrev: number[];
}

export interface TransformTraceResult {
  config: ConfigName;
  blocks: number;
  dim: number;
  blockWidth: number;
  rounds: number;
  alphas: number[];
  betas: number[];
  input: number[];
  output: number[];
  reconstructed: number[];
  exactMatch: boolean;
  inputUnmutated: boolean;
  stages: TransformStage[];
}

function diffArrays(prev: number[], next: number[], q = FIELD_MODULUS): { changed: number[]; deltas: number[] } {
  const changed: number[] = [];
  const deltas: number[] = [];
  for (let i = 0; i < next.length; i++) {
    const d = modPos(next[i] - prev[i], q);
    deltas.push(d);
    if (next[i] !== prev[i]) {
      changed.push(i);
    }
  }
  return { changed, deltas };
}

export function transformWithTrace(
  configName: ConfigName,
  x: number[],
  alphas: number[],
  betas: number[],
  rounds?: number,
  q = FIELD_MODULUS
): TransformTraceResult {
  const cfgName = validateConfigName(configName);
  const cfg = BLOCK_CONFIGS[cfgName];
  const xv = validateVector(x, cfg.dim, 'x', q);
  const inputSnapshot = [...xv];
  const nRounds = validateRounds(rounds ?? cfg.defaultRounds);
  const { alphas: av, betas: bv } = validateCoefficients(alphas, betas, cfg.dim, q);

  const stages: TransformStage[] = [];
  let idx = 0;

  stages.push({
    index: idx++,
    direction: 'forward',
    kind: 'input',
    roundNum: 0,
    title: 'Initial Input Vector x',
    formula: `x ∈ F_${q}^${cfg.dim}`,
    values: [...xv],
    changedIndices: [],
    deltasFromPrev: new Array(cfg.dim).fill(0),
  });

  // Scalar Layer S
  let prev = [...xv];
  let cur = scalarForwardVector(xv, av, bv, q);
  let diff = diffArrays(prev, cur, q);
  stages.push({
    index: idx++,
    direction: 'forward',
    kind: 'scalar',
    roundNum: 0,
    title: 'Forward Scalar Layer S',
    formula: `x_i ← (α_i · x_i^5 + β_i) mod ${q}`,
    values: [...cur],
    changedIndices: diff.changed,
    deltasFromPrev: diff.deltas,
  });

  // Forward Mixing Rounds L^n
  for (let r = 1; r <= nRounds; r++) {
    prev = [...cur];
    const sweep1 = [...cur];
    for (let i = 0; i < cfg.dim - 1; i++) {
      sweep1[i] = (sweep1[i] + sweep1[i + 1]) % q;
    }
    diff = diffArrays(prev, sweep1, q);
    stages.push({
      index: idx++,
      direction: 'forward',
      kind: 'sweep1',
      roundNum: r,
      title: `Forward Round L${r} · Sweep 1`,
      formula: `x_i ← (x_i + x_{i+1}) mod ${q}, i = 0..${cfg.dim - 2}`,
      values: [...sweep1],
      changedIndices: diff.changed,
      deltasFromPrev: diff.deltas,
    });

    prev = [...sweep1];
    const sweep2 = [...sweep1];
    for (let i = cfg.dim - 1; i >= 1; i--) {
      sweep2[i] = (sweep2[i] + sweep2[i - 1]) % q;
    }
    diff = diffArrays(prev, sweep2, q);
    stages.push({
      index: idx++,
      direction: 'forward',
      kind: 'sweep2',
      roundNum: r,
      title: `Forward Round L${r} · Sweep 2`,
      formula: `x_i ← (x_i + x_{i-1}) mod ${q}, i = ${cfg.dim - 1}..1`,
      values: [...sweep2],
      changedIndices: diff.changed,
      deltasFromPrev: diff.deltas,
    });
    cur = sweep2;
  }

  const output = [...cur];
  const outputDiff = diffArrays(xv, output, q);
  stages.push({
    index: idx++,
    direction: 'forward',
    kind: 'output',
    roundNum: nRounds,
    title: `Forward Output y = T_${nRounds}(x)`,
    formula: `y = (L^${nRounds} ∘ S)(x) · net Δ vs Input x`,
    values: [...output],
    changedIndices: outputDiff.changed,
    deltasFromPrev: outputDiff.deltas,
  });

  // Inverse Mixing Rounds L^{-n}
  let invCur = [...output];
  for (let r = nRounds; r >= 1; r--) {
    prev = [...invCur];
    const undoSweep2 = [...invCur];
    for (let i = 1; i < cfg.dim; i++) {
      undoSweep2[i] = modPos(undoSweep2[i] - undoSweep2[i - 1], q);
    }
    diff = diffArrays(prev, undoSweep2, q);
    stages.push({
      index: idx++,
      direction: 'inverse',
      kind: 'sweep2',
      roundNum: r,
      title: `Inverse Round L${r}⁻¹ · Undo Sweep 2`,
      formula: `x_i ← (x_i - x_{i-1}) mod ${q}, i = 1..${cfg.dim - 1}`,
      values: [...undoSweep2],
      changedIndices: diff.changed,
      deltasFromPrev: diff.deltas,
    });

    prev = [...undoSweep2];
    const undoSweep1 = [...undoSweep2];
    for (let i = cfg.dim - 2; i >= 0; i--) {
      undoSweep1[i] = modPos(undoSweep1[i] - undoSweep1[i + 1], q);
    }
    diff = diffArrays(prev, undoSweep1, q);
    stages.push({
      index: idx++,
      direction: 'inverse',
      kind: 'sweep1',
      roundNum: r,
      title: `Inverse Round L${r}⁻¹ · Undo Sweep 1`,
      formula: `x_i ← (x_i - x_{i+1}) mod ${q}, i = ${cfg.dim - 2}..0`,
      values: [...undoSweep1],
      changedIndices: diff.changed,
      deltasFromPrev: diff.deltas,
    });
    invCur = undoSweep1;
  }

  // Inverse Scalar Layer S^{-1}
  prev = [...invCur];
  const reconstructed = scalarInverseVector(invCur, av, bv, q);
  diff = diffArrays(prev, reconstructed, q);
  stages.push({
    index: idx++,
    direction: 'inverse',
    kind: 'scalar',
    roundNum: 0,
    title: 'Inverse Scalar Layer S⁻¹ (Recovered x̂)',
    formula: `x_i ← (α_i⁻¹ · (y_i - β_i))^52429 mod ${q}`,
    values: [...reconstructed],
    changedIndices: diff.changed,
    deltasFromPrev: diff.deltas,
  });

  const exactMatch = reconstructed.every((v, i) => v === inputSnapshot[i]);
  if (!exactMatch) {
    throw new QH4InternalStateError(
      'ROUND_TRIP_MISMATCH',
      'Internal invariant violation: reconstructed vector x̂ does not match input vector x.'
    );
  }

  const inputUnmutated = x.every((v, i) => v === inputSnapshot[i]);

  return {
    config: cfgName,
    blocks: cfg.blocks,
    dim: cfg.dim,
    blockWidth: cfg.blockWidth,
    rounds: nRounds,
    alphas: av,
    betas: bv,
    input: inputSnapshot,
    output,
    reconstructed,
    exactMatch,
    inputUnmutated,
    stages,
  };
}

export function fnvChecksumHex(vec: number[]): string {
  // Deterministic synchronous digest for UI display; SHA-256 is computed via SubtleCrypto or Node crypto
  let h1 = 0x811c9dc5 >>> 0;
  let h2 = 0x9e3779b9 >>> 0;
  for (let i = 0; i < vec.length; i++) {
    h1 ^= vec[i] & 0xffff;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= (vec[i] >>> 0) + i * 31;
    h2 = Math.imul(h2, 0x85ebca6b) >>> 0;
  }
  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}
