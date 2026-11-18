// SPDX-License-Identifier: Apache-2.0

import { BLOCK_CONFIGS, CONFIG_NAMES, FIELD_MODULUS, FORWARD_EXPONENT, INVERSE_EXPONENT } from './constants';
import {
  buildPresetVector,
  deriveCoefficients,
  gcd,
  isPrime,
  mixingForwardRound,
  mixingInverseRound,
  modInverse,
  modPow,
  scalarForwardElement,
  scalarInverseElement,
  transformForward,
  transformInverse,
} from './math';
import { QH4ValidationError } from './validation';

export interface MathCheckResult {
  id: string;
  name: string;
  passed: boolean;
  durationMs: number;
  detail: string;
}

export function runTypeScriptM01ToM10(): {
  allPassed: boolean;
  totalDurationMs: number;
  checks: MathCheckResult[];
} {
  const tStart = performance.now();
  const checks: MathCheckResult[] = [];

  const runCheck = (id: string, name: string, fn: () => string) => {
    const t0 = performance.now();
    try {
      const detail = fn();
      checks.push({ id, name, passed: true, durationMs: Number((performance.now() - t0).toFixed(2)), detail });
    } catch (err) {
      checks.push({
        id,
        name,
        passed: false,
        durationMs: Number((performance.now() - t0).toFixed(2)),
        detail: err instanceof Error ? err.message : String(err),
      });
    }
  };

  runCheck('M01', 'Modulus Primality (q = 65,537)', () => {
    if (FIELD_MODULUS !== 65537 || !isPrime(FIELD_MODULUS)) {
      throw new Error(`Expected prime 65537, got ${FIELD_MODULUS}`);
    }
    return 'q = 2^16 + 1 = 65,537 verified prime.';
  });

  runCheck('M02', 'Coprimality gcd(5, 65536) = 1 & gcd(α, q) = 1', () => {
    if (gcd(FORWARD_EXPONENT, FIELD_MODULUS - 1) !== 1) {
      throw new Error('gcd(5, q-1) != 1');
    }
    for (const a of [1, 2, 17, 257, 32768, 65536]) {
      if (gcd(a, FIELD_MODULUS) !== 1) throw new Error(`gcd(${a}, q) != 1`);
    }
    return 'Exponent d=5 coprime to phi(q)=65,536; nonzero alphas coprime to q.';
  });

  runCheck('M03', 'Inverse Exponent Identity (5·52429 ≡ 1 mod 65536)', () => {
    if ((FORWARD_EXPONENT * INVERSE_EXPONENT) % (FIELD_MODULUS - 1) !== 1) {
      throw new Error('5 * e != 1 mod 65536');
    }
    for (let u = 0; u < FIELD_MODULUS; u++) {
      const p5 = modPow(u, FORWARD_EXPONENT, FIELD_MODULUS);
      const rec = modPow(p5, INVERSE_EXPONENT, FIELD_MODULUS);
      if (rec !== u) throw new Error(`Exhaustive power inverse failed at u=${u}`);
    }
    return 'Exhaustive verification across all 65,537 elements passed.';
  });

  runCheck('M04', 'Exhaustive Unique Scalar Permutation S', () => {
    const seen = new Uint8Array(FIELD_MODULUS);
    const alpha = 31337;
    const beta = 42424;
    for (let x = 0; x < FIELD_MODULUS; x++) {
      const y = scalarForwardElement(x, alpha, beta);
      if (seen[y] !== 0) throw new Error(`Collision at y=${y}`);
      seen[y] = 1;
      if (scalarInverseElement(y, alpha, beta) !== x) {
        throw new Error(`Scalar inverse mismatch at x=${x}`);
      }
    }
    return '65,537 distinct outputs confirmed; bijection cardinality exact.';
  });

  runCheck('M05', 'Field Range & Input Validation Guards', () => {
    const badInputs: unknown[] = [-1, 65537, 1.5, Number.NaN, true, 'abc'];
    for (const bad of badInputs) {
      let caught = false;
      try {
        scalarForwardElement(bad as number, 1, 0);
      } catch (e) {
        if (e instanceof QH4ValidationError) caught = true;
      }
      if (!caught) throw new Error(`Validation failed to reject ${String(bad)}`);
    }
    return 'Rejected out-of-range, float, NaN, bool, and string inputs cleanly.';
  });

  runCheck('M06', 'Modular Inverse Accuracy (α·α⁻¹ ≡ 1 mod q)', () => {
    for (const a of [1, 2, 3, 5, 257, 4099, 65535, 65536]) {
      const inv = modInverse(a, FIELD_MODULUS);
      if ((a * inv) % FIELD_MODULUS !== 1) throw new Error(`Inverse failed for ${a}`);
    }
    return 'Verified α·α⁻¹ ≡ 1 (mod 65,537) across edge and boundary values.';
  });

  runCheck('M07', 'Two-Sweep Mixing Layer L / L⁻¹ Invertibility', () => {
    for (const dim of [8, 9, 16, 18, 32, 36]) {
      const vec = Array.from({ length: dim }, (_, i) => (i * 7919 + 13) % FIELD_MODULUS);
      const mixed = mixingForwardRound(vec);
      const recovered = mixingInverseRound(mixed);
      if (!recovered.every((v, i) => v === vec[i])) {
        throw new Error(`Mixing round-trip failed for dim=${dim}`);
      }
    }
    return 'Exact L⁻¹(L(x)) = x verified across dimensions 8, 9, 16, 18, 32, 36.';
  });

  runCheck('M08', 'Complete Transform Round-Trip (T_n⁻¹(T_n(x)) = x)', () => {
    for (const cfgName of CONFIG_NAMES) {
      const cfg = BLOCK_CONFIGS[cfgName];
      const { alphas, betas } = deriveCoefficients(cfgName, 808);
      const x = buildPresetVector(cfgName, 'seeded_random', 808);
      const y = transformForward(x, alphas, betas, cfg.defaultRounds);
      const rec = transformInverse(y, alphas, betas, cfg.defaultRounds);
      if (!rec.every((v, i) => v === x[i])) {
        throw new Error(`Full round-trip failed for ${cfgName}`);
      }
    }
    return 'All 8 block configurations reconstructed exact original input.';
  });

  runCheck('M09', 'Avalanche Diffusion (Single Coordinate Perturbation)', () => {
    const cfg = BLOCK_CONFIGS['8B-D16'];
    const { alphas, betas } = deriveCoefficients('8B-D16', 909);
    const x1 = new Array(cfg.dim).fill(0);
    const x2 = new Array(cfg.dim).fill(0);
    x2[0] = 1;
    const y1 = transformForward(x1, alphas, betas, cfg.defaultRounds);
    const y2 = transformForward(x2, alphas, betas, cfg.defaultRounds);
    const changed = y1.filter((v, i) => v !== y2[i]).length;
    if (changed < cfg.dim - 1) {
      throw new Error(`Insufficient diffusion: only ${changed}/${cfg.dim} elements changed`);
    }
    return `1-element perturbation diffused across ${changed}/${cfg.dim} coordinates.`;
  });

  runCheck('M10', 'Non-Mutation & Deterministic Execution', () => {
    const cfg = BLOCK_CONFIGS['9B-D18'];
    const { alphas, betas } = deriveCoefficients('9B-D18', 1010);
    const x = buildPresetVector('9B-D18', 'sequential_ramp', 1010);
    const snap = [...x];
    const y1 = transformForward(x, alphas, betas, cfg.defaultRounds);
    const y2 = transformForward(x, alphas, betas, cfg.defaultRounds);
    if (!x.every((v, i) => v === snap[i])) throw new Error('Input vector mutated');
    if (!y1.every((v, i) => v === y2[i])) throw new Error('Non-deterministic forward output');
    return 'Zero input mutation and bit-identical repeatability confirmed.';
  });

  return {
    allPassed: checks.every((c) => c.passed),
    totalDurationMs: Number((performance.now() - tStart).toFixed(2)),
    checks,
  };
}
