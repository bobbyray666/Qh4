// SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';
import { BLOCK_CONFIGS, CONFIG_NAMES, FIELD_MODULUS } from './constants';
import {
  buildPresetVector,
  deriveCoefficients,
  transformForward,
  transformInverse,
  transformWithTrace,
} from './math';
import { runTypeScriptM01ToM10 } from './tests';
import { parseVectorText, QH4ValidationError, validateAlpha, validateRounds, validateVector } from './validation';

test('M01-M10 mathematical verification suite passes completely', () => {
  const report = runTypeScriptM01ToM10();
  assert.equal(report.allPassed, true, JSON.stringify(report.checks.filter((c) => !c.passed)));
  assert.equal(report.checks.length, 10);
});

test('All 8 block configurations pass multi-seed round-trip and non-mutation', () => {
  for (const cfgName of CONFIG_NAMES) {
    const cfg = BLOCK_CONFIGS[cfgName];
    for (let seed = 100; seed < 110; seed++) {
      const { alphas, betas } = deriveCoefficients(cfgName, seed);
      const x = buildPresetVector(cfgName, 'seeded_random', seed);
      const copy = [...x];
      const trace = transformWithTrace(cfgName, x, alphas, betas, cfg.defaultRounds);
      assert.equal(trace.exactMatch, true);
      assert.equal(trace.inputUnmutated, true);
      assert.deepEqual(x, copy);
      assert.equal(trace.stages[0].values.length, cfg.dim);
      assert.equal(trace.stages[trace.stages.length - 1].values.length, cfg.dim);
      assert.deepEqual(trace.reconstructed, x);
    }
  }
});

test('Comprehensive input validation rejects malformed inputs with clear errors', () => {
  assert.throws(() => parseVectorText('', 8), (err: unknown) => {
    return err instanceof QH4ValidationError && err.code === 'EMPTY_VECTOR_INPUT';
  });

  assert.throws(() => parseVectorText('1, 2, 65537, 4, 5, 6, 7, 8', 8), (err: unknown) => {
    return err instanceof QH4ValidationError && err.code === 'FIELD_RANGE_ERROR';
  });

  assert.throws(() => parseVectorText('1, -5, 3, 4, 5, 6, 7, 8', 8), (err: unknown) => {
    return err instanceof QH4ValidationError && err.code === 'FIELD_RANGE_ERROR';
  });

  assert.throws(() => parseVectorText('1, 2.5, 3, 4, 5, 6, 7, 8', 8), (err: unknown) => {
    return err instanceof QH4ValidationError && err.code === 'INVALID_TOKEN';
  });

  assert.throws(() => parseVectorText('1, 2, 3', 8), (err: unknown) => {
    return err instanceof QH4ValidationError && err.code === 'DIMENSION_MISMATCH';
  });

  assert.throws(() => validateAlpha(0), (err: unknown) => {
    return err instanceof QH4ValidationError && err.code === 'ALPHA_ZERO_NON_INVERTIBLE';
  });

  assert.throws(() => validateRounds(0), (err: unknown) => {
    return err instanceof QH4ValidationError && err.code === 'INVALID_ROUND_COUNT';
  });

  assert.throws(() => validateVector([1, Number.NaN, 3, 4], 4), (err: unknown) => {
    return err instanceof QH4ValidationError && err.code === 'INVALID_NUMBER';
  });

  const parsed = parseVectorText('[0, 1, 2, 3, 4, 5, 6, 65536]', 8);
  assert.equal(parsed[7], FIELD_MODULUS - 1);
});

test('Forward and inverse custom round counts match trace output', () => {
  const { alphas, betas } = deriveCoefficients('8B-D8', 77);
  const x = [0, 1, 255, 1024, 32768, 50000, 65535, 65536];
  const y = transformForward(x, alphas, betas, 4);
  const rec = transformInverse(y, alphas, betas, 4);
  assert.deepEqual(rec, x);

  const trace = transformWithTrace('8B-D8', x, alphas, betas, 4);
  // Stages: input(1) + scalar(1) + 4*(sweep1+sweep2)(8) + output(1) + 4*(undo2+undo1)(8) + inv_scalar(1) = 20
  assert.equal(trace.stages.length, 20);
  const outputStage = trace.stages.find((s) => s.kind === 'output');
  assert.ok(outputStage);
  assert.deepEqual(outputStage.values, y);
});
