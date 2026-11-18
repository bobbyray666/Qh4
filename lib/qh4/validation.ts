// SPDX-License-Identifier: Apache-2.0

import { BLOCK_CONFIGS, ConfigName, FIELD_MODULUS } from './constants';

export class QH4ValidationError extends Error {
  readonly code: string;
  readonly field?: string;
  readonly details?: Record<string, unknown>;

  constructor(code: string, message: string, field?: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'QH4ValidationError';
    this.code = code;
    this.field = field;
    this.details = details;
  }
}

export class QH4InternalStateError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = 'QH4InternalStateError';
    this.code = code;
  }
}

export function validateConfigName(name: unknown): ConfigName {
  if (typeof name !== 'string' || !(name in BLOCK_CONFIGS)) {
    throw new QH4ValidationError(
      'UNKNOWN_CONFIGURATION',
      `Configuration "${String(name)}" is invalid. Select one of: ${Object.keys(BLOCK_CONFIGS).join(', ')}.`,
      'config'
    );
  }
  return name as ConfigName;
}

export function validateStrictInteger(value: unknown, field: string): number {
  if (typeof value === 'boolean') {
    throw new QH4ValidationError(
      'INVALID_TYPE_BOOL',
      `Parameter "${field}" must be an integer, received boolean (${String(value)}).`,
      field
    );
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new QH4ValidationError(
      'INVALID_NUMBER',
      `Parameter "${field}" must be a finite integer (received ${String(value)}).`,
      field
    );
  }
  if (!Number.isInteger(value)) {
    throw new QH4ValidationError(
      'NON_INTEGER_VALUE',
      `Parameter "${field}" must be an exact integer without decimals (received ${value}).`,
      field
    );
  }
  if (!Number.isSafeInteger(value)) {
    throw new QH4ValidationError(
      'UNSAFE_INTEGER',
      `Parameter "${field}" exceeds JavaScript safe integer bounds.`,
      field
    );
  }
  return value;
}

export function validateFieldElement(value: unknown, field = 'x', q = FIELD_MODULUS): number {
  const n = validateStrictInteger(value, field);
  if (n < 0 || n >= q) {
    throw new QH4ValidationError(
      'FIELD_RANGE_ERROR',
      `Element ${field} = ${n} is outside F_${q} valid range [0, ${q - 1}].`,
      field,
      { value: n, min: 0, max: q - 1 }
    );
  }
  return n;
}

export function validateAlpha(value: unknown, field = 'alpha', q = FIELD_MODULUS): number {
  const n = validateFieldElement(value, field, q);
  if (n === 0) {
    throw new QH4ValidationError(
      'ALPHA_ZERO_NON_INVERTIBLE',
      `Coefficient ${field} cannot be 0 because 0 has no inverse in F_${q}. Valid range is [1, ${q - 1}].`,
      field
    );
  }
  return n;
}

export function validateBeta(value: unknown, field = 'beta', q = FIELD_MODULUS): number {
  return validateFieldElement(value, field, q);
}

export function validateRounds(rounds: unknown, maxRounds = 256): number {
  const n = validateStrictInteger(rounds, 'rounds');
  if (n < 1 || n > maxRounds) {
    throw new QH4ValidationError(
      'INVALID_ROUND_COUNT',
      `Mixing rounds (${n}) must be an integer between 1 and ${maxRounds}.`,
      'rounds',
      { min: 1, max: maxRounds, value: n }
    );
  }
  return n;
}

export function validateSeed(seed: unknown): number {
  const n = validateStrictInteger(seed, 'seed');
  if (n < 0 || n > 2147483647) {
    throw new QH4ValidationError(
      'INVALID_SEED',
      `Seed (${n}) must be an integer between 0 and 2,147,483,647.`,
      'seed'
    );
  }
  return n;
}

export function validateVector(vec: unknown, expectedDim?: number, name = 'x', q = FIELD_MODULUS): number[] {
  if (!Array.isArray(vec)) {
    throw new QH4ValidationError(
      'INVALID_VECTOR_CONTAINER',
      `Vector "${name}" must be a 1D array of integers.`,
      name
    );
  }
  if (vec.length < 2) {
    throw new QH4ValidationError(
      'VECTOR_TOO_SHORT',
      `Vector "${name}" requires at least 2 elements for two-sweep mixing (received ${vec.length}).`,
      name
    );
  }
  if (expectedDim !== undefined && vec.length !== expectedDim) {
    throw new QH4ValidationError(
      'DIMENSION_MISMATCH',
      `Vector length mismatch: expected ${expectedDim} elements for selected configuration, received ${vec.length}.`,
      name,
      { expected: expectedDim, actual: vec.length }
    );
  }
  return vec.map((v, i) => validateFieldElement(v, `${name}[${i}]`, q));
}

export function validateCoefficients(
  alphas: unknown,
  betas: unknown,
  expectedDim: number,
  q = FIELD_MODULUS
): { alphas: number[]; betas: number[] } {
  if (!Array.isArray(alphas) || alphas.length !== expectedDim) {
    throw new QH4ValidationError(
      'ALPHA_LENGTH_MISMATCH',
      `Expected ${expectedDim} alpha coefficients, received ${Array.isArray(alphas) ? alphas.length : typeof alphas}.`,
      'alphas'
    );
  }
  if (!Array.isArray(betas) || betas.length !== expectedDim) {
    throw new QH4ValidationError(
      'BETA_LENGTH_MISMATCH',
      `Expected ${expectedDim} beta coefficients, received ${Array.isArray(betas) ? betas.length : typeof betas}.`,
      'betas'
    );
  }
  return {
    alphas: alphas.map((a, i) => validateAlpha(a, `alpha[${i}]`, q)),
    betas: betas.map((b, i) => validateBeta(b, `beta[${i}]`, q)),
  };
}

export function parseVectorText(raw: string, expectedDim: number, field = 'vectorText'): number[] {
  if (typeof raw !== 'string') {
    throw new QH4ValidationError('INVALID_VECTOR_TEXT', 'Vector input must be text.', field);
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    throw new QH4ValidationError(
      'EMPTY_VECTOR_INPUT',
      `Vector input is empty. Provide ${expectedDim} comma- or space-separated integers in [0, ${FIELD_MODULUS - 1}].`,
      field
    );
  }

  let tokens: string[];
  if (trimmed.startsWith('[')) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      throw new QH4ValidationError(
        'MALFORMED_JSON_VECTOR',
        'JSON vector syntax is invalid. Example: [10, 2048, 65536, ...].',
        field
      );
    }
    if (!Array.isArray(parsed)) {
      throw new QH4ValidationError('INVALID_JSON_ARRAY', 'JSON input must be a flat array of integers.', field);
    }
    return validateVector(parsed, expectedDim, 'x');
  } else {
    tokens = trimmed.split(/[\s,;]+/).filter(Boolean);
  }

  if (tokens.length !== expectedDim) {
    throw new QH4ValidationError(
      'DIMENSION_MISMATCH',
      `Expected ${expectedDim} elements for this configuration, but parsed ${tokens.length} value${tokens.length === 1 ? '' : 's'}.`,
      field,
      { expected: expectedDim, actual: tokens.length }
    );
  }

  const out: number[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (!/^-?\d+$/.test(tok)) {
      throw new QH4ValidationError(
        'INVALID_TOKEN',
        `Token "${tok}" at index ${i} is not a valid integer. Only base-10 integers in [0, ${FIELD_MODULUS - 1}] are permitted.`,
        field,
        { index: i, token: tok }
      );
    }
    const val = Number(tok);
    out.push(validateFieldElement(val, `x[${i}]`));
  }
  return out;
}
