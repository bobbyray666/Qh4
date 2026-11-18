// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto';
import { NextRequest, NextResponse } from 'next/server';
import { handleCorsPreflight, verifyApiAuthorization, withCors } from '@/lib/qh4/access';
import { BLOCK_CONFIGS } from '@/lib/qh4/constants';
import { deriveCoefficients, transformWithTrace } from '@/lib/qh4/math';
import {
  parseVectorText,
  QH4InternalStateError,
  QH4ValidationError,
  validateCoefficients,
  validateConfigName,
  validateRounds,
  validateSeed,
  validateVector,
} from '@/lib/qh4/validation';

function sha256Vector(vec: number[]): string {
  return createHash('sha256').update(vec.join(',')).digest('hex');
}

export async function OPTIONS(req: NextRequest) {
  return handleCorsPreflight(req);
}

export async function POST(req: NextRequest) {
  const auth = verifyApiAuthorization(req);
  if (!auth.authorized) {
    return auth.response;
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return withCors(
      NextResponse.json(
        {
          ok: false,
          error: {
            code: 'INVALID_JSON_BODY',
            message: 'Request body must be valid JSON.',
          },
        },
        { status: 400 }
      ),
      req
    );
  }

  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return withCors(
      NextResponse.json(
        {
          ok: false,
          error: {
            code: 'INVALID_PAYLOAD',
            message: 'Request payload must be a JSON object.',
          },
        },
        { status: 400 }
      ),
      req
    );
  }

  try {
    const payload = body as Record<string, unknown>;
    const configName = validateConfigName(payload.config);
    const cfg = BLOCK_CONFIGS[configName];
    const seed = validateSeed(payload.seed ?? 42);
    const rounds = validateRounds(payload.rounds ?? cfg.defaultRounds);

    let vector: number[];
    if (typeof payload.vectorText === 'string') {
      vector = parseVectorText(payload.vectorText, cfg.dim, 'vectorText');
    } else if (payload.vector !== undefined) {
      vector = validateVector(payload.vector, cfg.dim, 'vector');
    } else {
      throw new QH4ValidationError('MISSING_VECTOR', 'Provide either "vector" array or "vectorText" string.', 'vector');
    }

    const derived = deriveCoefficients(configName, seed);
    let alphas = derived.alphas;
    let betas = derived.betas;

    if (payload.alphas !== undefined || payload.betas !== undefined) {
      const checked = validateCoefficients(
        payload.alphas ?? derived.alphas,
        payload.betas ?? derived.betas,
        cfg.dim
      );
      alphas = checked.alphas;
      betas = checked.betas;
    }

    const trace = transformWithTrace(configName, vector, alphas, betas, rounds);

    return withCors(
      NextResponse.json({
        ok: true,
        trace,
        checksums: {
          inputSha256: sha256Vector(trace.input),
          outputSha256: sha256Vector(trace.output),
          reconstructedSha256: sha256Vector(trace.reconstructed),
        },
      }),
      req
    );
  } catch (err) {
    if (err instanceof QH4ValidationError) {
      return withCors(
        NextResponse.json(
          {
            ok: false,
            error: {
              code: err.code,
              message: err.message,
              field: err.field,
              details: err.details,
            },
          },
          { status: 400 }
        ),
        req
      );
    }
    if (err instanceof QH4InternalStateError) {
      return withCors(
        NextResponse.json(
          {
            ok: false,
            error: {
              code: err.code,
              message: err.message,
            },
          },
          { status: 500 }
        ),
        req
      );
    }
    return withCors(
      NextResponse.json(
        {
          ok: false,
          error: {
            code: 'UNEXPECTED_ERROR',
            message: err instanceof Error ? err.message : 'Unexpected server error.',
          },
        },
        { status: 500 }
      ),
      req
    );
  }
}
