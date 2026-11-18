// SPDX-License-Identifier: Apache-2.0

import { NextRequest, NextResponse } from 'next/server';
import appMetadata from '@/metadata.json';
import {
  handleCorsPreflight,
  resolvePublicBaseUrl,
  verifyApiAuthorization,
  withCors,
} from '@/lib/qh4/access';
import {
  BLOCK_CONFIGS,
  FIELD_MODULUS,
  FORWARD_EXPONENT,
  INVERSE_EXPONENT,
} from '@/lib/qh4/constants';
import { runTypeScriptM01ToM10 } from '@/lib/qh4/tests';
import { NEXTJS_APP_FILES, QH4_EXPERIMENTAL_FILES } from '@/lib/qh4/sourceManifest';

export async function OPTIONS(req: NextRequest) {
  return handleCorsPreflight(req);
}

export async function GET(req: NextRequest) {
  const auth = verifyApiAuthorization(req);
  if (!auth.authorized) {
    return auth.response;
  }

  const baseUrl = resolvePublicBaseUrl(req);
  const mathVerification = runTypeScriptM01ToM10();

  return withCors(
    NextResponse.json({
      ok: true,
      app: {
        name: appMetadata.name || 'QH4-FieldPermutation',
        version: '0.2.0',
        description: appMetadata.description,
        license: 'Apache-2.0',
        baseUrl,
        architecture: {
          frontend: 'Next.js 15 App Router (TypeScript + Tailwind CSS)',
          referenceEngine: 'Isolated Python 3.11 experimental workspace (qh4-experimental/)',
          gpuPlan: 'Triton GPU kernels + Pelagos-style multi-worker process orchestration',
        },
      },
      mathSpec: {
        fieldModulus: FIELD_MODULUS,
        forwardExponent: FORWARD_EXPONENT,
        inverseExponent: INVERSE_EXPONENT,
        scalarForward: 'S_i(x_i) = (alpha_i * x_i^5 + beta_i) mod 65537',
        scalarInverse: 'S_i^-1(y_i) = (alpha_i^-1 * (y_i - beta_i))^52429 mod 65537',
        mixingRoundL: [
          'Forward Sweep 1: x_i <- (x_i + x_{i+1}) mod q for i = 0..n-2',
          'Forward Sweep 2: x_i <- (x_i + x_{i-1}) mod q for i = n-1..1',
        ],
        completeTransform: 'T_n = L^n o S',
      },
      configurations: BLOCK_CONFIGS,
      mathVerification,
      endpoints: [
        { method: 'GET', path: '/api/openapi.json', purpose: 'OpenAPI 3.1.0 specification for OpenAI Actions and external tools' },
        { method: 'GET', path: '/api/qh4/overview', purpose: 'Structured architecture, math spec, configs, M01-M10 check results, and file manifest' },
        { method: 'GET', path: '/api/qh4/source?file=<path>', purpose: 'Read any whitelisted file from Next.js app or qh4-experimental/ (use ?list=1 for file list)' },
        { method: 'GET', path: '/api/qh4/bundle?format=json|markdown', purpose: 'Export all whitelisted source files in a single JSON or Markdown bundle' },
        { method: 'POST', path: '/api/qh4/trace', purpose: 'Run forward/inverse transformation trace and SHA-256 verification' },
        { method: 'POST', path: '/api/qh4/python-verify', purpose: 'Execute Python 3.11 M01-M10 checks, config tests, or Pelagos CPU benchmark' },
      ],
      repositoryFiles: {
        qh4Experimental: QH4_EXPERIMENTAL_FILES,
        nextjsApp: NEXTJS_APP_FILES,
      },
    }),
    req
  );
}
