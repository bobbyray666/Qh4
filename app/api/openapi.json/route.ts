// SPDX-License-Identifier: Apache-2.0

import { NextRequest, NextResponse } from 'next/server';
import { CONFIG_NAMES } from '@/lib/qh4/constants';
import { handleCorsPreflight, resolvePublicBaseUrl, withCors } from '@/lib/qh4/access';

export async function OPTIONS(req: NextRequest) {
  return handleCorsPreflight(req);
}

export async function GET(req: NextRequest) {
  const baseUrl = resolvePublicBaseUrl(req);
  const hasToken = Boolean(process.env.QH4_API_TOKEN?.trim());

  const schema = {
    openapi: '3.1.0',
    info: {
      title: 'QH4-FieldPermutation v0.2.0 API',
      version: '0.2.0',
      description:
        'Experimental F_65537 scalar and two-sweep mixing transformation workbench (T_n = L^n o S) with architecture overview, source inspection, mathematical verification, and Pelagos-style CPU reference execution.',
    },
    servers: [{ url: baseUrl }],
    ...(hasToken
      ? {
          components: {
            securitySchemes: {
              BearerAuth: {
                type: 'http',
                scheme: 'bearer',
              },
            },
          },
          security: [{ BearerAuth: [] }],
        }
      : {}),
    paths: {
      '/api/qh4/overview': {
        get: {
          operationId: 'getAppOverview',
          summary:
            'Retrieve complete application overview, math spec, all 8 block configs, live M01-M10 verification results, API endpoints, and readable source file inventory.',
          responses: {
            '200': { description: 'Complete structured application overview JSON.' },
            '401': { description: 'Unauthorized when QH4_API_TOKEN is enabled.' },
          },
        },
      },
      '/api/qh4/source': {
        get: {
          operationId: 'readAppSourceFile',
          summary:
            'List or read whitelisted source files from both the Next.js workbench and the isolated qh4-experimental Python repository.',
          parameters: [
            {
              name: 'file',
              in: 'query',
              required: false,
              description:
                'Relative file path (e.g. "app/page.tsx", "lib/qh4/math.ts", "qh4_experimental/cpu_reference.py"). Omit or pass list=1 to discover all files.',
              schema: { type: 'string' },
            },
            {
              name: 'list',
              in: 'query',
              required: false,
              description: 'Set to "1" to return only the whitelisted file lists without file content.',
              schema: { type: 'string', enum: ['1'] },
            },
          ],
          responses: {
            '200': { description: 'File contents and whitelisted file inventory.' },
            '400': { description: 'Requested file is not in the whitelisted file list.' },
            '401': { description: 'Unauthorized when QH4_API_TOKEN is enabled.' },
          },
        },
      },
      '/api/qh4/bundle': {
        get: {
          operationId: 'exportSourceBundle',
          summary: 'Export the complete whitelisted Next.js and qh4-experimental source codebase in a single JSON or Markdown bundle.',
          parameters: [
            {
              name: 'format',
              in: 'query',
              required: false,
              schema: { type: 'string', enum: ['json', 'markdown'], default: 'json' },
            },
            {
              name: 'scope',
              in: 'query',
              required: false,
              schema: { type: 'string', enum: ['all', 'qh4-experimental', 'nextjs-app'], default: 'all' },
            },
          ],
          responses: {
            '200': { description: 'Complete source bundle.' },
            '401': { description: 'Unauthorized when QH4_API_TOKEN is enabled.' },
          },
        },
      },
      '/api/qh4/trace': {
        post: {
          operationId: 'executeQh4Trace',
          summary: 'Execute forward and inverse QH4 transformation with full stage-by-stage trace.',
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  required: ['config'],
                  properties: {
                    config: { type: 'string', enum: CONFIG_NAMES },
                    vector: {
                      type: 'array',
                      description: 'Required unless vectorText is provided. Integer vector matching the config dimension.',
                      items: { type: 'integer', minimum: 0, maximum: 65536 },
                    },
                    vectorText: {
                      type: 'string',
                      description: 'Alternative comma/whitespace-separated integer string for the input vector.',
                    },
                    seed: { type: 'integer', minimum: 0, maximum: 2147483647, default: 42 },
                    rounds: { type: 'integer', minimum: 1, maximum: 256 },
                  },
                },
              },
            },
          },
          responses: {
            '200': { description: 'Validated forward/inverse transformation trace and SHA-256 digests.' },
            '400': { description: 'Validation error with structured code and field diagnostic.' },
            '401': { description: 'Unauthorized when QH4_API_TOKEN is enabled.' },
          },
        },
      },
      '/api/qh4/python-verify': {
        post: {
          operationId: 'runPythonReferenceVerification',
          summary: 'Run Python 3.11 M01-M10 checks, 8-config validation, or Pelagos CPU benchmark.',
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  required: ['mode'],
                  properties: {
                    mode: {
                      type: 'string',
                      enum: [
                        'm01_m10',
                        'configs',
                        'benchmark',
                        'triton_probe',
                        'differential_test',
                        'cuda_triton_compile',
                        'triton_interpret_test',
                        'remote_gpu_test',
                      ],
                    },
                    config: { type: 'string', enum: CONFIG_NAMES },
                    vectorsPerConfig: { type: 'integer', minimum: 10, maximum: 10000 },
                    batchSize: { type: 'integer', minimum: 1, maximum: 4096 },
                    workerCount: { type: 'integer', minimum: 1, maximum: 8 },
                    seed: { type: 'integer', minimum: 0, maximum: 2147483647 },
                  },
                },
              },
            },
          },
          responses: {
            '200': { description: 'Verification or benchmark telemetry report.' },
            '400': { description: 'Parameter validation error.' },
          },
        },
      },
    },
  };

  return withCors(NextResponse.json(schema), req);
}
