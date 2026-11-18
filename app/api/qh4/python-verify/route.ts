// SPDX-License-Identifier: Apache-2.0

import { execFile } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';
import { NextRequest, NextResponse } from 'next/server';
import { handleCorsPreflight, verifyApiAuthorization, withCors } from '@/lib/qh4/access';
import {
  QH4ValidationError,
  validateConfigName,
  validateSeed,
  validateStrictInteger,
} from '@/lib/qh4/validation';

const execFileAsync = promisify(execFile);

const QH4_ROOT = path.resolve(process.cwd(), 'qh4-experimental');
const TRITON_VENV_PYTHON = '/opt/qh4-triton/venv/bin/python';

function resolvePythonBin(): string {
  const configured = process.env.QH4_PYTHON_BIN?.trim();
  if (configured) return configured;
  if (fs.existsSync(TRITON_VENV_PYTHON)) return TRITON_VENV_PYTHON;
  return 'python3';
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
        { ok: false, error: { code: 'INVALID_JSON', message: 'Request body must be valid JSON.' } },
        { status: 400 }
      ),
      req
    );
  }

  try {
    const payload = (body && typeof body === 'object' ? body : {}) as Record<string, unknown>;
    const mode = payload.mode;
    if (
      mode !== 'm01_m10' &&
      mode !== 'configs' &&
      mode !== 'benchmark' &&
      mode !== 'triton_probe' &&
      mode !== 'differential_test' &&
      mode !== 'cuda_triton_compile' &&
      mode !== 'triton_interpret_test' &&
      mode !== 'remote_gpu_test'
    ) {
      throw new QH4ValidationError(
        'INVALID_MODE',
        'Mode must be one of: m01_m10, configs, benchmark, triton_probe, differential_test, cuda_triton_compile, triton_interpret_test, remote_gpu_test.',
        'mode'
      );
    }

    if (mode === 'remote_gpu_test') {
      const workerUrl = process.env.QH4_GPU_WORKER_URL?.trim();
      const batchSize = validateStrictInteger(payload.batchSize ?? 128, 'batchSize');
      const seed = validateSeed(payload.seed ?? 2026);
      const sm = validateStrictInteger(payload.sm ?? 80, 'sm');
      if (batchSize < 1 || batchSize > 4096) {
        throw new QH4ValidationError('INVALID_BATCH_SIZE', 'batchSize must be between 1 and 4,096.', 'batchSize');
      }
      if (!workerUrl) {
        return withCors(
          NextResponse.json({
            ok: true,
            mode,
            data: {
              status: 'worker_not_configured',
              message:
                'No remote physical GPU worker URL (QH4_GPU_WORKER_URL) is configured in server environment.',
              instructions: {
                colab_or_cuda_machine:
                  'python3 qh4-experimental/scripts/run_actual_cuda_triton_test.py --batch-size 256 --seed 2026',
                modal_serverless_t4:
                  'modal run qh4-experimental/scripts/modal_cuda_triton_test.py --batch-size 256 --seed 2026',
                remote_worker_server:
                  'python3 qh4-experimental/scripts/gpu_worker_server.py --port 8787  # then set QH4_GPU_WORKER_URL',
              },
            },
            stderr: null,
          }),
          req
        );
      }

      const target = `${workerUrl.replace(/\/+$/, '')}/run-differential`;
      const workerRes = await fetch(target, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batchSize, seed, sm }),
        signal: AbortSignal.timeout(120000),
      });
      const workerJson = await workerRes.json();
      return withCors(
        NextResponse.json({
          ok: workerRes.ok && Boolean(workerJson.ok),
          mode,
          data: workerJson.report ?? workerJson,
          stderr: null,
        }),
        req
      );
    }

    let script = '';
    const extraEnv: Record<string, string> = {};

    if (mode === 'm01_m10') {
      script =
        'import json; from tests.test_math_m01_m10 import run_suite_report; print(json.dumps(run_suite_report()))';
    } else if (mode === 'configs') {
      const count = validateStrictInteger(payload.vectorsPerConfig ?? 1000, 'vectorsPerConfig');
      if (count < 10 || count > 10000) {
        throw new QH4ValidationError(
          'INVALID_VECTOR_COUNT',
          'vectorsPerConfig must be between 10 and 10,000.',
          'vectorsPerConfig'
        );
      }
      const seed = validateSeed(payload.seed ?? 202602);
      script = `import json; from tests.test_configs_10k import validate_all_eight_configs; print(json.dumps(validate_all_eight_configs(vectors_per_config=${count}, base_seed=${seed})))`;
    } else if (mode === 'benchmark') {
      const configName = validateConfigName(payload.config ?? '8B-D16');
      const batchSize = validateStrictInteger(payload.batchSize ?? 64, 'batchSize');
      if (batchSize < 1 || batchSize > 4096) {
        throw new QH4ValidationError('INVALID_BATCH_SIZE', 'batchSize must be between 1 and 4,096.', 'batchSize');
      }
      const workerCount = validateStrictInteger(payload.workerCount ?? 2, 'workerCount');
      if (workerCount < 1 || workerCount > 8) {
        throw new QH4ValidationError('INVALID_WORKER_COUNT', 'workerCount must be between 1 and 8.', 'workerCount');
      }
      const seed = validateSeed(payload.seed ?? 2026);
      script = `import json; from qh4_experimental.benchmark import run_structured_benchmark; print(json.dumps(run_structured_benchmark(config_name=${JSON.stringify(
        configName
      )}, batch_size=${batchSize}, worker_count=${workerCount}, seed=${seed}, backend_mode='cpu_reference')))`;
    } else if (mode === 'differential_test') {
      const configName = validateConfigName(payload.config ?? '8B-D16');
      const batchSize = validateStrictInteger(payload.batchSize ?? 128, 'batchSize');
      if (batchSize < 1 || batchSize > 4096) {
        throw new QH4ValidationError('INVALID_BATCH_SIZE', 'batchSize must be between 1 and 4,096.', 'batchSize');
      }
      const seed = validateSeed(payload.seed ?? 2026);
      script = `import json; from tests.test_gpu_cpu_differential import run_gpu_cpu_differential_test; print(json.dumps(run_gpu_cpu_differential_test(config_name=${JSON.stringify(
        configName
      )}, batch_size=${batchSize}, seed=${seed})))`;
    } else if (mode === 'cuda_triton_compile') {
      const sm = validateStrictInteger(payload.sm ?? 80, 'sm');
      if (sm < 50 || sm > 100) {
        throw new QH4ValidationError('INVALID_SM_ARCH', 'sm target must be between 50 and 100.', 'sm');
      }
      script = `import json; from scripts.run_actual_cuda_triton_test import run_suite; print(json.dumps(run_suite(batch_size=16, seed=2026, target_sm=${sm}, run_interpret=False)))`;
    } else if (mode === 'triton_interpret_test') {
      const configName = validateConfigName(payload.config ?? '8B-D16');
      const batchSize = validateStrictInteger(payload.batchSize ?? 16, 'batchSize');
      if (batchSize < 1 || batchSize > 256) {
        throw new QH4ValidationError('INVALID_BATCH_SIZE', 'batchSize must be between 1 and 256 for Triton interpreter.', 'batchSize');
      }
      const seed = validateSeed(payload.seed ?? 2026);
      extraEnv.TRITON_INTERPRET = '1';
      script = `import json; from qh4_experimental.configs import BLOCK_CONFIGS, derive_coefficients, generate_random_vectors; from qh4_experimental.triton_kernels import validate_triton_arithmetic_kernels, validate_triton_against_cpu; cfg=BLOCK_CONFIGS[${JSON.stringify(
        configName
      )}]; a,b=derive_coefficients(cfg.name, seed=${seed}); batch=generate_random_vectors(cfg.name, count=${batchSize}, seed=${seed}); arith=validate_triton_arithmetic_kernels(seed=${seed}, allow_interpret=True); xform=validate_triton_against_cpu(batch, a, b, rounds=cfg.default_rounds, allow_interpret=True); print(json.dumps({'status':'passed','config':cfg.name,'dim':cfg.dim,'rounds':cfg.default_rounds,'arithmetic':arith,'transform':xform,'note':'Executed real @triton.jit kernels via Triton official TRITON_INTERPRET=1 CPU interpreter; not physical CUDA hardware.'}))`;
    } else {
      script =
        'import json; from qh4_experimental.benchmark import collect_environment_metadata; print(json.dumps(collect_environment_metadata()))';
    }

    const pythonBin = resolvePythonBin();
    const { stdout, stderr } = await execFileAsync(pythonBin, ['-c', script], {
      cwd: QH4_ROOT,
      env: { ...process.env, ...extraEnv, PYTHONPATH: QH4_ROOT },
      timeout: 120000,
      maxBuffer: 8 * 1024 * 1024,
    });

    const parsed = JSON.parse(stdout.trim());
    return withCors(
      NextResponse.json({
        ok: true,
        mode,
        pythonBin,
        data: parsed,
        stderr: stderr.trim() || null,
      }),
      req
    );
  } catch (err) {
    if (err instanceof QH4ValidationError) {
      return withCors(
        NextResponse.json(
          { ok: false, error: { code: err.code, message: err.message, field: err.field } },
          { status: 400 }
        ),
        req
      );
    }
    return withCors(
      NextResponse.json(
        {
          ok: false,
          error: {
            code: 'PYTHON_EXECUTION_ERROR',
            message: err instanceof Error ? err.message : 'Python verification failed.',
          },
        },
        { status: 500 }
      ),
      req
    );
  }
}
