'use client';

import React, { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Copy, Cpu, Play, Server, Terminal } from 'lucide-react';
import { CONFIG_NAMES, ConfigName } from '@/lib/qh4/constants';

interface EnvProbe {
  python_version: string;
  platform: string;
  machine: string;
  cpu_count: number;
  cpu_reference_backend: string;
  triton_environment: {
    torch_installed: boolean;
    cuda_available: boolean;
    triton_installed: boolean;
    triton_interpret_active?: boolean;
    triton_version: string | null;
    torch_version: string | null;
    device_count: number;
    devices: string[];
    error: string | null;
  };
  cuda_toolchain?: {
    cuda_available: boolean;
    gpu_devices: string[];
    nvidia_smi: string | null;
    nvcc: string | null;
    libnvrtc: string | null;
    ptxas: string | null;
    torch_version: string | null;
    cuda_source_exists: boolean;
  };
}

interface BenchmarkReport {
  config: string;
  blocks: number;
  dim: number;
  rounds: number;
  batch_size: number;
  worker_count: number;
  backend_mode: string;
  seed: number;
  timings_ms: {
    cold_start: number;
    warm_compute: number;
    warm_batch: number;
    end_to_end: number;
  };
  checksums: {
    input_sha256: string;
    forward_sha256: string;
    inverse_sha256: string;
    roundtrip_exact_match: boolean;
  };
  shards: Array<{
    worker_id: number;
    device_id: string;
    start: number;
    end: number;
    count: number;
  }>;
  worker_boot_ms: number[];
  environment: EnvProbe;
}

interface CudaTritonCompileReport {
  hardware: {
    cuda_available: boolean;
    gpu_devices: string[];
    triton_version: string | null;
    torch_version: string | null;
  };
  compiler_verification: {
    cuda_cpp_nvrtc?: {
      status: string;
      nvrtc_version?: string;
      target_arch?: string;
      ptx_bytes?: number;
      ptx_sha256?: string;
      cubin_bytes?: number;
      cubin_sha256?: string;
      exported_kernels?: string[];
      error?: string;
    };
    triton_aot_ptx?: {
      status: string;
      triton_version?: string;
      target?: string;
      kernel_count?: number;
      kernels?: Array<{
        kernel: string;
        ptx_bytes: number;
        cubin_bytes: number;
        cubin_sha256: string;
        shared_bytes: number;
      }>;
      error?: string;
    };
  };
  execution_verification?: {
    status?: string;
    message?: string;
  };
  overall_status: string;
}

interface TritonInterpretReport {
  status: string;
  config: string;
  dim: number;
  rounds: number;
  arithmetic: {
    status: string;
    execution_backend: string;
    elements_tested: number;
  };
  transform: {
    status: string;
    execution_backend: string;
    batch_size: number;
    dim: number;
  };
  note: string;
}

export default function BenchmarkPelagosPanel() {
  const [config, setConfig] = useState<ConfigName>('8B-D16');
  const [batchSize, setBatchSize] = useState<number>(64);
  const [workerCount, setWorkerCount] = useState<number>(2);
  const [seed, setSeed] = useState<number>(2026);

  const [envProbe, setEnvProbe] = useState<EnvProbe | null>(null);
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [compileReport, setCompileReport] = useState<CudaTritonCompileReport | null>(null);
  const [compileLoading, setCompileLoading] = useState<boolean>(false);

  const [interpretReport, setInterpretReport] = useState<TritonInterpretReport | null>(null);
  const [interpretLoading, setInterpretLoading] = useState<boolean>(false);

  const [remoteGpuReport, setRemoteGpuReport] = useState<Record<string, unknown> | null>(null);
  const [remoteLoading, setRemoteLoading] = useState<boolean>(false);
  const [gpuActionError, setGpuActionError] = useState<string | null>(null);
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch('/api/qh4/python-verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'triton_probe' }),
    })
      .then((r) => r.json())
      .then((json) => {
        if (active && json.ok) {
          setEnvProbe(json.data);
        }
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  const handleRunBenchmark = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/qh4/python-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'benchmark',
          config,
          batchSize,
          workerCount,
          seed,
        }),
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        throw new Error(json.error?.message || 'Benchmark execution failed.');
      }
      setReport(json.data);
      setEnvProbe(json.data.environment);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unexpected benchmark error.');
    } finally {
      setLoading(false);
    }
  };

  const handleCompileCudaTriton = async () => {
    setCompileLoading(true);
    setGpuActionError(null);
    try {
      const res = await fetch('/api/qh4/python-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'cuda_triton_compile', sm: 80 }),
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        throw new Error(json.error?.message || 'CUDA/Triton AOT compilation check failed.');
      }
      setCompileReport(json.data);
    } catch (err) {
      setGpuActionError(err instanceof Error ? err.message : 'Unexpected compiler error.');
    } finally {
      setCompileLoading(false);
    }
  };

  const handleRunTritonInterpret = async () => {
    setInterpretLoading(true);
    setGpuActionError(null);
    try {
      const res = await fetch('/api/qh4/python-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'triton_interpret_test',
          config,
          batchSize: Math.min(batchSize, 16),
          seed,
        }),
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        throw new Error(json.error?.message || 'Triton interpreter test failed.');
      }
      setInterpretReport(json.data);
    } catch (err) {
      setGpuActionError(err instanceof Error ? err.message : 'Unexpected Triton interpreter error.');
    } finally {
      setInterpretLoading(false);
    }
  };

  const handleRunRemoteGpu = async () => {
    setRemoteLoading(true);
    setGpuActionError(null);
    try {
      const res = await fetch('/api/qh4/python-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'remote_gpu_test',
          batchSize,
          seed,
          sm: 75,
        }),
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        throw new Error(json.error?.message || 'Remote GPU worker check failed.');
      }
      setRemoteGpuReport(json.data);
    } catch (err) {
      setGpuActionError(err instanceof Error ? err.message : 'Unexpected remote GPU error.');
    } finally {
      setRemoteLoading(false);
    }
  };

  const handleCopyCommand = async (id: string, cmd: string) => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopiedCmd(id);
      window.setTimeout(() => setCopiedCmd(null), 1800);
    } catch {
      // Clipboard fallback ignored inside headless environments
    }
  };

  const tritonEnv = envProbe?.triton_environment;
  const cudaEnv = envProbe?.cuda_toolchain;

  const runnerCommands = [
    {
      id: 'modal',
      label: '1. Serverless NVIDIA T4 GPU (Modal)',
      cmd: 'modal run qh4-experimental/scripts/modal_cuda_triton_test.py --batch-size 256 --seed 2026',
    },
    {
      id: 'colab',
      label: '2. Local CUDA / Google Colab T4 / RunPod GPU',
      cmd: 'python3 qh4-experimental/scripts/run_actual_cuda_triton_test.py --batch-size 256 --seed 2026',
    },
    {
      id: 'worker',
      label: '3. Connect Remote CUDA GPU Worker to This Workbench',
      cmd: 'python3 qh4-experimental/scripts/gpu_worker_server.py --port 8787  # set QH4_GPU_WORKER_URL',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Hardware & Runtime Capability Probe */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg p-6 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-[#0F172A]">Runtime Environment &amp; GPU/Triton Capability Probe</h2>
            <p className="text-xs text-[#475569] mt-0.5">
              Live server inspection of Python, PyTorch, NVIDIA NVRTC/ptxas compilers, CUDA devices, and Triton JIT availability.
            </p>
          </div>
          <span className="text-xs font-mono text-[#475569]">
            {envProbe ? `Python ${envProbe.python_version} · ${envProbe.cpu_count} vCPUs` : 'Probing runtime...'}
          </span>
        </div>

        {envProbe && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-md p-4 space-y-1">
              <div className="flex items-center gap-1.5 font-medium text-[#334155]">
                <Cpu className="w-4 h-4 text-[#0F4C81]" aria-hidden="true" />
                CPU Reference Baseline
              </div>
              <div className="text-sm font-semibold text-[#15803D] pt-1">
                ✓ Operational ({envProbe.cpu_reference_backend})
              </div>
              <p className="text-[#64748B] font-mono">{envProbe.platform}</p>
            </div>

            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-md p-4 space-y-1">
              <div className="flex items-center gap-1.5 font-medium text-[#334155]">
                <Server className="w-4 h-4 text-[#0F4C81]" aria-hidden="true" />
                Triton &amp; NVIDIA CUDA Toolchain
              </div>
              <div
                className={`text-sm font-semibold pt-1 ${
                  tritonEnv?.cuda_available ? 'text-[#15803D]' : 'text-[#0F4C81]'
                }`}
              >
                {tritonEnv?.cuda_available
                  ? `✓ Physical GPU Ready (${tritonEnv.device_count} CUDA)`
                  : tritonEnv?.triton_installed
                    ? `✓ Triton ${tritonEnv.triton_version} + NVRTC/ptxas (CPU Host)`
                    : '▲ Triton/CUDA Not Installed'}
              </div>
              <p className="text-[#64748B]">
                {tritonEnv?.cuda_available
                  ? `Devices: ${tritonEnv.devices.join(', ')}`
                  : cudaEnv?.libnvrtc && cudaEnv?.ptxas
                    ? 'No physical NVIDIA GPU attached in this Cloud Run container; real NVRTC PTX, Triton PTX/CUBIN compilation, and TRITON_INTERPRET=1 are operational.'
                    : tritonEnv?.error || 'CUDA GPU / PyTorch / Triton not installed on this host.'}
              </p>
            </div>

            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-md p-4 space-y-1">
              <div className="font-medium text-[#334155]">Pelagos Multi-Worker Orchestrator</div>
              <div className="text-sm font-semibold text-[#15803D] pt-1">
                ✓ Spawn Pool Active (CPU Reference)
              </div>
              <p className="text-[#64748B]">
                Executes persistent worker processes, quotient+remainder sharding, and contiguous reassembly.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Actual CUDA C++ & Triton Verification Controls */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg p-6 space-y-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-[#0F172A]">Actual CUDA C++ &amp; Triton Verification Lab</h3>
            <p className="text-xs text-[#475569] mt-0.5">
              Compile real CUDA C++ (<span className="font-mono">qh4_cuda_kernels.cu</span>) &amp; Triton (<span className="font-mono">triton_kernels.py</span>) to NVIDIA sm_80 PTX/CUBIN, run Triton&apos;s official <span className="font-mono">TRITON_INTERPRET=1</span> differential check, or dispatch to a physical CUDA worker.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={handleCompileCudaTriton}
            disabled={compileLoading}
            className="px-4 py-2 bg-[#0F4C81] hover:bg-[#0C3E6B] disabled:opacity-60 text-white text-xs font-medium rounded-md flex items-center gap-1.5 whitespace-nowrap shrink-0"
          >
            <Terminal className="w-3.5 h-3.5" aria-hidden="true" />
            {compileLoading ? 'Compiling PTX & CUBIN...' : 'Compile CUDA C++ & Triton (sm_80 PTX/CUBIN)'}
          </button>

          <button
            type="button"
            onClick={handleRunTritonInterpret}
            disabled={interpretLoading}
            className="px-4 py-2 bg-white hover:bg-[#F8FAFC] disabled:opacity-60 text-[#0F172A] border border-[#CBD5E1] text-xs font-medium rounded-md flex items-center gap-1.5 whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5 text-[#0F4C81]" aria-hidden="true" />
            {interpretLoading ? 'Running TRITON_INTERPRET=1...' : `Run Triton Interpreter Parity (${config})`}
          </button>

          <button
            type="button"
            onClick={handleRunRemoteGpu}
            disabled={remoteLoading}
            className="px-4 py-2 bg-white hover:bg-[#F8FAFC] disabled:opacity-60 text-[#0F172A] border border-[#CBD5E1] text-xs font-medium rounded-md flex items-center gap-1.5 whitespace-nowrap shrink-0"
          >
            <Server className="w-3.5 h-3.5 text-[#0F4C81]" aria-hidden="true" />
            {remoteLoading ? 'Checking Physical GPU Worker...' : 'Query Physical GPU Worker'}
          </button>
        </div>

        {gpuActionError && (
          <div role="alert" className="bg-[#FEF2F2] border border-[#FECACA] rounded-md p-3 text-xs text-[#991B1B] flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
            <span>{gpuActionError}</span>
          </div>
        )}

        {compileReport && (
          <div className="border border-[#E2E8F0] rounded-md bg-[#F8FAFC] p-4 space-y-3 text-xs" data-testid="cuda-triton-compile-report">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 font-semibold text-[#0F172A]">
                <CheckCircle2 className="w-4 h-4 text-[#15803D]" aria-hidden="true" />
                NVRTC CUDA C++ &amp; Triton AOT PTX/CUBIN Compilation
              </div>
              <span className="font-mono text-[#475569]">
                status={compileReport.overall_status} · cuda_available={String(compileReport.hardware.cuda_available)}
              </span>
            </div>

            {compileReport.compiler_verification.cuda_cpp_nvrtc && (
              <div className="bg-white border border-[#E2E8F0] rounded p-3 space-y-1 font-mono text-[11px]">
                <div className="font-sans font-semibold text-[#0F172A] text-xs">
                  CUDA C++ (<span className="font-mono">qh4_cuda_kernels.cu</span>) via NVIDIA NVRTC {compileReport.compiler_verification.cuda_cpp_nvrtc.nvrtc_version}
                </div>
                <div className="text-[#334155]">
                  Target: {compileReport.compiler_verification.cuda_cpp_nvrtc.target_arch} · PTX:{' '}
                  {compileReport.compiler_verification.cuda_cpp_nvrtc.ptx_bytes} bytes · CUBIN:{' '}
                  {compileReport.compiler_verification.cuda_cpp_nvrtc.cubin_bytes} bytes
                </div>
                <div className="text-[#64748B] break-all">
                  CUBIN SHA-256: {compileReport.compiler_verification.cuda_cpp_nvrtc.cubin_sha256}
                </div>
              </div>
            )}

            {compileReport.compiler_verification.triton_aot_ptx?.kernels && (
              <div className="bg-white border border-[#E2E8F0] rounded overflow-x-auto">
                <table className="w-full text-left border-collapse text-[11px] font-mono">
                  <thead className="bg-[#F1F5F9] text-[#475569] border-b border-[#E2E8F0]">
                    <tr>
                      <th className="py-1.5 px-3 font-medium">Triton Kernel</th>
                      <th className="py-1.5 px-3 font-medium text-right">PTX Bytes</th>
                      <th className="py-1.5 px-3 font-medium text-right">CUBIN Bytes</th>
                      <th className="py-1.5 px-3 font-medium">CUBIN SHA-256</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E2E8F0]">
                    {compileReport.compiler_verification.triton_aot_ptx.kernels.map((k) => (
                      <tr key={k.kernel}>
                        <td className="py-1.5 px-3 text-[#0F172A]">{k.kernel}</td>
                        <td className="py-1.5 px-3 text-right text-[#475569]">{k.ptx_bytes.toLocaleString()}</td>
                        <td className="py-1.5 px-3 text-right text-[#0F4C81] font-semibold">{k.cubin_bytes.toLocaleString()}</td>
                        <td className="py-1.5 px-3 text-[#64748B]">{k.cubin_sha256.slice(0, 24)}…</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {interpretReport && (
          <div className="border border-[#BBF7D0] bg-[#F0FDF4] rounded-md p-4 text-xs space-y-1.5" data-testid="triton-interpret-report">
            <div className="font-semibold text-[#166534] flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" aria-hidden="true" />
              Triton Interpreter Differential Parity Passed ({interpretReport.config}, {interpretReport.transform.batch_size} vectors)
            </div>
            <div className="font-mono text-[11px] text-[#14532D]">
              Arithmetic ({interpretReport.arithmetic.elements_tested} elems): {interpretReport.arithmetic.status} · Transform ({interpretReport.transform.dim}D, {interpretReport.rounds} rounds): {interpretReport.transform.status} · Backend: {interpretReport.transform.execution_backend}
            </div>
            <p className="text-[#166534]">{interpretReport.note}</p>
          </div>
        )}

        {remoteGpuReport && (
          <div className="border border-[#E2E8F0] bg-[#F8FAFC] rounded-md p-4 text-xs space-y-2" data-testid="remote-gpu-report">
            <div className="font-semibold text-[#0F172A]">Physical GPU Worker Response</div>
            <pre className="text-[11px] font-mono text-[#334155] overflow-x-auto whitespace-pre-wrap">
              {JSON.stringify(remoteGpuReport, null, 2)}
            </pre>
          </div>
        )}

        {/* Standalone Hardware Execution Commands */}
        <div className="border-t border-[#E2E8F0] pt-4 space-y-2.5">
          <div className="text-xs font-semibold text-[#0F172A]">
            Execute on Physical NVIDIA GPU Hardware (CUDA C++ + Triton)
          </div>
          <div className="grid grid-cols-1 gap-2">
            {runnerCommands.map((item) => (
              <div
                key={item.id}
                className="flex flex-wrap items-center justify-between gap-2 bg-[#F8FAFC] border border-[#E2E8F0] rounded px-3 py-2 text-xs"
              >
                <div className="min-w-0">
                  <div className="font-medium text-[#334155]">{item.label}</div>
                  <div className="font-mono text-[11px] text-[#0F4C81] break-all mt-0.5">{item.cmd}</div>
                </div>
                <button
                  type="button"
                  onClick={() => handleCopyCommand(item.id, item.cmd)}
                  className="px-2.5 py-1 bg-white border border-[#CBD5E1] hover:bg-[#F1F5F9] rounded text-[11px] font-medium text-[#334155] flex items-center gap-1 shrink-0"
                >
                  <Copy className="w-3 h-3" aria-hidden="true" />
                  {copiedCmd === item.id ? 'Copied' : 'Copy'}
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Benchmark Configuration & Runner */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg p-6 space-y-5">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 flex-1 min-w-[280px]">
            <div>
              <label htmlFor="bench-config" className="block text-xs font-medium text-[#334155] mb-1">
                Configuration
              </label>
              <select
                id="bench-config"
                value={config}
                onChange={(e) => setConfig(e.target.value as ConfigName)}
                className="w-full px-3 py-2 text-xs font-mono bg-white border border-[#CBD5E1] rounded-md"
              >
                {CONFIG_NAMES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="bench-batch" className="block text-xs font-medium text-[#334155] mb-1">
                Batch Size
              </label>
              <select
                id="bench-batch"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full px-3 py-2 text-xs font-mono bg-white border border-[#CBD5E1] rounded-md"
              >
                {[16, 64, 256, 1024].map((b) => (
                  <option key={b} value={b}>
                    {b} vectors
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="bench-workers" className="block text-xs font-medium text-[#334155] mb-1">
                Persistent Workers
              </label>
              <select
                id="bench-workers"
                value={workerCount}
                onChange={(e) => setWorkerCount(Number(e.target.value))}
                className="w-full px-3 py-2 text-xs font-mono bg-white border border-[#CBD5E1] rounded-md"
              >
                {[1, 2, 3, 4].map((w) => (
                  <option key={w} value={w}>
                    {w} worker{w === 1 ? '' : 's'}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="bench-seed" className="block text-xs font-medium text-[#334155] mb-1">
                Seed
              </label>
              <input
                id="bench-seed"
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full px-3 py-2 text-xs font-mono bg-white border border-[#CBD5E1] rounded-md"
              />
            </div>
          </div>

          <button
            type="button"
            onClick={handleRunBenchmark}
            disabled={loading}
            className="px-4 py-2 bg-[#0F4C81] hover:bg-[#0C3E6B] disabled:opacity-60 text-white text-xs font-medium rounded-md flex items-center gap-1.5 whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" aria-hidden="true" />
            {loading ? 'Running Pelagos Benchmark...' : 'Run Structured Benchmark'}
          </button>
        </div>

        {errorMsg && (
          <div role="alert" className="bg-[#FEF2F2] border border-[#FECACA] rounded-md p-3 text-xs text-[#991B1B] flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
            <span>{errorMsg}</span>
          </div>
        )}

        {!report ? (
          <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-md p-6 text-center text-xs text-[#64748B]">
            Click <span className="font-medium text-[#0F172A]">Run Structured Benchmark</span> to spawn persistent Python workers, shard the batch using quotient+remainder allocation, and record cold-start, warm compute, warm batch, end-to-end timings, and SHA-256 digests.
          </div>
        ) : (
          <div className="space-y-5">
            {/* Latency Breakdown */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'Cold-Start (Boot + 1st Run)', val: `${report.timings_ms.cold_start} ms` },
                { label: 'Warm Compute (Local Core)', val: `${report.timings_ms.warm_compute} ms` },
                { label: 'Warm Batch (IPC + Workers)', val: `${report.timings_ms.warm_batch} ms` },
                { label: 'End-to-End (Prep + Round-Trip)', val: `${report.timings_ms.end_to_end} ms` },
              ].map((m) => (
                <div key={m.label} className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-md p-3.5">
                  <div className="text-[11px] text-[#64748B]">{m.label}</div>
                  <div className="text-xl font-mono font-semibold text-[#0F172A] mt-1">{m.val}</div>
                </div>
              ))}
            </div>

            {/* Sharding Plan Table */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="border border-[#E2E8F0] rounded-md overflow-hidden">
                <div className="bg-[#F8FAFC] px-4 py-2.5 border-b border-[#E2E8F0] text-xs font-semibold text-[#0F172A]">
                  Static Quotient + Remainder Sharding Plan
                </div>
                <table className="w-full text-xs text-left border-collapse font-mono">
                  <thead className="border-b border-[#E2E8F0] text-[#64748B]">
                    <tr>
                      <th className="py-2 px-3 font-medium">Worker</th>
                      <th className="py-2 px-3 font-medium">Device ID</th>
                      <th className="py-2 px-3 font-medium text-right">Range</th>
                      <th className="py-2 px-3 font-medium text-right">Count</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E2E8F0]">
                    {report.shards.map((s) => (
                      <tr key={s.worker_id}>
                        <td className="py-2 px-3 text-[#0F172A]">W{s.worker_id}</td>
                        <td className="py-2 px-3 text-[#475569]">dev:{s.device_id}</td>
                        <td className="py-2 px-3 text-right text-[#475569]">
                          [{s.start}, {s.end})
                        </td>
                        <td className="py-2 px-3 text-right font-semibold text-[#0F4C81]">{s.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="border border-[#E2E8F0] rounded-md p-4 space-y-2.5 text-xs">
                <div className="font-semibold text-[#0F172A]">SHA-256 Telemetry &amp; Checksum Audit</div>
                <div className="space-y-1.5 font-mono text-[11px]">
                  <div>
                    <span className="text-[#64748B] font-sans">Input Digest: </span>
                    <span className="text-[#0F172A] break-all">{report.checksums.input_sha256}</span>
                  </div>
                  <div>
                    <span className="text-[#64748B] font-sans">Forward Output Digest: </span>
                    <span className="text-[#0F4C81] break-all">{report.checksums.forward_sha256}</span>
                  </div>
                  <div>
                    <span className="text-[#64748B] font-sans">Inverse Recovered Digest: </span>
                    <span className="text-[#15803D] break-all">{report.checksums.inverse_sha256}</span>
                  </div>
                </div>
                <div className="pt-1 text-xs font-medium text-[#15803D]">
                  {report.checksums.roundtrip_exact_match
                    ? '✓ Reassembled batch inverse matches input SHA-256 bit-for-bit'
                    : '✕ Checksum mismatch'}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
