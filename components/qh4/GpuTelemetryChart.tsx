'use client';

// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  Play,
  RotateCcw,
  Zap,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  BLOCK_CONFIGS,
  CONFIG_NAMES,
  ConfigName,
  FIELD_MODULUS,
} from '@/lib/qh4/constants';

interface StageMetricPoint {
  stage: string;
  round: number;
  stageName: string;
  vramMb: number;
  vramDeltaKb: number;
  throughputKVecsPerSec: number;
  elementsProcessed: number;
  simulatedLatencyUs: number;
}

interface StreamingSample {
  timestamp: string;
  batchSize: number;
  allocatedVramMb: number;
  cacheOverheadMb: number;
  totalVramMb: number;
  vectorsPerSec: number;
  elementsPerSecM: number;
  memoryBandwidthGBs: number;
}

const subscribeNoop = () => () => {};

export default function GpuTelemetryChart() {
  const isMounted = React.useSyncExternalStore(
    subscribeNoop,
    () => true,
    () => false
  );
  const [selectedConfig, setSelectedConfig] = useState<ConfigName>('16B-D32');
  const [batchSize, setBatchSize] = useState<number>(1024);
  const [rounds, setRounds] = useState<number>(32);
  const [isLiveStreaming, setIsLiveStreaming] = useState<boolean>(true);

  const cfg = BLOCK_CONFIGS[selectedConfig];

  // Mathematical GPU memory calculations for F_65537:
  // Each element is a uint16 (2 bytes).
  // In Triton / CUDA GPU architectures:
  // - Input Tensor: batchSize * dim * 2 bytes
  // - Coefficients (alpha, beta): dim * 2 * 2 bytes = 4 * dim bytes (held in GPU L1/constant cache)
  // - Forward/Inverse mixing workspace: 2 ping-pong buffers of batchSize * dim * 2 bytes to avoid read-after-write hazards in SIMD warp sweeps
  // - Triton / PyTorch CUDA context static footprint: ~120 MB base allocation
  const memoryBreakdown = useMemo(() => {
    const elementBytes = 2; // uint16 fits [0, 65536]
    const inputBytes = batchSize * cfg.dim * elementBytes;
    const workspaceBytes = inputBytes * 2; // Ping-pong sweep buffers
    const coeffBytes = cfg.dim * elementBytes * 2; // Alpha + Beta vectors
    const dynamicBytes = inputBytes + workspaceBytes + coeffBytes;
    const dynamicMb = Number((dynamicBytes / (1024 * 1024)).toFixed(3));
    const staticBaseMb = 128.0; // Simulated Triton runtime context base
    const totalPeakMb = Number((staticBaseMb + dynamicMb).toFixed(3));

    // Simulated throughput based on GPU hardware model (e.g., modern Tensor Core / CUDA FP32/INT32 execution model):
    // Scalar layer requires modPow(x, 5) -> 3 modular multiplications + 1 affine MAC
    // Mixing layer requires 2 sequential sweeps across lanes with carry chains
    // Vector operations = batchSize * (cfg.dim * 4 + rounds * cfg.dim * 2)
    const opsPerVector = cfg.dim * 5 + rounds * (cfg.dim * 2);
    const simulatedGpuGflops = 14200.0; // 14.2 TFLOPS effective integer modular throughput
    const estimatedLatencyMs = Number(
      Math.max(0.04, ((batchSize * opsPerVector) / (simulatedGpuGflops * 1e6)) * 1000).toFixed(3)
    );
    const throughputVecsSec = Math.round(batchSize / (estimatedLatencyMs / 1000.0));
    const throughputElementsSecM = Number(((throughputVecsSec * cfg.dim) / 1e6).toFixed(2));
    const memoryBandwidthGBs = Number(
      (((batchSize * cfg.dim * elementBytes * (rounds * 4 + 2)) / (estimatedLatencyMs / 1000.0)) / 1e9).toFixed(2)
    );

    return {
      elementBytes,
      inputBytes,
      workspaceBytes,
      coeffBytes,
      dynamicMb,
      staticBaseMb,
      totalPeakMb,
      opsPerVector,
      estimatedLatencyMs,
      throughputVecsSec,
      throughputElementsSecM,
      memoryBandwidthGBs,
    };
  }, [batchSize, cfg.dim, rounds]);

  const [history, setHistory] = useState<StreamingSample[]>(() => {
    const samples: StreamingSample[] = [];
    const baseTime = Date.now();
    for (let i = 5; i >= 0; i--) {
      const t = (baseTime - i * 1000) / 1000;
      const timeStr = new Date(baseTime - i * 1000).toTimeString().split(' ')[0];
      const varianceFactor = 1.0 + (Math.sin(t * 0.7) * 0.035 + Math.cos(t * 1.3) * 0.02);
      samples.push({
        timestamp: timeStr,
        batchSize: 1024,
        allocatedVramMb: 0.188,
        cacheOverheadMb: 128.0,
        totalVramMb: Number((128.188 + Math.sin(t) * 0.08).toFixed(3)),
        vectorsPerSec: Math.round(6440252 * varianceFactor),
        elementsPerSecM: Number((206.09 * varianceFactor).toFixed(2)),
        memoryBandwidthGBs: Number((53.58 * varianceFactor).toFixed(2)),
      });
    }
    return samples;
  });

  // Stage-by-stage memory and throughput distribution
  const stageData: StageMetricPoint[] = useMemo(() => {
    const points: StageMetricPoint[] = [];
    const baseMb = 128.0;
    const batchBytes = batchSize * cfg.dim * 2;
    const batchMb = batchBytes / (1024 * 1024);

    // Initial Host-to-Device Allocation
    points.push({
      stage: 'H2D Alloc',
      round: 0,
      stageName: 'Vector H2D Buffer Copy',
      vramMb: Number((baseMb + batchMb).toFixed(3)),
      vramDeltaKb: Number((batchBytes / 1024).toFixed(1)),
      throughputKVecsPerSec: Number((memoryBreakdown.throughputVecsSec / 1000 * 1.4).toFixed(1)),
      elementsProcessed: batchSize * cfg.dim,
      simulatedLatencyUs: 18,
    });

    // Scalar S Layer
    points.push({
      stage: 'Scalar S',
      round: 0,
      stageName: 'Scalar S_i(x) = (α·x⁵ + β) mod q',
      vramMb: Number((baseMb + batchMb * 1.5).toFixed(3)),
      vramDeltaKb: Number(((batchBytes * 0.5) / 1024).toFixed(1)),
      throughputKVecsPerSec: Number((memoryBreakdown.throughputVecsSec / 1000 * 0.9).toFixed(1)),
      elementsProcessed: batchSize * cfg.dim,
      simulatedLatencyUs: Math.round(memoryBreakdown.estimatedLatencyMs * 250),
    });

    // Sampling representative mixing rounds
    const sampleRounds = Math.min(rounds, 8);
    for (let r = 1; r <= sampleRounds; r++) {
      const isLast = r === sampleRounds;
      points.push({
        stage: `R${r} Swp1`,
        round: r,
        stageName: `Round ${r} Sweep 1 (Forward Neighbor)`,
        vramMb: Number((baseMb + batchMb * 2.0).toFixed(3)),
        vramDeltaKb: Number((batchBytes / 1024).toFixed(1)),
        throughputKVecsPerSec: Number((memoryBreakdown.throughputVecsSec / 1000 * 1.15).toFixed(1)),
        elementsProcessed: batchSize * (cfg.dim - 1),
        simulatedLatencyUs: Math.round((memoryBreakdown.estimatedLatencyMs * 750) / rounds),
      });

      points.push({
        stage: `R${r} Swp2`,
        round: r,
        stageName: `Round ${r} Sweep 2 (Backward Neighbor)`,
        vramMb: Number((baseMb + batchMb * (isLast ? 1.0 : 2.0)).toFixed(3)),
        vramDeltaKb: 0,
        throughputKVecsPerSec: Number((memoryBreakdown.throughputVecsSec / 1000 * 1.12).toFixed(1)),
        elementsProcessed: batchSize * (cfg.dim - 1),
        simulatedLatencyUs: Math.round((memoryBreakdown.estimatedLatencyMs * 750) / rounds),
      });
    }

    // Output Device-to-Host Retrieval
    points.push({
      stage: 'D2H Retr',
      round: rounds,
      stageName: 'Device-to-Host Output Stream',
      vramMb: Number((baseMb + batchMb).toFixed(3)),
      vramDeltaKb: Number((-batchBytes / 1024).toFixed(1)),
      throughputKVecsPerSec: Number((memoryBreakdown.throughputVecsSec / 1000 * 1.6).toFixed(1)),
      elementsProcessed: batchSize * cfg.dim,
      simulatedLatencyUs: 22,
    });

    return points;
  }, [batchSize, cfg.dim, memoryBreakdown, rounds]);

  // Live streaming simulator ticker
  // Live streaming simulator ticker updating history window
  useEffect(() => {
    if (!isLiveStreaming) return;
    const interval = window.setInterval(() => {
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];
      const t = Date.now() / 1000;
      const varianceFactor = 1.0 + (Math.sin(t * 0.7) * 0.035 + Math.cos(t * 1.3) * 0.02);
      const currentThroughput = Math.round(memoryBreakdown.throughputVecsSec * varianceFactor);
      const currentElementsM = Number((memoryBreakdown.throughputElementsSecM * varianceFactor).toFixed(2));
      const currentVram = Number((memoryBreakdown.totalPeakMb + Math.sin(t) * 0.08).toFixed(3));
      const currentBw = Number((memoryBreakdown.memoryBandwidthGBs * varianceFactor).toFixed(2));

      const sample: StreamingSample = {
        timestamp: timeStr,
        batchSize,
        allocatedVramMb: Number(memoryBreakdown.dynamicMb.toFixed(3)),
        cacheOverheadMb: memoryBreakdown.staticBaseMb,
        totalVramMb: currentVram,
        vectorsPerSec: currentThroughput,
        elementsPerSecM: currentElementsM,
        memoryBandwidthGBs: currentBw,
      };

      setHistory((prev) => [...prev, sample].slice(-20));
    }, 1000);
    return () => window.clearInterval(interval);
  }, [isLiveStreaming, memoryBreakdown, batchSize]);

  return (
    <div className="space-y-6">
      {/* Parameter Control Header */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-[#0F172A] flex items-center gap-2">
              <Activity className="w-5 h-5 text-[#0F4C81]" aria-hidden="true" />
              Simulated GPU Memory Allocation &amp; Real-Time Throughput
            </h2>
            <p className="text-xs text-[#475569] mt-0.5">
              Theoretical Triton CUDA memory layout over F<sub>{FIELD_MODULUS.toLocaleString()}</sub> with real-time streaming bandwidth, latency, and stage VRAM profiles rendered via Recharts.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIsLiveStreaming((v) => !v)}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                isLiveStreaming
                  ? 'bg-[#15803D] hover:bg-[#166534] text-white'
                  : 'bg-[#0F4C81] hover:bg-[#0C3E6B] text-white'
              }`}
            >
              <Play className="w-3.5 h-3.5" aria-hidden="true" />
              {isLiveStreaming ? 'Live Streaming Active' : 'Resume Live Stream'}
            </button>

            <button
              type="button"
              onClick={() => setHistory([])}
              className="px-3 py-1.5 border border-[#CBD5E1] bg-white hover:bg-[#F8FAFC] text-[#334155] rounded-md text-xs font-medium flex items-center gap-1 shrink-0"
              title="Reset streaming history"
            >
              <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
              Clear
            </button>
          </div>
        </div>

        {/* Configuration, Batch, and Rounds Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 border-t border-[#E2E8F0] pt-4">
          <div>
            <label htmlFor="sim-cfg" className="block text-xs font-medium text-[#334155] mb-1">
              Configuration ({cfg.blocks}B · {cfg.dim}D)
            </label>
            <select
              id="sim-cfg"
              value={selectedConfig}
              onChange={(e) => {
                const nextCfg = e.target.value as ConfigName;
                setSelectedConfig(nextCfg);
                setRounds(BLOCK_CONFIGS[nextCfg].defaultRounds);
              }}
              className="w-full px-3 py-1.5 text-xs font-mono bg-white border border-[#CBD5E1] rounded-md text-[#0F172A]"
            >
              {CONFIG_NAMES.map((name) => (
                <option key={name} value={name}>
                  {name} ({BLOCK_CONFIGS[name].blocks}B · {BLOCK_CONFIGS[name].dim}D)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="sim-batch" className="block text-xs font-medium text-[#334155] mb-1">
              Batch Size (Vectors)
            </label>
            <select
              id="sim-batch"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              className="w-full px-3 py-1.5 text-xs font-mono bg-white border border-[#CBD5E1] rounded-md text-[#0F172A]"
            >
              {[64, 256, 512, 1024, 2048, 4096, 8192].map((b) => (
                <option key={b} value={b}>
                  {b.toLocaleString()} vectors ({((b * cfg.dim * 2) / 1024).toFixed(1)} KB)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="sim-rounds" className="block text-xs font-medium text-[#334155] mb-1">
              Mixing Rounds (n = {rounds})
            </label>
            <input
              id="sim-rounds"
              type="range"
              min={1}
              max={64}
              value={rounds}
              onChange={(e) => setRounds(Number(e.target.value))}
              className="w-full accent-[#0F4C81] cursor-pointer"
            />
          </div>

          <div>
            <span className="block text-xs font-medium text-[#334155] mb-1">Theoretical Kernel Latency</span>
            <div className="text-sm font-mono font-semibold text-[#0F172A]">
              {memoryBreakdown.estimatedLatencyMs} ms
              <span className="text-xs font-sans font-normal text-[#64748B] ml-1.5">
                ({memoryBreakdown.opsPerVector.toLocaleString()} ops/vec)
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Real-Time Telemetry KPI Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white border border-[#E2E8F0] rounded-lg p-4 space-y-1">
          <div className="text-xs text-[#64748B] flex items-center gap-1.5">
            <Gauge className="w-3.5 h-3.5 text-[#0F4C81]" aria-hidden="true" />
            Throughput (Vectors/s)
          </div>
          <div className="text-2xl font-mono font-bold text-[#0F172A]">
            {memoryBreakdown.throughputVecsSec.toLocaleString()}
          </div>
          <div className="text-[11px] text-[#15803D] font-medium">
            ~{memoryBreakdown.throughputElementsSecM}M finite-field elements/sec
          </div>
        </div>

        <div className="bg-white border border-[#E2E8F0] rounded-lg p-4 space-y-1">
          <div className="text-xs text-[#64748B] flex items-center gap-1.5">
            <HardDrive className="w-3.5 h-3.5 text-[#0F4C81]" aria-hidden="true" />
            Simulated Peak VRAM
          </div>
          <div className="text-2xl font-mono font-bold text-[#0F172A]">
            {memoryBreakdown.totalPeakMb} <span className="text-sm font-normal text-[#64748B]">MB</span>
          </div>
          <div className="text-[11px] text-[#475569]">
            {memoryBreakdown.dynamicMb} MB dynamic + {memoryBreakdown.staticBaseMb} MB runtime
          </div>
        </div>

        <div className="bg-white border border-[#E2E8F0] rounded-lg p-4 space-y-1">
          <div className="text-xs text-[#64748B] flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-[#0F4C81]" aria-hidden="true" />
            Effective Memory Bandwidth
          </div>
          <div className="text-2xl font-mono font-bold text-[#0F172A]">
            {memoryBreakdown.memoryBandwidthGBs} <span className="text-sm font-normal text-[#64748B]">GB/s</span>
          </div>
          <div className="text-[11px] text-[#475569]">
            {rounds * 2} sweeps × 2 passes/round
          </div>
        </div>

        <div className="bg-white border border-[#E2E8F0] rounded-lg p-4 space-y-1">
          <div className="text-xs text-[#64748B] flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-[#0F4C81]" aria-hidden="true" />
            Active Vector Tensor
          </div>
          <div className="text-2xl font-mono font-bold text-[#0F172A]">
            {(memoryBreakdown.inputBytes / 1024).toFixed(1)} <span className="text-sm font-normal text-[#64748B]">KB</span>
          </div>
          <div className="text-[11px] text-[#475569]">
            {batchSize} × {cfg.dim} uint16 ({cfg.dim * 2} B/vec)
          </div>
        </div>
      </div>

      {/* Primary Recharts Visualization: Real-time Live Stream */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[#0F172A]">
              Live Processing Throughput Stream (Vectors/sec &amp; Bandwidth)
            </h3>
            <p className="text-xs text-[#64748B]">
              Real-time time-series window showing warp scheduling variance and effective memory bandwidth under active batch transformation.
            </p>
          </div>
          <span className="text-xs font-mono bg-[#F1F5F9] px-2.5 py-1 rounded text-[#334155]">
            Window: last {history.length} samples
          </span>
        </div>

        <div className="h-[270px] w-full pt-2">
          {isMounted && history.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="timestamp" stroke="#64748B" fontSize={11} tickMargin={6} />
                <YAxis
                  yAxisId="left"
                  stroke="#0F4C81"
                  fontSize={11}
                  tickFormatter={(v: number) => `${Math.round(v / 1000)}k`}
                  domain={['auto', 'auto']}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#059669"
                  fontSize={11}
                  tickFormatter={(v: number) => `${v} GB/s`}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '6px',
                    fontSize: '11px',
                    color: '#E2E8F0',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="vectorsPerSec"
                  name="Throughput (Vectors/s)"
                  stroke="#0F4C81"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  activeDot={{ r: 4 }}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="memoryBandwidthGBs"
                  name="Bandwidth (GB/s)"
                  stroke="#059669"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-[#64748B]">
              Initializing real-time telemetry stream...
            </div>
          )}
        </div>
      </div>

      {/* Secondary Recharts Visualization: Stage-by-Stage VRAM Profile */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[#0F172A]">
              Stage-by-Stage VRAM Footprint &amp; Ping-Pong Sweep Buffer Allocation
            </h3>
            <p className="text-xs text-[#64748B]">
              GPU memory profile across forward stages: H2D tensor staging → Scalar power layer → 2-sweep mixing rounds → D2H output retrieval.
            </p>
          </div>
          <span className="text-xs font-mono text-[#0F4C81]">
            Peak Dynamic VRAM: +{memoryBreakdown.dynamicMb} MB
          </span>
        </div>

        <div className="h-[270px] w-full pt-2">
          {isMounted ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={stageData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                <defs>
                  <linearGradient id="vramGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0F4C81" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#0F4C81" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="stage" stroke="#64748B" fontSize={11} tickMargin={6} />
                <YAxis
                  stroke="#0F4C81"
                  fontSize={11}
                  domain={[125, 'auto']}
                  tickFormatter={(v: number) => `${v} MB`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '6px',
                    fontSize: '11px',
                    color: '#E2E8F0',
                  }}
                  formatter={(value) => [`${Number(value ?? 0)} MB`, 'Total VRAM Allocated']}
                  labelFormatter={(label, payload) => {
                    const item = payload?.[0]?.payload as StageMetricPoint | undefined;
                    return item?.stageName || String(label);
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Area
                  type="monotone"
                  dataKey="vramMb"
                  name="Allocated VRAM (MB)"
                  stroke="#0F4C81"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#vramGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-[#64748B]">
              Loading stage memory model...
            </div>
          )}
        </div>
      </div>

      {/* Kernel Specification & Differential Test Note */}
      <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-5 space-y-2 text-xs text-[#475569]">
        <div className="font-semibold text-[#0F172A] flex items-center gap-1.5">
          <Cpu className="w-4 h-4 text-[#0F4C81]" aria-hidden="true" />
          Differential Test &amp; Architecture Note
        </div>
        <p className="leading-relaxed">
          The memory and bandwidth curves above model the independent Triton kernel pipeline in{' '}
          <code className="text-[#0F172A] font-semibold">qh4_experimental/triton_kernels.py</code>. Because this server container environment operates without physical NVIDIA CUDA accelerators, execution metrics are derived mathematically from the exact F<sub>65,537</sub> tensor sizes and sequential round mixing constraints. On GPU targets, the differential test compares the output vectors of{' '}
          <code className="text-[#0F172A]">qh4_transform_forward_kernel</code> directly against the pure Python CPU baseline{' '}
          <code className="text-[#0F172A]">cpu_reference.py</code> to ensure bit-identical arithmetic equivalence.
        </p>
      </div>
    </div>
  );
}
