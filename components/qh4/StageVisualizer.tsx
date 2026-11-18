'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
} from 'lucide-react';
import {
  BLOCK_CONFIGS,
  CONFIG_NAMES,
  ConfigName,
  FIELD_MODULUS,
  VECTOR_PRESETS,
  VectorPresetId,
} from '@/lib/qh4/constants';
import {
  buildPresetVector,
  deriveCoefficients,
  fnvChecksumHex,
  transformWithTrace,
  TransformTraceResult,
} from '@/lib/qh4/math';
import {
  parseVectorText,
  QH4ValidationError,
  validateCoefficients,
  validateRounds,
  validateSeed,
} from '@/lib/qh4/validation';

interface ValidationAlert {
  code: string;
  message: string;
  field?: string;
}

export default function StageVisualizer() {
  const [configName, setConfigName] = useState<ConfigName>('8B-D16');
  const [seedText, setSeedText] = useState<string>('42');
  const [roundsText, setRoundsText] = useState<string>('16');
  const [preset, setPreset] = useState<VectorPresetId>('single_spike');
  const [vectorText, setVectorText] = useState<string>(() =>
    buildPresetVector('8B-D16', 'single_spike', 42).join(', ')
  );

  // Optional custom alpha/beta override
  const [showCoeffEditor, setShowCoeffEditor] = useState<boolean>(false);
  const [useCustomCoeffs, setUseCustomCoeffs] = useState<boolean>(false);
  const [alphaText, setAlphaText] = useState<string>(() => deriveCoefficients('8B-D16', 42).alphas.join(', '));
  const [betaText, setBetaText] = useState<string>(() => deriveCoefficients('8B-D16', 42).betas.join(', '));

  const [trace, setTrace] = useState<TransformTraceResult>(() => {
    const { alphas, betas } = deriveCoefficients('8B-D16', 42);
    const vec = buildPresetVector('8B-D16', 'single_spike', 42);
    return transformWithTrace('8B-D16', vec, alphas, betas, 16);
  });

  const [validationError, setValidationError] = useState<ValidationAlert | null>(null);
  const [stageIdx, setStageIdx] = useState<number>(1);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [inspectedLane, setInspectedLane] = useState<number>(0);
  const [viewFilter, setViewFilter] = useState<'all' | 'changed'>('all');

  const cfg = BLOCK_CONFIGS[configName];

  const runTraceExecution = (
    targetConfig: ConfigName,
    targetSeedStr: string,
    targetRoundsStr: string,
    targetVecStr: string,
    customCoeffsEnabled: boolean,
    targetAlphaStr: string,
    targetBetaStr: string
  ) => {
    try {
      if (!/^-?\d+$/.test(targetSeedStr.trim())) {
        throw new QH4ValidationError('INVALID_SEED', 'Seed must be a base-10 integer in [0, 2,147,483,647].', 'seed');
      }
      const seedNum = validateSeed(Number(targetSeedStr.trim()));

      if (!/^-?\d+$/.test(targetRoundsStr.trim())) {
        throw new QH4ValidationError('INVALID_ROUND_COUNT', 'Mixing rounds must be an integer in [1, 256].', 'rounds');
      }
      const roundsNum = validateRounds(Number(targetRoundsStr.trim()));

      const targetCfg = BLOCK_CONFIGS[targetConfig];
      const vec = parseVectorText(targetVecStr, targetCfg.dim, 'vectorText');

      let alphas: number[];
      let betas: number[];
      if (customCoeffsEnabled) {
        const parsedA = parseVectorText(targetAlphaStr, targetCfg.dim, 'alphas');
        const parsedB = parseVectorText(targetBetaStr, targetCfg.dim, 'betas');
        const checked = validateCoefficients(parsedA, parsedB, targetCfg.dim);
        alphas = checked.alphas;
        betas = checked.betas;
      } else {
        const derived = deriveCoefficients(targetConfig, seedNum);
        alphas = derived.alphas;
        betas = derived.betas;
      }

      const nextTrace = transformWithTrace(targetConfig, vec, alphas, betas, roundsNum);
      setTrace(nextTrace);
      setValidationError(null);
      setStageIdx((prev) => Math.min(prev, nextTrace.stages.length - 1));
      setInspectedLane((prev) => Math.min(prev, nextTrace.dim - 1));
    } catch (err) {
      setIsPlaying(false);
      if (err instanceof QH4ValidationError) {
        setValidationError({ code: err.code, message: err.message, field: err.field });
      } else {
        setValidationError({
          code: 'EXECUTION_ERROR',
          message: err instanceof Error ? err.message : 'Unexpected transformation error.',
        });
      }
    }
  };

  const handleSelectConfig = (nextConfig: ConfigName) => {
    setIsPlaying(false);
    setConfigName(nextConfig);
    const nextCfg = BLOCK_CONFIGS[nextConfig];
    const safeSeed = /^\d+$/.test(seedText.trim()) ? Math.min(Number(seedText.trim()), 2147483647) : 42;
    const nextRoundsStr = String(nextCfg.defaultRounds);
    const nextVec = buildPresetVector(nextConfig, preset, safeSeed);
    const nextVecStr = nextVec.join(', ');
    const { alphas, betas } = deriveCoefficients(nextConfig, safeSeed);
    const nextAlphaStr = alphas.join(', ');
    const nextBetaStr = betas.join(', ');

    setRoundsText(nextRoundsStr);
    setVectorText(nextVecStr);
    setAlphaText(nextAlphaStr);
    setBetaText(nextBetaStr);
    setUseCustomCoeffs(false);
    setStageIdx(1);
    setInspectedLane(0);

    runTraceExecution(nextConfig, String(safeSeed), nextRoundsStr, nextVecStr, false, nextAlphaStr, nextBetaStr);
  };

  const handleApplyPreset = (nextPreset: VectorPresetId) => {
    setIsPlaying(false);
    setPreset(nextPreset);
    const safeSeed = /^\d+$/.test(seedText.trim()) ? Math.min(Number(seedText.trim()), 2147483647) : 42;
    const nextVecStr = buildPresetVector(configName, nextPreset, safeSeed).join(', ');
    setVectorText(nextVecStr);
    runTraceExecution(configName, seedText, roundsText, nextVecStr, useCustomCoeffs, alphaText, betaText);
  };

  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      setStageIdx((prev) => {
        if (prev >= trace.stages.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 320);
    return () => window.clearInterval(timer);
  }, [isPlaying, trace.stages.length]);

  const activeStage = trace.stages[Math.min(stageIdx, trace.stages.length - 1)];
  const prevStage =
    activeStage.kind === 'output'
      ? trace.stages[0]
      : stageIdx > 0
        ? trace.stages[stageIdx - 1]
        : activeStage;

  const changedSet = useMemo(() => new Set(activeStage.changedIndices), [activeStage]);

  const stageJumpIndices = useMemo(() => {
    const scalarIdx = trace.stages.findIndex((s) => s.kind === 'scalar' && s.direction === 'forward');
    const outputIdx = trace.stages.findIndex((s) => s.kind === 'output');
    const lastIdx = trace.stages.length - 1;
    return { inputIdx: 0, scalarIdx, outputIdx, lastIdx };
  }, [trace.stages]);

  const laneRows = useMemo(() => {
    const rows = activeStage.values.map((val, idx) => {
      const prevVal = prevStage.values[idx] ?? val;
      const delta = activeStage.deltasFromPrev[idx] ?? 0;
      const blockIdx = Math.floor(idx / trace.blockWidth);
      const changed = changedSet.has(idx);
      return {
        idx,
        blockIdx,
        alpha: trace.alphas[idx],
        beta: trace.betas[idx],
        prevVal,
        val,
        delta,
        changed,
      };
    });
    if (viewFilter === 'changed') {
      return rows.filter((r) => r.changed);
    }
    return rows;
  }, [activeStage, prevStage, trace.blockWidth, trace.alphas, trace.betas, changedSet, viewFilter]);

  const selectedLaneData = {
    idx: inspectedLane,
    blockIdx: Math.floor(inspectedLane / trace.blockWidth),
    val: activeStage.values[inspectedLane] ?? 0,
    prevVal: prevStage.values[inspectedLane] ?? 0,
    delta: activeStage.deltasFromPrev[inspectedLane] ?? 0,
    alpha: trace.alphas[inspectedLane] ?? 1,
    beta: trace.betas[inspectedLane] ?? 0,
    changed: changedSet.has(inspectedLane),
  };

  // SVG layout dimensions
  const svgWidth = 860;
  const svgHeight = 290;
  const padLeft = 56;
  const padRight = 20;
  const padTop = 24;
  const padBottom = 42;
  const plotW = svgWidth - padLeft - padRight;
  const plotH = svgHeight - padTop - padBottom;
  const slotW = plotW / trace.dim;
  const barW = Math.max(8, Math.min(28, slotW * 0.64));

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start">
      {/* Left Parameter & Input Validation Column */}
      <aside className="w-full lg:w-[360px] shrink-0 bg-white border border-[#E2E8F0] rounded-lg p-5 space-y-5">
        <div>
          <h2 className="text-base font-semibold text-[#0F172A]">Configuration & Vector Input</h2>
          <p className="text-xs text-[#475569] mt-1">
            Field F<sub>65,537</sub> · Exponent d = 5 · Inverse e = 52,429
          </p>
        </div>

        {/* 8 Block Configurations */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-[#334155]">Block Configuration</label>
            <span className="text-xs font-mono text-[#475569]">
              {cfg.blocks}B · {cfg.dim}D · w={cfg.blockWidth}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-1.5" role="group" aria-label="Block configuration selector">
            {CONFIG_NAMES.map((name) => {
              const active = name === configName;
              const c = BLOCK_CONFIGS[name];
              return (
                <button
                  key={name}
                  type="button"
                  onClick={() => handleSelectConfig(name)}
                  className={`px-3 py-2 text-left rounded-md border text-xs transition-colors whitespace-nowrap shrink-0 ${
                    active
                      ? 'bg-[#0F4C81] text-white border-[#0F4C81] font-medium'
                      : 'bg-[#F8FAFC] text-[#1E293B] border-[#E2E8F0] hover:bg-[#F1F5F9]'
                  }`}
                >
                  <span className="font-mono font-semibold">{name}</span>
                  <span className={`ml-1.5 ${active ? 'text-white/80' : 'text-[#64748B]'}`}>
                    ({c.blocks}×{c.blockWidth})
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Seed & Mixing Rounds */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label htmlFor="qh4-seed-input" className="block text-xs font-medium text-[#334155] mb-1">
              Coefficient Seed
            </label>
            <input
              id="qh4-seed-input"
              type="text"
              inputMode="numeric"
              value={seedText}
              onChange={(e) => setSeedText(e.target.value)}
              className="w-full px-3 py-2 text-sm font-mono bg-white border border-[#CBD5E1] rounded-md text-[#0F172A] focus:outline-none focus:ring-2 focus:ring-[#0F4C81]/30 focus:border-[#0F4C81]"
            />
          </div>
          <div>
            <label htmlFor="qh4-rounds-input" className="block text-xs font-medium text-[#334155] mb-1">
              Mixing Rounds (n)
            </label>
            <input
              id="qh4-rounds-input"
              type="text"
              inputMode="numeric"
              value={roundsText}
              onChange={(e) => setRoundsText(e.target.value)}
              className="w-full px-3 py-2 text-sm font-mono bg-white border border-[#CBD5E1] rounded-md text-[#0F172A] focus:outline-none focus:ring-2 focus:ring-[#0F4C81]/30 focus:border-[#0F4C81]"
            />
          </div>
        </div>

        {/* Presets */}
        <div className="space-y-1.5">
          <label htmlFor="qh4-preset-select" className="block text-xs font-medium text-[#334155]">
            Vector Generator Preset
          </label>
          <select
            id="qh4-preset-select"
            value={preset}
            onChange={(e) => handleApplyPreset(e.target.value as VectorPresetId)}
            className="w-full px-3 py-2 text-sm bg-white border border-[#CBD5E1] rounded-md text-[#0F172A] focus:outline-none focus:ring-2 focus:ring-[#0F4C81]/30 focus:border-[#0F4C81]"
          >
            {VECTOR_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-[#64748B]">
            {VECTOR_PRESETS.find((p) => p.id === preset)?.description}
          </p>
        </div>

        {/* Input Vector Textarea */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label htmlFor="qh4-vector-input" className="text-xs font-medium text-[#334155]">
              Input Vector x ({cfg.dim} elements in 0..65536)
            </label>
          </div>
          <textarea
            id="qh4-vector-input"
            rows={4}
            value={vectorText}
            onChange={(e) => setVectorText(e.target.value)}
            placeholder={`Enter ${cfg.dim} integers in [0, 65536], comma or JSON separated`}
            className="w-full px-3 py-2 text-xs font-mono bg-white border border-[#CBD5E1] rounded-md text-[#0F172A] leading-relaxed focus:outline-none focus:ring-2 focus:ring-[#0F4C81]/30 focus:border-[#0F4C81]"
          />
          <p className="text-[11px] text-[#64748B]">
            Accepts comma-separated integers or JSON arrays. Out-of-range values (&lt;0 or &gt;65536), floats, and wrong lengths are rejected explicitly.
          </p>
        </div>

        {/* Optional Custom Alpha/Beta Overrides */}
        <div className="border-t border-[#E2E8F0] pt-3">
          <button
            type="button"
            onClick={() => setShowCoeffEditor((v) => !v)}
            aria-expanded={showCoeffEditor}
            className="flex items-center justify-between w-full text-xs font-medium text-[#334155] hover:text-[#0F172A] py-1"
          >
            <span className="flex items-center gap-1.5">
              <SlidersHorizontal className="w-3.5 h-3.5" aria-hidden="true" />
              Alpha / Beta Coefficient Overrides
            </span>
            <span className="text-[#64748B]">{showCoeffEditor ? 'Hide' : 'Inspect / Edit'}</span>
          </button>

          {showCoeffEditor && (
            <div className="mt-3 space-y-3 bg-[#F8FAFC] border border-[#E2E8F0] rounded-md p-3">
              <label className="flex items-center gap-2 text-xs text-[#1E293B] cursor-pointer">
                <input
                  type="checkbox"
                  checked={useCustomCoeffs}
                  onChange={(e) => setUseCustomCoeffs(e.target.checked)}
                  className="rounded border-[#CBD5E1] text-[#0F4C81]"
                />
                Use custom α / β vectors instead of seed-derived
              </label>

              <div>
                <label htmlFor="qh4-alpha-input" className="block text-[11px] font-medium text-[#475569] mb-1">
                  Alpha α ({cfg.dim} nonzero integers in 1..65536)
                </label>
                <textarea
                  id="qh4-alpha-input"
                  rows={2}
                  disabled={!useCustomCoeffs}
                  value={alphaText}
                  onChange={(e) => setAlphaText(e.target.value)}
                  className="w-full px-2.5 py-1.5 text-[11px] font-mono bg-white disabled:bg-[#F1F5F9] border border-[#CBD5E1] rounded text-[#0F172A]"
                />
              </div>

              <div>
                <label htmlFor="qh4-beta-input" className="block text-[11px] font-medium text-[#475569] mb-1">
                  Beta β ({cfg.dim} integers in 0..65536)
                </label>
                <textarea
                  id="qh4-beta-input"
                  rows={2}
                  disabled={!useCustomCoeffs}
                  value={betaText}
                  onChange={(e) => setBetaText(e.target.value)}
                  className="w-full px-2.5 py-1.5 text-[11px] font-mono bg-white disabled:bg-[#F1F5F9] border border-[#CBD5E1] rounded text-[#0F172A]"
                />
              </div>
            </div>
          )}
        </div>

        {/* Inline Validation Error Banner */}
        {validationError && (
          <div
            role="alert"
            data-testid="validation-error-banner"
            className="bg-[#FEF2F2] border border-[#FECACA] rounded-md p-3 text-xs text-[#991B1B] space-y-1"
          >
            <div className="flex items-center gap-1.5 font-semibold">
              <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
              <span className="font-mono">{validationError.code}</span>
            </div>
            <p className="leading-relaxed">{validationError.message}</p>
          </div>
        )}

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() =>
              runTraceExecution(configName, seedText, roundsText, vectorText, useCustomCoeffs, alphaText, betaText)
            }
            className="flex-1 px-4 py-2.5 bg-[#0F4C81] hover:bg-[#0C3E6B] text-white text-sm font-medium rounded-md transition-colors whitespace-nowrap shrink-0"
          >
            Apply &amp; Trace Vector
          </button>
          <button
            type="button"
            aria-label="Reset vector and stage to preset"
            onClick={() => handleApplyPreset(preset)}
            className="px-3 py-2.5 bg-[#F1F5F9] hover:bg-[#E2E8F0] text-[#334155] border border-[#CBD5E1] rounded-md transition-colors shrink-0"
          >
            <RotateCcw className="w-4 h-4" aria-hidden="true" />
          </button>
        </div>
      </aside>

      {/* Main Stage Visualization Viewport */}
      <section className="flex-1 min-w-0 w-full space-y-5">
        {/* Summary Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-white border border-[#E2E8F0] rounded-lg px-4 py-3">
            <div className="text-[11px] font-medium text-[#64748B]">Active Stage</div>
            <div className="text-lg font-mono font-semibold text-[#0F172A] mt-0.5">
              {stageIdx + 1} / {trace.stages.length}
            </div>
            <div className="text-xs text-[#475569] mt-0.5 capitalize">
              {activeStage.direction} · {activeStage.kind}
            </div>
          </div>

          <div className="bg-white border border-[#E2E8F0] rounded-lg px-4 py-3">
            <div className="text-[11px] font-medium text-[#64748B]">Coordinates Modified</div>
            <div className="text-lg font-mono font-semibold text-[#0F172A] mt-0.5">
              {activeStage.changedIndices.length} / {trace.dim}
            </div>
            <div className="text-xs text-[#475569] mt-0.5">
              Round {activeStage.roundNum} of {trace.rounds}
            </div>
          </div>

          <div className="bg-white border border-[#E2E8F0] rounded-lg px-4 py-3">
            <div className="text-[11px] font-medium text-[#64748B]">Round-Trip Check</div>
            <div className="text-lg font-semibold text-[#15803D] mt-0.5 flex items-center gap-1.5">
              <Check className="w-4 h-4" aria-hidden="true" />
              {trace.exactMatch ? 'Exact Match' : 'Mismatch'}
            </div>
            <div className="text-xs text-[#475569] mt-0.5">
              {trace.inputUnmutated ? '✓ Input unmutated' : '✕ Input mutated'}
            </div>
          </div>

          <div className="bg-white border border-[#E2E8F0] rounded-lg px-4 py-3">
            <div className="text-[11px] font-medium text-[#64748B]">State Digest (FNV-64)</div>
            <div className="text-sm font-mono font-semibold text-[#0F172A] mt-1 truncate">
              {fnvChecksumHex(activeStage.values)}
            </div>
            <div className="text-xs font-mono text-[#64748B] mt-0.5 truncate">
              in={fnvChecksumHex(trace.input).slice(0, 8)} · out={fnvChecksumHex(trace.output).slice(0, 8)}
            </div>
          </div>
        </div>

        {/* Transport & Scrubber Panel */}
        <div className="bg-white border border-[#E2E8F0] rounded-lg p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold text-[#0F172A]">{activeStage.title}</h2>
              <p className="text-xs font-mono text-[#0F4C81] mt-0.5">{activeStage.formula}</p>
            </div>

            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setStageIdx((i) => Math.max(0, i - 1));
                }}
                disabled={stageIdx === 0}
                aria-label="Previous stage"
                className="px-2.5 py-1.5 border border-[#CBD5E1] rounded-md text-xs font-medium text-[#1E293B] hover:bg-[#F1F5F9] disabled:opacity-40"
              >
                <ChevronLeft className="w-4 h-4" aria-hidden="true" />
              </button>

              <button
                type="button"
                onClick={() => {
                  if (stageIdx >= trace.stages.length - 1) setStageIdx(0);
                  setIsPlaying((p) => !p);
                }}
                className="px-3.5 py-1.5 bg-[#0F4C81] hover:bg-[#0C3E6B] text-white rounded-md text-xs font-medium flex items-center gap-1.5 whitespace-nowrap shrink-0"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" aria-hidden="true" /> : <Play className="w-3.5 h-3.5" aria-hidden="true" />}
                {isPlaying ? 'Pause' : 'Play Trace'}
              </button>

              <button
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setStageIdx((i) => Math.min(trace.stages.length - 1, i + 1));
                }}
                disabled={stageIdx === trace.stages.length - 1}
                aria-label="Next stage"
                className="px-2.5 py-1.5 border border-[#CBD5E1] rounded-md text-xs font-medium text-[#1E293B] hover:bg-[#F1F5F9] disabled:opacity-40"
              >
                <ChevronRight className="w-4 h-4" aria-hidden="true" />
              </button>
            </div>
          </div>

          {/* Scrubber Slider + Quick Landmarks */}
          <div className="space-y-2.5">
            <input
              type="range"
              aria-label="Transformation stage scrubber"
              min={0}
              max={trace.stages.length - 1}
              value={stageIdx}
              onChange={(e) => {
                setIsPlaying(false);
                setStageIdx(Number(e.target.value));
              }}
              className="w-full accent-[#0F4C81] cursor-pointer"
            />

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-1.5">
                {[
                  { label: 'Input x', idx: stageJumpIndices.inputIdx },
                  { label: 'Scalar S', idx: stageJumpIndices.scalarIdx },
                  { label: 'Output y', idx: stageJumpIndices.outputIdx },
                  { label: 'Recovered x̂', idx: stageJumpIndices.lastIdx },
                ].map((jump) => (
                  <button
                    key={jump.label}
                    type="button"
                    onClick={() => {
                      setIsPlaying(false);
                      setStageIdx(jump.idx);
                    }}
                    className={`px-2.5 py-1 text-xs rounded border whitespace-nowrap shrink-0 ${
                      stageIdx === jump.idx
                        ? 'bg-[#0F4C81] text-white border-[#0F4C81]'
                        : 'bg-[#F8FAFC] text-[#334155] border-[#E2E8F0] hover:bg-[#F1F5F9]'
                    }`}
                  >
                    {jump.label}
                  </button>
                ))}
              </div>

              <select
                aria-label="Jump to specific transformation stage"
                value={stageIdx}
                onChange={(e) => {
                  setIsPlaying(false);
                  setStageIdx(Number(e.target.value));
                }}
                className="px-2.5 py-1 text-xs font-mono bg-[#F8FAFC] border border-[#CBD5E1] rounded text-[#0F172A] max-w-[280px]"
              >
                {trace.stages.map((st) => (
                  <option key={st.index} value={st.index}>
                    [{st.index + 1}] {st.title}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Vector Coordinate Plot (SVG) */}
          <div className="border border-[#E2E8F0] rounded-md bg-[#F8FAFC] p-3">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2 text-xs text-[#475569]">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-3 rounded-sm bg-[#0F4C81]" aria-hidden="true" />
                  Modified (Forward)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-3 rounded-sm bg-[#0284C7]" aria-hidden="true" />
                  Modified (Inverse)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-3 rounded-sm bg-[#CBD5E1]" aria-hidden="true" />
                  Unchanged
                </span>
              </div>
              <div className="font-mono text-[#0F172A]">
                Lane [{selectedLaneData.idx}] · B{selectedLaneData.blockIdx} · val={selectedLaneData.val} · Δ={selectedLaneData.delta}
              </div>
            </div>

            <div className="w-full overflow-x-auto">
              <svg
                viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                className="w-full min-w-[600px] h-auto select-none"
                role="img"
                aria-label={`Vector coordinate bar chart for ${activeStage.title}`}
                data-testid="vector-stage-chart"
              >
                {/* Horizontal gridlines for F_65537 */}
                {[0, 16384, 32768, 49152, 65536].map((tick) => {
                  const yPos = padTop + plotH - (tick / (FIELD_MODULUS - 1)) * plotH;
                  return (
                    <g key={tick}>
                      <line
                        x1={padLeft}
                        y1={yPos}
                        x2={svgWidth - padRight}
                        y2={yPos}
                        stroke="#E2E8F0"
                        strokeWidth={1}
                      />
                      <text
                        x={padLeft - 8}
                        y={yPos + 4}
                        textAnchor="end"
                        className="fill-[#64748B] text-[10px] font-mono"
                      >
                        {tick}
                      </text>
                    </g>
                  );
                })}

                {/* Block boundary dividers */}
                {Array.from({ length: trace.blocks + 1 }, (_, b) => {
                  const xPos = padLeft + b * trace.blockWidth * slotW;
                  return (
                    <g key={`blk-${b}`}>
                      <line
                        x1={xPos}
                        y1={padTop}
                        x2={xPos}
                        y2={padTop + plotH}
                        stroke={b === 0 || b === trace.blocks ? '#94A3B8' : '#CBD5E1'}
                        strokeDasharray={b === 0 || b === trace.blocks ? undefined : '3 3'}
                        strokeWidth={1}
                      />
                      {b < trace.blocks && (
                        <text
                          x={xPos + (trace.blockWidth * slotW) / 2}
                          y={padTop - 8}
                          textAnchor="middle"
                          className="fill-[#64748B] text-[10px] font-mono"
                        >
                          B{b}
                        </text>
                      )}
                    </g>
                  );
                })}

                {/* Coordinate bars */}
                {activeStage.values.map((val, idx) => {
                  const xCenter = padLeft + idx * slotW + slotW / 2;
                  const h = Math.max(2, (val / (FIELD_MODULUS - 1)) * plotH);
                  const yTop = padTop + plotH - h;
                  const isModified = changedSet.has(idx);
                  const isSelected = inspectedLane === idx;

                  let fill = '#CBD5E1';
                  if (isModified) {
                    fill = activeStage.direction === 'forward' ? '#0F4C81' : '#0284C7';
                  }

                  return (
                    <g
                      key={idx}
                      onMouseEnter={() => setInspectedLane(idx)}
                      onClick={() => setInspectedLane(idx)}
                      className="cursor-pointer"
                    >
                      <rect
                        x={xCenter - barW / 2}
                        y={yTop}
                        width={barW}
                        height={h}
                        rx={2}
                        fill={fill}
                        stroke={isSelected ? '#0F172A' : 'none'}
                        strokeWidth={isSelected ? 1.5 : 0}
                      />
                      <text
                        x={xCenter}
                        y={padTop + plotH + 16}
                        textAnchor="middle"
                        className={`text-[10px] font-mono ${
                          isSelected ? 'fill-[#0F172A] font-semibold' : 'fill-[#64748B]'
                        }`}
                      >
                        {idx}
                      </text>
                      {isModified && (
                        <circle cx={xCenter} cy={padTop + plotH + 28} r={2.5} fill={fill} />
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>

          {/* Selected Coordinate Probe */}
          <div className="grid grid-cols-2 sm:grid-cols-6 gap-3 bg-[#F8FAFC] border border-[#E2E8F0] rounded-md px-4 py-3 text-xs">
            <div>
              <div className="text-[#64748B]">Coordinate</div>
              <div className="font-mono font-semibold text-[#0F172A] mt-0.5">
                x[{selectedLaneData.idx}] · B{selectedLaneData.blockIdx}
              </div>
            </div>
            <div>
              <div className="text-[#64748B]">Previous</div>
              <div className="font-mono font-semibold text-[#0F172A] mt-0.5">{selectedLaneData.prevVal}</div>
            </div>
            <div>
              <div className="text-[#64748B]">Current</div>
              <div className="font-mono font-semibold text-[#0F4C81] mt-0.5">{selectedLaneData.val}</div>
            </div>
            <div>
              <div className="text-[#64748B]">Delta (mod q)</div>
              <div className="font-mono font-semibold text-[#0F172A] mt-0.5">+{selectedLaneData.delta}</div>
            </div>
            <div>
              <div className="text-[#64748B]">Alpha (α)</div>
              <div className="font-mono font-semibold text-[#0F172A] mt-0.5">{selectedLaneData.alpha}</div>
            </div>
            <div>
              <div className="text-[#64748B]">Beta (β)</div>
              <div className="font-mono font-semibold text-[#0F172A] mt-0.5">{selectedLaneData.beta}</div>
            </div>
          </div>
        </div>

        {/* Per-Coordinate Diff Matrix Table */}
        <div className="bg-white border border-[#E2E8F0] rounded-lg p-5 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-[#0F172A]">Coordinate State &amp; Delta Table</h3>
              <p className="text-xs text-[#64748B]">
                Inspecting {laneRows.length} of {trace.dim} lanes for {activeStage.title}
              </p>
            </div>
            <div className="flex items-center gap-1 p-1 bg-[#F1F5F9] rounded-md">
              <button
                type="button"
                onClick={() => setViewFilter('all')}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  viewFilter === 'all' ? 'bg-white text-[#0F172A] shadow-sm' : 'text-[#475569] hover:text-[#0F172A]'
                }`}
              >
                All Lanes ({trace.dim})
              </button>
              <button
                type="button"
                onClick={() => setViewFilter('changed')}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  viewFilter === 'changed' ? 'bg-white text-[#0F172A] shadow-sm' : 'text-[#475569] hover:text-[#0F172A]'
                }`}
              >
                Modified Only ({activeStage.changedIndices.length})
              </button>
            </div>
          </div>

          <div className="overflow-x-auto border border-[#E2E8F0] rounded-md">
            <table className="w-full text-xs text-left border-collapse">
              <thead className="bg-[#F8FAFC] text-[#475569] border-b border-[#E2E8F0]">
                <tr>
                  <th className="py-2 px-3 font-medium">Index</th>
                  <th className="py-2 px-3 font-medium">Block</th>
                  <th className="py-2 px-3 font-medium text-right">α_i</th>
                  <th className="py-2 px-3 font-medium text-right">β_i</th>
                  <th className="py-2 px-3 font-medium text-right">Previous</th>
                  <th className="py-2 px-3 font-medium text-right">Current</th>
                  <th className="py-2 px-3 font-medium text-right">Δ mod q</th>
                  <th className="py-2 px-3 font-medium">State</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E2E8F0] font-mono">
                {laneRows.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-6 text-center font-sans text-[#64748B]">
                      No coordinates changed during this stage (input or terminal landmark stage).
                    </td>
                  </tr>
                ) : (
                  laneRows.map((row) => (
                    <tr
                      key={row.idx}
                      onClick={() => setInspectedLane(row.idx)}
                      className={`cursor-pointer transition-colors ${
                        inspectedLane === row.idx ? 'bg-[#0F4C81]/5' : 'hover:bg-[#F8FAFC]'
                      }`}
                    >
                      <td className="py-2 px-3 text-[#0F172A] font-semibold">{row.idx}</td>
                      <td className="py-2 px-3 text-[#475569]">B{row.blockIdx}</td>
                      <td className="py-2 px-3 text-right text-[#475569]">{row.alpha}</td>
                      <td className="py-2 px-3 text-right text-[#475569]">{row.beta}</td>
                      <td className="py-2 px-3 text-right text-[#475569]">{row.prevVal}</td>
                      <td className="py-2 px-3 text-right text-[#0F172A] font-semibold">{row.val}</td>
                      <td className="py-2 px-3 text-right text-[#0F4C81]">+{row.delta}</td>
                      <td className="py-2 px-3 font-sans">
                        {row.changed ? (
                          <span className="text-[#0F4C81] font-medium">● Modified</span>
                        ) : (
                          <span className="text-[#64748B]">○ Unchanged</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
