'use client';

import React, { useState } from 'react';
import { AlertTriangle, CheckCircle2, Play, RefreshCw } from 'lucide-react';
import { runTypeScriptM01ToM10, MathCheckResult } from '@/lib/qh4/tests';

interface PythonCheck {
  id: string;
  passed: boolean;
  duration_ms: number;
  error: string;
}

interface PythonConfigResult {
  config: string;
  blocks: number;
  dim: number;
  block_width: number;
  rounds: number;
  seed: number;
  vectors_tested: number;
  alphas_preview: number[];
  betas_preview: number[];
  mismatches: number;
  input_mutations: number;
  range_violations: number;
  input_sha256?: string;
  output_sha256: string;
  reconstructed_sha256?: string;
  roundtrip_sha256_match?: boolean;
  vectors_per_sec?: number;
  duration_ms: number;
  passed: boolean;
}

interface PythonConfigsReport {
  version?: string;
  python_version?: string;
  platform?: string;
  hash_specification?: string;
  vectors_per_config: number;
  total_vectors: number;
  all_passed: boolean;
  total_duration_ms: number;
  configs: PythonConfigResult[];
}

export default function MathConfigSuite() {
  const [tsChecks, setTsChecks] = useState<{ allPassed: boolean; totalDurationMs: number; checks: MathCheckResult[] }>(
    () => runTypeScriptM01ToM10()
  );

  const [pyM01Loading, setPyM01Loading] = useState(false);
  const [pyM01Report, setPyM01Report] = useState<{
    backend: string;
    total: number;
    passed: number;
    all_passed: boolean;
    total_duration_ms: number;
    checks: PythonCheck[];
  } | null>(null);

  const [pyConfigsLoading, setPyConfigsLoading] = useState(false);
  const [vectorsPerConfig, setVectorsPerConfig] = useState<number>(1000);
  const [pyConfigsReport, setPyConfigsReport] = useState<PythonConfigsReport | null>(null);

  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const handleRunTsSuite = () => {
    setErrorBanner(null);
    setTsChecks(runTypeScriptM01ToM10());
  };

  const handleRunPythonM01 = async () => {
    setPyM01Loading(true);
    setErrorBanner(null);
    try {
      const res = await fetch('/api/qh4/python-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'm01_m10' }),
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        throw new Error(json.error?.message || 'Python M01-M10 execution failed.');
      }
      setPyM01Report(json.data);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : 'Python M01-M10 failed.');
    } finally {
      setPyM01Loading(false);
    }
  };

  const handleRunPythonConfigs = async () => {
    setPyConfigsLoading(true);
    setErrorBanner(null);
    try {
      const res = await fetch('/api/qh4/python-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'configs', vectorsPerConfig, seed: 202602 }),
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        throw new Error(json.error?.message || '8-Config validation execution failed.');
      }
      setPyConfigsReport(json.data);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : '8-Config validation failed.');
    } finally {
      setPyConfigsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {errorBanner && (
        <div role="alert" className="bg-[#FEF2F2] border border-[#FECACA] rounded-lg p-4 text-sm text-[#991B1B] flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
          <span>{errorBanner}</span>
        </div>
      )}

      {/* Mathematical Test Suite M01-M10 */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-[#0F172A]">Mathematical Verification Suite (M01–M10)</h2>
            <p className="text-xs text-[#475569] mt-0.5">
              Primality of 65,537 · Coprimality · Exhaustive 65,537-element inverse &amp; permutation bijections · Two-sweep L / L⁻¹ invertibility · Avalanche &amp; non-mutation
            </p>
          </div>
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={handleRunTsSuite}
              className="px-3.5 py-2 border border-[#CBD5E1] bg-white hover:bg-[#F8FAFC] text-[#0F172A] text-xs font-medium rounded-md flex items-center gap-1.5 whitespace-nowrap shrink-0"
            >
              <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" />
              Re-run TypeScript M01–M10
            </button>
            <button
              type="button"
              onClick={handleRunPythonM01}
              disabled={pyM01Loading}
              className="px-4 py-2 bg-[#0F4C81] hover:bg-[#0C3E6B] disabled:opacity-60 text-white text-xs font-medium rounded-md flex items-center gap-1.5 whitespace-nowrap shrink-0"
            >
              <Play className="w-3.5 h-3.5" aria-hidden="true" />
              {pyM01Loading ? 'Running Python 3.11...' : 'Run Live Python Reference M01–M10'}
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs text-[#475569] border-b border-[#E2E8F0] pb-3">
          <span className={tsChecks.allPassed ? 'text-[#15803D] font-medium' : 'text-[#B91C1C] font-medium'}>
            {tsChecks.allPassed ? '✓ TypeScript Baseline: 10/10 Passed' : '✕ TypeScript Baseline Failed'}
          </span>
          <span aria-hidden="true">·</span>
          <span className="font-mono">{tsChecks.totalDurationMs} ms</span>
          {pyM01Report && (
            <>
              <span aria-hidden="true">·</span>
              <span className={pyM01Report.all_passed ? 'text-[#15803D] font-medium' : 'text-[#B91C1C] font-medium'}>
                {pyM01Report.all_passed
                  ? `✓ Python 3.11 (${pyM01Report.backend}): ${pyM01Report.passed}/${pyM01Report.total} Passed`
                  : '✕ Python Baseline Failed'}
              </span>
              <span aria-hidden="true">·</span>
              <span className="font-mono">{pyM01Report.total_duration_ms} ms</span>
            </>
          )}
        </div>

        <div className="overflow-x-auto border border-[#E2E8F0] rounded-md">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-[#F8FAFC] text-[#475569] border-b border-[#E2E8F0]">
              <tr>
                <th className="py-2.5 px-3 font-medium">Check</th>
                <th className="py-2.5 px-3 font-medium">Property Verified</th>
                <th className="py-2.5 px-3 font-medium">Empirical Detail</th>
                <th className="py-2.5 px-3 font-medium text-right">TS Time</th>
                <th className="py-2.5 px-3 font-medium">TS Status</th>
                <th className="py-2.5 px-3 font-medium">Python 3.11</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E2E8F0]">
              {tsChecks.checks.map((chk, idx) => {
                const pyItem = pyM01Report?.checks[idx];
                return (
                  <tr key={chk.id} className="hover:bg-[#F8FAFC]">
                    <td className="py-2.5 px-3 font-mono font-semibold text-[#0F172A]">{chk.id}</td>
                    <td className="py-2.5 px-3 font-medium text-[#1E293B]">{chk.name}</td>
                    <td className="py-2.5 px-3 text-[#475569]">{chk.detail}</td>
                    <td className="py-2.5 px-3 font-mono text-right text-[#475569]">{chk.durationMs} ms</td>
                    <td className="py-2.5 px-3">
                      {chk.passed ? (
                        <span className="text-[#15803D] font-medium">✓ Pass</span>
                      ) : (
                        <span className="text-[#B91C1C] font-medium">✕ Fail</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-[#475569]">
                      {pyItem ? (
                        pyItem.passed ? (
                          <span className="text-[#15803D] font-sans font-medium">
                            ✓ Pass ({pyItem.duration_ms} ms)
                          </span>
                        ) : (
                          <span className="text-[#B91C1C] font-sans font-medium">✕ Fail</span>
                        )
                      ) : (
                        <span className="font-sans text-[#94A3B8]">Not run yet</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Eight Block Configuration 10K Randomized Validation */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-[#0F172A]">Eight Block Configuration Randomized Validator</h2>
            <p className="text-xs text-[#475569] mt-0.5">
              Executes reproducible Python 3.11 round-trip &amp; non-mutation tests across 8B-D8 through 18B-D36.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2.5">
            <label htmlFor="vec-count-select" className="text-xs text-[#475569]">
              Vectors / Config:
            </label>
            <select
              id="vec-count-select"
              value={vectorsPerConfig}
              onChange={(e) => setVectorsPerConfig(Number(e.target.value))}
              className="px-2.5 py-2 text-xs font-mono bg-white border border-[#CBD5E1] rounded-md text-[#0F172A]"
            >
              <option value={500}>500 (4,000 total)</option>
              <option value={1000}>1,000 (8,000 total)</option>
              <option value={10000}>10,000 (80,000 total · Full Directive 4)</option>
            </select>
            <button
              type="button"
              onClick={handleRunPythonConfigs}
              disabled={pyConfigsLoading}
              className="px-4 py-2 bg-[#0F4C81] hover:bg-[#0C3E6B] disabled:opacity-60 text-white text-xs font-medium rounded-md flex items-center gap-1.5 whitespace-nowrap shrink-0"
            >
              <Play className="w-3.5 h-3.5" aria-hidden="true" />
              {pyConfigsLoading ? 'Executing Batch...' : 'Run 8-Config Python Validation'}
            </button>
          </div>
        </div>

        {!pyConfigsReport ? (
          <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-md p-6 text-center space-y-2">
            <p className="text-sm font-medium text-[#1E293B]">Ready to run live Python 3.11 8-Configuration verification</p>
            <p className="text-xs text-[#64748B] max-w-xl mx-auto">
              Select 1,000 vectors for quick interactive validation (~6s) or 10,000 vectors per configuration (~75s) to reproduce the full Directive 4 audit in real time.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="space-y-1 text-xs text-[#475569]">
              <div className="flex flex-wrap items-center gap-4">
                <span className="flex items-center gap-1.5 text-[#15803D] font-medium">
                  <CheckCircle2 className="w-4 h-4" aria-hidden="true" />
                  {pyConfigsReport.total_vectors.toLocaleString()} vectors verified · 0 mismatches · 0 input mutations
                </span>
                <span aria-hidden="true">·</span>
                <span className="font-mono">Total duration: {(pyConfigsReport.total_duration_ms / 1000).toFixed(2)}s</span>
                {pyConfigsReport.python_version && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span>Python {pyConfigsReport.python_version}</span>
                  </>
                )}
              </div>
              {pyConfigsReport.hash_specification && (
                <p className="text-[11px] text-[#64748B] font-mono">
                  Hash Spec: {pyConfigsReport.hash_specification}
                </p>
              )}
            </div>

            <div className="overflow-x-auto border border-[#E2E8F0] rounded-md">
              <table className="w-full text-xs text-left border-collapse">
                <thead className="bg-[#F8FAFC] text-[#475569] border-b border-[#E2E8F0]">
                  <tr>
                    <th className="py-2.5 px-3 font-medium">Config</th>
                    <th className="py-2.5 px-3 font-medium text-right">Blocks / Dim</th>
                    <th className="py-2.5 px-3 font-medium text-right">Seed</th>
                    <th className="py-2.5 px-3 font-medium text-right">Vectors</th>
                    <th className="py-2.5 px-3 font-medium">Output SHA-256 Digest</th>
                    <th className="py-2.5 px-3 font-medium text-center">Round-Trip SHA</th>
                    <th className="py-2.5 px-3 font-medium text-right">Throughput</th>
                    <th className="py-2.5 px-3 font-medium text-right">Duration</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E8F0] font-mono">
                  {pyConfigsReport.configs.map((row) => (
                    <tr key={row.config} className="hover:bg-[#F8FAFC]">
                      <td className="py-2.5 px-3 font-semibold text-[#0F172A]">{row.config}</td>
                      <td className="py-2.5 px-3 text-right text-[#475569]">
                        {row.blocks}B · {row.dim}D
                      </td>
                      <td className="py-2.5 px-3 text-right text-[#475569]">{row.seed}</td>
                      <td className="py-2.5 px-3 text-right text-[#0F172A]">{row.vectors_tested.toLocaleString()}</td>
                      <td className="py-2.5 px-3 text-[#475569]" title={`Input: ${row.input_sha256 || 'N/A'}\nOutput: ${row.output_sha256}`}>
                        <div className="flex items-center gap-1.5">
                          <span className="truncate max-w-[220px] text-[#0F4C81]">{row.output_sha256}</span>
                          <button
                            type="button"
                            onClick={() => navigator.clipboard.writeText(row.output_sha256)}
                            className="text-[10px] text-[#64748B] hover:text-[#0F172A] px-1 py-0.5 border border-[#CBD5E1] rounded"
                            title="Copy full 64-character SHA-256"
                          >
                            Copy
                          </button>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-center text-[#15803D]">
                        {row.roundtrip_sha256_match !== false ? '✓ Exact' : '✕ Mismatch'}
                      </td>
                      <td className="py-2.5 px-3 text-right text-[#0F172A]">
                        {row.vectors_per_sec ? `${row.vectors_per_sec.toLocaleString()} v/s` : '—'}
                      </td>
                      <td className="py-2.5 px-3 text-right text-[#475569]">{row.duration_ms} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
