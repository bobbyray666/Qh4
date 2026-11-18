'use client';

import React, { useState } from 'react';
import StageVisualizer from './StageVisualizer';
import MathConfigSuite from './MathConfigSuite';
import BenchmarkPelagosPanel from './BenchmarkPelagosPanel';
import GpuTelemetryChart from './GpuTelemetryChart';
import RepositoryExplorer from './RepositoryExplorer';

type ActiveTab = 'visualizer' | 'suite' | 'benchmark' | 'gpu' | 'repository';

export default function Qh4Workbench() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('visualizer');

  const navItems: Array<{ id: ActiveTab; label: string }> = [
    { id: 'visualizer', label: 'Stage Visualizer' },
    { id: 'suite', label: 'Math & Config Suite' },
    { id: 'benchmark', label: 'Benchmarks & Pelagos' },
    { id: 'gpu', label: 'GPU Telemetry & Charts' },
    { id: 'repository', label: 'Repository' },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-[#0F172A]">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="bg-white border-b border-[#E2E8F0]">
        <div className="max-w-[1400px] mx-auto px-4 md:px-6 py-3.5 flex items-center justify-between gap-4 md:gap-8">
          {/* Zone 1: Single-line Brand Wordmark */}
          <span className="text-sm md:text-base font-semibold text-[#0F172A] whitespace-nowrap shrink-0">
            QH4 FieldPermutation
          </span>

          {/* Zone 2: Clean single-line navigation links */}
          <nav aria-label="Workbench sections" className="flex items-center gap-4 md:gap-6 overflow-x-auto">
            {navItems.map((item) => {
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  aria-current={active ? 'page' : undefined}
                  className={`py-1 text-xs md:text-sm font-medium border-b-2 transition-colors whitespace-nowrap shrink-0 ${
                    active
                      ? 'border-[#0F4C81] text-[#0F4C81]'
                      : 'border-transparent text-[#475569] hover:text-[#0F172A]'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: 1 Primary Action Button */}
          <div className="shrink-0 hidden sm:block">
            <button
              type="button"
              onClick={() => setActiveTab('suite')}
              className="px-4 py-2 bg-[#0F4C81] hover:bg-[#0C3E6B] text-white text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              Run Verification
            </button>
          </div>
        </div>
      </header>

      {/* Main Workbench Content */}
      <main className="flex-1 max-w-[1400px] w-full mx-auto px-6 py-6 space-y-6">
        <div className="border-b border-[#E2E8F0] pb-4 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold text-[#0F172A]">
              QH4-FieldPermutation v0.2.0 Experimental Workbench
            </h1>
            <p className="text-sm text-[#475569] mt-1 max-w-3xl">
              Isolated F<sub>65,537</sub> scalar &amp; two-sweep mixing implementation (T<sub>n</sub> = L<sup>n</sup> ∘ S) with CPU reference baseline, Triton GPU kernels, and Pelagos-style multi-worker orchestration.
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs text-[#64748B] font-mono">
            <span>q = 65,537 · d = 5 · e = 52,429 · Apache-2.0</span>
            <button
              type="button"
              onClick={() => setActiveTab('repository')}
              className="font-sans font-medium text-[#0F4C81] hover:underline whitespace-nowrap"
            >
              Copy Full Codebase →
            </button>
          </div>
        </div>

        {activeTab === 'visualizer' && <StageVisualizer />}
        {activeTab === 'suite' && <MathConfigSuite />}
        {activeTab === 'benchmark' && <BenchmarkPelagosPanel />}
        {activeTab === 'gpu' && <GpuTelemetryChart />}
        {activeTab === 'repository' && <RepositoryExplorer />}
      </main>
    </div>
  );
}
