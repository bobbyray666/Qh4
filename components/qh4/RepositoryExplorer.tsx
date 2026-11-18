'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Check, Copy, Download, FileCode2, FileText } from 'lucide-react';

function copyTextWithFallback(text: string): Promise<boolean> {
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    return navigator.clipboard
      .writeText(text)
      .then(() => true)
      .catch(() => execCommandCopy(text));
  }
  return Promise.resolve(execCommandCopy(text));
}

function execCommandCopy(text: string): boolean {
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '-9999px';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

export default function RepositoryExplorer() {
  const [files, setFiles] = useState<string[]>([]);
  const [appFiles, setAppFiles] = useState<string[]>([]);
  const [activeFile, setActiveFile] = useState<string>('qh4_experimental/cpu_reference.py');
  const [loadedFile, setLoadedFile] = useState<string>('');
  const [content, setContent] = useState<string>('');
  const [meta, setMeta] = useState<{ lines: number; bytes: number }>({ lines: 0, bytes: 0 });
  const [copiedFile, setCopiedFile] = useState<boolean>(false);

  // Full codebase one-click bundle state (preloaded for synchronous clipboard gesture)
  const [bundleText, setBundleText] = useState<string>('');
  const [bundleStatus, setBundleStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [copiedAll, setCopiedAll] = useState<'idle' | 'copied' | 'fallback'>('idle');
  const [showRawBundle, setShowRawBundle] = useState<boolean>(false);
  const rawTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  const loading = loadedFile !== activeFile;

  useEffect(() => {
    let mounted = true;
    fetch(`/api/qh4/source?file=${encodeURIComponent(activeFile)}`)
      .then((r) => r.json())
      .then((json) => {
        if (!mounted) return;
        if (json.ok) {
          setFiles(json.files ?? []);
          setAppFiles(json.appFiles ?? []);
          setContent(json.content);
          setMeta({ lines: json.lines, bytes: json.bytes });
          setLoadedFile(activeFile);
        }
      });
    return () => {
      mounted = false;
    };
  }, [activeFile]);

  useEffect(() => {
    let mounted = true;
    fetch('/api/qh4/bundle?format=markdown&scope=all')
      .then(async (res) => {
        if (!res.ok) throw new Error('Bundle fetch failed');
        return res.text();
      })
      .then((text) => {
        if (!mounted) return;
        setBundleText(text);
        setBundleStatus('ready');
      })
      .catch(() => {
        if (!mounted) return;
        setBundleStatus('error');
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleCopyActive = async () => {
    if (!content) return;
    await copyTextWithFallback(content);
    setCopiedFile(true);
    window.setTimeout(() => setCopiedFile(false), 1500);
  };

  const handleCopyEntireCodebase = async () => {
    let textToCopy = bundleText;
    if (!textToCopy) {
      const res = await fetch('/api/qh4/bundle?format=markdown&scope=all');
      if (!res.ok) return;
      textToCopy = await res.text();
      setBundleText(textToCopy);
      setBundleStatus('ready');
    }
    const ok = await copyTextWithFallback(textToCopy);
    if (ok) {
      setCopiedAll('copied');
      window.setTimeout(() => setCopiedAll('idle'), 2200);
    } else {
      setCopiedAll('fallback');
      setShowRawBundle(true);
      window.setTimeout(() => rawTextareaRef.current?.select(), 50);
    }
  };

  const handleDownloadBundle = async () => {
    let text = bundleText;
    if (!text) {
      const res = await fetch('/api/qh4/bundle?format=markdown&scope=all');
      if (!res.ok) return;
      text = await res.text();
      setBundleText(text);
      setBundleStatus('ready');
    }
    const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'qh4-source-bundle.md';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const totalFiles = files.length + appFiles.length;
  const bundleKb = bundleText ? (new TextEncoder().encode(bundleText).length / 1024).toFixed(1) : null;

  return (
    <div className="space-y-4">
      {/* One-Click Full Codebase Copy & Export Banner */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg px-5 py-4 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-0.5">
          <h2 className="text-sm font-semibold text-[#0F172A]">
            Full Codebase One-Click Copy &amp; Export
          </h2>
          <p className="text-xs text-[#475569]">
            {bundleStatus === 'ready' && totalFiles > 0
              ? `Ready: ${totalFiles} files (${files.length} qh4-experimental + ${appFiles.length} Next.js app) · ${bundleKb} KB Markdown bundle (secrets excluded).`
              : bundleStatus === 'error'
                ? 'Could not preload bundle; click Copy Full Codebase to retry.'
                : 'Preparing full codebase bundle...'}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleCopyEntireCodebase}
            className="px-4 py-2 bg-[#0F4C81] hover:bg-[#0C3E6B] text-white rounded-md text-xs font-medium flex items-center gap-1.5 whitespace-nowrap shrink-0"
          >
            {copiedAll === 'copied' ? (
              <Check className="w-3.5 h-3.5" aria-hidden="true" />
            ) : (
              <Copy className="w-3.5 h-3.5" aria-hidden="true" />
            )}
            {copiedAll === 'copied'
              ? `Copied All ${totalFiles || ''} Files!`
              : 'Copy Full Codebase (1-Click)'}
          </button>

          <button
            type="button"
            onClick={handleDownloadBundle}
            className="px-3.5 py-2 border border-[#CBD5E1] bg-white hover:bg-[#F8FAFC] text-[#1E293B] rounded-md text-xs font-medium flex items-center gap-1.5 whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" aria-hidden="true" />
            Download .md Bundle
          </button>

          <button
            type="button"
            onClick={() => setShowRawBundle((v) => !v)}
            className="px-3 py-2 border border-[#CBD5E1] bg-white hover:bg-[#F8FAFC] text-[#475569] rounded-md text-xs font-medium flex items-center gap-1.5 whitespace-nowrap shrink-0"
          >
            <FileText className="w-3.5 h-3.5" aria-hidden="true" />
            {showRawBundle ? 'Hide Raw Bundle' : 'View Raw Copy Box'}
          </button>
        </div>
      </div>

      {showRawBundle && (
        <div className="bg-white border border-[#E2E8F0] rounded-lg p-4 space-y-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-[#334155]">
              {copiedAll === 'fallback'
                ? 'Clipboard access was blocked by the browser frame — press Ctrl+C / Cmd+C below to copy all files:'
                : 'Complete Markdown Source Bundle (Select All + Copy):'}
            </span>
            <button
              type="button"
              onClick={() => rawTextareaRef.current?.select()}
              className="px-2.5 py-1 text-xs border border-[#CBD5E1] rounded bg-[#F8FAFC] hover:bg-[#F1F5F9] text-[#0F172A]"
            >
              Select All
            </button>
          </div>
          <textarea
            ref={rawTextareaRef}
            readOnly
            value={bundleText}
            rows={12}
            aria-label="Complete source bundle text"
            className="w-full p-3 text-[11px] font-mono bg-[#0F172A] text-[#E2E8F0] border border-[#334155] rounded-md"
          />
        </div>
      )}

      {/* Split Repository File Browser */}
      <div className="bg-white border border-[#E2E8F0] rounded-lg overflow-hidden flex flex-col lg:flex-row min-h-[540px]">
        <aside className="w-full lg:w-[290px] shrink-0 bg-[#F8FAFC] border-b lg:border-b-0 lg:border-r border-[#E2E8F0] p-4 space-y-4 max-h-[680px] overflow-y-auto">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#475569]">
              qh4-experimental ({files.length})
            </h3>
            <nav aria-label="Isolated repository files" className="space-y-1 mt-2">
              {files.map((f) => {
                const active = f === activeFile;
                return (
                  <button
                    key={f}
                    type="button"
                    onClick={() => setActiveFile(f)}
                    className={`w-full text-left px-2.5 py-1.5 rounded text-xs font-mono flex items-center gap-2 transition-colors ${
                      active
                        ? 'bg-[#0F4C81] text-white font-medium'
                        : 'text-[#334155] hover:bg-[#E2E8F0]/60'
                    }`}
                  >
                    <FileCode2 className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                    <span className="truncate">{f}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          {appFiles.length > 0 && (
            <div className="border-t border-[#E2E8F0] pt-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#475569]">
                Next.js App ({appFiles.length})
              </h3>
              <nav aria-label="Next.js application files" className="space-y-1 mt-2">
                {appFiles.map((f) => {
                  const active = f === activeFile;
                  return (
                    <button
                      key={f}
                      type="button"
                      onClick={() => setActiveFile(f)}
                      className={`w-full text-left px-2.5 py-1.5 rounded text-xs font-mono flex items-center gap-2 transition-colors ${
                        active
                          ? 'bg-[#0F4C81] text-white font-medium'
                          : 'text-[#334155] hover:bg-[#E2E8F0]/60'
                      }`}
                    >
                      <FileCode2 className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                      <span className="truncate">{f}</span>
                    </button>
                  );
                })}
              </nav>
            </div>
          )}
        </aside>

        <div className="flex-1 min-w-0 flex flex-col">
          <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 border-b border-[#E2E8F0] bg-white">
            <div className="min-w-0">
              <div className="text-sm font-mono font-semibold text-[#0F172A] truncate">{activeFile}</div>
              <div className="text-xs text-[#64748B]">
                {meta.lines} lines · {meta.bytes.toLocaleString()} bytes · SPDX-License-Identifier: Apache-2.0
              </div>
            </div>

            <button
              type="button"
              onClick={handleCopyActive}
              className="px-3 py-1.5 border border-[#CBD5E1] rounded-md text-xs font-medium text-[#1E293B] hover:bg-[#F8FAFC] flex items-center gap-1.5 whitespace-nowrap shrink-0"
            >
              {copiedFile ? (
                <Check className="w-3.5 h-3.5 text-[#15803D]" aria-hidden="true" />
              ) : (
                <Copy className="w-3.5 h-3.5" aria-hidden="true" />
              )}
              {copiedFile ? 'Copied' : 'Copy File'}
            </button>
          </div>

          <div className="flex-1 overflow-auto bg-[#0F172A] text-[#E2E8F0] p-5 max-h-[620px]">
            {loading ? (
              <div className="text-xs font-mono text-[#94A3B8]">Loading file contents...</div>
            ) : (
              <pre className="text-xs font-mono leading-relaxed whitespace-pre">{content}</pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
