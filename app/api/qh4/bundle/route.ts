// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs/promises';
import path from 'node:path';
import { NextRequest, NextResponse } from 'next/server';
import { handleCorsPreflight, verifyApiAuthorization, withCors } from '@/lib/qh4/access';
import { NEXTJS_APP_FILES, QH4_EXPERIMENTAL_FILES } from '@/lib/qh4/sourceManifest';

const WORKSPACE_ROOT = process.cwd();
const QH4_ROOT = path.resolve(WORKSPACE_ROOT, 'qh4-experimental');

interface BundleFileEntry {
  path: string;
  scope: 'qh4-experimental' | 'nextjs-app';
  lines: number;
  bytes: number;
  content: string;
}

function langForPath(filePath: string): string {
  if (filePath.endsWith('.py')) return 'python';
  if (filePath.endsWith('.ts') || filePath.endsWith('.tsx')) return 'tsx';
  if (filePath.endsWith('.json')) return 'json';
  if (filePath.endsWith('.css')) return 'css';
  if (filePath.endsWith('.toml')) return 'toml';
  if (filePath.endsWith('.md')) return 'markdown';
  return 'text';
}

export async function OPTIONS(req: NextRequest) {
  return handleCorsPreflight(req);
}

export async function GET(req: NextRequest) {
  const auth = verifyApiAuthorization(req);
  if (!auth.authorized) {
    return auth.response;
  }

  const scopeParam = req.nextUrl.searchParams.get('scope') ?? 'all';
  const formatParam = req.nextUrl.searchParams.get('format') ?? 'json';
  const download = req.nextUrl.searchParams.get('download') === '1';

  const entries: BundleFileEntry[] = [];

  try {
    if (scopeParam === 'all' || scopeParam === 'qh4-experimental') {
      for (const rel of QH4_EXPERIMENTAL_FILES) {
        const abs = path.resolve(QH4_ROOT, rel);
        if (!abs.startsWith(QH4_ROOT + path.sep)) continue;
        const content = await fs.readFile(abs, 'utf-8');
        entries.push({
          path: `qh4-experimental/${rel}`,
          scope: 'qh4-experimental',
          lines: content.split('\n').length,
          bytes: Buffer.byteLength(content, 'utf-8'),
          content,
        });
      }
    }

    if (scopeParam === 'all' || scopeParam === 'nextjs-app') {
      for (const rel of NEXTJS_APP_FILES) {
        const abs = path.resolve(WORKSPACE_ROOT, rel);
        if (!abs.startsWith(WORKSPACE_ROOT + path.sep)) continue;
        const content = await fs.readFile(abs, 'utf-8');
        entries.push({
          path: rel,
          scope: 'nextjs-app',
          lines: content.split('\n').length,
          bytes: Buffer.byteLength(content, 'utf-8'),
          content,
        });
      }
    }
  } catch (err) {
    return withCors(
      NextResponse.json(
        {
          ok: false,
          error: {
            code: 'BUNDLE_READ_ERROR',
            message: err instanceof Error ? err.message : 'Failed to assemble source bundle.',
          },
        },
        { status: 500 }
      ),
      req
    );
  }

  if (formatParam === 'markdown' || formatParam === 'txt') {
    const sections = [
      '# QH4-FieldPermutation v0.2.0 — Complete Source Bundle',
      '',
      `Exported ${entries.length} whitelisted source files (no secrets or .env files included).`,
      '',
      '## File Index',
      ...entries.map((e) => `- \`${e.path}\` (${e.lines} lines, ${e.bytes} bytes)`),
      '',
    ];

    for (const entry of entries) {
      const lang = langForPath(entry.path);
      sections.push(`## \`${entry.path}\``);
      sections.push('');
      sections.push(`\`\`\`${lang}`);
      sections.push(entry.content.replace(/\n+$/, ''));
      sections.push('```');
      sections.push('');
    }

    const body = sections.join('\n');
    const headers: Record<string, string> = {
      'Content-Type': 'text/markdown; charset=utf-8',
    };
    if (download) {
      headers['Content-Disposition'] = 'attachment; filename="qh4-source-bundle.md"';
    }
    return withCors(new NextResponse(body, { status: 200, headers }), req);
  }

  const res = NextResponse.json({
    ok: true,
    bundleVersion: '0.2.0',
    scope: scopeParam,
    fileCount: entries.length,
    totalBytes: entries.reduce((sum, e) => sum + e.bytes, 0),
    files: entries,
  });

  if (download) {
    res.headers.set('Content-Disposition', 'attachment; filename="qh4-source-bundle.json"');
  }
  return withCors(res, req);
}
