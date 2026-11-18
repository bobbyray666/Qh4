// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs/promises';
import path from 'node:path';
import { NextRequest, NextResponse } from 'next/server';
import { handleCorsPreflight, verifyApiAuthorization, withCors } from '@/lib/qh4/access';
import { NEXTJS_APP_FILES, QH4_EXPERIMENTAL_FILES } from '@/lib/qh4/sourceManifest';

const WORKSPACE_ROOT = process.cwd();
const QH4_ROOT = path.resolve(WORKSPACE_ROOT, 'qh4-experimental');

export async function OPTIONS(req: NextRequest) {
  return handleCorsPreflight(req);
}

export async function GET(req: NextRequest) {
  const auth = verifyApiAuthorization(req);
  if (!auth.authorized) {
    return auth.response;
  }

  const listOnly = req.nextUrl.searchParams.get('list') === '1';
  if (listOnly) {
    return withCors(
      NextResponse.json({
        ok: true,
        files: QH4_EXPERIMENTAL_FILES,
        appFiles: NEXTJS_APP_FILES,
      }),
      req
    );
  }

  const fileParam = req.nextUrl.searchParams.get('file') ?? 'qh4_experimental/cpu_reference.py';
  const isExperimentalFile = QH4_EXPERIMENTAL_FILES.includes(
    fileParam as (typeof QH4_EXPERIMENTAL_FILES)[number]
  );
  const isAppFile = NEXTJS_APP_FILES.includes(fileParam as (typeof NEXTJS_APP_FILES)[number]);

  if (!isExperimentalFile && !isAppFile) {
    return withCors(
      NextResponse.json(
        {
          ok: false,
          error: {
            code: 'INVALID_FILE_PATH',
            message: `File "${fileParam}" is not in the whitelisted application or qh4-experimental file list.`,
          },
          files: QH4_EXPERIMENTAL_FILES,
          appFiles: NEXTJS_APP_FILES,
        },
        { status: 400 }
      ),
      req
    );
  }

  try {
    const baseRoot = isExperimentalFile ? QH4_ROOT : WORKSPACE_ROOT;
    const absPath = path.resolve(baseRoot, fileParam);
    if (!absPath.startsWith(baseRoot + path.sep)) {
      return withCors(
        NextResponse.json(
          { ok: false, error: { code: 'FORBIDDEN_PATH', message: 'Invalid repository path.' } },
          { status: 403 }
        ),
        req
      );
    }
    const content = await fs.readFile(absPath, 'utf-8');
    return withCors(
      NextResponse.json({
        ok: true,
        file: fileParam,
        scope: isExperimentalFile ? 'qh4-experimental' : 'nextjs-app',
        files: QH4_EXPERIMENTAL_FILES,
        appFiles: NEXTJS_APP_FILES,
        content,
        lines: content.split('\n').length,
        bytes: Buffer.byteLength(content, 'utf-8'),
      }),
      req
    );
  } catch (err) {
    return withCors(
      NextResponse.json(
        {
          ok: false,
          error: {
            code: 'FILE_READ_ERROR',
            message: err instanceof Error ? err.message : 'Failed to read repository file.',
          },
        },
        { status: 500 }
      ),
      req
    );
  }
}
