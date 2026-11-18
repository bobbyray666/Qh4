// SPDX-License-Identifier: Apache-2.0

import { timingSafeEqual } from 'node:crypto';
import { NextRequest, NextResponse } from 'next/server';

const DEFAULT_OPENAI_ORIGINS = [
  'https://chatgpt.com',
  'https://chat.openai.com',
  'https://platform.openai.com',
  'https://openai.com',
];

export function getAllowedOrigins(envValue = process.env.OPENAI_ALLOWED_ORIGINS, appUrl = process.env.APP_URL): string[] {
  const origins = new Set<string>(DEFAULT_OPENAI_ORIGINS);
  if (envValue) {
    for (const item of envValue.split(',')) {
      const trimmed = item.trim();
      if (trimmed) origins.add(trimmed);
    }
  }
  if (appUrl) {
    try {
      origins.add(new URL(appUrl).origin);
    } catch {
      // Ignore malformed APP_URL
    }
  }
  return Array.from(origins);
}

export function isOriginAllowed(origin: string | null, allowedOrigins: string[]): boolean {
  if (!origin) return false;
  if (allowedOrigins.includes('*')) return true;
  if (allowedOrigins.includes(origin)) return true;
  try {
    const parsed = new URL(origin);
    if (parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1') {
      return true;
    }
    if (
      parsed.protocol === 'https:' &&
      (parsed.hostname === 'openai.com' ||
        parsed.hostname.endsWith('.openai.com') ||
        parsed.hostname === 'oaiusercontent.com' ||
        parsed.hostname.endsWith('.oaiusercontent.com'))
    ) {
      return true;
    }
  } catch {
    return false;
  }
  return false;
}

export function resolvePublicBaseUrl(req: NextRequest): string {
  const forwardedHost = req.headers.get('x-forwarded-host')?.split(',')[0]?.trim();
  const host = forwardedHost || req.headers.get('host')?.trim();
  if (host) {
    const forwardedProto = req.headers.get('x-forwarded-proto')?.split(',')[0]?.trim();
    const isLocal = host.startsWith('localhost') || host.startsWith('127.0.0.1');
    const proto = forwardedProto || (isLocal ? 'http' : 'https');
    return `${proto}://${host}`;
  }
  const rawAppUrl = process.env.APP_URL?.trim();
  if (rawAppUrl && rawAppUrl !== 'MY_APP_URL') {
    return rawAppUrl.replace(/\/+$/, '');
  }
  return 'http://localhost:3000';
}

export function buildCorsHeaders(
  origin: string | null,
  envValue = process.env.OPENAI_ALLOWED_ORIGINS,
  appUrl = process.env.APP_URL
): Record<string, string> {
  const allowed = getAllowedOrigins(envValue, appUrl);
  const headers: Record<string, string> = {
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
  };

  if (allowed.includes('*')) {
    headers['Access-Control-Allow-Origin'] = '*';
  } else if (origin && isOriginAllowed(origin, allowed)) {
    headers['Access-Control-Allow-Origin'] = origin;
  }
  return headers;
}

export function withCors(res: NextResponse, req: NextRequest): NextResponse {
  const headers = buildCorsHeaders(req.headers.get('origin'));
  for (const [k, v] of Object.entries(headers)) {
    res.headers.set(k, v);
  }
  return res;
}

export function handleCorsPreflight(req: NextRequest): NextResponse {
  const res = new NextResponse(null, { status: 204 });
  return withCors(res, req);
}

function constantTimeTokenMatch(provided: string, expected: string): boolean {
  const a = Buffer.from(provided, 'utf8');
  const b = Buffer.from(expected, 'utf8');
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

export function verifyApiAuthorization(
  req: NextRequest,
  configuredToken = process.env.QH4_API_TOKEN
): { authorized: true } | { authorized: false; response: NextResponse } {
  const token = configuredToken?.trim();
  if (!token) {
    return { authorized: true };
  }

  // Allow same-origin requests originating directly from this app's own UI
  const secFetchSite = req.headers.get('sec-fetch-site');
  if (secFetchSite === 'same-origin') {
    return { authorized: true };
  }

  const authHeader = req.headers.get('authorization') ?? '';
  const match = /^Bearer\s+(.+)$/i.exec(authHeader.trim());
  if (!match || !constantTimeTokenMatch(match[1], token)) {
    const res = NextResponse.json(
      {
        ok: false,
        error: {
          code: 'UNAUTHORIZED',
          message: 'Valid Authorization: Bearer <QH4_API_TOKEN> header is required for external access.',
        },
      },
      { status: 401 }
    );
    return { authorized: false, response: withCors(res, req) };
  }

  return { authorized: true };
}
