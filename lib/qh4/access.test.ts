// SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';
import { NextRequest } from 'next/server';
import {
  buildCorsHeaders,
  getAllowedOrigins,
  isOriginAllowed,
  resolvePublicBaseUrl,
  verifyApiAuthorization,
} from './access';

test('getAllowedOrigins includes default OpenAI origins and APP_URL', () => {
  const origins = getAllowedOrigins('https://custom-openai-app.example.com', 'https://my-app.run.app/path');
  assert.ok(origins.includes('https://chatgpt.com'));
  assert.ok(origins.includes('https://chat.openai.com'));
  assert.ok(origins.includes('https://platform.openai.com'));
  assert.ok(origins.includes('https://custom-openai-app.example.com'));
  assert.ok(origins.includes('https://my-app.run.app'));
});

test('buildCorsHeaders grants CORS to OpenAI/oaiusercontent origins and blocks untrusted or spoofed origins', () => {
  const openAiHeaders = buildCorsHeaders('https://chatgpt.com', '', '');
  assert.equal(openAiHeaders['Access-Control-Allow-Origin'], 'https://chatgpt.com');

  const sandboxHeaders = buildCorsHeaders('https://web-sandbox.oaiusercontent.com', '', '');
  assert.equal(sandboxHeaders['Access-Control-Allow-Origin'], 'https://web-sandbox.oaiusercontent.com');

  const untrustedHeaders = buildCorsHeaders('https://evil.example.net', '', '');
  assert.equal(untrustedHeaders['Access-Control-Allow-Origin'], undefined);
  assert.equal(isOriginAllowed('https://evil.example.net', getAllowedOrigins('', '')), false);
  assert.equal(isOriginAllowed('https://openai.com.attacker.net', getAllowedOrigins('', '')), false);
});

test('resolvePublicBaseUrl uses request forwarded host when present', () => {
  const req = new NextRequest('http://localhost:3000/api/openapi.json', {
    headers: {
      'x-forwarded-host': 'ais-pre-example.us-east1.run.app',
      'x-forwarded-proto': 'https',
    },
  });
  assert.equal(resolvePublicBaseUrl(req), 'https://ais-pre-example.us-east1.run.app');
});

test('verifyApiAuthorization allows open access when QH4_API_TOKEN is unset and enforces Bearer when set', () => {
  const reqNoAuth = new NextRequest('http://localhost:3000/api/qh4/trace', { method: 'POST' });
  assert.equal(verifyApiAuthorization(reqNoAuth, '').authorized, true);

  const denied = verifyApiAuthorization(reqNoAuth, 'secret-token-123');
  assert.equal(denied.authorized, false);
  if (!denied.authorized) {
    assert.equal(denied.response.status, 401);
  }

  const reqWithAuth = new NextRequest('http://localhost:3000/api/qh4/trace', {
    method: 'POST',
    headers: { authorization: 'Bearer secret-token-123' },
  });
  assert.equal(verifyApiAuthorization(reqWithAuth, 'secret-token-123').authorized, true);

  const reqSameOrigin = new NextRequest('http://localhost:3000/api/qh4/trace', {
    method: 'POST',
    headers: { 'sec-fetch-site': 'same-origin' },
  });
  assert.equal(verifyApiAuthorization(reqSameOrigin, 'secret-token-123').authorized, true);
});
