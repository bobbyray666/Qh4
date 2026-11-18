import type { NextConfig } from 'next';
import { PHASE_DEVELOPMENT_SERVER } from 'next/constants';

export default function createNextConfig(phase: string): NextConfig {
  return {
    // Isolate dev server artifacts from `next build` so concurrent dev/build runs never corrupt `.next` manifests.
    distDir: phase === PHASE_DEVELOPMENT_SERVER ? '.next-dev' : '.next',
    reactStrictMode: true,
    eslint: {
      ignoreDuringBuilds: true,
    },
    typescript: {
      ignoreBuildErrors: false,
    },
    // Allow access to remote image placeholder.
    images: {
      remotePatterns: [
        {
          protocol: 'https',
          hostname: 'picsum.photos',
          port: '',
          pathname: '/**', // This allows any path under the hostname
        },
      ],
    },
    output: 'standalone',
    outputFileTracingExcludes: {
      '*': [
        '**/node_modules/typescript/**',
        '**/node_modules/eslint*/**',
        '**/node_modules/firebase-tools/**',
        '**/node_modules/caniuse-lite/data/**',
        '**/.next-dev/**',
        '**/.git/**',
        '**/qh4-experimental/**/__pycache__/**',
      ],
    },
    outputFileTracingIncludes: {
      '/api/qh4/bundle': [
        'app/**/*',
        'components/**/*',
        'lib/**/*',
        'hooks/**/*',
        'qh4-experimental/**/*',
        'eslint.config.mjs',
        'next.config.ts',
        'postcss.config.mjs',
        'tsconfig.json',
      ],
      '/api/qh4/source': [
        'app/**/*',
        'components/**/*',
        'lib/**/*',
        'hooks/**/*',
        'qh4-experimental/**/*',
        'eslint.config.mjs',
        'next.config.ts',
        'postcss.config.mjs',
        'tsconfig.json',
      ],
    },
    transpilePackages: ['motion'],
    webpack: (config, { dev }) => {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      if (dev && process.env.DISABLE_HMR === 'true') {
        config.watchOptions = {
          ignored: /.*/,
        };
      }
      return config;
    },
  };
}
