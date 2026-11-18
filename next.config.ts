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
        '**/node_modules/next/dist/compiled/webpack/**',
        '**/node_modules/next/dist/compiled/babel/**',
        '**/node_modules/next/dist/compiled/terser/**',
        '**/node_modules/next/dist/compiled/@babel/**',
        '**/node_modules/next/dist/compiled/sass-loader/**',
        '**/node_modules/next/dist/compiled/css-loader/**',
        '**/node_modules/next/dist/compiled/postcss-loader/**',
        '**/node_modules/next/dist/compiled/mini-css-extract-plugin/**',
        '**/node_modules/next/dist/build/webpack/**',
        '**/node_modules/next/dist/build/babel/**',
        '**/node_modules/next/dist/build/swc/**',
        '**/node_modules/next/dist/build/analysis/**',
        '**/node_modules/next/dist/build/templates/**',
        '**/node_modules/next/dist/server/dev/next-dev-server*',
        '**/node_modules/next/dist/server/dev/hot-reloader-webpack*',
        '**/node_modules/next/dist/server/dev/hot-reloader-turbopack*',
        '**/node_modules/next/dist/server/dev/turbopack*',
        '**/node_modules/next/dist/client/dev/**',
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
