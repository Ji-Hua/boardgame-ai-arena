import { defineConfig } from '@playwright/test'
import path from 'node:path'
import { existsSync } from 'node:fs'

const evidence = process.env.E2E_RESULTS_DIR || `/home/jihua/quoridor-data/platform-p1-p2/acceptance/${new Date().toISOString().replaceAll(':', '-')}`
const localLibraries = '/home/jihua/.cache/ms-playwright/local-libs/usr/lib/x86_64-linux-gnu'
if (!path.isAbsolute(evidence) || evidence.includes('/workspace/')) throw new Error('E2E_RESULTS_DIR must be an absolute external evidence directory')

export default defineConfig({
  testDir: '.',
  testMatch: '**/*.spec.mjs',
  timeout: 300_000,
  expect: { timeout: 120_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  outputDir: path.join(evidence, 'artifacts'),
  reporter: [['list'], ['json', { outputFile: path.join(evidence, 'results.json') }], ['html', { outputFolder: path.join(evidence, 'html'), open: 'never' }]],
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://127.0.0.1:8751',
    browserName: 'chromium',
    viewport: { width: 1440, height: 1100 },
    actionTimeout: 15_000,
    navigationTimeout: 120_000,
    screenshot: 'on',
    trace: 'retain-on-failure',
    launchOptions: {
      ...(process.env.E2E_CHROMIUM_PATH ? { executablePath: process.env.E2E_CHROMIUM_PATH } : {}),
      ...(existsSync(localLibraries) ? { env: { ...process.env, LD_LIBRARY_PATH: [localLibraries, process.env.LD_LIBRARY_PATH].filter(Boolean).join(':') } } : {}),
    },
  },
})