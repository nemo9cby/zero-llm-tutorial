// The live trainer of content/train-in-browser.md, run in headless Chromium with the page's own settings,
// reproduces the recorded default run (web/data/train-in-browser/runs.json) step for step.
// It drives web/widgets/trainer-worker.js exactly as trainer-live.js does: init with the widget's config
// from the page, ask for the step-0 sample, then "开始训练".
//   node --test tests/js/trainer-browser.test.mjs        (needs Playwright's Chromium; skipped without it)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, statSync, createReadStream } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SETTINGS } from './fixtures/trainer/record_runs.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const WEB = path.join(ROOT, 'web');
const STEPS = 500; // covers the first crossing of the bigram line (step 475)
const rec = JSON.parse(readFileSync(path.join(WEB, 'data/train-in-browser/runs.json'), 'utf8'));
const r4 = (v) => Math.round(v * 1e4) / 1e4;

/** The JSON config of the ```widget trainer-live block on the page. */
function pageConfig() {
  const page = readFileSync(path.join(ROOT, 'content/train-in-browser.md'), 'utf8');
  const m = page.match(/```widget trainer-live[^\n]*\n(\{[^\n]*\})\n```/);
  assert.ok(m, 'trainer-live widget not found on the page');
  return JSON.parse(m[1]);
}

const MIME = { '.js': 'text/javascript', '.txt': 'text/plain; charset=utf-8', '.json': 'application/json' };
function serve() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (url === '/') { res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); res.end('<!doctype html><title>trainer</title>'); return; }
    const file = path.join(WEB, url);
    if (!file.startsWith(WEB) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
    createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

test(`headless Chromium: the first ${STEPS} steps of the live trainer equal the recorded run`, async (t) => {
  let chromium;
  try {
    ({ chromium } = await import('playwright'));
  } catch {
    t.skip('playwright is not installed');
    return;
  }
  const cfg = pageConfig();
  // "不改设置" means: the page's defaults are the recorded settings.
  for (const key of ['batch', 'evalEvery', 'sampleEvery', 'sampleChars', 'prompt', 'temperature', 'seed']) assert.equal(cfg[key], SETTINGS[key], key);
  assert.equal(cfg.lr, rec.runs.default.lr);
  // The same settings object trainer-live.js builds (its defaults, then the page's config), stopped after STEPS.
  const settings = {
    batch: cfg.batch ?? 8, lr: cfg.lr ?? 3e-3, maxSteps: STEPS, evalEvery: cfg.evalEvery ?? 25, sampleEvery: cfg.sampleEvery ?? 100,
    sampleChars: cfg.sampleChars ?? 120, prompt: cfg.prompt ?? 'def ', temperature: cfg.temperature ?? 0.8, mode: 'normal',
  };
  const model = { ctx: 64, embed: 64, heads: 4, layers: 2, ff: 192, ...(cfg.config || {}) };

  const server = await serve();
  let browser;
  try {
    browser = await chromium.launch();
  } catch (err) {
    server.close();
    t.skip(`Chromium is not available (${err.message.split('\n')[0]})`);
    return;
  }
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const out = await page.evaluate(({ model, settings, seed }) => new Promise((resolve) => {
      const w = new Worker('/widgets/trainer-worker.js', { type: 'module' });
      const got = { losses: [], evals: [], samples: [], ready: null, error: null, ua: navigator.userAgent };
      w.onerror = (e) => { got.error = e.message || 'worker error'; resolve(got); };
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') {
          got.ready = { simd: m.simd, initialVal: m.initialVal, baselines: m.baselines };
          w.postMessage({ type: 'sample' }); // trainer-live.js shows a sample of the untrained model
          w.postMessage({ type: 'start' });
        } else if (m.type === 'progress') {
          got.losses.push(...m.losses); got.evals.push(...m.evals);
        } else if (m.type === 'sample') {
          got.samples.push({ step: m.step, text: m.text });
        } else if (m.type === 'state' && m.done) {
          resolve(got);
        } else if (m.type === 'error') {
          got.error = m.message; resolve(got);
        }
      };
      w.postMessage({ type: 'init', trainURL: '/data/train-in-browser/tinycodes-train.txt',
        valURL: '/data/train-in-browser/tinycodes-val.txt', config: model, seed, settings });
    }), { model, settings, seed: cfg.seed ?? 0 });

    assert.equal(out.error, null);
    assert.match(out.ua, /HeadlessChrome/);
    assert.equal(out.ready.simd, true);
    assert.deepEqual(out.ready.baselines, rec.baselines);
    const run = rec.runs.default;
    const losses = out.losses.map(([, l]) => l);
    assert.deepEqual(out.losses.map(([s]) => s), [...Array(STEPS).keys()]);
    assert.deepEqual(losses.slice(0, 20), run.first); // bit for bit
    assert.deepEqual(losses.map(r4), run.losses.slice(0, STEPS));
    assert.deepEqual([[0, r4(out.ready.initialVal)], ...out.evals.map(([k, v]) => [k, r4(v)])], run.held.filter(([k]) => k <= STEPS));
    assert.deepEqual(out.samples, run.samples.filter((s) => s.step <= STEPS));
  } finally {
    await browser.close();
    server.close();
  }
});
