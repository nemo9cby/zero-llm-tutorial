// The widget's cost model (web/widgets/gpu-costmodel.js) must agree with gpu/costs.py.
// Expected values come from tests/js/fixtures/gpu/cases.json (regenerate: uv run python verify/_gpu_fixtures.py).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as cm from '../../web/widgets/gpu-costmodel.js';

const read = (rel) => JSON.parse(readFileSync(new URL(rel, import.meta.url), 'utf8'));
const fx = read('./fixtures/gpu/cases.json');
const { bots } = read('../../web/data/gpu/bots.json');
const gpus = read('../../web/data/gpu/gpus.json');
const byId = (list, id) => list.find((x) => x.id === id);

function close(actual, expected, what) {
  if (typeof expected === 'boolean' || Number.isInteger(expected)) {
    assert.equal(actual, expected, what);
  } else {
    const tol = 1e-12 * Math.max(1, Math.abs(expected));
    assert.ok(Math.abs(actual - expected) <= tol, `${what}: ${actual} vs ${expected}`);
  }
}

test('parameter counts and n_mm match the Python model', () => {
  for (const [name, [params, nmm]] of Object.entries(fx.params)) {
    assert.deepEqual(cm.paramCounts(bots[name]), [params, nmm], name);
    assert.equal(params, bots[name].params, `${name} params in bots.json`);
  }
});

test('evaluation counts follow `i % every == 0 or i == max_iters - 1`', () => {
  for (const [it, every, steps, expected] of fx.n_evals) {
    assert.equal(cm.nEvals(it, every, steps), expected, `${it}/${every}/${steps}`);
  }
});

test('every scenario x offer estimate matches gpu/costs.py', () => {
  assert.ok(fx.cases.length >= 30, 'fixture covers the presets');
  for (const c of fx.cases) {
    const b = bots[c.bot];
    const offer = byId(gpus.offers, c.offer);
    const gpu = byId(gpus.gpus, offer.gpu);
    const a = c.args;
    const shape = cm.runShape(b, { world: a.world, maxIters: a.max_iters, batch: a.batch, accum: a.accum, steps: a.steps });
    for (const [k, v] of Object.entries(c.shape)) close(shape[k], v, `${c.scenario}/${c.offer} shape.${k}`);
    const est = cm.estimate(b, gpu, cm.pricePerGpuHour(offer), shape, c.mfu, c.floor_ms);
    for (const [k, v] of Object.entries(c.estimate)) close(est[k], v, `${c.scenario}/${c.offer}/${c.floor_ms} ${k}`);
  }
});

test('the WebBot probe stops after 200 steps with one evaluation', () => {
  const shape = cm.runShape(bots.webbot, { steps: 200 });
  for (const [k, v] of Object.entries(fx.probe_shape)) close(shape[k], v, `probe ${k}`);
  assert.equal(shape.evals, 1);
});

test('probe-based ETA matches', () => {
  const r = cm.estimateMeasured(fx.measured.args);
  for (const [k, v] of Object.entries(fx.measured.expected)) close(r[k], v, `measured ${k}`);
});

test('multi-GPU comparison rows follow gpu/jobs.py', () => {
  const wb = bots.webbot;
  assert.equal(fx.multi_gpu.length, 6);
  for (const { n, scale, settings } of fx.multi_gpu) {
    const row = cm.multiGpuVariants(wb, n).find((r) => r.key === scale);
    assert.ok(row, `${n} GPUs, ${scale}`);
    const shape = cm.runShape(wb, row.args);
    assert.equal(shape.world, n);
    assert.equal(shape.max_iters, settings.max_iters, `${n}/${scale} max_iters`);
    assert.equal(shape.batch, settings.micro_batch_size, `${n}/${scale} micro-batch`);
    assert.equal(shape.accum, settings.accumulation_steps, `${n}/${scale} accumulation`);
    // either way the n GPUs see the single-GPU token budget
    assert.equal(shape.train_tokens, cm.runShape(wb).train_tokens, `${n}/${scale} tokens`);
  }
});

test('every multi-GPU preset is one row of its own comparison table', () => {
  // The rows come from the single-GPU job, so a preset that already divided max_iters is not divided twice.
  const { scenarios } = read('../../web/data/gpu/bots.json');
  const multi = scenarios.filter((s) => s.world > 1);
  assert.equal(multi.length, 3);
  for (const s of multi) {
    const b = bots[s.bot];
    const headline = cm.runShape(b, { world: s.world, maxIters: s.max_iters, batch: s.batch, accum: s.accum });
    const key = s.id.split('-').pop();                     // webbot-x8-trap | -iters | -batch
    const row = cm.multiGpuVariants(b, s.world).find((r) => r.key === key);
    assert.deepEqual(cm.runShape(b, row.args), headline, s.id);
    assert.ok(s.label.endsWith(row.label), `${s.label} / ${row.label}`);
  }
});

test('a GPU without BF16 refuses a BF16 bot', () => {
  const t4 = byId(gpus.gpus, 't4');
  const shape = cm.runShape(bots.storybot);
  assert.throws(() => cm.estimate(bots.storybot, t4, 0.59, shape, 0.25), /no published bf16/);
});
