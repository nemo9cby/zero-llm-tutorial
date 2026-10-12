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

test('a GPU without BF16 refuses a BF16 bot', () => {
  const t4 = byId(gpus.gpus, 't4');
  const shape = cm.runShape(bots.storybot);
  assert.throws(() => cm.estimate(bots.storybot, t4, 0.59, shape, 0.25), /no published bf16/);
});
