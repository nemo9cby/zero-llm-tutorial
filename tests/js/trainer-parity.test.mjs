// Parity of web/widgets/_tinygpt.js with DLFS6's own codebot.model.GPT run in PyTorch (float64).
// Fixtures: tests/js/fixtures/trainer/parity-*.json, written by `uv run python verify/_trainer_export.py`.
//   node --test tests/js/trainer-parity.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createModel, makeWorkspace, forward, backward, zeroGrad, AdamW, paramCount } from '../../web/widgets/_tinygpt.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/trainer/${name}`, import.meta.url), 'utf8'));
const unb64 = (s) => { const b = Buffer.from(s, 'base64'); return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
const jsConfig = (c) => ({ vocabSize: c.vocab_size, ctx: c.max_context_len, embed: c.embed_dim, heads: c.n_head, layers: c.n_layer, ff: c.ff_dim });
const relErr = (a, b) => {
  let d = 0, n = 0;
  for (let i = 0; i < a.length; i++) { d += (a[i] - b[i]) ** 2; n += b[i] ** 2; }
  return Math.sqrt(d / Math.max(n, 1e-300));
};
const batch = (rows) => {
  const T = rows[0].length - 1;
  return { x: Int32Array.from(rows.flatMap((r) => r.slice(0, T))), y: Int32Array.from(rows.flatMap((r) => r.slice(1))), B: rows.length, T };
};

function loadTiny(dtype) {
  const fx = fixture('parity-tiny.json');
  const m = createModel(jsConfig(fx.config), { dtype });
  assert.deepEqual(m.params.map((p) => p.name), fx.names, 'same parameters, same order as named_parameters()');
  assert.equal(fx.names.length, 29);
  for (const p of m.params) {
    assert.deepEqual(p.shape, fx.shapes[p.name], p.name);
    p.data.set(unb64(fx.weights[p.name]));
  }
  return { fx, m };
}

test('float64: loss and every gradient equal DLFS6 GPT + backward() (rel. error < 1e-12)', () => {
  const { fx, m } = loadTiny(Float64Array);
  const { x, y, B, T } = batch(fx.ids[0]);
  const ws = makeWorkspace(m, B, T);
  zeroGrad(m);
  const loss = forward(m, ws, x, y, B, T);
  backward(m, ws);
  assert.ok(Math.abs(loss - fx.loss) < 1e-12, `loss ${loss} vs torch ${fx.loss}`);
  for (const p of m.params) {
    const e = relErr(p.grad, unb64(fx.grads[p.name]));
    assert.ok(e < 1e-12, `${p.name}: ${e}`);
  }
});

test('float32 (what the browser trains with) stays within 1e-5 of the float64 reference', () => {
  const { fx, m } = loadTiny(Float32Array);
  const { x, y, B, T } = batch(fx.ids[0]);
  const ws = makeWorkspace(m, B, T);
  zeroGrad(m);
  const loss = forward(m, ws, x, y, B, T);
  backward(m, ws);
  assert.ok(Math.abs(loss - fx.loss) / fx.loss < 1e-6, `loss ${loss} vs ${fx.loss}`);
  for (const p of m.params) {
    const e = relErr(p.grad, unb64(fx.grads[p.name]));
    assert.ok(e < 1e-5, `${p.name}: ${e}`);
  }
});

test('three AdamW steps equal torch.optim.AdamW (losses and parameters)', () => {
  const { fx, m } = loadTiny(Float64Array);
  const { lr, betas, eps, weight_decay: weightDecay, losses, params_after: after } = fx.adamw;
  const opt = new AdamW(m.params, { lr, betas, eps, weightDecay });
  const B = fx.B, T = fx.T, ws = makeWorkspace(m, B, T);
  for (let s = 1; s <= 3; s++) {
    const { x, y } = batch(fx.ids[s]);
    zeroGrad(m);
    const loss = forward(m, ws, x, y, B, T);
    backward(m, ws);
    opt.step();
    assert.ok(Math.abs(loss - losses[s - 1]) < 1e-11, `step ${s}: ${loss} vs ${losses[s - 1]}`);
  }
  for (const p of m.params) {
    const e = relErr(p.data, unb64(after[p.name]));
    assert.ok(e < 1e-11, `${p.name}: ${e}`);
  }
});

test('the trainer\'s real size: parameter count, loss and gradient checksums equal DLFS6 GPT', () => {
  const fx = fixture('parity-default.json');
  const cfg = jsConfig(fx.config);
  const m = createModel(cfg, { dtype: Float64Array });
  assert.equal(paramCount(cfg), fx.param_count);
  assert.equal(m.params.reduce((s, p) => s + p.data.length, 0), fx.param_count);
  assert.deepEqual(m.params.map((p) => p.name), fx.names);
  // Same formula as formula_weights() in verify/_trainer_export.py.
  m.params.forEach((p, k) => {
    const ln = /norm[12]?\.weight$/.test(p.name);
    for (let i = 0; i < p.data.length; i++) p.data[i] = (ln ? 1 : 0) + 0.05 * Math.sin(1.7 * i + 0.3 * k + 0.1);
  });
  const { x, y, B, T } = batch(fx.ids);
  const ws = makeWorkspace(m, B, T);
  zeroGrad(m);
  const loss = forward(m, ws, x, y, B, T);
  backward(m, ws);
  assert.ok(Math.abs(loss - fx.loss) < 1e-11, `loss ${loss} vs ${fx.loss}`);
  for (const p of m.params) {
    const c = fx.grad_checks[p.name];
    let sum = 0, sumsq = 0;
    for (const g of p.grad) { sum += g; sumsq += g * g; }
    assert.ok(Math.abs(sumsq - c.sumsq) <= 1e-10 * c.sumsq + 1e-300, `${p.name} sumsq ${sumsq} vs ${c.sumsq}`);
    assert.ok(Math.abs(sum - c.sum) <= 1e-10 * Math.sqrt(c.sumsq * p.grad.length) + 1e-300, `${p.name} sum`);
    c.head.forEach((h, i) => assert.ok(Math.abs(p.grad[i] - h) <= 1e-10 * Math.sqrt(c.sumsq) + 1e-300, `${p.name}[${i}]`));
  }
});
