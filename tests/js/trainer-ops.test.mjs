// Gradient checks for every layer of web/widgets/_tinygpt.js (node --test tests/js/trainer-ops.test.mjs).
// Each backward pass is compared with central finite differences in float64, the "gradient check"
// of DLFS1 section 5.7.3 and DLFS3 step 10.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  erf, makeRng, CharTokenizer, EOT, embeddingForward, embeddingBackward, linearForward, linearBackward,
  layerNormForward, layerNormBackward, geluForward, geluBackward, attentionForward, attentionBackward,
  crossEntropyForward, crossEntropyBackward, createModel, initWeights, makeWorkspace, forward, backward,
  zeroGrad, paramCount, makeBatcher, generate,
} from '../../web/widgets/_tinygpt.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/trainer/${name}`, import.meta.url), 'utf8'));
const F = Float64Array;
const randn = (rng, n, s = 1) => F.from({ length: n }, () => s * rng.normal());

/** ||a - b|| / (||a|| + ||b||): a relative error that does not blow up on entries that are almost zero.
    Central differences with h = 1e-5 are good to about 1e-10 here (truncation ~h^2, rounding ~1e-16/h). */
function relErr(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += (a[i] - b[i]) ** 2; na += a[i] ** 2; nb += b[i] ** 2; }
  return Math.sqrt(d) / Math.max(1e-30, Math.sqrt(na) + Math.sqrt(nb));
}

/** Central differences of the scalar f() with respect to every entry of x (x is perturbed in place). */
function numericalGrad(f, x, h = 1e-5) {
  const g = new F(x.length);
  for (let i = 0; i < x.length; i++) {
    const old = x[i];
    x[i] = old + h; const fp = f();
    x[i] = old - h; const fm = f();
    x[i] = old;
    g[i] = (fp - fm) / (2 * h);
  }
  return g;
}

/** L = sum(y * R) for a fixed random R, so dL/dy = R. */
const dot = (y, R) => { let s = 0; for (let i = 0; i < R.length; i++) s += y[i] * R[i]; return s; };

/** Position of a float64 on the number line, counted in float64 steps (so b - a is "how many ulps apart"). */
const ordinal = (x) => {
  const i = new BigInt64Array(Float64Array.of(x).buffer)[0];
  return i < 0n ? -(i & 0x7fffffffffffffffn) : i;
};
const ulpsApart = (a, b) => { const d = ordinal(a) - ordinal(b); return Number(d < 0n ? -d : d); };
const unb64 = (s) => { const b = Buffer.from(s, 'base64'); return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };

test('erf is at most one ulp from the correctly rounded value (mpmath, 120 bits), on a grid and 10,000 random points in [-7, 7]', () => {
  const fx = fixture('erf.json');
  const x = unb64(fx.x), ref = unb64(fx.erf);
  assert.equal(x.length, fx.n);
  assert.ok(fx.n > 10_000);
  let worst = 0, at = 0, exact = 0;
  for (let i = 0; i < x.length; i++) {
    const d = ulpsApart(erf(x[i]), ref[i]);
    if (d > worst) { worst = d; at = x[i]; }
    if (d === 0) exact += 1;
  }
  assert.ok(worst <= 1, `erf(${at}) is ${worst} ulps from the reference`);
  assert.ok(exact / x.length > 0.9, `correctly rounded at ${exact} of ${x.length} points`);
  assert.equal(ulpsApart(1, 1 - 2 ** -53), 1); // the ruler itself: 1 and the float64 just below it are one step apart
  assert.equal(erf(0), 0);
  assert.equal(erf(-7), -1);
});

test('linear layer: dx, dW, db match finite differences', () => {
  const rng = makeRng(1), N = 5, nIn = 7, nOut = 6;
  const x = randn(rng, N * nIn), W = randn(rng, nOut * nIn), b = randn(rng, nOut), R = randn(rng, N * nOut);
  const y = new F(N * nOut);
  const L = () => { linearForward(x, W, b, y, N, nIn, nOut); return dot(y, R); };
  const dx = new F(N * nIn), dW = new F(nOut * nIn), db = new F(nOut);
  L(); linearBackward(R, x, W, dx, dW, db, N, nIn, nOut);
  for (const [name, a, p] of [['dx', dx, x], ['dW', dW, W], ['db', db, b]]) {
    const e = relErr(a, numericalGrad(L, p));
    assert.ok(e < 1e-8, `${name}: ${e}`);
  }
});

test('LayerNorm: dx, dgamma, dbeta match finite differences', () => {
  const rng = makeRng(2), N = 4, E = 8;
  const x = randn(rng, N * E, 2), gam = randn(rng, E), bet = randn(rng, E), R = randn(rng, N * E);
  const y = new F(N * E), xhat = new F(N * E), rstd = new F(N);
  const L = () => { layerNormForward(x, gam, bet, y, xhat, rstd, N, E); return dot(y, R); };
  const dx = new F(N * E), dg = new F(E), db = new F(E);
  L(); layerNormBackward(R, xhat, rstd, gam, dx, dg, db, N, E);
  for (const [name, a, p] of [['dx', dx, x], ['dgamma', dg, gam], ['dbeta', db, bet]]) {
    const e = relErr(a, numericalGrad(L, p));
    assert.ok(e < 1e-8, `${name}: ${e}`);
  }
});

test('GELU (erf version): dx matches finite differences', () => {
  const rng = makeRng(3), n = 40;
  const x = randn(rng, n, 2.5), R = randn(rng, n), y = new F(n), cdf = new F(n), dx = new F(n);
  const L = () => { geluForward(x, y, cdf, n); return dot(y, R); };
  L(); geluBackward(R, x, cdf, dx, n);
  assert.ok(relErr(dx, numericalGrad(L, x)) < 1e-8);
  // GELU(1) = Phi(1) = 0.8413447460685429 (exact value: 0.5 * (1 + erf(1/sqrt 2)))
  geluForward(F.of(1), y, cdf, 1);
  assert.ok(Math.abs(y[0] - 0.8413447460685429) < 1e-15);
});

test('causal attention: dq, dk, dv match finite differences; masked weights are exactly zero', () => {
  const rng = makeRng(4), B = 2, T = 5, H = 2, D = 3, E = H * D, n = B * T * E;
  const q = randn(rng, n), k = randn(rng, n), v = randn(rng, n), R = randn(rng, n);
  const out = new F(n), P = new F(B * H * T * T);
  const L = () => { attentionForward(q, k, v, out, P, B, T, H, D); return dot(out, R); };
  const dq = new F(n), dk = new F(n), dv = new F(n);
  L(); attentionBackward(R, q, k, v, P, dq, dk, dv, B, T, H, D);
  for (const [name, a, p] of [['dq', dq, q], ['dk', dk, k], ['dv', dv, v]]) {
    const e = relErr(a, numericalGrad(L, p));
    assert.ok(e < 1e-8, `${name}: ${e}`);
  }
  for (let row = 0; row < B * H * T; row++) {
    const i = row % T;
    let s = 0;
    for (let j = 0; j < T; j++) { s += P[row * T + j]; if (j > i) assert.equal(P[row * T + j], 0); }
    assert.ok(Math.abs(s - 1) < 1e-12);
  }
});

test('softmax cross-entropy: gradient is (softmax - onehot) / N, and matches finite differences', () => {
  const rng = makeRng(5), N = 6, V = 9;
  const logits = randn(rng, N * V, 3), targets = Int32Array.from({ length: N }, () => rng.int(V));
  const L = () => crossEntropyForward(F.from(logits), targets, N, V);
  const probs = F.from(logits);
  crossEntropyForward(probs, targets, N, V);
  const d = new F(N * V);
  crossEntropyBackward(probs, targets, d, N, V);
  assert.ok(relErr(d, numericalGrad(L, logits)) < 1e-8);
  // Uniform logits give exactly ln V.
  assert.ok(Math.abs(crossEntropyForward(new F(N * V), targets, N, V) - Math.log(V)) < 1e-15);
});

test('embedding: repeated ids accumulate their gradients', () => {
  const rng = makeRng(6), B = 2, T = 3, E = 4, V = 5;
  const ids = Int32Array.of(1, 3, 1, 1, 0, 3);
  const tok = randn(rng, V * E), pos = randn(rng, T * E), R = randn(rng, B * T * E), x = new F(B * T * E);
  const L = () => { embeddingForward(ids, tok, pos, x, B, T, E); return dot(x, R); };
  const dTok = new F(V * E), dPos = new F(T * E);
  L(); embeddingBackward(ids, R, dTok, dPos, B, T, E);
  assert.ok(relErr(dTok, numericalGrad(L, tok)) < 1e-8);
  assert.ok(relErr(dPos, numericalGrad(L, pos)) < 1e-8);
  // id 1 appears three times, so its row is the sum of three rows of R; id 2 never appears.
  for (let e = 0; e < E; e++) {
    assert.ok(Math.abs(dTok[1 * E + e] - (R[0 * E + e] + R[2 * E + e] + R[3 * E + e])) < 1e-15);
    assert.equal(dTok[2 * E + e], 0);
  }
});

const TINY = { vocabSize: 11, ctx: 6, embed: 8, heads: 2, layers: 2, ff: 16 };

test('whole model: every parameter gradient matches finite differences (float64)', () => {
  const m = createModel(TINY, { dtype: F }), rng = makeRng(7);
  initWeights(m, rng, 0.3); // a larger std than 0.02 so that every path carries signal
  for (const p of m.params) if (p.init !== 'normal') for (let i = 0; i < p.data.length; i++) p.data[i] += 0.1 * rng.normal();
  const B = 2, T = 6, ws = makeWorkspace(m, B, T);
  const x = Int32Array.from({ length: B * T }, () => rng.int(TINY.vocabSize));
  const y = Int32Array.from({ length: B * T }, () => rng.int(TINY.vocabSize));
  zeroGrad(m);
  forward(m, ws, x, y, B, T);
  backward(m, ws);
  const L = () => forward(m, makeWorkspace(m, B, T), x, y, B, T);
  for (const p of m.params) {
    const e = relErr(p.grad, numericalGrad(L, p.data));
    assert.ok(e < 1e-8, `${p.name}: relative error ${e}`);
  }
  assert.equal(m.params.reduce((s, p) => s + p.data.length, 0), paramCount(TINY));
});

test('whole model is causal: changing token t never changes the logits before t', () => {
  const m = createModel(TINY, { dtype: F }), rng = makeRng(8);
  initWeights(m, rng, 0.3);
  const T = 6, ws = makeWorkspace(m, 1, T);
  const a = Int32Array.from({ length: T }, () => rng.int(TINY.vocabSize));
  forward(m, ws, a, null, 1, T);
  const la = Float64Array.from(ws.logits);
  const b = Int32Array.from(a); b[3] = (b[3] + 1) % TINY.vocabSize;
  forward(m, ws, b, null, 1, T);
  const V = TINY.vocabSize;
  for (let i = 0; i < 3 * V; i++) assert.equal(ws.logits[i], la[i]);
  assert.ok(ws.logits.slice(3 * V).some((v, i) => v !== la[3 * V + i]));
});

test('generation: temperature 0 picks the argmax of a full forward; it stops at <|endoftext|>', () => {
  const m = createModel(TINY, { dtype: F }), rng = makeRng(9);
  initWeights(m, rng, 0.5);
  const ws = makeWorkspace(m, 1, TINY.ctx), full = makeWorkspace(m, 1, TINY.ctx);
  const prompt = [1, 2, 3];
  const out = generate(m, ws, prompt, { maxNewTokens: 10, temperature: 0, rng });
  const ids = [...prompt];
  for (const next of out) {
    const w = Int32Array.from(ids.slice(-TINY.ctx));
    forward(m, full, w, null, 1, w.length);
    const row = full.logits.subarray((w.length - 1) * TINY.vocabSize, w.length * TINY.vocabSize);
    assert.equal(next, row.indexOf(Math.max(...row)));
    ids.push(next);
  }
  assert.equal(out.length, 10); // more than ctx = 6 tokens: the window slides
  const first = out[0];
  assert.deepEqual(generate(m, ws, prompt, { maxNewTokens: 10, temperature: 0, rng, endId: first }), []);
});

test('batcher visits every window start exactly once per epoch', () => {
  const tokens = Int32Array.from({ length: 40 }, (_, i) => i);
  const B = 3, T = 4, nStarts = tokens.length - T;
  const next = makeBatcher(tokens, B, T, makeRng(10));
  const seen = [];
  for (let s = 0; s < Math.ceil(nStarts / B) + 1; s++) {
    const { x, y } = next();
    for (let b = 0; b < B; b++) {
      seen.push(x[b * T]);
      for (let t = 0; t < T; t++) assert.equal(y[b * T + t], x[b * T + t] + 1); // targets are shifted by one
    }
  }
  assert.deepEqual([...new Set(seen.slice(0, nStarts))].sort((a, b) => a - b), [...Array(nStarts).keys()]);
});

test('character tokenizer: sorted characters, then <|endoftext|> last; decode(encode(s)) == s', () => {
  const tok = new CharTokenizer(`ba\n${EOT}\nab`);
  assert.deepEqual(tok.chars, ['\n', 'a', 'b']);
  assert.equal(tok.endId, 3);
  const s = `ab${EOT}ba\n`;
  assert.deepEqual(tok.encode(s), [1, 2, 3, 2, 1, 0]);
  assert.equal(tok.decode(tok.encode(s)), s);
  assert.throws(() => tok.encode('c'));
});
