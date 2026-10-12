// The WebAssembly SIMD matmul in web/widgets/_tinygpt.js computes the same product as the JavaScript loop.
//   node --test tests/js/trainer-simd.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  matmulBT, setSimd, simdSupported, makeRng, createModel, initWeights, makeWorkspace, forward, backward, zeroGrad,
} from '../../web/widgets/_tinygpt.js';

const rel = (a, b) => {
  let d = 0, n = 0;
  for (let i = 0; i < a.length; i++) { d += (a[i] - b[i]) ** 2; n += b[i] ** 2; }
  return Math.sqrt(d / n);
};

test('this engine supports WebAssembly SIMD, and the switch works', () => {
  assert.equal(simdSupported, true); // Node 24 / V8 13 always has it; browsers without it fall back to JS
  assert.equal(setSimd(false), false);
  assert.equal(setSimd(true), true);
});

test('SIMD matmul equals the JavaScript matmul (float32 vs float64 accumulation: about 1e-7)', () => {
  const rng = makeRng(0);
  for (const [M, K, N] of [[512, 64, 64], [512, 64, 192], [512, 192, 64], [192, 512, 64], [512, 64, 96], [64, 16, 64], [64, 64, 16]]) {
    const A = Float32Array.from({ length: M * K }, () => rng.normal()), B = Float32Array.from({ length: N * K }, () => rng.normal());
    const fast = new Float32Array(M * N), slow = new Float32Array(M * N);
    setSimd(true); matmulBT(A, B, fast, M, K, N);
    setSimd(false); matmulBT(A, B, slow, M, K, N);
    setSimd(true);
    const e = rel(fast, slow);
    assert.ok(e > 0 && e < 1e-6, `${M}x${K}x${N}: ${e}`);
  }
});

test('shapes the kernel cannot take (M % 4, N % 2 or K % 4) fall back to the identical JavaScript loop', () => {
  const rng = makeRng(1);
  const [M, K, N] = [5, 7, 3];
  const A = Float32Array.from({ length: M * K }, () => rng.normal()), B = Float32Array.from({ length: N * K }, () => rng.normal());
  const a = new Float32Array(M * N), b = new Float32Array(M * N);
  setSimd(true); matmulBT(A, B, a, M, K, N);
  setSimd(false); matmulBT(A, B, b, M, K, N);
  setSimd(true);
  assert.deepEqual(a, b);
});

test('a full float32 training step agrees with and without SIMD to 1e-5', () => {
  const cfg = { vocabSize: 96, ctx: 64, embed: 64, heads: 4, layers: 2, ff: 192 };
  const run = (simd) => {
    setSimd(simd);
    const m = createModel(cfg), rng = makeRng(3);
    initWeights(m, rng);
    const B = 4, T = 64, ws = makeWorkspace(m, B, T);
    const x = Int32Array.from({ length: B * T }, () => rng.int(96)), y = Int32Array.from({ length: B * T }, () => rng.int(96));
    zeroGrad(m);
    const loss = forward(m, ws, x, y, B, T);
    backward(m, ws);
    setSimd(true);
    return { loss, grads: m.params.map((p) => Float32Array.from(p.grad)) };
  };
  const fast = run(true), slow = run(false);
  assert.ok(Math.abs(fast.loss - slow.loss) < 1e-5);
  fast.grads.forEach((g, i) => assert.ok(rel(g, slow.grads[i]) < 1e-5, `param ${i}: ${rel(g, slow.grads[i])}`));
});
