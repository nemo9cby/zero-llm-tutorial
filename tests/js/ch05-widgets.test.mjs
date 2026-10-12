// 第 5 章交互图里的计算，与正文验证单元的结论对拍：node --test tests/js/ch05-widgets.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { ropeDot2D } from '../../web/widgets/ch05-rope.js';
import { ropeVec, scoreMatrix, invFreq } from '../../web/widgets/ch05-ropeclock.js';
import { swish, gelu, argmin } from '../../web/widgets/ch05-act.js';
import { layerNorm, rmsNorm, rms, mean, FINFO_EPS } from '../../web/widgets/ch05-norm.js';
import { stepCells, stepCounts } from '../../web/widgets/ch05-kv.js';
import { kvBytesPerToken, PRESETS } from '../../web/widgets/ch05-kvmem.js';
import { requestSize, cumulative, prefillWithPrefixCache, firstOverflow } from '../../web/widgets/ch05-tokens.js';
import { ratios } from '../../web/widgets/ch05-speed.js';
import { rng, formatBytes } from '../../web/widgets/ch05-common.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test('2D RoPE: q·k depends only on n − m (验证 5.1-a)', () => {
  close(ropeDot2D(1, 3, 30), 0.5);
  close(ropeDot2D(6, 8, 30), 0.5);
  close(ropeDot2D(10, 12, 30), 0.5);
  close(ropeDot2D(1, 4, 30), 0);                  // 夹角 90°
});

test('RoPE score matrix is Toeplitz; additive absolute positions are not (验证 5.1-c)', () => {
  const r = rng(7), d = 16, T = 10;
  const q = Array.from({ length: d }, () => r.normal()), k = Array.from({ length: d }, () => r.normal());
  const P = Array.from({ length: T }, () => Array.from({ length: d }, () => r.normal()));
  const S = scoreMatrix(q, k, T, 'rope');
  for (let i = 0; i + 1 < T; i++) for (let j = 0; j + 1 < T; j++) close(S[i + 1][j + 1], S[i][j], 1e-9);
  const A = scoreMatrix(q, k, T, 'abs', 10000, P);
  const diag = Array.from({ length: T - 3 }, (_, i) => A[i + 3][i]);
  assert.ok(Math.max(...diag) - Math.min(...diag) > 1);
  const norm = (v) => Math.hypot(...v);
  close(norm(ropeVec(q, 37)), norm(q), 1e-9);      // 旋转不改变长度
  close(invFreq(64, 10000)[1], 10000 ** (-2 / 64), 1e-15);
  close(2 * Math.PI / invFreq(64, 10000)[31], 47117.24, 0.01);
});

test('Swish minimum is x* + 1 at x* ≈ −1.2785 (验证 5.2-b)', () => {
  const [x, v] = argmin(swish, -3, 0);
  close(x, -1.2785, 1e-4);
  close(v, -0.2785, 1e-4);
  close(v, x + 1, 1e-7);                           // x* 只精确到 1e-8 左右
  const [gx, gv] = argmin(gelu, -3, 0);            // tanh 近似式的 GELU
  assert.ok(gx > -0.8 && gx < -0.7 && gv > -0.18 && gv < -0.16);
});

test('LayerNorm vs RMSNorm hand example and invariances (验证 5.3-a, 5.3-b, 5.3-c)', () => {
  const x = [1, 2, 3, 4];
  layerNorm(x, 0).forEach((v, i) => close(v, [-1.3416, -0.4472, 0.4472, 1.3416][i], 1e-4));
  rmsNorm(x, 0).forEach((v, i) => close(v, [0.3651, 0.7303, 1.0954, 1.4606][i], 1e-4));
  const y = [1.2, -0.4, 2.0, 0.7, -1.1, 0.3, 1.6, 0.9];
  const shifted = y.map((v) => v + 5), scaled = y.map((v) => v * 10);
  layerNorm(shifted).forEach((v, i) => close(v, layerNorm(y)[i], 1e-5));          // LayerNorm 不怕平移
  assert.ok(Math.max(...rmsNorm(shifted).map((v, i) => Math.abs(v - rmsNorm(y)[i]))) > 0.5);   // RMSNorm 怕
  rmsNorm(scaled).forEach((v, i) => close(v, rmsNorm(y)[i], 1e-5));               // 两者都不怕缩放
  close(FINFO_EPS, 1.1920928955078125e-7, 0);
  const tiny = y.map((v) => v / rms(y) * 1e-3);                                     // 输入 RMS = 1e-3
  close(rms(rmsNorm(tiny, 1e-5)), 0.3015, 1e-4);
  close(rms(rmsNorm(tiny, FINFO_EPS)), 0.9452, 1e-4);
  close(mean(layerNorm(y)), 0, 1e-12);
});

test('KV cache animation counts (验证 5.5-a)', () => {
  assert.deepEqual(stepCounts('cache', 4, 0), { tokens: 4, scores: 10 });          // prefill：4 个 token，下三角 10 格
  assert.deepEqual(stepCounts('cache', 4, 0, true), { tokens: 4, scores: 16 });    // 不加 mask：16 格
  assert.deepEqual(stepCounts('cache', 1, 0, true), { tokens: 1, scores: 1 });     // prompt 只有 1 个 token：bug 不显形
  assert.deepEqual(stepCounts('cache', 4, 1), { tokens: 1, scores: 5 });
  assert.equal(stepCells('cache', 4, 0, true).filter((c) => c.kind === 'future').length, 6);
  // 从 1 个 token 出发生成 200 个：无 cache 处理 20,100 个 token，有 cache 处理 200 个
  let noCache = 0, cache = 0;
  for (let s = 0; s < 200; s++) { noCache += stepCounts('nocache', 1, s).tokens; cache += stepCounts('cache', 1, s).tokens; }
  assert.equal(noCache, 20100);
  assert.equal(cache, 200);
});

test('KV cache bytes per token and presets (验证 5.5-h)', () => {
  assert.equal(kvBytesPerToken(6, 6, 64, 4), 18432);
  assert.equal(kvBytesPerToken(4, 16, 32, 4), 16384);
  assert.equal(kvBytesPerToken(12, 4, 64, 2), 12288);
  assert.equal(kvBytesPerToken(12, 12, 64, 2), 36864);
  assert.equal(formatBytes(64 * 1024 * kvBytesPerToken(12, 4, 64, 2)), '768 MiB');
  assert.equal(formatBytes(256 * kvBytesPerToken(6, 6, 64, 4)), '4.50 MiB');
  // 预设里的参数量与第 5 章的公式一致（WebBot 的 K、V 投影只有 n_kv_head 个头）
  const improved = (V, E, L, F) => V * E + L * (4 * E * E + 3 * E * F + 2 * E) + E + E * V;
  const webbot = (V, E, L, F, KV, D) => V * E + L * (2 * E * E + 2 * E * KV * D + 3 * E * F + 2 * E) + E + E * V;
  assert.equal(PRESETS.demo.params, improved(1000, 384, 6, 1024));
  assert.equal(PRESETS.storybot.params, improved(10000, 512, 4, 1344));
  assert.equal(PRESETS.webbot.params, webbot(50000, 768, 12, 2048, 4, 64));
  assert.equal(PRESETS.webbotMHA.params, webbot(50000, 768, 12, 2048, 12, 64));
});

test('agent token meter (验证 5.5-k)', () => {
  assert.equal(requestSize(3000, 800, 300), 243000);
  assert.equal(cumulative(3000, 800, 300), 37020000);
  assert.equal(prefillWithPrefixCache(3000, 800, 300), 243000);
  assert.equal(firstOverflow(3000, 800, 256000, 300), Infinity);
  const t = firstOverflow(3000, 800, 256000, 600);
  assert.ok(requestSize(3000, 800, t - 1) <= 256000 && requestSize(3000, 800, t) > 256000);
  close(cumulative(3000, 800, 300) / cumulative(3000, 800, 150), 3.89, 0.01);
});

test('recorded CPU timing: measured per-step speedup stays far below t (验证 5.5-g)', () => {
  const rec = JSON.parse(fs.readFileSync(new URL('../../web/data/ch05/kv_timing_cpu.json', import.meta.url), 'utf8'));
  for (const r of ratios(rec.per_step)) if (r.t >= 25) assert.ok(r.measured < r.flops / 2, `t=${r.t}`);
});
