// 第 8 章交互图里的计算：与正文验证单元的结论逐条对拍（node --test tests/js/ch08-widgets.test.mjs）。
import test from 'node:test';
import assert from 'node:assert/strict';

import { lossFit, nOpt, chinchilla20, h100Hours, human, PRESETS as SCALE } from '../../web/widgets/ch08-scaling.js';
import { gaussianMatrix, naiveAttention, flashSteps, maxAbsDiff } from '../../web/widgets/ch08-flash.js';
import { kvBytesPerToken, groupOf, divisors, formatBytes, PRESETS as GQA } from '../../web/widgets/ch08-gqa.js';
import { softmax, topk, route, auxLoss, routerLogits, TOKENS, E } from '../../web/widgets/ch08-moe.js';
import { turns, slowPairs, ntkBase } from '../../web/widgets/ch08-ropeext.js';

const close = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b} (tol ${tol})`);

test('scaling: 闭式解 N_opt 就是等算力曲线的最低点（验证 8.1-e）', () => {
  const { N, D } = SCALE.webbot;
  const C = 6 * N * D;
  let best = Infinity, bestN = 0;
  for (let i = 0; i <= 3000; i++) {
    const n = 10 ** (7 + (3 * i) / 3000);
    const l = lossFit(n, C / (6 * n));
    if (l < best) { best = l; bestN = n; }
  }
  close(bestN / nOpt(C), 1, 0.002);
  close(nOpt(C) / 1e6, 288, 1);
  close(N / nOpt(C), 0.53, 0.005);
  const gap = lossFit(N, D) - lossFit(nOpt(C), C / (6 * nOpt(C)));
  close(gap, 0.028, 0.001);
  for (const c of [1e18, 1e20, 1e22, 1e24]) {
    const ratio = c / 6 / nOpt(c) ** 2;
    assert.ok(ratio > 17 && ratio < 26, `D/N = ${ratio} at C = ${c}`);
  }
});

test('scaling: 20 倍规则、D/N 与 H100 估算（验证 8.1-d、8.1-f）', () => {
  const { N, D } = SCALE.webbot;
  assert.equal(D, 13_107_200_000);
  assert.equal(Math.round(D / N), 86);
  close(chinchilla20(6 * N * D).N / 1e6, 316, 1);
  close(h100Hours(6 * N * D), 9.6, 0.05);
  close(h100Hours(6 * N * 20 * N), 2.23, 0.01);
  assert.equal(SCALE.chinchilla.D / SCALE.chinchilla.N, 20);
  assert.equal(human(152_316_672), '152M');
  assert.equal(human(1.4e12), '1.4T');
  assert.equal(human(13_107_200_000), '13.1B');
});

test('flash: 分块 + online softmax 与朴素 attention 相同（验证 8.2-c）', () => {
  const N = 8, d = 4;
  const Q = gaussianMatrix(N, d, 7), K = gaussianMatrix(N, d, 8), V = gaussianMatrix(N, d, 9);
  for (const causal of [true, false]) {
    const ref = naiveAttention(Q, K, V, causal);
    for (const block of [1, 2, 4]) {
      const run = flashSteps(Q, K, V, { block, causal });
      assert.ok(maxAbsDiff(run.out, ref) < 1e-12, `block ${block}, causal ${causal}`);
      const tiles = (N / block) ** 2;
      const nb = N / block;
      assert.equal(run.steps.length, causal ? (nb * (nb + 1)) / 2 : tiles);
      assert.equal(run.skipped.length, causal ? tiles - (nb * (nb + 1)) / 2 : 0);
    }
  }
  // 每行最后的 ℓ 与一次性计算的分母相同
  const run = flashSteps(Q, K, V, { block: 2, causal: true });
  const last = run.steps[run.steps.length - 1];
  for (let r = 0; r < N; r++) {
    const s = K.map((k, c) => (c > r ? -Infinity : k.reduce((a, x, j) => a + x * Q[r][j], 0) / 2));
    const m = Math.max(...s);
    close(last.m[r], m, 1e-12);
  }
});

test('gqa: KV cache 字节数与分组约定（验证 8.3-a、8.3-b）', () => {
  assert.equal(kvBytesPerToken(GQA.webbot), 12 * 1024);
  assert.equal(kvBytesPerToken(GQA.webbot_mha), 36 * 1024);
  assert.equal(kvBytesPerToken(GQA.webbot_mqa), 3 * 1024);
  assert.equal(kvBytesPerToken(GQA.mixtral) * GQA.mixtral.T, 4 * 2 ** 30);
  assert.deepEqual([0, 1, 2, 3, 4, 5].map((h) => groupOf(h, 6, 2, 'interleave')), [0, 0, 0, 1, 1, 1]);
  assert.deepEqual([0, 1, 2, 3, 4, 5].map((h) => groupOf(h, 6, 2, 'tile')), [0, 1, 0, 1, 0, 1]);
  assert.deepEqual(Array.from({ length: 12 }, (_, h) => groupOf(h, 12, 4)), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
  assert.deepEqual(divisors(12), [1, 2, 3, 4, 6, 12]);
  assert.equal(formatBytes(12 * 1024), '12 KiB');
  assert.equal(formatBytes(4 * 2 ** 30), '4 GiB');
});

test('moe: top-k softmax、分派与辅助 loss（验证 8.4-a、8.4-c、8.4-d）', () => {
  const z = [2.0, 1.0, 0.5, -1.0];
  const top = topk(z, 2);
  assert.deepEqual(top, [0, 1]);
  const g1 = softmax(top.map((e) => z[e]));
  const p = softmax(z);
  const g2 = top.map((e) => p[e] / (p[0] + p[1]));
  g1.forEach((x, i) => close(x, g2[i], 1e-12));
  close(g1[0], 0.7311, 1e-4);

  const logits = routerLogits(0);
  assert.equal(logits.length, TOKENS.length);
  for (const k of [1, 2]) {
    const r = route(logits, { k, capacityFactor: 100 });           // 容量足够大：一个也不丢
    assert.equal(r.dropped, 0);
    assert.ok(r.choices.every((c) => c.length === k && new Set(c).size === k));
    assert.equal(r.assigned.reduce((a, b) => a + b, 0), TOKENS.length * k);
    r.gates.forEach((g) => close(g.reduce((a, b) => a + b, 0), 1, 1e-12));
  }
  assert.equal(route(logits, { k: 1, capacityFactor: 1 }).capacity, TOKENS.length / E);   // Switch 式 (3)

  // 辅助 loss：均匀为 1，偏向一个专家时趋近 E
  close(auxLoss(TOKENS.map(() => [0, 0, 0, 0])).value, 1, 1e-12);
  close(auxLoss(TOKENS.map(() => [50, 0, 0, 0])).value, E, 1e-12);
  const a0 = auxLoss(routerLogits(0)).value, a4 = auxLoss(routerLogits(4)).value;
  assert.ok(a0 < 2 && a4 > 3 && a4 < E, `${a0} ${a4}`);
  assert.ok(route(routerLogits(4), { k: 2, capacityFactor: 1.25 }).dropped > route(routerLogits(0), { k: 2, capacityFactor: 1.25 }).dropped);
});

test('ropeext: 转不满一圈的维度与 NTK-aware 的两端（验证 8.6-b、8.6-c）', () => {
  const w = slowPairs(64, 10000, 1024);
  assert.equal(w.length, 14);
  assert.equal(w[0], 18);
  const s = slowPairs(32, 10000, 256);
  assert.equal(s.length, 9);
  assert.equal(s[0], 7);
  close(ntkBase(10000, 64, 2), 20452.23, 0.01);
  const cfg = { D: 64, theta: 10000, L: 1024, s: 2 };
  const train = turns(cfg, 'train'), extra = turns(cfg, 'extrapolate'), pi = turns(cfg, 'pi'), ntk = turns(cfg, 'ntk');
  close(ntk[0], extra[0], 1e-12);                                   // 最高频不变
  close(ntk[31] / train[31], (2 * 1024 - 1) / (2 * 1023), 1e-9);    // 最低频放慢 s 倍，转过的角度与训练时几乎相同
  pi.forEach((x, i) => assert.ok(x <= train[i] * (1023.5 / 1023) + 1e-12));
  assert.ok(extra[31] > train[31] * 1.9);
});
