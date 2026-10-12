// 第 9 章交互图里的计算：与正文结论一致（node --test tests/js/ch09-widgets.test.mjs）
import test from 'node:test';
import assert from 'node:assert/strict';

import { makeData, batchGrad, accumulate } from '../../web/widgets/ch09-accum.js';
import { initialGrads, ringTrace } from '../../web/widgets/ch09-allreduce.js';
import { ACCOUNTING, zeroBytes, COMM } from '../../web/widgets/ch09-zero.js';
import { COEFFS, makeMatrix, nsMatrix, singularValues, nsScalar, symEig, polyText } from '../../web/widgets/ch09-muon.js';
import { parseCalls, compactNewlines } from '../../web/widgets/ch09-cpurun.js';

const close = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

test('梯度累积：每份除以 K 时等于大 batch；不除时是 K 倍', () => {
  const { xs, ts } = makeData(16);
  const big = batchGrad(xs, ts, 0.5);
  for (const K of [1, 2, 4, 8, 16]) {
    assert.ok(close(accumulate(xs, ts, 0.5, K, true)[K], big, 1e-12), `K=${K}`);
    assert.ok(close(accumulate(xs, ts, 0.5, K, false)[K], K * big, 1e-12), `K=${K} 忘了除以 K`);
  }
});

test('Ring all-reduce：每张卡最后都是总和；每张卡发出 2(W−1) 块', () => {
  for (const W of [2, 3, 4, 6, 8]) {
    const g = initialGrads(W);
    const states = ringTrace(g);
    assert.equal(states.length - 1, 2 * (W - 1));
    const sums = g[0].map((_, c) => g.reduce((s, row) => s + row[c], 0));
    const end = states[states.length - 1];
    for (let r = 0; r < W; r++) {
      assert.deepEqual(end.vals[r], sums, `W=${W} GPU ${r}`);
      assert.ok(end.cnt[r].every((n) => n === W));
    }
    // reduce-scatter 结束时，GPU r 手里的第 (r+1) 块已经是总和
    const mid = states[W - 1];
    for (let r = 0; r < W; r++) assert.equal(mid.cnt[r][(r + 1) % W], W);
    const sent = states.slice(1).reduce((s, st) => s + st.moves.filter((m) => m.from === 0).length, 0);
    assert.equal(sent, 2 * (W - 1));
  }
});

test('ZeRO：论文图 1 的 7.5B、64 卡 → 120 / 31.4 / 16.6 / 1.9 GB', () => {
  const gbs = [0, 1, 2, 3].map((s) => {
    const p = zeroBytes(7.5e9, 64, s, ACCOUNTING.zero);
    return (p.params + p.grads + p.optim) / 1e9;
  });
  assert.deepEqual(gbs.map((v) => Math.round(v * 10) / 10), [120, 31.4, 16.6, 1.9]);
  assert.deepEqual(COMM, [1, 1, 1, 1.5]);
  for (const a of Object.values(ACCOUNTING)) assert.equal(a.p + a.g + a.k, 16);
});

test('Jacobi 求出的特征值正确', () => {
  const ev = symEig([[2, 1, 0], [1, 2, 0], [0, 0, 5]]).sort((p, q) => p - q);
  [1, 3, 5].forEach((v, i) => assert.ok(close(ev[i], v, 1e-12)));
});

test('Newton–Schulz：矩阵迭代的奇异值 = 标量多项式作用在奇异值上', () => {
  for (const coef of Object.values(COEFFS)) {
    for (const kappa of [1, 10, 100]) {
      const { G, sigma } = makeMatrix(kappa);
      assert.deepEqual(singularValues(G).map((v) => +v.toFixed(9)), sigma.map((v) => +v.toFixed(9)));
      const norm = Math.hypot(...sigma) + 1e-7;           // 与 nsMatrix 一样，范数加 eps
      const pred = sigma.map((s) => nsScalar(s / norm, coef.abc, 5)).sort((p, q) => q - p);
      const got = singularValues(nsMatrix(G, coef.abc, 5));
      got.forEach((v, i) => assert.ok(close(v, pred[i], 1e-9), `${coef.label} κ=${kappa}`));
    }
  }
});

test('Muon 的五次迭代：5 步把 [0.01, 1] 里的奇异值都送进 [0.68, 1.14]', () => {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i <= 400; i++) {
    const y = nsScalar(10 ** (-2 + (2 * i) / 400), COEFFS.quintic.abc, 5);
    lo = Math.min(lo, y); hi = Math.max(hi, y);
  }
  assert.ok(lo > 0.68 && hi < 1.14, `${lo} ${hi}`);
  assert.ok(nsScalar(0.01, COEFFS.cubic.abc, 5) < 0.1);       // 经典三次迭代 5 步时还远没到 1
});

test('运行记录解析：config、train、val 分开', () => {
  const text = [
    '{"call": "init", "project": "webbot-pretrain", "config": {}}',
    '{"call": "config.update", "data": {"vocab_size": 50000, "effective_batch_size": 32}}',
    '{"call": "log", "step": 0, "data": {"train/loss": 10.8, "train/lr": 0.0}}',
    '{"call": "log", "step": 0, "data": {"val/loss": 10.83}}',
    '{"call": "finish"}',
  ].join('\n');
  const r = parseCalls(text);
  assert.equal(r.cfg.effective_batch_size, 32);
  assert.equal(r.train.length, 1);
  assert.equal(r.val[0].data['val/loss'], 10.83);
});

test('续写的换行压缩：只改换行，其余文字不变', () => {
  assert.equal(compactNewlines('a\nb\n\n\nc'), 'a ↵ b ↵×3 c');
  const raw = 'The capital of Japan is to the.\n\nThe internet,';
  assert.equal(compactNewlines(raw).replace(/ ↵(×\d+)? /g, ''), raw.replace(/\n/g, ''));
});

test('多项式的文字形式：系数为 0 的项不写', () => {
  assert.equal(polyText([3.4445, -4.775, 2.0315]), '3.4445x − 4.775x³ + 2.0315x⁵');
  assert.equal(polyText([1.5, -0.5, 0]), '1.5x − 0.5x³');
});
