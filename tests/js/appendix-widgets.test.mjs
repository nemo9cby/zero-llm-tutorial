// 附录 B–E 交互图里的计算：与对应验证单元（Python）打印的数值对拍。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeRng, softmax, histogram } from '../../web/widgets/appendix-common.js';
import { attentionRow, meanMaxWeight } from '../../web/widgets/appendix-scaling.js';
import { bandit, histRange, PRESETS } from '../../web/widgets/appendix-baseline.js';
import { emaPath, uncorrectedRatio, ratioSummary, ratioText } from '../../web/widgets/appendix-adam.js';
import { optimalPolicy, kl, sigmoid } from '../../web/widgets/appendix-dpo.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b}`);

test('appendix-common：随机数可复现，均值方差、类别频率、直方图都对', () => {
  const a = makeRng(42), b = makeRng(42);
  for (let i = 0; i < 5; i++) assert.equal(a.normal(), b.normal());
  const rng = makeRng(1);
  const xs = Array.from({ length: 200000 }, rng.normal);
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
  const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
  close(mean, 0, 0.01, 'mean');
  close(v, 1, 0.02, 'var');
  const p = [0.1, 0.2, 0.7];
  const counts = [0, 0, 0];
  for (let i = 0; i < 100000; i++) counts[rng.categorical(p)] += 1;
  counts.forEach((c, i) => close(c / 100000, p[i], 0.01, `p${i}`));
  const hist = histogram(xs, -5, 5, 50);
  close(hist.reduce((s, [, d]) => s + d * 0.2, 0), 1, 1e-3, 'histogram integral');
  close(softmax([1, 2, 3]).reduce((s, x) => s + x, 0), 1, 1e-12);
});

test('histogram：边界上的值不被浮点误差劈开；hi 本身计入，[lo, hi] 之外不计', () => {
  // 0.7 − 0.4 = 0.29999999999999993，只比边界 0.3 小一点浮点误差，应当与 0.3 落进同一个区间；0.6 = hi 计入最后一个区间
  const counts = histogram([0.7 - 0.4, 0.3, 0.6, 0.61, -0.01], 0, 0.6, 2).map(([, d]) => Math.round(d * 5 * 0.3));
  assert.deepEqual(counts, [0, 3]);
  // N = 8 时梯度估计只取格点值 k·0.09375；让格点正好是区间的左边界，再加上左右各差一个 ulp 的值
  const lattice = Array.from({ length: 8 }, (_, k) => (k * 3) / 32);
  const xs = lattice.flatMap((v) => [v * (1 - 2 ** -52), v, v * (1 + 2 ** -52)]);
  const perBin = histogram(xs, 0, 0.75, 8).map(([, d]) => Math.round(d * xs.length * 0.09375));
  assert.deepEqual(perBin, [3, 3, 3, 3, 3, 3, 3, 3]);   // 不加容差时是 [4, 3, 3, 3, 3, 3, 3, 2]
});

test('appendix-baseline：直方图的横轴以真梯度为中心，真梯度在中间那个区间的中点', () => {
  const [lo, hi] = histRange(0.1875, 0.2679);
  const w = (hi - lo) / 41;
  close((0.1875 - lo) / w, 20.5, 1e-9, 'center bin');
  close(hi - lo, 41 * (8 * 0.2679) / 40, 1e-12, 'width');
});

test('appendix-scaling：给定 q 时，一行分数的标准差就是 |q|（验证 B-f）', () => {
  for (const scaled of [false, true]) {
    const { qNorm, scores } = attentionRow(makeRng(5), 64, 20000, scaled);
    const m = scores.reduce((a, b) => a + b, 0) / scores.length;
    const sd = Math.sqrt(scores.reduce((a, b) => a + (b - m) ** 2, 0) / scores.length);
    const expected = scaled ? qNorm / 8 : qNorm;
    close(sd / expected, 1, 0.03, `scaled=${scaled}`);   // 2 万个分数，标准差的相对标准误约 0.5%
  }
});

test('appendix-scaling：不缩放时最大权重随 d 上升；缩放后 d ≥ 16 几乎不变、d = 1 略低（验证 B-g）', () => {
  // 32 个键与验证 B-g 相同（3000 行）；16 个键是交互图的设置（1500 行），规律相同。
  // 容差：换 40 个种子，32 个键时 d ≥ 16 三个值的最大/最小不超过 1.032；换 60 个种子，16 个键时不超过 1.046。
  const dims = [1, 16, 256, 4096];
  for (const [n, trials, tol] of [[32, 3000, 1.05], [16, 1500, 1.08]]) {
    const rng = makeRng(7);
    const raw = dims.map((d) => meanMaxWeight(rng, d, n, false, trials));
    const scaled = dims.map((d) => meanMaxWeight(rng, d, n, true, trials));
    for (let i = 1; i < raw.length; i++) assert.ok(raw[i] > raw[i - 1], `n=${n} raw ${raw}`);
    assert.ok(raw[3] > 0.95, `n=${n} raw ${raw}`);
    const big = scaled.slice(1);                                     // d = 16、256、4096
    assert.ok(Math.max(...big) / Math.min(...big) < tol, `n=${n} scaled ${scaled}`);
    assert.ok(scaled[0] < Math.min(...big), `n=${n} scaled ${scaled}`);  // d = 1 时 |q|²/d 波动大，略低
    if (n === 32) {   // 与验证 B-g 的数值相符：d ≥ 16 约 0.164，d = 1 约 0.142（40 万行的估计）
      close(scaled[3], 0.164, 0.01, 'd=4096');
      close(scaled[0], 0.142, 0.01, 'd=1');
    }
  }
});

test('appendix-baseline：精确的方差、b*、E[r] 与验证 C-c 相同', () => {
  const u = bandit(PRESETS.uniform);
  u.grad.forEach((g, i) => close(g, [-1 / 16, -1 / 16, 3 / 16, -1 / 16][i], 1e-12));
  close(u.bStar, 0.25, 1e-12);
  close(u.totalVar(0), 9 / 64, 1e-12);
  close(u.totalVar(0.25), 6 / 64, 1e-12);
  close(u.totalVar(1), 33 / 64, 1e-12);
  const s = bandit(PRESETS.skewed);
  close(s.meanR, 0.2689, 1e-4);
  close(s.bStar, 0.4435, 1e-4);
  close(s.totalVar(0), 0.2333, 1e-4);
  close(s.totalVar(s.meanR), 0.1283, 1e-4);
  close(s.totalVar(s.bStar), 0.1090, 1e-4);
  // 抛物线：V(b) = V(b*) + E‖s‖²·(b − b*)²
  const es2 = s.pi.reduce((acc, p, a) => acc + p * s.s2[a], 0);
  for (const b of [-1, 0, 0.7, 2]) close(s.totalVar(b), s.totalVar(s.bStar) + es2 * (b - s.bStar) ** 2, 1e-12);
});

test('appendix-baseline：上图那一维的方差在它自己的 b*_k 处最小，不在 b*；b* 是各维 b*_k 的加权平均（验证 C-c）', () => {
  const u = bandit(PRESETS.uniform);
  [1 / 12, 1 / 12, 3 / 4, 1 / 12].forEach((v, k) => close(u.compStar(k), v, 1e-12, `b*_${k}`));
  close(u.compVar(0.75, 2), 0, 1e-15, 'b = 0.75 时第 2 维的估计恒为 3/16');
  close(u.totalVar(0.75), 3 * u.totalVar(0.25), 1e-12, '而四维之和是最小值的 3 倍');
  const grid = Array.from({ length: 30001 }, (_, i) => -1 + i * 1e-4);   // 交互图滑块的范围 [−1, 2]
  for (const name of Object.keys(PRESETS)) {
    const m = bandit(PRESETS[name]);
    const ek = [0, 1, 2, 3].map((k) => m.pi.reduce((acc, p, a) => acc + p * m.S[a][k] ** 2, 0));   // E[s_k²]
    close(ek.reduce((acc, e, k) => acc + e * m.compStar(k), 0) / ek.reduce((x, y) => x + y, 0), m.bStar, 1e-12, name);
    for (let k = 0; k < 4; k++) {
      const argmin = grid.reduce((x, y) => (m.compVar(y, k) < m.compVar(x, k) ? y : x));
      close(argmin, m.compStar(k), 1e-4, `${name} 第 ${k} 维`);
    }
  }
  // 不均匀策略：交互图默认画奖励最高的第 1 号臂，它的最优基线约 0.82，而 b* 约 0.44
  const s = bandit(PRESETS.skewed);
  assert.ok(s.compStar(1) > 0.8 && s.compStar(1) < 0.85 && Math.abs(s.bStar - 0.4435) < 1e-4, `${s.compStar(1)}`);
});

test('appendix-adam：m_t = 1 − 0.9ᵗ；f(t) 在第 12 步达到 6.5685（验证 D-a、D-d）', () => {
  emaPath([1, 1, 1, 1, 1], 0.9).forEach(([m, mh], i) => {
    close(m, 1 - 0.9 ** (i + 1), 1e-12);
    close(mh, 1, 1e-12);
  });
  close(uncorrectedRatio(1, 0.9, 0.999), 0.1 / Math.sqrt(0.001), 1e-12);
  const s = ratioSummary(0.9, 0.999);
  assert.equal(s.at, 12);
  close(s.best, 6.5685, 1e-4);
  assert.equal(s.settle, 3925);
  assert.match(ratioText(s), /最大 6\.57，在第 12 步；第 3,925 步之后/);
});

test('appendix-adam：β₁ ≥ β₂ 时 f(t) 从下方逼近 1，总结里不再报“最大值”', () => {
  // β₁ = β₂ = 0.9：f(t) = √(1 − 0.9ᵗ)，0.9ᵗ ≤ 0.0199 即 t ≥ 38 时进入 1 ± 0.01
  const same = ratioSummary(0.9, 0.9);
  assert.ok(same.best <= 1 && same.first < 1);
  assert.equal(same.settle, 38);
  // β₁ = 0.99、β₂ = 0.9：1 − 0.99ᵗ ≈ 0.99 即 t ≈ 459
  const slow = ratioSummary(0.99, 0.9);
  assert.equal(slow.settle, 459);
  const text = ratioText(slow);
  assert.match(text, /从未超过 1\.01/);
  assert.doesNotMatch(text, /最大/);
  // 在 tMax 步之内回不到 1 ± 0.01 时如实说明（β₂ = 0.9999 要到第 39,268 步）
  const short = ratioSummary(0.9, 0.9999, 1000);
  assert.equal(short.settle, null);
  assert.match(ratioText(short), /1,000 步之内都没有稳定/);
  assert.equal(ratioSummary(0.9, 0.9999).settle, 39268);
});

test('appendix-dpo：π*、Z、隐式奖励与验证 E-c、E-e 相同', () => {
  const piRef = [0.4, 0.3, 0.2, 0.1], r = [0, 1, 2, 0.5], beta = 0.5;
  const { pi, logZ } = optimalPolicy(piRef, r, beta);
  [0.02896835, 0.16053655, 0.79080907, 0.01968603].forEach((v, i) => close(pi[i], v, 1e-7));
  close(Math.exp(logZ), 13.808175, 1e-5);
  pi.forEach((p, i) => close(beta * Math.log(p / piRef[i]) + beta * logZ, r[i], 1e-12));
  close(kl(pi, piRef), 0.878735, 1e-5);
  close(sigmoid(2), 0.880797, 1e-6);
  // β 很小时不溢出，集中在奖励最高的回答上
  const sharp = optimalPolicy(piRef, r, 0.01).pi;
  assert.ok(sharp[2] > 0.999 && sharp.every(Number.isFinite));
});
