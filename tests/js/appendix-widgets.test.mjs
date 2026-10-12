// 附录 B–E 交互图里的计算：与对应验证单元（Python）打印的数值对拍。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeRng, softmax, histogram } from '../../web/widgets/appendix-common.js';
import { meanMaxWeight } from '../../web/widgets/appendix-scaling.js';
import { bandit, PRESETS } from '../../web/widgets/appendix-baseline.js';
import { emaPath, uncorrectedRatio, ratioSummary } from '../../web/widgets/appendix-adam.js';
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

test('appendix-scaling：不缩放时最大权重随 d 上升，缩放后基本不变（验证 B-g）', () => {
  const rng = makeRng(7);
  const raw = [1, 16, 256, 4096].map((d) => meanMaxWeight(rng, d, 32, false, 300));
  const scaled = [1, 16, 256, 4096].map((d) => meanMaxWeight(rng, d, 32, true, 300));
  for (let i = 1; i < raw.length; i++) assert.ok(raw[i] > raw[i - 1], `raw ${raw}`);
  assert.ok(raw[3] > 0.95, `raw ${raw}`);
  assert.ok(Math.max(...scaled) / Math.min(...scaled) < 1.2, `scaled ${scaled}`);
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
