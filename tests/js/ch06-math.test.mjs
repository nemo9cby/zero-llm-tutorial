// 第 6 章其余组件里的计算：与正文验证单元的输出一致（优化器轨迹、学习率调度、Wilson 区间与符号检验、DPO loss）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runSGD, runMomentum, runAdam, decayOnly } from '../../web/widgets/ch06-optim.js';
import { getLr, cosineLr, summarize } from '../../web/widgets/ch06-lrsched.js';
import { wilson, signTestP, wald } from '../../web/widgets/ch06-wilson.js';
import { dpo } from '../../web/widgets/ch06-dpo.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b}`);
const start = [-7, 2];

test('优化器轨迹与验证 6.1-a、6.1-b、6.1-d、6.1-e 一致', () => {
  const sgd = runSGD(start, 0.95, 30).path;
  close(sgd[3][0], -5.1885, 5e-5); close(sgd[3][1], -1.458, 5e-5);
  close(sgd[30][0], -0.3504, 5e-5); close(sgd[30][1], 0.0848, 5e-5);
  for (let t = 1; t <= 30; t++) close(sgd[t][1] / sgd[t - 1][1], -0.9, 1e-12);

  const mom = runMomentum(start, 0.1, 0.9, 30).path;
  close(mom[30][0], 0.8964, 5e-5); close(mom[30][1], 0.0881, 5e-5);

  const adam = runAdam(start, 0.3, 30);
  close(adam.path[1][0], -6.7, 1e-6); close(adam.path[1][1], 1.7, 1e-6);
  close(adam.path[2][0], -6.4004, 5e-5); close(adam.path[2][1], 1.4022, 5e-5);
  close(adam.path[30][0], 0.136, 5e-5); close(adam.path[30][1], 0.3182, 5e-5);

  // 碗在原点时，Adam + L2 的轨迹与 Adam 相同（耦合的 L2 被归一化吃掉）
  const l2 = runAdam(start, 0.3, 30, [0, 0], { wd: 0.1, coupled: true }).path;
  for (let t = 0; t <= 30; t++) for (let k = 0; k < 2; k++) close(l2[t][k], adam.path[t][k], 1e-6);

  close(decayOnly(10)[10], 0.999 ** 10, 1e-12);
  const coupled = decayOnly(10, { coupled: true });
  [1.0, 0.9, 0.8004, 0.7016, 0.6039, 0.508, 0.4142, 0.3234, 0.2363, 0.1536, 0.0763]
    .forEach((v, t) => close(coupled[t], v, 5e-5, `t=${t}`));
});

test('学习率调度与验证 6.3-c、6.3-d、6.3-e 一致', () => {
  assert.equal(getLr(0, 1e-3, 200, 40000), 0);
  close(getLr(1, 1e-3, 200, 40000), 5e-6, 1e-18);
  close(getLr(100, 1e-3, 200, 40000), 5e-4, 1e-18);
  close(getLr(20100, 1e-3, 200, 40000), 5e-4, 1e-15);
  close(getLr(39999, 1e-3, 200, 40000), 1e-3 / 39800, 1e-18);
  assert.equal(getLr(40000, 1e-3, 200, 40000), 0);
  const s = summarize((it) => getLr(it, 1e-3, 200, 40000), 40000, 1e-3);
  close(s.sum, 20.0, 1e-9); close(s.meanRatio, 0.5, 1e-12); close(s.shrink, Math.exp(-0.2), 1e-5);
  // CS336 测试的期望值（预热 7、到第 21 步、最小 0.1）
  close(cosineLr(8, 1, 7, 21, 0.1), 0.9887175604818206, 1e-12);
  close(cosineLr(14, 1, 7, 21, 0.1), 0.55, 1e-12);
  close(cosineLr(21, 1, 7, 21, 0.1), 0.1, 1e-12);
  close(cosineLr(24, 1, 7, 21, 0.1), 0.1, 1e-12);
  assert.equal(getLr(14, 1, 7, 21), 0.5);
  assert.equal(getLr(21, 1, 7, 21), 0);
  // lr_graph.py 的余弦：全程平均 ≈ 0.5475
  const c = summarize((it) => cosineLr(it, 1, 5000, 100000, 0.1), 100000, 1);
  close(c.meanRatio, 0.5475, 1e-3);
});

test('Wilson 区间与符号检验与验证 6.6-b 一致', () => {
  const [lo, hi] = wilson(5, 8);
  close(lo, 0.3057, 5e-5); close(hi, 0.8632, 5e-5);
  close(wilson(8, 8)[0], 0.6756, 5e-5);
  close(wilson(23, 32)[0], 0.5463, 5e-5); close(wilson(23, 32)[1], 0.8444, 5e-5);
  close(wilson(50, 100)[1] - 0.5, 0.0962, 5e-5);
  close(signTestP(13, 8), 0.38331031799316406, 1e-12);
  close(signTestP(23, 9), 0.020061607006937265, 1e-12);
  assert.equal(signTestP(5, 5), 1);
  const big = signTestP(600, 400);
  assert.ok(big > 0 && big < 1e-9);                // 大样本不下溢成 0
  const [wl, wh] = wald(8, 8);
  assert.equal(wl, 1); assert.equal(wh, 1);         // 全胜时 Wald 区间宽度为 0
});

test('DPO loss 与验证 6.7-f 一致', () => {
  close(dpo(0.1, 0, 0).loss, Math.LN2, 1e-15);
  // log π(c) = -10、log π(r) = -12、参考模型都是 -11 → s_c = 1，s_r = -1，z = 0.2
  const d = dpo(0.1, 1, -1);
  close(d.z, 0.2, 1e-15); close(d.loss, 0.598139, 1e-6); close(d.weight, 0.045017, 1e-6);
  // loss 只取决于 s_c − s_r
  close(dpo(0.1, 10, -10).loss, dpo(0.1, -10, -30).loss, 1e-15);
});
