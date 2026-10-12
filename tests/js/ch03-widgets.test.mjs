// 第 3 章交互图里的纯函数：与正文验证单元的结论一致（node --test tests/js/ch03-widgets.test.mjs）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  rng, softmax, entropy, topK, topP, sample, advantages, looAdvantages, pNoSignal,
} from '../../web/widgets/ch03-common.js';
import { LINEAGE, PRESETS } from '../../web/widgets/ch03-lineage.js';
import { ratioHistogram } from '../../web/widgets/ch03-dropout.js';
import { passAtK, correctProbs, MODELS } from '../../web/widgets/ch03-answers.js';
import { parseMetrics, accSummary, lossSummary, patchText } from '../../web/widgets/ch03-grporuns.js';
import { stdNote } from '../../web/widgets/ch03-group.js';
import { columnsFor } from '../../web/widgets/ch03-novar.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const close = (a, b, tol = 1e-3) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test('softmax：和为 1；T = 0 是 argmax 的 one-hot；温度越高熵越大', () => {
  const z = [2, 1, 0.5, 0, -1];
  const p = softmax(z);
  close(p.reduce((a, b) => a + b, 0), 1, 1e-12);
  [0.563, 0.207, 0.126, 0.076, 0.028].forEach((v, i) => close(p[i], v));
  assert.deepEqual(softmax(z, 0), [1, 0, 0, 0, 0]);
  const Hs = [0.25, 0.5, 1, 2, 4].map((T) => entropy(softmax(z, T)));
  Hs.slice(1).forEach((H, i) => assert.ok(H > Hs[i]));
  close(entropy(softmax(z, 1000)), Math.log(5), 1e-3);
});

test('Top-k 与 Top-p：与验证单元 3.2-d 的手算结果一致', () => {
  const p = softmax([2, 1, 0.5, 0, -1]);
  const k2 = topK(p, 2);
  [0.731, 0.269, 0, 0, 0].forEach((v, i) => close(k2[i], v));
  assert.deepEqual(topK(p, 1), [1, 0, 0, 0, 0]);
  const p9 = topP(p, 0.9);
  assert.equal(p9.filter((v) => v > 0).length, 4);
  [0.579, 0.213, 0.129, 0.078, 0].forEach((v, i) => close(p9[i], v));
  assert.deepEqual(topP(p, 1), p);
  assert.deepEqual(topK(p, 0), p);
});

test('rng 与 sample：可复现，频率收敛到概率', () => {
  const a = rng(42), b = rng(42);
  for (let i = 0; i < 5; i++) assert.equal(a(), b());
  const p = [0.5, 0.3, 0.2];
  const r = rng(1);
  const counts = [0, 0, 0];
  for (let i = 0; i < 20000; i++) counts[sample(p, r)] += 1;
  counts.forEach((c, i) => close(c / 20000, p[i], 0.015));
});

test('组内优势：和为 0；G=1 与全相同的组为 0；= (G-1)/G × RLOO；P(无信号)', () => {
  const A = advantages([1, 0, 0, 1, 1, 0, 0, 0]);
  [0.625, -0.375, -0.375, 0.625, 0.625, -0.375, -0.375, -0.375].forEach((v, i) => close(A[i], v, 1e-12));
  assert.deepEqual(advantages([1]), [0]);
  assert.ok(advantages([1, 1, 1, 1]).every((v) => v === 0));
  assert.ok(advantages([0, 0, 0], { stdNorm: true }).every((v) => v === 0));
  assert.ok(Number.isNaN(advantages([1], { stdNorm: true })[0]));
  const r = [0.3, 0.9, 0.1, 0.5, 0.7];
  const loo = looAdvantages(r);
  advantages(r).forEach((v, i) => close(v, (4 / 5) * loo[i], 1e-12));
  close(pNoSignal(0.5, 8), 2 / 256, 1e-15);
  assert.equal(pNoSignal(0.3, 1), 1);
});

test('除以标准差时，k 个答对的组里答对者的优势 ≈ sqrt((1-p)/p)（ddof=1 与 ddof=0 只差常数倍）', () => {
  for (const k of [1, 4, 7]) {
    const r = [...Array(k).fill(1), ...Array(8 - k).fill(0)];
    const a = advantages(r, { stdNorm: true, eps: 0 })[0];
    const p = k / 8;
    close(a, Math.sqrt((1 - p) / p) * Math.sqrt(7 / 8), 1e-12);
  }
});

test('家谱与配置解码器里引用的行号，确实指向原书代码的对应内容', () => {
  const grpo = fs.readFileSync(path.join(ROOT, 'third_party/dlfs6/ch03/09_grpo.py'), 'utf8').split('\n');
  const utils = fs.readFileSync(path.join(ROOT, 'third_party/dlfs6/codebot/utils.py'), 'utf8').split('\n');
  const line = (n) => grpo[n - 1];
  assert.match(line(89), /advantages = rewards - rewards\.mean\(\)/);
  assert.match(line(110), /def grpo_loss/);
  assert.match(line(129), /return -token_objective\.sum\(\) \/ n_samples/);
  assert.match(line(84), /temperature=1\.0/);
  assert.match(line(139), /learning_rate = 7e-6/);
  assert.match(line(141), /n_update_per_generation = 2/);
  assert.match(line(143), /epsilon = 0\.2/);
  assert.match(line(144), /group_size = 8/);
  assert.match(line(145), /batch_size = 32/);
  assert.match(line(156), /DataLoader\(dataset, batch_size=batch_size, shuffle=True\)/);
  assert.match(line(181), /for _ in range\(n_update_per_generation\)/);
  assert.match(utils[5], /max_new_tokens=1000/);
  // 解码器与家谱里出现的每个 “09_grpo.py 第 N 行” 都在文件范围内
  const refs = JSON.stringify([LINEAGE, PRESETS]).match(/09_grpo\.py 第 [\d、–]+ 行/g);
  assert.ok(refs.length >= 8);
  for (const ref of refs) for (const n of ref.match(/\d+/g).slice(1)) assert.ok(Number(n) <= grpo.length);
});

test('比值直方图：每格 0.1，最后一格是 ≥ 2.5', () => {
  const bins = ratioHistogram([0.05, 0.85, 0.95, 1.0, 1.19, 2.6, 12]);
  assert.equal(bins.length, 26);
  assert.equal(bins[0], 1);
  assert.equal(bins[8] + bins[9] + bins[10] + bins[11], 4);
  assert.equal(bins[25], 2);
});

test('pass@k 与验证单元 3.9-i 一致（SFT 的 pass@8 高于 GRPO）', () => {
  const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/data/ch03/answer_dists.json'), 'utf8'));
  const truth = data.prompts.map((s) => Number(s[0]) + Number(s[2]));
  const sft = data.sft.map((r, k) => r[truth[k]]);
  const grpo = data.grpo.map((r, k) => r[truth[k]]);
  close(passAtK(sft, 1), 0.450);
  close(passAtK(grpo, 1), 0.821);
  close(passAtK(sft, 8), 0.915);
  close(passAtK(grpo, 8), 0.834);
});

test('本书运行的答案分布：原样 GRPO 有 9 道题被压死，加 model.eval() 的一道也没有（与验证单元 3.9-i 一致）', () => {
  const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/data/ch03/answer_dists.json'), 'utf8'));
  assert.deepEqual(MODELS.map((m) => m.key).filter((k) => data[k]), ['sft', 'grpo', 'run_sft', 'run_grpo', 'run_evalfix']);
  const dead = (k) => correctProbs(data, k).filter((p) => p < 0.01).length;
  assert.deepEqual(['sft', 'grpo', 'run_sft', 'run_grpo', 'run_evalfix'].map(dead), [0, 12, 2, 9, 0]);
  close(passAtK(correctProbs(data, 'run_evalfix'), 8), 1.0);
  close(passAtK(correctProbs(data, 'run_grpo'), 8), 0.873);
});

test('两次 3060 Ti 运行的记录：第 51 轮 60.5% 对 98.8%；最终 86.4% 对 100%（第 131 轮起）；loss 的水位', () => {
  const read = (run) => parseMetrics(fs.readFileSync(path.join(ROOT, `web/data/runs/${run}/09_grpo.metrics.jsonl`), 'utf8'));
  const asIs = read('codebot-3060ti'), fixed = read('codebot-grpo-evalfix-3060ti');
  assert.equal(asIs.length, 501);
  assert.deepEqual(accSummary(asIs), { start: 53.1, at51: 60.5, final: 86.4, firstFinal: 291, allFrom: null });
  assert.deepEqual(accSummary(fixed), { start: 56.8, at51: 98.8, final: 100, firstFinal: 131, allFrom: 131 });
  const la = lossSummary(asIs), lf = lossSummary(fixed);
  close(la.mean, 0.121); close(la.max, 3.8697, 1e-4); assert.equal(la.maxIt, 433);
  close(lf.mean, 0.0066, 1e-4); close(lf.median, 0, 1e-4); close(lf.max, 0.3233, 1e-4);
  const report = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/data/runs/codebot-grpo-evalfix-3060ti/report.json'), 'utf8'));
  assert.equal(patchText(report.steps[0].patches[0]), 'model.eval()（加在 for _ in range(n_update_per_generation): 之前）');
  assert.deepEqual(parseMetrics('{"it": 0, "acc": 1}\r\n\r\n{"it": 1, "acc": 2}\r\n'), [{ it: 0, acc: 1 }, { it: 1, acc: 2 }]);
});

test('组内优势计算器“除以标准差”：答对者的优势与专栏公式 √((1−p)/p) 只差常数倍 √((G−1)/G)', () => {
  const r = [1, 0, 0, 1, 1, 0, 0, 0];
  const note = stdNote(r, advantages(r, { stdNorm: true }), true);
  assert.match(note, /A₊ = 1\.207/);                      // 无偏 std（除以 G−1）再加 1e-4
  assert.match(note, /√\(\(1−p\)\/p\) = 1\.291/);         // 3.7-h：除以 G 的 std
  assert.match(note, /√\(\(G−1\)\/G\) = 0\.935/);
  assert.equal(stdNote(r, advantages(r), false), '');      // 没打开开关时不说
  assert.equal(stdNote([1, 1, 1], advantages([1, 1, 1], { stdNorm: true }), true), '');   // 全对的组：没有 A₊ 可比
});

test('P(无信号) 热力图：放不下 11 列（每格 ≥ 30px）时只画 p = 0, 0.2, …, 1', () => {
  assert.equal(columnsFor(700).length, 11);
  assert.equal(columnsFor(343).length, 6);                // 375px 的手机，正文约 343px 宽
  assert.equal(columnsFor(0).length, 11);                 // 还不知道宽度时先画完整的网格
  assert.deepEqual(columnsFor(343), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  const cell = (w, C) => Math.min(44, Math.floor((w - 44 - 4) / C));   // 与 _plot.js heatmap 的格子大小相同
  assert.ok(cell(343, columnsFor(343).length) >= 30 && cell(378, columnsFor(378).length) >= 30);
});
