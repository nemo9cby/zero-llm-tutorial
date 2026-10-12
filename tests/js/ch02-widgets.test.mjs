// 第 2 章交互图的计算：与页面验证单元断言的数字逐一对拍。
//   node --test tests/js/ch02-widgets.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { softDict, MOVIES, entropy, hintFor } from '../../web/widgets/ch02-softdict.js';
import { dotSamples, variance, meanMaxWeight, effectiveKeys, histogram, N_KEYS } from '../../web/widgets/ch02-scaling.js';
import { leakReport, MODES, verdictFor } from '../../web/widgets/ch02-masklab.js';
import { panels, futureCells, elementAt, C, H, D } from '../../web/widgets/ch02-heads.js';
import { prevTokenMeans, PRESETS as HEAD_PRESETS } from '../../web/widgets/ch02-attnmaps.js';
import { countParams, PRESETS, approxNote } from '../../web/widgets/ch02-params.js';
import { erf, geluErf, geluTanh } from '../../web/widgets/ch02-gelu.js';
import { mulberry32, softmax } from '../../web/widgets/ch02-util.js';

const round2 = (x) => Math.round(x * 100) / 100;

test('软字典：78.66，72.62% / 26.71%（验证 2.1-b）', () => {
  const r = softDict([6, 4, 5]);
  assert.deepEqual(r.scores, [71, 59, 59, 75, 76, 70, 63]);
  assert.equal(round2(r.result), 78.66);
  assert.equal(round2(r.weights[4] * 100), 72.62);
  assert.equal(round2(r.weights[3] * 100), 26.71);
});

test('软字典：T→0 的并列 87.5 / 73.33，T→∞ 是平均 75（验证 2.1-d）', () => {
  assert.equal(round2(softDict([8, 2, 3], { T: 0.001 }).result), 87.5);
  assert.equal(round2(softDict([5, 5, 5], { T: 0.001 }).result), 73.33);
  assert.ok(Math.abs(softDict([6, 4, 5], { T: 1e6 }).result - 75) < 1e-3);
  for (const [i, m] of MOVIES.entries()) {
    if (i !== 0 && i !== 3) assert.equal(round2(softDict(m.key, { T: 0.001 }).result), m.value);
  }
});

test('软字典：相似度 ×10 时朴素 softmax 出现 NaN，稳定写法没有（验证 2.1-e）', () => {
  const naive = softDict([6, 4, 5], { scale: 10, naive: true }).weights;
  assert.deepEqual(naive.map((w) => (Number.isNaN(w) ? 'nan' : w)), ['nan', 0, 0, 'nan', 'nan', 0, 0]);
  const stable = softDict([6, 4, 5], { scale: 10 }).weights;
  assert.ok(stable.every(Number.isFinite));
  assert.ok(Math.abs(stable.reduce((a, b) => a + b, 0) - 1) < 1e-12);
});

test('软字典的提示：NaN 分溢出（inf/inf）与下溢（0/0）两种原因；余弦 ×10 时不说“几乎一样大”', () => {
  const state = (o) => ({ T: 1, kind: 'dot', scale: 1, naive: false, ...o });
  const over = state({ scale: 10, naive: true });
  const outOver = softDict([6, 4, 5], over);
  assert.ok(Number.isNaN(outOver.result));
  assert.match(hintFor(over, outOver), /溢出成 inf，inf \/ inf = NaN/);
  const under = state({ T: 0.01, kind: 'dist', naive: true });       // 负欧氏距离 / 很小的 T：最大也只有 −755
  const outUnder = softDict([10, 0, 10], under);
  assert.ok(Number.isNaN(outUnder.result) && Math.max(...outUnder.scores.map((s) => s / 0.01)) < -745);
  assert.match(hintFor(under, outUnder), /下溢成 0，0 \/ 0 = NaN/);
  assert.doesNotMatch(hintFor(under, outUnder), /溢出成 inf/);
  const cos10 = state({ kind: 'cos', scale: 10 });                   // 相似度 7.05–9.87，权重 2.35%–39.41%
  assert.doesNotMatch(hintFor(cos10, softDict([6, 4, 5], cos10)), /−1 到 1/);
  const cos1 = state({ kind: 'cos' });
  assert.match(hintFor(cos1, softDict([6, 4, 5], cos1)), /−1 到 1/);
});

test('软字典的熵：平均分给 7 部是 ln 7 nats = log2 7 bits；全给一部是 0', () => {
  assert.ok(Math.abs(entropy(new Array(7).fill(1 / 7)) - Math.log(7)) < 1e-12);
  assert.equal(entropy([0, 0, 1, 0]), 0);
  const w = softDict([6, 4, 5]).weights;
  assert.ok(entropy(w) > 0 && entropy(w) < Math.log(7));
});

test('缩放：Var(q·k) ≈ d，除以 √d 后 ≈ 1（验证 2.3-b）', () => {
  const rand = mulberry32(7);
  for (const d of [16, 64, 1024]) {
    const xs = dotSamples(d, 20000, rand);
    assert.ok(Math.abs(variance(xs) / d - 1) < 0.06, `d=${d}: ${variance(xs)}`);
    assert.ok(Math.abs(variance(xs.map((x) => x / Math.sqrt(d))) - 1) < 0.06);
  }
});

test('缩放：不缩放时越来越尖，缩放后与 d 无关（验证 2.3-c 的结论）', () => {
  const rand = mulberry32(3);
  const un = [4, 64, 1024].map((d) => meanMaxWeight(d, { nKeys: 256, trials: 300, rand }));
  const sc = [4, 64, 1024].map((d) => meanMaxWeight(d, { nKeys: 256, trials: 300, scaled: true, rand }));
  assert.ok(un[0] < un[1] && un[1] < un[2] && un[2] > 0.9, `unscaled ${un}`);
  assert.ok(Math.max(...sc) - Math.min(...sc) < 0.01 && Math.max(...sc) < 0.06, `scaled ${sc}`);
});

test('缩放图：与验证 2.3-c 同样 256 个键，缩放后最大权重约 0.04；有效键数 1/Σw² 在均分时等于 k', () => {
  assert.equal(N_KEYS, 256);
  const sc = meanMaxWeight(64, { scaled: true, rand: mulberry32(5) });     // 默认就是 256 个键
  assert.ok(sc > 0.03 && sc < 0.06, `scaled ${sc}`);
  assert.ok(Math.abs(effectiveKeys(new Array(10).fill(0.1)) - 10) < 1e-12 && effectiveKeys([1, 0, 0]) === 1);
  // 直方图的横轴固定：缩放后的样本几乎全落在 ±4 里，不缩放的样本 d 越大越少落在 ±4 里
  const rand = mulberry32(9);
  const inside = (xs) => histogram(xs, -4, 4, 40).reduce((a, [, c]) => a + c, 0) / xs.length;
  for (const d of [4, 1024]) {
    const raw = dotSamples(d, 4000, rand);
    assert.ok(inside(raw.map((x) => x / Math.sqrt(d))) > 0.99);
    if (d === 1024) assert.ok(inside(raw) < 0.2);
  }
});

test('掩码实验台：只有“先遮挡、再按行 softmax”不漏（验证 2.6-d/e）', () => {
  const past = (mode) => leakReport(mode).slice(0, 4);
  assert.ok(past('right').every((r) => !r.changed && !r.nan));
  for (const mode of ['nomask', 'axis', 'after']) assert.ok(past(mode).some((r) => r.changed), mode);
  const strict = past('strict');
  assert.ok(strict[0].nan && strict.slice(1).every((r) => !r.changed && !r.nan));
  assert.equal(MODES.length, 5);
  // 整行被遮住时，先减最大值的 softmax 得到的是 −∞ − (−∞)，不是 0/0（验证 2.6-d 的标题与 2.6.3 的正文）
  assert.match(verdictFor(leakReport('strict')), /−∞ − \(−∞\) = NaN/);
  assert.ok(Number.isNaN(-Infinity - -Infinity) && Number.isNaN(Math.exp(-Infinity - -Infinity)));
  assert.match(verdictFor(leakReport('right')), /^不漏/);
  assert.match(verdictFor(leakReport('axis')), /^泄漏/);
});

test('形状追踪：正确的拆法没有“未来”的格子，错误的拆法有（验证 2.8-e 的结论）', () => {
  assert.equal(futureCells('right').length, 0);
  assert.ok(futureCells('wrong').length > 0);
  // 正确拆法：第 a 个面板的第 r 行正是位置 r、头 a 的 D 个数
  panels('right').forEach((pan, a) => pan.forEach((row, r) => row.forEach((m, c) => {
    assert.deepEqual(elementAt(m), { p: r, h: a, d: c });
  })));
  // 两种拆法都是把同一块内存的 C·H·D 个数各用一次
  for (const kind of ['right', 'wrong']) {
    assert.deepEqual(panels(kind).flat(2).sort((a, b) => a - b), [...Array(C * H * D).keys()]);
  }
  // 第 4 步的说明：“头 0”面板的位置 1、2 装着位置 0 的头 1、头 2；“头 1”“头 2”面板的位置 0 装着位置 1、2 的数据
  const at = (a, r) => elementAt(panels('wrong')[a][r][0]);
  assert.deepEqual([at(0, 1), at(0, 2)].map(({ p, h }) => [p, h]), [[0, 1], [0, 2]]);
  assert.deepEqual([at(1, 0).p, at(2, 0).p], [1, 2]);
  assert.ok(futureCells('wrong').some(({ a, r }) => a === 1 && r === 0) && futureCells('wrong').some(({ a, r }) => a === 2 && r === 0));
});

test('真实注意力：L4H3 是最强的 previous-token head；预设的头与正文一致（验证 2.8-a）', () => {
  const data = JSON.parse(fs.readFileSync(new URL('../../web/data/ch02/attn_maps.json', import.meta.url), 'utf-8'));
  const prev = prevTokenMeans(data.snippets[0].attn);
  let best = [0, 0];
  prev.forEach((row, l) => row.forEach((v, k) => { if (v > prev[best[0]][best[1]]) best = [l, k]; }));
  assert.deepEqual(best, [4, 3]);
  const [p0, p1, , p3] = HEAD_PRESETS;
  assert.ok(data.snippets[p0.snippet].attn[p0.layer][p0.head][11][10] > 0.95);
  assert.ok(data.snippets[p1.snippet].attn[p1.layer][p1.head][9][1] > 0.9);
  const sq = data.snippets[p3.snippet];
  for (const t of [16, 17, 18]) {                 // L4H0 看前文 square 的“下一段”，也正是模型的预测
    const row = sq.attn[p3.layer][p3.head][t];
    const j = row.indexOf(Math.max(...row));
    assert.equal(j, t - 14);
    assert.ok(row[j] > 0.6);
    assert.equal(sq.pred[t], sq.tokens[j]);
  }
  assert.equal(p3.q, 17);
});

test('参数计算器：11,121,640 / 11,505,640 / GPT-2 small 两种写法（验证 2.10-a/b）', () => {
  const cb = countParams(PRESETS.codebot);
  assert.equal(cb.total, 11_121_640);
  assert.equal(cb.stateDictSum, 11_505_640);
  assert.equal(countParams({ ...PRESETS.codebot, tied: false }).total, 11_505_640);
  assert.equal(countParams({ ...PRESETS.gpt2, style: 'codebot' }).total, 124_453_201);
  assert.equal(countParams(PRESETS.gpt2).total, 124_439_808);
  // 12·L·E² 只在 ff = 4E 时成立；改了 FFN 宽度，说明文字要换成 L·(4E² + 2·E·ff)
  const r = countParams(PRESETS.codebot);
  assert.equal(r.approxBlocks, r.approx12);
  assert.match(approxNote(PRESETS.codebot, r), /^粗估 12·L·E² = 10,616,832/);
  const narrow = { ...PRESETS.codebot, ff: 768 };
  const rn = countParams(narrow);
  assert.equal(rn.approxBlocks, 6 * (4 * 384 * 384 + 2 * 384 * 768));
  assert.match(approxNote(narrow, rn), /ff ≠ 4E/);
});

test('GELU：tanh 近似与 erf 版最大差 4.73e-4，最低点 -0.16997（验证 2.9-b）', () => {
  assert.ok(Math.abs(erf(1) - 0.8427007929) < 2e-7);
  let maxDiff = 0, minVal = Infinity, argMin = 0;
  for (let i = 0; i <= 12000; i++) {
    const x = -6 + i * 0.001;
    maxDiff = Math.max(maxDiff, Math.abs(geluTanh(x) - geluErf(x)));
    if (geluErf(x) < minVal) { minVal = geluErf(x); argMin = x; }
  }
  assert.ok(Math.abs(maxDiff - 4.732e-4) < 2e-6, `maxDiff ${maxDiff}`);
  assert.ok(Math.abs(minVal + 0.16997) < 1e-4 && Math.abs(argMin + 0.752) < 2e-3);
});

test('softmax 工具：每行和为 1，对平移不变', () => {
  const p = softmax([1, 2, 3]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-15);
  softmax([101, 102, 103]).forEach((v, i) => assert.ok(Math.abs(v - p[i]) < 1e-12));
});
