// 第 2 章交互图的计算：与页面验证单元断言的数字逐一对拍。
//   node --test tests/js/ch02-widgets.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { softDict, MOVIES } from '../../web/widgets/ch02-softdict.js';
import { dotSamples, variance, meanMaxWeight } from '../../web/widgets/ch02-scaling.js';
import { leakReport, MODES } from '../../web/widgets/ch02-masklab.js';
import { panels, futureCells, elementAt, C, H, D } from '../../web/widgets/ch02-heads.js';
import { prevTokenMeans, PRESETS as HEAD_PRESETS } from '../../web/widgets/ch02-attnmaps.js';
import { countParams, PRESETS } from '../../web/widgets/ch02-params.js';
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

test('掩码实验台：只有“先遮挡、再按行 softmax”不漏（验证 2.6-d/e）', () => {
  const past = (mode) => leakReport(mode).slice(0, 4);
  assert.ok(past('right').every((r) => !r.changed && !r.nan));
  for (const mode of ['nomask', 'axis', 'after']) assert.ok(past(mode).some((r) => r.changed), mode);
  const strict = past('strict');
  assert.ok(strict[0].nan && strict.slice(1).every((r) => !r.changed && !r.nan));
  assert.equal(MODES.length, 5);
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
