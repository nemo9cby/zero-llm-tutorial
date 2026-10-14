/* 图：Attention 缩放（2.3 节）。拖动维数 d：不缩放时 q·k 的分布随 d 变宽、注意力越来越尖；
   除以 √d 之后形状与 d 无关。设定与验证 2.3-b/2.3-c 相同（一个查询对 256 个键；JS 测试对拍统计量）。
   采样技巧：固定 q 时，q·k_j（k_j ~ N(0, I)）独立同分布于 N(0, |q|²)，而 |q|² ~ χ²_d，
   所以 q·k = |q|·z 在分布上完全等价，d = 4096 也只需 O(1) 次采样。 */
import { h } from './_lib.js';
import { lineChart, seqColor, fmtNum } from './_plot.js';
import { mulberry32, gaussian, chiSquare, softmax, injectStyle, range, onThemeChange, seqLegend } from './ch02-util.js';

export const N_KEYS = 256;              // 与验证 2.3-c 相同
const RAW_LIM = 256;                    // 左图横轴固定在 ±4√4096：拖动 d 时能看见分布变宽
const SCALED_LIM = 4;                   // 右图横轴固定在 ±4：缩放后的分布不随 d 移动

/** n independent pairs (q, k) with q, k ~ N(0, I_d): their dot products. */
export function dotSamples(d, n, rand) {
  return range(n).map(() => Math.sqrt(chiSquare(d, rand)) * gaussian(rand));
}

/** One random query against n random keys: raw scores q·k_j. */
export function rowScores(d, n, rand) {
  const qn = Math.sqrt(chiSquare(d, rand));
  return range(n).map(() => qn * gaussian(rand));
}

export function variance(xs) {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length;
}

/** Effective number of keys 1 / Σw²: exactly k when the weight is spread evenly over k keys. */
export const effectiveKeys = (w) => 1 / w.reduce((a, x) => a + x * x, 0);

/** Mean (over trials) of the largest attention weight among nKeys keys. */
export function meanMaxWeight(d, { nKeys = N_KEYS, trials = 200, scaled = false, rand }) {
  let s = 0;
  for (let t = 0; t < trials; t++) {
    const sc = rowScores(d, nKeys, rand);
    s += Math.max(...softmax(scaled ? sc.map((x) => x / Math.sqrt(d)) : sc));
  }
  return s / trials;
}

export function histogram(xs, lo, hi, bins) {
  const counts = new Array(bins).fill(0);
  const w = (hi - lo) / bins;
  for (const x of xs) {
    const k = Math.floor((x - lo) / w);
    if (k >= 0 && k < bins) counts[k] += 1;
  }
  return counts.map((c, k) => [lo + (k + 0.5) * w, c]);
}

export function mount(root, config) {
  injectStyle();
  let logd = Math.round(Math.log2(config.d || 64));
  let seed = 1;

  const dOut = h('span', { class: 'w-ch02-num' });
  const dInput = h('input', { type: 'range', min: 1, max: 12, step: 1, value: logd, 'aria-label': '维数 d（2 的幂，2 到 4096）' });
  dInput.addEventListener('input', () => { logd = +dInput.value; render(); });
  const again = h('button', { type: 'button' }, '重新采样');
  again.addEventListener('click', () => { seed += 1; render(); });
  const stats = h('p', { class: 'w-label', 'aria-live': 'polite' });
  const rawBox = h('div');
  const scaledBox = h('div');
  const strips = h('div');
  const curveBox = h('div');
  root.append(
    h('div', { class: 'w-ch02-controls' }, h('label', {}, '维数', dInput, dOut), again),
    stats,
    h('div', { class: 'w-ch02-two' },
      h('div', {}, h('p', { class: 'w-label' }, '不缩放的 q·k，横轴固定在 ±256：d 越大越宽'), rawBox),
      h('div', {}, h('p', { class: 'w-label' }, '同一批样本，横轴固定在 ±4：除以 √d 之后不随 d 变'), scaledBox)),
    strips,
    h('p', { class: 'w-label', style: { marginTop: '.8rem' } }, `每个 d 下，一个查询对 ${N_KEYS} 个键的最大权重（200 次试验的平均）：`),
    curveBox,
  );

  // The curve over all d is computed once (fixed seed) and only the marker moves.
  const rc = mulberry32(2024);
  const curve = ['不缩放', '除以 √d'].map((name, s) => ({
    name, points: range(12).map((i) => [i + 1, meanMaxWeight(2 ** (i + 1), { trials: 200, scaled: !!s, rand: rc })]),
  }));
  const ticksFor = () => ((root.clientWidth || 600) < 520 ? [2, 6, 10] : [2, 4, 6, 8, 10, 12]);
  let ticks = ticksFor();
  const curveChart = lineChart(curveBox, {
    series: curve, xLabel: '维数 d', yLabel: '最大权重的平均', height: 220, yMin: 0, yMax: 1,
    xFormat: (v) => fmtNum(2 ** v), xTicks: ticks, vLines: [{ x: logd, label: `d = ${2 ** logd}` }],
  });
  let rawChart = null, scaledChart = null;

  // Each strip: 256 weights sorted from large to small, one pixel column each, color = seqColor(√w).
  function strip(weights, label) {
    const sorted = [...weights].sort((a, b) => b - a);
    const cv = h('canvas', { width: N_KEYS, height: 1, class: 'w-ch02-strip-canvas', role: 'img',
      'aria-label': `${label}：${N_KEYS} 个权重从大到小排列，最大 ${sorted[0].toFixed(3)}` });
    const ctx = cv.getContext('2d');
    sorted.forEach((w, j) => { ctx.fillStyle = seqColor(Math.sqrt(w)); ctx.fillRect(j, 0, 1, 1); });
    return h('div', {},
      h('div', { class: 'w-label' }, `${label}：最大权重 ${sorted[0].toFixed(3)}，有效键数 ${effectiveKeys(weights).toFixed(1)} / ${N_KEYS}`),
      cv);
  }

  function render() {
    const d = 2 ** logd;
    dOut.textContent = d;
    const rand = mulberry32(seed * 7919 + logd);
    const raw = dotSamples(d, 4000, rand);
    const scaled = raw.map((x) => x / Math.sqrt(d));
    stats.textContent = `q·k 的方差（4000 对随机向量）：不缩放 ${variance(raw).toFixed(2)}（理论值 d = ${d}）；`
      + `除以 √d 之后 ${variance(scaled).toFixed(3)}（理论值 1）。`;
    const rawOpts = { series: [{ name: '不缩放', points: histogram(raw, -RAW_LIM, RAW_LIM, 64) }],
      xLabel: 'q·k', yLabel: '样本数', height: 180, yMin: 0, table: false, xTicks: [-RAW_LIM, -128, 0, 128, RAW_LIM] };
    const scaledOpts = { series: [{ name: '不缩放', points: histogram(raw, -SCALED_LIM, SCALED_LIM, 40) },
      { name: '除以 √d', points: histogram(scaled, -SCALED_LIM, SCALED_LIM, 40) }],
    xLabel: 'q·k 或 q·k / √d', yLabel: '样本数', height: 180, yMin: 0, table: false, xTicks: [-4, -2, 0, 2, 4] };
    if (rawChart) rawChart.update(rawOpts); else rawChart = lineChart(rawBox, rawOpts);
    if (scaledChart) scaledChart.update(scaledOpts); else scaledChart = lineChart(scaledBox, scaledOpts);
    const row = rowScores(d, N_KEYS, rand);
    strips.replaceChildren(
      h('p', { class: 'w-label' }, `一个查询对 ${N_KEYS} 个键的注意力权重，从大到小排列（颜色按色标，权重越大越醒目）：`),
      strip(softmax(row), '不缩放'),
      strip(softmax(row.map((x) => x / Math.sqrt(d))), '除以 √d'),
      seqLegend([[0, '0'], [0.5, '0.25'], [1, '1']]),
      h('p', { class: 'w-ch02-note' }, '有效键数 = 1 / Σw²：权重平均分给 k 个键时，它恰好等于 k；全压在一个键上时是 1。'));
    curveChart.update({ vLines: [{ x: logd, label: `d = ${d}` }] });
  }
  render();
  onThemeChange(render);
  new ResizeObserver(() => {
    const next = ticksFor();
    if (next.length !== ticks.length) { ticks = next; curveChart.update({ xTicks: ticks }); }
  }).observe(root);
}
