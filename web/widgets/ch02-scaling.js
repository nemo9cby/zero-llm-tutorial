/* 图：Attention 缩放（2.3 节）。拖动维数 d：不缩放时 q·k 的方差随 d 增长、注意力越来越尖；
   除以 √d 之后形状与 d 无关。与验证 2.3-b/2.3-c 是同一个实验（JS 测试对拍统计量）。
   采样技巧：固定 q 时，q·k_j（k_j ~ N(0, I)）独立同分布于 N(0, |q|²)，而 |q|² ~ χ²_d，
   所以 q·k = |q|·z 在分布上完全等价，d = 4096 也只需 O(1) 次采样。 */
import { h } from './_lib.js';
import { lineChart, seqColor, fmtNum } from './_plot.js';
import { mulberry32, gaussian, chiSquare, softmax, injectStyle, range } from './ch02-util.js';

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

/** Mean (over trials) of the largest attention weight among nKeys keys. */
export function meanMaxWeight(d, { nKeys = 64, trials = 200, scaled = false, rand }) {
  let s = 0;
  for (let t = 0; t < trials; t++) {
    const sc = rowScores(d, nKeys, rand);
    s += Math.max(...softmax(scaled ? sc.map((x) => x / Math.sqrt(d)) : sc));
  }
  return s / trials;
}

function histogram(xs, lo, hi, bins) {
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
  const N_KEYS = 64;
  let logd = Math.round(Math.log2(config.d || 64));
  let seed = 1;

  const dOut = h('span', { class: 'w-ch02-num' });
  const dInput = h('input', { type: 'range', min: 1, max: 12, step: 1, value: logd, 'aria-label': '维数 d（2 的幂，2 到 4096）' });
  dInput.addEventListener('input', () => { logd = +dInput.value; render(); });
  const again = h('button', { type: 'button' }, '重新采样');
  again.addEventListener('click', () => { seed += 1; render(); });
  const stats = h('p', { class: 'w-label', 'aria-live': 'polite' });
  const histBox = h('div');
  const strips = h('div');
  const curveBox = h('div');
  root.append(
    h('div', { class: 'w-ch02-controls' }, h('label', {}, '维数', dInput, dOut), again),
    stats,
    histBox,
    strips,
    h('p', { class: 'w-label', style: { marginTop: '.8rem' } }, '每个 d 下，一个查询对 64 个键的最大权重（200 次试验的平均）：'),
    curveBox,
  );

  // The curve over all d is computed once (fixed seed) and only the marker moves.
  const rc = mulberry32(2024);
  const curve = ['不缩放', '除以 √d'].map((name, s) => ({
    name, points: range(12).map((i) => [i + 1, meanMaxWeight(2 ** (i + 1), { nKeys: N_KEYS, trials: 200, scaled: !!s, rand: rc })]),
  }));
  const curveChart = lineChart(curveBox, {
    series: curve, xLabel: '维数 d', yLabel: '最大权重的平均', height: 220, yMin: 0, yMax: 1,
    xFormat: (v) => fmtNum(2 ** v), vLines: [{ x: logd, label: `d = ${2 ** logd}` }],
  });
  let histChart = null;

  function strip(weights, label) {
    const sorted = [...weights].sort((a, b) => b - a);
    const max = sorted[0];
    const eff = 1 / weights.reduce((a, w) => a + w * w, 0);
    return h('div', {},
      h('div', { class: 'w-label' }, `${label}：最大权重 ${max.toFixed(3)}，有效键数 ${eff.toFixed(1)} / ${N_KEYS}`),
      h('div', { class: 'w-ch02-strip', style: { gridTemplateColumns: `repeat(${N_KEYS}, 1fr)` }, role: 'img',
        'aria-label': `${label}：64 个权重从大到小排列，最大 ${max.toFixed(3)}` },
      sorted.map((w) => h('span', { style: { background: seqColor(Math.sqrt(w)) } }))));
  }

  function render() {
    const d = 2 ** logd;
    dOut.textContent = d;
    const rand = mulberry32(seed * 7919 + logd);
    const raw = dotSamples(d, 4000, rand);
    const scaled = raw.map((x) => x / Math.sqrt(d));
    stats.textContent = `q·k 的方差（4000 对随机向量）：不缩放 ${variance(raw).toFixed(2)}（理论值 d = ${d}）；`
      + `除以 √d 之后 ${variance(scaled).toFixed(3)}（理论值 1）。`;
    const lim = 4 * Math.sqrt(d);
    const series = [{ name: '不缩放', points: histogram(raw, -lim, lim, 41) },
      { name: '除以 √d', points: histogram(scaled, -lim, lim, 41) }];
    const histOpts = { series, xLabel: 'q·k 的值', yLabel: '样本数', height: 200, yMin: 0, table: false };
    if (histChart) histChart.update(histOpts); else histChart = lineChart(histBox, histOpts);
    const row = rowScores(d, N_KEYS, rand);
    strips.replaceChildren(
      h('p', { class: 'w-label' }, `一个查询对 ${N_KEYS} 个键的注意力权重（从大到小排列，颜色越深权重越大）：`),
      strip(softmax(row), '不缩放'),
      strip(softmax(row.map((x) => x / Math.sqrt(d))), '除以 √d'));
    curveChart.update({ vLines: [{ x: logd, label: `d = ${d}` }] });
  }
  render();
}
