/* 图：09_grpo.py 的 dropout bug。
   左：第一次更新时（新旧权重完全相同）283 个回答 token 的比值分布，train 模式（脚本原样）对 eval 模式。
   右：同一个种子下，dropout 开 / 关两条 GRPO 运行的逐轮指标（data/ch03/grpo_dropout_runs.json）。
   config: { ratios, runs } */
import { h } from './_lib.js';
import { lineChart, seriesColor } from './_plot.js';
import { injectStyle, loadJSON, segmented } from './ch03-common.js';

const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, ...kids) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of kids) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
};

const METRICS = [
  { key: 'greedy_acc', label: 'greedy 准确率（%）', sparse: true },
  { key: 'reward_mean', label: '采样回答的平均奖励' },
  { key: 'zero_var_frac', label: '没有信号的组的比例' },
  { key: 'entropy', label: '回答 token 的熵（nats）' },
  { key: 'grad_norm0', label: '裁剪前的梯度范数' },
  { key: 'loss1', label: '第 2 次更新的 loss' },
];

/** 比值的直方图：[0, 2.5) 每 0.1 一格，≥ 2.5 合成一格。 */
export function ratioHistogram(values, width = 0.1, top = 2.5) {
  const n = Math.round(top / width);
  const bins = Array(n + 1).fill(0);
  for (const v of values) bins[v >= top ? n : Math.floor(v / width + 1e-9)] += 1;
  return bins;
}

export async function mount(root, config) {
  injectStyle();
  const [ratios, index] = await Promise.all([loadJSON(root, config.ratios), loadJSON(root, config.runs)]);
  if (!ratios || !index) return;
  root.classList.add('w-ch03');
  root.append(h('style', {}, `.w-ch03-drop svg.hist { display: block; width: 100%; height: auto; }
.w-ch03-drop .axis { fill: var(--ink-3); font: 11px var(--mono); }`));
  root.classList.add('w-ch03-drop');

  // ---------------- 左：比值直方图
  const left = h('div'), right = h('div');
  root.append(h('div', { class: 'two' }, left, right));
  let mode = 'train';
  const histBox = h('div');
  const histNote = h('div', { class: 'stat', 'aria-live': 'polite' });
  left.append(h('div', { class: 'w-label' }, '第一次更新时的比值 π_θ / π_old（权重完全相同）'),
    segmented(['train 模式（脚本原样）', 'eval 模式（修复后）'], 0, (i) => { mode = i ? 'eval' : 'train'; drawHist(); }, '模型模式'),
    histBox, histNote);

  function drawHist() {
    const vals = ratios[mode];
    const bins = ratioHistogram(vals);
    const W = 360, H = 190, m = { l: 30, r: 8, t: 10, b: 28 };
    const maxc = Math.max(...bins);
    const bw = (W - m.l - m.r) / bins.length;
    const sy = (c) => m.t + (1 - c / maxc) * (H - m.t - m.b);
    const s = svg('svg', { class: 'hist', viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': `比值直方图（${mode} 模式），每格宽 0.1，最后一格是 2.5 以上` });
    // 裁剪区间 [0.8, 1.2]
    s.append(svg('rect', { x: m.l + 8 * bw, y: m.t, width: 4 * bw, height: H - m.t - m.b, fill: 'var(--ai-2)' }));
    bins.forEach((c, i) => {
      if (!c) return;
      s.append(svg('rect', { x: m.l + i * bw + 0.5, y: sy(c), width: Math.max(1, bw - 1), height: H - m.b - sy(c), fill: seriesColor(mode === 'train' ? 0 : 1) },
        svg('title', {}, `${i === bins.length - 1 ? '≥ 2.5' : `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`}：${c} 个 token`)));
    });
    for (const [x, t] of [[0, '0'], [8, '0.8'], [12, '1.2'], [20, '2'], [25, '≥2.5']]) {
      s.append(svg('text', { class: 'axis', x: m.l + x * bw, y: H - 10, 'text-anchor': 'middle' }, t));
    }
    s.append(svg('text', { class: 'axis', x: m.l - 4, y: m.t + 9, 'text-anchor': 'end' }, String(maxc)));
    s.append(svg('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: 'var(--ink-3)' }));
    histBox.replaceChildren(s);
    const out = vals.filter((v) => v < 0.8 || v > 1.2).length;
    histNote.textContent = `${vals.length} 个 token，落在裁剪区间 [0.8, 1.2]（浅色带）之外的有 ${out} 个（${((out / vals.length) * 100).toFixed(1)}%）；`
      + `最小 ${Math.min(...vals).toPrecision(3)}，最大 ${Math.max(...vals).toPrecision(3)}。`;
  }
  drawHist();

  // ---------------- 右：对照运行
  const runs = index.runs;
  const seeds = [...new Set(runs.map((r) => r.seed))].sort((a, b) => a - b);
  const rows = new Map();
  await Promise.all(runs.map(async (r) => {
    try {
      const resp = await fetch(`data/ch03/${r.file}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      rows.set(r.file, (await resp.text()).split('\n').filter(Boolean).map((l) => JSON.parse(l)));
    } catch (err) { rows.set(r.file, []); }
  }));
  let si = 0, mi = 0;
  const chartBox = h('div');
  const runNote = h('div', { class: 'stat', 'aria-live': 'polite' });
  right.append(h('div', { class: 'w-label' }, '同一种子的两条 GRPO 运行'),
    segmented(seeds.map((s) => `seed ${s}`), 0, (i) => { si = i; drawRuns(); }, '选择种子'),
    h('div', { class: 'ctl' }, h('label', {}, '指标 ', (() => {
      const sel = h('select', { 'aria-label': '选择指标' }, METRICS.map((m, i) => h('option', { value: i }, m.label)));
      sel.addEventListener('change', () => { mi = Number(sel.value); drawRuns(); });
      return sel;
    })())), chartBox, runNote);
  let chart = null;

  function drawRuns() {
    const metric = METRICS[mi];
    const sel = runs.filter((r) => r.seed === seeds[si]).sort((a, b) => Number(b.dropout) - Number(a.dropout));
    const series = sel.map((r) => ({
      name: r.dropout ? 'dropout 开' : 'dropout 关',
      points: (rows.get(r.file) || []).filter((p) => p[metric.key] != null).map((p) => [p.it, p[metric.key]]),
    })).filter((s) => s.points.length);
    const opts = { series, xLabel: 'iteration', yLabel: metric.label, height: 220, table: true };
    if (!series.length) { chartBox.replaceChildren(h('p', { class: 'w-label' }, '这一组运行还没有数据。')); chart = null; return; }
    if (!chart) { chartBox.replaceChildren(); chart = lineChart(chartBox, opts); } else chart.update(opts);
    runNote.textContent = sel.map((r) => `${r.dropout ? 'dropout 开' : 'dropout 关'}：最终 greedy 准确率 ${r.final_greedy_acc.toFixed(1)}%`).join('；')
      + `（${sel[0]?.who || ''}，${sel[0]?.iters || ''} 轮，CPU）。`;
  }
  drawRuns();
}
