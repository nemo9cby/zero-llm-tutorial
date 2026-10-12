/* 图：D2Z（原书的 get_lr）与余弦退火（lr_graph.py 的公式，降到 η_min 后保持）的对比（6.3 节）。
   读数：全程平均学习率、学习率之和，以及 AdamW 默认 λ = 0.01 时，一个从未收到梯度的参数全程被衰减到的比例。 */
import { h } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';

/** ch06/03_lr_scheduler.py 的 get_lr，逐行照搬。 */
export function getLr(it, maxLr, warmupIters, maxIters) {
  if (it < warmupIters) return maxLr * (it / warmupIters);
  if (it < maxIters) {
    const progress = (it - warmupIters) / (maxIters - warmupIters);
    return maxLr * (1.0 - progress);
  }
  return 0.0;
}

/** lr_graph.py 的余弦（预热相同），到 maxIters 之后保持在 η_min（与 CS336 作业 1 的定义一致）。 */
export function cosineLr(it, maxLr, warmupIters, maxIters, minRatio) {
  if (it < warmupIters) return maxLr * (it / warmupIters);
  if (it <= maxIters) {
    const progress = (it - warmupIters) / (maxIters - warmupIters);
    return maxLr * (minRatio + 0.5 * (1 - minRatio) * (1 + Math.cos(Math.PI * progress)));
  }
  return maxLr * minRatio;
}

/** 训练实际用到的 it = 0 … maxIters-1 上的统计量。 */
export function summarize(fn, maxIters, maxLr, wd = 0.01) {
  let sum = 0;
  let logShrink = 0;
  for (let it = 0; it < maxIters; it++) {
    const lr = fn(it);
    sum += lr;
    logShrink += Math.log1p(-wd * lr);
  }
  return { sum, mean: sum / maxIters, meanRatio: sum / maxIters / maxLr, shrink: Math.exp(logShrink) };
}

export const PRESETS = {
  storybot: { label: 'StoryBot（05_pretrain.py）', maxLr: 1e-3, warmup: 200, total: 40000, minRatio: 0.1 },
  webbot: { label: 'WebBot（ch09/01_pretrain.py）', maxLr: 6e-4, warmup: 500, total: 100000, minRatio: 0.1 },
  lrgraph: { label: 'lr_graph.py（预热 5%）', maxLr: 1, warmup: 50, total: 1000, minRatio: 0.1 },
  cs336: { label: 'CS336 测试（预热 7，到第 21 步）', maxLr: 1, warmup: 7, total: 21, minRatio: 0.1 },
};

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch06-lr .ctrl { display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: .4rem 1.2rem; margin: .3rem 0 .6rem; font-size: .88rem; }
.w-ch06-lr label { display: flex; flex-direction: column; gap: .15rem; color: var(--ink-2); }
.w-ch06-lr label b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
.w-ch06-lr input[type=range] { width: 100%; margin: 0; accent-color: var(--ai); }
.w-ch06-lr select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .15rem .3rem; max-width: 100%; }
.w-ch06-lr .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: .2rem 1.2rem; font-size: .86rem; color: var(--ink-2); margin-top: .5rem; }
.w-ch06-lr .stats b { color: var(--ink); font-variant-numeric: tabular-nums; }
`));
}

export function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch06-lr');
  const presetSel = h('select', { 'aria-label': '预置的调度参数' },
    Object.entries(PRESETS).map(([k, p]) => h('option', { value: k }, p.label)));
  presetSel.value = PRESETS[config.preset] ? config.preset : 'storybot';
  const warmIn = h('input', { type: 'range', min: 0, 'aria-label': '预热的步数' });
  const minIn = h('input', { type: 'range', min: 0, max: 50, step: 1, 'aria-label': '余弦退火的下限（占最大学习率的百分比）' });
  const atIn = h('input', { type: 'range', min: 0, max: 1000, step: 1, value: 500, 'aria-label': '查看某一步的学习率' });
  const warmLabel = h('b'), minLabel = h('b'), atLabel = h('b');
  const chartBox = h('div');
  const stats = h('div', { class: 'stats' });
  root.append(
    h('div', { class: 'ctrl' },
      h('label', {}, '预置参数', presetSel),
      h('label', {}, h('span', {}, '预热 ', warmLabel), warmIn),
      h('label', {}, h('span', {}, '余弦的下限 ', minLabel), minIn),
      h('label', {}, h('span', {}, '查看第 ', atLabel, ' 步'), atIn)),
    chartBox, stats);

  let chart = null;
  // 预热按步数调：最多到全程的一半；长训练每格 100 步（各预置的预热步数都是它的整数倍）
  const loadPreset = () => {
    const p = PRESETS[presetSel.value];
    warmIn.max = Math.floor(p.total / 2);
    warmIn.step = p.total > 2000 ? 100 : 1;
    warmIn.value = p.warmup;
    minIn.value = Math.round(p.minRatio * 100);
    atIn.max = p.total;
    atIn.value = Math.round(p.total / 2);
  };

  function render() {
    const p = PRESETS[presetSel.value];
    const total = p.total;
    const warmup = Math.min(Math.round(+warmIn.value), total);
    const minRatio = +minIn.value / 100;
    warmLabel.textContent = `${warmup.toLocaleString('en-US')} 步（全程的 ${((100 * warmup) / total).toFixed(1)}%）`;
    minLabel.textContent = `${minRatio * 100}%`;
    const d2z = (it) => getLr(it, p.maxLr, warmup, total);
    const cos = (it) => cosineLr(it, p.maxLr, warmup, total, minRatio);
    const nPts = Math.min(total + 1, 400);
    const xs = [...new Set([...Array(nPts).keys()].map((i) => Math.round((i / (nPts - 1)) * total)).concat([0, 1, warmup, total]))]
      .filter((x) => x >= 0 && x <= total).sort((a, b) => a - b);
    const series = [
      { name: 'D2Z', points: xs.map((x) => [x, d2z(x)]) },
      { name: '余弦', points: xs.map((x) => [x, cos(x)]) },
    ];
    const yFormat = p.maxLr < 0.01 ? (v) => (v === 0 ? '0' : v.toExponential(1)) : (v) => +v.toFixed(3);
    // 预热结束处只画一条竖线、不加文字：竖线顶端正是两条曲线的最高点，文字会压在曲线上（步数已写在滑块上）
    const opts = { series, xLabel: 'iteration', yLabel: '学习率', height: 240, yMin: 0, table: false, yFormat,
      vLines: warmup > 0 ? [{ x: warmup, label: '' }] : [] };
    if (!chart) chart = lineChart(chartBox, opts); else chart.update(opts);

    const at = Math.min(+atIn.value, total);
    atLabel.textContent = at.toLocaleString('en-US');
    const sD = summarize(d2z, total, p.maxLr), sC = summarize(cos, total, p.maxLr);
    stats.replaceChildren(
      h('div', {}, `第 ${at.toLocaleString('en-US')} 步：D2Z `, h('b', {}, fmtNum(d2z(at))), '，余弦 ', h('b', {}, fmtNum(cos(at)))),
      h('div', {}, 'get_lr(0) = ', h('b', {}, fmtNum(d2z(0))), '：第 0 步参数不动，但 AdamW 的状态已经更新'),
      h('div', {}, '全程平均（÷ 最大值）：D2Z ', h('b', {}, sD.meanRatio.toFixed(4)), '，余弦 ', h('b', {}, sC.meanRatio.toFixed(4))),
      h('div', {}, '学习率之和：D2Z ', h('b', {}, fmtNum(sD.sum)), '，余弦 ', h('b', {}, fmtNum(sC.sum))),
      h('div', {}, '无梯度参数被 λ = 0.01 衰减到：D2Z ', h('b', {}, `${(sD.shrink * 100).toFixed(1)}%`), '，余弦 ',
        h('b', {}, `${(sC.shrink * 100).toFixed(1)}%`)),
    );
  }

  presetSel.addEventListener('change', () => { loadPreset(); render(); });
  for (const el of [warmIn, minIn, atIn]) el.addEventListener('input', render);
  loadPreset();
  render();
}
