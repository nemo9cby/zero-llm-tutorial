/* 附录 C 的交互图：基线只改变方差，不改变期望。
   config: { preset: "uniform" | "skewed", n: 8, comp?: 0–3 }
   4 臂老虎机、softmax 策略。梯度有 4 维，上图只画其中一维（comp，默认是奖励最高的那一臂）：
   400 次“用 N 个样本估计梯度”的结果的分布，竖线是精确的期望梯度。
   下图：单个样本的方差随 b 的变化。实线 V(b) 是四维之和（与验证 C-c 的公式相同），最低点是 b*；
   虚线 V_k(b) 是上图那一维，最低点是这一维自己的最优基线 b*_k。上图的宽窄跟着虚线走，不是跟着实线走。
   抽样时对所有 b 使用同一批样本（公共随机数），拖动 b 时看到的变化只来自 b 本身。 */
import { h } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';
import { makeRng, softmax, dot, histogram, stairs, nextId, appendixRoot } from './appendix-common.js';

export const PRESETS = {
  uniform: { label: '均匀策略：θ = (0, 0, 0, 0)，r = (0, 0, 1, 0)', theta: [0, 0, 0, 0], r: [0, 0, 1, 0] },
  skewed: { label: '不均匀策略：θ = (1, 0, 0, −1)，r = (0, 1, 0, 1)', theta: [1, 0, 0, -1], r: [0, 1, 0, 1] },
};

/** 精确量：π、s(a) = e_a − π、∇J、‖s(a)‖²、E[r]、b*、单样本总方差 V(b)，
    以及第 c 维的方差 V_c(b) 与这一维自己的最优基线 b*_c = E[r s_c²] / E[s_c²]。 */
export function bandit({ theta, r }) {
  const pi = softmax(theta);
  const S = pi.map((_, a) => pi.map((p, j) => (a === j ? 1 : 0) - p));
  const grad = pi.map((_, j) => pi.reduce((acc, p, a) => acc + p * r[a] * S[a][j], 0));
  const s2 = S.map((row) => dot(row, row));
  const meanR = dot(pi, r);
  const bStar = pi.reduce((acc, p, a) => acc + p * r[a] * s2[a], 0) / dot(pi, s2);
  const totalVar = (b) => pi.reduce((acc, p, a) => acc + p * (r[a] - b) ** 2 * s2[a], 0) - dot(grad, grad);
  const compVar = (b, c) => pi.reduce((acc, p, a) => acc + p * ((r[a] - b) * S[a][c]) ** 2, 0) - grad[c] ** 2;
  const compStar = (c) => pi.reduce((acc, p, a) => acc + p * r[a] * S[a][c] ** 2, 0)
    / pi.reduce((acc, p, a) => acc + p * S[a][c] ** 2, 0);
  return { pi, S, grad, s2, meanR, bStar, totalVar, compVar, compStar };
}

const B_MIN = -1, B_MAX = 2;
const BINS = 41;   // 奇数个区间：真梯度正好落在中间那个区间的中点，竖线不会被柱子的边线盖住
const SUB = ['₀', '₁', '₂', '₃'];

/** 直方图的横轴范围：以真梯度为中心、左右各 4 个标准差，区间宽度为 8sd/40。 */
export function histRange(center, sd) {
  const w = (8 * sd) / (BINS - 1);
  return [center - (BINS / 2) * w, center + (BINS / 2) * w];
}

/** 把 b 显示成最多 4 位小数（滑块的 0.01 步长与 b* 这样的精确值都能看清）。 */
const fmtB = (x) => String(+x.toFixed(4));

export function mount(root, config = {}) {
  appendixRoot(root);
  let preset = PRESETS[config.preset] ? config.preset : 'uniform';
  let n = config.n ?? 8;
  const topArm = (p) => PRESETS[p].r.indexOf(Math.max(...PRESETS[p].r));   // 奖励最高的那一臂
  let comp = Number.isInteger(config.comp) && config.comp >= 0 && config.comp < 4 ? config.comp : topArm(preset);
  let b = 0;          // 当前的基线；“b = b*”等按钮给出精确值，不受滑块步长的限制
  let seed = 1;
  const M = 400;

  const presetSel = h('select', { 'aria-label': '选择策略和奖励' },
    Object.entries(PRESETS).map(([k, v]) => h('option', { value: k, selected: k === preset }, v.label)));
  const nSel = h('select', { 'aria-label': '每次估计用几个样本' },
    [1, 8, 64].map((v) => h('option', { value: v, selected: v === n }, `N = ${v}`)));
  const compSel = h('select', { 'aria-label': '上图画梯度的哪一维' },
    [0, 1, 2, 3].map((k) => h('option', { value: k, selected: k === comp }, `上图：第 ${k} 号臂那一维`)));
  const slider = h('input', { type: 'range', min: B_MIN, max: B_MAX, step: 0.01, value: b, 'aria-label': '基线 b' });
  const bOut = h('output', { style: 'min-width:4.5em;font-variant-numeric:tabular-nums' });
  const setB = (value) => { b = value; slider.value = String(value); render(); };
  const jump = (label, get) => h('button', { type: 'button', onclick: () => setB(get()) }, label);
  const compJump = jump('', () => model.compStar(comp));
  const stats = h('div', { class: 'w-label', 'aria-live': 'polite', style: 'margin:.3rem 0 .5rem;line-height:1.7' });
  const histBox = h('div');
  const varBox = h('div', { style: 'margin-top:.4rem' });
  const sliderId = nextId('w-appendix-c-b');
  slider.id = sliderId;
  root.append(
    h('div', { class: 'w-row' }, presetSel, nSel, compSel),
    h('div', { class: 'w-row' }, h('label', { for: sliderId }, 'b = ', bOut), slider,
      jump('b = 0', () => 0), jump('b = E[r]', () => model.meanR), jump('b = b*', () => model.bStar), compJump,
      h('button', { type: 'button', onclick: () => { seed += 1; resample(); render(); } }, '重新抽样')),
    stats, histBox, varBox);

  // 两张图只建一次，换策略、样本数或维度时用 update() 换数据
  const placeholder = [{ name: '', points: [[0, 0], [1, 0]] }];
  const hist = lineChart(histBox, { series: placeholder, yLabel: `${M} 次估计的分布（密度）`, height: 200, table: false });
  const varChart = lineChart(varBox, {
    series: placeholder, xLabel: '基线 b',
    yLabel: '单个样本的方差：实线是四维之和 V(b)，虚线是上图这一维（没有标注的竖线是当前的 b）',
    height: 220, yMin: 0, table: true,
  });

  let model, arms, xRange;
  const resample = () => {
    const rng = makeRng(seed * 104729 + n);
    arms = Array.from({ length: M }, () => Array.from({ length: n }, () => rng.categorical(model.pi)));
  };
  const setup = () => {
    model = bandit(PRESETS[preset]);
    // 横轴范围按滑块两端（这一维方差最大处）定好，拖动 b 时坐标不动，宽窄的变化一目了然
    const sd = Math.sqrt(Math.max(model.compVar(B_MIN, comp), model.compVar(B_MAX, comp)) / n);
    xRange = histRange(model.grad[comp], sd);
    resample();
    const bs = Array.from({ length: 61 }, (_, i) => B_MIN + (i * (B_MAX - B_MIN)) / 60);
    varChart.update({ series: [
      { name: 'V(b)', points: bs.map((x) => [x, model.totalVar(x)]) },
      { name: `V${SUB[comp]}(b)`, points: bs.map((x) => [x, model.compVar(x, comp)]), dash: '5 4' },
    ] });
    compJump.textContent = `b = b*${SUB[comp]}`;
  };

  const render = () => {
    bOut.textContent = fmtB(b);
    slider.setAttribute('aria-valuetext', `b = ${fmtB(b)}`);
    const { S } = model;
    const { r } = PRESETS[preset];
    const est = arms.map((group) => group.reduce((acc, a) => acc + (r[a] - b) * S[a][comp], 0) / n);
    const mean = est.reduce((acc, x) => acc + x, 0) / M;
    const spread = est.reduce((acc, x) => acc + (x - mean) ** 2, 0) / M;
    const bins = histogram(est, xRange[0], xRange[1], BINS);
    const top = Math.max(...bins.map(([, d]) => d));
    hist.update({
      series: [{ name: '估计值的分布', points: stairs(bins) }],
      xLabel: `第 ${comp} 号臂那一维的梯度估计`, vLines: [{ x: model.grad[comp], label: '真梯度' }],
      yMax: 1.3 * top,   // 留出顶部的空白：“真梯度”几个字不压在最高的那根柱子上
    });
    const bk = model.compStar(comp);
    // 当前 b 的竖线不加文字：它可能贴着右端，文字会伸出图外；图的标题说明了它是什么
    varChart.update({
      vLines: [{ x: b }, { x: model.bStar, label: 'b*' }, { x: bk, label: `b*${SUB[comp]}`, labelY: 'bottom' }],
      refLines: [{ y: model.totalVar(model.bStar), label: `V(b) 的最小值 ${fmtNum(model.totalVar(model.bStar))}` }],
    });
    stats.replaceChildren(
      h('div', {}, `精确的期望梯度 ∇J = (${model.grad.map((g) => g.toFixed(4)).join(', ')})，与 b 无关。`),
      h('div', {}, `四维之和（单个样本）：V(b) = ${model.totalVar(b).toFixed(4)}；`
        + `最低点 b* = ${model.bStar.toFixed(4)}，为 ${model.totalVar(model.bStar).toFixed(4)}。E[r] = ${model.meanR.toFixed(4)}。`),
      h('div', {}, `上图这一维（第 ${comp} 号臂）：自己的最优基线 b*${SUB[comp]} = ${bk.toFixed(4)}；`
        + `${M} 次估计（每次 ${n} 个样本）的平均 ${mean.toFixed(4)}，方差 ${spread.toFixed(5)}，`
        + `理论值 V${SUB[comp]}(b)/N = ${(model.compVar(b, comp) / n).toFixed(5)}。`));
  };

  presetSel.addEventListener('change', () => { preset = presetSel.value; setup(); render(); });
  nSel.addEventListener('change', () => { n = Number(nSel.value); setup(); render(); });
  compSel.addEventListener('change', () => { comp = Number(compSel.value); setup(); render(); });
  slider.addEventListener('input', () => { b = Number(slider.value); render(); });
  setup();
  render();
}
