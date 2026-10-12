/* 附录 D 的交互图：偏差校正。
   config: { beta1: 0.9, beta2: 0.999, mu: 0.5, sigma: 1.0 }
   上图：同一串带噪声的梯度 g_t ~ N(μ, σ²)，未校正的 m_t、校正后的 m̂_t 与 E[m_t] = (1 − β₁ᵗ)μ；
   下图：不校正时更新量是校正后的 f(t) = (1 − β₁ᵗ)/√(1 − β₂ᵗ) 倍（验证 D-d），横轴是对数刻度。 */
import { h } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';
import { makeRng, nextId } from './appendix-common.js';

/** 指数移动平均：返回每一步的 [m_t, m̂_t]（m_0 = 0）。 */
export function emaPath(grads, beta) {
  let m = 0;
  return grads.map((g, i) => {
    m = beta * m + (1 - beta) * g;
    return [m, m / (1 - beta ** (i + 1))];
  });
}

/** 不校正时的更新量 / 校正后的更新量。 */
export const uncorrectedRatio = (t, beta1, beta2) => (1 - beta1 ** t) / Math.sqrt(1 - beta2 ** t);

/** f(t) 在 t = 1…tMax 上的最大值与位置，以及第一次回落到 1.01 以下的步数。 */
export function ratioSummary(beta1, beta2, tMax = 100000) {
  let best = 0, at = 1, settle = null;
  for (let t = 1; t <= tMax; t++) {
    const f = uncorrectedRatio(t, beta1, beta2);
    if (f > best) { best = f; at = t; }
    if (settle === null && t > at && f < 1.01) settle = t;
  }
  return { first: uncorrectedRatio(1, beta1, beta2), best, at, settle };
}

const T = 60;

export function mount(root, config = {}) {
  let beta1 = config.beta1 ?? 0.9;
  let beta2 = config.beta2 ?? 0.999;
  let sigma = config.sigma ?? 1.0;
  const mu = config.mu ?? 0.5;
  let seed = 3;

  const b1Id = nextId('w-appendix-d-b1'), sId = nextId('w-appendix-d-sigma');
  const b1 = h('input', { id: b1Id, type: 'range', min: 0.5, max: 0.99, step: 0.01, value: beta1, 'aria-label': 'β₁' });
  const b1Out = h('output', { style: 'font-variant-numeric:tabular-nums' });
  const b2 = h('select', { 'aria-label': 'β₂' },
    [0.9, 0.99, 0.999, 0.9999].map((v) => h('option', { value: v, selected: v === beta2 }, `β₂ = ${v}`)));
  const sig = h('input', { id: sId, type: 'range', min: 0, max: 2, step: 0.1, value: sigma, 'aria-label': '梯度噪声的标准差 σ' });
  const sigOut = h('output', { style: 'font-variant-numeric:tabular-nums' });
  const again = h('button', { type: 'button' }, '换一串梯度');
  const emaBox = h('div');
  const ratioBox = h('div', { style: 'margin-top:.5rem' });
  const stats = h('div', { class: 'w-label', 'aria-live': 'polite', style: 'margin-top:.3rem' });
  root.append(
    h('div', { class: 'w-row' }, h('label', { for: b1Id }, 'β₁ = ', b1Out), b1, b2),
    h('div', { class: 'w-row' }, h('label', { for: sId }, '噪声 σ = ', sigOut), sig, again),
    emaBox, ratioBox, stats);

  const ts = [...new Set(Array.from({ length: 121 }, (_, i) => Math.round(10 ** (i / 24))))];  // 1 … 10⁵，对数等距
  let ema = null, ratio = null;
  const render = () => {
    b1Out.textContent = beta1.toFixed(2);
    sigOut.textContent = sigma.toFixed(1);
    b1.setAttribute('aria-valuetext', `β₁ = ${beta1.toFixed(2)}`);
    sig.setAttribute('aria-valuetext', `σ = ${sigma.toFixed(1)}`);
    const rng = makeRng(seed);
    const grads = Array.from({ length: T }, () => mu + sigma * rng.normal());
    const path = emaPath(grads, beta1);
    // 再独立跑 500 串梯度，把校正后的 m̂_t 逐步平均：它应当贴着 μ（无偏），单次的那条则很吵
    const avg = new Array(T).fill(0);
    const rngAvg = makeRng(seed + 1000);
    for (let run = 0; run < 500; run++) {
      const g = Array.from({ length: T }, () => mu + sigma * rngAvg.normal());
      emaPath(g, beta1).forEach(([, mh], i) => { avg[i] += mh / 500; });
    }
    const series = [
      { name: 'm_t（未校正，单次）', points: path.map(([m], i) => [i + 1, m]) },
      { name: 'm̂_t（校正后，单次）', points: path.map(([, mh], i) => [i + 1, mh]) },
      { name: 'E[m_t] = (1−β₁ᵗ)μ', points: path.map((_, i) => [i + 1, (1 - beta1 ** (i + 1)) * mu]) },
      { name: 'm̂_t（500 次平均）', points: avg.map((v, i) => [i + 1, v]) },
    ];
    const emaOpts = { series, xLabel: '步数 t', yLabel: '一阶矩的估计', height: 220,
      refLines: [{ y: mu, label: `真实均值 μ = ${mu}` }], table: false };
    if (ema) ema.update(emaOpts); else ema = lineChart(emaBox, emaOpts);

    const ratioOpts = {
      series: [{ name: 'f(t)', points: ts.map((t) => [Math.log10(t), uncorrectedRatio(t, beta1, beta2)]) }],
      xLabel: '步数 t（对数刻度）', yLabel: '不校正的更新量 ÷ 校正后的更新量', height: 200, yMin: 0,
      xFormat: (v) => fmtNum(Math.round(10 ** v)), refLines: [{ y: 1, label: '校正后 = 1' }], table: true,
    };
    if (ratio) ratio.update(ratioOpts); else ratio = lineChart(ratioBox, ratioOpts);

    const s = ratioSummary(beta1, beta2);
    stats.textContent = `f(1) = ${s.first.toFixed(2)}；最大 ${s.best.toFixed(2)}，在第 ${s.at} 步；`
      + (s.settle ? `第 ${fmtNum(s.settle)} 步之后回落到 1.01 以内。` : '十万步之内都没有回落到 1.01 以内。');
  };
  b1.addEventListener('input', () => { beta1 = Number(b1.value); render(); });
  b2.addEventListener('change', () => { beta2 = Number(b2.value); render(); });
  sig.addEventListener('input', () => { sigma = Number(sig.value); render(); });
  again.addEventListener('click', () => { seed += 1; render(); });
  render();
}
