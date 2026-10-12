/* 附录 D 的交互图：偏差校正。
   config: { beta1: 0.9, beta2: 0.999, mu: 0.5, sigma: 1.0 }
   上图：同一串带噪声的梯度 g_t ~ N(μ, σ²)，未校正的 m_t、校正后的 m̂_t 与 E[m_t] = (1 − β₁ᵗ)μ，
         以及单次 m̂_t 的 μ ± σ/√n_eff(t)（附录 D.5 的有效样本数）；
   下图：不校正时更新量是校正后的 f(t) = (1 − β₁ᵗ)/√(1 − β₂ᵗ) 倍（验证 D-d），横轴是对数刻度。 */
import { h } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';
import { makeRng, nextId, appendixRoot } from './appendix-common.js';

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

/** 有效样本数 n_eff(t) = 1/Σw_i²（附录 D.5 的公式，验证 D-g）；单次 m̂_t 的标准差是 σ/√n_eff(t)。 */
export const nEff = (t, beta) => ((1 - beta ** t) * (1 + beta)) / ((1 - beta) * (1 + beta ** t));
export const mHatSd = (t, beta, sigma) => sigma / Math.sqrt(nEff(t, beta));

/** f(t) 在 t = 1…tMax 上的最大值与位置，以及从第几步起 f(t) 一直留在 1 ± 0.01 以内
    （到 tMax 还没做到时为 null）。β₁ 不比 β₂ 小时，f(t) 从下方逼近 1，最大值没有意义，只看 settle。 */
export function ratioSummary(beta1, beta2, tMax = 100000) {
  let best = -Infinity, at = 1, lastOut = 0;
  for (let t = 1; t <= tMax; t++) {
    const f = uncorrectedRatio(t, beta1, beta2);
    if (f > best) { best = f; at = t; }
    if (Math.abs(f - 1) >= 0.01) lastOut = t;
  }
  return { first: uncorrectedRatio(1, beta1, beta2), best, at, settle: lastOut < tMax ? lastOut + 1 : null, tMax };
}

/** 交互图下方的一句话总结。 */
export function ratioText(s) {
  const parts = [`f(1) = ${s.first.toFixed(2)}`];
  parts.push(s.best > 1.01
    ? `最大 ${s.best.toFixed(2)}，在第 ${fmtNum(s.at)} 步`
    : 'f(t) 从未超过 1.01：这组 β 下，不校正的步子不会偏大（f < 1 时反而偏小）');
  parts.push(s.settle
    ? `第 ${fmtNum(s.settle)} 步之后一直在 1 ± 0.01 以内`
    : `${fmtNum(s.tMax)} 步之内都没有稳定在 1 ± 0.01 以内`);
  return `${parts.join('；')}。`;
}

const T = 60;

export function mount(root, config = {}) {
  appendixRoot(root);
  let beta1 = config.beta1 ?? 0.9;
  let beta2 = config.beta2 ?? 0.999;
  let sigma = config.sigma ?? 1.0;
  const mu = config.mu ?? 0.5;
  // 默认这一串梯度是“典型”的：单次的 m̂_t 约七成时间落在 ±1 个标准差以内，几次穿过 μ。
  // 一路偏在一边的一串（例如种子 3）容易被误读成校正后还有偏差；“换一串梯度”可以看到各种情况。
  let seed = 12;

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

  // 1 … 约 8 万步，对数等距（止于 10^4.9：最右边的刻度是 10,000，标签不会伸出图外；β₂ = 0.9999 时 f(t) 也已回落到 1 附近）
  const ts = [...new Set(Array.from({ length: 119 }, (_, i) => Math.round(10 ** (i / 24))))];
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
    // 两条细虚线：单次 m̂_t 的 μ ± 1 个标准差（σ/√n_eff(t)），说明单次的值本来就会在 μ 上下漂移
    const sd = path.map((_, i) => mHatSd(i + 1, beta1, sigma));
    const series = [
      { name: 'm_t（未校正，单次）', points: path.map(([m], i) => [i + 1, m]) },
      { name: 'm̂_t（校正后，单次）', points: path.map(([, mh], i) => [i + 1, mh]) },
      { name: 'E[m_t] = (1−β₁ᵗ)μ', points: path.map((_, i) => [i + 1, (1 - beta1 ** (i + 1)) * mu]) },
      { name: 'm̂_t（500 次平均）', points: avg.map((v, i) => [i + 1, v]) },
      { name: 'μ + 1 个标准差（单次 m̂_t）', points: sd.map((s, i) => [i + 1, mu + s]), dash: '2 4', width: 1.25 },
      { name: 'μ − 1 个标准差', points: sd.map((s, i) => [i + 1, mu - s]), dash: '2 4', width: 1.25 },
    ];
    const emaOpts = { series, xLabel: '步数 t', yLabel: '一阶矩的估计', height: 220,
      refLines: [{ y: mu, label: `真实均值 μ = ${mu}` }], table: false };
    if (ema) ema.update(emaOpts); else ema = lineChart(emaBox, emaOpts);

    const ratioOpts = {
      series: [{ name: 'f(t)', points: ts.map((t) => [t, uncorrectedRatio(t, beta1, beta2)]) }], logX: true,
      xLabel: '步数 t（对数刻度）', yLabel: '不校正的更新量 ÷ 校正后的更新量', height: 200, yMin: 0,
      refLines: [{ y: 1, label: '校正后 = 1' }], table: true,
    };
    if (ratio) ratio.update(ratioOpts); else ratio = lineChart(ratioBox, ratioOpts);

    stats.textContent = ratioText(ratioSummary(beta1, beta2));
  };
  b1.addEventListener('input', () => { beta1 = Number(b1.value); render(); });
  b2.addEventListener('change', () => { beta2 = Number(b2.value); render(); });
  sig.addEventListener('input', () => { sigma = Number(sig.value); render(); });
  again.addEventListener('click', () => { seed += 1; render(); });
  render();
}
