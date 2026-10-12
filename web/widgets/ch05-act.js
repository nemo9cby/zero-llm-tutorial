/* 图：ReLU、GELU（graph.py 的近似式）与 Swish（SiLU）。可切换函数值与导数，并标出 Swish 的最小值。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle, radioButtons } from './ch05-common.js';

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
export const relu = (x) => Math.max(0, x);
export const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
export const swish = (x) => x * sigmoid(x);
const d = (f) => (x) => (f(x + 1e-5) - f(x - 1e-5)) / 2e-5;     // central difference
export const derivatives = { relu: (x) => (x > 0 ? 1 : 0), gelu: d(gelu), swish: d(swish) };

/** Minimum of f on [lo, hi] by golden-section search (f unimodal there). */
export function argmin(f, lo, hi, iters = 100) {
  const g = (Math.sqrt(5) - 1) / 2;
  let a = lo, b = hi;
  for (let i = 0; i < iters; i++) {
    const c = b - g * (b - a), e = a + g * (b - a);
    if (f(c) < f(e)) b = e; else a = c;
  }
  const x = (a + b) / 2;
  return [x, f(x)];
}

export function mount(root) {
  injectStyle();
  const box = h('div', { class: 'w-ch05' });
  root.append(box);
  const xs = Array.from({ length: 121 }, (_, i) => +(-3 + i * 0.05).toFixed(2));
  const [sx, sv] = argmin(swish, -3, 0);
  const [gx, gv] = argmin(gelu, -3, 0);
  let mode = 'f';
  const chartBox = h('div');
  const info = h('div', { class: 'readout' },
    'Swish 的最小值 ', h('b', {}, sv.toFixed(4)), `（x = ${sx.toFixed(4)}）；`,
    'GELU（近似式）的最小值 ', h('b', {}, gv.toFixed(4)), `（x = ${gx.toFixed(4)}）`);
  const sel = radioButtons([['f', '函数值 f(x)'], ['df', '导数 f′(x)']], mode, (v) => { mode = v; chart.update(opts()); }, '显示函数值还是导数');
  box.append(h('div', { class: 'ctl' }, sel), chartBox, info);
  const opts = () => {
    const fs = mode === 'f' ? { ReLU: relu, GELU: gelu, Swish: swish }
      : { ReLU: derivatives.relu, GELU: derivatives.gelu, Swish: derivatives.swish };
    return {
      series: Object.entries(fs).map(([name, f]) => ({ name, points: xs.map((x) => [x, +f(x).toFixed(5)]) })),
      xLabel: 'x', yLabel: mode === 'f' ? 'f(x)' : 'f′(x)', height: 260, xTicks: [-3, -2, -1, 0, 1, 2, 3],
      yMin: mode === 'f' ? -0.5 : -0.2, yMax: mode === 'f' ? 3 : 1.2,
      refLines: [{ y: 0 }],
      vLines: [{ x: sx, label: mode === 'f' ? 'Swish 最低点' : 'Swish′ = 0' }],
      ariaLabel: mode === 'f' ? 'ReLU、GELU、Swish 的函数曲线' : 'ReLU、GELU、Swish 的导数曲线',
    };
  };
  const chart = lineChart(chartBox, opts());
}
