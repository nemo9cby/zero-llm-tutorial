/* 图：ReLU 与 GELU（2.9.3 节，复现 ch02/graph.py 的图）。勾选“放大差值”显示 tanh 近似与 erf 精确版之差 × 1000。
   结论（最大差 4.73e-4、最低点 −0.170）由验证 2.9-b 断言，JS 测试核对本组件的数值与之一致。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle, range } from './ch02-util.js';

/** erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). */
export function erf(x) {
  const s = Math.sign(x), a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
export const geluErf = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
export const geluTanh = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
export const relu = (x) => Math.max(0, x);

export function mount(root) {
  injectStyle();
  const box = h('input', { type: 'checkbox', 'aria-label': '放大显示 tanh 近似与 erf 版之差' });
  const note = h('p', { class: 'w-label', 'aria-live': 'polite' });
  const chartBox = h('div');
  root.append(h('div', { class: 'w-ch02-controls' }, h('label', {}, box, '放大差值（tanh 版 − erf 版）× 1000')), chartBox, note);
  const xs = range(301).map((i) => -3 + i * 0.02);
  const xsWide = range(1201).map((i) => -6 + i * 0.01);
  const curves = {
    series: [
      { name: 'ReLU', points: xs.map((x) => [x, relu(x)]) },
      { name: 'GELU（erf，codebot）', points: xs.map((x) => [x, geluErf(x)]) },
      { name: 'GELU（tanh 近似，graph.py）', points: xs.map((x) => [x, geluTanh(x)]) },
    ],
    xLabel: 'x', yLabel: 'f(x)', yMin: -0.5, yMax: 3, height: 260,
  };
  const diffs = {
    series: [{ name: '(tanh − erf) × 1000', points: xsWide.map((x) => [x, 1000 * (geluTanh(x) - geluErf(x))]) }],
    xLabel: 'x', yLabel: '差值 × 1000', yMin: undefined, yMax: undefined, height: 260,
  };
  const chart = lineChart(chartBox, curves);
  const render = () => {
    chart.update(box.checked ? diffs : curves);
    note.textContent = box.checked
      ? '两个版本最多相差约 4.7e-4（在 x ≈ ±2.7 处），差值是 x 的偶函数。'
      : 'GELU 在负半轴有一个凹陷，最低点约为 −0.17（x ≈ −0.75）；x 大于 2 之后几乎与 ReLU 重合。三条曲线里，两条 GELU 在这个尺度上看不出区别。';
  };
  box.addEventListener('change', render);
  render();
}
