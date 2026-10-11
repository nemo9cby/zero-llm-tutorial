/* 样板间：演示 _plot.js 的三种图。 */
import { h } from './_lib.js';
import { lineChart, barChart, heatmap } from './_plot.js';

export function mount(root) {
  const steps = [...Array(60).keys()];
  const decay = steps.map((i) => [i, 6.9 * Math.exp(-i / 18) + 1.2 + 0.08 * Math.sin(i)]);
  const val = steps.map((i) => [i, 6.9 * Math.exp(-i / 20) + 1.35]);
  const lineBox = h('div');
  const barBox = h('div');
  const heatBox = h('div');
  root.append(lineBox, barBox, heatBox);
  lineChart(lineBox, {
    series: [{ name: 'train', points: decay }, { name: 'valid', points: val }],
    xLabel: 'iteration', yLabel: 'loss（nats）', refLines: [{ y: Math.log(1000), label: 'ln 1000' }],
  });
  barChart(barBox, {
    bars: [{ label: '朴素', value: 64 * 60 }, { label: '去重', value: 300 }, { label: '增量', value: 25 }, { label: '并行', value: 19 }],
    yLabel: '秒', highlight: '并行',
  });
  const n = 6;
  const w = [...Array(n)].map((_, r) => [...Array(n)].map((_, c) => (c <= r ? 1 / (r + 1) : 0)));
  heatmap(heatBox, {
    values: w, rowLabels: [...Array(n)].map((_, i) => `q${i}`), colLabels: [...Array(n)].map((_, i) => `k${i}`),
    domain: [0, 1], masked: (r, c) => c > r, cellFormat: (v) => v.toFixed(2),
  });
}
