/* 图：P(无信号) = p^G + (1−p)^G 的热力图，以及“题目难度分布”如何让平均奖励与零方差比例背离。
   真实分布来自官方 SFT / GRPO checkpoint 在 81 道题上答对的精确概率（data/ch03/answer_dists.json）。
   config: { src } */
import { h } from './_lib.js';
import { heatmap, barChart } from './_plot.js';
import { injectStyle, loadJSON, pNoSignal, segmented } from './ch03-common.js';

const GS = [1, 2, 4, 8, 16, 32];
const PS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

export async function mount(root, config) {
  injectStyle();
  const data = await loadJSON(root, config.src);
  if (!data) return;
  root.classList.add('w-ch03');
  const truth = data.prompts.map((s) => Number(s[0]) + Number(s[2]));
  const pc = (name) => data[name].map((row, k) => row[truth[k]]);
  const dists = [
    { name: '均匀（每题 p = 0.5）', ps: Array(81).fill(0.5) },
    { name: '两极（一半必错，一半必对）', ps: [...Array(41).fill(0), ...Array(40).fill(1)] },
    { name: 'CodeBot：SFT 起点', ps: pc('sft') },
    { name: 'CodeBot：GRPO 终点', ps: pc('grpo') },
  ];

  const heatBox = h('div', { class: 'hm-seq' });
  root.append(h('div', { class: 'w-label' }, '每格：一组 G 个回答全对或全错（没有学习信号）的概率。横轴是这道题的正确率 p，纵轴是组大小 G。'), heatBox);
  heatmap(heatBox, {
    values: GS.map((G) => PS.map((p) => pNoSignal(p, G))),
    rowLabels: GS.map((G) => `G=${G}`), colLabels: PS.map((p) => p.toFixed(1)),
    domain: [0, 1], cellFormat: (v) => v.toFixed(2), size: 44, ariaLabel: 'P(无信号) 热力图',
  });

  let di = 2, G = 8;
  const info = h('div', { class: 'stat', 'aria-live': 'polite' });
  const histBox = h('div');
  const segD = segmented(dists.map((d) => d.name), di, (i) => { di = i; render(); }, '题目难度分布');
  const segG = segmented(GS.map((g) => `G=${g}`), GS.indexOf(G), (i) => { G = GS[i]; render(); }, '组大小');
  root.append(h('div', { class: 'w-label', style: 'margin-top:1rem' }, '换一种题目难度的分布，比较“平均奖励”和“没有信号的组的比例”：'),
    segD, h('div', { class: 'ctl' }, segG), info,
    h('div', { class: 'w-label' }, '各正确率区间的题数（0.0 表示 0 ≤ p < 0.1，0.9 表示 0.9 ≤ p ≤ 1）：'), histBox);
  let chart = null;

  function render() {
    const ps = dists[di].ps;
    const meanR = ps.reduce((a, b) => a + b, 0) / ps.length;
    const zero = ps.reduce((a, p) => a + pNoSignal(p, G), 0) / ps.length;
    info.textContent = `${ps.length} 道题：平均奖励 ${meanR.toFixed(3)}；G = ${G} 时，没有学习信号的组的期望比例 ${(zero * 100).toFixed(1)}%。`;
    const bins = Array(10).fill(0);
    for (const p of ps) bins[Math.min(9, Math.floor(p * 10))] += 1;
    const bars = bins.map((n, i) => ({ label: `${(i / 10).toFixed(1)}`, value: n }));
    if (!chart) chart = barChart(histBox, { bars, yLabel: '题数', height: 180, table: false });
    else chart.update({ bars });
  }
  render();
}
