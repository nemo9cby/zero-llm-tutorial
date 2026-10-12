/* 图：81 道加法题答对的概率（官方 SFT / GRPO checkpoint 的精确答案分布，data/ch03/answer_dists.json），
   以及 pass@k = mean_q [1 − (1 − p_q)^k] 随 k 的变化。config: { src } */
import { h } from './_lib.js';
import { heatmap, lineChart } from './_plot.js';
import { injectStyle, loadJSON, segmented } from './ch03-common.js';

export const passAtK = (ps, k) => ps.reduce((a, p) => a + (1 - (1 - p) ** k), 0) / ps.length;

export async function mount(root, config) {
  injectStyle();
  const data = await loadJSON(root, config.src);
  if (!data) return;
  root.classList.add('w-ch03');
  const truth = data.prompts.map((s) => Number(s[0]) + Number(s[2]));
  const pc = { sft: data.sft.map((r, k) => r[truth[k]]), grpo: data.grpo.map((r, k) => r[truth[k]]) };
  const grid = (ps) => Array.from({ length: 9 }, (_, i) => ps.slice(i * 9, i * 9 + 9));
  const views = [
    { name: 'SFT 模型', values: grid(pc.sft), scale: 'seq', domain: [0, 1] },
    { name: 'GRPO 模型', values: grid(pc.grpo), scale: 'seq', domain: [0, 1] },
    { name: 'GRPO − SFT', values: grid(pc.grpo.map((p, k) => p - pc.sft[k])), scale: 'div', domain: [-1, 1] },
  ];
  const labels = Array.from({ length: 9 }, (_, i) => String(i + 1));
  const heatBox = h('div', { style: 'max-width:30rem' });
  const note = h('div', { class: 'stat', 'aria-live': 'polite' });
  let hm = null;
  const show = (i) => {
    const v = views[i];
    const opts = { values: v.values, rowLabels: labels, colLabels: labels, scale: v.scale, domain: v.domain,
      cellFormat: (x) => (Math.abs(x) < 0.005 ? '0' : x.toFixed(2)), size: 40, ariaLabel: `${v.name}：每道题答对的概率` };
    heatBox.classList.toggle('hm-seq', v.scale === 'seq');
    if (!hm) hm = heatmap(heatBox, opts); else hm.update(opts);
    const ps = i === 0 ? pc.sft : i === 1 ? pc.grpo : null;
    note.textContent = ps
      ? `${v.name}：平均 P(答对) ${(ps.reduce((a, b) => a + b, 0) / 81).toFixed(3)}；P(答对) < 0.01 的题 ${ps.filter((p) => p < 0.01).length} 道。行 = 第一个加数，列 = 第二个加数。`
      : '蓝色：GRPO 之后答对的概率降低了；红色：提高了。';
  };
  root.append(segmented(views.map((v) => v.name), 0, show, '选择模型'), heatBox, note);
  show(0);

  const ks = [1, 2, 4, 8, 16, 32, 64];
  const lineBox = h('div');
  root.append(h('div', { class: 'w-label', style: 'margin-top:1rem' }, 'pass@k：采样 k 次至少答对一次的概率（横轴是 log₂ k）'), lineBox);
  lineChart(lineBox, {
    series: [{ name: 'SFT', points: ks.map((k) => [Math.log2(k), passAtK(pc.sft, k)]) },
      { name: 'GRPO', points: ks.map((k) => [Math.log2(k), passAtK(pc.grpo, k)]) }],
    xLabel: 'k', xFormat: (x) => String(2 ** Math.round(x)), yLabel: 'pass@k', yMin: 0.4, yMax: 1, height: 220,
  });
}
