/* 图：81 道加法题答对的概率（data/ch03/answer_dists.json 里的精确答案分布：官方 SFT / GRPO checkpoint，
   以及本书 3060 Ti 运行的 checkpoint，键 run_*），以及 pass@k = mean_q [1 − (1 − p_q)^k] 随 k 的变化。
   config: { src } */
import { h } from './_lib.js';
import { heatmap, lineChart } from './_plot.js';
import { injectStyle, loadJSON, segmented } from './ch03-common.js';

export const passAtK = (ps, k) => ps.reduce((a, p) => a + (1 - (1 - p) ** k), 0) / ps.length;

// answer_dists.json 的键 → 名字；run_* 只在数据里有这一行时出现
export const MODELS = [
  { key: 'sft', name: '官方 SFT' },
  { key: 'grpo', name: '官方 GRPO' },
  { key: 'run_sft', name: '本书 SFT' },
  { key: 'run_grpo', name: '本书 GRPO（原样）' },
  { key: 'run_evalfix', name: '本书 GRPO（加 model.eval()）' },
];

/** 每道题答对的概率：answer_dists 的第 k 行、第 (i + j) 列。 */
export function correctProbs(data, key) {
  const truth = data.prompts.map((s) => Number(s[0]) + Number(s[2]));
  return data[key].map((r, k) => r[truth[k]]);
}

export async function mount(root, config) {
  injectStyle();
  const data = await loadJSON(root, config.src);
  if (!data) return;
  root.classList.add('w-ch03');
  const models = MODELS.filter((m) => data[m.key]);
  const pc = Object.fromEntries(models.map((m) => [m.key, correctProbs(data, m.key)]));
  const grid = (ps) => Array.from({ length: 9 }, (_, i) => ps.slice(i * 9, i * 9 + 9));
  const views = [
    ...models.filter((m) => m.key !== 'run_sft').map((m) => ({ name: m.name, key: m.key, values: grid(pc[m.key]), scale: 'seq', domain: [0, 1] })),
    { name: '官方 GRPO − 官方 SFT', values: grid(pc.grpo.map((p, k) => p - pc.sft[k])), scale: 'div', domain: [-1, 1] },
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
    const ps = v.key ? pc[v.key] : null;
    note.textContent = ps
      ? `${v.name}：平均 P(答对) ${(ps.reduce((a, b) => a + b, 0) / 81).toFixed(3)}；P(答对) < 0.01 的题 ${ps.filter((p) => p < 0.01).length} 道。行 = 第一个加数，列 = 第二个加数，格子里的数字就是答对的概率。`
      : '格子里是两个模型答对概率之差：负数（蓝）表示 GRPO 之后降低了，正数（红）表示提高了。';
  };
  root.append(segmented(views.map((v) => v.name), 0, show, '选择模型'), heatBox, note);
  show(0);

  const ks = [1, 2, 4, 8, 16, 32, 64];
  const lineBox = h('div');
  root.append(h('div', { class: 'w-label', style: 'margin-top:1rem' }, 'pass@k：采样 k 次至少答对一次的概率（横轴是 log₂ k）'), lineBox);
  lineChart(lineBox, {
    series: models.map((m) => ({ name: m.name, points: ks.map((k) => [Math.log2(k), passAtK(pc[m.key], k)]) })),
    xLabel: 'k', xFormat: (x) => String(2 ** Math.round(x)), yLabel: 'pass@k', yMin: 0.4, yMax: 1, height: 240,
  });
}
