/* E01 的证据：GRPO 原样运行 vs 更新前 model.eval() 的 greedy 准确率曲线。
   config: { data: "data/errata/grpo_dropout_runs.json" }
   数据来自研究阶段的记录（research/data/ch03_grpo_metrics_*.jsonl，由 verify/test_reference.py 校验）。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';

const SEEDS = [
  { seed: 0, runs: ['base', 'nodrop'] },
  { seed: 1, runs: ['base_s1', 'nodrop_s1'] },
];
const SFT_BASELINE = 54.3; // 44 / 81, verify/test_reference.py::test_e15_sft_greedy_baseline_is_44_of_81

export function seriesFor(data, seed) {
  const pick = SEEDS.find((s) => s.seed === seed) || SEEDS[0];
  return pick.runs.map((name) => {
    const r = data.runs[name];
    // short names: they are drawn as end labels next to the curves
    return { name: name.startsWith('base') ? '原样' : '加 eval()', points: r.acc_it.map((it, i) => [it, r.greedy_acc[i]]) };
  });
}

export async function mount(root, config) {
  let data;
  try {
    const res = await fetch(config.data || 'data/errata/grpo_dropout_runs.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `数据加载失败：${err.message}`));
    return;
  }
  let seed = 0;
  const buttons = SEEDS.map(({ seed: s }) => h('button', {
    type: 'button', 'aria-pressed': String(s === seed), class: 'w-grpo-seed',
  }, `seed ${s}`));
  const bar = h('div', { class: 'w-row', role: 'group', 'aria-label': '选择随机种子' },
    h('span', { class: 'w-label' }, '随机种子：'), ...buttons);
  const chartBox = h('div');
  root.append(bar, chartBox, h('p', { class: 'w-label' }, data.note));
  const chart = lineChart(chartBox, {
    series: seriesFor(data, seed), xLabel: '迭代', yLabel: 'greedy 准确率（%，81 道题）',
    yMin: 50, yMax: 100, refLines: [{ y: SFT_BASELINE, label: `SFT 起点 ${SFT_BASELINE}%` }],
    yFormat: (v) => `${Math.round(v)}`,
  });
  buttons.forEach((b, i) => b.addEventListener('click', () => {
    seed = SEEDS[i].seed;
    buttons.forEach((x, j) => x.setAttribute('aria-pressed', String(i === j)));
    chart.update({ series: seriesFor(data, seed) });
  }));
  root.append(h('style', {}, `
.w-grpo-seed { font: inherit; font-size: .85rem; padding: .15rem .65rem; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--rule); background: transparent; color: var(--ink-2); }
.w-grpo-seed[aria-pressed="true"] { background: var(--ai); border-color: var(--ai); color: var(--paper); }
`));
}
