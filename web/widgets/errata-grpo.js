/* E01 的证据：GRPO 原样运行 vs 更新前 model.eval() 的 greedy 准确率曲线。
   config: {
     runs: { base: "data/runs/codebot-3060ti/09_grpo.metrics.jsonl",          // 本机 RTX 3060 Ti，原书脚本原样
             fix: "data/runs/codebot-grpo-evalfix-3060ti/09_grpo.metrics.jsonl" }, // 同一 SFT 起点，只加 model.eval()
     data: "data/errata/grpo_dropout_runs.json"                                // 研究阶段的 CPU 对照（两个 seed）
   }
   默认显示两次完整的 GPU 运行；另外两个选项是研究阶段的 CPU 记录（verify/test_reference.py 校验）。
   横轴是“已完成的迭代次数”：脚本在每次迭代的两次更新之后才评估（勘误 E15），所以点画在 1、11、21……处。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';

const SFT_BASELINE = 54.3; // 44 / 81：两个 SFT 起点在 CPU 上都是 44 道（test_e15_*、test_e01_recorded_*）
const CPU_SEEDS = [
  { key: 'cpu0', label: 'CPU 对照 seed 0', runs: ['base', 'nodrop'] },
  { key: 'cpu1', label: 'CPU 对照 seed 1', runs: ['base_s1', 'nodrop_s1'] },
];

/** Parse a run's metrics JSONL: one {"it", "loss", "acc"} object per line. */
export function parseJsonl(text) {
  return text.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
}

/** Evaluation points of a recorded 09_grpo.py run: the progress bar shows a new accuracy after iterations
 *  1, 11, 21, … (evaluation at loop index 0, 10, 20, … happens after that iteration's two updates). */
export function evalPoints(rows) {
  return rows.filter((r) => r.it >= 1 && (r.it - 1) % 10 === 0).map((r) => [r.it, r.acc]);
}

export function gpuSeries(baseRows, fixRows) {
  return [{ name: '原样', points: evalPoints(baseRows) }, { name: '加 eval()', points: evalPoints(fixRows) }];
}

/** Research-phase CPU runs: greedy_acc logged after the updates of loop index acc_it → x = acc_it + 1. */
export function cpuSeries(data, key) {
  const pick = CPU_SEEDS.find((s) => s.key === key) || CPU_SEEDS[0];
  return pick.runs.map((name) => {
    const r = data.runs[name];
    return { name: name.startsWith('base') ? '原样' : '加 eval()', points: r.acc_it.map((it, i) => [it + 1, r.greedy_acc[i]]) };
  });
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.text();
}

export async function mount(root, config) {
  let gpu, cpu;
  try {
    const runs = config.runs || {};
    const [b, f, c] = await Promise.all([
      fetchText(runs.base || 'data/runs/codebot-3060ti/09_grpo.metrics.jsonl'),
      fetchText(runs.fix || 'data/runs/codebot-grpo-evalfix-3060ti/09_grpo.metrics.jsonl'),
      fetchText(config.data || 'data/errata/grpo_dropout_runs.json'),
    ]);
    gpu = gpuSeries(parseJsonl(b), parseJsonl(f));
    cpu = JSON.parse(c);
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `数据加载失败：${err.message}`));
    return;
  }
  const OPTIONS = [
    { key: 'gpu', label: 'GPU，原书脚本 500 次', series: () => gpu,
      note: '本机 RTX 3060 Ti 上的两次完整运行（运行记录 codebot-3060ti 与 codebot-grpo-evalfix-3060ti），从同一个 SFT 检查点出发；各一次运行，脚本没有设随机种子。' },
    ...CPU_SEEDS.map((s) => ({ key: s.key, label: s.label, series: () => cpuSeries(cpu, s.key), note: cpu.note })),
  ];
  let current = OPTIONS[0];
  const buttons = OPTIONS.map((o) => h('button', {
    type: 'button', 'aria-pressed': String(o === current), class: 'w-grpo-seed',
  }, o.label));
  const bar = h('div', { class: 'w-row', role: 'group', 'aria-label': '选择要看的运行' },
    h('span', { class: 'w-label' }, '运行：'), ...buttons);
  const chartBox = h('div');
  const note = h('p', { class: 'w-label' });
  const firstNote = h('p', { class: 'w-label' },
    '每条曲线的第一个点都已经在两次更新之后（勘误 E15），所以不等于虚线标出的 SFT 起点。');
  root.append(bar, chartBox, note, firstNote);
  const chart = lineChart(chartBox, {
    series: current.series(), xLabel: '已完成的迭代次数', yLabel: 'greedy 准确率（%，81 道题）',
    yMin: 50, yMax: 100, refLines: [{ y: SFT_BASELINE, label: `SFT 起点 ${SFT_BASELINE}%` }],
    yFormat: (v) => `${Math.round(v)}`,
  });
  note.textContent = current.note;
  buttons.forEach((b, i) => b.addEventListener('click', () => {
    current = OPTIONS[i];
    buttons.forEach((x, j) => x.setAttribute('aria-pressed', String(i === j)));
    chart.update({ series: current.series() });
    note.textContent = current.note;
  }));
  root.append(h('style', {}, `
.w-grpo-seed { font: inherit; font-size: .85rem; padding: .15rem .65rem; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--rule); background: transparent; color: var(--ink-2); }
.w-grpo-seed[aria-pressed="true"] { background: var(--ai); border-color: var(--ai); color: var(--paper); }
`));
}
