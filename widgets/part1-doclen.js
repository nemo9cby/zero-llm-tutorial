/* 图：文档长度（token 数）的累计分布，竖线是模型的上下文长度。用在第 I–III 部的扉页。
   Lengths come from data/<part>/doclen.json: tokens between two <|endoftext|> in the book's .bin files
   (verify/test_part{1,2,3}.py recompute them).
   Each dataset keeps its colour on every page: the series are drawn in the fixed order TinyCodes (part1),
   TinyStories (part2), OpenWebText (part3), and every page that shows a later dataset also shows the earlier ones.
   config: { datasets: [{ name, file }], xMax: 1024 } */
import { h, fmt } from './_lib.js';
import { lineChart } from './_plot.js';

const ORDER = ['data/part1/doclen.json', 'data/part2/doclen.json', 'data/part3/doclen.json'];

async function load(file) {
  const r = await fetch(file);
  if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
  return r.json();
}

export async function mount(root, config) {
  let sets;
  try {
    sets = await Promise.all((config.datasets || []).map(async (d) => ({ ...d, data: await load(d.file) })));
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `无法读取文档长度数据（${err.message}）。`));
    return;
  }
  const rank = (s) => (ORDER.includes(s.file) ? ORDER.indexOf(s.file) : ORDER.length);
  sets.sort((a, b) => rank(a) - rank(b));
  const xMax = config.xMax || 1024;
  const steps = 96;
  const series = sets.map((s) => {
    const sorted = Float64Array.from(s.data.lengths).sort();
    const n = sorted.length;
    let k = 0;
    const points = [];
    for (let q = 0; q <= steps; q++) {
      const x = Math.round((xMax * q) / steps);
      while (k < n && sorted[k] <= x) k++;
      points.push([x, (100 * k) / n]);
    }
    return { name: s.name, points };
  });
  const ctxs = [...new Set(sets.map((s) => s.data.context_len))].filter((c) => c <= xMax);
  const chart = h('div');
  root.append(chart);
  lineChart(chart, {
    series, xLabel: '文档长度（token）', yLabel: '不超过该长度的文档占比（%）', yMin: 0, yMax: 100, height: 240,
    // labels at the bottom: right of each line the cumulative curves are already high, so the text stays clear
    vLines: ctxs.map((c) => ({ x: c, label: `上下文 ${fmt(c)}`, labelY: 'bottom' })),
    yFormat: (v) => `${Math.round(v)}`,
  });
  root.append(...sets.map((s) => {
    const sm = s.data.summary, ctx = s.data.context_len;
    const within = (100 * (1 - sm.over_ctx_frac)).toFixed(1);
    return h('p', { class: 'w-label', style: 'margin:.3rem 0 0' },
      `${s.name}：${fmt(sm.n)} 篇，中位数 ${fmt(sm.median)} 个 token，${within}% 不超过上下文 ${fmt(ctx)}，`
      + `最长 ${fmt(sm.max)}。`);
  }));
}
