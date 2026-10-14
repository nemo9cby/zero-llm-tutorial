/* 图：去重。左边是一段可编辑的文字按出现顺序切出的预分词，右边是去重后的计数表；
   下方是实测的 Heaps 定律（双对数坐标，虚线是拟合的斜率 β）：语料变大时，不同预分词的个数涨得很慢。
   config: { text: "data/ch04/train_head10k.txt", heaps: "data/ch04/bench.json", chars: 1200 } */
import { h, chip, visible, fmt } from './_lib.js';
import { lineChart } from './_plot.js';
import { pretokenizeCorpus } from './ch04-bpe.js';

const STYLE = `
/* token chips: theme-aware colours (the shared dark-mode .tok filter also darkens the text) */
.w-ch04-dedup .tok { filter: none !important; color: var(--ink) !important; }
.w-ch04-dedup .tok i { color: var(--ink-2) !important; }
.w-ch04-dedup .tok.c0 { background: color-mix(in srgb, var(--ai) 22%, var(--paper)) !important; }
.w-ch04-dedup .tok.c1 { background: color-mix(in srgb, var(--moss) 24%, var(--paper)) !important; }
.w-ch04-dedup .tok.c2 { background: color-mix(in srgb, var(--kincha) 26%, var(--paper)) !important; }
.w-ch04-dedup .tok.c3 { background: color-mix(in srgb, var(--plum) 24%, var(--paper)) !important; }
.w-ch04-dedup .tok.c4 { background: color-mix(in srgb, var(--ai) 40%, var(--paper)) !important; }
.w-ch04-dedup .tok.c5 { background: color-mix(in srgb, var(--shu) 18%, var(--paper)) !important; }
.w-ch04-dedup .tok.plain { background: var(--paper-2) !important; }
.w-ch04-dedup, .w-ch04-dedup * { font-variant-ligatures: none !important; }
.w-ch04-dedup textarea { width: 100%; min-height: 6.5rem; font: 13px/1.5 var(--mono); color: var(--ink); background: var(--paper);
  border: 1px solid var(--rule); border-radius: 4px; padding: .5rem .6rem; resize: vertical; }
.w-ch04-dedup .stats { display: flex; flex-wrap: wrap; gap: .3rem 1.4rem; margin: .5rem 0 .8rem; font-size: .86rem; color: var(--ink-2); }
.w-ch04-dedup .stats b { color: var(--ink); font-variant-numeric: tabular-nums; }
.w-ch04-dedup .cols { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 1.2rem; }
.w-ch04-dedup h4 { margin: 0 0 .3rem; font-size: .82rem; font-weight: 600; color: var(--ink-2); }
.w-ch04-dedup .tokens { max-height: 13.5rem; overflow: hidden; }
.w-ch04-dedup .tok.plain { background: var(--paper-2); }
.w-ch04-dedup table { display: table; width: 100%; margin: 0; font-size: .82rem; }
.w-ch04-dedup td { padding: .12rem .4rem .12rem 0; border-bottom: 1px solid var(--rule); vertical-align: middle; }
.w-ch04-dedup td.n { text-align: right; font: 12px var(--mono); font-variant-numeric: tabular-nums; width: 3.5rem; }
.w-ch04-dedup td.bar span { display: block; height: .55rem; border-radius: 2px; background: var(--ai); opacity: .55; }
.w-ch04-dedup code { font-size: .8rem; white-space: pre; }
.w-ch04-dedup .more { font-size: .78rem; color: var(--ink-3); margin-top: .3rem; }
.w-ch04-dedup .heaps { margin-top: 1.2rem; }
@media (max-width: 640px) { .w-ch04-dedup .cols { grid-template-columns: minmax(0, 1fr); } }
`;

/** Short labels for counts: 4741 → 5K, 2,524,909 → 2.5M, 1,000,000 → 1M (no trailing .0). */
export const compact = (v) => (v >= 1e6 ? `${+(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}K` : String(Math.round(v)));

/** The Heaps' law chart from data/ch04/bench.json: the measured prefixes as they are, on lineChart's log axes, and the
    dashed power law U = U₀ (N / N₀)^β through the training prefix's first point, β fitted from its first and last points
    (the same fit as verify cell 4.1-b). On log-log axes a power law is a straight line of slope β. */
export function heapsChart(heaps) {
  const runs = [
    { name: '验证集的前缀', rows: heaps.filter((r) => r.file === 'tiny_stories_valid.txt') },
    { name: '训练集的前缀', rows: heaps.filter((r) => r.file === 'train_prefix.txt') },
  ];
  const train = runs[1].rows, first = train[0], last = train[train.length - 1];
  const nRatio = last.occurrences / first.occurrences, uRatio = last.unique / first.unique;
  const beta = Math.log(uRatio) / Math.log(nRatio);
  const ns = runs.flatMap((s) => s.rows.map((r) => r.occurrences));
  const fit = [Math.min(...ns), Math.max(...ns)].map((n) => [n, first.unique * (n / first.occurrences) ** beta]);
  const opts = {
    series: [...runs.map((s) => ({ name: s.name, markers: true, points: s.rows.map((r) => [r.occurrences, r.unique]) })),
      { name: `斜率 ${beta.toFixed(2)}`, points: fit, dash: '5 4', width: 1.5 }],
    logX: true, logY: true, height: 240, table: false, xLabel: '预分词出现次数（对数刻度）', yLabel: '不同的预分词（个，对数刻度）',
    xFormat: compact, yFormat: compact, ariaLabel: 'Heaps 定律：出现次数与不同预分词数的双对数图',
  };
  return { runs, beta, nRatio, uRatio, opts };
}

export async function mount(root, config) {
  root.classList.add('w-ch04-dedup');
  root.append(h('style', {}, STYLE));
  let text = '', heaps = [];
  try {
    const [t, b] = await Promise.all([fetch(config.text), fetch(config.heaps)]);
    if (!t.ok || !b.ok) throw new Error(`HTTP ${t.status}/${b.status}`);
    text = (await t.text()).slice(0, config.chars ?? 1200);
    heaps = (await b.json()).heaps;
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `读取数据失败：${err.message}`));
    return;
  }
  const area = h('textarea', { 'aria-label': '可编辑的语料：改动后计数表会立即更新', spellcheck: 'false' });
  area.value = text;
  const stats = h('div', { class: 'stats', 'aria-live': 'polite' });
  const occBox = h('div', { class: 'tokens' });
  const occMore = h('div', { class: 'more' });
  const tbody = h('tbody');
  const tableMore = h('div', { class: 'more' });
  root.append(
    h('div', { class: 'w-label' }, '语料（可以改）：TinyStories 训练集的开头'), area, stats,
    h('div', { class: 'cols' },
      h('div', {}, h('h4', {}, '朴素版：每一次出现各存一份'), occBox, occMore),
      h('div', {}, h('h4', {}, '去重版：每个不同的预分词一行'), h('table', {}, tbody), tableMore)),
  );

  function render() {
    const occ = pretokenizeCorpus(area.value);
    const counts = new Map();
    for (const w of occ) counts.set(w, (counts.get(w) || 0) + 1);
    const ranked = [...counts].sort((a, b) => b[1] - a[1]);
    const top = new Map(ranked.slice(0, 5).map(([w], i) => [w, i]));
    const bytes = (w) => new TextEncoder().encode(w).length;
    const occBytes = occ.reduce((s, w) => s + bytes(w), 0);
    const uniqBytes = ranked.reduce((s, [w]) => s + bytes(w), 0);
    stats.replaceChildren(
      h('span', {}, '出现 ', h('b', {}, fmt(occ.length)), ' 次'),
      h('span', {}, '不同的 ', h('b', {}, fmt(counts.size)), ' 个'),
      h('span', {}, '平均每个重复 ', h('b', {}, counts.size ? (occ.length / counts.size).toFixed(1) : '0'), ' 次'),
      h('span', {}, '每步要扫描的字节：朴素 ', h('b', {}, fmt(occBytes)), '，去重 ', h('b', {}, fmt(uniqBytes))),
    );
    const shown = occ.slice(0, 90);
    occBox.replaceChildren(...shown.map((w) => {
      const c = chip(w, '', top.has(w) ? top.get(w) : 0, `“${visible(w)}” 一共出现 ${counts.get(w)} 次`);
      if (!top.has(w)) c.classList.add('plain');
      c.lastChild.remove();             // no id line under occurrences
      return c;
    }));
    occMore.textContent = occ.length > shown.length ? `……后面还有 ${fmt(occ.length - shown.length)} 次出现。前 5 名用彩色标出。` : '前 5 名用彩色标出。';
    const maxN = ranked.length ? ranked[0][1] : 1;
    tbody.replaceChildren(...ranked.slice(0, 12).map(([w, n]) => h('tr', {},
      h('td', {}, h('code', {}, `'${visible(w)}'`)),
      h('td', { class: 'n' }, `× ${n}`),
      h('td', { class: 'bar' }, h('span', { style: { width: `${(n / maxN) * 100}%` } })))));
    tableMore.textContent = ranked.length > 12 ? `……共 ${fmt(ranked.length)} 行，只显示次数最多的 12 行` : '';
  }
  area.addEventListener('input', render);
  render();

  // Heaps' law U = k N^beta: a straight line of slope beta on the chart's log-log axes.
  const box = h('div', { class: 'heaps' });
  const { runs, beta, nRatio, uRatio, opts } = heapsChart(heaps);
  root.append(h('div', { class: 'w-label' },
    `实测（双对数坐标）：训练集的前缀里，出现次数涨了 ${Math.round(nRatio)} 倍，不同的预分词只涨了 ${uRatio.toFixed(1)} 倍；`
    + `两条线几乎是直线，斜率就是 β ≈ ${beta.toFixed(2)}`), box);
  lineChart(box, opts);
  box.append(h('details', {}, h('summary', {}, '查看数据表'),
    h('table', {}, h('thead', {}, h('tr', {}, ['语料', '读到的字节', '出现次数', '不同的预分词'].map((c) => h('th', {}, c)))),
      h('tbody', {}, runs.flatMap((s) => s.rows.map((r) => h('tr', {},
        h('td', {}, s.name), h('td', {}, fmt(r.bytes)), h('td', {}, fmt(r.occurrences)), h('td', {}, fmt(r.unique)))))))));
}
