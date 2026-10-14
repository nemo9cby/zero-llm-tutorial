/* 图：三台机器人的分词器读同一句网页文本，以及它们在三种数据上的压缩率（字节 / token）。
   Data from data/part3/tokenizers.json (verify/test_part3.py re-encodes everything with the book's tokenizers).
   Chips and the 3×3 table are tinted locally (shared palette mixed into var(--paper)) so their text stays
   legible in both themes.
   config: {} */
import { h, visible, fmt } from './_lib.js';
import { seriesVars } from './_plot.js';

const NAMES = { codebot: 'CodeBot（1,000）', storybot: 'StoryBot（10,000）', webbot: 'WebBot（50,000）' };
const CORPUS = { TinyCodes: '代码', TinyStories: '故事', OpenWebText: '网页' };
const decoder = new TextDecoder('utf-8', { fatal: true });

const CSS = `
.w-part3-tok .chips { display: flex; flex-wrap: wrap; gap: 3px; font: 13.5px/1.35 var(--mono); margin: .3rem 0 .9rem; }
.w-part3-tok .chip { display: inline-flex; flex-direction: column; align-items: center; border-radius: 3px;
  padding: 2px 5px 1px; min-width: 1.6em; color: var(--ink); }
.w-part3-tok .chip b { font-weight: 400; white-space: pre; }
.w-part3-tok .chip i { font-style: normal; font-size: 10px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.w-part3-tok table.heat { display: table; margin: .4rem 0 0; font: 14px/1.4 var(--mono); font-variant-numeric: tabular-nums; }
.w-part3-tok table.heat th { font: 600 13px/1.4 var(--serif); color: var(--ink-2); padding: .3rem .6rem; border: 0; }
.w-part3-tok table.heat td { text-align: center; padding: .45rem .9rem; border: 2px solid var(--paper); color: var(--ink); }`;

function label(hex) {
  const bytes = Uint8Array.from(hex.match(/../g).map((b) => parseInt(b, 16)));
  try {
    return visible(decoder.decode(bytes));
  } catch {
    return `‹${hex.match(/../g).join(' ')}›`;     // part of a multi-byte character
  }
}

export async function mount(root) {
  root.append(h('style', {}, CSS));
  const box = h('div', { class: 'w-part3-tok' });
  root.append(box);
  seriesVars(box);                                  // the tints follow theme switches
  let data;
  try {
    const r = await fetch('data/part3/tokenizers.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    box.append(h('p', { class: 'w-label' }, `无法读取 data/part3/tokenizers.json（${err.message}）。`));
    return;
  }
  box.append(h('p', { class: 'w-label', style: 'margin:0 0 .3rem' }, `例句（${data.sentence_source}）：`),
    h('p', { style: 'margin:0 0 .8rem;font-size:.95rem' }, data.sentence));
  let partial = false;
  for (const [tok, chips] of Object.entries(data.chips)) {
    box.append(h('div', { class: 'w-label' }, `${NAMES[tok]}：${chips.length} 个 token`),
      h('div', { class: 'chips' }, chips.map((c, i) => {
        const text = label(c.bytes);
        partial ||= text.startsWith('‹');
        return h('span', {
          class: 'chip', title: `id ${c.id}，字节 ${c.bytes.match(/../g).join(' ')}`,
          style: { background: `color-mix(in oklab, var(--series-${i % 6}) 24%, var(--paper))` },
        }, h('b', {}, text), h('i', {}, c.id));
      })));
  }
  if (partial) {
    box.append(h('p', { class: 'w-label', style: 'margin:-.4rem 0 .9rem' },
      '‹e2› 这样的记号是单独一个字节：弯引号 “ 在 UTF-8 里占 3 个字节（e2 80 9c）。词表里没有把它们合起来的 token 时，'
      + '只能切成 3 个 token，单独一个字节显示不出字来。'));
  }
  // A small shaded table rather than _plot.heatmap: the shade is mixed into var(--paper), so the text
  // (var(--ink)) stays readable in both themes.
  const toks = Object.keys(NAMES);
  const vals = data.matrix.map((row) => toks.map((t) => row.bytes / row.tokens[t]));
  const all = vals.flat(), lo = Math.min(...all), hi = Math.max(...all);
  const shade = (v) => `color-mix(in oklab, var(--series-0) ${Math.round(8 + (62 * (v - lo)) / (hi - lo))}%, var(--paper))`;
  box.append(h('div', { class: 'w-label' }, '每个 token 平均装下多少字节（越大越省；行是数据，列是分词器）'),
    h('table', { class: 'heat', 'aria-label': '三种分词器在三种数据上的压缩率（字节 / token）' },
      h('thead', {}, h('tr', {}, h('th', {}, ''), ['CodeBot', 'StoryBot', 'WebBot'].map((t) => h('th', {}, t)))),
      h('tbody', {}, data.matrix.map((row, r) => h('tr', {}, h('th', {}, CORPUS[row.corpus]),
        vals[r].map((v, c) => h('td', {
          style: { background: shade(v), fontWeight: v === Math.max(...vals[r]) ? 700 : 400 },
          title: `${row.corpus} × ${toks[c]}：${fmt(row.bytes)} 字节 / ${fmt(row.tokens[toks[c]])} token`,
        }, v.toFixed(2))))))));
  const approx = Math.round(Math.min(...data.matrix.map((r) => r.bytes)) / 1e4);
  box.append(h('p', { class: 'w-label', style: 'margin:.3rem 0 0' },
    `行：代码 = TinyCodes，故事 = TinyStories 验证集，网页 = OpenWebText 验证集，各取开头的整篇文档，约 ${fmt(approx)} 万字节`
    + `（${data.matrix.map((r) => fmt(r.bytes)).join(' / ')} 字节）。列：三台机器人各自的分词器。`));
}
