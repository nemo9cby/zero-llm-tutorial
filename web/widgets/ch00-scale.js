/* 图：三台机器人的规模。对数刻度上的点，每格是 10 倍；点下给出数值和相对 CodeBot 的倍数。
   Dots rather than bars: on a log axis a bar's length depends on where the axis starts, so it would not show
   the true ratio. Values come from data/ch00/bots.json (checked by verify/test_ch00.py); 6ND is computed here
   from N and D, and verify cell 0.5-e asserts the same products.
   config: { metric: "params" } */
import { h, fmt } from './_lib.js';
import { seriesColor } from './_plot.js';

const METRICS = [
  { id: 'params', name: '参数量 N', get: (b) => b.params },
  { id: 'tokens', name: '训练 token 数 D', get: (b) => b.pretrain.tokens },
  { id: 'flops', name: '训练计算量 6ND', get: (b) => 6 * b.params * b.pretrain.tokens, sci: true },
  { id: 'bytes', name: '训练语料（字节）', get: (b) => b.corpus.text_bytes },
  { id: 'vocab', name: '词表大小', get: (b) => b.model.vocab_size },
  { id: 'ctx', name: '上下文长度', get: (b) => b.model.context_len },
];

const CSS = `
.w-ch00-scale .metrics { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .9rem; }
.w-ch00-scale .metrics button[aria-pressed="true"] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-ch00-scale .rows { display: grid; grid-template-columns: 5.2rem minmax(0, 1fr); gap: .35rem .6rem; align-items: start; }
.w-ch00-scale .name { font-size: .88rem; font-weight: 600; line-height: 1.3rem; }
.w-ch00-scale .track { position: relative; height: 1.3rem; border-left: 1px solid var(--ink-3); }
.w-ch00-scale .track::before { content: ''; position: absolute; left: 0; right: 0; top: 50%; border-top: 1px solid var(--rule); }
.w-ch00-scale .dot { position: absolute; top: 50%; width: .8rem; height: .8rem; border-radius: 50%;
  transform: translate(-50%, -50%); box-shadow: 0 0 0 2px var(--paper); transition: left .35s ease; }
.w-ch00-scale .val { font: 13px/1.4 var(--serif); font-variant-numeric: tabular-nums; color: var(--ink-2);
  margin: .1rem 0 .3rem; overflow-wrap: anywhere; }
.w-ch00-scale sup { font-size: .75em; line-height: 0; }
.w-ch00-scale .axis { position: relative; height: 1.3rem; margin-left: calc(5.2rem + .6rem); font: 12px/1.3rem var(--serif); color: var(--ink-3); }
.w-ch00-scale .axis span { position: absolute; white-space: nowrap; }
.w-ch00-scale .grid { position: absolute; top: 0; bottom: 0; border-left: 1px dashed var(--rule); }
.w-ch00-scale .note { margin-top: .6rem; font-size: .88rem; color: var(--ink-2); }
@media (prefers-reduced-motion: reduce) { .w-ch00-scale .dot { transition: none; } }`;

const sup = (e) => String(e).split('').map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('');   // for aria-labels (plain text)
const sci = (v) => {
  const e = Math.floor(Math.log10(v));
  return `${(v / 10 ** e).toFixed(2)}×10${sup(e)}`;
};
const sciNode = (v) => {                                   // the same number with a real <sup> for the page
  const e = Math.floor(Math.log10(v));
  return [`${(v / 10 ** e).toFixed(2)}×10`, h('sup', {}, String(e))];
};
const pow10 = (e) => ['10', h('sup', {}, String(e))];
const ratio = (r) => (r >= 100 ? fmt(Math.round(r)) : r >= 10 ? r.toFixed(1) : r.toFixed(2));

export async function mount(root, config) {
  root.append(h('style', {}, CSS));
  const box = h('div', { class: 'w-ch00-scale' });
  root.append(box);
  let bots;
  try {
    const r = await fetch('data/ch00/bots.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    bots = (await r.json()).bots;
  } catch (err) {
    box.append(h('p', { class: 'w-label' }, `无法读取 data/ch00/bots.json（${err.message}）。`));
    return;
  }
  let metric = METRICS.find((m) => m.id === config.metric) || METRICS[0];
  const btnRow = h('div', { class: 'metrics', role: 'group', 'aria-label': '选择要比较的量' });
  const rows = h('div', { class: 'rows' });
  const axis = h('div', { class: 'axis', 'aria-hidden': 'true' });
  const note = h('p', { class: 'note', 'aria-live': 'polite' });
  box.append(btnRow, rows, axis, note);

  let lo = 0, hi = 1;
  const pos = (v) => ((Math.log10(v) - lo) / (hi - lo)) * 100;
  // Tick labels need about 48 px each: on a narrow (phone) axis, label every 2nd or 3rd decade.
  const renderTicks = () => {
    const every = Math.max(1, Math.ceil(((hi - lo) * 48) / Math.max(axis.clientWidth, 1)));
    const ticks = [];
    for (let e = lo; e <= hi; e++) {
      if ((e - lo) % every) continue;
      const left = pos(10 ** e);
      const shift = left > 94 ? '-100%' : left < 6 ? '0' : '-50%';   // keep the end labels inside the box
      ticks.push(h('span', { style: { left: `${left}%`, transform: `translateX(${shift})` } }, pow10(e)));
    }
    axis.replaceChildren(...ticks);
  };
  if (typeof ResizeObserver === 'function') new ResizeObserver(() => renderTicks()).observe(axis);

  const render = () => {
    btnRow.replaceChildren(...METRICS.map((m) => h('button', {
      type: 'button', 'aria-pressed': String(m === metric), onclick: () => { metric = m; render(); },
    }, m.name)));
    const vals = bots.map(metric.get);
    lo = Math.floor(Math.log10(Math.min(...vals))) - 1;
    hi = Math.ceil(Math.log10(Math.max(...vals)));
    rows.replaceChildren(...bots.flatMap((b, i) => {
      const v = vals[i];
      const times = i ? `（CodeBot 的 ${ratio(v / vals[0])} 倍）` : '';
      const label = `${metric.sci ? sci(v) : fmt(v)}${times}`;
      const grid = [];
      for (let e = lo + 1; e <= hi; e++) grid.push(h('span', { class: 'grid', style: { left: `${pos(10 ** e)}%` } }));
      return [
        h('div', { class: 'name' }, b.name),
        h('div', {},
          h('div', { class: 'track', role: 'img', 'aria-label': `${b.name}：${label}` }, ...grid,
            h('div', { class: 'dot', style: { left: `${pos(v)}%`, background: seriesColor(i) } })),
          h('div', { class: 'val' }, metric.sci ? sciNode(v) : fmt(v), times)),
      ];
    }));
    renderTicks();
    const [c, , w] = vals;
    note.textContent = `${metric.name}：WebBot 是 CodeBot 的 ${ratio(w / c)} 倍。横轴每过一条虚线，数值就乘以 10。`;
  };
  render();
}
