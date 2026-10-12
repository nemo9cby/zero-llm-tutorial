/* 图：分块边界。上半部分是验证集开头的 60 个故事（valid_head.txt），等分的切点（虚线）向后吸附到
   下一个 <|endoftext|>（实线）；下方放大显示某个切点附近的文字。表格用整份验证集的 27,630 个
   <|endoftext|> 偏移，现场算出 find_chunk_boundaries 的边界数（与 Python 逐个相同，见 tests/js）。
   config: { eot: "data/ch04/valid_eot.json", text: "data/ch04/valid_head.txt" } */
import { h, fmt } from './_lib.js';
import { chunkBoundaries, decodeOffsets, EOT } from './ch04-bpe.js';

const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

const STYLE = `
.w-ch04-chunks, .w-ch04-chunks * { font-variant-ligatures: none !important; }
.w-ch04-chunks .ctl { display: flex; flex-wrap: wrap; gap: .5rem 1rem; align-items: center; font-size: .86rem; margin-bottom: .5rem; }
.w-ch04-chunks input[type=range] { flex: 1 1 12rem; min-width: 10rem; }
.w-ch04-chunks svg { display: block; width: 100%; height: auto; }
.w-ch04-chunks .legend { display: flex; flex-wrap: wrap; gap: .3rem 1.2rem; font-size: .78rem; color: var(--ink-2); margin: .3rem 0 .6rem; }
.w-ch04-chunks .legend i { display: inline-block; width: 16px; height: 0; vertical-align: middle; margin-right: 5px; border-top: 2px solid var(--ai); }
.w-ch04-chunks .legend i.raw { border-top: 2px dashed var(--kincha); }
.w-ch04-chunks .legend i.eot { border-top: 2px solid var(--ink-3); width: 8px; }
.w-ch04-chunks .zoom { font: 12.5px/1.6 var(--mono); background: var(--paper-2); border-radius: 4px; padding: .5rem .7rem; white-space: pre-wrap; word-break: break-all; }
.w-ch04-chunks .zoom .cut { color: var(--kincha); font-weight: 700; }
.w-ch04-chunks .zoom .snap { color: var(--ai); font-weight: 700; }
.w-ch04-chunks .zoom .mk { background: color-mix(in srgb, var(--ai-2) 70%, transparent); }
.w-ch04-chunks .zoomnav { display: flex; gap: .5rem; align-items: center; margin: .6rem 0 .3rem; font-size: .84rem; }
.w-ch04-chunks table { display: table; font-size: .82rem; margin: .8rem 0 0; }
.w-ch04-chunks td, .w-ch04-chunks th { padding: .2rem .9rem .2rem 0; font-variant-numeric: tabular-nums; }
`;

export async function mount(root, config) {
  root.classList.add('w-ch04-chunks');
  root.append(h('style', {}, STYLE));
  let eot, text;
  try {
    const [a, b] = await Promise.all([fetch(config.eot), fetch(config.text)]);
    if (!a.ok || !b.ok) throw new Error(`HTTP ${a.status}/${b.status}`);
    eot = await a.json();
    text = await b.text();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `读取数据失败：${err.message}`));
    return;
  }
  const bytes = new TextEncoder().encode(text);
  const size = bytes.length;
  const marker = new TextEncoder().encode(EOT);
  const offs = [];
  for (let i = 0; i + marker.length <= size; i++) {
    if (bytes[i] === marker[0] && marker.every((m, j) => bytes[i + j] === m)) offs.push(i);
  }
  const dec = new TextDecoder();
  let k = 4, sel = 1;

  const slider = h('input', { type: 'range', min: 1, max: 120, value: k, 'aria-label': '块数 num_chunks' });
  const kLabel = h('span', { 'aria-live': 'polite' });
  const strip = h('div');
  const zoomLabel = h('span', { class: 'w-label' });
  const prev = h('button', { type: 'button', 'aria-label': '上一个切点' }, '◀');
  const next = h('button', { type: 'button', 'aria-label': '下一个切点' }, '▶');
  const zoom = h('div', { class: 'zoom' });
  root.append(
    h('div', { class: 'ctl' }, h('label', {}, 'num_chunks'), slider, kLabel),
    strip,
    h('div', { class: 'legend' }, h('span', {}, h('i', { class: 'raw' }), '等分的初始切点'), h('span', {}, h('i'), '吸附后的边界'),
      h('span', {}, h('i', { class: 'eot' }), '<|endoftext|>（故事之间）')),
    h('div', { class: 'zoomnav' }, prev, next, zoomLabel), zoom,
  );

  // whole validation file: counts for a few num_chunks values, computed from the real offsets
  const full = decodeOffsets(eot.deltas);
  const rows = [8, 64, 1000, 100000].map((n) => {
    const r = chunkBoundaries(full, eot.size, n);
    return h('tr', {}, h('td', {}, fmt(n)), h('td', {}, fmt(r.boundaries.length)),
      h('td', {}, `${(eot.size / (r.boundaries.length - 1) / 1e6).toFixed(n >= 1000 ? 3 : 2)} MB`));
  });
  root.append(h('div', { class: 'w-label', style: { marginTop: '1rem' } },
    `整份验证集（${fmt(eot.size)} 字节，${fmt(eot.count)} 个 <|endoftext|>）上，同一个函数给出的边界数：`),
  h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'num_chunks'), h('th', {}, '边界数（含首尾）'), h('th', {}, '平均每块'))),
    h('tbody', {}, rows)));

  function render() {
    const r = chunkBoundaries(offs, size, k);
    const internal = r.raw.slice(1, -1);
    kLabel.textContent = `${k} 块：${internal.length} 个初始切点 → 去重后 ${r.boundaries.length} 个边界（含首尾）`;
    const W = 720, H = 74, x = (b) => 4 + (b / size) * (W - 8);
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `文件条：${k} 块，${r.boundaries.length} 个边界` });
    let startByte = 0;
    [...offs, size].forEach((end, i) => {
      s.append(svg('rect', { x: x(startByte), y: 22, width: Math.max(0.5, x(end) - x(startByte)), height: 26,
        fill: i % 2 ? 'var(--paper-3)' : 'var(--paper-2)' }));
      startByte = end;
    });
    for (const o of offs) s.append(svg('line', { x1: x(o), x2: x(o), y1: 22, y2: 48, stroke: 'var(--ink-3)', 'stroke-width': 1 }));
    internal.forEach((raw, i) => {
      const snapped = r.snapped[i + 1];
      const active = i + 1 === sel;
      s.append(svg('line', { x1: x(raw), x2: x(raw), y1: 6, y2: 64, stroke: 'var(--kincha)', 'stroke-width': active ? 2 : 1.2, 'stroke-dasharray': '3 2' }));
      s.append(svg('path', { d: `M${x(raw)},12 L${x(snapped)},12`, stroke: 'var(--kincha)', 'stroke-width': 1, fill: 'none' }));
      s.append(svg('line', { x1: x(snapped), x2: x(snapped), y1: 10, y2: 60, stroke: 'var(--ai)', 'stroke-width': active ? 3 : 2 }));
    });
    s.append(svg('rect', { x: 4, y: 22, width: W - 8, height: 26, fill: 'none', stroke: 'var(--rule)' }));
    strip.replaceChildren(s);

    sel = Math.min(Math.max(1, sel), Math.max(1, internal.length));
    prev.disabled = sel <= 1;
    next.disabled = sel >= internal.length;
    if (!internal.length) {
      zoomLabel.textContent = '只有一块，没有内部切点。';
      zoom.replaceChildren();
      return;
    }
    const raw = internal[sel - 1], snapped = r.snapped[sel];
    zoomLabel.textContent = `第 ${sel} 个切点：初始位置第 ${fmt(raw)} 字节，吸附到第 ${fmt(snapped)} 字节（向后挪了 ${fmt(snapped - raw)} 字节）`;
    const piece = (a, b) => dec.decode(bytes.slice(Math.max(0, a), Math.min(size, b))).replace(/\n/g, '↵');
    const ctx = 34;
    const lines = [h('div', {}, '初始切点：…', piece(raw - ctx, raw), h('span', { class: 'cut' }, '│'), piece(raw, raw + ctx), '…')];
    if (snapped < size) {
      lines.push(h('div', {}, '吸附之后：…', piece(snapped - ctx, snapped), h('span', { class: 'snap' }, '│'),
        h('span', { class: 'mk' }, piece(snapped, snapped + marker.length)), piece(snapped + marker.length, snapped + marker.length + 20), '…'));
    } else {
      lines.push(h('div', {}, '吸附之后：文件末尾（后面再也没有 <|endoftext|>）'));
    }
    zoom.replaceChildren(...lines);
  }

  slider.addEventListener('input', () => { k = Number(slider.value); render(); });
  prev.addEventListener('click', () => { sel -= 1; render(); });
  next.addEventListener('click', () => { sel += 1; render(); });
  render();
}
