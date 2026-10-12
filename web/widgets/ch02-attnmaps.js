/* 图：训练好的 CodeBot 的注意力（2.8.1 节）。数据 data/ch02/attn_maps.json 由原书作者公开的 model_pretrain.pt
   导出，verify/test_ch02.py 会从 checkpoint 重新计算并核对；正文的结论由验证 2.8-a 断言。层和头从 0 开始编号。 */
import { h, visible } from './_lib.js';
import { seqColor } from './_plot.js';
import { injectStyle, range, seqLegend } from './ch02-util.js';

const NS = 'http://www.w3.org/2000/svg';
export const PRESETS = [
  { label: '上一个 token（L4H3）', snippet: 0, layer: 4, head: 3, q: 11 },
  { label: '函数名（L5H2）', snippet: 0, layer: 5, head: 2, q: 9 },
  { label: '开头的 token（L1H2）', snippet: 0, layer: 1, head: 2, q: 8 },
  { label: '像在抄前文（L4H0）', snippet: 1, layer: 4, head: 0, q: 17 },   // 只是注意力的样子，见 2.8.1 节的消融
];

/** Mean weight each head puts on the previous token, as [layer][head]. */
export function prevTokenMeans(attn) {
  return attn.map((layer) => layer.map((W) => {
    let s = 0;
    for (let t = 1; t < W.length; t++) s += W[t][t - 1];
    return s / (W.length - 1);
  }));
}

function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const k of kids) el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return el;
}

const short = (t, n = 7) => { const v = visible(t); return v.length > n ? `${v.slice(0, n - 1)}…` : v; };
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#ccc';

export async function mount(root, config) {
  injectStyle();
  let data;
  try {
    const r = await fetch('data/ch02/attn_maps.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `注意力数据加载失败（${err.message}）。`));
    return;
  }
  const st = { snippet: config.snippet ?? 0, layer: config.layer ?? 4, head: config.head ?? 3, q: 11, k: 10 };

  const snipSel = h('select', { 'aria-label': '选择代码片段' },
    data.snippets.map((s, i) => h('option', { value: i }, visible(s.text).slice(0, 28) + (s.text.length > 28 ? '…' : ''))));
  const layerSel = h('select', { 'aria-label': '层' }, range(6).map((l) => h('option', { value: l }, `L${l}`)));
  const headSel = h('select', { 'aria-label': '头' }, range(6).map((k) => h('option', { value: k }, `H${k}`)));
  snipSel.addEventListener('change', () => { st.snippet = +snipSel.value; resetCursor(); render(); });
  layerSel.addEventListener('change', () => { st.layer = +layerSel.value; resetCursor(); render(); });
  headSel.addEventListener('change', () => { st.head = +headSel.value; resetCursor(); render(); });
  const presets = PRESETS.map((p) => {
    const b = h('button', { type: 'button' }, p.label);
    b.addEventListener('click', () => { Object.assign(st, p); st.k = argmaxKey(); render(); });
    return b;
  });
  const mapBox = h('div', { style: { overflowX: 'auto' } });
  const legend = h('div');
  const readout = h('p', { class: 'w-label', 'aria-live': 'polite', style: { minHeight: '3.2em' } });
  const thumbs = h('div', { class: 'w-ch02-grid', style: { gridTemplateColumns: '2em repeat(6, minmax(0, 2.8rem))', maxWidth: '20rem' } });
  const meta = data.meta;
  root.append(
    h('div', { class: 'w-ch02-controls' }, snipSel, h('label', {}, '层', layerSel), h('label', {}, '头', headSel)),
    h('div', { class: 'w-ch02-controls' }, h('span', { class: 'w-ch02-seg', role: 'group', 'aria-label': '典型的头' }, presets)),
    h('div', { class: 'w-ch02-two', style: { gridTemplateColumns: 'repeat(auto-fit, minmax(17rem, 1fr))', alignItems: 'start' } },
      h('div', {}, mapBox, legend, readout),
      h('div', {}, h('div', { class: 'w-label' }, '全部 36 个头（点击选择）'), thumbs)),
    h('p', { class: 'w-ch02-note' },
      `数据：原书作者在 Hugging Face 公开的 ${meta.checkpoint}（${meta.bytes.toLocaleString('en-US')} 字节，sha256 ${meta.sha256}…），`
      + 'eval 模式，保留 4 位小数。层和头都从 0 开始编号；换一个 checkpoint，头的分工会不同。'),
  );

  const cur = () => data.snippets[st.snippet];
  const W = () => cur().attn[st.layer][st.head];
  function argmaxKey() {
    const row = W()[st.q];
    return row.reduce((best, v, j) => (v > row[best] ? j : best), 0);
  }
  function resetCursor() {
    st.q = Math.min(st.q, cur().tokens.length - 1);
    st.k = argmaxKey();
  }

  function renderMap() {
    const toks = cur().tokens, n = toks.length, A = W();
    const left = 62, top = 58;
    const avail = Math.max(220, Math.min(root.clientWidth || 600, 640)) - left - 10;
    const cs = Math.max(9, Math.min(26, Math.floor(avail / n)));
    const Wd = left + n * cs + 8, Ht = top + n * cs + 6;
    const s = svg('svg', { viewBox: `0 0 ${Wd} ${Ht}`, width: Wd, height: Ht, tabindex: 0, role: 'img',
      'aria-label': `L${st.layer}H${st.head} 的注意力权重，${n} 个 token；用方向键移动光标`, style: 'max-width:100%;height:auto;display:block' });
    const fs = Math.min(11, cs * 0.85);
    toks.forEach((t, i) => {
      s.append(svg('text', { x: left - 5, y: top + (i + 0.5) * cs + fs * 0.35, 'text-anchor': 'end', 'font-size': fs,
        fill: i === st.q ? 'var(--ink)' : 'var(--ink-3)', 'font-family': 'var(--mono)' }, short(t)));
      const x = left + (i + 0.5) * cs, y = top - 5;
      s.append(svg('text', { x, y, transform: `rotate(-55 ${x} ${y})`, 'font-size': fs,
        fill: i === st.k ? 'var(--ink)' : 'var(--ink-3)', 'font-family': 'var(--mono)' }, short(t)));
    });
    for (let r = 0; r < n; r++) {
      for (let c = 0; c <= r; c++) {                     // the future (c > r) is masked: left blank
        s.append(svg('rect', { x: left + c * cs, y: top + r * cs, width: cs - 1, height: cs - 1, fill: seqColor(A[r][c]) }));
      }
    }
    s.append(svg('rect', { x: left - 0.5, y: top - 0.5, width: n * cs, height: n * cs, fill: 'none', stroke: 'var(--rule)' }));
    s.append(svg('rect', { x: left + st.k * cs - 0.5, y: top + st.q * cs - 0.5, width: cs, height: cs, fill: 'none',
      stroke: 'var(--ink)', 'stroke-width': 2 }));
    const pick = (e) => {
      const b = s.getBoundingClientRect(), sc = b.width / Wd;
      const c = Math.floor(((e.clientX - b.left) / sc - left) / cs), r = Math.floor(((e.clientY - b.top) / sc - top) / cs);
      if (r >= 0 && r < n && c >= 0 && c <= r && (r !== st.q || c !== st.k)) { st.q = r; st.k = c; renderMapAndReadout(); }
    };
    s.addEventListener('pointermove', pick);
    s.addEventListener('keydown', (e) => {
      const mv = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
      if (!mv) return;
      e.preventDefault();
      st.q = Math.max(0, Math.min(n - 1, st.q + mv[0]));
      st.k = Math.max(0, Math.min(st.q, st.k + mv[1]));
      renderMapAndReadout(true);
    });
    mapBox.replaceChildren(s);
    return s;
  }

  function renderReadout() {
    const sn = cur(), A = W();
    const q = st.q, k = st.k, row = A[q];
    const best = row.reduce((b, v, j) => (v > row[b] ? j : b), 0);
    readout.replaceChildren(
      `L${st.layer}H${st.head}：查询 ${q}「${visible(sn.tokens[q])}」→ 键 ${k}「${visible(sn.tokens[k])}」的权重 `,
      h('b', {}, row[k].toFixed(2)), `；这一行最大的权重 ${row[best].toFixed(2)} 落在键 ${best}「${visible(sn.tokens[best])}」。`,
      h('br'), `模型在位置 ${q} 预测的下一个 token 是「${visible(sn.pred[q])}」（概率 ${sn.pred_prob[q].toFixed(2)}）。`);
  }

  function renderMapAndReadout(refocus = false) {
    const s = renderMap();
    renderReadout();
    if (refocus) s.focus();
  }

  function renderThumbs() {
    const sn = cur(), n = sn.tokens.length, masked = cssVar('--paper');   // the masked future reads as empty
    const cells = [h('span', { class: 'cell head' }, ''), ...range(6).map((k) => h('span', { class: 'cell head' }, `H${k}`))];
    for (let l = 0; l < 6; l++) {
      cells.push(h('span', { class: 'cell head' }, `L${l}`));
      for (let k = 0; k < 6; k++) {
        const cv = h('canvas', { width: n, height: n, style: { width: '100%', aspectRatio: '1', imageRendering: 'pixelated', display: 'block' } });
        const ctx = cv.getContext('2d');
        const A = sn.attn[l][k];
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { ctx.fillStyle = c > r ? masked : seqColor(A[r][c]); ctx.fillRect(c, r, 1, 1); }
        const sel = l === st.layer && k === st.head;
        const b = h('button', { type: 'button', 'aria-label': `L${l}H${k}（层和头从 0 数）`, 'aria-pressed': String(sel),
          style: { padding: '1px', border: sel ? '2px solid var(--ink)' : '1px solid var(--rule)', lineHeight: 0 } }, cv);
        b.addEventListener('click', () => { st.layer = l; st.head = k; resetCursor(); render(); });
        cells.push(b);
      }
    }
    thumbs.replaceChildren(...cells);
  }

  function render() {
    snipSel.value = st.snippet; layerSel.value = st.layer; headSel.value = st.head;
    renderMapAndReadout();
    renderThumbs();
    legend.replaceChildren(seqLegend());          // seqColor flips in the dark theme: the key says which end is large
  }

  st.k = argmaxKey();
  render();
  let lastW = root.clientWidth;
  new ResizeObserver(() => { if (root.clientWidth !== lastW) { lastW = root.clientWidth; renderMapAndReadout(); } }).observe(root);
  const rerender = () => render();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', rerender);
  new MutationObserver(rerender).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}
