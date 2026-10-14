/* P.5 Where Attention looks: the softmax weights of all 6 layers x 6 heads for the current input (thumbnails), and
   one head enlarged with token labels (rows = queries, columns = keys; the causal mask hides the upper triangle).
   config: { checkpoint, prompt, template, layer, head } (layer and head count from 0, written L0H0 as in the code and ch02) */
import { h } from './_lib.js';
import { heatmap, seqColor, onThemeChange } from './_plot.js';
import {
  injectStyle, getIndex, getTokenizer, engine, ensureModel, checkpointSwitch, statusLine, alpaca, tokenLabel,
  whenVisible, pct,
} from './playground-common.js';

const MAX_TOKENS = 40;
const N_LAYER = 6, N_HEAD = 6;
const headName = (l, hh) => `L${l}H${hh}`;
const clampIndex = (v, n) => Math.min(n - 1, Math.max(0, Math.trunc(Number(v)) || 0));

let gridStyled = false;
function injectGridStyle() {
  if (gridStyled) return;
  gridStyled = true;
  document.head.append(h('style', {}, `
.w-pg-grid { display: grid; grid-template-columns: max-content repeat(${N_HEAD}, minmax(0, 52px)); justify-content: start; gap: 4px; align-items: center; margin: .5rem 0; }
.w-pg-grid .hd { font-size: .75rem; color: var(--ink-3); text-align: center; white-space: nowrap; padding-right: .2rem; }
.w-pg-thumb { padding: 2px; line-height: 0; border: 1px solid var(--rule); background: var(--paper); }
.w-pg-thumb canvas { width: 100%; height: auto; image-rendering: pixelated; }
.w-pg-thumb[aria-pressed="true"] { border: 2px solid var(--ai); padding: 1px; }
`));
}

function thumbnail(values, n, label, onPick) {
  const px = Math.max(1, Math.floor(44 / n));
  const canvas = h('canvas', { width: n * px, height: n * px, 'aria-hidden': 'true' });
  const ctx = canvas.getContext('2d');
  const masked = getComputedStyle(document.documentElement).getPropertyValue('--paper-3').trim() || '#e3e8f0';
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      ctx.fillStyle = c > r ? masked : seqColor(values[r * n + c]);
      ctx.fillRect(c * px, r * px, px, px);
    }
  }
  return h('button', { type: 'button', class: 'w-pg-thumb', 'aria-label': label, 'aria-pressed': 'false', onclick: onPick }, canvas);
}

export async function mount(root, config = {}) {
  injectStyle();
  const cfg = { checkpoint: 'pretrain', prompt: 'def add(a, b):\n    return a + b', template: false, layer: 0, head: 0, ...config };
  const box = h('div', { class: 'w-pg' }, h('p', { class: 'w-pg-note' }, '正在准备……'));
  root.append(box);
  let entries, tok;
  try {
    [entries, tok] = await Promise.all([getIndex(), getTokenizer()]);
  } catch (err) {
    box.replaceChildren(h('p', { class: 'w-pg-error' }, `这张图没能启动：${err.message}`));
    return;
  }
  injectGridStyle();
  let name = cfg.checkpoint, sel = { l: clampIndex(cfg.layer, N_LAYER), h: clampIndex(cfg.head, N_HEAD) };
  let data = null, busy = false, again = false;
  const sw = checkpointSwitch(entries, name, (n) => { name = n; refresh(); });
  const text = h('textarea', { class: 'w-input', rows: 2, spellcheck: 'false', 'aria-label': 'prompt' });
  text.value = cfg.prompt;
  const tpl = h('input', { type: 'checkbox', checked: cfg.template });
  const status = statusLine();
  const grid = h('div', { class: 'w-pg-grid', role: 'group', 'aria-label': '6 层 × 6 个头的注意力权重缩略图（层和头从 0 编号）' });
  const title = h('div', { class: 'w-label' });
  const big = h('div');
  const note = h('div', { class: 'w-pg-note', 'aria-live': 'polite' });
  box.replaceChildren(h('div', { class: 'w-row' }, sw.el, h('label', { class: 'w-pg-field' }, tpl, '套用 Alpaca 模板')),
    text, status.el, grid, title, big, note);
  let hm = null;

  function head(l, hh) {
    const { n, attention } = data;
    const off = ((l * N_HEAD) + hh) * n * n;
    return attention.subarray(off, off + n * n);
  }

  function show() {
    const { n, ids } = data;
    for (const b of grid.querySelectorAll('.w-pg-thumb')) b.setAttribute('aria-pressed', String(+b.dataset.l === sel.l && +b.dataset.h === sel.h));
    const w = head(sel.l, sel.h);
    const full = ids.map((id) => tokenLabel(tok, id));
    // A cut label ends in "…". Row labels get the 44 px margin of the heatmap; a column label gets one cell
    // (at most 26 px, narrower on a phone), about 7 px per character.
    const cut = (s, room) => ([...s].length > room ? `${[...s].slice(0, room - 1).join('')}…` : s);
    const cellPx = Math.min(26, Math.floor(((root.clientWidth || 600) - 48) / n));
    const rowLabels = full.map((s) => cut(s, 6));
    const colLabels = full.map((s) => cut(s, Math.max(2, Math.floor(cellPx / 7))));
    const values = Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => (c > r ? NaN : w[r * n + c])));
    const opts = { values, rowLabels, colLabels, scale: 'seq', domain: [0, 1], size: 26, showValues: false,
      cellFormat: (v) => pct(v), masked: (r, c) => c > r, ariaLabel: `${headName(sel.l, sel.h)} 的注意力权重` };
    title.textContent = `${headName(sel.l, sel.h)}（行：查询 token；列：被看的 token）`;
    if (hm) hm.update(opts); else hm = heatmap(big, opts);
    const last = w.subarray((n - 1) * n, n * n);
    let best = 0;
    for (let j = 1; j < n; j++) if (last[j] > last[best]) best = j;
    note.textContent = `最后一个 token「${full[n - 1]}」把 ${pct(last[best], 0)} 的注意力给了位置 ${best} 的「${full[best]}」。`;
  }

  function buildGrid() {
    grid.replaceChildren(h('span'), ...Array.from({ length: N_HEAD }, (_, j) => h('span', { class: 'hd' }, `H${j}`)));
    for (let l = 0; l < N_LAYER; l++) {
      grid.append(h('span', { class: 'hd' }, `L${l}`));
      for (let hh = 0; hh < N_HEAD; hh++) {
        const b = thumbnail(head(l, hh), data.n, `${headName(l, hh)}（层和头从 0 数）`, () => { sel = { l, h: hh }; show(); });
        b.dataset.l = l; b.dataset.h = hh;
        grid.append(b);
      }
    }
  }

  async function refresh() {
    if (busy) { again = true; return; }
    busy = true;
    sw.set(name);
    try {
      let ids = tok.encode(tpl.checked ? alpaca(text.value) : text.value);
      if (!ids.length) { status.say('先输入一点文字。'); return; }
      const cut = ids.length > MAX_TOKENS;
      ids = ids.slice(0, MAX_TOKENS);
      await ensureModel(name, status);
      const res = await (await engine().call('forward', name, { ids, attention: true })).promise;
      data = { n: res.n, attention: res.attention, ids };
      buildGrid();
      show();
      status.say(`${res.n} 个 token${cut ? `（只取前 ${MAX_TOKENS} 个）` : ''}，前向 ${res.ms.toFixed(0)} ms。点缩略图放大某个头。`);
    } catch (err) {
      status.error(err);
    } finally {
      busy = false;
      if (again) { again = false; refresh(); }
    }
  }

  let t = 0;
  text.addEventListener('input', () => { clearTimeout(t); t = setTimeout(refresh, 300); });
  tpl.addEventListener('change', refresh);
  // The thumbnails are canvases painted with seqColor and --paper-3: repaint them on every theme switch (toggle or OS).
  onThemeChange(() => { if (data) { buildGrid(); show(); } });
  status.say('这张图进入视野时才加载模型。');
  whenVisible(box, refresh);
}
