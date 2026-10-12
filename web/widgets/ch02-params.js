/* 图：参数计算器（2.10.2 节）。逐项列出 GPT 的参数从哪里来；公式与验证 2.10-a/2.10-b 相同，
   CodeBot 写法的结果由 verify/test_ch02.py 在多组配置上与 PyTorch 核对。 */
import { h, fmt } from './_lib.js';
import { seriesColor } from './_plot.js';
import { injectStyle } from './ch02-util.js';

export const PRESETS = {
  codebot: { label: 'CodeBot（本章）', V: 1000, ctx: 256, E: 384, L: 6, ff: 1536, style: 'codebot', tied: true },
  gpt2: { label: 'GPT-2 small', V: 50257, ctx: 1024, E: 768, L: 12, ff: 3072, style: 'gpt2', tied: true },
};

/** Parameter breakdown. style 'codebot': no qkv/proj bias, unembed has a bias;
    style 'gpt2' (original GPT-2): qkv and output projection have biases, the output layer has none. */
export function countParams({ V, ctx, E, L, ff, tied = true, style = 'codebot' }) {
  const attnBias = style === 'gpt2' ? 4 * E : 0;
  const parts = [
    { key: 'tok', name: tied ? 'token 嵌入（与 unembed 共享）' : 'token 嵌入', n: V * E },
    { key: 'pos', name: '位置嵌入', n: ctx * E },
    { key: 'ln', name: `块内 LayerNorm（${L} × 2 × 2E）`, n: L * 2 * 2 * E },
    { key: 'attn', name: `块内 Attention（${L} × (4E²${attnBias ? ' + 4E' : ''})）`, n: L * (4 * E * E + attnBias) },
    { key: 'ffn', name: `块内 FFN（${L} × (2·E·ff + ff + E)）`, n: L * (E * ff + ff + ff * E + E) },
    { key: 'lnf', name: '最终 LayerNorm（2E）', n: 2 * E },
  ];
  if (!tied) parts.push({ key: 'unw', name: 'unembed 权重（不共享）', n: V * E });
  if (style === 'codebot') parts.push({ key: 'unb', name: 'unembed 偏置', n: V });
  const total = parts.reduce((a, p) => a + p.n, 0);
  return { parts, total, stateDictSum: total + (tied ? V * E : 0), approx12: 12 * L * E * E };
}

export function mount(root) {
  injectStyle();
  const st = { ...PRESETS.codebot };
  const fields = [['V', '词表 V'], ['ctx', '上下文'], ['E', '宽度 E'], ['L', '层数 L'], ['ff', 'FFN 宽度']];
  const inputs = Object.fromEntries(fields.map(([k, label]) => {
    const inp = h('input', { type: 'number', min: 1, max: k === 'V' ? 300000 : 100000, step: 1, value: st[k], 'aria-label': label });
    inp.addEventListener('input', () => { const v = Math.floor(+inp.value); if (v > 0) { st[k] = v; render(); } });
    return [k, [label, inp]];
  }));
  const tiedBox = h('input', { type: 'checkbox', checked: true, 'aria-label': '权重共享' });
  tiedBox.addEventListener('change', () => { st.tied = tiedBox.checked; render(); });
  const styleSel = h('select', { 'aria-label': '偏置的写法' },
    h('option', { value: 'codebot' }, 'CodeBot 的偏置'), h('option', { value: 'gpt2' }, '原版 GPT-2 的偏置'));
  styleSel.addEventListener('change', () => { st.style = styleSel.value; render(); });
  const presetBtns = Object.entries(PRESETS).map(([k, p]) => {
    const b = h('button', { type: 'button' }, p.label);
    b.addEventListener('click', () => { Object.assign(st, p); sync(); render(); });
    return b;
  });
  const out = h('div', { 'aria-live': 'polite' });
  root.append(
    h('div', { class: 'w-ch02-controls' }, ...presetBtns),
    h('div', { class: 'w-ch02-controls' }, Object.values(inputs).map(([label, inp]) => h('label', {}, label, inp))),
    h('div', { class: 'w-ch02-controls' }, h('label', {}, tiedBox, '权重共享'), styleSel),
    out,
  );

  function sync() {
    for (const [k, [, inp]] of Object.entries(inputs)) inp.value = st[k];
    tiedBox.checked = st.tied;
    styleSel.value = st.style;
  }

  function render() {
    const r = countParams(st);
    const colors = r.parts.map((_, i) => seriesColor(i));
    out.replaceChildren(
      h('div', { class: 'w-row' }, h('span', { class: 'w-label' }, '参数总数'), h('span', { class: 'w-ch02-big' }, fmt(r.total)),
        h('span', { class: 'w-label' }, `float32 约 ${(r.total * 4 / 1e6).toFixed(1)} MB`)),
      h('div', { class: 'w-ch02-stack', role: 'img', 'aria-label': '各部分参数占比' },
        r.parts.map((p, i) => h('span', { title: `${p.name}：${fmt(p.n)}`, style: { width: `${(100 * p.n) / r.total}%`, background: colors[i] } }))),
      h('table', { class: 'w-ch02-table' }, h('tbody', {}, r.parts.map((p, i) => h('tr', {},
        h('td', {}, h('i', { style: { display: 'inline-block', width: '.7rem', height: '.7rem', borderRadius: '2px', marginRight: '.4rem', background: colors[i] } }), p.name),
        h('td', { class: 'n' }, fmt(p.n)), h('td', { class: 'n' }, `${((100 * p.n) / r.total).toFixed(1)}%`))))),
      h('p', { class: 'w-label' }, `对 state_dict 的张量求和：${fmt(r.stateDictSum)}`
        + (st.tied ? `（共享的 ${fmt(st.V * st.E)} 个参数被算了两次）` : '（不共享时与总数相同）')),
      h('p', { class: 'w-label' }, `粗估 12·L·E² = ${fmt(r.approx12)}：只算块内的大矩阵，嵌入和偏置不在其中。`),
    );
  }
  sync();
  render();
}
