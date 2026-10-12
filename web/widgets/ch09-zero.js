/* 图：ZeRO 显存计算器 —— 每张卡上的“模型状态”（参数、梯度、优化器状态）在 DDP 和 ZeRO-1/2/3 下各占多少。
   公式取自 ZeRO 论文（Rajbhandari et al., 2020, arXiv:1910.02054）：
     DDP      (p + g + k)·Ψ
     ZeRO-1   (p + g)·Ψ + k·Ψ/N
     ZeRO-2    p·Ψ + (g + k)·Ψ/N
     ZeRO-3   (p + g + k)·Ψ/N
   Ψ 是参数量，N 是数据并行的卡数，p/g/k 是每个参数的参数、梯度、优化器状态各占几个字节。
   激活不在这里面（见 9.2 节）。 */
import { h } from './_lib.js';

export const ACCOUNTING = {
  zero: { label: 'ZeRO 论文（2 + 2 + 12）', p: 2, g: 2, k: 12,
    note: 'ZeRO 论文的记账：fp16 参数 2 字节 + fp16 梯度 2 字节 + 优化器状态 12 字节（fp32 主参数、m、v 各 4 字节）。' },
  dlfs6: { label: 'DLFS6（4 + 4 + 8）', p: 4, g: 4, k: 8,
    note: '原书的写法：fp32 参数 4 字节 + fp32 梯度 4 字节 + 优化器状态 8 字节（m、v 各 4 字节），计算时由 autocast 临时转成 bf16。' },
};
export const STAGES = ['DDP（不切分）', 'ZeRO-1：切分优化器状态', 'ZeRO-2：再切分梯度', 'ZeRO-3：连参数也切分'];
export const COMM = [1, 1, 1, 1.5];   // 相对 DDP 的通信量（论文第 7 节）

/** 每张卡上的模型状态（字节），分成三部分。 */
export function zeroBytes(psi, nd, stage, acct) {
  const parts = { params: acct.p * psi, grads: acct.g * psi, optim: acct.k * psi };
  if (stage >= 1) parts.optim /= nd;
  if (stage >= 2) parts.grads /= nd;
  if (stage >= 3) parts.params /= nd;
  return parts;
}

const PRESETS = [
  { label: 'WebBot（152,316,672 个参数）', psi: 152_316_672 },
  { label: 'ZeRO 论文的例子（75 亿参数）', psi: 7.5e9 },
  { label: '700 亿参数（示意）', psi: 70e9 },
];
const NDS = [1, 2, 4, 8, 16, 64, 256, 1024];
const CAPS = [24, 40, 80, 141];
const PART = [['params', '参数', 'var(--ai)'], ['grads', '梯度', 'var(--moss)'], ['optim', '优化器状态', 'var(--kincha)']];

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch09-zero .bars { display: grid; grid-template-columns: minmax(7.5rem, 11rem) 1fr; gap: .55rem .8rem; align-items: center; margin: .8rem 0 .4rem; }
.w-ch09-zero .lab { font-size: .82rem; color: var(--ink-2); line-height: 1.3; }
.w-ch09-zero .track { position: relative; height: 1.5rem; background: var(--paper-2); border-radius: 3px; }
.w-ch09-zero .seg { position: absolute; top: 0; bottom: 0; }
.w-ch09-zero .cap { position: absolute; top: -.25rem; bottom: -.25rem; width: 0; border-left: 2px dashed var(--shu); }
.w-ch09-zero .val { position: absolute; top: 1.55rem; left: 0; font-size: .75rem; color: var(--ink-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
.w-ch09-zero .row { margin-bottom: .9rem; }
.w-ch09-zero .legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .8rem; color: var(--ink-2); }
.w-ch09-zero .legend span::before { content: ''; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; background: var(--c); }
.w-ch09-zero select { font: inherit; font-size: .85rem; max-width: 100%; }
.w-ch09-zero label { max-width: 100%; }
@media (max-width: 520px) { .w-ch09-zero .bars { grid-template-columns: 1fr; gap: .15rem; } .w-ch09-zero .row { margin-bottom: 1.2rem; } }
`));
}

const gb = (b) => b / 1e9;
const fmtGB = (b) => {
  const v = gb(b);
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3)} GB`;
};

export function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch09-zero');
  const state = { preset: config.preset ?? 1, nd: config.nd ?? 64, acct: config.acct ?? 'zero', cap: config.cap ?? 80 };
  const mk = (label, options, key, parse = Number) => {
    const s = h('select', { 'aria-label': label },
      options.map(([v, text]) => h('option', { value: v, selected: String(v) === String(state[key]) }, text)));
    s.addEventListener('change', () => { state[key] = parse(s.value); render(); });
    return h('label', { class: 'w-label' }, `${label} `, s);
  };
  const controls = h('div', {},
    h('div', { class: 'w-row' }, mk('模型', PRESETS.map((p, i) => [i, p.label]), 'preset')),
    h('div', { class: 'w-row' },
      mk('数据并行的卡数', NDS.map((n) => [n, `${n} 张`]), 'nd'),
      mk('每张卡的显存', CAPS.map((c) => [c, `${c} GB`]), 'cap')),
    h('div', { class: 'w-row' }, mk('记账方式', Object.entries(ACCOUNTING).map(([k, a]) => [k, a.label]), 'acct', String)),
  );
  const legend = h('div', { class: 'legend' },
    PART.map(([, name, c]) => h('span', { style: `--c:${c}` }, name)),
    h('span', { style: '--c:transparent;border-left:2px dashed var(--shu);padding-left:4px' }, '每张卡的显存'));
  const bars = h('div', { class: 'bars' });
  const acctNote = h('p', { class: 'w-label', style: 'margin:.2rem 0 .6rem' });
  const summary = h('p', { class: 'w-label', 'aria-live': 'polite' });
  root.append(controls, acctNote, legend, bars, summary);

  function render() {
    const psi = PRESETS[state.preset].psi;
    const acct = ACCOUNTING[state.acct];
    const capBytes = state.cap * 1e9;
    acctNote.textContent = acct.note;
    const rows = STAGES.map((_, s) => zeroBytes(psi, state.nd, s, acct));
    const totals = rows.map((r) => r.params + r.grads + r.optim);
    const scale = Math.max(totals[0], capBytes) * 1.04;
    bars.replaceChildren(...rows.flatMap((parts, s) => {
      let left = 0;
      const segs = PART.map(([key, name, color]) => {
        const w = (100 * parts[key]) / scale;
        const seg = h('div', { class: 'seg', style: `left:${left}%;width:${w}%;background:${color}`, title: `${name} ${fmtGB(parts[key])}` });
        left += w;
        return seg;
      });
      const fits = totals[s] <= capBytes;
      return [
        h('div', { class: 'lab' }, STAGES[s]),
        h('div', { class: 'row' }, h('div', {
          class: 'track', role: 'img',
          'aria-label': `${STAGES[s]}：每张卡 ${fmtGB(totals[s])}，${fits ? '放得下' : '放不下'}，通信量是 DDP 的 ${COMM[s]} 倍`,
        }, segs, h('div', { class: 'cap', style: `left:${(100 * capBytes) / scale}%` }),
        h('div', { class: 'val' }, `${fmtGB(totals[s])}/卡 · ${fits ? '放得下' : '放不下'} · 通信 ×${COMM[s]}`))),
      ];
    }));
    summary.textContent = `${PRESETS[state.preset].label}，${state.nd} 张卡：每张卡上的模型状态依次是 `
      + totals.map(fmtGB).join(' → ')
      + `。参数、梯度、优化器状态合计 ${acct.p + acct.g + acct.k} 字节/参数；ZeRO-1、2 的通信量与 DDP 相同，ZeRO-3 是 1.5 倍。`
      + (state.nd === 1 ? '只有 1 张卡时没有什么可切分的。' : '');
  }
  render();
}
