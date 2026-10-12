/* 图：多头的形状追踪（2.8.4 节）。同一块内存（C=4 个位置 × H=3 个头 × D=2 维），
   正确的拆法 view(C,H,D) → transpose 让每个头拿到全部位置；错误的拆法直接 view(H,C,D)，
   把不同位置的数据塞进同一个“头”，标出来自“未来”的格子。与验证 2.8-e 的结论相同（JS 测试对拍）。 */
import { h } from './_lib.js';
import { seriesColor } from './_plot.js';
import { injectStyle, range } from './ch02-util.js';

export const C = 4, H = 3, D = 2;

/** Element stored at flat memory index m of the (C, H*D) row-major buffer. */
export const elementAt = (m) => ({ p: Math.floor(m / (H * D)), h: Math.floor((m % (H * D)) / D), d: m % D });

/** Panels of the (H, C, D) tensor: panel[a][r][c] = memory index.
    right: view(C, H, D).transpose(0, 1) -> index p*H*D + a*D + c with p = r.
    wrong: view(H, C, D)                 -> index a*C*D + r*D + c (consecutive chunks). */
export function panels(kind) {
  return range(H).map((a) => range(C).map((r) => range(D).map((c) =>
    (kind === 'right' ? r * H * D + a * D + c : a * C * D + r * D + c))));
}

/** Cells whose data comes from a later position than the row they are placed in. */
export function futureCells(kind) {
  const out = [];
  panels(kind).forEach((pan, a) => pan.forEach((row, r) => row.forEach((m) => {
    if (elementAt(m).p > r) out.push({ a, r, m });
  })));
  return out;
}

const STEPS = [
  { key: 'buf', code: 'Q = W_q(x)', shape: '(C, H·D) = (4, 6)',
    text: '线性层一次算出所有头的查询：每一行是一个位置，6 个数依次属于头 0、头 1、头 2。内存里按行连续存放。' },
  { key: 'view', code: 'Q.view(C, H, D)', shape: '(4, 3, 2)',
    text: 'view 不移动任何数据，只是换一种“数法”：把每一行切成 3 段，每段 2 个数。位置还是那个位置。' },
  { key: 'right', code: '.transpose(0, 1)', shape: '(H, C, D) = (3, 4, 2)',
    text: 'transpose 把“头”换到前面：每个面板是一个头，4 行正好是位置 0–3。数据仍然没有搬动，只是读的顺序变了，所以张量不再“连续”。' },
  { key: 'wrong', code: 'Q.view(H, C, D)  # 错误', shape: '(3, 4, 2)',
    text: '形状一样，含义却错了：每个面板只是内存里连续的 8 个数。面板里名义上的“位置 0”可能装着别的头的数据，描了边的格子，装的是比所在行更靠后的位置的数据。因果掩码只遮住名义上的未来，真正的未来就这样漏了进来。' },
  { key: 'merge', code: '.transpose(0, 1).contiguous().view(C, H*D)', shape: '(4, 6)',
    text: '合并时先 transpose 回 (C, H, D)。此时张量不连续，PyTorch 的 view 会报 RuntimeError；先 .contiguous() 按新的顺序复制一份（或改用 reshape），再 view 成 (4, 6)，每行又是一个位置的全部 6 个数。' },
];

export function mount(root) {
  injectStyle();
  let step = 0;
  const btns = STEPS.map((s, i) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(i === step) }, `${i + 1}. ${s.code.split('  #')[0]}`);
    b.addEventListener('click', () => { step = i; render(); });
    return b;
  });
  const shape = h('p', { class: 'w-label', 'aria-live': 'polite' });
  const view = h('div');
  const text = h('p', { class: 'w-label' });
  const legend = h('div', { class: 'w-ch02-legend' },
    range(H).map((k) => h('span', {}, h('i', { style: { background: seriesColor(k) } }), `头 ${k} 的数据`)),
    h('span', {}, h('i', { style: { background: 'transparent', outline: '2px solid var(--kincha)' } }), '来自“未来”的位置'));
  root.append(h('div', { class: 'w-ch02-controls' }, h('span', { class: 'w-ch02-seg', role: 'group', 'aria-label': '形状变换的步骤' }, btns)),
    shape, legend, view, text);

  const chipFor = (m, r) => {
    const e = elementAt(m);
    const future = r != null && e.p > r;
    return h('span', {
      class: 'cell', title: `内存第 ${m} 个数：位置 ${e.p}、头 ${e.h}、第 ${e.d} 维`,
      'aria-label': `位置 ${e.p} 头 ${e.h}${future ? '，来自未来' : ''}`,
      style: { background: `color-mix(in srgb, ${seriesColor(e.h)} 28%, var(--paper))`, color: 'var(--ink)',
        outline: future ? '2px solid var(--kincha)' : 'none', outlineOffset: '-2px' },
    }, `p${e.p}`);
  };
  const gridOf = (rows, cols, cells, labels) => h('div', {
    class: 'w-ch02-grid', style: { gridTemplateColumns: `2.6em repeat(${cols}, minmax(1.8em, 2.6em))` },
  }, rows.flatMap((r) => [h('span', { class: 'cell head' }, labels[r]), ...cells(r)]));

  function render() {
    btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === step)));
    const s = STEPS[step];
    shape.replaceChildren(h('code', {}, s.code), ` → 形状 ${s.shape}`);
    text.textContent = s.text;
    const rowLabels = range(C).map((p) => `位置${p}`);
    if (s.key === 'buf' || s.key === 'merge') {
      view.replaceChildren(gridOf(range(C), H * D, (p) => range(H * D).map((j) => chipFor(p * H * D + j)), rowLabels));
    } else if (s.key === 'view') {
      view.replaceChildren(h('div', { class: 'w-ch02-grid', style: { gridTemplateColumns: `2.6em repeat(${H}, auto)` } },
        range(C).flatMap((p) => [h('span', { class: 'cell head' }, rowLabels[p]),
          ...range(H).map((k) => h('span', { style: { display: 'flex', gap: '2px' } }, range(D).map((d) => chipFor(p * H * D + k * D + d))))])));
    } else {
      const pans = panels(s.key);
      view.replaceChildren(h('div', { class: 'w-ch02-two', style: { gridTemplateColumns: 'repeat(auto-fit, minmax(8.5rem, 1fr))' } },
        pans.map((pan, a) => h('div', {}, h('div', { class: 'w-label' }, `头 ${a}`),
          gridOf(range(C), D, (r) => pan[r].map((m) => chipFor(m, r)), rowLabels)))));
    }
  }
  render();
}
