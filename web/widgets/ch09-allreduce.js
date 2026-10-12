/* 图：Ring all-reduce —— W 张 GPU 排成一个环。每张卡把自己的梯度切成 W 块：
   先做 W−1 步 reduce-scatter（每步把一块发给右边的邻居，对方加到自己那块上），
   再做 W−1 步 all-gather（把已经求好和的块沿环传一圈）。最后每张卡都拿到总和，
   而每张卡只发送了 2(W−1)/W·N 个数。 */
import { h } from './_lib.js';
import { mulberry32 } from './ch09-util.js';

/** GPU r 的梯度：W 块，每块是 1–9 的一个整数，方便心算。种子固定；各张卡读的数据不同，梯度也互不相同。 */
export function initialGrads(W, seed = 90) {
  const rnd = mulberry32(seed + W);
  return Array.from({ length: W }, () => Array.from({ length: W }, () => 1 + Math.floor(9 * rnd())));
}

const mod = (a, W) => ((a % W) + W) % W;

/** 每一步之后的状态：vals[r][c] 是 GPU r 上第 c 块的值，cnt[r][c] 是它已经包含了几张卡的贡献。 */
export function ringTrace(grads) {
  const W = grads.length;
  let vals = grads.map((row) => row.slice());
  let cnt = grads.map((row) => row.map(() => 1));
  const states = [{ vals, cnt, phase: 'start', moves: [] }];
  for (const phase of ['reduce-scatter', 'all-gather']) {
    for (let s = 0; s < W - 1; s++) {
      const nv = vals.map((row) => row.slice());
      const nc = cnt.map((row) => row.slice());
      const moves = [];
      for (let r = 0; r < W; r++) {
        const to = (r + 1) % W;
        const c = phase === 'reduce-scatter' ? mod(r - s, W) : mod(r + 1 - s, W);
        if (phase === 'reduce-scatter') {        // 收到的块加到自己的同一块上
          nv[to][c] = vals[to][c] + vals[r][c];
          nc[to][c] = cnt[to][c] + cnt[r][c];
        } else {                                  // 收到的块已经是总和，直接覆盖
          nv[to][c] = vals[r][c];
          nc[to][c] = cnt[r][c];
        }
        moves.push({ from: r, to, chunk: c });
      }
      vals = nv;
      cnt = nc;
      states.push({ vals, cnt, phase, moves });
    }
  }
  return states;
}

const PHASE = {
  start: '开始：每张 GPU 只有自己那份梯度。',
  'reduce-scatter': 'reduce-scatter：每张卡把一块发给右边的邻居，对方把它加到自己的同一块上。',
  'all-gather': 'all-gather：把已经求好和的块沿环往右传，收到的直接覆盖。',
};

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch09-ar table { border-collapse: separate; border-spacing: 3px; margin: .4rem 0; display: table; font-variant-numeric: tabular-nums; }
.w-ch09-ar th { font-weight: 600; font-size: .78rem; color: var(--ink-2); padding: 0 .3rem; border: 0; text-align: center; min-width: 0; }
.w-ch09-ar th[scope=row] { text-align: right; white-space: nowrap; }
.w-ch09-ar td { width: 2.4rem; min-width: 0; height: 2.3rem; text-align: center; border-radius: 4px; padding: 0; border: 1px solid var(--rule);
  font: 600 .95rem/1.1 var(--mono); position: relative; background: var(--paper); }
.w-ch09-ar td small { display: block; font: 400 .62rem/1 var(--serif); color: var(--ink-3); }
.w-ch09-ar td.full { border-color: var(--ai); box-shadow: inset 0 0 0 1px var(--ai); }
.w-ch09-ar td.recv { outline: 2px solid var(--kincha); outline-offset: 1px; }
.w-ch09-ar td.sent { outline: 2px dashed var(--ink-3); outline-offset: 1px; }
.w-ch09-ar .sum td { border-style: dashed; color: var(--ink-2); background: transparent; }
.w-ch09-ar .moves { font-size: .82rem; color: var(--ink-2); min-height: 1.4em; }
@media (max-width: 420px) {
  .w-ch09-ar table { border-spacing: 2px; }
  .w-ch09-ar th { padding: 0 .15rem; font-size: .7rem; }
  .w-ch09-ar td { width: 1.75rem; height: 2rem; font-size: .8rem; }
  .w-ch09-ar td small { font-size: .55rem; }
}
`));
}

export function mount(root, config = {}) {
  injectStyle();
  let W, grads, sums, states;
  let k = 0;
  const setW = (w) => {
    W = w;
    grads = initialGrads(W);
    sums = grads[0].map((_, c) => grads.reduce((acc, row) => acc + row[c], 0));
    states = ringTrace(grads);
    k = 0;
  };
  setW(config.W ?? 4);

  const sel = h('select', { 'aria-label': 'GPU 的张数 W' },
    [2, 3, 4, 6, 8].map((n) => h('option', { value: n, selected: n === W }, `${n} 张 GPU`)));
  const prev = h('button', { type: 'button' }, '‹ 上一步');
  const next = h('button', { type: 'button' }, '下一步 ›');
  const reset = h('button', { type: 'button' }, '重置');
  const status = h('div', { class: 'w-label', 'aria-live': 'polite' });
  const grid = h('div', { style: 'overflow-x:auto' });
  const moves = h('div', { class: 'moves' });
  const traffic = h('div', { class: 'w-label' });
  root.classList.add('w-ch09-ar');
  root.append(h('div', { class: 'w-row' }, sel, prev, next, reset), status, grid, moves, traffic);

  function cell(st, r, c, recv, sent) {
    const v = st.vals[r][c], n = st.cnt[r][c], key = `${r},${c}`;
    const cls = [n === W && 'full', recv.has(key) && 'recv', sent.has(key) && !recv.has(key) && 'sent'].filter(Boolean).join(' ');
    return h('td', {
      class: cls, style: `background: color-mix(in srgb, var(--ai-2) ${Math.round((100 * n) / W)}%, var(--paper))`,
      'aria-label': `GPU ${r} 的块 ${c}：${v}，已包含 ${n} 张卡的贡献`,
    }, String(v), h('small', {}, `${n}/${W}`));
  }

  function describeMoves(st) {
    const legend = `虚线框发出，橙框接收，蓝框表示已经汇总了全部 ${W} 张卡。`;
    if (!st.moves.length) return `每格下方的 n/W 表示这个值已经汇总了几张卡的梯度。${legend}`;
    const what = W > 4 ? `${W} 张卡同时各发出 1 块`
      : st.moves.map((m) => `GPU ${m.from} → GPU ${m.to}：块 ${m.chunk}`).join('；');
    return `${what}。${legend}`;
  }

  function render() {
    const st = states[k];
    const total = states.length - 1;
    const recv = new Set(st.moves.map((m) => `${m.to},${m.chunk}`));
    const sent = new Set(st.moves.map((m) => `${m.from},${m.chunk}`));
    const head = h('tr', {}, h('th', {}, ''), sums.map((_, c) => h('th', { scope: 'col' }, `块${c}`)));
    const rows = st.vals.map((row, r) => h('tr', {}, h('th', { scope: 'row' }, `GPU ${r}`),
      row.map((_, c) => cell(st, r, c, recv, sent))));
    const sumRow = h('tr', { class: 'sum' }, h('th', { scope: 'row' }, '各卡之和'), sums.map((v) => h('td', {}, String(v))));
    grid.replaceChildren(h('table', {}, h('thead', {}, head), h('tbody', {}, rows, sumRow)));

    const done = k === total;
    status.textContent = `第 ${k} / ${total} 步。${PHASE[st.phase]}`;
    moves.textContent = describeMoves(st);
    traffic.textContent = `到这一步为止，每张 GPU 发送了 ${k} 块（每块 N/W 个数）。全部完成需要 ${total} 块，即 2(W−1)/W·N `
      + `≈ ${(2 * (W - 1) / W).toFixed(2)}·N；如果都先发给 GPU 0 再广播，GPU 0 要收 ${W - 1}·N、再发 ${W - 1}·N。`
      + (done ? ' 现在每张卡都拿到了总和，再除以 W 就是平均梯度。' : '');
    prev.disabled = k === 0;
    next.disabled = done;
  }
  next.addEventListener('click', () => { if (k < states.length - 1) { k += 1; render(); } });
  prev.addEventListener('click', () => { if (k > 0) { k -= 1; render(); } });
  reset.addEventListener('click', () => { k = 0; render(); });
  sel.addEventListener('change', () => { setW(+sel.value); render(); });
  render();
}
