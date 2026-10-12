/* 图 8-2：Flash Attention 的分块顺序。
   外层逐块取 query，内层逐块取 key/value；每处理一块，就用 online softmax 修正每行的最大值 m 与指数和 ℓ。
   计算全部用 Float64，在页面上与朴素 softmax attention 对拍；同一结论由验证 8.2-c 钉住。 */
import { h } from './_lib.js';

/** 可复现的伪随机数（mulberry32）加 Box–Muller，生成 N×d 的高斯矩阵。 */
export function gaussianMatrix(rows, cols, seed) {
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const u = Math.max(rand(), 1e-12), v = rand();
      row.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
    }
    out.push(row);
  }
  return out;
}

const dot = (x, y) => x.reduce((s, xi, i) => s + xi * y[i], 0);

export function naiveAttention(Q, K, V, causal = true) {
  const d = Q[0].length;
  return Q.map((q, r) => {
    const s = K.map((k, c) => (causal && c > r ? -Infinity : dot(q, k) / Math.sqrt(d)));
    const m = Math.max(...s);
    const p = s.map((x) => Math.exp(x - m));
    const l = p.reduce((a, b) => a + b, 0);
    return V[0].map((_, j) => p.reduce((acc, pc, c) => acc + pc * V[c][j], 0) / l);
  });
}

/** 分块计算，并记录每一步之后每行的 m、ℓ（用于动画）。 */
export function flashSteps(Q, K, V, { block, causal = true }) {
  const N = Q.length, d = Q[0].length;
  const m = new Array(N).fill(-Infinity), l = new Array(N).fill(0);
  const acc = Q.map(() => new Array(d).fill(0));
  const steps = [];
  const skipped = [];
  for (let i0 = 0; i0 < N; i0 += block) {
    const rows = [];
    for (let r = i0; r < Math.min(i0 + block, N); r++) rows.push(r);
    for (let j0 = 0; j0 < N; j0 += block) {
      if (causal && j0 > rows[rows.length - 1]) { skipped.push([i0, j0]); continue; } // 整块在对角线右上方
      const cols = [];
      for (let c = j0; c < Math.min(j0 + block, N); c++) cols.push(c);
      const raised = [];
      for (const r of rows) {
        const s = cols.map((c) => (causal && c > r ? -Infinity : dot(Q[r], K[c]) / Math.sqrt(d)));
        const mNew = Math.max(m[r], ...s);
        if (mNew === -Infinity) continue; // 这一行在本块里全被遮住（对齐的方块里不会发生）
        const scale = Math.exp(m[r] - mNew); // 最大值变了，旧的累加量要重新缩放
        const p = s.map((x) => Math.exp(x - mNew));
        if (mNew > m[r] && m[r] !== -Infinity) raised.push(r);
        l[r] = scale * l[r] + p.reduce((a, b) => a + b, 0);
        for (let j = 0; j < d; j++) acc[r][j] = scale * acc[r][j] + p.reduce((t, pc, k) => t + pc * V[cols[k]][j], 0);
        m[r] = mNew;
      }
      steps.push({ i0, j0, rows, cols, raised, m: m.slice(), l: l.slice() });
    }
  }
  const out = acc.map((row, r) => row.map((x) => x / l[r]));
  return { steps, skipped, out };
}

/** 每行在最终基准 m* 下的指数和 ℓ*（朴素做法一次算出的分母）。 */
export function finalDenominators(Q, K, causal = true) {
  const d = Q[0].length;
  return Q.map((q, r) => {
    const s = K.map((k, c) => (causal && c > r ? -Infinity : dot(q, k) / Math.sqrt(d)));
    const m = Math.max(...s);
    return { m, l: s.reduce((a, x) => a + Math.exp(x - m), 0) };
  });
}

export const maxAbsDiff = (A, B) => Math.max(...A.flatMap((row, i) => row.map((x, j) => Math.abs(x - B[i][j]))));

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch08-flash { outline: none; }
.w-ch08-flash:focus-visible { outline: 2px solid var(--ai); outline-offset: 4px; }
.w-ch08-flash .bar { display: flex; flex-wrap: wrap; gap: .4rem .6rem; align-items: center; font-size: .88rem; margin-bottom: .6rem; }
.w-ch08-flash .bar select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; }
.w-ch08-flash .stage { display: flex; flex-wrap: wrap; gap: 1rem 1.5rem; align-items: flex-start; }
.w-ch08-flash svg { width: 100%; max-width: 300px; height: auto; flex: 0 1 300px; }
.w-ch08-flash .side { flex: 1 1 15rem; min-width: 0; font-size: .86rem; }
.w-ch08-flash .rowstat { display: grid; grid-template-columns: 2.4em minmax(0, 1fr) 7.6em; gap: .25rem .5rem; align-items: center; margin: .15rem 0; font-variant-numeric: tabular-nums; }
.w-ch08-flash .meter { height: .55rem; background: var(--paper-3); border-radius: 3px; overflow: hidden; }
.w-ch08-flash .meter i { display: block; height: 100%; background: var(--ai); }
.w-ch08-flash .note { color: var(--ink-2); margin: .5rem 0 0; }
.w-ch08-flash .done { color: var(--ink); font-weight: 600; }
`));
}

const SVGNS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, text) => {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
};

export function mount(root, config = {}) {
  injectStyle();
  const N = config.n || 8, d = config.d || 4, seed = config.seed ?? 7;
  const Q = gaussianMatrix(N, d, seed), K = gaussianMatrix(N, d, seed + 1), V = gaussianMatrix(N, d, seed + 2);
  let block = config.block || 2, causal = true, k = 0, timer = null;
  let run, finals, ref;

  const box = h('div', { class: 'w-ch08-flash', tabindex: 0, 'aria-label': 'Flash Attention 分块演示，可用左右方向键单步' });
  const btn = (label, aria, fn) => h('button', { type: 'button', 'aria-label': aria, onclick: fn }, label);
  const stepLabel = h('span', { class: 'w-label', 'aria-live': 'polite' });
  const playBtn = btn('自动播放', '自动播放或暂停', () => (timer ? stop() : play()));
  const blockSel = h('select', { 'aria-label': '块的大小' }, [1, 2, 4].map((b) => h('option', { value: b }, `块大小 ${b}`)));
  blockSel.value = String(block);
  const causalBox = h('input', { type: 'checkbox', checked: true, 'aria-label': '因果掩码' });
  const svgEl = svg('svg', { role: 'img', 'aria-label': '分数矩阵的分块：行是 query，列是 key' });
  const side = h('div', { class: 'side' });
  box.append(
    h('div', { class: 'bar' },
      btn('重置', '回到第 0 步', () => { stop(); k = 0; draw(); }),
      btn('上一步', '上一步', () => { stop(); k = Math.max(0, k - 1); draw(); }),
      btn('下一步', '下一步', () => { stop(); k = Math.min(run.steps.length, k + 1); draw(); }),
      playBtn, blockSel, h('label', {}, causalBox, ' 因果掩码'), stepLabel),
    h('div', { class: 'stage' }, svgEl, side),
  );
  root.append(box);

  function compute() {
    run = flashSteps(Q, K, V, { block, causal });
    finals = finalDenominators(Q, K, causal);
    ref = naiveAttention(Q, K, V, causal);
    k = 0;
  }

  function play() {
    if (k >= run.steps.length) k = 0;
    playBtn.textContent = '暂停';
    timer = setInterval(() => {
      if (k >= run.steps.length) { stop(); return; }
      k += 1;
      draw();
    }, 900);
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    playBtn.textContent = '自动播放';
  }

  function draw() {
    const cell = 30, lab = 26, W = lab + N * cell + 2;
    svgEl.setAttribute('viewBox', `0 0 ${W} ${W}`);
    svgEl.replaceChildren();
    const defs = svg('defs');
    const pat = svg('pattern', { id: 'w-ch08-hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
    pat.append(svg('rect', { width: 6, height: 6, fill: 'var(--paper)' }), svg('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: 'var(--rule)', 'stroke-width': 3 }));
    defs.append(pat);
    svgEl.append(defs);
    const done = new Set(run.steps.slice(0, k).map((s) => `${s.i0},${s.j0}`));
    const cur = k > 0 ? run.steps[k - 1] : null;
    const skipped = new Set(run.skipped.map(([i, j]) => `${i},${j}`));
    for (let r = 0; r < N; r++) {
      svgEl.append(svg('text', { x: lab - 4, y: lab + r * cell + cell / 2 + 4, 'text-anchor': 'end', 'font-size': 10, fill: 'var(--ink-3)' }, `q${r}`));
      svgEl.append(svg('text', { x: lab + r * cell + cell / 2, y: lab - 6, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--ink-3)' }, `k${r}`));
      for (let c = 0; c < N; c++) {
        const tile = `${Math.floor(r / block) * block},${Math.floor(c / block) * block}`;
        let fill = 'var(--paper-2)';
        if (skipped.has(tile)) fill = 'url(#w-ch08-hatch)';
        else if (causal && c > r) fill = 'var(--paper-3)';
        else if (cur && tile === `${cur.i0},${cur.j0}`) fill = 'color-mix(in srgb, var(--ai) 60%, var(--paper))';
        else if (done.has(tile)) fill = 'color-mix(in srgb, var(--ai) 22%, var(--paper))';
        svgEl.append(svg('rect', { x: lab + c * cell + 1, y: lab + r * cell + 1, width: cell - 2, height: cell - 2, rx: 2, fill }));
        if (causal && c > r && !skipped.has(tile)) {
          svgEl.append(svg('text', { x: lab + c * cell + cell / 2, y: lab + r * cell + cell / 2 + 4, 'text-anchor': 'middle',
            'font-size': 10, fill: 'var(--ink-3)' }, '−∞'));
        }
      }
    }
    for (let i = 0; i < N; i += block) {
      for (let j = 0; j < N; j += block) {
        svgEl.append(svg('rect', { x: lab + j * cell, y: lab + i * cell, width: block * cell, height: block * cell,
          fill: 'none', stroke: 'var(--ink-3)', 'stroke-width': 1 }));
      }
    }
    if (cur) {
      svgEl.append(svg('rect', { x: lab + cur.j0 * cell, y: lab + cur.i0 * cell, width: cur.cols.length * cell, height: cur.rows.length * cell,
        fill: 'none', stroke: 'var(--ai)', 'stroke-width': 3 }));
    }

    const total = run.steps.length;
    stepLabel.textContent = `第 ${k} / ${total} 步`;
    const lastState = cur || { m: new Array(N).fill(-Infinity), l: new Array(N).fill(0) };
    const rows = cur ? cur.rows : Array.from({ length: Math.min(block, N) }, (_, i) => i);
    const stats = rows.map((r) => {
      const m = lastState.m[r], l = lastState.l[r];
      const frac = m === -Infinity ? 0 : (l * Math.exp(m - finals[r].m)) / finals[r].l; // 把 ℓ 换算到最终基准
      return h('div', { class: 'rowstat' },
        h('span', {}, `q${r}`),
        h('span', { class: 'meter', role: 'meter', 'aria-valuemin': 0, 'aria-valuemax': 1, 'aria-valuenow': frac.toFixed(3), 'aria-label': `q${r} 的分母已累加的比例` },
          h('i', { style: { width: `${(100 * frac).toFixed(1)}%` } })),
        h('span', {}, m === -Infinity ? 'm = −∞' : `m=${m.toFixed(2)} ℓ=${l.toFixed(2)}`));
    });
    const msg = !cur
      ? '还没有开始：每行的 m = −∞，ℓ = 0。点击“下一步”处理第一块。'
      : `第 ${cur.i0 / block + 1} 块 query（q${cur.rows[0]}–q${cur.rows[cur.rows.length - 1]}）× 第 ${cur.j0 / block + 1} 块 key（k${cur.cols[0]}–k${cur.cols[cur.cols.length - 1]}）`
        + (cur.j0 === 0
          ? '：这几行第一次读到分数，m 从 −∞ 变成这一块里的最大值。'
          : cur.raised.length
            ? `：${cur.raised.map((r) => `q${r}`).join('、')} 遇到了更大的分数，旧的 ℓ 和输出乘以 e^(m−m′) 修正。`
            : '：最大值没有变化，直接累加。');
    const err = maxAbsDiff(run.out, ref);
    side.replaceChildren(...[
      h('div', { class: 'w-label' }, '当前这一块 query 的各行：进度条是分母 ℓ 已累加的比例（换算到最终的最大值之后）'),
      ...stats,
      h('p', { class: 'note' }, msg),
      h('p', { class: 'note' }, `朴素实现要同时摆出 ${N}×${N} = ${N * N} 个分数；分块实现此刻只有 ${block}×${block} = ${block * block} 个，外加每行的 m、ℓ 和输出累加器。`
        + (causal ? `因果掩码让斜线所示的 ${run.skipped.length} 块被整块跳过，标着 −∞ 的格子是块内被遮住的分数。` : '')),
      k === total ? h('p', { class: 'note done' }, `全部 ${total} 块处理完毕：与朴素 softmax attention 的最大差 ${err.toExponential(1)}。`) : null,
    ].filter(Boolean));
  }

  blockSel.addEventListener('change', () => { stop(); block = +blockSel.value; compute(); draw(); });
  causalBox.addEventListener('change', () => { stop(); causal = causalBox.checked; compute(); draw(); });
  box.addEventListener('keydown', (e) => {
    if (e.target.closest('select, input')) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); stop(); k = Math.min(run.steps.length, k + 1); draw(); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); stop(); k = Math.max(0, k - 1); draw(); }
  });
  compute();
  draw();
}
