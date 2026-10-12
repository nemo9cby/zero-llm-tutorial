/* 图：KV cache 的 prefill 与 decode。行是 query（正在计算的 token），列是 key。
   无 cache：每一步重算整个下三角；有 cache：prefill 算一次 prompt 的下三角，之后每一步只算新的一行。
   “prefill 不加 mask”重现 8b3911f 之前的 bug：prompt 内部出现指向未来的格子（prompt 只有 1 个 token 时不会出现）。 */
import { h } from './_lib.js';
import { seriesColor } from './_plot.js';
import { injectStyle, slider, radioButtons, button, reducedMotion, svg, fmtInt } from './ch05-common.js';

/** Cells (query i, key j) computed at step s (0-based). kind: 'new' | 'redo' | 'future'. */
export function stepCells(mode, P, s, buggy = false) {
  const cells = [];
  if (mode === 'nocache') {
    const T = P + s;                                     // all tokens so far are fed again
    for (let i = 0; i < T; i++) {
      for (let j = 0; j <= i; j++) cells.push({ i, j, kind: s > 0 && i < T - 1 ? 'redo' : 'new' });
    }
    return cells;
  }
  if (s === 0) {                                         // prefill: the whole prompt at once
    for (let i = 0; i < P; i++) {
      for (let j = 0; j < P; j++) {
        if (j <= i) cells.push({ i, j, kind: 'new' });
        else if (buggy) cells.push({ i, j, kind: 'future' });
      }
    }
    return cells;
  }
  const i = P + s - 1;                                   // decode: only the newest token's row
  for (let j = 0; j <= i; j++) cells.push({ i, j, kind: 'new' });
  return cells;
}

/** {tokens, scores} processed at step s. */
export function stepCounts(mode, P, s, buggy = false) {
  const cells = stepCells(mode, P, s, buggy);
  return { tokens: new Set(cells.map((c) => c.i)).size, scores: cells.length };
}

export function mount(root, config = {}) {
  injectStyle();
  const N = config.total ?? 12;
  let P = config.prompt ?? 4, mode = 'cache', buggy = false, s = 0, timer = 0;
  const steps = () => N - P;                             // step s produces token P + s
  const box = h('div', { class: 'w-ch05 w-ch05-kv' });
  root.append(box);
  box.append(h('style', {}, `
.w-ch05-kv .stage { display: flex; flex-wrap: wrap; gap: .6rem 1.4rem; align-items: flex-start; margin-top: .4rem; }
.w-ch05-kv .grid svg { display: block; max-width: 100%; height: auto; }
.w-ch05-kv .side { flex: 1 1 14rem; min-width: 0; }
.w-ch05-kv .chips { display: flex; flex-wrap: wrap; gap: 2px; margin: .15rem 0 .5rem; font: 11px/1 var(--mono); }
.w-ch05-kv .chip { padding: 3px 4px; border-radius: 3px; background: var(--paper-3); color: var(--ink-2); min-width: 1.9em; text-align: center; }
.w-ch05-kv .chip.fresh { background: var(--ai); color: var(--paper); }
.w-ch05-kv .chip.redo { background: color-mix(in srgb, ${seriesColor(1)} 35%, var(--paper)); color: var(--ink); }
.w-ch05-kv .legend { display: flex; flex-wrap: wrap; gap: .2rem .9rem; font-size: .78rem; color: var(--ink-2); margin: .3rem 0; }
.w-ch05-kv .legend span::before { content: ''; display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; background: var(--c); }
.w-ch05-kv .phase { font-weight: 600; color: var(--ink); margin: .2rem 0; }
`));

  const colors = () => ({ new: seriesColor(0), redo: `color-mix(in srgb, ${seriesColor(1)} 55%, var(--paper))`,
    past: 'var(--ai-2)', future: seriesColor(7) });

  const modeSel = radioButtons([['nocache', '无 cache（每步全部重算）'], ['cache', '有 cache（prefill + decode）']], mode,
    (v) => { mode = v; bugBox.disabled = v !== 'cache'; s = 0; draw(); }, '生成方式');
  const sP = slider({ label: 'prompt 长度', min: 1, max: 6, value: P, onInput: (v) => { P = v; s = 0; draw(); } });
  const bugBox = h('input', { type: 'checkbox', 'aria-label': 'prefill 时不加因果掩码（commit 8b3911f 之前的写法）' });
  bugBox.addEventListener('change', () => { buggy = bugBox.checked; draw(); });
  const prev = button('上一步', () => { stop(); s = Math.max(0, s - 1); draw(); });
  const next = button('下一步', () => { stop(); s = Math.min(steps() - 1, s + 1); draw(); });
  const play = button('播放', () => (timer ? stop() : start()), '自动播放每一步');
  const reset = button('从头开始', () => { stop(); s = 0; draw(); });
  box.append(h('div', { class: 'ctl' }, modeSel),
    h('div', { class: 'ctl' }, sP.el, h('label', {}, bugBox, 'prefill 不加 mask（8b3911f 之前）')),
    h('div', { class: 'btns' }, prev, next, play, reset));

  const legend = h('div', { class: 'legend' });
  const gridBox = h('div', { class: 'grid' });
  const phase = h('div', { class: 'phase', 'aria-live': 'polite' });
  const kRow = h('div', { class: 'chips' }), vRow = h('div', { class: 'chips' });
  const counts = h('div', { class: 'readout' });
  const side = h('div', { class: 'side' }, phase, h('div', { class: 'panel-title' }, 'K cache'), kRow,
    h('div', { class: 'panel-title' }, 'V cache'), vRow, counts);
  box.append(legend, h('div', { class: 'stage' }, gridBox, side));

  function start() {
    if (reducedMotion()) { s = Math.min(steps() - 1, s + 1); draw(); return; }
    play.textContent = '暂停';
    timer = setInterval(() => { if (s >= steps() - 1) { stop(); return; } s += 1; draw(); }, 1100);
  }
  function stop() { clearInterval(timer); timer = 0; play.textContent = '播放'; }

  function draw() {
    const c = colors();
    legend.replaceChildren(
      h('span', { style: `--c:${c.new}` }, '本步新算'),
      mode === 'nocache' ? h('span', { style: `--c:${c.redo}` }, '本步重复计算') : h('span', { style: `--c:${c.past}` }, '之前算过（结果已用掉）'),
      ...(mode === 'cache' && buggy ? [h('span', { style: `--c:${c.future}` }, '看见了未来')] : []));
    const avail = Math.max(220, (gridBox.parentElement?.clientWidth || 360));
    const lab = 26, cs = Math.max(14, Math.min(26, Math.floor((Math.min(avail, 420) - lab - 4) / N)));
    const W = lab + cs * N + 2, H = lab + cs * N + 2;
    const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
      'aria-label': `第 ${s + 1} 步计算的注意力分数：行是 query，列是 key` });
    for (let t = 0; t < N; t++) {
      const isPrompt = t < P;
      g.append(svg('text', { x: lab + cs * (t + 0.5), y: lab - 7, 'text-anchor': 'middle', 'font-size': 10,
        fill: isPrompt ? 'var(--ink)' : 'var(--ink-3)' }, `k${t}`));
      g.append(svg('text', { x: lab - 4, y: lab + cs * (t + 0.5) + 3.5, 'text-anchor': 'end', 'font-size': 10,
        fill: isPrompt ? 'var(--ink)' : 'var(--ink-3)' }, `q${t}`));
    }
    const cur = new Map(stepCells(mode, P, s, buggy).map((x) => [`${x.i},${x.j}`, x.kind]));
    const past = new Set();
    if (mode === 'cache') for (let t = 0; t < s; t++) for (const x of stepCells(mode, P, t, buggy)) past.add(`${x.i},${x.j}`);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const key = `${i},${j}`, kind = cur.get(key);
        const fill = kind ? c[kind] : past.has(key) ? c.past : 'transparent';
        g.append(svg('rect', { x: lab + j * cs + 1, y: lab + i * cs + 1, width: cs - 2, height: cs - 2, rx: 2, fill,
          stroke: j <= i ? 'var(--rule)' : 'none', 'stroke-width': 1 }));
      }
    }
    // prompt boundary
    g.append(svg('rect', { x: lab, y: lab, width: cs * P, height: cs * P, fill: 'none', stroke: 'var(--ink-3)',
      'stroke-width': 1.2, 'stroke-dasharray': '3 2' }));
    gridBox.replaceChildren(g);

    const len = mode === 'cache' ? (s === 0 ? P : P + s) : P + s;     // tokens whose K, V exist after this step
    const fresh = mode === 'cache' ? (s === 0 ? [...Array(P).keys()] : [P + s - 1]) : [...Array(len).keys()];
    const chip = (t) => h('span', { class: `chip ${fresh.includes(t) ? (mode === 'nocache' && s > 0 && t < len - 1 ? 'redo' : 'fresh') : ''}` }, String(t));
    kRow.replaceChildren(...Array.from({ length: len }, (_, t) => chip(t)));
    vRow.replaceChildren(...Array.from({ length: len }, (_, t) => chip(t)));
    const { tokens, scores } = stepCounts(mode, P, s, buggy);
    let cumT = 0, cumS = 0;
    for (let t = 0; t <= s; t++) { const x = stepCounts(mode, P, t, buggy); cumT += x.tokens; cumS += x.scores; }
    phase.textContent = mode === 'nocache'
      ? `第 ${s + 1} 步：把全部 ${P + s} 个 token 重新送进模型，生成第 ${P + s} 号 token`
      : s === 0 ? `第 1 步 prefill：一次处理 prompt 的 ${P} 个 token${buggy && P > 1 ? '（没有掩码！）' : ''}`
        : `第 ${s + 1} 步 decode：只处理第 ${P + s - 1} 号 token，用缓存里的 ${P + s} 组 K、V`;
    counts.replaceChildren(
      h('div', {}, '本步处理的 token：', h('b', {}, fmtInt(tokens)), '，注意力分数：', h('b', {}, fmtInt(scores))),
      h('div', {}, `到第 ${s + 1} 步为止累计：token `, h('b', {}, fmtInt(cumT)), '，分数 ', h('b', {}, fmtInt(cumS))),
      h('div', {}, mode === 'nocache' ? 'K、V 不保存，每一步都重算' : '新算的 K、V 追加到缓存的末尾'));
    prev.disabled = s === 0;
    next.disabled = s >= steps() - 1;
  }
  bugBox.disabled = mode !== 'cache';
  let lastW = 0;
  new ResizeObserver(() => { const w = box.clientWidth; if (w && w !== lastW) { lastW = w; draw(); } }).observe(box);
  draw();
}
