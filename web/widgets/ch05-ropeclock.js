/* 图：高维 RoPE。上：位置 m 被编码成 d/2 根转速不同的指针（第 i 根转 m·θ_i，θ_i = Θ^(−2i/d)）。
   下：同一个 q、k 放在不同位置时的分数矩阵 S[m, n]：RoPE 下沿对角线不变（Toeplitz），
   换成“加到 q、k 上的绝对位置向量”后条纹消失。 */
import { h } from './_lib.js';
import { heatmap, seriesColor } from './_plot.js';
import { injectStyle, slider, radioButtons, rng, svg } from './ch05-common.js';

export const invFreq = (d, base) => Array.from({ length: d / 2 }, (_, i) => base ** (-2 * i / d));

/** RoPE of one vector at position pos (interleaved pairs, like the book). */
export function ropeVec(x, pos, base = 10000) {
  const f = invFreq(x.length, base);
  const out = x.slice();
  f.forEach((w, i) => {
    const c = Math.cos(pos * w), s = Math.sin(pos * w), e = x[2 * i], o = x[2 * i + 1];
    out[2 * i] = e * c - o * s;
    out[2 * i + 1] = e * s + o * c;
  });
  return out;
}

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

/** S[m][n] for m, n in 0..T-1. mode 'rope' rotates q, k; mode 'abs' adds position vectors P[m]. */
export function scoreMatrix(q, k, T, mode, base = 10000, P = null) {
  return Array.from({ length: T }, (_, m) => Array.from({ length: T }, (_, n) => (mode === 'rope'
    ? dot(ropeVec(q, m, base), ropeVec(k, n, base))
    : dot(q.map((v, j) => v + P[m][j]), k.map((v, j) => v + P[n][j])))));
}

export function mount(root, config = {}) {
  injectStyle();
  const d = config.d ?? 16, T = 10;
  let base = config.base ?? 10000, m = config.m ?? 3, mode = 'rope';
  const box = h('div', { class: 'w-ch05 w-ch05-clock' });
  root.append(box);
  box.append(h('style', {}, `
.w-ch05-clock .dials { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: .3rem .4rem; margin: .3rem 0 .2rem; }
.w-ch05-clock .dial { text-align: center; font-size: 11px; color: var(--ink-2); line-height: 1.35; }
.w-ch05-clock .dial svg { width: 52px; height: 52px; display: block; margin: 0 auto; }
.w-ch05-clock .heat { margin-top: .4rem; max-width: 420px; }
`));

  // ---- panel A: the multi-hand clock
  const dials = h('div', { class: 'dials', role: 'img', 'aria-label': `位置 m 上 ${d / 2} 根指针的角度` });
  const sM = slider({ label: '位置 m', min: 0, max: 200, value: m, onInput: (v) => { m = v; drawDials(); } });
  const baseSel = radioButtons([[10000, 'Θ = 10,000'], [1000000, 'Θ = 1,000,000']], base,
    (v) => { base = v; drawDials(); drawHeat(); }, 'RoPE 的常数 Θ');
  box.append(h('div', { class: 'panel-title' }, `上：第 i 根指针转过 m·θ_i 弧度（d = ${d}，共 ${d / 2} 根）`),
    h('div', { class: 'ctl' }, sM.el, baseSel), dials);

  function drawDials() {
    dials.replaceChildren();
    invFreq(d, base).forEach((w, i) => {
      const a = m * w;                                   // radians turned so far
      const x = 26 + 20 * Math.cos(a), y = 26 - 20 * Math.sin(a);
      const turns = a / (2 * Math.PI);
      const s = svg('svg', { viewBox: '0 0 52 52', 'aria-hidden': 'true' },
        svg('circle', { cx: 26, cy: 26, r: 23, fill: 'none', stroke: 'var(--rule)', 'stroke-width': 1.2 }),
        svg('line', { x1: 26, y1: 26, x2: 48, y2: 26, stroke: 'var(--rule)', 'stroke-width': 1 }),
        svg('line', { x1: 26, y1: 26, x2: x, y2: y, stroke: seriesColor(0), 'stroke-width': 2.5, 'stroke-linecap': 'round' }),
        svg('circle', { cx: 26, cy: 26, r: 2.2, fill: 'var(--ink-2)' }));
      const wl = 2 * Math.PI / w;
      dials.append(h('div', { class: 'dial', title: `θ_${i} = ${w.toPrecision(3)}，波长 ${wl.toFixed(1)}，已转 ${turns.toFixed(2)} 圈` },
        s, `i=${i}`, h('br'), `波长 ${wl < 100 ? wl.toFixed(1) : Math.round(wl).toLocaleString('en-US')}`));
    });
  }

  // ---- panel B: S[m, n] heatmap
  const r = rng(7);
  const q = Array.from({ length: d }, () => r.normal());
  const k = Array.from({ length: d }, () => r.normal());
  const P = Array.from({ length: T }, () => Array.from({ length: d }, () => r.normal()));
  const heatBox = h('div', { class: 'heat' });
  const modeSel = radioButtons([['rope', 'RoPE'], ['abs', '绝对位置（相加）']], mode,
    (v) => { mode = v; drawHeat(); }, '位置编码方式');
  const note = h('div', { class: 'readout' });
  box.append(h('div', { class: 'panel-title', style: 'margin-top:1rem' },
    '下：同一个 q 放在位置 m（行）、同一个 k 放在位置 n（列）时的分数 S[m, n]'),
  h('div', { class: 'ctl' }, modeSel), heatBox, note);
  let chart = null;

  function drawHeat() {
    const S = scoreMatrix(q, k, T, mode, base, P);
    const lim = Math.max(...S.flat().map(Math.abs));
    const opts = {
      values: S, rowLabels: S.map((_, i) => `m=${i}`), colLabels: S.map((_, i) => `n=${i}`),
      scale: 'div', domain: [-lim, lim], cellFormat: (v) => v.toFixed(1), size: 34,
      ariaLabel: '分数矩阵 S[m, n] 的热力图',
    };
    if (chart) chart.update(opts); else chart = heatmap(heatBox, opts);
    const diag = Array.from({ length: T - 3 }, (_, i) => S[i + 3][i]);
    const spread = Math.max(...diag) - Math.min(...diag);
    note.textContent = mode === 'rope'
      ? `m − n = 3 的 ${diag.length} 个格子：分数都是 ${diag[0].toFixed(3)}（最大差 ${spread.toExponential(1)}）`
      : `m − n = 3 的 ${diag.length} 个格子：分数从 ${Math.min(...diag).toFixed(1)} 到 ${Math.max(...diag).toFixed(1)}`;
  }

  drawDials();
  drawHeat();
}
