/* 图：同一个向量的 LayerNorm 与 RMSNorm（γ = 1、β = 0）。
   拖动左边输入向量的柱子（或用方向键），整体平移、缩放，观察两种归一化的输出；虚线是各自的均值。 */
import { h } from './_lib.js';
import { seriesColor } from './_plot.js';
import { injectStyle, radioButtons, button, svg } from './ch05-common.js';

export const mean = (x) => x.reduce((s, v) => s + v, 0) / x.length;
export const rms = (x) => Math.sqrt(mean(x.map((v) => v * v)));
export function layerNorm(x, eps = 1e-5) {
  const mu = mean(x), v = mean(x.map((t) => (t - mu) ** 2));
  return x.map((t) => (t - mu) / Math.sqrt(v + eps));
}
export function rmsNorm(x, eps = 1e-5) {
  const r = Math.sqrt(mean(x.map((t) => t * t)) + eps);
  return x.map((t) => t / r);
}

export const FINFO_EPS = 2 ** -23;    // torch.finfo(torch.float32).eps，nn.RMSNorm(eps=None) 的默认值

const fmt = (v) => {
  const a = Math.abs(v);
  if (a < 1e-9) return '0.000';                      // 均值这类量的舍入噪声，例如 -1.39e-17
  if (a < 1e-3 || a >= 1e4) return v.toExponential(2);
  return v.toFixed(a >= 100 ? 0 : 3);
};

/** One bar panel drawn at the holder's pixel width; set(values) updates the bars in place. */
function barPanel(holder, { lim, color, editable, label }) {
  const W = Math.max(180, holder.clientWidth || 220), H = 150, pad = 8;
  const n = 8, band = (W - 2 * pad) / n, bw = band * 0.62;
  const sy = (v) => pad + (1 - (Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim)) * (H - 2 * pad);
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: editable ? 'group' : 'img', 'aria-label': label });
  s.append(svg('line', { x1: pad, x2: W - pad, y1: sy(0), y2: sy(0), stroke: 'var(--ink-3)', 'stroke-width': 1 }));
  const rects = Array.from({ length: n }, (_, i) => {
    const r = svg('rect', { class: 'bar', x: pad + band * (i + 0.5) - bw / 2, width: bw, rx: 2, fill: color,
      tabindex: editable ? 0 : null, 'data-i': i });
    s.append(r);
    return r;
  });
  const meanLine = svg('line', { x1: pad, x2: W - pad, stroke: 'var(--ink-2)', 'stroke-width': 1, 'stroke-dasharray': '4 3' });
  s.append(meanLine, svg('text', { x: W - pad, y: pad + 9, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--ink-3)' }, `±${fmt(lim)}`));
  holder.replaceChildren(s);
  return {
    s, lim, H, pad, band,
    set(values) {
      values.forEach((v, i) => {
        const y0 = sy(0), y1 = sy(v);
        rects[i].setAttribute('y', Math.min(y0, y1));
        rects[i].setAttribute('height', Math.max(1, Math.abs(y1 - y0)));
        rects[i].setAttribute('aria-label', `第 ${i + 1} 个分量 = ${fmt(v)}`);
      });
      const m = sy(mean(values));
      meanLine.setAttribute('y1', m);
      meanLine.setAttribute('y2', m);
    },
  };
}

export function mount(root, config = {}) {
  injectStyle();
  const x0 = (config.x || [1.2, -0.4, 2.0, 0.7, -1.1, 0.3, 1.6, 0.9]).slice(0, 8);
  let x = x0.slice(), eps = 1e-5;
  const box = h('div', { class: 'w-ch05 w-ch05-norm' });
  root.append(box);
  box.append(h('style', {}, `
.w-ch05-norm .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .8rem 1rem; margin-top: .5rem; }
.w-ch05-norm .panel svg { display: block; width: 100%; overflow: visible; }
.w-ch05-norm .panel.input svg { cursor: ns-resize; touch-action: none; }
.w-ch05-norm .panel .stat { font-size: .8rem; color: var(--ink-2); font-variant-numeric: tabular-nums; margin-top: .15rem; }
.w-ch05-norm rect.bar:focus { outline: none; stroke: var(--ink); stroke-width: 2; }
`));

  const epsSel = radioButtons([[1e-5, 'eps = 1e-5（03_rmsnorm.py）'], [FINFO_EPS, 'eps = 1.19e-7（nn.RMSNorm 默认）']], eps,
    (v) => { eps = v; update(); }, 'eps 的取值');
  const ops = h('div', { class: 'btns' },
    button('+1（平移）', () => { x = x.map((v) => v + 1); render(); }, '每个分量加 1'),
    button('×10（缩放）', () => { x = x.map((v) => v * 10); render(); }, '每个分量乘以 10'),
    button('×0.001（缩小）', () => { x = x.map((v) => v * 0.001); render(); }, '每个分量乘以 0.001'),
    button('复原', () => { x = x0.slice(); render(); }));
  const panels = h('div', { class: 'panels' });
  box.append(h('div', { class: 'ctl' }, ops), h('div', { class: 'ctl' }, epsSel), panels);

  const mk = (cls, title) => {
    const holder = h('div'), stat = h('div', { class: 'stat', 'aria-live': cls === 'input' ? null : 'polite' });
    panels.append(h('div', { class: `panel ${cls}` }, h('div', { class: 'panel-title' }, title), holder, stat));
    return { holder, stat };
  };
  const pIn = mk('input', '输入 x（可拖动柱子）');
  const pLN = mk('ln', 'LayerNorm(x)');
  const pRMS = mk('rms', 'RMSNorm(x)');
  let gIn, gLN, gRMS, dragging = -1;

  function update() {
    const ln = layerNorm(x, eps), rn = rmsNorm(x, eps);
    gIn.set(x); gLN.set(ln); gRMS.set(rn);
    pIn.stat.textContent = `均值 ${fmt(mean(x))}，RMS ${fmt(rms(x))}`;
    pLN.stat.textContent = `输出：均值 ${fmt(mean(ln))}，RMS ${fmt(rms(ln))}`;
    pRMS.stat.textContent = `输出：均值 ${fmt(mean(rn))}，RMS ${fmt(rms(rn))}`;
  }

  function render() {        // rebuild the panels (new input scale); outputs always use ±3
    const lim = Math.max(...x.map(Math.abs)) * 1.25 || 1;
    gIn = barPanel(pIn.holder, { lim, color: seriesColor(0), editable: true, label: '输入向量：每根柱子可以拖动，或聚焦后用上下方向键调整' });
    gLN = barPanel(pLN.holder, { lim: 3, color: seriesColor(2), label: 'LayerNorm 输出的柱状图' });
    gRMS = barPanel(pRMS.holder, { lim: 3, color: seriesColor(1), label: 'RMSNorm 输出的柱状图' });
    wire(gIn);
    update();
  }

  function valueAt(g, clientY) {
    const rect = g.s.getBoundingClientRect(), scale = rect.height / g.H;
    const frac = 1 - ((clientY - rect.top) / scale - g.pad) / (g.H - 2 * g.pad);
    return +Math.max(-g.lim, Math.min(g.lim, frac * 2 * g.lim - g.lim)).toFixed(3);
  }
  function wire(g) {
    g.s.addEventListener('pointerdown', (e) => {
      const rect = g.s.getBoundingClientRect(), scale = rect.width / g.s.viewBox.baseVal.width;
      const i = Math.floor(((e.clientX - rect.left) / scale - g.pad) / g.band);
      if (i < 0 || i >= x.length) return;
      dragging = i;
      g.s.setPointerCapture(e.pointerId);
      x[i] = valueAt(g, e.clientY);
      update();
    });
    g.s.addEventListener('pointermove', (e) => { if (dragging >= 0) { x[dragging] = valueAt(g, e.clientY); update(); } });
    const end = () => { if (dragging >= 0) { dragging = -1; render(); } };
    g.s.addEventListener('pointerup', end);
    g.s.addEventListener('pointercancel', end);
    g.s.addEventListener('keydown', (e) => {
      const i = +e.target?.dataset?.i;
      if (!(e.key === 'ArrowUp' || e.key === 'ArrowDown') || Number.isNaN(i)) return;
      e.preventDefault();
      x[i] = +(x[i] + (e.shiftKey ? 1 : 0.1) * (e.key === 'ArrowUp' ? 1 : -1)).toFixed(3);
      if (Math.abs(x[i]) > g.lim) { render(); gIn.s.querySelector(`rect[data-i="${i}"]`)?.focus(); } else update();
    });
  }

  let lastW = 0;
  new ResizeObserver(() => { const w = panels.clientWidth; if (w && w !== lastW && dragging < 0) { lastW = w; render(); } }).observe(panels);
  render();
}
