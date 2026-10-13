/* Shared SVG charts: lineChart, barChart, heatmap.
   Palette = the validated dataviz reference instance (categorical slots in fixed order,
   sequential blue, diverging blue↔red with a gray midpoint), light and dark steps.
   Every chart ships a hover/focus tooltip and an optional data table (<details>).
   Also the theme helpers isDark() / onThemeChange(fn) for widgets that colour things in JS.
   Nothing here touches the DOM at import time, so Node tests can import it. */
import { h } from './_lib.js';

const NS = 'http://www.w3.org/2000/svg';
const LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
const DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
  '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
const DIV_NEG = '#2a78d6', DIV_POS = '#e34948';
const DIV_MID_LIGHT = '#f0efec', DIV_MID_DARK = '#383835';
const ON_DARK = '#ffffff', ON_LIGHT = '#1b2233';      // value labels on heat-map cells: white or the ink

// ---------------------------------------------------------------- theme

/** True when the page shows the dark theme: the toggle's data-theme wins, otherwise the OS setting. */
export function isDark() {
  if (typeof document === 'undefined') return false;
  const t = document.documentElement.dataset.theme;
  if (t) return t === 'dark';
  return typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches;
}

const themeListeners = new Set();
let themeWatched = false, lastDark = false;

/** Call fn(dark) once per change of the effective theme, whether it comes from the page's toggle (data-theme)
    or from the OS setting while the page follows it. Returns an unsubscribe function; outside a browser it
    does nothing. A listener that throws is reported and does not stop the others. */
export function onThemeChange(fn) {
  if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') return () => {};
  if (!themeWatched) {
    themeWatched = true;
    lastDark = isDark();
    const check = () => {
      const dark = isDark();
      if (dark === lastDark) return;
      lastDark = dark;
      for (const f of [...themeListeners]) {
        try { f(dark); } catch (err) { console.error(err); }
      }
    };
    if (typeof matchMedia === 'function') matchMedia('(prefers-color-scheme: dark)').addEventListener('change', check);
    new MutationObserver(check).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }
  themeListeners.add(fn);
  return () => { themeListeners.delete(fn); };
}

export const seriesColor = (i) => (isDark() ? DARK : LIGHT)[i % 8];

// ---------------------------------------------------------------- colour arithmetic

const hexRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const fromLinear = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
const rgbHex = (rgb) => `#${rgb.map((c) => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, '0')).join('')}`;

/** WCAG 2 relative luminance of a #rrggbb colour. */
export function relativeLuminance(hex) {
  const [r, g, b] = hexRgb(hex).map(toLinear);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG 2 contrast ratio of two #rrggbb colours, from 1 to 21. */
export function contrastRatio(a, b) {
  const [hi, lo] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** The colour for text drawn on `fill` (#rrggbb): white or the ink, whichever has the higher WCAG contrast. */
export function textColorOn(fill) {
  return contrastRatio(ON_DARK, fill) > contrastRatio(ON_LIGHT, fill) ? ON_DARK : ON_LIGHT;
}

// Oklab (Björn Ottosson's matrices on linear sRGB), the space CSS color-mix(in oklab, ...) interpolates in.
function toOklab(hex) {
  const [r, g, b] = hexRgb(hex).map(toLinear);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
}
function fromOklab([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  return rgbHex([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s].map(fromLinear));
}

/** color-mix(in oklab, a p, b) for 0 ≤ p ≤ 1, as an sRGB #rrggbb (clipped to the gamut). */
export function mixOklab(a, b, p) {
  const A = toOklab(a), B = toOklab(b);
  return fromOklab(A.map((v, i) => v * p + B[i] * (1 - p)));
}

function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const k of kids.flat()) if (k != null) el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return el;
}

/** "Nice" tick values covering [lo, hi]. */
export function niceTicks(lo, hi, count = 5) {
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const start = Math.ceil(lo / step - 1e-9) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

/** niceTicks, asked for more ticks until at least `min` of them fall inside [lo, hi]: on a narrow chart
    niceTicks(−3, 3, 2) is just [0], autoTicks(−3, 3, 2) is [−2, 0, 2]. Unchanged when niceTicks has enough. */
export function autoTicks(lo, hi, count = 5, min = 3) {
  let ticks = niceTicks(lo, hi, count);
  for (let c = count + 1; ticks.length < min && c <= count + 20; c++) ticks = niceTicks(lo, hi, c);
  return ticks;
}

const MANTISSAS = [[1], [1, 2, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9]];

/** Ticks for a log axis on [lo, hi]: the powers of ten if at least `min` of them fit, else 1-2-5, else every
    integer mantissa; a span too short for any of these (1.3 to 2.6) gets linear ticks, which on so short a
    span read correctly on a log axis too. */
export function logTicks(lo, hi, min = 3) {
  lo = Math.max(lo, 1e-12);
  if (!(hi >= lo)) return [];
  for (const ms of MANTISSAS) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const m of ms) {
        const v = +(m * 10 ** e).toPrecision(12);
        if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(v);
      }
    }
    if (out.length >= min) return out;
  }
  return autoTicks(lo, hi, 4, min);
}

/** Rough width in px of an 11–11.5 px chart label (CJK characters are about twice as wide). */
const textWidth = (t) => [...String(t)].reduce((w, ch) => w + (ch.charCodeAt(0) > 0x2e80 ? 11.5 : 6.6), 0);

export const fmtNum = (v) => {
  if (v == null || !isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e7)) return v.toExponential(2);
  if (Number.isInteger(v)) return v.toLocaleString('en-US');
  return (+v.toPrecision(4)).toLocaleString('en-US', { maximumFractionDigits: 6 });
};

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.viz { position: relative; font: 12px/1.4 var(--serif); color: var(--ink-2); }
.viz svg { display: block; width: 100%; overflow: visible; }
.viz svg.viz-fixed { width: auto; max-width: 100%; height: auto; }
.viz .grid line { stroke: var(--rule); stroke-width: 1; }
.viz .axis text { fill: var(--ink-3); font-size: 11px; font-variant-numeric: tabular-nums; }
.viz .axis-title { fill: var(--ink-2); font-size: 11.5px; }
.viz .end-label { fill: var(--ink-2); font-size: 11.5px; }
.viz .crosshair { stroke: var(--ink-3); stroke-width: 1; }
.viz-ylabel { font-size: 11.5px; color: var(--ink-2); margin-bottom: .2rem; }
.viz-legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; margin: 0 0 .4rem; font-size: 12px; color: var(--ink-2); }
.viz-legend span::before { content: ''; display: inline-block; width: 14px; height: 2px; margin-right: 6px; vertical-align: middle; background: var(--c); }
.viz-legend.rect span::before { height: 10px; width: 10px; border-radius: 2px; }
.viz-tip { position: absolute; pointer-events: none; z-index: 3; background: var(--paper); color: var(--ink);
  border: 1px solid var(--rule); border-radius: 4px; padding: .35rem .55rem; font-size: 12px; line-height: 1.5;
  box-shadow: 0 2px 8px color-mix(in srgb, var(--ink) 12%, transparent); white-space: nowrap; }
.viz-tip .k { color: var(--ink-3); }
.viz-tip .row::before { content: ''; display: inline-block; width: 10px; height: 2px; margin-right: 6px; vertical-align: middle; background: var(--c); }
.viz-tip b { font-variant-numeric: tabular-nums; }
.viz details { margin-top: .4rem; font-size: 12px; }
.viz details summary { cursor: pointer; color: var(--ink-3); }
.viz table { font-size: 12px; margin: .4rem 0; }
.viz .cell:focus, .viz .bar:focus { outline: 2px solid var(--ink); outline-offset: 1px; }
`));
}

function tooltip(root) {
  const tip = h('div', { class: 'viz-tip', hidden: true, role: 'status' });
  root.append(tip);
  return {
    show(x, y, build) {
      tip.replaceChildren(...build());
      tip.hidden = false;
      const rw = root.clientWidth, tw = tip.offsetWidth;
      tip.style.left = `${Math.max(0, Math.min(x + 12, rw - tw))}px`;
      tip.style.top = `${Math.max(0, y - tip.offsetHeight - 8)}px`;
    },
    hide() { tip.hidden = true; },
  };
}

function dataTable(columns, rows) {
  return h('details', {}, h('summary', {}, '查看数据表'),
    h('table', {}, h('thead', {}, h('tr', {}, columns.map((c) => h('th', {}, c)))),
      h('tbody', {}, rows.map((r) => h('tr', {}, r.map((v) => h('td', {}, typeof v === 'number' ? fmtNum(v) : v)))))));
}

function onResize(el, fn) {
  let w = 0;
  const ro = new ResizeObserver(() => { const nw = el.clientWidth; if (nw && nw !== w) { w = nw; fn(nw); } });
  ro.observe(el);
  onThemeChange(() => fn(el.clientWidth || w || 600));       // series colours differ between the themes
}

/** Keep the ticks whose labels do not collide with the previous kept label (pos and size in px). */
function spread(ticks, pos, size, gap) {
  const out = [];
  let end = -Infinity;
  for (const v of [...ticks].sort((a, b) => pos(a) - pos(b))) {
    if (pos(v) - size(v) / 2 < end + gap) continue;
    out.push(v);
    end = pos(v) + size(v) / 2;
  }
  return out;
}

/**
 * Line chart.
 * opts: { series: [{name, points: [[x, y], ...], dash?: '4 3', width?: 2, markers?: false | true | [[x, y], ...]}],
 *         xLabel, yLabel, height=260, yMin, yMax, xMin, xMax, xFormat, yFormat, logY=false, logX=false,
 *         xTicks?: number[], marginRight?: px (default 84 with 2–4 series, else 16),
 *         refLines: [{y, label, labelAt?: 'right'|'left'}],
 *         vLines: [{x, label, labelAt?: 'right'|'left', labelY?: 'top'|'bottom'}], table=true }
 * xMin/xMax fix the x domain like yMin/yMax fix the y domain. markers: true marks every point, an array marks
 * only those points (e.g. where a sparse metric was recorded). Without xTicks there are at least 3 x ticks,
 * a log axis gets 1-2-5 ticks when powers of ten are too few, and edge labels stay inside the chart.
 * Direct end labels are drawn only when they fit in the right margin and do not collide;
 * otherwise the legend carries identity.
 * Returns { update(newOpts) } so widgets can re-render with new data.
 */
export function lineChart(root, opts) {
  injectStyle();
  const box = h('div', { class: 'viz' });
  root.append(box);
  let o = opts;
  const render = (width) => {
    box.replaceChildren();
    const { series, height = 260 } = o;
    const fx = o.xFormat || fmtNum, fy = o.yFormat || fmtNum;
    if (o.yLabel) box.append(h('div', { class: 'viz-ylabel' }, o.yLabel));
    if (series.length > 1) {
      box.append(h('div', { class: 'viz-legend' }, series.map((s, i) => h('span', { style: `--c:${seriesColor(i)}` }, s.name))));
    }
    const m = { l: 52, r: o.marginRight ?? (series.length > 1 && series.length <= 4 ? 84 : 16), t: 10, b: 38 };
    const W = Math.max(width, 240), H = height;
    const all = series.flatMap((s) => s.points);
    const tf = (v) => (o.logY ? Math.log10(Math.max(v, 1e-12)) : v);
    const gx = (v) => (o.logX ? Math.log10(Math.max(v, 1e-12)) : v);
    let x0 = o.xMin ?? Math.min(...all.map((p) => p[0])), x1 = o.xMax ?? Math.max(...all.map((p) => p[0]));
    let y0 = o.yMin ?? Math.min(...all.map((p) => p[1]), ...(o.refLines || []).map((r) => r.y));
    let y1 = o.yMax ?? Math.max(...all.map((p) => p[1]), ...(o.refLines || []).map((r) => r.y));
    if (x0 === x1) x1 = x0 + 1;
    const yt = o.logY ? null : autoTicks(y0, y1, 5);
    if (yt) { y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]); }
    const ty0 = tf(y0), ty1 = tf(y1) === ty0 ? ty0 + 1 : tf(y1);
    const gx0 = gx(x0), gx1 = gx(x1) === gx(x0) ? gx(x0) + 1 : gx(x1);
    const sx = (x) => m.l + ((gx(x) - gx0) / (gx1 - gx0)) * (W - m.l - m.r);
    const sy = (y) => m.t + (1 - (tf(y) - ty0) / (ty1 - ty0)) * (H - m.t - m.b);
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.ariaLabel || o.yLabel || '折线图' });
    const grid = svg('g', { class: 'grid' }), axis = svg('g', { class: 'axis' });
    const yTicks = yt || spread(logTicks(y0, y1).filter((v) => tf(v) >= ty0 - 1e-9 && tf(v) <= ty1 + 1e-9),
      (v) => -sy(v), () => 11, 3);
    for (const v of yTicks) {
      if (tf(v) < ty0 - 1e-9 || tf(v) > ty1 + 1e-9) continue;
      grid.append(svg('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }));
      axis.append(svg('text', { x: m.l - 6, y: sy(v) + 4, 'text-anchor': 'end' }, fy(v)));
    }
    const inX = (v) => v >= Math.min(x0, x1) && v <= Math.max(x0, x1);
    const xTicks = o.xTicks || spread((o.logX ? logTicks(x0, x1) : autoTicks(x0, x1, Math.max(2, Math.floor((W - m.l - m.r) / 90))))
      .filter(inX), sx, (v) => textWidth(fx(v)), 6);
    for (const v of xTicks) {
      if (!inX(v)) continue;
      // A label centred on a tick at the plot's edge may not fit inside the SVG: shift it in (gpu figure 3).
      const half = textWidth(fx(v)) / 2;
      axis.append(svg('text', { x: Math.min(Math.max(sx(v), half), W - half), y: H - m.b + 16, 'text-anchor': 'middle' }, fx(v)));
    }
    if (o.xLabel) axis.append(svg('text', { class: 'axis-title', x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, o.xLabel));
    s.append(grid, axis);
    for (const r of o.refLines || []) {
      s.append(svg('line', { x1: m.l, x2: W - m.r, y1: sy(r.y), y2: sy(r.y), stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 3' }));
      if (r.label) {
        const left = r.labelAt === 'left';
        s.append(svg('text', { class: 'end-label', x: left ? m.l + 4 : W - m.r - 4, y: sy(r.y) - 5,
          'text-anchor': left ? 'start' : 'end' }, r.label));
      }
    }
    for (const v of o.vLines || []) {
      s.append(svg('line', { x1: sx(v.x), x2: sx(v.x), y1: m.t, y2: H - m.b, stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 3' }));
      if (v.label) {
        // labelAt: 'right' (default) or 'left' of the line; labelY: 'top' (default) or 'bottom' of the plot.
        const left = v.labelAt === 'left' || (v.labelAt == null && sx(v.x) > W - m.r - 80);
        s.append(svg('text', { class: 'end-label', x: sx(v.x) + (left ? -4 : 4),
          y: v.labelY === 'bottom' ? H - m.b - 6 : m.t + 10, 'text-anchor': left ? 'end' : 'start' }, v.label));
      }
    }
    series.forEach((ser, i) => {
      const d = ser.points.map((p, j) => `${j ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('');
      s.append(svg('path', { d, fill: 'none', stroke: seriesColor(i), 'stroke-width': ser.width || 2,
        'stroke-dasharray': ser.dash || null, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
      const marks = ser.markers === true ? ser.points : (Array.isArray(ser.markers) ? ser.markers : []);
      // 4 px dots while they have room (every chart that had markers before); dense ones shrink to 1.5 px.
      const r = +Math.min(4, Math.max(1.5, (W - m.l - m.r) / Math.max(1, marks.length) / 3)).toFixed(2);
      for (const p of marks) {
        s.append(svg('circle', { cx: sx(p[0]), cy: sy(p[1]), r, fill: seriesColor(i), stroke: 'var(--paper)', 'stroke-width': Math.min(2, r / 2) }));
      }
    });
    // Direct end labels only when they stay apart; converging lines fall back to the legend.
    const ends = series.map((ser) => ser.points[ser.points.length - 1]).filter(Boolean);
    const ys = ends.map((p) => sy(p[1])).sort((a, b) => a - b);
    const apart = ys.every((y, j) => j === 0 || y - ys[j - 1] >= 13);
    const fits = series.every((ser) => textWidth(ser.name) <= m.r - 8);
    if (series.length > 1 && series.length <= 4 && apart && fits) {
      series.forEach((ser) => {
        const last = ser.points[ser.points.length - 1];
        if (last) s.append(svg('text', { class: 'end-label', x: sx(last[0]) + 6, y: sy(last[1]) + 4 }, ser.name));
      });
    }
    // Crosshair + tooltip listing every series at the nearest x.
    const xs = [...new Set(all.map((p) => p[0]))].sort((a, b) => a - b);
    const cross = svg('line', { class: 'crosshair', y1: m.t, y2: H - m.b, visibility: 'hidden' });
    const hit = svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent', tabindex: 0,
      'aria-label': '用左右方向键查看数据点' });
    s.append(cross, hit);
    box.append(s);
    const tip = tooltip(box);
    let idx = -1;
    const showAt = (k) => {
      idx = Math.max(0, Math.min(xs.length - 1, k));
      const x = xs[idx];
      cross.setAttribute('x1', sx(x)); cross.setAttribute('x2', sx(x)); cross.setAttribute('visibility', 'visible');
      const rect = s.getBoundingClientRect(), scale = rect.width / W;
      tip.show(sx(x) * scale, m.t * scale + 20, () => [
        h('div', { class: 'k' }, `${o.xLabel || 'x'} = ${fx(x)}`),
        ...series.map((ser, i) => {
          if (!ser.points.length) return null;
          const p = ser.points.reduce((best, q) => (Math.abs(q[0] - x) < Math.abs(best[0] - x) ? q : best), ser.points[0]);
          return h('div', { class: 'row', style: `--c:${seriesColor(i)}` }, h('b', {}, fy(p[1])), ' ', h('span', { class: 'k' }, ser.name));
        }).filter(Boolean),
      ]);
    };
    hit.addEventListener('pointermove', (e) => {
      const rect = s.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      const gv = gx0 + ((px - m.l) / (W - m.l - m.r)) * (gx1 - gx0);     // the pointer, in axis units (log10 on a log axis)
      let k = 0; for (let j = 1; j < xs.length; j++) if (Math.abs(gx(xs[j]) - gv) < Math.abs(gx(xs[k]) - gv)) k = j;
      showAt(k);
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hide(); });
    hit.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); showAt((idx < 0 ? 0 : idx) + (e.key === 'ArrowRight' ? 1 : -1)); }
    });
    hit.addEventListener('blur', () => { cross.setAttribute('visibility', 'hidden'); tip.hide(); });
    if (o.table !== false) {
      const cols = [o.xLabel || 'x', ...series.map((x) => x.name)];
      const rows = xs.slice(0, 400).map((x) => [x, ...series.map((ser) => ser.points.find((p) => p[0] === x)?.[1] ?? '')]);
      box.append(dataTable(cols, rows));
    }
  };
  onResize(root, render);
  render(root.clientWidth || 600);
  return { update(next) { o = { ...o, ...next }; render(root.clientWidth || 600); } };
}

/**
 * Bar chart (vertical columns).
 * opts: { bars: [{label, value, series?}], yLabel, yFormat, height=220, highlight: label, table=true }
 */
export function barChart(root, opts) {
  injectStyle();
  const box = h('div', { class: 'viz' });
  root.append(box);
  let o = opts;
  const render = (width) => {
    box.replaceChildren();
    const { bars, height = 220 } = o;
    const fy = o.yFormat || fmtNum;
    if (o.yLabel) box.append(h('div', { class: 'viz-ylabel' }, o.yLabel));
    const m = { l: 48, r: 10, t: 16, b: 34 };
    const W = Math.max(width, 240), H = height;
    const vmax = Math.max(0, ...bars.map((b) => b.value)), vmin = Math.min(0, ...bars.map((b) => b.value));
    const ticks = niceTicks(vmin, vmax, 4);
    const y0 = Math.min(vmin, ticks[0]), y1 = Math.max(vmax, ticks[ticks.length - 1]);
    const sy = (v) => m.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
    const band = (W - m.l - m.r) / bars.length;
    const bw = Math.min(24, band * 0.6);
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.ariaLabel || o.yLabel || '柱状图' });
    const grid = svg('g', { class: 'grid' }), axis = svg('g', { class: 'axis' });
    for (const v of ticks) {
      grid.append(svg('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }));
      axis.append(svg('text', { x: m.l - 6, y: sy(v) + 4, 'text-anchor': 'end' }, fy(v)));
    }
    s.append(grid, axis);
    const tip = tooltip(box);
    bars.forEach((b, i) => {
      const cx = m.l + band * (i + 0.5);
      const top = sy(Math.max(b.value, 0)), bot = sy(Math.min(b.value, 0));
      const hgt = Math.max(1, bot - top);
      const r = Math.min(4, hgt / 2);
      const up = b.value >= 0;
      // 4px rounded data-end, square at the baseline.
      const x = cx - bw / 2;
      const d = up
        ? `M${x},${bot}V${top + r}Q${x},${top} ${x + r},${top}H${x + bw - r}Q${x + bw},${top} ${x + bw},${top + r}V${bot}Z`
        : `M${x},${top}V${bot - r}Q${x},${bot} ${x + r},${bot}H${x + bw - r}Q${x + bw},${bot} ${x + bw},${bot - r}V${top}Z`;
      const color = seriesColor(b.series ?? 0);
      const bar = svg('path', { class: 'bar', d, fill: color, tabindex: 0, opacity: o.highlight && o.highlight !== b.label ? 0.45 : 1,
        'aria-label': `${b.label}: ${fy(b.value)}` });
      const show = () => {
        const rect = s.getBoundingClientRect(), sc = rect.width / W;
        tip.show(cx * sc, top * sc, () => [h('div', {}, h('b', {}, fy(b.value))), h('div', { class: 'k' }, b.label)]);
      };
      bar.addEventListener('pointerenter', show); bar.addEventListener('focus', show);
      bar.addEventListener('pointerleave', () => tip.hide()); bar.addEventListener('blur', () => tip.hide());
      s.append(bar);
      if (bars.length <= 12) s.append(svg('text', { class: 'end-label', x: cx, y: (up ? top - 4 : bot + 12), 'text-anchor': 'middle' }, fy(b.value)));
      axis.append(svg('text', { x: cx, y: H - m.b + 15, 'text-anchor': 'middle' }, b.label));
    });
    s.append(svg('line', { x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0), stroke: 'var(--ink-3)', 'stroke-width': 1 }));
    box.append(s);
    if (o.table !== false) box.append(dataTable(['类别', o.yLabel || '值'], bars.map((b) => [b.label, b.value])));
  };
  onResize(root, render);
  render(root.clientWidth || 600);
  return { update(next) { o = { ...o, ...next }; render(root.clientWidth || 600); } };
}

const seqIndex = (t) => {
  // Light theme: low = light, high = dark. Dark theme: the ramp runs the other way, so low values
  // recede into the dark surface and high values stand out.
  const k = Math.max(0, Math.min(SEQ.length - 1, Math.round(t * (SEQ.length - 1))));
  return isDark() ? SEQ.length - 1 - k : k;
};
const SEQ_WHITE = SEQ.map((c) => textColorOn(c) === ON_DARK);        // per step: does white text win?
/** Sequential (one hue, light→dark) color for t in [0,1]; divColor below is the diverging (blue↔gray↔red) one. */
export function seqColor(t) {
  return SEQ[seqIndex(t)];
}
/** True when a label on seqColor(t) should be white: white has the higher WCAG contrast on that exact fill. */
export function seqIsDark(t) {
  return SEQ_WHITE[seqIndex(t)];
}
/** The diverging fill for t in [−1, 1], as #rrggbb: the oklab mix that color-mix(in oklab, end |t|, mid) gives,
    computed here so that the label colour can be chosen against the very same colour. */
export function divColor(t) {
  const mid = isDark() ? DIV_MID_DARK : DIV_MID_LIGHT;
  const a = Math.round(Math.min(1, Math.abs(t)) * 100) / 100;       // whole percent, as before
  if (!(a > 0)) return mid;
  const end = t < 0 ? DIV_NEG : DIV_POS;
  return a >= 1 ? end : mixOklab(end, mid, a);
}

/**
 * Heatmap. opts: { values: number[][], rowLabels, colLabels, scale: 'seq'|'div', domain: [lo, hi],
 *   cellFormat, masked: (r, c) => bool, size (max cell px, default 34), title, table=true }
 */
export function heatmap(root, opts) {
  injectStyle();
  const box = h('div', { class: 'viz' });
  root.append(box);
  let o = opts;
  const render = (width) => {
    box.replaceChildren();
    const { values } = o;
    const R = values.length, C = values[0]?.length || 0;
    const fv = o.cellFormat || fmtNum;
    const flat = values.flat().filter((v) => isFinite(v));
    const [lo, hi] = o.domain || [Math.min(...flat), Math.max(...flat)];
    const lab = 44;
    const cell = Math.max(10, Math.min(o.size || 34, Math.floor((Math.max(width, 200) - lab - 4) / C)));
    const W = lab + cell * C + 2, H = lab + cell * R + 2;
    const s = svg('svg', { class: 'viz-fixed', viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
      'aria-label': o.ariaLabel || '热力图' });
    const axis = svg('g', { class: 'axis' });
    (o.colLabels || []).forEach((t, c) => axis.append(svg('text', { x: lab + cell * (c + 0.5), y: lab - 6, 'text-anchor': 'middle' }, t)));
    (o.rowLabels || []).forEach((t, r) => axis.append(svg('text', { x: lab - 6, y: lab + cell * (r + 0.5) + 4, 'text-anchor': 'end' }, t)));
    s.append(axis);
    const tip = tooltip(box);
    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        const v = values[r][c];
        const masked = o.masked?.(r, c) || !isFinite(v);
        const t = o.scale === 'div' ? v / Math.max(Math.abs(lo), Math.abs(hi) || 1) : (v - lo) / (hi - lo || 1);
        const fill = masked ? 'var(--paper-3)' : (o.scale === 'div' ? divColor(t) : seqColor(t));
        const rect = svg('rect', { class: 'cell', x: lab + c * cell + 1, y: lab + r * cell + 1, width: cell - 2, height: cell - 2, rx: 2,
          fill, tabindex: 0, 'aria-label': `${o.rowLabels?.[r] ?? r} → ${o.colLabels?.[c] ?? c}: ${masked ? '被遮住' : fv(v)}` });
        const show = () => {
          const b = s.getBoundingClientRect(), sc = b.width / W;
          tip.show((lab + (c + 1) * cell) * sc, (lab + r * cell) * sc, () => [
            h('div', {}, h('b', {}, masked ? '被遮住' : fv(v))),
            h('div', { class: 'k' }, `行 ${o.rowLabels?.[r] ?? r} · 列 ${o.colLabels?.[c] ?? c}`),
          ]);
        };
        rect.addEventListener('pointerenter', show); rect.addEventListener('focus', show);
        rect.addEventListener('pointerleave', () => tip.hide()); rect.addEventListener('blur', () => tip.hide());
        s.append(rect);
        if (cell >= 30 && !masked && o.showValues !== false) {
          // White or ink, whichever contrasts more with this cell's actual fill (the fills differ by theme).
          s.append(svg('text', { x: lab + cell * (c + 0.5), y: lab + cell * (r + 0.5) + 4, 'text-anchor': 'middle',
            'font-size': 10, fill: textColorOn(fill), 'pointer-events': 'none' }, fv(v)));
        }
      }
    }
    box.append(s);
    if (o.table !== false) {
      box.append(dataTable(['', ...(o.colLabels || [...Array(C).keys()])],
        values.map((row, r) => [o.rowLabels?.[r] ?? r, ...row.map((v, c) => (o.masked?.(r, c) ? '—' : v))])));
    }
  };
  onResize(root, render);
  render(root.clientWidth || 600);
  return { update(next) { o = { ...o, ...next }; render(root.clientWidth || 600); } };
}
