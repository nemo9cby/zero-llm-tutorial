/* Shared SVG charts: lineChart, barChart, heatmap.
   Palette = the validated dataviz reference instance (categorical slots in fixed order,
   sequential blue, diverging blue↔red with a gray midpoint), light and dark steps.
   Every chart ships a hover/focus tooltip and an optional data table (<details>). */
import { h } from './_lib.js';

const NS = 'http://www.w3.org/2000/svg';
const LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
const DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
  '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
const DIV_NEG = '#2a78d6', DIV_POS = '#e34948';

const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
};
export const seriesColor = (i) => (isDark() ? DARK : LIGHT)[i % 8];

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
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => fn(el.clientWidth));
  new MutationObserver(() => fn(el.clientWidth)).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}

/**
 * Line chart.
 * opts: { series: [{name, points: [[x, y], ...]}], xLabel, yLabel, height=260, yMin, yMax,
 *         xFormat, yFormat, logY=false, refLines: [{y, label}], vLines: [{x, label}], table=true }
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
    const m = { l: 52, r: series.length > 1 && series.length <= 4 ? 84 : 16, t: 10, b: 38 };
    const W = Math.max(width, 240), H = height;
    const all = series.flatMap((s) => s.points);
    const tf = (v) => (o.logY ? Math.log10(Math.max(v, 1e-12)) : v);
    let x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
    let y0 = o.yMin ?? Math.min(...all.map((p) => p[1]), ...(o.refLines || []).map((r) => r.y));
    let y1 = o.yMax ?? Math.max(...all.map((p) => p[1]), ...(o.refLines || []).map((r) => r.y));
    if (x0 === x1) x1 = x0 + 1;
    const yt = o.logY ? null : niceTicks(y0, y1, 5);
    if (yt) { y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]); }
    const ty0 = tf(y0), ty1 = tf(y1) === ty0 ? ty0 + 1 : tf(y1);
    const sx = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
    const sy = (y) => m.t + (1 - (tf(y) - ty0) / (ty1 - ty0)) * (H - m.t - m.b);
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.ariaLabel || o.yLabel || '折线图' });
    const grid = svg('g', { class: 'grid' }), axis = svg('g', { class: 'axis' });
    const yTicks = yt || (() => { const out = []; for (let e = Math.floor(ty0); e <= Math.ceil(ty1); e++) out.push(10 ** e); return out; })();
    for (const v of yTicks) {
      if (tf(v) < ty0 - 1e-9 || tf(v) > ty1 + 1e-9) continue;
      grid.append(svg('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }));
      axis.append(svg('text', { x: m.l - 6, y: sy(v) + 4, 'text-anchor': 'end' }, fy(v)));
    }
    for (const v of niceTicks(x0, x1, Math.max(2, Math.floor((W - m.l - m.r) / 90)))) {
      if (v < x0 || v > x1) continue;
      axis.append(svg('text', { x: sx(v), y: H - m.b + 16, 'text-anchor': 'middle' }, fx(v)));
    }
    if (o.xLabel) axis.append(svg('text', { class: 'axis-title', x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, o.xLabel));
    s.append(grid, axis);
    for (const r of o.refLines || []) {
      s.append(svg('line', { x1: m.l, x2: W - m.r, y1: sy(r.y), y2: sy(r.y), stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 3' }));
      if (r.label) s.append(svg('text', { class: 'end-label', x: W - m.r - 4, y: sy(r.y) - 5, 'text-anchor': 'end' }, r.label));
    }
    for (const v of o.vLines || []) {
      s.append(svg('line', { x1: sx(v.x), x2: sx(v.x), y1: m.t, y2: H - m.b, stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 3' }));
      if (v.label) s.append(svg('text', { class: 'end-label', x: sx(v.x) + 4, y: m.t + 10 }, v.label));
    }
    series.forEach((ser, i) => {
      const d = ser.points.map((p, j) => `${j ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('');
      s.append(svg('path', { d, fill: 'none', stroke: seriesColor(i), 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
      const last = ser.points[ser.points.length - 1];
      if (last && series.length > 1 && series.length <= 4) {
        s.append(svg('text', { class: 'end-label', x: sx(last[0]) + 6, y: sy(last[1]) + 4 }, ser.name));
      }
    });
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
          const p = ser.points.reduce((best, q) => (Math.abs(q[0] - x) < Math.abs(best[0] - x) ? q : best), ser.points[0]);
          return h('div', { class: 'row', style: `--c:${seriesColor(i)}` }, h('b', {}, fy(p[1])), ' ', h('span', { class: 'k' }, ser.name));
        }),
      ]);
    };
    hit.addEventListener('pointermove', (e) => {
      const rect = s.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      const xv = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
      let k = 0; for (let j = 1; j < xs.length; j++) if (Math.abs(xs[j] - xv) < Math.abs(xs[k] - xv)) k = j;
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

/** Sequential (one hue, light→dark) or diverging (blue↔gray↔red) color for t in [0,1] / [-1,1]. */
export function seqColor(t) {
  const k = Math.max(0, Math.min(SEQ.length - 1, Math.round(t * (SEQ.length - 1))));
  return SEQ[k];
}
export function divColor(t) {
  const mid = isDark() ? '#383835' : '#f0efec';
  const a = Math.min(1, Math.abs(t));
  const end = t < 0 ? DIV_NEG : DIV_POS;
  return `color-mix(in oklab, ${end} ${Math.round(a * 100)}%, ${mid})`;
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
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: `max-width:100%;height:auto`, role: 'img',
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
          const dark = o.scale === 'div' ? Math.abs(t) > 0.55 : t > 0.55;
          s.append(svg('text', { x: lab + cell * (c + 0.5), y: lab + cell * (r + 0.5) + 4, 'text-anchor': 'middle',
            'font-size': 10, fill: dark ? '#fff' : 'var(--ink)', 'pointer-events': 'none' }, fv(v)));
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
