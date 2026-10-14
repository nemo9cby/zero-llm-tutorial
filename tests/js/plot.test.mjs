// web/widgets/_plot.js: theme helpers, label colours on the heat-map fills, ticks and the lineChart options.
//   node --test tests/js/plot.test.mjs
// The charts are rendered into a small fake DOM (no dependency): enough of document, Node, MutationObserver,
// ResizeObserver and matchMedia for _plot.js and _lib.js.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  isDark, onThemeChange, seriesColor, seriesVars, seqColor, seqIsDark, divColor, textColorOn, contrastRatio, mixOklab,
  niceTicks, autoTicks, logTicks, lineChart, heatmap,
} from '../../web/widgets/_plot.js';
import { recordedPoints } from '../../web/widgets/runcurve.js';

// Run before the fake DOM below is installed (test bodies run later).
const noDom = (() => {
  const off = onThemeChange(() => { throw new Error('never called'); });
  const r = { hadDocument: typeof document !== 'undefined', dark: isDark(), off: typeof off };
  off();
  return r;
})();
test('without a DOM (Node), the theme helpers are inert instead of throwing', () => {
  assert.deepEqual(noDom, { hadDocument: false, dark: false, off: 'function' });
});

// ---------------------------------------------------------------- a fake DOM
class FakeNode {
  constructor() { this.childNodes = []; this.parentNode = null; }
  get children() { return this.childNodes.filter((n) => n instanceof FakeElement); }
  append(...nodes) {
    for (const n0 of nodes) {
      const n = typeof n0 === 'string' ? new FakeText(n0) : n0;
      if (n.parentNode) n.parentNode.childNodes.splice(n.parentNode.childNodes.indexOf(n), 1);
      n.parentNode = this;
      this.childNodes.push(n);
    }
  }
  replaceChildren(...nodes) {
    for (const c of this.childNodes) c.parentNode = null;
    this.childNodes = [];
    this.append(...nodes);
  }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(''); }
  set textContent(v) { this.replaceChildren(String(v)); }
}
class FakeText extends FakeNode {
  constructor(t) { super(); this.data = t; }
  get textContent() { return this.data; }
}
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag; this.attrs = new Map(); this.dataset = {}; this.listeners = {}; this.hidden = false;
    this.style = { setProperty(k, v) { this[k] = String(v); }, getPropertyValue(k) { return this[k] ?? ''; } };
    this.width = 0;
  }
  setAttribute(k, v) { this.attrs.set(k, String(v)); }
  getAttribute(k) { return this.attrs.has(k) ? this.attrs.get(k) : null; }
  removeAttribute(k) { this.attrs.delete(k); }
  get className() { return this.attrs.get('class') || ''; }
  set className(v) { this.attrs.set('class', v); }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  get clientWidth() { return this.width; }
  get offsetWidth() { return 0; }
  get offsetHeight() { return 0; }
  getBoundingClientRect() { return { left: 0, top: 0, right: this.width, bottom: 0, width: this.width, height: 0 }; }
}

const observers = [];
const mediaListeners = [];
let osDark = false;
globalThis.Node = FakeNode;
globalThis.document = {
  documentElement: new FakeElement('html'),
  head: new FakeElement('head'),
  createElement: (tag) => new FakeElement(tag),
  createElementNS: (ns, tag) => new FakeElement(tag),
  createTextNode: (t) => new FakeText(t),
};
globalThis.MutationObserver = class { constructor(cb) { this.cb = cb; observers.push(this); } observe() {} };
globalThis.ResizeObserver = class { constructor(cb) { this.cb = cb; } observe() {} };
globalThis.matchMedia = () => ({ get matches() { return osDark; }, addEventListener: (type, fn) => mediaListeners.push(fn) });

const root = document.documentElement;
function setTheme(t) {                       // what runtime.js's toggle does: a data-theme attribute change
  if (t == null) delete root.dataset.theme; else root.dataset.theme = t;
  for (const o of observers) o.cb([]);
}
function setOs(dark) { osDark = dark; for (const fn of mediaListeners) fn({ matches: dark }); }
const all = (node, pred) => {
  const out = [];
  const walk = (n) => { if (n instanceof FakeElement && pred(n)) out.push(n); n.childNodes.forEach(walk); };
  walk(node);
  return out;
};
const tag = (node, name) => all(node, (e) => e.tagName === name);
const mount = (width) => { const el = new FakeElement('div'); el.width = width; return el; };

// ---------------------------------------------------------------- theme
test('isDark: an explicit data-theme wins over the OS setting', () => {
  setOs(false); setTheme(null);
  assert.equal(isDark(), false);
  setOs(true);
  assert.equal(isDark(), true);
  setTheme('light');
  assert.equal(isDark(), false);
  setOs(false); setTheme('dark');
  assert.equal(isDark(), true);
  setTheme(null);
});

test('onThemeChange: one call per effective change, from the toggle or from the OS; unsubscribe works', () => {
  setOs(false); setTheme(null);
  const seen = [];
  const off = onThemeChange((dark) => seen.push(dark));
  setTheme('light');                 // still light: no call
  setTheme('dark');                  // light -> dark
  setOs(true);                       // the page says dark already: no call
  setTheme(null);                    // follows the OS: still dark, no call
  setOs(false);                      // dark -> light (OS-level change, no data-theme)
  assert.deepEqual(seen, [true, false]);
  off();
  setTheme('dark');
  assert.deepEqual(seen, [true, false]);
  setTheme(null);
});

test('onThemeChange: a listener that throws does not stop the others', () => {
  const errors = [];
  const orig = console.error;
  console.error = (e) => errors.push(e);
  let ok = 0;
  const offA = onThemeChange(() => { throw new Error('boom'); });
  const offB = onThemeChange(() => { ok += 1; });
  try {
    setTheme('dark');
    setTheme(null);
  } finally {
    console.error = orig;
    offA(); offB();
  }
  assert.equal(ok, 2);
  assert.equal(errors.length, 2);
});

test('seriesVars: --series-0 … --series-7 follow the effective theme, from the toggle and from the OS', () => {
  setOs(false); setTheme(null);
  const el = mount(100);
  const vars = () => Array.from({ length: 8 }, (_, i) => el.style.getPropertyValue(`--series-${i}`));
  const palette = () => Array.from({ length: 8 }, (_, i) => seriesColor(i));
  const off = seriesVars(el);
  const light = palette();
  assert.deepEqual(vars(), light);
  setTheme('dark');                  // the page's toggle
  const dark = palette();
  assert.notDeepEqual(dark, light);
  assert.deepEqual(vars(), dark);
  setTheme(null);                    // follows the OS again: light
  assert.deepEqual(vars(), light);
  setOs(true);                       // an OS-level switch, no data-theme
  assert.deepEqual(vars(), dark);
  off();                             // unsubscribed: no longer updated
  setOs(false);
  assert.deepEqual(vars(), dark);
  assert.equal(typeof off, 'function');
});

// ---------------------------------------------------------------- colour
test('WCAG contrast: known values', () => {
  assert.ok(Math.abs(contrastRatio('#ffffff', '#000000') - 21) < 1e-9);
  assert.ok(Math.abs(contrastRatio('#777777', '#ffffff') - 4.48) < 0.01);
  assert.equal(contrastRatio('#2a78d6', '#ffffff'), contrastRatio('#ffffff', '#2a78d6'));
});

test('textColorOn picks the label with the higher contrast', () => {
  assert.equal(textColorOn('#ffffff'), '#1b2233');
  assert.equal(textColorOn('#000000'), '#ffffff');
  assert.equal(textColorOn('#f0efec'), '#1b2233');      // light diverging midpoint
  assert.equal(textColorOn('#383835'), '#ffffff');      // dark diverging midpoint (ch05 R7)
  for (const fill of ['#2a78d6', '#3987e5', '#e34948', '#cde2fb', '#0d366b']) {
    const pick = textColorOn(fill), other = pick === '#ffffff' ? '#1b2233' : '#ffffff';
    assert.ok(contrastRatio(pick, fill) >= contrastRatio(other, fill), fill);
  }
});

test('seqIsDark is exact: true exactly when white beats the ink on the fill seqColor returns', () => {
  for (const theme of ['light', 'dark']) {
    setTheme(theme);
    for (let i = 0; i <= 200; i++) {
      const t = i / 200;
      assert.equal(seqIsDark(t), textColorOn(seqColor(t)) === '#ffffff', `${theme} t=${t}`);
    }
  }
  setTheme(null);
});

// Chrome 14x: a 1×1 canvas filled with `color-mix(in oklab, <end> p%, <mid>)`, read back as sRGB bytes.
const CHROME = {
  '#2a78d6 37% #f0efec': '#a9c4e7', '#2a78d6 50% #f0efec': '#90b5e4', '#2a78d6 90% #f0efec': '#4285d9',
  '#e34948 10% #f0efec': '#f2e0db', '#e34948 55% #f0efec': '#f19a91', '#e34948 75% #f0efec': '#ec7971',
  '#2a78d6 10% #383835': '#393f44', '#2a78d6 50% #383835': '#375983', '#2a78d6 75% #383835': '#3369ac',
  '#e34948 25% #383835': '#62403b', '#e34948 63% #383835': '#a24843', '#e34948 90% #383835': '#d14a47',
};
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

test('mixOklab matches the browser\'s color-mix(in oklab) to within one step per channel', () => {
  for (const [k, want] of Object.entries(CHROME)) {
    const [end, p, mid] = k.split(' ');
    const got = mixOklab(end, mid, parseFloat(p) / 100);
    const d = rgb(got).map((v, i) => Math.abs(v - rgb(want)[i]));
    assert.ok(Math.max(...d) <= 1, `${k}: ${got} vs ${want}`);
  }
});

test('divColor: exact sRGB fills; the midpoint is the theme\'s grey and the ends are the two hues', () => {
  setTheme('light');
  assert.equal(divColor(0), '#f0efec');
  assert.equal(divColor(-1), '#2a78d6');
  assert.equal(divColor(1), '#e34948');
  assert.equal(divColor(-3), '#2a78d6');                 // clamped
  setTheme('dark');
  assert.equal(divColor(0), '#383835');
  assert.equal(divColor(-1), '#2a78d6');
  assert.equal(divColor(1), '#e24848');                  // the dark theme's red end (next test)
  assert.equal(divColor(0.5), mixOklab('#e24848', '#383835', 0.5));
  assert.match(divColor(-0.37), /^#[0-9a-f]{6}$/);
  setTheme(null);
});

test('dark-theme red end: darkened just enough that white beats the ink, as on every other dark red step', () => {
  const W = '#ffffff', INK = '#1b2233';
  // On #e34948 the ink wins by a hair, 4.015:1 against 3.953:1, so in the dark theme the strongest red cells alone
  // carried ink labels among white-labelled neighbours (ch05 figure 5-2).
  assert.ok(Math.abs(contrastRatio(INK, '#e34948') - 4.015) < 5e-4 && Math.abs(contrastRatio(W, '#e34948') - 3.953) < 5e-4);
  // #e24848 is #e34948 with oklab lightness 0.0015 lower (one sRGB step on red and green): 3.997:1 against 3.971:1.
  assert.ok(Math.abs(contrastRatio(W, '#e24848') - 3.997) < 5e-4 && Math.abs(contrastRatio(INK, '#e24848') - 3.971) < 5e-4);
  // Every whole-percent step of every half-ramp, in both themes: the label is still the exact-contrast pick, and it
  // changes at most once along a half-ramp. The dark red half is white throughout; light red stays ink throughout.
  const runs = {};
  for (const theme of ['light', 'dark']) {
    setTheme(theme);
    for (const sign of [-1, 1]) {
      const labels = Array.from({ length: 101 }, (_, p) => textColorOn(divColor((sign * p) / 100)) === W ? 'W' : 'i').join('');
      runs[`${theme} ${sign < 0 ? 'blue' : 'red'}`] = labels;
      assert.ok(labels.replace(/(.)\1*/g, '$1').length <= 2, `${theme} ${sign}: ${labels}`);
    }
  }
  setTheme(null);
  assert.equal(runs['dark red'], 'W'.repeat(101));
  assert.equal(runs['light red'], 'i'.repeat(101));
  assert.equal(runs['dark blue'], 'W'.repeat(101));
  assert.match(runs['light blue'], /^i+W+$/);
});

test('heatmap: every value label uses the higher-contrast colour on its actual fill, in both themes', () => {
  for (const theme of ['light', 'dark']) {
    setTheme(theme);
    for (const scale of ['div', 'seq']) {
      const el = mount(420);
      heatmap(el, { values: [[-1, -0.6, -0.2, 0, 0.2, 0.6, 1]], scale, domain: [-1, 1], size: 34, table: false });
      const svg = tag(el, 'svg')[0];
      const rects = tag(svg, 'rect').filter((r) => r.getAttribute('class') === 'cell');
      const labels = tag(svg, 'text').filter((t) => t.getAttribute('font-size') === '10');
      assert.equal(labels.length, 7);
      labels.forEach((t, i) => assert.equal(t.getAttribute('fill'), textColorOn(rects[i].getAttribute('fill')), `${theme} ${scale} ${i}`));
      if (scale === 'div') assert.equal(labels[3].getAttribute('fill'), theme === 'dark' ? '#ffffff' : '#1b2233');
      // the red end (value 1) is labelled like its neighbour (0.6): white in the dark theme, ink in the light one
      if (scale === 'div') assert.equal(labels[6].getAttribute('fill'), labels[5].getAttribute('fill'), `${theme}: red end`);
    }
  }
  setTheme(null);
});

// ---------------------------------------------------------------- ticks
test('niceTicks is unchanged', () => {
  assert.deepEqual(niceTicks(0, 10, 5), [0, 2, 4, 6, 8, 10]);
  assert.deepEqual(niceTicks(-3, 3, 2), [0]);
  assert.deepEqual(niceTicks(1, 800, 2), [500]);
  assert.deepEqual(niceTicks(0.5, 0.5), [-0.5, 0, 0.5, 1, 1.5]);
  assert.deepEqual(niceTicks(0, 20000, 3), [0, 10000, 20000]);
});

test('autoTicks gives at least three ticks where niceTicks gave one (ch05 R13)', () => {
  for (const [lo, hi] of [[-3, 3], [1, 800], [0, 20000], [0, 1], [1, 9], [0.001, 0.0013]]) {
    const t = autoTicks(lo, hi, 2);
    assert.ok(t.length >= 3, `${lo}..${hi}: ${t}`);
    assert.ok(t.every((v) => v >= lo - 1e-12 && v <= hi + 1e-12), `${lo}..${hi}: ${t}`);
  }
  assert.deepEqual(autoTicks(0, 10, 5), niceTicks(0, 10, 5));   // enough already: same as before
});

test('logTicks: powers of ten when they are enough, 1-2-5 or finer when the span is short', () => {
  assert.deepEqual(logTicks(0.001, 1), [0.001, 0.01, 0.1, 1]);
  assert.deepEqual(logTicks(1, 10), [1, 2, 5, 10]);                 // ch06 R11: was [1, 10]
  assert.deepEqual(logTicks(2.2, 9), [3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(logTicks(0.6, 55), [1, 2, 5, 10, 20, 50]);
  const short = logTicks(1.3, 2.6);                                  // no 1-2-5 value inside: linear ticks
  assert.ok(short.length >= 3 && short.every((v) => v >= 1.3 && v <= 2.6), String(short));
});

// ---------------------------------------------------------------- lineChart
const xLabels = (svg) => tag(svg, 'text').filter((t) => t.getAttribute('text-anchor') === 'middle' && !t.getAttribute('class'));
const yLabels = (svg) => tag(svg, 'text').filter((t) => t.getAttribute('text-anchor') === 'end' && !t.getAttribute('class'));
const viewW = (svg) => +svg.getAttribute('viewBox').split(' ')[2];
const line = (n, f) => Array.from({ length: n }, (_, i) => [i, f(i)]);

test('lineChart: at least three x tick labels at phone width (ch05 R13)', () => {
  for (const [x0, x1] of [[-3, 3], [1, 800]]) {
    const el = mount(300);                   // a chart inside a figure at 375 px: niceTicks was asked for 2 ticks
    lineChart(el, { series: [{ name: 'a', points: [[x0, 0], [x1, 1]] }], table: false });
    assert.ok(xLabels(tag(el, 'svg')[0]).length >= 3, `${x0}..${x1}`);
  }
});

test('lineChart: explicit xTicks are still used as given', () => {
  const el = mount(343);
  lineChart(el, { series: [{ name: 'a', points: [[-3, 0], [3, 1]] }], xTicks: [-3, 0, 3], table: false });
  assert.deepEqual(xLabels(tag(el, 'svg')[0]).map((t) => t.textContent), ['-3', '0', '3']);
});

test('lineChart: xMin / xMax fix the x domain (trainer-R12) and update() keeps them', () => {
  const el = mount(600);
  const chart = lineChart(el, { series: [{ name: 'a', points: line(101, (i) => i) }], xMin: 0, xMax: 1000, table: false });
  let svg = tag(el, 'svg')[0];
  assert.ok(xLabels(svg).map((t) => t.textContent).includes('1,000'));
  const lastX = (s) => +tag(s, 'path')[0].getAttribute('d').split('L').at(-1).split(',')[0];
  const m = { l: 52, r: 16 };
  assert.ok(Math.abs(lastX(svg) - (m.l + 0.1 * (600 - m.l - m.r))) < 0.2);
  chart.update({ series: [{ name: 'a', points: line(201, (i) => i) }] });
  svg = tag(el, 'svg')[0];
  assert.ok(Math.abs(lastX(svg) - (m.l + 0.2 * (600 - m.l - m.r))) < 0.2);
});

test('lineChart: marginRight overrides the series-count default', () => {
  const two = [{ name: 'a', points: [[0, 0], [1, 1]] }, { name: 'b', points: [[0, 1], [1, 0]] }];
  const gridRight = (el) => +tag(tag(el, 'svg')[0], 'line')[0].getAttribute('x2');
  const a = mount(500), b = mount(500), c = mount(500);
  lineChart(a, { series: two, table: false });
  lineChart(b, { series: two, marginRight: 16, table: false });
  lineChart(c, { series: two.slice(0, 1), marginRight: 84, table: false });
  assert.equal(gridRight(a), 500 - 84);
  assert.equal(gridRight(b), 500 - 16);
  assert.equal(gridRight(c), 500 - 84);
});

test('lineChart: x tick labels at the plot edges stay inside the SVG (gpu figure 3 at 375 px)', () => {
  const el = mount(343);
  lineChart(el, { series: [{ name: 'loss', points: line(201, (i) => 7 - i / 50).map(([i, y]) => [i * 100, y]) }], table: false });
  const svg = tag(el, 'svg')[0];
  const W = viewW(svg);
  const labels = xLabels(svg);
  assert.equal(labels.at(-1).textContent, '20,000');
  for (const t of labels) {
    const half = t.textContent.length * 6.6 / 2, x = +t.getAttribute('x');
    assert.ok(x - half >= -0.01 && x + half <= W + 0.01, `${t.textContent} at ${x} of ${W}`);
  }
});

test('lineChart: a short log axis is labelled 1, 2, 5, 10 (ch06 R11)', () => {
  const el = mount(600);
  lineChart(el, { series: [{ name: 'loss', points: line(50, (i) => 9 - i * 0.15) }], logY: true, yMin: 1, yMax: 10, table: false });
  assert.deepEqual(yLabels(tag(el, 'svg')[0]).map((t) => t.textContent), ['1', '2', '5', '10']);
});

test('lineChart: markers: true marks every point; an array marks only the given points', () => {
  const el = mount(600);
  lineChart(el, {
    series: [{ name: 'a', points: line(5, (i) => i), markers: true },
      { name: 'b', points: line(400, (i) => i % 7), markers: [[10, 3], [200, 4], [390, 5]] }],
    table: false,
  });
  const circles = tag(tag(el, 'svg')[0], 'circle');
  assert.equal(circles.length, 5 + 3);
  assert.ok(circles.slice(0, 5).every((c) => c.getAttribute('r') === '4'));
});

test('lineChart tooltip: on a log x axis, each series lists its point nearest as drawn (ch04 figure 4-2)', () => {
  // b has points at 10 and 10,000 only. At x = 2,000, 10 is nearer in raw units (1,990 against 8,000), but on a log
  // axis 10,000 is (0.7 decades against 2.3): that is the point the reader sees next to the crosshair.
  const series = [{ name: 'a', points: [[10, 1], [2000, 2], [10000, 3]] }, { name: 'b', points: [[10, 5], [10000, 7]] }];
  const rowsAt2000 = (logX) => {
    const el = mount(600);
    lineChart(el, { series, logX, table: false });
    const hit = all(el, (e) => e.getAttribute('aria-label') === '用左右方向键查看数据点')[0];
    for (const fn of hit.listeners.keydown) fn({ key: 'ArrowRight', preventDefault() {} });      // x = 10 -> 2,000
    const tip = all(el, (e) => e.className === 'viz-tip')[0];
    return all(tip, (e) => e.className === 'row').map((r) => r.textContent);
  };
  assert.deepEqual(rowsAt2000(true), ['2 a', '7 b']);
  assert.deepEqual(rowsAt2000(false), ['2 a', '5 b']);         // a linear axis: raw distance, as before
});

test('lineChart: reference-line labels keep their end-label class (trainer-replay liftLabels relies on it)', () => {
  const el = mount(600);
  lineChart(el, { series: [{ name: 'a', points: line(10, (i) => i) }], refLines: [{ y: 3, label: 'ln 3' }], table: false });
  assert.deepEqual(tag(el, 'text').filter((t) => t.getAttribute('class') === 'end-label').map((t) => t.textContent), ['ln 3']);
});

test('lineChart re-renders with the dark palette when the theme changes', () => {
  setTheme(null); setOs(false);
  const el = mount(600);
  lineChart(el, { series: [{ name: 'a', points: line(10, (i) => i) }], table: false });
  const stroke = () => tag(tag(el, 'svg')[0], 'path')[0].getAttribute('stroke');
  assert.equal(stroke(), seriesColor(0));
  const light = stroke();
  setTheme('dark');
  assert.notEqual(stroke(), light);
  assert.equal(stroke(), seriesColor(0));
  setTheme(null);
  assert.equal(stroke(), light);
});

// ---------------------------------------------------------------- runcurve.js: dots where a metric was measured
const metrics = (run, step) => fs.readFileSync(new URL(`../../web/data/runs/${run}/${step}.metrics.jsonl`, import.meta.url), 'utf8')
  .split('\n').filter(Boolean).map((l) => JSON.parse(l));

test('runcurve: the StoryBot validation loss is marked where it was measured, not on every logged row (ch06 R11)', () => {
  const rows = metrics('storybot-h100', '05_pretrain');
  const val = recordedPoints(rows, 'val_loss');
  assert.equal(val.length, 81);                                  // step 0 and every 500 steps up to 40,000
  assert.deepEqual(val.slice(0, 3).map(([it]) => it), [0, 505, 1002]);   // the first logged row after each evaluation
  assert.ok(val.every(([, v], i) => i === 0 || v !== val[i - 1][1]));
  assert.equal(recordedPoints(rows, 'loss'), null);              // measured on every row: no dots
});

test('runcurve: a metric logged on only some rows is marked on those rows; constant or dense ones are not', () => {
  const rows = Array.from({ length: 100 }, (_, it) => ({ it, loss: 5 - it / 50, ...(it % 10 === 0 ? { acc: it / 100 } : {}) }));
  assert.deepEqual(recordedPoints(rows, 'acc').map(([it]) => it), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
  assert.equal(recordedPoints(rows, 'loss'), null);
  assert.equal(recordedPoints(rows, 'missing'), null);
  const many = Array.from({ length: 4000 }, (_, it) => ({ it, val: Math.floor(it / 10) }));   // 400 measurements
  assert.equal(recordedPoints(many, 'val'), null);
});
