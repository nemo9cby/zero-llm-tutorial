/* 图：二维 RoPE 实验台。q 位于位置 m，转 m·θ；k 位于位置 n，转 n·θ。
   “q、k 同时 +1”时两支箭头一起转，夹角与点积都不变；只挪动其中一个，点积才会变。 */
import { h } from './_lib.js';
import { seriesColor } from './_plot.js';
import { injectStyle, slider, button, reducedMotion, svg } from './ch05-common.js';

const DEG = Math.PI / 180;

export function rotate([x, y], a) {
  return [Math.cos(a) * x - Math.sin(a) * y, Math.sin(a) * x + Math.cos(a) * y];
}

/** q·k after rotating q = (1,0) by m·θ and k = (1,0) by n·θ (θ in degrees). Equals cos((n−m)θ). */
export function ropeDot2D(m, n, thetaDeg, q = [1, 0], k = [1, 0]) {
  const a = rotate(q, m * thetaDeg * DEG), b = rotate(k, n * thetaDeg * DEG);
  return a[0] * b[0] + a[1] * b[1];
}

const norm360 = (d) => ((d % 360) + 360) % 360;

/** q·k to 3 decimals; a rounding residue such as −1.8e-16 is shown as 0.000, never as −0.000. */
export const fmtDot = (v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);

/** The same rotation as an angle in (−180°, 180°], e.g. 720° → 0°, 270° → −90°. */
export function equivalentAngle(deg) {
  const a = ((deg % 360) + 540) % 360 - 180;
  return a === -180 ? 180 : a;
}

export function mount(root, config = {}) {
  injectStyle();
  const state = { m: config.m ?? 1, n: config.n ?? 3, theta: config.theta ?? 30 };
  const shown = { qa: 0, ka: 0 };                       // angles currently drawn (degrees), for tweening
  const box = h('div', { class: 'w-ch05 w-ch05-rope' });
  root.append(box);
  box.append(h('style', {}, `
.w-ch05-rope .stage { display: flex; flex-wrap: wrap; gap: .8rem 1.4rem; align-items: center; }
.w-ch05-rope svg.plane { width: 240px; max-width: 100%; height: auto; flex: 0 0 auto; }
.w-ch05-rope .side { flex: 1 1 15rem; min-width: 0; }
.w-ch05-rope .hist { font: 12.5px/1.6 var(--mono); color: var(--ink-2); margin-top: .4rem; min-height: 1.6em; overflow-wrap: anywhere; }
`));

  const S = 240, C = S / 2, R = 92;
  const plane = svg('svg', { class: 'plane', viewBox: `0 0 ${S} ${S}`, role: 'img', 'aria-label': 'q 与 k 两支箭头在平面上的位置' });
  const defs = svg('defs');
  const marker = (id, color) => svg('marker', { id, viewBox: '0 0 10 10', refX: 8.5, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
    svg('path', { d: 'M0,0 L10,5 L0,10 z', fill: color }));
  const uid = Math.random().toString(36).slice(2, 8);
  const qColor = seriesColor(0), kColor = seriesColor(1);
  defs.append(marker(`aq${uid}`, qColor), marker(`ak${uid}`, kColor));
  plane.append(defs,
    svg('circle', { cx: C, cy: C, r: R, fill: 'none', stroke: 'var(--rule)', 'stroke-width': 1 }),
    svg('line', { x1: C - R - 12, x2: C + R + 12, y1: C, y2: C, stroke: 'var(--rule)' }),
    svg('line', { y1: C - R - 12, y2: C + R + 12, x1: C, x2: C, stroke: 'var(--rule)' }));
  const arc = svg('path', { fill: 'none', stroke: 'var(--ink-3)', 'stroke-width': 1.2, 'stroke-dasharray': '3 3' });
  const arcLabel = svg('text', { 'font-size': 12, fill: 'var(--ink-2)', 'text-anchor': 'middle' });
  const kLine = svg('line', { x1: C, y1: C, stroke: kColor, 'stroke-width': 3, 'marker-end': `url(#ak${uid})` });
  const qLine = svg('line', { x1: C, y1: C, stroke: qColor, 'stroke-width': 3, 'marker-end': `url(#aq${uid})` });
  const qLabel = svg('text', { 'font-size': 13, fill: qColor, 'font-weight': 700, 'text-anchor': 'middle' }, 'q');
  const kLabel = svg('text', { 'font-size': 13, fill: kColor, 'font-weight': 700, 'text-anchor': 'middle' }, 'k');
  plane.append(arc, arcLabel, kLine, qLine, qLabel, kLabel);

  const readout = h('div', { class: 'readout', 'aria-live': 'polite' });
  const hist = h('div', { class: 'hist' });
  const history = [];
  const sm = slider({ label: 'q 的位置 m', min: 0, max: 12, value: state.m, onInput: (v) => { state.m = v; update(); } });
  const sn = slider({ label: 'k 的位置 n', min: 0, max: 12, value: state.n, onInput: (v) => { state.n = v; update(); } });
  const st = slider({ label: '每步转 θ', min: 5, max: 90, step: 5, value: state.theta, format: (v) => `${v}°`,
    onInput: (v) => { state.theta = v; history.length = 0; update(); } });
  const both = button('q、k 同时 +1', () => { if (state.m < 12 && state.n < 12) { state.m += 1; state.n += 1; sync(); } },
    '把 q 和 k 同时往后挪一个位置');
  const kOnly = button('只有 k +1', () => { if (state.n < 12) { state.n += 1; sync(); } }, '只把 k 往后挪一个位置');
  const reset = button('复原', () => {
    Object.assign(state, { m: config.m ?? 1, n: config.n ?? 3, theta: config.theta ?? 30 });
    history.length = 0; st.set(state.theta); sync();
  });
  const side = h('div', { class: 'side' },
    h('div', { class: 'ctl' }, sm.el, sn.el, st.el),
    h('div', { class: 'btns' }, both, kOnly, reset),
    readout, hist);
  box.append(h('div', { class: 'stage' }, plane, side));

  function sync() { sm.set(state.m); sn.set(state.n); update(); }

  function draw(qa, ka) {
    const tip = (a, r) => [C + r * Math.cos(a * DEG), C - r * Math.sin(a * DEG)];
    const [qx, qy] = tip(qa, R - 4), [kx, ky] = tip(ka, R - 4);
    qLine.setAttribute('x2', qx); qLine.setAttribute('y2', qy);
    kLine.setAttribute('x2', kx); kLine.setAttribute('y2', ky);
    const [qlx, qly] = tip(qa, R + 14), [klx, kly] = tip(ka, R + 14);
    const same = Math.abs(norm360(qa) - norm360(ka)) < 8 || Math.abs(norm360(qa) - norm360(ka)) > 352;
    qLabel.setAttribute('x', qlx); qLabel.setAttribute('y', qly + 4);
    kLabel.setAttribute('x', klx + (same ? 12 : 0)); kLabel.setAttribute('y', kly + 4 + (same ? 10 : 0));
    // arc from q to k (counter-clockwise by the signed difference)
    const diff = ka - qa, r = 34;
    const [ax, ay] = tip(qa, r), [bx, by] = tip(ka, r);
    const large = Math.abs(diff) % 360 > 180 ? 1 : 0;
    const sweep = diff > 0 ? 0 : 1;
    arc.setAttribute('d', Math.abs(diff) < 0.5 ? '' : `M${ax},${ay} A${r},${r} 0 ${large} ${sweep} ${bx},${by}`);
    const mid = tip(qa + diff / 2, r + 14);
    arcLabel.setAttribute('x', mid[0]); arcLabel.setAttribute('y', mid[1] + 4);
    arcLabel.textContent = Math.abs(diff) < 0.5 ? '' : `${Math.round(diff)}°`;
  }

  let raf = 0;
  function update() {
    const qa = state.m * state.theta, ka = state.n * state.theta;
    const dot = fmtDot(ropeDot2D(state.m, state.n, state.theta));
    const diff = (state.n - state.m) * state.theta;
    const wrapped = equivalentAngle(diff);
    readout.replaceChildren(
      h('div', {}, 'q 转了 m·θ = ', h('b', {}, `${qa}°`), '，k 转了 n·θ = ', h('b', {}, `${ka}°`)),
      h('div', {}, '转角之差 (n − m)·θ = ', h('b', {}, `${diff}°`),
        wrapped === diff ? '' : `（相当于 ${wrapped}°）`,
        '，q·k = cos(转角之差) = ', h('b', {}, dot)),
    );
    history.push(`(${state.m},${state.n})→${dot}`);
    if (history.length > 6) history.shift();
    hist.textContent = `最近几次 (m,n)→q·k：${history.join('  ')}`;
    cancelAnimationFrame(raf);
    const from = { ...shown }, t0 = performance.now(), dur = reducedMotion() ? 0 : 260;
    const step = (now) => {
      const s = dur ? Math.min(1, (now - t0) / dur) : 1, e = 1 - (1 - s) ** 3;
      shown.qa = from.qa + (qa - from.qa) * e; shown.ka = from.ka + (ka - from.ka) * e;
      draw(shown.qa, shown.ka);
      if (s < 1) raf = requestAnimationFrame(step);
    };
    if (dur) raf = requestAnimationFrame(step); else step(t0);
  }
  update();
}
