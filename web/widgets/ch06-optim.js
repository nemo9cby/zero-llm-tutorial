/* 图：四种优化器在《深度学习入门》的碗 f(x, y) = (x-cx)²/20 + (y-cy)² 上的轨迹（6.1 节）。
   SGD、Momentum（《深度学习入门》的写法）、Adam、AdamW（解耦的权重衰减），可选“Adam + L2”。
   下方的小图：一个没有数据梯度、只有权重衰减的参数（验证 6.1-e）。
   纯函数都导出，供 tests/js/ch06-math.test.mjs 与页面上的验证单元对拍。 */
import { h } from './_lib.js';
import { lineChart, seriesColor } from './_plot.js';

export const grad = (p, c = [0, 0]) => [(p[0] - c[0]) / 10, 2 * (p[1] - c[1])];

export function runSGD(start, lr, steps, c) {
  let p = [...start];
  const path = [p];
  for (let t = 0; t < steps; t++) {
    const g = grad(p, c);
    p = [p[0] - lr * g[0], p[1] - lr * g[1]];
    path.push(p);
  }
  return { path };
}

/** 《深度学习入门》式 (6.3)(6.4)：v ← αv − η∇，W ← W + v */
export function runMomentum(start, lr, alpha, steps, c) {
  let p = [...start];
  let v = [0, 0];
  const path = [p];
  for (let t = 0; t < steps; t++) {
    const g = grad(p, c);
    v = [alpha * v[0] - lr * g[0], alpha * v[1] - lr * g[1]];
    p = [p[0] + v[0], p[1] + v[1]];
    path.push(p);
  }
  return { path };
}

/** Adam / AdamW / Adam + L2。wd 为衰减系数；coupled=true 时把 wd·p 加进梯度（L2），否则解耦（AdamW）。 */
export function runAdam(start, lr, steps, c, { wd = 0, coupled = false, b1 = 0.9, b2 = 0.999, eps = 1e-8 } = {}) {
  let p = [...start];
  let m = [0, 0];
  let v = [0, 0];
  const path = [p];
  const states = [null];
  for (let t = 1; t <= steps; t++) {
    let g = grad(p, c);
    if (coupled) g = [g[0] + wd * p[0], g[1] + wd * p[1]];
    m = [b1 * m[0] + (1 - b1) * g[0], b1 * m[1] + (1 - b1) * g[1]];
    v = [b2 * v[0] + (1 - b2) * g[0] * g[0], b2 * v[1] + (1 - b2) * g[1] * g[1]];
    const mh = m.map((x) => x / (1 - b1 ** t));
    const vh = v.map((x) => x / (1 - b2 ** t));
    const step = [mh[0] / (Math.sqrt(vh[0]) + eps), mh[1] / (Math.sqrt(vh[1]) + eps)];
    const decay = coupled ? [0, 0] : [lr * wd * p[0], lr * wd * p[1]];
    p = [p[0] - lr * step[0] - decay[0], p[1] - lr * step[1] - decay[1]];
    path.push(p);
    states.push({ g, mh, vh, step });
  }
  return { path, states };
}

/** 只有衰减、没有数据梯度的标量参数（验证 6.1-e）。 */
export function decayOnly(steps, { lr = 0.1, wd = 0.01, coupled = false, b1 = 0.9, b2 = 0.999, eps = 1e-8 } = {}) {
  let theta = 1;
  let m = 0;
  let v = 0;
  const out = [1];
  for (let t = 1; t <= steps; t++) {
    const g = coupled ? wd * theta : 0;
    m = b1 * m + (1 - b1) * g;
    v = b2 * v + (1 - b2) * g * g;
    const step = (m / (1 - b1 ** t)) / (Math.sqrt(v / (1 - b2 ** t)) + eps);
    theta = theta - lr * step - (coupled ? 0 : lr * wd * theta);
    out.push(theta);
  }
  return out;
}

const NS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs = {}) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};
const f4 = (x) => (Math.abs(x) < 5e-5 ? '0.0000' : x.toFixed(4));

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch06-optim .ctrl { display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: .4rem 1.2rem; margin: .3rem 0 .6rem; font-size: .88rem; }
.w-ch06-optim label { display: flex; flex-direction: column; gap: .15rem; color: var(--ink-2); }
.w-ch06-optim label span b { color: var(--ink); font-variant-numeric: tabular-nums; font-weight: 600; }
.w-ch06-optim input[type=range] { width: 100%; margin: 0; accent-color: var(--ai); }
.w-ch06-optim select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .15rem .3rem; }
.w-ch06-optim .bowl { width: 100%; height: auto; display: block; background: var(--paper-2); border-radius: 4px; touch-action: manipulation; }
.w-ch06-optim .legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: 12.5px; color: var(--ink-2); margin: .4rem 0; }
.w-ch06-optim .legend span::before { content: ''; display: inline-block; width: 14px; height: 3px; margin-right: 6px; vertical-align: middle; background: var(--c); }
.w-ch06-optim .legend label { flex-direction: row; align-items: center; gap: .3rem; }
.w-ch06-optim table { font: 12.5px/1.5 var(--mono); margin: .3rem 0; display: table; width: 100%; }
.w-ch06-optim td, .w-ch06-optim th { padding: .1rem .6rem .1rem 0; border-bottom: 1px solid var(--rule); white-space: nowrap; }
.w-ch06-optim .tablewrap { overflow-x: auto; }
.w-ch06-optim .note { font-size: .85rem; color: var(--ink-2); margin: .3rem 0 .8rem; }
.w-ch06-optim .row { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .2rem 0 .4rem; }
`));
}

export function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch06-optim');
  const start = config.start || [-7, 2];
  const STEPS = 60;                      // 碗底挪开后，60 步才看得清权重衰减把终点拉向哪里
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const tIn = h('input', { type: 'range', min: 0, max: STEPS, step: 1, value: STEPS, 'aria-label': '显示到第几步' });
  const lrIn = h('input', { type: 'range', min: 0.05, max: 1, step: 0.05, value: 0.3, 'aria-label': 'Adam 与 AdamW 的学习率' });
  const wdIn = h('input', { type: 'range', min: 0, max: 0.5, step: 0.05, value: 0.2, 'aria-label': 'AdamW 的权重衰减系数' });
  const bowlSel = h('select', { 'aria-label': '碗底的位置' },
    h('option', { value: '0,0' }, '原点 (0, 0)（《深度学习入门》）'),
    h('option', { value: '5,1' }, '挪到 (5, 1)'));
  const l2Box = h('input', { type: 'checkbox', 'aria-label': '同时显示 Adam + L2' });
  const playBtn = h('button', { type: 'button' }, '播放');
  const tLabel = h('b'), lrLabel = h('b'), wdLabel = h('b');

  const svg = svgEl('svg', { class: 'bowl', viewBox: '0 0 400 200', role: 'img', 'aria-label': '优化器在碗状函数上的轨迹' });
  const legend = h('div', { class: 'legend' });
  const tableWrap = h('div', { class: 'tablewrap' });
  const decayBox = h('div');

  root.append(
    h('div', { class: 'ctrl' },
      h('label', {}, h('span', {}, '显示到第 ', tLabel, ' 步'), tIn),
      h('label', {}, h('span', {}, 'Adam / AdamW 的学习率 ', lrLabel), lrIn),
      h('label', {}, h('span', {}, 'AdamW 的权重衰减 λ ', wdLabel), wdIn),
      h('label', {}, h('span', {}, '碗底的位置'), bowlSel)),
    h('div', { class: 'row' }, playBtn,
      h('label', { style: 'flex-direction:row;align-items:center;gap:.35rem;font-size:.85rem;color:var(--ink-2)' }, l2Box, '同时显示 Adam + L2（耦合的衰减）')),
    svg, legend, tableWrap,
    h('div', { class: 'w-label', style: 'margin-top:.8rem' }, '只有权重衰减、没有数据梯度的参数（lr = 0.1，λ = 0.01，验证 6.1-e）'),
    decayBox,
  );

  // 6.1-e 的小图：固定参数，与正文的验证单元一致
  const steps30 = [...Array(31).keys()];
  const dW = decayOnly(30), dL2 = decayOnly(30, { coupled: true });
  lineChart(decayBox, {
    series: [{ name: 'AdamW', points: steps30.map((t) => [t, dW[t]]) },
      { name: 'Adam + L2', points: steps30.map((t) => [t, dL2[t]]) }],
    xLabel: '步数', yLabel: 'θ', height: 200, refLines: [{ y: 0, label: '0' }],
  });

  const X = (x) => ((x + 10) / 20) * 400;
  const Y = (y) => ((5 - y) / 10) * 200;

  function render() {
    const t = +tIn.value, lr = +lrIn.value, wd = +wdIn.value;
    const c = bowlSel.value.split(',').map(Number);
    tLabel.textContent = t; lrLabel.textContent = lr.toFixed(2); wdLabel.textContent = wd.toFixed(2);
    const runs = [
      { name: 'SGD（lr 0.95）', ...runSGD(start, 0.95, STEPS, c) },
      { name: 'Momentum（lr 0.1，α 0.9）', ...runMomentum(start, 0.1, 0.9, STEPS, c) },
      { name: `Adam（lr ${lr.toFixed(2)}）`, ...runAdam(start, lr, STEPS, c) },
      { name: `AdamW（λ ${wd.toFixed(2)}）`, ...runAdam(start, lr, STEPS, c, { wd }) },
    ];
    if (l2Box.checked) runs.push({ name: `Adam + L2（λ ${wd.toFixed(2)}）`, dash: '5 4', ...runAdam(start, lr, STEPS, c, { wd, coupled: true }) });

    svg.replaceChildren();
    const clip = svgEl('clipPath', { id: 'w-ch06-optim-clip' });
    clip.append(svgEl('rect', { x: 0, y: 0, width: 400, height: 200 }));
    const defs = svgEl('defs'); defs.append(clip); svg.append(defs);
    const g = svgEl('g', { 'clip-path': 'url(#w-ch06-optim-clip)' });
    for (const L of [0.25, 1, 4, 9, 16, 25, 36, 49, 64]) {
      g.append(svgEl('ellipse', { cx: X(c[0]), cy: Y(c[1]), rx: Math.sqrt(20 * L) * 20, ry: Math.sqrt(L) * 20,
        fill: 'none', stroke: 'var(--rule)', 'stroke-width': 1 }));
    }
    g.append(svgEl('line', { x1: 0, x2: 400, y1: Y(0), y2: Y(0), stroke: 'var(--ink-3)', 'stroke-width': 0.5, 'stroke-dasharray': '2 3' }));
    g.append(svgEl('line', { x1: X(0), x2: X(0), y1: 0, y2: 200, stroke: 'var(--ink-3)', 'stroke-width': 0.5, 'stroke-dasharray': '2 3' }));
    g.append(svgEl('circle', { cx: X(c[0]), cy: Y(c[1]), r: 3, fill: 'var(--ink)' }));
    runs.forEach((r, i) => {
      const pts = r.path.slice(0, t + 1).map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
      const color = seriesColor(i);
      g.append(svgEl('polyline', { points: pts, fill: 'none', stroke: color, 'stroke-width': 1.8,
        'stroke-linejoin': 'round', ...(r.dash ? { 'stroke-dasharray': r.dash } : {}) }));
      const last = r.path[t];
      g.append(svgEl('circle', { cx: X(last[0]), cy: Y(last[1]), r: 3.5, fill: color, stroke: 'var(--paper)', 'stroke-width': 1 }));
    });
    g.append(svgEl('circle', { cx: X(start[0]), cy: Y(start[1]), r: 3, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.2 }));
    svg.append(g);

    legend.replaceChildren(...runs.map((r, i) => h('span', { style: `--c:${seriesColor(i)}` }, r.name)));
    const adam = runs[2];
    const st = adam.states[t];
    tableWrap.replaceChildren(h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, `第 ${t} 步`), h('th', {}, 'x'), h('th', {}, 'y'))),
      h('tbody', {}, runs.map((r) => h('tr', {}, h('td', {}, r.name.split('（')[0]),
        h('td', {}, f4(r.path[t][0])), h('td', {}, f4(r.path[t][1])))))),
    h('div', { class: 'note' }, st
      ? `Adam 第 ${t} 步：m̂ = (${f4(st.mh[0])}, ${f4(st.mh[1])})，√v̂ = (${f4(Math.sqrt(st.vh[0]))}, ${f4(Math.sqrt(st.vh[1]))})，`
        + `更新量 / lr = (${f4(st.step[0])}, ${f4(st.step[1])})`
      : '第 0 步：所有优化器都在起点 (−7, 2)。把滑块往右拖，看看第 1 步时 Adam 的“更新量 / lr”为什么两个方向都约等于 ±1。'));
  }

  let timer = null;
  playBtn.addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; playBtn.textContent = '播放'; return; }
    if (reduce) { tIn.value = STEPS; render(); return; }
    tIn.value = 0; render();
    playBtn.textContent = '暂停';
    timer = setInterval(() => {
      if (+tIn.value >= STEPS) { clearInterval(timer); timer = null; playBtn.textContent = '播放'; return; }
      tIn.value = +tIn.value + 1; render();
    }, 160);
  });
  for (const el of [tIn, lrIn, wdIn, bowlSel, l2Box]) el.addEventListener('input', render);
  bowlSel.addEventListener('change', render);
  l2Box.addEventListener('change', render);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  render();
}
