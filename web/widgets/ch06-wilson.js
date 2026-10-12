/* 图：胜率的 Wilson 区间与符号检验（6.6.3 节、6.8.3 节）。与验证 6.6-b 使用同一组公式。 */
import { h } from './_lib.js';

export const Z = { '90%': 1.6448536269514722, '95%': 1.959963984540054, '99%': 2.5758293035489004 };

/** 式 (6.10)：Wilson 区间。 */
export function wilson(k, n, z = Z['95%']) {
  if (n === 0) return [0, 1];
  const p = k / n;
  const den = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / den;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / den;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

/** 常见的 p ± z·sqrt(p(1-p)/n)（Wald 区间），用来对比；不截断，好让越界看得见。 */
export function wald(k, n, z = Z['95%']) {
  if (n === 0) return [0, 1];
  const p = k / n;
  const half = z * Math.sqrt((p * (1 - p)) / n);
  return [p - half, p + half];
}

/** 双侧符号检验（精确二项检验，p = 0.5），平局不计。在对数空间里求和，n 很大也不会下溢。 */
export function signTestP(wins, losses) {
  const n = wins + losses;
  if (n === 0) return 1;
  const k = Math.max(wins, losses);
  let logC = 0;
  const logs = [];
  for (let i = 0; i <= n; i++) {
    if (i > 0) logC += Math.log(n - i + 1) - Math.log(i);
    if (i >= k) logs.push(logC - n * Math.LN2);
  }
  const mx = Math.max(...logs);
  const tail = Math.exp(mx) * logs.reduce((s, l) => s + Math.exp(l - mx), 0);
  return Math.min(1, 2 * tail);
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch06-ci .ctrl { display: flex; flex-wrap: wrap; gap: .5rem 1rem; align-items: end; font-size: .88rem; color: var(--ink-2); }
.w-ch06-ci label { display: flex; flex-direction: column; gap: .15rem; }
.w-ch06-ci input[type=number] { width: 5.5em; font: 15px var(--mono); color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .2rem .4rem; }
.w-ch06-ci select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .15rem .3rem; }
.w-ch06-ci .presets { display: flex; flex-wrap: wrap; gap: .3rem; margin: .5rem 0; }
.w-ch06-ci .presets button { font-size: .8rem; padding: .1rem .5rem; }
.w-ch06-ci svg { width: 100%; height: auto; display: block; margin: .4rem 0; }
.w-ch06-ci .read { font-size: .9rem; color: var(--ink-2); line-height: 1.7; }
.w-ch06-ci .read b { color: var(--ink); font-variant-numeric: tabular-nums; }
`));
}

export function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch06-ci');
  const num = (label, value) => h('input', { type: 'number', min: 0, max: 100000, step: 1, value, 'aria-label': label });
  const winIn = num('胜场数', config.wins ?? 5);
  const lossIn = num('负场数', config.losses ?? 3);
  const tieIn = num('平局数', config.ties ?? 0);
  const confSel = h('select', { 'aria-label': '置信水平' }, Object.keys(Z).map((k) => h('option', { value: k }, k)));
  confSel.value = '95%';
  const presetList = [
    { label: '5 胜 3 负', w: 5, l: 3, t: 0 }, { label: '8 胜 0 负', w: 8, l: 0, t: 0 },
    { label: '13 胜 8 负', w: 13, l: 8, t: 0 }, { label: '50 胜 50 负', w: 50, l: 50, t: 0 },
    { label: '71 胜 29 负', w: 71, l: 29, t: 0 }];
  const presets = h('div', { class: 'presets', role: 'group', 'aria-label': '例子' }, presetList.map((p) =>
    h('button', { type: 'button', onclick: () => { winIn.value = p.w; lossIn.value = p.l; tieIn.value = p.t; render(); } }, p.label)));
  const svgBox = h('div');
  const read = h('div', { class: 'read', 'aria-live': 'polite' });
  root.append(h('div', { class: 'ctrl' },
    h('label', {}, '胜', winIn), h('label', {}, '负', lossIn), h('label', {}, '平', tieIn), h('label', {}, '置信水平', confSel)),
  presets, svgBox, read);

  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  };

  function render() {
    const w = Math.max(0, Math.floor(+winIn.value || 0));
    const l = Math.max(0, Math.floor(+lossIn.value || 0));
    const t = Math.max(0, Math.floor(+tieIn.value || 0));
    const z = Z[confSel.value];
    const n = w + l;
    const [lo, hi] = wilson(w, n, z);
    const [wlo, whi] = wald(w, n, z);
    const p = n ? w / n : NaN;
    const X = (v) => 20 + v * 360;
    const s = el('svg', { viewBox: '0 0 400 70', role: 'img', 'aria-label': `胜率 ${n ? p.toFixed(3) : '无'}，区间 ${lo.toFixed(3)} 到 ${hi.toFixed(3)}` });
    s.append(el('line', { x1: X(0), x2: X(1), y1: 30, y2: 30, stroke: 'var(--rule)', 'stroke-width': 2 }));
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      s.append(el('line', { x1: X(v), x2: X(v), y1: 24, y2: 36, stroke: v === 0.5 ? 'var(--ink-2)' : 'var(--rule)', 'stroke-width': 1 }));
      s.append(el('text', { x: X(v), y: 56, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-3)' }, String(v)));
    }
    if (n) {
      s.append(el('rect', { x: X(lo), y: 22, width: Math.max(1, X(hi) - X(lo)), height: 16, rx: 3, fill: 'var(--ai)', opacity: 0.35 }));
      s.append(el('circle', { cx: X(p), cy: 30, r: 5, fill: 'var(--ai)' }));
    }
    svgBox.replaceChildren(s);
    if (!n) {
      read.replaceChildren(h('div', {}, '去掉平局后没有胜负，无法估计胜率。'));
      return;
    }
    const pv = signTestP(w, l);
    read.replaceChildren(
      h('div', {}, `不计平局的胜率 `, h('b', {}, `${w}/${n} = ${p.toFixed(3)}`), `（另有 ${t} 局平局）`),
      h('div', {}, `${confSel.value} Wilson 区间 `, h('b', {}, `[${lo.toFixed(3)}, ${hi.toFixed(3)}]`),
        lo > 0.5 ? '：整个区间都在 0.5 以上' : hi < 0.5 ? '：整个区间都在 0.5 以下' : '：区间包含 0.5，分不出谁更好'),
      h('div', {}, `对比：常见的 p ± z·√(p(1−p)/n) 给出 `, h('b', {}, `[${wlo.toFixed(3)}, ${whi.toFixed(3)}]`),
        wlo < 0 || whi > 1 ? '，越出了 [0, 1]' : (w === 0 || l === 0 ? '，宽度为 0，过于自信' : '')),
      h('div', {}, '符号检验（双侧）p = ', h('b', {}, pv < 1e-4 ? pv.toExponential(2) : pv.toFixed(4)),
        pv < 0.05 ? '：两个模型一样好时，很少出现这么悬殊的结果' : '：两个模型一样好时，这样的结果并不罕见'));
  }
  for (const e of [winIn, lossIn, tieIn, confSel]) e.addEventListener('input', render);
  confSel.addEventListener('change', render);
  render();
}
