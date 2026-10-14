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
.w-ch06-ci .scale { position: relative; height: 2.7rem; margin: .7rem .9rem .6rem; line-height: 1.2; }
.w-ch06-ci .scale > * { position: absolute; }
.w-ch06-ci .axis { left: 0; right: 0; top: calc(1rem - 1px); height: 2px; background: var(--rule); }
.w-ch06-ci .tick { top: .625rem; width: 1px; height: .75rem; background: var(--rule); }
.w-ch06-ci .tick.mid { background: var(--ink-2); }
.w-ch06-ci .lab { top: 1.7rem; transform: translateX(-50%); font-size: .78rem; color: var(--ink-3); font-variant-numeric: tabular-nums; }
.w-ch06-ci .band { top: .5rem; height: 1rem; min-width: 2px; border-radius: 3px; background: var(--ai); opacity: .35; }
.w-ch06-ci .dot { top: .625rem; width: .75rem; height: .75rem; margin-left: -.375rem; border-radius: 50%; background: var(--ai); }
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
  const scale = h('div', { class: 'scale', role: 'img' });
  const read = h('div', { class: 'read', 'aria-live': 'polite' });
  root.append(h('div', { class: 'ctrl' },
    h('label', {}, '胜', winIn), h('label', {}, '负', lossIn), h('label', {}, '平', tieIn), h('label', {}, '置信水平', confSel)),
  presets, scale, read);

  // 0–1 的数轴用 HTML 画：位置用百分比，字号固定，桌面和手机上一样清楚
  const at = (v) => `${(100 * v).toFixed(3)}%`;
  function drawScale(n, p, lo, hi) {
    const parts = [h('div', { class: 'axis' })];
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      parts.push(h('div', { class: v === 0.5 ? 'tick mid' : 'tick', style: { left: at(v) } }),
        h('span', { class: 'lab', 'aria-hidden': 'true', style: { left: at(v) } }, String(v)));
    }
    if (n) {
      parts.push(h('div', { class: 'band', style: { left: at(lo), width: at(hi - lo) } }),
        h('div', { class: 'dot', style: { left: at(p) } }));
    }
    scale.setAttribute('aria-label', n ? `胜率 ${p.toFixed(3)}，区间 ${lo.toFixed(3)} 到 ${hi.toFixed(3)}` : '没有胜负记录');
    scale.replaceChildren(...parts);
  }

  function render() {
    const w = Math.max(0, Math.floor(+winIn.value || 0));
    const l = Math.max(0, Math.floor(+lossIn.value || 0));
    const t = Math.max(0, Math.floor(+tieIn.value || 0));
    const z = Z[confSel.value];
    const n = w + l;
    const [lo, hi] = wilson(w, n, z);
    const [wlo, whi] = wald(w, n, z);
    const p = n ? w / n : NaN;
    drawScale(n, p, lo, hi);
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
