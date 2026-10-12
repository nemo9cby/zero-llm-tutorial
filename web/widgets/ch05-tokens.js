/* 图：智能体的 token 计量表。第 t 轮的请求 = 开头 B + t × Δ（每轮把全部历史重新发送）。
   单次请求受上下文窗口限制；累计输入随轮数平方增长；复用前缀的 KV cache 后，需要 prefill 的总量只随轮数线性增长。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle, slider, fmtInt } from './ch05-common.js';

export const requestSize = (B, d, t) => B + d * t;                         // t = 1, 2, …
export const cumulative = (B, d, n) => n * B + d * n * (n + 1) / 2;
export const prefillWithPrefixCache = (B, d, n) => B + d * n;
/** First turn whose request no longer fits in the window W (Infinity if none within n). */
export function firstOverflow(B, d, W, n) {
  const t = Math.floor((W - B) / d) + 1;
  return t >= 1 && t <= n ? t : Infinity;
}

const WINDOWS = [[32_000, '32K'], [128_000, '128K'], [256_000, '256K'], [1_000_000, '1M']];
const short = (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}K` : String(v));

export function mount(root, config = {}) {
  injectStyle();
  const st = { B: config.base ?? 3000, d: config.perTurn ?? 800, n: config.turns ?? 300, W: config.window ?? 256_000 };
  const box = h('div', { class: 'w-ch05' });
  root.append(box);
  const sB = slider({ label: '开头的 token（系统提示 + 任务）', min: 0, max: 20000, step: 500, value: st.B, format: fmtInt,
    onInput: (v) => { st.B = v; update(); } });
  const sD = slider({ label: '每轮新增', min: 100, max: 5000, step: 100, value: st.d, format: fmtInt, onInput: (v) => { st.d = v; update(); } });
  const sN = slider({ label: '轮数', min: 1, max: 600, value: st.n, onInput: (v) => { st.n = v; update(); } });
  const wSel = h('select', { 'aria-label': '上下文窗口' }, WINDOWS.map(([v, t]) => h('option', { value: v, selected: v === st.W }, t)));
  wSel.addEventListener('change', () => { st.W = +wSel.value; update(); });
  const read = h('div', { class: 'readout', 'aria-live': 'polite' });
  const c1 = h('div'), c2 = h('div');
  box.append(h('div', { class: 'ctl' }, sB.el, sD.el), h('div', { class: 'ctl' }, sN.el, h('label', {}, '上下文窗口', wSel)), read,
    h('div', { class: 'panel-title', style: 'margin-top:.6rem' }, '每一次请求的长度（token）'), c1,
    h('div', { class: 'panel-title', style: 'margin-top:.6rem' }, '到第 t 轮为止的累计（token）'), c2);
  let ch1 = null, ch2 = null;

  function update() {
    const { B, d, n, W } = st;
    const turns = Array.from({ length: n }, (_, i) => i + 1);
    const step = Math.max(1, Math.ceil(n / 150));                          // ≤ ~150 points per series
    const ts = turns.filter((t) => (t - 1) % step === 0 || t === n);
    const last = requestSize(B, d, n), total = cumulative(B, d, n), fresh = prefillWithPrefixCache(B, d, n);
    const over = firstOverflow(B, d, W, n);
    read.replaceChildren(
      h('div', {}, '最后一次请求 ', h('b', {}, fmtInt(last)), ` 个 token，`,
        over === Infinity ? `在 ${fmtInt(W)} 的窗口之内` : h('b', {}, `从第 ${over} 轮起超出 ${fmtInt(W)} 的窗口`)),
      h('div', {}, `${n} 轮累计输入 `, h('b', {}, fmtInt(total)), ` 个 token，是窗口的 `, h('b', {}, (total / W).toFixed(total / W < 10 ? 1 : 0)), ' 倍'),
      h('div', {}, '复用前缀的 KV cache 后，需要 prefill 的一共 ', h('b', {}, fmtInt(fresh)), ' 个'));
    const o1 = {
      series: [{ name: '单次请求', points: ts.map((t) => [t, requestSize(B, d, t)]) }],
      xLabel: '轮数 t', yLabel: 'token', height: 200, yMin: 0, refLines: [{ y: W, label: `窗口 ${fmtInt(W)}` }],
      yFormat: short, ariaLabel: '每一轮请求的长度与上下文窗口',
    };
    const o2 = {
      series: [
        { name: '累计输入', points: ts.map((t) => [t, cumulative(B, d, t)]) },
        { name: '前缀缓存', points: ts.map((t) => [t, prefillWithPrefixCache(B, d, t)]) },
      ],
      xLabel: '轮数 t', yLabel: 'token', height: 220, yMin: 0, yFormat: short, ariaLabel: '累计输入与前缀缓存后的计算量',
    };
    if (ch1) { ch1.update(o1); ch2.update(o2); } else { ch1 = lineChart(c1, o1); ch2 = lineChart(c2, o2); }
  }
  update();
}
