/* 图：智能体的 token 计量表。第 t 轮要发送的完整历史 = 开头 B + t × Δ（每轮把全部历史重新发送）。
   单次请求不能超过上下文窗口 W：历史超出之后，每次只能发送最近的 W 个 token，更早的必须截掉或压缩。
   累计输入在超出之前随轮数平方增长，超出之后每轮正好多 W 个（线性）；复用前缀的 KV cache 后，
   需要 prefill 的总量只随轮数线性增长，但只在历史还装得下的时候成立。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle, slider, fmtInt } from './ch05-common.js';

/** Length of the request sent at turn t (t = 1, 2, …): the whole history, capped at the window W. */
export const requestSize = (B, d, t, W = Infinity) => Math.min(B + d * t, W);
/** Sum of the first n requests. Without a window: n·B + d·n(n+1)/2 (quadratic); past the window, +W per turn. */
export function cumulative(B, d, n, W = Infinity) {
  const t0 = Math.min(n, firstOverflow(B, d, W, n) - 1);           // turns whose full history still fits
  const head = t0 * B + d * t0 * (t0 + 1) / 2;
  return n > t0 ? head + (n - t0) * W : head;
}
/** Tokens that need prefill when the server keeps the previous turn's KV cache (prefix caching). */
export const prefillWithPrefixCache = (B, d, n) => B + d * n;
/** First turn whose history no longer fits in the window W (Infinity if none within n). */
export function firstOverflow(B, d, W, n) {
  if (!isFinite(W)) return Infinity;
  const t = Math.floor((W - B) / d) + 1;
  return t <= n ? Math.max(t, 1) : Infinity;
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
    const over = firstOverflow(B, d, W, n);
    const step = Math.max(1, Math.ceil(n / 150));                          // ≤ ~150 points per series
    const ts = Array.from({ length: n }, (_, i) => i + 1).filter((t) => (t - 1) % step === 0 || t === n || t === over - 1 || t === over);
    const last = requestSize(B, d, n, W), total = cumulative(B, d, n, W);
    const fits = over === Infinity;
    const fresh = prefillWithPrefixCache(B, d, fits ? n : over - 1);
    read.replaceChildren(
      fits
        ? h('div', {}, '最后一次请求 ', h('b', {}, fmtInt(last)), ` 个 token，在 ${fmtInt(W)} 的窗口之内`)
        : h('div', {}, h('b', {}, `从第 ${over} 轮起，完整历史已装不进 ${fmtInt(W)} 的窗口`),
          '：每次只能发送最近的 ', h('b', {}, fmtInt(W)), ' 个 token，更早的历史必须截掉或压缩'),
      h('div', {}, `${n} 轮累计输入 `, h('b', {}, fmtInt(total)), ` 个 token，是窗口的 `, h('b', {}, (total / W).toFixed(total / W < 10 ? 1 : 0)), ' 倍'
        + (fits ? '' : `（第 ${over} 轮起每轮正好多 ${fmtInt(W)} 个，改成线性增长）`)),
      fits
        ? h('div', {}, '复用前缀的 KV cache 后，需要 prefill 的一共 ', h('b', {}, fmtInt(fresh)), ' 个')
        : h('div', {}, `复用前缀的 KV cache：前 ${over - 1} 轮需要 prefill 的一共 `, h('b', {}, fmtInt(fresh)),
          ` 个；从第 ${over} 轮起，截断处后面的 token 位置全都变了，它们的 K、V 只能重算，图中不再画这条线`));
    const xTicks = [...new Set([1, Math.round(n / 2), n])];
    const o1 = {
      series: [{ name: '单次请求', points: ts.map((t) => [t, requestSize(B, d, t, W)]) }],
      xLabel: '轮数 t', yLabel: 'token', height: 200, yMin: 0, refLines: [{ y: W, label: `窗口 ${fmtInt(W)}`, labelAt: 'left' }],
      yFormat: short, xTicks, ariaLabel: '每一轮请求的长度：随轮数增长，涨到上下文窗口为止',
    };
    const prefix = ts.filter((t) => t < over).map((t) => [t, prefillWithPrefixCache(B, d, t)]);
    const o2 = {
      series: [
        { name: '累计输入', points: ts.map((t) => [t, cumulative(B, d, t, W)]) },
        ...(prefix.length ? [{ name: '复用前缀缓存时需要 prefill 的累计', points: prefix }] : []),
      ],
      xLabel: '轮数 t', yLabel: 'token', height: 220, yMin: 0, yFormat: short, xTicks,
      ariaLabel: '累计输入与复用前缀缓存时需要 prefill 的累计',
    };
    if (ch1) { ch1.update(o1); ch2.update(o2); } else { ch1 = lineChart(c1, o1); ch2 = lineChart(c2, o2); }
  }
  update();
}
