/* 第 3 章各交互图共用的小工具：纯函数（可在 Node 里单元测试）+ 少量 DOM 控件。
   不是交互图本身，页面上没有 figure 引用它。 */
import { h } from './_lib.js';

// ---------------------------------------------------------------- 纯函数

/** 可复现的伪随机数（mulberry32），返回 [0, 1) 的浮点数。 */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 数值稳定的 softmax；T = 0 时返回 argmax 的 one-hot（与原书 generate() 的特例一致）。 */
export function softmax(z, T = 1) {
  const n = z.length;
  if (T === 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (z[i] > z[best]) best = i;
    return Array.from({ length: n }, (_, i) => (i === best ? 1 : 0));
  }
  let m = -Infinity;
  for (const v of z) if (v / T > m) m = v / T;
  const e = z.map((v) => Math.exp(v / T - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** 熵（nats）。 */
export function entropy(p) {
  let H = 0;
  for (const v of p) if (v > 0) H -= v * Math.log(v);
  return H;
}

/** 按概率从大到小的下标。 */
export function argsortDesc(p) {
  return p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map((x) => x[1]);
}

/** Top-k：只保留概率最大的 k 个，再归一化；k <= 0 表示不截断。 */
export function topK(p, k) {
  if (!k || k <= 0 || k >= p.length) return p.slice();
  const keep = new Set(argsortDesc(p).slice(0, k));
  const q = p.map((v, i) => (keep.has(i) ? v : 0));
  const s = q.reduce((a, b) => a + b, 0);
  return q.map((v) => v / s);
}

/** Top-p：保留累计概率首次达到 thr 的最少一批 token，再归一化；thr >= 1 表示不截断。 */
export function topP(p, thr) {
  if (thr >= 1) return p.slice();
  const order = argsortDesc(p);
  const keep = new Set();
  let cum = 0;
  for (const i of order) {
    keep.add(i);
    cum += p[i];
    if (cum >= thr) break;
  }
  const q = p.map((v, i) => (keep.has(i) ? v : 0));
  const s = q.reduce((a, b) => a + b, 0);
  return q.map((v) => v / s);
}

/** 按分布 p 抽一个下标（逆 CDF）。 */
export function sample(p, rand) {
  const u = rand();
  let c = 0;
  for (let i = 0; i < p.length; i++) {
    c += p[i];
    if (u < c) return i;
  }
  for (let i = p.length - 1; i >= 0; i--) if (p[i] > 0) return i;
  return p.length - 1;
}

/** 09_grpo.py 的组内优势：r - mean(r)。stdNorm 时再除以 (std + eps)，std 用 torch 默认的无偏估计。 */
export function advantages(rewards, { stdNorm = false, eps = 1e-4 } = {}) {
  const G = rewards.length;
  const mean = rewards.reduce((a, b) => a + b, 0) / G;
  const A = rewards.map((r) => r - mean);
  if (!stdNorm) return A;
  const std = G > 1 ? Math.sqrt(A.reduce((a, b) => a + b * b, 0) / (G - 1)) : NaN;
  return A.map((a) => a / (std + eps));
}

/** 除自己之外其余回答的平均作基线（RLOO）。 */
export function looAdvantages(rewards) {
  const G = rewards.length;
  if (G < 2) return rewards.map(() => NaN);
  const s = rewards.reduce((a, b) => a + b, 0);
  return rewards.map((r) => r - (s - r) / (G - 1));
}

/** 一组 G 个回答没有学习信号（全对或全错）的概率。 */
export const pNoSignal = (p, G) => p ** G + (1 - p) ** G;

// ---------------------------------------------------------------- DOM 控件

let styled = false;
export function injectStyle() {
  if (styled || typeof document === 'undefined') return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch03 { font-size: .9rem; color: var(--ink); }
.w-ch03, .w-ch03 * { font-variant-ligatures: none !important; }
/* 顺序色阶的热力图在暗色主题下也是浅色格子：格内数字固定用深色，保证对比度 */
.w-ch03 .hm-seq .viz svg text[fill="var(--ink)"] { fill: #1b2233; }
.w-ch03 .ctl { display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; align-items: center; margin: .4rem 0 .6rem; }
.w-ch03 label.slider { display: inline-flex; align-items: center; gap: .45rem; min-width: 0; }
.w-ch03 label.slider input[type=range] { width: 9.5rem; max-width: 42vw; accent-color: var(--ai); }
.w-ch03 .val { font-family: var(--mono); font-size: .82rem; min-width: 3.2em; color: var(--ai); font-variant-numeric: tabular-nums; }
.w-ch03 .seg { display: inline-flex; flex-wrap: wrap; gap: .3rem; }
.w-ch03 .seg button[aria-pressed=true] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-ch03 .stat { color: var(--ink-2); font-size: .85rem; margin: .3rem 0; font-variant-numeric: tabular-nums; }
.w-ch03 .flag { display: inline-block; padding: .05rem .5rem; border-radius: 3px; border: 1px solid var(--kincha); color: var(--kincha); font-size: .82rem; }
.w-ch03 .muted { color: var(--ink-3); }
.w-ch03 .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1rem; }
@media (max-width: 640px) { .w-ch03 .two { grid-template-columns: minmax(0, 1fr); } }
.w-ch03 .tokc { font-family: var(--mono); white-space: pre; }
@media (prefers-reduced-motion: reduce) { .w-ch03 * { transition: none !important; animation: none !important; } }
`));
}

/** 带标签与数值显示的滑块。onInput(value) 在拖动时调用。 */
export function slider({ label, min, max, step, value, format = (v) => String(v), aria, onInput }) {
  const out = h('span', { class: 'val' }, format(value));
  const input = h('input', { type: 'range', min, max, step, value, 'aria-label': aria || label });
  input.addEventListener('input', () => {
    const v = Number(input.value);
    out.textContent = format(v);
    onInput(v);
  });
  return { el: h('label', { class: 'slider' }, h('span', {}, label), input, out), input,
    set(v) { input.value = v; out.textContent = format(Number(input.value)); } };
}

/** 一组互斥按钮。onSelect(index) 在点击时调用。 */
export function segmented(labels, active, onSelect, aria) {
  const btns = labels.map((t, i) => h('button', { type: 'button', 'aria-pressed': String(i === active) }, t));
  const wrap = h('div', { class: 'seg', role: 'group', 'aria-label': aria || '选择' }, btns);
  btns.forEach((b, i) => b.addEventListener('click', () => {
    btns.forEach((x, j) => x.setAttribute('aria-pressed', String(i === j)));
    onSelect(i);
  }));
  return wrap;
}

/** 复选框。 */
export function toggle(label, checked, onChange) {
  const input = h('input', { type: 'checkbox' });
  input.checked = !!checked;
  input.addEventListener('change', () => onChange(input.checked));
  return h('label', {}, input, ' ', label);
}

/** 读取 JSON；失败时在 root 里显示原因并返回 null。 */
export async function loadJSON(root, url) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `数据 ${url} 读取失败：${err.message}`));
    return null;
  }
}

/** 让空白字符可见（与 _lib.visible 相同，但保留全角字符原样）。 */
export const showTok = (s) => s.replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥');
