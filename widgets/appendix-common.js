/* 附录 A–E 的交互图共用的小工具（不是交互图本身，没有 mount）。 */
import { h } from './_lib.js';

let styled = false;
/** 给交互图的根元素加上 w-appendix 类，并注入共用的表单样式（下拉框、滑块与复选框的颜色跟随主题）。 */
export function appendixRoot(root) {
  if (!styled) {
    styled = true;
    document.head.append(h('style', {}, `
.w-appendix select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule);
  border-radius: 4px; padding: .15rem .3rem; max-width: 100%; }
.w-appendix input[type=range], .w-appendix input[type=checkbox] { accent-color: var(--ai); }
`));
  }
  root.classList.add('w-appendix');
  return root;
}

/** 可复现的随机数：mulberry32 均匀分布 + Box–Muller 正态分布。 */
export function makeRng(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0;
    while (u === 0) u = uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * uniform();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
  /** 按概率 p（和为 1）抽一个下标。 */
  const categorical = (p) => {
    let u = uniform();
    for (let i = 0; i < p.length - 1; i++) { u -= p[i]; if (u < 0) return i; }
    return p.length - 1;
  };
  return { uniform, normal, categorical };
}

export function softmax(s) {
  const m = Math.max(...s);
  const e = s.map((x) => Math.exp(x - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / z);
}

export const sum = (xs) => xs.reduce((a, b) => a + b, 0);
export const dot = (a, b) => a.reduce((acc, x, i) => acc + x * b[i], 0);

/** 直方图：返回 [[区间中点, 密度], ...]，密度的积分为 1（落在 [lo, hi] 之外的样本不计）。
    恰好落在区间边界上的值（误差在 1e-9 个区间宽度以内）一律算进右边的区间：离散分布的取值常常正好在边界上，
    而浮点误差会让同一个值时而偏左、时而偏右，把一根柱子劈成两半。 */
export function histogram(xs, lo, hi, bins) {
  const width = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const x of xs) {
    const k = Math.floor((x - lo) / width + 1e-9);
    if (k >= 0 && k < bins) counts[k] += 1;
    else if (k === bins && x <= hi) counts[bins - 1] += 1;   // 右端点 hi 本身
  }
  return counts.map((c, k) => [lo + (k + 0.5) * width, c / (xs.length * width)]);
}

/** 把 histogram() 的结果画成阶梯形的轮廓（每个区间一段水平线），离散分布也不会画成锯齿。 */
export function stairs(hist) {
  if (hist.length < 2) return hist;
  const w = hist[1][0] - hist[0][0];
  const pts = [[hist[0][0] - w / 2, 0]];
  for (const [x, d] of hist) pts.push([x - w / 2, d], [x + w / 2, d]);
  pts.push([hist[hist.length - 1][0] + w / 2, 0]);
  return pts;
}

/** 给同一页里的多个控件生成不重复的 id（label 的 for 要用）。 */
let uid = 0;
export const nextId = (prefix) => `${prefix}-${(uid += 1)}`;
