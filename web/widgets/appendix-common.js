/* 附录 A–E 的交互图共用的小工具（不是交互图本身，没有 mount）。 */

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

/** 直方图：返回 [[区间中点, 密度], ...]，密度的积分为 1（落在区间外的样本不计）。 */
export function histogram(xs, lo, hi, bins) {
  const width = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const x of xs) {
    const k = Math.floor((x - lo) / width);
    if (k >= 0 && k < bins) counts[k] += 1;
    else if (x === hi) counts[bins - 1] += 1;
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
