/* 第 9 章各交互图共用的小工具（它本身不是交互图）。纯计算，tests/js/ch09-widgets.test.mjs 在 Node 里直接调用。 */

/** 固定种子的伪随机数（mulberry32），保证每次打开页面数据都一样。返回 [0, 1) 的数。 */
export function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 原书 ch09/01_pretrain.py 的 get_lr：warmup 线性升到 maxLr，再线性降到 0。 */
export function bookLr(it, maxLr, warmup, maxIters) {
  if (it < warmup) return maxLr * (it / warmup);
  if (it < maxIters) return maxLr * (1 - (it - warmup) / (maxIters - warmup));
  return 0;
}
