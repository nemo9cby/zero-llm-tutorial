/* 附录 B 的交互图：缩放前后的一行注意力。
   config: { d: 64, n: 16 }  —— d 取 2 的幂（1…4096），n 是一行里键的个数。
   上图：随机取一个查询 q 和 n 个键，各分量独立标准正态，画出 softmax 之后的一行权重；
   下图：每个 d 重复 300 次的平均最大权重（不缩放 vs 除以 √d），与验证 B-g 是同一个实验。 */
import { h } from './_lib.js';
import { barChart, lineChart, fmtNum } from './_plot.js';
import { makeRng, softmax, nextId } from './appendix-common.js';

/** 一行注意力：q 与 n 个键的点积（可选除以 √d），再取 softmax。 */
export function attentionRow(rng, d, n, scaled) {
  const q = Array.from({ length: d }, rng.normal);
  const scores = Array.from({ length: n }, () => {
    let s = 0;
    for (let i = 0; i < d; i++) s += q[i] * rng.normal();
    return scaled ? s / Math.sqrt(d) : s;
  });
  return { scores, p: softmax(scores) };
}

/** 平均最大权重；给定 q 时分数服从 N(0, |q|²)（验证 B-f），所以只需采样 |q|。 */
export function meanMaxWeight(rng, d, n, scaled, trials = 300) {
  let total = 0;
  for (let t = 0; t < trials; t++) {
    let norm2 = 0;
    for (let i = 0; i < d; i++) { const x = rng.normal(); norm2 += x * x; }
    const sd = Math.sqrt(scaled ? norm2 / d : norm2);
    const p = softmax(Array.from({ length: n }, () => sd * rng.normal()));
    total += Math.max(...p);
  }
  return total / trials;
}

const std = (xs) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
};

export function mount(root, config = {}) {
  const n = config.n ?? 16;
  let k = Math.round(Math.log2(config.d ?? 64));
  let scaled = false;
  let seed = 1;

  const slider = h('input', { type: 'range', min: 0, max: 12, step: 1, value: k, 'aria-label': '维度 d（2 的幂）' });
  const dOut = h('output', { style: 'min-width:5.5em;font-variant-numeric:tabular-nums' });
  const box = h('input', { type: 'checkbox', id: nextId('w-appendix-b-scale') });
  const again = h('button', { type: 'button' }, '换一个 q');
  const stats = h('div', { class: 'w-label', 'aria-live': 'polite', style: 'margin:.2rem 0 .6rem' });
  const barBox = h('div');
  const lineBox = h('div', { style: 'margin-top:.6rem' });
  root.append(
    h('div', { class: 'w-row' }, h('label', {}, 'd = ', dOut), slider,
      h('label', { for: box.id, style: 'display:inline-flex;gap:.35rem;align-items:center' }, box, '除以 √d'), again),
    stats, h('div', { class: 'w-label' }, `一行注意力：${n} 个键各自分到的权重（最大的那根颜色最深）`), barBox, lineBox);

  // 下图：平均最大权重随 d 的变化（固定种子，只算一次）
  const curveRng = makeRng(2024);
  const ks = [...Array(13).keys()];
  const curve = (sc) => ks.map((kk) => [kk, meanMaxWeight(curveRng, 2 ** kk, n, sc)]);
  const series = [{ name: '不缩放', points: curve(false) }, { name: '除以 √d', points: curve(true) }];
  const line = lineChart(lineBox, {
    series, xLabel: 'd', yLabel: `一行 ${n} 个键里最大的权重（300 次平均）`, height: 220, yMin: 0, yMax: 1,
    xFormat: (v) => fmtNum(2 ** v), vLines: [{ x: k, label: '当前 d' }], table: true,
  });

  let bars = null;
  const render = () => {
    const d = 2 ** k;
    dOut.textContent = String(d);
    slider.setAttribute('aria-valuetext', `d = ${d}`);
    const { scores, p } = attentionRow(makeRng(seed * 7919 + k), d, n, scaled);
    const top = p.indexOf(Math.max(...p));
    const H = -p.reduce((a, x) => a + (x > 0 ? x * Math.log(x) : 0), 0);
    stats.textContent = `分数的标准差 ${std(scores).toFixed(2)}（理论值 ${scaled ? '1' : `√d = ${Math.sqrt(d).toFixed(1)}`}）；`
      + `最大权重 ${p[top].toFixed(3)}；熵 ${H.toFixed(2)} nats（均匀时是 ln ${n} = ${Math.log(n).toFixed(2)}）`;
    const opts = {
      bars: p.map((v, i) => ({ label: String(i + 1), value: v })), yLabel: '注意力权重', yFormat: (v) => fmtNum(v),
      height: 200, highlight: String(top + 1), ariaLabel: `一行注意力的 ${n} 个权重`,
    };
    if (bars) bars.update(opts); else bars = barChart(barBox, opts);
    line.update({ vLines: [{ x: k, label: `d = ${d}` }] });
  };
  slider.addEventListener('input', () => { k = Number(slider.value); render(); });
  box.addEventListener('change', () => { scaled = box.checked; render(); });
  again.addEventListener('click', () => { seed += 1; render(); });
  render();
}
