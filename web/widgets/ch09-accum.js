/* 图：梯度累积 —— 把 16 个样本的大 batch 拆成 K 个 micro-batch。每份 loss 先除以 K，
   梯度逐份加起来，最后恰好等于一次算完的大 batch 梯度；忘了除以 K，就变成 K 倍。
   模型是一元线性回归 y = w·x，loss = 平均 (w·x − t)²，对 w 的梯度可以心算。 */
import { h, fmt } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';

/** 固定种子的伪随机数（mulberry32），保证每次打开页面数据都一样。 */
function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeData(n = 16, seed = 9) {
  const rnd = mulberry32(seed);
  const xs = [], ts = [];
  for (let i = 0; i < n; i++) {
    const x = 0.5 + 3.5 * rnd();
    xs.push(x);
    ts.push(1.6 * x + 2 * (rnd() - 0.5));
  }
  return { xs, ts };
}

/** 一个 batch 的平均 loss 对 w 的梯度：mean(2·(w·x − t)·x)。 */
export function batchGrad(xs, ts, w) {
  let g = 0;
  for (let i = 0; i < xs.length; i++) g += 2 * (w * xs[i] - ts[i]) * xs[i];
  return g / xs.length;
}

/** 依次累积 K 个 micro-batch 的梯度；path[k] 是累积了 k 份之后 w.grad 里的值。 */
export function accumulate(xs, ts, w, K, divide = true) {
  const b = xs.length / K;
  const path = [0];
  for (let k = 0; k < K; k++) {
    const g = batchGrad(xs.slice(k * b, (k + 1) * b), ts.slice(k * b, (k + 1) * b), w);
    path.push(path[k] + (divide ? g / K : g));
  }
  return path;
}

const KS = [1, 2, 4, 8, 16];
const W0 = 0.5;

export function mount(root, config = {}) {
  const { xs, ts } = makeData(16);
  const big = batchGrad(xs, ts, W0);
  let K = config.K ?? 4;
  let divide = true;

  const kLabel = h('b', {}, '');
  const slider = h('input', {
    type: 'range', min: 0, max: KS.length - 1, step: 1, value: KS.indexOf(K),
    'aria-label': '把大 batch 拆成几个 micro-batch（K）', style: 'width:10rem',
  });
  const box = h('input', { type: 'checkbox', checked: true, id: 'w-ch09-accum-div' });
  const chartBox = h('div');
  const readout = h('div', { class: 'w-row', 'aria-live': 'polite' });
  const cost = h('div', { class: 'w-label' });
  root.append(
    h('div', { class: 'w-row' },
      h('label', {}, '拆成 K = ', kLabel, ' 个 micro-batch '), slider),
    h('div', { class: 'w-row' },
      h('label', { for: 'w-ch09-accum-div' }, box, ' 每个 micro-batch 的 loss 先除以 K（原书 loss = loss / accumulation_steps）')),
    chartBox, readout, cost,
  );

  // path[k]：w.grad 里累积到第 k 份时的值；“大 batch”是一次算完 16 个样本的梯度
  const seriesFor = (path) => [
    { name: '累积值', points: path.map((g, k) => [k, g]) },
    { name: '大 batch', points: [[0, big], [K, big]] },
  ];
  const chart = lineChart(chartBox, {
    series: seriesFor(accumulate(xs, ts, W0, K, divide)),
    xLabel: '已经做完 backward 的 micro-batch 个数', yLabel: '∂L/∂w', height: 230,
  });

  function render() {
    const path = accumulate(xs, ts, W0, K, divide);
    const acc = path[K];
    kLabel.textContent = String(K);
    chart.update({ series: seriesFor(path) });
    const same = Math.abs(acc - big) < 1e-12 * Math.abs(big);
    readout.replaceChildren(
      h('span', {}, '累积结果 ', h('b', {}, fmtNum(acc))),
      h('span', {}, '大 batch ', h('b', {}, fmtNum(big))),
      h('span', {}, same ? '两者相同' : `累积结果是大 batch 的 ${fmtNum(acc / big)} 倍`),
    );
    cost.textContent = `同一时刻只需把 ${fmt(16 / K)} 个样本的激活放进显存；代价是要做 ${fmt(K)} 次前向和反向。`;
  }
  slider.addEventListener('input', () => { K = KS[+slider.value]; render(); });
  box.addEventListener('change', () => { divide = box.checked; render(); });
  render();
}
