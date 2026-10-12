/* 图：因果掩码实验台（2.6 节）。五种写法 × “改动未来的 token”：看哪些位置的输出跟着变了。
   与验证 2.6-c/d/e 是同一组实验（JS 测试对拍泄漏检测的结论）。输入和权重是随机生成的玩具数据，
   TOKENS 只是给 6 个位置起的名字。 */
import { h, visible } from './_lib.js';
import { seqColor, seqIsDark } from './_plot.js';
import { mulberry32, gaussian, injectStyle, range, onThemeChange, seqLegend } from './ch02-util.js';

export const TOKENS = ['def', ' add', '(', 'a', ',', ' b'];
export const MODES = [
  ['right', '先遮挡，再按行 softmax（原书）'],
  ['nomask', '不加掩码'],
  ['strict', '掩码不含对角线（k=−1）'],
  ['axis', 'softmax 沿列做（axis=−2）'],
  ['after', '先 softmax，再遮挡'],
];
const C = TOKENS.length, E = 8;
export const CHANGED_FROM = 4;          // positions 4 and 5 get new inputs

const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const randMat = (r, c, rand, s = 1) => range(r).map(() => range(c).map(() => s * gaussian(rand)));

/** Deterministic toy inputs and weights; `changed` replaces the inputs at positions ≥ CHANGED_FROM. */
export function toyInputs(seed = 0) {
  const rand = mulberry32(seed + 11);
  const x = randMat(C, E, rand);
  // Q/K weights scaled by 0.5 keep the scores moderate, so the rows are not one-hot and the row sums say something.
  const Wq = randMat(E, E, rand, 0.5), Wk = randMat(E, E, rand, 0.5), Wv = randMat(E, E, rand);
  const xChanged = x.map((row, i) => (i >= CHANGED_FROM ? row.map(() => gaussian(rand)) : row.slice()));
  return { x, xChanged, Wq, Wk, Wv };
}

const visibleAt = (mode, i, j) => (mode === 'nomask' ? true : mode === 'strict' ? j < i : j <= i);

/** Attention weights for one mode. -Infinity marks a masked score; NaN appears when a whole row is masked. */
export function weightsFor(S, mode) {
  if (mode === 'axis') {                                   // mask, then normalize each COLUMN
    const M = S.map((row, i) => row.map((s, j) => (j <= i ? s : -Infinity)));
    const cols = transpose(M).map((col) => {
      const m = Math.max(...col);
      const e = col.map((v) => Math.exp(v - m));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / z);
    });
    return transpose(cols);
  }
  if (mode === 'after') {                                  // softmax the full row, then zero the future
    return S.map((row, i) => {
      const m = Math.max(...row);
      const e = row.map((v) => Math.exp(v - m));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map((v, j) => (j <= i ? v / z : 0));
    });
  }
  return S.map((row, i) => {                               // right / nomask / strict
    const r = row.map((s, j) => (visibleAt(mode, i, j) ? s : -Infinity));
    const m = Math.max(...r);
    const e = r.map((v) => Math.exp(v - m));               // a fully masked row: exp(-inf - -inf) = NaN
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  });
}

export function forward(x, mode, { Wq, Wk, Wv }) {
  const Q = matmul(x, Wq), K = matmul(x, Wk), V = matmul(x, Wv);
  const S = matmul(Q, transpose(K)).map((row) => row.map((s) => s / Math.sqrt(E)));
  const W = weightsFor(S, mode);
  return { S, W, out: matmul(W, V) };
}

/** For each position: did its output change when only positions ≥ CHANGED_FROM changed? NaN counts as broken. */
export function leakReport(mode, seed = 0) {
  const p = toyInputs(seed);
  const a = forward(p.x, mode, p), b = forward(p.xChanged, mode, p);
  return range(C).map((i) => {
    const nan = a.out[i].some(Number.isNaN);
    const changed = a.out[i].some((v, k) => Math.abs(v - b.out[i][k]) > 1e-12);
    return { nan, changed };
  });
}

export function mount(root) {
  injectStyle();
  const params = toyInputs(0);
  let mode = 'right', perturbed = false;

  const modeBtns = MODES.map(([key, label]) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(key === mode) }, label);
    b.addEventListener('click', () => { mode = key; render(); });
    return [key, b];
  });
  const poke = h('button', { type: 'button', 'aria-pressed': 'false' }, '改动未来的 token（位置 4、5）');
  poke.addEventListener('click', () => { perturbed = !perturbed; render(); });
  const grid = h('div', { class: 'w-ch02-grid', role: 'table', 'aria-label': '注意力权重矩阵，行是查询，列是键' });
  const legend = h('div');
  const chips = h('div', { class: 'w-ch02-chips', 'aria-live': 'polite' });
  const verdict = h('p', { class: 'w-label' });
  root.append(
    h('div', { class: 'w-ch02-controls' }, h('span', { class: 'w-ch02-seg', role: 'group', 'aria-label': '掩码的写法' }, modeBtns.map(([, b]) => b))),
    h('div', { class: 'w-ch02-controls' }, poke),
    grid,
    legend,
    h('p', { class: 'w-ch02-note' }, '输入和权重都是随机生成的玩具数据；行列上的 token 只是给 6 个位置起的名字，不是 CodeBot 的真实注意力。'),
    h('div', { class: 'w-label' }, '与“没改动”时相比，每个位置的输出：'),
    chips, verdict,
  );

  function cell(v, i, j) {
    const masked = (mode === 'nomask' ? false : mode === 'strict' ? j >= i : j > i);
    if (Number.isNaN(v)) return h('span', { class: 'cell nan', role: 'cell', 'aria-label': `第 ${i} 行第 ${j} 列：NaN` }, 'NaN');
    if (masked && v === 0) return h('span', { class: 'cell masked', role: 'cell', 'aria-label': `第 ${i} 行第 ${j} 列：被遮住，权重 0` }, '0');
    return h('span', { class: 'cell', role: 'cell', 'aria-label': `第 ${i} 行第 ${j} 列：${v.toFixed(2)}`,
      // seqColor depends on the theme (its ramp flips in dark mode), and so does seqIsDark: render() reruns on theme change.
      style: { background: seqColor(v), color: seqIsDark(v) ? '#ffffff' : '#1b2233' } }, v.toFixed(2));
  }

  // Header: the position on the first line, the (toy) token on a second, so narrow screens do not cut it off.
  const posHeader = (j, t, role) => h('span', { class: 'cell head', role, title: `位置 ${j}` }, String(j), h('span', { class: 'tok' }, visible(t)));

  function render() {
    for (const [key, b] of modeBtns) b.setAttribute('aria-pressed', String(key === mode));
    poke.setAttribute('aria-pressed', String(perturbed));
    const { W } = forward(perturbed ? params.xChanged : params.x, mode, params);
    grid.style.gridTemplateColumns = `minmax(2.6em, auto) repeat(${C}, minmax(0, 1fr)) minmax(2.6em, auto)`;
    const head = [h('span', { class: 'cell head', role: 'columnheader' }, '查询＼键'),
      ...TOKENS.map((t, j) => posHeader(j, t, 'columnheader')),
      h('span', { class: 'cell head', role: 'columnheader' }, '行和')];
    const rows = W.flatMap((row, i) => {
      const sum = row.reduce((a, b) => a + b, 0);
      return [posHeader(i, TOKENS[i], 'rowheader'),
        ...row.map((v, j) => cell(v, i, j)),
        h('span', { class: `cell head${Number.isFinite(sum) && Math.abs(sum - 1) < 1e-9 ? '' : ' w-ch02-bad'}` },
          Number.isFinite(sum) ? sum.toFixed(2) : 'NaN')];
    });
    grid.replaceChildren(...head, ...rows);
    legend.replaceChildren(seqLegend());

    const rep = leakReport(mode, 0);
    chips.replaceChildren(...rep.map((r, i) => {
      if (i >= CHANGED_FROM) return h('span', { class: 'muted' }, `${i}：被改动`);
      if (r.nan) return h('span', { class: 'bad' }, `${i}：NaN`);
      return r.changed ? h('span', { class: 'bad' }, `${i}：变了`) : h('span', { class: 'ok' }, `${i}：不变`);
    }));
    verdict.textContent = verdictFor(rep);
  }
  render();
  onThemeChange(render);
}

/** The sentence under the grid. A fully masked row is NaN because the max-subtracting softmax computes −∞ − (−∞). */
export function verdictFor(rep) {
  const past = rep.slice(0, CHANGED_FROM);
  if (past.some((r) => r.nan)) return '第 0 行一个可见的键也没有：先减最大值时得到 −∞ − (−∞) = NaN（不减最大值则是 0/0）。这一行的 NaN 还会经过下一层传给所有位置（验证 2.6-d）。';
  if (past.some((r) => r.changed)) return '泄漏：只改动了位置 4、5，排在它们前面的位置却变了，未来的信息漏进了过去。';
  return '不漏：位置 0–3 的输出与改动前逐位相同。只有“先遮挡、再按行 softmax”能做到这一点。';
}
