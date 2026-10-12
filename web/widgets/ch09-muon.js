/* 图：Newton–Schulz 迭代 —— Muon 用 X ← aX + b(XXᵀ)X + c(XXᵀ)²X 把梯度矩阵“正交化”。
   这个迭代不改变奇异向量，只把每个奇异值 x 送进同一个多项式 p(x) = ax + bx³ + cx⁵。
   上图：一个 16×8 矩阵的奇异值（真的用矩阵乘法迭代，再用 Jacobi 法求奇异值）；
   下图：标量多项式迭代 k 次后的形状。两者的数字相同。 */
import { h } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';

export const COEFFS = {
  quintic: { label: '五次多项式（Muon 的系数）', abc: [3.4445, -4.7750, 2.0315] },
  cubic: { label: '三次多项式（经典的 1.5, −0.5）', abc: [1.5, -0.5, 0] },
};

function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const frob = (A) => Math.sqrt(A.reduce((s, row) => s + row.reduce((t, v) => t + v * v, 0), 0));

/** 列正交化（Gram–Schmidt）。 */
function orthonormalColumns(A) {
  const cols = transpose(A);
  const out = [];
  for (const c of cols) {
    const v = c.slice();
    for (const q of out) {
      const d = q.reduce((s, qi, i) => s + qi * v[i], 0);
      for (let i = 0; i < v.length; i++) v[i] -= d * q[i];
    }
    const n = Math.hypot(...v);
    out.push(v.map((x) => x / n));
  }
  return transpose(out);
}

/** G = U·diag(σ)·Vᵀ，σ 从 1 等比降到 1/κ。U、V 由固定种子生成，所以 UVᵀ（理想答案）是已知的。 */
export function makeMatrix(kappa, m = 16, n = 8, seed = 11) {
  const rnd = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const U = orthonormalColumns(Array.from({ length: m }, () => Array.from({ length: n }, gauss)));
  const V = orthonormalColumns(Array.from({ length: n }, () => Array.from({ length: n }, gauss)));
  const sigma = Array.from({ length: n }, (_, i) => kappa ** (-i / (n - 1)));
  const US = U.map((row) => row.map((v, j) => v * sigma[j]));
  return { G: matmul(US, transpose(V)), polar: matmul(U, transpose(V)), sigma };
}

/** Newton–Schulz：先除以 Frobenius 范数，再迭代 steps 次（与 PyTorch 的 _zeropower_via_newtonschulz 同一个公式）。 */
export function nsMatrix(G, [a, b, c], steps, eps = 1e-7) {
  const tall = G.length > G[0].length;
  const nrm = frob(G) + eps;
  let X = (tall ? transpose(G) : G).map((row) => row.map((v) => v / nrm));
  for (let s = 0; s < steps; s++) {
    const A = matmul(X, transpose(X));
    const B = matmul(A, A).map((row, i) => row.map((v, j) => b * A[i][j] + c * v));
    const BX = matmul(B, X);
    X = X.map((row, i) => row.map((v, j) => a * v + BX[i][j]));
  }
  return tall ? transpose(X) : X;
}

/** 对称矩阵的特征值（循环 Jacobi 旋转）。 */
export function symEig(Ain) {
  const n = Ain.length;
  const A = Ain.map((row) => row.slice());
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] ** 2;
    if (off < 1e-26) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = cs * akp - sn * akq;
          A[k][q] = sn * akp + cs * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = cs * apk - sn * aqk;
          A[q][k] = sn * apk + cs * aqk;
        }
      }
    }
  }
  return A.map((row, i) => row[i]);
}

/** 奇异值（从大到小）：XᵀX 或 XXᵀ（取小的那个）的特征值开平方。 */
export function singularValues(X) {
  const gram = X.length >= X[0].length ? matmul(transpose(X), X) : matmul(X, transpose(X));
  return symEig(gram).map((v) => Math.sqrt(Math.max(v, 0))).sort((p, q) => q - p);
}

/** p(x) = ax + bx³ + cx⁵ 的文字形式（系数为 0 的项不写）。 */
export function polyText([a, b, c]) {
  const terms = [[a, 'x'], [b, 'x³'], [c, 'x⁵']].filter(([v]) => v !== 0);
  return terms.map(([v, t], i) => `${i === 0 ? (v < 0 ? '−' : '') : (v < 0 ? ' − ' : ' + ')}${Math.abs(v)}${t}`).join('');
}

/** 标量版：一个奇异值 x 迭代 steps 次。 */
export function nsScalar(x, [a, b, c], steps) {
  for (let s = 0; s < steps; s++) x = a * x + b * x ** 3 + c * x ** 5;
  return x;
}

export function mount(root, config = {}) {
  const state = { coef: config.coef ?? 'quintic', steps: config.steps ?? 5, logk: config.logk ?? 2 };
  const coefSel = h('select', { 'aria-label': '迭代多项式的系数' },
    Object.entries(COEFFS).map(([k, v]) => h('option', { value: k, selected: k === state.coef }, v.label)));
  const stepIn = h('input', { type: 'range', min: 0, max: 10, step: 1, value: state.steps, 'aria-label': '迭代步数 k', style: 'width:9rem' });
  const kapIn = h('input', { type: 'range', min: 0, max: 3, step: 0.25, value: state.logk, 'aria-label': '最大与最小奇异值之比 κ（对数）', style: 'width:9rem' });
  const stepLab = h('b', {}), kapLab = h('b', {});
  const svBox = h('div'), mapBox = h('div');
  const readout = h('p', { class: 'w-label', 'aria-live': 'polite' });
  root.append(
    h('div', { class: 'w-row' }, coefSel),
    h('div', { class: 'w-row' }, h('label', {}, '迭代 k = ', stepLab, ' 步 '), stepIn,
      h('label', {}, '  奇异值之比 κ = ', kapLab, ' '), kapIn),
    svBox, readout, mapBox,
  );

  const xs = Array.from({ length: 401 }, (_, i) => -3 + (3 * i) / 400);   // log10(x)；p⁽ᵏ⁾ 振荡得厉害，点要密
  const svChart = lineChart(svBox, {
    series: [{ name: '迭代前', points: [[1, 1]] }], xLabel: '第几个奇异值（从大到小）',
    yLabel: '奇异值（G 先除以 Frobenius 范数）', refLines: [{ y: 1, label: 'UVᵀ 的奇异值都是 1' }], height: 220, yMin: 0,
  });
  const mapChart = lineChart(mapBox, {
    series: [{ name: 'y = x', points: xs.map((v) => [v, 10 ** v]) }], xLabel: '输入的奇异值 x（对数刻度）',
    yLabel: 'k 步之后的值', xFormat: (v) => fmtNum(+(10 ** v).toPrecision(2)), refLines: [{ y: 1 }],
    height: 220, yMin: 0,
  });

  function render() {
    const abc = COEFFS[state.coef].abc;
    const kappa = 10 ** state.logk;
    stepLab.textContent = String(state.steps);
    kapLab.textContent = fmtNum(+kappa.toPrecision(3));
    const { G, polar, sigma } = makeMatrix(kappa);
    const norm = Math.hypot(...sigma) + 1e-7;          // 与 nsMatrix 一样：范数加 eps
    const before = sigma.map((s) => s / norm);
    const X = nsMatrix(G, abc, state.steps);
    const after = singularValues(X);
    const predicted = before.map((x) => nsScalar(x, abc, state.steps)).sort((p, q) => q - p);
    const gap = Math.max(...after.map((v, i) => Math.abs(v - predicted[i])));
    const err = frob(X.map((row, i) => row.map((v, j) => v - polar[i][j]))) / frob(polar);
    svChart.update({
      series: [
        { name: '迭代前', points: before.map((v, i) => [i + 1, v]) },
        { name: `${state.steps} 步之后`, points: after.map((v, i) => [i + 1, v]) },
      ],
    });
    mapChart.update({
      series: [
        { name: 'y = x', points: xs.map((v) => [v, 10 ** v]) },
        { name: `${state.steps} 步之后`, points: xs.map((v) => [v, nsScalar(10 ** v, abc, state.steps)]) },
      ],
    });
    readout.textContent = `p(x) = ${polyText(abc)}。${state.steps} 步之后：最大/最小奇异值 = ${fmtNum(+(after[0] / after[after.length - 1]).toPrecision(3))}`
      + `（迭代前是 ${fmtNum(+kappa.toPrecision(3))}），与 UVᵀ 的相对误差 ${fmtNum(+err.toPrecision(2))}。`
      + `矩阵迭代算出的奇异值与标量公式 p⁽ᵏ⁾(x) 的最大差 ${gap.toExponential(1)}：奇异向量不动，只有奇异值被映射。`;
  }
  coefSel.addEventListener('change', () => { state.coef = coefSel.value; render(); });
  stepIn.addEventListener('input', () => { state.steps = +stepIn.value; render(); });
  kapIn.addEventListener('input', () => { state.logk = +kapIn.value; render(); });
  render();
}
