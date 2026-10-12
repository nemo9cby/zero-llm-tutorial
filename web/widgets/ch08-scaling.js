/* 图 8-1：IsoFLOP 曲线与 Chinchilla 计算器。
   loss 模型取 Besiroglu 等 (2024, arXiv:2404.10102) 对 Chinchilla 数据的重新拟合：
   L(N, D) = E + A/N^α + B/D^β。只借用曲线的形状，绝对数值不能套到 WebBot 上。
   同一组结论由验证 8.1-d、8.1-e、8.1-f 钉住；这里的纯函数由 tests/js/ch08-widgets.test.mjs 对拍。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';

export const FIT = { E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 };

export const lossFit = (N, D, f = FIT) => f.E + f.A / N ** f.alpha + f.B / D ** f.beta;

/** 在 D = C/(6N) 上令 dL/dN = 0 的闭式解。 */
export function nOpt(C, f = FIT) {
  const G = ((f.alpha * f.A) / (f.beta * f.B)) ** (1 / (f.alpha + f.beta));
  return G * (C / 6) ** (f.beta / (f.alpha + f.beta));
}

/** “20 token/参数”规则：C = 6·N·20N = 120N²。 */
export function chinchilla20(C) {
  const N = Math.sqrt(C / 120);
  return { N, D: 20 * N };
}

export const H100_BF16 = 1979e12 / 2; // CS336 讲义代码 facts.py 的取值
export const h100Hours = (C, mfu = 0.35) => C / (H100_BF16 * mfu) / 3600;

export const PRESETS = {
  codebot: { label: 'CodeBot（第 3 章）', N: 11_121_640, D: 20_000 * 32 * 256 },
  storybot: { label: 'StoryBot（第 6 章）', N: 22_696_448, D: 40_000 * 32 * 256 },
  webbot: { label: 'WebBot（原书）', N: 152_316_672, D: 100_000 * 32 * 4 * 1024 },
  webbot20: { label: 'WebBot 预算版（20 token/参数）', N: 152_316_672, D: 20 * 152_316_672 },
  chinchilla: { label: 'Chinchilla 70B', N: 70e9, D: 1.4e12 },
  llama7b: { label: 'LLaMA-7B（6.7B）', N: 6.7e9, D: 1.0e12 },
};

/** 152316672 -> "152M"，1.4e12 -> "1.4T"。 */
export function human(n) {
  const units = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
  for (const [u, s] of units) {
    if (Math.abs(n) >= u) {
      const v = n / u;
      return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1).replace(/\.0$/, '') : v.toFixed(2).replace(/\.?0+$/, '')}${s}`;
    }
  }
  return String(Math.round(n));
}

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export const sci = (x) => {
  const e = Math.floor(Math.log10(x));
  return `${(x / 10 ** e).toFixed(2)}×10${String(e).split('').map((c) => SUP[c]).join('')}`;
};

/** 两条竖线靠得太近时（窄屏上常见），只在左边那条上写一个合并的标签，免得文字叠在一起。 */
export function vLineLabels(xNow, xOpt, x0, x1, width) {
  const px = (Math.abs(xNow - xOpt) / (x1 - x0)) * Math.max(width - 68, 1);
  if (px >= 64) return [{ x: xNow, label: '当前 N' }, { x: xOpt, label: '最优' }];
  const [left, right] = xNow <= xOpt ? [xNow, xOpt] : [xOpt, xNow];
  return [{ x: left, label: xNow <= xOpt ? '当前 N｜最优' : '最优｜当前 N' }, { x: right }];
}

/** 一条等算力曲线：x = log10 N，y = 拟合的 loss。 */
export function isoCurve(C, x0, x1, n = 121) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = x0 + ((x1 - x0) * i) / (n - 1);
    const N = 10 ** x;
    pts.push([x, lossFit(N, C / (6 * N))]);
  }
  return pts;
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch08-scaling .ctl { display: grid; grid-template-columns: 7.5em minmax(0, 1fr) 4.2em; gap: .35rem .6rem; align-items: center; font-size: .9rem; margin: .4rem 0 .8rem; }
.w-ch08-scaling .ctl input[type=range] { width: 100%; min-width: 0; }
.w-ch08-scaling .ctl output { font-variant-numeric: tabular-nums; text-align: right; }
.w-ch08-scaling select { font: inherit; max-width: 100%; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .15rem .3rem; }
.w-ch08-scaling .readout { display: grid; grid-template-columns: repeat(auto-fit, minmax(13.5rem, 1fr)); gap: .3rem 1.2rem; font-size: .88rem; margin: .2rem 0 .9rem; }
.w-ch08-scaling .readout div { border-left: 2px solid var(--rule); padding-left: .55rem; }
.w-ch08-scaling .readout b { font-variant-numeric: tabular-nums; font-weight: 600; }
.w-ch08-scaling .readout .k { display: block; color: var(--ink-3); font-size: .78rem; }
`));
}

export function mount(root, config = {}) {
  injectStyle();
  const box = h('div', { class: 'w-ch08-scaling' });
  root.append(box);

  const preset = h('select', { 'aria-label': '选择一个预设模型' },
    Object.entries(PRESETS).map(([k, p]) => h('option', { value: k }, p.label)),
    h('option', { value: 'custom' }, '自定义'));
  const nRange = h('input', { type: 'range', min: 6, max: 11.5, step: 0.01, 'aria-label': '参数量 N（对数刻度）' });
  const dRange = h('input', { type: 'range', min: 7, max: 13, step: 0.01, 'aria-label': '训练 token 数 D（对数刻度）' });
  const nOut = h('output', {});
  const dOut = h('output', {});
  const readout = h('div', { class: 'readout', 'aria-live': 'polite' });
  const chartBox = h('div');
  box.append(
    h('div', { class: 'w-row' }, h('span', { class: 'w-label' }, '预设'), preset),
    h('div', { class: 'ctl' },
      h('label', {}, '参数量 N'), nRange, nOut,
      h('label', {}, '训练 token D'), dRange, dOut),
    readout, chartBox,
    h('p', { class: 'w-label' }, '曲线来自 Chinchilla 数据的拟合（Besiroglu 等 2024），只能看形状；H100 时间按 bf16 稠密峰值 989 TFLOPS、MFU 35% 估算。'),
  );

  let N = PRESETS.webbot.N;
  let D = PRESETS.webbot.D;
  const start = PRESETS[config.preset];
  if (start) { N = start.N; D = start.D; preset.value = config.preset; }

  let chart = null;
  const item = (k, v) => h('div', {}, h('span', { class: 'k' }, k), h('b', {}, v));

  function render() {
    nRange.value = Math.log10(N).toFixed(2);
    dRange.value = Math.log10(D).toFixed(2);
    nOut.value = human(N);
    dOut.value = human(D);
    const C = 6 * N * D;
    const No = nOpt(C);
    const Do = C / (6 * No);
    const r20 = chinchilla20(C);
    const gap = lossFit(N, D) - lossFit(No, Do);
    readout.replaceChildren(
      item('算力 C = 6ND', `${sci(C)} FLOPs`),
      item('每个参数的 token 数 D/N', `${(D / N).toFixed(1)}（Chinchilla ≈ 20）`),
      item('这个算力下的最优（拟合）', `N ≈ ${human(No)}，D ≈ ${human(Do)}`),
      item('按 20 token/参数分配', `N ≈ ${human(r20.N)}，D ≈ ${human(r20.D)}`),
      item('比最优多付出的 loss', `+${gap.toFixed(3)}（N 是最优的 ${(N / No).toFixed(2)} 倍）`),
      item('单张 H100（MFU 35%）', h100Hours(C) < 1 ? `约 ${(h100Hours(C) * 60).toFixed(0)} 分钟` : `约 ${h100Hours(C).toFixed(1)} 小时`),
    );
    // 三条等算力曲线：C/10、C、10C。U 形两臂太高的部分截掉，免得把谷底压扁。
    const xo = Math.log10(No);
    const x0 = Math.max(4, xo - 2.6), x1 = xo + 2.6;
    const curves = [[C / 10, 'C/10'], [C, '当前 C'], [C * 10, '10×C']].map(([c, name]) => ({ name, c, pts: isoCurve(c, x0, x1) }));
    const yMin = lossFit(nOpt(C * 10), (C * 10) / (6 * nOpt(C * 10))) - 0.05;
    const yMax = lossFit(nOpt(C / 10), C / 10 / (6 * nOpt(C / 10))) + 0.45;
    const series = curves.map((s) => ({ name: s.name, points: s.pts.filter((p) => p[1] <= yMax) }));
    const opts = {
      series, xLabel: '参数量 N', yLabel: '拟合的 loss（只看形状）', yMin, yMax, height: 250,
      xFormat: (v) => human(10 ** v), yFormat: (v) => v.toFixed(2),
      vLines: vLineLabels(Math.log10(N), xo, x0, x1, chartBox.clientWidth || 600),
      ariaLabel: '三条等算力曲线：横轴是参数量（对数刻度），纵轴是拟合的 loss',
    };
    if (chart) chart.update(opts); else chart = lineChart(chartBox, opts);
  }

  preset.addEventListener('change', () => {
    const p = PRESETS[preset.value];
    if (p) { N = p.N; D = p.D; render(); }
  });
  nRange.addEventListener('input', () => { N = 10 ** +nRange.value; preset.value = 'custom'; render(); });
  dRange.addEventListener('input', () => { D = 10 ** +dRange.value; preset.value = 'custom'; render(); });
  render();
}
