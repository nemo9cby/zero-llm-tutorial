/* 图 8-5：每对 RoPE 维度在整个上下文里最多转几圈。
   训练时转不满一圈的低频维度，直接外推会转进没见过的角度；PI 把位置压回训练范围；NTK-aware 改底数，高频不变、低频放慢。
   同一组结论由验证 8.6-b、8.6-c 钉住；纯函数由 tests/js/ch08-widgets.test.mjs 对拍。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';

export const ntkBase = (theta, D, s) => theta * s ** (D / (D - 2));

/** 第 i 对维度在 [0, 最大位置] 内转过的最大圈数。method: train | extrapolate | pi | ntk。 */
export function turns({ D, theta, L, s }, method) {
  const out = [];
  for (let i = 0; i < D / 2; i++) {
    const base = method === 'ntk' ? ntkBase(theta, D, s) : theta;
    const omega = base ** ((-2 * i) / D);
    const maxPos = method === 'train' ? L - 1 : method === 'pi' ? (s * L - 1) / s : s * L - 1;
    out.push((maxPos * omega) / (2 * Math.PI));
  }
  return out;
}

/** 训练时转不满一圈（波长比上下文长）的维度对。 */
export function slowPairs(D, theta, L) {
  const idx = [];
  for (let i = 0; i < D / 2; i++) if ((2 * Math.PI) / theta ** ((-2 * i) / D) > L) idx.push(i);
  return idx;
}

const SERIES = [['train', '训练时'], ['extrapolate', '直接外推'], ['pi', '位置插值'], ['ntk', 'NTK-aware']];

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch08-ropeext .ctl { display: flex; flex-wrap: wrap; gap: .4rem 1rem; font-size: .88rem; margin-bottom: .6rem; }
.w-ch08-ropeext select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; }
.w-ch08-ropeext .facts { font-size: .88rem; margin-top: .4rem; }
`));
}

export function mount(root, config = {}) {
  injectStyle();
  const st = { D: config.D || 64, theta: config.theta || 10000, L: config.L || 1024, s: config.s || 2 };
  const sel = (key, label, opts) => {
    const el = h('select', { 'aria-label': label }, opts.map(([v, t]) => h('option', { value: v }, t)));
    el.value = String(st[key]);
    el.addEventListener('change', () => { st[key] = +el.value; draw(); });
    return h('label', {}, `${label} `, el);
  };
  const chartBox = h('div');
  const facts = h('div', { class: 'facts', 'aria-live': 'polite' });
  root.append(h('div', { class: 'w-ch08-ropeext' },
    h('div', { class: 'ctl' },
      sel('D', '每头维度', [[32, '32（StoryBot）'], [64, '64（WebBot）'], [128, '128']]),
      sel('theta', '底数 θ', [[10000, '10,000'], [500000, '500,000'], [1000000, '1,000,000']]),
      sel('L', '训练长度', [[256, '256'], [1024, '1,024'], [4096, '4,096']]),
      sel('s', '扩展倍数 s', [[2, '2'], [4, '4'], [8, '8']])),
    chartBox, facts));

  let chart = null;
  function draw() {
    const series = SERIES.map(([m, name]) => ({ name, points: turns(st, m).map((y, i) => [i, y]) }));
    const opts = {
      series, logY: true, height: 260, xLabel: '维度对的编号 i', yLabel: '最多转过的圈数（对数刻度）',
      xFormat: (v) => String(Math.round(v)),
      yFormat: (v) => (v >= 1 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : v.toPrecision(1)),
      refLines: [{ y: 1, label: '一圈' }],
      ariaLabel: '每对维度在上下文里最多转过的圈数：训练时、直接外推、位置插值、NTK-aware',
    };
    if (chart) chart.update(opts); else chart = lineChart(chartBox, opts);
    const slow = slowPairs(st.D, st.theta, st.L);
    facts.replaceChildren(
      h('div', {}, `训练时转不满一圈的维度对：${slow.length} / ${st.D / 2}`, slow.length ? `（i ≥ ${slow[0]}）` : ''),
      h('div', {}, `NTK-aware 的新底数 θ′ = θ·s^(d/(d−2)) ≈ ${Math.round(ntkBase(st.theta, st.D, st.s)).toLocaleString('en-US')}；最高频不变，最低频恰好放慢 ${st.s} 倍。`),
      h('div', { class: 'w-label' }, '位置插值（绿）与训练时（蓝）几乎完全重合，蓝线被绿线盖住了：PI 恰好把每一对的角度压回训练时的范围。'),
    );
  }
  draw();
}
