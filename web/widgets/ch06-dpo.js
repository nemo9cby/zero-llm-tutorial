/* 图：DPO 的 β 与 margin（6.7.4 节）。
   s_c、s_r 分别是 chosen、rejected 的 log π − log π_ref（整条回答求和，单位 nats）。
   z = β (s_c − s_r)，loss = −log σ(z) = log(1 + e^(−z))，每条 pair 的梯度权重是 β σ(−z)。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';

export const softplus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export function dpo(beta, sc, sr) {
  const z = beta * (sc - sr);
  return { z, loss: softplus(-z), prob: sigmoid(z), weight: beta * sigmoid(-z), rc: beta * sc, rr: beta * sr };
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch06-dpo .ctrl { display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: .4rem 1.2rem; font-size: .88rem; }
.w-ch06-dpo label { display: flex; flex-direction: column; gap: .15rem; color: var(--ink-2); }
.w-ch06-dpo label b { color: var(--ink); font-variant-numeric: tabular-nums; }
.w-ch06-dpo input[type=range] { width: 100%; margin: 0; accent-color: var(--ai); }
.w-ch06-dpo .presets { display: flex; flex-wrap: wrap; gap: .3rem; margin: .5rem 0; }
.w-ch06-dpo .presets button { font-size: .8rem; padding: .1rem .5rem; }
.w-ch06-dpo .read { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: .1rem 1rem; font-size: .88rem; color: var(--ink-2); margin: .5rem 0; }
.w-ch06-dpo .read b { color: var(--ink); font-variant-numeric: tabular-nums; }
`));
}

export function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch06-dpo');
  const range = (label, min, max, step, value) => h('input', { type: 'range', min, max, step, value, 'aria-label': label });
  const betaIn = range('β', 0.01, 1, 0.01, 0.1);
  const scIn = range('chosen 的 log π − log π_ref（nats）', -60, 60, 0.5, 0);
  const srIn = range('rejected 的 log π − log π_ref（nats）', -60, 60, 0.5, 0);
  const bL = h('b'), cL = h('b'), rL = h('b');
  const read = h('div', { class: 'read', 'aria-live': 'polite' });
  const chartBox = h('div');
  const presetBox = h('div', { class: 'presets', role: 'group', 'aria-label': '例子' });
  const presets = [
    { label: '初始：π = π_ref', sc: 0, sr: 0 },
    { label: 'chosen 升、rejected 降', sc: 10, sr: -10 },
    { label: '两者都降，rejected 降得更多', sc: -10, sr: -40 },
  ];
  root.append(
    h('div', { class: 'ctrl' },
      h('label', {}, h('span', {}, 'β ', bL), betaIn),
      h('label', {}, h('span', {}, 's_c = chosen 的 log π − log π_ref ', cL), scIn),
      h('label', {}, h('span', {}, 's_r = rejected 的 log π − log π_ref ', rL), srIn)),
    presetBox, read, chartBox);
  const addPreset = (p) => presetBox.append(h('button', {
    type: 'button', onclick: () => { scIn.value = p.sc; srIn.value = p.sr; render(); },
  }, p.label));
  presets.forEach(addPreset);

  // 若有记录运行，再加一个“CPU 上训练 1000 步后的平均值”
  fetch(config.src || 'data/ch06/dpo_cpu.json').then((r) => (r.ok ? r.json() : null)).then((rec) => {
    const tp = rec?.post?.train_pairs;
    if (tp) addPreset({ label: '本章 CPU 实测：1000 步后的平均', sc: +tp.chosen_shift_nats.toFixed(1), sr: +tp.rejected_shift_nats.toFixed(1) });
  }).catch(() => { /* 没有记录也能用 */ });

  const zs = [...Array(121).keys()].map((i) => -6 + i * 0.1);
  let chart = null;
  function render() {
    const beta = +betaIn.value, sc = +scIn.value, sr = +srIn.value;
    bL.textContent = beta.toFixed(2); cL.textContent = sc.toFixed(1); rL.textContent = sr.toFixed(1);
    const d = dpo(beta, sc, sr);
    read.replaceChildren(
      h('div', {}, 'z = β(s_c − s_r) = ', h('b', {}, d.z.toFixed(3))),
      h('div', {}, 'loss = −log σ(z) = ', h('b', {}, d.loss.toFixed(4))),
      h('div', {}, '隐式奖励 r_c = β·s_c = ', h('b', {}, d.rc.toFixed(2)), '，r_r = ', h('b', {}, d.rr.toFixed(2))),
      h('div', {}, 'P(chosen ≻ rejected) = σ(z) = ', h('b', {}, d.prob.toFixed(3))),
      h('div', {}, '这条 pair 的梯度权重 βσ(−z) = ', h('b', {}, d.weight.toFixed(4))),
      h('div', {}, 'loss 只取决于 s_c − s_r = ', h('b', {}, (sc - sr).toFixed(1))));
    const zc = Math.max(-6, Math.min(6, d.z));
    const opts = {
      series: [{ name: 'loss', points: zs.map((z) => [+z.toFixed(1), softplus(-z)]) }],
      xLabel: 'z', yLabel: 'DPO loss = −log σ(z)', height: 220, yMin: 0, table: false,
      refLines: [{ y: Math.LN2, label: 'ln 2' }],
      vLines: [{ x: zc, label: d.z > 6 ? '当前 z（> 6）' : d.z < -6 ? '当前 z（< −6）' : '当前 z' }],
    };
    if (!chart) chart = lineChart(chartBox, opts); else chart.update(opts);
  }
  for (const el of [betaIn, scIn, srIn]) el.addEventListener('input', render);
  render();
}
