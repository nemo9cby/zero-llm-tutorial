/* 附录 E 的交互图：β 与最优策略 π* ∝ π_ref · exp(r/β)。
   config: { pi_ref: [...], r: [...], beta: 0.5 }
   每个回答一行：奖励滑块、π_ref 与 π* 的条形、隐式奖励（β 乘以 π* 与 π_ref 之比的对数）；下方是 Z、KL 与一对回答的偏好概率。
   计算与验证 E-c 到 E-f 相同。 */
import { h } from './_lib.js';
import { nextId } from './appendix-common.js';

/** π*(y) = π_ref(y)·e^(r(y)/β) / Z，用对数域计算以免 β 很小时溢出；同时返回 log Z。 */
export function optimalPolicy(piRef, r, beta) {
  const logits = piRef.map((p, i) => Math.log(p) + r[i] / beta);
  const m = Math.max(...logits);
  const e = logits.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return { pi: e.map((x) => x / s), logZ: m + Math.log(s) };
}

export const kl = (p, q) => p.reduce((acc, x, i) => acc + (x > 0 ? x * Math.log(x / q[i]) : 0), 0);
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-appendix-e-rows { display: grid; gap: .55rem; margin: .5rem 0 .7rem; }
.w-appendix-e-row { display: grid; grid-template-columns: 2.2rem minmax(0, 1fr); gap: .15rem .6rem; align-items: center; }
.w-appendix-e-name { font: 600 .95rem/1 var(--mono); grid-row: span 3; align-self: start; padding-top: .2rem; }
.w-appendix-e-ctl { display: flex; flex-wrap: wrap; gap: .2rem .6rem; align-items: center; font-size: .85rem; color: var(--ink-2); }
.w-appendix-e-ctl input { flex: 1 1 8rem; min-width: 6rem; }
.w-appendix-e-bar { display: grid; grid-template-columns: 3.4rem minmax(0, 1fr) 3.6rem; gap: .5rem; align-items: center;
  font-size: .8rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.w-appendix-e-track { height: .7rem; background: var(--paper-2); border-radius: 2px; overflow: hidden; }
.w-appendix-e-fill { height: 100%; border-radius: 2px; }
.w-appendix-e-sum { font-size: .85rem; color: var(--ink-2); line-height: 1.75; }
.w-appendix-e-sum b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
`));
}

/** 一根水平条：返回 { el, set(value) }，更新时只改宽度和数字，不重建 DOM。 */
function makeBar(label, color) {
  const fill = h('div', { class: 'w-appendix-e-fill', style: `background:${color}` });
  const track = h('div', { class: 'w-appendix-e-track', role: 'img' }, fill);
  const num = h('span', {});
  const el = h('div', { class: 'w-appendix-e-bar' }, h('span', {}, label), track, num);
  return {
    el,
    set(value) {
      fill.style.width = `${(100 * value).toFixed(1)}%`;
      num.textContent = value.toFixed(3);
      track.setAttribute('aria-label', `${label} ${value.toFixed(3)}`);
    },
  };
}

export function mount(root, config = {}) {
  injectStyle();
  const piRef = config.pi_ref ?? [0.4, 0.3, 0.2, 0.1];
  const r = [...(config.r ?? [0, 1, 2, 0.5])];
  let logBeta = Math.log10(config.beta ?? 0.5);
  let w = 2, l = 0;

  const betaId = nextId('w-appendix-e-beta');
  const betaIn = h('input', { id: betaId, type: 'range', min: -2, max: 2, step: 0.05, value: logBeta, 'aria-label': 'β（对数刻度）' });
  const betaOut = h('output', { style: 'min-width:4em;font-variant-numeric:tabular-nums' });
  const rows = h('div', { class: 'w-appendix-e-rows' });
  const pick = (value, onChange, label) => {
    const sel = h('select', { 'aria-label': label }, piRef.map((_, i) => h('option', { value: i, selected: i === value }, `y${i}`)));
    sel.addEventListener('change', () => { onChange(Number(sel.value)); render(); });
    return sel;
  };
  const summary = h('div', { class: 'w-appendix-e-sum', 'aria-live': 'polite' });
  const pairSummary = h('div', { class: 'w-appendix-e-sum', 'aria-live': 'polite' });
  root.append(
    h('div', { class: 'w-row' }, h('label', { for: betaId }, 'β = ', betaOut), betaIn),
    rows, summary,
    h('div', { class: 'w-row' }, '比较一对回答：', pick(w, (v) => { w = v; }, '更受偏好的回答 y_w'), ' ≻ ',
      pick(l, (v) => { l = v; }, '不受偏好的回答 y_l')),
    pairSummary);

  // 每个回答一行，只建一次；拖动滑块时只更新文字和条形，滑块本身不离开 DOM
  const rowEls = piRef.map((p0, i) => {
    const rLabel = h('span', {});
    const implicitText = h('span', {});
    const slider = h('input', { type: 'range', min: -1, max: 3, step: 0.1, value: r[i], 'aria-label': `回答 y${i} 的奖励` });
    slider.addEventListener('input', () => { r[i] = Number(slider.value); render(); });
    const refBar = makeBar('π_ref', 'var(--ink-3)');
    const optBar = makeBar('π*', 'var(--ai)');
    refBar.set(p0);
    rows.append(h('div', { class: 'w-appendix-e-row' },
      h('span', { class: 'w-appendix-e-name' }, `y${i}`),
      h('div', { class: 'w-appendix-e-ctl' }, rLabel, slider, implicitText), refBar.el, optBar.el));
    return { rLabel, implicitText, optBar };
  });

  const render = () => {
    const beta = 10 ** logBeta;
    betaOut.textContent = beta >= 1 ? beta.toFixed(2) : beta.toPrecision(2);
    betaIn.setAttribute('aria-valuetext', `β = ${betaOut.textContent}`);
    const { pi, logZ } = optimalPolicy(piRef, r, beta);
    rowEls.forEach(({ rLabel, implicitText, optBar }, i) => {
      const implicit = beta * Math.log(pi[i] / piRef[i]);
      rLabel.textContent = `r = ${r[i].toFixed(1)}`;
      implicitText.textContent = Number.isFinite(implicit)
        ? `隐式奖励 ${implicit.toFixed(3)}，加上 β·log Z 得 ${(implicit + beta * logZ).toFixed(3)}`
        : '隐式奖励 −∞（π* 下溢为 0）';
      optBar.set(pi[i]);
    });
    const expR = (p) => p.reduce((acc, x, i) => acc + x * r[i], 0);
    summary.replaceChildren(
      h('div', {}, 'Z = ', h('b', {}, Math.exp(logZ).toPrecision(4)), '，最大目标值 β·log Z = ', h('b', {}, (beta * logZ).toFixed(3))),
      h('div', {}, 'KL(π* ‖ π_ref) = ', h('b', {}, kl(pi, piRef).toFixed(3)), ' nats；期望奖励从 ',
        h('b', {}, expR(piRef).toFixed(3)), '（π_ref）升到 ', h('b', {}, expR(pi).toFixed(3)), '（π*）'));
    if (w === l) {
      pairSummary.textContent = '请选两个不同的回答。';
      return;
    }
    const pref = sigmoid(r[w] - r[l]);
    pairSummary.replaceChildren(
      h('div', {}, `Bradley–Terry：P(y${w} ≻ y${l}) = σ(r${w} − r${l}) = `, h('b', {}, pref.toFixed(4))),
      h('div', {}, '这一对的 DPO 损失：π_θ = π_ref 时是 ', h('b', {}, Math.log(2).toFixed(4)), '（ln 2）；π_θ = π* 时是 ',
        h('b', {}, (-Math.log(pref)).toFixed(4)), '（−log σ(r_w − r_l)）'));
  };
  betaIn.addEventListener('input', () => { logBeta = Number(betaIn.value); render(); });
  render();
}
