/* 图：GRPO 的组内优势计算器。点击每个回答切换对错，拖动 G，或按正确率 p 随机抽一组。
   算法与 09_grpo.py 相同：A = r − mean(r)；可选“除以标准差”（DeepSeekMath 原版，std 为无偏估计 + 1e-4）。
   config: { G: 8, rewards: [1,0,...] } */
import { h } from './_lib.js';
import { seriesColor } from './_plot.js';
import { injectStyle, advantages, looAdvantages, pNoSignal, rng, slider, toggle } from './ch03-common.js';

const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, ...kids) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of kids) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
};

export function mount(root, config) {
  injectStyle();
  let rewards = (config.rewards || [1, 0, 0, 1, 1, 0, 0, 0]).slice(0, 16);
  let G = rewards.length, stdNorm = false, p = 0.5, seed = 7;

  root.classList.add('w-ch03');
  root.append(h('style', {}, `
.w-ch03-grp .chips { display: flex; flex-wrap: wrap; gap: 4px; margin: .5rem 0; }
.w-ch03-grp .chips button { min-width: 2.6rem; padding: .2rem .4rem; font: 13px/1.4 var(--mono); }
.w-ch03-grp .chips button[aria-pressed=true] { border-color: var(--moss); color: var(--moss); font-weight: 700; }
.w-ch03-grp svg { display: block; width: 100%; height: auto; }
.w-ch03-grp .axis { fill: var(--ink-3); font: 11px var(--mono); }
`));
  root.classList.add('w-ch03-grp');

  const sG = slider({ label: '组大小 G', min: 1, max: 16, step: 1, value: G, aria: '组的大小 G',
    onInput: (v) => { G = v; while (rewards.length < G) rewards.push(0); rewards = rewards.slice(0, G); render(); } });
  const sp = slider({ label: '正确率 p', min: 0, max: 1, step: 0.05, value: p, format: (v) => v.toFixed(2), aria: '随机抽组时每个回答答对的概率',
    onInput: (v) => { p = v; render(); } });
  const drawBtn = h('button', { type: 'button' }, '按 p 随机抽一组');
  drawBtn.addEventListener('click', () => {
    const rand = rng(seed++);
    rewards = Array.from({ length: G }, () => (rand() < p ? 1 : 0));
    render();
  });
  const stdToggle = toggle('除以标准差（DeepSeekMath 原版）', false, (v) => { stdNorm = v; render(); });
  const chips = h('div', { class: 'chips', role: 'group', 'aria-label': '点击切换每个回答的对错' });
  const chart = h('div');
  const stats = h('div', { class: 'stat', 'aria-live': 'polite' });
  const flag = h('div', { 'aria-live': 'polite' });
  root.append(h('div', { class: 'ctl' }, sG.el, sp.el, drawBtn), h('div', { class: 'ctl' }, stdToggle),
    h('div', { class: 'w-label' }, '这一组的 G 个回答（点击切换 ✓ 答对 / ✗ 答错）：'), chips, chart, stats, flag);

  function render() {
    chips.replaceChildren(...rewards.map((r, i) => {
      const b = h('button', { type: 'button', 'aria-pressed': String(r === 1), 'aria-label': `回答 ${i + 1}：${r ? '答对' : '答错'}` },
        `${i + 1} ${r ? '✓' : '✗'}`);
      b.addEventListener('click', () => { rewards[i] = 1 - rewards[i]; render(); });
      return b;
    }));
    const A = advantages(rewards, { stdNorm });
    const mean = rewards.reduce((a, b) => a + b, 0) / G;
    const allSame = rewards.every((r) => r === rewards[0]);
    // 优势条形图：正的向上，负的向下
    const W = Math.max(280, Math.min(720, chart.clientWidth || root.clientWidth || 560)), H = 170, m = { l: 46, r: 8, t: 12, b: 26 };
    const finite = A.filter(Number.isFinite);
    const lim = Math.max(1, ...finite.map((a) => Math.abs(a)));
    const sy = (v) => m.t + (1 - (v + lim) / (2 * lim)) * (H - m.t - m.b);
    const band = (W - m.l - m.r) / G, bw = Math.min(28, band * 0.7);
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `组内优势：${A.map((a) => (Number.isFinite(a) ? a.toFixed(3) : 'NaN')).join('，')}` });
    for (const v of [lim, 0, -lim]) {
      s.append(svg('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: v === 0 ? 'var(--ink-3)' : 'var(--rule)', 'stroke-width': 1 }));
      s.append(svg('text', { class: 'axis', x: m.l - 4, y: sy(v) + 4, 'text-anchor': 'end' }, v === 0 ? '0' : (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))));
    }
    A.forEach((a, i) => {
      const cx = m.l + band * (i + 0.5);
      if (!Number.isFinite(a)) {
        s.append(svg('text', { class: 'axis', x: cx, y: sy(0) - 4, 'text-anchor': 'middle' }, 'NaN'));
      } else if (a !== 0) {
        const y0 = sy(0), y1 = sy(a);
        s.append(svg('rect', { x: cx - bw / 2, y: Math.min(y0, y1), width: bw, height: Math.max(1, Math.abs(y1 - y0)), rx: 2,
          fill: seriesColor(a > 0 ? 0 : 1) }));
      }
      s.append(svg('text', { class: 'axis', x: cx, y: H - 8, 'text-anchor': 'middle' }, rewards[i] ? '✓' : '✗'));
    });
    chart.replaceChildren(s);
    const sum = finite.reduce((x, y) => x + y, 0);
    const loo = looAdvantages(rewards);
    const looNote = G >= 2 && !stdNorm
      ? `；RLOO 优势 × (G−1)/G = ${(((G - 1) / G) * loo[0]).toFixed(3)}，与 A₁ = ${A[0].toFixed(3)} 相同` : '';
    stats.textContent = `平均奖励 ${mean.toFixed(3)}；优势之和 ${Math.abs(sum) < 1e-12 ? '0' : sum.toFixed(3)}${looNote}。`
      + ` 按 p = ${p.toFixed(2)} 抽一组，没有学习信号的概率 p^G + (1−p)^G = ${pNoSignal(p, G).toFixed(4)}。`;
    flag.replaceChildren(allSame
      ? h('span', { class: 'flag' }, G === 1
        ? (stdNorm ? 'G = 1：无偏标准差除以 G − 1 = 0，结果是 NaN' : 'G = 1：A = r − r = 0，这一组永远没有梯度')
        : '组内奖励全相同：优势全为 0，这一组白算了')
      : h('span', { class: 'muted' }, `答对的回答被推高（蓝），答错的被压低（橙）${stdNorm ? '；除以标准差后，幅度随组内对错比例变化' : ''}。`));
  }
  render();
}
