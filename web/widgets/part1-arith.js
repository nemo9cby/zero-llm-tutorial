/* 图：81 道个位数加法，SFT 与强化学习（GRPO）之后各答对了哪些。格子里是模型的回答。
   Answers come from data/part1/preview.json (greedy decoding, scored with ch03/09_grpo.py's calculate_reward,
   re-checked on CPU by verify/test_part1.py).
   config: { mode: "diff" }   // "sft" | "grpo" | "diff" */
import { h } from './_lib.js';
import { seriesVars } from './_plot.js';

const CSS = `
.w-part1-arith .pick { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .7rem; }
.w-part1-arith .pick button[aria-pressed="true"] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-part1-arith .grid { display: grid; grid-template-columns: 1.7rem repeat(9, minmax(0, 1fr)); gap: 2px;
  max-width: 25rem; font: 12px/1 var(--mono); font-variant-numeric: tabular-nums; }
.w-part1-arith .hd { display: grid; place-items: center; color: var(--ink-3); height: 1.6rem; }
.w-part1-arith .cell { display: grid; place-items: center; align-content: center; height: 2.1rem; border-radius: 3px; color: var(--ink); }
.w-part1-arith .cell:focus { outline: 2px solid var(--ink); outline-offset: 1px; }
.w-part1-arith .cell .was { font-size: 9px; line-height: 1; color: var(--ink-2); }
.w-part1-arith .cell .now { line-height: 1.1; }
.w-part1-arith .legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; margin: .7rem 0 .2rem; font-size: .82rem; color: var(--ink-2); }
.w-part1-arith .legend span::before { content: ''; display: inline-block; width: .8rem; height: .8rem; border-radius: 2px;
  margin-right: .35rem; vertical-align: -.1rem; background: var(--c); box-shadow: inset 0 0 0 1px var(--ink-3); }
.w-part1-arith .hint { font-size: .8rem; color: var(--ink-3); margin: .1rem 0 0; }
.w-part1-arith .sum { font-size: .9rem; margin: .4rem 0 0; }`;

const tint = (c, pct) => `color-mix(in oklab, ${c} ${pct}%, var(--paper))`;

export async function mount(root, config) {
  root.append(h('style', {}, CSS));
  const box = h('div', { class: 'w-part1-arith' });
  root.append(box);
  seriesVars(box);                                  // the cell colours follow theme switches
  let data;
  try {
    const r = await fetch('data/part1/preview.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    box.append(h('p', { class: 'w-label' }, `无法读取 data/part1/preview.json（${err.message}）。`));
    return;
  }
  const sft = data.models.sft.addition_greedy, grpo = data.models.grpo.addition_greedy;
  const ok = (ans, i, j) => {   // calculate_reward: the last integer in the reply equals i + j
    const nums = ans.match(/-?\d+/g);
    return !!nums && Number(nums[nums.length - 1]) === i + j;
  };
  const C = {   // blue = correct / fixed, red = wrong, amber = broken by RL (distinct hues in both themes)
    right: tint('var(--series-0)', 28), wrong: tint('var(--series-7)', 55),
    fixed: tint('var(--series-0)', 62), broke: tint('var(--series-3)', 75), same: 'var(--paper-2)',
  };
  let mode = config.mode || 'diff';
  let active = 0;                                  // roving tabindex: the one answer cell that takes Tab focus
  const pick = h('div', { class: 'pick', role: 'group', 'aria-label': '选择要看的模型' });
  const grid = h('div', { class: 'grid', role: 'table',
    'aria-label': 'i + j 的回答，行是 i，列是 j；用方向键在格子之间移动' });
  const legend = h('div', { class: 'legend' });
  const hint = h('p', { class: 'hint' });
  const sum = h('p', { class: 'sum', 'aria-live': 'polite' });
  box.append(pick, grid, legend, hint, sum);
  grid.addEventListener('keydown', (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 9, ArrowUp: -9 }[e.key];
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? 80 : step == null ? null : active + step;
    if (next == null) return;
    e.preventDefault();
    if (next < 0 || next > 80) return;
    const cells = grid.querySelectorAll('.cell');
    cells[active].tabIndex = -1;
    active = next;
    cells[active].tabIndex = 0;
    cells[active].focus();
  });

  const render = () => {
    pick.replaceChildren(...[['sft', 'SFT 之后'], ['grpo', '强化学习之后'], ['diff', '前后对比']]
      .map(([k, label]) => h('button', { type: 'button', 'aria-pressed': String(k === mode),
        onclick: () => { mode = k; render(); } }, label)));
    const cells = [h('div', { class: 'hd' }, '+')];
    for (let j = 1; j <= 9; j++) cells.push(h('div', { class: 'hd' }, j));
    const count = { sftOk: 0, grpoOk: 0, fixed: 0, broke: 0 };
    for (let i = 1; i <= 9; i++) {
      cells.push(h('div', { class: 'hd' }, i));
      for (let j = 1; j <= 9; j++) {
        const key = `${i}+${j}`, a = sft[key], b = grpo[key];
        const okA = ok(a, i, j), okB = ok(b, i, j);
        count.sftOk += okA; count.grpoOk += okB; count.fixed += !okA && okB; count.broke += okA && !okB;
        let bg, shown;
        if (mode === 'sft') { bg = okA ? C.right : C.wrong; shown = a; }
        else if (mode === 'grpo') { bg = okB ? C.right : C.wrong; shown = b; }
        else {
          bg = okA && okB ? C.same : !okA && !okB ? C.wrong : okB ? C.fixed : C.broke;
          // a cell that RL fixed or broke shows both answers: SFT's small on top, RL's below
          shown = okA === okB ? b : [h('span', { class: 'was' }, a), h('span', { class: 'now' }, b)];
        }
        const index = (i - 1) * 9 + (j - 1);
        cells.push(h('div', {
          class: 'cell', role: 'cell', tabindex: index === active ? 0 : -1, style: { background: bg },
          title: `${key}=${i + j}　SFT 答 ${a}，强化学习后答 ${b}`,
          'aria-label': `${key} 等于 ${i + j}；SFT 答 ${a}${okA ? '（对）' : '（错）'}，强化学习后答 ${b}${okB ? '（对）' : '（错）'}`,
          onfocus: () => { active = index; },
        }, shown));
      }
    }
    grid.replaceChildren(...cells);
    const items = mode === 'diff'
      ? [['都答对', C.same], ['被强化学习修好', C.fixed], ['被强化学习弄坏', C.broke], ['都答错', C.wrong]]
      : [['答对', C.right], ['答错', C.wrong]];
    legend.replaceChildren(...items.map(([t, c]) => {
      const s = h('span', {}, t);
      s.style.setProperty('--c', c);                 // custom properties need setProperty
      return s;
    }));
    hint.textContent = mode === 'diff'
      ? '被修好或弄坏的格子里，上面的小字是 SFT 的回答，下面是强化学习之后的回答；其余格子只显示强化学习之后的回答。'
      : '';
    const pct = (n) => (100 * n / 81).toFixed(1);
    sum.textContent = mode === 'sft' ? `SFT 之后答对 ${count.sftOk} / 81 道（${pct(count.sftOk)}%）。`
      : mode === 'grpo' ? `强化学习之后答对 ${count.grpoOk} / 81 道（${pct(count.grpoOk)}%）。`
        : `答对数从 ${count.sftOk} 升到 ${count.grpoOk}：修好了 ${count.fixed} 道，同时弄坏了 ${count.broke} 道。`;
  };
  render();
}
