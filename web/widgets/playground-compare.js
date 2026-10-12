/* P.8 Three checkpoints, one prompt: pretrain continues text, SFT follows the template, GRPO gets the sums right.
   Then the 81 single-digit additions of 09_grpo.py (and two sets it never trained on), asked in this browser, next
   to the answers PyTorch gave with the same int8 weights (web/data/playground/addition.json).
   config: { prompt, template, maxNewTokens, set } */
import { h } from './_lib.js';
import {
  injectStyle, getIndex, getTokenizer, engine, ensureModel, statusLine, alpaca, reward, DATA,
} from './playground-common.js';

const PRESETS = ['3+4=', '7+8=', 'print hello', 'read a file line by line', "What's a stack?"];
const SETS = [
  { key: 'train', label: '训练用的 81 道' },
  { key: 'zero', label: '含 0 的 19 道' },
  { key: 'teens', label: '两位数 + 一位数 81 道' },
];

/** Lay a set of "a+b=" prompts out as a table: one row per first operand (or per "0+b" / "a+0" family),
    one column per varying operand. Returns {cols, corner, rows: [{label, cells}]}; a cell is an index into
    prompts, or -1 for an empty slot. */
export function layout(prompts) {
  const ab = prompts.map((p) => p.slice(0, -1).split('+').map(Number));
  const zeroSet = ab.some(([a]) => a === 0) && ab.some(([a, b]) => a > 0 && b === 0);
  if (zeroSet) {                                 // "0+b" for b = 0..9, then "a+0" for a = 1..9
    const cols = [...Array(10).keys()];
    const rowA = cols.map((b) => ab.findIndex(([a, x]) => a === 0 && x === b));
    const rowB = cols.map((a) => (a === 0 ? -1 : ab.findIndex(([x, b]) => x === a && b === 0)));
    return { cols: cols.map(String), rows: [{ label: '0+b', cells: rowA }, { label: 'a+0', cells: rowB }], corner: '' };
  }
  const as = [...new Set(ab.map(([a]) => a))], bs = [...new Set(ab.map(([, b]) => b))];
  return { cols: bs.map(String), corner: '',
    rows: as.map((a) => ({ label: `${a}+`, cells: bs.map((b) => ab.findIndex(([x, y]) => x === a && y === b)) })) };
}

let styled = false;
function style() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-pg-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: .6rem; margin: .5rem 0; }
.w-pg-card { border: 1px solid var(--rule); border-radius: 4px; padding: .5rem .6rem; background: var(--paper); min-width: 0; }
.w-pg-card h4 { margin: 0 0 .2rem; font-size: .9rem; }
.w-pg-card .w-pg-out { min-height: 3em; margin: .3rem 0 0; font-size: 13px; }
.w-pg-sum { display: grid; gap: 2px; font: 11px/1.2 var(--mono); margin: .3rem 0; }
.w-pg-sum > span { display: grid; place-items: center; min-height: 1.9em; border-radius: 2px; overflow: hidden; white-space: nowrap; }
.w-pg-sum > .hd { color: var(--ink-3); font-size: 10px; }
.w-pg-sum .ok { background: color-mix(in srgb, var(--moss) 30%, var(--paper)); color: var(--ink); }
.w-pg-sum .bad { background: var(--paper-3); color: var(--ink-3); }
.w-pg-sum .todo { background: transparent; border: 1px dashed var(--rule); }
.w-pg-pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: .8rem; }
`));
}

export async function mount(root, config = {}) {
  injectStyle();
  style();
  const cfg = { prompt: '3+4=', template: true, maxNewTokens: 48, set: 'train', ...config };
  const box = h('div', { class: 'w-pg' }, h('p', { class: 'w-pg-note' }, '正在准备……'));
  root.append(box);
  let entries, tok, record;
  try {
    [entries, tok] = await Promise.all([getIndex(), getTokenizer()]);
  } catch (err) {
    box.replaceChildren(h('p', { class: 'w-pg-error' }, `这张图没能启动：${err.message}`));
    return;
  }
  const input = h('input', { class: 'w-input', value: cfg.prompt, spellcheck: 'false', 'aria-label': '三个检查点共用的 prompt' });
  const tpl = h('input', { type: 'checkbox', checked: cfg.template });
  const askBtn = h('button', { type: 'button', class: 'w-pg-btn primary' }, '三个检查点一起回答（greedy）');
  const status = statusLine();
  const cards = h('div', { class: 'w-pg-cards' });
  const outs = new Map();
  for (const c of entries) {
    const out = h('div', { class: 'w-pg-out', 'aria-live': 'polite' });
    outs.set(c.name, out);
    cards.append(h('div', { class: 'w-pg-card' }, h('h4', {}, c.label), h('div', { class: 'w-pg-note' }, c.description), out));
  }
  const presets = h('div', { class: 'w-pg-presets' }, '示例：', PRESETS.map((p) => h('button', { type: 'button', class: 'w-pg-btn',
    onclick: () => { input.value = p; ask(); } }, p)));

  // addition
  let setKey = cfg.set;
  const setSwitch = h('div', { class: 'w-pg-switch', role: 'group', 'aria-label': '选择题目集' }, SETS.map((s) => h('button', {
    type: 'button', 'aria-pressed': String(s.key === setKey), onclick: () => { setKey = s.key; drawEmpty(); },
  }, s.label)));
  const runBtn = h('button', { type: 'button', class: 'w-pg-btn primary' }, '在浏览器里逐题测试 SFT 与 GRPO');
  const addStatus = statusLine();
  const pair = h('div', { class: 'w-pg-pair' });
  const addNote = h('div', { class: 'w-label', 'aria-live': 'polite' });
  box.replaceChildren(
    h('div', { class: 'w-row' }, h('label', { class: 'w-pg-field' }, tpl, '套用 Alpaca 模板')), presets,
    input, h('div', { class: 'w-row' }, askBtn), status.el, cards,
    h('div', { class: 'w-label', style: 'margin-top:1rem' }, '个位数加法：SFT 与 GRPO 逐题比较（绿色 = 答对，格子里是模型的回答）'),
    h('div', { class: 'w-row' }, setSwitch, runBtn), addStatus.el, pair, addNote,
  );
  let busy = false;
  const setBusy = (b) => { busy = b; askBtn.disabled = b; runBtn.disabled = b; };

  async function ask() {
    if (busy) return;
    setBusy(true);
    const ids = tok.encode(tpl.checked ? alpaca(input.value) : input.value).slice(-256);
    try {
      for (const c of entries) {
        const out = outs.get(c.name);
        out.replaceChildren(h('span', { class: 'w-pg-note' }, '……'));
        await ensureModel(c.name, status);
        status.say(`${c.label} 正在回答……`);
        const dec = tok.streamDecoder();
        out.replaceChildren();
        const handle = await engine().call('generate', c.name,
          { ids, opts: { temperature: 0, maxNewTokens: cfg.maxNewTokens, stopAtEnd: true, top: 0 } }, (ev) => {
            if (ev.token.end) out.append(h('span', { class: 'eos' }, '<|endoftext|>'));
            else out.append(dec.push(ev.token.id));
          });
        const res = await handle.promise;
        if (res.stopped === 'length') out.append(h('span', { class: 'w-pg-note' }, ' ……（到达上限）'));
      }
      status.say('三个检查点用的是同一串 token、同样的 greedy 解码，不同的只是权重。');
    } catch (err) {
      status.error(err);
    } finally {
      setBusy(false);
    }
  }

  function grid(set, answers, label) {
    const { cols, rows, corner } = layout(set.prompts);
    const g = h('div', { class: 'w-pg-sum', role: 'table', 'aria-label': label,
      style: `grid-template-columns:2.6em repeat(${cols.length}, minmax(0, 1fr))` });
    g.append(h('span', { class: 'hd', role: 'columnheader' }, corner), ...cols.map((c) => h('span', { class: 'hd', role: 'columnheader' }, c)));
    for (const row of rows) {
      g.append(h('span', { class: 'hd', role: 'rowheader' }, row.label));
      for (const i of row.cells) {
        if (i < 0) { g.append(h('span')); continue; }
        const p = set.prompts[i], a = answers?.[i];
        const cls = a == null ? 'todo' : a.ok ? 'ok' : 'bad';
        g.append(h('span', { class: cls, role: 'cell', title: `${p} → ${a == null ? '（未测）' : JSON.stringify(a.text)}`,
          'aria-label': `${p} ${a == null ? '未测' : `回答 ${a.text || '（空）'}，${a.ok ? '对' : '错'}`}` },
        a == null ? '' : (a.text.trim() || '∅')));
      }
    }
    return g;
  }

  async function loadRecord() {
    if (!record) {
      const r = await fetch(new URL('addition.json', DATA));
      if (!r.ok) throw new Error(`读取 addition.json 失败（HTTP ${r.status}）`);
      record = await r.json();
    }
    return record;
  }

  function currentSet(rec) {
    return { key: setKey, prompts: rec.sets[setKey], answers: rec.answers[setKey] };
  }

  async function drawEmpty() {
    for (const b of setSwitch.children) b.setAttribute('aria-pressed', String(b.textContent === SETS.find((s) => s.key === setKey).label));
    try {
      const rec = await loadRecord();
      const set = currentSet(rec);
      pair.replaceChildren(...['sft', 'grpo'].map((n) => h('div', {}, h('div', { class: 'w-label' }, entries.find((c) => c.name === n)?.label || n),
        grid(set, null, n))));
      addNote.textContent = '';
    } catch (err) {
      addStatus.error(err);
    }
  }

  async function runAddition() {
    if (busy) return;
    setBusy(true);
    try {
      const rec = await loadRecord();
      const set = currentSet(rec);
      const prompts = set.prompts.map((p) => tok.encode(alpaca(p)));
      const summary = [];
      const blocks = [];
      for (const n of ['sft', 'grpo']) {
        await ensureModel(n, addStatus);
        const answers = [];
        const holder = h('div', {}, h('div', { class: 'w-label' }, entries.find((c) => c.name === n)?.label || n), grid(set, answers, n));
        blocks.push(holder);
        pair.replaceChildren(...blocks, ...(blocks.length < 2 ? [h('div')] : []));
        const handle = await engine().call('addition', n, { prompts, maxNew: rec.max_new_tokens }, (ev) => {
          const text = tok.decode(ev.ids);
          answers[ev.index] = { text, ok: reward(text, set.answers[ev.index]), ids: ev.ids };
          if (ev.index % 9 === 8 || ev.index === prompts.length - 1) holder.lastChild.replaceWith(grid(set, answers, n));
          addStatus.say(`${entries.find((c) => c.name === n)?.label}：${ev.index + 1} / ${prompts.length}`);
        });
        await handle.promise;
        const ref = rec.models[`ours-${n}-q8`].results[setKey];
        const same = answers.filter((a, i) => a && a.text === ref.responses[i]).length;
        const right = answers.filter((a) => a?.ok).length;
        summary.push(`${entries.find((c) => c.name === n)?.label} 答对 ${right}/${prompts.length}（PyTorch 用同一份 int8 权重：${ref.n_correct}/${prompts.length}；逐题回答相同 ${same}/${prompts.length}）`);
      }
      addNote.textContent = `${summary.join('；')}。`;
      addStatus.say('完成。');
    } catch (err) {
      addStatus.error(err);
    } finally {
      setBusy(false);
    }
  }

  askBtn.addEventListener('click', ask);
  runBtn.addEventListener('click', runAddition);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
  status.say('点按钮后，三个检查点依次下载（各 11.3 MB）并回答同一个 prompt。');
  addStatus.say('每道题都按 ch03 的模板提问，greedy 解码，直到模型输出 <|endoftext|>。');
  drawEmpty();
}
