/* P.8 Three checkpoints, one prompt: pretrain continues text, SFT follows the template, GRPO gets the sums right.
   Then the 81 single-digit additions of 09_grpo.py (and two sets GRPO never trained on). Each set first shows the
   answers PyTorch gave with the same int8 weights (web/data/playground/addition.json); a button re-asks every
   problem in this browser. Browser results are kept per set, so switching sets back and forth loses nothing.
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
  const tested = new Map();                      // set key -> {answers: {sft, grpo}, summary}: answers computed here
  const setSwitch = h('div', { class: 'w-pg-switch', role: 'group', 'aria-label': '选择题目集' }, SETS.map((s) => h('button', {
    type: 'button', 'aria-pressed': String(s.key === setKey), 'data-key': s.key, onclick: () => { setKey = s.key; drawSet(); },
  }, s.label)));
  const runBtn = h('button', { type: 'button', class: 'w-pg-btn primary' }, '在浏览器里逐题测试 SFT 与 GRPO');
  const addStatus = statusLine();
  const pair = h('div', { class: 'w-pg-pair' });
  const addNote = h('div', { class: 'w-label', 'aria-live': 'polite' });
  box.replaceChildren(
    h('div', { class: 'w-row' }, h('label', { class: 'w-pg-field' }, tpl, '套用 Alpaca 模板')), presets,
    input, h('div', { class: 'w-row' }, askBtn), status.el, cards,
    h('div', { class: 'w-label', style: 'margin-top:1rem' },
      '加法题：SFT 与 GRPO 逐题比较。绿色 = 答对，格子里是模型的回答，∅ = 空回答（模型直接输出了 <|endoftext|>）'),
    h('div', { class: 'w-row' }, setSwitch, runBtn), addStatus.el, pair, addNote,
  );
  let busy = false;
  const setBusy = (b) => {
    busy = b; askBtn.disabled = b; runBtn.disabled = b;
    for (const btn of setSwitch.children) btn.disabled = b;
  };
  const labelOf = (n) => entries.find((c) => c.name === n)?.label || n;

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

  const block = (n, set, answers, where) => h('div', {},
    h('div', { class: 'w-label' }, `${labelOf(n)}（${where}）`), grid(set, answers, n));

  // Show the current set: the answers computed in this browser if it was tested here, else PyTorch's record.
  async function drawSet() {
    for (const b of setSwitch.children) b.setAttribute('aria-pressed', String(b.dataset.key === setKey));
    try {
      const rec = await loadRecord();
      const set = currentSet(rec);
      const done = tested.get(setKey);
      if (done) {
        pair.replaceChildren(...['sft', 'grpo'].map((n) => block(n, set, done.answers[n], '在这个浏览器里算出的回答')));
        addNote.textContent = done.summary;
        addStatus.say('这一组已经在浏览器里测过了。');
        return;
      }
      pair.replaceChildren(...['sft', 'grpo'].map((n) => {
        const r = rec.models[`ours-${n}-q8`].results[setKey];
        return block(n, set, r.responses.map((text, i) => ({ text, ok: r.correct[i] > 0 })), 'PyTorch 记录的回答');
      }));
      addNote.textContent = '';
      addStatus.say('这一组还没在浏览器里测过；格子里先显示 PyTorch 用同一份 int8 权重的回答（ch03 的模板，greedy 解码）。'
        + '点“在浏览器里逐题测试”，可以在这里逐题重算。');
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
      const all = {};
      for (const n of ['sft', 'grpo']) {
        await ensureModel(n, addStatus);
        const answers = [];
        const holder = block(n, set, answers, '在这个浏览器里算出的回答');
        blocks.push(holder);
        pair.replaceChildren(...blocks, ...(blocks.length < 2 ? [h('div')] : []));
        const handle = await engine().call('addition', n, { prompts, maxNew: rec.max_new_tokens }, (ev) => {
          const text = tok.decode(ev.ids);
          answers[ev.index] = { text, ok: reward(text, set.answers[ev.index]), ids: ev.ids };
          if (ev.index % 9 === 8 || ev.index === prompts.length - 1) holder.lastChild.replaceWith(grid(set, answers, n));
          addStatus.say(`${labelOf(n)}：${ev.index + 1} / ${prompts.length}`);
        });
        await handle.promise;
        all[n] = answers;
        const ref = rec.models[`ours-${n}-q8`].results[set.key];
        const same = answers.filter((a, i) => a && a.text === ref.responses[i]).length;
        const right = answers.filter((a) => a?.ok).length;
        summary.push(`${labelOf(n)} 答对 ${right}/${prompts.length}（PyTorch 用同一份 int8 权重：${ref.n_correct}/${prompts.length}；逐题回答相同 ${same}/${prompts.length}）`);
      }
      tested.set(set.key, { answers: all, summary: `${summary.join('；')}。` });
      addNote.textContent = tested.get(set.key).summary;
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
  drawSet();
}
