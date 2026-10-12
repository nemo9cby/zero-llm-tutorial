/* P.3 What the model actually reads: the Alpaca template of ch03 wrapped around an instruction, as CodeBot's BPE
   tokens. No model is needed here, only the tokenizer (web/widgets/_bpe.js, id-for-id equal to codebot.tokenizer).
   config: { instruction, template } */
import { h, chip } from './_lib.js';
import { injectStyle, getTokenizer, alpaca } from './playground-common.js';

const PREFIX = '### Instruction:\n';

export async function mount(root, config = {}) {
  injectStyle();
  const cfg = { instruction: '3+4=', template: true, ...config };
  const box = h('div', { class: 'w-pg' }, h('p', { class: 'w-pg-note' }, '正在加载分词器……'));
  root.append(box);
  let tok;
  try {
    tok = await getTokenizer();
  } catch (err) {
    box.replaceChildren(h('p', { class: 'w-pg-error' }, `分词器没能载入：${err.message}`));
    return;
  }
  const input = h('input', { class: 'w-input', value: cfg.instruction, spellcheck: 'false', 'aria-label': '指令' });
  const tpl = h('input', { type: 'checkbox', checked: cfg.template });
  const chips = h('div', { class: 'tokens' });
  const summary = h('div', { class: 'w-label', 'aria-live': 'polite' });
  const legend = h('div', { class: 'w-pg-note' }, '颜色较淡的是模板本身的 token，颜色较深的是指令的 token；小字是 token id。');

  function render() {
    const instr = input.value;
    const text = tpl.checked ? alpaca(instr) : instr;
    const lo = tpl.checked ? PREFIX.length : 0, hi = lo + instr.length;
    const nodes = [];
    let off = 0, k = 0, nInstr = 0, nAll = 0;
    for (const piece of tok.encodeChunks(text)) {
      const a = off, b = off + piece.text.length;
      off = b;
      const inInstr = a < hi && b > lo;            // a pre-token overlapping the instruction counts as instruction
      for (const id of piece.ids) {
        const c = chip(tok.decode([id]), id, inInstr ? k++ : 0, `id ${id}${inInstr ? '' : '（模板）'}`);
        if (!inInstr) c.style.opacity = '0.55';
        nodes.push(c);
        nAll += 1;
        if (inInstr) nInstr += 1;
      }
    }
    chips.replaceChildren(...nodes);
    summary.textContent = tpl.checked
      ? `共 ${nAll} 个 token：指令 ${nInstr} 个，模板 ${nAll - nInstr} 个。模型从这串 token 之后开始写回答。`
      : `共 ${nAll} 个 token（没有模板）。`;
  }

  input.addEventListener('input', render);
  tpl.addEventListener('change', render);
  box.replaceChildren(h('div', { class: 'w-row' }, h('label', { class: 'w-pg-field' }, tpl, '套用 Alpaca 模板')),
    input, chips, summary, legend);
  render();
}
