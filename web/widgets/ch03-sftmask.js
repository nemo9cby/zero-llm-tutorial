/* 图：SFT 样本的输入、标签与掩码。数据是原书分词器对 tiny_codes_sft.json 的分词结果
   （data/ch03/sft_tokens.json，test_ch03.py 已与 SFTDataset 逐位核对）。
   每一列 = 位置 t：上面是输入 ids[t]，下面是要预测的 labels[t]（−100 表示不算 loss）。
   config: { src, vocab, samples: [instruction, ...], context_len: 256 } */
import { h } from './_lib.js';
import { injectStyle, loadJSON, segmented, showTok } from './ch03-common.js';

export async function mount(root, config) {
  injectStyle();
  const [data, vocab] = await Promise.all([loadJSON(root, config.src), loadJSON(root, config.vocab)]);
  if (!data || !vocab) return;
  const tokens = vocab.tokens;
  const C = config.context_len || data.context_len || 256;
  const picks = (config.samples || ['3+4=']).map((s) => data.instructions.indexOf(s)).filter((k) => k >= 0);
  if (!picks.length) { root.append(h('p', { class: 'w-label' }, '找不到指定的样本。')); return; }

  root.classList.add('w-ch03');
  root.append(h('style', {}, `
.w-ch03-sft .cells { display: flex; flex-wrap: wrap; gap: 3px; margin: .5rem 0; }
.w-ch03-sft .cell { display: inline-flex; flex-direction: column; align-items: stretch; min-width: 1.9em; border-radius: 3px; overflow: hidden;
  font: 12.5px/1.35 var(--mono); border: 1px solid var(--rule); }
.w-ch03-sft .cell span { padding: 1px 4px; white-space: pre; text-align: center; }
.w-ch03-sft .cell .in { background: var(--paper-2); }
.w-ch03-sft .cell .out { border-top: 1px dashed var(--rule); }
.w-ch03-sft .cell.prompt .out { color: var(--ink-3); }
.w-ch03-sft .cell.resp { border-color: var(--moss); }
.w-ch03-sft .cell.resp .out { background: color-mix(in srgb, var(--moss) 22%, var(--paper)); color: var(--ink); font-weight: 600; }
.w-ch03-sft .cell.pad { opacity: .45; }
.w-ch03-sft .meter { display: flex; height: 12px; border-radius: 2px; overflow: hidden; margin: .3rem 0 .2rem; border: 1px solid var(--rule); }
.w-ch03-sft .meter i { display: block; height: 100%; }
.w-ch03-sft .key { font-size: .8rem; color: var(--ink-3); }
.w-ch03-sft .key b { display: inline-block; width: 10px; height: 10px; margin: 0 .25rem 0 .7rem; vertical-align: middle; border-radius: 2px; }
.w-ch03-sft pre.src { font: 12.5px/1.5 var(--mono); background: var(--paper-2); padding: .5rem .7rem; border-radius: 4px; white-space: pre-wrap; margin: .4rem 0; }
`));
  root.classList.add('w-ch03-sft');

  const seg = segmented(picks.map((k) => data.instructions[k]), 0, (i) => render(picks[i]), '选择 SFT 样本');
  const src = h('pre', { class: 'src' });
  const stats = h('div', { class: 'stat', 'aria-live': 'polite' });
  const meter = h('div', { class: 'meter', 'aria-hidden': 'true' });
  const key = h('div', { class: 'key' }, '256 个位置：',
    h('b', { style: 'background:var(--ink-3)' }), 'prompt（标签 −100）',
    h('b', { style: 'background:var(--moss)' }), '回答（算 loss）',
    h('b', { style: 'background:var(--paper-3)' }), 'padding（id 0，标签 −100）');
  const cells = h('div', { class: 'cells', role: 'list', 'aria-label': '每个位置的输入与标签' });
  root.append(seg, src, stats, meter, key, cells);

  function render(k) {
    const pIds = data.prompt_ids[k], rIds = data.response_ids[k];
    // 与 SFTDataset._create_sample 相同：拼接、错开一位、补齐
    const ids = [...pIds, ...rIds].slice(0, -1);
    const labels = [...pIds.map(() => -100), ...rIds].slice(1);
    const nReal = ids.length, nPad = Math.max(0, C - nReal);
    const nResp = labels.filter((l) => l !== -100).length;
    const nPromptPos = nReal - nResp;
    src.textContent = `### Instruction:\n${data.instructions[k]}\n\n### Response:\n`
      + `${rIds.map((t) => tokens[t]).join('')}`;
    stats.textContent = `prompt ${pIds.length} 个 token，回答 ${rIds.length} 个（含 EOS）。错开一位后：`
      + `${nPromptPos} 个位置标签为 −100，${nResp} 个位置算 loss，补了 ${nPad} 个 padding；`
      + `带标签的位置只占 ${((nResp / C) * 100).toFixed(1)}%。`;
    meter.replaceChildren(
      h('i', { style: `width:${(nPromptPos / C) * 100}%;background:var(--ink-3)` }),
      h('i', { style: `width:${(nResp / C) * 100}%;background:var(--moss)` }),
      h('i', { style: `width:${(nPad / C) * 100}%;background:var(--paper-3)` }));
    const showPad = Math.min(nPad, 3);
    const list = [];
    for (let t = 0; t < nReal + showPad; t++) {
      const pad = t >= nReal;
      const input = pad ? 0 : ids[t];
      const lab = pad ? -100 : labels[t];
      const kind = pad ? 'pad' : (lab === -100 ? 'prompt' : 'resp');
      const outText = lab === -100 ? '−100' : showTok(tokens[lab]);
      list.push(h('span', { class: `cell ${kind}`, role: 'listitem',
        title: `位置 ${t}：输入 id ${input}，标签 ${lab}` },
      h('span', { class: 'in' }, pad ? '0' : showTok(tokens[input])),
      h('span', { class: 'out' }, outText)));
    }
    if (nPad > showPad) list.push(h('span', { class: 'cell pad', role: 'listitem' }, h('span', { class: 'in' }, `… 还有 ${nPad - showPad} 个`), h('span', { class: 'out' }, '−100')));
    cells.replaceChildren(...list);
  }
  render(picks[0]);
}
