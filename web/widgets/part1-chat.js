/* 图：和官方 CodeBot 检查点对话（贪心解码的录制结果）。
   Replies come from data/part1/preview.json, regenerated on CPU by verify/test_part1.py.
   config: { model: "grpo" } */
import { h } from './_lib.js';

const CSS = `
.w-part1-chat .pick { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .7rem; }
.w-part1-chat .pick button[aria-pressed="true"] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-part1-chat .log { font: 13.5px/1.6 var(--mono); background: var(--paper-2); border-radius: 6px; padding: .7rem .9rem; }
.w-part1-chat .turn { margin: 0 0 .7rem; }
.w-part1-chat .turn:last-child { margin-bottom: 0; }
.w-part1-chat .who { color: var(--ink-3); }
.w-part1-chat .you { color: var(--ai); }
.w-part1-chat pre { margin: .1rem 0 0 1.5rem; white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }
.w-part1-chat .mark { font-family: var(--serif); font-size: .8rem; margin-left: .6rem; color: var(--ink-2); }
.w-part1-chat .meta { margin-top: .5rem; }`;

export async function mount(root, config) {
  root.append(h('style', {}, CSS));
  const box = h('div', { class: 'w-part1-chat' });
  root.append(box);
  let data;
  try {
    const r = await fetch('data/part1/preview.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    box.append(h('p', { class: 'w-label' }, `无法读取 data/part1/preview.json（${err.message}）。`));
    return;
  }
  let which = config.model || 'grpo';
  const pick = h('div', { class: 'pick', role: 'group', 'aria-label': '选择检查点' });
  const log = h('div', { class: 'log', 'aria-live': 'polite' });
  const prov = data._provenance;
  const meta = h('p', { class: 'w-label meta' });
  box.append(pick, log, meta);

  const MAX_LINES = 8;
  const render = () => {
    pick.replaceChildren(...[['pretrain', '只做了预训练'], ['sft', 'SFT 之后'], ['grpo', '强化学习之后']]
      .map(([k, label]) => h('button', { type: 'button', 'aria-pressed': String(k === which),
        onclick: () => { which = k; render(); } }, `${label}（model_${k}.pt）`)));
    const m = data.models[which];
    log.replaceChildren(...data.prompts.map((p, i) => {
      const reply = m.greedy[i];
      const sum = p.match(/^(\d+)\+(\d+)=$/);
      const mark = !reply ? '（一开口就输出了结束记号，什么也没说）'
        : sum ? (Number(reply) === Number(sum[1]) + Number(sum[2])
          ? '✓ 答对了' : `✗ 正确答案是 ${Number(sum[1]) + Number(sum[2])}`) : '';
      const lines = reply.split('\n');
      const shown = lines.length > MAX_LINES
        ? `${lines.slice(0, MAX_LINES).join('\n')}\n…（共 ${lines.length} 行，后面省略）` : reply;
      return h('div', { class: 'turn' },
        h('div', {}, h('span', { class: 'who you' }, 'You: '), p),
        h('div', {}, h('span', { class: 'who' }, 'Bot:'), mark && h('span', { class: 'mark' }, mark)),
        reply ? h('pre', {}, shown) : null);
    }));
    meta.textContent = `贪心解码（temperature 0），每个回答最多 200 个 token；检查点 sha256 ${m.sha256.slice(0, 12)}…；`
      + `${prov.date} 在 CPU 上用 PyTorch ${prov.torch} 生成。原书的 04_chat.py 用 temperature 1.0 采样，所以你自己运行时回答会有变化。`;
  };
  render();
}
