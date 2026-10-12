/* 图：H100 运行 storybot-h100 的两个 checkpoint（DPO 之前与之后）用同样的开头、同样的种子各写一遍故事（6.8.3 节）。
   数据来自 data/ch06/h100_dpo.json（verify/_ch06_h100_dpo.py 生成，verify/test_ch06.py::test_h100_dpo_record_reproduces 复核）。
   纯函数 openings()、storyMeta()、shortLabel() 由 tests/js/ch06-dposamples.test.mjs 检查。 */
import { h } from './_lib.js';

/** 按开头分组，保持记录里的顺序：[{ prompt, kind, rows: [{ seed, pretrain, dpo }] }]。 */
export function openings(rec) {
  const out = [];
  for (const row of rec.stories) {
    let group = out.find((g) => g.prompt === row.prompt);
    if (!group) {
      group = { prompt: row.prompt, kind: row.kind, rows: [] };
      out.push(group);
    }
    group.rows.push(row);
  }
  return out;
}

/** 故事下方的一行说明：新写了多少个 token，是否写到了 <|endoftext|>。 */
export function storyMeta(story) {
  return story.ended
    ? `新写了 ${story.new_tokens} 个 token，写到 <|endoftext|> 自然结束`
    : `写满了上限 ${story.max_new_tokens} 个 token，没有写完`;
}

/** 下拉框里的短标签：留出 pair 的开头取 “there was” 之后的部分。 */
export function shortLabel(prompt) {
  const m = prompt.match(/there was (.*?)[.!]?\s*$/);
  return m ? m[1] : prompt;
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch06-ds .ctrl { display: flex; flex-wrap: wrap; gap: .4rem 1rem; align-items: center; font-size: .88rem; color: var(--ink-2); }
.w-ch06-ds .ctrl label { display: flex; flex-wrap: wrap; gap: .3rem; align-items: center; min-width: 0; max-width: 100%; }
.w-ch06-ds select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .15rem .3rem; max-width: 100%; min-width: 0; }
.w-ch06-ds .seeds { display: flex; flex-wrap: wrap; gap: .3rem; align-items: center; }
.w-ch06-ds .seeds button[aria-pressed="true"] { background: var(--ai-2); border-color: var(--ai); color: var(--ink); }
.w-ch06-ds .pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: .6rem; margin-top: .6rem; }
.w-ch06-ds .story { font: 14.5px/1.65 var(--serif); background: var(--paper-2); border-radius: 4px; padding: .6rem .8rem; white-space: pre-wrap; overflow-wrap: anywhere; }
.w-ch06-ds .story h4 { margin: 0 0 .3rem; font-size: .85rem; color: var(--ink-2); font-weight: 600; white-space: normal; }
.w-ch06-ds .story .prompt { color: var(--ink-3); }
.w-ch06-ds .story .meta { display: block; margin-top: .45rem; font-size: .8rem; line-height: 1.4; color: var(--ink-2); white-space: normal; }
`));
}

export async function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch06-ds');
  const wait = h('p', { class: 'w-label' }, '正在读取记录……');
  root.append(wait);
  let rec;
  try {
    const r = await fetch(config.src || 'data/ch06/h100_dpo.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    rec = await r.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `记录文件读取失败（${err.message}）。运行 verify/_ch06_h100_dpo.py 可以重新生成它。`));
    return;
  } finally {
    wait.remove();
  }

  const groups = openings(rec);
  let gi = 0;
  let si = 0;
  const select = h('select', { 'aria-label': '故事的开头' }, groups.map((g, i) =>
    h('option', { value: String(i) }, g.kind === 'heldout' ? `留出开头：${shortLabel(g.prompt)}` : g.prompt)));
  const seeds = h('div', { class: 'seeds', role: 'group', 'aria-label': '选择随机种子' });
  const pair = h('div', { class: 'pair', 'aria-live': 'polite' });
  root.append(h('div', { class: 'ctrl' }, h('label', {}, '开头', select), seeds), pair);

  const panel = (title, prompt, story) => h('div', { class: 'story' }, h('h4', {}, title),
    h('span', { class: 'prompt' }, prompt), story.text, h('span', { class: 'meta' }, storyMeta(story)));

  function render() {
    const group = groups[gi];
    si = Math.min(si, group.rows.length - 1);
    seeds.replaceChildren(h('span', {}, '种子'), ...group.rows.map((row, i) => h('button', {
      type: 'button', 'aria-pressed': String(i === si), onclick: () => { si = i; render(); },
    }, String(row.seed))));
    const row = group.rows[si];
    pair.replaceChildren(panel('DPO 之前（预训练模型）', group.prompt, row.pretrain), panel('DPO 之后', group.prompt, row.dpo));
  }
  select.addEventListener('change', () => { gi = +select.value; si = 0; render(); });
  render();
}
