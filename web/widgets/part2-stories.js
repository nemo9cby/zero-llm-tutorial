/* 图：同一个随机种子，三个训练阶段的 StoryBot 各写一个故事。
   Stories come from data/part2/preview.json: the official checkpoints at iteration 500 / 5,000 / 40,000,
   sampled on CPU with storybot/utils.generate (re-checked by verify/test_part2.py).
   config: { iter: 40000, seed: 1 } */
import { h, fmt } from './_lib.js';

const CSS = `
.w-part2-stories .pick { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .5rem; }
.w-part2-stories .pick button { text-align: left; line-height: 1.35; }
.w-part2-stories .pick button small { display: block; font-size: .75rem; color: inherit; opacity: .8; }
.w-part2-stories .pick button[aria-pressed="true"] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-part2-stories .story { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--paper-2); border-radius: 6px;
  padding: .8rem 1rem; font: 15px/1.7 var(--serif); margin: .6rem 0 .3rem; min-height: 9rem; }`;

const seen = (n) => (n >= 1e8 ? `${(n / 1e8).toFixed(2)} 亿` : `${fmt(n / 1e4)} 万`);

export async function mount(root, config) {
  root.append(h('style', {}, CSS));
  const box = h('div', { class: 'w-part2-stories' });
  root.append(box);
  let data;
  try {
    const r = await fetch('data/part2/preview.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    box.append(h('p', { class: 'w-label' }, `无法读取 data/part2/preview.json（${err.message}）。`));
    return;
  }
  const cks = data.checkpoints;
  let ck = cks.find((c) => c.iters === config.iter) || cks[cks.length - 1];
  let seed = config.seed ?? cks[0].samples[0].seed;
  const ckRow = h('div', { class: 'pick', role: 'group', 'aria-label': '选择训练阶段' });
  const seedRow = h('div', { class: 'pick', role: 'group', 'aria-label': '选择随机种子' });
  const story = h('div', { class: 'story', 'aria-live': 'polite' });
  const meta = h('p', { class: 'w-label' });
  box.append(ckRow, seedRow, story, meta);

  const render = () => {
    ckRow.replaceChildren(...cks.map((c) => h('button', {
      type: 'button', 'aria-pressed': String(c === ck), onclick: () => { ck = c; render(); },
    }, `第 ${fmt(c.iters)} 步`, h('small', {}, `看过 ${seen(c.tokens_seen)} token · loss ${c.val_loss_subset.toFixed(2)}`))));
    seedRow.replaceChildren(...ck.samples.map((s, k) => h('button', {
      type: 'button', 'aria-pressed': String(s.seed === seed), onclick: () => { seed = s.seed; render(); },
    }, `样本 ${k + 1}（种子 ${s.seed}）`)));
    const s = ck.samples.find((x) => x.seed === seed) || ck.samples[0];
    story.textContent = s.text.replace(/^\n/, '');
    meta.textContent = `${ck.file}，${s.tokens} 个 token。prompt 是 <|endoftext|>（“从头写一个新故事”），temperature 1.0；`
      + `loss 是验证集上 32 个随机窗口的平均交叉熵。同一个种子意味着采样用的随机数完全相同，差别只来自训练。`;
  };
  render();
}
