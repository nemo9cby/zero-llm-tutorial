/* 图：官方 StoryBot checkpoint 的留出 loss 与同一种子下写出的故事（6.6.2 节）。
   数据来自记录运行 data/ch06/checkpoints.json（verify/_ch06_ckpt_record.py 生成，test_ch06.py 复核）。 */
import { h } from './_lib.js';
import { barChart } from './_plot.js';

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch06-ck .seg { display: flex; flex-wrap: wrap; gap: .3rem; margin: .4rem 0; }
.w-ch06-ck .seg button[aria-pressed="true"] { background: var(--ai-2); border-color: var(--ai); color: var(--ink); }
.w-ch06-ck .story { font: 15px/1.7 var(--serif); background: var(--paper-2); border-radius: 4px; padding: .7rem .9rem; margin: .4rem 0; white-space: pre-wrap; }
.w-ch06-ck .meta { font-size: .85rem; color: var(--ink-2); }
.w-ch06-ck .meta b { color: var(--ink); font-variant-numeric: tabular-nums; }
`));
}

export async function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch06-ck');
  const wait = h('p', { class: 'w-label' }, '正在读取记录……');
  root.append(wait);
  let rec;
  try {
    const r = await fetch(config.src || 'data/ch06/checkpoints.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    rec = await r.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `记录文件读取失败（${err.message}）。运行 verify/_ch06_ckpt_record.py 可以重新生成它。`));
    return;
  } finally {
    wait.remove();
  }
  const rows = rec.checkpoints;
  const label = (c) => (c.iter === 0 ? '随机初始化' : `第 ${c.iter.toLocaleString('en-US')} 步`);
  const chartBox = h('div');
  root.append(h('div', { class: 'w-label' }, `留出 loss（nats/token，${rec.heldout.tokens.toLocaleString('en-US')} 个 token）`), chartBox);
  barChart(chartBox, {
    bars: rows.map((c) => ({ label: label(c), value: +c.val_loss.toFixed(3) })),
    yLabel: 'loss', height: 200, yFormat: (v) => v.toFixed(2),
  });

  const withSamples = rows.filter((c) => c.samples);
  let ci = withSamples.length - 1;
  let seedIdx = rec.sampling.seeds.length - 1;
  const ckSeg = h('div', { class: 'seg', role: 'group', 'aria-label': '选择 checkpoint' });
  const seedSeg = h('div', { class: 'seg', role: 'group', 'aria-label': '选择随机种子' });
  const meta = h('div', { class: 'meta' });
  const story = h('div', { class: 'story', 'aria-live': 'polite' });
  root.append(ckSeg, seedSeg, meta, story);

  function render() {
    ckSeg.replaceChildren(...withSamples.map((c, i) => h('button', {
      type: 'button', 'aria-pressed': String(i === ci), onclick: () => { ci = i; render(); },
    }, label(c))));
    seedSeg.replaceChildren(h('span', { class: 'w-label', style: 'align-self:center' }, '种子'),
      ...rec.sampling.seeds.map((s, i) => h('button', {
        type: 'button', 'aria-pressed': String(i === seedIdx), onclick: () => { seedIdx = i; render(); },
      }, String(s))));
    const c = withSamples[ci];
    meta.replaceChildren(label(c), '：读过 ', h('b', {}, `${(c.tokens_seen / 1e6).toFixed(1)}M`), ' 个 token，留出 loss ',
      h('b', {}, c.val_loss.toFixed(3)), '，困惑度 ', h('b', {}, c.ppl.toFixed(2)),
      `；prompt 为 <|endoftext|>，温度 ${rec.sampling.temperature}，最多 ${rec.sampling.max_new_tokens} 个 token`);
    story.textContent = c.samples[seedIdx];
  }
  render();
}
