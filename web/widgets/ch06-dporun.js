/* 图：本章在 CPU 上原样运行 ch06/08_dpo.py（1000 步）的记录（6.8.2、6.8.3 节）。
   数据来自 data/ch06/dpo_cpu.json（verify/_ch06_dpo_run.py 生成，test_ch06.py 复核前几步）。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { wilson } from './ch06-wilson.js';

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch06-dr .facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: .15rem 1.2rem; font-size: .86rem; color: var(--ink-2); margin: .5rem 0; }
.w-ch06-dr .facts b { color: var(--ink); font-variant-numeric: tabular-nums; }
.w-ch06-dr .seg { display: flex; flex-wrap: wrap; gap: .3rem; margin: .6rem 0 .3rem; align-items: center; }
.w-ch06-dr .seg button[aria-pressed="true"] { background: var(--ai-2); border-color: var(--ai); color: var(--ink); }
.w-ch06-dr .pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: .6rem; }
.w-ch06-dr .story { font: 14.5px/1.65 var(--serif); background: var(--paper-2); border-radius: 4px; padding: .6rem .8rem; white-space: pre-wrap; }
.w-ch06-dr .story h4 { margin: 0 0 .3rem; font-size: .85rem; color: var(--ink-2); font-weight: 600; }
`));
}

export async function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch06-dr');
  const wait = h('p', { class: 'w-label' }, '正在读取记录……');
  root.append(wait);
  let rec;
  try {
    const r = await fetch(config.src || 'data/ch06/dpo_cpu.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    rec = await r.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `记录文件读取失败（${err.message}）。运行 verify/_ch06_dpo_run.py 可以重新生成它。`));
    return;
  } finally {
    wait.remove();
  }
  const losses = rec.losses;
  const pts = losses.map((v, i) => [i, v]).filter(([i]) => i < 40 || i % 4 === 0 || i === losses.length - 1);
  const chartBox = h('div');
  root.append(chartBox);
  lineChart(chartBox, {
    series: [{ name: 'DPO loss', points: pts }], xLabel: 'iteration', yLabel: 'DPO loss（每个 batch 8 条 pair）',
    height: 230, yMin: 0, refLines: [{ y: Math.LN2, label: 'ln 2' }], table: false,
  });

  const tp = rec.post.train_pairs, ho = rec.post.heldout_pairs;
  const hoK = Math.round(ho.reward_acc * ho.n);
  const [lo, hi] = wilson(hoK, ho.n);
  const last100 = losses.slice(-100).reduce((a, b) => a + b, 0) / Math.min(100, losses.length);
  const pv = rec.provenance;
  root.append(h('div', { class: 'facts' },
    h('div', {}, '第 0 步 loss ', h('b', {}, losses[0].toFixed(4)), '（ln 2 = 0.6931）'),
    h('div', {}, '最后 100 步的平均 loss ', h('b', {}, last100.toFixed(4))),
    h('div', {}, `训练 pair（${tp.n} 条）隐式奖励准确率 `, h('b', {}, tp.reward_acc.toFixed(3))),
    h('div', {}, `留出 pair（${ho.n} 条）准确率 `, h('b', {}, `${hoK}/${ho.n}`), `，95% 区间 [${lo.toFixed(2)}, ${hi.toFixed(2)}]`),
    h('div', {}, 'chosen 的 log π − log π_ref 平均 ', h('b', {}, `${tp.chosen_shift_nats >= 0 ? '+' : ''}${tp.chosen_shift_nats.toFixed(1)}`), ' nats'),
    h('div', {}, 'rejected 的 log π − log π_ref 平均 ', h('b', {}, tp.rejected_shift_nats.toFixed(1)), ' nats'),
    h('div', {}, `留出 TinyStories 文本的 loss：`, h('b', {}, rec.post.lm_loss_pretrain.toFixed(4)), ' → ', h('b', {}, rec.post.lm_loss_dpo.toFixed(4))),
    h('div', {}, `CPU 训练 ${(pv.train_seconds / 60).toFixed(0)} 分钟（${pv.threads} 线程，torch ${pv.torch}）`)));

  let si = 0;
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': '选择随机种子' });
  const pair = h('div', { class: 'pair', 'aria-live': 'polite' });
  root.append(h('div', { class: 'w-label', style: 'margin-top:.6rem' },
    '同一个种子下，预训练模型与 DPO 模型从 “Once upon a time” 续写 120 个 token：'), seg, pair);
  function render() {
    seg.replaceChildren(h('span', { class: 'w-label' }, '种子'), ...rec.samples.map((s, i) => h('button', {
      type: 'button', 'aria-pressed': String(i === si), onclick: () => { si = i; render(); },
    }, String(s.seed))));
    const s = rec.samples[si];
    pair.replaceChildren(
      h('div', { class: 'story' }, h('h4', {}, '预训练模型'), s.pretrain),
      h('div', { class: 'story' }, h('h4', {}, 'DPO 之后'), s.dpo));
  }
  render();
}
