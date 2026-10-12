/* 图：本章在 CPU 上运行 ch06/08_dpo.py（1000 步）的记录（6.8.2 节）。与原书相比只改了 HF 权重的键名、
   context_len 96、随机种子与线程数（见 verify/_ch06_dpo_run.py 的说明）。
   数据来自 data/ch06/dpo_cpu.json（verify/_ch06_dpo_run.py 生成，test_ch06.py 复核前几步）。
   记录里的 3 段样本只续写了 120 个 token，大多停在句子中间，看不出结局，所以这里不再显示；
   DPO 前后写出的完整故事见 6.8.3 节的 ch06-dposamples（H100 运行的两个 checkpoint）。 */
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
  // 每一步都画（不抽样），纵轴取对数：loss 降到 0.01 以下以后，每 18 步一次的尖峰仍然看得出来（验证 6.8-e）
  const pts = losses.map((v, i) => [i, v]);
  const chartBox = h('div');
  root.append(chartBox);
  lineChart(chartBox, {
    series: [{ name: 'DPO loss', points: pts, width: 1.5 }], xLabel: 'iteration',
    yLabel: 'DPO loss（每个 batch 8 条 pair，对数刻度）', height: 230, logY: true, yMin: 0.001, yMax: 1,
    refLines: [{ y: Math.LN2, label: 'ln 2' }], table: false,
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
    h('div', {}, `留出 pair（${ho.n} 条）隐式奖励准确率 `, h('b', {}, `${hoK}/${ho.n}`), `，95% 区间 [${lo.toFixed(2)}, ${hi.toFixed(2)}]`),
    h('div', {}, 'chosen 的 log π − log π_ref 平均 ', h('b', {}, `${tp.chosen_shift_nats >= 0 ? '+' : ''}${tp.chosen_shift_nats.toFixed(1)}`), ' nats'),
    h('div', {}, 'rejected 的 log π − log π_ref 平均 ', h('b', {}, tp.rejected_shift_nats.toFixed(1)), ' nats'),
    h('div', {}, `留出 TinyStories 文本的 loss：`, h('b', {}, rec.post.lm_loss_pretrain.toFixed(4)), ' → ', h('b', {}, rec.post.lm_loss_dpo.toFixed(4))),
    h('div', {}, `CPU 训练 ${(pv.train_seconds / 60).toFixed(0)} 分钟（${pv.threads} 线程，torch ${pv.torch}）`)));
}
