/* Replay of the recorded runs in data/train-in-browser/runs.json (written by
   tests/js/fixtures/trainer/record_runs.mjs): loss curves, the samples at every 100 steps, provenance.
   config: { view: 'default' | 'overfit' | 'lr', views: ['default', 'overfit', 'lr'] } */
import { h, fmt } from './_lib.js';
import { lineChart } from './_plot.js';

const VIEWS = {
  default: '正常训练 2,000 步',
  overfit: '过拟合一个 batch',
  lr: '四种学习率',
};
const LR_RUNS = ['lr3e-4', 'lr1e-3', 'default', 'lr1e-2'];
const BUCKET = 10; // the training curve shows the mean loss of every 10 steps

const EOT = '<|endoftext|>';
/** The prompt in bold, then the sampled text with every <|endoftext|> shown as a small marker. */
function sampleNodes(prompt, text) {
  const nodes = [h('b', {}, prompt)];
  text.split(EOT).forEach((part, i) => {
    if (i > 0) nodes.push(h('span', { class: 'eot', title: '文档结束 token' }, EOT));
    nodes.push(part);
  });
  return nodes;
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-replay fieldset { border: 0; padding: 0; margin: 0 0 .5rem; display: flex; flex-wrap: wrap; gap: .3rem .9rem; }
.w-replay legend { float: left; margin-right: .6rem; color: var(--ink-2); font-size: .85rem; }
.w-replay label { font-size: .88rem; color: var(--ink-2); display: inline-flex; align-items: center; gap: .3rem; }
.w-replay input[type=range] { flex: 1 1 8rem; max-width: 22rem; min-width: 0; }
.w-replay .w-replay-step { white-space: nowrap; min-width: 5.5em; }
.w-replay .w-replay-sample { margin: .3rem 0 0; font: 12.5px/1.5 var(--mono); font-variant-ligatures: none; background: var(--paper-2); border-radius: 4px;
  padding: .45rem .6rem; white-space: pre-wrap; word-break: break-all; min-height: 4.5em; }
.w-replay .w-replay-sample b { color: var(--ai); font-weight: 600; }
.w-replay .w-replay-note { font-size: .82rem; color: var(--ink-3); margin: .6rem 0 0; }
.w-replay .eot { color: var(--ink-3); font-size: .85em; border: 1px solid var(--rule); border-radius: 3px; padding: 0 .2em; margin: 0 .1em; }
.w-replay .w-replay-meta { font-size: .8rem; color: var(--ink-3); margin-top: .5rem; }
.w-replay svg text.end-label { paint-order: stroke; stroke: var(--scroll-bg, var(--paper));
  stroke-width: 4px; stroke-linejoin: round; pointer-events: none; }
`));
}

/** lineChart draws the reference-line labels first and the curves over them, so a curve that passes under a
    label hides its text. Move every label of the charts inside `box` to the top (after each re-render, too);
    the style above gives them a halo in the background color. Also used by trainer-live.js. */
export function liftLabels(box) {
  const lift = () => {
    for (const svg of box.querySelectorAll('svg')) {
      for (const t of svg.querySelectorAll('text.end-label:not([data-lifted])')) {
        t.setAttribute('data-lifted', '');
        svg.append(t);
      }
    }
  };
  new MutationObserver(lift).observe(box, { childList: true, subtree: true });
  lift();
}

/** Mean of every `k` consecutive training losses, plotted at the last step of each group (10, 20, 30, ...),
    so that every 50 steps a training point and a held-out point share the same x. */
function buckets(losses, k = BUCKET) {
  const out = [];
  for (let i = 0; i + k <= losses.length; i += k) {
    let s = 0;
    for (let j = i; j < i + k; j++) s += losses[j];
    out.push([i + k, s / k]);
  }
  return out;
}

export async function mount(root, cfg = {}) {
  injectStyle();
  let rec;
  try {
    const r = await fetch('data/train-in-browser/runs.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    rec = await r.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `没有读到录制的运行记录（${err.message}）。可以用 node tests/js/fixtures/trainer/record_runs.mjs 重新生成。`));
    return;
  }
  const views = cfg.views || Object.keys(VIEWS);
  let view = cfg.view || views[0];
  const name = `replay-${Math.random().toString(36).slice(2)}`;
  const radios = views.map((v) => {
    const input = h('input', { type: 'radio', name, value: v, checked: v === view });
    input.addEventListener('change', () => { if (input.checked) { view = v; render(); } });
    return h('label', {}, input, VIEWS[v]);
  });
  const chartBox = h('div');
  const slider = h('input', { type: 'range', min: 0, max: 0, step: 1, value: 0, 'aria-label': '选择第几步的样例' });
  const sliderLabel = h('span', { class: 'w-label w-replay-step' });
  const sampleLabel = h('span', { class: 'w-label' });
  const sampleBox = h('pre', { class: 'w-replay-sample', 'aria-live': 'polite' });
  const extra = h('div'); // overfit view: the memorized text, and what the model writes away from it
  const sampleWrap = h('div', {}, h('div', { class: 'w-row' }, sampleLabel, slider, sliderLabel), sampleBox, extra);
  const p = rec.provenance;
  const meta = h('div', { class: 'w-replay-meta' });
  root.append(h('div', { class: 'w-replay' },
    views.length > 1 ? h('fieldset', {}, h('legend', {}, '查看'), radios) : null, chartBox, sampleWrap, meta));

  // The bigram label goes where fewer curves pass under it: in the learning-rate view the 1e-2 curve runs
  // along it on the right, in the other two views the curves cross it on the left.
  const refs = (bigramAt) => [
    { y: rec.lnV, label: `ln ${rec.config.vocabSize} = ${rec.lnV.toFixed(2)}` },
    { y: rec.baselines.unigram, label: `unigram ${rec.baselines.unigram.toFixed(2)}` },
    { y: rec.baselines.bigram, label: `bigram ${rec.baselines.bigram.toFixed(2)}`, labelAt: bigramAt },
  ];
  let chart = null, samples = [], prompt = rec.settings.prompt;
  const showSample = () => {
    const s = samples[Number(slider.value)];
    if (!s) return;
    sliderLabel.textContent = `第 ${fmt(s.step)} 步`;
    sampleBox.replaceChildren(...sampleNodes(prompt, s.text));
  };
  slider.addEventListener('input', showSample);

  function render() {
    let series;
    if (view === 'lr') {
      const cut = rec.runs['lr3e-4'].steps;
      series = LR_RUNS.map((id) => ({
        name: `lr ${rec.runs[id].lr.toExponential(0)}`,
        points: rec.runs[id].held.filter(([k]) => k <= cut),
      }));
    } else {
      const run = rec.runs[view];
      series = [
        { name: '训练 loss', points: buckets(run.losses) },
        { name: '留出 loss', points: run.held },
      ];
    }
    const opts = { series, xLabel: '步数（参数更新次数）', yLabel: 'loss（nats）', yMin: 0, height: 280,
      refLines: refs(view === 'lr' ? 'left' : 'right'), ariaLabel: `${VIEWS[view]}：loss 随步数变化的曲线` };
    if (!chart) { chart = lineChart(chartBox, opts); liftLabels(chartBox); } else chart.update(opts);

    sampleWrap.hidden = view === 'lr';
    extra.replaceChildren();
    if (view !== 'lr') {
      const run = rec.runs[view];
      samples = run.samples;
      prompt = run.prompt ?? rec.settings.prompt;
      sampleLabel.textContent = view === 'overfit' ? '样例（从第 1 段的开头写起）' : '样例';
      slider.max = String(samples.length - 1);
      const preferred = samples.findIndex((s) => s.step === (view === 'overfit' ? 300 : 1000));
      slider.value = String(preferred >= 0 ? preferred : samples.length - 1);
      showSample();
      if (view === 'overfit') {
        extra.append(
          h('p', { class: 'w-replay-note' }, `被反复训练的 batch 是 ${run.batch.length} 段、每段 ${rec.config.ctx} 个字符的代码。第 1 段的原文：`),
          h('pre', { class: 'w-replay-sample' }, sampleNodes(run.prompt, run.batch[0].slice(run.prompt.length))),
          h('p', { class: 'w-replay-note' }, `离开这一段：第 ${fmt(run.offBatch.step)} 步时，从 “${rec.settings.prompt}” 开始写：`),
          h('pre', { class: 'w-replay-sample' }, sampleNodes(rec.settings.prompt, run.offBatch.text)));
      }
    }
    const runs = (view === 'lr' ? LR_RUNS : [view]).map((id) => rec.runs[id].label).join('；');
    meta.textContent = `录制：${p.engine}，${p.cpu}，WebAssembly SIMD ${p.simd ? '开' : '关'}，${p.recorded}。${runs}。`
      + `批大小 ${rec.settings.batch}，上下文 ${rec.config.ctx}，${fmt(rec.params)} 个参数，随机种子 ${rec.settings.seed}；`
      + `留出 loss 在整份留出文本（${fmt(rec.heldOut.targets)} 个预测位置）上计算；样例温度 ${rec.settings.temperature}。`
      + '在 V8 里，曲线只取决于随机种子和 SIMD 开关。';
  }
  render();
}
