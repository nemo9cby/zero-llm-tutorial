/* Replay of the recorded runs in data/train-in-browser/runs.json (written by
   tests/js/fixtures/trainer/record_runs.mjs): loss curves, the samples at every 100 steps, provenance.
   config: { view: 'default' | 'overfit' | 'lr', views: ['default', 'overfit', 'lr'] } */
import { h, fmt } from './_lib.js';
import { lineChart } from './_plot.js';

const VIEWS = {
  default: '正常训练 2,000 步',
  overfit: '过拟合一个 batch',
  lr: '三种学习率',
};

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
  padding: .45rem .6rem; white-space: pre-wrap; word-break: break-all; min-height: 7.5em; }
.w-replay .w-replay-sample b { color: var(--ai); font-weight: 600; }
.w-replay .eot { color: var(--ink-3); font-size: .85em; border: 1px solid var(--rule); border-radius: 3px; padding: 0 .2em; margin: 0 .1em; }
.w-replay .w-replay-meta { font-size: .8rem; color: var(--ink-3); margin-top: .5rem; }
`));
}

/** Mean of every `k` consecutive training losses, plotted at the middle step of each group. */
function buckets(losses, k = 10) {
  const out = [];
  for (let i = 0; i + k <= losses.length; i += k) {
    let s = 0;
    for (let j = i; j < i + k; j++) s += losses[j];
    out.push([i + (k - 1) / 2, s / k]);
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
  const sampleBox = h('pre', { class: 'w-replay-sample', 'aria-live': 'polite' });
  const sampleWrap = h('div', {}, h('div', { class: 'w-row' }, h('span', { class: 'w-label' }, '样例'), slider, sliderLabel), sampleBox);
  const p = rec.provenance;
  const meta = h('div', { class: 'w-replay-meta' });
  root.append(h('div', { class: 'w-replay' },
    views.length > 1 ? h('fieldset', {}, h('legend', {}, '查看'), radios) : null, chartBox, sampleWrap, meta));

  const refs = [
    { y: rec.lnV, label: `ln ${rec.config.vocabSize} = ${rec.lnV.toFixed(2)}` },
    { y: rec.baselines.unigram, label: `unigram ${rec.baselines.unigram.toFixed(2)}` },
    { y: rec.baselines.bigram, label: `bigram ${rec.baselines.bigram.toFixed(2)}` },
  ];
  let chart = null, samples = [];
  const showSample = () => {
    const s = samples[Number(slider.value)];
    if (!s) return;
    sliderLabel.textContent = `第 ${fmt(s.step)} 步`;
    sampleBox.replaceChildren(...sampleNodes(rec.settings.prompt, s.text));
  };
  slider.addEventListener('input', showSample);

  function render() {
    let series, runIds;
    if (view === 'lr') {
      runIds = ['lr3e-4', 'default', 'lr1e-2'];
      const cut = rec.runs['lr3e-4'].steps;
      series = runIds.map((id) => ({
        name: `lr ${rec.runs[id].lr.toExponential(0)}`,
        points: rec.runs[id].held.filter(([k]) => k <= cut),
      }));
    } else {
      const run = rec.runs[view];
      runIds = [view];
      series = [
        { name: '训练 loss', points: buckets(run.losses) },
        { name: '留出 loss', points: run.held },
      ];
    }
    const opts = { series, xLabel: '步数（参数更新次数）', yLabel: 'loss（nats）', yMin: 0, refLines: refs, height: 280,
      ariaLabel: `${VIEWS[view]}：loss 随步数变化的曲线` };
    if (!chart) chart = lineChart(chartBox, opts);
    else chart.update(opts);
    const run = rec.runs[runIds.length === 1 ? runIds[0] : 'default'];
    samples = view === 'lr' ? rec.runs.default.samples.filter((s) => s.step <= rec.runs['lr3e-4'].steps) : run.samples;
    slider.max = String(samples.length - 1);
    const preferred = samples.findIndex((s) => s.step === (view === 'overfit' ? 100 : 1000));
    slider.value = String(preferred >= 0 ? preferred : samples.length - 1);
    sampleWrap.hidden = view === 'lr';
    showSample();
    const secs = runIds.map((id) => `${rec.runs[id].label}：${fmt(rec.runs[id].steps)} 步用时 ${rec.runs[id].seconds} 秒`).join('；');
    meta.textContent = `录制：${p.engine}，${p.cpu}，WebAssembly SIMD ${p.simd ? '开' : '关'}，${p.date}。${secs}。${p.note}`
      + ` 批大小 ${rec.settings.batch}，上下文 ${rec.config.ctx}，${fmt(rec.params)} 个参数，随机种子 ${rec.settings.seed}；`
      + `样例温度 ${rec.settings.temperature}。`;
  }
  render();
}
