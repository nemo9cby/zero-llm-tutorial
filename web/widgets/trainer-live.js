/* Live trainer: a tiny GPT learns TinyCodes in a Web Worker (trainer-worker.js) while you watch.
   config: { config: {ctx, embed, heads, layers, ff}, batch, lr, maxSteps, evalEvery, sampleEvery,
             sampleChars, prompt, temperature, seed } */
import { h, fmt } from './_lib.js';
import { lineChart } from './_plot.js';

const LRS = [3e-4, 1e-3, 3e-3, 1e-2];
const STEPS = [300, 1000, 3000];
const TEMPS = [0.5, 0.8, 1.0];
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
.w-trainer .w-row { gap: .5rem .8rem; }
.w-trainer fieldset { border: 0; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: .3rem .9rem; align-items: center; }
.w-trainer legend { float: left; margin-right: .6rem; color: var(--ink-2); font-size: .85rem; }
.w-trainer label { font-size: .88rem; color: var(--ink-2); display: inline-flex; align-items: center; gap: .3rem; }
.w-trainer select { font: inherit; font-size: .88rem; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .1rem .3rem; }
.w-trainer button.primary { border-color: var(--ai); color: var(--ai); font-weight: 600; }
.w-trainer button:disabled { opacity: .45; cursor: default; }
.w-trainer .w-trainer-status { font: 13px/1.6 var(--mono); font-variant-ligatures: none; color: var(--ink-2); font-variant-numeric: tabular-nums; margin: .4rem 0 .6rem; }
.w-trainer .w-trainer-status b { color: var(--ink); font-weight: 600; }
.w-trainer .w-trainer-samples { margin-top: .8rem; }
.w-trainer .w-trainer-sample { margin: 0 0 .7rem; }
.w-trainer .w-trainer-sample header { font-size: .82rem; color: var(--ink-3); margin-bottom: .15rem; }
.w-trainer .w-trainer-sample pre { margin: 0; font: 12.5px/1.5 var(--mono); font-variant-ligatures: none; background: var(--paper-2); border-radius: 4px;
  padding: .45rem .6rem; white-space: pre-wrap; word-break: break-all; max-height: 11.5em; overflow: auto; }
.w-trainer .w-trainer-sample pre b { color: var(--ai); font-weight: 600; }
.w-trainer .eot, .w-replay .eot { color: var(--ink-3); font-size: .85em; border: 1px solid var(--rule); border-radius: 3px; padding: 0 .2em; margin: 0 .1em; }
.w-trainer .w-trainer-error { color: var(--shu); font-size: .9rem; }
.w-trainer .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
`));
}

const fmtLr = (v) => v.toExponential(0);
const select = (label, values, current, format) => h('select', { 'aria-label': label },
  values.map((v) => h('option', { value: String(v), selected: v === current }, format(v))));

export function mount(root, cfg = {}) {
  injectStyle();
  const settings = {
    batch: cfg.batch ?? 8, lr: cfg.lr ?? 3e-3, maxSteps: cfg.maxSteps ?? 1000, evalEvery: cfg.evalEvery ?? 25,
    sampleEvery: cfg.sampleEvery ?? 100, sampleChars: cfg.sampleChars ?? 120, prompt: cfg.prompt ?? 'def ',
    temperature: cfg.temperature ?? 0.8, mode: 'normal',
  };
  const model = { ctx: 64, embed: 64, heads: 4, layers: 2, ff: 192, ...(cfg.config || {}) };

  const startBtn = h('button', { type: 'button', class: 'primary', disabled: true }, '开始训练');
  const pauseBtn = h('button', { type: 'button', disabled: true }, '暂停');
  const resetBtn = h('button', { type: 'button', disabled: true }, '重置');
  const sampleBtn = h('button', { type: 'button', disabled: true }, '现在采样一段');
  const modeNormal = h('input', { type: 'radio', name: `trainer-mode-${Math.random().toString(36).slice(2)}`, value: 'normal', checked: true });
  const modeOverfit = h('input', { type: 'radio', name: modeNormal.name, value: 'overfit' });
  const lrSel = select('学习率', LRS, settings.lr, fmtLr);
  const stepSel = select('训练步数', STEPS, settings.maxSteps, (v) => `${fmt(v)} 步`);
  const tempSel = select('采样温度', TEMPS, settings.temperature, (v) => v.toFixed(1));
  const simdBox = h('input', { type: 'checkbox', checked: true, disabled: true });
  const status = h('div', { class: 'w-trainer-status' }, '正在加载数据并初始化模型……');
  const live = h('div', { class: 'sr-only', 'aria-live': 'polite' });
  const chartBox = h('div');
  const samples = h('div', { class: 'w-trainer-samples', 'aria-label': '生成的样例（最新的在上面）' });
  const errorBox = h('div', { class: 'w-trainer-error', role: 'alert' });

  root.append(h('div', { class: 'w-trainer' },
    h('div', { class: 'w-row' }, startBtn, pauseBtn, resetBtn, sampleBtn),
    h('div', { class: 'w-row' }, h('fieldset', {}, h('legend', {}, '模式'),
      h('label', {}, modeNormal, '正常训练（每步换一批随机窗口）'),
      h('label', {}, modeOverfit, '过拟合一个 batch'))),
    h('div', { class: 'w-row' },
      h('label', {}, '学习率', lrSel), h('label', {}, '步数', stepSel), h('label', {}, '采样温度', tempSel),
      h('label', {}, simdBox, 'WebAssembly SIMD')),
    status, live, errorBox, chartBox, samples));

  // ------------------------------------------------------------ chart state
  let facts = null, trainPts = [], evalPts = [], bucket = [], chart = null, lastDraw = 0, drawTimer = 0;
  let step = 0, seconds = 0, running = false, lastLoss = null;

  const refLines = () => [
    { y: facts.lnV, label: `ln ${facts.vocabSize} = ${facts.lnV.toFixed(2)}（瞎猜）` },
    { y: facts.baselines.unigram, label: `unigram ${facts.baselines.unigram.toFixed(2)}` },
    { y: facts.baselines.bigram, label: `bigram ${facts.baselines.bigram.toFixed(2)}` },
  ];
  const chartOpts = () => ({
    series: [
      { name: '训练 loss', points: trainPts.length ? trainPts : [[0, facts.initialLoss]] },
      { name: '留出 loss', points: evalPts },
    ],
    xLabel: '步数（参数更新次数）', yLabel: 'loss（nats）', yMin: 0, refLines: refLines(), table: false, height: 280,
    ariaLabel: '训练 loss 与留出 loss 随步数变化的曲线',
  });
  const draw = () => {
    drawTimer = 0;
    lastDraw = performance.now();
    if (!chart) chart = lineChart(chartBox, chartOpts());
    else chart.update(chartOpts());
  };
  const scheduleDraw = () => {
    if (drawTimer) return;
    drawTimer = setTimeout(draw, Math.max(0, 250 - (performance.now() - lastDraw)));
  };
  const showStatus = () => {
    const rate = seconds > 0 ? step / seconds : 0;
    const lossTxt = lastLoss == null ? '' : ` · 训练 loss <b>${lastLoss.toFixed(3)}</b>`;
    const valTxt = evalPts.length ? ` · 留出 loss <b>${evalPts.at(-1)[1].toFixed(3)}</b>` : '';
    const speed = seconds > 0 ? ` · ${rate.toFixed(1)} 步/秒 · ${fmt(Math.round(rate * settings.batch * model.ctx))} token/秒`
      + ` · 训练用时 ${seconds.toFixed(1)} 秒` : '';
    status.innerHTML = `第 <b>${fmt(step)}</b> / ${fmt(settings.maxSteps)} 步${speed}${lossTxt}${valTxt}`;
  };
  const setRunning = (on) => {
    running = on;
    startBtn.disabled = on || step >= settings.maxSteps;
    startBtn.textContent = step > 0 && !on ? '继续训练' : '开始训练';
    pauseBtn.disabled = !on;
    resetBtn.disabled = false;
    sampleBtn.disabled = on;
  };
  const addSample = ({ step: s, prompt, text, val }) => {
    const item = h('figure', { class: 'w-trainer-sample' },
      h('header', {}, `第 ${fmt(s)} 步${val != null ? `（留出 loss ${val.toFixed(2)}）` : ''}，温度 ${settings.temperature}，从 “${prompt}” 开始：`),
      h('pre', {}, sampleNodes(prompt, text)));
    samples.prepend(item);
    while (samples.children.length > 4) samples.lastChild.remove();
  };
  const clearRun = (initialLoss, initialVal) => {
    trainPts = []; bucket = []; evalPts = [[0, initialVal]]; step = 0; seconds = 0; lastLoss = initialLoss;
    samples.replaceChildren();
    facts.initialLoss = initialLoss;
    draw();
    showStatus();
  };

  // ------------------------------------------------------------ worker
  let worker;
  try {
    worker = new Worker(new URL('./trainer-worker.js', import.meta.url), { type: 'module' });
  } catch (err) {
    status.textContent = '';
    errorBox.textContent = `这个浏览器无法启动模块 Web Worker（${err.message}），训练器不可用。`;
    return;
  }
  worker.onerror = (e) => { errorBox.textContent = `训练线程出错：${e.message || '未知错误'}`; setRunning(false); };
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') {
      facts = msg;
      simdBox.disabled = !msg.simd;
      simdBox.checked = msg.simd;
      clearRun(msg.initialLoss, msg.initialVal);
      setRunning(false);
      live.textContent = `模型就绪：${fmt(msg.params)} 个参数，词表 ${msg.vocabSize}。`;
      worker.postMessage({ type: 'sample' });
    } else if (msg.type === 'progress') {
      step = msg.step; seconds = msg.seconds;
      for (const [s, loss] of msg.losses) {
        bucket.push(loss);
        lastLoss = loss;
        if (bucket.length === BUCKET) {
          trainPts.push([s - (BUCKET - 1) / 2, bucket.reduce((a, b) => a + b, 0) / BUCKET]);
          bucket = [];
        }
      }
      evalPts.push(...msg.evals);
      showStatus();
      scheduleDraw();
    } else if (msg.type === 'sample') {
      addSample(msg);
    } else if (msg.type === 'state') {
      setRunning(msg.running);
      if (msg.done) live.textContent = `训练完成：${fmt(msg.step)} 步。`;
    } else if (msg.type === 'reset') {
      clearRun(msg.initialLoss, msg.initialVal);
      setRunning(false);
      worker.postMessage({ type: 'sample' });
    } else if (msg.type === 'simd') {
      simdBox.checked = msg.on;
    } else if (msg.type === 'error') {
      errorBox.textContent = `出错了：${msg.message}`;
      setRunning(false);
    }
  };
  const base = new URL('../data/train-in-browser/', import.meta.url);
  worker.postMessage({
    type: 'init', trainURL: new URL('tinycodes-train.txt', base).href, valURL: new URL('tinycodes-val.txt', base).href,
    config: model, seed: cfg.seed ?? 0, settings,
  });

  // ------------------------------------------------------------ controls
  const set = (patch) => { Object.assign(settings, patch); worker.postMessage({ type: 'set', settings: patch }); };
  startBtn.addEventListener('click', () => { errorBox.textContent = ''; worker.postMessage({ type: 'start' }); });
  pauseBtn.addEventListener('click', () => worker.postMessage({ type: 'pause' }));
  resetBtn.addEventListener('click', () => worker.postMessage({ type: 'reset' }));
  sampleBtn.addEventListener('click', () => worker.postMessage({ type: 'sample' }));
  for (const r of [modeNormal, modeOverfit]) r.addEventListener('change', () => r.checked && set({ mode: r.value }));
  lrSel.addEventListener('change', () => set({ lr: Number(lrSel.value) }));
  stepSel.addEventListener('change', () => { set({ maxSteps: Number(stepSel.value) }); setRunning(running); showStatus(); });
  tempSel.addEventListener('change', () => set({ temperature: Number(tempSel.value) }));
  simdBox.addEventListener('change', () => set({ simd: simdBox.checked }));
}
