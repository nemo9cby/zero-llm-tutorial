/* GPU 成本计算器：模型与超参数 + GPU + MFU -> 训练 token、FLOPs、小时与美元。
   数据：data/gpu/bots.json（结构与超参数，抄自原书脚本）、data/gpu/gpus.json（峰值算力与价格，附来源）。
   计算：gpu-costmodel.js（与 gpu/costs.py 逐行对应，有 node 测试对拍）。
   config: { "scenario": "webbot" }   // 初始预设，取 bots.json 的 scenarios[].id */
import { h, fmt } from './_lib.js';
import * as cm from './gpu-costmodel.js';

let uid = 0;
let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-gpu-calc { font-size: .92rem; }
.w-gpu-calc .ctrls { display: grid; grid-template-columns: repeat(auto-fit, minmax(13.5rem, 1fr)); gap: .7rem 1.1rem; align-items: end; }
.w-gpu-calc .wide { grid-column: 1 / -1; }
.w-gpu-calc label { display: flex; flex-direction: column; gap: .25rem; font-size: .8rem; color: var(--ink-2); }
.w-gpu-calc select, .w-gpu-calc input[type=number] { width: 100%; font: 14px/1.4 var(--serif); color: var(--ink);
  background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .3rem .45rem; box-sizing: border-box; }
.w-gpu-calc input[type=range] { width: 100%; accent-color: var(--ai); }
.w-gpu-calc .val { font-variant-numeric: tabular-nums; color: var(--ink); font-weight: 600; }
.w-gpu-calc details { margin: .7rem 0 0; }
.w-gpu-calc details > summary { cursor: pointer; color: var(--ink-2); font-size: .85rem; }
.w-gpu-calc details .ctrls { margin-top: .5rem; }
.w-gpu-calc .big { margin: 1rem 0 .3rem; font-size: 1.45rem; font-weight: 800; line-height: 1.35; font-variant-numeric: tabular-nums; }
.w-gpu-calc .big small { font-size: .85rem; font-weight: 400; color: var(--ink-2); }
.w-gpu-calc dl { display: grid; grid-template-columns: minmax(6.5em, max-content) minmax(0, 1fr); gap: .2rem 1rem; margin: .5rem 0 .6rem; }
.w-gpu-calc dt { color: var(--ink-3); }
.w-gpu-calc dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.w-gpu-calc .note { margin: .5rem 0; padding: .35rem .75rem; border-left: 2px solid var(--kincha); font-size: .86rem; }
.w-gpu-calc .note.info { border-left-color: var(--rule); color: var(--ink-2); }
.w-gpu-calc .cmp { overflow-x: auto; }
.w-gpu-calc table { margin: .5rem 0; font-size: .84rem; }
.w-gpu-calc th, .w-gpu-calc td { padding: .25rem .5rem .25rem 0; white-space: nowrap; vertical-align: top; }
.w-gpu-calc th { white-space: normal; text-align: left; }
.w-gpu-calc td.lab { white-space: normal; min-width: 5.5em; }
.w-gpu-calc td.num { text-align: right; font-variant-numeric: tabular-nums; }
.w-gpu-calc tr.hl td { font-weight: 700; }
.w-gpu-calc .src { font-size: .78rem; color: var(--ink-3); overflow-wrap: anywhere; }
.w-gpu-calc .src a { color: inherit; }
`));
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} ${url}`);
  return r.json();
}

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export function sci(x) {
  if (!x) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  const m = x / 10 ** e;
  return `${m.toFixed(2)} × 10${String(e).split('').map((c) => SUP[c]).join('')}`;
}
export function hours(hr) {
  if (hr < 1) return `${Math.max(1, Math.round(hr * 60))} 分钟`;
  if (hr < 48) return `${hr.toFixed(hr < 10 ? 2 : 1)} 小时`;
  return `${(hr / 24).toFixed(1)} 天`;
}
export function usd(x) {
  if (x === 0) return '0 美元';
  return `${x < 10 ? x.toFixed(2) : x.toFixed(1)} 美元`;
}
export function tokens(n) {
  return n >= 1e8 ? `${fmt(n)}（约 ${(n / 1e8).toFixed(n >= 1e10 ? 0 : 1)} 亿）` : fmt(n);
}
const sameShape = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export async function mount(root, config = {}) {
  injectStyle();
  let gpus, botsData;
  try {
    [gpus, botsData] = await Promise.all([getJSON('data/gpu/gpus.json'), getJSON('data/gpu/bots.json')]);
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `计算器的数据没有加载成功：${err.message}`));
    return;
  }
  const { bots, scenarios } = botsData;
  const id = `gpucalc${++uid}`;
  const offerById = Object.fromEntries(gpus.offers.map((o) => [o.id, o]));
  const gpuById = Object.fromEntries(gpus.gpus.map((g) => [g.id, g]));
  // maxIters: what the run uses (a preset or the reader); userIters: only what the reader typed. The multi-GPU
  // comparison starts from the single-GPU job, so a preset's already divided max_iters never feeds back into it.
  const st = { scenario: null, bot: 'codebot', offer: 'local:rtx3060ti', world: 1, mfu: 0.3, floorMs: 15,
    maxIters: null, userIters: null, batch: null, accum: null, nmm: null, sPerIt: null, evalS: null };

  // ---------------------------------------------------------------- controls
  const field = (label, control, cls) => h('label', { for: control.id, class: cls || null }, label, control);
  const customOpt = h('option', { value: 'custom', disabled: true, hidden: true }, '自定义（改过预设）');
  const selScenario = h('select', { id: `${id}-sc` },
    scenarios.map((s) => h('option', { value: s.id }, s.label)), customOpt);
  const selOffer = h('select', { id: `${id}-of` });
  const groups = {};
  for (const o of gpus.offers) {
    const prov = gpus.providers[o.provider];
    groups[o.provider] ||= h('optgroup', { label: prov.label });
    groups[o.provider].append(h('option', { value: o.id }, o.label));
  }
  selOffer.append(...Object.values(groups));
  const selWorld = h('select', { id: `${id}-w` });
  const mfu = h('input', { id: `${id}-mfu`, type: 'range', min: 1, max: 60, step: 1, 'aria-label': 'MFU（百分比）' });
  const mfuOut = h('span', { class: 'val' });
  const floor = h('input', { id: `${id}-fl`, type: 'number', min: 0, max: 500, step: 1 });
  // Once the reader changes anything a preset sets, the preset's name no longer describes the run.
  const markCustom = () => {
    customOpt.hidden = false;
    selScenario.value = 'custom';
    st.scenario = null;
  };
  const num = (key, min, label) => {
    const el = h('input', { id: `${id}-${key}`, type: 'number', min, step: 1, placeholder: '原书默认' });
    el.addEventListener('input', () => {
      const v = parseInt(el.value, 10);
      st[key] = v > 0 ? v : null;
      if (key === 'maxIters') st.userIters = st.maxIters;
      markCustom();
      render();
    });
    return field(label, el);
  };
  const real = (key, label) => {
    const el = h('input', { id: `${id}-${key}`, type: 'number', min: 0, step: 'any', placeholder: '未填' });
    el.addEventListener('input', () => { const v = parseFloat(el.value); st[key] = v > 0 ? v : null; render(); });
    return field(label, el);
  };
  const sPerItField = real('sPerIt', '实测：每步秒数（读进度条中段）');
  const evalSField = real('evalS', '实测：每次评估秒数');
  const advIters = num('maxIters', 1, 'max_iters（总步数）');
  const advBatch = num('batch', 1, '每卡 micro-batch');
  const advAccum = num('accum', 1, '梯度累积步数');
  const advNmm = num('nmm', 1, 'n_mm：每 token 的矩阵权重');

  const out = h('div', { 'aria-live': 'polite' });
  root.append(h('div', { class: 'w-gpu-calc' },
    h('div', { class: 'ctrls' },
      field('预设', selScenario, 'wide'),
      field('平台与 GPU', selOffer),
      field('GPU 数量', selWorld),
      field(h('span', {}, 'MFU（实际算力 ÷ 峰值）', ' ', mfuOut), mfu),
      field('每步最少耗时（毫秒）', floor)),
    h('details', {}, h('summary', {}, '改超参数（留空 = 原书脚本的值）'),
      h('div', { class: 'ctrls' }, advIters, advBatch, advAccum, advNmm)),
    h('details', {}, h('summary', {}, '用实测数字估算（填入探针或 tqdm 中段的读数）'),
      h('div', { class: 'ctrls' }, sPerItField, evalSField)),
    out));

  // Only ch09 (WebBot) supports torchrun; the other bots run on one GPU, so 8-GPU-only offers are disabled.
  function setWorldOptions() {
    const multi = st.bot === 'webbot';
    for (const opt of selOffer.querySelectorAll('option')) opt.disabled = !multi && !offerById[opt.value].counts.includes(1);
    if (selOffer.selectedOptions[0]?.disabled) {
      st.offer = [...selOffer.querySelectorAll('option')].find((o) => !o.disabled).value;
      selOffer.value = st.offer;
    }
    const counts = multi ? offerById[st.offer].counts : [1];
    if (!counts.includes(st.world)) st.world = counts[0];
    selWorld.replaceChildren(...counts.map((n) => h('option', { value: n, selected: n === st.world }, `${n} 张`)));
  }
  function applyScenario(sid) {
    const s = scenarios.find((x) => x.id === sid) || scenarios[0];
    if (s.bot !== st.bot) {
      // Probe readings belong to one bot: CodeBot's preset must not inherit WebBot's 0.317 s/step.
      st.sPerIt = st.evalS = null;
      for (const wrap of [sPerItField, evalSField]) wrap.querySelector('input').value = '';
    }
    Object.assign(st, { scenario: s.id, bot: s.bot, offer: s.offer, world: s.world, mfu: s.mfu,
      maxIters: s.max_iters || null, userIters: null, batch: s.batch || null, accum: s.accum || null, nmm: null });
    customOpt.hidden = true;
    selScenario.value = s.id;
    selOffer.value = st.offer;
    for (const [key, wrap] of [['maxIters', advIters], ['batch', advBatch], ['accum', advAccum], ['nmm', advNmm]]) {
      wrap.querySelector('input').value = st[key] ?? '';
    }
    setWorldOptions();
  }

  selScenario.addEventListener('change', () => { applyScenario(selScenario.value); render(); });
  selOffer.addEventListener('change', () => { st.offer = selOffer.value; setWorldOptions(); markCustom(); render(); });
  selWorld.addEventListener('change', () => { st.world = parseInt(selWorld.value, 10); markCustom(); render(); });
  mfu.addEventListener('input', () => { st.mfu = parseInt(mfu.value, 10) / 100; markCustom(); render(); });
  floor.addEventListener('input', () => { const v = parseFloat(floor.value); st.floorMs = v >= 0 ? v : 0; render(); });

  // ---------------------------------------------------------------- output
  function render() {
    const b = st.nmm ? { ...bots[st.bot], n_mm: st.nmm } : bots[st.bot];
    const offer = offerById[st.offer];
    const gpu = gpuById[offer.gpu];
    const price = cm.pricePerGpuHour(offer);
    mfu.value = Math.round(st.mfu * 100);
    mfuOut.textContent = `${Math.round(st.mfu * 100)}%`;
    floor.value = st.floorMs;
    const shape = cm.runShape(b, { world: st.world, maxIters: st.maxIters, batch: st.batch, accum: st.accum });
    const rows = [];
    const notes = [];
    if (cm.peakTflops(b, gpu) == null) {
      out.replaceChildren(h('p', { class: 'note' },
        `${gpu.name} 没有公布 ${b.precision.toUpperCase()} 峰值${gpu.note ? `（${gpu.note}）` : ''}。`
        + `${b.label} 用 ${b.precision.toUpperCase()} 训练，请换一张卡。`));
      return;
    }
    const est = cm.estimate(b, gpu, price, shape, st.mfu, st.floorMs);
    rows.push(['参数量', `${fmt(b.params)}；每 token 的矩阵权重 n_mm ${fmt(b.n_mm)}${st.nmm ? '（自定义）' : ''}`]);
    rows.push(['训练 token', `${tokens(shape.train_tokens)}`]);
    rows.push(['每步 token', `${fmt(shape.tokens_per_step)}（每卡 ${shape.batch} × ${shape.accum} × ${b.ctx}${st.world > 1 ? ` × ${st.world} 卡` : ''}）`]);
    rows.push(['优化器步数', fmt(shape.steps)]);
    rows.push(['完整评估', shape.evals ? `${shape.evals} 次，共 ${(shape.eval_tokens / 1e8).toFixed(1)} 亿 token，只在 rank 0 上跑` : '无']);
    rows.push(['计算量', `${sci(est.train_flops)} 训练 + ${sci(est.eval_flops)} 评估 FLOPs`]);
    rows.push(['峰值算力', `${gpu.name} ${b.precision.toUpperCase()} ${cm.peakTflops(b, gpu)} TFLOP/s（稠密）`]);
    rows.push(['每步计算', `${est.step_ms.toFixed(1)} 毫秒${est.floor_bound ? `，低于下限，按 ${st.floorMs} 毫秒计` : ''}`]);
    rows.push(['GPU·小时', est.gpu_hours.toFixed(2)]);
    rows.push(['单价', price ? `${price.toFixed(2)} 美元 / GPU·小时${offer.provider === 'modal' ? '（只算 GPU；CPU 与内存另计，见 G.5）' : ''}`
      : '0（本机，电费不计）']);

    const big = h('div', { class: 'big' }, `${hours(est.wall_hours)} · ${usd(est.usd)} `,
      h('small', {}, st.world > 1 ? `（${st.world} 张卡同时计费）` : ''));
    const defaults = st.maxIters == null && st.batch == null && st.accum == null;
    if (st.world > 1 && defaults) {
      notes.push(h('p', { class: 'note' }, `原样启动 ${st.world} 个进程：每张卡都跑满 ${fmt(b.max_iters)} 步、各取各的 batch，`
        + `总 token 是单卡的 ${st.world} 倍，墙钟时间却和单卡一样。见下表的另外两种跑法。`));
    }
    if (est.floor_bound) {
      notes.push(h('p', { class: 'note info' }, `每步只有 ${est.step_ms.toFixed(1)} 毫秒的计算：小模型遇上快 GPU，kernel 启动、`
        + 'Python 循环和取 batch 的时间比计算还长。这时 MFU 已经没有意义，请用 200 步探针实测。'));
    }
    const children = [big, h('dl', {}, rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)])), ...notes];

    if (st.sPerIt) {
      const m = cm.estimateMeasured({ steps: shape.steps, sPerIt: st.sPerIt, evals: shape.evals,
        evalS: st.evalS || 0, world: st.world, usdPerGpuHour: price });
      children.push(h('p', { class: 'note info' }, `按实测：${fmt(shape.steps)} 步 × ${st.sPerIt} 秒`
        + (shape.evals ? ` + ${shape.evals} 次评估 × ${st.evalS || 0} 秒` : '') + ` = ${hours(m.wall_hours)}，${usd(m.usd)}。`
        + (st.evalS ? '' : (shape.evals ? '（还没填评估耗时，这部分按 0 计）' : ''))));
    }

    if (st.bot === 'webbot' && st.world > 1) {
      // Rows come from the single-GPU job (the book's max_iters, or what the reader typed), never from the preset.
      const variants = cm.multiGpuVariants(b, st.world, st.userIters || b.max_iters);
      let matched = false;
      const body = variants.map(({ label, args }) => {
        const sh = cm.runShape(b, args);
        const e = cm.estimate(b, gpu, price, sh, st.mfu, st.floorMs);
        const same = sameShape(sh, shape);
        matched ||= same;
        return h('tr', { class: same ? 'hl' : null },
          h('td', { class: 'lab' }, label), h('td', { class: 'num' }, usd(e.usd)),
          h('td', { class: 'num' }, hours(e.wall_hours)), h('td', { class: 'num' }, fmt(sh.evals)),
          h('td', { class: 'num' }, `${(sh.train_tokens / 1e8).toFixed(0)} 亿`));
      });
      children.push(h('div', { class: 'cmp' }, h('table', {},
        h('thead', {}, h('tr', {}, ['跑法', '费用', '墙钟', '评估', '训练 token'].map((t) => h('th', {}, t)))),
        h('tbody', {}, body))));
      children.push(h('p', { class: 'note info' }, (matched ? '加粗的一行就是上面的配置。' : '')
        + `max_iters ÷ ${st.world} 的全局 batch 是单卡的 ${st.world} 倍、更新次数只有单卡的 1/${st.world}；`
        + `每卡 batch ÷ ${st.world} 与单卡的优化过程相同。`));
    }
    const prov = gpus.providers[offer.provider];
    children.push(h('p', { class: 'src' },
      prov.src ? ['价格：', h('a', { href: prov.src, target: '_blank', rel: 'noopener' }, prov.src), `（${gpus.checked} 查证）；`] : '',
      '峰值：', ...gpu.src.flatMap((u, i) => [i ? '、' : '', h('a', { href: u, target: '_blank', rel: 'noopener' }, new URL(u).hostname)]),
      gpu.derived ? `（${gpu.derived}）` : ''));
    out.replaceChildren(...children);
  }

  applyScenario(config.scenario || scenarios[0].id);
  render();
}
