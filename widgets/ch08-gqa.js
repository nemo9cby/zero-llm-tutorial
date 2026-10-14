/* 图 8-3：GQA 的分组与 KV cache 显存。
   分组约定与 PyTorch 的 enable_gqa 相同：第 h 个 query 头用第 h // (H/KV) 个 KV 头（repeat_interleave）。
   显存公式 2·L·KV·d_h·bytes 与验证 8.3-b 相同；纯函数由 tests/js/ch08-widgets.test.mjs 对拍。 */
import { h } from './_lib.js';
import { barChart, seriesColor, onThemeChange } from './_plot.js';

export const kvBytesPerToken = ({ L, KV, D, bytes }) => 2 * L * KV * D * bytes;

/** 第 h 个 query 头使用哪个 KV 头。'interleave' = repeat_interleave（PyTorch 的约定），'tile' = repeat（平铺）。 */
export function groupOf(h_, H, KV, mode = 'interleave') {
  return mode === 'interleave' ? Math.floor(h_ / (H / KV)) : h_ % KV;
}

/** 误用 repeat（平铺）时拿到别组 K、V 的 query 头：两种分组不一致的那些 h（验证 8.3-a）。 */
export const misgrouped = (H, KV) =>
  Array.from({ length: H }, (_, h) => h).filter((h) => groupOf(h, H, KV, 'interleave') !== groupOf(h, H, KV, 'tile'));

export const divisors = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((k) => n % k === 0);

export function formatBytes(b) {
  const units = [[2 ** 30, 'GiB'], [2 ** 20, 'MiB'], [2 ** 10, 'KiB']];
  for (const [u, s] of units) if (b >= u) return `${(b / u).toFixed(b / u >= 100 ? 0 : 1).replace(/\.0$/, '')} ${s}`;
  return `${b} B`;
}

/** 柱状图的单位：按最大的那根柱子选，刻度上的数字保持在 1 万以内。 */
export function chartUnit(maxBytes) {
  if (maxBytes >= 10240 * 2 ** 30) return [2 ** 40, 'TiB'];
  if (maxBytes >= 10240 * 2 ** 20) return [2 ** 30, 'GiB'];
  if (maxBytes >= 2 ** 20) return [2 ** 20, 'MiB'];
  return [2 ** 10, 'KiB'];
}

export const PRESETS = {
  webbot: { label: 'WebBot（GQA：12 个 query 头，4 个 KV 头）', L: 12, H: 12, KV: 4, D: 64, T: 1024, batch: 1, bytes: 2 },
  webbot_mha: { label: 'WebBot 改用 MHA（KV 头 = 12）', L: 12, H: 12, KV: 12, D: 64, T: 1024, batch: 1, bytes: 2 },
  webbot_mqa: { label: 'WebBot 改用 MQA（KV 头 = 1）', L: 12, H: 12, KV: 1, D: 64, T: 1024, batch: 1, bytes: 2 },
  mixtral: { label: 'Mixtral 8x7B（32 个 query 头，8 个 KV 头）', L: 32, H: 32, KV: 8, D: 128, T: 32768, batch: 1, bytes: 2 },
};

const HEADS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const DIMS = [32, 64, 128, 256];
const DTYPES = [[4, 'fp32（4 字节）'], [2, 'bf16（2 字节）'], [1, 'fp8（1 字节）']];

let styled = false;
let instances = 0;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch08-gqa select { font: inherit; max-width: 100%; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .1rem .3rem; }
.w-ch08-gqa .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap: .45rem 1rem; font-size: .88rem; margin: .5rem 0 .8rem; }
.w-ch08-gqa .grid label { display: flex; flex-direction: column; gap: .15rem; color: var(--ink-2); }
.w-ch08-gqa .grid input[type=range] { width: 100%; }
.w-ch08-gqa .grid output { color: var(--ink); font-variant-numeric: tabular-nums; }
.w-ch08-gqa svg { max-width: 100%; height: auto; display: block; margin: .3rem 0 .5rem; }
.w-ch08-gqa .modes { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .88rem; }
.w-ch08-gqa .result { font-size: .9rem; margin: .4rem 0 .2rem; }
.w-ch08-gqa .result b { font-variant-numeric: tabular-nums; }
`));
}

const SVGNS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, text) => {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
};

export function mount(root, config = {}) {
  injectStyle();
  const s = { ...(PRESETS[config.preset] || PRESETS.webbot) };
  let mode = 'interleave';
  const box = h('div', { class: 'w-ch08-gqa' });
  root.append(box);

  const preset = h('select', { 'aria-label': '选择预设' },
    Object.entries(PRESETS).map(([k, p]) => h('option', { value: k }, p.label)), h('option', { value: 'custom' }, '自定义'));
  preset.value = PRESETS[config.preset] ? config.preset : 'webbot';
  const range = (min, max, step, aria) => h('input', { type: 'range', min, max, step, 'aria-label': aria });
  const sel = (opts, aria) => h('select', { 'aria-label': aria }, opts.map(([v, t]) => h('option', { value: v }, t)));
  const ctl = {
    L: range(1, 80, 1, '层数'),
    H: sel(HEADS.map((x) => [x, String(x)]), 'query 头数'),
    KV: h('select', { 'aria-label': 'KV 头数' }),
    D: sel(DIMS.map((x) => [x, String(x)]), '每个头的维度'),
    T: range(7, 17, 1, '上下文长度（2 的幂）'),
    batch: range(1, 64, 1, 'batch 大小'),
    bytes: sel(DTYPES, '每个数的精度'),
  };
  const outs = { L: h('output'), T: h('output'), batch: h('output') };
  const field = (label, input, out) => h('label', {}, h('span', {}, label, out ? ' ' : '', out || ''), input);
  const radioName = `w-ch08-gqa-mode-${++instances}`;
  const modeRadios = [['interleave', 'repeat_interleave（PyTorch 的约定）'], ['tile', 'repeat（平铺，分组错了）']].map(([v, t]) => {
    const r = h('input', { type: 'radio', name: radioName, value: v, checked: v === mode });
    r.addEventListener('change', () => { if (r.checked) { mode = v; draw(); } });
    return h('label', {}, r, ` ${t}`);
  });
  const mapSvg = svg('svg', { role: 'img' });
  const result = h('div', { class: 'result', 'aria-live': 'polite' });
  const chartBox = h('div');
  box.append(
    h('div', { class: 'w-row' }, h('span', { class: 'w-label' }, '预设'), preset),
    h('div', { class: 'grid' },
      field('层数 L', ctl.L, outs.L), field('query 头数 H', ctl.H), field('KV 头数', ctl.KV),
      field('每头维度 d_h', ctl.D), field('上下文长度 T', ctl.T, outs.T), field('batch', ctl.batch, outs.batch),
      field('精度', ctl.bytes)),
    h('div', { class: 'modes', role: 'radiogroup', 'aria-label': '分组方式' }, modeRadios),
    mapSvg, result, chartBox,
  );

  let chart = null;
  function syncControls() {
    ctl.L.value = s.L; outs.L.value = s.L;
    ctl.H.value = s.H;
    ctl.KV.replaceChildren(...divisors(s.H).map((x) => h('option', { value: x }, String(x))));
    if (s.H % s.KV) s.KV = s.H;
    ctl.KV.value = s.KV;
    ctl.D.value = s.D;
    ctl.T.value = Math.round(Math.log2(s.T)); outs.T.value = s.T.toLocaleString('en-US');
    ctl.batch.value = s.batch; outs.batch.value = s.batch;
    ctl.bytes.value = s.bytes;
  }

  function drawMap() {
    const H = s.H, KV = s.KV;
    mapSvg.replaceChildren();
    if (H > 32) {
      mapSvg.setAttribute('viewBox', '0 0 320 24');
      mapSvg.setAttribute('width', 320);
      mapSvg.setAttribute('aria-label', '头数太多，不画连线');
      mapSvg.append(svg('text', { x: 0, y: 16, 'font-size': 12, fill: 'var(--ink-2)' }, `头数超过 32，不画连线：每 ${H / KV} 个 query 头共用 1 个 KV 头。`));
      return;
    }
    const W = Math.max(280, Math.min(box.clientWidth || 320, 640)), pad = 6, top = 18, bottom = 92, bh = 18;
    const qw = (W - 2 * pad) / H, kw = (W - 2 * pad) / KV;
    mapSvg.setAttribute('viewBox', `0 0 ${W} ${bottom + bh + 18}`);
    mapSvg.setAttribute('width', W);
    mapSvg.setAttribute('aria-label', `query 头到 KV 头的对应：${Array.from({ length: H }, (_, i) => groupOf(i, H, KV, mode)).join(',')}`);
    mapSvg.append(svg('text', { x: pad, y: 11, 'font-size': 10, fill: 'var(--ink-3)' }, `query 头（${H} 个）`));
    mapSvg.append(svg('text', { x: pad, y: bottom + bh + 14, 'font-size': 10, fill: 'var(--ink-3)' }, `KV 头（${KV} 个），在 KV cache 里只存这些`));
    for (let i = 0; i < H; i++) {
      const g = groupOf(i, H, KV, mode);
      const x1 = pad + qw * (i + 0.5), x2 = pad + kw * (g + 0.5);
      mapSvg.append(svg('line', { x1, y1: top + bh, x2, y2: bottom, stroke: seriesColor(g % 8), 'stroke-width': 1.5, opacity: 0.8 }));
    }
    for (let i = 0; i < H; i++) {
      const color = seriesColor(groupOf(i, H, KV, 'interleave') % 8);
      mapSvg.append(svg('rect', { x: pad + qw * i + 1, y: top, width: Math.max(qw - 2, 1), height: bh, rx: 2, fill: color, opacity: 0.85 }));
    }
    for (let g = 0; g < KV; g++) {
      mapSvg.append(svg('rect', { x: pad + kw * g + 1, y: bottom, width: Math.max(kw - 2, 1), height: bh, rx: 2, fill: seriesColor(g % 8) }));
    }
  }

  function draw() {
    drawMap();
    const per = kvBytesPerToken(s);
    const seq = per * s.T, all = seq * s.batch;
    const wrong = misgrouped(s.H, s.KV).length;
    result.replaceChildren(...[
      h('div', {}, '每个 token：', h('b', {}, formatBytes(per)), `（2 × ${s.L} 层 × ${s.KV} 个 KV 头 × ${s.D} 维 × ${s.bytes} 字节）`),
      h('div', {}, `一条序列（${s.T.toLocaleString('en-US')} 个 token）：`, h('b', {}, formatBytes(seq)),
        `；整个 batch（×${s.batch}）：`, h('b', {}, formatBytes(all))),
      mode === 'tile' && wrong
        ? h('div', { class: 'w-label' }, (s.H > 32 ? `平铺时有 ${wrong} 个 query 头（共 ${s.H} 个）` : `平铺时连线交错：连线颜色与上排方块颜色不同的 ${wrong} 个头（共 ${s.H} 个）`)
          + '拿到的是别组的 K、V。显存一样，算出来的结果却不对（用 SDPA 训练出来的权重按相邻分组）。')
        : null,
    ].filter(Boolean));
    const mha = kvBytesPerToken({ ...s, KV: s.H }) * s.T * s.batch;
    const [unit, unitName] = chartUnit(mha);                 // MHA 那根柱子总是最高
    const bars = [
      { label: `MHA（${s.H}）`, value: mha / unit },
      { label: `当前（${s.KV}）`, value: all / unit, series: 1 },
      { label: 'MQA（1）', value: (kvBytesPerToken({ ...s, KV: 1 }) * s.T * s.batch) / unit, series: 2 },
    ];
    // 不用 highlight：三根柱子颜色已经各不相同，暗色主题下调低的不透明度又几乎看不见
    const opts = { bars, yLabel: `KV cache（${unitName}）`, height: 190,
      yFormat: (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toPrecision(2)) };
    if (chart) chart.update(opts); else chart = barChart(chartBox, opts);
  }

  const custom = () => { preset.value = 'custom'; };
  preset.addEventListener('change', () => { if (PRESETS[preset.value]) { Object.assign(s, PRESETS[preset.value]); syncControls(); draw(); } });
  ctl.L.addEventListener('input', () => { s.L = +ctl.L.value; outs.L.value = s.L; custom(); draw(); });
  ctl.H.addEventListener('change', () => { s.H = +ctl.H.value; if (s.H % s.KV) s.KV = s.H; syncControls(); custom(); draw(); });
  ctl.KV.addEventListener('change', () => { s.KV = +ctl.KV.value; custom(); draw(); });
  ctl.D.addEventListener('change', () => { s.D = +ctl.D.value; custom(); draw(); });
  ctl.T.addEventListener('input', () => { s.T = 2 ** +ctl.T.value; outs.T.value = s.T.toLocaleString('en-US'); custom(); draw(); });
  ctl.batch.addEventListener('input', () => { s.batch = +ctl.batch.value; outs.batch.value = s.batch; custom(); draw(); });
  ctl.bytes.addEventListener('change', () => { s.bytes = +ctl.bytes.value; custom(); draw(); });
  onThemeChange(() => drawMap());                  // the map's lines and blocks use the palette; the bar chart redraws itself
  let lastWidth = 0;
  new ResizeObserver(() => { if (box.clientWidth !== lastWidth) { lastWidth = box.clientWidth; drawMap(); } }).observe(box);
  syncControls();
  draw();
}
