/* 图：KV cache 显存计算器。每个 token 占 2 × L × H_kv × D × b 字节，总量再乘 token 数和 batch。 */
import { h } from './_lib.js';
import { injectStyle, fmtInt, formatBytes } from './ch05-common.js';

export const kvBytesPerToken = (L, Hkv, D, b) => 2 * L * Hkv * D * b;

// 三台机器人的配置都取自原书代码（ch05/05_kvcache.py、ch06/05_pretrain.py、ch09/01_pretrain.py）
export const PRESETS = {
  demo: { name: '05 的演示模型（fp32）', L: 6, H: 6, D: 64, b: 4, T: 256, B: 1, params: 11_389_824 },
  storybot: { name: 'StoryBot（fp32）', L: 4, H: 16, D: 32, b: 4, T: 256, B: 1, params: 22_696_448 },
  storybot16: { name: 'StoryBot（bf16）', L: 4, H: 16, D: 32, b: 2, T: 256, B: 1, params: 22_696_448 },
  webbot: { name: 'WebBot，GQA 4 个 KV 头（bf16）', L: 12, H: 4, D: 64, b: 2, T: 1024, B: 64, params: 152_316_672 },
  webbotMHA: { name: 'WebBot 若不用 GQA（bf16）', L: 12, H: 12, D: 64, b: 2, T: 1024, B: 64, params: 161_753_856 },
};

export function mount(root, config = {}) {
  injectStyle();
  const box = h('div', { class: 'w-ch05 w-ch05-kvmem' });
  root.append(box);
  box.append(h('style', {}, `
.w-ch05-kvmem .fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr)); gap: .5rem .9rem; margin: .5rem 0; }
.w-ch05-kvmem .fields label { display: flex; flex-direction: column; gap: .15rem; font-size: .8rem; color: var(--ink-2); }
.w-ch05-kvmem .fields input, .w-ch05-kvmem .fields select { font: 14px/1.4 var(--mono); color: var(--ink); background: var(--paper);
  border: 1px solid var(--rule); border-radius: 4px; padding: .25rem .4rem; width: 100%; }
.w-ch05-kvmem .formula { font: 13.5px/1.7 var(--mono); background: var(--paper-2); border-radius: 4px; padding: .5rem .7rem; overflow-x: auto; }
.w-ch05-kvmem .meter { height: 10px; border-radius: 5px; background: var(--paper-3); overflow: hidden; margin: .35rem 0 .1rem; }
.w-ch05-kvmem .meter > div { height: 100%; background: var(--ai); }
`));
  const presetSel = h('select', { 'aria-label': '预设配置' },
    Object.entries(PRESETS).map(([k, p]) => h('option', { value: k, selected: k === (config.preset || 'storybot') }, p.name)));
  const field = (label, key, attrs) => {
    const input = h('input', { type: 'number', min: 1, step: 1, ...attrs, 'aria-label': label });
    input.addEventListener('input', update);
    return [key, input, h('label', {}, label, input)];
  };
  const fields = [
    field('层数 L', 'L', { max: 200 }), field('KV 头数 H_kv', 'H', { max: 128 }), field('head_dim D', 'D', { max: 1024 }),
    field('token 数', 'T', { max: 10_000_000 }), field('batch（同时生成的序列数）', 'B', { max: 4096 }),
  ];
  const bytesSel = h('select', { 'aria-label': '每个数的字节数' },
    h('option', { value: 4 }, '4（fp32）'), h('option', { value: 2 }, '2（bf16 / fp16）'), h('option', { value: 1 }, '1（int8 / fp8）'));
  bytesSel.addEventListener('change', update);
  const inputs = Object.fromEntries(fields.map(([k, input]) => [k, input]));
  const out = h('div', { class: 'formula', 'aria-live': 'polite' });
  const compare = h('div', { class: 'readout' });
  box.append(h('div', { class: 'ctl' }, h('label', {}, '预设', presetSel)),
    h('div', { class: 'fields' }, ...fields.map((f) => f[2]), h('label', {}, '每个数的字节数 b', bytesSel)), out, compare);

  let params = 0;
  function load(key) {
    const p = PRESETS[key];
    for (const k of ['L', 'H', 'D', 'T', 'B']) inputs[k].value = p[k];
    bytesSel.value = p.b;
    params = p.params;
    update();
  }
  presetSel.addEventListener('change', () => load(presetSel.value));

  function update() {
    const v = (k) => Math.max(0, Math.floor(+inputs[k].value || 0));
    const L = v('L'), H = v('H'), D = v('D'), T = v('T'), B = v('B'), b = +bytesSel.value;
    const per = kvBytesPerToken(L, H, D, b), total = per * T * B;
    out.replaceChildren(
      h('div', {}, `每个 token：2 × ${L} × ${H} × ${D} × ${b} = ${fmtInt(per)} 字节`),
      h('div', {}, `全部：${fmtInt(per)} × ${fmtInt(T)} 个 token × batch ${fmtInt(B)} = ${formatBytes(total)}`));
    const preset = PRESETS[presetSel.value];
    const same = ['L', 'H', 'D'].every((k) => v(k) === preset[k]);
    if (same && params) {
      const weights = params * b;
      const frac = total / (total + weights);
      compare.replaceChildren(
        h('div', {}, `同精度下模型权重 ${fmtInt(params)} × ${b} B = `, h('b', {}, formatBytes(weights)),
          '；KV cache 是它的 ', h('b', {}, (total / weights).toFixed(total / weights < 0.1 ? 3 : 2)), ' 倍'),
        h('div', { class: 'meter', role: 'img', 'aria-label': `KV cache 占二者之和的 ${(100 * frac).toFixed(0)}%` },
          h('div', { style: { width: `${(100 * frac).toFixed(1)}%` } })));
    } else {
      compare.textContent = '（改动了层数、头数或 head_dim，不再与预设的模型权重比较）';
    }
  }
  load(presetSel.value);
}
