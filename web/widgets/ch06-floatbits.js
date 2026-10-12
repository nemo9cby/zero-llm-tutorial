/* 图：一个数在 fp32、fp16、bf16 下的位（6.4 节）。
   fp16：由 JS 的 double 直接就近舍入（平局取偶），与 np.float16(x) 相同；
   bf16：先舍入成 fp32，再对 fp32 的位做就近舍入（与验证 6.4-d 的 to_bf16、以及 PyTorch 相同）。
   转换函数都导出，tests/js/ch06-floatbits.test.mjs 用 NumPy 生成的样例逐位对拍。 */
import { h } from './_lib.js';

const buf = new DataView(new ArrayBuffer(4));

export function f32Bits(x) {
  buf.setFloat32(0, x);
  return buf.getUint32(0);
}
export function f32FromBits(u) {
  buf.setUint32(0, u >>> 0);
  return buf.getFloat32(0);
}

function roundHalfEven(v) {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** double → fp16 的位（就近舍入、平局取偶；含次正规数、上溢到 inf）。 */
export function f16Bits(x) {
  if (Number.isNaN(x)) return 0x7e00;
  const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  const a = Math.abs(x);
  if (a === Infinity) return sign | 0x7c00;
  if (a === 0) return sign;
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e -= 1;
  else if (2 ** (e + 1) <= a) e += 1;
  if (e < -14) return sign | roundHalfEven(a * 2 ** 24);       // 次正规数：a = m · 2^-24（m = 1024 时恰好进位成最小正规数）
  let m = roundHalfEven((a / 2 ** e - 1) * 1024);                // 这几步在 double 里都是精确的
  let E = e + 15;
  if (m === 1024) { m = 0; E += 1; }
  if (E >= 31) return sign | 0x7c00;                              // 上溢
  return sign | (E << 10) | m;
}
export function f16FromBits(b) {
  const s = b & 0x8000 ? -1 : 1;
  const E = (b >> 10) & 0x1f;
  const m = b & 0x3ff;
  if (E === 0x1f) return m ? NaN : s * Infinity;
  if (E === 0) return s * m * 2 ** -24;
  return s * (1 + m / 1024) * 2 ** (E - 15);
}

/** fp32 的位 → bf16 的位（与 PyTorch 的 round_to_nearest_even 相同）。 */
export function bf16BitsFromF32Bits(u) {
  if ((u & 0x7fffffff) >>> 0 > 0x7f800000) return ((u >>> 16) & 0x8000) | 0x7fc0;   // NaN
  const lsb = (u >>> 16) & 1;
  return ((u + 0x7fff + lsb) >>> 16) & 0xffff;
}
export const bf16Bits = (x) => bf16BitsFromF32Bits(f32Bits(x));
export const bf16FromBits = (b) => f32FromBits((b & 0xffff) << 16);

export const FORMATS = [
  { name: 'fp32', E: 8, M: 23, bits: (x) => f32Bits(x), value: (b) => f32FromBits(b), maxNorm: 3.4028234663852886e38 },
  { name: 'fp16', E: 5, M: 10, bits: (x) => f16Bits(x), value: (b) => f16FromBits(b), maxNorm: 65504 },
  { name: 'bf16', E: 8, M: 7, bits: (x) => bf16Bits(x), value: (b) => bf16FromBits(b), maxNorm: 3.3895313892515355e38 },
];

/** 解析输入：支持 1e-8、70000、pi、1/3 这样的写法。 */
export function parseInput(text) {
  const t = text.trim().toLowerCase().replace(/π/g, 'pi').replace(/\s+/g, '');
  if (t === 'pi' || t === '-pi') return (t[0] === '-' ? -1 : 1) * Math.PI;
  const frac = t.match(/^(-?[\d.]+(?:e[-+]?\d+)?)\/([\d.]+(?:e[-+]?\d+)?)$/);
  if (frac) return Number(frac[1]) / Number(frac[2]);
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$|^[-+]?inf(inity)?$/.test(t)) return NaN;
  return t.includes('inf') ? (t[0] === '-' ? -Infinity : Infinity) : Number(t);
}

const fmtVal = (v) => {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : (v > 0 ? 'inf' : '-inf');
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(8).replace(/\.?0+e/, 'e');
  return String(+v.toPrecision(12));
};

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch06-fb .inrow { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin-bottom: .6rem; }
.w-ch06-fb .inrow input { flex: 1 1 10rem; min-width: 0; }
.w-ch06-fb .presets { display: flex; flex-wrap: wrap; gap: .3rem; }
.w-ch06-fb .presets button { font: 12.5px var(--mono); padding: .1rem .45rem; }
.w-ch06-fb .fmt { margin: .7rem 0 .9rem; }
.w-ch06-fb .fmt-head { display: flex; flex-wrap: wrap; gap: .2rem .9rem; align-items: baseline; font-size: .88rem; color: var(--ink-2); }
.w-ch06-fb .fmt-head b { color: var(--ink); font-size: .95rem; }
.w-ch06-fb .fmt-head .val { font-family: var(--mono); color: var(--ink); }
.w-ch06-fb .bits { display: flex; flex-wrap: wrap; gap: 1px; margin: .3rem 0 .2rem; font: 11px/1 var(--mono); }
.w-ch06-fb .bit { width: 10px; height: 18px; display: grid; place-items: center; border-radius: 2px; color: var(--ink); }
.w-ch06-fb .bit.s { background: color-mix(in srgb, var(--plum) 30%, var(--paper)); }
.w-ch06-fb .bit.e { background: color-mix(in srgb, var(--ai) 28%, var(--paper)); }
.w-ch06-fb .bit.m { background: color-mix(in srgb, var(--moss) 24%, var(--paper)); }
.w-ch06-fb .bit.gap { margin-left: 4px; }
.w-ch06-fb .flag { font-size: .8rem; padding: 0 .35rem; border-radius: 3px; border: 1px solid var(--kincha); color: var(--kincha); }
.w-ch06-fb .key { display: flex; flex-wrap: wrap; gap: .2rem 1rem; font-size: .8rem; color: var(--ink-2); }
.w-ch06-fb .key span::before { content: ''; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; background: var(--c); }
`));
}

export function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch06-fb');
  const input = h('input', { class: 'w-input', value: String(config.value ?? 1000), 'aria-label': '输入一个数，例如 1000、1e-8、70000、pi、1/3' });
  const presets = h('div', { class: 'presets', role: 'group', 'aria-label': '常用的例子' },
    ['1000', '0.01', '1e-8', '3e-8', '65504', '65520', '70000', 'pi', '1/3'].map((v) =>
      h('button', { type: 'button', onclick: () => { input.value = v; render(); } }, v)));
  const out = h('div', { 'aria-live': 'polite' });
  root.append(h('div', { class: 'inrow' }, input), presets,
    h('div', { class: 'key', style: 'margin-top:.6rem' },
      h('span', { style: '--c:color-mix(in srgb, var(--plum) 30%, var(--paper))' }, '符号'),
      h('span', { style: '--c:color-mix(in srgb, var(--ai) 28%, var(--paper))' }, '指数'),
      h('span', { style: '--c:color-mix(in srgb, var(--moss) 24%, var(--paper))' }, '尾数')),
    out);

  function render() {
    const x = parseInput(input.value);
    if (Number.isNaN(x) && !/nan/i.test(input.value)) {
      out.replaceChildren(h('p', { class: 'w-label' }, '请输入一个数，例如 1000、1e-8、70000、pi 或 1/3。'));
      return;
    }
    out.replaceChildren(...FORMATS.map((f) => {
      const b = f.bits(x);
      const v = f.value(b);
      const n = 1 + f.E + f.M;
      const bitEls = [];
      for (let i = n - 1; i >= 0; i--) {
        const pos = n - 1 - i;
        const cls = pos === 0 ? 's' : pos <= f.E ? 'e' : 'm';
        const gap = pos === 1 || pos === 1 + f.E ? ' gap' : '';
        bitEls.push(h('span', { class: `bit ${cls}${gap}` }, String((b >>> i) & 1)));
      }
      const expField = (b >>> f.M) & ((1 << f.E) - 1);
      const flags = [];
      if (Number.isFinite(x) && !Number.isFinite(v)) flags.push('上溢 → inf');
      else if (x !== 0 && Number.isFinite(x) && v === 0) flags.push('下溢 → 0');
      else if (expField === 0 && v !== 0) flags.push('次正规数');
      const rel = Number.isFinite(v) && x !== 0 && Number.isFinite(x) ? Math.abs(v - x) / Math.abs(x) : null;
      // 相邻两个可表示数的间隔（ulp）：2^(指数 - bias - M)，次正规数区间按最小指数算
      const bias = 2 ** (f.E - 1) - 1;
      const ulp = Number.isFinite(v) ? 2 ** (Math.max(expField, 1) - bias - f.M) : null;
      return h('div', { class: 'fmt' },
        h('div', { class: 'fmt-head' }, h('b', {}, f.name),
          h('span', {}, `0x${b.toString(16).toUpperCase().padStart(n / 4, '0')}`),
          h('span', {}, '存下来的值 ', h('span', { class: 'val' }, fmtVal(v))),
          rel != null ? h('span', {}, `相对误差 ${rel === 0 ? '0' : rel.toExponential(2)}`) : null,
          ulp != null ? h('span', {}, `这一段的间隔 ${fmtVal(ulp)}`) : null,
          ...flags.map((t) => h('span', { class: 'flag' }, t))),
        h('div', { class: 'bits', 'aria-label': `${f.name} 的 ${n} 个二进制位` }, bitEls));
    }));
  }
  input.addEventListener('input', render);
  render();
}
