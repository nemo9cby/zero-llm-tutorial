/* 第 5 章各交互图共用的小工具：控件（滑块、按钮组、单选组）、样式、确定性的随机数。
   只依赖 _lib.js；颜色一律用 CSS 变量。 */
import { h } from './_lib.js';

let styled = false;
export function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch05 { font-size: .9rem; color: var(--ink); }
.w-ch05 .ctl { display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; align-items: center; margin: .4rem 0 .6rem; }
.w-ch05 .ctl label { display: inline-flex; align-items: center; gap: .45rem; color: var(--ink-2); font-size: .86rem; }
.w-ch05 .ctl input[type=range] { width: 9rem; max-width: 46vw; accent-color: var(--ai); }
.w-ch05 .ctl select { font: inherit; font-size: .86rem; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .1rem .3rem; }
.w-ch05 .ctl .val { font-variant-numeric: tabular-nums; color: var(--ink); min-width: 2.6em; display: inline-block; }
.w-ch05 .btns { display: flex; flex-wrap: wrap; gap: .4rem; }
.w-ch05 .btns button { font-size: .84rem; }
.w-ch05 button[aria-pressed=true] { background: var(--ai-2); border-color: var(--ai); }
.w-ch05 .readout { font-variant-numeric: tabular-nums; color: var(--ink-2); font-size: .86rem; line-height: 1.7; }
.w-ch05 .readout b { color: var(--ink); font-weight: 600; }
.w-ch05 .panel-title { font-size: .84rem; color: var(--ink-2); margin: .2rem 0 .3rem; }
.w-ch05 svg text { font-family: var(--serif); }
.w-ch05 .num { font-variant-numeric: tabular-nums; }
`));
}

export const reducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

/** A labelled range slider. Returns {el, input, set(v), get()}. */
export function slider({ label, min, max, step = 1, value, format = (v) => String(v), onInput, aria }) {
  const out = h('span', { class: 'val' }, format(value));
  const input = h('input', { type: 'range', min, max, step, value, 'aria-label': aria || label });
  input.addEventListener('input', () => { out.textContent = format(+input.value); onInput?.(+input.value); });
  const el = h('label', {}, label, input, out);
  return {
    el, input,
    get: () => +input.value,
    set(v) { input.value = v; out.textContent = format(+input.value); },
  };
}

/** A row of mutually exclusive toggle buttons (aria-pressed). */
export function radioButtons(options, value, onChange, aria) {
  const wrap = h('div', { class: 'btns', role: 'group', 'aria-label': aria || '' });
  const buttons = options.map(([key, text]) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(key === value) }, text);
    b.addEventListener('click', () => {
      for (const o of buttons) o.setAttribute('aria-pressed', String(o === b));
      onChange(key);
    });
    return b;
  });
  wrap.append(...buttons);
  return wrap;
}

export function button(text, onClick, aria) {
  const b = h('button', { type: 'button', 'aria-label': aria || text }, text);
  b.addEventListener('click', onClick);
  return b;
}

/** Deterministic PRNG (mulberry32) and a standard-normal sampler on top of it. */
export function rng(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(uniform(), 1e-12), v = uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { uniform, normal };
}

export const fmtInt = (n) => Math.round(n).toLocaleString('en-US');

/** Human-readable bytes with binary units, e.g. 4718592 → "4.5 MiB". */
export function formatBytes(b) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (b >= 1024 && i < units.length - 1) { b /= 1024; i += 1; }
  const s = i === 0 ? String(Math.round(b)) : (b >= 100 ? b.toFixed(0) : b >= 10 ? b.toFixed(1) : b.toFixed(2));
  return `${s} ${units[i]}`;
}

export const NS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const k of kids.flat()) if (k != null) el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return el;
}
