/* Shared helpers for the chapter 2 widgets (not a widget itself).
   Pure math is exported so tests/js/ch02-widgets.test.mjs can check it in Node against the
   numbers the page's verify cells assert. */
import { h } from './_lib.js';

/** Seeded uniform RNG in [0, 1) (mulberry32). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample (Box–Muller). */
export function gaussian(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/** Gamma(shape, 1) sample (Marsaglia–Tsang); shape ≥ 1. */
function gamma(shape, rand) {
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = gaussian(rand); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = rand();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** Chi-square sample with k ≥ 2 degrees of freedom: |q|² for q ~ N(0, I_k). */
export function chiSquare(k, rand) {
  return 2 * gamma(k / 2, rand);
}

/** Row softmax of a plain array. naive = true skips the max subtraction (the book's 01 script). */
export function softmax(xs, naive = false) {
  const m = naive ? 0 : Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export const range = (n) => [...Array(n).keys()];

let styled = false;
/** Inject the chapter-2 widget styles once (all class names start with w-ch02-). */
export function injectStyle() {
  if (styled || typeof document === 'undefined') return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch02-controls { display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; align-items: center; margin: .2rem 0 .6rem; font-size: .88rem; }
.w-ch02-controls label { display: inline-flex; align-items: center; gap: .4rem; color: var(--ink-2); }
.w-ch02-controls input[type=range] { width: 7.5rem; accent-color: var(--ai); }
.w-ch02-controls select, .w-ch02-controls input[type=number] { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .1rem .3rem; }
.w-ch02-controls input[type=number] { width: 5.6rem; }
.w-ch02-num { font-variant-numeric: tabular-nums; color: var(--ink); min-width: 2.2em; display: inline-block; }
.w-ch02-seg { display: inline-flex; flex-wrap: wrap; gap: .3rem; }
.w-ch02-controls button[aria-pressed=true] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-ch02-big { font-size: 1.25rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.w-ch02-ok { color: var(--moss); }
.w-ch02-bad { color: var(--kincha); }
.w-ch02-note { font-size: .82rem; color: var(--ink-3); margin: .3rem 0; }
.w-ch02-bars { display: grid; grid-template-columns: fit-content(14em) minmax(3em, 1fr) 4.2em; gap: .25rem .5rem; align-items: center; font-size: .82rem; margin: .4rem 0; }
.w-ch02-bars .lab { color: var(--ink-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.w-ch02-bars .lab b { color: var(--ink); font-weight: 600; font-family: var(--mono); font-size: .78rem; }
.w-ch02-bars .track { height: .8rem; background: var(--paper-2); border-radius: 2px; overflow: hidden; }
.w-ch02-bars .fill { height: 100%; background: var(--ai); border-radius: 2px; }
.w-ch02-bars .fill.top { background: var(--plum); }
.w-ch02-bars .pct { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); }
.w-ch02-strip { display: grid; gap: 1px; margin: .2rem 0 .5rem; }
.w-ch02-strip span { height: 1.1rem; border-radius: 1px; }
.w-ch02-grid { display: grid; gap: 2px; font: 11px/1 var(--mono); margin: .4rem 0; }
.w-ch02-grid .cell { display: grid; place-items: center; min-height: 1.9rem; border-radius: 2px; font-variant-numeric: tabular-nums; }
.w-ch02-grid .head { color: var(--ink-3); white-space: nowrap; overflow: hidden; }
.w-ch02-grid .masked { background: var(--paper-3); color: var(--ink-3); }
.w-ch02-grid .head.w-ch02-bad { color: var(--kincha); font-weight: 600; }
.w-ch02-grid .nan { background: transparent; color: var(--kincha); border: 1px dashed var(--kincha); font-weight: 600; }
.w-ch02-chips { display: flex; flex-wrap: wrap; gap: .3rem; margin: .3rem 0; font-size: .8rem; }
.w-ch02-chips span { padding: .05rem .45rem; border-radius: 3px; border: 1px solid var(--rule); }
.w-ch02-chips .ok { color: var(--moss); border-color: var(--moss); }
.w-ch02-chips .bad { color: var(--kincha); border-color: var(--kincha); font-weight: 600; }
.w-ch02-chips .muted { color: var(--ink-3); }
.w-ch02-two { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: .4rem 1.2rem; }
.w-ch02-table { font-size: .84rem; margin: .4rem 0; display: table; }
.w-ch02-table td, .w-ch02-table th { padding: .15rem .8rem .15rem 0; }
.w-ch02-table td.n { text-align: right; font-variant-numeric: tabular-nums; }
.w-ch02-stack { display: flex; height: 1rem; border-radius: 2px; overflow: hidden; margin: .4rem 0 .2rem; }
.w-ch02-legend { display: flex; flex-wrap: wrap; gap: .2rem .9rem; font-size: .78rem; color: var(--ink-2); }
.w-ch02-legend i { display: inline-block; width: .7rem; height: .7rem; border-radius: 2px; margin-right: .3rem; vertical-align: -1px; }
@media (prefers-reduced-motion: no-preference) { .w-ch02-bars .fill { transition: width .15s ease-out; } }
`));
}
