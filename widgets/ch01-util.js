/* 第 1 章组件共用的小工具（不是组件，没有 mount）。纯函数部分没有 DOM 依赖，Node 测试可以直接 import。 */
import { h, visible } from './_lib.js';

const STRICT = new TextDecoder('utf-8', { fatal: true });

/** A token's bytes as chip text: the decoded string with visible whitespace, or hex if it is not valid UTF-8 alone. */
export function tokenLabel(bytes) {
  try {
    return { text: visible(STRICT.decode(bytes)), partial: false };
  } catch (e) {
    return { text: [...bytes].map((b) => b.toString(16).padStart(2, '0')).join(' '), partial: true };
  }
}

export const hex = (bytes) => [...bytes].map((b) => b.toString(16).padStart(2, '0')).join(' ');

/** Python-like repr of a short byte string, for tables: 'is ' or b'\xe2\x80'. */
export function bytesRepr(bytes) {
  const t = tokenLabel(bytes);
  return t.partial ? `b'${[...bytes].map((b) => `\\x${b.toString(16).padStart(2, '0')}`).join('')}'` : `'${t.text}'`;
}

export const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

const SCOPE = 'figure.widget[data-widget^="ch01-"]';
const DARK_BG = ['#22385a', '#1d4634', '#4a3b1d', '#41304e', '#1d4450', '#4d2a2a'];
const DARK_CHIPS = (root) => DARK_BG.map((bg, i) => `${root} ${SCOPE} .tok.c${i} { background: ${bg}; }`).join('\n');

let styled = false;
/** Styles shared by the chapter-1 widgets (prefix w-ch01-). Safe to call more than once; no-op without a DOM. */
export function injectStyle() {
  if (styled || typeof document === 'undefined') return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch01-bar { display: flex; flex-wrap: wrap; gap: .45rem .9rem; align-items: center; margin: .3rem 0 .55rem; font-size: .86rem; }
.w-ch01-bar label { display: inline-flex; align-items: center; gap: .35rem; color: var(--ink-2); }
.w-ch01-bar select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .1rem .3rem; max-width: 100%; }
.w-ch01-bar input[type=checkbox] { accent-color: var(--ai); }
.w-ch01-seg { display: inline-flex; flex-wrap: wrap; gap: .3rem; }
.w-ch01-seg button[aria-pressed=true] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-ch01-presets { display: flex; flex-wrap: wrap; gap: .3rem; margin: .2rem 0 .45rem; }
.w-ch01-presets button { font-size: .78rem; padding: .12rem .5rem; }
textarea.w-input { resize: vertical; min-height: 3.2em; display: block; }
.w-ch01-stats { font-size: .86rem; color: var(--ink-2); margin: .2rem 0 .4rem; font-variant-numeric: tabular-nums; }
.w-ch01-stats b { color: var(--ink); font-weight: 650; }
.w-ch01-ok { color: var(--moss); font-weight: 600; }
.w-ch01-bad { color: var(--kincha); font-weight: 600; }
.w-ch01-note { font-size: .8rem; color: var(--ink-3); margin: .25rem 0; }
.w-ch01-br { flex-basis: 100%; height: 0; }
.tokens .tok.w-ch01-special { outline: 1.5px dashed var(--shu); outline-offset: -1px; }
.tokens .tok.w-ch01-partial b { font-size: 11px; letter-spacing: -.02em; }
.tokens .tok.w-ch01-gap { background: transparent; border: 1px dashed var(--ink-3); color: var(--ink-3); }
.w-ch01-tokens { max-height: 22rem; overflow-y: auto; }
.w-ch01-table { width: 100%; border-collapse: collapse; font-size: .8rem; margin: .3rem 0 .6rem; display: table; }
.w-ch01-table th, .w-ch01-table td { border-bottom: 1px solid var(--rule); padding: .18rem .4rem; text-align: left; vertical-align: top; }
.w-ch01-table th { color: var(--ink-3); font-weight: 600; white-space: nowrap; }
.w-ch01-table td code, .w-ch01-mono { font-family: var(--mono); font-size: .78rem; white-space: pre-wrap; word-break: break-all; }
.w-ch01-table tr.pick td { background: color-mix(in srgb, var(--ai-2) 70%, transparent); }
.w-ch01-table td.diff { color: var(--kincha); font-weight: 650; }
.w-ch01-table tr.first-diff td { box-shadow: inset 0 2px 0 var(--kincha); }
.w-ch01-num { text-align: right !important; font-variant-numeric: tabular-nums; white-space: nowrap; }
.w-ch01-wrap { overflow-x: auto; max-width: 100%; }
.w-ch01-progress { height: .55rem; background: var(--paper-3); border-radius: 3px; overflow: hidden; margin: .35rem 0; }
.w-ch01-progress > div { height: 100%; width: 0; background: var(--ai); }
.w-ch01-legend { display: flex; flex-wrap: wrap; gap: .3rem .8rem; font-size: .8rem; color: var(--ink-2); margin: .2rem 0 .4rem; }
.w-ch01-legend label { display: inline-flex; gap: .3rem; align-items: center; }
.w-ch01-swatch { display: inline-block; width: .85rem; height: .85rem; border-radius: 2px; vertical-align: middle; }
.w-ch01-bars { display: grid; grid-template-columns: minmax(6.5em, max-content) minmax(3em, 1fr) 3.2em; gap: .22rem .5rem; align-items: center; font-size: .82rem; margin: .4rem 0; }
.w-ch01-bars .lab { color: var(--ink-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.w-ch01-bars .lab small { color: var(--ink-3); }
.w-ch01-bars .track { height: .75rem; background: var(--paper-2); border-radius: 2px; overflow: hidden; }
.w-ch01-bars .fill { height: 100%; background: var(--ink-3); border-radius: 2px; }
.w-ch01-bars .fill.me { background: var(--ai); }
.w-ch01-bars .val { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); }
.w-ch01-history { display: flex; flex-direction: column; gap: .15rem; font-size: .8rem; margin: .3rem 0; max-height: 11rem; overflow-y: auto; }
.w-ch01-history div { font-variant-numeric: tabular-nums; }
/* JetBrains Mono draws <| and |> as triangles: in a tokenizer widget every character must look like itself. */
${SCOPE} :is(.tokens, .w-input, code, .w-ch01-mono) { font-variant-ligatures: none; font-feature-settings: "liga" 0, "calt" 0; }
/* Dark chips with light text (the shared dark style dims the text together with the background). */
:root[data-theme="dark"] ${SCOPE} .tok { filter: none; color: var(--ink); }
${DARK_CHIPS(':root[data-theme="dark"]')}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) ${SCOPE} .tok { filter: none; color: var(--ink); }
  ${DARK_CHIPS(':root:not([data-theme="light"])')}
}
`));
}

/** A row of preset buttons; onPick(preset) is called with the clicked entry. */
export function presetButtons(presets, onPick, label = '预设') {
  return h('div', { class: 'w-ch01-presets', role: 'group', 'aria-label': label },
    presets.map((p) => h('button', { type: 'button', onclick: () => onPick(p) }, p.label)));
}

/** Segmented control: [{value, label}], returns {el, set(value)}; onChange(value) on click. */
export function segmented(options, value, onChange, label) {
  const buttons = options.map((o) => h('button', {
    type: 'button', 'aria-pressed': String(o.value === value),
    onclick: () => { set(o.value); onChange(o.value); },
  }, o.label));
  const el = h('div', { class: 'w-ch01-seg', role: 'group', 'aria-label': label }, buttons);
  function set(v) { buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(options[i].value === v))); }
  return { el, set };
}
