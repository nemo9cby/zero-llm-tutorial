/* 图 7-1：同一段文字，交给三台机器人的分词器（CodeBot 1,000 / StoryBot 10,000 / WebBot 50,000），
   再加一行“字节级”（不用分词器）作基线。
   config: { "presets": "data/ch07/presets.json", "preset": 0 }
   compare() is pure (no DOM) so tests/js/ch07-compare.test.mjs can check it against the book's Python code. */
import { h, chip, fmt, visible } from './_lib.js';
import { Tokenizer, utf8Encode } from './_bpe.js';

export const BOTS = Object.freeze([
  { key: 'codebot', label: 'CodeBot', corpus: '代码', url: 'data/tokenizers/codebot.json' },
  { key: 'storybot', label: 'StoryBot', corpus: '儿童故事', url: 'data/tokenizers/storybot.json' },
  { key: 'webbot', label: 'WebBot', corpus: '网页文章', url: 'data/tokenizers/webbot.json' },
]);

const MAX_CHIPS = 300;          // per row; longer texts show a count instead of every chip
const MAX_CHIPS_NARROW = 80;    // on phones (every preset still fits: the longest row has 78 tokens)
const MAX_LINES = 12;           // the textarea grows with its text up to this many lines, then scrolls
const strict = new TextDecoder('utf-8', { fatal: true });
let mounted = 0;                // unique ids when a page mounts the widget more than once

/**
 * Tokenize one text with every tokenizer.
 * @param {Tokenizer[]} toks
 * @param {string} text
 * @param {{special?: boolean}} [opts] special: false treats "<|endoftext|>" as plain text (tiktoken's encode_ordinary)
 * @returns {{bytes: number, rows: {ids: number[], chunks: object[], tokens: number, bytesPerToken: number}[]}}
 */
export function compare(toks, text, { special = true } = {}) {
  const bytes = utf8Encode(text).length;
  const rows = toks.map((tok) => {
    const chunks = tok.encodeChunks(text, { special });
    const ids = chunks.flatMap((c) => c.ids);
    return { ids, chunks, tokens: ids.length, bytesPerToken: ids.length ? bytes / ids.length : 0 };
  });
  return { bytes, rows };
}

/** How a token looks on a chip: its text, or Python-style escapes (\xe4\xb8) when its bytes are not valid UTF-8. */
export function tokenLabel(tok, id) {
  const bytes = tok.tokenBytes(id);
  try {
    return { text: strict.decode(bytes), partial: false };
  } catch {
    return { text: [...bytes].map((b) => `\\x${b.toString(16).padStart(2, '0')}`).join(''), partial: true };
  }
}

/**
 * The status-line text for one chip: which token it is, its id and its bytes.
 * @param {string} bot  label of the tokenizer ("WebBot")
 * @param {number} index  0-based position of the token in the row
 * @param {number} id
 * @param {Uint8Array} bytes
 * @param {{special?: boolean, partial?: boolean, text?: string}} lab
 */
export function tokenInfo(bot, index, id, bytes, lab) {
  const head = `${bot} 的第 ${fmt(index + 1)} 个 token：id ${fmt(id)}`;
  if (lab.special) return `${head}，特殊 token ${lab.text}`;
  const hex = [...bytes].map((x) => x.toString(16).padStart(2, '0')).join(' ');
  // spaces, newlines and tabs shown as on the chips (␣ ↵ ⇥)
  return `${head}，字节 ${hex}` + (lab.partial ? '（单独不是完整的 UTF-8 字符）' : `（“${visible(lab.text)}”）`);
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch07-presets { display: flex; flex-wrap: wrap; gap: .35rem; margin: 0 0 .5rem; }
.w-ch07-presets button[aria-pressed="true"] { border-color: var(--ai); color: var(--ai); font-weight: 600; }
.w-ch07-text { min-height: 4.6rem; resize: vertical; font-size: 14px; overflow-y: hidden; }
.w-ch07-opt { display: flex; align-items: flex-start; gap: .4rem; font-size: .85rem; color: var(--ink-2); margin: .3rem 0 .6rem; }
.w-ch07-opt input { margin-top: .3rem; flex: none; }
.w-ch07-row { margin: .7rem 0 .2rem; }
.w-ch07-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .1rem .8rem; font-size: .88rem; }
.w-ch07-name { font-weight: 700; color: var(--ink); }
.w-ch07-meta { color: var(--ink-3); font-size: .8rem; margin-left: .4rem; font-weight: 400; }
.w-ch07-count { color: var(--ink-2); font-variant-numeric: tabular-nums; }
.w-ch07-count b { color: var(--ink); }
.w-ch07-bar { height: 5px; background: var(--paper-3); border-radius: 3px; margin: .25rem 0 .1rem; overflow: hidden; }
.w-ch07-bar span { display: block; height: 100%; background: var(--ai); border-radius: 3px; }
.w-ch07-bar.base span { background: var(--ink-3); }
.w-ch07-row .tokens { margin: .3rem 0 .2rem; }
.w-ch07-row .tok.partial b { font-size: 11px; color: var(--ink-2); }
.w-ch07-row .tok.special { outline: 1.5px solid var(--plum); outline-offset: -1.5px; }
.w-ch07-row .tok { cursor: default; }
.w-ch07-row .tok:focus-visible { outline: 2px solid var(--ai); outline-offset: 1px; }
.w-ch07-row .tok.picked { box-shadow: 0 0 0 2px var(--ai); }
.w-ch07-more { font-size: .8rem; color: var(--ink-3); }
.w-ch07-pick { margin: .6rem 0 0; padding: .35rem 0; background: var(--paper);
  border-top: 1px solid var(--rule); font-size: .85rem; color: var(--ink-2); overflow-wrap: anywhere; min-height: 1.5em; }
.w-ch07-pick.on { position: sticky; bottom: 0; z-index: 1; }
.w-ch07-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
`));
}

export async function mount(root, config = {}) {
  injectStyle();
  root.classList.add('w-ch07');
  const status = h('p', { class: 'w-label' }, '正在加载三个分词器（WebBot 的合并表约 450 KB）……');
  root.append(status);
  let toks, presets;
  try {
    [toks, presets] = await Promise.all([
      Promise.all(BOTS.map((b) => Tokenizer.load(b.url))),
      fetch(config.presets || 'data/ch07/presets.json').then((r) => {
        if (!r.ok) throw new Error(`presets: HTTP ${r.status}`);
        return r.json();
      }).then((d) => d.presets),
    ]);
  } catch (err) {
    status.textContent = `分词器加载失败：${err.message}`;
    return;
  }
  status.remove();

  const text = h('textarea', { class: 'w-input w-ch07-text', 'aria-label': '输入任意文字，比较三个分词器', spellcheck: 'false' });
  const presetBar = h('div', { class: 'w-ch07-presets', role: 'group', 'aria-label': '示例文字' });
  const presetButtons = presets.map((p, i) => {
    const b = h('button', { type: 'button', 'aria-pressed': 'false', onclick: () => choose(i) }, p.name);
    presetBar.append(b);
    return b;
  });
  const plain = h('input', { type: 'checkbox', id: `w-ch07-plain-${++mounted}` });
  const opt = h('label', { class: 'w-ch07-opt', for: plain.id }, plain,
    '把文字里的 <|endoftext|> 当作普通字符（tiktoken 的 encode_ordinary；原书的 encode 总是把它变成特殊 token）');
  const rowsBox = h('div');
  const live = h('div', { class: 'w-ch07-sr', 'aria-live': 'polite' });   // counts only, for screen readers
  const HINT = '点按或把鼠标移到某个 token 上，这里显示它的 id 和字节（键盘：Tab 进入一行，方向键逐个移动）';
  const pick = h('p', { class: 'w-ch07-pick' }, HINT);
  root.append(presetBar, text, opt, live, rowsBox, pick);
  const narrow = matchMedia('(max-width: 600px)');

  function fit() {                       // grow the textarea with its text, up to MAX_LINES lines
    text.style.height = 'auto';
    if (!text.scrollHeight) {            // not laid out (hidden): keep the CSS size
      text.style.height = '';
      return;
    }
    const cs = getComputedStyle(text);
    const pad = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
    const border = parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth);
    const box = cs.boxSizing === 'border-box';
    const need = box ? text.scrollHeight + border : text.scrollHeight - pad;
    const cap = MAX_LINES * (parseFloat(cs.lineHeight) || 21) + (box ? pad + border : 0);
    text.style.height = `${Math.min(need, cap)}px`;
    text.style.overflowY = need > cap ? 'auto' : 'hidden';
  }

  function choose(i) {
    text.value = presets[i].text;
    presetButtons.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
    render();
    fit();
  }

  let picked = null;
  function show(el) {                    // a chip was hovered, focused or tapped
    if (!el || !el.dataset.info) return;
    if (picked && picked !== el) picked.classList.remove('picked');
    picked = el;
    el.classList.add('picked');
    pick.textContent = el.dataset.info;
    pick.classList.add('on');            // from now on the line sticks to the bottom of the screen
  }
  rowsBox.addEventListener('mouseover', (e) => show(e.target.closest('.tok')));
  rowsBox.addEventListener('focusin', (e) => show(e.target.closest('.tok')));
  rowsBox.addEventListener('click', (e) => show(e.target.closest('.tok')));
  rowsBox.addEventListener('keydown', (e) => {   // roving tabindex: one tab stop per row, arrows move inside it
    const el = e.target.closest('.tok');
    if (!el) return;
    const all = [...el.parentElement.querySelectorAll('.tok')];
    const i = all.indexOf(el);
    const j = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: all.length - 1 }[e.key];
    if (j === undefined || j < 0 || j >= all.length) return;
    e.preventDefault();
    el.tabIndex = -1;
    all[j].tabIndex = 0;
    all[j].focus();
  });

  function row(bot, tok, r, maxTokens) {
    const chips = h('div', { class: 'tokens', role: 'list', 'aria-label': `${bot.label} 的 token` });
    const limit = narrow.matches ? MAX_CHIPS_NARROW : MAX_CHIPS;
    let shown = 0;
    let color = 0;
    for (const c of r.chunks) {
      for (const id of c.ids) {
        if (shown >= limit) break;
        const lab = c.special ? { text: c.text, partial: false, special: true } : tokenLabel(tok, id);
        const info = tokenInfo(bot.label, shown, id, tok.tokenBytes(id), lab);
        const el = chip(lab.text, id, color++, info);
        el.dataset.info = info;
        el.setAttribute('role', 'listitem');
        el.setAttribute('aria-label', info);
        el.tabIndex = shown === 0 ? 0 : -1;
        if (lab.partial) el.classList.add('partial');
        if (c.special) el.classList.add('special');
        chips.append(el);
        shown++;
      }
    }
    const more = r.tokens > shown ? h('div', { class: 'w-ch07-more' }, `……其余 ${fmt(r.tokens - shown)} 个 token 未显示`) : null;
    return h('div', { class: 'w-ch07-row' },
      h('div', { class: 'w-ch07-head' },
        h('span', { class: 'w-ch07-name' }, bot.label,
          h('span', { class: 'w-ch07-meta' }, `词表 ${fmt(tok.vocabSize)} · 学的是${bot.corpus}`)),
        h('span', { class: 'w-ch07-count' }, h('b', {}, fmt(r.tokens)), ' 个 token',
          r.tokens ? ` · 每个 ${r.bytesPerToken.toFixed(2)} 字节` : '')),
      h('div', { class: 'w-ch07-bar', role: 'presentation' },
        h('span', { style: { width: `${maxTokens ? (100 * r.tokens) / maxTokens : 0}%` } })),
      chips, more);
  }

  function render() {
    const { bytes, rows } = compare(toks, text.value, { special: !plain.checked });
    const maxTokens = Math.max(bytes, ...rows.map((r) => r.tokens));
    const base = h('div', { class: 'w-ch07-row' },
      h('div', { class: 'w-ch07-head' },
        h('span', { class: 'w-ch07-name' }, '字节级',
          h('span', { class: 'w-ch07-meta' }, '不用分词器 · 词表 256（另加 1 个特殊 token）')),
        h('span', { class: 'w-ch07-count' }, h('b', {}, fmt(bytes)), ' 个 token', bytes ? ' · 每个 1 字节' : '')),
      h('div', { class: 'w-ch07-bar base', role: 'presentation' },
        h('span', { style: { width: `${maxTokens ? (100 * bytes) / maxTokens : 0}%` } })));
    picked = null;
    pick.textContent = HINT;
    pick.classList.remove('on');         // the hint stays in place: it must not cover the text box
    rowsBox.replaceChildren(...BOTS.map((b, i) => row(b, toks[i], rows[i], maxTokens)), base);
    live.textContent = `${fmt(bytes)} 字节：` + BOTS.map((b, i) => `${b.label} ${fmt(rows[i].tokens)} 个 token`).join('，');
  }

  text.addEventListener('input', () => {
    presetButtons.forEach((b) => b.setAttribute('aria-pressed', 'false'));
    render();
    fit();
  });
  plain.addEventListener('change', render);
  narrow.addEventListener('change', render);     // the chip limit depends on the width
  addEventListener('resize', fit);               // wrapping, and so the textarea's height, depends on the width
  choose(Math.min(Math.max(0, config.preset ?? 0), presets.length - 1));
}
