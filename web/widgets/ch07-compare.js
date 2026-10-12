/* 图 7-1：同一段文字，交给三台机器人的分词器（CodeBot 1,000 / StoryBot 10,000 / WebBot 50,000），
   再加一行“字节级”（不用分词器）作基线。
   config: { "presets": "data/ch07/presets.json", "preset": 0 }
   compare() is pure (no DOM) so tests/js/ch07-compare.test.mjs can check it against the book's Python code. */
import { h, chip, fmt } from './_lib.js';
import { Tokenizer, utf8Encode } from './_bpe.js';

export const BOTS = Object.freeze([
  { key: 'codebot', label: 'CodeBot', corpus: '代码', url: 'data/tokenizers/codebot.json' },
  { key: 'storybot', label: 'StoryBot', corpus: '儿童故事', url: 'data/tokenizers/storybot.json' },
  { key: 'webbot', label: 'WebBot', corpus: '网页文章', url: 'data/tokenizers/webbot.json' },
]);

const MAX_CHIPS = 300;          // per row; longer texts show a count instead of every chip
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

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch07-presets { display: flex; flex-wrap: wrap; gap: .35rem; margin: 0 0 .5rem; }
.w-ch07-presets button[aria-pressed="true"] { border-color: var(--ai); color: var(--ai); font-weight: 600; }
.w-ch07-text { width: 100%; min-height: 4.6rem; resize: vertical; font: 14px/1.5 var(--mono); color: var(--ink);
  background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .5rem .7rem; }
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
.w-ch07-row .tok.partial b { font-size: 11px; color: var(--ink-2) !important; }
.w-ch07-row .tok.special { outline: 1.5px solid var(--kincha); outline-offset: -1.5px; }
.w-ch07-more { font-size: .8rem; color: var(--ink-3); }
/* Chips tinted from the paper colour, so text keeps its contrast in both themes (the shared dark-mode rule
   dims text and background together with a filter). */
.w-ch07 .tok { filter: none !important; color: var(--ink) !important; }
.w-ch07 .tok i { color: var(--ink-2); }
${['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'].map((c, i) =>
    `.w-ch07 .tok.c${i} { background: color-mix(in srgb, ${c} 20%, var(--paper)) !important; }`).join('\n')}
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

  const text = h('textarea', { class: 'w-ch07-text', 'aria-label': '输入任意文字，比较三个分词器', spellcheck: 'false' });
  const presetBar = h('div', { class: 'w-ch07-presets', role: 'group', 'aria-label': '示例文字' });
  const presetButtons = presets.map((p, i) => {
    const b = h('button', { type: 'button', 'aria-pressed': 'false', onclick: () => choose(i) }, p.name);
    presetBar.append(b);
    return b;
  });
  const plain = h('input', { type: 'checkbox', id: `w-ch07-plain-${++mounted}` });
  const opt = h('label', { class: 'w-ch07-opt', for: plain.id }, plain,
    '把文字里的 <|endoftext|> 当作普通字符（tiktoken 的 encode_ordinary；原书的 encode 总是把它变成特殊 token）');
  const rowsBox = h('div', { 'aria-live': 'polite' });
  root.append(presetBar, text, opt, rowsBox);

  function choose(i) {
    text.value = presets[i].text;
    presetButtons.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
    render();
  }

  function row(bot, tok, r, maxTokens) {
    const chips = h('div', { class: 'tokens' });
    let shown = 0;
    let color = 0;
    for (const c of r.chunks) {
      for (const id of c.ids) {
        if (shown >= MAX_CHIPS) break;
        const lab = c.special ? { text: c.text, partial: false } : tokenLabel(tok, id);
        const bytesHex = [...tok.tokenBytes(id)].map((x) => x.toString(16).padStart(2, '0')).join(' ');
        const el = chip(lab.text, id, color++, c.special ? `id ${id}：特殊 token` : `id ${id}：字节 ${bytesHex}`
          + (lab.partial ? '（单独不是完整的 UTF-8 字符）' : ''));
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
        h('span', { class: 'w-ch07-name' }, '字节级', h('span', { class: 'w-ch07-meta' }, '不用分词器 · 词表 256')),
        h('span', { class: 'w-ch07-count' }, h('b', {}, fmt(bytes)), ' 个 token · 每个 1 字节')),
      h('div', { class: 'w-ch07-bar base', role: 'presentation' },
        h('span', { style: { width: `${maxTokens ? (100 * bytes) / maxTokens : 0}%` } })));
    rowsBox.replaceChildren(...BOTS.map((b, i) => row(b, toks[i], rows[i], maxTokens)), base);
  }

  text.addEventListener('input', () => {
    presetButtons.forEach((b) => b.setAttribute('aria-pressed', 'false'));
    render();
  });
  plain.addEventListener('change', render);
  choose(Math.min(Math.max(0, config.preset ?? 0), presets.length - 1));
}
