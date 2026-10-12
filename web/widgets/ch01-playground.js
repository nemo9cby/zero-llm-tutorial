/* 图：CodeBot 等分词器的游乐场（1.8.3 节，以及章首的终点预览）。
   输入任意文字，看它被切成哪些 token、各自的 id、字节数与压缩率。编码由 _bpe.js 完成，
   它与原书的 BPETokenizer 逐个 id 相同（tests/js/lib-bpe.test.mjs）。
   config: { text, compact: true|false, tokenizer: 'codebot'|'storybot'|'webbot' } */
import { h, fmt } from './_lib.js';
import { Tokenizer, ENDOFTEXT, utf8Encode } from './_bpe.js';
import { injectStyle, tokenLabel, hex, presetButtons } from './ch01-util.js';

export const TOKENIZERS = [
  { name: 'codebot', label: 'CodeBot（词表 1,000）' },
  { name: 'storybot', label: 'StoryBot（词表 10,000）' },
  { name: 'webbot', label: 'WebBot（词表 50,000）' },
];

export const PRESETS = [
  { label: 'Python 代码', text: "def greet(name):\n    print(f'Hello, {name}!')" },
  { label: '深缩进', text: 'class Counter:\n    def add(self, items):\n        for item in items:\n            self.total += item\n' },
  { label: 'snake_case', text: 'total_time = start_time + elapsed_time' },
  { label: '儿童故事', text: 'Once upon a time, there was a little girl named Lily. She loved to play outside with her dog.' },
  { label: '中文与 emoji', text: '你好，世界！hello世界😁' },
  { label: '特殊 token 注入', text: "print('<|endoftext|>')" },
];

const MAX_CHIPS = 1500;

/** Everything the widget shows, computed without the DOM (tested in tests/js/ch01-widgets.test.mjs). */
export function analyze(tok, text, { special = true } = {}) {
  const chunks = tok.encodeChunks(text, { special });
  const ids = chunks.flatMap((c) => c.ids);
  const bytes = utf8Encode(text).length;
  return {
    chunks, ids, bytes,
    ratio: ids.length ? bytes / ids.length : 0,
    roundTrip: tok.decode(ids) === text,
    specialCount: chunks.filter((c) => c.special).length,
  };
}

export function mount(root, config = {}) {
  injectStyle();
  const compact = !!config.compact;
  let current = config.tokenizer || 'codebot';
  const input = h('textarea', { class: 'w-input', rows: compact ? 3 : 4, spellcheck: 'false', 'aria-label': '输入要分词的文字' });
  input.value = config.text ?? PRESETS[0].text;
  const select = h('select', { 'aria-label': '选择分词器' },
    TOKENIZERS.map((t) => h('option', { value: t.name, selected: t.name === current }, t.label)));
  const special = h('input', { type: 'checkbox', checked: true, 'aria-label': '识别特殊 token' });
  const chips = h('div', { class: 'tokens w-ch01-tokens', 'aria-live': 'polite' });
  const stats = h('div', { class: 'w-ch01-stats' }, '正在载入分词器…');
  if (!compact) {
    root.append(
      h('div', { class: 'w-ch01-bar' }, h('label', {}, '分词器', select),
        h('label', {}, special, '识别特殊 token（关掉 = tiktoken 的 encode_ordinary）')),
      presetButtons(PRESETS, (p) => { input.value = p.text; render(); }, '示例文字'),
    );
  }
  root.append(input, chips, stats);

  let seq = 0;
  async function render() {
    const mine = ++seq;
    let tok;
    try {
      tok = await Tokenizer.load(`data/tokenizers/${current}.json`);
    } catch (err) {
      stats.textContent = `分词器载入失败：${err.message}`;
      return;
    }
    if (mine !== seq) return;
    const a = analyze(tok, input.value, { special: special.checked });
    chips.replaceChildren();
    let n = 0;
    outer:
    for (const c of a.chunks) {
      for (const id of c.ids) {
        if (n >= MAX_CHIPS) { chips.append(h('span', { class: 'w-label' }, `……（只显示前 ${fmt(MAX_CHIPS)} 个）`)); break outer; }
        const b = tok.tokenBytes(id);
        const lab = c.special ? { text: ENDOFTEXT, partial: false } : tokenLabel(b);
        const cls = `tok c${n % 6}${c.special ? ' w-ch01-special' : ''}${lab.partial ? ' w-ch01-partial' : ''}`;
        const title = c.special ? `特殊 token，id ${id}` : `id ${id}；${b.length} 个字节：${hex(b)}${lab.partial ? '（单独解码不是完整的字符）' : ''}`;
        chips.append(h('span', { class: cls, title }, h('b', {}, lab.text), h('i', {}, id)));
        n++;
        if (!c.special && b.includes(10)) chips.append(h('span', { class: 'w-ch01-br', 'aria-hidden': 'true' }));
      }
    }
    stats.replaceChildren(
      h('b', {}, fmt(a.bytes)), ' 个字节 → ', h('b', {}, fmt(a.ids.length)), ' 个 token，压缩率 ',
      h('b', {}, a.ids.length ? a.ratio.toFixed(2) : '—'), ' 字节/token',
      a.specialCount ? `；其中 ${a.specialCount} 个特殊 token（id ${tok.vocabSize - 1}）` : '',
      '；解码还原：', h('span', { class: a.roundTrip ? 'w-ch01-ok' : 'w-ch01-bad' }, a.roundTrip ? '一字不差' : '不一致'),
    );
  }
  input.addEventListener('input', render);
  select.addEventListener('change', () => { current = select.value; render(); });
  special.addEventListener('change', render);
  render();
}
