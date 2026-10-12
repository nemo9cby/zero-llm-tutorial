/* 封面：书名在三台机器人的分词器眼里是什么样子。
   Tokens come from the real merge tables (web/data/tokenizers/*.json) through _bpe.js, which
   tests/js/lib-bpe.test.mjs checks id-for-id against the book's Python tokenizers. */
import { h, visible } from './_lib.js';
import { Tokenizer, utf8Decode } from './_bpe.js';

const BOTS = [
  { key: 'codebot', label: 'CodeBot', short: '1k', vocab: '1,000', data: 'Python 小程序' },
  { key: 'storybot', label: 'StoryBot', short: '10k', vocab: '10,000', data: '英文童话' },
  { key: 'webbot', label: 'WebBot', short: '50k', vocab: '50,000', data: '英文网页' },
];
const SAMPLES = [
  { label: '书名', text: '从零构建大语言模型' },
  { label: '英文书名', text: 'Build a Large Language Model from Scratch' },
  { label: '一行代码', text: 'def train(model, data):' },
];

const cache = new Map();
const load = (key) => {
  if (!cache.has(key)) cache.set(key, Tokenizer.load(`data/tokenizers/${key}.json`));
  return cache.get(key);
};

/** What to print on a chip: the token's text, or its bytes in hex when they are not valid UTF-8 on their own. */
function chipText(tok, id) {
  const bytes = tok.tokenBytes(id);
  const s = utf8Decode(bytes);
  return s.includes('�') ? [...bytes].map((b) => b.toString(16).padStart(2, '0')).join(' ') : visible(s);
}

export function mount(root, config) {
  let bot = BOTS[0];
  const input = h('input', { class: 'w-input hero-input', value: config.text || SAMPLES[0].text,
    'aria-label': '输入任意文字，看它被切成哪些 token' });
  const tape = h('div', { class: 'tokens hero-tape', 'aria-live': 'polite' });
  const count = h('p', { class: 'hero-count' });
  const botButtons = BOTS.map((b) => h('button', {
    type: 'button', 'aria-pressed': String(b === bot), 'aria-label': `${b.label}，词表 ${b.vocab}`,
    onclick: () => { bot = b; botButtons.forEach((x, i) => x.setAttribute('aria-pressed', String(BOTS[i] === bot))); render(); },
  }, `${b.label}（${b.short}）`));
  const sampleButtons = SAMPLES.map((s) => h('button', {
    type: 'button', class: 'hero-sample', onclick: () => { input.value = s.text; render(); },
  }, s.label));

  root.append(
    h('div', { class: 'hero-bots', role: 'group', 'aria-label': '选择分词器' }, botButtons),
    tape, count,
    h('div', { class: 'hero-try' }, h('span', { class: 'w-label' }, '换一句试试：'), sampleButtons), input,
  );

  let seq = 0;
  async function render() {
    const my = ++seq;
    const text = input.value;
    let tok;
    try {
      tok = await load(bot.key);
    } catch (err) {
      count.textContent = `分词表加载失败：${err.message}`;
      return;
    }
    if (my !== seq) return; // a newer render is in flight
    const ids = tok.encode(text);
    tape.replaceChildren(...ids.map((id, i) => h('span', { class: `tok c${i % 6}`, title: `token id ${id}` },
      h('b', {}, chipText(tok, id)), h('i', {}, id))));
    const chars = [...text].length;
    count.textContent = `${bot.label} 的分词器（词表 ${bot.vocab}，训练数据：${bot.data}）把这 ${chars} 个字符切成了 ${ids.length} 个 token。`;
  }
  input.addEventListener('input', render);
  render();
}
