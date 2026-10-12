/* 图：GPT-2 预分词正则的切分（1.6.2 节）。每块的颜色表示是哪个分支匹配的；关掉某个分支，
   就用剩下的分支做 findall（与 Python 的 regex.findall 相同），没被任何分支匹配的字符会被跳过。
   切分由 _bpe.js 的 pretokenizeSpans 完成，它与 Python regex 逐块相同（tests/js/lib-bpe.test.mjs）。 */
import { h, fmt, visible } from './_lib.js';
import { GPT2_BRANCHES, pretokenizeSpans } from './_bpe.js';
import { injectStyle, presetButtons } from './ch01-util.js';

export const BRANCH_LABELS = ['缩约', '字母', '数字', '其他符号', '空白（留一格）', '其余空白'];

export const PRESETS = [
  { label: '原书例句', text: 'Say hello! Why hello? Just hello.' },
  { label: "f'done' 与 DON'T", text: "print(f'done')  # I'd say DON'T" },
  { label: '缩进与数字', text: 'def total_time(x):\n    return x + 1234567\n\n' },
  { label: '多语言', text: '日本語のテキスト。中文也一样，e\u0301 café 😁' },
];

/**
 * findall with the enabled branches, plus the characters no branch matched (as gaps), in text order.
 * @returns {{text: string, branch: number|null}[]}  branch null = skipped by findall
 */
export function piecesWithGaps(text, enabled) {
  const branches = GPT2_BRANCHES.filter((_, i) => enabled[i]);
  const index = GPT2_BRANCHES.map((b, i) => (enabled[i] ? branches.indexOf(b) : -1));
  const spans = pretokenizeSpans(text, branches);
  const out = [];
  let at = 0;
  for (const s of spans) {
    if (s.start > at) out.push({ text: text.slice(at, s.start), branch: null });
    out.push({ text: s.text, branch: index.indexOf(s.branch) });
    at = s.end;
  }
  if (at < text.length) out.push({ text: text.slice(at), branch: null });
  return out;
}

export function mount(root, config = {}) {
  injectStyle();
  const enabled = GPT2_BRANCHES.map(() => true);
  const input = h('textarea', { class: 'w-input', rows: 3, spellcheck: 'false', 'aria-label': '输入要预分词的文字' });
  input.value = config.text ?? PRESETS[1].text;
  const legend = h('div', { class: 'w-ch01-legend', role: 'group', 'aria-label': '正则的 6 个分支，可以分别关掉' },
    BRANCH_LABELS.map((label, i) => {
      const box = h('input', { type: 'checkbox', checked: true, 'aria-label': `启用分支 ${i + 1}：${label}` });
      box.addEventListener('change', () => { enabled[i] = box.checked; render(); });
      return h('label', { title: GPT2_BRANCHES[i].py }, box, h('span', { class: `tok c${i} w-ch01-swatch` }), `${i + 1} ${label}`);
    }));
  const out = h('div', { class: 'tokens', 'aria-live': 'polite' });
  const stats = h('div', { class: 'w-ch01-stats' });
  root.append(presetButtons(PRESETS, (p) => { input.value = p.text; render(); }, '示例文字'), input, legend, out, stats);

  function render() {
    const pieces = piecesWithGaps(input.value, enabled);
    out.replaceChildren(...pieces.flatMap((p) => {
      const chip = p.branch === null
        ? h('span', { class: 'tok w-ch01-gap', title: '没有分支匹配，findall 跳过了这些字符' }, h('b', {}, visible(p.text)), h('i', {}, '跳过'))
        : h('span', { class: `tok c${p.branch}`, title: `分支 ${p.branch + 1}：${BRANCH_LABELS[p.branch]}` },
          h('b', {}, visible(p.text)), h('i', {}, p.branch + 1));
      return p.text.includes('\n') ? [chip, h('span', { class: 'w-ch01-br', 'aria-hidden': 'true' })] : [chip];
    }));
    const kept = pieces.filter((p) => p.branch !== null);
    const skipped = pieces.filter((p) => p.branch === null).reduce((s, p) => s + [...p.text].length, 0);
    stats.replaceChildren(`切成 `, h('b', {}, fmt(kept.length)), ' 块；BPE 只在每一块内部合并，不会跨过块的边界',
      skipped ? h('span', { class: 'w-ch01-bad' }, `；有 ${skipped} 个字符没被任何分支匹配，被丢掉了`) : '');
  }
  input.addEventListener('input', render);
  render();
}
