/* 图：压缩率随词表大小的变化（1.8.2 节）。
   数据是 data/ch01/compression.json：只保留 merge_rules.pkl 的前 k 条规则时，样本被编成多少个 token
   （k = 0…743），以及其他分词器在同一样本上的 token 数。verify/test_ch01.py 会重新计算并逐个核对。 */
import { h, fmt } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle, segmented } from './ch01-util.js';

const NAMES = { storybot: 'StoryBot', webbot: 'WebBot', 'GPT-2': 'GPT-2', cl100k_base: 'cl100k_base', o200k_base: 'o200k_base' };

/** The CodeBot curve (x = vocab size 257 + k, y = bytes/token) and the bar list for one sample. */
export function curveFor(data, sample) {
  const s = data[sample];
  const points = s.tokens.map((n, k) => [257 + k, s.bytes / n]);
  const bars = [{ name: 'CodeBot', vocab: 1000, ratio: s.bytes / s.tokens[s.tokens.length - 1], me: true },
    ...s.refs.map((r) => ({ name: NAMES[r.name] || r.name, vocab: r.vocab, ratio: s.bytes / r.tokens }))]
    .sort((a, b) => a.ratio - b.ratio);
  return { points, bars, label: s.label, bytes: s.bytes };
}

export function mount(root) {
  injectStyle();
  const chartBox = h('div');
  const barsBox = h('div', { class: 'w-ch01-bars', role: 'list', 'aria-label': '同一样本上各分词器的压缩率' });
  const note = h('div', { class: 'w-ch01-note' });
  root.append(h('div', { class: 'w-ch01-note' }, '正在载入…'));
  fetch('data/ch01/compression.json')
    .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then((data) => {
      let sample = 'full';
      const seg = segmented([{ value: 'full', label: '全语料' }, { value: 'head', label: '前 10,000 个字符' }], sample,
        (v) => { sample = v; draw(); }, '选择样本');
      root.replaceChildren(h('div', { class: 'w-ch01-bar' }, h('span', { class: 'w-label' }, '样本'), seg.el), chartBox,
        h('div', { class: 'w-label' }, '同一样本上，各分词器的压缩率（字节/token）'), barsBox, note);
      let chart = null;
      function draw() {
        const c = curveFor(data, sample);
        const gpt2 = c.bars.find((b) => b.name === 'GPT-2');
        const opts = {
          series: [{ name: `CodeBot（前 k 条规则）`, points: c.points }],
          xLabel: '词表大小（256 个字节 + k 条规则 + 1 个特殊 token）', yLabel: '压缩率（字节/token）',
          yMin: 1, yMax: 3.2, height: 250, table: false,
          refLines: [{ y: gpt2.ratio, label: `GPT-2（词表 50,257）${gpt2.ratio.toFixed(2)}` }],
          ariaLabel: `${c.label}上 CodeBot 的压缩率随词表大小上升，从 ${c.points[0][1].toFixed(2)} 到 ${c.points[c.points.length - 1][1].toFixed(2)}`,
        };
        if (chart) chart.update(opts);
        else chart = lineChart(chartBox, opts);
        const max = Math.max(...c.bars.map((b) => b.ratio));
        barsBox.replaceChildren(...c.bars.flatMap((b) => [
          h('div', { class: 'lab', role: 'listitem' }, b.name, ' ', h('small', {}, `词表 ${fmt(b.vocab)}`)),
          h('div', { class: 'track', 'aria-hidden': 'true' }, h('div', { class: `fill${b.me ? ' me' : ''}`, style: `width:${(100 * b.ratio) / max}%` })),
          h('div', { class: 'val' }, b.ratio.toFixed(2)),
        ]));
        const k100 = c.points[100][1] - c.points[0][1];
        const last = c.points[743][1] - c.points[600][1];
        note.textContent = `${c.label}（${fmt(c.bytes)} 字节）：前 100 条规则让压缩率提高了 ${k100.toFixed(2)}，最后 143 条只提高了 ${last.toFixed(2)}。`;
      }
      draw();
    })
    .catch((err) => { root.replaceChildren(h('div', { class: 'w-ch01-bad' }, `数据载入失败：${err.message}`)); });
}
