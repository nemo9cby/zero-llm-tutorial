/* 图：增量更新的单步演示（ch04/02 的循环体）。每点一次“下一步”：
   挑出次数最多的 pair（平票比 id），从倒排索引里取出含有它的词，只更新这几个词；
   计数表里分两栏列出本步减去的旧计数（−旧）和加上的新计数（+新），减了又加回的行也照样列出；
   最右一栏是倒排索引：现在含有这个 pair 的词。右上角的指示灯把增量结果与全量重算比一次。
   config: { corpus: "toy" } */
import { h, chip, visible, fmt } from './_lib.js';
import { IncrementalBPE, unpair, pairKey, pretokenizeCorpus, countWords } from './ch04-bpe.js';

const STYLE = `
/* token chips: theme-aware colours (the shared dark-mode .tok filter also darkens the text) */
.w-ch04-incr .tok { filter: none !important; color: var(--ink) !important; }
.w-ch04-incr .tok i { color: var(--ink-2) !important; }
.w-ch04-incr .tok.c0 { background: color-mix(in srgb, var(--ai) 22%, var(--paper)) !important; }
.w-ch04-incr .tok.c1 { background: color-mix(in srgb, var(--moss) 24%, var(--paper)) !important; }
.w-ch04-incr .tok.c2 { background: color-mix(in srgb, var(--kincha) 26%, var(--paper)) !important; }
.w-ch04-incr .tok.c3 { background: color-mix(in srgb, var(--plum) 24%, var(--paper)) !important; }
.w-ch04-incr .tok.c4 { background: color-mix(in srgb, var(--ai) 40%, var(--paper)) !important; }
.w-ch04-incr .tok.c5 { background: color-mix(in srgb, var(--shu) 18%, var(--paper)) !important; }
.w-ch04-incr .tok.plain { background: var(--paper-2) !important; }
.w-ch04-incr, .w-ch04-incr * { font-variant-ligatures: none !important; }
.w-ch04-incr .bar { display: flex; flex-wrap: wrap; gap: .5rem .8rem; align-items: center; margin-bottom: .7rem; font-size: .86rem; }
.w-ch04-incr .lamp { margin-left: auto; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem; border: 1px solid var(--moss); color: var(--moss); }
.w-ch04-incr .lamp.bad { border-color: var(--shu); color: var(--shu); }
.w-ch04-incr .cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.2rem; }
.w-ch04-incr h4 { margin: 0 0 .35rem; font-size: .82rem; font-weight: 600; color: var(--ink-2); }
.w-ch04-incr .word { display: flex; align-items: center; gap: .4rem; padding: .15rem .3rem; border-radius: 4px; border: 1px solid transparent; }
.w-ch04-incr .word.hit { border-color: var(--ai); background: color-mix(in srgb, var(--ai-2) 55%, transparent); }
.w-ch04-incr .word .tokens { margin: .1rem 0; flex-wrap: nowrap; }
.w-ch04-incr .word .cnt { font: 12px var(--mono); color: var(--ink-2); white-space: nowrap; }
.w-ch04-incr .word .tag { font-size: .72rem; color: var(--ai); white-space: nowrap; }
.w-ch04-incr table { display: table; width: 100%; margin: 0; font-size: .82rem; }
.w-ch04-incr td, .w-ch04-incr th { padding: .15rem .45rem .15rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
.w-ch04-incr th { font-weight: 600; text-align: left; }
.w-ch04-incr td.pair code { white-space: pre; }
.w-ch04-incr td.num, .w-ch04-incr th.num { white-space: nowrap; text-align: right; }
.w-ch04-incr td.num { font: 12px var(--mono); font-variant-numeric: tabular-nums; }
.w-ch04-incr td.idx { font-size: .76rem; color: var(--ink-2); overflow-wrap: anywhere; }
.w-ch04-incr td.idx, .w-ch04-incr th.idx { padding-left: .6rem; }
.w-ch04-incr td.idx em { font-style: normal; color: var(--ink-3); }
.w-ch04-incr .legend { margin-top: .35rem; font-size: .76rem; color: var(--ink-3); }
.w-ch04-incr tr.next td { background: color-mix(in srgb, var(--ai-2) 55%, transparent); }
.w-ch04-incr tr.gone td { color: var(--ink-3); text-decoration: line-through; }
.w-ch04-incr .minus { color: var(--shu); } .w-ch04-incr .plus { color: var(--moss); }
.w-ch04-incr .foot { margin-top: .7rem; font-size: .84rem; color: var(--ink-2); line-height: 1.7; }
.w-ch04-incr .foot code { font-size: .8rem; }
@media (max-width: 640px) { .w-ch04-incr .cols { grid-template-columns: minmax(0, 1fr); } }
`;

const CORPORA = {
  toy: { name: '玩具语料（CS336 讲义的例子）', words: [['low', 5], ['lower', 2], ['newest', 6], ['widest', 3]] },
  story: { name: '一个小故事', text: 'Once upon a time, there was a little girl. She saw a little bird. The bird was sad. The girl gave the bird a seed. The bird was happy and sang a song for the little girl.' },
};

export function mount(root, config) {
  root.classList.add('w-ch04-incr');
  root.append(h('style', {}, STYLE));
  let key = config.corpus in CORPORA ? config.corpus : 'toy';
  let bpe, last, stepNo;

  const select = h('select', { 'aria-label': '选择语料' },
    Object.entries(CORPORA).map(([k, c]) => h('option', { value: k, selected: k === key }, c.name)));
  const nextBtn = h('button', { type: 'button', 'aria-label': '执行一次合并' }, '下一步 ▶');
  const resetBtn = h('button', { type: 'button' }, '重置');
  const stepLabel = h('span', { class: 'w-label', 'aria-live': 'polite' });
  const lamp = h('span', { class: 'lamp', role: 'status' });
  const wordsBox = h('div');
  const table = h('table', {});
  const legend = h('div', { class: 'legend' });
  const foot = h('div', { class: 'foot' });
  root.append(
    h('div', { class: 'bar' }, select, nextBtn, resetBtn, stepLabel, lamp),
    h('div', { class: 'cols' },
      h('div', {}, h('h4', {}, '不同的词（当前的 token 序列 × 出现次数）'), wordsBox),
      h('div', {}, h('h4', {}, 'pair 计数表与倒排索引'), table, legend)),
    foot,
  );

  function reset() {
    const c = CORPORA[key];
    const words = c.words ? c.words.map(([word, count]) => ({ word, count })) : countWords(pretokenizeCorpus(c.text));
    bpe = new IncrementalBPE(words);
    last = null;
    stepNo = 0;
    render();
  }

  const pairLabel = (k) => { const [a, b] = unpair(k); return `${visible(bpe.text(a))} + ${visible(bpe.text(b))}`; };

  function render() {
    const nextKey = bpe.done ? null : [...bpe.pairCounts].reduce((best, [k, c]) =>
      (!best || c > best[1] || (c === best[1] && k > best[0]) ? [k, c] : best), null)[0];
    stepLabel.textContent = `已合并 ${stepNo} 次`;
    nextBtn.disabled = bpe.done;
    const ok = bpe.consistent();
    lamp.className = ok ? 'lamp' : 'lamp bad';
    lamp.textContent = ok ? '增量 == 全量重算 ✔' : '不一致 ✘';

    const hit = new Set(last ? last.affected : []);
    const order = bpe.seqs.map((s, i) => i);
    const many = order.length > 10;
    const shown = many ? order.filter((i) => hit.has(i)).slice(0, 8) : order;
    wordsBox.replaceChildren(...shown.map((si) => {
      const s = bpe.seqs[si];
      return h('div', { class: `word${hit.has(si) ? ' hit' : ''}` },
        h('div', { class: 'tokens' }, s.ids.map((id, j) => chip(bpe.text(id), id, id >= 256 ? 2 : 0))),
        h('span', { class: 'cnt' }, `× ${s.count}`), hit.has(si) ? h('span', { class: 'tag' }, '本步更新') : null);
    }));
    if (many) wordsBox.append(h('div', { class: 'w-label' }, last ? `只显示本步受影响的词；其余 ${fmt(order.length - hit.size)} 个词原封不动。`
      : `共 ${fmt(order.length)} 个不同的词。点“下一步”后，这里只显示受影响的那几个。`));

    const minus = last ? last.minus : new Map(), plus = last ? last.plus : new Map();
    const mergedKey = last ? pairKey(last.pair[0], last.pair[1]) : null;
    const rows = new Map([...bpe.pairCounts].sort((x, y) => y[1] - x[1] || y[0] - x[0]).slice(0, 9));
    for (const k of [...minus.keys(), ...plus.keys()]) if (!rows.has(k)) rows.set(k, bpe.pairCounts.get(k) || 0);
    const amount = (m, k, sign, cls) => (m.has(k) ? h('span', { class: cls }, `${sign}${fmt(m.get(k))}`) : '');
    const index = (k) => {
      if (k === mergedKey) return h('em', {}, '本步合并，条目已删除');
      const words = bpe.wordsWith(k);
      if (!words.length) return h('em', {}, '（空）');
      const shown = words.slice(0, 3).map((w) => visible(w)).join('，');
      return words.length > 3 ? `${shown} 等 ${fmt(words.length)} 个` : shown;
    };
    table.replaceChildren(
      h('thead', {}, h('tr', {}, h('th', {}, 'pair'), h('th', { class: 'num' }, '次数'), h('th', { class: 'num' }, '−旧'),
        h('th', { class: 'num' }, '+新'), h('th', { class: 'idx' }, '倒排索引'))),
      h('tbody', {}, [...rows].map(([k, c]) => h('tr', {
        class: k === nextKey ? 'next' : (c === 0 ? 'gone' : ''), 'aria-label': k === nextKey ? `下一步要合并的 pair：${pairLabel(k)}` : null },
      h('td', { class: 'pair' }, h('code', {}, pairLabel(k))),
      h('td', { class: 'num' }, fmt(c)),
      h('td', { class: 'num' }, amount(minus, k, '−', 'minus')),
      h('td', { class: 'num' }, amount(plus, k, '+', 'plus')),
      h('td', { class: 'idx' }, index(k))))));
    legend.textContent = `高亮的一行是下一步要合并的 pair（次数最多，平票时比 id）；倒排索引一栏是现在含有这个 pair 的词。`;

    const totalBytes = bpe.seqs.reduce((s, q) => s + new TextEncoder().encode(q.word).length * q.count, 0);
    const touched = new Set([...minus.keys(), ...plus.keys()]);
    const backAgain = [...touched].filter((k) => minus.get(k) === plus.get(k)).length;
    foot.replaceChildren(last
      ? h('div', {}, '刚才合并了 ', h('code', {}, `${visible(bpe.text(last.pair[0]))} + ${visible(bpe.text(last.pair[1]))} → ${visible(bpe.text(last.newId))}`),
        `（id ${last.newId}，出现 ${fmt(last.count)} 次）。倒排索引里含有它的词只有 ${last.affected.length} 个，所以只需要更新这 ${last.affected.length} 个词：`,
        `先减去它们的全部旧 pair，再加上全部新 pair，计数表里动过的有 ${fmt(touched.size)} 行，其中 ${fmt(backAgain)} 行减了又加回同样的数。`,
        `朴素版这一步要把全部 ${fmt(totalBytes)} 个字节重新数一遍。`)
      : h('div', {}, `pair 次数相同时，原书比较 id：(count, id0, id1) 最大的那个先合并。`),
    );
  }

  nextBtn.addEventListener('click', () => { if (!bpe.done) { last = bpe.step(); stepNo += 1; render(); nextBtn.focus(); } });
  resetBtn.addEventListener('click', reset);
  select.addEventListener('change', () => { key = select.value; reset(); });
  reset();
}
