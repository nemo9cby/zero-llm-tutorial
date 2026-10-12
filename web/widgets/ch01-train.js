/* 图：在浏览器里训练 CodeBot 的分词器（1.7.3 节，端到端例子）。
   数据是 TinyCodes 的预分词表（data/ch01/tinycodes_pretokens.json：30,535 种词及其次数，
   verify/test_ch01.py 核对过它与 tiny_codes.txt 一致），算法是 _bpe.js 的 BPETrainer：
   与原书相同的合并规则，只是像第 4 章那样按“种类 × 次数”计数、只更新受影响的词。
   训练完与原书的 merge_rules.pkl（data/tokenizers/codebot.json）逐条对照。 */
import { h, fmt } from './_lib.js';
import { BPETrainer, Tokenizer } from './_bpe.js';
import { injectStyle, bytesRepr, tokenLabel } from './ch01-util.js';

export const RULES = [
  { value: 'ids', label: 'id 最大者胜（原书 codebot）' },
  { value: 'first', label: '先出现者胜（ch01/03–06）' },
  { value: 'bytes', label: '字节串最大者胜（CS336）' },
];
export const NUM_MERGES = 743;

/** Every occurrence of every pre-token, in first-occurrence order of the distinct ones (1,826,876 strings). */
export function expandPieces(table) {
  const out = [];
  for (const [piece, count] of table.pieces) for (let i = 0; i < count; i++) out.push(piece);
  return out;
}

/** A trainer over the whole corpus. The text is '' and the pieces come from the table, so no end tokens are seen;
    they never take part in merges anyway (training splits on them first). */
export function makeTrainer(expanded, rule) {
  return new BPETrainer('', { tieBreak: rule, pretokenize: () => expanded, specialToken: null });
}

/** Compare learned merges with the official ones: positions equal, first difference, and token sets. */
export function compareWithOfficial(trainer, official, records) {
  const merges = trainer.merges;
  let same = 0;
  let first = -1;
  merges.forEach((m, i) => {
    const o = official[i];
    if (m[0] === o[0] && m[1] === o[1]) same++;
    else if (first < 0) first = i;
  });
  const key = (b) => Array.from(b).join(',');
  const officialVocab = [];
  const v = Array.from({ length: 256 }, (_, i) => Uint8Array.of(i));
  for (const [a, b] of official) {
    const t = new Uint8Array(v[a].length + v[b].length);
    t.set(v[a]); t.set(v[b], v[a].length);
    v.push(t); officialVocab.push(t);
  }
  const mine = new Set(trainer.vocab.slice(256).map(key));
  const theirs = new Set(officialVocab.map(key));
  const onlyMine = trainer.vocab.slice(256).filter((b) => !theirs.has(key(b)));
  const onlyTheirs = officialVocab.filter((b) => !mine.has(key(b)));
  return {
    same, first, onlyMine, onlyTheirs,
    firstRecord: first >= 0 ? records[first] : null,
    officialAtFirst: first >= 0 ? officialVocab[first] : null,
  };
}

export function mount(root) {
  injectStyle();
  const select = h('select', { 'aria-label': '平票规则' }, RULES.map((r) => h('option', { value: r.value }, r.label)));
  const go = h('button', { type: 'button', disabled: true }, '开始训练');
  const status = h('div', { class: 'w-ch01-stats', 'aria-live': 'polite' }, '正在载入 TinyCodes 的预分词表（约 450 KB）…');
  const bar = h('div', { class: 'w-ch01-progress', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': NUM_MERGES, 'aria-valuenow': 0 }, h('div'));
  const latest = h('div', { class: 'tokens' });
  const verdict = h('div', { class: 'w-ch01-stats' });
  root.append(h('div', { class: 'w-ch01-bar' }, h('label', {}, '平票规则', select), go), status, bar, latest, verdict);

  let table, official, expanded;
  let run = null;  // { trainer, records, rule, t0, elapsed, paused, done }

  Promise.all([
    fetch('data/ch01/tinycodes_pretokens.json').then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
    Tokenizer.load('data/tokenizers/codebot.json'),
  ]).then(([t, tok]) => {
    table = t;
    official = tok.merges;
    go.disabled = false;
    status.replaceChildren('已载入：', h('b', {}, fmt(t.pretokens)), ' 个词（', h('b', {}, fmt(t.pieces.length)), ' 种），',
      fmt(t.bytes), ' 个字节。选好平票规则，点“开始训练”。');
  }).catch((err) => { status.textContent = `数据载入失败：${err.message}`; });

  const ratio = (tokens) => table.bytes / (tokens + table.end_tokens);   // each <|endoftext|> is 1 token

  function show() {
    const { trainer, records } = run;
    const k = trainer.merges.length;
    bar.firstElementChild.style.width = `${(100 * k) / NUM_MERGES}%`;
    bar.setAttribute('aria-valuenow', k);
    status.replaceChildren('第 ', h('b', {}, k), ` / ${NUM_MERGES} 条规则，用时 `, h('b', {}, (run.elapsed / 1000).toFixed(1)), ' 秒，压缩率 ',
      h('b', {}, ratio(trainer.tokens).toFixed(3)), ' 字节/token');
    latest.replaceChildren(...records.slice(-8).map((rec) => h('span', { class: `tok c${rec.newId % 6}`, title: `出现 ${fmt(rec.count)} 次` },
      h('b', {}, tokenLabel(trainer.vocab[rec.newId]).text), h('i', {}, rec.newId))));
  }

  function finish() {
    const { trainer, records, rule } = run;
    const c = compareWithOfficial(trainer, official, records);
    const r = ratio(trainer.tokens).toFixed(2);
    const parts = [];
    if (c.first < 0) {
      parts.push(h('span', { class: 'w-ch01-ok' }, `✓ ${NUM_MERGES} 条规则与原书的 merge_rules.pkl 逐条相同，顺序也相同。`));
    } else {
      const rec = c.firstRecord;
      parts.push(h('span', { class: 'w-ch01-bad' }, `前 ${c.first} 条与原书相同，第 ${c.first + 1} 条开始不同：`),
        '这里学的是 ', h('code', {}, bytesRepr(trainer.vocab[256 + c.first])), '，原书是 ', h('code', {}, bytesRepr(c.officialAtFirst)),
        `（那一步有 ${rec.tied} 个 pair 并列，都出现了 ${fmt(rec.count)} 次）。${NUM_MERGES} 条里位置相同的有 ${c.same} 条。`);
      parts.push(h('br'), c.onlyMine.length === 0 ? '不过，最终学到的 743 个 token 与原书完全是同一批，只是顺序（也就是 id）不同。'
        : `最终学到的 743 个 token 里，与原书不同的只有 ${c.onlyMine.length} 个：这里多了 ${c.onlyMine.map(bytesRepr).join('、')}，原书是 ${c.onlyTheirs.map(bytesRepr).join('、')}。`);
    }
    parts.push(h('br'), `全语料压缩率 ${r}（原书 2.42）；${RULES.find((x) => x.value === rule).label}，浏览器里用时 ${(run.elapsed / 1000).toFixed(1)} 秒。`);
    verdict.replaceChildren(...parts);
    go.textContent = '重新训练';
  }

  function loop() {
    if (!run || run.paused || run.done) return;
    const t0 = performance.now();
    while (performance.now() - t0 < 14) {
      if (run.trainer.merges.length >= NUM_MERGES) { run.done = true; break; }
      const rec = run.trainer.step();
      if (!rec) { run.done = true; break; }
      run.records.push(rec);
    }
    run.elapsed += performance.now() - t0;
    show();
    if (run.done) finish();
    else setTimeout(loop, 0);
  }

  go.addEventListener('click', () => {
    if (run && !run.done) {                       // pause / resume
      run.paused = !run.paused;
      go.textContent = run.paused ? '继续' : '暂停';
      if (!run.paused) loop();
      return;
    }
    verdict.replaceChildren();
    status.textContent = '正在准备：展开 1,826,876 个词，数一遍所有相邻 pair…';
    go.textContent = '暂停';
    setTimeout(() => {
      const t0 = performance.now();
      expanded = expanded || expandPieces(table);
      const trainer = makeTrainer(expanded, select.value);
      run = { trainer, records: [], rule: select.value, elapsed: performance.now() - t0, paused: false, done: false };
      loop();
    }, 30);
  });
  select.addEventListener('change', () => {
    if (run && !run.done) run.paused = true;
    run = null;
    go.textContent = '开始训练';
    verdict.replaceChildren();
    latest.replaceChildren();
    bar.firstElementChild.style.width = '0%';
  });
}
