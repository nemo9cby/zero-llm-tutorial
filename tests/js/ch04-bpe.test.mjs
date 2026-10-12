// 第 4 章组件逻辑的测试：期望值全部由原书的 Python 代码生成（verify/_ch04_data.py -> fixtures/ch04/）。
//   node --test tests/js/ch04-bpe.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  pretokenize, pretokenizeCorpus, countWords, IncrementalBPE, chunkBoundaries, decodeOffsets, amdahl, EOT,
} from '../../web/widgets/ch04-bpe.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const fixture = (name) => JSON.parse(fs.readFileSync(path.join(here, 'fixtures/ch04', name), 'utf8'));
const data = (name) => fs.readFileSync(path.join(root, 'web/data/ch04', name), 'utf8');

test('JS pretokenizer == Python regex on tricky inputs and on the 10,000-character sample', () => {
  const fx = fixture('pretokenize.json');
  for (const [text, tokens] of fx.tricky) assert.deepEqual(pretokenize(text), tokens, JSON.stringify(text));
  assert.deepEqual(pretokenizeCorpus(data('train_head10k.txt')), fx.sample);
});

test('IncrementalBPE reproduces ch04/02 merges (toy corpus and the story sample)', () => {
  const fx = fixture('merges.json');
  const toy = 'low low low low low lower lower widest widest widest newest newest newest newest newest newest';
  const run = (text, n) => {
    const bpe = new IncrementalBPE(countWords(pretokenizeCorpus(text)));
    for (let i = 0; i < n && !bpe.done; i++) bpe.step();
    return bpe;
  };
  const t = run(toy, fx.toy.length);
  assert.deepEqual(t.merges, fx.toy);
  assert.deepEqual(t.merges.slice(0, 6).map(([a, b]) => `${t.text(a)} ${t.text(b)}`),
    ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']);   // CS336 handout example
  assert.deepEqual(run(data('train_head10k.txt'), fx.sample400.length).merges, fx.sample400);
});

test('every incremental step equals a full recount, and only affected words change', () => {
  const bpe = new IncrementalBPE(countWords(pretokenizeCorpus(data('train_head10k.txt'))));
  for (let i = 0; i < 80; i++) {
    const before = bpe.seqs.map((s) => s.ids.join(','));
    const r = bpe.step();
    assert.ok(bpe.consistent(), `step ${i}`);
    bpe.seqs.forEach((s, si) => {
      if (!r.affected.includes(si)) assert.equal(s.ids.join(','), before[si]);
    });
  }
});

test('first step on the toy corpus: (s,t) beats the tie with (e,s), deltas as computed by hand', () => {
  const bpe = new IncrementalBPE([{ word: 'low', count: 5 }, { word: 'lower', count: 2 },
    { word: 'newest', count: 6 }, { word: 'widest', count: 3 }]);
  const r = bpe.step();
  assert.deepEqual(r.pair, [115, 116]);              // 's','t'
  assert.equal(r.count, 9);
  assert.deepEqual(r.affected, [2, 3]);              // newest, widest
  const d = Object.fromEntries([...r.delta].map(([k, v]) => [k, v]));
  const key = (a, b) => a * 1048576 + b;
  assert.equal(d[key(101, 115)], -9);                // (e,s) 9 -> 0
  assert.equal(d[key(115, 116)], -9);                // (s,t) 9 -> 0
  assert.equal(d[key(101, 256)], 9);                 // (e,st) new
  assert.equal(d[key(110, 101)], undefined);         // (n,e): -6 then +6 = no net change
  // the two halves of the update, as the figure shows them: subtract every old pair, add every new pair
  const minus = Object.fromEntries(r.minus), plus = Object.fromEntries(r.plus);
  assert.equal(minus[key(110, 101)], 6);              // (n,e) in newest x6: -6 ...
  assert.equal(plus[key(110, 101)], 6);               // ... and +6 again
  assert.equal(minus[key(101, 115)], 9);              // (e,s): newest x6 + widest x3
  assert.equal(plus[key(101, 115)], undefined);
  assert.equal(minus[key(115, 116)], 9);              // the merged pair itself
  assert.equal(plus[key(101, 256)], 9);               // (e,st) appears
  assert.equal(minus[key(108, 111)], undefined);      // low / lower are not touched
  assert.deepEqual(bpe.wordsWith(key(101, 256)), ['newest', 'widest']);   // inverted index after the step
  assert.deepEqual(bpe.wordsWith(key(101, 115)), []);                     // emptied, like the book's defaultdict
  for (const [k, v] of r.delta) assert.equal(v, (r.plus.get(k) ?? 0) - (r.minus.get(k) ?? 0));   // delta = net
});

test('chunkBoundaries (simulated from offsets) == find_chunk_boundaries on the real files', () => {
  const fx = fixture('boundaries.json');
  const eot = JSON.parse(data('valid_eot.json'));
  const offs = decodeOffsets(eot.deltas);
  assert.equal(offs.length, 27630);
  for (const [k, expected] of Object.entries(fx.valid)) {
    assert.deepEqual(chunkBoundaries(offs, eot.size, Number(k)).boundaries, expected, `valid k=${k}`);
  }
  assert.equal(chunkBoundaries(offs, eot.size, 100000).boundaries.length, fx.valid_count_100000);
  assert.equal(chunkBoundaries(offs, eot.size, 1000000).boundaries.length, fx.valid_count_1000000);   // fewer again
  const head = data('valid_head.txt');
  const bytes = Buffer.from(head, 'utf8');
  const headOffs = [];
  for (let i = bytes.indexOf(EOT); i !== -1; i = bytes.indexOf(EOT, i + 1)) headOffs.push(i);
  for (const [k, expected] of Object.entries(fx.head)) {
    assert.deepEqual(chunkBoundaries(headOffs, bytes.length, Number(k)).boundaries, expected, `head k=${k}`);
  }
});

test('amdahl: hand example from the text', () => {
  assert.ok(Math.abs(amdahl(12, 30, 8) - 42 / 31.5) < 1e-12);
  assert.ok(Math.abs(amdahl(12, 30, 1e9) - 1.4) < 1e-6);
  assert.equal(amdahl(10, 0, 4), 4);
});
