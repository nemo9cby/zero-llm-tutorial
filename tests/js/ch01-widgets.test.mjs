// Chapter 1 widgets: the pure parts of web/widgets/ch01-*.js, checked against the numbers the page states.
// The BPE engine itself (_bpe.js) is checked against the Python code by lib-bpe.test.mjs; here we check what the
// widgets build on top of it, with the real data files the page loads.
//   run:  node --test tests/js/ch01-widgets.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Tokenizer, pretokenize, utf8Encode } from '../../web/widgets/_bpe.js';
import { tokenLabel, bytesRepr } from '../../web/widgets/ch01-util.js';
import { analyze, PRESETS as PLAY } from '../../web/widgets/ch01-playground.js';
import { piecesWithGaps, PRESETS as PRE } from '../../web/widgets/ch01-pretok.js';
import { Race, PRESETS as STEP } from '../../web/widgets/ch01-bpestep.js';
import { expandPieces, makeTrainer, compareWithOfficial, NUM_MERGES } from '../../web/widgets/ch01-train.js';
import { curveFor } from '../../web/widgets/ch01-compress.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const readJSON = (rel) => JSON.parse(readFileSync(join(ROOT, rel), 'utf8'));
const toks = Object.fromEntries(['codebot', 'storybot', 'webbot'].map((b) => [b, Tokenizer.fromJSON(readJSON(`web/data/tokenizers/${b}.json`))]));
const text = (bs) => new TextDecoder().decode(bs);

test('tokenLabel: visible whitespace, hex for a token that is only part of a character', () => {
  assert.deepEqual(tokenLabel(utf8Encode('\n   ')), { text: '↵␣␣␣', partial: false });
  assert.deepEqual(tokenLabel(Uint8Array.of(0xe2, 0x80)), { text: 'e2 80', partial: true });
  assert.equal(bytesRepr(utf8Encode('is ')), "'is␣'");
  assert.equal(bytesRepr(Uint8Array.of(0xe2, 0x80)), "b'\\xe2\\x80'");
});

test('playground: the preset counts the caption quotes (deep indentation vs a children\'s story)', () => {
  const count = (label, bot) => analyze(toks[bot], PLAY.find((p) => p.label === label).text).ids.length;
  assert.deepEqual(['codebot', 'storybot', 'webbot'].map((b) => count('深缩进', b)), [30, 41, 45]);
  assert.deepEqual(['codebot', 'storybot', 'webbot'].map((b) => count('儿童故事', b)), [43, 22, 22]);
  for (const p of PLAY) for (const tok of Object.values(toks)) assert.equal(analyze(tok, p.text).roundTrip, true, p.label);
  // the injection preset: one special token with the special-token switch on, plain bytes with it off
  const inj = PLAY.find((p) => p.label === '特殊 token 注入').text;
  assert.deepEqual(analyze(toks.codebot, inj).ids, [350, 338, 999, 343]);
  const ordinary = analyze(toks.codebot, inj, { special: false });
  assert.equal(ordinary.specialCount, 0);
  assert.ok(!ordinary.ids.includes(999) && ordinary.roundTrip);
});

test('figure 1-1 (chapter opener): the default text round-trips; each Chinese character of its comment is 3 byte tokens', () => {
  const page = readFileSync(join(ROOT, 'content', 'ch01.md'), 'utf8');
  const m = page.match(/```widget ch01-playground[^\n]*\n(.*)\n```/);
  const { text: opener, compact } = JSON.parse(m[1]);
  assert.equal(compact, true);
  const { ids, roundTrip } = analyze(toks.codebot, opener);
  assert.ok(roundTrip);
  const comment = [...utf8Encode('打招呼')];                  // 9 bytes, each one a token of its own (ids 0-255 are bytes)
  assert.equal(comment.length, 9);
  const at = ids.findIndex((_, i) => comment.every((b, k) => ids[i + k] === b));
  assert.ok(at >= 0, 'the comment is encoded byte by byte');
  // the indentation before '#' and 'print' is merged: newline + 3 spaces is one token (265), as in verify 1.9-a
  assert.equal(ids.filter((id) => id === 265).length, 2);
});

test('pretok viewer: all branches == pretokenize; switching branches off behaves as the caption says', () => {
  const all = [true, true, true, true, true, true];
  for (const p of PRE) {
    const pieces = piecesWithGaps(p.text, all);
    assert.deepEqual(pieces.map((x) => x.text), pretokenize(p.text), p.label);
    assert.ok(pieces.every((x) => x.branch !== null));
  }
  const noContraction = [false, true, true, true, true, true];
  assert.deepEqual(piecesWithGaps("print(f'done')", noContraction).map((x) => x.text), ['print', '(', 'f', "'", 'done', "')"]);
  const noLookahead = [true, true, true, true, false, true];
  assert.deepEqual(piecesWithGaps('    return x', noLookahead).map((x) => x.text), ['    ', 'return', ' x']);
  assert.deepEqual(piecesWithGaps('    return x', all).map((x) => x.text), ['   ', ' return', ' x']);
  // without the two whitespace branches, findall skips the spaces it cannot match
  const gaps = piecesWithGaps('a  b', [true, true, true, true, false, false]);
  assert.deepEqual(gaps.map((x) => [x.text, x.branch]), [['a', 1], [' ', null], [' b', 1]]);
});

test('BPE stepper: the book example, the zz/aa tie, the overlap preset, and the segmentation', () => {
  const book = new Race(STEP[0].text, STEP[0].pretokenize);
  for (let i = 0; i < 4; i++) book.step();
  assert.deepEqual(book.history.first.map((r) => text(r.bytes)), ['is', 'is ', 'in', 'He']);
  assert.equal(book.trainers.first.tokens, 27);
  assert.equal(book.firstDivergence(), 0);                    // 'is' vs 's ' already at step 1 (verify 1.3-c)
  assert.deepEqual(book.history.ids.map((r) => text(r.bytes)).slice(0, 1), ['s ']);

  const tie = new Race(STEP[1].text, STEP[1].pretokenize);
  for (let i = 0; i < 3; i++) tie.step();
  assert.deepEqual(Object.fromEntries(Object.entries(tie.history).map(([k, v]) => [k, v.map((r) => text(r.bytes))])),
    { first: ['zz', 'aa', 'zzq'], ids: ['zz', 'aa', 'aaq'], bytes: ['zz', 'aa', 'zzq'] });
  assert.equal(tie.firstDivergence(), 2);                     // the caption: the tie comes at step 3

  const overlap = new Race(STEP[2].text, STEP[2].pretokenize);
  overlap.step();
  assert.deepEqual([overlap.history.ids[0].count, overlap.history.ids[0].replaced], [6, 3]);

  const code = new Race(STEP[4].text, STEP[4].pretokenize);
  for (let i = 0; i < 25; i++) code.step();
  for (const rule of ['first', 'ids', 'bytes']) {
    const t = code.trainers[rule];
    const bytes = code.segmentation(rule).flatMap((s) => s.ids.flatMap((id) => [...t.vocab[id]]));
    assert.equal(text(Uint8Array.from(bytes)), STEP[4].text, rule);
  }
  // nextPicks does not disturb the trainer it inspects
  const before = code.trainers.ids.best();
  code.nextPicks('ids');
  assert.deepEqual(code.trainers.ids.best(), before);
});

test('in-browser training on the TinyCodes table: the numbers of 1.7.3 and the column', { timeout: 120_000 }, () => {
  const table = readJSON('web/data/ch01/tinycodes_pretokens.json');
  const expanded = expandPieces(table);
  assert.equal(expanded.length, 1_826_876);
  const official = toks.codebot.merges;
  const expect = { ids: [-1, 743, 0, 0], first: [122, 635, 0, 0], bytes: [289, 657, 1, 1] };
  for (const [rule, [first, same, extra, missing]] of Object.entries(expect)) {
    const tr = makeTrainer(expanded, rule);
    const records = [];
    while (tr.merges.length < NUM_MERGES) records.push(tr.step());
    const c = compareWithOfficial(tr, official, records);
    assert.deepEqual([c.first, c.same, c.onlyMine.length, c.onlyTheirs.length], [first, same, extra, missing], rule);
    assert.equal(tr.tokens + table.end_tokens, 2_677_398, rule);          // same compression for every rule
    if (rule === 'bytes') assert.deepEqual([text(c.onlyMine[0]), text(c.onlyTheirs[0])], ['tiful', 'eauti']);
    if (rule !== 'ids') assert.equal(c.firstRecord.tied, 2);
  }
});

test('compression curve: endpoints and the reference tokenizers of 1.8.2/1.8.3', () => {
  const data = readJSON('web/data/ch01/compression.json');
  const full = curveFor(data, 'full');
  assert.equal(full.points.length, 744);
  assert.deepEqual([full.points[0][0], full.points[743][0]], [257, 1000]);
  assert.deepEqual([full.points[0][1].toFixed(3), full.points[100][1].toFixed(3), full.points[743][1].toFixed(3)], ['1.042', '1.620', '2.423']);
  const r = (c, name) => c.bars.find((b) => b.name === name).ratio.toFixed(2);
  const head = curveFor(data, 'head');
  assert.deepEqual(['CodeBot', 'StoryBot', 'GPT-2'].map((n) => r(full, n)), ['2.42', '2.06', '2.64']);
  assert.deepEqual(['CodeBot', 'StoryBot', 'GPT-2'].map((n) => r(head, n)), ['2.12', '2.18', '3.09']);
  // the caption: GPT-2's lead shrinks from 0.97 to 0.21, StoryBot and CodeBot swap places
  assert.equal((head.bars.find((b) => b.name === 'GPT-2').ratio - head.bars.find((b) => b.me).ratio).toFixed(2), '0.97');
  assert.equal((full.bars.find((b) => b.name === 'GPT-2').ratio - full.bars.find((b) => b.me).ratio).toFixed(2), '0.21');
  assert.ok(full.bars.findIndex((b) => b.name === 'StoryBot') < full.bars.findIndex((b) => b.me));
  assert.ok(head.bars.findIndex((b) => b.name === 'StoryBot') > head.bars.findIndex((b) => b.me));
});
