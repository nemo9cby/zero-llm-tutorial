// Figure 7-1 (web/widgets/ch07-compare.js) against the book's Python tokenizers.
//   fixtures:  uv run python verify/_ch07_export.py        (storybot.tokenizer.BPETokenizer + the three .pkl)
//   run:       node --test tests/js/ch07-compare.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Tokenizer } from '../../web/widgets/_bpe.js';
import { BOTS, compare, tokenInfo, tokenLabel } from '../../web/widgets/ch07-compare.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const readJSON = (rel) => JSON.parse(readFileSync(join(ROOT, rel), 'utf8'));
const fixture = readJSON('tests/js/fixtures/ch07/compare.json');
const presets = readJSON('web/data/ch07/presets.json').presets;
const toks = BOTS.map((b) => Tokenizer.fromJSON(readJSON(`web/${b.url}`)));

test('the widget loads the three bots in order of vocabulary size', () => {
  assert.deepEqual(BOTS.map((b) => b.key), ['codebot', 'storybot', 'webbot']);
  assert.deepEqual(toks.map((t) => t.vocabSize), [1000, 10000, 50000]);
});

test('presets.json and the fixture describe the same texts', () => {
  assert.deepEqual(presets.map((p) => p.text), fixture.cases.map((c) => c.text));
});

test('compare() gives the ids of the book\'s BPETokenizer for every preset and bot', () => {
  for (const c of fixture.cases) {
    const { bytes, rows } = compare(toks, c.text);
    assert.equal(bytes, c.bytes, c.name);
    BOTS.forEach((b, i) => {
      assert.deepEqual(rows[i].ids, c.ids[b.key], `${c.name} / ${b.key}`);
      assert.equal(rows[i].tokens, c.ids[b.key].length);
      assert.equal(rows[i].bytesPerToken, c.bytes / c.ids[b.key].length);
    });
  }
});

test('special: false encodes <|endoftext|> as plain text, like encode_ordinary', () => {
  for (const c of fixture.cases) {
    const { rows } = compare(toks, c.text, { special: false });
    BOTS.forEach((b, i) => assert.deepEqual(rows[i].ids, c.ordinary[b.key], `${c.name} / ${b.key}`));
  }
  const inj = fixture.cases.find((c) => c.text.includes('<|endoftext|>'));
  const [special, plain] = [compare(toks, inj.text), compare(toks, inj.text, { special: false })];
  toks.forEach((t, i) => {
    assert.ok(special.rows[i].ids.includes(t.vocabSize - 1), 'the special id appears');
    assert.ok(!plain.rows[i].ids.includes(t.vocabSize - 1), 'and disappears in ordinary mode');
  });
});

test('chips show half characters as Python-style byte escapes', () => {
  const web = toks[2];
  const partial = [];
  for (let id = 256; id < web.vocabSize - 1; id++) if (tokenLabel(web, id).partial) partial.push(id);
  // 157 WebBot tokens are not valid UTF-8 on their own (repo-ch01 F35); the first is b'\xe2\x80'
  assert.equal(partial.length, 157);
  assert.equal(tokenLabel(web, partial[0]).text, '\\xe2\\x80');
  assert.deepEqual(tokenLabel(web, 256), { text: ' t', partial: false });
});

test('the status line names the token, its id and its bytes', () => {
  const web = toks[2];
  const label = (id) => tokenLabel(web, id);
  assert.equal(tokenInfo('WebBot', 0, 256, web.tokenBytes(256), label(256)),
    'WebBot 的第 1 个 token：id 256，字节 20 74（“ t”）');
  const half = [...Array(web.vocabSize - 257).keys()].map((k) => k + 256).find((id) => label(id).partial);
  assert.equal(tokenInfo('WebBot', 1_233, half, web.tokenBytes(half), label(half)),
    `WebBot 的第 1,234 个 token：id ${half.toLocaleString('en-US')}，字节 e2 80（单独不是完整的 UTF-8 字符）`);
  const eot = { text: '<|endoftext|>', partial: false, special: true };
  assert.equal(tokenInfo('CodeBot', 2, 999, new Uint8Array(), eot),
    'CodeBot 的第 3 个 token：id 999，特殊 token <|endoftext|>');
});
