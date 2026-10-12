// web/widgets/_bpe.js against the real DLFS6 Python tokenizers.
//   fixtures:  uv run python tests/js/fixtures/lib/make_fixtures.py
//   run:       node --test tests/js/lib-bpe.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as bpe from '../../web/widgets/_bpe.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const readJSON = (rel) => JSON.parse(readFileSync(join(ROOT, rel), 'utf8'));
const fixture = (name) => readJSON(`tests/js/fixtures/lib/${name}`);
const BOTS = ['codebot', 'storybot', 'webbot'];
const toks = Object.fromEntries(BOTS.map((b) => [b, bpe.Tokenizer.fromJSON(readJSON(`web/data/tokenizers/${b}.json`))]));
const enc = fixture('encode.json');
const caseText = (name) => enc.cases.find((c) => c.name === name).text;
const hex = (bytes) => Buffer.from(bytes).toString('hex');

// ─── pre-tokenization ──────────────────────────────────────────────────────────────────────────────────────
test('pretokenize == regex.findall(GPT2_PATTERN, text) on the whole mixed corpus', () => {
  enc.cases.forEach(({ name, text }, i) => assert.deepEqual(bpe.pretokenize(text), enc.pretokens[i], name));
  // F17: the pieces cover every character, so pre-tokenization loses nothing.
  for (const { text } of enc.cases) assert.equal(bpe.pretokenize(text).join(''), text);
});

test('character classes match Python regex on every code point the JS engine knows', (t) => {
  const py = fixture('unicode.json');
  const flags = (ranges) => {
    const f = new Uint8Array(0x110000);
    for (const [a, b] of ranges) f.fill(1, a, b + 1);
    return f;
  };
  const L = flags(py.L), N = flags(py.N), S = flags(py.S);
  const reL = /^\p{L}$/u, reN = /^\p{N}$/u, reWS = new RegExp(`^[${bpe.WHITESPACE}]$`, 'u');
  const reNative = /^\s$/u, unassigned = /^\p{Cn}$/u;
  const gap = { L: 0, N: 0 };
  const nativeDiff = [];
  for (let c = 0; c < 0x110000; c++) {
    if (c >= 0xd800 && c <= 0xdfff) continue;
    const ch = String.fromCodePoint(c);
    for (const [k, re, flag] of [['L', reL, L], ['N', reN, N]]) {
      if (re.test(ch) !== (flag[c] === 1)) {
        assert.ok(unassigned.test(ch), `U+${c.toString(16)} classified differently but assigned in JS`);
        gap[k]++;
      }
    }
    assert.equal(reWS.test(ch), S[c] === 1, `WHITESPACE vs Python \\s at U+${c.toString(16)}`);
    if (reNative.test(ch) !== (S[c] === 1)) nativeDiff.push(c);
  }
  // Why WHITESPACE exists: JS's own \s differs from Python's in exactly these two code points.
  assert.deepEqual(nativeDiff, [0x85, 0xfeff]);
  t.diagnostic(`Python regex ${py.regex}: letters/numbers newer than this engine's Unicode tables: `
    + `${gap.L} L, ${gap.N} N (all unassigned in JS, Unicode ${process.versions.unicode})`);
});

test('pretokenizer() builds findall over a subset of the branches', () => {
  assert.equal(bpe.GPT2_PATTERN, readJSON('web/data/tokenizers/codebot.json').pattern);
  const noContractions = bpe.pretokenizer(bpe.GPT2_BRANCHES.slice(1));
  assert.deepEqual(bpe.pretokenize("print(f'done')"), ['print', '(', 'f', "'d", 'one', "')"]); // F16
  assert.deepEqual(noContractions("print(f'done')"), ['print', '(', 'f', "'", 'done', "')"]);
  assert.deepEqual(bpe.pretokenizer([])('abc'), []);
  assert.throws(() => bpe.pretokenize(42), TypeError);
});

test('pretokenizeSpans: the same pieces, their offsets, and the branch Python matched', () => {
  enc.cases.forEach(({ name, text }, i) => {
    const spans = bpe.pretokenizeSpans(text);
    assert.deepEqual(spans.map((s) => s.text), enc.pretokens[i], name);
    assert.deepEqual(spans.map((s) => s.branch), enc.branches[i], name);
    for (const s of spans) assert.equal(text.slice(s.start, s.end), s.text, name);
  });
  const pairs = (spans) => spans.map((s) => [s.text, s.branch]);
  assert.deepEqual(pairs(bpe.pretokenizeSpans("it's  2!\n")),
    [['it', 1], ["'s", 0], [' ', 4], [' 2', 2], ['!', 3], ['\n', 4]]);
  // With a subset, branch indexes refer to the array passed in; capturing groups inside a branch are allowed.
  assert.deepEqual(pairs(bpe.pretokenizeSpans("it's", bpe.GPT2_BRANCHES.slice(1))), [['it', 0], ["'", 2], ['s', 0]]);
  assert.deepEqual(pairs(bpe.pretokenizeSpans('abc', [{ js: '(a)(b)' }, { js: 'c' }])), [['ab', 0], ['c', 1]]);
});

test('splitSpecial behaves like re.split with a capturing group (F13), longest special first', () => {
  const E = bpe.ENDOFTEXT;
  assert.deepEqual(bpe.splitSpecial(`A${E}B${E}`, [E]), ['A', E, 'B', E, '']);
  assert.deepEqual(bpe.splitSpecial('no specials', [E]), ['no specials']);
  assert.deepEqual(bpe.splitSpecial(`${E}${E}`, [E, E + E]), ['', E + E, '']); // CS336 overlapping specials
  assert.deepEqual(bpe.splitSpecial('a.b*c', ['.', '*']), ['a', '.', 'b', '*', 'c']); // regex characters are literal
  assert.throws(() => bpe.splitSpecial('x', ['']), TypeError);
});

// ─── helpers from ch01/03 ──────────────────────────────────────────────────────────────────────────────────
test('countPairs and merge reproduce the book examples (F07, F08)', () => {
  const pairs = (m) => [...m].map(([k, c]) => [...bpe.unpair(k), c]);
  assert.deepEqual(pairs(bpe.countPairs([1, 2, 3, 1, 2])), [[1, 2, 2], [2, 3, 1], [3, 1, 1]]);
  assert.deepEqual(bpe.merge([1, 2, 3, 1, 2], [1, 2], 4), [4, 3, 4]);
  assert.deepEqual(pairs(bpe.countPairs([7, 7, 7])), [[7, 7, 2]]); // overlapping pairs are counted ...
  assert.deepEqual(bpe.merge([7, 7, 7], [7, 7], 8), [8, 7]);        // ... but merged left to right once
  assert.deepEqual(bpe.merge([7, 7, 7, 7], [7, 7], 8), [8, 8]);
  assert.deepEqual(pairs(bpe.countPairs([1, 2], bpe.countPairs([1, 2]), 3)), [[1, 2, 4]]);
});

// ─── encoding and decoding ─────────────────────────────────────────────────────────────────────────────────
test('encode == BPETokenizer.encode for codebot, storybot and webbot', () => {
  for (const bot of BOTS) {
    enc.cases.forEach(({ name, text }, i) => assert.deepEqual(toks[bot].encode(text), enc.ids[bot][i], `${bot}: ${name}`));
  }
});

test('decode(encode(text)) == text for every case and tokenizer', () => {
  for (const bot of BOTS) for (const { name, text } of enc.cases) {
    assert.equal(toks[bot].decode(toks[bot].encode(text)), text, `${bot}: ${name}`);
  }
});

test('the book examples: ch01/04, ch01/05 and special-token injection (F11, F12, F23)', () => {
  const rules04 = [[105, 115], [256, 32], [105, 110], [72, 101]];
  const t04 = new bpe.Tokenizer({ merges: rules04, pretokenize: false });
  assert.deepEqual(t04.encode('Hello世界😁'), [259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]);
  assert.equal(t04.decode(t04.encode('Hello世界😁')), 'Hello世界😁');
  const t05 = new bpe.Tokenizer({ merges: rules04.slice(0, 3), specialTokens: [bpe.ENDOFTEXT], pretokenize: false });
  assert.deepEqual([t05.specialTokens[bpe.ENDOFTEXT], t05.vocabSize], [259, 260]);
  assert.deepEqual(t05.encode('Hello world!<|endoftext|>'), [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 259]);
  const cb = toks.codebot;
  assert.deepEqual(cb.encode("print('<|endoftext|>')"), [350, 338, 999, 343]);
  const ordinary = cb.encode("print('<|endoftext|>')", { special: false });
  assert.ok(!ordinary.includes(999));
  assert.equal(cb.decode(ordinary), "print('<|endoftext|>')");
});

test('decode == Python decode(errors="replace") on random id sequences', () => {
  const dec = fixture('decode.json');
  for (const bot of BOTS) for (const [ids, text] of dec.ids[bot]) assert.equal(toks[bot].decode(ids), text, `${bot}: ${ids}`);
  assert.equal(toks.codebot.decode([]), '');
  const head = toks.codebot.decode([116, 528]);
  assert.equal(toks.codebot.decode(new Uint16Array([116, 528])), head); // a slice of a .bin file works
  assert.equal(toks.codebot.decode((function* () { yield 116; yield 528; })()), head); // so does any iterable
});

test('utf8Decode == bytes.decode("utf-8", "replace") on malformed UTF-8 (BOM kept)', () => {
  for (const [h, text] of fixture('decode.json').bytes) assert.equal(bpe.utf8Decode(Buffer.from(h, 'hex')), text, h);
  assert.equal(bpe.utf8Decode(Uint8Array.of(0xef, 0xbb, 0xbf, 0x41)), '\ufeffA'); // TextDecoder's default drops it
});

test('streamDecoder == codecs.getincrementaldecoder("utf-8")(errors="replace"), token by token', () => {
  const dec = fixture('decode.json');
  for (const bot of BOTS) for (const [ids, pieces] of dec.streams[bot]) {
    const s = toks[bot].streamDecoder();
    assert.deepEqual([...ids.map((id) => s.push(id)), s.flush()], pieces, `${bot}: ${ids}`);
  }
});

test('every token has the same bytes as id_to_bytes (sha256 over the whole vocabulary)', () => {
  const dec = fixture('decode.json');
  for (const bot of BOTS) {
    const tok = toks[bot];
    const { vocab_size: n, sha256, partial_utf8: partial } = dec.digest[bot];
    assert.equal(tok.vocabSize, n);
    const h = createHash('sha256');
    const strict = new TextDecoder('utf-8', { fatal: true });
    let broken = 0;
    for (let id = 0; id < n; id++) {
      const b = tok.tokenBytes(id);
      const len = Buffer.alloc(4);
      len.writeUInt32LE(b.length);
      h.update(len).update(b);
      try {
        strict.decode(b);
      } catch {
        broken++;
      }
    }
    assert.equal(h.digest('hex'), sha256, bot);
    assert.equal(broken, partial, `${bot}: tokens that are not valid UTF-8 on their own`);
  }
  assert.equal(dec.digest.webbot.partial_utf8 - 128, 157); // F35: 157 merged tokens are "half a character"
});

test('encodeChunks, mergeSteps, rank and tokenBytes agree with encode', () => {
  const cb = toks.codebot;
  const text = caseText('tinycodes');
  const chunks = cb.encodeChunks(text);
  assert.deepEqual(chunks.flatMap((c) => c.ids), cb.encode(text));
  assert.equal(chunks.map((c) => c.text).join(''), text);
  assert.ok(chunks.some((c) => c.special && c.ids[0] === 999));
  const steps = cb.mergeSteps(' BeautifulSoup');
  assert.deepEqual(steps.at(-1).ids, cb.encode(' BeautifulSoup'));
  assert.deepEqual(steps.at(-1).ids, [777]); // F20: ' BeautifulSoup' is a single CodeBot token
  const ranks = steps.slice(1).map((s) => s.newId);
  assert.deepEqual(ranks, [...ranks].sort((x, y) => x - y)); // merges are applied in training order
  assert.equal(cb.rank(32, 32), 256);
  assert.equal(cb.rank(999, 999), undefined);
  const copy = cb.tokenBytes(777);
  copy.fill(0);
  assert.equal(cb.decode([777]), ' BeautifulSoup');
  // A tokenizer is shared by every widget on a page, so its merge table is read-only.
  assert.throws(() => { cb.merges[0][0] = 1; }, TypeError);
  assert.throws(() => { steps[1].pair[0] = 1; }, TypeError);
  assert.deepEqual(cb.encode(' BeautifulSoup'), [777]);
});

test('invalid input fails loudly', () => {
  const T = bpe.Tokenizer;
  assert.throws(() => toks.codebot.decode([1000]), RangeError);
  assert.throws(() => toks.codebot.decode([-1]), RangeError);
  assert.throws(() => toks.codebot.tokenBytes(1.5), RangeError);
  assert.throws(() => toks.codebot.encode(null), TypeError);
  assert.throws(() => new T({ merges: [[300, 1]] }), RangeError);
  assert.throws(() => new T({ merges: [[1, 2], [1, 2]] }), RangeError);
  assert.throws(() => new T({ merges: [[1, 2, 300]] }), RangeError);
  assert.throws(() => new T({ merges: [[1, 2]], specialTokens: { '<|x|>': 300 } }), RangeError);
  assert.throws(() => new T({ merges: [], specialTokens: ['<|x|>', '<|x|>'] }), RangeError);
  assert.throws(() => new T({ merges: 'nope' }), TypeError);
  const json = readJSON('web/data/tokenizers/codebot.json');
  assert.throws(() => T.fromJSON({ ...json, format: 'other' }), TypeError);
  assert.throws(() => T.fromJSON({ ...json, vocab_size: 1001 }), RangeError);
  assert.throws(() => T.fromJSON({ ...json, pattern: '\\w+' }), TypeError);
  assert.throws(() => T.fromJSON({ ...json, merges: [1, 2, 3] }), TypeError);
  assert.throws(() => bpe.trainBPE('abc', 256), RangeError); // 256 bytes + <|endoftext|> need 257
  assert.throws(() => bpe.trainBPE('abc', 300.5), RangeError);
  assert.throws(() => bpe.trainBPE('abc', 300, { tieBreak: 'random' }), TypeError);
  assert.throws(() => bpe.trainBPE('abc', 300, { pretokenize: 'yes' }), TypeError);
});

// fetch() refuses these ports (Fetch standard, "bad port"); Windows may hand them out as ephemeral ports.
const FETCH_BAD_PORTS = new Set([1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69, 77, 79, 87, 95,
  101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 139, 143, 161, 179, 389, 427, 465, 512, 513,
  514, 515, 526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989, 990, 993, 995, 1719, 1720, 1723, 2049,
  3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669, 6679, 6697, 10080]);

async function listenFetchable(server) {
  for (;;) {
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    if (!FETCH_BAD_PORTS.has(server.address().port)) return `http://127.0.0.1:${server.address().port}`;
    await new Promise((r) => server.close(r));
  }
}

test('Tokenizer.load fetches once per url and does not cache failures', async () => {
  const body = readFileSync(join(ROOT, 'web/data/tokenizers/codebot.json'));
  let hits = 0;
  const server = createServer((req, res) => {
    hits++;
    if (req.url === '/codebot.json') res.writeHead(200, { 'content-type': 'application/json' }).end(body);
    else res.writeHead(404).end();
  });
  const base = await listenFetchable(server);
  try {
    const [a, b] = await Promise.all([bpe.Tokenizer.load(`${base}/codebot.json`), bpe.Tokenizer.load(`${base}/codebot.json`)]);
    assert.equal(a, b);
    assert.equal(hits, 1);
    assert.deepEqual(a.encode('print(x)'), toks.codebot.encode('print(x)'));
    await assert.rejects(bpe.Tokenizer.load(`${base}/missing.json`), /HTTP 404/);
    await assert.rejects(bpe.Tokenizer.load(`${base}/missing.json`), /HTTP 404/);
    assert.equal(hits, 3);
  } finally {
    server.close();
    server.closeAllConnections(); // fetch keeps sockets alive; don't hold the test process open
  }
});

test('encodes 10 KB of OpenWebText with the webbot merges in < 300 ms', (t) => {
  const json = readJSON('web/data/tokenizers/webbot.json');
  let t0 = performance.now();
  const web = bpe.Tokenizer.fromJSON(json);
  const build = performance.now() - t0;
  const text = caseText('owt-10k');
  assert.ok(bpe.utf8Encode(text).length >= 10_000);
  t0 = performance.now();
  const ids = web.encode(text); // cold: nothing cached yet
  const cold = performance.now() - t0;
  assert.deepEqual(ids, enc.ids.webbot[enc.cases.findIndex((c) => c.name === 'owt-10k')]);
  t.diagnostic(`webbot: build ${build.toFixed(0)} ms, `
    + `encode ${bpe.utf8Encode(text).length} B -> ${ids.length} ids in ${cold.toFixed(1)} ms`);
  assert.ok(cold < 300, `${cold.toFixed(0)} ms`);
});

// ─── training ──────────────────────────────────────────────────────────────────────────────────────────────
test('trainBPE == the Python trainers for every tie-break (merges, counts, ties, top-5)', () => {
  const { corpora, cases } = fixture('train.json');
  for (const c of cases) {
    const label = `${c.corpus} / ${c.tie_break} / pretokenize=${c.pretokenize} / special=${c.special_token}`;
    const text = corpora[c.corpus];
    const specials = c.special_token ? [c.special_token] : [];
    const res = bpe.trainBPE(text, c.vocab_size,
      { tieBreak: c.tie_break, pretokenize: c.pretokenize, specialToken: c.special_token });
    assert.deepEqual(res.merges, c.merges, label);
    const steps = res.trace.map((r) => [r.count, r.tied, r.top.map(({ pair, count }) => [...pair, count])]);
    assert.deepEqual(steps, c.steps, label);
    assert.equal(res.stoppedEarly, c.merges.length < c.vocab_size - 256 - specials.length, label);
    // Training leaves the text segmented exactly as encode() with the learned merges would.
    const tok = new bpe.Tokenizer({ merges: res.merges, specialTokens: specials, pretokenize: c.pretokenize });
    assert.equal(res.tokens, tok.encode(text).length, label);
    assert.equal(res.textBytes, bpe.utf8Encode(text).length, label);
  }
});

test('the ch01/03 example and the three tie-breaks on one sentence (F09, F10)', () => {
  const text = 'Hello world! This is BPE training.';
  const ch01 = { pretokenize: false, specialToken: null };
  const first = bpe.trainBPE(text, 260, { ...ch01, tieBreak: 'first' });
  assert.deepEqual(first.merges, [[105, 115], [256, 32], [105, 110], [72, 101]]);
  assert.deepEqual(first.trace.map((r) => r.tokens), [32, 30, 28, 27]); // 34 bytes -> 27 tokens
  assert.deepEqual(bpe.trainBPE(text, 257, { ...ch01, tieBreak: 'ids' }).merges, [[115, 32]]); // 's '
  assert.equal(first.trace[0].tied, 3); // 'is', 's ', 'in' all occur twice
  assert.equal(first.initialTokens, 34);
  const docs = [...Array(3).fill('zzq'), ...Array(3).fill('zz'), ...Array(3).fill('aaq'), 'aa', 'aa'].join(bpe.ENDOFTEXT);
  const words = (r) => r.merges.map((_, i) => bpe.utf8Decode(r.vocab[256 + i]));
  assert.deepEqual(words(bpe.trainBPE(docs, 260, { tieBreak: 'ids' })), ['zz', 'aa', 'aaq']);   // F44
  assert.deepEqual(words(bpe.trainBPE(docs, 260, { tieBreak: 'bytes' })), ['zz', 'aa', 'zzq']); // CS336
});

test('BPETrainer steps one merge at a time, consistently with trainBPE', () => {
  const text = caseText('book-say');
  const trainer = new bpe.BPETrainer(text, { tieBreak: 'first' });
  const ref = bpe.trainBPE(text, 270, { tieBreak: 'first' });
  assert.deepEqual(trainer.words.slice(0, 3).map((w) => [w.text, w.count]), [['Say', 1], [' hello', 3], ['!', 2]]);
  for (let i = 0; i < ref.merges.length; i++) {
    const best = trainer.best();
    assert.deepEqual(best.pair, trainer.top(3)[0].pair);
    const rec = trainer.step();
    assert.deepEqual(rec.pair, ref.merges[i]);
    assert.equal(trainer.pairCount(...rec.pair), 0);
  }
  assert.deepEqual(trainer.specialTokenIds(), { [bpe.ENDOFTEXT]: 269 });
  assert.equal(trainer.words.find((w) => w.text === ' hello').ids.length, 1); // F18: ' hello' became one token
});

test('the bytes tie-break reproduces CS336 A1 reference merges (corpus.en, vocab 500)', (t) => {
  const cs = fixture('cs336.json');
  const opts = { specialToken: cs.special_tokens, trace: false };
  const t0 = performance.now();
  const res = bpe.trainBPE(cs.text, cs.vocab_size, { ...opts, tieBreak: 'bytes' });
  const ms = performance.now() - t0;
  assert.deepEqual(res.merges.map(([a, b]) => [hex(res.vocab[a]), hex(res.vocab[b])]), cs.merges_hex);
  const ids = bpe.trainBPE(cs.text, cs.vocab_size, { ...opts, tieBreak: 'ids' });
  const diverge = ids.merges.findIndex(([a, b], i) => a !== res.merges[i][0] || b !== res.merges[i][1]);
  assert.equal(diverge + 1, cs.ids_rule_first_divergence); // the book's key parts ways with CS336 here
  t.diagnostic(`${cs.merges_hex.length}/${cs.merges_hex.length} CS336 merges in ${ms.toFixed(0)} ms; `
    + `the (count, id0, id1) key diverges at merge #${diverge + 1}`);
  assert.ok(ms < 1500, 'CS336 test_train_bpe_speed allows 1.5 s');
});

test('full TinyCodes: trainBPE reproduces merge_rules.pkl, encode reproduces tiny_codes.bin', { timeout: 180_000 }, (t) => {
  // Python reads the file in text mode, which turns a CRLF checkout back into LF (research F40).
  const text = readFileSync(join(ROOT, 'third_party/dlfs6/codebot/tiny_codes.txt'), 'utf8').replace(/\r\n/g, '\n');
  let t0 = performance.now();
  const res = bpe.trainBPE(text, 1000, { trace: 0 });
  const trainMs = performance.now() - t0;
  assert.deepEqual(res.merges.flat(), readJSON('web/data/tokenizers/codebot.json').merges);
  // Compression ratio (bytes per token) with the first k merges, as in research F50.
  const ratio = (vocab) => {
    const k = vocab - 257;
    const tokens = k === 0 ? res.initialTokens : res.trace[k - 1].tokens;
    return Math.round((res.textBytes / tokens) * 100) / 100;
  };
  assert.deepEqual([257, 357, 457, 657, 1000].map(ratio), [1.04, 1.62, 1.86, 2.14, 2.42]);
  t0 = performance.now();
  const ids = toks.codebot.encode(text);
  const encodeMs = performance.now() - t0;
  const buf = readFileSync(join(ROOT, 'third_party/dlfs6/codebot/tiny_codes.bin'));
  const bin = new Uint16Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  assert.equal(ids.length, 2_677_398);
  assert.equal(res.tokens, ids.length);
  assert.ok(ids.every((id, i) => id === bin[i]) && bin.length === ids.length, 'ids differ from tiny_codes.bin');
  t.diagnostic(`train 743 merges on ${res.textBytes.toLocaleString('en-US')} B in ${(trainMs / 1000).toFixed(1)} s; `
    + `encode -> ${ids.length.toLocaleString('en-US')} ids in ${(encodeMs / 1000).toFixed(1)} s`);
});
