/**
 * @module _bpe
 * @file Byte-level BPE for the browser, id-for-id identical to the DLFS6 Python tokenizers.
 *
 * Shared library (not a widget: it has no mount()). Import it like _lib.js:
 *
 *   import { Tokenizer, pretokenize, trainBPE, BPETrainer } from './_bpe.js';
 *   const tok = await Tokenizer.load('data/tokenizers/codebot.json');   // also storybot.json, webbot.json
 *   tok.encode("print('<|endoftext|>')")   // [350, 338, 999, 343]: same ids as codebot.tokenizer.BPETokenizer
 *   tok.decode([228, 184])                 // '\ufffd': same as b"\xe4\xb8".decode("utf-8", errors="replace")
 *
 * Every function is checked against the real Python code by tests/js/lib-bpe.test.mjs (reference outputs from
 * tests/js/fixtures/lib/make_fixtures.py). The JSON files come from tools/export_tokenizers.py.
 *
 * ── Pre-tokenization ──────────────────────────────────────────────────────────────────────────────────────
 * GPT2_PATTERN          string  the regex exactly as written in ch01/06 and codebot/tokenizer.py (Python syntax)
 * GPT2_BRANCHES         [{ py, js }] its six alternatives, as Python source and as JS RegExp source
 * WHITESPACE            string  JS character-class body equal to Python `regex`'s \s (Unicode White_Space,
 *                               25 code points). JS's own \s differs: it adds U+FEFF and lacks U+0085.
 * pretokenize(text)     string[]  = regex.findall(GPT2_PATTERN, text)
 * pretokenizer(branches = GPT2_BRANCHES) -> (text) => string[]   findall over a subset of the branches
 * pretokenizeSpans(text, branches?) -> [{ text, branch, start, end }]  the same pieces, with UTF-16 offsets and
 *                       the index of the branch that matched (0 contraction … 5 remaining spaces)
 * splitSpecial(text, specials) string[]  = re.split("(" + "|".join(map(re.escape, specials)) + ")", text):
 *                       even indices are ordinary text, odd indices special tokens (the longest one wins)
 *
 * ── Ports of the book's helpers ───────────────────────────────────────────────────────────────────────────
 * countPairs(ids, counts = new Map(), weight = 1) -> Map<pairKey, count>   overlapping pairs count: [7,7,7] -> 2
 * merge(ids, [a, b], newId) -> number[]   left to right, non-overlapping: merge([7,7,7], [7,7], 8) -> [8, 7]
 * pairKey(a, b) / unpair(key)             the numeric Map keys countPairs uses (ids must be < 2**21)
 * utf8Encode(text) -> Uint8Array          TextEncoder; a lone surrogate becomes U+FFFD (Python would raise)
 * utf8Decode(bytes) -> string             = bytes.decode("utf-8", errors="replace"): same U+FFFD, BOM kept
 *
 * ── Training ──────────────────────────────────────────────────────────────────────────────────────────────
 * trainBPE(text, vocabSize, options)
 *     -> { merges, trace, vocab, specialTokens, initialTokens, tokens, textBytes, stoppedEarly }
 *   options.tieBreak     'ids'   (default) codebot/storybot/webbot: max of (count, id0, id1)
 *                        'first' ch01/03–06: max(counts, key=counts.get), the first pair in scan order
 *                        'bytes' CS336 A1: the lexicographically greater (bytes0, bytes1)
 *   options.pretokenize  true (default, GPT-2 regex) | false (whole segments, ch01/03–05) | (text) => string[]
 *   options.specialToken '<|endoftext|>' (default) | null (ch01/03) | string[]. The text is split on it first, so
 *                        no merge crosses a document boundary, and it takes a vocab slot:
 *                        number of merges = vocabSize - 256 - (number of special tokens), as from ch01/05 on.
 *   options.trace        how many top pairs to record per step: 5 (default), 0 (none), or false (no trace at all)
 *   options.onStep(rec, trainer)  called after every merge (progress bars in a Web Worker)
 *   merges   [a, b][]    merge i creates id 256 + i: the order of merge_rules.pkl
 *   trace    [{ step, pair, newId, count, tied, top, tokens }]  count = frequency of the chosen pair; tied = number
 *            of pairs sharing that maximum; top = [{ pair, count }] in the tie-break's order (top[0] is the chosen
 *            pair); tokens = length of the whole training text after the merge (a special token counts 1), so
 *            textBytes / tokens is the compression ratio of ch01/08_eval.py
 *   vocab    Uint8Array[] bytes of ids 0 … 255 + merges.length;  specialTokens { token: id } after the merges
 *   initialTokens / tokens  length of the training text in tokens before the first / after the last merge
 *   stoppedEarly  true when the text ran out of pairs (the book's `if not counts: break`)
 * new BPETrainer(text, options)  the same algorithm one merge at a time (stepper widgets):
 *   .step({ top }) -> record | null   .best() -> { pair, count, tied } | null   .top(k) -> [{ pair, count }]
 *   .pairCount(a, b)   .merges   .vocab   .tokens   .textBytes   .specialCount
 *   .words -> [{ text, ids, count }]  distinct pre-tokens in first-occurrence order, ids = current segmentation
 *
 * ── Encoding and decoding ─────────────────────────────────────────────────────────────────────────────────
 * new Tokenizer({ merges, specialTokens = [], pretokenize = true, name })
 *   merges         [a, b][] in rank order ([a, b, newId] is accepted when newId === 256 + i)
 *   specialTokens  string[] (ids 256 + merges.length + i, the DLFS6 layout) or { token: id } using those ids
 * Tokenizer.load(url) -> Promise<Tokenizer>  JSON from tools/export_tokenizers.py, fetched once per url
 * Tokenizer.fromJSON(object)
 * tok.encode(text, { special = true })   special: false encodes "<|endoftext|>" as plain text (tiktoken's
 *                                        encode_ordinary); DLFS6 itself always maps it to the special id
 * tok.encodeChunks(text, { special })    [{ text, ids, special }] pre-token by pre-token, for token-chip widgets
 * tok.mergeSteps(piece)                  [{ ids, pair, newId }] how one pre-token is merged, step by step
 * tok.decode(ids)                        errors="replace"; an unknown id throws RangeError (Python: KeyError)
 * tok.tokenBytes(id) -> Uint8Array       a copy of id_to_bytes[id]
 * tok.streamDecoder() -> { push(id) -> string, flush() -> string }   incremental decoding for streamed output
 * tok.rank(a, b) -> newId | undefined;   tok.merges, tok.specialTokens, tok.numMerges, tok.vocabSize, tok.name
 *   (tok.merges and the pairs in trainer records are frozen: one tokenizer is shared by every widget on a page)
 *
 * Known limit (tested): \p{L} and \p{N} follow the JS engine's Unicode tables. The `regex` module pinned in
 * uv.lock (2026.9.29) ships newer tables than Node 24 (Unicode 16.0): 17,144 letters and 336 numbers it knows are
 * unassigned code points to V8, so text containing such recently encoded characters can pre-tokenize
 * differently. Every code point the engine knows is classified exactly as in Python.
 */

export const ENDOFTEXT = '<|endoftext|>';

// ─── pre-tokenization ──────────────────────────────────────────────────────────────────────────────────────
export const WHITESPACE = String.raw`\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000`;

export const GPT2_BRANCHES = Object.freeze([
  [String.raw`'(?:[sdmt]|ll|ve|re)`, String.raw`'(?:[sdmt]|ll|ve|re)`],     // contractions (lower case only)
  [String.raw` ?\p{L}+`, String.raw` ?\p{L}+`],                              // letters, one leading space
  [String.raw` ?\p{N}+`, String.raw` ?\p{N}+`],                              // numbers, one leading space
  [String.raw` ?[^\s\p{L}\p{N}]+`, String.raw` ?[^${WHITESPACE}\p{L}\p{N}]+`], // everything else
  [String.raw`\s+(?!\S)`, String.raw`[${WHITESPACE}]+(?![^${WHITESPACE}])`], // spaces, leaving one for the next word
  [String.raw`\s+`, String.raw`[${WHITESPACE}]+`],                            // the remaining spaces
].map(([py, js]) => Object.freeze({ py, js })));

export const GPT2_PATTERN = GPT2_BRANCHES.map((b) => b.py).join('|');

/**
 * A findall-style splitter over (a subset of) the GPT-2 branches, e.g. for a widget that switches branches off.
 * @param {ReadonlyArray<{py: string, js: string}>} [branches]
 * @returns {(text: string) => string[]}
 */
export function pretokenizer(branches = GPT2_BRANCHES) {
  const re = branches.length ? new RegExp(branches.map((b) => b.js).join('|'), 'gu') : null;
  return (text) => {
    if (typeof text !== 'string') throw new TypeError('pretokenize: text must be a string');
    return (re && text.match(re)) || [];
  };
}

/**
 * Pre-tokenize like DLFS6: `regex.findall(GPT2_PATTERN, text)`.
 * @type {(text: string) => string[]}
 */
export const pretokenize = pretokenizer();

const spanRegExps = new WeakMap(); // branches array -> [RegExp with one group per branch, group number per branch]

/**
 * pretokenize() with positions and the branch that matched each piece (for a pre-tokenizer viewer).
 * start/end are UTF-16 offsets into text; branch indexes the branches array (GPT2_BRANCHES: 0 = contraction,
 * 1 = letters, 2 = numbers, 3 = other symbols, 4 = spaces before a word, 5 = remaining spaces).
 * @param {string} text
 * @param {ReadonlyArray<{py: string, js: string}>} [branches]
 * @returns {{text: string, branch: number, start: number, end: number}[]}
 */
export function pretokenizeSpans(text, branches = GPT2_BRANCHES) {
  if (typeof text !== 'string') throw new TypeError('pretokenizeSpans: text must be a string');
  if (!branches.length) return [];
  let compiled = spanRegExps.get(branches);
  if (!compiled) {
    const groups = [];
    let group = 1;
    for (const b of branches) {
      groups.push(group);
      group += new RegExp(`${b.js}|`, 'u').exec('').length; // 1 + the branch's own capturing groups
    }
    compiled = [new RegExp(branches.map((b) => `(${b.js})`).join('|'), 'gu'), groups];
    spanRegExps.set(branches, compiled);
  }
  const [re, groups] = compiled;
  return [...text.matchAll(re)].map((m) => ({
    text: m[0],
    branch: groups.findIndex((g) => m[g] !== undefined),
    start: m.index,
    end: m.index + m[0].length,
  }));
}

const escapeRe = (s) => s.replace(/[\\^$.*+?()[\]{}|/]/g, '\\$&');

function specialRegExp(specials) {
  if (!specials.length) return null;
  for (const s of specials) {
    if (typeof s !== 'string' || !s) throw new TypeError('special tokens must be non-empty strings');
  }
  const longestFirst = [...specials].sort((a, b) => b.length - a.length);
  return new RegExp(longestFirst.map(escapeRe).join('|'), 'gu');
}

function splitWith(text, re) {
  if (!re) return [text];
  const out = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    out.push(text.slice(last, m.index), m[0]);
    last = m.index + m[0].length;
  }
  out.push(text.slice(last));
  return out;
}

/**
 * `re.split("(" + "|".join(map(re.escape, specials)) + ")", text)`: odd indices hold the special tokens.
 * @param {string} text
 * @param {string|string[]} specials
 * @returns {string[]}
 */
export function splitSpecial(text, specials) {
  if (typeof text !== 'string') throw new TypeError('splitSpecial: text must be a string');
  return splitWith(text, specialRegExp([].concat(specials ?? [])));
}

function pieceSplitter(option) {
  if (option === true) return pretokenize;
  if (option === false) return (text) => (text ? [text] : []);
  if (typeof option === 'function') return option;
  throw new TypeError('pretokenize must be true, false or a function (text) => string[]');
}

// ─── small ports of the book's helpers ─────────────────────────────────────────────────────────────────────
const PAIR = 2 ** 21; // ids < 2^21 keep a * PAIR + b exact in a double, and key order == (a, b) order
export const pairKey = (a, b) => a * PAIR + b;
export const unpair = (key) => [Math.floor(key / PAIR), key % PAIR];

export function countPairs(ids, counts = new Map(), weight = 1) {
  for (let i = 0; i + 1 < ids.length; i++) {
    const k = pairKey(ids[i], ids[i + 1]);
    counts.set(k, (counts.get(k) ?? 0) + weight);
  }
  return counts;
}

export function merge(ids, [a, b], newId) {
  const out = [];
  for (let i = 0; i < ids.length;) {
    if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) {
      out.push(newId);
      i += 2;
    } else {
      out.push(ids[i++]);
    }
  }
  return out;
}

const ENCODER = new TextEncoder();
const DECODER = new TextDecoder('utf-8', { ignoreBOM: true }); // not fatal: U+FFFD like errors="replace"
export const utf8Encode = (text) => ENCODER.encode(text);
export const utf8Decode = (bytes) => DECODER.decode(bytes);

function compareBytes(x, y) {
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
}

// ─── training ──────────────────────────────────────────────────────────────────────────────────────────────
const TIE_BREAKS = ['ids', 'first', 'bytes'];
const specialList = (option) => (option == null ? [] : [].concat(option));

/**
 * Step-by-step BPE training with the book's semantics (options as for trainBPE).
 *
 * Counts pairs once, then after each merge updates only the pre-tokens that contained the merged pair
 * (ch04's idea). Distinct pre-tokens are kept with their frequencies in first-occurrence order, which is the
 * order in which the book's full recount inserts pairs into its dict; that makes 'first' exact as well.
 */
export class BPETrainer {
  constructor(text, { tieBreak = 'ids', pretokenize: pre = true, specialToken = ENDOFTEXT } = {}) {
    if (typeof text !== 'string') throw new TypeError('BPETrainer: text must be a string');
    if (!TIE_BREAKS.includes(tieBreak)) throw new TypeError(`tieBreak must be one of: ${TIE_BREAKS.join(', ')}`);
    this.tieBreak = tieBreak;
    this.specials = specialList(specialToken);
    const pieces = pieceSplitter(pre);
    const parts = splitWith(text, specialRegExp(this.specials));
    this.specialCount = (parts.length - 1) / 2;

    this.words = [];
    const index = new Map(); // piece -> position in this.words
    for (let p = 0; p < parts.length; p += 2) {
      for (const piece of pieces(parts[p])) {
        const w = index.get(piece);
        if (w !== undefined) this.words[w].count++;
        else {
          index.set(piece, this.words.length);
          this.words.push({ text: piece, ids: Array.from(utf8Encode(piece)), count: 1 });
        }
      }
    }
    this.counts = new Map(); // pairKey -> weighted frequency
    this.where = new Map();  // pairKey -> Set of word positions containing the pair
    this.tokens = this.specialCount;
    this.words.forEach((word, w) => {
      this.tokens += word.ids.length * word.count;
      this._add(w);
    });
    this.textBytes = utf8Encode(text).length;
    this.vocab = Array.from({ length: 256 }, (_, i) => Uint8Array.of(i));
    this.merges = [];
  }

  _add(w) {
    const { ids, count } = this.words[w];
    for (let i = 0; i + 1 < ids.length; i++) {
      const k = pairKey(ids[i], ids[i + 1]);
      this.counts.set(k, (this.counts.get(k) ?? 0) + count);
      let set = this.where.get(k);
      if (!set) this.where.set(k, (set = new Set()));
      set.add(w);
    }
  }

  _remove(w) {
    const { ids, count } = this.words[w];
    for (let i = 0; i + 1 < ids.length; i++) {
      const k = pairKey(ids[i], ids[i + 1]);
      const c = this.counts.get(k) - count;
      if (c > 0) this.counts.set(k, c);
      else this.counts.delete(k);
      const set = this.where.get(k);
      if (set) {
        set.delete(w);
        if (!set.size) this.where.delete(k);
      }
    }
  }

  // Where the book's scan meets the pair first: (pre-token position, position inside its current ids).
  _first(k) {
    let w = Infinity;
    for (const i of this.where.get(k)) if (i < w) w = i;
    const [a, b] = unpair(k);
    const ids = this.words[w].ids;
    for (let i = 0; i + 1 < ids.length; i++) if (ids[i] === a && ids[i + 1] === b) return [w, i];
    throw new Error('BPETrainer: pair index out of sync');
  }

  // compare(x, y) > 0 when pair key x wins a tie against y.
  _comparator() {
    const memo = new Map();
    const first = (k) => {
      let r = memo.get(k);
      if (!r) memo.set(k, (r = this._first(k)));
      return r;
    };
    const earlier = (x, y) => {
      const [xw, xi] = first(x);
      const [yw, yi] = first(y);
      return (yw - xw) || (yi - xi);
    };
    if (this.tieBreak === 'ids') return (x, y) => x - y;
    if (this.tieBreak === 'first') return earlier;
    return (x, y) => { // 'bytes'; two ids with equal bytes would fall back to scan order, like Python's max()
      const [xa, xb] = unpair(x);
      const [ya, yb] = unpair(y);
      return compareBytes(this.vocab[xa], this.vocab[ya]) || compareBytes(this.vocab[xb], this.vocab[yb])
        || earlier(x, y);
    };
  }

  pairCount(a, b) {
    return this.counts.get(pairKey(a, b)) ?? 0;
  }

  best() {
    let max = 0;
    for (const c of this.counts.values()) if (c > max) max = c;
    if (max === 0) return null;
    const better = this._comparator();
    let key = -1;
    let tied = 0;
    for (const [k, c] of this.counts) {
      if (c !== max) continue;
      tied++;
      if (key < 0 || better(k, key) > 0) key = k;
    }
    return { key, pair: unpair(key), count: max, tied };
  }

  top(k = 5) {
    k = Math.floor(k);
    if (!(k > 0) || !this.counts.size) return [];
    const largest = []; // the k largest counts seen so far, descending
    for (const c of this.counts.values()) {
      if (largest.length === k && c <= largest[k - 1]) continue;
      if (largest.length === k) largest[k - 1] = c;
      else largest.push(c);
      for (let i = largest.length - 1; i > 0 && largest[i - 1] < largest[i]; i--) {
        [largest[i - 1], largest[i]] = [largest[i], largest[i - 1]];
      }
    }
    const threshold = largest[largest.length - 1];
    const better = this._comparator();
    const cands = [];
    for (const [key, c] of this.counts) if (c >= threshold) cands.push([key, c]);
    cands.sort((x, y) => (y[1] - x[1]) || better(y[0], x[0]));
    return cands.slice(0, k).map(([key, count]) => ({ pair: unpair(key), count }));
  }

  step({ top = 0 } = {}) {
    const best = this.best();
    if (!best) return null;
    const { key, count, tied } = best;
    const pair = Object.freeze(best.pair); // shared by the record and this.merges
    const newId = 256 + this.merges.length;
    const rec = { step: this.merges.length, pair, newId, count, tied };
    if (top > 0) rec.top = this.top(top);
    for (const w of [...this.where.get(key)]) {
      const word = this.words[w];
      this._remove(w);
      const before = word.ids.length;
      word.ids = merge(word.ids, pair, newId);
      this.tokens -= (before - word.ids.length) * word.count;
      this._add(w);
    }
    this.counts.delete(key); // already zero: a merged pair cannot reappear
    this.where.delete(key);
    const [a, b] = pair;
    const bytes = new Uint8Array(this.vocab[a].length + this.vocab[b].length);
    bytes.set(this.vocab[a]);
    bytes.set(this.vocab[b], this.vocab[a].length);
    this.vocab.push(bytes);
    this.merges.push(pair);
    rec.tokens = this.tokens;
    return rec;
  }

  specialTokenIds() {
    return Object.fromEntries(this.specials.map((t, i) => [t, 256 + this.merges.length + i]));
  }
}

/**
 * Train byte-level BPE exactly like the book's train_bpe (the module header describes every option and field).
 * @param {string} text
 * @param {number} vocabSize 256 byte ids + learned merges + special tokens
 * @param {{tieBreak?: 'ids'|'first'|'bytes', pretokenize?: boolean|((text: string) => string[]),
 *          specialToken?: string|string[]|null, trace?: number|boolean,
 *          onStep?: (rec: object, trainer: BPETrainer) => void}} [options]
 * @returns {{merges: number[][], trace: object[], vocab: Uint8Array[], specialTokens: Object<string, number>,
 *            initialTokens: number, tokens: number, textBytes: number, stoppedEarly: boolean}}
 */
export function trainBPE(text, vocabSize, options = {}) {
  const { trace = 5, onStep } = options;
  const reserved = 256 + specialList(options.specialToken === undefined ? ENDOFTEXT : options.specialToken).length;
  if (!Number.isInteger(vocabSize) || vocabSize < reserved) {
    throw new RangeError(`vocabSize must be an integer >= ${reserved} (256 bytes + special tokens)`);
  }
  const trainer = new BPETrainer(text, options);
  const numMerges = vocabSize - reserved;
  const top = trace === true ? 5 : trace === false ? 0 : trace;
  const steps = [];
  const initialTokens = trainer.tokens;
  while (trainer.merges.length < numMerges) {
    const rec = trainer.step({ top });
    if (!rec) break;
    if (trace !== false) steps.push(rec);
    onStep?.(rec, trainer);
  }
  return {
    merges: trainer.merges,
    trace: steps,
    vocab: trainer.vocab,
    specialTokens: trainer.specialTokenIds(),
    initialTokens,
    tokens: trainer.tokens,
    textBytes: trainer.textBytes,
    stoppedEarly: trainer.merges.length < numMerges,
  };
}

// ─── encoding and decoding ─────────────────────────────────────────────────────────────────────────────────
const CACHE_LIMIT = 50_000; // distinct pre-tokens remembered per tokenizer
const loading = new Map();  // url -> Promise<Tokenizer>

/** Encoder/decoder with the semantics of DLFS6's BPETokenizer (codebot and storybot versions give equal ids). */
export class Tokenizer {
  /**
   * @param {{merges: ReadonlyArray<ReadonlyArray<number>>, specialTokens?: string[]|Object<string, number>,
   *          pretokenize?: boolean|((text: string) => string[]), name?: string}} options
   */
  constructor({ merges, specialTokens = [], pretokenize: pre = true, name = '' } = {}) {
    if (!Array.isArray(merges)) throw new TypeError('Tokenizer: merges must be an array of [a, b] pairs');
    const n = merges.length;
    this.name = name;
    this._rank = new Map(); // pairKey -> id the pair merges into (smaller id = learned earlier = applied first)
    const pairs = new Array(n);
    const lens = new Uint32Array(256 + n).fill(1); // byte length of every token
    for (let i = 0; i < n; i++) {
      const m = merges[i];
      const id = 256 + i;
      const a = m?.[0];
      const b = m?.[1];
      if (!(Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b >= 0 && a < id && b < id)) {
        throw new RangeError(`merge #${i} [${a}, ${b}] must join ids that exist before id ${id}`);
      }
      if (m.length > 2 && m[2] !== id) throw new RangeError(`merge #${i} creates id ${m[2]}, expected ${id}`);
      const size = this._rank.size;
      this._rank.set(pairKey(a, b), id);
      if (this._rank.size === size) throw new RangeError(`merge #${i} [${a}, ${b}] appears twice`);
      pairs[i] = Object.freeze([a, b]); // shared with every widget on the page: read-only
      lens[id] = lens[a] + lens[b];
    }
    this.merges = Object.freeze(pairs);

    const specials = Array.isArray(specialTokens)
      ? specialTokens.map((t, i) => [t, 256 + n + i])
      : Object.entries(specialTokens ?? {});
    specials.sort((x, y) => x[1] - y[1]);
    specials.forEach(([, id], i) => {
      if (id !== 256 + n + i) {
        throw new RangeError(`special token ids must be ${256 + n} … ${256 + n + specials.length - 1}, got ${id}`);
      }
    });
    if (new Set(specials.map(([t]) => t)).size !== specials.length) throw new RangeError('duplicate special token');
    this._specialRe = specialRegExp(specials.map(([t]) => t));
    this._specialId = new Map(specials);
    this.specialTokens = Object.fromEntries(specials);
    this.numMerges = n;
    this.vocabSize = 256 + n + specials.length;

    // Every token's bytes, stored back to back in id order: token i is buf[off[i] .. off[i + 1]).
    const specialBytes = specials.map(([t]) => utf8Encode(t));
    const total = lens.reduce((s, x) => s + x, 0) + specialBytes.reduce((s, b) => s + b.length, 0);
    const buf = (this._buf = new Uint8Array(total));
    const off = (this._off = new Uint32Array(this.vocabSize + 1));
    for (let i = 0; i < 256; i++) buf[i] = off[i] = i;
    let at = 256;
    for (let i = 0; i < n; i++) {
      const [a, b] = pairs[i];
      off[256 + i] = at;
      for (let j = off[a], end = j + lens[a]; j < end; j++) buf[at++] = buf[j];
      for (let j = off[b], end = j + lens[b]; j < end; j++) buf[at++] = buf[j];
    }
    specialBytes.forEach((bytes, i) => {
      off[256 + n + i] = at;
      buf.set(bytes, at);
      at += bytes.length;
    });
    off[this.vocabSize] = at;

    this._pieces = pieceSplitter(pre);
    this._cache = new Map();
  }

  /**
   * Build from the parsed JSON written by tools/export_tokenizers.py; validates format, pattern and vocab size.
   * @param {{format: string, merges: number[], special_tokens?: Object<string, number>, vocab_size?: number,
   *          pattern?: string, name?: string}} data
   * @returns {Tokenizer}
   */
  static fromJSON(data) {
    if (data?.format !== 'dlfs6-bpe/1') {
      throw new TypeError('not a tokenizer exported by tools/export_tokenizers.py (format "dlfs6-bpe/1")');
    }
    if (data.pattern !== undefined && data.pattern !== GPT2_PATTERN) {
      throw new TypeError('this tokenizer uses a pre-tokenizer pattern _bpe.js does not implement');
    }
    const flat = data.merges;
    if (!Array.isArray(flat) || flat.length % 2) throw new TypeError('merges must be a flat array of id pairs');
    const merges = [];
    for (let i = 0; i < flat.length; i += 2) merges.push([flat[i], flat[i + 1]]);
    const tok = new Tokenizer({ merges, specialTokens: data.special_tokens ?? {}, name: data.name ?? '' });
    if (data.vocab_size !== undefined && data.vocab_size !== tok.vocabSize) {
      throw new RangeError(`vocab_size ${data.vocab_size} does not match ${tok.vocabSize} ids`);
    }
    return tok;
  }

  /**
   * Fetch a tokenizer JSON; each url is fetched once per page. A relative url resolves against the page, so
   * widgets use 'data/tokenizers/codebot.json' (or storybot.json, webbot.json).
   * @param {string|URL} url
   * @returns {Promise<Tokenizer>}
   */
  static load(url) {
    const key = String(url);
    let promise = loading.get(key);
    if (!promise) {
      promise = fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(`${key}: HTTP ${r.status}`);
          return r.json();
        })
        .then((data) => Tokenizer.fromJSON(data));
      loading.set(key, promise);
      promise.catch(() => loading.delete(key)); // let a later call retry
    }
    return promise;
  }

  /** @returns {number|undefined} the id that the pair (a, b) merges into, if it is a merge rule */
  rank(a, b) {
    return this._rank.get(pairKey(a, b));
  }

  // storybot's _encode_text: keep merging the adjacent pair with the lowest rank. Applying all merges in
  // training order (codebot's version) gives the same ids, because a merge only creates pairs of later rank.
  _bpe(bytes, onMerge) {
    let ids = Array.from(bytes);
    while (ids.length > 1) {
      let best = Infinity;
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = this._rank.get(pairKey(ids[i], ids[i + 1]));
        if (r !== undefined && r < best) best = r;
      }
      if (best === Infinity) break;
      const pair = this.merges[best - 256];
      ids = merge(ids, pair, best);
      onMerge?.(ids, pair, best);
    }
    return ids;
  }

  _encodePiece(piece) {
    let ids = this._cache.get(piece);
    if (!ids) {
      if (this._cache.size >= CACHE_LIMIT) this._cache.clear();
      ids = this._bpe(utf8Encode(piece));
      this._cache.set(piece, ids);
    }
    return ids;
  }

  _parts(text, special) {
    if (typeof text !== 'string') throw new TypeError('encode: text must be a string');
    return special ? splitWith(text, this._specialRe) : [text];
  }

  /**
   * Text to token ids, exactly as BPETokenizer.encode in DLFS6.
   * @param {string} text
   * @param {{special?: boolean}} [opts] special: false encodes special-token strings as plain text
   * @returns {number[]}
   */
  encode(text, { special = true } = {}) {
    const out = [];
    const parts = this._parts(text, special);
    for (let p = 0; p < parts.length; p++) {
      if (p % 2) {
        out.push(this._specialId.get(parts[p]));
        continue;
      }
      for (const piece of this._pieces(parts[p])) {
        const ids = this._encodePiece(piece);
        for (let i = 0; i < ids.length; i++) out.push(ids[i]);
      }
    }
    return out;
  }

  /**
   * encode() pre-token by pre-token; concatenating every chunk's ids gives encode(text).
   * @returns {{text: string, ids: number[], special: boolean}[]}
   */
  encodeChunks(text, { special = true } = {}) {
    const out = [];
    this._parts(text, special).forEach((part, p) => {
      if (p % 2) {
        out.push({ text: part, ids: [this._specialId.get(part)], special: true });
        return;
      }
      for (const piece of this._pieces(part)) {
        out.push({ text: piece, ids: this._encodePiece(piece).slice(), special: false });
      }
    });
    return out;
  }

  /**
   * How one pre-token is merged: its UTF-8 bytes first, then one entry per merge (lowest rank first).
   * @param {string} piece
   * @returns {{ids: number[], pair: ReadonlyArray<number>|null, newId: number|null}[]}
   */
  mergeSteps(piece) {
    const bytes = utf8Encode(piece);
    const steps = [{ ids: Array.from(bytes), pair: null, newId: null }];
    this._bpe(bytes, (ids, pair, newId) => steps.push({ ids, pair, newId }));
    return steps;
  }

  _check(id) {
    if (!(Number.isInteger(id) && id >= 0 && id < this.vocabSize)) throw new RangeError(`unknown token id ${id}`);
  }

  _view(id) {
    this._check(id);
    return this._buf.subarray(this._off[id], this._off[id + 1]);
  }

  /** @param {number} id @returns {Uint8Array} a copy of the token's bytes (id_to_bytes[id]) */
  tokenBytes(id) {
    return this._view(id).slice();
  }

  /**
   * Token ids to text: `b"".join(id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")`.
   * @param {Iterable<number>} ids an array, a typed array (e.g. a slice of a .bin file) or any iterable
   * @returns {string}
   */
  decode(ids) {
    const list = Array.isArray(ids) || ArrayBuffer.isView(ids) ? ids : Array.from(ids); // read twice below
    const { _buf: buf, _off: off } = this;
    let total = 0;
    for (const id of list) {
      this._check(id);
      total += off[id + 1] - off[id];
    }
    const bytes = new Uint8Array(total);
    let at = 0;
    for (const id of list) for (let j = off[id], end = off[id + 1]; j < end; j++) bytes[at++] = buf[j];
    return utf8Decode(bytes);
  }

  /**
   * Decode a stream one token at a time without printing a broken character between two tokens.
   * @returns {{push: (id: number) => string, flush: () => string}}
   */
  streamDecoder() {
    const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
    return {
      push: (id) => decoder.decode(this._view(id), { stream: true }),
      flush: () => decoder.decode(),
    };
  }
}
