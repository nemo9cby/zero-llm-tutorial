/* 第 4 章组件共用的纯函数（无 DOM，Node 测试直接 import）。
   pretokenize   GPT-2 预分词正则的 JS 版（与 Python regex 的切分逐个相同，见 tests/js/ch04-bpe.test.mjs）
   IncrementalBPE  ch04/02 的增量训练：计数表 + 倒排索引，每步返回“谁受影响、计数怎样增减”
   chunkBoundaries ch04/03 的 find_chunk_boundaries，只用 <|endoftext|> 的字节偏移来模拟（含 4096 字节缓冲区的细节）
   amdahl          Amdahl 定律 */

export const PRETOKENIZE_RE = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
export const EOT = '<|endoftext|>';
const enc = new TextEncoder();

export function pretokenize(text) {
  return text.match(PRETOKENIZE_RE) || [];
}

/** Split on <|endoftext|> and pretokenize each piece (what every ch04 trainer does first). */
export function pretokenizeCorpus(text) {
  return text.split(EOT).flatMap(pretokenize);
}

/** Occurrence list -> [{word, count}] in first-seen order (Python dict order). */
export function countWords(words) {
  const m = new Map();
  for (const w of words) m.set(w, (m.get(w) || 0) + 1);
  return [...m].map(([word, count]) => ({ word, count }));
}

export const pairKey = (a, b) => a * 1048576 + b;           // ids stay far below 2^20
export const unpair = (k) => [Math.floor(k / 1048576), k % 1048576];

export function merge(ids, a, b, newId) {
  const out = [];
  for (let i = 0; i < ids.length;) {
    if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(newId); i += 2; } else { out.push(ids[i]); i += 1; }
  }
  return out;
}

export function pairsOf(ids) {
  const m = new Map();
  for (let i = 0; i + 1 < ids.length; i++) {
    const k = pairKey(ids[i], ids[i + 1]);
    m.set(k, (m.get(k) || 0) + 1);
  }
  return m;
}

/** The book's choice: max over (count, id0, id1) — a total order, so dict order never matters. */
export function bestPair(pairCounts) {
  let best = null, bc = -1;
  for (const [k, c] of pairCounts) {
    if (c > bc || (c === bc && k > best)) { best = k; bc = c; }   // k orders like (id0, id1)
  }
  return best;
}

/**
 * ch04/02 as a state machine. seqs: [{word, count, ids}] (unique pretokens).
 * step() mirrors the book's loop body and reports what changed.
 */
export class IncrementalBPE {
  constructor(words) {
    this.seqs = words.map(({ word, count }) => ({ word, count, ids: [...enc.encode(word)] }));
    this.vocab = new Map([...Array(256).keys()].map((i) => [i, new Uint8Array([i])]));
    this.merges = [];
    this.pairCounts = new Map();
    this.pairToSeqs = new Map();
    this.seqs.forEach((s, si) => {
      for (const [k, c] of pairsOf(s.ids)) {
        this.pairCounts.set(k, (this.pairCounts.get(k) || 0) + c * s.count);
        if (!this.pairToSeqs.has(k)) this.pairToSeqs.set(k, new Set());
        this.pairToSeqs.get(k).add(si);
      }
    });
  }

  get done() { return this.pairCounts.size === 0; }

  step() {
    if (this.done) return null;
    const k = bestPair(this.pairCounts);
    const [a, b] = unpair(k);
    const newId = 256 + this.merges.length;
    const count = this.pairCounts.get(k);
    this.merges.push([a, b]);
    const va = this.vocab.get(a), vb = this.vocab.get(b);
    const joined = new Uint8Array(va.length + vb.length); joined.set(va); joined.set(vb, va.length);
    this.vocab.set(newId, joined);
    const affected = [...(this.pairToSeqs.get(k) || [])].sort((x, y) => x - y);
    this.pairToSeqs.delete(k);
    const delta = new Map();                       // pair -> net change this step
    const add = (p, d) => delta.set(p, (delta.get(p) || 0) + d);
    const changes = [];
    for (const si of affected) {
      const s = this.seqs[si];
      const before = s.ids;
      const after = merge(before, a, b, newId);
      for (const [p, c] of pairsOf(before)) {        // subtract the whole old sequence
        const v = this.pairCounts.get(p) - c * s.count;
        if (v <= 0) this.pairCounts.delete(p); else this.pairCounts.set(p, v);
        this.pairToSeqs.get(p)?.delete(si);
        add(p, -c * s.count);
      }
      for (const [p, c] of pairsOf(after)) {         // add the whole new sequence
        this.pairCounts.set(p, (this.pairCounts.get(p) || 0) + c * s.count);
        if (!this.pairToSeqs.has(p)) this.pairToSeqs.set(p, new Set());
        this.pairToSeqs.get(p).add(si);
        add(p, c * s.count);
      }
      s.ids = after;
      changes.push({ si, before, after });
    }
    for (const [p, d] of delta) if (d === 0) delta.delete(p);
    return { pair: [a, b], newId, count, affected, changes, delta };
  }

  /** Full recount (the 01 way) — used to show that the incremental table is always exact. */
  recount() {
    const m = new Map();
    for (const s of this.seqs) for (const [k, c] of pairsOf(s.ids)) m.set(k, (m.get(k) || 0) + c * s.count);
    return m;
  }

  consistent() {
    const full = this.recount();
    if (full.size !== this.pairCounts.size) return false;
    for (const [k, c] of full) if (this.pairCounts.get(k) !== c) return false;
    for (const [k, set] of this.pairToSeqs) {
      for (const si of set) if (!pairsOf(this.seqs[si].ids).has(k)) return false;
    }
    return true;
  }

  text(id) { return new TextDecoder().decode(this.vocab.get(id)); }
}

/**
 * ch04/03 find_chunk_boundaries, simulated from the byte offsets of every <|endoftext|>.
 * Mirrors the 4096-byte read loop: a marker that straddles two reads is not seen.
 */
export function chunkBoundaries(eotOffsets, fileSize, numChunks, bufferSize = 4096, markerLen = 13) {
  const chunkSize = Math.floor(fileSize / numChunks);
  const raw = [...Array(numChunks).keys()].map((i) => i * chunkSize);
  raw.push(fileSize);
  const snapped = raw.slice();
  for (let bi = 1; bi < raw.length - 1; bi++) {
    let pos = raw[bi];
    let j = lowerBound(eotOffsets, pos);
    for (;;) {
      if (pos >= fileSize) { snapped[bi] = fileSize; break; }
      const end = Math.min(pos + bufferSize, fileSize);
      while (j < eotOffsets.length && eotOffsets[j] < pos) j++;
      // first marker fully inside [pos, end)
      let hit = -1;
      for (let t = j; t < eotOffsets.length && eotOffsets[t] < end; t++) {
        if (eotOffsets[t] + markerLen <= end) { hit = eotOffsets[t]; break; }
      }
      if (hit >= 0) { snapped[bi] = hit; break; }
      pos += bufferSize;
    }
  }
  return { raw, boundaries: [...new Set(snapped)].sort((x, y) => x - y), snapped };
}

function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}

/** Delta-decoded offsets from web/data/ch04/valid_eot.json. */
export function decodeOffsets(deltas) {
  const out = new Array(deltas.length);
  let acc = 0;
  deltas.forEach((d, i) => { acc += d; out[i] = acc; });
  return out;
}

/** Amdahl: (Tp + Ts) / (Tp/n + Ts + To). */
export function amdahl(tPar, tSer, n, tOver = 0) {
  return (tPar + tSer) / (tPar / n + tSer + tOver);
}
