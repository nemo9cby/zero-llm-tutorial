/**
 * @module _gpt
 * @file The DLFS6 CodeBot GPT (third_party/dlfs6/codebot/model.py) in dependency-free JavaScript, for browsers,
 * Web Workers and Node. Shared library of the workbench page (content/playground.md); it has no mount().
 *
 * The model is the book's: learned position embedding, 6 pre-LN blocks (nn.LayerNorm eps 1e-5, causal multi-head
 * Attention without bias, FFN with the exact erf GELU of nn.GELU()), a final LayerNorm and an unembedding tied to the
 * token embedding (plus its own bias). Weights come from tools/export_codebot.py: int8 per output row + float32
 * row scales for 2-D weights, float32 for 1-D parameters. They are dequantized once, at load time, to float32.
 *
 *   import { fetchCheckpoint, CodeBot, generate } from './_gpt.js';
 *   const { manifest, buffer } = await fetchCheckpoint('data/playground/pretrain/manifest.json', { onProgress });
 *   const bot = new CodeBot(manifest, buffer);               // backend 'wasm-simd' when available, else 'js'
 *   const { logits, attention, lens } = bot.forward(ids, { attention: true, lens: 5 });
 *   for (const tok of generate(bot, ids, { temperature: 0.8, topK: 40, topP: 0.95, seed: 1 })) { ... }
 *
 * ── Loading ─────────────────────────────────────────────────────────────────────────────────────────────
 * fetchCheckpoint(manifestURL, { onProgress(loaded, total), verify, fetch }) -> { manifest, buffer, url, verified }
 *     downloads the shards in parallel, concatenates them, checks their sizes (and SHA-256 when verify is set and
 *     crypto.subtle exists)
 * unpackTensors(manifest, buffer) -> Map(name -> Float32Array)   the dequantized float32 weights
 * sha256Hex(buffer) -> Promise<string | null>                    null where crypto.subtle is unavailable
 *
 * ── The model ───────────────────────────────────────────────────────────────────────────────────────────
 * new CodeBot(manifest, buffer, { backend: 'auto' | 'wasm-simd' | 'js' })
 *   .cfg { V, T, E, H, D, L, FF }   .backend   .pos (tokens in the KV cache)   .bytes (memory held)
 *   .reset()                         empty the KV cache
 *   .step(id) -> logits              one token at position .pos (decode); the returned view is overwritten later
 *   .prefill(ids, capture?) -> logits  several tokens at positions .pos, .pos + 1, ... in one pass (matrix-matrix
 *                                    products); bit-identical to calling step() on each of them
 *   .forward(ids, { attention, lens }) -> { logits, attention, n, lens }   reset + prefill, with captures:
 *       attention  Float32Array [L][H][n][n], softmax weights (zero above the diagonal)
 *       lens       [{ layer, ids, probs }]: the logit lens, i.e. the final LayerNorm + unembedding applied to the
 *                  residual stream after the embedding (layer 0) and after each block (1..L), last position only
 *
 * ── Decoding ────────────────────────────────────────────────────────────────────────────────────────────
 * softmax(logits, temperature = 1) -> Float64Array
 * filterDistribution(logits, { temperature, topK, topP }) -> { probs, kept }   temperature -> softmax -> top-k ->
 *     renormalize -> top-p (smallest set whose cumulative probability reaches topP) -> renormalize; the order of
 *     content/ch03.md 3.2.3. temperature 0 is greedy (argmax), as in codebot/utils.generate.
 * mulberry32(seed) -> () => number in [0, 1)                    the seeded generator behind `seed`
 * generate(bot, promptIds, opts) -> iterator of { id, prob, top, step, end, refill, window, computed }
 *     the loop of codebot/utils.generate: before every forward pass the sequence is cropped to its last T tokens.
 *     A full cache cannot slide. Cropping the oldest token shifts every remaining position (CodeBot's position
 *     embedding is a learned table), and from layer 2 up every cached K and V was computed after attending to the
 *     dropped token, which holds under any position scheme. So once the sequence outgrows the window, every new
 *     token re-runs the last T tokens from scratch (refill: true), exactly what the book's code computes.
 * compareToReference(bot, ref) -> per prompt: max |Δlogit| against PyTorch (q8 and f32 weights), greedy agreement,
 *     attention and logit-lens agreement. ref is web/data/playground/<name>/ref.json.
 */

// ─── WebAssembly SIMD kernels ───────────────────────────────────────────────────────────────────────────────
// Source: web/data/playground/kernels.wat, compiled with wabt (tests/js/fixtures/playground/compile_kernels.cjs).
//   matvec(W, x, y, nIn, nOut)        y[o] = Σ_i W[o·nIn + i]·x[i]                  nIn % 4 == 0
//   matmul(X, W, Y, n, nIn, nOut)     Y[t·nOut + o] = Σ_i X[t·nIn + i]·W[o·nIn + i]   n % 4 == 0, nOut % 2 == 0
// Arguments are byte offsets into the module's memory. Each output keeps one f32x4 accumulator that walks i in
// steps of 4 and is summed as (l0 + l1) + (l2 + l3) at the end, in both kernels: so prefill (matmul) and decode
// (matvec) produce bit-identical numbers.
const KERNELS_WASM = 'AGFzbQEAAAABFwNgAXsBfWAFf39/f38AYAZ/f39/f38AAwQDAAECBQMBAAEHGQMDbWVtAgAGbWF0dmVjAAEGbWF0bXVsAAIKpQgDGQAgAP0fACAA'
  + '/R8BkiAA/R8CIAD9HwOSkgu5AwIJfwV7IANBAnQhCCAEQXxxIQYCQANAIAUgBk8NASAAIAUgCGxqIQkgCSAIaiEKIAogCGohCyALIAhqIQz9DAAA'
  + 'AAAAAAAAAAAAAAAAAAAhD/0MAAAAAAAAAAAAAAAAAAAAACEQ/QwAAAAAAAAAAAAAAAAAAAAAIRH9DAAAAAAAAAAAAAAAAAAAAAAhEkEAIQcCQANA'
  + 'IAcgCE8NASABIAdq/QAEACEOIA8gDiAJIAdq/QAEAP3mAf3kASEPIBAgDiAKIAdq/QAEAP3mAf3kASEQIBEgDiALIAdq/QAEAP3mAf3kASERIBIg'
  + 'DiAMIAdq/QAEAP3mAf3kASESIAdBEGohBwwACwsgAiAFQQJ0aiENIA0gDxAAOAIAIA0gEBAAOAIEIA0gERAAOAIIIA0gEhAAOAIMIAVBBGohBQwA'
  + 'CwsCQANAIAUgBE8NASAAIAUgCGxqIQn9DAAAAAAAAAAAAAAAAAAAAAAhD0EAIQcCQANAIAcgCE8NASAPIAEgB2r9AAQAIAkgB2r9AAQA/eYB/eQB'
  + 'IQ8gB0EQaiEHDAALCyACIAVBAnRqIA8QADgCACAFQQFqIQUMAAsLC80EAgx/C3sgBEECdCEJIAVBAnQhCgJAA0AgBiADTw0BIAAgBiAJbGohCyAL'
  + 'IAlqIQwgDCAJaiENIA0gCWohDkEAIQcCQANAIAcgBU8NASABIAcgCWxqIQ8gDyAJaiEQ/QwAAAAAAAAAAAAAAAAAAAAAIRX9DAAAAAAAAAAAAAAA'
  + 'AAAAAAAhFv0MAAAAAAAAAAAAAAAAAAAAACEX/QwAAAAAAAAAAAAAAAAAAAAAIRj9DAAAAAAAAAAAAAAAAAAAAAAhGf0MAAAAAAAAAAAAAAAAAAAA'
  + 'ACEa/QwAAAAAAAAAAAAAAAAAAAAAIRv9DAAAAAAAAAAAAAAAAAAAAAAhHEEAIQgCQANAIAggCU8NASAPIAhq/QAEACESIBAgCGr9AAQAIRMgCyAI'
  + 'av0ABAAhFCAVIBQgEv3mAf3kASEVIBYgFCAT/eYB/eQBIRYgDCAIav0ABAAhFCAXIBQgEv3mAf3kASEXIBggFCAT/eYB/eQBIRggDSAIav0ABAAh'
  + 'FCAZIBQgEv3mAf3kASEZIBogFCAT/eYB/eQBIRogDiAIav0ABAAhFCAbIBQgEv3mAf3kASEbIBwgFCAT/eYB/eQBIRwgCEEQaiEIDAALCyACIAYg'
  + 'CmwgB0ECdGpqIREgESAVEAA4AgAgESAWEAA4AgQgESAKaiERIBEgFxAAOAIAIBEgGBAAOAIEIBEgCmohESARIBkQADgCACARIBoQADgCBCARIApq'
  + 'IREgESAbEAA4AgAgESAcEAA4AgQgB0ECaiEHDAALCyAGQQRqIQYMAAsLCw==';

let kernelModule = null;
/** The compiled kernel module, or null where WebAssembly SIMD is unavailable (old engine, or a CSP without wasm). */
function kernels() {
  if (kernelModule === null) {
    try {
      const bytes = Uint8Array.from(atob(KERNELS_WASM), (ch) => ch.charCodeAt(0));
      kernelModule = WebAssembly.validate(bytes) ? new WebAssembly.Module(bytes) : false; // 1.1 KB: sync compile is fine
    } catch (err) {
      kernelModule = false;
    }
  }
  return kernelModule || null;
}
/** True when this engine can run the SIMD kernels. */
export const simdSupported = () => kernels() !== null;

// ─── Loading ────────────────────────────────────────────────────────────────────────────────────────────────
export const MANIFEST_FORMAT = 'dlfs6-codebot/1';

async function readWithProgress(resp, onChunk) {
  if (!resp.body || typeof resp.body.getReader !== 'function') {
    const buf = new Uint8Array(await resp.arrayBuffer());
    onChunk(buf.length);
    return buf;
  }
  const reader = resp.body.getReader();
  const parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    n += value.length;
    onChunk(value.length);
  }
  const out = new Uint8Array(n);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

/** SHA-256 of an ArrayBuffer as hex, or null where crypto.subtle is unavailable (e.g. plain-http pages). */
export async function sha256Hex(buffer) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = new Uint8Array(await subtle.digest('SHA-256', buffer));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Download a checkpoint exported by tools/export_codebot.py.
 * @param {string} manifestURL absolute, or relative to the document (in a worker: to the worker script)
 * @param {{onProgress?: (loaded: number, total: number) => void, verify?: boolean, fetch?: typeof fetch}} [opts]
 */
export async function fetchCheckpoint(manifestURL, { onProgress, verify = false, fetch: get = globalThis.fetch } = {}) {
  const base = new URL(manifestURL, globalThis.location?.href ?? undefined);
  const r = await get(base.href);
  if (!r.ok) throw new Error(`读取 ${base.pathname} 失败（HTTP ${r.status}）`);
  const manifest = await r.json();
  if (manifest.format !== MANIFEST_FORMAT) throw new Error(`不认识的权重格式 ${manifest.format}`);
  const total = manifest.total_bytes;
  const out = new Uint8Array(total);
  let loaded = 0;
  onProgress?.(0, total);
  const starts = [];
  manifest.shards.reduce((off, s) => { starts.push(off); return off + s.bytes; }, 0);
  const shardBytes = await Promise.all(manifest.shards.map(async (s, i) => {
    const resp = await get(new URL(s.file, base).href);
    if (!resp.ok) throw new Error(`读取 ${s.file} 失败（HTTP ${resp.status}）`);
    const bytes = await readWithProgress(resp, (n) => { loaded += n; onProgress?.(Math.min(loaded, total), total); });
    if (bytes.length !== s.bytes) throw new Error(`${s.file} 应有 ${s.bytes} 字节，实际收到 ${bytes.length} 字节`);
    out.set(bytes, starts[i]);
    return bytes;
  }));
  let verified = null;
  if (verify) {
    const hashes = await Promise.all(shardBytes.map((b) => sha256Hex(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))));
    if (hashes.every((x) => x !== null)) {
      const bad = manifest.shards.filter((s, i) => s.sha256 !== hashes[i]).map((s) => s.file);
      if (bad.length) throw new Error(`权重文件校验失败：${bad.join('、')} 的 SHA-256 与清单不符`);
      verified = true;
    }
  }
  return { manifest, buffer: out.buffer, url: base.href, verified };
}

function configOf(manifest) {
  const c = manifest.config;
  const cfg = { V: c.vocab_size, T: c.max_context_len, E: c.embed_dim, H: c.n_head, L: c.n_layer, FF: c.ff_dim };
  cfg.D = cfg.E / cfg.H;
  if (!Number.isInteger(cfg.D) || cfg.E % 4 || cfg.FF % 4) throw new Error(`不支持的模型形状 ${JSON.stringify(c)}`);
  return cfg;
}

function expectedShapes({ V, T, E, L, FF }) {
  const s = new Map([['embed.weight', [V, E]], ['pos_embed.weight', [T, E]]]);
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    s.set(p + 'norm1.weight', [E]); s.set(p + 'norm1.bias', [E]);
    for (const m of 'qkvo') s.set(p + `attn.W_${m}.weight`, [E, E]);
    s.set(p + 'norm2.weight', [E]); s.set(p + 'norm2.bias', [E]);
    s.set(p + 'ffn.layers.0.weight', [FF, E]); s.set(p + 'ffn.layers.0.bias', [FF]);
    s.set(p + 'ffn.layers.2.weight', [E, FF]); s.set(p + 'ffn.layers.2.bias', [E]);
  }
  s.set('norm.weight', [E]); s.set('norm.bias', [E]); s.set('unembed.bias', [V]);
  return s;
}

/** Dequantize every tensor of an export into float32: Map(name -> Float32Array). */
export function unpackTensors(manifest, buffer) {
  const cfg = configOf(manifest);
  const want = expectedShapes(cfg);
  const out = new Map();
  for (const t of manifest.tensors) {
    const shape = want.get(t.name);
    if (!shape || shape.join() !== t.shape.join()) throw new Error(`权重 ${t.name} 的形状 ${t.shape} 不符合模型配置`);
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (t.offset < 0 || t.offset + t.nbytes > buffer.byteLength) throw new Error(`权重 ${t.name} 超出了文件范围`);
    let data;
    if (t.dtype === 'f32') {
      if (t.offset % 4) throw new Error(`权重 ${t.name} 没有按 4 字节对齐`);
      data = new Float32Array(buffer.slice(t.offset, t.offset + n * 4));
    } else if (t.dtype === 'q8') {
      const [rows, cols] = t.shape;
      const q = new Int8Array(buffer, t.offset, n);
      const scales = new Float32Array(buffer.slice(t.scales_offset, t.scales_offset + rows * 4));
      data = new Float32Array(n);
      for (let r = 0; r < rows; r++) {
        const s = scales[r], base = r * cols;
        for (let c = 0; c < cols; c++) data[base + c] = q[base + c] * s;   // one float32 rounding, as in NumPy
      }
    } else {
      throw new Error(`权重 ${t.name} 的数据类型 ${t.dtype} 无法识别`);
    }
    out.set(t.name, data);
  }
  for (const name of want.keys()) if (!out.has(name)) throw new Error(`导出文件缺少权重 ${name}`);
  return out;
}

// ─── Small numeric helpers ──────────────────────────────────────────────────────────────────────────────────
/** erf by Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7, below float32 resolution of GELU's output). */
export function erf(x) {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
/** nn.GELU() default: x·Φ(x) with the exact (erf) Φ. The book's own tanh GELU class is not used by CodeBot. */
export const gelu = (x) => 0.5 * x * (1 + erf(x * Math.SQRT1_2));

/** Softmax at a temperature (> 0), in float64. */
export function softmax(logits, temperature = 1) {
  const n = logits.length, p = new Float64Array(n);
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > m) m = logits[i];
  let z = 0;
  for (let i = 0; i < n; i++) { p[i] = Math.exp((logits[i] - m) / temperature); z += p[i]; }
  for (let i = 0; i < n; i++) p[i] /= z;
  return p;
}

/** First index of the maximum (torch.argmax returns the first one too). */
export function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

/** Indices sorted by value, descending; ties by index, ascending. */
export function argsortDesc(a) {
  return Array.from(a.keys()).sort((i, j) => a[j] - a[i] || i - j);
}

/** Entropy in nats. */
export function entropy(p) {
  let h = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log(p[i]);
  return h;
}

/**
 * The distribution a token is drawn from: temperature, then top-k, then top-p (content/ch03.md 3.2.3).
 * @returns {{probs: Float64Array, kept: number[]}} kept = ids with non-zero probability, most probable first
 */
export function filterDistribution(logits, { temperature = 1, topK = 0, topP = 1 } = {}) {
  if (!(temperature >= 0)) throw new RangeError(`temperature must be >= 0, got ${temperature}`);
  const n = logits.length;
  if (temperature === 0) {                       // greedy, like `if temperature == 0: argmax` in the book
    const probs = new Float64Array(n), best = argmax(logits);
    probs[best] = 1;
    return { probs, kept: [best] };
  }
  const probs = softmax(logits, temperature);
  if (!probs.every(Number.isFinite)) throw new RangeError(`temperature ${temperature} 太小，softmax 溢出了`);
  let kept = argsortDesc(probs);
  if (topK > 0 && topK < n) kept = kept.slice(0, topK);
  let total = 0;
  for (const i of kept) total += probs[i];
  if (topP < 1) {
    let cum = 0, cut = kept.length;
    for (let r = 0; r < kept.length; r++) {
      cum += probs[kept[r]] / total;
      if (cum >= topP) { cut = r + 1; break; }
    }
    kept = kept.slice(0, cut);
    total = 0;
    for (const i of kept) total += probs[i];
  }
  const out = new Float64Array(n);
  for (const i of kept) out[i] = probs[i] / total;
  return { probs: out, kept };
}

/** Draw from a filtered distribution with a uniform number u in [0, 1). */
export function sampleFrom({ probs, kept }, u) {
  let r = u;
  for (const i of kept) {
    r -= probs[i];
    if (r < 0) return i;
  }
  return kept[kept.length - 1];                  // u ≈ 1 and rounding: the least likely kept token
}

/** A small, fast, seeded generator (Tommy Ettinger's Mulberry32). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ─── The model ──────────────────────────────────────────────────────────────────────────────────────────────
const align4 = (n) => Math.ceil(n / 4) * 4;

export class CodeBot {
  /**
   * @param {object} manifest manifest.json of an export
   * @param {ArrayBuffer} buffer the concatenated shards
   * @param {{backend?: 'auto' | 'wasm-simd' | 'js'}} [opts]
   */
  constructor(manifest, buffer, { backend = 'auto' } = {}) {
    const cfg = configOf(manifest);
    const { V, T, E, L, FF } = cfg;
    this.cfg = cfg;
    this.name = manifest.name;
    this.endId = manifest.end_token_id ?? V - 1;
    const tensors = unpackTensors(manifest, buffer);
    const T4 = align4(T);
    // Float offsets of every region in one heap (WebAssembly memory or a plain Float32Array).
    let top = 0;
    const alloc = (n) => { const off = top; top += align4(n); return off; };
    this.w = {};
    for (const [name, data] of tensors) this.w[name] = alloc(data.length);
    this.kv = Array.from({ length: L }, () => ({ K: alloc(T * E), V: alloc(T * E) }));
    const b = {};
    for (const [k, n] of [['x', E], ['h', E], ['q', E], ['a', E], ['o', E], ['f', FF], ['logits', V], ['lens', V],
      ['X', T4 * E], ['HN', T4 * E], ['Q', T4 * E], ['A', T4 * E], ['O', T4 * E], ['F', T4 * FF]]) b[k] = alloc(n);
    this.b = b;
    const floats = top;
    const mod = backend === 'js' ? null : kernels();
    if (backend === 'wasm-simd' && !mod) throw new Error('此环境不支持 WebAssembly SIMD');
    if (mod) {
      const inst = new WebAssembly.Instance(mod);
      const mem = inst.exports.mem;
      const need = Math.ceil((floats * 4) / 65536) - mem.buffer.byteLength / 65536;
      if (need > 0) mem.grow(need);
      this.heap = new Float32Array(mem.buffer, 0, floats);
      this.kmv = inst.exports.matvec;
      this.kmm = inst.exports.matmul;
      this.backend = 'wasm-simd';
    } else {
      this.heap = new Float32Array(floats);
      this.backend = 'js';
    }
    for (const [name, data] of tensors) this.heap.set(data, this.w[name]);
    this.bytes = floats * 4;
    this.att = new Float64Array(T);
    this.pos = 0;
  }

  /** Empty the KV cache. */
  reset() { this.pos = 0; }

  // y[o] = W[o,:]·x  (float offsets into the heap)
  matvec(W, x, y, nIn, nOut) {
    if (this.kmv) { this.kmv(W * 4, x * 4, y * 4, nIn, nOut); return; }
    const m = this.heap;
    for (let o = 0; o < nOut; o++) m[y + o] = dotJS(m, W + o * nIn, x, nIn);
  }

  // Y[t,:] = W·X[t,:] for t < n4 (n4 % 4 == 0; the padding rows are scratch)
  matmul(X, W, Y, n4, nIn, nOut) {
    if (this.kmm) { this.kmm(X * 4, W * 4, Y * 4, n4, nIn, nOut); return; }
    const m = this.heap;
    for (let t = 0; t < n4; t++) {
      const xr = X + t * nIn, yr = Y + t * nOut;
      for (let o = 0; o < nOut; o++) m[yr + o] = dotJS(m, W + o * nIn, xr, nIn);
    }
  }

  layerNorm(dst, src, g, b) {
    const m = this.heap, n = this.cfg.E;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += m[src + i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = m[src + i] - mean; v += d * d; }
    const r = 1 / Math.sqrt(v / n + 1e-5);
    for (let i = 0; i < n; i++) m[dst + i] = (m[src + i] - mean) * r * m[g + i] + m[b + i];
  }

  // Attention of one query row (heap[q..q+E)) at absolute position t over cache rows 0..t of layer l -> heap[out..].
  // cap (optional): {attn, n} receives the softmax weights at [l][h][t][j].
  attendRow(l, q, t, out, cap) {
    const m = this.heap, { E, H, D } = this.cfg, att = this.att, scale = 1 / Math.sqrt(D);
    const K = this.kv[l].K, Vc = this.kv[l].V;
    for (let hd = 0; hd < H; hd++) {
      const off = hd * D;
      let mx = -Infinity;
      for (let j = 0; j <= t; j++) {
        let s = 0;
        const kr = K + j * E + off;
        for (let d = 0; d < D; d++) s += m[q + off + d] * m[kr + d];
        s *= scale;
        att[j] = s;
        if (s > mx) mx = s;
      }
      let z = 0;
      for (let j = 0; j <= t; j++) { att[j] = Math.exp(att[j] - mx); z += att[j]; }
      for (let d = 0; d < D; d++) m[out + off + d] = 0;
      const base = cap ? ((l * H + hd) * cap.n + t) * cap.n : 0;
      for (let j = 0; j <= t; j++) {
        const w = att[j] / z;
        if (cap) cap.attn[base + j] = w;
        const vr = Vc + j * E + off;
        for (let d = 0; d < D; d++) m[out + off + d] += w * m[vr + d];
      }
    }
  }

  // Final LayerNorm + tied unembedding of heap[src..src+E) into heap[dst..dst+V).
  unembed(src, dst) {
    const { E, V } = this.cfg, m = this.heap, w = this.w, h = this.b.h;
    this.layerNorm(h, src, w['norm.weight'], w['norm.bias']);
    this.matvec(w['embed.weight'], h, dst, E, V);
    const ub = w['unembed.bias'];
    for (let i = 0; i < V; i++) m[dst + i] += m[ub + i];
    return this.heap.subarray(dst, dst + V);
  }

  /** Feed one token at position this.pos; returns the next-token logits (a view, overwritten by the next call). */
  step(id) {
    const { E, L, FF, V, T } = this.cfg, m = this.heap, w = this.w, b = this.b;
    const t = this.pos;
    if (t >= T) throw new RangeError(`KV cache 已满（${T} 个位置）：位置 embedding 只有 ${T} 行`);
    if (!Number.isInteger(id) || id < 0 || id >= V) throw new RangeError(`token id ${id} 不在 0..${V - 1} 之内`);
    const wte = w['embed.weight'], wpe = w['pos_embed.weight'];
    for (let i = 0; i < E; i++) m[b.x + i] = m[wte + id * E + i] + m[wpe + t * E + i];
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`, { K, V: Vc } = this.kv[l];
      this.layerNorm(b.h, b.x, w[p + 'norm1.weight'], w[p + 'norm1.bias']);
      this.matvec(w[p + 'attn.W_q.weight'], b.h, b.q, E, E);
      this.matvec(w[p + 'attn.W_k.weight'], b.h, K + t * E, E, E);
      this.matvec(w[p + 'attn.W_v.weight'], b.h, Vc + t * E, E, E);
      this.attendRow(l, b.q, t, b.a, null);
      this.matvec(w[p + 'attn.W_o.weight'], b.a, b.o, E, E);
      for (let i = 0; i < E; i++) m[b.x + i] += m[b.o + i];
      this.layerNorm(b.h, b.x, w[p + 'norm2.weight'], w[p + 'norm2.bias']);
      this.matvec(w[p + 'ffn.layers.0.weight'], b.h, b.f, E, FF);
      const b1 = w[p + 'ffn.layers.0.bias'];
      for (let i = 0; i < FF; i++) m[b.f + i] = gelu(m[b.f + i] + m[b1 + i]);
      this.matvec(w[p + 'ffn.layers.2.weight'], b.f, b.o, FF, E);
      const b2 = w[p + 'ffn.layers.2.bias'];
      for (let i = 0; i < E; i++) m[b.x + i] += m[b.o + i] + m[b2 + i];
    }
    this.pos = t + 1;
    return this.unembed(b.x, b.logits);
  }

  /**
   * Feed several tokens at positions pos, pos+1, ... in one pass; returns the logits after the last one.
   * capture (optional, only from an empty cache): {attn: Float32Array(L*H*n*n), n, resid: Float32Array((L+1)*E)}.
   */
  prefill(ids, capture = null) {
    const { E, L, FF, T } = this.cfg, m = this.heap, w = this.w, b = this.b;
    const n = ids.length, p0 = this.pos;
    if (n === 0) throw new RangeError('prefill 需要至少一个 token');
    if (p0 + n > T) throw new RangeError(`${p0} + ${n} 个 token 超出了 ${T} 个位置`);
    if (capture && p0 !== 0) throw new Error('只有从空的 KV cache 开始时才能记录注意力');
    const n4 = align4(n), last = n - 1;
    const wte = w['embed.weight'], wpe = w['pos_embed.weight'];
    for (let t = 0; t < n; t++) {
      const id = ids[t];
      if (!Number.isInteger(id) || id < 0 || id >= this.cfg.V) throw new RangeError(`token id ${id} 不在 0..${this.cfg.V - 1} 之内`);
      for (let i = 0; i < E; i++) m[b.X + t * E + i] = m[wte + id * E + i] + m[wpe + (p0 + t) * E + i];
    }
    if (capture?.resid) capture.resid.set(m.subarray(b.X + last * E, b.X + n * E), 0);
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`, { K, V: Vc } = this.kv[l];
      for (let t = 0; t < n; t++) this.layerNorm(b.HN + t * E, b.X + t * E, w[p + 'norm1.weight'], w[p + 'norm1.bias']);
      this.matmul(b.HN, w[p + 'attn.W_q.weight'], b.Q, n4, E, E);
      this.matmul(b.HN, w[p + 'attn.W_k.weight'], b.O, n4, E, E);       // O is free until the output projection
      m.copyWithin(K + p0 * E, b.O, b.O + n * E);
      this.matmul(b.HN, w[p + 'attn.W_v.weight'], b.O, n4, E, E);
      m.copyWithin(Vc + p0 * E, b.O, b.O + n * E);
      for (let t = 0; t < n; t++) this.attendRow(l, b.Q + t * E, p0 + t, b.A + t * E, capture?.attn ? capture : null);
      this.matmul(b.A, w[p + 'attn.W_o.weight'], b.O, n4, E, E);
      for (let i = 0; i < n * E; i++) m[b.X + i] += m[b.O + i];
      for (let t = 0; t < n; t++) this.layerNorm(b.HN + t * E, b.X + t * E, w[p + 'norm2.weight'], w[p + 'norm2.bias']);
      this.matmul(b.HN, w[p + 'ffn.layers.0.weight'], b.F, n4, E, FF);
      const b1 = w[p + 'ffn.layers.0.bias'];
      for (let t = 0; t < n; t++) {
        const r = b.F + t * FF;
        for (let i = 0; i < FF; i++) m[r + i] = gelu(m[r + i] + m[b1 + i]);
      }
      this.matmul(b.F, w[p + 'ffn.layers.2.weight'], b.O, n4, FF, E);
      const b2 = w[p + 'ffn.layers.2.bias'];
      for (let t = 0; t < n; t++) {
        const r = t * E;
        for (let i = 0; i < E; i++) m[b.X + r + i] += m[b.O + r + i] + m[b2 + i];
      }
      if (capture?.resid) capture.resid.set(m.subarray(b.X + last * E, b.X + n * E), (l + 1) * E);
    }
    this.pos = p0 + n;
    return this.unembed(b.X + last * E, b.logits);
  }

  /**
   * Run ids from an empty cache. lens = k > 0 adds the top-k logit lens per layer (last position).
   * @returns {{logits: Float32Array, attention: Float32Array | null, n: number,
   *            resid: Float32Array | null, lens: Array | null}}
   */
  forward(ids, { attention = false, lens = 0 } = {}) {
    const { L, H, E } = this.cfg, n = ids.length;
    this.reset();
    const capture = (attention || lens) ? {
      attn: attention ? new Float32Array(L * H * n * n) : null, n,
      resid: lens ? new Float32Array((L + 1) * E) : null,
    } : null;
    const logits = Float32Array.from(this.prefill(ids, capture));
    const resid = capture?.resid ?? null;
    return { logits, attention: capture?.attn ?? null, n, resid, lens: lens ? this.logitLens(resid, lens) : null };
  }

  /** softmax(unembed(LN(residual of layer l))): what the model would predict if it stopped after block l. */
  lensDistribution(resid, l) {
    const { E } = this.cfg, b = this.b;
    this.heap.set(resid.subarray(l * E, (l + 1) * E), b.x);
    return softmax(this.unembed(b.x, b.lens));
  }

  /** Top-k of the logit lens for every captured layer (0 = embedding, l = after block l). */
  logitLens(resid, k = 5) {
    const out = [];
    for (let l = 0; l <= this.cfg.L; l++) {
      const p = this.lensDistribution(resid, l);
      const ids = argsortDesc(p).slice(0, k);
      out.push({ layer: l, ids, probs: ids.map((i) => p[i]) });
    }
    return out;
  }
}

// Plain-JavaScript dot product used by both matvec and matmul, so the two agree bit for bit in this backend too.
function dotJS(m, w, x, n) {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (let i = 0; i < n; i += 4) {
    s0 += m[w + i] * m[x + i]; s1 += m[w + i + 1] * m[x + i + 1];
    s2 += m[w + i + 2] * m[x + i + 2]; s3 += m[w + i + 3] * m[x + i + 3];
  }
  return (s0 + s1) + (s2 + s3);
}

// ─── Generation ─────────────────────────────────────────────────────────────────────────────────────────────
/**
 * The book's generation loop (codebot/utils.generate), one token per iteration.
 * @param {CodeBot} bot
 * @param {number[]} promptIds
 * @param {{temperature?: number, topK?: number, topP?: number, maxNewTokens?: number, seed?: number,
 *          rand?: () => number, stopAtEnd?: boolean, top?: number, cache?: boolean}} [opts]
 *   cache: false re-runs the whole window for every token, exactly like the book's generate (no KV cache);
 *          the tokens are the same, because prefill and step are bit-identical.
 * Yields {id, prob, top, step, end, refill, window, computed}:
 *   prob     probability of id in the distribution it was drawn from (after temperature/top-k/top-p)
 *   top      [[id, p], ...] the model's own top candidates at temperature 1 (for display)
 *   end      id is <|endoftext|> and stopAtEnd: the book stops here and does not append it
 *   refill   this token's forward pass re-ran the whole window (the sequence had outgrown T, or cache: false)
 *   window   tokens the forward pass saw (≤ T);  computed  positions computed so far (prompt included)
 */
export function* generate(bot, promptIds, opts = {}) {
  const { temperature = 1, topK = 0, topP = 1, maxNewTokens = 128, seed = 0, stopAtEnd = true, top = 5, cache = true } = opts;
  const rand = opts.rand ?? mulberry32(seed);
  const T = bot.cfg.T;
  if (!promptIds.length) throw new RangeError('prompt 至少要有一个 token');
  const seq = promptIds.slice();
  let window = seq.slice(-T);
  bot.reset();
  let logits = bot.prefill(window);
  let computed = window.length, refill = false;
  for (let step = 0; step < maxNewTokens; step++) {
    const dist = filterDistribution(logits, { temperature, topK, topP });
    const id = temperature === 0 ? dist.kept[0] : sampleFrom(dist, rand());
    const raw = softmax(logits);
    const best = top > 0 ? argsortDesc(raw).slice(0, top).map((i) => [i, raw[i]]) : [];
    const end = stopAtEnd && id === bot.endId;
    yield { id, prob: dist.probs[id], top: best, step, end, refill, window: window.length, computed };
    if (end) return;
    seq.push(id);
    if (step === maxNewTokens - 1) return;       // the last token needs no forward pass
    if (cache && seq.length <= T) {              // the window still starts at position 0: extend the cache
      window = seq;
      logits = bot.step(id);
      computed += 1;
      refill = false;
    } else {                                     // no cache, or the window slid (positions shifted, and the K/V of
      window = seq.slice(-T);                    // layers 2+ saw the dropped token): recompute the window
      bot.reset();
      logits = bot.prefill(window);
      computed += window.length;                 // ≤ T: the book's loop recomputes min(sequence, T) positions
      refill = true;
    }
  }
}

// ─── Parity with PyTorch ────────────────────────────────────────────────────────────────────────────────────
const maxAbsDiff = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i])); return d; };
const agreement = (a, b) => {
  const first = a.findIndex((x, i) => x !== b[i]);
  return { same: first === -1 ? a.length : first, total: b.length, firstMismatch: first };
};

/**
 * Re-run the prompts of ref.json (tools/export_codebot.py) and compare with PyTorch.
 * q8: PyTorch with exactly these dequantized weights (checks the implementation);
 * f32: PyTorch with the checkpoint's original weights (adds the int8 rounding).
 */
export function compareToReference(bot, ref, { onPrompt } = {}) {
  const results = [];
  for (const p of ref.prompts) {
    const t0 = Date.now();
    const fwd = bot.forward(p.ids, { attention: Boolean(p.q8.attention), lens: 5 });
    const greedy = [];
    for (const tok of generate(bot, p.ids, { temperature: 0, maxNewTokens: p.new_tokens, stopAtEnd: false, top: 0 })) greedy.push(tok.id);
    let lensTop1 = true, lensMaxDiff = 0;
    p.q8.lens.forEach((x, l) => {                 // compare at PyTorch's top-5 ids, so near-ties cannot reorder them
      const P = bot.lensDistribution(fwd.resid, l);
      if (argmax(P) !== x.ids[0]) lensTop1 = false;
      x.ids.forEach((id, r) => { lensMaxDiff = Math.max(lensMaxDiff, Math.abs(P[id] - x.probs[r])); });
    });
    const r = {
      name: p.name, n: p.ids.length, newTokens: p.new_tokens,
      maxDiffQ8: maxAbsDiff(fwd.logits, p.q8.logits), maxDiffF32: maxAbsDiff(fwd.logits, p.f32.logits),
      greedyQ8: agreement(greedy, p.q8.greedy), greedyF32: agreement(greedy, p.f32.greedy),
      lensTop1, lensMaxDiff, eosAt: p.f32.greedy.indexOf(ref.end_token_id), greedy, ms: Date.now() - t0,
    };
    if (p.q8.attention) r.attentionMaxDiff = maxAbsDiff(fwd.attention, p.q8.attention.flat(3));
    results.push(r);
    onPrompt?.(r);
  }
  return results;
}
