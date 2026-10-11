// Dependency-free forward pass for the DLFS6 CodeBot GPT (pre-LN, LayerNorm, exact GELU,
// learned positional embedding, tied embedding + unembed bias). Works in browsers, workers and Node.
const F16 = (() => {                       // fp16 bit pattern -> float32 lookup table (256 KB)
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export function loadModel(manifest, buf) {
  const W = {};
  for (const t of manifest.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === 'f32') W[t.name] = new Float32Array(buf, t.offset, n);
    else if (t.dtype === 'f16') {                        // LUT loop: ~70 ms for 11M values (Array.from with a callback: ~4 s)
      const u = new Uint16Array(buf, t.offset, n), out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = F16[u[i]];
      W[t.name] = out;
    }
    else if (t.dtype === 'q8') {                         // per-row symmetric int8 -> float32
      const q = new Int8Array(buf, t.offset, n), s = new Float32Array(buf, t.scales_offset, t.shape[0]);
      const cols = t.shape[1], out = new Float32Array(n);
      for (let r = 0; r < t.shape[0]; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = q[r * cols + c] * s[r];
      W[t.name] = out;
    } else throw new Error(`unknown dtype ${t.dtype} for ${t.name}`);
  }
  const c = manifest.config;
  return { cfg: { V: c.vocab_size, T: c.max_context_len, E: c.embed_dim, H: c.n_head, L: c.n_layer, FF: c.ff_dim }, W };
}

// y[o] = W[o,:] . x + b[o]   (PyTorch nn.Linear layout: W is (out, in), row-major)
function linear(y, W, x, b, nIn, nOut) {
  for (let o = 0; o < nOut; o++) {
    const r = o * nIn; let s0 = 0, s1 = 0, s2 = 0, s3 = 0, i = 0;
    for (; i + 3 < nIn; i += 4) { s0 += W[r + i] * x[i]; s1 += W[r + i + 1] * x[i + 1]; s2 += W[r + i + 2] * x[i + 2]; s3 += W[r + i + 3] * x[i + 3]; }
    for (; i < nIn; i++) s0 += W[r + i] * x[i];
    y[o] = s0 + s1 + s2 + s3 + (b ? b[o] : 0);
  }
}
function layerNorm(y, x, g, b, n) {
  let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
  let v = 0; for (let i = 0; i < n; i++) v += (x[i] - m) ** 2; v /= n;
  const r = 1 / Math.sqrt(v + 1e-5);
  for (let i = 0; i < n; i++) y[i] = (x[i] - m) * r * g[i] + b[i];
}
function erf(x) {                          // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));   // nn.GELU() default (exact, erf-based)

export function createSession(model) {
  const { E, T, L, FF, V } = model.cfg;
  return {
    model, pos: 0,
    K: Array.from({ length: L }, () => new Float32Array(T * E)), Vc: Array.from({ length: L }, () => new Float32Array(T * E)),
    x: new Float32Array(E), h: new Float32Array(E), q: new Float32Array(E), a: new Float32Array(E), o: new Float32Array(E),
    f: new Float32Array(FF), att: new Float32Array(T), logits: new Float32Array(V),
  };
}

// Feed one token at position sess.pos, append its K/V to the cache, return logits for the next token.
export function step(sess, id) {
  const { cfg: { E, H, L, FF, V, T }, W } = sess.model, { x, h, q, a, o, f, att } = sess;
  const t = sess.pos, D = E / H, scale = 1 / Math.sqrt(D);
  if (t >= T) throw new Error('context full: re-prefill the last T tokens (learned positions cannot slide)');
  const wte = W['embed.weight'], wpe = W['pos_embed.weight'];
  for (let i = 0; i < E; i++) x[i] = wte[id * E + i] + wpe[t * E + i];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`, Kc = sess.K[l], Vc = sess.Vc[l];
    layerNorm(h, x, W[p + 'norm1.weight'], W[p + 'norm1.bias'], E);
    linear(q, W[p + 'attn.W_q.weight'], h, null, E, E);
    linear(Kc.subarray(t * E, (t + 1) * E), W[p + 'attn.W_k.weight'], h, null, E, E);
    linear(Vc.subarray(t * E, (t + 1) * E), W[p + 'attn.W_v.weight'], h, null, E, E);
    for (let hd = 0; hd < H; hd++) {
      const off = hd * D; let mx = -Infinity;
      for (let j = 0; j <= t; j++) {                       // causal mask for free: the cache only holds positions <= t
        let s = 0; for (let d = 0; d < D; d++) s += q[off + d] * Kc[j * E + off + d];
        att[j] = s * scale; if (att[j] > mx) mx = att[j];
      }
      let z = 0; for (let j = 0; j <= t; j++) { att[j] = Math.exp(att[j] - mx); z += att[j]; }
      for (let d = 0; d < D; d++) a[off + d] = 0;
      for (let j = 0; j <= t; j++) { const w = att[j] / z; for (let d = 0; d < D; d++) a[off + d] += w * Vc[j * E + off + d]; }
    }
    linear(o, W[p + 'attn.W_o.weight'], a, null, E, E);
    for (let i = 0; i < E; i++) x[i] += o[i];
    layerNorm(h, x, W[p + 'norm2.weight'], W[p + 'norm2.bias'], E);
    linear(f, W[p + 'ffn.layers.0.weight'], h, W[p + 'ffn.layers.0.bias'], E, FF);
    for (let i = 0; i < FF; i++) f[i] = gelu(f[i]);
    linear(o, W[p + 'ffn.layers.2.weight'], f, W[p + 'ffn.layers.2.bias'], FF, E);
    for (let i = 0; i < E; i++) x[i] += o[i];
  }
  layerNorm(h, x, W['norm.weight'], W['norm.bias'], E);
  linear(sess.logits, wte, h, W['unembed.bias'], E, V);   // tied: unembed.weight is embed.weight
  sess.pos++;
  return sess.logits;
}

export function sample(logits, temperature = 1.0, rand = Math.random) {
  let best = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  if (temperature === 0) return best;
  let z = 0; const p = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - logits[best]) / temperature); z += p[i]; }
  let r = rand() * z; for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; }
  return p.length - 1;
}

// BPE tokenizer port of codebot/tokenizer.py (same regex pre-tokenizer, merges applied in training order).
export function makeTokenizer(tok) {
  const PAT = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const bytes = Array.from({ length: 256 }, (_, i) => [i]);
  for (const [a, b, n] of tok.merges) bytes[n] = bytes[a].concat(bytes[b]);
  bytes[tok.end_token_id] = [...new TextEncoder().encode(tok.end_token)];
  const enc = new TextEncoder(), dec = new TextDecoder();
  const encodePiece = (s) => {
    let ids = [...enc.encode(s)];
    for (const [a, b, n] of tok.merges) {
      if (ids.length < 2) break;
      const out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(n); i++; } else out.push(ids[i]);
      }
      ids = out;
    }
    return ids;
  };
  return {
    endId: tok.end_token_id,
    encode: (text) => text.split(/(<\|endoftext\|>)/).flatMap((part) =>
      part === tok.end_token ? [tok.end_token_id] : (part.match(PAT) || []).flatMap(encodePiece)),
    decode: (ids) => dec.decode(new Uint8Array(ids.flatMap((i) => bytes[i]))),
  };
}
