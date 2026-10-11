// CodeBot forward pass where every nn.Linear runs as a WASM SIMD (f32x4) matvec.
// All weights, the KV cache and the activation buffers live inside one WebAssembly.Memory,
// so JS only passes byte offsets. Same math as ../jsinfer/gpt.mjs (parity-checked against PyTorch).
export async function createSimdModel(manifest, buf, wasmBytes) {
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { matvec, mem } = instance.exports;
  const c = manifest.config;
  const cfg = { V: c.vocab_size, T: c.max_context_len, E: c.embed_dim, H: c.n_head, L: c.n_layer, FF: c.ff_dim };
  const floats = manifest.tensors.reduce((s, t) => s + t.shape.reduce((a, b) => a * b, 1), 0)
    + 2 * cfg.L * cfg.T * cfg.E + 6 * cfg.E + cfg.FF + cfg.V;
  const need = Math.ceil((floats * 4) / 65536) + 1 - mem.buffer.byteLength / 65536;
  if (need > 0) mem.grow(need);
  const f32 = new Float32Array(mem.buffer);
  let top = 0;
  const alloc = (n) => { const off = top; top += n; return { off: off * 4, a: f32.subarray(off, off + n) }; };
  const W = {};
  for (const t of manifest.tensors) {            // f32 manifest only in this prototype
    const n = t.shape.reduce((a, b) => a * b, 1), r = alloc(n);
    r.a.set(new Float32Array(buf, t.offset, n)); W[t.name] = r;
  }
  const kv = Array.from({ length: cfg.L }, () => ({ K: alloc(cfg.T * cfg.E), V: alloc(cfg.T * cfg.E) }));
  const B = { x: alloc(cfg.E), h: alloc(cfg.E), q: alloc(cfg.E), a: alloc(cfg.E), o: alloc(cfg.E), f: alloc(cfg.FF), logits: alloc(cfg.V) };
  return { cfg, W, kv, B, matvec, pos: 0, att: new Float32Array(cfg.T) };
}

function layerNorm(y, x, g, b, n) {
  let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
  let v = 0; for (let i = 0; i < n; i++) v += (x[i] - m) ** 2; v /= n;
  const r = 1 / Math.sqrt(v + 1e-5);
  for (let i = 0; i < n; i++) y[i] = (x[i] - m) * r * g[i] + b[i];
}
function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}

export function stepSimd(m, id) {
  const { cfg: { E, H, L, FF, V, T }, W, B, kv, matvec, att } = m;
  const t = m.pos, D = E / H, scale = 1 / Math.sqrt(D);
  if (t >= T) throw new Error('context full');
  const x = B.x.a, h = B.h.a, q = B.q.a, a = B.a.a, o = B.o.a, f = B.f.a;
  const wte = W['embed.weight'].a, wpe = W['pos_embed.weight'].a;
  for (let i = 0; i < E; i++) x[i] = wte[id * E + i] + wpe[t * E + i];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`, Kc = kv[l].K, Vc = kv[l].V;
    layerNorm(h, x, W[p + 'norm1.weight'].a, W[p + 'norm1.bias'].a, E);
    matvec(W[p + 'attn.W_q.weight'].off, B.h.off, B.q.off, E, E);
    matvec(W[p + 'attn.W_k.weight'].off, B.h.off, Kc.off + t * E * 4, E, E);
    matvec(W[p + 'attn.W_v.weight'].off, B.h.off, Vc.off + t * E * 4, E, E);
    const K = Kc.a, Vv = Vc.a;
    for (let hd = 0; hd < H; hd++) {
      const off = hd * D; let mx = -Infinity;
      for (let j = 0; j <= t; j++) { let s = 0; for (let d = 0; d < D; d++) s += q[off + d] * K[j * E + off + d]; att[j] = s * scale; if (att[j] > mx) mx = att[j]; }
      let z = 0; for (let j = 0; j <= t; j++) { att[j] = Math.exp(att[j] - mx); z += att[j]; }
      for (let d = 0; d < D; d++) a[off + d] = 0;
      for (let j = 0; j <= t; j++) { const w = att[j] / z; for (let d = 0; d < D; d++) a[off + d] += w * Vv[j * E + off + d]; }
    }
    matvec(W[p + 'attn.W_o.weight'].off, B.a.off, B.o.off, E, E);
    for (let i = 0; i < E; i++) x[i] += o[i];
    layerNorm(h, x, W[p + 'norm2.weight'].a, W[p + 'norm2.bias'].a, E);
    matvec(W[p + 'ffn.layers.0.weight'].off, B.h.off, B.f.off, E, FF);
    const b1 = W[p + 'ffn.layers.0.bias'].a;
    for (let i = 0; i < FF; i++) { const u = f[i] + b1[i]; f[i] = 0.5 * u * (1 + erf(u / Math.SQRT2)); }
    matvec(W[p + 'ffn.layers.2.weight'].off, B.f.off, B.o.off, FF, E);
    const b2 = W[p + 'ffn.layers.2.bias'].a;
    for (let i = 0; i < E; i++) x[i] += o[i] + b2[i];
  }
  layerNorm(h, x, W['norm.weight'].a, W['norm.bias'].a, E);
  matvec(W['embed.weight'].off, B.h.off, B.logits.off, E, V);
  const logits = B.logits.a, ub = W['unembed.bias'].a;
  for (let i = 0; i < V; i++) logits[i] += ub[i];
  m.pos++;
  return logits;
}
