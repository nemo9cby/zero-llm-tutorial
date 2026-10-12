// Unit tests of web/widgets/_gpt.js on a small synthetic CodeBot (no downloads, a few seconds).
//   node --test tests/js/playground-gpt.test.mjs
// The real checkpoints are compared with PyTorch in tests/js/playground-parity.test.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import {
  CodeBot, unpackTensors, erf, gelu, softmax, filterDistribution, sampleFrom, mulberry32, argmax, entropy, generate,
  simdSupported,
} from '../../web/widgets/_gpt.js';

// ── a synthetic checkpoint in the export format of tools/export_codebot.py ─────────────────────────────────────
function synthetic({ V = 24, T = 8, E = 16, H = 2, L = 2, FF = 32 } = {}, seed = 7) {
  const rand = mulberry32(seed);
  const normal = () => Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-12))) * Math.cos(2 * Math.PI * rand());
  const shapes = [['embed.weight', [V, E]], ['pos_embed.weight', [T, E]]];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    shapes.push([p + 'norm1.weight', [E]], [p + 'norm1.bias', [E]]);
    for (const m of 'qkvo') shapes.push([p + `attn.W_${m}.weight`, [E, E]]);
    shapes.push([p + 'norm2.weight', [E]], [p + 'norm2.bias', [E]], [p + 'ffn.layers.0.weight', [FF, E]],
      [p + 'ffn.layers.0.bias', [FF]], [p + 'ffn.layers.2.weight', [E, FF]], [p + 'ffn.layers.2.bias', [E]]);
  }
  shapes.push(['norm.weight', [E]], ['norm.bias', [E]], ['unembed.bias', [V]]);
  const parts = [], tensors = [];
  let off = 0;
  for (const [name, shape] of shapes) {
    const n = shape.reduce((a, b) => a * b, 1);
    const isGain = name.endsWith('norm1.weight') || name.endsWith('norm2.weight') || name === 'norm.weight';
    const w = Float32Array.from({ length: n }, () => (isGain ? 1 + 0.1 * normal() : 0.4 * normal()));
    let bytes;
    const entry = { name, shape, offset: off };
    if (shape.length === 2) {
      const [rows, cols] = shape;
      const q = new Int8Array(n), scales = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        let mx = 0;
        for (let c = 0; c < cols; c++) mx = Math.max(mx, Math.abs(w[r * cols + c]));
        scales[r] = Math.fround(mx / 127) || 1;
        for (let c = 0; c < cols; c++) q[r * cols + c] = Math.max(-127, Math.min(127, Math.round(w[r * cols + c] / scales[r])));
      }
      entry.dtype = 'q8';
      entry.scales_offset = off + n;
      bytes = new Uint8Array(n + rows * 4);
      bytes.set(new Uint8Array(q.buffer), 0);
      bytes.set(new Uint8Array(scales.buffer), n);
    } else {
      entry.dtype = 'f32';
      bytes = new Uint8Array(w.buffer);
    }
    entry.nbytes = bytes.length;
    const pad = (4 - (bytes.length % 4)) % 4;
    parts.push(bytes, new Uint8Array(pad));
    tensors.push(entry);
    off += bytes.length + pad;
  }
  const buffer = new Uint8Array(off);
  let o = 0;
  for (const p of parts) { buffer.set(p, o); o += p.length; }
  const manifest = {
    format: 'dlfs6-codebot/1', name: 'synthetic', end_token_id: V - 1, total_bytes: off, shards: [], tensors,
    config: { vocab_size: V, max_context_len: T, embed_dim: E, n_head: H, n_layer: L, ff_dim: FF },
  };
  return { manifest, buffer: buffer.buffer };
}

// A plain float64 forward pass written from codebot/model.py, recomputing everything (no cache, no kernels).
function referenceLogits(manifest, buffer, ids) {
  const w = unpackTensors(manifest, buffer);
  const { vocab_size: V, embed_dim: E, n_head: H, n_layer: L, ff_dim: FF } = manifest.config;
  const D = E / H, n = ids.length;
  const lin = (x, W, nIn, nOut, b) => Array.from({ length: nOut }, (_, o) => {
    let s = b ? b[o] : 0;
    for (let i = 0; i < nIn; i++) s += W[o * nIn + i] * x[i];
    return s;
  });
  const ln = (x, g, b) => {
    const mu = x.reduce((a, v) => a + v, 0) / E;
    const va = x.reduce((a, v) => a + (v - mu) ** 2, 0) / E;
    return x.map((v, i) => ((v - mu) / Math.sqrt(va + 1e-5)) * g[i] + b[i]);
  };
  let X = ids.map((id, t) => Array.from({ length: E }, (_, i) => w.get('embed.weight')[id * E + i] + w.get('pos_embed.weight')[t * E + i]));
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    const Hn = X.map((x) => ln(x, w.get(p + 'norm1.weight'), w.get(p + 'norm1.bias')));
    const Q = Hn.map((x) => lin(x, w.get(p + 'attn.W_q.weight'), E, E));
    const K = Hn.map((x) => lin(x, w.get(p + 'attn.W_k.weight'), E, E));
    const Vv = Hn.map((x) => lin(x, w.get(p + 'attn.W_v.weight'), E, E));
    const A = X.map((_, t) => {
      const out = new Array(E).fill(0);
      for (let hd = 0; hd < H; hd++) {
        const s = [];
        for (let j = 0; j <= t; j++) {
          let d = 0;
          for (let k = 0; k < D; k++) d += Q[t][hd * D + k] * K[j][hd * D + k];
          s.push(d / Math.sqrt(D));
        }
        const m = Math.max(...s), e = s.map((v) => Math.exp(v - m)), z = e.reduce((a, v) => a + v, 0);
        for (let j = 0; j <= t; j++) for (let k = 0; k < D; k++) out[hd * D + k] += (e[j] / z) * Vv[j][hd * D + k];
      }
      return out;
    });
    X = X.map((x, t) => {
      const o = lin(A[t], w.get(p + 'attn.W_o.weight'), E, E);
      const x1 = x.map((v, i) => v + o[i]);
      const f = lin(ln(x1, w.get(p + 'norm2.weight'), w.get(p + 'norm2.bias')), w.get(p + 'ffn.layers.0.weight'), E, FF,
        w.get(p + 'ffn.layers.0.bias')).map((v) => 0.5 * v * (1 + erf(v / Math.SQRT2)));
      const g = lin(f, w.get(p + 'ffn.layers.2.weight'), FF, E, w.get(p + 'ffn.layers.2.bias'));
      return x1.map((v, i) => v + g[i]);
    });
  }
  const last = ln(X[n - 1], w.get('norm.weight'), w.get('norm.bias'));
  return lin(last, w.get('embed.weight'), E, V, w.get('unembed.bias'));
}

const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const backends = simdSupported() ? ['wasm-simd', 'js'] : ['js'];
const ckpt = synthetic();
const ids = [3, 1, 4, 1, 5, 9, 2];

test('Node runs the WebAssembly SIMD kernels (otherwise only the JS backend is tested)', () => {
  assert.equal(simdSupported(), true);
});

test('erf and the exact GELU of nn.GELU() agree with high-precision values (|error| < 2e-7)', () => {
  const cases = [[0, 0], [0.5, 0.5204998778130465], [1, 0.8427007929497149], [2, 0.9953222650189527], [-1.5, -0.9661051464753108]];
  for (const [x, y] of cases) assert.ok(Math.abs(erf(x) - y) < 2e-7, `erf(${x})`);
  for (const [x, y] of [[1, 0.8413447460685429], [-1, -0.15865525393145707], [0.5, 0.34573123063700656]]) {
    assert.ok(Math.abs(gelu(x) - y) < 2e-7, `gelu(${x})`);
  }
});

for (const backend of backends) {
  test(`${backend}: the forward pass equals a plain float64 implementation of codebot/model.py (< 1e-4)`, () => {
    const bot = new CodeBot(ckpt.manifest, ckpt.buffer, { backend });
    assert.equal(bot.backend, backend);
    const got = bot.forward(ids).logits;
    assert.ok(maxAbs(Array.from(got), referenceLogits(ckpt.manifest, ckpt.buffer, ids)) < 1e-4);
  });

  test(`${backend}: prefill, one step at a time, and a split prefill are bit-identical`, () => {
    const bot = new CodeBot(ckpt.manifest, ckpt.buffer, { backend });
    bot.reset();
    const a = Float32Array.from(bot.prefill(ids));
    bot.reset();
    let b;
    for (const id of ids) b = bot.step(id);
    b = Float32Array.from(b);
    bot.reset();
    bot.prefill(ids.slice(0, 3));
    const c = Float32Array.from(bot.prefill(ids.slice(3)));
    assert.deepEqual(b, a);
    assert.deepEqual(c, a);
  });

  test(`${backend}: attention rows sum to 1, nothing above the diagonal; lens of the last block = the output`, () => {
    const bot = new CodeBot(ckpt.manifest, ckpt.buffer, { backend });
    const { attention, n, resid, logits, lens } = bot.forward(ids, { attention: true, lens: 3 });
    const { L, H } = bot.cfg;
    for (let k = 0; k < L * H; k++) {
      for (let r = 0; r < n; r++) {
        let s = 0;
        for (let c = 0; c < n; c++) {
          const v = attention[(k * n + r) * n + c];
          if (c > r) assert.equal(v, 0);
          s += v;
        }
        assert.ok(Math.abs(s - 1) < 1e-6);
      }
    }
    assert.equal(lens.length, L + 1);
    const last = bot.lensDistribution(resid, L), out = softmax(logits);
    assert.ok(maxAbs(Array.from(last), Array.from(out)) < 1e-7);
    assert.equal(lens[L].ids[0], argmax(logits));
  });

  test(`${backend}: generate() crops to the last T tokens like codebot/utils.generate; cache on or off, same tokens`, () => {
    const bot = new CodeBot(ckpt.manifest, ckpt.buffer, { backend });
    const T = bot.cfg.T, prompt = ids.slice(0, 5), N = 9;
    const cached = [...generate(bot, prompt, { temperature: 0, maxNewTokens: N, stopAtEnd: false, top: 0 })];
    const plain = [...generate(bot, prompt, { temperature: 0, maxNewTokens: N, stopAtEnd: false, top: 0, cache: false })];
    assert.deepEqual(plain.map((t) => t.id), cached.map((t) => t.id));
    // brute force: recompute the cropped window from scratch for every token
    const seq = prompt.slice(), brute = [];
    for (let k = 0; k < N; k++) {
      const next = argmax(bot.forward(seq.slice(-T)).logits);
      brute.push(next);
      seq.push(next);
    }
    assert.deepEqual(cached.map((t) => t.id), brute);
    // the cache serves tokens until the sequence outgrows T, then every token re-runs the window
    const firstRefill = cached.findIndex((t) => t.refill);
    assert.equal(firstRefill, T - prompt.length + 1);
    assert.ok(cached.slice(firstRefill).every((t) => t.refill && t.window === T));
    assert.ok(plain.slice(1).every((t) => t.refill));
  });
}

test('generate(): computed counts the positions of every forward pass, as positions_computed in P.7 of the page', () => {
  const bot = new CodeBot(ckpt.manifest, ckpt.buffer);
  const T = bot.cfg.T;
  // The first token prefills the prompt; after that, 1 position per token while the cached sequence fits in T,
  // otherwise min(sequence, T): the book's loop recomputes the cropped window (content/playground.md, P.7.2).
  const expected = (nPrompt, nNew, cache) => {
    let total = Math.min(nPrompt, T);
    for (let seq = nPrompt + 1; seq < nPrompt + nNew; seq++) total += cache && seq <= T ? 1 : Math.min(seq, T);
    return total;
  };
  const run = (n, cache) => [...generate(bot, ids.slice(0, 5), { temperature: 0, maxNewTokens: n, stopAtEnd: false, top: 0, cache })]
    .map((t) => t.computed);
  assert.deepEqual(run(3, false), [5, 11, 18]);    // without a cache: 5, then 6, then 7 positions (not T = 8 each)
  for (const cache of [true, false]) {
    const got = run(9, cache);
    assert.deepEqual(got, got.map((_, k) => expected(5, k + 1, cache)), `cache: ${cache}`);
  }
});

test('the two backends agree to float32 rounding (< 1e-5)', () => {
  const a = new CodeBot(ckpt.manifest, ckpt.buffer, { backend: 'wasm-simd' }).forward(ids).logits;
  const b = new CodeBot(ckpt.manifest, ckpt.buffer, { backend: 'js' }).forward(ids).logits;
  assert.ok(maxAbs(Array.from(a), Array.from(b)) < 1e-5);
});

test('the cache refuses position T: there is no position embedding for it', () => {
  const bot = new CodeBot(ckpt.manifest, ckpt.buffer);
  bot.reset();
  bot.prefill(Array(bot.cfg.T).fill(1));
  assert.throws(() => bot.step(1), RangeError);
  bot.reset();
  assert.throws(() => bot.prefill(Array(bot.cfg.T + 1).fill(1)), RangeError);
  assert.throws(() => bot.step(bot.cfg.V), RangeError);
});

test('filterDistribution: the hand example of ch03 3.2-d, temperature 0 is greedy, top-k before top-p', () => {
  const logits = [2, 1, 0.5, 0, -1];
  const k2 = filterDistribution(logits, { topK: 2 });
  assert.ok(maxAbs(Array.from(k2.probs), [0.731, 0.269, 0, 0, 0]) < 1e-3);
  assert.equal(filterDistribution(logits, { topP: 0.9 }).kept.length, 4);
  assert.deepEqual(filterDistribution(logits, { temperature: 0 }).kept, [0]);
  assert.deepEqual(Array.from(filterDistribution(logits, { topK: 1 }).probs), [1, 0, 0, 0, 0]);
  // top-p measures the cumulative probability inside the top-k set: 0.731 >= 0.7 keeps 1 token, not 2
  assert.equal(filterDistribution(logits, { topK: 2, topP: 0.7 }).kept.length, 1);
  assert.equal(filterDistribution(logits, { topP: 0.7 }).kept.length, 2);
  // temperature flattens: entropy grows with T
  const H = (t) => entropy(filterDistribution(logits, { temperature: t }).probs);
  assert.ok(H(0.5) < H(1) && H(1) < H(2));
  assert.throws(() => filterDistribution(logits, { temperature: -1 }), RangeError);
});

test('sampleFrom with mulberry32 reproduces the distribution (20,000 draws, |freq - p| < 0.01)', () => {
  const dist = filterDistribution([2, 1, 0.5, 0, -1], { topP: 0.9 });
  const rand = mulberry32(42), counts = new Array(5).fill(0);
  for (let i = 0; i < 20000; i++) counts[sampleFrom(dist, rand())] += 1;
  counts.forEach((c, i) => assert.ok(Math.abs(c / 20000 - dist.probs[i]) < 0.01, `token ${i}`));
  assert.equal(counts[4], 0);                       // removed by top-p: never drawn
  assert.equal(sampleFrom(dist, 0), 0);
  assert.equal(sampleFrom(dist, 0.999999999), dist.kept[dist.kept.length - 1]);
  const a = mulberry32(5), b = mulberry32(5);
  for (let i = 0; i < 100; i++) assert.equal(a(), b());
});

test('the embedded WebAssembly equals kernels.wat compiled with wabt (skipped without wabt)', async (t) => {
  const require = createRequire(import.meta.url);
  const { compileKernels, embeddedBase64 } = require('./fixtures/playground/compile_kernels.cjs');
  const bytes = await compileKernels();
  if (!bytes) { t.skip('wabt not installed (npm install --no-save wabt@1.0.39, or WABT=<path>)'); return; }
  assert.equal(Buffer.from(bytes).toString('base64'), embeddedBase64());
});
