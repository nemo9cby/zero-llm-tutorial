/* _tinygpt.js: a dependency-free, character-level GPT that trains in a browser tab.

   The architecture and the initialization are those of DLFS6 codebot/model.py (GPT-2 style):
   learned position embeddings, pre-LN blocks, bias-free Q/K/V/O projections, an erf-GELU FFN with
   biases, a final LayerNorm, and an unembedding that shares its weight with the token embedding but
   keeps its own bias. Parameter names and shapes follow the PyTorch state_dict, so weights can be
   exchanged with the PyTorch reference (tests/js/fixtures/trainer, verify/_trainer_export.py).

   Every layer has a forward and a hand-written backward, in the style of DLFS1 chapter 5. The optimizer
   is AdamW with torch.optim.AdamW's update rule. Buffers are typed arrays: Float32Array for fast
   training, Float64Array for gradient checks (pass { dtype: Float64Array } to createModel).

   Layout: activations are row-major [N, width] with row n = b * T + t (batch b, position t).
   Backward functions overwrite the gradient of their inputs and accumulate (+=) into parameter
   gradients, like PyTorch's .grad. */

export const EOT = '<|endoftext|>';

// ------------------------------------------------------------------ tokenizer
/** Character-level tokenizer: the sorted characters of the training text, then <|endoftext|> last. */
export class CharTokenizer {
  constructor(text) {
    this.chars = [...new Set(text.split(EOT).join(''))].sort();
    this.endId = this.chars.length;
    this.vocabSize = this.chars.length + 1;
    this.index = new Map(this.chars.map((c, i) => [c, i]));
  }

  encode(text) {
    const ids = [];
    text.split(EOT).forEach((piece, k) => {
      if (k > 0) ids.push(this.endId);
      for (const ch of piece) {
        const id = this.index.get(ch);
        if (id === undefined) throw new Error(`字符 ${JSON.stringify(ch)} 不在词表里`);
        ids.push(id);
      }
    });
    return ids;
  }

  decode(ids) {
    return Array.from(ids, (i) => (i === this.endId ? EOT : this.chars[i])).join('');
  }
}

// ------------------------------------------------------------------ random numbers
/** Seeded generator (mulberry32): rng() is uniform in [0, 1); rng.normal() is N(0, 1); rng.int(n) is 0..n-1. */
export function makeRng(seed = 0) {
  let a = seed >>> 0;
  const rng = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  rng.normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0;
    while (u === 0) u = rng();
    const r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * rng();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
  rng.int = (n) => Math.floor(rng() * n);
  return rng;
}

// ------------------------------------------------------------------ erf (fdlibm s_erf.c, as in Go's math.Erf)
const F64 = new Float64Array(1), U32 = new Uint32Array(F64.buffer);
const LOW_WORD = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1 ? 0 : 1;

/** Error function, accurate to about one ulp (JavaScript has no Math.erf). Rational approximations of
    fdlibm on [0, 0.84375), [0.84375, 1.25), [1.25, 1/0.35), [1/0.35, 6); erf(x) = 1 beyond 6. */
export function erf(x) {
  if (Number.isNaN(x)) return NaN;
  const a = Math.abs(x);
  let r;
  if (a < 0.84375) {
    if (a < 3.725290298461914e-9) {
      r = a + 1.28379167095512586316e-01 * a;
    } else {
      const z = a * a;
      const p = 1.28379167095512558561e-01 + z * (-3.25042107247001499370e-01 + z * (-2.84817495755985104766e-02
        + z * (-5.77027029648944159157e-03 + z * -2.37630166566501626084e-05)));
      const q = 1 + z * (3.97917223959155352819e-01 + z * (6.50222499887672944485e-02 + z * (5.08130628187576562776e-03
        + z * (1.32494738004321644526e-04 + z * -3.96022827877536812320e-06))));
      r = a + a * (p / q);
    }
  } else if (a < 1.25) {
    const s = a - 1;
    const p = -2.36211856075265944077e-03 + s * (4.14856118683748331666e-01 + s * (-3.72207876035701323847e-01
      + s * (3.18346619901161753674e-01 + s * (-1.10894694282396677476e-01 + s * (3.54783043256182359371e-02
      + s * -2.16637559486879084300e-03)))));
    const q = 1 + s * (1.06420880400844228286e-01 + s * (5.40397917702171048937e-01 + s * (7.18286544141962662868e-02
      + s * (1.26171219808761642112e-01 + s * (1.36370839120290507362e-02 + s * 1.19844998467991074170e-02)))));
    r = 8.45062911510467529297e-01 + p / q;
  } else if (a >= 6) {
    r = 1;
  } else {
    const s = 1 / (a * a);
    let R, S;
    if (a < 1 / 0.35) {
      R = -9.86494403484714822705e-03 + s * (-6.93858572707181764372e-01 + s * (-1.05586262253232909814e+01
        + s * (-6.23753324503260060396e+01 + s * (-1.62396669462573470355e+02 + s * (-1.84605092906711035994e+02
        + s * (-8.12874355063065934246e+01 + s * -9.81432934416914548592e+00))))));
      S = 1 + s * (1.96512716674392571292e+01 + s * (1.37657754143519042600e+02 + s * (4.34565877475229228821e+02
        + s * (6.45387271733267880336e+02 + s * (4.29008140027567833386e+02 + s * (1.08635005541779435134e+02
        + s * (6.57024977031928170135e+00 + s * -6.04244152148580987438e-02)))))));
    } else {
      R = -9.86494292470009928597e-03 + s * (-7.99283237680523006574e-01 + s * (-1.77579549177547519889e+01
        + s * (-1.60636384855821916062e+02 + s * (-6.37566443368389627722e+02 + s * (-1.02509513161107724954e+03
        + s * -4.83519191608651397019e+02)))));
      S = 1 + s * (3.03380607434824582924e+01 + s * (3.25792512996573918826e+02 + s * (1.53672958608443695994e+03
        + s * (3.19985821950859553908e+03 + s * (2.55305040643316442583e+03 + s * (4.74528541206955367215e+02
        + s * -2.24409524465858183362e+01))))));
    }
    F64[0] = a; U32[LOW_WORD] = 0; // z = a with the low 32 bits cleared, so z*z is exact
    const z = F64[0];
    r = 1 - Math.exp(-z * z - 0.5625) * Math.exp((z - a) * (z + a) + R / S) / a;
  }
  return x < 0 ? -r : r;
}

// ------------------------------------------------------------------ layers: forward and backward
/** Token + position embedding: x[n] = tok[ids[n]] + pos[t]. */
export function embeddingForward(ids, tok, pos, x, B, T, E) {
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const n = b * T + t, xo = n * E, to = ids[n] * E, po = t * E;
      for (let e = 0; e < E; e++) x[xo + e] = tok[to + e] + pos[po + e];
    }
  }
}

/** Backward of the embedding lookup: scatter-add the rows of dx (repeated ids add up). */
export function embeddingBackward(ids, dx, dTok, dPos, B, T, E) {
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const n = b * T + t, xo = n * E, to = ids[n] * E, po = t * E;
      for (let e = 0; e < E; e++) {
        dTok[to + e] += dx[xo + e];
        dPos[po + e] += dx[xo + e];
      }
    }
  }
}

/** C = A B^T, i.e. C[m, n] = sum_k A[m, k] B[n, k], for A: [M, K], B: [N, K], C: [M, N].
    Every matrix product of the model goes through here. Each pass of the inner loop keeps a 4 x 4 block
    of C in registers, which in V8 runs about 3x faster than one dot product at a time. Float32 inputs
    whose sizes fit go to the WebAssembly SIMD version of the same loop instead (see below). */
export function matmulBT(A, Bm, C, M, K, N) {
  if (useSimd && A instanceof Float32Array && Bm instanceof Float32Array && C instanceof Float32Array
      && M % 4 === 0 && N % 2 === 0 && K % 4 === 0) {
    simdMatmulBT(A, Bm, C, M, K, N);
    return;
  }
  let m = 0;
  for (; m + 3 < M; m += 4) {
    const a0 = m * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K, o0 = m * N, o1 = o0 + N, o2 = o1 + N, o3 = o2 + N;
    let n = 0;
    for (; n + 3 < N; n += 4) {
      const b0 = n * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
      let c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, c12 = 0, c13 = 0;
      let c20 = 0, c21 = 0, c22 = 0, c23 = 0, c30 = 0, c31 = 0, c32 = 0, c33 = 0;
      for (let k = 0; k < K; k++) {
        const x0 = A[a0 + k], x1 = A[a1 + k], x2 = A[a2 + k], x3 = A[a3 + k];
        const u0 = Bm[b0 + k], u1 = Bm[b1 + k], u2 = Bm[b2 + k], u3 = Bm[b3 + k];
        c00 += x0 * u0; c01 += x0 * u1; c02 += x0 * u2; c03 += x0 * u3;
        c10 += x1 * u0; c11 += x1 * u1; c12 += x1 * u2; c13 += x1 * u3;
        c20 += x2 * u0; c21 += x2 * u1; c22 += x2 * u2; c23 += x2 * u3;
        c30 += x3 * u0; c31 += x3 * u1; c32 += x3 * u2; c33 += x3 * u3;
      }
      C[o0 + n] = c00; C[o0 + n + 1] = c01; C[o0 + n + 2] = c02; C[o0 + n + 3] = c03;
      C[o1 + n] = c10; C[o1 + n + 1] = c11; C[o1 + n + 2] = c12; C[o1 + n + 3] = c13;
      C[o2 + n] = c20; C[o2 + n + 1] = c21; C[o2 + n + 2] = c22; C[o2 + n + 3] = c23;
      C[o3 + n] = c30; C[o3 + n + 1] = c31; C[o3 + n + 2] = c32; C[o3 + n + 3] = c33;
    }
    for (; n < N; n++) {
      const b0 = n * K;
      let c0 = 0, c1 = 0, c2 = 0, c3 = 0;
      for (let k = 0; k < K; k++) {
        const u = Bm[b0 + k];
        c0 += A[a0 + k] * u; c1 += A[a1 + k] * u; c2 += A[a2 + k] * u; c3 += A[a3 + k] * u;
      }
      C[o0 + n] = c0; C[o1 + n] = c1; C[o2 + n] = c2; C[o3 + n] = c3;
    }
  }
  for (; m < M; m++) {
    const a0 = m * K;
    for (let n = 0; n < N; n++) {
      const b0 = n * K;
      let c = 0;
      for (let k = 0; k < K; k++) c += A[a0 + k] * Bm[b0 + k];
      C[m * N + n] = c;
    }
  }
}

// ------------------------------------------------------------------ WebAssembly SIMD (optional speed-up)
/* The matmulBT loop again, in WebAssembly with 128-bit SIMD: four float32 multiply-adds per
   instruction, a 4 x 2 block of C in eight vector registers. In V8 it runs 6-10x faster than the
   JavaScript loop, copies in and out included. Accumulation is in float32 (the JavaScript loop
   accumulates in float64), so results differ by about 1e-7 relative; tests/js/trainer-simd.test.mjs
   checks it against the JavaScript version. The bytes below were compiled with wabt 1.0.39
   (parseWat with SIMD enabled, then toBinary) from exactly this source:

  (module
    (memory (export "mem") 1)
    ;; C = A B^T in float32: C[m*N + n] = sum_k A[m*K + k] * B[n*K + k].
    ;; A, B, C are byte offsets into "mem". Needs M % 4 == 0, N % 2 == 0, K % 4 == 0.
    ;; Each inner step multiplies 4 rows of A by 2 rows of B, four k at a time (f32x4),
    ;; and keeps the 4 x 2 block of C in eight vector registers.
    (func (export "mm") (param $A i32) (param $B i32) (param $C i32) (param $M i32) (param $K i32) (param $N i32)
      (local $m i32) (local $n i32) (local $k i32) (local $rb i32) (local $cb i32)
      (local $a0 i32) (local $a1 i32) (local $a2 i32) (local $a3 i32) (local $b0 i32) (local $b1 i32) (local $c i32)
      (local $c00 v128) (local $c01 v128) (local $c10 v128) (local $c11 v128)
      (local $c20 v128) (local $c21 v128) (local $c30 v128) (local $c31 v128)
      (local $x v128) (local $u v128) (local $w v128)
      (local.set $rb (i32.shl (local.get $K) (i32.const 2)))
      (local.set $cb (i32.shl (local.get $N) (i32.const 2)))
      (block $mdone (loop $mloop
        (br_if $mdone (i32.ge_u (local.get $m) (local.get $M)))
        (local.set $a0 (i32.add (local.get $A) (i32.mul (local.get $m) (local.get $rb))))
        (local.set $a1 (i32.add (local.get $a0) (local.get $rb)))
        (local.set $a2 (i32.add (local.get $a1) (local.get $rb)))
        (local.set $a3 (i32.add (local.get $a2) (local.get $rb)))
        (local.set $n (i32.const 0))
        (block $ndone (loop $nloop
          (br_if $ndone (i32.ge_u (local.get $n) (local.get $N)))
          (local.set $b0 (i32.add (local.get $B) (i32.mul (local.get $n) (local.get $rb))))
          (local.set $b1 (i32.add (local.get $b0) (local.get $rb)))
          (local.set $c00 (v128.const i32x4 0 0 0 0)) (local.set $c01 (v128.const i32x4 0 0 0 0))
          (local.set $c10 (v128.const i32x4 0 0 0 0)) (local.set $c11 (v128.const i32x4 0 0 0 0))
          (local.set $c20 (v128.const i32x4 0 0 0 0)) (local.set $c21 (v128.const i32x4 0 0 0 0))
          (local.set $c30 (v128.const i32x4 0 0 0 0)) (local.set $c31 (v128.const i32x4 0 0 0 0))
          (local.set $k (i32.const 0))
          (block $kdone (loop $kloop
            (br_if $kdone (i32.ge_u (local.get $k) (local.get $rb)))
            (local.set $u (v128.load (i32.add (local.get $b0) (local.get $k))))
            (local.set $w (v128.load (i32.add (local.get $b1) (local.get $k))))
            (local.set $x (v128.load (i32.add (local.get $a0) (local.get $k))))
            (local.set $c00 (f32x4.add (local.get $c00) (f32x4.mul (local.get $x) (local.get $u))))
            (local.set $c01 (f32x4.add (local.get $c01) (f32x4.mul (local.get $x) (local.get $w))))
            (local.set $x (v128.load (i32.add (local.get $a1) (local.get $k))))
            (local.set $c10 (f32x4.add (local.get $c10) (f32x4.mul (local.get $x) (local.get $u))))
            (local.set $c11 (f32x4.add (local.get $c11) (f32x4.mul (local.get $x) (local.get $w))))
            (local.set $x (v128.load (i32.add (local.get $a2) (local.get $k))))
            (local.set $c20 (f32x4.add (local.get $c20) (f32x4.mul (local.get $x) (local.get $u))))
            (local.set $c21 (f32x4.add (local.get $c21) (f32x4.mul (local.get $x) (local.get $w))))
            (local.set $x (v128.load (i32.add (local.get $a3) (local.get $k))))
            (local.set $c30 (f32x4.add (local.get $c30) (f32x4.mul (local.get $x) (local.get $u))))
            (local.set $c31 (f32x4.add (local.get $c31) (f32x4.mul (local.get $x) (local.get $w))))
            (local.set $k (i32.add (local.get $k) (i32.const 16)))
            (br $kloop)))
          ;; c = &C[m, n]; write the horizontal sums of the eight accumulators
          (local.set $c (i32.add (local.get $C) (i32.add (i32.mul (local.get $m) (local.get $cb)) (i32.shl (local.get $n) (i32.const 2)))))
          (f32.store offset=0 (local.get $c) (call $hsum (local.get $c00)))
          (f32.store offset=4 (local.get $c) (call $hsum (local.get $c01)))
          (local.set $c (i32.add (local.get $c) (local.get $cb)))
          (f32.store offset=0 (local.get $c) (call $hsum (local.get $c10)))
          (f32.store offset=4 (local.get $c) (call $hsum (local.get $c11)))
          (local.set $c (i32.add (local.get $c) (local.get $cb)))
          (f32.store offset=0 (local.get $c) (call $hsum (local.get $c20)))
          (f32.store offset=4 (local.get $c) (call $hsum (local.get $c21)))
          (local.set $c (i32.add (local.get $c) (local.get $cb)))
          (f32.store offset=0 (local.get $c) (call $hsum (local.get $c30)))
          (f32.store offset=4 (local.get $c) (call $hsum (local.get $c31)))
          (local.set $n (i32.add (local.get $n) (i32.const 2)))
          (br $nloop)))
        (local.set $m (i32.add (local.get $m) (i32.const 4)))
        (br $mloop))))
    (func $hsum (param $v v128) (result f32)
      (f32.add (f32.add (f32x4.extract_lane 0 (local.get $v)) (f32x4.extract_lane 1 (local.get $v)))
               (f32.add (f32x4.extract_lane 2 (local.get $v)) (f32x4.extract_lane 3 (local.get $v)))))
  )
*/
const MM_WASM = 'AGFzbQEAAAABDwJgBn9/f39/fwBgAXsBfQMDAgABBQMBAAEHDAIDbWVtAgACbW0AAArqBALNBAIMfwt7IARBAnQhCSAFQQJ0IQ'
  + 'oCQANAIAYgA08NASAAIAYgCWxqIQsgCyAJaiEMIAwgCWohDSANIAlqIQ5BACEHAkADQCAHIAVPDQEgASAHIAlsaiEPIA8gCWohEP0MAAAAAAA'
  + 'AAAAAAAAAAAAAACES/QwAAAAAAAAAAAAAAAAAAAAAIRP9DAAAAAAAAAAAAAAAAAAAAAAhFP0MAAAAAAAAAAAAAAAAAAAAACEV/QwAAAAAAAAAAA'
  + 'AAAAAAAAAAIRb9DAAAAAAAAAAAAAAAAAAAAAAhF/0MAAAAAAAAAAAAAAAAAAAAACEY/QwAAAAAAAAAAAAAAAAAAAAAIRlBACEIAkADQCAIIAlPD'
  + 'QEgDyAIav0ABAAhGyAQIAhq/QAEACEcIAsgCGr9AAQAIRogEiAaIBv95gH95AEhEiATIBogHP3mAf3kASETIAwgCGr9AAQAIRogFCAaIBv95g'
  + 'H95AEhFCAVIBogHP3mAf3kASEVIA0gCGr9AAQAIRogFiAaIBv95gH95AEhFiAXIBogHP3mAf3kASEXIA4gCGr9AAQAIRogGCAaIBv95gH95AEhG'
  + 'CAZIBogHP3mAf3kASEZIAhBEGohCAwACwsgAiAGIApsIAdBAnRqaiERIBEgEhABOAIAIBEgExABOAIEIBEgCmohESARIBQQATgCACARIBUQATgC'
  + 'BCARIApqIREgESAWEAE4AgAgESAXEAE4AgQgESAKaiERIBEgGBABOAIAIBEgGRABOAIEIAdBAmohBwwACwsgBkEEaiEGDAALCwsZACAA/R8AIAD'
  + '9HwGSIAD9HwIgAP0fA5KSCw==';

let wasm = null;
try {
  const bytes = Uint8Array.from(atob(MM_WASM), (ch) => ch.charCodeAt(0));
  wasm = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports; // 670 bytes: sync compile is fine
} catch (err) {
  wasm = null; // no WebAssembly SIMD (old browser, or a CSP without wasm-unsafe-eval): stay in JavaScript
}
/** True when this engine can run the SIMD kernel. */
export const simdSupported = wasm !== null;
let useSimd = simdSupported;
/** Turn the SIMD path on or off (it stays off where unsupported). Returns the new setting. */
export function setSimd(on) {
  useSimd = Boolean(on) && simdSupported;
  return useSimd;
}

function simdMatmulBT(A, Bm, C, M, K, N) {
  const need = (M * K + N * K + M * N) * 4;
  if (wasm.mem.buffer.byteLength < need) wasm.mem.grow(Math.ceil((need - wasm.mem.buffer.byteLength) / 65536));
  const f = new Float32Array(wasm.mem.buffer);
  f.set(A.subarray(0, M * K), 0);
  f.set(Bm.subarray(0, N * K), M * K);
  wasm.mm(0, M * K * 4, (M * K + N * K) * 4, M, K, N);
  C.set(f.subarray(M * K + N * K, M * K + N * K + M * N));
}

/** out[c, r] = a[r, c] for a: [R, C]. */
export function transpose(a, out, R, C) {
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) out[c * R + r] = a[r * C + c];
}

// Reusable scratch buffers (one set per array type), so the training loop allocates nothing.
const pools = new Map();
function scratch(Arr, name, size) {
  const key = `${Arr.name}:${name}`;
  let buf = pools.get(key);
  if (!buf || buf.length < size) { buf = new Arr(size); pools.set(key, buf); }
  return buf;
}

/** Linear layer, PyTorch layout: y = x W^T + b with W of shape [nOut, nIn]; bias may be null. */
export function linearForward(x, W, bias, y, N, nIn, nOut) {
  matmulBT(x, W, y, N, nIn, nOut);
  if (bias) for (let n = 0; n < N; n++) for (let o = 0; o < nOut; o++) y[n * nOut + o] += bias[o];
}

/** Backward of the linear layer: dx = dy W, dW += dy^T x, db += sum of dy over rows. */
export function linearBackward(dy, x, W, dx, dW, dBias, N, nIn, nOut) {
  const Arr = dy.constructor;
  const Wt = scratch(Arr, 'Wt', nIn * nOut), dyT = scratch(Arr, 'dyT', nOut * N);
  const xT = scratch(Arr, 'xT', nIn * N), g = scratch(Arr, 'gW', nOut * nIn);
  transpose(W, Wt, nOut, nIn);
  matmulBT(dy, Wt, dx, N, nOut, nIn); // dx = dy W
  transpose(dy, dyT, N, nOut);
  transpose(x, xT, N, nIn);
  matmulBT(dyT, xT, g, nOut, N, nIn); // dy^T x
  for (let i = 0; i < nOut * nIn; i++) dW[i] += g[i];
  if (dBias) {
    for (let o = 0; o < nOut; o++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += dyT[o * N + n];
      dBias[o] += s;
    }
  }
}

/** LayerNorm over the last axis (biased variance, eps = 1e-5, like nn.LayerNorm). Caches xhat and 1/std. */
export function layerNormForward(x, gamma, beta, y, xhat, rstd, N, E, eps = 1e-5) {
  for (let n = 0; n < N; n++) {
    const o = n * E;
    let mean = 0;
    for (let e = 0; e < E; e++) mean += x[o + e];
    mean /= E;
    let v = 0;
    for (let e = 0; e < E; e++) { const d = x[o + e] - mean; v += d * d; }
    const r = 1 / Math.sqrt(v / E + eps);
    rstd[n] = r;
    for (let e = 0; e < E; e++) {
      const h = (x[o + e] - mean) * r;
      xhat[o + e] = h;
      y[o + e] = h * gamma[e] + beta[e];
    }
  }
}

/** Backward of LayerNorm: dx = rstd * (g - mean(g) - xhat * mean(g * xhat)), where g = dy * gamma. */
export function layerNormBackward(dy, xhat, rstd, gamma, dx, dGamma, dBeta, N, E) {
  for (let n = 0; n < N; n++) {
    const o = n * E;
    let mg = 0, mgx = 0;
    for (let e = 0; e < E; e++) {
      const d = dy[o + e];
      dGamma[e] += d * xhat[o + e];
      dBeta[e] += d;
      const g = d * gamma[e];
      mg += g;
      mgx += g * xhat[o + e];
    }
    mg /= E; mgx /= E;
    const r = rstd[n];
    for (let e = 0; e < E; e++) dx[o + e] = r * (dy[o + e] * gamma[e] - mg - xhat[o + e] * mgx);
  }
}

const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

/** Exact GELU, like nn.GELU(): y = x * Phi(x). Caches Phi(x) for the backward pass. */
export function geluForward(x, y, cdf, n) {
  for (let i = 0; i < n; i++) {
    const p = 0.5 * (1 + erf(x[i] * Math.SQRT1_2));
    cdf[i] = p;
    y[i] = x[i] * p;
  }
}

/** Backward of GELU: d/dx [x * Phi(x)] = Phi(x) + x * phi(x). */
export function geluBackward(dy, x, cdf, dx, n) {
  for (let i = 0; i < n; i++) {
    const v = x[i];
    dx[i] = dy[i] * (cdf[i] + v * INV_SQRT_2PI * Math.exp(-0.5 * v * v));
  }
}

/** Copy head h of batch row b out of [B*T, H*D] into a contiguous [T, D] block (transposed: [D, T]). */
function gatherHead(src, dst, b, h, T, D, E, transposed = false) {
  for (let t = 0; t < T; t++) {
    const s = (b * T + t) * E + h * D;
    if (transposed) for (let d = 0; d < D; d++) dst[d * T + t] = src[s + d];
    else for (let d = 0; d < D; d++) dst[t * D + d] = src[s + d];
  }
}

/** The inverse of gatherHead: write a [T, D] block back into head h of batch row b. */
function scatterHead(src, dst, b, h, T, D, E) {
  for (let t = 0; t < T; t++) {
    const s = (b * T + t) * E + h * D;
    for (let d = 0; d < D; d++) dst[s + d] = src[t * D + d];
  }
}

/** Causal multi-head attention, the core of MultiHeadAttention.forward in codebot/model.py.
    q, k, v, out: [B*T, H*D] (heads side by side, as W_q/W_k/W_v produce them); P: [B, H, T, T].
    Per (batch, head): S = Q K^T / sqrt(D), mask the future (j > i), softmax each row into P, O = P V. */
export function attentionForward(q, k, v, out, P, B, T, H, D) {
  const E = H * D, scale = 1 / Math.sqrt(D), Arr = q.constructor;
  const Q = scratch(Arr, 'aQ', T * D), K = scratch(Arr, 'aK', T * D);
  const Vt = scratch(Arr, 'aVt', D * T), O = scratch(Arr, 'aO', T * D);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      gatherHead(q, Q, b, h, T, D, E);
      gatherHead(k, K, b, h, T, D, E);
      gatherHead(v, Vt, b, h, T, D, E, true);
      const S = P.subarray((b * H + h) * T * T, (b * H + h + 1) * T * T);
      matmulBT(Q, K, S, T, D, T); // S[i, j] = q_i . k_j
      for (let i = 0; i < T; i++) {
        const r = i * T;
        let max = -Infinity;
        for (let j = 0; j <= i; j++) { S[r + j] *= scale; if (S[r + j] > max) max = S[r + j]; }
        let sum = 0;
        for (let j = 0; j <= i; j++) { const w = Math.exp(S[r + j] - max); S[r + j] = w; sum += w; }
        for (let j = 0; j <= i; j++) S[r + j] /= sum;
        for (let j = i + 1; j < T; j++) S[r + j] = 0; // masked_fill(-inf) then softmax gives exactly 0
      }
      matmulBT(S, Vt, O, T, T, D); // O = P V
      scatterHead(O, out, b, h, T, D, E);
    }
  }
}

/** Backward of attentionForward, per (batch, head), with G = dO V^T:
    dS = P * (G - rowsum(P * G)) / sqrt(D)   (softmax backward; zero where P is masked)
    dQ = dS K,   dK = dS^T Q,   dV = P^T dO. */
export function attentionBackward(dout, q, k, v, P, dq, dk, dv, B, T, H, D) {
  const E = H * D, scale = 1 / Math.sqrt(D), Arr = q.constructor;
  const Q = scratch(Arr, 'aQ', T * D), K = scratch(Arr, 'aK', T * D), V = scratch(Arr, 'aV', T * D);
  const dO = scratch(Arr, 'adO', T * D), X = scratch(Arr, 'aX', T * D), out = scratch(Arr, 'aO', T * D);
  const G = scratch(Arr, 'aG', T * T), dS = scratch(Arr, 'adS', T * T), Mt = scratch(Arr, 'aMt', T * T);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      gatherHead(q, Q, b, h, T, D, E);
      gatherHead(k, K, b, h, T, D, E);
      gatherHead(v, V, b, h, T, D, E);
      gatherHead(dout, dO, b, h, T, D, E);
      const Pb = P.subarray((b * H + h) * T * T, (b * H + h + 1) * T * T);
      matmulBT(dO, V, G, T, D, T); // G[i, j] = dout_i . v_j
      for (let i = 0; i < T; i++) {
        const r = i * T;
        let dot = 0;
        for (let j = 0; j <= i; j++) dot += Pb[r + j] * G[r + j];
        for (let j = 0; j < T; j++) dS[r + j] = j <= i ? Pb[r + j] * (G[r + j] - dot) * scale : 0;
      }
      transpose(Pb, Mt, T, T);
      gatherHead(dout, X, b, h, T, D, E, true);
      matmulBT(Mt, X, out, T, T, D); // dV = P^T dO
      scatterHead(out, dv, b, h, T, D, E);
      gatherHead(k, X, b, h, T, D, E, true);
      matmulBT(dS, X, out, T, T, D); // dQ = dS K
      scatterHead(out, dq, b, h, T, D, E);
      transpose(dS, Mt, T, T);
      gatherHead(q, X, b, h, T, D, E, true);
      matmulBT(Mt, X, out, T, T, D); // dK = dS^T Q
      scatterHead(out, dk, b, h, T, D, E);
    }
  }
}

/** Mean cross-entropy of logits [N, V] against integer targets. Turns the logits into probabilities in place. */
export function crossEntropyForward(logits, targets, N, V) {
  let total = 0;
  for (let n = 0; n < N; n++) {
    const o = n * V;
    let max = -Infinity;
    for (let c = 0; c < V; c++) if (logits[o + c] > max) max = logits[o + c];
    const target = logits[o + targets[n]];
    let sum = 0;
    for (let c = 0; c < V; c++) { const w = Math.exp(logits[o + c] - max); logits[o + c] = w; sum += w; }
    for (let c = 0; c < V; c++) logits[o + c] /= sum;
    total += Math.log(sum) + max - target; // -log softmax(target), without log(0)
  }
  return total / N;
}

/** Backward of the mean cross-entropy: dlogits = (softmax - onehot) / N. May write over probs. */
export function crossEntropyBackward(probs, targets, dlogits, N, V) {
  for (let n = 0; n < N; n++) {
    const o = n * V;
    for (let c = 0; c < V; c++) dlogits[o + c] = probs[o + c] / N;
    dlogits[o + targets[n]] -= 1 / N;
  }
}

// ------------------------------------------------------------------ the model
/** Count of parameters, exactly as sum(p.numel() for p in model.parameters()) on the DLFS6 GPT. */
export function paramCount({ vocabSize: V, ctx: C, embed: E, layers: L, ff: F }) {
  const block = 2 * E + 4 * E * E + 2 * E + (E * F + F) + (F * E + E);
  return V * E + C * E + L * block + 2 * E + V;
}

/** Build a GPT with the DLFS6 parameter names. cfg: { vocabSize, ctx, embed, heads, layers, ff }. */
export function createModel(cfg, { dtype = Float32Array } = {}) {
  const { vocabSize: V, ctx: C, embed: E, heads: H, layers: L, ff: F } = cfg;
  if (E % H !== 0) throw new Error(`embed (${E}) 必须能被 heads (${H}) 整除`);
  const params = [];
  const add = (name, shape, init) => {
    const size = shape.reduce((a, b) => a * b, 1);
    const p = { name, shape, init, data: new dtype(size), grad: new dtype(size) };
    params.push(p);
    return p;
  };
  const m = { cfg: { ...cfg }, dtype, params, blocks: [] };
  m.tok = add('embed.weight', [V, E], 'normal'); // also unembed.weight (tied)
  m.pos = add('pos_embed.weight', [C, E], 'normal');
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    m.blocks.push({
      ln1g: add(`${p}norm1.weight`, [E], 'ones'), ln1b: add(`${p}norm1.bias`, [E], 'zeros'),
      wq: add(`${p}attn.W_q.weight`, [E, E], 'normal'), wk: add(`${p}attn.W_k.weight`, [E, E], 'normal'),
      wv: add(`${p}attn.W_v.weight`, [E, E], 'normal'), wo: add(`${p}attn.W_o.weight`, [E, E], 'normal'),
      ln2g: add(`${p}norm2.weight`, [E], 'ones'), ln2b: add(`${p}norm2.bias`, [E], 'zeros'),
      w1: add(`${p}ffn.layers.0.weight`, [F, E], 'normal'), b1: add(`${p}ffn.layers.0.bias`, [F], 'zeros'),
      w2: add(`${p}ffn.layers.2.weight`, [E, F], 'normal'), b2: add(`${p}ffn.layers.2.bias`, [E], 'zeros'),
    });
  }
  m.lnfg = add('norm.weight', [E], 'ones');
  m.lnfb = add('norm.bias', [E], 'zeros');
  m.ubias = add('unembed.bias', [V], 'zeros');
  return m;
}

/** DLFS6 GPT._init_weights: Linear/Embedding weights ~ N(0, 0.02^2), biases 0, LayerNorm weight 1 / bias 0. */
export function initWeights(m, rng, std = 0.02) {
  for (const p of m.params) {
    if (p.init === 'normal') for (let i = 0; i < p.data.length; i++) p.data[i] = std * rng.normal();
    else p.data.fill(p.init === 'ones' ? 1 : 0);
  }
}

/** Activation buffers for batches of up to B sequences of up to T tokens. */
export function makeWorkspace(m, B, T) {
  const { vocabSize: V, embed: E, heads: H, ff: F } = m.cfg;
  const A = m.dtype, N = B * T;
  const layer = () => ({
    xhat1: new A(N * E), rstd1: new A(N), h1: new A(N * E), q: new A(N * E), k: new A(N * E), v: new A(N * E),
    P: new A(B * H * T * T), att: new A(N * E), xhat2: new A(N * E), rstd2: new A(N), h2: new A(N * E),
    u: new A(N * F), cdf: new A(N * F), g: new A(N * F),
  });
  return {
    x: new A(N * E), tmp: new A(N * E), layers: m.blocks.map(layer),
    xhatf: new A(N * E), rstdf: new A(N), hf: new A(N * E), logits: new A(N * V),
    dx: new A(N * E), dh: new A(N * E), dtmp: new A(N * E), datt: new A(N * E),
    dq: new A(N * E), dk: new A(N * E), dv: new A(N * E), dg: new A(N * F), du: new A(N * F),
    ids: null, targets: null, bt: [B, T], // the last forward()'s batch, for backward()
  };
}

/** Forward pass for ids [B*T] (B and T may be smaller than the workspace's).
    With targets: returns the mean cross-entropy and keeps what backward() needs.
    Without targets: leaves logits in ws.logits. With onlyRow = r (generation), only row r's logits are
    computed, into ws.logits[0 .. V-1]. */
export function forward(m, ws, ids, targets, B, T, { onlyRow = -1 } = {}) {
  const { vocabSize: V, embed: E, heads: H, ff: F } = m.cfg;
  const N = B * T, D = E / H, x = ws.x;
  if (T > m.cfg.ctx) throw new Error(`序列长度 ${T} 超过了上下文长度 ${m.cfg.ctx}`);
  embeddingForward(ids, m.tok.data, m.pos.data, x, B, T, E);
  m.blocks.forEach((blk, l) => {
    const c = ws.layers[l];
    layerNormForward(x, blk.ln1g.data, blk.ln1b.data, c.h1, c.xhat1, c.rstd1, N, E);
    linearForward(c.h1, blk.wq.data, null, c.q, N, E, E);
    linearForward(c.h1, blk.wk.data, null, c.k, N, E, E);
    linearForward(c.h1, blk.wv.data, null, c.v, N, E, E);
    attentionForward(c.q, c.k, c.v, c.att, c.P, B, T, H, D);
    linearForward(c.att, blk.wo.data, null, ws.tmp, N, E, E);
    for (let i = 0; i < N * E; i++) x[i] += ws.tmp[i]; // x = x + attn(norm1(x))
    layerNormForward(x, blk.ln2g.data, blk.ln2b.data, c.h2, c.xhat2, c.rstd2, N, E);
    linearForward(c.h2, blk.w1.data, blk.b1.data, c.u, N, E, F);
    geluForward(c.u, c.g, c.cdf, N * F);
    linearForward(c.g, blk.w2.data, blk.b2.data, ws.tmp, N, F, E);
    for (let i = 0; i < N * E; i++) x[i] += ws.tmp[i]; // x = x + ffn(norm2(x))
  });
  if (onlyRow >= 0) { // generation: one position's logits are enough
    const o = onlyRow * E;
    layerNormForward(x.subarray(o), m.lnfg.data, m.lnfb.data, ws.hf, ws.xhatf, ws.rstdf, 1, E);
    linearForward(ws.hf, m.tok.data, m.ubias.data, ws.logits, 1, E, V);
    return null;
  }
  layerNormForward(x, m.lnfg.data, m.lnfb.data, ws.hf, ws.xhatf, ws.rstdf, N, E);
  linearForward(ws.hf, m.tok.data, m.ubias.data, ws.logits, N, E, V); // unembed.weight is embed.weight
  ws.ids = ids; ws.targets = targets; ws.bt = [B, T];
  return targets ? crossEntropyForward(ws.logits, targets, N, V) : null;
}

/** Backward pass after forward(..., targets): accumulates every parameter's .grad. */
export function backward(m, ws) {
  const { vocabSize: V, embed: E, heads: H, ff: F } = m.cfg;
  const [B, T] = ws.bt, N = B * T, D = E / H, dx = ws.dx;
  crossEntropyBackward(ws.logits, ws.targets, ws.logits, N, V);
  // The unembedding shares embed.weight, so its gradient lands in the same buffer as the lookup's.
  linearBackward(ws.logits, ws.hf, m.tok.data, ws.dh, m.tok.grad, m.ubias.grad, N, E, V);
  layerNormBackward(ws.dh, ws.xhatf, ws.rstdf, m.lnfg.data, dx, m.lnfg.grad, m.lnfb.grad, N, E);
  for (let l = m.blocks.length - 1; l >= 0; l--) {
    const blk = m.blocks[l], c = ws.layers[l];
    // x_out = x_mid + ffn(norm2(x_mid)): the residual passes dx through unchanged, the branch adds to it.
    linearBackward(dx, c.g, blk.w2.data, ws.dg, blk.w2.grad, blk.b2.grad, N, F, E);
    geluBackward(ws.dg, c.u, c.cdf, ws.du, N * F);
    linearBackward(ws.du, c.h2, blk.w1.data, ws.dh, blk.w1.grad, blk.b1.grad, N, E, F);
    layerNormBackward(ws.dh, c.xhat2, c.rstd2, blk.ln2g.data, ws.dtmp, blk.ln2g.grad, blk.ln2b.grad, N, E);
    for (let i = 0; i < N * E; i++) dx[i] += ws.dtmp[i];
    // x_mid = x_in + attn(norm1(x_in))
    linearBackward(dx, c.att, blk.wo.data, ws.datt, blk.wo.grad, null, N, E, E);
    attentionBackward(ws.datt, c.q, c.k, c.v, c.P, ws.dq, ws.dk, ws.dv, B, T, H, D);
    linearBackward(ws.dq, c.h1, blk.wq.data, ws.dh, blk.wq.grad, null, N, E, E);
    linearBackward(ws.dk, c.h1, blk.wk.data, ws.dtmp, blk.wk.grad, null, N, E, E);
    for (let i = 0; i < N * E; i++) ws.dh[i] += ws.dtmp[i];
    linearBackward(ws.dv, c.h1, blk.wv.data, ws.dtmp, blk.wv.grad, null, N, E, E);
    for (let i = 0; i < N * E; i++) ws.dh[i] += ws.dtmp[i];
    layerNormBackward(ws.dh, c.xhat1, c.rstd1, blk.ln1g.data, ws.dtmp, blk.ln1g.grad, blk.ln1b.grad, N, E);
    for (let i = 0; i < N * E; i++) dx[i] += ws.dtmp[i];
  }
  embeddingBackward(ws.ids, dx, m.tok.grad, m.pos.grad, B, T, E);
}

export function zeroGrad(m) {
  for (const p of m.params) p.grad.fill(0);
}

// ------------------------------------------------------------------ optimizer
/** AdamW with torch.optim.AdamW's update (defaults betas (0.9, 0.999), eps 1e-8, weight_decay 0.01,
    applied to every parameter, as in DLFS6 ch03 `torch.optim.AdamW(model.parameters(), lr=...)`). */
export class AdamW {
  constructor(params, { lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0.01 } = {}) {
    Object.assign(this, { params, lr, betas, eps, weightDecay, t: 0 });
    this.m = params.map((p) => new p.data.constructor(p.data.length));
    this.v = params.map((p) => new p.data.constructor(p.data.length));
  }

  step() {
    this.t += 1;
    const [b1, b2] = this.betas, { lr, eps } = this;
    const stepSize = lr / (1 - b1 ** this.t);
    const bc2Sqrt = Math.sqrt(1 - b2 ** this.t);
    const decay = 1 - lr * this.weightDecay;
    this.params.forEach((p, k) => {
      const w = p.data, g = p.grad, m = this.m[k], v = this.v[k];
      for (let i = 0; i < w.length; i++) {
        w[i] *= decay; // decoupled weight decay
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        w[i] -= (stepSize * m[i]) / (Math.sqrt(v[i]) / bc2Sqrt + eps);
      }
    });
  }
}

// ------------------------------------------------------------------ text generation
/** Sample like DLFS6 codebot.utils.generate: keep the last ctx tokens, take the last position's
    logits, divide by the temperature (0 means argmax), stop at <|endoftext|> (endId) unless
    stopAtEnd is false. Returns the new ids. The window is padded at the end to a multiple of 4 rows so
    that the SIMD kernel applies; causal attention guarantees that the padding cannot change the logits
    of the last real position. */
export function generate(m, ws, promptIds, { maxNewTokens = 120, temperature = 1, rng, endId = -1, stopAtEnd = true } = {}) {
  const V = m.cfg.vocabSize, C = m.cfg.ctx;
  const ids = Array.from(promptIds), out = [], probs = new Float64Array(V);
  const window = new Int32Array(C);
  for (let s = 0; s < maxNewTokens; s++) {
    const ctx = ids.slice(-C), T = ctx.length, Tp = Math.min(C, Math.ceil(T / 4) * 4);
    window.fill(0);
    window.set(ctx);
    forward(m, ws, window, null, 1, Tp, { onlyRow: T - 1 });
    const logits = ws.logits;
    let next = 0;
    if (temperature === 0) {
      for (let c = 1; c < V; c++) if (logits[c] > logits[next]) next = c;
    } else {
      let max = -Infinity, sum = 0;
      for (let c = 0; c < V; c++) max = Math.max(max, logits[c] / temperature);
      for (let c = 0; c < V; c++) { probs[c] = Math.exp(logits[c] / temperature - max); sum += probs[c]; }
      let u = rng() * sum;
      next = V - 1;
      for (let c = 0; c < V; c++) { u -= probs[c]; if (u < 0) { next = c; break; } }
    }
    if (next === endId && stopAtEnd) break;
    ids.push(next);
    out.push(next);
  }
  return out;
}

// ------------------------------------------------------------------ data
/** Every window start 0 .. n-T-1 once per epoch in shuffled order, like DataLoader(TokenDataset, shuffle=True). */
export function makeBatcher(tokens, B, T, rng) {
  const nStarts = tokens.length - T;
  const order = new Int32Array(nStarts);
  let pos = nStarts;
  const x = new Int32Array(B * T), y = new Int32Array(B * T);
  return function next() {
    for (let b = 0; b < B; b++) {
      if (pos === nStarts) { // new epoch: Fisher-Yates shuffle of all window starts
        for (let i = 0; i < nStarts; i++) order[i] = i;
        for (let i = nStarts - 1; i > 0; i--) { const j = rng.int(i + 1); const t = order[i]; order[i] = order[j]; order[j] = t; }
        pos = 0;
      }
      const s = order[pos++];
      for (let t = 0; t < T; t++) { x[b * T + t] = tokens[s + t]; y[b * T + t] = tokens[s + t + 1]; }
    }
    return { x, y };
  };
}

/** B evenly spaced windows of a token array (a fixed held-out batch). */
export function fixedWindows(tokens, B, T) {
  const x = new Int32Array(B * T), y = new Int32Array(B * T);
  const span = tokens.length - T - 1;
  for (let b = 0; b < B; b++) {
    const s = Math.floor((b * span) / Math.max(1, B - 1));
    for (let t = 0; t < T; t++) { x[b * T + t] = tokens[s + t]; y[b * T + t] = tokens[s + t + 1]; }
  }
  return { x, y };
}

/** Count-based reference points for the loss curve: the cross-entropy (nats) on `heldOut` of a unigram
    model and of a bigram model, both counted on `train` with add-k smoothing. A GPT whose held-out loss
    is below the bigram line is using more than the previous character. */
export function ngramBaselines(train, heldOut, V, k = 0.1) {
  const uni = new Float64Array(V).fill(k), big = new Float64Array(V * V).fill(k);
  for (let i = 0; i < train.length; i++) uni[train[i]] += 1;
  for (let i = 1; i < train.length; i++) big[train[i - 1] * V + train[i]] += 1;
  const uniTotal = uni.reduce((a, b) => a + b, 0), rowTotal = new Float64Array(V);
  for (let a = 0; a < V; a++) for (let b = 0; b < V; b++) rowTotal[a] += big[a * V + b];
  let h1 = 0, h2 = 0;
  for (let i = 0; i < heldOut.length; i++) h1 -= Math.log(uni[heldOut[i]] / uniTotal);
  for (let i = 1; i < heldOut.length; i++) h2 -= Math.log(big[heldOut[i - 1] * V + heldOut[i]] / rowTotal[heldOut[i - 1]]);
  return { unigram: h1 / heldOut.length, bigram: h2 / (heldOut.length - 1) };
}

/** One training step on (x, y): zero_grad, forward, backward, AdamW. Returns the loss. */
export function trainStep(m, ws, opt, x, y, B, T) {
  zeroGrad(m);
  const loss = forward(m, ws, x, y, B, T);
  backward(m, ws);
  opt.step();
  return loss;
}

/** One training run, as in ch03/01_pretrain.py: model + AdamW + shuffled windows, plus a fixed held-out
    batch and a sampler. The Web Worker, the recorded runs and the tests all drive this same object.
    mode 'overfit' keeps training on the first batch drawn (the "overfit one batch" check). */
export function createSession({ trainIds, valIds, config, seed = 0, batch = 8, lr = 3e-3, evalBatch = 16 }) {
  const T = config.ctx, rng = makeRng(seed);
  const model = createModel(config);
  initWeights(model, rng);
  const opt = new AdamW(model.params, { lr });
  const ws = makeWorkspace(model, batch, T), evalWs = makeWorkspace(model, evalBatch, T), genWs = makeWorkspace(model, 1, T);
  const next = makeBatcher(trainIds, batch, T, rng);
  const first = next();
  const fixed = { x: Int32Array.from(first.x), y: Int32Array.from(first.y) };
  const held = fixedWindows(valIds, evalBatch, T);
  const sampleRng = makeRng(seed + 1);
  const session = {
    model, opt, batch, fixed, step: 0, mode: 'normal',
    /** Train on the next batch (or the fixed one); returns that batch's loss before the update. */
    trainStep() {
      const { x, y } = session.mode === 'overfit' ? fixed : next();
      const loss = trainStep(model, ws, opt, x, y, batch, T);
      session.step += 1;
      return loss;
    },
    heldOutLoss: () => forward(model, evalWs, held.x, held.y, evalBatch, T),
    fixedBatchLoss: () => forward(model, ws, fixed.x, fixed.y, batch, T),
    sample: (promptIds, opts) => generate(model, genWs, promptIds, { ...opts, rng: sampleRng }),
  };
  return session;
}
