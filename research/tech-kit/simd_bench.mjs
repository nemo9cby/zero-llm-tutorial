// node simd_bench.mjs : WASM SIMD (f32x4) matvec vs plain JS, same shapes as CodeBot's linear layers.
import wabtInit from 'wabt';
import os from 'node:os';
try { os.setPriority(0, os.constants.priority.PRIORITY_HIGH); } catch {}

const WAT = `(module
  (memory (export "mem") 512)
  ;; y[o] = sum_i W[o*nIn + i] * x[i]   (W row-major (nOut, nIn), nIn % 16 == 0, byte offsets)
  (func (export "matvec") (param $W i32) (param $x i32) (param $y i32) (param $nIn i32) (param $nOut i32)
    (local $o i32) (local $i i32) (local $row i32) (local $xi i32) (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (block $done_o (loop $lo
      (br_if $done_o (i32.ge_u (local.get $o) (local.get $nOut)))
      (local.set $row (i32.add (local.get $W) (i32.shl (i32.mul (local.get $o) (local.get $nIn)) (i32.const 2))))
      (local.set $a0 (v128.const f32x4 0 0 0 0)) (local.set $a1 (v128.const f32x4 0 0 0 0))
      (local.set $a2 (v128.const f32x4 0 0 0 0)) (local.set $a3 (v128.const f32x4 0 0 0 0))
      (local.set $i (i32.const 0))
      (block $done_i (loop $li
        (br_if $done_i (i32.ge_u (local.get $i) (local.get $nIn)))
        (local.set $xi (i32.add (local.get $x) (i32.shl (local.get $i) (i32.const 2))))
        (local.set $a0 (f32x4.add (local.get $a0) (f32x4.mul (v128.load offset=0 (local.get $row)) (v128.load offset=0 (local.get $xi)))))
        (local.set $a1 (f32x4.add (local.get $a1) (f32x4.mul (v128.load offset=16 (local.get $row)) (v128.load offset=16 (local.get $xi)))))
        (local.set $a2 (f32x4.add (local.get $a2) (f32x4.mul (v128.load offset=32 (local.get $row)) (v128.load offset=32 (local.get $xi)))))
        (local.set $a3 (f32x4.add (local.get $a3) (f32x4.mul (v128.load offset=48 (local.get $row)) (v128.load offset=48 (local.get $xi)))))
        (local.set $row (i32.add (local.get $row) (i32.const 64)))
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $li)))
      (local.set $a0 (f32x4.add (f32x4.add (local.get $a0) (local.get $a1)) (f32x4.add (local.get $a2) (local.get $a3))))
      (f32.store (i32.add (local.get $y) (i32.shl (local.get $o) (i32.const 2)))
        (f32.add (f32.add (f32x4.extract_lane 0 (local.get $a0)) (f32x4.extract_lane 1 (local.get $a0)))
                 (f32.add (f32x4.extract_lane 2 (local.get $a0)) (f32x4.extract_lane 3 (local.get $a0)))))
      (local.set $o (i32.add (local.get $o) (i32.const 1)))
      (br $lo))))
)`;

const wabt = await wabtInit();
const mod = wabt.parseWat('matvec.wat', WAT, { simd: true });
const { buffer } = mod.toBinary({});
const { instance } = await WebAssembly.instantiate(buffer);
const { matvec, mem } = instance.exports;
console.log(`wasm module: ${buffer.byteLength} bytes`);

function jsMatvec(y, W, x, nIn, nOut) {
  for (let o = 0; o < nOut; o++) {
    const r = o * nIn; let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (let i = 0; i < nIn; i += 4) { s0 += W[r + i] * x[i]; s1 += W[r + i + 1] * x[i + 1]; s2 += W[r + i + 2] * x[i + 2]; s3 += W[r + i + 3] * x[i + 3]; }
    y[o] = s0 + s1 + s2 + s3;
  }
}
const bench = (fn, flops, minMs = 400) => { fn(); let n = 0; const t0 = performance.now(); let t; do { fn(); n++; t = performance.now() - t0; } while (t < minMs); return +(flops * n / t / 1e6).toFixed(2); };

for (const [nOut, nIn] of [[384, 384], [1536, 384], [384, 1536], [1000, 384]]) {
  const f32 = new Float32Array(mem.buffer);
  const Wp = 0, xp = nOut * nIn * 4, yp = xp + nIn * 4;
  const W = f32.subarray(Wp / 4, Wp / 4 + nOut * nIn), x = f32.subarray(xp / 4, xp / 4 + nIn), y = f32.subarray(yp / 4, yp / 4 + nOut);
  for (let i = 0; i < W.length; i++) W[i] = Math.random() - 0.5;
  for (let i = 0; i < nIn; i++) x[i] = Math.random() - 0.5;
  const yj = new Float32Array(nOut);
  jsMatvec(yj, W, x, nIn, nOut); matvec(Wp, xp, yp, nIn, nOut);
  let maxDiff = 0; for (let i = 0; i < nOut; i++) maxDiff = Math.max(maxDiff, Math.abs(yj[i] - y[i]));
  const flops = 2 * nOut * nIn;
  console.log(`matvec ${nOut}x${nIn}`.padEnd(18), `JS ${bench(() => jsMatvec(yj, W, x, nIn, nOut), flops)} GFLOP/s`.padEnd(20),
    `WASM-SIMD ${bench(() => matvec(Wp, xp, yp, nIn, nOut), flops)} GFLOP/s`.padEnd(26), `max|diff| ${maxDiff.toExponential(1)}`);
}
