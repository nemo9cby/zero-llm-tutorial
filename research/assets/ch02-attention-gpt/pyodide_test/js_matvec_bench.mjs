// Per-token decode cost of codebot GPT with a KV cache ~ all weight matvecs once:
// 6 layers x (4*384*384 + 2*384*1536) + 384*1000 multiply-adds.
function matvec(W, x, y, rows, cols) {
  for (let r = 0; r < rows; r++) {
    let s = 0, o = r * cols;
    for (let c = 0; c < cols; c++) s += W[o + c] * x[c];
    y[r] = s;
  }
}
const shapes = [];
for (let l = 0; l < 6; l++) shapes.push([384, 384], [384, 384], [384, 384], [384, 384], [1536, 384], [384, 1536]);
shapes.push([1000, 384]);
const Ws = shapes.map(([r, c]) => Float32Array.from({ length: r * c }, () => Math.random() - 0.5));
const x = Float32Array.from({ length: 1536 }, () => Math.random());
const y = new Float32Array(1536);
let macs = shapes.reduce((a, [r, c]) => a + r * c, 0);
for (let w = 0; w < 3; w++) shapes.forEach(([r, c], i) => matvec(Ws[i], x, y, r, c));
const t = performance.now(), N = 50;
for (let n = 0; n < N; n++) shapes.forEach(([r, c], i) => matvec(Ws[i], x, y, r, c));
const ms = (performance.now() - t) / N;
console.log(`MACs/token = ${macs.toLocaleString()} | ${ms.toFixed(1)} ms/token | ${(macs / ms / 1e6).toFixed(2)} GMAC/s (plain JS, 1 thread)`);
