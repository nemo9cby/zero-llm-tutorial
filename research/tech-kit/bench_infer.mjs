// node bench_infer.mjs <out_dir> : parity vs PyTorch + tokens/sec for f32/f16/q8 weights.
import fs from 'node:fs';
import { loadModel, createSession, step, sample, makeTokenizer } from './gpt.mjs';

const dir = process.argv[2];
const ref = JSON.parse(fs.readFileSync(`${dir}/ref.json`, 'utf8'));
const tok = makeTokenizer(JSON.parse(fs.readFileSync(`${dir}/tokenizer.json`, 'utf8')));
const jsIds = tok.encode(ref.prompt);
console.log(`tokenizer parity: ${JSON.stringify(jsIds) === JSON.stringify(ref.prompt_ids) ? 'OK' : 'MISMATCH'} ${JSON.stringify(jsIds)}`);

for (const fmt of ['f32', 'f16', 'q8']) {
  const man = JSON.parse(fs.readFileSync(`${dir}/codebot.${fmt}.json`, 'utf8'));
  const b = fs.readFileSync(`${dir}/codebot.${fmt}.bin`);
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  let t0 = performance.now();
  const model = loadModel(man, buf);
  const loadMs = performance.now() - t0;
  const s = createSession(model);
  t0 = performance.now();
  let logits;
  for (const id of ref.prompt_ids) logits = step(s, id);
  const prefillMs = performance.now() - t0;
  let maxDiff = 0;
  for (let i = 0; i < logits.length; i++) maxDiff = Math.max(maxDiff, Math.abs(logits[i] - ref.last_logits[i]));
  const out = [];
  t0 = performance.now();
  for (let i = 0; i < ref.greedy_ids.length; i++) { const id = sample(logits, 0); out.push(id); logits = step(s, id); }
  const decMs = performance.now() - t0;
  const firstMismatch = out.findIndex((id, i) => id !== ref.greedy_ids[i]);
  t0 = performance.now();
  let n = 0;
  while (s.pos < model.cfg.T - 20) { logits = step(s, sample(logits, 1.0)); n++; }   // positions ~72..236
  const longMs = performance.now() - t0;
  console.log(JSON.stringify({
    fmt, fileMB: +(b.byteLength / 1e6).toFixed(2), loadMs: Math.round(loadMs), maxAbsLogitDiff: +maxDiff.toExponential(2),
    greedy64: firstMismatch === -1 ? 'identical to PyTorch' : `first mismatch at token ${firstMismatch}`,
    prefillTokPerS: Math.round(ref.prompt_ids.length / prefillMs * 1e3),
    decodeTokPerS_pos8to72: Math.round(64 / decMs * 1e3), decodeTokPerS_pos72to236: Math.round(n / longMs * 1e3),
  }));
  if (fmt === 'f32') console.log('greedy text (JS):', JSON.stringify(tok.decode(out).slice(0, 100)));
}
