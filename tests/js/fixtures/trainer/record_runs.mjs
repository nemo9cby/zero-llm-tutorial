// Record the reference runs that content/train-in-browser.md shows before the reader trains anything.
//   node tests/js/fixtures/trainer/record_runs.mjs        -> web/data/train-in-browser/runs.json
// Each run drives createSession() from web/widgets/_tinygpt.js exactly like the Web Worker does
// (held-out loss every 25 steps, a sample at step 0 and every 100 steps), with seed 0.
// tests/js/trainer-runs.test.mjs re-runs the first steps and checks them against this file.
import { readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import { CharTokenizer, createSession, paramCount, ngramBaselines, simdSupported, setSimd } from '../../../../web/widgets/_tinygpt.js';

const DATA = new URL('../../../../web/data/train-in-browser/', import.meta.url);
export const CONFIG = { ctx: 64, embed: 64, heads: 4, layers: 2, ff: 192 };
export const SETTINGS = { batch: 8, evalEvery: 25, sampleEvery: 100, prompt: 'def ', temperature: 0.8, sampleChars: 120, seed: 0 };
export const RUNS = [
  { id: 'default', label: '正常训练，学习率 3e-3', lr: 3e-3, steps: 2000, mode: 'normal' },
  { id: 'overfit', label: '过拟合一个 batch，学习率 3e-3', lr: 3e-3, steps: 300, mode: 'overfit' },
  { id: 'lr3e-4', label: '学习率 3e-4（CodeBot 的值）', lr: 3e-4, steps: 1000, mode: 'normal' },
  { id: 'lr1e-2', label: '学习率 1e-2', lr: 1e-2, steps: 1000, mode: 'normal' },
];

export function loadData() {
  const train = readFileSync(new URL('tinycodes-train.txt', DATA), 'utf8');
  const val = readFileSync(new URL('tinycodes-val.txt', DATA), 'utf8');
  const tok = new CharTokenizer(train);
  return { tok, trainIds: Uint8Array.from(tok.encode(train)), valIds: Uint8Array.from(tok.encode(val)) };
}

/** Run one recorded configuration; `steps` may be cut short (the test replays only the first few). */
export function runOnce({ tok, trainIds, valIds }, run, steps = run.steps) {
  const config = { ...CONFIG, vocabSize: tok.vocabSize };
  const s = createSession({ trainIds, valIds, config, seed: SETTINGS.seed, batch: SETTINGS.batch, lr: run.lr });
  s.mode = run.mode;
  const prompt = tok.encode(SETTINGS.prompt);
  const sample = () => ({
    step: s.step,
    text: tok.decode(s.sample(prompt, {
      maxNewTokens: SETTINGS.sampleChars, temperature: SETTINGS.temperature, endId: tok.endId, stopAtEnd: false,
    })),
  });
  const out = { losses: [], held: [[0, s.heldOutLoss()]], samples: [sample()], seconds: 0 };
  const t0 = performance.now();
  while (s.step < steps) {
    out.losses.push(s.trainStep());
    if (s.step % SETTINGS.evalEvery === 0) out.held.push([s.step, s.heldOutLoss()]);
    if (s.step % SETTINGS.sampleEvery === 0) out.samples.push(sample());
  }
  out.seconds = (performance.now() - t0) / 1000;
  return out;
}

/** Median steps per second over 30 training steps, with the SIMD kernel on or off. */
function speed(data, simd) {
  setSimd(simd);
  const config = { ...CONFIG, vocabSize: data.tok.vocabSize };
  const s = createSession({ trainIds: data.trainIds, valIds: data.valIds, config, batch: SETTINGS.batch, lr: 3e-3 });
  for (let i = 0; i < 3; i++) s.trainStep();
  const times = [];
  for (let i = 0; i < 30; i++) { const t = performance.now(); s.trainStep(); times.push(performance.now() - t); }
  setSimd(true);
  times.sort((a, b) => a - b);
  return Math.round(10000 / times[15]) / 10;
}

function main() {
  const data = loadData();
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  const runs = {};
  for (const run of RUNS) {
    const res = runOnce(data, run);
    runs[run.id] = {
      ...run, seconds: Math.round(res.seconds * 10) / 10,
      first: res.losses.slice(0, 20), // full precision: the test replays these exactly
      losses: res.losses.map(r4), held: res.held.map(([k, v]) => [k, r4(v)]), samples: res.samples,
    };
    console.log(`${run.id}: ${run.steps} steps in ${res.seconds.toFixed(1)} s, held-out ${r4(res.held.at(-1)[1])}`);
  }
  const { tok, trainIds, valIds } = data;
  const config = { ...CONFIG, vocabSize: tok.vocabSize };
  const record = {
    provenance: {
      command: 'node tests/js/fixtures/trainer/record_runs.mjs', engine: `Node ${process.version} (V8 ${process.versions.v8})`,
      cpu: os.cpus()[0]?.model?.trim(), simd: simdSupported, date: new Date().toISOString().slice(0, 10),
      note: '录制时这台机器上还有其他任务，CPU 占用接近 100%，所以秒数偏慢；曲线只取决于步数和随机种子。',
    },
    config, params: paramCount(config), settings: SETTINGS, lnV: Math.log(tok.vocabSize),
    baselines: ngramBaselines(trainIds, valIds, tok.vocabSize), runs,
    stepsPerSecond: simdSupported ? { simd: speed(data, true), js: speed(data, false) } : { js: speed(data, false) },
  };
  writeFileSync(new URL('runs.json', DATA), `${JSON.stringify(record)}\n`);
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/').split('/').pop())) main();
