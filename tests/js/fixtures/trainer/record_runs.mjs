// Record the reference runs that content/train-in-browser.md shows before the reader trains anything.
//   node tests/js/fixtures/trainer/record_runs.mjs [--jobs 4]  -> web/data/train-in-browser/runs.json (runs + timing)
//   node tests/js/fixtures/trainer/record_runs.mjs --timing    -> re-measure only the "timing" block
// Each run drives createSession() from web/widgets/_tinygpt.js exactly like the Web Worker does: the held-out
// loss on the whole held-out text every 25 steps, a sample at step 0 and every 100 steps. The losses depend only
// on the seed (and on the SIMD switch, see the test); the timing block depends on the machine and on how busy
// it is, so it is measured separately and records the CPU load at the time. After re-measuring, update the
// numbers that content/train-in-browser.md quotes from it (T.1, T.5, T.6 and T.8; their verify cells assert them).
// tests/js/trainer-runs.test.mjs replays the start of every run and the first 500 steps of the default run.
import { readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { CharTokenizer, createSession, paramCount, ngramBaselines, simdSupported, setSimd } from '../../../../web/widgets/_tinygpt.js';

const DATA = new URL('../../../../web/data/train-in-browser/', import.meta.url);
const OUT = new URL('runs.json', DATA);
export const CONFIG = { ctx: 64, embed: 64, heads: 4, layers: 2, ff: 192 };
export const SETTINGS = { batch: 8, evalEvery: 25, sampleEvery: 100, prompt: 'def ', temperature: 0.8, sampleChars: 120, seed: 0 };
/** Every run trains from seed 0. `context1`: also score the held-out text with a context of one character
    every 100 steps. The overfit run samples from the first 8 characters of the batch it memorizes. */
export const RUNS = [
  { id: 'default', label: '正常训练，学习率 3e-3', lr: 3e-3, steps: 2000, mode: 'normal', samples: true, context1: true },
  { id: 'overfit', label: '过拟合一个 batch，学习率 3e-3', lr: 3e-3, steps: 300, mode: 'overfit', samples: true },
  { id: 'lr3e-4', label: '学习率 3e-4（CodeBot 的值）', lr: 3e-4, steps: 1000, mode: 'normal' },
  { id: 'lr1e-3', label: '学习率 1e-3', lr: 1e-3, steps: 1000, mode: 'normal' },
  { id: 'lr1e-2', label: '学习率 1e-2', lr: 1e-2, steps: 1000, mode: 'normal' },
];
/** The learning-rate comparison again with seeds 1 and 2 (seed 0 is in RUNS); held-out loss every 100 steps. */
export const SEED_RUNS = [3e-4, 1e-3, 3e-3, 1e-2].flatMap((lr) => [1, 2].map((seed) => ({
  id: `lr${lr.toExponential(0)}-seed${seed}`, lr, seed, steps: 1000, mode: 'normal', evalEvery: 100,
})));
export const OVERFIT_PROMPT = 8; // characters of the memorized batch's first window given as the prompt

export function loadData() {
  const train = readFileSync(new URL('tinycodes-train.txt', DATA), 'utf8');
  const val = readFileSync(new URL('tinycodes-val.txt', DATA), 'utf8');
  const tok = new CharTokenizer(train);
  return { tok, trainIds: Uint8Array.from(tok.encode(train)), valIds: Uint8Array.from(tok.encode(val)) };
}

/** Run one recorded configuration; `steps` may be cut short (the test replays only the start).
    heldOut: false skips every held-out evaluation (the cheap replay of the first losses). */
export function runOnce({ tok, trainIds, valIds }, run, steps = run.steps, { heldOut = true } = {}) {
  const config = { ...CONFIG, vocabSize: tok.vocabSize };
  const s = createSession({ trainIds, valIds, config, seed: run.seed ?? SETTINGS.seed, batch: SETTINGS.batch, lr: run.lr });
  s.mode = run.mode;
  const evalEvery = run.evalEvery ?? SETTINGS.evalEvery;
  const overfit = run.mode === 'overfit';
  const prompt = overfit ? Array.from(s.fixed.x.subarray(0, OVERFIT_PROMPT)) : tok.encode(SETTINGS.prompt);
  const sample = (ids = prompt, chars = overfit ? CONFIG.ctx - OVERFIT_PROMPT : SETTINGS.sampleChars) => ({
    step: s.step,
    text: tok.decode(s.sample(ids, { maxNewTokens: chars, temperature: SETTINGS.temperature, endId: tok.endId, stopAtEnd: false })),
  });
  const out = { losses: [], held: [], held1: [], samples: [] };
  const evaluate = () => {
    if (!heldOut) return;
    if (s.step % evalEvery === 0) out.held.push([s.step, s.heldOutLoss()]);
    if (run.context1 && s.step % 100 === 0) out.held1.push([s.step, s.heldOutLoss(1)]);
  };
  evaluate();
  if (run.samples) out.samples.push(sample());
  while (s.step < steps) {
    out.losses.push(s.trainStep());
    evaluate();
    if (run.samples && s.step % SETTINGS.sampleEvery === 0) out.samples.push(sample());
  }
  if (overfit && run.samples) {
    out.prompt = tok.decode(prompt);
    out.batch = Array.from({ length: SETTINGS.batch }, (_, b) => tok.decode(s.fixed.x.subarray(b * CONFIG.ctx, (b + 1) * CONFIG.ctx)));
    out.offBatch = sample(tok.encode(SETTINGS.prompt), SETTINGS.sampleChars); // the memorized model, away from its batch
  }
  return out;
}

const r4 = (v) => Math.round(v * 1e4) / 1e4;

/** What runs.json keeps of a run: full-precision first 20 losses (the test replays them exactly), the rest at 4 decimals. */
function summarize(run, res) {
  const rec = { ...run, first: res.losses.slice(0, 20), held: res.held.map(([k, v]) => [k, r4(v)]) };
  delete rec.samples; delete rec.context1;
  if (run.samples) Object.assign(rec, { losses: res.losses.map(r4), samples: res.samples });
  if (run.context1) rec.held1 = res.held1.map(([k, v]) => [k, r4(v)]);
  for (const key of ['prompt', 'batch', 'offBatch']) if (res[key]) rec[key] = res[key];
  return rec;
}

// ------------------------------------------------------------------ timing
const cpuTimes = () => os.cpus().reduce((a, { times: t }) => {
  const busy = t.user + t.nice + t.sys + t.irq;
  return { busy: a.busy + busy, all: a.all + busy + t.idle };
}, { busy: 0, all: 0 });
const median = (a) => { const b = [...a].sort((x, y) => x - y); return b[b.length >> 1]; };
const time = (f) => { const t = performance.now(); f(); return performance.now() - t; };

/** Median milliseconds of a training step, a full held-out evaluation and a 120-character sample, with the
    SIMD kernel on and off, and the step time of a few other model sizes (for the page's "what if" column).
    Configurations are measured in interleaved rounds, so a change of machine load hits all of them alike. */
export function measureTiming(data, rounds = 5) {
  const { tok, trainIds, valIds } = data;
  const prompt = tok.encode(SETTINGS.prompt);
  const variants = [
    { id: 'base', change: '现在的设置', config: {} },
    { id: 'ff256', change: 'ff_dim 256（4 倍）', config: { ff: 256 } },
    { id: 'embed128', change: 'embed_dim 128（ff_dim 384）', config: { embed: 128, ff: 384 } },
    { id: 'layers4', change: 'n_layer 4', config: { layers: 4 } },
    { id: 'heads8', change: 'n_head 8（每个头 8 维）', config: { heads: 8 } },
    { id: 'ctx128', change: 'context_len 128', config: { ctx: 128 } },
    { id: 'batch16', change: 'batch 16', config: {}, batch: 16 },
  ];
  const session = (v, simd) => {
    setSimd(simd);
    const config = { ...CONFIG, ...v.config, vocabSize: tok.vocabSize };
    const s = createSession({ trainIds, valIds, config, batch: v.batch ?? SETTINGS.batch, lr: 3e-3 });
    for (let i = 0; i < 3; i++) s.trainStep(); // warm up
    return { s };
  };
  const step = { simd: [], js: [] }, held = [], sample = [], vt = variants.map(() => []);
  const c0 = cpuTimes(), t0 = Date.now();
  for (let r = 0; r < rounds; r++) {
    for (const simd of [true, false]) {
      const { s } = session(variants[0], simd);
      for (let i = 0; i < 10; i++) step[simd ? 'simd' : 'js'].push(time(() => s.trainStep()));
      if (simd) {
        held.push(time(() => s.heldOutLoss()));
        sample.push(time(() => s.sample(prompt, { maxNewTokens: SETTINGS.sampleChars, temperature: SETTINGS.temperature })));
      }
    }
    variants.forEach((v, k) => {
      const { s } = session(v, true);
      for (let i = 0; i < 6; i++) vt[k].push(time(() => s.trainStep()));
    });
  }
  setSimd(true);
  const c1 = cpuTimes();
  const ms = (a) => Math.round(median(a) * 10) / 10;
  return {
    stepMs: { simd: ms(step.simd), js: ms(step.js) }, heldOutMs: ms(held), sampleMs: ms(sample),
    variants: variants.map((v, k) => ({
      id: v.id, change: v.change, params: paramCount({ ...CONFIG, ...v.config, vocabSize: tok.vocabSize }), stepMs: ms(vt[k]),
    })),
    cpuBusy: Math.round(((c1.busy - c0.busy) / (c1.all - c0.all)) * 100) / 100, cores: os.cpus().length,
    seconds: Math.round((Date.now() - t0) / 100) / 10,
  };
}

// ------------------------------------------------------------------ main
function provenance(timing, recorded) {
  const today = new Date().toISOString().slice(0, 10);
  return {
    command: 'node tests/js/fixtures/trainer/record_runs.mjs', engine: `Node ${process.version} (V8 ${process.versions.v8})`,
    cpu: os.cpus()[0]?.model?.trim(), simd: simdSupported, recorded: recorded ?? today, timed: today,
    timingLoad: `计时时整台机器的 CPU 占用 ${Math.round(timing.cpuBusy * 100)}%（${timing.cores} 个逻辑核，计时程序自己占 1 个）`,
  };
}

function runInWorker(run) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL(import.meta.url), { workerData: { run } });
    w.once('message', resolve);
    w.once('error', reject);
    w.once('exit', (code) => { if (code !== 0) reject(new Error(`${run.id}: worker exited with ${code}`)); });
  });
}

async function recordAll(jobs) {
  const queue = [...RUNS, ...SEED_RUNS], results = new Map();
  const t0 = Date.now();
  const lane = async () => {
    for (let run = queue.shift(); run; run = queue.shift()) {
      results.set(run.id, summarize(run, await runInWorker(run)));
      const held = results.get(run.id).held.at(-1);
      console.log(`${run.id}: ${run.steps} steps, held-out ${held[1]} at step ${held[0]} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
    }
  };
  await Promise.all(Array.from({ length: jobs }, lane));
  const strip = ({ id, lr, seed, steps, first, held }) => ({ id, lr, seed, steps, first, held });
  return { runs: Object.fromEntries(RUNS.map((r) => [r.id, results.get(r.id)])), seeds: SEED_RUNS.map((r) => strip(results.get(r.id))) };
}

async function main() {
  const data = loadData();
  const argv = process.argv.slice(2);
  const jobs = Number(argv[argv.indexOf('--jobs') + 1]) || 4;
  const { tok, trainIds, valIds } = data;
  const config = { ...CONFIG, vocabSize: tok.vocabSize };
  let record;
  if (argv.includes('--timing')) {
    record = JSON.parse(readFileSync(OUT, 'utf8'));
  } else {
    const { runs, seeds } = await recordAll(jobs);
    record = {
      config, params: paramCount(config), settings: SETTINGS, lnV: Math.log(tok.vocabSize),
      heldOut: { tokens: valIds.length, targets: valIds.length - 1 },
      baselines: ngramBaselines(trainIds, valIds, tok.vocabSize), runs, seeds,
    };
  }
  const timing = measureTiming(data);
  console.log('timing:', JSON.stringify(timing));
  const { provenance: old, timing: _, ...rest } = record;
  record = { provenance: provenance(timing, old?.recorded), ...rest, timing };
  writeFileSync(OUT, `${JSON.stringify(record)}\n`);
}

if (!isMainThread && workerData?.run) {
  parentPort.postMessage(runOnce(loadData(), workerData.run));
} else if (isMainThread && process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/').split('/').pop())) {
  main();
}
