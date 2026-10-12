// The recorded runs in web/data/train-in-browser/runs.json (shown on content/train-in-browser.md):
// they replay step for step with the current code, and every milestone the page quotes holds in them.
//   node --test tests/js/trainer-runs.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ngramBaselines, paramCount } from '../../web/widgets/_tinygpt.js';
import { CONFIG, SETTINGS, RUNS, SEED_RUNS, OVERFIT_PROMPT, loadData, runOnce } from './fixtures/trainer/record_runs.mjs';

const rec = JSON.parse(readFileSync(new URL('../../web/data/train-in-browser/runs.json', import.meta.url), 'utf8'));
const data = loadData();
const heldAt = (run, s) => run.held.find(([k]) => k === s)[1];
const firstBelow = (points, y) => points.find(([, x]) => x < y)?.[0];
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const r4 = (v) => Math.round(v * 1e4) / 1e4;

test('the record was made with the current settings, data and model size', () => {
  assert.deepEqual(rec.settings, SETTINGS);
  assert.deepEqual(rec.config, { ...CONFIG, vocabSize: 96 });
  assert.equal(rec.params, paramCount(rec.config));
  assert.equal(rec.params, 93_408);
  assert.deepEqual(Object.keys(rec.runs), RUNS.map((r) => r.id));
  assert.deepEqual(rec.seeds.map((r) => r.id), SEED_RUNS.map((r) => r.id));
  assert.equal(data.tok.vocabSize, 96);
  assert.deepEqual([data.trainIds.length, data.valIds.length], [211_113, 25_151]);
  assert.deepEqual(rec.heldOut, { tokens: 25_151, targets: 25_150 }); // every held-out character after the first is predicted once
  const b = ngramBaselines(data.trainIds, data.valIds, 96);
  assert.ok(Math.abs(b.unigram - 3.3867) < 1e-4 && Math.abs(b.bigram - 2.6931) < 1e-4); // same as verify T.2-b
  assert.deepEqual(rec.baselines, b);
});

test('every run replays: its first 20 losses match the record exactly', () => {
  for (const run of [...RUNS, ...SEED_RUNS]) {
    const res = runOnce(data, run, 20, { heldOut: false });
    const recorded = rec.runs[run.id] ?? rec.seeds.find((r) => r.id === run.id);
    assert.deepEqual(res.losses, recorded.first, run.id);
  }
});

test('default run: the first 500 steps replay exactly (every loss, held-out loss and sample)', () => {
  const run = rec.runs.default, res = runOnce(data, RUNS[0], 500);
  assert.deepEqual(res.losses.slice(0, 20), run.first);
  assert.deepEqual(res.losses.map(r4), run.losses.slice(0, 500));
  assert.deepEqual(res.held.map(([k, v]) => [k, r4(v)]), run.held.filter(([k]) => k <= 500)); // includes the bigram crossing at 475
  assert.deepEqual(res.held1.map(([k, v]) => [k, r4(v)]), run.held1.filter(([k]) => k <= 500));
  assert.deepEqual(res.samples, run.samples.filter((s) => s.step <= 500));
});

test('default run: the milestones quoted on the page', () => {
  const run = rec.runs.default, b = rec.baselines;
  assert.ok(Math.abs(heldAt(run, 0) - 4.5459) < 1e-4 && heldAt(run, 0) < rec.lnV);
  assert.equal(firstBelow(run.held, b.unigram), 25);
  assert.equal(firstBelow(run.held, b.bigram), 475);
  assert.ok(heldAt(run, 450) - b.bigram < 0.01 && b.bigram - heldAt(run, 475) > 0.015); // just above at 450, clearly below at 475
  assert.ok(Math.abs(heldAt(run, 1000) - 2.2783) < 1e-4);
  assert.ok(Math.abs(heldAt(run, 2000) - 1.9377) < 1e-4);
  assert.ok(Math.abs(mean(run.losses.slice(-10)) - 1.8686) < 1e-4); // "training 1.87 vs held-out 1.94"
});

test('default run: it uses more than the previous character long before it crosses the bigram line', () => {
  const run = rec.runs.default, one = Object.fromEntries(run.held1), full = Object.fromEntries(run.held);
  assert.ok(one[100] - full[100] < 0.06);                          // step 100: hardly any gain from a longer context
  assert.ok(one[200] - full[200] > 0.14 && full[200] > rec.baselines.bigram + 0.2); // step 200: 0.15 better, still far above bigram
  assert.ok(one[2000] - full[2000] > 0.9);
  assert.ok(run.held1.every(([, x]) => x > rec.baselines.bigram + 0.15)); // with one character it never reaches the count bigram
});

test('default run: what the samples look like at the steps the page describes', () => {
  const at = (s) => rec.runs.default.samples.find((x) => x.step === s).text;
  const plain = (t) => t.split('<|endoftext|>').join('');
  const punct = (t) => new Set([...plain(t)].filter((c) => /[^A-Za-z0-9\s]/.test(c))).size;
  const spaces = (t) => [...plain(t)].filter((c) => c === ' ').length / plain(t).length;
  assert.ok(punct(at(0)) >= 20 && spaces(at(0)) === 0); // symbol soup
  assert.ok(spaces(at(100)) > 0.4); // spaces take over
  for (const piece of ['# ', ' = [', ' for ']) assert.ok(at(1000).includes(piece), piece);
  for (const piece of ['self.', '(string', 'import']) assert.ok(at(2000).includes(piece), piece);
});

test('overfit run: the batch is memorized word for word, the held-out loss climbs above ln V', () => {
  const run = rec.runs.overfit, L = run.losses;
  assert.ok(mean(L.slice(90, 100)) < 0.035 && mean(L.slice(290, 300)) < 0.003);
  assert.ok(Math.abs(mean(L.slice(290, 300)) - 0.0027) < 5e-5);
  assert.ok(Math.abs(heldAt(run, 25) - 3.378) < 1e-4); // first a little better ...
  for (let i = 2; i < run.held.length; i++) assert.ok(run.held[i][1] > run.held[i - 1][1]); // ... then only worse
  assert.equal(run.held.find(([, x]) => x > rec.lnV)[0], 75);
  assert.ok(Math.abs(heldAt(run, 300) - 6.7584) < 1e-4);
  assert.equal(run.batch.length, 8);
  assert.equal(run.prompt, run.batch[0].slice(0, OVERFIT_PROMPT));
  for (const s of run.samples.filter((x) => x.step >= 100)) assert.equal(run.prompt + s.text, run.batch[0], `step ${s.step}`);
  assert.notEqual(run.prompt + run.samples[0].text, run.batch[0]);
});

test('learning rates: 3e-4 too slow, 1e-2 too aggressive, 1e-3 and 3e-3 within the spread of seeds', () => {
  const r = rec.runs, b = rec.baselines.bigram;
  const at1000 = { '3e-4': heldAt(r['lr3e-4'], 1000), '1e-3': heldAt(r['lr1e-3'], 1000), '3e-3': heldAt(r.default, 1000), '1e-2': heldAt(r['lr1e-2'], 1000) };
  assert.deepEqual(at1000, { '3e-4': 2.6041, '1e-3': 2.28, '3e-3': 2.2783, '1e-2': 2.7633 });
  assert.deepEqual(['lr3e-4', 'lr1e-3', 'default', 'lr1e-2'].map((id) => firstBelow(r[id].held, b) ?? null), [675, 350, 475, null]);
  const seeds = {};
  for (const [lr, id] of [[3e-4, 'lr3e-4'], [1e-3, 'lr1e-3'], [3e-3, 'default'], [1e-2, 'lr1e-2']]) {
    seeds[lr] = [heldAt(r[id], 1000), ...rec.seeds.filter((s) => s.lr === lr).map((s) => heldAt(s, 1000))];
  }
  const spread = (a) => Math.max(...a) - Math.min(...a);
  assert.ok(mean(seeds[3e-3]) < mean(seeds[1e-3]) && mean(seeds[1e-3]) - mean(seeds[3e-3]) < spread(seeds[3e-3]));
  assert.ok(Math.min(...seeds[3e-4]) > Math.max(...seeds[1e-3]) + 0.25 && Math.min(...seeds[1e-2]) > Math.max(...seeds[1e-3]) + 0.4);
});

test('recorded speed: the SIMD kernel makes a training step at least 3x faster', () => {
  const { simd, js } = rec.timing.stepMs;
  assert.ok(js / simd > 3, `${simd} ms vs ${js} ms`);
});
