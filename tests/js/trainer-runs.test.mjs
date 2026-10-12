// The recorded runs in web/data/train-in-browser/runs.json (shown on content/train-in-browser.md):
// they replay step for step with the current code, and every milestone the page quotes holds in them.
//   node --test tests/js/trainer-runs.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ngramBaselines, paramCount } from '../../web/widgets/_tinygpt.js';
import { CONFIG, SETTINGS, RUNS, loadData, runOnce } from './fixtures/trainer/record_runs.mjs';

const rec = JSON.parse(readFileSync(new URL('../../web/data/train-in-browser/runs.json', import.meta.url), 'utf8'));
const data = loadData();
const heldAt = (run, s) => run.held.find(([k]) => k === s)[1];
const firstBelow = (run, y) => run.held.find(([, x]) => x < y)?.[0];
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

test('the record was made with the current settings, data and model size', () => {
  assert.deepEqual(rec.settings, SETTINGS);
  assert.deepEqual(rec.config, { ...CONFIG, vocabSize: 96 });
  assert.equal(rec.params, paramCount(rec.config));
  assert.equal(rec.params, 93_408);
  assert.deepEqual(Object.keys(rec.runs), RUNS.map((r) => r.id));
  assert.equal(data.tok.vocabSize, 96);
  assert.deepEqual([data.trainIds.length, data.valIds.length], [211_113, 25_151]);
  const b = ngramBaselines(data.trainIds, data.valIds, 96);
  assert.ok(Math.abs(b.unigram - 3.3867) < 1e-4 && Math.abs(b.bigram - 2.6931) < 1e-4); // same as verify T.2-b
  assert.deepEqual(rec.baselines, b);
});

test('every run replays: the first 20 losses match the record exactly', () => {
  for (const run of RUNS) {
    const res = runOnce(data, run, 20);
    assert.deepEqual(res.losses, rec.runs[run.id].first, run.id);
    assert.ok(Math.abs(res.held[0][1] - rec.runs[run.id].held[0][1]) < 1e-4);
  }
});

test('default run: the milestones quoted on the page', () => {
  const run = rec.runs.default;
  assert.ok(Math.abs(heldAt(run, 0) - 4.5406) < 1e-4 && heldAt(run, 0) < rec.lnV);
  assert.equal(firstBelow(run, rec.baselines.unigram), 25);
  assert.equal(firstBelow(run, rec.baselines.bigram), 475);
  assert.ok(Math.abs(heldAt(run, 1000) - 2.2784) < 1e-4);
  assert.ok(Math.abs(heldAt(run, 2000) - 1.9511) < 1e-4);
  assert.ok(Math.abs(mean(run.losses.slice(-10)) - 1.8686) < 1e-4); // "training 1.87 vs held-out 1.95"
  // Timing quoted in T.1: 2,000 steps in 205 s, so step 475 at about 49 s.
  assert.ok(Math.abs(run.seconds - 204.7) < 0.05);
  assert.ok(Math.abs((475 / 2000) * run.seconds - 48.6) < 0.1);
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

test('overfit run: the training loss collapses, the held-out loss climbs above ln V', () => {
  const run = rec.runs.overfit, L = run.losses;
  assert.ok(mean(L.slice(90, 100)) < 0.035 && mean(L.slice(290, 300)) < 0.003);
  assert.ok(Math.abs(mean(L.slice(290, 300)) - 0.0027) < 5e-5);
  assert.ok(Math.abs(heldAt(run, 25) - 3.4193) < 1e-4); // first a little better ...
  for (let i = 2; i < run.held.length; i++) assert.ok(run.held[i][1] > run.held[i - 1][1]); // ... then only worse
  assert.ok(heldAt(run, 100) > rec.lnV && Math.abs(heldAt(run, 300) - 6.6048) < 1e-4);
});

test('learning rates at step 1,000: 3e-3 is best; 3e-4 crosses bigram only at 900; 1e-2 never', () => {
  const r = rec.runs;
  assert.ok(Math.abs(heldAt(r['lr3e-4'], 1000) - 2.6476) < 1e-4);
  assert.ok(Math.abs(heldAt(r['lr1e-2'], 1000) - 2.8184) < 1e-4);
  assert.equal(firstBelow(r['lr3e-4'], rec.baselines.bigram), 900);
  assert.equal(firstBelow(r['lr1e-2'], rec.baselines.bigram), undefined);
});

test('recorded speed: the SIMD kernel makes a training step at least 3x faster', () => {
  const { simd, js } = rec.stepsPerSecond;
  assert.ok(simd / js > 3, `${simd} vs ${js}`);
});
