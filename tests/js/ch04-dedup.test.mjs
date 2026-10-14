// web/widgets/ch04-dedup.js: the Heaps' law chart under figure 4-2, on lineChart's log axes.
//   node --test tests/js/ch04-dedup.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { heapsChart, compact } from '../../web/widgets/ch04-dedup.js';
import { logTicks } from '../../web/widgets/_plot.js';

const heaps = JSON.parse(fs.readFileSync(new URL('../../web/data/ch04/bench.json', import.meta.url), 'utf8')).heaps;

test('heapsChart: the data go in as measured, on log axes (not log10 values on linear axes)', () => {
  const { opts, runs } = heapsChart(heaps);
  assert.equal(opts.logX, true);
  assert.equal(opts.logY, true);
  assert.deepEqual(runs.map((r) => r.rows.length), [5, 4]);
  runs.forEach((run, i) => {
    assert.equal(opts.series[i].markers, true);
    assert.deepEqual(opts.series[i].points, run.rows.map((r) => [r.occurrences, r.unique]));
  });
  assert.deepEqual(opts.series[1].points.at(-1), [25254556, 21808]);
});

test('heapsChart: beta is verify cell 4.1-b\'s fit, and the dashed line is U = U0 (N / N0)^beta through the first training point', () => {
  const { beta, nRatio, uRatio, opts } = heapsChart(heaps);
  assert.ok(nRatio > 99 && uRatio < 5 && beta > 0.30 && beta < 0.36);       // the assertions of 4.1-b
  assert.equal(beta.toFixed(2), '0.33');                                      // the caption's "约 0.33 次方"
  const fit = opts.series[2];
  assert.equal(fit.name, `斜率 ${beta.toFixed(2)}`);
  assert.equal(fit.dash, '5 4');
  const [[n0, u0], [n1, u1]] = fit.points;
  assert.deepEqual([n0, n1], [253296, 25254556]);                             // spans every measured N
  const slope = (Math.log10(u1) - Math.log10(u0)) / (Math.log10(n1) - Math.log10(n0));
  assert.ok(Math.abs(slope - beta) < 1e-12);
  assert.ok(Math.abs(u0 - 4741 * (n0 / 253296) ** beta) < 1e-9);              // through (253,296, 4,741)
});

test('the axes get 1-2-5 ticks with round labels (they read 316K, 1.0M, 3.2M, 10M on linear log10 axes)', () => {
  const { opts } = heapsChart(heaps);
  const xs = opts.series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = opts.series.slice(0, 2).flatMap((s) => s.points.map((p) => p[1]));
  assert.deepEqual(logTicks(Math.min(...xs), Math.max(...xs)).map(opts.xFormat), ['500K', '1M', '2M', '5M', '10M', '20M']);
  assert.deepEqual(logTicks(Math.min(...ys), Math.max(...ys)).map(opts.yFormat), ['5K', '10K', '20K']);
});

test('compact: short labels without a trailing .0', () => {
  assert.deepEqual([999, 4741, 253296, 1e6, 2524909, 1e7, 25254556].map(compact),
    ['999', '5K', '253K', '1M', '2.5M', '10M', '25M']);
});
