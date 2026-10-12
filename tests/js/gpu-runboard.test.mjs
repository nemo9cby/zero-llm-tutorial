// The run board's text helpers (web/widgets/gpu-runboard.js) on the real run records.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { finalMetrics, costOf, duration, settledSince } from '../../web/widgets/gpu-runboard.js';

const runs = new URL('../../web/data/runs/', import.meta.url);
const report = (run) => JSON.parse(readFileSync(new URL(`${run}/report.json`, runs), 'utf8'));
const metrics = (run, stem) => readFileSync(new URL(`${run}/${stem}.metrics.jsonl`, runs), 'utf8')
  .trim().split('\n').map((ln) => JSON.parse(ln));

test('the probe shows when its val_loss was measured', () => {
  // the 200-step probe evaluates once, at step 0: val 10.98 next to train 6.14 is not a bug
  const since = settledSince(metrics('webbot-probe', '01_pretrain'), 'val_loss');
  assert.equal(since, 0);
  const final = report('webbot-probe').steps[0].final;
  assert.equal(finalMetrics(final, since), 'loss 6.139，val_loss 10.98（第 0 步的评估），lr 0.000239 @ 200');
});

test('a val_loss measured at the last step gets no note', () => {
  const pre = report('storybot-h100').steps[0];
  const since = settledSince(metrics('storybot-h100', '05_pretrain'), 'val_loss');
  assert.equal(since, pre.final.it);
  assert.equal(finalMetrics(pre.final, since), 'loss 1.331，val_loss 1.304 @ 40,000');
});

test('accuracy is printed as a percentage', () => {
  const grpo = report('codebot-3060ti').steps.find((s) => s.script === 'ch03/09_grpo.py');
  assert.equal(finalMetrics(grpo.final), 'loss -0.0046，acc 86.4% @ 500');
  assert.equal(finalMetrics(null), '—');
});

test('Modal costs are labelled as lower-bound estimates, local runs as free', () => {
  assert.match(costOf(report('storybot-h100')), /^估算费用 1\.17 美元（.*下限）$/);
  assert.equal(costOf(report('codebot-3060ti')), '费用 0（本机）');
  assert.equal(costOf({}), '费用未记录');
});

test('durations switch units at 90 s and 90 min', () => {
  assert.equal(duration(34.2), '34 秒');
  assert.equal(duration(973.6), '16.2 分钟');
  assert.equal(duration(5379.1), '89.7 分钟');
  assert.equal(duration(37_426), '10.40 小时');
  assert.equal(duration(null), '—');
});
