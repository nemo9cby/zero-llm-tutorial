// Pure logic of the CS336 A1 results widget, checked against the recorded run it displays.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { dotFor, everFailed, ratioPos, summarize } from '../../web/widgets/cs336-a1.js';

const rec = JSON.parse(fs.readFileSync(new URL('../../web/data/cs336/a1_results.json', import.meta.url), 'utf8'));
const byName = Object.fromEntries(rec.tests.map((t) => [t.test, t]));
const SPEED = 'test_train_bpe_speed';   // a wall-clock gate: its outcome depends on how busy the machine was

// The ten tests that fail as shipped whatever the machine load (the cs336 page, X.2).
const SHIPPED_FAILURES = ['test_checkpointing', 'test_encode_iterable_tinystories_matches_tiktoken',
  'test_encode_iterable_tinystories_sample_roundtrip', 'test_get_batch', 'test_get_lr_cosine_schedule',
  'test_overlapping_special_tokens', 'test_train_bpe', 'test_train_bpe_special_tokens', 'test_transformer_lm',
  'test_transformer_lm_truncated_input'];

test('the tolerance line (ratio = 1) sits two thirds along the 1e-4 … 1e2 log axis', () => {
  assert.ok(Math.abs(ratioPos(1) - 200 / 3) < 1e-9);
  assert.equal(ratioPos(1e-4), 0);
  assert.equal(ratioPos(1e2), 100);
  assert.equal(ratioPos(1e7), 100);   // clamped (the cosine-schedule test)
  assert.equal(ratioPos(0), 0);       // exact matches sit at the left edge
});

test('summaries match the record for both variants', () => {
  for (const key of ['as_shipped', 'one_line_fixes']) {
    assert.deepEqual(summarize(rec.tests, key), rec.summary[key]);
  }
});

test('without the speed test the outcomes are fixed: 35/10/2 as shipped, 40/5/2 after the three fixes', () => {
  const others = rec.tests.filter((t) => t.test !== SPEED);
  assert.deepEqual(summarize(others, 'as_shipped'), { passed: 35, failed: 10, skipped: 2 });
  assert.deepEqual(summarize(others, 'one_line_fixes'), { passed: 40, failed: 5, skipped: 2 });
});

test('passing comparisons sit below the tolerance line; tolerance failures above it', () => {
  let failures = 0;
  for (const t of rec.tests) {
    for (const key of ['as_shipped', 'one_line_fixes']) {
      const r = t[key];
      if (!r.worst || r.worst.kind !== 'numpy.testing.assert_allclose') continue;
      const above = ratioPos(r.worst.tol_ratio) > ratioPos(1);
      if (r.outcome === 'passed') assert.ok(!above, `${t.test} / ${key}`);
      if (r.outcome === 'failed' && /Not equal to tolerance/.test(r.message || '')) {
        assert.ok(above, `${t.test} / ${key}`);
        failures += 1;
      }
    }
  }
  assert.equal(failures, 4);   // two LM tests as shipped + the cosine schedule in both variants
});

test('a failure that is not a tolerance failure gets no dot (test_get_batch as shipped)', () => {
  const gb = byName.test_get_batch;
  assert.equal(gb.as_shipped.outcome, 'failed');
  assert.ok(gb.as_shipped.worst && !(gb.as_shipped.worst.tol_ratio > 1));   // the passing x+1 == y comparison
  assert.equal(dotFor(gb.as_shipped), null);
  assert.equal(dotFor(gb.one_line_fixes), gb.one_line_fixes.worst.tol_ratio);   // passed: drawn
  assert.ok(dotFor(byName.test_transformer_lm.as_shipped) > 1);                // a real tolerance failure: drawn
  assert.equal(dotFor({ outcome: 'passed' }), null);                           // no numeric comparison at all
});

test('“失败过的” lists the 10 as-shipped failures, plus the speed test whenever it failed', () => {
  const failed = rec.tests.filter(everFailed).map((t) => t.test).sort();
  const expected = [...SHIPPED_FAILURES, ...(everFailed(byName[SPEED]) ? [SPEED] : [])].sort();
  assert.deepEqual(failed, expected);
  const shipped = rec.tests.filter((t) => t.test !== SPEED && t.as_shipped.outcome === 'failed').map((t) => t.test);
  assert.deepEqual(shipped.sort(), SHIPPED_FAILURES);
});

test('the “设计不同” and “无对应” filters show exactly the items the page names', () => {
  const kind = (k) => rec.tests.filter((t) => t.kind === k).map((t) => t.test).sort();
  assert.deepEqual(kind('design'), ['test_get_lr_cosine_schedule']);
  assert.deepEqual(kind('none'), ['test_checkpointing', 'test_encode_iterable_memory_usage',
    'test_encode_iterable_tinystories_matches_tiktoken', 'test_encode_iterable_tinystories_sample_roundtrip',
    'test_overlapping_special_tokens']);
  // the memory test is skipped on Windows; the other five fail in both variants
  assert.equal(byName.test_encode_iterable_memory_usage.as_shipped.outcome, 'skipped');
  for (const t of [...kind('design'), ...kind('none')].filter((n) => n !== 'test_encode_iterable_memory_usage')) {
    assert.equal(byName[t].as_shipped.outcome, 'failed', t);
    assert.equal(byName[t].one_line_fixes.outcome, 'failed', t);
  }
});
