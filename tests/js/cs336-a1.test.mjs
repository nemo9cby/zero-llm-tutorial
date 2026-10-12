// Pure logic of the CS336 A1 results widget, checked against the recorded run it displays.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { everFailed, ratioPos, summarize } from '../../web/widgets/cs336-a1.js';

const rec = JSON.parse(fs.readFileSync(new URL('../../web/data/cs336/a1_results.json', import.meta.url), 'utf8'));

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

test('“失败过的” lists the 10 as-shipped failures plus the speed test that only fails after the fix', () => {
  const failed = rec.tests.filter(everFailed).map((t) => t.test);
  assert.equal(failed.length, rec.summary.as_shipped.failed + 1);
  assert.equal(failed.length, 11);
  const onlyAfterFix = rec.tests.filter((t) => t.as_shipped.outcome !== 'failed' && everFailed(t)).map((t) => t.test);
  assert.deepEqual(onlyAfterFix, ['test_train_bpe_speed']);
});
