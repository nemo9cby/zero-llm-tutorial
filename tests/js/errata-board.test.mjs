// Pure logic of the errata widgets: how index rows are classified, and the GRPO chart series.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { effectOf, partsOf } from '../../web/widgets/errata-board.js';
import { seriesFor } from '../../web/widgets/errata-grpo.js';

test('“影响结果” cells: only a plain “否” or “几乎没有” counts as no effect', () => {
  for (const t of ['否', '否（直接报错）', '否（无法运行）', '几乎没有']) assert.equal(effectOf(t), 'no', t);
  for (const t of ['是', '仅 Windows', '仅长生成', '部分', '未测量', '曲线偏一格']) assert.equal(effectOf(t), 'yes', t);
});

test('locations map to the book parts', () => {
  assert.deepEqual(partsOf('ch03/09_grpo.py'), ['I']);
  assert.deepEqual(partsOf('storybot/model.py；ch06/06–09'), ['II']);
  assert.deepEqual(partsOf('ch06/05、ch09/01'), ['II', 'III']);
  assert.deepEqual(partsOf('README.md'), ['repo']);
});

test('every row of the errata index is classified (no orphan locations)', () => {
  const md = fs.readFileSync(new URL('../../content/errata.md', import.meta.url), 'utf8');
  const rows = md.split('\n').filter((l) => /^\| \[E\d\d\]/.test(l));
  assert.equal(rows.length, 42);
  for (const row of rows) {
    const cells = row.split(/(?<!\\)\|/).map((c) => c.trim());   // ['', id, title, location, severity, effect, '']
    assert.ok(partsOf(cells[3]).length > 0, cells[1]);
    assert.ok(['高', '中', '低'].includes(cells[4]), cells[1]);
  }
});

test('GRPO chart series come from the recorded runs', () => {
  const data = JSON.parse(fs.readFileSync(new URL('../../web/data/errata/grpo_dropout_runs.json', import.meta.url), 'utf8'));
  const [base, nodrop] = seriesFor(data, 0);
  assert.equal(base.name, '原样');
  assert.deepEqual(base.points.at(-1), [59, 70.37]);
  assert.deepEqual(nodrop.points.at(-1), [59, 95.06]);
  const [b1, n1] = seriesFor(data, 1);
  assert.deepEqual([b1.points.at(-1)[1], n1.points.at(-1)[1]], [74.07, 95.06]);
});
