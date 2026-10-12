// Pure logic of the errata widgets: how index rows are classified, and the GRPO chart series.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { columnsOf, effectOf, partsOf } from '../../web/widgets/errata-board.js';
import { cpuSeries, evalPoints, gpuSeries, parseJsonl } from '../../web/widgets/errata-grpo.js';

const read = (rel) => fs.readFileSync(new URL(`../../${rel}`, import.meta.url), 'utf8');

test('“影响结果” cells: errors are their own bucket; only a plain “否” or “几乎没有” counts as no effect', () => {
  for (const t of ['否', '否（书中流程不触发）', '否（曲线缺了起点）', '否（梯度相同，只是多通信）', '几乎没有']) {
    assert.equal(effectOf(t), 'no', t);
  }
  for (const t of ['否（只在使用下载的权重时报错）', '否（无法运行）']) assert.equal(effectOf(t), 'error', t);
  for (const t of ['是', '仅 Windows', '仅长生成', '部分', '未测量', '仅多卡', '仅中文 Windows']) {
    assert.equal(effectOf(t), 'yes', t);
  }
});

test('locations map to the book parts', () => {
  assert.deepEqual(partsOf('ch03/09_grpo.py'), ['I']);
  assert.deepEqual(partsOf('storybot/model.py；ch06/06–09'), ['II']);
  assert.deepEqual(partsOf('ch06/05、ch09/01'), ['II', 'III']);
  assert.deepEqual(partsOf('storybot、webbot 的 model.py'), ['II', 'III']);
  assert.deepEqual(partsOf('README.md'), ['repo']);
});

test('columns are found by their header text, whatever the order', () => {
  assert.deepEqual(columnsOf(['编号', '严重度', '影响结果', '问题', '位置']), { severity: 1, effect: 2, location: 4 });
  assert.deepEqual(columnsOf(['编号', '问题', '位置', '严重度', '影响结果']), { severity: 3, effect: 4, location: 2 });
  assert.throws(() => columnsOf(['编号', '问题']));
});

test('every row of the errata index is classified, anchored and numbered in order', () => {
  const md = read('content/errata.md');
  const lines = md.split('\n');
  const header = lines.find((l) => l.startsWith('| 编号 | 严重度'));
  const split = (row) => row.split(/(?<!\\)\|/).map((c) => c.trim()).slice(1, -1);
  const col = columnsOf(split(header));
  const rows = lines.filter((l) => /^\| \[E\d\d\]/.test(l));
  const stated = Number(md.match(/整理出了 (\d+) 个问题/)[1]);       // the count the lede promises
  assert.equal(rows.length, stated);
  assert.ok(md.includes(`\n${stated} 个问题里`), 'the summary repeats the count');
  const ids = rows.map((r) => r.match(/^\| \[(E\d\d)\]/)[1]);
  assert.deepEqual(ids, Array.from({ length: stated }, (_, i) => `E${String(i + 1).padStart(2, '0')}`));
  for (const id of ids) assert.ok(md.includes(`id="${id.toLowerCase()}"`), `anchor for ${id}`);
  const effects = new Set();
  for (const row of rows) {
    const cells = split(row);
    assert.ok(partsOf(cells[col.location]).length > 0, cells[0]);
    assert.ok(['高', '中', '低'].includes(cells[col.severity]), cells[0]);
    effects.add(effectOf(cells[col.effect]));
  }
  assert.deepEqual([...effects].sort(), ['error', 'no', 'yes']);   // every filter bucket has rows
});

test('GPU chart series: evaluations after iterations 1, 11, …, 491 of the two recorded 500-iteration runs', () => {
  const base = parseJsonl(read('web/data/runs/codebot-3060ti/09_grpo.metrics.jsonl'));
  const fix = parseJsonl(read('web/data/runs/codebot-grpo-evalfix-3060ti/09_grpo.metrics.jsonl'));
  const [b, f] = gpuSeries(base, fix);
  assert.deepEqual([b.name, f.name], ['原样', '加 eval()']);
  assert.equal(b.points.length, 50);
  assert.deepEqual(b.points[0], [1, 53.1]);
  assert.deepEqual(f.points[0], [1, 56.8]);
  assert.deepEqual([b.points[5], f.points[5]], [[51, 60.5], [51, 98.8]]);
  assert.deepEqual([b.points.at(-1), f.points.at(-1)], [[491, 86.4], [491, 100]]);
  assert.deepEqual(evalPoints([{ it: 0, acc: 1 }, { it: 1, acc: 2 }, { it: 2, acc: 3 }, { it: 11, acc: 4 }]), [[1, 2], [11, 4]]);
});

test('CPU chart series come from the recorded research runs, shifted to completed iterations', () => {
  const data = JSON.parse(read('web/data/errata/grpo_dropout_runs.json'));
  const [base, nodrop] = cpuSeries(data, 'cpu0');
  assert.equal(base.name, '原样');
  assert.deepEqual(base.points[0], [1, 55.56]);       // logged as "iteration 0", after two updates (E15)
  assert.deepEqual(base.points.at(-1), [60, 70.37]);
  assert.deepEqual(nodrop.points.at(-1), [60, 95.06]);
  const [b1, n1] = cpuSeries(data, 'cpu1');
  assert.deepEqual([b1.points.at(-1)[1], n1.points.at(-1)[1]], [74.07, 95.06]);
});
