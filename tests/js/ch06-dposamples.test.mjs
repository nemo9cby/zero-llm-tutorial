// 第 6 章 DPO 前后的故事对照（ch06-dposamples）：分组、说明文字与记录 data/ch06/h100_dpo.json 一致。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { openings, storyMeta, shortLabel } from '../../web/widgets/ch06-dposamples.js';

const rec = JSON.parse(readFileSync(new URL('../../web/data/ch06/h100_dpo.json', import.meta.url), 'utf8'));

test('按开头分组：8 条留出开头各 3 个种子，"Once upon a time" 8 个种子，顺序与记录相同', () => {
  const groups = openings(rec);
  assert.equal(groups.length, 9);
  const held = groups.filter((g) => g.kind === 'heldout');
  assert.equal(held.length, 8);
  assert.ok(held.every((g) => g.rows.map((r) => r.seed).join() === '1,2,3'));
  const open = groups.find((g) => g.kind === 'open');
  assert.equal(open.prompt, 'Once upon a time');
  assert.deepEqual(open.rows.map((r) => r.seed), [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(groups.flatMap((g) => g.rows).length, rec.stories.length);
});

test('每个故事的说明：自然结束与写满上限，与验证 6.8-i 的计数一致', () => {
  const all = rec.stories;
  const ended = (name) => all.filter((s) => s[name].ended).length;
  assert.equal(ended('pretrain'), 30);
  assert.equal(ended('dpo'), 32);
  const cut = all.find((s) => !s.pretrain.ended).pretrain;
  assert.equal(storyMeta(cut), `写满了上限 ${cut.max_new_tokens} 个 token，没有写完`);
  const done = all[0].dpo;
  assert.ok(done.ended && done.new_tokens < done.max_new_tokens);
  assert.equal(storyMeta(done), `新写了 ${done.new_tokens} 个 token，写到 <|endoftext|> 自然结束`);
});

test('下拉框的短标签', () => {
  assert.equal(shortLabel('Once upon a time, there was a little bird who could not fly. '), 'a little bird who could not fly');
  assert.equal(shortLabel('Once upon a time'), 'Once upon a time');
});
