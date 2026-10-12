// Pure logic of the glossary widgets: the config translator and the table search normalisation.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { translate, blockParams, dlfs6Call } from '../../web/widgets/glossary-config.js';
import { normalize } from '../../web/widgets/glossary-search.js';

const src = JSON.parse(fs.readFileSync(new URL('../../web/data/glossary/sources.json', import.meta.url), 'utf8'));
const value = (rows, key) => rows.find((r) => r.key === key).value;

test('GPT-2 config: n_embd is the residual width, n_inner defaults to 4 x n_embd', () => {
  const { rows, derived } = translate(src.hf.gpt2.config);
  assert.equal(value(rows, 'width'), 768);
  assert.equal(value(rows, 'ctx'), 1024);
  assert.equal(value(rows, 'ff'), 3072);
  assert.equal(value(rows, 'kv'), 12);
  assert.equal(derived.headDim, 64);
  // 12 GPT-2 blocks without biases: (4·d² + 2·d·4d + 2d) · 12 — known ≈ 85M non-embedding parameters
  assert.equal(derived.blocks, 84_953_088);
});

test('OLMo-2-0425-1B: hidden_size 2048, MHA (16 KV heads), SwiGLU with 3 matrices', () => {
  const { rows, derived } = translate(src.hf.olmo2_0425_1b.config);
  assert.equal(value(rows, 'width'), 2048);
  assert.equal(value(rows, 'kv'), 16);
  assert.equal(derived.groupSize, 1);
  assert.equal(derived.blocks, 16 * (2 * 2048 ** 2 + 2 * 2048 * 2048 + 3 * 2048 * 8192 + 2 * 2048));
});

test('Qwen2.5-Math-1.5B: GQA with 6 query heads per KV head', () => {
  const { derived } = translate(src.hf.qwen25_math_1p5b.config);
  assert.equal(derived.headDim, 128);
  assert.equal(derived.groupSize, 6);
});

test('block counts reproduce the DLFS6 bots exactly', () => {
  const story = { intermediate_size: 1344 };
  assert.equal(4 * blockParams(story, 512, 16, 16, 1344), 12_455_936);      // StoryBot blocks
  assert.equal(blockParams({ intermediate_size: 2048 }, 768, 12, 4, 2048), 6_292_992); // one WebBot block
});

test('the DLFS6 constructor call uses the WebBot signature only when there is GQA', () => {
  const web = translate({ vocab_size: 50000, max_position_embeddings: 1024, hidden_size: 768, num_attention_heads: 12,
    num_key_value_heads: 4, num_hidden_layers: 12, intermediate_size: 2048, rope_theta: 10000 });
  assert.match(dlfs6Call(web.rows), /webbot\/model\.py[\s\S]*n_kv_head=4/);
  const story = translate({ vocab_size: 10000, max_position_embeddings: 256, hidden_size: 512, num_attention_heads: 16,
    num_hidden_layers: 4, intermediate_size: 1344, rope_theta: 10000 });
  assert.match(dlfs6Call(story.rows), /storybot\/model\.py[\s\S]*GPT\(vocab_size=10000, max_context_len=256, embed_dim=512, n_head=16, n_layer=4, ff_dim=1344, theta=10000\)/);
});

test('search normalisation ignores case, spaces, underscores and hyphens', () => {
  assert.equal(normalize('Hidden_Size'), normalize('hidden size'));
  assert.equal(normalize('hidden-size'), 'hiddensize');
  assert.ok(normalize('critic/rewards/mean').includes(normalize('Critic')));
});
