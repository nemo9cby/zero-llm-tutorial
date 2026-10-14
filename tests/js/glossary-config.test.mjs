// Pure logic of the glossary widgets: the config translator and the table search normalisation.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { translate, blockParams, dlfs6Call, roughNote, PRESETS } from '../../web/widgets/glossary-config.js';
import { normalize, findMatches } from '../../web/widgets/glossary-search.js';

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

test('search matches section and box titles as well as rows', () => {
  const items = ['probe', '“先 probe 一下”到底指什么？', 'Y.3 强化学习训练指标对照', 'global batch size (gbs)']
    .map((t) => ({ title: t, text: normalize(t) }));
  assert.deepEqual(findMatches(items, 'PROBE').map((s) => s.title), ['probe', '“先 probe 一下”到底指什么？']);
  assert.deepEqual(findMatches(items, 'gbs').map((s) => s.title), ['global batch size (gbs)']);
  assert.equal(findMatches(items, '').length, items.length);
  assert.equal(findMatches(items, 'rollout').length, 0);
});

test('Qwen3-0.6B: the stated head_dim (128) is used, and DLFS6 cannot express it', () => {
  const cfg = src.hf.qwen3_0p6b.config;
  const { rows, derived } = translate(cfg);
  assert.equal(value(rows, 'headDim'), 128);
  assert.equal(derived.headDim, 128);
  assert.equal(derived.headDimMismatch, 64);                       // hidden_size / heads = 1024 / 16
  // q 1024×2048, k and v 1024×1024, o 2048×1024, SwiGLU 3×1024×3072, two norms (q/k norms are not modelled)
  assert.equal(derived.blocks, 28 * (2 * 1024 * 2048 + 2 * 1024 * 1024 + 3 * 1024 * 3072 + 2 * 1024));
  assert.equal(derived.blocks, 440_459_264);
  assert.match(dlfs6Call(rows, cfg), /写不出这个模型[\s\S]*得到 64[\s\S]*head_dim = 128/);
  assert.doesNotMatch(dlfs6Call(rows, cfg), /GPT\(/);
});

test('configs without head_dim: D = hidden_size / heads, and the block count is unchanged', () => {
  for (const cfg of [src.hf.gpt2.config, src.hf.olmo2_0425_1b.config, src.hf.qwen25_math_1p5b.config]) {
    const { rows, derived } = translate(cfg);
    assert.equal(derived.headDimMismatch, undefined);
    assert.equal(value(rows, 'headDim'), value(rows, 'width') / value(rows, 'heads'));
  }
});

test('the rough 12·L·d² is explained when it is more than 5 % off', () => {
  const olmo = translate(src.hf.olmo2_0425_1b.config).derived;
  assert.equal(olmo.blocks, 1_073_807_360);
  assert.equal(olmo.roughNonEmbedding, 805_306_368);
  assert.match(roughNote(olmo), /每层约 16d²：Attention 4d²，FFN 12d²（3 个矩阵），所以粗估低了约 25%/);
  const qwen = translate(src.hf.qwen25_math_1p5b.config).derived;
  assert.match(roughNote(qwen), /Attention 2\.33d²，FFN 17\.5d²/);
  assert.equal(roughNote(translate(src.hf.gpt2.config).derived), null);           // 0.02 % off
  const story = translate({ vocab_size: 10000, max_position_embeddings: 256, hidden_size: 512, num_attention_heads: 16,
    num_hidden_layers: 4, intermediate_size: 1344, rope_theta: 10000 }).derived;
  assert.equal(roughNote(story), null);                                             // 1 % off
});

test('every generated DLFS6 call passes exactly the constructor arguments of its module', () => {
  const sigs = src.dlfs6.signatures;
  let checked = 0;
  for (const p of PRESETS(src)) {
    const call = dlfs6Call(translate(p.cfg).rows, p.cfg);
    const m = call.match(/^# (\w+)\/model\.py 的写法\nGPT\((.*)\)$/);
    if (!m) continue;                                                // a config DLFS6 cannot express (Qwen3)
    const args = [...m[2].matchAll(/(\w+)=/g)].map((x) => x[1]);
    assert.deepEqual(args, sigs[`${m[1]}/model.py GPT.__init__`], p.id);
    assert.doesNotMatch(m[2], /=\?/, p.id);                          // no missing values
    checked += 1;
  }
  assert.equal(checked, PRESETS(src).length - 1);
  assert.match(dlfs6Call(translate(src.hf.gpt2.config).rows, src.hf.gpt2.config), /dropout_rate=0\.1\)$/);
});
