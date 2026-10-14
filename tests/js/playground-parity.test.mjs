// Parity of web/widgets/_gpt.js with PyTorch on the workbench's real checkpoints.
// References: web/data/playground/<name>/ref.json, written by tools/export_codebot.py with the book's own
// codebot.model.GPT, once with exactly the int8 weights the browser rebuilds ("q8") and once with the checkpoint's
// float32 weights ("f32"). verify/test_playground.py re-derives those references with torch.
//   node --test tests/js/playground-parity.test.mjs          (about a minute on a laptop CPU)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { CodeBot, compareToReference, unpackTensors } from '../../web/widgets/_gpt.js';
import { Tokenizer } from '../../web/widgets/_bpe.js';
import { float32Forks, float32Verdict } from '../../web/widgets/playground-parity.js';

const DIR = new URL('../../web/data/playground/', import.meta.url);
const json = (rel) => JSON.parse(readFileSync(new URL(rel, DIR), 'utf8'));
const index = json('index.json').checkpoints;
const tok = Tokenizer.fromJSON(JSON.parse(readFileSync(new URL('../../web/data/tokenizers/codebot.json', import.meta.url), 'utf8')));

// Tolerances, measured on the three exports (see the report printed by the tests):
const TOL_Q8 = 1e-4;     // max |Δlogit| against PyTorch with the same weights: measured ≤ 1.5e-5 (float rounding at
                         // different places: summation order, float64 LayerNorm/softmax in JS, the erf approximation)
const TOL_ATTN = 1e-5;   // max |Δ| of the softmax weights: measured ≈ 2e-6 (ref.json keeps 6 decimals)
const TOL_LENS = 1e-4;   // max |Δp| of the logit lens at PyTorch's top-5 ids: measured ≈ 1e-6
const TOL_F32 = 0.5;     // the int8 rounding itself moves logits by up to ≈ 0.3 on these prompts

function loadExport(name) {
  const manifest = json(`${name}/manifest.json`);
  const parts = manifest.shards.map((s) => {
    const b = readFileSync(new URL(`${name}/${s.file}`, DIR));
    assert.equal(b.length, s.bytes, `${s.file} size`);
    assert.equal(createHash('sha256').update(b).digest('hex'), s.sha256, `${s.file} sha256`);
    return b;
  });
  const all = Buffer.concat(parts);
  assert.equal(createHash('sha256').update(all).digest('hex'), manifest.sha256);
  return { manifest, buffer: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength), ref: json(`${name}/ref.json`) };
}

test('figure P-8: the float32 sentence of the verdict tells forks before <|endoftext|> from forks after it', () => {
  assert.equal(float32Verdict({ early: 0, late: 0 }), '换成原始的 float32 权重，贪心生成的序列也全部相同。');
  assert.match(float32Verdict({ early: 2, late: 0 }), /^换成原始的 float32 权重时，有 2 个 prompt 贪心生成的序列在中途分岔/);
  assert.doesNotMatch(float32Verdict({ early: 2, late: 0 }), /之后分岔/);
  assert.match(float32Verdict({ early: 0, late: 1 }), /^换成原始的 float32 权重时，有 1 个 prompt 只在 <\|endoftext\|> 之后分岔/);
  assert.doesNotMatch(float32Verdict({ early: 0, late: 1 }), /中途分岔/);
  assert.match(float32Verdict({ early: 1, late: 2 }), /有 1 个 prompt 贪心生成的序列在中途分岔.*另有 2 个 prompt 只在/);
});

test('index.json lists pretrain, SFT and GRPO, and the folder stays under 36 MB', () => {
  assert.deepEqual(index.map((c) => c.name), ['pretrain', 'sft', 'grpo']);
  const total = index.reduce((a, c) => a + c.bytes, 0);
  assert.equal(total, 3 * 11_277_120);
  assert.ok(total < 36e6);
});

for (const { name } of index) {
  const { manifest, buffer, ref } = loadExport(name);

  test(`${name}: the manifest records where the weights came from`, () => {
    const p = manifest.provenance;
    assert.match(p.checkpoint, /model_(pretrain|sft|grpo)\.pt$/);
    assert.match(p.checkpoint_sha256, /^[0-9a-f]{64}$/);
    if (p.run) {                                     // the tutorial's own run: the producing step of the run report
      assert.equal(p.run_step.exit_code, 0);
      assert.match(p.run_step.script, /^ch03\/0[139]_/);
    }
    const w = unpackTensors(manifest, buffer);
    const params = [...w.values()].reduce((a, t) => a + t.length, 0);
    assert.equal(params, 11_121_640);               // ch02: unembed.weight is embed.weight, counted once
  });

  test(`${name} (WASM SIMD): logits, greedy tokens, attention and logit lens match PyTorch`, () => {
    const bot = new CodeBot(manifest, buffer, { backend: 'wasm-simd' });
    const rows = compareToReference(bot, ref);
    for (const r of rows) {
      const where = `${name}/${r.name}`;
      assert.ok(r.maxDiffQ8 < TOL_Q8, `${where}: max|Δlogit| ${r.maxDiffQ8}`);
      assert.equal(r.greedyQ8.firstMismatch, -1, `${where}: greedy tokens differ from PyTorch at ${r.greedyQ8.firstMismatch}`);
      assert.ok(r.lensTop1 && r.lensMaxDiff < TOL_LENS, `${where}: logit lens`);
      if (r.attentionMaxDiff != null) assert.ok(r.attentionMaxDiff < TOL_ATTN, `${where}: attention ${r.attentionMaxDiff}`);
      assert.ok(r.maxDiffF32 < TOL_F32, `${where}: int8 vs float32 ${r.maxDiffF32}`);
      console.log(`  ${where.padEnd(14)} max|Δlogit| ${r.maxDiffQ8.toExponential(1)} (int8) ${r.maxDiffF32.toFixed(3)} (float32); `
        + `greedy ${r.greedyQ8.same}/${r.greedyQ8.total} (int8) ${r.greedyF32.same}/${r.greedyF32.total} (float32 prefix)`);
    }
    // Figure P-8's verdict, counted again from ref.json alone: where PyTorch's int8 and float32 greedy sequences part,
    // before or after the float32 sequence's first <|endoftext|> (the book's generate() stops there).
    const expected = { early: 0, late: 0 };
    for (const p of ref.prompts) {
      const fork = p.q8.greedy.findIndex((id, i) => id !== p.f32.greedy[i]);
      const end = p.f32.greedy.indexOf(ref.end_token_id);
      if (fork >= 0) expected[end >= 0 && end < fork ? 'late' : 'early'] += 1;
    }
    assert.deepEqual(float32Forks(rows), expected);
    if (name !== 'pretrain') assert.equal(expected.early, 0, `${name}: forks only after <|endoftext|> (P.2.3 of the page)`);
  });

  test(`${name}: the page's tokenizer gives the ids codebot.tokenizer gave to PyTorch`, () => {
    for (const p of ref.prompts) assert.deepEqual(tok.encode(p.text), p.ids, p.name);
  });

  test(`${name}: on the real weights, prefill and step-by-step decoding are bit-identical (both backends)`, () => {
    const ids = ref.prompts.find((p) => p.name === 'chat').ids;
    for (const backend of ['wasm-simd', 'js']) {
      const bot = new CodeBot(manifest, buffer, { backend });
      bot.reset();
      const a = Float32Array.from(bot.prefill(ids));
      bot.reset();
      let b;
      for (const id of ids) b = bot.step(id);
      assert.deepEqual(Float32Array.from(b), a, backend);
    }
  });

  test(`${name} (plain JS): the short prompts match PyTorch too`, () => {
    const bot = new CodeBot(manifest, buffer, { backend: 'js' });
    const short = { ...ref, prompts: ref.prompts.filter((p) => p.name !== 'long') };
    for (const r of compareToReference(bot, short)) {
      assert.ok(r.maxDiffQ8 < TOL_Q8, `${name}/${r.name}: max|Δlogit| ${r.maxDiffQ8}`);
      assert.equal(r.greedyQ8.firstMismatch, -1, `${name}/${r.name}: greedy`);
      console.log(`  ${`${name}/${r.name}`.padEnd(14)} plain JS: max|Δlogit| ${r.maxDiffQ8.toExponential(1)} (int8); `
        + `greedy ${r.greedyQ8.same}/${r.greedyQ8.total} (int8)`);
    }
  });
}
