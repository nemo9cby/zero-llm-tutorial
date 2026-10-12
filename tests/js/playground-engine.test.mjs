// The workbench engine (web/widgets/playground-engine.js) driven in-process, as the module worker drives it.
// It must load a checkpoint with progress, stream generation, honour cancel, and answer the addition problems
// exactly as PyTorch did with the same int8 weights (web/data/playground/addition.json, verify/_playground_addition.py).
//   node --test tests/js/playground-engine.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createEngine } from '../../web/widgets/playground-engine.js';
import { Tokenizer } from '../../web/widgets/_bpe.js';

const DIR = new URL('../../web/data/playground/', import.meta.url);
const fileFetch = async (url) => {
  try {
    return new Response(await readFile(fileURLToPath(url)));
  } catch {
    return new Response(null, { status: 404 });
  }
};
const tok = Tokenizer.fromJSON(JSON.parse(readFileSync(new URL('../../web/data/tokenizers/codebot.json', import.meta.url), 'utf8')));
const alpaca = (s) => `### Instruction:\n${s}\n\n### Response:\n`;
const record = JSON.parse(readFileSync(new URL('addition.json', DIR), 'utf8'));

function harness() {
  const replies = [];
  const waiters = new Map();
  const engine = createEngine({
    fetch: fileFetch,
    post: (msg) => {
      replies.push(msg);
      if (msg.kind === 'result' || msg.kind === 'error') waiters.get(msg.id)?.(msg);
    },
  });
  let seq = 0;
  const call = (type, payload) => {
    const id = ++seq;
    const done = new Promise((resolve) => waiters.set(id, resolve));
    engine.handle({ id, type, manifestURL: new URL(`${payload.name}/manifest.json`, DIR).href, ...payload });
    return { id, done, events: () => replies.filter((r) => r.id === id && r.kind === 'event') };
  };
  return { engine, call, replies };
}

test('load reports download progress and the backend, once per checkpoint', async () => {
  const { call, replies } = harness();
  const a = call('load', { name: 'grpo' });
  const b = call('load', { name: 'grpo' });
  const [ra, rb] = await Promise.all([a.done, b.done]);
  assert.equal(ra.kind, 'result');
  assert.equal(ra.backend, 'wasm-simd');
  assert.equal(ra.downloaded, 11_277_120);
  assert.equal(rb.downloaded, ra.downloaded);
  const progress = replies.filter((r) => r.kind === 'progress');
  assert.equal(progress.at(-1).loaded, 11_277_120);
  assert.equal(progress.filter((r) => r.loaded === 0).length, 1, 'downloaded once');
  assert.equal(ra.provenance.run, 'codebot-3060ti');
  assert.equal(ra.verified, true, 'every shard matches the SHA-256 in manifest.json');
});

test('forward: 6 x 6 attention maps, the logit lens, and the final top tokens tracked through the layers', async () => {
  const { call } = harness();
  const ids = tok.encode('def add(a, b):\n');
  const r = await call('forward', { name: 'pretrain', ids, attention: true, lens: 5 }).done;
  assert.equal(r.kind, 'result');
  assert.equal(r.logits.length, 1000);
  assert.equal(r.attention.length, 6 * 6 * ids.length * ids.length);
  assert.equal(r.lens.length, 7);
  assert.deepEqual(r.trackIds, r.lens[6].ids);
  assert.ok(Math.abs(r.track[6][0] - r.lens[6].probs[0]) < 1e-12);
  assert.equal(r.lens[0].ids[0], ids.at(-1), 'layer 0 (embedding only) predicts the current token itself');
});

test('generate streams one event per token and stops at <|endoftext|> like the book', async () => {
  const { call } = harness();
  const ids = tok.encode(alpaca('3+4='));
  const job = call('generate', { name: 'grpo', ids, opts: { temperature: 0, maxNewTokens: 16, top: 3 } });
  const r = await job.done;
  const ev = job.events();
  assert.equal(r.stopped, 'end');
  assert.equal(tok.decode(r.tokens.slice(0, -1)), '7');
  assert.equal(ev.length, r.tokens.length);
  assert.equal(ev.at(-1).token.end, true);
  assert.ok(ev.every((e) => e.ms >= 0 && e.token.top.length === 3));
});

test('cancel stops a long generation between two tokens', async () => {
  const { call, engine } = harness();
  const ids = tok.encode('def ');
  const job = call('generate', { name: 'pretrain', ids, opts: { temperature: 1, seed: 3, maxNewTokens: 200, stopAtEnd: false } });
  while (job.events().length < 3) await new Promise((res) => setTimeout(res, 1));
  engine.handle({ id: null, type: 'cancel', target: job.id });
  const r = await job.done;
  assert.equal(r.stopped, 'cancel');
  assert.ok(r.tokens.length < 200);
});

test('bench: decoding one token with the KV cache is cheaper than recomputing the window', async () => {
  const { call } = harness();
  const long = JSON.parse(readFileSync(new URL('pretrain/ref.json', DIR), 'utf8')).prompts.find((p) => p.name === 'long');
  const job = call('bench', { name: 'pretrain', ids: long.ids.concat(long.q8.greedy), lengths: [16, 256], repeats: 3 });
  await job.done;
  const rows = job.events();
  assert.deepEqual(rows.map((r) => r.n), [16, 256]);
  assert.ok(rows[1].prefillMs > 5 * rows[1].decodeMs, JSON.stringify(rows));
});

for (const name of ['sft', 'grpo']) {
  test(`${name}: every addition answer equals PyTorch with the same int8 weights (181 problems)`, async () => {
    const { call } = harness();
    const ref = record.models[`ours-${name}-q8`].results;
    for (const set of Object.keys(record.sets)) {
      const prompts = record.sets[set].map((p) => tok.encode(alpaca(p)));
      const job = call('addition', { name, prompts, maxNew: record.max_new_tokens });
      await job.done;
      const got = job.events();
      assert.equal(got.length, prompts.length);
      got.forEach((e) => {
        assert.deepEqual(e.ids, ref[set].ids[e.index], `${set} ${record.sets[set][e.index]}`);
        assert.equal(e.eos, ref[set].eos[e.index]);
      });
    }
  });
}

test('parity request: per-prompt results stream back, the long prompt can be skipped', async () => {
  const { call } = harness();
  const job = call('parity', { name: 'sft', refURL: new URL('sft/ref.json', DIR).href, skip: ['long'] });
  const r = await job.done;
  assert.deepEqual(r.results.map((x) => x.name), ['code', 'chat', 'math']);
  assert.ok(r.results.every((x) => x.maxDiffQ8 < 1e-4 && x.greedyQ8.firstMismatch === -1));
  assert.equal(job.events().length, 3);
});

test('errors come back as replies: unknown checkpoint, bad token id', async () => {
  const { call } = harness();
  const bad = await call('load', { name: 'nope' }).done;
  assert.equal(bad.kind, 'error');
  const r = await call('forward', { name: 'pretrain', ids: [5000] }).done;
  assert.equal(r.kind, 'error');
  assert.match(r.message, /token id/);
});
