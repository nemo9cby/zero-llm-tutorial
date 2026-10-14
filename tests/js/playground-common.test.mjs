// web/widgets/playground-common.js: the next-token candidate list that playground-next and playground-bench show.
// At temperature 0 the distribution is one-hot, so ordering by probability alone ties every other token at 0 and the
// list used to fill with the lowest ids. The runners-up are ordered by logit when probabilities tie.
//   node --test tests/js/playground-common.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { candidateOrder, candidateBars, statusLine, ensureModel } from '../../web/widgets/playground-common.js';
import { filterDistribution, softmax, argsortDesc } from '../../web/widgets/_gpt.js';
import { Tokenizer } from '../../web/widgets/_bpe.js';

// A fake DOM, just enough for _lib.js's h(): elements with attributes and children, and text nodes.
class FakeNode {
  constructor() { this.childNodes = []; }
  append(...nodes) { this.childNodes.push(...nodes); }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(''); }
}
class FakeText extends FakeNode {
  constructor(t) { super(); this.data = t; }
  get textContent() { return this.data; }
}
class FakeElement extends FakeNode {
  constructor(tag) { super(); this.tagName = tag; this.attrs = {}; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; }
  get className() { return this.attrs.class || ''; }
  set className(v) { this.attrs.class = v; }
  addEventListener() {}
}
globalThis.Node = FakeNode;
globalThis.document = { createElement: (t) => new FakeElement(t), createTextNode: (t) => new FakeText(t) };

const tok = Tokenizer.fromJSON(JSON.parse(readFileSync(new URL('../../web/data/tokenizers/codebot.json', import.meta.url), 'utf8')));
const ref = (name) => JSON.parse(readFileSync(new URL(`../../web/data/playground/${name}/ref.json`, import.meta.url), 'utf8'));

/** The rows of a candidateBars grid: [{id, label, off, width, num}]. */
function rows(grid) {
  const kids = grid.childNodes, out = [];
  for (let i = 0; i < kids.length; i += 3) {
    const [lab, bar, num] = kids.slice(i, i + 3);
    out.push({
      id: Number(lab.getAttribute('title').replace('id ', '')), label: lab.textContent,
      off: lab.className.includes('off-lab'), width: bar.getAttribute('style'), num: num.textContent,
    });
  }
  return out;
}

test('candidateOrder: most probable first; ties are broken by logit, then by id', () => {
  const logits = [0.5, 3, -1, 2, 2, 0];
  const oneHot = filterDistribution(logits, { temperature: 0 }).probs;
  assert.deepEqual(candidateOrder(logits, oneHot, 6), [1, 3, 4, 0, 5, 2]);
  assert.deepEqual(argsortDesc(oneHot).slice(0, 6), [1, 0, 2, 3, 4, 5], 'what ordering by probability alone gave');
  // no ties: the same order as by probability
  const p = softmax(logits, 0.8);
  assert.deepEqual(candidateOrder(logits, p, 4), argsortDesc(p).slice(0, 4));
  // a tiny temperature underflows the tail to exactly 0: the tie is broken by logit too
  const under = softmax([1, 5, 4, 3, 2], 0.001);
  assert.deepEqual([...under], [0, 1, 0, 0, 0]);
  assert.deepEqual(candidateOrder([1, 5, 4, 3, 2], under, 5), [1, 2, 3, 4, 0]);
  assert.deepEqual(candidateOrder(logits, oneHot, 2), [1, 3], 'n limits the list');
});

for (const name of ['pretrain', 'sft', 'grpo']) {
  test(`${name}: at temperature 0 the runners-up are the next tokens by logit, shown at 0% (real logits, ref.json)`, () => {
    for (const prompt of ref(name).prompts) {
      const logits = Float32Array.from(prompt.q8.logits);
      const { grid } = candidateBars(tok, logits, { temperature: 0, topK: 0, topP: 1 }, 15);
      const r = rows(grid);
      assert.deepEqual(r.map((x) => x.id), argsortDesc(logits).slice(0, 15), prompt.name);
      assert.equal(r[0].id, prompt.q8.greedy[0], 'the first candidate is the greedy token');
      assert.deepEqual(r.map((x) => [x.off, x.num, x.width]),
        [[false, '100.0%', 'width:100%'], ...Array(14).fill([true, '0%', 'width:0.5%'])], 'probabilities as before');
    }
  });
}

test('at temperature > 0 the list and its numbers are unchanged: raw probabilities, kept ones with their share', () => {
  const logits = Float32Array.from(ref('sft').prompts[0].q8.logits);
  const s = { temperature: 0.8, topK: 5, topP: 0.95 };
  const { grid, summary } = candidateBars(tok, logits, s, 8);
  const raw = softmax(logits, 0.8), dist = filterDistribution(logits, s), kept = new Set(dist.kept);
  const order = argsortDesc(raw).slice(0, 8);
  const r = rows(grid);
  assert.deepEqual(r.map((x) => x.id), order);
  r.forEach((x, i) => {
    const on = kept.has(order[i]);
    assert.equal(x.off, !on);
    assert.equal(x.num, on ? `${(dist.probs[order[i]] * 100).toFixed(1)}%` : '0%');
    assert.equal(x.width, `width:${Math.max(0.5, (raw[order[i]] / raw[order[0]]) * 100)}%`);
  });
  assert.match(summary, new RegExp(`^保留 ${dist.kept.length} 个 token；`));
});

test('the status line is aria-busy while a checkpoint loads (check_pages waits for it), and not after', () => {
  const s = statusLine();
  assert.equal(s.el.getAttribute('aria-busy'), null);
  s.busy(true);
  s.busy(true);                                      // two loads at once (a quick switch of checkpoint)
  assert.equal(s.el.getAttribute('aria-busy'), 'true');
  s.busy(false);
  assert.equal(s.el.getAttribute('aria-busy'), 'true', 'still one load in flight');
  s.busy(false);
  assert.equal(s.el.getAttribute('aria-busy'), null);
  s.busy(false);                                     // never below zero
  s.busy(true);
  assert.equal(s.el.getAttribute('aria-busy'), 'true');
});

test('ensureModel clears aria-busy when the load fails', async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error('offline'); };      // the checkpoint list cannot be read
  try {
    const s = statusLine();
    const job = ensureModel('pretrain', s);
    assert.equal(s.el.getAttribute('aria-busy'), 'true');
    await assert.rejects(job, /offline/);
    assert.equal(s.el.getAttribute('aria-busy'), null);
  } finally {
    globalThis.fetch = realFetch;
  }
});
