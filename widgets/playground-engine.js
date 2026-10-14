/* The workbench's compute side (content/playground.md). It owns the CodeBot instances of _gpt.js and answers
   requests; it never touches the DOM or the tokenizer (ids in, ids out).

   It runs inside playground-worker.js (a module Web Worker). playground-common.js falls back to running it on
   the page when module workers are unavailable, and tests/js/playground-engine.test.mjs drives it in Node.

   createEngine({ post, fetch }) -> { handle(msg) }      post(reply, transfer?) sends a reply to the page
   Every request is {id, type, ...}; every reply carries the request id and a kind:
     'progress' (broadcast, name instead of id)  'event' (streamed items)  'result' (last)  'error' (last)
   Requests:
     load      {name, manifestURL}                  -> progress*, result {name, backend, cfg, bytes, ms, provenance,
                                                       verified: true once the shards' SHA-256 match, null without
                                                       crypto.subtle (plain-http pages); a mismatch is an error}
     forward   {name, ids, attention?, lens?, track?} -> result {logits, attention, n, lens, trackIds, track, ms}
                 lens = k: top-k of the logit lens per layer; track[l][j] = lens probability at layer l of
                 trackIds[j] (the final top-k, then the extra ids asked for in track)
     generate  {name, ids, opts}                    -> event {token: {...}, ms}*, result {tokens, ms, stopped}
                 opts as in _gpt.js generate() (temperature, topK, topP, maxNewTokens, seed, stopAtEnd, cache)
     cancel    {target}                             stops a running generate/bench/addition between two tokens
     bench     {name, ids, lengths}                 -> event {n, decodeMs, prefillMs}*, result {}
     parity    {name, refURL, backend, skip}        -> event {result}*, result {results}
     addition  {name, prompts: [ids...], maxNew}    -> event {index, ids, eos}*, result {}
*/
import { fetchCheckpoint, CodeBot, generate, compareToReference } from './_gpt.js';

const now = () => (globalThis.performance ?? Date).now();

/** Yield to the event loop, so that cancel messages and other requests get a turn. A MessageChannel round trip is
    fast but can starve timers (Node drains a port's queue in one go), so at least every 50 ms it is a setTimeout. */
const tick = (() => {
  const slow = () => new Promise((resolve) => setTimeout(resolve, 0));
  if (typeof MessageChannel !== 'function') return slow;
  const ch = new MessageChannel();
  const queue = [];
  // Node: an idle port must not keep the process alive, a pending tick must (ref/unref do not exist in browsers)
  ch.port1.onmessage = () => {
    queue.shift()?.();
    if (!queue.length) ch.port1.unref?.();
  };
  ch.port1.unref?.();
  let last = now();
  return () => {
    if (now() - last > 50) { last = now(); return slow(); }
    return new Promise((resolve) => { queue.push(resolve); ch.port1.ref?.(); ch.port2.postMessage(0); });
  };
})();
const median = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];

export function createEngine({ post, fetch: get = globalThis.fetch }) {
  const models = new Map();       // name -> Promise<{bot, manifest, buffer, info}>
  const locks = new Map();        // name -> tail of that model's queue (a CodeBot has one KV cache)
  const active = new Set();       // ids of requests in flight
  const cancelled = new Set();

  const reply = (id, kind, data = {}, transfer) => post({ id, kind, ...data }, transfer);

  function exclusive(name, fn) {
    const prev = locks.get(name) || Promise.resolve();
    const run = prev.then(fn, fn);
    locks.set(name, run.catch(() => {}));
    return run;
  }

  function load(name, manifestURL) {
    if (!models.has(name)) {
      const job = (async () => {
        if (!manifestURL) throw new Error(`checkpoint ${name} 还没有加载`);
        const t0 = now();
        const { manifest, buffer, verified } = await fetchCheckpoint(manifestURL, {
          fetch: get, verify: true, onProgress: (loaded, total) => post({ id: null, kind: 'progress', name, loaded, total }),
        });
        const bot = new CodeBot(manifest, buffer);
        const info = { name, backend: bot.backend, cfg: bot.cfg, bytes: bot.bytes, downloaded: manifest.total_bytes, verified,
          ms: now() - t0, provenance: manifest.provenance, label: manifest.label, endId: bot.endId };
        return { bot, manifest, buffer, info, js: null };
      })();
      models.set(name, job);
      job.catch(() => models.delete(name));      // a failed download can be retried
    }
    return models.get(name);
  }

  // lens = k: top-k of the logit lens per layer, plus the probability, at every layer, of the k tokens the model
  // finally predicts (trackIds). track = extra ids to follow through the layers.
  async function forward(m, { ids, attention = false, lens = 0, track = [] }) {
    const { L } = m.bot.cfg;
    if (attention && ids.length > 128) throw new RangeError('注意力图最多显示 128 个 token');
    const t0 = now();
    const out = m.bot.forward(ids, { attention, lens: lens || (track.length ? 1 : 0) });
    const trackIds = [...(lens ? out.lens[L].ids : []), ...track.filter((t) => !(lens && out.lens[L].ids.includes(t)))];
    const tracked = trackIds.length ? Array.from({ length: L + 1 }, (_, l) => {
      const p = m.bot.lensDistribution(out.resid, l);
      return trackIds.map((t) => p[t]);
    }) : null;
    const res = { logits: out.logits, attention: out.attention, n: out.n, lens: lens ? out.lens : null,
      trackIds, track: tracked, ms: now() - t0 };
    return { res, transfer: [out.logits.buffer, ...(out.attention ? [out.attention.buffer] : [])] };
  }

  async function runGenerate(id, m, { ids, opts = {} }) {
    const it = generate(m.bot, ids, opts);
    const tokens = [];
    const t0 = now();
    let stopped = 'length';
    for (;;) {
      if (cancelled.has(id)) { stopped = 'cancel'; break; }
      const s = now();
      const { value, done } = it.next();
      if (done) break;
      const ms = now() - s;
      tokens.push(value.id);
      reply(id, 'event', { token: value, ms });
      if (value.end) { stopped = 'end'; break; }
      await tick();
    }
    return { tokens, ms: now() - t0, stopped };
  }

  async function bench(id, m, { ids, lengths, repeats = 5 }) {
    const bot = m.bot;
    for (const n of lengths) {
      if (cancelled.has(id)) break;
      if (n < 2 || n > ids.length || n > bot.cfg.T) throw new RangeError(`长度 ${n} 超出范围`);
      bot.reset();
      bot.prefill(ids.slice(0, n - 1));          // the cache holds n - 1 tokens
      const decode = [];
      for (let r = 0; r < repeats; r++) {         // one new token at position n - 1; rewinding rewrites the same row
        bot.pos = n - 1;
        const s = now();
        bot.step(ids[n - 1]);
        decode.push(now() - s);
      }
      await tick();
      const prefill = [];
      for (let r = 0; r < (n <= 64 ? 3 : 1); r++) { // no cache: the whole window again for one new token
        bot.reset();
        const s = now();
        bot.prefill(ids.slice(0, n));
        prefill.push(now() - s);
      }
      reply(id, 'event', { n, decodeMs: median(decode), prefillMs: median(prefill) });
      await tick();
    }
    bot.reset();
    return {};
  }

  async function parity(id, m, { refURL, backend = 'auto', skip = [] }) {
    const r = await get(refURL);
    if (!r.ok) throw new Error(`读取参考数据失败（HTTP ${r.status}）`);
    const ref = await r.json();
    let bot = m.bot;
    if (backend === 'js' && bot.backend !== 'js') {
      m.js ??= new CodeBot(m.manifest, m.buffer, { backend: 'js' });
      bot = m.js;
    }
    const results = [];
    for (const p of ref.prompts) {
      if (cancelled.has(id)) break;
      if (skip.includes(p.name)) continue;
      const [res] = compareToReference(bot, { ...ref, prompts: [p] });
      res.backend = bot.backend;
      results.push(res);
      reply(id, 'event', { result: res });
      await tick();
    }
    return { results, backend: bot.backend };
  }

  async function addition(id, m, { prompts, maxNew = 16 }) {
    for (let i = 0; i < prompts.length; i++) {
      if (cancelled.has(id)) break;
      const out = [];
      let eos = false;
      for (const tok of generate(m.bot, prompts[i], { temperature: 0, maxNewTokens: maxNew, top: 0 })) {
        if (tok.end) { eos = true; break; }
        out.push(tok.id);
      }
      reply(id, 'event', { index: i, ids: out, eos });
      if (i % 4 === 3) await tick();
    }
    return {};
  }

  async function handle(msg) {
    const { id, type } = msg;
    if (type === 'cancel') {
      if (active.has(msg.target)) cancelled.add(msg.target);
      return;
    }
    active.add(id);
    try {
      if (type === 'load') {
        const m = await load(msg.name, msg.manifestURL);
        reply(id, 'result', m.info);
        return;
      }
      const m = await load(msg.name, msg.manifestURL);
      const out = await exclusive(msg.name, async () => {
        if (type === 'forward') return forward(m, msg);
        if (type === 'generate') return { res: await runGenerate(id, m, msg) };
        if (type === 'bench') return { res: await bench(id, m, msg) };
        if (type === 'parity') return { res: await parity(id, m, msg) };
        if (type === 'addition') return { res: await addition(id, m, msg) };
        throw new Error(`未知的请求类型 ${type}`);
      });
      reply(id, 'result', out.res, out.transfer);
    } catch (err) {
      reply(id, 'error', { message: err?.message || String(err) });
    } finally {
      active.delete(id);
      cancelled.delete(id);
    }
  }

  return { handle };
}
