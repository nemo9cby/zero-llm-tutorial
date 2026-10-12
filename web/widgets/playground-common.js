/* Shared by the workbench widgets of content/playground.md (not a widget itself: it has no mount()).
   One engine per page: every widget talks to the same module Web Worker (playground-worker.js), so a checkpoint
   is downloaded once however many figures use it. Without module workers, the engine runs on the page instead. */
import { h, visible } from './_lib.js';
import { Tokenizer } from './_bpe.js';
import { filterDistribution, softmax, argsortDesc, entropy } from './_gpt.js';

export const DATA = new URL('../data/playground/', import.meta.url);
const TOKENIZER = new URL('../data/tokenizers/codebot.json', import.meta.url);

/** The chat template of ch03 (03_sft.py, 04_chat.py, 09_grpo.py). */
export const alpaca = (instruction) => `### Instruction:\n${instruction}\n\n### Response:\n`;

/** 09_grpo.py's calculate_reward: the last integer in the text equals the ground truth. */
export function reward(text, gt) {
  const m = text.match(/-?\d+/g);
  return m ? Number.parseInt(m[m.length - 1], 10) === gt : false;
}

let indexPromise = null;
/** web/data/playground/index.json: [{name, label, description, template, manifest, ref, bytes}] in switch order. */
export function getIndex() {
  indexPromise ??= fetch(new URL('index.json', DATA)).then((r) => {
    if (!r.ok) throw new Error(`读取检查点列表失败（HTTP ${r.status}）`);
    return r.json();
  }).then((j) => j.checkpoints);
  return indexPromise;
}

let tokPromise = null;
export function getTokenizer() {
  tokPromise ??= Tokenizer.load(TOKENIZER.href);
  return tokPromise;
}

const refCache = new Map();
/** ref.json of a checkpoint (the PyTorch reference written by tools/export_codebot.py). */
export async function getRef(name) {
  if (!refCache.has(name)) {
    refCache.set(name, (async () => {
      const entry = (await getIndex()).find((c) => c.name === name);
      const r = await fetch(new URL(entry.ref, DATA));
      if (!r.ok) throw new Error(`读取参考数据失败（HTTP ${r.status}）`);
      return r.json();
    })());
  }
  return refCache.get(name);
}

// ─── The engine client ──────────────────────────────────────────────────────────────────────────────────────
class EngineClient {
  constructor() {
    this.seq = 0;
    this.pending = new Map();      // id -> {resolve, reject, onEvent, msg}
    this.progress = new Map();     // name -> Set(listener)
    this.last = new Map();         // name -> [loaded, total]
    this.loads = new Map();        // name -> Promise<info>
    this.mode = 'worker';
    this.ready = this.startWorker();
  }

  startWorker() {
    return new Promise((resolve) => {
      let worker;
      try {
        worker = new Worker(new URL('./playground-worker.js', import.meta.url), { type: 'module' });
      } catch (err) {
        resolve(this.startInPage());
        return;
      }
      let alive = false;
      worker.onmessage = (e) => {
        if (e.data.kind === 'hello') { alive = true; resolve((msg, transfer) => worker.postMessage(msg, transfer || [])); return; }
        this.receive(e.data);
      };
      worker.onerror = (e) => {
        e.preventDefault?.();
        if (!alive) { worker.terminate(); resolve(this.startInPage()); return; }   // never started: run on the page
        this.failAll(new Error(`计算线程出错：${e.message || '未知错误'}`));
      };
    });
  }

  async startInPage() {
    this.mode = 'page';
    const { createEngine } = await import('./playground-engine.js');
    const engine = createEngine({ post: (msg) => queueMicrotask(() => this.receive(msg)) });
    return (msg) => { engine.handle(msg); };
  }

  failAll(err) {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
    this.loads.clear();
  }

  receive(msg) {
    if (msg.kind === 'progress') {
      this.last.set(msg.name, [msg.loaded, msg.total]);
      for (const f of this.progress.get(msg.name) || []) f(msg.loaded, msg.total);
      return;
    }
    const p = this.pending.get(msg.id);
    if (!p) return;
    if (msg.kind === 'event') { p.onEvent?.(msg); return; }
    this.pending.delete(msg.id);
    if (msg.kind === 'error') p.reject(new Error(msg.message));
    else p.resolve(msg);
  }

  /** Send a request; returns {promise, cancel}. Events stream to onEvent. */
  request(type, payload, onEvent) {
    const id = ++this.seq;
    const promise = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onEvent });
    });
    this.ready.then((send) => send({ id, type, ...payload }));
    return { id, promise, cancel: () => this.ready.then((send) => send({ id: null, type: 'cancel', target: id })) };
  }

  async manifestURL(name) {
    const entry = (await getIndex()).find((c) => c.name === name);
    if (!entry) throw new Error(`没有名为 ${name} 的检查点`);
    return new URL(entry.manifest, DATA).href;
  }

  /** Load a checkpoint once per page; onProgress(loaded, total) is called for every listener. */
  load(name, onProgress) {
    if (onProgress) {
      if (!this.progress.has(name)) this.progress.set(name, new Set());
      this.progress.get(name).add(onProgress);
      const last = this.last.get(name);
      if (last) onProgress(...last);
    }
    if (!this.loads.has(name)) {
      const job = (async () => {
        const manifestURL = await this.manifestURL(name);
        return this.request('load', { name, manifestURL }).promise;
      })();
      this.loads.set(name, job);
      job.catch(() => this.loads.delete(name));
    }
    return this.loads.get(name);
  }

  async call(type, name, payload = {}, onEvent) {
    const manifestURL = await this.manifestURL(name);
    return this.request(type, { name, manifestURL, ...payload }, onEvent);
  }
}

let client = null;
/** The page's single engine client. */
export function engine() {
  client ??= new EngineClient();
  return client;
}

// ─── Small UI pieces ────────────────────────────────────────────────────────────────────────────────────────
let styled = false;
export function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-pg { font-size: .9rem; }
.w-pg-switch { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--rule); border-radius: 5px; overflow: hidden; }
.w-pg-switch button { border: 0; border-radius: 0; padding: .25rem .8rem; background: transparent; color: var(--ink-2); }
.w-pg-switch button + button { border-left: 1px solid var(--rule); }
.w-pg-switch button[aria-pressed="true"] { background: var(--ai-2); color: var(--ink); font-weight: 600; }
.w-pg-btn { padding: .25rem .8rem; }
.w-pg-btn.primary { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-pg-btn:disabled { opacity: .45; cursor: default; }
.w-pg-status { min-height: 1.4em; color: var(--ink-2); font-size: .82rem; font-variant-numeric: tabular-nums; }
.w-pg-progress { position: relative; height: 4px; background: var(--paper-3); border-radius: 2px; overflow: hidden; margin: .35rem 0; }
.w-pg-progress > span { position: absolute; inset: 0 auto 0 0; background: var(--ai); width: 0; }
.w-pg-progress[hidden] { display: none; }
.w-pg-field { display: inline-flex; align-items: center; gap: .4rem; color: var(--ink-2); font-size: .85rem; }
.w-pg-field input[type="range"] { width: 7.5rem; accent-color: var(--ai); }
.w-pg-field output { min-width: 2.6em; font-variant-numeric: tabular-nums; color: var(--ink); }
.w-pg-field input[type="number"], .w-pg-field select { font: inherit; color: var(--ink); background: var(--paper);
  border: 1px solid var(--rule); border-radius: 4px; padding: .1rem .3rem; width: 5em; }
.w-pg-field select { width: auto; }
.w-pg textarea.w-input { resize: vertical; min-height: 3.2em; }
.w-pg-chips { max-height: 10.5rem; overflow: auto; }
.w-pg-out { font: 14px/1.55 var(--mono); background: var(--paper-2); border: 1px solid var(--rule); border-radius: 4px;
  padding: .6rem .75rem; min-height: 5.5em; white-space: pre-wrap; overflow-wrap: anywhere; margin: .5rem 0; }
.w-pg-out .prompt { color: var(--ink-3); }
.w-pg-out .gen { border-radius: 2px; cursor: default; }
.w-pg-out .gen:focus { outline: 1px solid var(--ai); }
.w-pg-out .eos { color: var(--ink-3); font-size: 12px; border: 1px dashed var(--rule); border-radius: 3px; padding: 0 .3em; margin-left: .2em; white-space: nowrap; }
.w-pg-bars { display: grid; grid-template-columns: minmax(4.5em, max-content) 1fr 3.6em; gap: 2px .6rem; align-items: center;
  font-size: .82rem; margin: .4rem 0; }
.w-pg-bars .lab { font-family: var(--mono); white-space: pre; overflow: hidden; text-overflow: ellipsis; max-width: 9em; }
.w-pg-bars .bar { height: .8em; border-radius: 2px; background: var(--ai); min-width: 1px; }
.w-pg-bars .bar.off { background: var(--paper-3); }
.w-pg-bars .num { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink-2); }
.w-pg-bars .off-lab { color: var(--ink-3); text-decoration: line-through; }
.w-pg-note { color: var(--ink-3); font-size: .8rem; }
.w-pg-error { color: var(--kincha); font-size: .85rem; }
.w-pg-table { width: 100%; border-collapse: collapse; font-size: .82rem; font-variant-numeric: tabular-nums; }
.w-pg-table th, .w-pg-table td { border-bottom: 1px solid var(--rule); padding: .25rem .4rem; text-align: left; vertical-align: top; }
.w-pg-table th { color: var(--ink-2); font-weight: 600; }
.w-pg-tight { white-space: nowrap; }
.w-pg-scroll { overflow-x: auto; max-width: 100%; }
.w-pg-presets { display: flex; flex-wrap: wrap; gap: .3rem .4rem; align-items: center; color: var(--ink-2); font-size: .85rem; }
.w-pg-presets button { white-space: nowrap; }
@media (max-width: 640px) {
  .w-pg-rtable thead { display: none; }
  .w-pg-rtable tr { display: block; border-bottom: 1px solid var(--rule); padding: .35rem 0; }
  .w-pg-rtable td { display: flex; justify-content: space-between; gap: 1rem; border: 0; padding: .1rem 0; text-align: right; }
  .w-pg-rtable td::before { content: attr(data-label); color: var(--ink-3); text-align: left; }
  .w-pg-tight th, .w-pg-tight td { padding-left: .2rem; padding-right: .2rem; }
}
@media (prefers-reduced-motion: no-preference) { .w-pg-progress > span { transition: width .15s linear; } }
`));
}

/** Segmented checkpoint switch. entries: index.json checkpoints. */
export function checkpointSwitch(entries, value, onChange, label = '选择检查点') {
  const box = h('div', { class: 'w-pg-switch', role: 'group', 'aria-label': label });
  const buttons = entries.map((c) => h('button', { type: 'button', 'aria-pressed': String(c.name === value), 'data-name': c.name,
    onclick: () => { set(c.name); onChange(c.name); } }, c.label));
  const set = (name) => { for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.name === name)); };
  box.append(...buttons);
  return { el: box, set, buttons };
}

/** A labelled range slider with its value. */
export function slider({ label, min, max, step, value, format = (v) => String(v), onInput, aria }) {
  const out = h('output', {}, format(value));
  const input = h('input', { type: 'range', min, max, step, value, 'aria-label': aria || label });
  input.addEventListener('input', () => { out.textContent = format(+input.value); onInput(+input.value); });
  const el = h('label', { class: 'w-pg-field' }, label, input, out);
  return { el, input, get value() { return +input.value; }, set(v) { input.value = v; out.textContent = format(+v); } };
}

/** Download progress + status line for one widget. */
export function statusLine() {
  const bar = h('span');
  const progress = h('div', { class: 'w-pg-progress', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, hidden: true }, bar);
  const text = h('div', { class: 'w-pg-status', 'aria-live': 'polite' });
  const err = h('div', { class: 'w-pg-error', role: 'alert' });
  return {
    el: h('div', {}, progress, text, err),
    progress(loaded, total) {
      const pct = total ? Math.round((loaded / total) * 100) : 0;
      progress.hidden = loaded >= total;
      bar.style.width = `${pct}%`;
      progress.setAttribute('aria-valuenow', pct);
      if (loaded < total) text.textContent = `正在下载权重：${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
    },
    say(s) { text.textContent = s; err.textContent = ''; },
    error(e) { progress.hidden = true; err.textContent = `出错了：${e?.message || e}`; },
  };
}

/** Load a checkpoint for a widget, reporting progress on its status line. */
export async function ensureModel(name, status) {
  const info = await engine().load(name, (l, t) => status?.progress(l, t));
  status?.progress(1, 1);
  return info;
}

/** Text of one token for display (whitespace made visible). */
export function tokenLabel(tok, id) {
  const text = tok.decode([id]);
  return Object.values(tok.specialTokens).includes(id) ? text : visible(text);
}

/** Call fn once el is near the viewport (or right away without IntersectionObserver). */
export function whenVisible(el, fn) {
  if (typeof IntersectionObserver !== 'function') { fn(); return; }
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
  }, { rootMargin: '300px 0px' });
  io.observe(el);
}

export const pct = (p, digits = 1) => `${(p * 100).toFixed(digits)}%`;

/** One-line provenance of a checkpoint from its manifest (written by tools/export_codebot.py). */
export function provenanceText(info) {
  const p = info?.provenance || {};
  const s = p.run_step;
  if (s) {
    const fin = s.final ? Object.entries(s.final).filter(([k]) => k !== 'it')
      .map(([k, v]) => (k === 'acc' ? `训练时准确率 ${v}%` : `最终 ${k} ${v}`)).join('，') : '';
    const gpu = (s.gpu || '').replace(/^NVIDIA GeForce /, '');
    return `${s.script}，${gpu}，用时 ${(s.seconds / 60).toFixed(1)} 分钟${fin ? `，${fin}` : ''}`;
  }
  return p.source ? `来源：${p.source}` : (p.checkpoint || '');
}

/** Next-token candidates: the top raw probabilities at temperature T, marking what top-k / top-p removed. */
export function candidateBars(tok, logits, { temperature, topK, topP }, n = 8) {
  const dist = filterDistribution(logits, { temperature, topK, topP });
  const raw = temperature > 0 ? softmax(logits, temperature) : dist.probs;
  const order = argsortDesc(raw).slice(0, n);
  const kept = new Set(dist.kept);
  const max = raw[order[0]] || 1;
  const grid = h('div', { class: 'w-pg-bars', role: 'list', 'aria-label': '下一个 token 的候选' });
  for (const id of order) {
    const on = kept.has(id);
    grid.append(
      h('span', { class: `lab${on ? '' : ' off-lab'}`, role: 'listitem', title: `id ${id}` }, tokenLabel(tok, id)),
      h('span', { class: `bar${on ? '' : ' off'}`, style: `width:${Math.max(0.5, (raw[id] / max) * 100)}%` }),
      h('span', { class: 'num' }, on ? pct(dist.probs[id]) : '0%'),
    );
  }
  const H = entropy(dist.probs);
  return { grid, summary: `保留 ${dist.kept.length} 个 token；过滤后分布的熵 ${H.toFixed(2)} nats（温度 1、不过滤时 ${entropy(softmax(logits)).toFixed(2)} nats）` };
}
