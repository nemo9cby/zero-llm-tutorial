/* P.1 The workbench: the whole life of a prompt in one figure. Checkpoint switch (pretrain / SFT / GRPO), the
   chat template of ch03, the tokens the model sees, sampling settings, streamed generation (each token shaded by
   the probability it was drawn with), and the candidates for the next token.
   config: { checkpoint, prompt, template, temperature, topK, topP, maxNewTokens, seed, autorun } */
import { h, chip } from './_lib.js';
import {
  injectStyle, getIndex, getTokenizer, engine, ensureModel, checkpointSwitch, slider, statusLine, alpaca,
  tokenLabel, whenVisible, pct, provenanceText, candidateBars,
} from './playground-common.js';

const PRESETS = [
  { label: 'read a file line by line', text: 'read a file line by line', template: true },
  { label: '3+4=', text: '3+4=', template: true },
  { label: 'def add(a, b):', text: 'def add(a, b):\n', template: false },
];

export async function mount(root, config = {}) {
  injectStyle();
  const cfg = {
    checkpoint: 'sft', prompt: 'read a file line by line', template: true, temperature: 0.8, topK: 40, topP: 0.95,
    maxNewTokens: 128, seed: 1, autorun: true, ...config,
  };
  const box = h('div', { class: 'w-pg w-pg-bench' }, h('p', { class: 'w-pg-note' }, '正在准备 CodeBot 工作台……'));
  root.append(box);
  let entries, tok;
  try {
    [entries, tok] = await Promise.all([getIndex(), getTokenizer()]);
  } catch (err) {
    box.replaceChildren(h('p', { class: 'w-pg-error' }, `工作台没能启动：${err.message}`));
    return;
  }

  let name = entries.some((c) => c.name === cfg.checkpoint) ? cfg.checkpoint : entries[0].name;
  let info = null, job = null, gen = [], decoder = null, lastLogits = null, started = false;
  const settings = { temperature: cfg.temperature, topK: cfg.topK, topP: cfg.topP };

  // ── controls
  const desc = h('div', { class: 'w-label' });
  const prov = h('div', { class: 'w-pg-note' });
  const sw = checkpointSwitch(entries, name, (n) => { name = n; onCheckpoint(); }, '选择 CodeBot 的 checkpoint');
  const text = h('textarea', { class: 'w-input', rows: 2, spellcheck: 'false', 'aria-label': '输入 prompt 或指令' });
  text.value = cfg.prompt;
  const tpl = h('input', { type: 'checkbox', checked: cfg.template });
  const presets = h('div', { class: 'w-pg-presets' }, '示例：', PRESETS.map((p) => h('button', { type: 'button', class: 'w-pg-btn',
    onclick: () => { text.value = p.text; tpl.checked = p.template; onPrompt(); } }, p.label)));
  const chipsHead = h('div', { class: 'w-label' });
  const chips = h('div', { class: 'tokens w-pg-chips' });
  const temp = slider({ label: '温度', min: 0, max: 2, step: 0.05, value: settings.temperature, format: (v) => (v === 0 ? '0（贪心解码）' : v.toFixed(2)),
    onInput: (v) => { settings.temperature = v; showNext(); } });
  const topk = slider({ label: 'Top-k', min: 0, max: 100, step: 1, value: settings.topK, format: (v) => (v === 0 ? '不限' : String(v)),
    onInput: (v) => { settings.topK = v; showNext(); } });
  const topp = slider({ label: 'Top-p', min: 0.05, max: 1, step: 0.05, value: settings.topP, format: (v) => v.toFixed(2),
    onInput: (v) => { settings.topP = v; showNext(); } });
  const maxNew = h('select', { 'aria-label': '最多生成多少个 token' }, [32, 64, 128, 256].map((v) => h('option', { value: v, selected: v === cfg.maxNewTokens }, v)));
  const seed = h('input', { type: 'number', min: 0, max: 99999, step: 1, value: cfg.seed, 'aria-label': '随机种子' });
  const genBtn = h('button', { type: 'button', class: 'w-pg-btn primary', onclick: () => run(false) }, '生成');
  const stepBtn = h('button', { type: 'button', class: 'w-pg-btn', onclick: () => run(true), title: '在当前序列后面再生成 1 个 token' }, '单步');
  const stopBtn = h('button', { type: 'button', class: 'w-pg-btn', disabled: true, onclick: () => job?.cancel() }, '停止');
  const out = h('div', { class: 'w-pg-out', 'aria-live': 'off', 'aria-label': '生成结果' });
  const detail = h('div', { class: 'w-pg-note', 'aria-live': 'polite' }, '把鼠标移到（或用 Tab 键移到）生成的 token 上，可以看到它被抽中的概率和当时的候选。');
  const status = statusLine();
  const nextHead = h('div', { class: 'w-label' });
  const nextBox = h('div');
  const nextNote = h('div', { class: 'w-pg-note' });

  box.replaceChildren(
    h('div', { class: 'w-row' }, sw.el), desc, prov,
    h('div', { class: 'w-row' }, h('label', { class: 'w-pg-field' }, tpl, '套用 ch03 的 Alpaca 模板')), presets,
    text, chipsHead, chips,
    h('div', { class: 'w-row' }, temp.el, topk.el, topp.el,
      h('label', { class: 'w-pg-field' }, '最多', maxNew, '个 token'), h('label', { class: 'w-pg-field' }, '种子', seed)),
    h('div', { class: 'w-row' }, genBtn, stepBtn, stopBtn),
    status.el, out, detail, nextHead, nextBox, nextNote,
  );

  // ── state helpers
  const promptText = () => (tpl.checked ? alpaca(text.value) : text.value);
  const promptIds = () => tok.encode(promptText());
  const seqIds = () => [...promptIds(), ...gen.filter((g) => !g.end).map((g) => g.id)];
  const setBusy = (busy) => {
    genBtn.disabled = busy; stepBtn.disabled = busy; stopBtn.disabled = !busy;
    sw.buttons.forEach((b) => { b.disabled = busy; });
  };

  function renderChips() {
    const ids = promptIds();
    chipsHead.textContent = `模型看到的输入：${ids.length} 个 token${tpl.checked ? '（含模板）' : ''}`;
    const shown = ids.slice(0, 160);
    chips.replaceChildren(...shown.map((id, i) => chip(tok.decode([id]), id, i, `位置 ${i}，id ${id}`)));
    if (ids.length > shown.length) chips.append(h('span', { class: 'w-pg-note' }, `…… 还有 ${ids.length - shown.length} 个`));
    if (ids.length > 256) chipsHead.textContent += '。超过 256 个位置，模型只看得见最后 256 个';
  }

  function renderOut() {
    out.replaceChildren(h('span', { class: 'prompt' }, promptText()));
    decoder = tok.streamDecoder();
    for (const g of gen) appendToken(g);
  }

  function appendToken(g) {
    if (g.end) {
      out.append(h('span', { class: 'eos', title: `id ${g.id}：生成到这里停止，结束 token 不接进结果` }, '<|endoftext|> 停止'));
      return;
    }
    const piece = decoder.push(g.id);
    const shade = Math.round((1 - g.prob) * 45);
    const span = h('span', { class: 'gen', tabindex: 0, style: `background:color-mix(in srgb, var(--ai) ${shade}%, transparent)` }, piece);
    const tell = () => {
      const cands = g.top.map(([id, p]) => `${tokenLabel(tok, id)} ${pct(p)}`).join(' · ');
      detail.textContent = `第 ${g.step + 1} 个新 token「${tokenLabel(tok, g.id)}」（id ${g.id}）：从过滤后的分布里以 ${pct(g.prob)} 的概率抽中。`
        + `温度 1 时的前 5 个候选：${cands}${g.refill ? '。这个 token 的前向重算了整个 256 窗口' : ''}`;
    };
    span.addEventListener('pointerenter', tell);
    span.addEventListener('focus', tell);
    out.append(span);
  }

  // After an <|endoftext|> stop, the interesting distribution is the one the answer's first token came from;
  // otherwise (nothing generated yet, a single step, a stop at the length limit) it is the next one.
  const showsFirst = () => gen.length > 0 && gen[gen.length - 1].end;

  async function showNext() {
    if (!info) return;
    if (!lastLogits) {
      const ids = (showsFirst() ? promptIds() : seqIds()).slice(-256);
      try {
        const handle = await engine().call('forward', name, { ids });
        lastLogits = (await handle.promise).logits;
      } catch (err) { status.error(err); return; }
    }
    const { grid, summary } = candidateBars(tok, lastLogits, settings);
    const which = showsFirst() ? '回答的第 1 个 token 的候选（prompt 之后）' : '下一个 token 的候选';
    nextHead.textContent = `${which}，按当前设置：温度 ${settings.temperature}、Top-k ${settings.topK || '不限'}、Top-p ${settings.topP}`;
    nextBox.replaceChildren(grid);
    nextNote.textContent = `${summary}。灰色的候选被 Top-k / Top-p 截掉了。`;
  }

  async function run(single) {
    if (job || !info) return;
    if (!single) { gen = []; renderOut(); }
    const ids = seqIds();
    if (gen.some((g) => g.end)) { gen = gen.filter((g) => !g.end); renderOut(); }
    setBusy(true);
    const opts = {
      temperature: settings.temperature, topK: settings.topK, topP: settings.topP, stopAtEnd: true, top: 5,
      maxNewTokens: single ? 1 : +maxNew.value, seed: (+seed.value || 0) + (single ? gen.length : 0),
    };
    const times = [];
    let lastTok = null;
    try {
      job = await engine().call('generate', name, { ids, opts }, (ev) => {
        const g = { ...ev.token, step: gen.length };
        gen.push(g);
        times.push(ev.ms);
        lastTok = g;
        appendToken(g);
      });
      const res = await job.promise;
      const decodeMs = times.slice(1).reduce((a, b) => a + b, 0);
      const rate = times.length > 1 ? `之后每秒 ${Math.round(((times.length - 1) / decodeMs) * 1000)} 个 token` : '';
      const ctx = lastTok ? `上下文 ${Math.min(256, lastTok.window + 1)}/256` : '';
      status.say([`${info.backend === 'wasm-simd' ? 'WASM SIMD' : '纯 JavaScript'}`,
        `首个 token 用时 ${times[0]?.toFixed(0) ?? '-'} ms（含 prefill ${ids.length > 256 ? 256 : ids.length} 个 token）`, rate, ctx,
        res.stopped === 'end' ? '遇到 <|endoftext|> 停止' : res.stopped === 'cancel' ? '已停止' : '到达生成上限']
        .filter(Boolean).join(' · '));
    } catch (err) {
      status.error(err);
    } finally {
      job = null;
      setBusy(false);
      lastLogits = null;
      showNext();
    }
  }

  async function onCheckpoint() {
    sw.set(name);
    const entry = entries.find((c) => c.name === name);
    desc.textContent = entry.description || '';
    prov.textContent = '';
    gen = []; lastLogits = null;
    renderOut();
    setBusy(true);
    try {
      info = await ensureModel(name, status);
      prov.textContent = `训练记录：${provenanceText(info)}。权重 ${(info.downloaded / 1e6).toFixed(1)} MB`
        + `${info.verified ? '，SHA-256 与清单一致' : ''}。`;
      status.say(`已加载 ${entry.label}：${(info.downloaded / 1e6).toFixed(1)} MB 的 int8 权重，`
        + `${info.verified ? 'SHA-256 与清单一致' : '分片大小与清单一致'}，后端 ${info.backend === 'wasm-simd' ? 'WASM SIMD' : '纯 JavaScript'}`);
      setBusy(false);
      await showNext();
    } catch (err) {
      status.error(err);
      setBusy(false);
    }
  }

  function onPrompt() {
    if (job) job.cancel();
    gen = []; lastLogits = null;
    renderChips();
    renderOut();
    clearTimeout(onPrompt.t);
    onPrompt.t = setTimeout(showNext, 250);
  }

  text.addEventListener('input', onPrompt);
  tpl.addEventListener('change', onPrompt);
  renderChips();
  renderOut();
  desc.textContent = entries.find((c) => c.name === name)?.description || '';
  status.say('权重在这张图进入视野时才下载（每个 checkpoint 11.3 MB）。');
  setBusy(true);
  whenVisible(box, async () => {
    if (started) return;
    started = true;
    await onCheckpoint();
    if (cfg.autorun && info) run(false);
  });
}
