/* P.6 From logits to the next token: the 1,000 logits of the last position become a distribution (temperature),
   lose their long tail (top-k, then top-p), and one token is drawn with a seeded generator.
   config: { checkpoint, prompt, template, temperature, topK, topP, seed } */
import { h } from './_lib.js';
import { filterDistribution, sampleFrom, mulberry32 } from './_gpt.js';
import {
  injectStyle, getIndex, getTokenizer, engine, ensureModel, checkpointSwitch, slider, statusLine, alpaca,
  tokenLabel, whenVisible, pct, candidateBars,
} from './playground-common.js';

export async function mount(root, config = {}) {
  injectStyle();
  const cfg = { checkpoint: 'pretrain', prompt: 'for i in', template: false, temperature: 1, topK: 0, topP: 1, seed: 0, ...config };
  const box = h('div', { class: 'w-pg' }, h('p', { class: 'w-pg-note' }, '正在准备……'));
  root.append(box);
  let entries, tok;
  try {
    [entries, tok] = await Promise.all([getIndex(), getTokenizer()]);
  } catch (err) {
    box.replaceChildren(h('p', { class: 'w-pg-error' }, `这张图没能启动：${err.message}`));
    return;
  }
  let name = cfg.checkpoint, logits = null, busy = false, again = false, draws = 0;
  const s = { temperature: cfg.temperature, topK: cfg.topK, topP: cfg.topP };
  const sw = checkpointSwitch(entries, name, (n) => { name = n; refresh(); });
  const text = h('input', { class: 'w-input', value: cfg.prompt, spellcheck: 'false', 'aria-label': 'prompt' });
  const tpl = h('input', { type: 'checkbox', checked: cfg.template });
  const temp = slider({ label: '温度', min: 0, max: 2, step: 0.05, value: s.temperature, format: (v) => (v === 0 ? '0（greedy）' : v.toFixed(2)),
    onInput: (v) => { s.temperature = v; draw(); } });
  const topk = slider({ label: 'Top-k', min: 0, max: 50, step: 1, value: s.topK, format: (v) => (v === 0 ? '不限' : String(v)),
    onInput: (v) => { s.topK = v; draw(); } });
  const topp = slider({ label: 'Top-p', min: 0.05, max: 1, step: 0.05, value: s.topP, format: (v) => v.toFixed(2),
    onInput: (v) => { s.topP = v; draw(); } });
  const status = statusLine();
  const bars = h('div');
  const summary = h('div', { class: 'w-label' });
  const pick = h('div', { class: 'w-label', 'aria-live': 'polite' });
  const drawBtn = h('button', { type: 'button', class: 'w-pg-btn primary', onclick: sample }, '按这个分布抽一个 token');
  box.replaceChildren(h('div', { class: 'w-row' }, sw.el, h('label', { class: 'w-pg-field' }, tpl, '套用 Alpaca 模板')),
    text, status.el, h('div', { class: 'w-row' }, temp.el, topk.el, topp.el), bars, summary,
    h('div', { class: 'w-row' }, drawBtn), pick);

  function draw() {
    if (!logits) return;
    const { grid, summary: line } = candidateBars(tok, logits, s, 15);
    bars.replaceChildren(grid);
    summary.textContent = line;
  }

  function sample() {
    if (!logits) return;
    const dist = filterDistribution(logits, s);
    const u = mulberry32(cfg.seed + draws)();
    const id = s.temperature === 0 ? dist.kept[0] : sampleFrom(dist, u);
    draws += 1;
    pick.textContent = `第 ${draws} 次（种子 ${cfg.seed + draws - 1}，u = ${u.toFixed(3)}）：抽到「${tokenLabel(tok, id)}」，它在过滤后的分布里占 ${pct(dist.probs[id])}。`;
  }

  async function refresh() {
    if (busy) { again = true; return; }
    busy = true;
    sw.set(name);
    try {
      const ids = tok.encode(tpl.checked ? alpaca(text.value) : text.value).slice(-256);
      if (!ids.length) { status.say('先输入一点文字。'); return; }
      await ensureModel(name, status);
      const res = await (await engine().call('forward', name, { ids })).promise;
      logits = res.logits;
      draws = 0;
      pick.textContent = '';
      draw();
      status.say(`${ids.length} 个 token 的输入；下面是第 ${ids.length + 1} 个位置的 1,000 个候选里概率最大的 15 个。`);
    } catch (err) {
      status.error(err);
    } finally {
      busy = false;
      if (again) { again = false; refresh(); }
    }
  }

  let t = 0;
  text.addEventListener('input', () => { clearTimeout(t); t = setTimeout(refresh, 300); });
  tpl.addEventListener('change', refresh);
  status.say('这张图进入视野时才加载模型。');
  whenVisible(box, refresh);
}
