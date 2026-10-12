/* 图：采样台 —— 温度、Top-k、Top-p 怎样改变下一个 token 的分布。
   分布来自官方 checkpoint（data/ch03/next_token.json，由 verify/_ch03_export.py 导出、test_ch03.py 核对）。
   config: { src, vocab, top: 12 } */
import { h } from './_lib.js';
import { injectStyle, loadJSON, softmax, entropy, argsortDesc, topK, topP, sample, rng, slider, segmented, showTok } from './ch03-common.js';

const MODEL_NAME = { pretrain: '预训练模型', sft: 'SFT 模型', grpo: 'GRPO 模型' };

export async function mount(root, config) {
  injectStyle();
  const [data, vocab] = await Promise.all([loadJSON(root, config.src), loadJSON(root, config.vocab)]);
  if (!data || !vocab) return;
  const tokens = vocab.tokens;
  const ctxs = data.contexts;
  const nShow = config.top || 12;
  let ci = 0, T = 1, k = 0, p = 1, seed = 1, counts = null, nDraw = 0;

  root.classList.add('w-ch03');
  const style = h('style', {}, `
.w-ch03-samp .rows { display: grid; grid-template-columns: minmax(4.5rem, max-content) minmax(0, 1fr) 4.2rem; gap: 2px .6rem; align-items: center; margin: .5rem 0; }
.w-ch03-samp .lab { font: 13px/1.5 var(--mono); white-space: pre; overflow: hidden; text-overflow: ellipsis; max-width: 9rem; }
.w-ch03-samp .lab.cut { color: var(--ink-3); text-decoration: line-through; }
.w-ch03-samp .track { position: relative; height: 16px; background: var(--paper-2); border-radius: 2px; }
.w-ch03-samp .bar { position: absolute; left: 0; top: 0; bottom: 0; background: var(--ai); border-radius: 2px; }
.w-ch03-samp .ghost { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--ink-3); }
.w-ch03-samp .dot { position: absolute; top: 4px; width: 8px; height: 8px; margin-left: -4px; border-radius: 50%; background: var(--kincha); border: 1px solid var(--paper); }
.w-ch03-samp .pct { font: 12px/1.4 var(--mono); text-align: right; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.w-ch03-samp .legend { font-size: .8rem; color: var(--ink-3); }
.w-ch03-samp .legend i { display: inline-block; vertical-align: middle; margin: 0 .25rem 0 .6rem; }
`);
  root.classList.add('w-ch03-samp');

  const ctxSeg = segmented(ctxs.map((c) => c.label), 0, (i) => { ci = i; counts = null; render(); }, '选择上下文');
  const sT = slider({ label: '温度 T', min: 0, max: 3, step: 0.05, value: 1, format: (v) => (v === 0 ? '0（greedy）' : v.toFixed(2)),
    onInput: (v) => { T = v; counts = null; render(); } });
  const sK = slider({ label: 'Top-k', min: 0, max: 50, step: 1, value: 0, format: (v) => (v === 0 ? '不截断' : String(v)),
    onInput: (v) => { k = v; counts = null; render(); } });
  const sP = slider({ label: 'Top-p', min: 0.05, max: 1, step: 0.05, value: 1, format: (v) => (v >= 1 ? '不截断' : v.toFixed(2)),
    onInput: (v) => { p = v; counts = null; render(); } });
  const drawBtn = h('button', { type: 'button' }, '采样 200 次');
  const resetBtn = h('button', { type: 'button' }, '恢复默认');
  const prompt = h('div', { class: 'stat' });
  const stats = h('div', { class: 'stat', 'aria-live': 'polite' });
  const rows = h('div', { class: 'rows', role: 'list', 'aria-label': '概率最大的若干 token' });
  const drawInfo = h('div', { class: 'stat' });
  const legend = h('div', { class: 'legend' }, '蓝条：调整后的概率',
    h('i', { style: 'width:2px;height:12px;background:var(--ink-3)' }), '原始分布（T=1、不截断）',
    h('i', { style: 'width:8px;height:8px;border-radius:50%;background:var(--kincha)' }), '采样频率');

  root.append(style, ctxSeg, h('div', { class: 'ctl' }, sT.el, sK.el, sP.el), h('div', { class: 'ctl' }, drawBtn, resetBtn),
    prompt, stats, legend, rows, drawInfo);

  drawBtn.addEventListener('click', () => {
    const q = current();
    const rand = rng(seed++);
    counts = new Map();
    nDraw = 200;
    for (let i = 0; i < nDraw; i++) {
      const t = sample(q, rand);
      counts.set(t, (counts.get(t) || 0) + 1);
    }
    render();
  });
  resetBtn.addEventListener('click', () => {
    T = 1; k = 0; p = 1; counts = null;
    sT.set(1); sK.set(0); sP.set(1);
    render();
  });

  function current() {
    const z = ctxs[ci].logits;
    return topP(topK(softmax(z, T), k), p);
  }

  function render() {
    const c = ctxs[ci];
    const base = softmax(c.logits, 1);
    const q = current();
    const kept = q.filter((v) => v > 0).length;
    const H = entropy(q);
    prompt.replaceChildren(`${MODEL_NAME[c.model] || c.model}，已读入：`, h('code', { class: 'tokc' }, showTok(c.prompt.length > 60 ? `…${c.prompt.slice(-60)}` : c.prompt)));
    stats.textContent = `保留 ${kept} / ${q.length} 个 token；熵 ${H.toFixed(3)} nats（${(H / Math.LN2).toFixed(2)} bits）；`
      + `最大概率 ${(Math.max(...q) * 100).toFixed(1)}%`;
    // 显示调整后概率最大的若干个；如果截断掉了原分布里的前几名，也一并列出（划掉）
    const show = [];
    for (const i of argsortDesc(q)) { if (show.length >= nShow || q[i] === 0) break; show.push(i); }
    for (const i of argsortDesc(base).slice(0, nShow)) if (show.length < nShow + 4 && !show.includes(i) && q[i] === 0) show.push(i);
    const maxv = Math.max(...show.map((i) => Math.max(q[i], base[i])), 1e-9);
    rows.replaceChildren(...show.flatMap((i) => {
      const cut = q[i] === 0;
      const track = h('div', { class: 'track' },
        h('div', { class: 'bar', style: `width:${(q[i] / maxv) * 100}%` }),
        h('div', { class: 'ghost', style: `left:calc(${(base[i] / maxv) * 100}% - 1px)` }));
      if (counts && counts.get(i)) {
        track.append(h('div', { class: 'dot', style: `left:${((counts.get(i) / nDraw) / maxv) * 100}%`, title: `${counts.get(i)} 次` }));
      }
      const label = `${showTok(tokens[i])}`;
      return [
        h('div', { class: `lab${cut ? ' cut' : ''}`, title: `id ${i}`, role: 'listitem' }, label),
        track,
        h('div', { class: 'pct' }, cut ? '截掉' : `${(q[i] * 100).toFixed(q[i] < 0.001 ? 3 : 1)}%`),
      ];
    }));
    if (counts) {
      const distinct = counts.size;
      const outside = [...counts.keys()].filter((t) => !show.includes(t)).reduce((a, t) => a + counts.get(t), 0);
      drawInfo.textContent = `200 次采样里出现了 ${distinct} 种不同的 token；落在上表之外的有 ${outside} 次。`;
    } else {
      drawInfo.textContent = '';
    }
  }
  render();
}
