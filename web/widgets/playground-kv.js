/* P.7 One token at a time: what the KV cache saves, and what happens at the 256-position wall.
   (1) The cost of one new token at context length n: with the KV cache (one step) vs without it (the book's
       generate recomputes the whole window). (2) A generation that crosses 256 positions: learned position
       embeddings cannot slide, so every token after the wall recomputes the last 256 tokens.
   config: { checkpoint, lengths, cross } */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle, getTokenizer, getRef, engine, ensureModel, statusLine, tokenLabel } from './playground-common.js';

export async function mount(root, config = {}) {
  injectStyle();
  const cfg = { checkpoint: 'pretrain', lengths: [8, 16, 32, 64, 128, 192, 256], cross: 12, ...config };
  const box = h('div', { class: 'w-pg' }, h('p', { class: 'w-pg-note' }, '正在准备……'));
  root.append(box);
  let tok;
  try {
    tok = await getTokenizer();
  } catch (err) {
    box.replaceChildren(h('p', { class: 'w-pg-error' }, `这张图没能启动：${err.message}`));
    return;
  }
  const status = statusLine();
  const benchBtn = h('button', { type: 'button', class: 'w-pg-btn primary' }, '测量：生成 1 个新 token 要多久');
  const crossBtn = h('button', { type: 'button', class: 'w-pg-btn' }, `越过 256：从 251 个 token 再生成 ${cfg.cross} 个`);
  const chartBox = h('div');
  const table = h('div', { class: 'w-pg-scroll' });
  const strip = h('div', { class: 'w-pg-scroll' });
  const note = h('div', { class: 'w-pg-note', 'aria-live': 'polite' });
  box.replaceChildren(h('div', { class: 'w-row' }, benchBtn, crossBtn), status.el, chartBox, table, strip, note);
  let chart = null, busy = false;
  const setBusy = (b) => { busy = b; benchBtn.disabled = b; crossBtn.disabled = b; };

  // The 251-token TinyCodes prompt of ref.json; with its recorded greedy continuation it reaches 267 tokens.
  async function longIds(extended) {
    const p = (await getRef(cfg.checkpoint)).prompts.find((x) => x.name === 'long');
    return extended ? p.ids.concat(p.q8.greedy) : p.ids;
  }

  benchBtn.addEventListener('click', async () => {
    if (busy) return;
    setBusy(true);
    try {
      const ids = await longIds(true);
      const info = await ensureModel(cfg.checkpoint, status);
      status.say(`测量中（${info.backend === 'wasm-simd' ? 'WASM SIMD' : '纯 JavaScript'}）……`);
      const rows = [];
      const handle = await engine().call('bench', cfg.checkpoint, { ids, lengths: cfg.lengths }, (ev) => {
        rows.push(ev);
        const series = [
          { name: '有 KV cache', points: rows.map((r) => [r.n, r.decodeMs]) },
          { name: '没有 KV cache', points: rows.map((r) => [r.n, r.prefillMs]) },
        ];
        const opts = { series, xLabel: '上下文长度 n（个 token）', yLabel: '生成第 n 个位置的 token 用时（ms，对数刻度）', logY: true, height: 230 };
        if (chart) chart.update(opts); else chart = lineChart(chartBox, opts);
      });
      await handle.promise;
      table.replaceChildren(h('table', { class: 'w-pg-table' },
        h('thead', {}, h('tr', {}, h('th', {}, 'n'), h('th', {}, '有 KV cache'), h('th', {}, '没有 KV cache'), h('th', {}, '倍数'))),
        h('tbody', {}, rows.map((r) => h('tr', {}, h('td', {}, r.n), h('td', {}, `${r.decodeMs.toFixed(2)} ms`),
          h('td', {}, `${r.prefillMs.toFixed(1)} ms`), h('td', {}, `${(r.prefillMs / r.decodeMs).toFixed(0)}×`))))));
      const last = rows[rows.length - 1];
      status.say(`n = ${last.n} 时，没有 KV cache 要重算 ${last.n} 个位置，用时是一步解码的 ${(last.prefillMs / last.decodeMs).toFixed(0)} 倍。计时取决于你的设备。`);
    } catch (err) {
      status.error(err);
    } finally {
      setBusy(false);
    }
  });

  crossBtn.addEventListener('click', async () => {
    if (busy) return;
    setBusy(true);
    try {
      const ids = await longIds(false);
      await ensureModel(cfg.checkpoint, status);
      status.say(`从 ${ids.length} 个 token 开始，greedy 生成 ${cfg.cross} 个……`);
      const cells = [];
      const handle = await engine().call('generate', cfg.checkpoint,
        { ids, opts: { temperature: 0, maxNewTokens: cfg.cross, stopAtEnd: false, top: 0 } }, (ev) => {
          const t = ev.token;
          cells.push(h('tr', {}, h('td', {}, ids.length + t.step + 1), h('td', {}, tokenLabel(tok, t.id)),
            h('td', {}, t.refill ? `重算 ${t.window} 个` : t.step === 0 ? `prefill ${t.window} 个` : 'KV cache，算 1 个'),
            h('td', {}, `${ev.ms.toFixed(1)} ms`), h('td', {}, t.computed)));
          strip.replaceChildren(h('table', { class: 'w-pg-table' },
            h('thead', {}, h('tr', {}, h('th', {}, '序列中的第几个'), h('th', {}, 'token'), h('th', {}, '算出它的前向传播'), h('th', {}, '用时'),
              h('th', {}, '累计计算的位置'))), h('tbody', {}, cells)));
        });
      const res = await handle.promise;
      const inside = 256 - ids.length + 1;
      note.textContent = `第 1 个新 token 来自 prompt 的 prefill；第 2 到第 ${inside} 个还在 256 个位置之内，各只算 1 个位置；`
        + `从第 ${inside + 1} 个起，每一个都要把最近的 256 个 token 从头算一遍。总用时 ${(res.ms / 1000).toFixed(1)} 秒。`;
      status.say('完成。');
    } catch (err) {
      status.error(err);
    } finally {
      setBusy(false);
    }
  });
  status.say('点按钮开始。测量会在后台线程里加载预训练权重（11.3 MB）。');
}
