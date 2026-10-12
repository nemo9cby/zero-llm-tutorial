/* P.4 Through the six blocks: the logit lens. After the embedding and after every block, the residual stream of the
   last position is sent through the model's own final LayerNorm and unembedding, as if the model stopped there.
   config: { checkpoint, prompt, template } */
import { h, chip } from './_lib.js';
import { lineChart } from './_plot.js';
import {
  injectStyle, getIndex, getTokenizer, engine, ensureModel, checkpointSwitch, statusLine, alpaca, tokenLabel,
  whenVisible, pct,
} from './playground-common.js';

export async function mount(root, config = {}) {
  injectStyle();
  const cfg = { checkpoint: 'pretrain', prompt: 'def add(a, b):\n', template: false, ...config };
  const box = h('div', { class: 'w-pg' }, h('p', { class: 'w-pg-note' }, '正在准备……'));
  root.append(box);
  let entries, tok;
  try {
    [entries, tok] = await Promise.all([getIndex(), getTokenizer()]);
  } catch (err) {
    box.replaceChildren(h('p', { class: 'w-pg-error' }, `这张图没能启动：${err.message}`));
    return;
  }
  let name = cfg.checkpoint, busy = false, again = false;
  const sw = checkpointSwitch(entries, name, (n) => { name = n; refresh(); });
  const text = h('textarea', { class: 'w-input', rows: 2, spellcheck: 'false', 'aria-label': 'prompt' });
  text.value = cfg.prompt;
  const tpl = h('input', { type: 'checkbox', checked: cfg.template });
  const status = statusLine();
  const chips = h('div', { class: 'tokens' });
  const table = h('div', { class: 'w-pg-scroll' });
  const chartBox = h('div');
  const note = h('div', { class: 'w-pg-note' });
  box.replaceChildren(h('div', { class: 'w-row' }, sw.el, h('label', { class: 'w-pg-field' }, tpl, '套用 Alpaca 模板')),
    text, chips, status.el, table, chartBox, note);
  let chart = null;

  async function refresh() {
    if (busy) { again = true; return; }
    busy = true;
    sw.set(name);
    try {
      const ids = tok.encode(tpl.checked ? alpaca(text.value) : text.value).slice(-256);
      if (!ids.length) { status.say('先输入一点文字。'); return; }
      const shown = ids.slice(-12);
      chips.replaceChildren(...shown.map((id, i) => chip(tok.decode([id]), id, i, `id ${id}`)));
      if (ids.length > shown.length) chips.prepend(h('span', { class: 'w-pg-note' }, '…… '));
      await ensureModel(name, status);
      const res = await (await engine().call('forward', name, { ids, lens: 5 })).promise;
      const L = res.lens.length - 1;
      const rows = res.lens.map((x) => h('tr', {},
        h('th', { scope: 'row' }, x.layer === 0 ? '0：embedding' : x.layer === L ? `${x.layer}：块 ${x.layer}（输出）` : `${x.layer}：块 ${x.layer}`),
        h('td', {}, h('span', { class: 'tokens', style: 'margin:0' },
          x.ids.slice(0, 3).map((id, r) => h('span', { title: `id ${id}` }, chip(tok.decode([id]), pct(x.probs[r], 0), r)))))));
      table.replaceChildren(h('table', { class: 'w-pg-table', 'aria-label': '每一层的 logit lens：最可能的 3 个下一个 token' },
        h('thead', {}, h('tr', {}, h('th', {}, '层'), h('th', {}, '如果在这一层停下，最可能的下一个 token（概率）'))),
        h('tbody', {}, rows)));
      const series = res.trackIds.slice(0, 3).map((id, j) => ({
        name: tokenLabel(tok, id), points: res.track.map((row, l) => [l, row[j]]),
      }));
      const opts = { series, xLabel: '层（0 = 只有 embedding）', yLabel: '概率：模型最终的前 3 个候选，在每一层的 logit lens 里', yMin: 0, yMax: 1,
        yFormat: (v) => pct(v, 0), height: 220 };
      if (chart) chart.update(opts); else chart = lineChart(chartBox, opts);
      const first = res.lens[0].ids[0], last = ids[ids.length - 1];
      note.textContent = `第 0 层（还没进任何块）的第一名是「${tokenLabel(tok, first)}」${first === last ? '，正是输入的最后一个 token 自己' : ''}；`
        + `最终的第一名「${tokenLabel(tok, res.trackIds[0])}」从第 ${firstLayerTop(res)} 层起一直排在第一。`;
      status.say(`${ids.length} 个 token，前向 ${res.ms.toFixed(0)} ms`);
    } catch (err) {
      status.error(err);
    } finally {
      busy = false;
      if (again) { again = false; refresh(); }
    }
  }

  // The first layer from which the final top-1 token stays the lens's top-1.
  function firstLayerTop(res) {
    const target = res.trackIds[0];
    let from = res.lens.length - 1;
    for (let l = res.lens.length - 1; l >= 0 && res.lens[l].ids[0] === target; l--) from = l;
    return from;
  }

  let t = 0;
  const later = () => { clearTimeout(t); t = setTimeout(refresh, 300); };
  text.addEventListener('input', later);
  tpl.addEventListener('change', refresh);
  status.say('这张图进入视野时才加载模型。');
  whenVisible(box, refresh);
}
