/* 图 8-4：MoE 路由玩具。16 个 token、4 个专家，router 选 top-k，专家有容量上限，超出的 token 被丢弃。
   容量 = 每批 token 数 × k ÷ 专家数 × capacity factor：一批要分派 T·k 次，所以乘 k；k = 1 时就是 Switch 式 (3)。
   辅助 loss 按 Switch Transformer 式 (4)：E·Σ f_i·P_i（以 α 为单位），f_i 按 router 的最大值计数。
   同一组结论由验证 8.4-a 到 8.4-d 钉住；纯函数由 tests/js/ch08-widgets.test.mjs 对拍。 */
import { h } from './_lib.js';
import { seriesColor, onThemeChange } from './_plot.js';

export const TOKENS = ['The', ' cat', ' sat', ' on', ' the', ' mat', '.', ' It',
  'def', ' add', '(', 'a', ',', ' b', '):', ' return'];
export const IS_CODE = TOKENS.map((_, i) => i >= 8);
export const E = 4;

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 每个 token 对 4 个专家的 router logits：英文偏向专家 0，代码偏向专家 3，再加一点噪声；bias 加在专家 0 上。 */
export function routerLogits(bias = 0, seed = 11) {
  const r = rng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(r(), 1e-12))) * Math.cos(2 * Math.PI * r());
  return TOKENS.map((_, t) => {
    const base = IS_CODE[t] ? [-1, -1 / 3, 1 / 3, 1] : [1, 1 / 3, -1 / 3, -1];
    return base.map((b, e) => b + 0.9 * gauss() + (e === 0 ? bias : 0));
  });
}

export function softmax(z) {
  const m = Math.max(...z);
  const ex = z.map((x) => Math.exp(x - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / s);
}

export const topk = (z, k) => z.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map((p) => p[1]);

/** 路由：先满足所有 token 的第一选择，再处理第二选择；容量 = 每批 token 数 × k ÷ 专家数 × capacity factor（向下取整）。 */
export function route(logits, { k = 2, capacityFactor = 1.25 } = {}) {
  const T = logits.length, nE = logits[0].length;
  const capacity = Math.floor(((T * k) / nE) * capacityFactor);
  const choices = logits.map((z) => topk(z, k));
  const gates = logits.map((z, t) => softmax(choices[t].map((e) => z[e]))); // 只在选中的 k 个上做 softmax
  const kept = choices.map((c) => c.map(() => false));
  const assigned = new Array(nE).fill(0), used = new Array(nE).fill(0);
  for (let rank = 0; rank < k; rank++) {
    for (let t = 0; t < T; t++) {
      const e = choices[t][rank];
      assigned[e] += 1;
      if (used[e] < capacity) { used[e] += 1; kept[t][rank] = true; }
    }
  }
  const dropped = assigned.reduce((a, b) => a + b, 0) - used.reduce((a, b) => a + b, 0);
  const fullyDropped = kept.filter((row) => row.every((x) => !x)).length;
  return { capacity, choices, gates, kept, assigned, used, dropped, fullyDropped };
}

/** Switch Transformer 的辅助 loss（不含 α）：E·Σ f_i·P_i。 */
export function auxLoss(logits) {
  const T = logits.length, nE = logits[0].length;
  const probs = logits.map(softmax);
  const f = new Array(nE).fill(0), P = new Array(nE).fill(0);
  probs.forEach((p) => {
    f[p.indexOf(Math.max(...p))] += 1 / T;
    p.forEach((x, e) => { P[e] += x / T; });
  });
  return { value: nE * f.reduce((s, fi, e) => s + fi * P[e], 0), f, P };
}

let styled = false;
let instances = 0;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-ch08-moe .ctl { display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; align-items: center; font-size: .88rem; margin-bottom: .7rem; }
.w-ch08-moe .ctl label { display: inline-flex; gap: .35rem; align-items: center; }
.w-ch08-moe .ctl input[type=range] { width: 8.5rem; }
.w-ch08-moe .ctl output { font-variant-numeric: tabular-nums; min-width: 2.2em; }
.w-ch08-moe .toks { display: flex; flex-wrap: wrap; gap: 4px; font: 14px/1.4 var(--mono); margin: .2rem 0 .9rem; }
.w-ch08-moe .tk { padding: 2px 6px 1px; border-radius: 3px; border: 1.5px solid transparent; border-bottom-width: 4px; white-space: pre; color: var(--ink); }
.w-ch08-moe .tk.drop1 { border-style: dashed; border-color: var(--ink-3); }
.w-ch08-moe .tk.gone { text-decoration: line-through; opacity: .65; }
.w-ch08-moe .experts { display: grid; grid-template-columns: 4.2em minmax(0, 1fr) auto; gap: .35rem .6rem; align-items: center; font-size: .86rem; }
.w-ch08-moe .track { position: relative; height: .9rem; background: var(--paper-2); border-radius: 3px; }
.w-ch08-moe .track i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }
.w-ch08-moe .track b { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--ink); }
.w-ch08-moe .nums { font-variant-numeric: tabular-nums; color: var(--ink-2); white-space: nowrap; line-height: 1.3; }
.w-ch08-moe .nums small { display: block; color: var(--ink-3); font-size: .78rem; }
.w-ch08-moe .stats { font-size: .9rem; margin-top: .7rem; }
.w-ch08-moe .stats b { font-variant-numeric: tabular-nums; }
`));
}

export function mount(root, config = {}) {
  injectStyle();
  let k = config.k || 2, cf = config.capacity ?? 1.25, bias = config.bias ?? 0;
  const name = `w-ch08-moe-k-${++instances}`;
  const kRadios = [1, 2].map((v) => {
    const r = h('input', { type: 'radio', name, value: v, checked: v === k, 'aria-label': `每个 token 选 ${v} 个专家` });
    r.addEventListener('change', () => { if (r.checked) { k = v; draw(); } });
    return h('label', {}, r, `top-${v}`);
  });
  const cfRange = h('input', { type: 'range', min: 0.5, max: 2, step: 0.25, value: cf, 'aria-label': 'capacity factor' });
  const biasRange = h('input', { type: 'range', min: 0, max: 4, step: 0.5, value: bias, 'aria-label': 'router 偏向专家 0 的程度' });
  const cfOut = h('output'), biasOut = h('output');
  const toks = h('div', { class: 'toks', role: 'list', 'aria-label': '每个 token 被送去的专家' });
  const experts = h('div', { class: 'experts' });
  const stats = h('div', { class: 'stats', 'aria-live': 'polite' });
  root.append(h('div', { class: 'w-ch08-moe' },
    h('div', { class: 'ctl' },
      h('span', { role: 'radiogroup', 'aria-label': 'top-k' }, kRadios),
      h('label', {}, 'capacity factor', cfRange, cfOut),
      h('label', {}, '偏向专家 0', biasRange, biasOut)),
    toks, experts, stats,
    h('p', { class: 'w-label' }, '色块的底色是第一选择的专家，底边是第二选择的专家；虚线框表示第一选择超出了容量（可能仍由第二选择处理），删除线表示所有选择都被丢弃、只走残差连接。容量 = 每批 token 数 × k ÷ 专家数 × capacity factor（向下取整；k = 1 时就是 Switch Transformer 的定义），先满足所有 token 的第一选择。'),
  ));

  function draw() {
    cfOut.value = cf.toFixed(2);
    biasOut.value = bias.toFixed(1);
    const logits = routerLogits(bias);
    const r = route(logits, { k, capacityFactor: cf });
    const aux = auxLoss(logits);
    const tint = (e, pct) => `color-mix(in srgb, ${seriesColor(e)} ${pct}%, var(--paper))`;
    toks.replaceChildren(...TOKENS.map((tok, t) => {
      const [e1, e2] = r.choices[t];
      const cls = ['tk'];
      if (!r.kept[t][0]) cls.push('drop1');
      if (r.kept[t].every((x) => !x)) cls.push('gone');
      const desc = r.choices[t].map((e, i) => `专家 ${e}（权重 ${r.gates[t][i].toFixed(2)}${r.kept[t][i] ? '' : '，被丢弃'}）`).join('，');
      return h('span', { class: cls.join(' '), role: 'listitem', title: desc, 'aria-label': `${tok.trim() || tok}：${desc}`,
        style: { background: tint(e1, 30), borderBottomColor: e2 != null ? seriesColor(e2) : tint(e1, 30) } }, tok);
    }));
    const T = TOKENS.length;
    experts.replaceChildren(...Array.from({ length: E }, (_, e) => [
      h('span', {}, `专家 ${e}`),
      h('span', { class: 'track', role: 'img', 'aria-label': `专家 ${e}：收到 ${r.assigned[e]}，容量 ${r.capacity}` },
        h('i', { style: { width: `${(100 * r.assigned[e]) / T}%`, background: seriesColor(e) } }),
        h('b', { style: { left: `calc(${(100 * Math.min(r.capacity, T)) / T}% - 1px)` } })),
      h('span', { class: 'nums' }, `收到 ${r.assigned[e]}，容量 ${r.capacity}${r.assigned[e] > r.capacity ? `，丢弃 ${r.assigned[e] - r.capacity}` : ''}`,
        h('small', {}, `f = ${aux.f[e].toFixed(2)}，P = ${aux.P[e].toFixed(2)}`)),
    ]).flat());
    stats.replaceChildren(
      h('div', {}, '辅助 loss：', h('b', {}, `${aux.value.toFixed(2)} α`), '（全部挤到一个专家时为 4α）'),
      h('div', { class: 'w-label' }, '辅助 loss 里的 f 只数第一选择，P 是 router 给出的平均概率。P 接近均匀时它总在 1α 附近，不管 f 怎样；只有 f 和 P 一起偏向同一个专家，它才会升高。'),
      h('div', {}, `丢弃 ${r.dropped} / ${T * k} 次分派；${r.fullyDropped} 个 token 一个专家也没分到。每个 token 只激活 ${k}/${E} 的专家参数。`),
    );
  }

  cfRange.addEventListener('input', () => { cf = +cfRange.value; draw(); });
  biasRange.addEventListener('input', () => { bias = +biasRange.value; draw(); });
  onThemeChange(() => draw());                     // the token tints and expert bars use the palette
  draw();
}
