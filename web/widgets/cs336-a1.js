/* CS336 作业 1 的测试，跑在 DLFS6 的代码上：结果浏览器。
   config: {
     data: "data/cs336/a1_results.json",      // verify/cs336/run_a1.py 的输出（B 层记录）
     notes: { "test_name": "一句话解释", ... },  // 可选：展开后显示
     filter: "failed"                           // 可选：初始筛选（all | failed | dlfs6 | torch | design | none），默认 all
   }
   每一行：测试名、DLFS6 用的是什么、两种变体的结果，以及“实测误差 ÷ 容差”（对数刻度，竖线 = 1，超过即失败）。 */
import { h } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';

const VARIANTS = [
  { key: 'as_shipped', label: '原样' },
  { key: 'one_line_fixes', label: '三处修改后' },
];
const KIND = {
  dlfs6: 'DLFS6 自写', torch: 'PyTorch 内置', design: '设计不同', none: '无对应',
};
const OUTCOME = { passed: '通过', failed: '✗ 失败', skipped: '跳过' };
const FILES = ['test_model.py', 'test_nn_utils.py', 'test_optimizer.py', 'test_data.py', 'test_serialization.py',
  'test_train_bpe.py', 'test_tokenizer.py'];
const LOG_MIN = -4, LOG_MAX = 2; // ratio axis: 1e-4 … 1e2

/** Position (0–100 %) of a tolerance ratio on the log axis; ratios of 0 sit at the left edge. */
export function ratioPos(ratio) {
  if (!(ratio > 0)) return 0;
  const l = Math.min(LOG_MAX, Math.max(LOG_MIN, Math.log10(ratio)));
  return ((l - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 100;
}

export function summarize(tests, key) {
  const out = { passed: 0, failed: 0, skipped: 0 };
  for (const t of tests) out[t[key]?.outcome || 'skipped'] += 1;
  return out;
}

/** A test that failed in at least one of the two variants. */
export const everFailed = (t) => VARIANTS.some((v) => t[v.key]?.outcome === 'failed');

/** The tolerance ratio to draw for one variant's result, or null when the recorded comparison does not decide the
 *  outcome: no numeric comparison at all, or a failure that came from another assertion (ratio within tolerance). */
export function dotFor(result) {
  const w = result?.worst;
  if (!w) return null;
  if (result.outcome === 'failed' && !(w.tol_ratio > 1)) return null;
  return w.tol_ratio;
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-a1 { font-size: .9rem; }
.w-a1-summary { display: flex; flex-wrap: wrap; gap: .4rem 1.4rem; margin: .2rem 0 .7rem; color: var(--ink-2); }
.w-a1-summary b { color: var(--ink); font-variant-numeric: tabular-nums; }
.w-a1-filters { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .8rem; }
.w-a1-filters button, .w-a1-row-head { font: inherit; cursor: pointer; }
.w-a1-filters button { font-size: .82rem; padding: .15rem .6rem; border-radius: 999px; border: 1px solid var(--rule);
  background: transparent; color: var(--ink-2); }
.w-a1-filters button[aria-pressed="true"] { background: var(--ai); border-color: var(--ai); color: var(--paper); }
.w-a1-file { margin: .9rem 0 .25rem; font: 600 .8rem/1.4 var(--mono); color: var(--ink-3); }
.w-a1-row { border-top: 1px solid var(--rule); }
.w-a1-row-head { display: grid; grid-template-columns: minmax(0, 1fr) 6.2em 3.4em 3.4em minmax(5rem, 9rem);
  gap: .2rem .7rem; align-items: center; width: 100%; text-align: left; padding: .35rem .1rem;
  background: transparent; border: 0; color: var(--ink); }
.w-a1-row-head:hover { background: var(--paper-2); }
.w-a1-name { font: .8rem/1.35 var(--mono); overflow-wrap: anywhere; }
.w-a1-kind { font-size: .72rem; color: var(--ink-2); border: 1px solid var(--rule); border-radius: 3px; padding: 0 .3rem;
  text-align: center; white-space: nowrap; }
.w-a1-st { font-size: .78rem; text-align: center; border-radius: 3px; padding: .05rem 0; white-space: nowrap; }
.w-a1-st.passed { color: var(--moss); }
.w-a1-st.failed { color: var(--kincha); font-weight: 600; }
.w-a1-st.skipped { color: var(--ink-3); }
.w-a1-bar { position: relative; height: 1.1rem; background: var(--paper-2); border-radius: 2px; }
.w-a1-bar .tick { position: absolute; top: -2px; bottom: -2px; width: 1px; background: var(--ink-3); left: calc(4.5px + (100% - 9px) * ${ratioPos(1) / 100}); }
.w-a1-bar .dot { position: absolute; top: 50%; width: 9px; height: 9px; margin: -4.5px 0 0 -4.5px; border-radius: 50%; }
.w-a1-bar .dot.as_shipped { background: transparent; border: 2px solid var(--kincha); }
.w-a1-bar .dot.one_line_fixes { background: var(--ai); }
.w-a1-detail { padding: .3rem .2rem .8rem; color: var(--ink-2); font-size: .85rem; line-height: 1.6; }
.w-a1-detail pre { white-space: pre-wrap; overflow-wrap: anywhere; font: .75rem/1.5 var(--mono); background: var(--paper-2);
  padding: .5rem .6rem; border-radius: 3px; margin: .4rem 0; color: var(--ink); }
.w-a1-legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .78rem; color: var(--ink-2); margin: .6rem 0 0; }
.w-a1-legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .35rem; vertical-align: -1px; }
.w-a1-colhead { font-size: .72rem; color: var(--ink-3);
  border-bottom: 1px solid var(--rule); cursor: default; }
.w-a1-colhead:hover { background: var(--paper); }
.w-a1-colhead > span { text-align: center; }
.w-a1-colhead > span:first-child { text-align: left; }
@media (max-width: 560px) {
  .w-a1-row-head { grid-template-columns: 1fr 3.2em 3.2em; }
  .w-a1-name { grid-column: 1 / -1; }
  .w-a1-bar { grid-column: 1 / -1; }
  .w-a1-kind { justify-self: start; }
  .w-a1-colhead > span:first-child, .w-a1-colhead > span:last-child { display: none; }
}
`));
}

function measureText(w) {
  if (!w) return null;
  const tol = `atol ${fmtNum(w.atol)}，rtol ${fmtNum(w.rtol)}`;
  return `最大 |实测 − 期望| = ${fmtNum(w.max_abs_diff)}（${tol}），误差 ÷ 容差 = ${fmtNum(w.tol_ratio)}`;
}

export async function mount(root, config) {
  injectStyle();
  let rec;
  try {
    const res = await fetch(config.data || 'data/cs336/a1_results.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    rec = await res.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `数据加载失败：${err.message}`));
    return;
  }
  const notes = config.notes || {};
  const box = h('div', { class: 'w-a1' });
  root.append(box);

  const sums = VARIANTS.map((v) => {
    const s = summarize(rec.tests, v.key);
    return h('span', {}, `${v.label}：`, h('b', {}, s.passed), ' 通过 · ', h('b', {}, s.failed), ' 失败 · ',
      h('b', {}, s.skipped), ' 跳过');
  });
  box.append(h('div', { class: 'w-a1-summary' }, ...sums));

  const FILTERS = [
    { key: 'all', label: '全部 48 项', test: () => true },
    { key: 'failed', label: '失败过的', test: everFailed },
    ...Object.entries(KIND).map(([k, label]) => ({ key: k, label, test: (t) => t.kind === k })),
  ];
  let filter = FILTERS.some((f) => f.key === config.filter) ? config.filter : 'all';
  const fbtns = FILTERS.map((f) => h('button', { type: 'button', 'aria-pressed': String(f.key === filter) }, f.label));
  box.append(h('div', { class: 'w-a1-filters', role: 'group', 'aria-label': '筛选测试' }, ...fbtns));
  const list = h('div');
  const colhead = h('div', { class: 'w-a1-row-head w-a1-colhead', 'aria-hidden': 'true' },
    h('span', {}, '测试（点击展开细节）'), h('span', {}, 'DLFS6 用的是'),
    ...VARIANTS.map((v) => h('span', {}, v.key === 'as_shipped' ? '原样' : '修改后')), h('span', {}, '误差 ÷ 容差'));
  box.append(colhead, list, h('div', { class: 'w-a1-legend' },
    h('span', {}, h('i', { style: 'border:2px solid var(--kincha)' }), '原样的误差 ÷ 容差'),
    h('span', {}, h('i', { style: 'background:var(--ai)' }), '三处修改后'),
    h('span', {}, '横轴是对数刻度 10⁻⁴ … 10²，竖线 = 1：越过竖线即超出容差'),
    h('span', {}, '没有点的行不是数值比较（精确匹配、计时或功能缺失），或者失败与数值容差无关；看“原样 / 修改后”两列即可')));

  const row = (t) => {
    const detail = h('div', { class: 'w-a1-detail', hidden: true });
    const bar = h('div', { class: 'w-a1-bar', 'aria-hidden': 'true' }, h('span', { class: 'tick' }));
    for (const v of VARIANTS) {
      const ratio = dotFor(t[v.key]);
      if (ratio != null) bar.append(h('span', { class: `dot ${v.key}`,   // keep the 9px dot inside the bar at both ends
        style: `left:calc(4.5px + (100% - 9px) * ${ratioPos(ratio) / 100})` }));
    }
    const head = h('button', { type: 'button', class: 'w-a1-row-head', 'aria-expanded': 'false' },
      h('span', { class: 'w-a1-name' }, t.test),
      h('span', { class: 'w-a1-kind' }, KIND[t.kind] || t.kind),
      ...VARIANTS.map((v) => {
        const o = t[v.key]?.outcome || 'skipped';
        return h('span', { class: `w-a1-st ${o}`, title: v.label }, OUTCOME[o]);
      }),
      bar);
    let built = false;
    head.addEventListener('click', () => {
      const open = head.getAttribute('aria-expanded') !== 'true';
      head.setAttribute('aria-expanded', String(open));
      detail.hidden = !open;
      if (open && !built) {
        built = true;
        detail.append(h('div', {}, `DLFS6 这一侧：${t.dlfs6}（适配函数 ${t.adapter}）`));
        if (notes[t.test]) detail.append(h('div', {}, notes[t.test]));
        for (const v of VARIANTS) {
          const r = t[v.key] || {};
          const m = dotFor(r) != null ? measureText(r.worst) : (r.worst ? '失败来自其他断言，不是数值容差' : null);
          detail.append(h('div', {}, `${v.label}：${OUTCOME[r.outcome] || '—'}${r.duration != null ? `（${r.duration} s）` : ''}`
            + (m ? `；${m}` : '')));
          if (r.message && !(v.key !== 'as_shipped' && r.message === t.as_shipped?.message)) detail.append(h('pre', {}, r.message));
          else if (r.message) detail.append(h('div', { class: 'w-label' }, '（失败信息同上）'));
        }
        const w = t.as_shipped?.worst;
        if (w?.actual && w.actual.length > 4) {
          const chartBox = h('div');
          detail.append(chartBox);
          lineChart(chartBox, {
            series: [
              { name: 'DLFS6 D2Z', points: w.actual.map((y, i) => [i, y]) },
              { name: 'A1 cosine', points: w.desired.map((y, i) => [i, y]) },
            ],
            xLabel: 'it', yLabel: '学习率', height: 200,
          });
        }
      }
    });
    return h('div', { class: 'w-a1-row', 'data-file': t.file }, head, detail);
  };

  const render = () => {
    const f = FILTERS.find((x) => x.key === filter);
    list.replaceChildren();
    for (const file of FILES) {
      const tests = rec.tests.filter((t) => t.file === file && f.test(t));
      if (!tests.length) continue;
      list.append(h('div', { class: 'w-a1-file' }, `${file}（${tests.length}）`), ...tests.map(row));
    }
  };
  fbtns.forEach((b, i) => b.addEventListener('click', () => {
    filter = FILTERS[i].key;
    fbtns.forEach((x, j) => x.setAttribute('aria-pressed', String(i === j)));
    render();
  }));
  render();
  box.append(h('p', { class: 'w-label' },
    `CS336 A1 @ ${rec.cs336_a1.commit.slice(0, 7)} · DLFS6 @ ${rec.dlfs6_commit.slice(0, 7)} · `
    + `PyTorch ${rec.env.torch} · CPU · 记录于 ${rec.generated.slice(0, 10)}`
    + (rec.env.note ? `（${rec.env.note}）` : '')));
}
