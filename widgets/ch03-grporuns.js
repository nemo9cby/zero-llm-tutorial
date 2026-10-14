/* 图：本书在 RTX 3060 Ti 上的两次 GRPO 运行，从同一个 SFT checkpoint 出发：原书脚本原样，与更新前加一行
   model.eval()。两条曲线画在同一张图上，数据直接读 gpu/run_local.py 写下的运行记录
   （data/runs/<run>/report.json 与 <step>.metrics.jsonl），图下给出每次运行的溯源。
   config: { runs: [{ run, label }], step: "09_grpo", metric: "acc" | "loss", yLabel } */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle } from './ch03-common.js';

/** metrics.jsonl 的文本 → [{it, ...}]（容忍 CRLF 与空行）。 */
export function parseMetrics(text) {
  return text.split('\n').map((l) => l.trim()).filter(Boolean).map((l) => JSON.parse(l));
}

/** 准确率曲线的要点：第 51 轮的值、最终值、最终值第一次出现的轮次、从哪一轮起一直是 100%。 */
export function accSummary(rows) {
  const pts = rows.filter((r) => r.acc != null);
  const at = (it) => pts.find((r) => r.it === it)?.acc;
  const final = pts[pts.length - 1].acc;
  const firstFinal = pts.find((r) => r.acc === final).it;
  let allFrom = null;                      // 从这一轮起，之后每一个记录都是 100%
  for (let i = pts.length - 1; i >= 0 && pts[i].acc === 100; i--) allFrom = pts[i].it;
  return { start: pts[0].acc, at51: at(51), final, firstFinal, allFrom };
}

/** loss 曲线的要点：平均、中位数、最大值及其轮次。 */
export function lossSummary(rows) {
  const v = rows.filter((r) => r.loss != null);
  const xs = v.map((r) => r.loss).sort((a, b) => a - b);
  const n = xs.length;
  const median = n % 2 ? xs[(n - 1) / 2] : (xs[n / 2 - 1] + xs[n / 2]) / 2;
  const top = v.reduce((a, b) => (b.loss > a.loss ? b : a));
  return { mean: xs.reduce((a, b) => a + b, 0) / n, median, max: top.loss, maxIt: top.it };
}

/** 运行报告里的一处补丁 {old, new} → “新增的代码（加在 原来那一行 之前）”。 */
export function patchText(p) {
  const added = p.new.endsWith(p.old) ? p.new.slice(0, p.new.length - p.old.length) : p.new;
  const code = added.trim().split('\n')[0].split('#')[0].trim();
  return p.new.endsWith(p.old) ? `${code}（加在 ${p.old.trim()} 之前）` : code;
}

const pct = (x) => `${x.toFixed(1)}%`;
const sgn = (x, d = 3) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(d)}`;

function summaryText(metric, label, rows) {
  if (metric === 'acc') {
    const s = accSummary(rows);
    return `${label}：第 51 轮 ${pct(s.at51)}，最终 ${pct(s.final)}`
      + (s.allFrom != null ? `，第 ${s.allFrom} 轮起一直是 100%` : `（第 ${s.firstFinal} 轮第一次达到）`);
  }
  const s = lossSummary(rows);
  return `${label}：loss 平均 ${sgn(s.mean)}，中位数 ${sgn(s.median)}，最高 ${sgn(s.max, 2)}（第 ${s.maxIt} 轮）`;
}

function provenance(label, report, stem) {
  const host = report.host || {};
  const step = (report.steps || []).find((s) => s.script.endsWith(`${stem}.py`)) || {};
  const bits = [
    host.gpu ? `${host.gpu}` : null,
    host.torch ? `PyTorch ${host.torch}` : null,
    step.seconds ? `用时 ${(step.seconds / 60).toFixed(1)} 分钟` : null,
    report.dlfs6_commit ? `原书代码 ${report.dlfs6_commit.slice(0, 7)}` : null,
    step.patches?.length ? `补丁：${step.patches.map(patchText).join('；')}` : '脚本原样，没有补丁',
    ...(report.seeded || []).map((s) => `${s.path} 来自 ${s.from.replace('tutorial run ', '本书的运行 ')}（sha256 ${s.sha256.slice(0, 8)}…）`),
  ].filter(Boolean);
  return h('div', { class: 'w-label prov' }, h('b', {}, `${label}（${report.run}）`), '：', bits.join('；'));
}

export async function mount(root, config) {
  injectStyle();
  root.classList.add('w-ch03');
  root.append(h('style', {}, '.w-ch03-runs .prov { margin-top: .25rem; overflow-wrap: anywhere; }'));
  root.classList.add('w-ch03-runs');
  const stem = config.step || '09_grpo';
  const metric = config.metric || 'acc';
  let loaded;
  try {
    loaded = await Promise.all(config.runs.map(async ({ run, label }) => {
      const [rep, met] = await Promise.all([fetch(`data/runs/${run}/report.json`), fetch(`data/runs/${run}/${stem}.metrics.jsonl`)]);
      if (!rep.ok || !met.ok) throw new Error(`${run}：HTTP ${rep.ok ? met.status : rep.status}`);
      return { label, report: await rep.json(), rows: parseMetrics(await met.text()) };
    }));
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `运行记录读取失败（${err.message}）。记录这两次运行的命令（自己重跑时，把 `,
      h('code', {}, '--run'), ' 后面换成新的运行名）：'),
      h('pre', { class: 'cmd' }, h('code', {}, 'uv run python gpu/run_local.py --run codebot-3060ti --job codebot\n'
        + 'uv run python gpu/run_local.py --run codebot-grpo-evalfix-3060ti --job codebot-grpo-evalfix')));
    return;
  }
  const chartBox = h('div');
  root.append(chartBox);
  lineChart(chartBox, {
    series: loaded.map(({ label, rows }) => ({ name: label, points: rows.filter((r) => r[metric] != null).map((r) => [r.it, r[metric]]) })),
    xLabel: 'iteration（轮）', yLabel: config.yLabel || metric, height: 240,
    ...(metric === 'acc' ? { yMax: 100 } : {}),
    ariaLabel: `${loaded.map((x) => x.label).join(' 与 ')}：${config.yLabel || metric}`,
  });
  root.append(...loaded.map(({ label, rows }) => h('div', { class: 'stat' }, summaryText(metric, label, rows))));
  root.append(...loaded.map(({ label, report }) => provenance(label, report, stem)));
}
