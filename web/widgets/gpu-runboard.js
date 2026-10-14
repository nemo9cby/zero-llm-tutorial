/* 运行记录：列出 data/runs/index.json 里的每一次运行，展示 report.json 的溯源信息
   （在哪台机器、什么 GPU 与 PyTorch、哪两个提交、每个脚本的用时与结果、改过什么、花了多少钱），
   再从 data/gpu/plan.json 补上记录这次运行的命令、书里用到它的位置，以及报告里的教程提交不含代码时的真正提交；
   最后列出计划了但还没有记录的运行，附上命令。
   只请求 index.json 里登记过的报告，所以未记录的运行不会产生 404。config: {} */
import { h, fmt } from './_lib.js';

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-gpu-board { font-size: .9rem; }
.w-gpu-board .sum { color: var(--ink-2); margin: 0 0 .8rem; }
.w-gpu-board article { border-top: 1px solid var(--rule); padding: .8rem 0 .4rem; }
.w-gpu-board h4 { margin: 0 0 .3rem; font-size: 1rem; font-family: var(--mono); font-weight: 600; overflow-wrap: anywhere; }
.w-gpu-board .meta { display: flex; flex-wrap: wrap; gap: .2rem .9rem; color: var(--ink-2); font-size: .83rem; margin-bottom: .4rem; }
.w-gpu-board .ok { color: var(--moss); }
.w-gpu-board .bad { color: var(--shu); font-weight: 600; }
.w-gpu-board .warn { color: var(--kincha); }
.w-gpu-board table { margin: .4rem 0 .5rem; font-size: .82rem; }
.w-gpu-board th, .w-gpu-board td { padding: .25rem .8rem .25rem 0; white-space: nowrap; vertical-align: top; }
.w-gpu-board td.wrap { white-space: normal; min-width: 8em; }
.w-gpu-board td.num { text-align: right; font-variant-numeric: tabular-nums; }
.w-gpu-board .line { font-size: .8rem; color: var(--ink-2); margin: .25rem 0 .25rem 1rem; overflow-wrap: anywhere; }
.w-gpu-board .line a { color: inherit; }
.w-gpu-board pre { margin: .2rem 0 .1rem; font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
.w-gpu-board .plan li { margin: .5rem 0; }
.w-gpu-board .legend { font-size: .8rem; color: var(--ink-3); margin: .6rem 0 0; }
@media (max-width: 640px) {
  .w-gpu-board table, .w-gpu-board tbody, .w-gpu-board tr, .w-gpu-board td { display: block; }
  .w-gpu-board thead { display: none; }
  .w-gpu-board tr { padding: .35rem 0; border-top: 1px dotted var(--rule); }
  .w-gpu-board td, .w-gpu-board td.num { padding: .05rem 0; white-space: normal; text-align: left; min-width: 0; }
  .w-gpu-board td[data-label]::before { content: attr(data-label) "："; color: var(--ink-3); }
  .w-gpu-board td.script { font-weight: 600; }
}
`));
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/** The label above a recorded run's command from plan.json. The command carries the recorded run id, and the runner
 *  refuses to append to a report recorded on another kind of machine (on the same machine it would append duplicate
 *  steps), so a reader re-running it needs a new run id. */
export const RECORDED_COMMAND = '记录这次运行的命令（自己重跑时，把 --run 后面换成新的运行名）：';
/** The label's text with the flag set as inline code (in the body font, "--" reads like a dash). */
const withFlag = (text) => text.split('--run').flatMap((part, i) => (i ? [h('code', {}, '--run'), part] : [part]));

const short = (sha) => (sha && sha !== 'unknown' ? sha.slice(0, 7) : '未知');
const stemOf = (script) => script.split('/').pop().replace(/\.py$/, '');
export function duration(s) {
  if (s == null) return '—';
  if (s < 90) return `${s.toFixed(0)} 秒`;
  if (s < 5400) return `${(s / 60).toFixed(1)} 分钟`;
  return `${(s / 3600).toFixed(2)} 小时`;
}
/** "loss 6.139，val_loss 10.98（第 0 步的评估），acc 86.4% @ 500". evalIt: where the val_* values were set. */
export function finalMetrics(final, evalIt = null) {
  if (!final) return '—';
  return Object.entries(final).filter(([k]) => k !== 'it').map(([k, v]) => {
    const num = Number.isInteger(v) ? v : (+v.toPrecision(4));
    if (k === 'acc') return `acc ${num}%`;
    const when = k.startsWith('val_') && evalIt != null && evalIt < final.it ? `（第 ${fmt(evalIt)} 步的评估）` : '';
    return `${k} ${num}${when}`;
  }).join('，') + ` @ ${fmt(final.it)}`;
}
export function where(report) {
  const host = report.host || {};
  if (report.provider === 'modal') return 'Modal';
  const p = host.platform || '';
  if (/windows/i.test(p)) return '本机 Windows';
  return p ? `Linux（${p.split('-')[0]}）` : '未知平台';
}
export function gpuOf(report) {
  const host = report.host || {};
  if (host.gpu) return `${host.gpu_count > 1 ? `${host.gpu_count} × ` : ''}${host.gpu}`;
  if (host.nvidia_smi) return `${host.nvidia_smi.split(',')[0]}（CUDA 不可见）`;
  return 'CPU';
}
export function costOf(report) {
  if (report.cost_usd == null) return '费用未记录';
  if (report.provider === 'modal') return `估算费用 ${report.cost_usd} 美元（容器计时 × 标价，是下限）`;
  return report.cost_usd ? `费用 ${report.cost_usd} 美元` : '费用 0（本机）';
}

/** The iteration from which `key` has held its final value to the end of the metrics: for a val_* metric,
 *  the evaluation the final value came from (the WebBot probe only evaluates at step 0). */
export function settledSince(points, key) {
  let i = points.length - 1;
  while (i > 0 && points[i - 1][key] === points[i][key]) i -= 1;
  return points[i].it;
}

async function evalIterations(run, report, indexed) {
  const out = {};
  await Promise.all((report.steps || []).map(async (s) => {
    const key = Object.keys(s.final || {}).find((k) => k.startsWith('val_'));
    if (!key || !indexed.includes(stemOf(s.script))) return;
    try {
      const r = await fetch(`data/runs/${run}/${stemOf(s.script)}.metrics.jsonl`);
      if (!r.ok) return;
      const pts = (await r.text()).trim().split('\n').map((ln) => JSON.parse(ln));
      if (pts.at(-1)[key] === s.final[key]) out[s.script] = settledSince(pts, key);
    } catch (err) {
      // no annotation: the table still shows the recorded value
    }
  }));
  return out;
}

function card(run, report, pinned, plan, evalAt) {
  if (!report) return h('article', {}, h('h4', {}, run), h('p', { class: 'bad' }, 'index.json 里登记了，但 report.json 读不到。'));
  const host = report.host || {};
  const steps = report.steps || [];
  const pinnedOk = report.dlfs6_commit === pinned;
  const fix = plan?.code_commit;
  const meta = h('div', { class: 'meta' },
    h('span', {}, where(report)),
    h('span', {}, gpuOf(report)),
    host.torch ? h('span', {}, `PyTorch ${host.torch}${host.cuda ? ` · CUDA ${host.cuda}` : ''}`) : null,
    h('span', {}, `教程提交 ${short(report.tutorial_commit)}`,
      fix ? h('span', { class: 'warn' }, `（${fix.why}；代码在 ${short(fix.sha)}）`) : null),
    h('span', {}, '原书提交 ', short(report.dlfs6_commit), ' ',
      h('span', { class: pinnedOk ? 'ok' : 'bad' }, pinnedOk ? '（与子模块一致）' : '（与子模块不一致）')),
    h('span', {}, costOf(report)),
    steps[0]?.started ? h('span', {}, `开始于 ${steps[0].started.slice(0, 16).replace('T', ' ')} UTC`) : null);
  const showRate = steps.some((s) => s.s_per_it || s.rate);
  const cols = ['脚本', '用时', '结果', '最后的指标', showRate ? '后半程 s/it' : null].filter(Boolean);
  const rows = steps.map((s) => h('tr', {},
    h('td', { class: 'script' }, s.script),
    h('td', { class: 'num', 'data-label': '用时' }, duration(s.seconds)),
    h('td', { 'data-label': '结果' },
      s.exit_code === 0 ? h('span', { class: 'ok' }, '完成') : h('span', { class: 'bad' }, `退出码 ${s.exit_code}`)),
    h('td', { class: 'wrap', 'data-label': '最后的指标' }, finalMetrics(s.final, evalAt[s.script])),
    showRate ? h('td', { class: 'num', 'data-label': '后半程 s/it' },
      s.s_per_it ? String(s.s_per_it) : (s.rate ? `${s.rate}（最后读数）` : '—')) : null));
  const table = h('table', {}, h('thead', {}, h('tr', {}, cols.map((t) => h('th', {}, t)))), h('tbody', {}, rows));
  const extras = [];
  for (const s of steps) {
    for (const p of s.patches || []) {
      const diff = [...p.old.split('\n').map((l) => `- ${l}`), ...p.new.split('\n').map((l) => `+ ${l}`)].join('\n');
      extras.push(h('div', { class: 'line' }, `${s.script} 的改动：${p.why}`, h('pre', {}, h('code', {}, diff))));
    }
    if (s.launcher) extras.push(h('div', { class: 'line' }, `${s.script}：用 ${s.launcher} 启动`));
  }
  for (const sd of report.seeded || []) {
    extras.push(h('div', { class: 'line' }, `起点 checkpoint ${sd.path} 来自 ${sd.from}${sd.fix ? `（${sd.fix}）` : ''}`));
  }
  if (plan?.used_in?.length) {
    extras.push(h('div', { class: 'line' }, '用在：',
      ...plan.used_in.flatMap((u, i) => [i ? '、' : '', h('a', { href: u.href }, u.label)])));
  }
  if (plan?.command) {
    extras.push(h('div', { class: 'line' }, ...withFlag(RECORDED_COMMAND), h('pre', {}, h('code', {}, plan.command))));
  }
  return h('article', {}, h('h4', {}, run), meta, steps.length ? table : h('p', {}, '还没有完成的脚本。'), ...extras);
}

export async function mount(root) {
  injectStyle();
  let index, plan, bots;
  try {
    [index, plan, bots] = await Promise.all([getJSON('data/runs/index.json'), getJSON('data/gpu/plan.json'),
      getJSON('data/gpu/bots.json')]);
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `运行记录没有加载成功：${err.message}`));
    return;
  }
  const runs = Object.keys(index);
  const planned = Object.fromEntries(plan.planned.map((p) => [p.run, p]));
  const reports = await Promise.all(runs.map((r) => getJSON(`data/runs/${r}/report.json`).catch(() => null)));
  const evalAts = await Promise.all(runs.map((r, i) => (reports[i] ? evalIterations(r, reports[i], index[r].steps || []) : {})));
  const nSteps = reports.reduce((n, rep) => n + (rep?.steps?.length || 0), 0);
  const cost = reports.reduce((c, rep) => c + (rep?.cost_usd || 0), 0);
  const box = h('div', { class: 'w-gpu-board' });
  box.append(h('p', { class: 'sum' }, runs.length
    ? `已记录 ${runs.length} 次运行、${nSteps} 个脚本；云端费用合计约 ${cost.toFixed(2)} 美元（按容器计时 × 标价估算，是下限）。`
    : '还没有记录任何运行。下面任意一条命令跑完后重新构建，记录就会出现在这里。'));
  runs.forEach((r, i) => box.append(card(r, reports[i], bots.dlfs6_commit, planned[r], evalAts[i])));

  const allSteps = reports.flatMap((rep) => rep?.steps || []);
  const legend = [];
  if (allSteps.some((s) => s.s_per_it)) legend.push('后半程 s/it：用 tqdm 的计时算出的后半程平均每步（GRPO 是每轮）秒数，包含这段时间里的评估。');
  if (allSteps.some((s) => s.final && 'acc' in s.final)) legend.push('acc：81 道训练题上贪心解码的准确率（每 10 轮评估一次），不是留出集上的成绩。');
  if (legend.length) box.append(h('p', { class: 'legend' }, legend.join(' ')));

  const pending = plan.planned.filter((p) => !runs.includes(p.run));
  if (pending.length) {
    box.append(h('p', { class: 'sum', style: 'margin-top:1rem' }, '计划中、尚未记录的运行：'));
    box.append(h('ul', { class: 'plan' }, pending.map((p) => h('li', {},
      h('b', {}, p.run), `：${p.what}（${p.where}，估计 ${p.estimate_usd[1] ? `${p.estimate_usd[0]}–${p.estimate_usd[1]} 美元` : '0 美元'}）`,
      h('pre', {}, h('code', {}, p.command))))));
  }
  root.append(box);
}
