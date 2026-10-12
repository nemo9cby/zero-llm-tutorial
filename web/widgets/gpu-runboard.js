/* 运行记录：列出 data/runs/index.json 里的每一次运行，展示 report.json 的溯源信息
   （在哪台机器、什么 GPU 与 PyTorch、哪两个提交、每个脚本的用时与结果、改过什么、花了多少钱），
   再列出 data/gpu/plan.json 里计划了但还没有记录的运行，附上命令。
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
.w-gpu-board table { margin: .4rem 0 .5rem; font-size: .82rem; }
.w-gpu-board th, .w-gpu-board td { padding: .25rem .8rem .25rem 0; white-space: nowrap; }
.w-gpu-board td.wrap { white-space: normal; min-width: 8em; }
.w-gpu-board td.num { text-align: right; font-variant-numeric: tabular-nums; }
.w-gpu-board .patch { font-size: .8rem; color: var(--ink-2); margin: .15rem 0 .15rem 1rem; overflow-wrap: anywhere; }
.w-gpu-board .patch code { font-size: .78rem; }
.w-gpu-board .plan li { margin: .5rem 0; }
.w-gpu-board pre.cmd { margin: .3rem 0 .2rem; font-size: 12.5px; white-space: pre-wrap; overflow-wrap: anywhere; }
`));
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

const short = (sha) => (sha && sha !== 'unknown' ? sha.slice(0, 7) : '未知');
export function duration(s) {
  if (s == null) return '—';
  if (s < 90) return `${s.toFixed(0)} 秒`;
  if (s < 5400) return `${(s / 60).toFixed(1)} 分钟`;
  return `${(s / 3600).toFixed(2)} 小时`;
}
export function finalMetrics(final) {
  if (!final) return '—';
  return Object.entries(final).filter(([k]) => k !== 'it')
    .map(([k, v]) => `${k} ${Number.isInteger(v) ? v : (+v.toPrecision(4))}`).join('，') + ` @ ${fmt(final.it)}`;
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

function card(run, report, pinned) {
  if (!report) return h('article', {}, h('h4', {}, run), h('p', { class: 'bad' }, 'index.json 里登记了，但 report.json 读不到。'));
  const host = report.host || {};
  const steps = report.steps || [];
  const pinnedOk = report.dlfs6_commit === pinned;
  const meta = h('div', { class: 'meta' },
    h('span', {}, where(report)),
    h('span', {}, gpuOf(report)),
    host.torch ? h('span', {}, `PyTorch ${host.torch}${host.cuda ? ` · CUDA ${host.cuda}` : ''}`) : null,
    h('span', {}, `教程提交 ${short(report.tutorial_commit)}`),
    h('span', {}, '原书提交 ', short(report.dlfs6_commit), ' ',
      h('span', { class: pinnedOk ? 'ok' : 'bad' }, pinnedOk ? '（与子模块一致）' : '（与子模块不一致）')),
    h('span', {}, `费用 ${report.cost_usd != null ? `${report.cost_usd} 美元` : '未记录'}`),
    steps[0]?.started ? h('span', {}, `开始于 ${steps[0].started.slice(0, 16).replace('T', ' ')} UTC`) : null);
  const showRate = steps.some((s) => s.s_per_it || s.rate);
  const showGpu = steps.some((s) => s.gpu);
  const cols = ['脚本', '用时', '结果', '最后的指标', showRate ? '吞吐' : null, showGpu ? 'GPU' : null].filter(Boolean);
  const rows = steps.map((s) => h('tr', {},
    h('td', {}, s.script),
    h('td', { class: 'num' }, duration(s.seconds)),
    h('td', {}, s.exit_code === 0 ? h('span', { class: 'ok' }, '完成') : h('span', { class: 'bad' }, `退出码 ${s.exit_code}`)),
    h('td', { class: 'wrap' }, finalMetrics(s.final)),
    showRate ? h('td', { class: 'num' }, s.s_per_it ? `${s.s_per_it} s/it` : (s.rate || '—')) : null,
    showGpu ? h('td', {}, s.gpu || '') : null));
  const table = h('table', {}, h('thead', {}, h('tr', {}, cols.map((t) => h('th', {}, t)))), h('tbody', {}, rows));
  const extras = [];
  for (const s of steps) {
    for (const p of s.patches || []) {
      extras.push(h('div', { class: 'patch' }, `${s.script}：`, h('code', {}, p.old), ' → ', h('code', {}, p.new), `（${p.why}）`));
    }
    if (s.launcher) extras.push(h('div', { class: 'patch' }, `${s.script}：用 ${s.launcher} 启动`));
  }
  for (const sd of report.seeded || []) {
    extras.push(h('div', { class: 'patch' }, `起点检查点 ${sd.path} 来自 ${sd.from}${sd.fix ? `（${sd.fix}）` : ''}`));
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
  const reports = await Promise.all(runs.map((r) => getJSON(`data/runs/${r}/report.json`).catch(() => null)));
  const nSteps = reports.reduce((n, rep) => n + (rep?.steps?.length || 0), 0);
  const cost = reports.reduce((c, rep) => c + (rep?.cost_usd || 0), 0);
  const box = h('div', { class: 'w-gpu-board' });
  box.append(h('p', { class: 'sum' }, runs.length
    ? `已记录 ${runs.length} 次运行、${nSteps} 个脚本，云端费用合计 ${cost.toFixed(2)} 美元。`
    : '还没有记录任何运行。下面任意一条命令跑完后重新构建，记录就会出现在这里。'));
  runs.forEach((r, i) => box.append(card(r, reports[i], bots.dlfs6_commit)));
  const pending = plan.planned.filter((p) => !runs.includes(p.run));
  if (pending.length) {
    box.append(h('p', { class: 'sum', style: 'margin-top:1rem' }, '计划中、尚未记录的运行：'));
    box.append(h('ul', { class: 'plan' }, pending.map((p) => h('li', {},
      h('b', {}, p.run), `：${p.what}（${p.where}，估计 ${p.estimate_usd[1] ? `${p.estimate_usd[0]}–${p.estimate_usd[1]} 美元` : '0 美元'}）`,
      h('pre', { class: 'cmd' }, h('code', {}, p.command))))));
  }
  root.append(box);
}
