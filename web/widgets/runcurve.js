/* 训练运行回放：读取 data/runs/<run>/ 下的运行报告与指标，画曲线并给出溯源信息。
   config: {
     run: "codebot-3060ti",                  // web/data/runs/<run>/
     step: "01_pretrain",                     // which script's metrics (<stem>.metrics.jsonl)
     metrics: ["loss"],                       // postfix keys to plot (one chart, same unit)
     yLabel: "loss（nats）", xLabel: "iteration",
     refLines: [{ "y": 6.9078, "label": "ln 1000" }],
     command: "uv run python gpu/run_local.py --run codebot-3060ti --script ch03/01_pretrain.py",
     maxPoints: 600                           // downsample for display
   }
   If the run has not been recorded yet, the widget says so and shows the command instead of a chart. */
import { h } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function downsample(points, n) {
  if (points.length <= n) return points;
  const step = points.length / n;
  const out = [];
  for (let i = 0; i < n; i++) out.push(points[Math.floor(i * step)]);
  out.push(points[points.length - 1]);
  return out;
}

function provenance(report, stepInfo) {
  const host = report.host || {};
  const bits = [
    host.gpu ? `GPU：${host.gpu}` : `平台：${host.platform || '未知'}`,
    host.torch ? `PyTorch ${host.torch}` : null,
    stepInfo ? `用时 ${(stepInfo.seconds / 60).toFixed(1)} 分钟` : null,
    `费用 ${report.cost_usd != null ? `${report.cost_usd} 美元` : '未记录'}`,
    report.dlfs6_commit ? `原书代码 ${report.dlfs6_commit.slice(0, 7)}` : null,
    stepInfo?.started ? `运行于 ${stepInfo.started.slice(0, 10)}` : null,
  ].filter(Boolean);
  return h('div', { class: 'w-label', style: 'margin-top:.3rem' }, bits.join('；'));
}

export async function mount(root, config) {
  const base = `data/runs/${config.run}/`;
  const stem = config.step;
  let report, text;
  try {
    // index.json lists recorded runs, so a missing run is a message, not a 404.
    const index = await getJSON('data/runs/index.json');
    if (!index[config.run]?.steps?.includes(stem)) throw new Error('not recorded');
    report = await getJSON(base + 'report.json');
    const r = await fetch(`${base}${stem}.metrics.jsonl`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    text = await r.text();
  } catch (err) {
    root.append(h('p', { class: 'w-label' },
      `这次运行还没有记录（${config.run} / ${stem}）。在本机或云端运行下面的命令后重新构建，曲线就会出现在这里：`),
    h('pre', { class: 'cmd' }, h('code', {}, config.command || `uv run python gpu/run_local.py --run ${config.run} ...`)));
    return;
  }
  const rows = text.split('\n').filter(Boolean).map((l) => JSON.parse(l));
  const keys = config.metrics || ['loss'];
  const series = keys.map((k) => ({
    name: k, points: downsample(rows.filter((p) => p[k] != null).map((p) => [p.it, p[k]]), config.maxPoints || 600),
  })).filter((s) => s.points.length);
  if (!series.length) {
    root.append(h('p', { class: 'w-label' }, `运行记录里没有指标 ${keys.join('、')}。`));
    return;
  }
  const chartBox = h('div');
  root.append(chartBox);
  lineChart(chartBox, {
    series, xLabel: config.xLabel || 'iteration', yLabel: config.yLabel || keys.join(' / '),
    refLines: config.refLines, yMin: config.yMin, yMax: config.yMax, logY: config.logY,
  });
  const stepInfo = (report.steps || []).find((s) => s.script.endsWith(`${stem}.py`));
  const last = series.map((s) => `${s.name} 最终 ${fmtNum(s.points[s.points.length - 1][1])}`).join('，');
  root.append(h('div', { class: 'w-label' }, last), provenance(report, stepInfo));
}
