/* 图：缩小版 WebBot 在 CPU 上的真实训练记录。
   数据来自 verify/_ch09_shrunk_run.py：原书 ch09/01_pretrain.py 只改了模型尺寸、步数等几行（列表见“改动”），
   wandb 换成记录调用的替身，所以下面的每个点都是脚本自己 wandb.log 出来的值。
   config: { run: "webbot-cpu-shrunk", samples: true } */
import { h } from './_lib.js';
import { lineChart, fmtNum } from './_plot.js';

async function get(url, as = 'json') {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return as === 'json' ? r.json() : r.text();
}

/** 换行显示成 ↵，连续 n 个显示成 ↵×n，免得几十个空行把版面撑长。 */
export function compactNewlines(text) {
  return text.replace(/\n+/g, (m) => (m.length === 1 ? ' ↵ ' : ` ↵×${m.length} `));
}

export function parseCalls(text) {
  const calls = text.split('\n').filter(Boolean).map((l) => JSON.parse(l));
  const cfg = calls.find((c) => c.call === 'config.update')?.data || {};
  const train = calls.filter((c) => c.call === 'log' && 'train/loss' in c.data);
  const val = calls.filter((c) => c.call === 'log' && 'val/loss' in c.data);
  return { calls, cfg, train, val };
}

export async function mount(root, config) {
  const base = `data/ch09/${config.run}/`;
  let report, log, samples = null;
  try {
    [report, log] = await Promise.all([get(base + 'report.json'), get(base + 'wandb.jsonl', 'text')]);
    if (config.samples) samples = await get(base + 'samples.json');
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `运行记录 ${config.run} 读取失败（${err.message}）。`));
    return;
  }
  const { cfg, train, val } = parseCalls(log);
  const lossBox = h('div'), lrBox = h('div');
  root.append(lossBox);
  lineChart(lossBox, {
    series: [
      { name: 'train/loss（每步）', points: train.map((c) => [c.step, c.data['train/loss']]) },
      { name: 'val/loss', points: val.map((c) => [c.step, c.data['val/loss']]) },
    ],
    xLabel: 'step', yLabel: 'loss（nats）', height: 250,
    refLines: [{ y: Math.log(cfg.vocab_size || 50000), label: `ln ${(cfg.vocab_size || 50000).toLocaleString('en-US')}` }],
  });
  if (config.lr !== false) {
    root.append(lrBox);
    lineChart(lrBox, {
      series: [{ name: 'train/lr', points: train.map((c) => [c.step, c.data['train/lr']]) }],
      xLabel: 'step', yLabel: '学习率 train/lr', height: 150, yMin: 0, table: false,
    });
  }

  const last = val[val.length - 1];
  const host = report.host || {};
  const minutes = (report.seconds?.pretrain ?? 0) / 60;
  const tokensPerStep = cfg.micro_batch_size * cfg.accumulation_steps * cfg.context_len * (report.world_size || 1);
  root.append(
    h('div', { class: 'w-label' },
      `val/loss ${fmtNum(val[0].data['val/loss'])} → ${fmtNum(last.data['val/loss'])}（第 ${last.step} 步）；`
      + `每步 ${tokensPerStep.toLocaleString('en-US')} 个 token，共 ${train.length} 步。`
      + (report.world_size > 1
        ? ` ${report.world_size} 个进程，wandb 调用次数按 rank：${Object.entries(report.wandb_calls_per_rank).map(([r, n]) => `rank ${r} ${n} 次`).join('，')}；`
          + `记下的 effective_batch_size = ${cfg.effective_batch_size}，实际全局 batch = ${cfg.effective_batch_size * report.world_size} 条序列。`
        : '')),
    h('div', { class: 'w-label', style: 'margin-top:.3rem' },
      [`CPU：${host.processor || host.platform}（${host.cpu_count} 个逻辑核）`, `PyTorch ${host.torch}`,
        `用时 ${minutes.toFixed(1)} 分钟（机器同时在跑别的任务）`, '费用 0 美元',
        `原书代码 ${String(report.dlfs6_commit).slice(0, 7)}`, `运行于 ${String(report.started).slice(0, 10)}`].join('；')),
    h('details', {}, h('summary', { class: 'w-label' }, `相对原书改动的 ${report.patches.length} 行（其余一字未改）`),
      h('ul', { style: 'font-size:.82rem;margin:.4rem 0;padding-left:1.2rem' },
        report.patches.map((p) => h('li', {}, h('code', {}, p.file), '：', h('code', {}, p.old), ' → ', h('code', {}, p.new)))),
      h('p', { class: 'w-label' }, `复现：${report.command}`)),
  );
  if (samples) {
    root.append(h('details', {},
      h('summary', { class: 'w-label' }, `02_generate.py 用这个缩小版模型续写的 5 段文字（temperature ${samples.temperature}，seed ${samples.seed}）`),
      h('p', { class: 'w-label' }, '续写里有大段的连续换行，这里把换行显示成 ↵，连续 n 个显示成 ↵×n；其余文字原样照录。'),
      samples.samples.map((s) => h('div', { style: 'margin:.5rem 0' },
        h('div', { class: 'w-label' }, h('b', {}, s.prompt)),
        h('pre', { style: 'white-space:pre-wrap;word-break:break-word;font:12.5px/1.6 var(--mono);background:var(--paper-2);padding:.5rem .7rem;border-radius:4px;margin:.2rem 0' },
          compactNewlines(s.text))))));
  }
}
