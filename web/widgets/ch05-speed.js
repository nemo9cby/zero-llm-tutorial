/* 图：KV cache 的逐步耗时（记录在 data/ch05/kv_timing_<device>.json，由 verify/_ch05_kvbench.py 生成）。
   上：第 t 步的耗时，全部重算 vs. 只 decode 新 token；下：两者之比（实测的单步加速）与计算量之比 t。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle } from './ch05-common.js';

/** Measured per-step speedup full/decode and the FLOP ratio (≈ t tokens recomputed vs 1). */
export function ratios(perStep) {
  return perStep.map((r) => ({ t: r.t, measured: r.full_ms / r.decode_ms, flops: r.t }));
}

export async function mount(root, config = {}) {
  injectStyle();
  const box = h('div', { class: 'w-ch05' });
  root.append(box);
  let rec;
  try {
    const r = await fetch(config.data || 'data/ch05/kv_timing_cpu.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    rec = await r.json();
  } catch (err) {
    box.append(h('p', { class: 'w-label' }, `测速记录读取失败（${err.message}）。运行 uv run python verify/_ch05_kvbench.py 生成它。`));
    return;
  }
  const steps = rec.per_step;
  const top = h('div'), bottom = h('div');
  box.append(h('div', { class: 'panel-title' }, '第 t 步的耗时（毫秒）'), top,
    h('div', { class: 'panel-title', style: 'margin-top:.8rem' }, '单步加速：全部重算 ÷ decode 一步（对数坐标）'), bottom);
  lineChart(top, {
    series: [
      { name: '全部重算', points: steps.map((r) => [r.t, r.full_ms]) },
      { name: 'decode', points: steps.map((r) => [r.t, r.decode_ms]) },
    ],
    xLabel: 't（上下文长度）', yLabel: '毫秒', yMin: 0, height: 230,
    yFormat: (v) => v.toFixed(0), ariaLabel: '逐步耗时随上下文长度的变化',
  });
  const rs = ratios(steps);
  lineChart(bottom, {
    series: [
      { name: '计算量 t', points: rs.map((r) => [r.t, r.flops]) },
      { name: '实测', points: rs.map((r) => [r.t, +r.measured.toFixed(2)]) },
    ],
    xLabel: 't（上下文长度）', yLabel: '倍', logY: true, height: 230,
    yFormat: (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1)), ariaLabel: '实测加速比与计算量之比',
  });
  const last = rs[rs.length - 1];
  const load = rec.cpu_load_before_percent != null ? `；测量前 CPU 占用 ${rec.cpu_load_before_percent}%` : '';
  box.append(h('div', { class: 'readout', style: 'margin-top:.4rem' },
    `t = ${last.t} 时，计算量相差 ${last.flops} 倍，实测只快 ${last.measured.toFixed(1)} 倍。`),
  h('div', { class: 'w-label' },
    `记录：${rec.device_name}，${rec.device === 'cpu' ? `${rec.threads} 个线程，` : ''}PyTorch ${rec.torch}，${rec.date}${load}。`
    + `命令：${rec.command}`));
}
