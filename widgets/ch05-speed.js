/* 图：KV cache 的逐步耗时（记录在 data/ch05/kv_timing_<device>.json，由 verify/_ch05_kvbench.py 生成）。
   上：第 t 步的耗时，全部重算 vs. 只 decode 新 token；下：两者之比（实测的单步加速）与计算量之比 t。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle } from './ch05-common.js';

/** Measured per-step speedup full/decode and the FLOP ratio (≈ t tokens recomputed vs 1). */
export function ratios(perStep) {
  return perStep.map((r) => ({ t: r.t, measured: r.full_ms / r.decode_ms, flops: r.t }));
}

/** "快 4.9 倍"; near 1 "只快 3%" (never the misleading "快 1.0 倍"); below 1 "反而慢（0.84 倍）". */
export function speedupText(x) {
  if (x < 1) return `反而慢（${x.toFixed(2)} 倍）`;
  if (x < 1.2) return Math.round((x - 1) * 100) === 0 ? '几乎一样快' : `只快 ${Math.round((x - 1) * 100)}%`;
  return `快 ${x.toFixed(1)} 倍`;
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
  const tMax = Math.max(...steps.map((r) => r.t));
  const xTicks = [1, ...[1, 2, 3, 4].map((k) => Math.round((k * tMax) / 4))];   // 1, 200, 400, 600, 800
  const top = h('div'), bottom = h('div');
  box.append(h('div', { class: 'panel-title' }, '第 t 步的耗时（毫秒）'), top,
    h('div', { class: 'panel-title', style: 'margin-top:.8rem' }, '单步加速：全部重算 ÷ decode 一步（对数坐标）'), bottom);
  lineChart(top, {
    series: [
      { name: '全部重算', points: steps.map((r) => [r.t, r.full_ms]) },
      { name: 'decode', points: steps.map((r) => [r.t, r.decode_ms]) },
    ],
    xLabel: 't（上下文长度）', yLabel: '毫秒', yMin: 0, height: 230, xTicks,
    yFormat: (v) => v.toFixed(0), ariaLabel: '逐步耗时随上下文长度的变化',
  });
  const rs = ratios(steps);
  lineChart(bottom, {
    series: [
      { name: '计算量 t', points: rs.map((r) => [r.t, r.flops]) },
      { name: '实测', points: rs.map((r) => [r.t, +r.measured.toFixed(2)]) },
    ],
    xLabel: 't（上下文长度）', yLabel: '倍', logY: true, height: 230, xTicks,
    yFormat: (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1)), ariaLabel: '实测加速比与计算量之比',
  });
  const last = rs[rs.length - 1];
  const gen = rec.generate?.[rec.generate.length - 1];
  const loads = [rec.cpu_load_before_percent, rec.cpu_load_after_percent];
  const load = loads[0] == null ? '' : loads[1] == null ? `；测量前 CPU 占用 ${loads[0]}%` : `；测量前后 CPU 占用 ${loads[0]}% / ${loads[1]}%`;
  const reps = rec.gen_repeats ? `（${rec.gen_repeats} 次运行的中位数）` : '（只计时一次）';
  box.append(h('div', { class: 'readout', style: 'margin-top:.4rem' },
    h('div', {}, `t = ${last.t} 时，计算量相差 ${last.flops} 倍，decode 一步实测${last.measured >= 1.2 ? '只' : ''}${speedupText(last.measured)}。`),
    gen ? h('div', {}, `生成 ${gen.n} 个 token：不用 cache ${gen.no_cache_s.toFixed(2)} 秒，用 cache ${gen.cache_s.toFixed(2)} 秒${reps}，`
      + `用 cache ${speedupText(gen.no_cache_s / gen.cache_s)}。`) : null),
  h('div', { class: 'w-label' },
    `记录：${rec.device_name}，${rec.device === 'cpu' ? `${rec.threads} 个线程，` : ''}PyTorch ${rec.torch}，${rec.date}${load}。`
    + `命令：${rec.command}`));
}
