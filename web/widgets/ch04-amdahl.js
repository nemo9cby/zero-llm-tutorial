/* 图：Amdahl 计算器。数据来自 data/ch04/bench.json（TinyStories 验证集，本机实测）：
   单进程预分词 T_p、各进程数下 Pool 的实测耗时、合并循环 T_s（原书的 max() 版或扩展的堆版）。
   进程开销按实测拟合成 a + b·n；曲线是模型，圆点处是实测。
   config: { data: "data/ch04/bench.json" } */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { amdahl } from './ch04-bpe.js';

const STYLE = `
.w-ch04-amdahl, .w-ch04-amdahl * { font-variant-ligatures: none !important; }
.w-ch04-amdahl .ctl { display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; align-items: center; font-size: .86rem; margin-bottom: .6rem; }
.w-ch04-amdahl input[type=range] { flex: 1 1 10rem; min-width: 9rem; }
.w-ch04-amdahl .stack { margin: .2rem 0 .3rem; }
.w-ch04-amdahl .stackrow { display: grid; grid-template-columns: 5.5rem minmax(0, 1fr); gap: .6rem; align-items: center; margin: .3rem 0; font-size: .8rem; color: var(--ink-2); }
.w-ch04-amdahl .seg { display: flex; height: 1.2rem; border-radius: 3px; overflow: hidden; background: var(--paper-2); }
.w-ch04-amdahl .seg span { display: block; height: 100%; }
.w-ch04-amdahl .keys { display: flex; flex-wrap: wrap; gap: .2rem 1rem; font-size: .78rem; color: var(--ink-2); margin: .2rem 0 .6rem; }
.w-ch04-amdahl .keys i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.w-ch04-amdahl .sum { font-size: .86rem; margin: .3rem 0 .8rem; }
.w-ch04-amdahl .sum b { font-variant-numeric: tabular-nums; }
`;

const COLORS = { pre: 'var(--ai)', over: 'var(--kincha)', loop: 'var(--plum)' };

export async function mount(root, config) {
  root.classList.add('w-ch04-amdahl');
  root.append(h('style', {}, STYLE));
  let p;
  try {
    const r = await fetch(config.data);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    p = (await r.json()).pretok;
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `读取实测数据失败：${err.message}`));
    return;
  }
  const Tp = p.serial_wall;
  // overhead(n) = measured wall(n) - Tp/n, fitted as a + b*n (least squares)
  const pts = p.pool.map(({ n, wall }) => [n, wall - Tp / n]);
  const mx = pts.reduce((s, q) => s + q[0], 0) / pts.length, my = pts.reduce((s, q) => s + q[1], 0) / pts.length;
  const b = pts.reduce((s, q) => s + (q[0] - mx) * (q[1] - my), 0) / pts.reduce((s, q) => s + (q[0] - mx) ** 2, 0);
  const a = my - b * mx;
  const over = (n) => Math.max(0, a + b * n);
  const loops = { book: p.merge_loop_wall, heap: p.heap_loop_wall };

  let n = 8, which = 'book';
  const slider = h('input', { type: 'range', min: 1, max: 16, value: n, 'aria-label': '进程数' });
  const nLabel = h('b');
  const group = `amdahl-${Math.random().toString(36).slice(2)}`;   // one name per widget instance
  const radios = h('span', { role: 'radiogroup', 'aria-label': '合并循环的实现' },
    ...[['book', '合并循环：原书（max）'], ['heap', '合并循环：扩展（堆）']].filter(([k]) => loops[k] != null).map(([k, label]) => {
      const input = h('input', { type: 'radio', name: group, value: k, checked: k === which });
      input.addEventListener('change', () => { which = k; render(); });
      return h('label', { style: { marginRight: '.8rem' } }, input, ' ', label);
    }));
  const stack = h('div', { class: 'stack' });
  const sum = h('div', { class: 'sum', 'aria-live': 'polite' });
  const chart = h('div');
  root.append(
    h('div', { class: 'ctl' }, h('label', {}, '进程数 n = ', nLabel), slider, radios),
    stack,
    h('div', { class: 'keys' }, h('span', {}, h('i', { style: { background: COLORS.pre } }), '预分词（可并行）'),
      h('span', {}, h('i', { style: { background: COLORS.over } }), '进程启动等开销'),
      h('span', {}, h('i', { style: { background: COLORS.loop } }), '合并循环（串行）')),
    sum, chart,
  );

  const ns = [...Array(16).keys()].map((i) => i + 1);
  let chartApi = null;

  function render() {
    const Ts = loops[which];
    nLabel.textContent = String(n);
    const total1 = Tp + Ts;
    const totalN = Tp / n + over(n) + Ts;
    const scale = (t) => `${(t / total1) * 100}%`;
    const bar = (label, parts) => h('div', { class: 'stackrow' }, h('span', {}, label),
      h('div', { class: 'seg', role: 'img', 'aria-label': `${label}：${parts.map(([, t, name]) => `${name} ${t.toFixed(1)} 秒`).join('，')}` },
        parts.map(([color, t]) => h('span', { style: { width: scale(t), background: color } }))));
    stack.replaceChildren(
      bar('单进程', [[COLORS.pre, Tp, '预分词'], [COLORS.loop, Ts, '合并循环']]),
      bar(`${n} 个进程`, [[COLORS.pre, Tp / n, '预分词'], [COLORS.over, over(n), '开销'], [COLORS.loop, Ts, '合并循环']]),
    );
    // a ratio below 1 is a slowdown: say so instead of "faster by 0.9x"
    const gain = (r, digits) => (r < 1 ? ['反而慢了 ', h('b', {}, `${Math.round((1 / r - 1) * 100)}%`)]
      : ['快 ', h('b', {}, `${r.toFixed(digits)} 倍`)]);
    sum.replaceChildren('总时间 ', h('b', {}, `${total1.toFixed(1)} s → ${totalN.toFixed(1)} s`), '，整个训练',
      ...gain(total1 / totalN, 2), '；只看预分词', ...gain(Tp / (Tp / n + over(n)), 1),
      `。进程再多，上限也只有 ${(total1 / Ts).toFixed(2)} 倍。`);
    const measured = p.pool.map(({ n: k, wall }) => [k, Tp / wall]);
    const series = [
      { name: '预分词·实测', points: measured },
      { name: '预分词·模型', points: ns.map((k) => [k, Tp / (Tp / k + over(k))]) },
      { name: `整个训练·模型（${which === 'book' ? '原书' : '堆'}）`, points: ns.map((k) => [k, amdahl(Tp, Ts, k, over(k))]) },
    ];
    if (chartApi) chartApi.update({ series });
    else {
      chartApi = lineChart(chart, { series, height: 240, xLabel: '进程数 n', yLabel: '加速比（倍，相对单进程）', yMin: 0,
        xFormat: (v) => (Number.isInteger(v) ? String(v) : '') });
    }
  }
  slider.addEventListener('input', () => { n = Number(slider.value); render(); });
  render();
}
