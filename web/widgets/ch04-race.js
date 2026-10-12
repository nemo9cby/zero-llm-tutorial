/* 图：优化赛跑。读取 data/ch04/bench.json 里的实测耗时（同一台机器、同一份输入、输出逐项相同），
   条形长度与耗时成正比（可切换对数刻度）；“开跑”按真实耗时的比例压缩成几秒钟的动画。
   config: { data: "data/ch04/bench.json" } */
import { h, fmt } from './_lib.js';
import { seriesColor } from './_plot.js';

const STYLE = `
.w-ch04-race, .w-ch04-race * { font-variant-ligatures: none !important; }
.w-ch04-race .seg { display: flex; flex-wrap: wrap; gap: .35rem; margin: 0 0 .6rem; }
.w-ch04-race button[aria-pressed="true"] { background: var(--ai-2); border-color: var(--ai); color: var(--ink); }
.w-ch04-race .tools { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .2rem 0 .8rem; font-size: .85rem; }
.w-ch04-race .row { display: grid; grid-template-columns: 8.5rem minmax(0, 1fr) 9.6rem; gap: .2rem .7rem; align-items: center; margin: .35rem 0; font-size: .86rem; }
.w-ch04-race .name { color: var(--ink); line-height: 1.3; }
.w-ch04-race .name small { display: block; color: var(--ink-3); font-size: .74rem; }
.w-ch04-race .track { position: relative; height: 1.15rem; background: var(--paper-2); border-radius: 3px; overflow: hidden; }
.w-ch04-race .bar { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; min-width: 2px; }
.w-ch04-race .bar.ext { background-image: repeating-linear-gradient(135deg, transparent 0 5px, color-mix(in srgb, var(--paper) 45%, transparent) 5px 8px); }
.w-ch04-race .val { white-space: nowrap; font: 12.5px/1.3 var(--mono); color: var(--ink-2); font-variant-numeric: tabular-nums; }
.w-ch04-race .val b { color: var(--ink); font-weight: 600; }
.w-ch04-race .note { font-size: .8rem; color: var(--ink-3); margin-top: .6rem; line-height: 1.6; }
.w-ch04-race .ok { color: var(--moss); font-weight: 600; }
.w-ch04-race details { margin-top: .4rem; font-size: .8rem; }
.w-ch04-race table { font-size: .8rem; margin: .3rem 0; }
@media (max-width: 560px) {
  .w-ch04-race .row { grid-template-columns: minmax(0, 1fr) auto; }
  .w-ch04-race .track { grid-column: 1 / -1; grid-row: 2; }
}
`;

function fmtSec(s) {
  if (s >= 100) return `${s.toFixed(0)} s`;
  if (s >= 10) return `${s.toFixed(1)} s`;
  if (s >= 1) return `${s.toFixed(2)} s`;
  return `${(s * 1000).toFixed(0)} ms`;
}

export async function mount(root, config) {
  root.classList.add('w-ch04-race');
  root.append(h('style', {}, STYLE));
  let bench;
  try {
    const r = await fetch(config.data);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    bench = await r.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `读取实测数据失败：${err.message}`));
    return;
  }
  const races = bench.races;
  let current = 0, logScale = false, anim = null;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': '选择一组实验' });
  const runBtn = h('button', { type: 'button', 'aria-label': '按真实耗时的比例播放赛跑动画' }, '开跑 ▶');
  const scaleBtn = h('button', { type: 'button', 'aria-pressed': 'false', 'aria-label': '切换对数刻度' }, '对数刻度');
  const clock = h('span', { class: 'w-label', 'aria-live': 'polite' });
  const rowsBox = h('div');
  const note = h('div', { class: 'note' });
  const table = h('details');
  root.append(seg, h('div', { class: 'tools' }, runBtn, scaleBtn, clock), rowsBox, note, table);

  races.forEach((race, i) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(i === 0) }, race.short || race.title);
    b.addEventListener('click', () => { current = i; stop(); [...seg.children].forEach((x, j) => x.setAttribute('aria-pressed', String(j === i))); render(); });
    seg.append(b);
  });

  const width = (s, max, min) => {
    if (!logScale) return (s / max) * 100;
    const lo = Math.log10(min) - 0.3, hi = Math.log10(max);
    return Math.max(1.5, ((Math.log10(Math.max(s, 1e-9)) - lo) / (hi - lo)) * 100);
  };

  let bars = [];
  function render() {
    const race = races[current];
    const max = Math.max(...race.rows.map((r) => r.seconds));
    const min = Math.min(...race.rows.map((r) => r.seconds));
    const base = race.rows[0].seconds;
    rowsBox.replaceChildren();
    bars = race.rows.map((row, i) => {
      const bar = h('div', { class: `bar${row.ext ? ' ext' : ''}`, style: { background: seriesColor(i), width: `${width(row.seconds, max, min)}%` } });
      const speed = i === 0 ? '基准' : `快 ${fmt(Math.round(base / row.seconds))} 倍`;
      const val = h('div', { class: 'val' }, h('b', {}, fmtSec(row.seconds)), ' · ', speed);
      rowsBox.append(h('div', { class: 'row', role: 'img', 'aria-label': `${row.name}：${fmtSec(row.seconds)}，${speed}` },
        h('div', { class: 'name' }, row.name, row.sub ? h('small', {}, row.sub) : null),
        h('div', { class: 'track' }, bar), val));
      return { bar, row, val };
    });
    const allSame = race.rows.every((r) => r.identical);
    note.replaceChildren(
      h('div', {}, `输入：${race.input}。指标：${race.metric}。`),
      h('div', {}, allSame ? h('span', { class: 'ok' }, '各版本的输出逐项相同 ✓') : '注意：有版本的输出不同', '　', race.extra || ''),
      h('div', {}, `机器：${bench.env.cpu}（${bench.env.physical_cores} 核 ${bench.env.logical_cpus} 线程），${bench.env.os.split('-').slice(0, 2).join(' ')}，Python ${bench.env.python}。`
        + `测量时整机 CPU 占用平均约 ${Math.round(race.load)}%（同机还有其他任务），绝对时间偏慢，比值更可信。`),
    );
    table.replaceChildren(h('summary', {}, '查看数据表'),
      h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, '版本'), h('th', {}, race.metric), h('th', {}, '加速比'), h('th', {}, '输出相同'))),
        h('tbody', {}, race.rows.map((r) => h('tr', {}, h('td', {}, r.name), h('td', {}, r.seconds.toFixed(3)),
          h('td', {}, (base / r.seconds).toFixed(1)), h('td', {}, r.identical ? '是' : '否'))))));
    clock.textContent = '';
  }

  function stop() { if (anim) cancelAnimationFrame(anim); anim = null; }

  runBtn.addEventListener('click', () => {
    stop();
    const race = races[current];
    const max = Math.max(...race.rows.map((r) => r.seconds));
    const min = Math.min(...race.rows.map((r) => r.seconds));
    if (reduce) { render(); clock.textContent = '（已关闭动画：直接显示结果）'; return; }
    const duration = 6000;                       // the slowest version takes 6 s on screen
    const t0 = performance.now();
    const tick = (now) => {
      const sim = Math.min(max, ((now - t0) / duration) * max);
      for (const { bar, row, val } of bars) {
        const s = Math.min(sim, row.seconds);
        bar.style.width = `${width(Math.max(s, min * 0.02), max, min)}%`;
        bar.style.opacity = s >= row.seconds ? '1' : '.55';
        val.firstChild.textContent = fmtSec(s);
      }
      clock.textContent = `模拟时钟 ${fmtSec(sim)}（真实耗时压缩到 6 秒）`;
      if (sim < max) anim = requestAnimationFrame(tick); else { anim = null; render(); clock.textContent = '全部完成'; }
    };
    anim = requestAnimationFrame(tick);
  });

  scaleBtn.addEventListener('click', () => {
    logScale = !logScale;
    scaleBtn.setAttribute('aria-pressed', String(logScale));
    stop();
    render();
  });

  // bar colours come from the light/dark palette: redraw when the theme changes
  const redraw = () => { if (!anim) render(); };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  render();
}
