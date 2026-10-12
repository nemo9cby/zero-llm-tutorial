/* 勘误一览表的筛选器：按严重度、所属部分、是否影响书中结果过滤页面上的表格行。
   config: { table: "#errata-index table" }  —— 表格本身写在 Markdown 里，组件只负责显示/隐藏行。
   列的顺序：编号 | 问题 | 位置 | 严重度 | 影响书中结果 */
import { h } from './_lib.js';

const PARTS = [
  { key: 'all', label: '全部' },
  { key: 'I', label: '第 I 部 CodeBot', test: /ch0[123]|codebot/ },
  { key: 'II', label: '第 II 部 StoryBot', test: /ch0[456]|storybot/ },
  { key: 'III', label: '第 III 部 WebBot', test: /ch0[789]|webbot/ },
  { key: 'repo', label: '仓库与文档', test: /README|HF|download\.sh/ },
];
const SEVERITIES = ['高', '中', '低'];
const EFFECTS = [
  { key: 'all', label: '全部' },
  { key: 'yes', label: '可能影响' },
  { key: 'no', label: '不影响' },
];

/** Classify the “影响书中结果” cell: anything that is not a plain “否” / “几乎没有” may change a number. */
export function effectOf(text) {
  const t = text.trim();
  if (t.startsWith('否') || t.startsWith('几乎没有')) return 'no';
  return 'yes';
}

export function partsOf(location) {
  return PARTS.filter((p) => p.test && p.test.test(location)).map((p) => p.key);
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-errata-controls { display: grid; gap: .55rem; font-size: .9rem; }
.w-errata-group { display: flex; flex-wrap: wrap; align-items: center; gap: .35rem .4rem; }
.w-errata-group > span { color: var(--ink-2); min-width: 5.5em; }
.w-errata-controls button { font: inherit; font-size: .85rem; padding: .18rem .6rem; border-radius: 999px;
  border: 1px solid var(--rule); background: transparent; color: var(--ink-2); cursor: pointer; }
.w-errata-controls button[aria-pressed="true"] { background: var(--ai); border-color: var(--ai); color: var(--paper); }
.w-errata-count { color: var(--ink-2); font-variant-numeric: tabular-nums; }
tr.w-errata-hidden { display: none; }
#errata-index th:nth-child(1), #errata-index td:nth-child(1),
#errata-index th:nth-child(4), #errata-index td:nth-child(4) { white-space: nowrap; }
`));
}

export function mount(root, config) {
  injectStyle();
  const table = document.querySelector(config.table || '#errata-index table');
  if (!table) {
    root.append(h('p', { class: 'w-label' }, '没有找到要筛选的表格。'));
    return;
  }
  const rows = [...table.tBodies[0].rows].map((tr) => {
    const cells = [...tr.cells].map((c) => c.textContent);
    return { tr, severity: cells[3].trim(), parts: partsOf(cells[2]), effect: effectOf(cells[4]) };
  });
  const state = { severity: new Set(SEVERITIES), part: 'all', effect: 'all' };
  const count = h('span', { class: 'w-errata-count', role: 'status', 'aria-live': 'polite' });

  const apply = () => {
    let shown = 0;
    for (const r of rows) {
      const ok = state.severity.has(r.severity)
        && (state.part === 'all' || r.parts.includes(state.part))
        && (state.effect === 'all' || r.effect === state.effect);
      r.tr.classList.toggle('w-errata-hidden', !ok);
      shown += ok;
    }
    count.textContent = `显示 ${shown} / ${rows.length} 条`;
  };

  const toggleButtons = (label, items, isOn, onClick) => {
    const group = h('div', { class: 'w-errata-group', role: 'group', 'aria-label': label }, h('span', {}, label));
    const buttons = items.map((it) => {
      const b = h('button', { type: 'button', 'aria-pressed': String(isOn(it)) }, it.label ?? it);
      b.addEventListener('click', () => {
        onClick(it);
        buttons.forEach((x, i) => x.setAttribute('aria-pressed', String(isOn(items[i]))));
        apply();
      });
      return b;
    });
    group.append(...buttons);
    return group;
  };

  const sev = toggleButtons('严重度', SEVERITIES, (s) => state.severity.has(s), (s) => {
    if (state.severity.has(s) && state.severity.size > 1) state.severity.delete(s);
    else state.severity.add(s);
  });
  const part = toggleButtons('所属部分', PARTS, (p) => state.part === p.key, (p) => { state.part = p.key; });
  const eff = toggleButtons('影响结果', EFFECTS, (e) => state.effect === e.key, (e) => { state.effect = e.key; });
  root.append(h('div', { class: 'w-errata-controls' }, sev, part, eff, count));
  apply();
}
