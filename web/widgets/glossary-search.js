/* 在本页的所有表格里查找：输入中文或英文（大小写、空格、下划线都不计较），只留下匹配的行。
   config: { scope: "article.chapter" } */
import { h } from './_lib.js';

/** “hidden_size”“Hidden Size”“hidden-size” 都归一成 hiddensize。 */
export function normalize(s) {
  return s.toLowerCase().replace(/[\s_\-·.]/g, '');
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-gloss-search { display: flex; flex-wrap: wrap; gap: .5rem .8rem; align-items: center; }
.w-gloss-search .w-input { flex: 1 1 16rem; }
.w-gloss-search button { font: inherit; font-size: .85rem; padding: .3rem .7rem; border-radius: 4px;
  border: 1px solid var(--rule); background: transparent; color: var(--ink-2); cursor: pointer; }
tr.w-gloss-hidden { display: none; }
tr.w-gloss-hit td:first-child { box-shadow: inset 3px 0 0 var(--ai); padding-left: .5rem; }
`));
}

export function mount(root, config) {
  injectStyle();
  const scope = document.querySelector(config.scope || 'article.chapter') || document.body;
  const rows = [...scope.querySelectorAll('table tbody tr')]
    .filter((tr) => !tr.closest('figure.widget'))                     // only the page's own tables
    .map((tr) => ({ tr, text: normalize(tr.textContent) }));
  const input = h('input', { class: 'w-input', type: 'search', 'aria-label': '在本页的表格里查找',
    placeholder: '例如 hidden_size、epoch、critic、上下文' });
  const clear = h('button', { type: 'button' }, '清除');
  const status = h('span', { class: 'w-label', role: 'status', 'aria-live': 'polite' });
  const apply = () => {
    const q = normalize(input.value);
    let hits = 0;
    for (const r of rows) {
      const ok = !q || r.text.includes(q);
      r.tr.classList.toggle('w-gloss-hidden', !ok);
      r.tr.classList.toggle('w-gloss-hit', Boolean(q) && ok);
      hits += ok;
    }
    status.textContent = q ? `匹配 ${hits} 行（共 ${rows.length} 行）` : `本页的表格共 ${rows.length} 行`;
  };
  input.addEventListener('input', apply);
  input.addEventListener('keydown', (e) => { if (e.key === 'Escape') { input.value = ''; apply(); } });
  clear.addEventListener('click', () => { input.value = ''; apply(); input.focus(); });
  root.append(h('div', { class: 'w-gloss-search' }, input, clear, status));
  if (config.initial) input.value = config.initial;
  apply();
}
