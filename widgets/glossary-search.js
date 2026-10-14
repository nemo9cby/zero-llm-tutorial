/* 在本页的所有表格里查找：输入中文或英文（大小写、空格、下划线都不计较），只留下匹配的行；
   同时列出标题里含这个词的小节（h2、h3）和方框（callout），点击即可跳过去。
   config: { scope: "article.chapter" } */
import { h } from './_lib.js';

/** “hidden_size”“Hidden Size”“hidden-size” 都归一成 hiddensize。 */
export function normalize(s) {
  return s.toLowerCase().replace(/[\s_\-·.]/g, '');
}

/** The items whose normalised text contains the normalised query (all of them for an empty query). */
export function findMatches(items, query) {
  const q = normalize(query);
  return q ? items.filter((it) => it.text.includes(q)) : items.slice();
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-gloss-search { display: flex; flex-wrap: wrap; gap: .5rem .8rem; align-items: center; }
.w-gloss-search .w-input { flex: 1 1 16rem; min-width: 0; }
.w-gloss-search button { font: inherit; font-size: .85rem; padding: .3rem .7rem; border-radius: 4px;
  border: 1px solid var(--rule); background: transparent; color: var(--ink-2); cursor: pointer; }
.w-gloss-sections { margin: .5rem 0 0; font-size: .88rem; color: var(--ink-2); overflow-wrap: anywhere; }
.w-gloss-sections a { margin-right: .7rem; }
tr.w-gloss-hidden { display: none; }
tr.w-gloss-hit td:first-child { box-shadow: inset 3px 0 0 var(--ai); padding-left: .5rem; }
`));
}

export function mount(root, config) {
  injectStyle();
  const scope = document.querySelector(config.scope || 'article.chapter') || document.body;
  const outsideWidgets = (el) => !el.closest('figure.widget');          // only the page's own content
  const rows = [...scope.querySelectorAll('table tbody tr')].filter(outsideWidgets)
    .map((tr) => ({ tr, text: normalize(tr.textContent) }));
  const sections = [...scope.querySelectorAll('h2, h3, aside.callout')].filter(outsideWidgets).map((el, i) => {
    const title = (el.matches('aside') ? el.querySelector('.callout-text')?.textContent : el.textContent) || '';
    if (!el.id) el.id = `w-gloss-sec-${i}`;                            // callouts have no id of their own
    return { el, title: title.trim(), text: normalize(title) };
  }).filter((s) => s.title);
  const input = h('input', { class: 'w-input', type: 'search', 'aria-label': '在本页的表格和标题里查找',
    placeholder: '例如 hidden_size、epoch、critic、上下文' });
  const clear = h('button', { type: 'button' }, '清除');
  const status = h('span', { class: 'w-label', role: 'status', 'aria-live': 'polite' });
  const secBox = h('p', { class: 'w-gloss-sections', hidden: true });
  const apply = () => {
    const q = normalize(input.value);
    const hitRows = new Set(findMatches(rows, input.value));
    for (const r of rows) {
      const ok = hitRows.has(r);
      r.tr.classList.toggle('w-gloss-hidden', !ok);
      r.tr.classList.toggle('w-gloss-hit', Boolean(q) && ok);
    }
    const hitSecs = q ? findMatches(sections, input.value) : [];
    secBox.hidden = hitSecs.length === 0;
    secBox.replaceChildren(...(hitSecs.length ? ['标题里含这个词的小节和方框：',
      ...hitSecs.map((s) => h('a', { href: `#${s.el.id}` }, s.title))] : []));
    if (!q) status.textContent = `本页的表格共 ${rows.length} 行`;
    else if (hitRows.size || hitSecs.length) status.textContent = `匹配 ${hitRows.size} 行（共 ${rows.length} 行）`;
    else status.textContent = '没有找到。换成英文或中文的另一种说法试试';
  };
  input.addEventListener('input', apply);
  input.addEventListener('keydown', (e) => { if (e.key === 'Escape') { input.value = ''; apply(); } });
  clear.addEventListener('click', () => { input.value = ''; apply(); input.focus(); });
  root.append(h('div', { class: 'w-gloss-search' }, input, clear, status), secBox);
  if (config.initial) input.value = config.initial;
  apply();
}
