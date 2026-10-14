/* 图：字符 vs 字节 —— 同一段文字在字符级与字节级 tokenizer 下的样子。
   Hovering a character (or moving the keyboard focus to it with Tab) highlights the UTF-8 bytes it became,
   and the other way round. */
import { h, chip, utf8, fmt } from './_lib.js';

const STYLE = `
.w-tokentape .tok.w-tokentape-on { outline: 2px solid var(--ai); }
.w-tokentape .tok:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
`;

const hex = (cp) => `U+${cp.toString(16).toUpperCase().padStart(4, '0')}`;

export function mount(root, config) {
  root.classList.add('w-tokentape');
  root.append(h('style', {}, STYLE));
  const input = h('input', { class: 'w-input', value: config.text ?? 'hello世界😁', 'aria-label': '输入任意文字' });
  const charRow = h('div', { class: 'tokens', role: 'list', 'aria-label': '字符级 token' });
  const byteRow = h('div', { class: 'tokens', role: 'list', 'aria-label': '字节级 token' });
  const stats = h('div', { class: 'w-row w-label' });
  root.append(
    input,
    h('div', { class: 'w-label' }, '字符级：每个 Unicode 码位一个 token（id = ord(c)）'), charRow,
    h('div', { class: 'w-label' }, '字节级：UTF-8 编码后的每个字节一个 token（id ∈ 0…255）'), byteRow,
    stats,
  );

  /** A character chip and its byte chips light up together while the pointer is on one of them
      or one of them has the keyboard focus. */
  function linkGroup(group) {
    let hover = false, focus = false;
    const sync = () => { for (const el of group) el.classList.toggle('w-tokentape-on', hover || focus); };
    for (const el of group) {
      el.tabIndex = 0;
      el.setAttribute('role', 'listitem');
      el.addEventListener('mouseenter', () => { hover = true; sync(); });
      el.addEventListener('mouseleave', () => { hover = false; sync(); });
      el.addEventListener('focus', () => { focus = true; sync(); });
      el.addEventListener('blur', () => { focus = false; sync(); });
    }
  }

  function render() {
    const chars = [...input.value]; // iterates code points, not UTF-16 units
    charRow.replaceChildren();
    byteRow.replaceChildren();
    let nBytes = 0;
    chars.forEach((c, i) => {
      const cp = c.codePointAt(0);
      const bytes = [...utf8.encode(c)];
      nBytes += bytes.length;
      const cChip = chip(c, cp, i, `${hex(cp)}，${bytes.length} 个字节`);
      cChip.setAttribute('aria-label', `字符 “${c}”，${hex(cp)}，id ${cp}，${bytes.length} 个字节`);
      const bChips = bytes.map((b, j) => {
        const el = chip(b.toString(16).padStart(2, '0'), b, i, `来自字符 “${c}”`);
        el.setAttribute('aria-label', `字节 0x${b.toString(16).padStart(2, '0')}（id ${b}），字符 “${c}” 的第 ${j + 1} / ${bytes.length} 个字节`);
        return el;
      });
      linkGroup([cChip, ...bChips]);
      charRow.append(cChip);
      byteRow.append(...bChips);
    });
    stats.textContent = `${fmt(chars.length)} 个字符 → ${fmt(nBytes)} 个字节`
      + (chars.length ? `（平均每字符 ${(nBytes / chars.length).toFixed(2)} 字节）` : '');
  }
  input.addEventListener('input', render);
  render();
}
