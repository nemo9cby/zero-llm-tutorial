/* 图：字符 vs 字节 —— 同一段文字在字符级与字节级 tokenizer 下的样子。
   Hovering a character highlights the UTF-8 bytes it became. */
import { h, chip, utf8, fmt } from './_lib.js';

export function mount(root, config) {
  const input = h('input', { class: 'w-input', value: config.text ?? 'hello世界😁', 'aria-label': '输入任意文字' });
  const charRow = h('div', { class: 'tokens' });
  const byteRow = h('div', { class: 'tokens' });
  const stats = h('div', { class: 'w-row w-label' });
  root.append(
    input,
    h('div', { class: 'w-label' }, '字符级：每个 Unicode 码位一个 token（id = ord(c)）'), charRow,
    h('div', { class: 'w-label' }, '字节级：UTF-8 编码后的每个字节一个 token（id ∈ 0…255）'), byteRow,
    stats,
  );

  function render() {
    const chars = [...input.value]; // iterates code points, not UTF-16 units
    charRow.replaceChildren();
    byteRow.replaceChildren();
    let nBytes = 0;
    chars.forEach((c, i) => {
      const cp = c.codePointAt(0);
      const bytes = [...utf8.encode(c)];
      nBytes += bytes.length;
      const cChip = chip(c, cp, i, `U+${cp.toString(16).toUpperCase().padStart(4, '0')}，${bytes.length} 个字节`);
      const bChips = bytes.map((b) => chip(b.toString(16).padStart(2, '0'), b, i, `来自字符 “${c}”`));
      const link = (on) => { for (const el of [cChip, ...bChips]) el.style.outline = on ? '2px solid var(--ai)' : ''; };
      for (const el of [cChip, ...bChips]) {
        el.addEventListener('mouseenter', () => link(true));
        el.addEventListener('mouseleave', () => link(false));
      }
      charRow.append(cChip);
      byteRow.append(...bChips);
    });
    stats.textContent = `${fmt(chars.length)} 个字符 → ${fmt(nBytes)} 个字节`
      + (chars.length ? `（平均每字符 ${(nBytes / chars.length).toFixed(2)} 字节）` : '');
  }
  input.addEventListener('input', render);
  render();
}
