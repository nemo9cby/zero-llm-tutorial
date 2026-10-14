/* Small DOM helpers shared by widgets. Widgets export mount(root, config). */

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** A token chip: visible text on top, small id underneath. */
export function chip(text, id, colorIndex, title) {
  return h('span', { class: `tok c${((colorIndex % 6) + 6) % 6}`, title }, h('b', {}, visible(text)), h('i', {}, id));
}

/** Make whitespace and control characters visible inside chips. */
export function visible(s) {
  return s.replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥');
}

export const utf8 = new TextEncoder();
export const fmt = (n) => n.toLocaleString('en-US');
