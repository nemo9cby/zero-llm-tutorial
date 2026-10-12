/* 附录 A 的交互图：UTF-8 位模式显微镜。
   config.mode = "encode"：输入文字，逐个码位展示二进制位怎样被切开、填进模板（config.text 为初始文字）。
   config.mode = "decode"：输入十六进制字节，展示解码器怎样划分“最大子部分”，并与浏览器的 TextDecoder 对照
                          （config.hex 为初始字节）。
   编解码算法与附录 A 的验证 A-d、A-j 相同；tests/js/appendix-utf8.test.mjs 用 TextEncoder / TextDecoder 对拍。 */
import { h } from './_lib.js';
import { makeRng, appendixRoot } from './appendix-common.js';

const REPLACEMENT = String.fromCodePoint(0xfffd);  // U+FFFD
const hex = (v, width = 2) => v.toString(16).toUpperCase().padStart(width, '0');

/** 把一个码位编码成 UTF-8 字节值的数组（与验证 A-d 的 utf8_encode 相同）。 */
export function encodeCodePoint(cp) {
  if (!Number.isInteger(cp) || cp < 0 || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
    throw new RangeError(`U+${hex(Math.max(0, cp | 0), 4)} 不是可以编码的字符`);
  }
  if (cp < 0x80) return [cp];
  if (cp < 0x800) return [0xc0 | (cp >> 6), 0x80 | (cp & 0x3f)];
  if (cp < 0x10000) return [0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f)];
  return [0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f)];
}

/** 按 Unicode 表 3-7 解码（与验证 A-j 的 utf8_decode 相同）。
    返回分段：合法的序列是 {start, end, cp}，不合法的“最大子部分”是 {start, end, reason}。 */
export function decodeSegments(bytes) {
  const segs = [];
  const n = bytes.length;
  let i = 0;
  while (i < n) {
    const b0 = bytes[i];
    if (b0 < 0x80) { segs.push({ start: i, end: i + 1, cp: b0 }); i += 1; continue; }
    let len = 0, lo = 0x80, hi = 0xbf;
    if (b0 >= 0xc2 && b0 <= 0xdf) len = 2;
    else if (b0 === 0xe0) { len = 3; lo = 0xa0; }        // 排除超长编码
    else if (b0 === 0xed) { len = 3; hi = 0x9f; }        // 排除代理区
    else if (b0 >= 0xe1 && b0 <= 0xef) len = 3;
    else if (b0 === 0xf0) { len = 4; lo = 0x90; }        // 排除超长编码
    else if (b0 === 0xf4) { len = 4; hi = 0x8f; }        // 不超过 U+10FFFF
    else if (b0 >= 0xf1 && b0 <= 0xf3) len = 4;
    let j = i + 1;                                       // 停在第一个“不对”的字节上
    while (len > 0 && j < i + len && j < n && bytes[j] >= lo && bytes[j] <= hi) { lo = 0x80; hi = 0xbf; j += 1; }
    if (len > 0 && j === i + len) {
      let cp = b0 & (0x7f >> len);
      for (let k = i + 1; k < j; k++) cp = (cp << 6) | (bytes[k] & 0x3f);
      segs.push({ start: i, end: j, cp });
    } else {
      const reason = len === 0 ? 'invalid start byte' : j === n ? 'unexpected end of data' : 'invalid continuation byte';
      segs.push({ start: i, end: j, reason });
    }
    i = j;
  }
  return segs;
}

/** errors="replace"：每个最大子部分换成一个 U+FFFD。 */
export const decodeReplace = (bytes) =>
  decodeSegments(bytes).map((s) => (s.reason ? REPLACEMENT : String.fromCodePoint(s.cp))).join('');

/** 解析 "E4 B8 0x41, c0" 这样的十六进制字节；出错时抛出带中文说明的 Error。 */
export function parseHex(text) {
  const parts = text.trim().split(/[\s,]+/).filter(Boolean);
  if (parts.length > 64) throw new Error('最多输入 64 个字节');
  return parts.map((p) => {
    const m = /^(?:0x)?([0-9a-f]{1,2})$/i.exec(p);
    if (!m) throw new Error(`“${p}”不是一个十六进制字节（00–FF）`);
    return parseInt(m[1], 16);
  });
}

const GROUP_SIZES = [[7], [5, 6], [4, 6, 6], [3, 6, 6, 6]];
const LEAD_MARK = ['0', '110', '1110', '11110'];
const REASON_ZH = {
  'invalid start byte': (b) => (b >= 0x80 && b <= 0xbf ? '孤立的后续字节' : '不能当首字节'),
  'unexpected end of data': () => '序列被截断',
  'invalid continuation byte': () => '下一个字节不在允许的范围内',
};

// 每个字节的角色和它在小方块底边上的颜色
function role(b) {
  if (b < 0x80) return ['ASCII', 'var(--moss)'];
  if (b <= 0xbf) return ['后续', 'var(--ink-3)'];
  if (b >= 0xc2 && b <= 0xf4) return ['首字节', 'var(--ai)'];
  return ['禁用', 'var(--kincha)'];
}

function shown(cp) {
  if (cp < 0x20 || (cp >= 0x7f && cp < 0xa0)) return '·';  // 控制字符没有字形
  if (cp === 0x20) return '␣';
  return String.fromCodePoint(cp);
}

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-appendix-a-picks { display: flex; flex-wrap: wrap; gap: .3rem; margin: .45rem 0 .2rem; }
.w-appendix-a-picks button { font-size: .82rem; padding: .1rem .5rem; }
.w-appendix-a-cards { display: grid; gap: .7rem; margin-top: .6rem; }
.w-appendix-a-card { border: 1px solid var(--rule); border-radius: 6px; padding: .55rem .75rem; }
.w-appendix-a-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: .2rem .9rem; }
.w-appendix-a-glyph { font-size: 1.55rem; line-height: 1.1; min-width: 1.4em; }
.w-appendix-a-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: .15rem .7rem; margin-top: .35rem; }
.w-appendix-a-row > .w-label { min-width: 6.5em; }
.w-appendix-a-bits { font: 13px/1.5 var(--mono); letter-spacing: .02em; }
.w-appendix-a-bits span + span { margin-left: .45em; }
.w-appendix-a-bytes { display: flex; flex-wrap: wrap; gap: .25rem .7rem; }
.w-appendix-a-byte { display: inline-flex; flex-direction: column; align-items: center; font: 13px/1.5 var(--mono); }
.w-appendix-a-byte small { font-size: 11px; color: var(--ink-2); }
.w-appendix-a-m { color: var(--ink-3); }
.w-appendix-a-g0 { color: var(--ai); font-weight: 600; }
.w-appendix-a-g1 { color: var(--moss); font-weight: 600; }
.w-appendix-a-g2 { color: var(--plum); font-weight: 600; }
.w-appendix-a-g3 { color: var(--kincha); font-weight: 600; }
.w-appendix-a-strip { display: flex; flex-wrap: wrap; gap: .45rem; margin: .7rem 0 .4rem; }
.w-appendix-a-seg { display: flex; flex-direction: column; align-items: center; gap: .1rem;
  border: 1px solid var(--rule); border-radius: 5px; padding: .25rem .35rem .2rem; background: var(--paper); }
.w-appendix-a-seg.bad { border-style: dashed; border-color: var(--kincha); }
.w-appendix-a-chips { display: flex; gap: 3px; }
.w-appendix-a-chip { display: inline-flex; flex-direction: column; align-items: center; min-width: 2.3em;
  padding: 2px 5px 1px; border-radius: 3px; background: var(--paper-2); font: 14px/1.35 var(--mono); color: var(--ink);
  box-shadow: inset 0 -2px 0 var(--chip, var(--ink-3)); }
.w-appendix-a-chip i { font-style: normal; font-size: 10.5px; color: var(--ink-2); }
.w-appendix-a-res { font-size: .78rem; color: var(--ink-2); text-align: center; max-width: 10em; line-height: 1.35; }
.w-appendix-a-res b { font-size: 1.05rem; font-weight: 400; color: var(--ink); }
.w-appendix-a-out { font: 13px/1.7 var(--mono); white-space: pre-wrap; word-break: break-all; margin: .3rem 0 0; }
.w-appendix-a-err { color: var(--kincha); font-size: .85rem; }
`));
}

function encodeCard(cp) {
  const bytes = encodeCodePoint(cp);
  const n = bytes.length;
  const bits = cp.toString(2).padStart(GROUP_SIZES[n - 1].reduce((a, b) => a + b, 0), '0');
  let pos = 0;
  const groups = GROUP_SIZES[n - 1].map((s) => { const g = bits.slice(pos, pos + s); pos += s; return g; });
  return h('div', { class: 'w-appendix-a-card' },
    h('div', { class: 'w-appendix-a-head' },
      h('span', { class: 'w-appendix-a-glyph', 'aria-hidden': 'true' }, shown(cp)),
      h('span', {}, `U+${hex(cp, 4)}`),
      h('span', { class: 'w-label' }, `${bits.length} 位有效 → ${n} 个字节`)),
    h('div', { class: 'w-appendix-a-row' },
      h('span', { class: 'w-label' }, '码位的二进制'),
      h('span', { class: 'w-appendix-a-bits' }, groups.map((g, k) => h('span', { class: `w-appendix-a-g${k}` }, g)))),
    h('div', { class: 'w-appendix-a-row' },
      h('span', { class: 'w-label' }, 'UTF-8 字节'),
      h('span', { class: 'w-appendix-a-bytes' }, bytes.map((b, k) => h('span', { class: 'w-appendix-a-byte' },
        h('span', {}, h('span', { class: 'w-appendix-a-m' }, k === 0 ? LEAD_MARK[n - 1] : '10'),
          h('span', { class: `w-appendix-a-g${k}` }, groups[k])),
        h('small', {}, `${hex(b)} = ${b}`))))));
}

function mountEncode(root, config) {
  const input = h('input', { class: 'w-input', value: config.text ?? 'Aé世😁', 'aria-label': '输入任意文字，逐个码位查看 UTF-8 编码' });
  const cards = h('div', { class: 'w-appendix-a-cards', 'aria-live': 'polite' });
  const ch = (v) => String.fromCodePoint(v);
  const picks = [['A', 'A'], ['é', 'é'], ['あ', 'あ'], ['世', '世'], ['😁', '😁'], ['U+07FF', ch(0x7ff)],
    ['U+0800', ch(0x800)], ['U+FFFD', ch(0xfffd)], ['U+10000', ch(0x10000)], ['U+10FFFF', ch(0x10ffff)]];
  const render = () => {
    const cps = [...input.value].map((c) => c.codePointAt(0));
    cards.replaceChildren(...cps.slice(0, 12).map(encodeCard));
    if (cps.length > 12) cards.append(h('p', { class: 'w-label' }, `只显示前 12 个字符（共 ${cps.length} 个）。`));
    if (!cps.length) cards.append(h('p', { class: 'w-label' }, '输入一些文字试试。'));
  };
  root.append(input,
    h('div', { class: 'w-appendix-a-picks' }, picks.map(([label, text]) => h('button', {
      type: 'button', 'aria-label': `只看 ${label}`, onclick: () => { input.value = text; render(); },
    }, label))),
    h('div', { class: 'w-label' }, '同一种颜色的位，在码位和字节里是同一组；灰色是标记位。'), cards);
  input.addEventListener('input', render);
  render();
}

function mountDecode(root, config) {
  const input = h('input', { class: 'w-input', value: config.hex ?? 'E4 B8 41 C0 80', spellcheck: 'false',
    'aria-label': '输入十六进制字节，例如 E4 B8 96' });
  const strip = h('div', { class: 'w-appendix-a-strip' });
  const out = h('div', { 'aria-live': 'polite' });
  const picks = [['半个“世”', 'E4 B8'], ['完整的“世”', 'E4 B8 96'], ['超长编码', 'C0 80'], ['代理区', 'ED A0 80'],
    ['超出上限', 'F4 90 80 80'], ['😁', 'F0 9F 98 81'], ['😁 少一个字节', 'F0 9F 98 41']];
  // 随机字节专挑各个区间的边界值，更容易撞上不合法的情况（与验证 A-k 的思路相同）
  const rng = makeRng(7);
  const edges = [0x41, 0x80, 0x9f, 0xa0, 0xbf, 0xc0, 0xc2, 0xdf, 0xe0, 0xe4, 0xed, 0xef, 0xf0, 0xf4, 0xf5];
  const randomBytes = () => Array.from({ length: 8 }, () => hex(edges[Math.floor(rng.uniform() * edges.length)])).join(' ');
  const render = () => {
    let bytes;
    try { bytes = parseHex(input.value); } catch (err) {
      strip.replaceChildren();
      out.replaceChildren(h('p', { class: 'w-appendix-a-err' }, err.message));
      return;
    }
    const segs = decodeSegments(bytes);
    strip.replaceChildren(...segs.map((s) => {
      const chips = bytes.slice(s.start, s.end).map((b) => {
        const [name, color] = role(b);
        return h('span', { class: 'w-appendix-a-chip', style: `--chip:${color}`, title: name }, hex(b), h('i', {}, name));
      });
      const res = s.reason
        ? h('span', { class: 'w-appendix-a-res' }, h('b', {}, REPLACEMENT), ' ', REASON_ZH[s.reason](bytes[s.start]))
        : h('span', { class: 'w-appendix-a-res' }, h('b', {}, shown(s.cp)), ` U+${hex(s.cp, 4)}`);
      return h('div', { class: `w-appendix-a-seg${s.reason ? ' bad' : ''}` }, h('span', { class: 'w-appendix-a-chips' }, chips), res);
    }));
    const replaced = decodeReplace(bytes);
    const firstBad = segs.find((s) => s.reason);
    let browser = null;   // ignoreBOM：与 Python 一样，不吞掉开头的 EF BB BF
    try { browser = new TextDecoder('utf-8', { ignoreBOM: true }).decode(new Uint8Array(bytes)); } catch { /* 没有 TextDecoder */ }
    const row = (label, ...value) => h('div', { class: 'w-row' }, h('span', { class: 'w-label' }, label), ...value);
    out.replaceChildren(
      row('errors="replace"', h('code', { class: 'w-appendix-a-out' }, JSON.stringify(replaced))),
      row('严格模式', firstBad
        ? `位置 ${firstBad.start}–${firstBad.end} 报错（${firstBad.reason}）`
        : '解码成功'),
      browser === null
        ? row('TextDecoder', '这个浏览器不支持 TextDecoder，无法对照')
        : row('浏览器的 TextDecoder', h('code', { class: 'w-appendix-a-out' }, JSON.stringify(browser)),
          browser === replaced ? '与上面一致' : '与上面不一致！'));
  };
  root.append(input,
    h('div', { class: 'w-appendix-a-picks' },
      picks.map(([label, value]) => h('button', {
        type: 'button', 'aria-label': `输入 ${label} 的字节 ${value}`, onclick: () => { input.value = value; render(); },
      }, label)),
      h('button', { type: 'button', 'aria-label': '随机生成 8 个字节', onclick: () => { input.value = randomBytes(); render(); } }, '随机 8 个字节')),
    strip, out);
  input.addEventListener('input', render);
  render();
}

export function mount(root, config = {}) {
  injectStyle();
  appendixRoot(root);
  if (config.mode === 'decode') mountDecode(root, config);
  else mountEncode(root, config);
}
