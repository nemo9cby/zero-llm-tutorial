// 附录 A 交互图的编解码逻辑：与浏览器 / Node 内置的 TextEncoder、TextDecoder 对拍。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { encodeCodePoint, decodeSegments, decodeReplace, parseHex, bitsLabel, rejectReason, shown, quoteVisible }
  from '../../web/widgets/appendix-utf8.js';

test('encodeCodePoint 在全部 1,112,064 个码位上与 TextEncoder 逐字节相同', () => {
  const enc = new TextEncoder();
  let n = 0;
  for (let start = 0; start <= 0x10ffff; start += 0x10000) {
    const ours = [];
    const chars = [];
    for (let cp = start; cp < start + 0x10000; cp++) {
      if (cp >= 0xd800 && cp <= 0xdfff) continue;
      ours.push(...encodeCodePoint(cp));
      chars.push(String.fromCodePoint(cp));
      n += 1;
    }
    assert.deepEqual(Uint8Array.from(ours), enc.encode(chars.join('')), `block U+${start.toString(16)}`);
  }
  assert.equal(n, 1_112_064);
});

test('encodeCodePoint 拒绝代理区和超出范围的码位', () => {
  for (const cp of [0xd800, 0xdbff, 0xdc00, 0xdfff, 0x110000, -1, 1.5]) {
    assert.throws(() => encodeCodePoint(cp), RangeError);
  }
});

test('decodeReplace 与 TextDecoder 在随机字节串上逐字相同（最大子部分替换）', () => {
  const dec = new TextDecoder('utf-8', { ignoreBOM: true });
  const edges = [0x00, 0x41, 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf,
    0xe0, 0xe1, 0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf3, 0xf4, 0xf5, 0xff];
  let seed = 12345;
  const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  for (let t = 0; t < 20000; t++) {
    const len = 1 + Math.floor(rand() * 8);
    const bytes = Array.from({ length: len },
      () => (t % 2 ? edges[Math.floor(rand() * edges.length)] : Math.floor(rand() * 256)));
    assert.equal(decodeReplace(bytes), dec.decode(Uint8Array.from(bytes)), bytes.join(','));
  }
});

test('decodeSegments 给出与 Python 相同的出错位置和原因', () => {
  // 期望值来自 Python：bytes.decode("utf-8") 抛出的 UnicodeDecodeError 的 (start, end, reason)，见验证 A-j
  const cases = [
    [[0x80], [0, 1, 'invalid start byte']],
    [[0xe4, 0xb8], [0, 2, 'unexpected end of data']],
    [[0xc0, 0x80], [0, 1, 'invalid start byte']],
    [[0xe0, 0x80, 0x80], [0, 1, 'invalid continuation byte']],
    [[0xed, 0xa0, 0x80], [0, 1, 'invalid continuation byte']],
    [[0xf4, 0x90, 0x80, 0x80], [0, 1, 'invalid continuation byte']],
    [[0xe4, 0xb8, 0x41], [0, 2, 'invalid continuation byte']],
  ];
  for (const [bytes, expected] of cases) {
    const bad = decodeSegments(bytes).find((s) => s.reason);
    assert.deepEqual([bad.start, bad.end, bad.reason], expected);
  }
  assert.deepEqual(decodeSegments([0xe4, 0xb8, 0x96]), [{ start: 0, end: 3, cp: 0x4e16 }]);
});

test('卡片标题：码位本身的位数与模板能装的位数分开写（é 是 8 位，装进 11 位的模板）', () => {
  assert.equal(bitsLabel(0x41), '7 位有效，装进 7 位的模板 → 1 个字节');
  assert.equal(bitsLabel(0xe9), '8 位有效，装进 11 位的模板 → 2 个字节');
  assert.equal(bitsLabel(0x4e16), '15 位有效，装进 16 位的模板 → 3 个字节');   // 世
  assert.equal(bitsLabel(0x1f601), '17 位有效，装进 21 位的模板 → 4 个字节');  // 😁
});

test('落单的代理区码位：编码模式给出原因，不抛异常', () => {
  const cps = [...'A\ud800B'].map((c) => c.codePointAt(0));   // 粘贴进来的文字里可能有
  assert.deepEqual(cps, [0x41, 0xd800, 0x42]);
  assert.throws(() => encodeCodePoint(0xd800), RangeError);
  assert.match(rejectReason(0xd800), /U\+D800 是代理区的码位.*A\.4 节/);
  assert.equal(shown(0xd800), String.fromCodePoint(0xfffd));
});

test('看不见的字符：字形显示成 ·，输出行写成 \\uXXXX', () => {
  for (const cp of [0x00, 0x1f, 0x7f, 0x85, 0xad, 0x200b, 0x200d, 0x2028, 0x2029, 0xfeff, 0xe0001]) {
    assert.equal(shown(cp), '·', `U+${cp.toString(16)}`);
  }
  assert.equal(shown(0x20), '␣');
  assert.equal(shown(0x3000), '␣');
  assert.equal(shown(0x4e16), '世');
  // EF BB BF 41：解出来是 U+FEFF 加 A，不能只显示成 "A"
  const bomA = String.fromCodePoint(0xfeff, 0x41);
  assert.equal(decodeReplace([0xef, 0xbb, 0xbf, 0x41]), bomA);
  assert.equal(quoteVisible(bomA), '"\\uFEFFA"');
  assert.equal(quoteVisible(String.fromCodePoint(0x61, 0x200b, 0x62, 0x2028)), '"a\\u200Bb\\u2028"');
  assert.equal(quoteVisible(String.fromCodePoint(0xe0001)), '"\\u{E0001}"');
  const plain = String.fromCodePoint(0x4e16, 0x754c, 0xfffd);   // 世界�：普通字符和 � 原样显示
  assert.equal(quoteVisible(plain), JSON.stringify(plain));
});

test('parseHex 接受常见写法，拒绝非法输入', () => {
  assert.deepEqual(parseHex('E4 b8, 0x41\n96'), [0xe4, 0xb8, 0x41, 0x96]);
  assert.deepEqual(parseHex('   '), []);
  assert.throws(() => parseHex('E4 G1'), /不是一个十六进制字节/);
  assert.throws(() => parseHex('100'), /不是一个十六进制字节/);
  assert.throws(() => parseHex(Array(65).fill('00').join(' ')), /最多输入 64 个字节/);
});
