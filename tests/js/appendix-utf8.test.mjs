// 附录 A 交互图的编解码逻辑：与浏览器 / Node 内置的 TextEncoder、TextDecoder 对拍。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { encodeCodePoint, decodeSegments, decodeReplace, parseHex } from '../../web/widgets/appendix-utf8.js';

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

test('parseHex 接受常见写法，拒绝非法输入', () => {
  assert.deepEqual(parseHex('E4 b8, 0x41\n96'), [0xe4, 0xb8, 0x41, 0x96]);
  assert.deepEqual(parseHex('   '), []);
  assert.throws(() => parseHex('E4 G1'), /不是一个十六进制字节/);
  assert.throws(() => parseHex('100'), /不是一个十六进制字节/);
  assert.throws(() => parseHex(Array(65).fill('00').join(' ')), /最多输入 64 个字节/);
});
