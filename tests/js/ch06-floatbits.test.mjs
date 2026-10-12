// 第 6 章 浮点位探索器：JS 的 fp16 / bf16 转换与 NumPy（以及经 verify/test_ch06.py 核对过的 PyTorch）逐位相同。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  f16Bits, f16FromBits, bf16Bits, bf16FromBits, f32Bits, f32FromBits, parseInput,
} from '../../web/widgets/ch06-floatbits.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/ch06/floats.json', import.meta.url), 'utf8'));

test('fp16: double → half 的位与 np.float16 相同（4030 个样例，含次正规数、平局与上溢）', () => {
  for (const [xs, h16] of fixture.rows) {
    assert.equal(f16Bits(Number(xs)), h16, `x = ${xs}`);
  }
});

test('bf16: 先舍入成 fp32，再按就近、平局取偶舍入，位与 NumPy 的模拟相同', () => {
  for (const [xs, , b16] of fixture.rows) {
    if (b16 === null) continue;
    assert.equal(bf16Bits(Number(xs)), b16, `x = ${xs}`);
  }
});

test('正文里的几个数', () => {
  assert.equal(f16Bits(1000), 0x63d0);
  assert.equal(bf16Bits(1000), 0x447a);
  assert.equal(f16FromBits(f16Bits(70000)), Infinity);
  assert.equal(f16FromBits(f16Bits(65519)), 65504);
  assert.equal(f16FromBits(f16Bits(65520)), Infinity);
  assert.equal(f16FromBits(f16Bits(1e-8)), 0);
  assert.equal(f16FromBits(f16Bits(3e-8)), 2 ** -24);
  assert.equal(bf16FromBits(bf16Bits(70000)), 70144);
  assert.equal(bf16FromBits(bf16Bits(Math.PI)), 3.140625);
  assert.equal(bf16FromBits(bf16Bits(1 / 3)), 0.333984375);
  assert.ok(Math.abs(bf16FromBits(bf16Bits(1e-8)) - 1.0011717677e-8) < 1e-17);
  assert.ok(Number.isNaN(f16FromBits(f16Bits(NaN))) && Number.isNaN(bf16FromBits(bf16Bits(NaN))));
});

test('fp32 的位来回转换不变；解码与编码互逆', () => {
  for (const x of [0, -0, 1, -2.5, 1e-40, 3.4028234663852886e38, Infinity]) {
    assert.ok(Object.is(f32FromBits(f32Bits(x)), Math.fround(x)), `x = ${x}`);
  }
  for (let b = 0; b < 0x7c00; b += 7) assert.equal(f16Bits(f16FromBits(b)), b);
});

test('输入解析', () => {
  assert.equal(parseInput(' 1e-8 '), 1e-8);
  assert.equal(parseInput('pi'), Math.PI);
  assert.equal(parseInput('1/3'), 1 / 3);
  assert.equal(parseInput('-inf'), -Infinity);
  assert.ok(Number.isNaN(parseInput('abc')));
});
