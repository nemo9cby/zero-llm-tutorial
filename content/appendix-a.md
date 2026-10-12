---
chapter: A
slug: appendix-a
order: 110
kind: appendix
title: UTF-8 编码的原理
lede: 第 1 章把 str.encode("utf-8") 当作黑盒来用。本附录打开这个黑盒：码位怎样变成字节，为什么结果总在 0–255 之间，哪些字节序列不合法，以及 errors="replace" 究竟替换了什么。
pyodide_packages: [numpy]
---

[1.2 节](ch01.html#s1-2)里，我们把 `str.encode("utf-8")` 当作一个黑盒来用：`A` 变成 1 个字节，`あ` 变成 3 个字节，`😁` 变成 4 个字节。字节级分词器之所以成立，靠的是两条性质：任何文字都能写成一串 0–255 的整数；这串整数还能原样变回文字。这两条性质都来自 UTF-8 的设计。

黑盒也留下了几个疑问。为什么 ASCII 字符只占 1 个字节，汉字却要 3 个？[验证 1.2-b](ch01.html#v-1.2-b) 里，半个汉字为什么解不出来？用 `errors="replace"` 时，两个坏字节为什么只变成了一个 `�`？

本附录就来打开这个黑盒。我们先比较 UTF-32、UTF-16、UTF-8 三种编码，然后从位模式出发，亲手实现 UTF-8 的编码器和解码器，并把它们与 Python 内置的实现逐字节对拍。最后回到分词器，看看这些规则藏在 DLFS6 代码的哪些地方。

## A.1 三种编码：UTF-32、UTF-16 与 UTF-8

Unicode 给每个字符规定了一个码位，范围是 0 到 0x10FFFF（[1.1 节](ch01.html#s1-1)）。码位只是一个整数。要把它存进文件、在网络上传输，还得规定怎样把这个整数写成字节，这种规定就是**编码**（encoding）。常用的编码有三种：UTF-32、UTF-16 和 UTF-8。

先来看结果。下面把 `A`、`あ`、`😁` 分别用三种编码写成字节。编码名后面的 `-be` 表示高位字节在前（big-endian），这样写出来的字节顺序与码位的十六进制写法一致：

```verify id=A-a title="同一个字符，三种编码"
for ch in ["A", "あ", "😁"]:
    print(f"{ch}  U+{ord(ch):04X}")
    for enc in ["utf-32-be", "utf-16-be", "utf-8"]:
        b = ch.encode(enc)
        print(f"    {enc[:6].upper():<6}  {len(b)} 字节  {b.hex(' ').upper()}")

assert [len(c.encode("utf-32-be")) for c in "Aあ😁"] == [4, 4, 4]
assert [len(c.encode("utf-16-be")) for c in "Aあ😁"] == [2, 2, 4]
assert [len(c.encode("utf-8")) for c in "Aあ😁"] == [1, 3, 4]
```

UTF-32 最简单：每个码位固定占 4 个字节，字节就是码位本身。代价是浪费，`A` 只需要 7 位，却占了 32 位，前 3 个字节全是 0。

UTF-16 以 2 个字节为一个单位。U+FFFF 以内的码位直接写成一个单位；更大的码位放不下，就拆成两个单位，叫作**代理对**（surrogate pair）。为此，Unicode 专门把 U+D800–U+DFFF 这一段码位留了出来，永远不分配给任何字符。拆法是先减去 0x10000，剩下的 20 位一分为二，高 10 位加到 0xD800 上，低 10 位加到 0xDC00 上。以 `😁` 为例：

$$
\begin{aligned}
&v = \text{0x1F601} - \text{0x10000} = \underbrace{0000111101}_{\text{0x03D}}\;\underbrace{1000000001}_{\text{0x201}} && \text{（20 位，一分为二）}\\
&\text{0xD800} + \text{0x03D} = \text{0xD83D} && \text{（高 10 位）}\\
&\text{0xDC00} + \text{0x201} = \text{0xDE01} && \text{（低 10 位）}
\end{aligned}
$$

两半拼起来正是上面输出里的 `D8 3D DE 01`。用代码再算一遍：

```verify id=A-b title="UTF-16 的代理对：高 10 位与低 10 位"
cp = ord("😁")
v = cp - 0x10000                 # 剩下 20 位
hi = 0xD800 + (v >> 10)          # 高 10 位
lo = 0xDC00 + (v & 0x3FF)        # 低 10 位
print(f"v      = {v >> 10:010b} {v & 0x3FF:010b}")
print(f"代理对 = {hi:04X} {lo:04X}")
pair = hi.to_bytes(2, "big") + lo.to_bytes(2, "big")
assert "😁".encode("utf-16-be") == pair
assert 0xD800 <= hi <= 0xDBFF and 0xDC00 <= lo <= 0xDFFF  # 都在留出的区间里
```

::: note 网页里的字符串是 UTF-16
JavaScript 的字符串按 UTF-16 的单位计数，所以 `"😁".length` 是 2，而 Python 的 `len("😁")` 是 1（[MDN：String.length](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/length)）。所以本附录的交互图在逐字符处理文字时，用 `[...text]` 按码位切分，避免把一个 emoji 劈成两半。
:::

UTF-8 是变长的：ASCII 字符只占 1 个字节，而且这个字节就等于码位本身。所以一个纯 ASCII 的文件，原封不动就是合法的 UTF-8 文件。另外，除了 U+0000 本身，UTF-8 的输出里不会出现值为 0 的字节；而用 UTF-16、UTF-32 写英文时，每个字符都带着值为 0 的字节（上面 `A` 的两行就是例子）。

那么，为什么“任何字符都能写成 0–255 的数列”呢？答案其实很朴素：三种编码的输出都是字节，一个字节有 8 位，取值只能是 0–255。字节级分词器用哪一种编码都行，真正的区别在于长度：

```verify id=A-c title="英文和代码用 UTF-8 最短，中文用 UTF-16 更短"
def size_in(text, enc):
    return len(text.encode(enc))

code, zh = "def add(a, b):\n    return a + b", "深度学习入门"
for text in (code, zh):
    n8, n16, n32 = (size_in(text, e)
                    for e in ("utf-8", "utf-16-be", "utf-32-be"))
    print(f"{len(text):2d} 个字符：UTF-8 {n8:3d}，UTF-16 {n16:3d}，"
          f"UTF-32 {n32:3d} 字节")

assert size_in(code, "utf-8") == len(code)
assert size_in(code, "utf-16-be") == 2 * len(code)
assert size_in(zh, "utf-8") == 3 * len(zh)          # 中文：UTF-8 每字 3 字节
assert size_in(zh, "utf-16-be") == 2 * len(zh)      # 而 UTF-16 每字 2 字节
assert all(0x4E00 <= ord(c) <= 0x9FFF for c in zh)  # 这几个字的码位范围
```

对英文和代码来说，UTF-8 的长度是 UTF-16 的一半、UTF-32 的四分之一；对中文来说，UTF-8 反而比 UTF-16 多出一半。DLFS6 的三份语料（Python 代码、英文童话、英文网页）几乎全是 ASCII，选 UTF-8 是自然的。它的分词器第一步就是 `text.encode("utf-8")`：

```listing codebot/tokenizer.py#L81-L85
```

::: cs336 A1 §2.2 unicode2
CS336 作业 1 的 `unicode2` 小题问了三件事：为什么在 UTF-8 的字节上训练分词器，而不是 UTF-16 或 UTF-32；一个逐字节解码的函数错在哪里；什么样的两个字节无法解码。这三件事分别对应本附录的 A.1、A.5 和 A.4 节。读完之后，建议你合上本页，自己作答一遍（[A1 仓库](https://github.com/stanford-cs336/assignment1-basics)）。
:::

## A.2 UTF-8 的位模式

UTF-8 的规则可以写成一张表。码位越大，需要的有效位数越多，就用越多的字节来装：

| 码位范围 | 有效位数 | 字节序列（x 是码位的二进制位） | 字节数 |
|---|---|---|---|
| U+0000–U+007F | 7 | `0xxxxxxx` | 1 |
| U+0080–U+07FF | 11 | `110xxxxx 10xxxxxx` | 2 |
| U+0800–U+FFFF | 16 | `1110xxxx 10xxxxxx 10xxxxxx` | 3 |
| U+10000–U+10FFFF | 21 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` | 4 |

每个字节开头的几位是固定的标记，剩下的 x 位用来装码位。规律只有两条：第一个字节开头有几个 1，这个字符就占几个字节（单字节的 ASCII 例外，开头是 0）；其余的字节一律以 `10` 开头。我们把第一个字节叫作**首字节**（lead byte），把以 `10` 开头的字节叫作**后续字节**（continuation byte）。数一数 x 的个数：2 字节能装 5+6=11 位，3 字节能装 4+6+6=16 位，4 字节能装 3+6+6+6=21 位，正好与表中的有效位数对上。

开头的第一个问题也有了答案：ASCII 字符的码位小于 0x80，落在第 1 行；`あ` 是 U+3042，`深度学习入门` 这几个字的码位都在 U+4E00–U+9FFF 之间（[验证 A-c](#v-A-c) 的最后一行），都落在第 3 行，所以要 3 个字节。

下面手算 `あ` 的编码，每一步的理由写在右边：

$$
\begin{aligned}
&\operatorname{ord}(\text{あ}) = \text{0x3042} && \text{（查码位）}\\
&\text{0x0800} \le \text{0x3042} \le \text{0xFFFF} && \text{（表的第 3 行：3 个字节）}\\
&\text{0x3042} = 0011\;000001\;000010_{(2)} && \text{（16 位，按 4+6+6 切开）}\\
&\to 1110\mathbf{0011}\;\;10\mathbf{000001}\;\;10\mathbf{000010} && \text{（依次填进模板）}\\
&= \text{E3 81 82} = 227,\ 129,\ 130 && \text{（与 1.2 节的输出一致）}
\end{aligned}
$$

把表格翻译成代码，就是下面的 `utf8_encode`。`>>` 把码位右移，取出高位的那一组；`& 0x3F` 只保留低 6 位；`|` 把标记位和数据位拼在一起：

```verify id=A-d title="亲手实现 UTF-8 编码器，手算 あ"
def utf8_encode(cp):
    """按 UTF-8 的位模式，把一个码位编码成字节值的列表。"""
    if not 0 <= cp <= 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
        raise ValueError(f"U+{cp:04X} 不是可以编码的字符")  # 原因见 A.4 节
    if cp < 0x80:       # 7 位 ：0xxxxxxx
        return [cp]
    if cp < 0x800:      # 11 位：110xxxxx 10xxxxxx
        return [0xC0 | (cp >> 6),
                0x80 | (cp & 0x3F)]
    if cp < 0x10000:    # 16 位：1110xxxx 10xxxxxx 10xxxxxx
        return [0xE0 | (cp >> 12),
                0x80 | ((cp >> 6) & 0x3F),
                0x80 | (cp & 0x3F)]
    # 21 位：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    return [0xF0 | (cp >> 18),
            0x80 | ((cp >> 12) & 0x3F),
            0x80 | ((cp >> 6) & 0x3F),
            0x80 | (cp & 0x3F)]

cp = ord("あ")
b = utf8_encode(cp)
groups = f"{cp >> 12:04b} {(cp >> 6) & 0x3F:06b} {cp & 0x3F:06b}"
steps = [(f"U+{cp:04X}", "码位"),
         (f"{cp:016b}", "写成 16 位"),
         (groups, "按 4+6+6 切开"),
         (" ".join(f"{x:08b}" for x in b), "填进模板"),
         (" ".join(f"{x:02X}" for x in b) + f" = {b}", "字节")]
for value, why in steps:
    print(f"{value:<28}（{why}）")
assert b == [227, 129, 130] == list("あ".encode("utf-8"))
```

::: think 手算 😁（U+1F601）的 UTF-8 编码，结果应该与 1.2 节的哪几个 id 一致？
:::
::: answer
0x1F601 落在第 4 行，要 4 个字节。写成 21 位是 `000 011111 011000 000001`（按 3+6+6+6 切开），填进模板得到 `11110000 10011111 10011000 10000001`，也就是 `F0 9F 98 81` = 240、159、152、129。这正是 1.2 节 `hello世界😁` 编码结果的最后 4 个 id。
:::

一个例子对上了，不等于所有码位都对。下面从四类码位里各随机抽 2,000 个，再加上每一类的上下边界，与 Python 的 `str.encode` 逐个比较。代理区 U+D800–U+DFFF 两边都应该拒绝：

```verify id=A-e title="四类码位抽样对拍，代理区两边都拒绝"
import numpy as np

rng = np.random.default_rng(0)
classes = [(0x0000, 0x007F), (0x0080, 0x07FF),
           (0x0800, 0xFFFF), (0x10000, 0x10FFFF)]
for n, (lo, hi) in enumerate(classes, start=1):
    cps = [lo, hi] + [int(c) for c in rng.integers(lo, hi + 1, size=2000)]
    cps = [c for c in cps if not 0xD800 <= c <= 0xDFFF]  # 代理区另行检查
    for c in cps:
        b = utf8_encode(c)
        assert len(b) == n and bytes(b) == chr(c).encode("utf-8")
    print(f"U+{lo:04X}–U+{hi:04X}：{len(cps)} 个码位，都是 {n} 字节，"
          "与 str.encode 一致")

def try_encode(encode, c):
    try:
        return encode(c).hex(" ").upper()
    except (ValueError, UnicodeEncodeError):
        return "拒绝"

for c in (0xD7FF, 0xD800, 0xDFFF, 0xE000):
    mine = try_encode(lambda c: bytes(utf8_encode(c)), c)
    ref = try_encode(lambda c: chr(c).encode("utf-8"), c)
    print(f"U+{c:04X} → {mine}（Python：{ref}）")
    assert mine == ref
```

抽样能发现大多数错误，但我们可以做得更彻底：把全部 1,112,064 个可编码的码位一次性对拍。码位从小到大排列时，字节数也是从小到大的，所以把四类码位分别用数组运算编好，再依次拼起来，就应该等于“所有字符连成的字符串”的 UTF-8 编码。构造这个字符串时，我们借用了 A.1 节的结论：UTF-32 的字节就是码位本身，所以把码位数组按 UTF-32 解码，就得到了包含全部字符的字符串。

```verify id=A-f title="全部 1,112,064 个码位逐字节对拍（NumPy 向量化）"
cps = np.arange(0x110000, dtype=np.uint32)
cps = cps[(cps < 0xD800) | (cps > 0xDFFF)]                 # 去掉代理区
c1, c2, c3, c4 = (cps[(cps >= lo) & (cps <= hi)] for lo, hi in classes)
# 与 utf8_encode 完全相同的位运算，只是一次算一整个数组
b1 = c1[:, None]
b2 = np.stack([0xC0 | (c2 >> 6), 0x80 | (c2 & 0x3F)], axis=1)
b3 = np.stack([0xE0 | (c3 >> 12), 0x80 | ((c3 >> 6) & 0x3F),
               0x80 | (c3 & 0x3F)], axis=1)
b4 = np.stack([0xF0 | (c4 >> 18), 0x80 | ((c4 >> 12) & 0x3F),
               0x80 | ((c4 >> 6) & 0x3F), 0x80 | (c4 & 0x3F)], axis=1)
blocks = [b.ravel() for b in (b1, b2, b3, b4)]
ours = np.concatenate(blocks).astype(np.uint8).tobytes()

# UTF-32 的字节就是码位本身：按 UTF-32 解码，得到全部字符连成的字符串
everything = cps.astype("<u4").tobytes().decode("utf-32-le")
print(f"{len(cps):,} 个码位 → {len(ours):,} 个字节")
assert len(cps) == 1_112_064 and len(everything) == len(cps)
assert ours == everything.encode("utf-8")
```

一百多万个码位，四百多万个字节，一个不差。下面的交互图可以输入任意文字，看看每个字符的码位是怎样被拆开、填进模板的。

```widget appendix-utf8 caption="码位的每一位都原样出现在字节里：首字节的标记告诉我们一共几个字节，后续字节都以 10 开头。"
{"mode": "encode", "text": "Aé世😁"}
```

## A.3 解码与自同步

解码是编码的逆过程：读首字节，数开头有几个 1，就知道这个字符一共几个字节；去掉每个字节的标记位，把剩下的数据位拼起来，就得到码位。以 `E3 81 82` 为例：

$$
\begin{aligned}
&\text{E3 81 82} = 1110\,0011\;\;10\,000001\;\;10\,000010_{(2)} && \text{（写成二进制）}\\
&\to 0011\;000001\;000010_{(2)} && \text{（去掉标记位）}\\
&= \text{0x3042} = \text{あ} && \text{（拼起来就是码位）}
\end{aligned}
$$

这里有一个容易被忽略的设计：三种字节的取值范围互不重叠。ASCII 是 `00–7F`，后续字节是 `80–BF`，首字节则落在 `C2–F4`（剩下的 `C0`、`C1`、`F5–FF` 永远不会出现，原因见 A.4 节）。我们用 A.2 节的全部编码结果来确认这一点：

```verify id=A-g title="三种字节互不重叠；13 个字节值永远不出现"
first = np.concatenate([b2[:, 0], b3[:, 0], b4[:, 0]])
rest = np.concatenate([b2[:, 1:].ravel(), b3[:, 1:].ravel(),
                       b4[:, 1:].ravel()])
lead, cont = set(np.unique(first).tolist()), set(np.unique(rest).tolist())
unused = sorted(set(range(256)) - set(range(0x80)) - lead - cont)
print(f"首字节  ：{min(lead):02X}–{max(lead):02X}，共 {len(lead)} 个")
print(f"后续字节：{min(cont):02X}–{max(cont):02X}，共 {len(cont)} 个")
print("从不出现：", " ".join(f"{x:02X}" for x in unused))
assert lead == set(range(0xC2, 0xF5)) and cont == set(range(0x80, 0xC0))
assert len(unused) == 13 and not lead & cont

# 手算的解码：去掉标记位，把数据位拼起来
cp = ((0xE3 & 0x0F) << 12) | ((0x81 & 0x3F) << 6) | (0x82 & 0x3F)
assert chr(cp) == "あ"
```

范围不重叠，带来了两个好处。第一，随便从文件中间的哪个字节开始，只要跳过开头的后续字节（最多 3 个），就一定站在某个字符的起点上，从那里开始解码即可。这种性质叫作**自同步**（self-synchronizing）。第二，多字节字符里的每个字节都不小于 0x80，所以 ASCII 字节绝不会出现在某个汉字或 emoji 的“肚子”里。

```verify id=A-h title="从任意字节出发，最多跳过 3 个后续字节就能对齐"
text = "hello世界😁"
data = text.encode("utf-8")
max_skip = 0
for start in range(len(data)):
    i = start
    while i < len(data) and 0x80 <= data[i] <= 0xBF:  # 跳过后续字节
        i += 1
    max_skip = max(max_skip, i - start)
    assert text.endswith(data[i:].decode("utf-8"))   # 剩下的都能解码
print(f"{len(data)} 个起点都能对齐，最多跳过 {max_skip} 个字节")
assert max_skip == 3
```

第二个好处在 DLFS6 里派上了用场。[4.3 节](ch04.html#s4-3)训练分词器时，要把好几 GB 的文件切成若干块分给多个进程。`find_chunk_boundaries` 先按字节数把文件等分，再从每个切点往后找最近的 `<|endoftext|>`，把切点挪到那里：

```listing storybot/tokenizer.py#L53-L71
```

等分的切点可能正好落在一个汉字的中间，挪过之后就不会了：`<|endoftext|>` 全是 ASCII 字节，它的起点不可能在任何多字节字符的内部。于是每一块都能单独解码。我们在内存里模拟一遍：

```verify id=A-i title="把切点挪到 <|endoftext|>，每一块都能单独解码"
end = b"<|endoftext|>"
docs = ["从前有一只猫，它每天都在窗台上晒太阳。", "Hello, 世界! 😁",
        "def f(x):\n    return x"] * 37
blob = end.join(d.encode("utf-8") for d in docs)

def decodable(chunk):
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False

def move_to_end_token(cut):
    """从切点往后找最近的 <|endoftext|>，找不到就挪到末尾。"""
    p = blob.find(end, cut)
    return len(blob) if p == -1 else p

def count_bad(cuts):
    return sum(not decodable(blob[a:b]) for a, b in zip(cuts, cuts[1:]))

num_chunks = 8
size = len(blob) // num_chunks
naive = [i * size for i in range(num_chunks)] + [len(blob)]     # 等分
moved = map(move_to_end_token, naive[1:-1])
aligned = sorted({0, len(blob), *moved})                          # 挪过之后
print(f"等分：{num_chunks} 块里有 {count_bad(naive)} 块无法单独解码")
print(f"对齐：{len(aligned) - 1} 块里有 {count_bad(aligned)} 块无法单独解码")
assert count_bad(naive) > 0 and count_bad(aligned) == 0
chunks = [blob[a:b] for a, b in zip(aligned, aligned[1:])]
assert b"".join(chunks) == blob
```

这里的模拟把 `find_chunk_boundaries` 的循环读取简化成了一次 `bytes.find`。真正的函数在磁盘文件上是不是也这样，可以用下面的本地测试确认：

```localcheck verify/test_appendix.py::test_a_chunk_boundaries title="真实的 find_chunk_boundaries 切出的每一块都能单独解码"
```

## A.4 不合法的字节序列

UTF-8 的规则也决定了哪些字节序列不合法。归纳起来有下面五种情况：

| 情况 | 例子 | 为什么不合法 |
|---|---|---|
| 孤立的后续字节 | `80` | 后续字节前面必须有首字节 |
| 序列被截断 | `E4 B8` | 首字节 `E4` 说一共 3 个字节，只来了 2 个 |
| 超长编码（overlong） | `C0 80`、`E0 80 80` | 同一个码位只允许最短的写法 |
| 代理区 | `ED A0 80`（U+D800） | 这段码位留给 UTF-16 的代理对，不是字符 |
| 超出 U+10FFFF | `F4 90 80 80` | Unicode 的码位到 0x10FFFF 为止 |

前两种情况直接违反了 A.2 节的模板，后三种则需要解释一下。

先看超长编码。如果不管规矩，硬把 `C0 80` 按模板拆开，数据位是 `00000` 和 `000000`，拼起来就是 U+0000。可 U+0000 明明只需要一个字节 `00`。如果两种写法都算合法，同一段文字就会有多种字节表示，分词器给出的 token 序列也就不再唯一。更危险的是安全检查：一个程序禁止输入里出现 NUL 字节 `00`，却放过了 `C0 80`，后面的解码器又把它还原成了 NUL。这正是 UTF-8 的规范 RFC 3629 在第 10 节举的例子，所以规范要求只接受最短的写法（[RFC 3629](https://www.rfc-editor.org/rfc/rfc3629)）。这也解释了首字节为什么从 `C2` 开始：`C0` 和 `C1` 开头的 2 字节序列只能装下小于 0x80 的码位，必然是超长编码。

再看后两种。代理区 U+D800–U+DFFF 是 A.1 节里留给 UTF-16 的“零件”，单独出现没有意义。至于上限，早期的规范 RFC 2279 允许最多 6 个字节；RFC 3629 把范围收窄到 0x10FFFF，也就是 UTF-16 能表示的范围，于是最多 4 个字节。这样一来，`F5–F7` 开头的 4 字节序列表示的码位都超过了 0x10FFFF，`F8–FD` 属于已经废除的 5、6 字节格式，`FE` 和 `FF` 则连旧规范里都没有用到（[RFC 2279](https://www.rfc-editor.org/rfc/rfc2279)），所以首字节 `F5–FF` 永远不会出现。

把这些规则合起来，就是 Unicode 标准中的“合法 UTF-8 字节序列”表（[Unicode 16.0 第 3.9.3 节，表 3-7](https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-3/#G27506)）。它的关键在于：第 2 个字节允许的范围取决于首字节：

| 码位范围 | 首字节 | 第 2 字节 | 第 3 字节 | 第 4 字节 |
|---|---|---|---|---|
| U+0000–U+007F | `00–7F` | | | |
| U+0080–U+07FF | `C2–DF` | `80–BF` | | |
| U+0800–U+0FFF | `E0` | **`A0–BF`** | `80–BF` | |
| U+1000–U+CFFF | `E1–EC` | `80–BF` | `80–BF` | |
| U+D000–U+D7FF | `ED` | **`80–9F`** | `80–BF` | |
| U+E000–U+FFFF | `EE–EF` | `80–BF` | `80–BF` | |
| U+10000–U+3FFFF | `F0` | **`90–BF`** | `80–BF` | `80–BF` |
| U+40000–U+FFFFF | `F1–F3` | `80–BF` | `80–BF` | `80–BF` |
| U+100000–U+10FFFF | `F4` | **`80–8F`** | `80–BF` | `80–BF` |

加粗的四格各自排除了一种不合法的情况：`E0` 后面不能是 `80–9F`（否则是超长编码），`ED` 后面不能是 `A0–BF`（否则落进代理区），`F0` 后面不能是 `80–8F`（超长编码），`F4` 后面不能是 `90–BF`（超出 0x10FFFF）。

把这张表原样搬进代码，就是下面的 `TABLE_3_7` 和解码器 `utf8_decode`。遇到不合法的序列时，它与 Python 一样抛出 `UnicodeDecodeError`，并且报告出错的起止位置和原因。`errors="replace"` 那一支在 A.5 节解释：

```verify id=A-j title="照着表 3-7 实现解码器：五种不合法情况与 Python 报错一致"
# 表 3-7 的每一行：(首字节范围, 总字节数, 第 2 个字节的范围)
TABLE_3_7 = [((0xC2, 0xDF), 2, (0x80, 0xBF)),
             ((0xE0, 0xE0), 3, (0xA0, 0xBF)),   # 排除超长编码
             ((0xE1, 0xEC), 3, (0x80, 0xBF)),
             ((0xED, 0xED), 3, (0x80, 0x9F)),   # 排除代理区
             ((0xEE, 0xEF), 3, (0x80, 0xBF)),
             ((0xF0, 0xF0), 4, (0x90, 0xBF)),   # 排除超长编码
             ((0xF1, 0xF3), 4, (0x80, 0xBF)),
             ((0xF4, 0xF4), 4, (0x80, 0x8F))]   # 不超过 U+10FFFF

def lookup(b0):
    """首字节 → (总字节数, 第 2 个字节的下限, 上限)；不能当首字节时长度为 0。"""
    for (first, last), length, (lo, hi) in TABLE_3_7:
        if first <= b0 <= last:
            return length, lo, hi
    return 0, 0, 0

def utf8_decode(data, errors="strict"):
    """按表 3-7 解码。errors 取 "strict" 或 "replace"。"""
    out, i, n = [], 0, len(data)
    while i < n:
        b0 = data[i]
        if b0 < 0x80:                          # ASCII
            out.append(chr(b0))
            i += 1
            continue
        length, lo, hi = lookup(b0)
        j = i + 1                              # j 停在第一个“不对”的字节上
        while j < i + length and j < n and lo <= data[j] <= hi:
            lo, hi = 0x80, 0xBF                # 第 3、4 个字节都是 80–BF
            j += 1
        if length and j == i + length:         # 完整而且合法
            cp = b0 & (0x7F >> length)         # 首字节里的数据位
            for k in range(i + 1, j):
                cp = (cp << 6) | (data[k] & 0x3F)  # 每个后续字节 6 位
            out.append(chr(cp))
        elif errors == "strict":
            reason = ("invalid start byte" if length == 0 else
                      "unexpected end of data" if j == n else
                      "invalid continuation byte")
            raise UnicodeDecodeError("utf-8", bytes(data), i, j, reason)
        else:
            out.append(chr(0xFFFD))  # data[i:j] 换成一个 U+FFFD（A.5 节）
        i = j
    return "".join(out)

def outcome(decode, data):
    """解码成功返回文字；失败返回 (起点, 终点, 原因)。"""
    try:
        return decode(data)
    except UnicodeDecodeError as e:
        return (e.start, e.end, e.reason)

def python_decode(data):
    return data.decode("utf-8")

bad = {"孤立的后续字节": b"\x80", "序列被截断": b"\xe4\xb8",
       "超长编码": b"\xc0\x80", "超长编码（3 字节）": b"\xe0\x80\x80",
       "代理区 U+D800": b"\xed\xa0\x80",
       "超出 U+10FFFF": b"\xf4\x90\x80\x80"}
for name, data in bad.items():
    mine = outcome(utf8_decode, data)
    assert mine == outcome(python_decode, data)
    start, stop, reason = mine
    print(f"{data.hex(' ').upper():<12} {start}–{stop}  {reason:<25} {name}")
assert utf8_decode("hello世界😁".encode("utf-8")) == "hello世界😁"
```

六个例子都对上了，连出错的位置和原因的措辞都与 Python 相同。不过，六个例子仍然只是六个例子。下面随机生成 2,000 段 1–7 个字节的字节串，一半从全部 256 个字节值里抽，另一半专门从各个区间的边界值里抽（这样更容易撞上刁钻的情况），逐一比较两个解码器的结果：

```verify id=A-k title="2,000 段随机字节：严格模式下的结果与报错逐一相同"
rng = np.random.default_rng(1)
edges = [0x00, 0x41, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF,
         0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1, 0xEC, 0xED, 0xEE,
         0xEF, 0xF0, 0xF1, 0xF3, 0xF4, 0xF5, 0xFF]   # 各个区间的边界
cases = []
for t in range(2000):
    pool = edges if t % 2 else range(256)
    n = int(rng.integers(1, 8))
    cases.append(bytes(int(x) for x in rng.choice(pool, size=n)))
n_bad = 0
for data in cases:
    ref = outcome(python_decode, data)
    assert outcome(utf8_decode, data) == ref, data
    n_bad += isinstance(ref, tuple)
print(f"{len(cases)} 段字节串，其中 {n_bad} 段不合法；"
      "成功的结果与报错位置全部一致")
```

## A.5 errors="replace"：一段坏字节换成几个 �

现在可以回答开头的问题了。严格模式遇到坏字节就报错；`errors="replace"` 则把坏字节换成替换字符 U+FFFD（显示为 `�`）然后继续。问题在于“换几个”：两个坏字节，是换成一个 `�` 还是两个？

Unicode 标准推荐的做法叫“按最大子部分替换”（[第 3.9.6 节](https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-3/#G66453)）。从出错的位置开始，找出最长的一段字节，这段字节要么是某个合法序列的开头，要么只有 1 个字节；这一段叫作**最大子部分**（maximal subpart），整段只换成一个 U+FFFD，然后从它后面接着解码。标准本身并不强制这种做法，但 Python 的解码器遵守它（下面的验证会逐字对拍），浏览器里的 `TextDecoder` 也遵守它（[WHATWG Encoding 标准的 UTF-8 解码器](https://encoding.spec.whatwg.org/#utf-8-decoder)）。

A.4 节解码器里的变量 `j` 正是为此设计的：它从首字节往前走，停在第一个“不对”的字节上，`data[i:j]` 就是最大子部分。于是：

- 半个“世” `E4 B8`：这两个字节是合法序列 `E4 B8 96` 的开头，属于同一个最大子部分，只换成**一个** `�`。这就是[验证 1.2-b](ch01.html#v-1.2-b) 的输出。
- 超长编码 `C0 80`：`C0` 不可能是任何合法序列的开头，单独成一段；`80` 是孤立的后续字节，又成一段，所以换成**两个** `�`。
- 代理区 `ED A0 80`：`ED` 后面只允许 `80–9F`，`A0` 不对，于是 `ED` 单独成一段，`A0`、`80` 各成一段，一共**三个** `�`。

```verify id=A-l title="按最大子部分替换：与 errors=replace 的结果逐字相同"
examples = [b"\xe4\xb8", b"\xe4\xb8A", b"\xc0\x80", b"\xed\xa0\x80",
            b"\xf0\x9f\x98"]
for data in examples:
    mine = utf8_decode(data, errors="replace")
    assert mine == data.decode("utf-8", errors="replace")
    k = mine.count(chr(0xFFFD))
    print(f"{data.hex(' ').upper():<12} → {mine!r}（{k} 个 U+FFFD）")

for data in cases:                       # A-k 的 2,000 段随机字节
    ref = data.decode("utf-8", errors="replace")
    assert utf8_decode(data, errors="replace") == ref
print(f"{len(cases)} 段随机字节：替换结果与 Python 逐字相同")
```

DLFS6 的 BPE 分词器从 [1.4 节](ch01.html#s1-4)起，`decode` 都用 `errors="replace"`：

```listing ch01/04_bpe_tokenizer.py#L47-L56
```

它解码的是拼好的整段字节，所以只要模型最终生成的是完整的字符，就不会出现 `�`。麻烦出在“逐个 token 显示”的场合：一个汉字的 3 个字节如果分属 3 个 token，逐个解码就会得到 3 个 `�`。另外，替换是有损的，`�` 再编码回去是 `EF BF BD`，原来的字节已经找不回来了。流式输出的正确做法是先攒字节，等一个字符完整了再输出，Python 的增量解码器做的正是这件事：

```verify id=A-m title="逐 token 解码得到 3 个 �；增量解码器先攒字节再输出"
ids = list("世".encode("utf-8"))                     # [228, 184, 150]
per_token = "".join(utf8_decode(bytes([i]), errors="replace") for i in ids)
print("逐个解码：", per_token)
assert per_token == chr(0xFFFD) * 3

half = bytes([228, 184])
lossy = half.decode("utf-8", errors="replace").encode("utf-8")
print("半个“世”替换后再编码：", lossy.hex(" ").upper())
assert lossy == b"\xef\xbf\xbd" != half              # 原来的字节找不回来了

import codecs
dec = codecs.getincrementaldecoder("utf-8")()
pieces = [dec.decode(bytes([b])) for b in "世界😁".encode("utf-8")]
print("增量解码：", pieces)
assert "".join(pieces) == "世界😁" and chr(0xFFFD) not in "".join(pieces)
```

“半个字符”的 token 并不是理论上的担心。WebBot 的分词器在 OpenWebText 上训练出了 5 万个 token（[7.1 节](ch07.html#s7-1)），其中有 157 个合并出来的 token 单独拿出来不是合法的 UTF-8：150 个以不完整的序列结尾，例如 id 290 是 `E2 80`，它是弯引号 `’` `“` `”` 和破折号 `—` 共同的前两个字节；另外 7 个以后续字节开头，例如 id 6091 是 `82 AC`，它是欧元符号 `€`（`E2 82 AC`）的后两个字节。下面的本地测试读取真实的 WebBot 词表，确认这些数字，并用本页的解码器对 5 万个 token 逐个、对随机拼成的 token 序列逐段，与 DLFS6 的 `decode` 对拍：

```localcheck verify/test_appendix.py::test_a_decode_matches_dlfs6 title="WebBot 词表：157 个“半个字符”的 token；本页解码器与 DLFS6 的 decode 逐字相同"
```

下面的交互图可以输入任意字节，看看解码器怎样划分最大子部分。图里同时给出了浏览器内置 `TextDecoder` 的结果作为对照。

```widget appendix-utf8 caption="每个框是一个最大子部分：实线框是合法的序列，解出一个字符；虚线框是不合法的一段，整段只换成一个 �。"
{"mode": "decode", "hex": "E4 B8 41 C0 80 ED A0 80 F0 9F 98 81"}
```

## 小结

本附录打开了 `str.encode("utf-8")` 这个黑盒。UTF-8 用首字节开头的 1 的个数表示字节数，用 `10` 标记后续字节，把码位的二进制位依次填进模板。我们照着规则实现了编码器，并在全部 1,112,064 个码位上与 Python 逐字节对拍。三类字节的取值范围互不重叠，这让 UTF-8 可以从任意位置同步，也让 DLFS6 能放心地在原始字节里寻找 `<|endoftext|>` 来切分大文件。

规则的另一面是“不合法”：截断、孤立的后续字节、超长编码、代理区、超出 0x10FFFF。我们照着 Unicode 的表 3-7 实现了解码器，它的报错位置、原因和替换结果都与 Python 完全一致。`errors="replace"` 按最大子部分替换，所以半个汉字只换成一个 `�`；而逐个 token 解码会把一个汉字变成三个 `�`。

### 本附录所学的内容

- 三种编码的输出都是字节，所以任何字符都能写成 0–255 的数列；对英文和代码，UTF-8 最短（[验证 A-a](#v-A-a)、[验证 A-c](#v-A-c)）。
- UTF-8 按“首字节标记 + 每个后续字节 6 位”的模板编码，`あ` 是 `E3 81 82`；这条规则在全部 1,112,064 个码位上与 Python 一致（[验证 A-d](#v-A-d)、[验证 A-f](#v-A-f)）。
- 首字节、后续字节、ASCII 的取值范围互不重叠，有 13 个字节值永远不出现；因此 UTF-8 可以自同步（[验证 A-g](#v-A-g)、[验证 A-h](#v-A-h)）。
- 把切点挪到 `<|endoftext|>` 上，每一块都能单独解码，这就是 `find_chunk_boundaries` 的做法（[验证 A-i](#v-A-i)）。
- 超长编码、代理区、超出 0x10FFFF 都不合法，表 3-7 用“第 2 个字节的范围”把它们一并排除（[验证 A-j](#v-A-j)、[验证 A-k](#v-A-k)）。
- `errors="replace"` 把每个最大子部分换成一个 U+FFFD：半个“世”是一个 `�`，逐 token 解码是三个；替换有损，流式输出要用增量解码（[验证 A-l](#v-A-l)、[验证 A-m](#v-A-m)）。

字节的问题到此为止。[附录 B](appendix-b.html) 回到第 2 章的 Attention，证明那个让 softmax 不再饱和的 $\sqrt{d}$ 从何而来。
