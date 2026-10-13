---
chapter: T
slug: train-in-browser
order: 165
kind: extra
title: 在浏览器里从零训练一个迷你 GPT
nav_title: 浏览器里训练迷你 GPT
lede: 不安装任何东西，就在这个标签页里训练一个 9 万参数的字符级 GPT。前向、反向传播和 AdamW 全部手写，每一个梯度都与 DLFS6 的 GPT 逐个比对过；在一台普通的台式机上，半分钟左右就能看到 loss 从 ln V 降到 bigram 线以下。
pyodide_packages: [numpy]
---

[第 3 章](ch03.html)用 `ch03/01_pretrain.py` 训练 CodeBot：1,112 万个参数，每一步 32 条长 256 个 token 的序列，一共 2 万步。这样的训练要交给 GPU，而跑起来以后，我们能看到的只有终端里不停刷新的 loss。参数是怎样一步步被推到“会写代码”的位置上的？隔着一行 `loss.backward()`，这些都看不见。

这一页把同一件事缩小到浏览器的一个标签页里。模型与 CodeBot 是同一个类，只是小得多：2 层，9 万多个参数。分词器换成了最朴素的字符级，训练数据是从 TinyCodes（CodeBot 的语料）里切下的一小片。不同的是，这里不借助任何框架：前向传播、反向传播和 AdamW 都用 JavaScript 手写，在后台线程里运行。为了确认手写的代码没有算错，每一层的梯度都做了**梯度确认**（gradient check），也就是与数值微分逐个比较；整个模型的 loss 和梯度也都与 DLFS6 的 `codebot.model.GPT` 逐个**对拍**（parity check）过：用同样的输入运行两份独立的实现，逐个比较输出。

我们按这样的顺序走：先看一次录好的训练（T.1）；然后准备数据和分词器（T.2），搭模型（T.3），逐层手写反向传播（T.4），与 PyTorch 对拍（T.5）；接着亲手训练（T.6），并练习一个调试的好习惯，先让模型过拟合一个 batch（T.7）；最后把这个玩具放大回 CodeBot（T.8）。

## T.1 先来看结果

先来看一次已经录好的训练。下图是这个迷你 GPT 在 TinyCodes 上训练 2,000 步的过程，横轴是参数更新的次数，纵轴是 loss。拖动图下方的滑块，可以看到它在不同阶段从 `def ` 开始续写的文字。

```widget trainer-replay caption="训练 2,000 步，留出 loss 从 4.55 降到 1.94；第 450 步时还略高于 bigram 线，第 475 步已经低于它。这次录制的样例里，第 0 步是符号乱码；第 1,000 步出现了 # 注释、= 和 for；到第 2,000 步，已经能写出 self.row_string(string, …) 和 import 这样的片段。"
{"views": ["default"]}
```

图中有两条曲线。**训练 loss** 是模型在每一步新取来的那批文字（8 段，每段 64 个字符）上的成绩，在更新参数之前算，每 10 步取一次平均，画在这 10 步的最后一步上；**留出 loss**（held-out loss）是它在 86 篇从不参加训练的留出文档上的成绩：每个字符都预测一次，再取平均。三条虚线是三个参照点，从上到下依次是瞎猜（ln 96）、只看字符频率（unigram）、只看前一个字符（bigram）。它们为什么是这几个数，T.2 会亲手算出来。

先记住这条曲线的样子：开头几十步掉得飞快，之后越来越慢。下面的验证单元直接读取录下来的运行记录，把几个关键时刻列出来：

```verify id=T.1-a title="录制的训练：几个关键时刻" files=data/train-in-browser/runs.json
import json

rec = json.load(open("data/train-in-browser/runs.json", encoding="utf-8"))
run, base = rec["runs"]["default"], rec["baselines"]
held = dict((s, x) for s, x in run["held"])            # 每 25 步测一次的留出 loss
first_below = lambda y: next(s for s, x in run["held"] if x < y)
for s in (0, 100, 450, 475, 1000, 2000):
    print(f"第 {s:>5,} 步  留出 loss {held[s]:.3f}")
print(f"第一次低于 unigram 线（{base['unigram']:.3f}）：第 {first_below(base['unigram'])} 步")
print(f"第一次低于 bigram 线（{base['bigram']:.3f}）：第 {first_below(base['bigram'])} 步")
train_end = sum(run["losses"][-10:]) / 10              # 最后 10 步的训练 loss
print(f"第 2,000 步：训练 loss（最后 10 步平均）{train_end:.3f}，留出 loss {held[2000]:.3f}")
assert abs(held[0] - 4.5459) < 1e-3 and abs(held[1000] - 2.2783) < 1e-3 and abs(held[2000] - 1.9377) < 1e-3
assert abs(train_end - 1.8686) < 1e-3
assert first_below(base["unigram"]) == 25 and first_below(base["bigram"]) == 475 and held[450] > base["bigram"]
assert rec["heldOut"]["targets"] == 25_150              # 留出文本 25,151 个 token，除第一个外每个都被预测一次
```

这条曲线可以原样复现。录制脚本 `record_runs.mjs`（在 `tests/js/fixtures/trainer/` 目录下）用的就是本页的同一份代码，随机种子固定为 0。下面的测试先把每一次录制的前 20 步重新跑一遍，与录下的 loss 逐位对照；再把这次训练的前 500 步完整地重跑一遍，每一步的 loss、每一次的留出 loss、每一段样例都要与录制相同；最后检查本页引用的每一个里程碑：

```localcheck verify/test_trainer.py::test_recorded_runs_replay title="录制的运行可以逐步复现"
```

要花多少时间呢？在录制用的台式机上（Intel i7-11700F），一步训练约 34 毫秒，在整份留出文本上测一次 loss 约 0.7 秒，写一段样例约 0.15 秒：

```verify id=T.1-b title="录制的机器上，训练器走到第 475 步、第 1,000 步要多久"
tm = rec["timing"]                                    # 录制的机器上的计时（中位数，单位毫秒）
def live(steps):                                      # T.6 的训练器从点下“开始训练”起，走到第 steps 步要几秒
    return (steps * tm["stepMs"]["simd"] + steps // 25 * tm["heldOutMs"] + steps // 100 * tm["sampleMs"]) / 1000
print(f"一步训练 {tm['stepMs']['simd']:.1f} 毫秒，测一次留出 loss {tm['heldOutMs'] / 1000:.2f} 秒，写一段样例 {tm['sampleMs'] / 1000:.2f} 秒")
print(f"第 475 步约在第 {live(475):.0f} 秒，1,000 步约 {live(1000):.0f} 秒，其中测留出 loss 占 {40 * tm['heldOutMs'] / 1000 / live(1000):.0%}")
print(rec["provenance"]["timingLoad"])
assert 30 < tm["stepMs"]["simd"] < 37 and 650 < tm["heldOutMs"] < 750 and 120 < tm["sampleMs"] < 180
assert round(live(475)) == 30 and round(live(1000)) == 63
assert 0.4 < 40 * tm["heldOutMs"] / 1000 / live(1000) < 0.5 and round(tm["cpuBusy"] * 100) == 42
```

照这个速度，T.6 的训练器从点下“开始训练”算起，约 30 秒走到越过 bigram 线的第 475 步，1,000 步约 1 分钟，其中将近一半的时间花在测留出 loss 上。计时的时候，这台机器上还有别的任务在跑（整机 CPU 占用 42%，计时程序只用 1 个核）。你的机器有多快，T.6 的训练器会实时显示。

## T.2 数据与分词器

### 从 TinyCodes 里切一小片

CodeBot 的预训练数据是 `codebot/tiny_codes.txt`：一份 6,487,033 字节的 Python 代码语料，21,836 篇短文档之间用 `<|endoftext|>` 隔开。本书把网页上的单个数据文件控制在 2 MB 以内，所以我们只取一小片：每 25 篇取 1 篇（共 874 篇），跳过其中 10 篇含有制表符或控制字符的文档；剩下的文档中，每 10 篇留 1 篇不参加训练，专门用来测留出 loss。

这样得到 778 篇训练文档（220,437 个字符）和 86 篇留出文档（26,171 个字符），两个文件加起来不到 250 KB。这里数的是文件里的字符：每个 `<|endoftext|>` 在文件里占 13 个字符，分词后只算 1 个 token，所以下面数出来的 token 要少一些。切片的规则写在 `verify/_trainer_data.py` 里，下面的测试会从原书语料重新切一遍，确认与网页上的文件逐字节相同。

```localcheck verify/test_trainer.py::test_data_slice_is_reproducible title="网页上的数据就是原书语料的切片"
```

### 字符级分词器

分词器用最朴素的一种：一个字符就是一个 token。这正是 [1.1 节](ch01.html#s1-1)的字符级分词器，只改了两处。一是只给训练文本里真正出现过的字符编号，按码位排好，从 0 开始（1.1 节直接拿 Unicode 码位当 id，词表有 111 万多行）；二是像 CodeBot 的分词器那样，把 `<|endoftext|>` 放在词表的最后（CodeBot 是 id 999）。

```js
export class CharTokenizer {
  constructor(text) {
    this.chars = [...new Set(text.split(EOT).join(''))].sort();
    this.endId = this.chars.length;
    this.vocabSize = this.chars.length + 1;
    this.index = new Map(this.chars.map((c, i) => [c, i]));
  }
```

这是训练器的 JavaScript 源码（`web/widgets/_tinygpt.js`）。本页引用的 JavaScript 片段都从这个文件原样摘出，下面的测试会逐段核对：

```localcheck verify/test_trainer.py::test_page_js_excerpts_are_verbatim title="本页的 JavaScript 片段与源码逐字相同"
```

再用 NumPy 写一个同样的分词器，拿网页上的两个数据文件数一数：

```verify id=T.2-a title="字符级分词器：95 种字符加 <|endoftext|>，V = 96" files=data/train-in-browser/tinycodes-train.txt,data/train-in-browser/tinycodes-val.txt
import math
import numpy as np

EOT = "<|endoftext|>"
train = open("data/train-in-browser/tinycodes-train.txt", encoding="utf-8", newline="").read()
val = open("data/train-in-browser/tinycodes-val.txt", encoding="utf-8", newline="").read()

chars = sorted(set(train.replace(EOT, "")))     # 训练文本里出现过的字符，按码位排序
V = len(chars) + 1                              # 最后一个 id 留给 <|endoftext|>
end_id = V - 1
lut = np.full(128, -1)                          # ASCII 码 -> token id
lut[[ord(c) for c in chars]] = np.arange(len(chars))
table = np.frombuffer("".join(chars).encode("ascii"), np.uint8)   # token id -> ASCII 码

def encode(text):
    pieces = [lut[np.frombuffer(p.encode("ascii"), np.uint8)] for p in text.split(EOT)]
    out = pieces[:1]
    for p in pieces[1:]:
        out += [np.array([end_id]), p]
    return np.concatenate(out)

def decode(ids):
    pieces = np.split(ids, np.flatnonzero(ids == end_id))
    return EOT.join(table[p[p != end_id]].tobytes().decode("ascii") for p in pieces)

t, v = encode(train), encode(val)
print(f"字符 {len(chars)} 种 + <|endoftext|>，V = {V}")
print("没出现的可打印 ASCII 字符:", [chr(c) for c in range(32, 127) if lut[c] < 0])
print(f"训练 {len(t):,} 个 token，留出 {len(v):,} 个 token")
print("encode('def ') =", encode("def ").tolist())
assert V == 96 and end_id == 95 and t.min() >= 0
assert decode(t) == train and decode(v) == val
assert (len(t), len(v)) == (211_113, 25_151)
assert len(t) == len(train) - train.count(EOT) * (len(EOT) - 1)   # 每个 <|endoftext|>：文件里 13 个字符，分词后 1 个 token
```

TinyCodes 是纯 ASCII 文本，出现过的字符有 95 种：换行符，加上 94 个可打印字符（只有 `~` 一次也没出现）。再加上 `<|endoftext|>`，词表大小 V = 96。

字符级的代价，[1.2 节](ch01.html#s1-2)已经见过：序列变长。同一段留出文本，CodeBot 的 BPE 分词器切成 10,659 个 token，字符级切成 25,151 个，是 BPE 的 2.36 倍。上下文同样是 64 个 token，字符级模型一次只能看到 64 个字符。在浏览器里，我们接受这个代价，换来一张只有 96 行的 embedding 表。

```localcheck verify/test_trainer.py::test_bpe_sequences_are_shorter title="同一段文字，BPE 的 token 数是字符级的 1/2.36"
```

### loss 的三个参照点

训练开始之前，先想清楚 loss 应该落在哪里，看曲线时心里才有数。我们准备三个参照点。

第一个是瞎猜。如果模型对 96 个 token 一视同仁，每个都给 1/96 的概率，那么不管正确答案是什么，交叉熵都是 $-\ln(1/96) = \ln 96 \approx 4.564$。随机初始化的模型输出几乎是均匀的，所以 loss 应该从这附近出发。

第二个是 **unigram**：只统计每个字符出现的频率，完全不看上下文。代码里最常见的字符是空格，占 16.9%（缩进贡献了很多）；如果下一个字符恰好是空格，unigram 模型的 loss 是 $-\ln 0.169 \approx 1.78$，遇到罕见字符时则会很大。把留出文本上所有位置的 loss 平均起来，就是

$$H_1 = -\frac{1}{N}\sum_{i=1}^{N} \ln p(x_i)$$

这里，$x_i$ 是留出文本的第 $i$ 个 token，$p(x_i)$ 是这个字符在训练文本中出现的频率，$N$ 是留出文本的 token 数。

第三个是 **bigram**：统计“当前字符是 a 时，下一个字符是 b”的频率，用前一个字符来预测下一个：

$$H_2 = -\frac{1}{N-1}\sum_{i=2}^{N} \ln p(x_i \mid x_{i-1})$$

这里，$p(x_i \mid x_{i-1})$ 是训练文本里字符 $x_{i-1}$ 后面紧跟着 $x_i$ 的比例。比如空格后面，下一个还是空格的比例是 36.4%（缩进里的连续空格），bigram 在这样的位置只付 $-\ln 0.364 \approx 1.01$；unigram 不看前一个字符，同一个位置要付 1.78。

两个计数模型都在每个计数上加了 0.1。这叫**加 k 平滑**（add-k smoothing）：留出文本里可能出现训练时没见过的组合，不加平滑，这些组合的概率就是 0，loss 会变成无穷大。

```verify id=T.2-b title="三个参照点：ln V、unigram、bigram"
k = 0.1                                       # 加 k 平滑：没见过的组合也留一点概率
uni = np.bincount(t, minlength=V) + k
p1 = uni / uni.sum()                          # p(c)
big = np.bincount(t[:-1] * V + t[1:], minlength=V * V).reshape(V, V) + k
p2 = big / big.sum(axis=1, keepdims=True)     # p(下一个 | 当前)

ln_v = math.log(V)
h1 = -np.log(p1[v]).mean()                    # 在留出文本上的交叉熵
h2 = -np.log(p2[v[:-1], v[1:]]).mean()
for name, x in [("瞎猜  ln V", ln_v), ("unigram", h1), ("bigram ", h2)]:
    print(f"{name}: {x:.4f} nats = {x / math.log(2):.4f} bits")
top = np.argsort(-p1)[:3]
print("最常见的字符:", ", ".join(f"{chars[i]!r} {p1[i]:.1%}" for i in top))
sp = chars.index(" ")
print(f"空格后面还是空格：p = {p2[sp, sp]:.3f}，bigram 付 {-math.log(p2[sp, sp]):.2f}，unigram 付 {-math.log(p1[sp]):.2f}")
assert abs(ln_v - 4.5643) < 1e-4 and abs(h1 - 3.3867) < 1e-4 and abs(h2 - 2.6931) < 1e-4
assert abs(ln_v / math.log(2) - math.log2(96)) < 1e-12   # 1 nat = 1/ln 2 = 1.4427 bits
assert [chars[i] for i in top] == [" ", "e", "t"] and abs(p1[sp] - 0.169) < 5e-4   # 正文：空格占 16.9%，其次是 e 和 t
assert abs(p2[sp, sp] - 0.364) < 5e-4 and abs(-math.log(p1[sp]) - 1.78) < 0.005   # 正文：36.4%，1.01 对 1.78
```

这样我们就有了一把尺子：瞎猜 4.564，unigram 3.387，bigram 2.693。loss 降到 unigram 线以下，说明模型至少学会了字符的频率；降到 bigram 线以下，说明它至少做到了“只看前一个字符”的计数模型能做到的程度。

那么，loss 一低于 bigram 线，是不是就说明模型用上了更远的上下文呢？还不能这么说。bigram 线只是一个估计：平滑常数换一换，它就会挪动几个千分点；而且它的计数来自训练文本，未必是“只看前一个字符”所能做到的最好。要下这个结论，需要一条任何只看前一个字符的模型都越不过去的线。这条线可以这样得到：直接在留出文本上数 bigram，再用它给同一份留出文本打分。这等于偷看了答案，只看前一个字符的模型，没有谁能比它更低。

```verify id=T.2-c title="只看前一个字符，留出 loss 不可能低于 2.562"
lines = {}
for kk in (0.03, 0.1, 0.3):                           # 平滑常数不同，bigram 线略有不同
    bk = np.bincount(t[:-1] * V + t[1:], minlength=V * V).reshape(V, V) + kk
    lines[kk] = -np.log((bk / bk.sum(1, keepdims=True))[v[:-1], v[1:]]).mean()
    print(f"k = {kk}: bigram 线 {lines[kk]:.4f}")
cv = np.bincount(v[:-1] * V + v[1:], minlength=V * V).reshape(V, V)   # 在留出文本上数 bigram（偷看答案）
pv = cv / np.maximum(cv.sum(1, keepdims=True), 1)
floor = -np.log(pv[v[:-1], v[1:]]).mean()                          # 再给同一份留出文本打分
print(f"只看前一个字符的下限: {floor:.4f}")
assert 0.003 < max(abs(x - h2) for x in lines.values()) < 0.01     # 正文：挪动几个千分点
assert abs(floor - 2.5622) < 1e-4 and floor < h2 - 0.13
```

算出来是 2.562，比 bigram 线低 0.13。对每一种“前一个字符”，用留出文本自己的比例去猜，交叉熵最小，换成任何别的分布都只会更大（这是交叉熵的性质：真实分布本身让它最小）。所以，loss 降到 2.562 以下，模型就一定用上了比前一个字符更远的上下文。而把远处的字符搬过来，正是 Attention 的工作。

::: note nats 与 bits
本页的 loss 都以 e 为底，单位是 **nats**；以 2 为底时，单位是 bits，1 nat = 1/ln 2 ≈ 1.4427 bits。所以 ln 96 nats 正好等于 log₂ 96 ≈ 6.585 bits，也就是从 96 个符号里均匀地挑一个所需的信息量。PyTorch 的 `F.cross_entropy` 用自然对数，CodeBot 训练日志里的 loss 也是 nats（T.5 的对拍会确认两边的 loss 一致到 1e-12）。
:::

## T.3 模型：同一张图纸

数据和尺子都有了，下面来搭模型。迷你 GPT 的结构与 CodeBot 完全相同。下面是 `codebot/model.py` 的 GPT 类：token embedding 加上可学习的位置 embedding，`n_layer` 个 Transformer 块，一个 LayerNorm，最后由 `unembed` 映射回词表。注意倒数第二行的 `self.embed.weight = self.unembed.weight`：`unembed` 与 `embed` 共享同一个权重矩阵（**权重共享**，weight tying），但 `unembed` 保留了自己的 bias。

```listing codebot/model.py#L102-L126
```

不同的只是尺寸。下表把两者并排：

| | CodeBot（`ch03/01_pretrain.py`） | 迷你版 | 为什么这样改 |
|---|---|---|---|
| 词表 `vocab_size` | 1,000（BPE） | 96（字符） | 字符级分词，见 T.2 |
| 上下文 `context_len` | 256 | 64 | Attention 的打分矩阵是 64 × 64，计算量随长度的平方增长 |
| embedding 维度 `embed_dim` | 384 | 64 | 让每一步只要几十毫秒（见 T.6） |
| 头数 `n_head` | 6 | 4 | 每个头 16 维 |
| 层数 `n_layer` | 6 | 2 | 同上 |
| FFN 宽度 `ff_dim` | 1,536（4 倍） | 192（3 倍） | 参数预算 10 万以内（4 倍是 109,920） |
| `dropout_rate` | 0.1 | 0 | 见下文 |
| 参数量 | 11,121,640 | 93,408 | 约为 1/119 |

`embed_dim` 是残差流的宽度。同一个量，CS336 叫 `d_model`，Hugging Face 的模型配置一般叫 `hidden_size`，GPT-2 的配置里叫 `n_embd`。dropout 设为 0 有两个原因：一是 2,000 步时训练 loss（1.87）和留出 loss（1.94）还很接近，看不出过拟合，用不着这种正则化；二是 dropout 会让每次前向传播都带随机性，梯度确认和对拍都会变得麻烦。

参数量可以逐项加出来。以迷你版为例（权重共享的矩阵只算一次）：

| 部分 | 形状 | 参数 |
|---|---|---|
| `embed.weight`（也是 `unembed.weight`） | 96 × 64 | 6,144 |
| `pos_embed.weight` | 64 × 64 | 4,096 |
| 每个块的 `norm1`、`norm2` | 2 × (64 + 64) | 256 |
| 每个块的 `W_q`、`W_k`、`W_v`、`W_o`（没有 bias） | 4 × 64 × 64 | 16,384 |
| 每个块的 FFN | 64 × 192 + 192 + 192 × 64 + 64 | 24,832 |
| 两个块 | 2 × (256 + 16,384 + 24,832) | 82,944 |
| 最后的 `norm` | 64 + 64 | 128 |
| `unembed.bias` | 96 | 96 |
| 合计 | | 93,408 |

```verify id=T.3-a title="参数量：迷你版 93,408，CodeBot 11,121,640"
def gpt_params(V, C, E, L, F):
    block = 2 * E + 4 * E * E + 2 * E + (E * F + F) + (F * E + E)  # norm1 + Q/K/V/O（无 bias）+ norm2 + FFN
    return V * E + C * E + L * block + 2 * E + V                   # embed（与 unembed 共享）+ 位置 + 各块 + norm + unembed 的 bias

mini = gpt_params(V=96, C=64, E=64, L=2, F=192)
codebot = gpt_params(V=1000, C=256, E=384, L=6, F=4 * 384)
print(f"迷你版 : {mini:>10,}")
print(f"CodeBot: {codebot:>10,}  （约 {codebot / mini:.0f} 倍）")
assert mini == 93_408 and codebot == 11_121_640
assert 96 * 64 + 64 * 64 + 2 * (256 + 16_384 + 24_832) + 128 + 96 == mini   # 正文表格里逐项相加的结果
assert gpt_params(V=96, C=64, E=64, L=2, F=256) == 109_920                  # FFN 取 4 倍时
```

公式算出的数，还要拿原书的类核对一遍。下面的测试用这组超参数构造 `codebot.model.GPT`，确认 `sum(p.numel() for p in model.parameters())` 也是 93,408：

```localcheck verify/test_trainer.py::test_param_counts title="DLFS6 的 GPT 在迷你尺寸下有 93,408 个参数"
```

JavaScript 这边，每个参数都按 PyTorch 的 `state_dict` 取名，形状也与 PyTorch 相同。这样，两边的权重可以直接互相搬运，T.5 的对拍就靠这一点：

```js
  m.tok = add('embed.weight', [V, E], 'normal'); // also unembed.weight (tied)
  m.pos = add('pos_embed.weight', [C, E], 'normal');
```

### 初始 loss

初始化同样照搬 `_init_weights`：所有线性层和 embedding 的权重取自 $\mathcal{N}(0, 0.02^2)$，bias 为 0。LayerNorm 不在 `_init_weights` 里，用的是 `nn.LayerNorm` 的默认值：γ 为 1，β 为 0。

```listing codebot/model.py#L128-L134
```

这样初始化之后，loss 应该是多少？最后一个 LayerNorm 的输出每一维的方差约为 1，乘上标准差 0.02 的权重之后，每个 logit 近似服从 $\mathcal{N}(0, \sigma^2)$，其中 $\sigma = 0.02 \times \sqrt{64} = 0.16$。如果 logits 与正确答案毫无关系，交叉熵的期望约为

$$\mathbb{E}[\text{loss}] \approx \ln V + \frac{\sigma^2}{2}$$

这里，$\ln V$ 是瞎猜的 loss，$\sigma^2/2$ 来自 $\mathbb{E}[e^{z}] = e^{\sigma^2/2}$（$z \sim \mathcal{N}(0,\sigma^2)$）：logits 有了起伏，归一化项 $\ln\sum e^{z}$ 平均会变大一点。代入数字，$4.564 + 0.16^2/2 = 4.577$。

```verify id=T.3-b title="随机初始化：loss ≈ ln V + σ²/2"
rng = np.random.default_rng(0)
E, n = 64, 4096
W = 0.02 * rng.standard_normal((V, E))           # embed.weight（与 unembed.weight 共享）的初始值
h = rng.standard_normal((n, E))
h = (h - h.mean(1, keepdims=True)) / h.std(1, keepdims=True)   # LayerNorm 之后：均值 0、方差 1
logits = h @ W.T
targets = rng.integers(0, V, n)                  # 与 logits 毫无关系的目标
ce = (np.log(np.exp(logits).sum(1)) - logits[np.arange(n), targets]).mean()
sigma = logits.std()
print(f"logits 的标准差 {sigma:.4f}（0.02 * sqrt(64) = {0.02 * 8:.4f}）")
print(f"交叉熵 {ce:.4f}，ln V + sigma^2/2 = {ln_v + sigma ** 2 / 2:.4f}，ln V = {ln_v:.4f}")
assert abs(sigma - 0.16) < 0.005
assert abs(ce - (ln_v + sigma ** 2 / 2)) < 0.01 and ce > ln_v
```

::: think T.1 的曲线在第 0 步的留出 loss 是 4.546，比瞎猜的 ln 96 = 4.564 还低。一个随机初始化的模型，怎么会比瞎猜还准？
:::
::: answer
原因在权重共享。初始时，残差流里带着当前字符自己的 embedding 向量 $e_c$，而 `unembed` 用的恰好是同一张表，所以“下一个字符还是 c”这一项的 logit 会比别的字符高出一截。代码里，下一个字符与当前字符相同的情况并不少见（缩进里连续的空格、`==`、`''`），占了 8.58%。模型一出生就带着这个偏好，而这个偏好碰巧对了一部分，loss 就略低于 ln V。

```verify id=T.3-c title="权重共享让初始模型偏爱“重复当前字符”"
same = (t[1:] == t[:-1]).mean()
c = rng.integers(0, V, n)                         # 当前字符
x = W[c] + 0.02 * rng.standard_normal((n, E))     # 残差流 ≈ 字符 embedding + 位置 embedding
hx = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
lg = hx @ W.T                                     # unembed 用的正是同一个 W
boost = (lg[np.arange(n), c] - lg.mean(1)).mean()
print(f"训练文本里“下一个字符与当前字符相同”的比例: {same:.2%}")
print(f"初始化时，“与当前字符相同”的 logit 平均高出其他字符 {boost:.2f}")
assert abs(same - 0.0858) < 5e-4 and boost > 0.5
```

这个玩具省略了两个 Transformer 块的输出。真实模型的残差流里还有它们，偏好被冲淡了一些：在 DLFS6 的 GPT 上实测，高出约 0.6。下面的测试还把三种情况都量了一遍：在真实代码上，初始 loss 低于 ln V；换成均匀随机的 token，loss 回到 ln V + σ²/2；解除权重共享之后，在真实代码上也回到 ln V 附近。

```localcheck verify/test_trainer.py::test_initial_loss_is_below_ln_v_because_of_tied_embeddings title="DLFS6 的 GPT：真实文本、随机 token、解除权重共享"
```

这个偏好能让 loss 低多少？可以粗算一下。在 8.58% 的位置上，正确答案恰好与当前字符相同，它的 logit 高出约 0.6，这些位置的 loss 就少了约 0.6。代价是归一化项变大：$\ln\sum e^{z}$ 里多出一项 $e^{0.6} - 1$，平均每个位置要多付 $\ln\left(1 + (e^{0.6} - 1)/96\right) \approx 0.0085$。合起来，$4.577 - 0.0858 \times 0.6 + 0.0085 \approx 4.534$，与录制的第 0 步（4.546）只差 0.01 左右。

```verify id=T.3-d title="粗算：权重共享让初始 loss 低了约 0.04"
gain = same * 0.6                                       # 8.58% 的位置，正确答案的 logit 高出约 0.6
cost = math.log(1 + (math.exp(0.6) - 1) / V)            # 每个位置的归一化项多付的部分
est = ln_v + 0.16 ** 2 / 2 - gain + cost
print(f"{ln_v + 0.16 ** 2 / 2:.3f} - {gain:.4f} + {cost:.4f} = {est:.3f}（录制的第 0 步：{held[0]:.3f}）")
assert abs(cost - 0.0085) < 1e-4 and abs(gain - cost - 0.043) < 1e-3 and abs(est - 4.534) < 1e-3
assert abs(est - held[0]) < 0.015
```
:::

## T.4 反向传播：每一层都自己写

在 PyTorch 里，`loss.backward()` 一行就算完所有参数的梯度。浏览器里没有 PyTorch，梯度要我们自己算。好在 GPT 只由几种零件搭成，每种零件只需要写一次 forward 和一次 backward。

如果你读过《深度学习入门：基于Python的理论与实现》第 5 章，对这种写法一定不陌生：每一层都是一个带 forward 和 backward 的小零件，正向传播按顺序调用，反向传播按逆序调用。迷你 GPT 一共只用到下面 6 种零件（残差连接是加法，梯度原样流过，就不单独列了）。表中 dy 是上游传来的梯度 ∂L/∂y。

| 层 | 前向 | 对输入的梯度 | 对参数的梯度 |
|---|---|---|---|
| embedding | x = E[id] + P[t] | — | 按 id 累加 dy |
| 线性 | y = xWᵀ + b | dx = dy·W | dW = dyᵀx，db = Σdy |
| LayerNorm | y = γ⊙x̂ + β | 见下文 | dγ = Σ dy⊙x̂，dβ = Σ dy |
| GELU | y = x·Φ(x) | dx = dy⊙(Φ(x) + x·φ(x)) | — |
| 因果 Attention | O = softmax(QKᵀ/√D + M)·V | 见下文 | — |
| softmax + 交叉熵 | L = −(1/N)·Σ ln p(答案) | (p − onehot)/N | — |

这六种零件，在 PyTorch 里依次是 `nn.Embedding`、`nn.Linear`、`nn.LayerNorm`、`nn.GELU()`、`codebot/model.py` 里的 `MultiHeadAttention` 和 `F.cross_entropy`。

下面按反向传播的顺序，从 loss 出发往回走：先看 softmax 与交叉熵，最后看 embedding。每个零件都配一个 NumPy 的验证单元：先按公式算出梯度，再用**数值微分**（中心差分）独立算一遍，两者应当一致。这就是章首说的梯度确认。

### softmax 与交叉熵

如果你读过《深度学习入门：基于Python的理论与实现》5.6.3 节，会记得 Softmax-with-Loss 层的反向传播“漂亮地”等于 $y - t$。语言模型的输出层是同一个结构，只是类别从 10 个变成了 V 个，再对 N 个位置取平均，所以 logits 的梯度是 $(p - \text{onehot})/N$。

举一个能心算的例子。3 个类别的 logits 是 (2, 1, 0.1)，正确答案是第 0 类。softmax 得到 p = (0.659, 0.242, 0.099)，减去 one-hot (1, 0, 0)，梯度就是 (−0.341, 0.242, 0.099)：正确答案的 logit 被往上推，其余的被往下压，推压的力度正好是“还差多少”。

```verify id=T.4-a title="交叉熵的梯度 = softmax − one-hot"
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def num_grad(f, x, h=1e-5):
    """中心差分：每次只动 x 的一个元素。"""
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + h; fp = f()
        x[i] = old - h; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g

def rel_err(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))

z = np.array([[2.0, 1.0, 0.1]])                  # 一个位置，3 个类别
y = np.array([0])                                # 正确答案是第 0 类
p = softmax(z)
grad = p.copy()
grad[0, y[0]] -= 1                               # (softmax - one-hot) / N，这里 N = 1
print("softmax:", np.array2string(p, precision=4))
print("梯度   :", np.array2string(grad, precision=4))
ce_loss = lambda: -np.log(softmax(z)[0, y[0]])
assert rel_err(grad, num_grad(ce_loss, z)) < 1e-9
assert np.allclose(grad, [[-0.3410, 0.2424, 0.0986]], atol=1e-4)
```

### 线性层

线性层 $y = xW^\top + b$ 的反向传播有三个公式：$dx = dy\,W$，$dW = dy^\top x$，$db$ 是 $dy$ 沿 batch 方向的和。这里，$x$ 的形状是 $[N, \text{in}]$，$W$ 是 $[\text{out}, \text{in}]$（PyTorch 的存法），$dy$ 是 $[N, \text{out}]$。举一个能心算的例子：$x = (1, 2)$，$W = (3, 4)$（out = 1，in = 2），$dy = (1)$，于是 $dx = dy\,W = (3, 4)$，$dW = dy^\top x = (1, 2)$，$db = 1$。

公式不必死记：$dW$ 的形状必须与 $W$ 相同，是 $[\text{out}, \text{in}]$，能凑出这个形状的乘法只有 $dy^\top x$。如果你读过《深度学习入门：基于Python的理论与实现》5.6.2 节或《深度学习入门2：自制框架》步骤 41，用的正是这个“按形状反推”的技巧。不过那两本书把 $W$ 存成 $[\text{in}, \text{out}]$，所以写作 $dX = dY\,W^\top$、$dW = X^\top dY$；PyTorch 存成 $[\text{out}, \text{in}]$，转置换了位置，两套公式说的是同一件事。

```verify id=T.4-b title="线性层：dx = dy·W，dW = dyᵀ·x，db = Σdy"
x1, W1, dy1 = np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]), np.array([[1.0]])   # 正文里心算的例子
assert (dy1 @ W1).tolist() == [[3.0, 4.0]] and (dy1.T @ x1).tolist() == [[1.0, 2.0]] and dy1.sum(0).tolist() == [1.0]
rng = np.random.default_rng(1)
x, Wl, b = rng.normal(size=(3, 4)), rng.normal(size=(2, 4)), rng.normal(size=2)
R = rng.normal(size=(3, 2))                      # 上游传来的梯度 dL/dy（L = sum(y * R)）
L = lambda: ((x @ Wl.T + b) * R).sum()
dx, dW, db = R @ Wl, R.T @ x, R.sum(axis=0)      # 反向传播的三个公式
for name, a, ref in [("dx", dx, x), ("dW", dW, Wl), ("db", db, b)]:
    e = rel_err(a, num_grad(L, ref))
    assert e < 1e-9 and a.shape == ref.shape
    print(f"{name}: 形状 {a.shape}，与数值微分一致（相对误差 < 1e-9）")
Win = Wl.T                                        # 同一个矩阵存成 [in, out]，即《深度学习入门》的写法
assert np.allclose(R @ Win.T, dx) and np.allclose(x.T @ R, dW.T)   # dX = dY·Wᵀ、dW = Xᵀ·dY：同一件事
```

JavaScript 版做的是同样的三件事。`matmulBT(A, B, C, …)` 计算 $C = AB^\top$：模型里所有的矩阵乘法，配上转置，都能写成这一种形式，于是只需要把这一个函数写快。

```js
export function linearBackward(dy, x, W, dx, dW, dBias, N, nIn, nOut) {
  const Arr = dy.constructor;
  const Wt = scratch(Arr, 'Wt', nIn * nOut), dyT = scratch(Arr, 'dyT', nOut * N);
  const xT = scratch(Arr, 'xT', nIn * N), g = scratch(Arr, 'gW', nOut * nIn);
  transpose(W, Wt, nOut, nIn);
  matmulBT(dy, Wt, dx, N, nOut, nIn); // dx = dy W
  transpose(dy, dyT, N, nOut);
  transpose(x, xT, N, nIn);
  matmulBT(dyT, xT, g, nOut, N, nIn); // dy^T x
  for (let i = 0; i < nOut * nIn; i++) dW[i] += g[i];
  if (dBias) {
    for (let o = 0; o < nOut; o++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += dyT[o * N + n];
      dBias[o] += s;
    }
  }
}
```

### LayerNorm

LayerNorm 先把每个 token 的向量标准化，再缩放和平移：

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \odot \hat{x} + \beta$$

这里，$\mu$ 和 $\sigma^2$ 是这个向量 64 个分量的均值和（有偏）方差，$\epsilon = 10^{-5}$。反向传播的难点在于，$\mu$ 和 $\sigma$ 本身也依赖 $x$。记 $r = 1/\sqrt{\sigma^2 + \epsilon}$，则 $\hat{x}_j = (x_j - \mu)\,r$。对 $x_i$ 求导，$\mu$ 和 $r$ 都要跟着变：

$$\frac{\partial \hat{x}_j}{\partial x_i} = r\left(\delta_{ij} - \frac{1}{64}\right) - \frac{r}{64}\,\hat{x}_i\,\hat{x}_j$$

这里，$\delta_{ij}$ 在 $i = j$ 时为 1，否则为 0。括号里的 $-1/64$ 来自 $\mu$（$\partial\mu/\partial x_i = 1/64$），后一项来自 $\sigma$（$\partial r/\partial x_i = -r^2\hat{x}_i/64$）。记 $g = dy \odot \gamma$，它是 loss 对 $\hat{x}$ 的梯度；按链式法则 $dx_i = \sum_j g_j\,\partial\hat{x}_j/\partial x_i$，就得到

$$dx = r\left(g - \overline{g} - \hat{x} \odot \overline{g \odot \hat{x}}\right)$$

其中上划线表示对这 64 个分量取平均。它有一个可以心算的性质：给 $x$ 的所有分量加上同一个数，输出不变，所以 $dx$ 各分量之和必须为 0。

```verify id=T.4-c title="LayerNorm 的反向传播，以及“dx 各行之和为 0”"
def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                # 有偏方差，与 nn.LayerNorm 相同
    xhat = (x - mu) / np.sqrt(var + eps)
    return xhat * g + b, xhat, 1 / np.sqrt(var + eps)

x = rng.normal(size=(3, 5)) * 2
g, bb = rng.normal(size=5), rng.normal(size=5)
R = rng.normal(size=(3, 5))
y, xhat, rstd = layer_norm(x, g, bb)
gx = R * g                                        # 对 xhat 的梯度
dx = rstd * (gx - gx.mean(-1, keepdims=True) - xhat * (gx * xhat).mean(-1, keepdims=True))
dg, db = (R * xhat).sum(0), R.sum(0)
L = lambda: (layer_norm(x, g, bb)[0] * R).sum()
for name, a, ref in [("dx", dx, x), ("dgamma", dg, g), ("dbeta", db, bb)]:
    e = rel_err(a, num_grad(L, ref))
    assert e < 1e-8
    print(f"{name}: 与数值微分一致（相对误差 < 1e-8）")
assert np.allclose(dx.sum(-1), 0, atol=1e-12)     # 平移 x 不改变输出，所以梯度各行之和为 0
print("dx 每一行的和: 0（误差 < 1e-12）")
k5 = x.shape[1]                                   # 这里的向量长 5，正文的 64 换成 5
J = rstd[0] * (np.eye(k5) - 1 / k5) - rstd[0] / k5 * np.outer(xhat[0], xhat[0])   # J[i, j] = ∂x̂_j/∂x_i
Jn = np.stack([num_grad(lambda: layer_norm(x, g, bb)[1][0, j], x)[0] for j in range(k5)], axis=1)
assert rel_err(J, Jn) < 1e-8
print("∂x̂_j/∂x_i 的式子与数值微分一致（相对误差 < 1e-8）")
```

如果你读过《深度学习入门：基于Python的理论与实现》6.3 节，那里的 Batch Norm 没有展开反向传播的推导。上面两行正好可以补上：Batch Norm 的推导与它是同一回事，只是均值和方差沿 batch 方向而不是特征方向计算。JavaScript 版逐行对应上面的公式：

```js
    for (let e = 0; e < E; e++) {
      const d = dy[o + e];
      dGamma[e] += d * xhat[o + e];
      dBeta[e] += d;
      const g = d * gamma[e];
      mg += g;
      mgx += g * xhat[o + e];
    }
    mg /= E; mgx /= E;
    const r = rstd[n];
    for (let e = 0; e < E; e++) dx[o + e] = r * (dy[o + e] * gamma[e] - mg - xhat[o + e] * mgx);
```

### GELU

`codebot/model.py` 的 FFN 用的是 `nn.GELU()`，也就是精确版 $\text{GELU}(x) = x\,\Phi(x)$，其中 $\Phi$ 是标准正态分布的累积分布函数，$\Phi(x) = \frac{1}{2}(1 + \text{erf}(x/\sqrt{2}))$。按乘法法则求导：

$$\text{GELU}'(x) = \Phi(x) + x\,\varphi(x)$$

这里，$\varphi(x) = e^{-x^2/2}/\sqrt{2\pi}$ 是标准正态分布的密度。代入 $x = 1$：$\Phi(1) = 0.8413$，$\varphi(1) = 0.2420$，所以 $\text{GELU}'(1) = 1.0833$，比 ReLU 在这里的导数 1 稍大一点。

```verify id=T.4-d title="GELU 的导数：Φ(x) + x·φ(x)"
erf = np.vectorize(math.erf)
gelu = lambda x: x * 0.5 * (1 + erf(x / math.sqrt(2)))                 # nn.GELU()，精确版
dgelu = lambda x: 0.5 * (1 + erf(x / math.sqrt(2))) + x * np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
print(f"GELU(1) = {gelu(1.0):.4f}，GELU'(1) = Phi(1) + phi(1) = {dgelu(1.0):.4f}")
xs = rng.normal(size=8) * 2
R = rng.normal(size=8)
e = rel_err(dgelu(xs) * R, num_grad(lambda: (gelu(xs) * R).sum(), xs))
assert abs(dgelu(1.0) - (0.8413 + 0.2420)) < 1e-4 and e < 1e-9
print("8 个随机点上，公式与数值微分一致（相对误差 < 1e-9）")
```

JavaScript 有 `Math.exp`，却没有 `Math.erf`。训练器里的 `erf` 移植自 fdlibm 的有理逼近（[Go 语言的 `math.Erf`](https://go.dev/src/math/erf.go) 用的也是这一份）。在 −7 到 7 之间，它与 120 位精度算出的参考值（mpmath）最多差 1 ulp，也就是双精度浮点数末位的一个单位；我们测了 10,908 个点，绝大多数完全相同。下一节会看到，这里如果偷懒用近似公式，对拍就过不去。

```localcheck verify/test_trainer.py::test_js_erf_within_one_ulp title="JavaScript 的 erf 与正确舍入的值最多差 1 ulp"
```

### 因果 Attention

Attention 的前向传播与 `codebot/model.py` 一一对应：打分、遮住未来、softmax、加权求和。

```listing codebot/model.py#L33-L41
```

反向传播从后往前走。记 $P$ 为 softmax 之后的注意力权重，$dO$ 为输出的梯度：

- $O = PV$ 是乘法，所以 $dV = P^\top dO$，而 $P$ 的梯度是 $G = dO\,V^\top$；
- softmax 的反向传播：$dS = P \odot \left(G - \text{rowsum}(P \odot G)\right)$，再乘上缩放 $1/\sqrt{D}$；
- $S = QK^\top$ 又是乘法：$dQ = dS\,K$，$dK = dS^\top Q$。

被遮住的位置，$P$ 恰好为 0，于是 $dS$ 也恰好为 0：第 $i$ 个位置的输出，不会把梯度传给它后面（$j > i$）的 key 和 value。因果性在反向传播中同样成立。

```verify id=T.4-e title="Attention 的反向传播：dQ、dK、dV 与数值微分一致"
T, Dh = 4, 3
q, kk, vv = rng.normal(size=(T, Dh)), rng.normal(size=(T, Dh)), rng.normal(size=(T, Dh))
dO = rng.normal(size=(T, Dh))                    # 上游梯度 dL/dO
mask = np.tril(np.ones((T, T), dtype=bool))

def attn(q, k, v):
    s = np.where(mask, q @ k.T / math.sqrt(Dh), -np.inf)
    P = softmax(s)
    return P @ v, P

O, P = attn(q, kk, vv)
G = dO @ vv.T                                    # dL/dP
dS = P * (G - (P * G).sum(-1, keepdims=True)) / math.sqrt(Dh)   # softmax 的反向，再乘缩放
dq, dk, dv = dS @ kk, dS.T @ q, P.T @ dO
L = lambda: (attn(q, kk, vv)[0] * dO).sum()
for name, a, ref in [("dQ", dq, q), ("dK", dk, kk), ("dV", dv, vv)]:
    e = rel_err(a, num_grad(L, ref))
    assert e < 1e-8
    print(f"{name}: 与数值微分一致（相对误差 < 1e-8）")
print("被遮住的位置（上三角）的 dS:", np.abs(dS[~mask]).max())
assert np.all(dS[~mask] == 0) and np.allclose(P.sum(-1), 1)
dO0 = np.zeros_like(dO); dO0[0] = dO[0]           # 只让第 0 个位置的输出带梯度
G0 = dO0 @ vv.T
dS0 = P * (G0 - (P * G0).sum(-1, keepdims=True)) / math.sqrt(Dh)
assert np.all((dS0.T @ q)[1:] == 0) and np.all((P.T @ dO0)[1:] == 0)   # 它后面的 key、value 一点梯度也拿不到
print("第 0 个位置的输出传给第 1–3 个 key、value 的梯度: 0")
```

JavaScript 版对每个 batch、每个头分别做这几步。其中算出 $G$、再做 softmax 反向传播的是这几行：

```js
      matmulBT(dO, V, G, T, D, T); // G[i, j] = dout_i . v_j
      for (let i = 0; i < T; i++) {
        const r = i * T;
        let dot = 0;
        for (let j = 0; j <= i; j++) dot += Pb[r + j] * G[r + j];
        for (let j = 0; j < T; j++) dS[r + j] = j <= i ? Pb[r + j] * (G[r + j] - dot) * scale : 0;
      }
```

### embedding 与权重共享

embedding 层是查表：第 t 个位置取出词表的第 id 行。反向传播就是把梯度按 id 加回去。同一个 token 出现几次，它那一行就收到几份梯度，所以是累加，不是覆盖。如果你读过《深度学习入门2：自制框架》附录 B，`np.add.at` 做的正是这件事。

权重共享让事情多了一层：同一个矩阵既用来查表，又在最后当 `unembed`。它从两条路径各得到一份梯度，两份相加。这就是《深度学习入门：基于Python的理论与实现》附录 A 说的“正向传播时有分支流出，反向传播时梯度相加”。

```verify id=T.4-f title="权重共享：查表与 unembed 两条路径的梯度相加"
Vs, Es = 5, 3
Wt = rng.normal(size=(Vs, Es))                    # 共享的权重：既查表，又当 unembed
ids = np.array([1, 3, 1])                        # 第 1 号 token 出现了两次
H = rng.normal(size=(3, Es))                     # 某一层的输出（这里当作常数）
R1, R2 = rng.normal(size=(3, Es)), rng.normal(size=(3, Vs))
L = lambda: (Wt[ids] * R1).sum() + ((H @ Wt.T) * R2).sum()   # 查表一次，unembed 一次
d_lookup = np.zeros_like(Wt)
np.add.at(d_lookup, ids, R1)                     # 查表的反向：按 id 累加（不是覆盖）
d_unembed = R2.T @ H                             # unembed 的反向：dW = dy^T x
e = rel_err(d_lookup + d_unembed, num_grad(L, Wt))
assert np.allclose(d_lookup[1], R1[0] + R1[2]) and e < 1e-9
print("第 1 行（查表部分）=", np.array2string(d_lookup[1], precision=3), "= R1[0] + R1[2]")
print("两条路径的梯度相加后，与数值微分一致（相对误差 < 1e-9）")
```

在训练器里，`unembed` 的梯度直接写进 `embed.weight` 的梯度缓冲区，反向传播走到最后，查表的梯度再加进同一个缓冲区：

```js
  // The unembedding shares embed.weight, so its gradient lands in the same buffer as the lookup's.
  linearBackward(ws.logits, ws.hf, m.tok.data, ws.dh, m.tok.grad, m.ubias.grad, N, E, V);
// … 中间是各个 Transformer 块的反向传播 …
  embeddingBackward(ws.ids, dx, m.tok.grad, m.pos.grad, B, T, E);
```

### 梯度确认

上面的验证单元用的是 NumPy 写的“参考版”。训练器本身是 JavaScript，同样的确认还要在 JavaScript 里再做一遍：对每一种零件，以及一个缩小的完整模型（2 层，全部参数），用中心差分逐个元素地求数值梯度，与手写的反向传播比较。用 float64 计算时，所有零件的相对误差都在 1e-8 以下。

如果你读过《深度学习入门：基于Python的理论与实现》5.7.3 节，会记得梯度确认的误差不会恰好为 0：数值微分本身有截断误差和舍入误差。我们用的判据是 $\|a - b\| / (\|a\| + \|b\|)$，它不会因为某几个梯度分量接近 0 而失真。

```localcheck verify/test_trainer.py::test_js_gradient_checks title="JavaScript 版每一层的梯度确认"
```

## T.5 与 PyTorch 对拍

梯度确认只能说明一件事：反向传播与我们自己的前向传播是一致的。可如果前向传播本身就与 CodeBot 不同呢？梯度确认照样会通过。

举个例子。[第 2 章](ch02.html#s2-9)的 `ch02/09_norm_gelu.py` 自己实现了 tanh 近似版的 GELU，而 `codebot/model.py` 的 FFN 实际用的是 `nn.GELU()`，也就是精确版：

```listing codebot/model.py#L73-L84
```

两者最多相差 4.7e-4。假如我们图省事用了 tanh 版，每一层的梯度确认依然全部通过，但模型已经不是 CodeBot 的那个函数了。

```verify id=T.5-a title="tanh 近似版 GELU 与精确版最多相差 4.7e-4"
gelu_tanh = lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
xs = np.linspace(-6, 6, 4001)
diff = np.abs(gelu_tanh(xs) - gelu(xs))
print(f"tanh 近似与精确 GELU 的最大差 {diff.max():.2e}，出现在 x = {xs[diff.argmax()]:.3f}")
assert 4.6e-4 < diff.max() < 4.8e-4
```

所以还需要第二道检查：与原书的模型本身对拍。对拍用一个很小的 DLFS6 GPT：词表 23、embedding 16 维，上下文 16，其余结构与迷你版相同（4 个头，2 层，FFN 64）。词表取 23，是因为这个数与其他尺寸都不同：哪个矩阵的两个维度一旦弄反，数字就对不上。`verify/_trainer_export.py` 用 float64 跑这个模型，关掉 dropout，把随机权重、一批 token、loss 和全部梯度存成文件；接着用 `torch.optim.AdamW`（学习率 0.05）走 3 步，把每一步的 loss 和最后的参数也存下来。JavaScript 按 `state_dict` 的名字读入同样的权重，再算一遍。

| 对拍的内容 | 容差 |
|---|---|
| 一批数据上的 loss | 绝对误差 < 1e-12 |
| 全部 29 个参数张量的梯度 | 相对误差 < 1e-12 |
| 3 步 AdamW 之后的 loss 和参数 | < 1e-11 |
| 同样的权重，换成 float32 计算（浏览器训练时用的精度） | 相对误差 < 1e-5 |
| 实际尺寸（V = 96，93,408 个参数）的 loss 与梯度校验和 | < 1e-10 |

全部通过。下面第一个测试确认这些文件确实出自 DLFS6 的 GPT：重新导出一遍，JavaScript 一侧读到的每一个数（loss、梯度、3 步 AdamW 的 loss 和参数、实际尺寸的梯度校验和）都与文件里的相同。第二个测试运行 JavaScript 一侧的对拍：

```localcheck verify/test_trainer.py::test_parity_fixtures_come_from_dlfs6_gpt title="对拍文件可以从 DLFS6 的 GPT 重新生成"
```

```localcheck verify/test_trainer.py::test_js_matches_dlfs6_gpt title="JavaScript 训练器与 DLFS6 的 GPT 对拍"
```

::: cs336 A1 的测试也是“与 PyTorch 对拍”
[CS336 作业 1](https://github.com/stanford-cs336/assignment1-basics) 规定，除了 `nn.Parameter`、容器类和 `optim.Optimizer` 基类，不许使用 `torch.nn`、`torch.nn.functional` 和 `torch.optim` 里现成的定义。它的测试是这样对拍的：

| 对拍的内容 | 容差 |
|---|---|
| 交叉熵，对照 `F.cross_entropy` | atol 1e-4 |
| 手写的 AdamW，对照 `torch.optim.AdamW`（在 `Linear(3, 2)` 上跑 1,000 步） | atol 1e-4 |
| SwiGLU、注意力、多头注意力、RoPE，对照固定权重下的快照 | atol 1e-5 |
| RMSNorm、Transformer 块，同上 | atol 1e-4 |
| 整个语言模型，同上 | atol 1e-4，另加 rtol 1e-2 |
| Linear、embedding，同上 | 默认的 rtol 1e-4、atol 1e-2 |

思路与本节相同。不同的是，CS336 的学生仍然可以用 autograd 求梯度；我们连反向传播也是手写的，所以多了 T.4 的梯度确认这一道。（测试代码见 A1 仓库 2026-04-07 的提交 [a158843 的 `tests/` 目录](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712/tests)，查证于 2026-10。）
:::

::: note WebAssembly SIMD
浏览器支持 WebAssembly 的 SIMD 指令时，训练器会把 `matmulBT` 换成一个 670 字节的 WebAssembly 版本：一条 `f32x4.mul` 同时做 4 个 float32 乘法，下一条 `f32x4.add` 再做 4 个加法（WebAssembly 的 128 位 SIMD 没有融合乘加）。它与 JavaScript 版算的是同一个乘积，只是累加用 float32 而不是 float64，两者相差约 1e-7（相对）。上表里 float32 那一行的对拍，大部分乘法走的正是这条路径；形状不满足条件的（行数要是 4 的倍数，列数要是偶数，内积的长度要是 4 的倍数），例如 V = 23 的 unembed，仍然回退到 JavaScript。在录制用的机器上，它让一步训练的速度变成原来的 3 倍多：

```verify id=T.5-b title="WebAssembly SIMD 对训练速度的影响（录制时测得）"
ms = rec["timing"]["stepMs"]
speedup = ms["js"] / ms["simd"]
print(f"一步训练：SIMD {ms['simd']} 毫秒，纯 JavaScript {ms['js']} 毫秒，速度是 {speedup:.1f} 倍")
print(f"所以纯 JavaScript 版里，矩阵乘法至少占了每一步 1 - 1/{speedup:.1f} = {1 - 1 / speedup:.0%} 的时间")
assert speedup > 3 and 1 - 1 / speedup > 2 / 3
```

SIMD 只换掉了矩阵乘法，整步的速度却变成了 3 倍多。由此可以反推：在纯 JavaScript 版里，矩阵乘法至少占了每一步三分之二的时间。这是 Amdahl 定律：如果矩阵乘法占了时间的比例 $f$，那么即使它快到不花时间，整步最多也只能快 $1/(1-f)$ 倍。T.6 的训练器上有一个开关，可以在你自己的浏览器里关掉它比较。

```localcheck verify/test_trainer.py::test_simd_kernel title="SIMD 内核：670 字节就是注释里的源码，结果与 JavaScript 版一致"
```
:::

## T.6 训练：取窗口、AdamW 与 Web Worker

模型和梯度都准备好了，剩下的是训练循环。先回顾 `ch03/01_pretrain.py` 的核心：

```listing ch03/01_pretrain.py#L71-L80
```

迷你 GPT 的一步训练也是这四件事：梯度清零，前向传播算 loss，反向传播，AdamW 更新参数。

```js
export function trainStep(m, ws, opt, x, y, B, T) {
  zeroGrad(m);
  const loss = forward(m, ws, x, y, B, T);
  backward(m, ws);
  opt.step();
  return loss;
}
```

顺序与原书略有不同：原书先算 loss，再把梯度清零（第 75–78 行）。前向传播不碰梯度，梯度只在反向传播时才写入，所以两种顺序是等价的。

### 取窗口

训练数据的组织方式也与 CodeBot 相同。`TokenDataset` 从 token 序列里切出长度为 `context_len` 的窗口，目标 `y` 是输入 `x` 错开一位；`DataLoader(shuffle=True)` 把所有窗口打乱后依次取用。

```listing ch03/01_pretrain.py#L33-L50
```

训练器的 `makeBatcher` 做同样的事：每个 epoch 把全部窗口的起点打乱一次，依次取 8 个组成一个 batch，取完再重新打乱。这一点与原书略有不同：原书用 `itertools.cycle` 包住 DataLoader，第二轮起会按原来的顺序重放第一轮的 batch（见 [3.3.5 节](ch03.html#s3-3-5)）。不过两边都远没有走完一轮，所以不影响这里的结果。

我们的训练文本有 211,113 个 token，可以切出 211,049 个窗口。2,000 步用掉 16,000 个窗口，按 DataLoader 的算法还不到一个 epoch 的 8%，同一个窗口不会出现第二次。但窗口是重叠的。按 token 算，2,000 步一共有 1,024,000 个预测目标，是训练文本长度的 4.85 倍：每个字符平均已经当过近 5 次预测目标（第 3 章的 CodeBot 是 61 次，见 [3.1.3 节](ch03.html#s3-1-3)）。“没见过的窗口”不等于“没见过的文字”。

```verify id=T.6-a title="窗口数与 2,000 步用掉的比例：按窗口不到 8%，按 token 是 4.85 遍"
n_windows = len(t) - 64                          # 起点 0 .. N-C-1，与 TokenDataset.__len__ 相同
used = 2000 * 8
print(f"窗口数 {n_windows:,}，2,000 步用掉 {used:,} 个，占 {used / n_windows:.1%}")
print(f"预测目标 {used * 64:,} 个，是训练文本 {len(t):,} 个 token 的 {used * 64 / len(t):.2f} 倍")
assert n_windows == 211_049 and used / n_windows < 0.08
assert abs(used * 64 / len(t) - 4.85) < 0.01
```

### AdamW

优化器是 `torch.optim.AdamW`，超参数全部用默认值：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$，weight decay $\lambda = 0.01$，并且作用于所有参数（包括 LayerNorm 和 bias），与 `01_pretrain.py` 第 62 行一致。

```listing ch03/01_pretrain.py#L62-L62
```

一步 AdamW 是这样的：

$$\begin{aligned} m &\leftarrow \beta_1 m + (1-\beta_1)\,g \\ v &\leftarrow \beta_2 v + (1-\beta_2)\,g^2 \\ w &\leftarrow w\,(1-\eta\lambda) - \eta\,\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\epsilon} \end{aligned}$$

这里，$g$ 是梯度，$m$ 和 $v$ 是梯度及其平方的滑动平均，$t$ 是步数，$\eta$ 是学习率。式中的 $1-\beta_1^t$ 和 $1-\beta_2^t$ 是偏差校正（推导见[附录 D](appendix-d.html)）。第一步时，校正后的 $m$ 就是 $g$，$v$ 就是 $g^2$，于是更新量变成 $\eta\,g/|g| = \pm\eta$：**无论梯度是 0.02 还是 2，每个参数第一步都走 $\eta$ 那么远**。以 $w = 0.5$、$g = 0.02$、$\eta = 0.003$ 为例：$0.5 \times (1 - 3\times10^{-5}) - 0.003 = 0.496985$。

```verify id=T.6-b title="AdamW 的第一步：每个参数都移动约 lr"
def adamw_step(w, g, m, v, t, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    w = w * (1 - lr * wd)                        # 解耦的权重衰减（torch.optim.AdamW 先做这一步）
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps), m, v

w0 = np.array([0.5, 0.5, 0.5])
gs = np.array([0.02, 2.0, -1e-4])                # 梯度相差 2 万倍
w1, _, _ = adamw_step(w0, gs, np.zeros(3), np.zeros(3), t=1)
print("第一步的移动量:", np.array2string(w1 - w0, precision=6))
assert abs(w1[0] - 0.4969850015) < 1e-9        # 0.5*(1-3e-5) - 3e-3*0.02/(0.02+1e-8)
assert np.allclose(np.abs(w1 - w0 * (1 - 3e-5)), 3e-3, rtol=1e-3)   # 无论梯度多大，都走约 lr 那么远
```

训练器里的 AdamW 逐元素地做同样的计算（T.5 已经确认它与 `torch.optim.AdamW` 一致到 1e-11）。与参数无关的三个数先算好：`stepSize` 是 $\eta/(1-\beta_1^t)$，`bc2Sqrt` 是 $\sqrt{1-\beta_2^t}$，`decay` 是 $1-\eta\lambda$。循环里的最后一行，正是上面的第三个式子：

```js
    const stepSize = lr / (1 - b1 ** this.t);
    const bc2Sqrt = Math.sqrt(1 - b2 ** this.t);
    const decay = 1 - lr * this.weightDecay;
// …
      for (let i = 0; i < w.length; i++) {
        w[i] *= decay; // decoupled weight decay
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        w[i] -= (stepSize * m[i]) / (Math.sqrt(v[i]) / bc2Sqrt + eps);
      }
```

### 这些数字从哪来

训练器的超参数有三种来历：照搬原书的，我们量出来的，还有为了让演示在浏览器里跑得动而选的。先看前两种：

| 超参数 | 取值 | 出处 |
|---|---|---|
| 学习率 | 3e-3 | 量出来的：在 3e-4、1e-3、3e-3、1e-2 上各训练 1,000 步，每种换 3 个随机种子。3e-4 太慢，1e-2 太激进；1e-3 与 3e-3 的差别比换一个种子还小，我们取了平均更低的 3e-3（见本节末尾的“你可能会困惑”） |
| AdamW 的 β、ε、weight decay | 0.9 / 0.999、1e-8、0.01 | `torch.optim.AdamW` 的默认值，与 `01_pretrain.py` 相同 |
| 初始化 | 正态分布 N(0, 0.02²) | `codebot/model.py` 的 `_init_weights` |
| 留出 loss | 每 25 步测一次，用整份留出文本 | 按 64 个字符一段切开，每个字符预测一次，共 25,150 个位置。测一次的时间约等于 20 步训练，测得太勤会拖慢训练 |
| 样例 | 每 100 步一次，温度 0.8，从 `def ` 开始写 120 个字符 | 遇到文档结束 token 不停下，以标记显示后接着写（原书的 `generate` 遇到它就停） |
| 随机种子 | 0 | 每次“重置”之后都能从头复现 |

模型的尺寸和 batch 属于第三种，没有做扫描。我们的标准是：在录制用的机器上，一步训练只要几十毫秒，而且 1,000 步之内能越过 bigram 线。把其中任何一项改大，参数量和每步的时间会这样变：

| 改动（其余照旧） | 参数量 | 每步的时间 |
|---|---|---|
| 现在的设置 | 93,408 | 1 |
| `ff_dim` 192 → 256（4 倍，与原书相同） | 109,920 | 1.2 |
| `embed_dim` 64 → 128（`ff_dim` 随之变成 384） | 350,560 | 2.6 |
| `n_layer` 2 → 4 | 176,352 | 2.0 |
| `n_head` 4 → 8（每个头 8 维） | 93,408 | 1.1 |
| `context_len` 64 → 128 | 97,504 | 2.3 |
| batch 8 → 16 | 93,408 | 2.0 |

参数量是算出来的；每步的时间是在录制用的机器上实测的，以现在的设置为 1。头数加倍、FFN 加到 4 倍，每步只慢一两成；embedding 维度、层数、上下文或 batch 翻一倍，每步的时间就要翻一倍左右，甚至更多。

```verify id=T.6-c title="放大模型或 batch 的代价：参数量与每步的时间"
var = {x["id"]: x for x in rec["timing"]["variants"]}
base_ms = var["base"]["stepMs"]
for x in rec["timing"]["variants"]:
    print(f"{x['change']}：参数 {x['params']:,}，每步的时间 {x['stepMs'] / base_ms:.1f}")
print(f"测一次留出 loss 的时间 = {rec['timing']['heldOutMs'] / rec['timing']['stepMs']['simd']:.0f} 步训练")
expect = {"base": (64, 64, 2, 192), "ff256": (64, 64, 2, 256), "embed128": (64, 128, 2, 384),
          "layers4": (64, 64, 4, 192), "heads8": (64, 64, 2, 192), "ctx128": (128, 64, 2, 192), "batch16": (64, 64, 2, 192)}
assert all(var[k]["params"] == gpt_params(96, C, E, L, F) for k, (C, E, L, F) in expect.items())
ratio = {k: round(x["stepMs"] / base_ms, 1) for k, x in var.items()}
assert ratio == {"base": 1.0, "ff256": 1.2, "embed128": 2.6, "layers4": 2.0, "heads8": 1.1, "ctx128": 2.3, "batch16": 2.0}
assert max(ratio["ff256"], ratio["heads8"]) <= 1.2 and min(ratio[k] for k in ("embed128", "layers4", "ctx128", "batch16")) >= 1.95
assert base_ms < 100 and first_below(base["bigram"]) < 1000          # 选尺寸的两条标准
assert 15 < rec["timing"]["heldOutMs"] / rec["timing"]["stepMs"]["simd"] < 30
```

### 为什么要放进 Web Worker

浏览器的页面只有一个主线程，它既要跑 JavaScript，又要响应点击、重绘页面。如果训练循环直接在主线程里跑，在它跑完之前，页面就会卡住不动。**Web Worker** 是浏览器提供的后台线程：训练放在 Worker 里（`web/widgets/trainer-worker.js`），每训练约 0.1 秒就把这段时间的 loss 发回页面一次，页面只负责画图。暂停、换学习率这些操作，也是作为消息发给 Worker 的。

### 亲手训练

现在轮到你了。点“开始训练”，看 loss 往下走；每 100 步，下方会多出一段模型写的文字。训练中可以随时暂停、换学习率，或者切换到“过拟合一个 batch”模式（T.7 会用到）。

```widget trainer-live caption="浏览器里的迷你 GPT。曲线是实时的：训练 loss 每 10 步取平均，留出 loss 每 25 步测一次；三条虚线依次是瞎猜、unigram 和 bigram 的水平。"
{"batch": 8, "lr": 0.003, "maxSteps": 1000, "evalEvery": 25, "sampleEvery": 100, "sampleChars": 120, "prompt": "def ", "temperature": 0.8, "seed": 0}
```

你应该看到这样几个阶段（数字来自 T.1 录制的同一份设置）：

1. **第 0 步**：留出 loss 约 4.55，样例是各种符号的乱码。
2. **前 25 步**：loss 掉到 unigram 线以下。模型最先学会的是字符的频率（空格最多，其次是 e 和 t）。录制中第 100 步的样例四成多是空格，比语料里的 16.9% 还多：温度 0.8 会放大最常见的字符（只按频率、以温度 0.8 抽样，空格约占四分之一），而一段 120 个字符的样例本身起伏也很大。
3. **第 475 步**：越过 bigram 线（第 450 步还略高），此后它比只看前一个字符的计数模型更准。到第 625 步，loss 降到 T.2 的那条下限 2.562 以下，这时可以断定，它用上了更远的上下文。其实它更早就开始用了：第 200 步时，只让它看前一个字符，留出 loss 是 3.05；让它看满 64 个字符，是 2.90，已经低了 0.15。
4. **第 1,000 步**：留出 loss 约 2.28。样例里能认出 `#` 注释、`=` 和 `for` 这样的零件，但还拼不成像样的代码。

```verify id=T.6-d title="录制的训练：频率先学会，更远的上下文在越过 bigram 线之前就已经用上"
q = p1 ** (1 / 0.8)                                   # 只按频率、以温度 0.8 抽样
q /= q.sum()
held1 = dict((s, x) for s, x in run["held1"])         # 每段只给 1 个字符时的留出 loss
first_floor = first_below(floor)
s100 = next(x["text"] for x in run["samples"] if x["step"] == 100).replace(EOT, "")
print(f"第 100 步的样例里，空格占 {s100.count(' ') / len(s100):.0%}；温度 0.8 下，只按频率抽到空格的概率 {q[sp]:.1%}（语料里 {p1[sp]:.1%}）")
print(f"第一次低于下限 {floor:.3f}：第 {first_floor} 步")
for s in (100, 200, 1000, 2000):
    print(f"第 {s:>5,} 步：只看前一个字符 {held1[s]:.3f}，看满 64 个字符 {held[s]:.3f}，差 {held1[s] - held[s]:.2f}")
assert 0.4 < s100.count(" ") / len(s100) < 0.5 and abs(q[sp] - 0.2445) < 1e-3
assert all(x < base["bigram"] for s, x in run["held"] if s >= 475) and first_floor == 625
assert abs(held1[200] - 3.0546) < 1e-3 and abs(held[200] - 2.9039) < 1e-3 and held[200] > base["bigram"] + 0.2
assert min(held1.values()) > base["bigram"] + 0.15     # 只看一个字符，它始终不如计数的 bigram
```

最后一行还说明了一件事：只看一个字符时，这个模型直到第 2,000 步也不如计数的 bigram（2.90 对 2.69）。它能越过 bigram 线，靠的是更远的上下文。

在 Chrome、Edge 这类以 V8 为引擎的浏览器里，不改设置、只按“开始训练”，曲线会与 T.1 的录制逐步相同。下面的测试在无头 Chromium 里照这样跑了前 500 步，每一步的 loss、每一次的留出 loss 和每一段样例都与录制相同。其他浏览器没有逐位核对过：数学函数的实现若有细微差别，数字会略有出入，但走势相同。关掉 WebAssembly SIMD 也一样：累加的精度变了，数字也会略有不同。

```localcheck verify/test_trainer.py::test_live_trainer_in_chromium_matches_recording title="无头 Chromium 里的训练器：前 500 步与录制相同"
```

::: confusion 学习率为什么不用 CodeBot 的 3e-4？
直觉上，原书的超参数应该最可靠，照搬总没错。可是 3e-4 是 CodeBot 用的值，它面对的是另一个问题：1,112 万个参数，训练 2 万步。迷你 GPT 只有它的 1/119，训练只有一两千步，最合适的学习率没有理由一样。下图是同一个模型、同一个随机种子，只换学习率，各训练 1,000 步的留出 loss：

```widget trainer-replay caption="3e-4 太保守：1,000 步时 2.60，第 675 步才越过 bigram 线。1e-2 太激进：很快在高处走平（2.76），一次也没有越过。1e-3 与 3e-3 在第 1,000 步几乎一样（2.28），1e-3 还更早越过 bigram 线（第 350 步）。"
{"view": "lr", "views": ["lr"]}
```

一个随机种子说明不了太多，所以我们又换了两个种子，把四种学习率各重跑了一遍：

```verify id=T.6-e title="四种学习率、三个随机种子：第 1,000 步的留出 loss"
lrs = {"lr3e-4": 3e-4, "lr1e-3": 1e-3, "default": 3e-3, "lr1e-2": 1e-2}
H = {rid: dict((s, x) for s, x in rec["runs"][rid]["held"]) for rid in lrs}
seeds = {rid: [H[rid][1000]] + [dict((s, x) for s, x in r["held"])[1000] for r in rec["seeds"] if r["lr"] == lr]
         for rid, lr in lrs.items()}
cross = {rid: next((s for s, x in rec["runs"][rid]["held"] if x < base["bigram"]), None) for rid in lrs}
print("学习率  第 25 步  第 1,000 步（种子 0 / 1 / 2）  平均   极差   越过 bigram 线")
for rid, lr in lrs.items():
    sv = seeds[rid]
    print(f"{lr:.0e}   {H[rid][25]:.3f}    {sv[0]:.3f} / {sv[1]:.3f} / {sv[2]:.3f}   {np.mean(sv):.3f}  {np.ptp(sv):.3f}  {cross[rid] or '—'}")
print("移动一个标准差（0.02）要几步:", ", ".join(f"{lr:.0e}: {0.02 / lr:.0f}" for lr in lrs.values()))
assert [round(seeds[rid][0], 4) for rid in lrs] == [2.6041, 2.28, 2.2783, 2.7633]
assert list(cross.values()) == [675, 350, 475, None]
best = seeds["lr1e-3"] + seeds["default"]
assert min(seeds["lr3e-4"]) > max(best) + 0.25 and min(seeds["lr1e-2"]) > max(best) + 0.4   # 换了种子照样落后
gap = np.mean(seeds["lr1e-3"]) - np.mean(seeds["default"])
assert 0.05 < gap < 0.07 and 0.13 < np.ptp(seeds["default"]) < 0.14                       # 平均差 0.06，换种子差 0.13
assert [round(0.02 / lr) for lr in lrs.values()] == [67, 20, 7, 2]
assert abs(H["lr3e-4"][25] - 3.961) < 1e-3 and abs(H["default"][25] - 3.331) < 1e-3
assert abs(H["lr1e-2"][500] - H["lr1e-2"][1000] - 0.12) < 0.005 and abs(H["default"][500] - H["default"][1000] - 0.37) < 0.005
```

3e-4 和 1e-2 换了种子照样落后，这个结论站得住。1e-3 和 3e-3 就分不出高下了：三个种子的平均，3e-3 低 0.06；可是 3e-3 自己换一个种子，第 1,000 步就能差出 0.13。本页取 3e-3，你也可以在训练器里选 1e-3 试试。

那么，3e-4 在这里为什么这么慢呢？T.6 的验证已经给出了一半答案：AdamW 每一步让每个参数移动的距离大约就是 lr 这个量级，与梯度的大小无关。权重初始化时的标准差是 0.02。直观上，lr = 3e-4 时，一个权重要沿同一个方向被连推约 67 步，才移动一个标准差；3e-3 只要 7 步，1e-2 只要 2 步。在只有 1,000 步的训练里，3e-4 开头就落后了：第 25 步时，它的留出 loss 还是 3.96，3e-3 已经到了 3.33。1e-2 正相反，每一步都推得太远，loss 很快就在高处走平：后 500 步只降了 0.12，3e-3 降了 0.37。严格地讲，“移动一个标准差要几步”只是一把粗略的尺子，算不出最合适的学习率；最合适的值只能量出来。

可以带走的结论是：**学习率是量出来的，不是抄来的**。换了模型的尺寸或训练的长度，就重新扫一遍，最好还多换几个随机种子。[第 6 章](ch06.html)还会在学习率上加 warmup 和衰减。
:::

## T.7 先让它过拟合一个 batch

开始一次很长的训练之前，有一个花不了几秒钟的检查：只拿一个 batch 反复训练。如果模型和训练代码都没有错，它应该能把这一个 batch 背下来，loss 一路降到接近 0。做不到，多半说明代码里有 bug，或者学习率错得离谱；这时候加数据、加步数都无济于事。

把训练器的模式切换到“过拟合一个 batch”再开始训练，就是这个检查。下图是录好的结果：

```widget trainer-replay caption="反复训练同一个 batch：训练 loss 在 100 步内降到 0.03，300 步时降到 0.003。从第 100 步起，给它第 1 段开头的 8 个字符，它就能一字不差地写出这一段剩下的 56 个字符；换成从 def 开始写，它写出的只是一串支离破碎的字符。留出 loss 在第 25 步降到 3.38 之后便一路上升，第 75 步已超过瞎猜的水平，300 步时升到 6.76。"
{"view": "overfit", "views": ["overfit"]}
```

```verify id=T.7-a title="录制的过拟合运行：背得一字不差，换了文本就不行"
of = rec["runs"]["overfit"]
mean10 = lambda a: sum(of["losses"][a:a + 10]) / 10      # 第 a..a+9 步的平均训练 loss
held_of = dict((s, x) for s, x in of["held"])
for a in (0, 40, 90, 290):
    print(f"第 {a:>3}–{a + 9:<3} 步  训练 loss {mean10(a):.4f}")
print(f"留出 loss：第 25 步 {held_of[25]:.2f}，第 75 步 {held_of[75]:.2f}，第 300 步 {held_of[300]:.2f}（ln 96 = {ln_v:.2f}）")
recited = [s["step"] for s in of["samples"] if of["prompt"] + s["text"] == of["batch"][0]]
print(f"从 {of['prompt']!r} 写起，与第 1 段原文一字不差的样例：第 {recited} 步")
assert mean10(90) < 0.035 and mean10(290) < 0.003 and abs(mean10(290) - 0.0027) < 5e-5
assert min(held_of, key=held_of.get) == 25 and next(s for s, x in of["held"] if x > ln_v) == 75
assert abs(held_of[300] - 6.76) < 0.01
assert recited == [100, 200, 300] and len(of["prompt"]) == 8 and len(of["batch"][0]) == 64
```

300 步之后，这 512 个位置的平均 loss 是 0.0027，换算成概率，模型给正确答案的概率（几何平均）是 $e^{-0.0027} \approx 99.7\%$：它把这 8 段代码背得滚瓜烂熟。从第 100 步起，只要给它第 1 段的开头 `value = `，它就能把这一段剩下的 56 个字符一字不差地写出来。可是离开这 8 段，它就不会写了；在没见过的文本上，它比瞎猜还差。

同样的检查在 PyTorch 里也成立：用 DLFS6 的 GPT 以同样的尺寸、同样的学习率反复训练一个 batch，150 步内训练 loss 就降到 0.05 以下。

```localcheck verify/test_trainer.py::test_dlfs6_gpt_overfits_one_batch title="DLFS6 的 GPT 同样能过拟合一个 batch"
```

::: think 过拟合之后，留出 loss 为什么会比瞎猜（4.56）还高？
:::
::: answer
瞎猜对每个字符都给 1/96 的概率，错也错得不多。背熟了一个 batch 的模型却非常“自信”：它把几乎全部概率押在“在那个 batch 里出现过的下一个字符”上。到了新文本，这种自信的猜测经常押错，而交叉熵对“自信地猜错”的惩罚是 $-\ln p$：给正确答案的概率只剩 0.1% 时，这一个位置的 loss 就是 $-\ln 0.001 \approx 6.9$。
:::

::: confusion loss 不降：该怪数据、学习率，还是步数不够？
这三种怀疑都很自然，也都可能对。麻烦在于，它们表现出来是同一个样子：一条不怎么往下走的曲线。要把它们分开，可以分两步。

第一步，过拟合一个 batch。连一个 batch 都背不下来，问题就出在模型或训练代码上（或者学习率错了几个数量级），与数据量、步数无关，先查代码。

第二步，一个 batch 能背下来，正常训练却不见起色，代码大概率没问题，接下来看曲线的形状：

- **开头就掉得慢**：学习率偏小。[图 T-3](#fig-T-3) 里，3e-4 在第 25 步还是 3.96，3e-3 已经到了 3.33。
- **开头掉得快，随后早早在较高的位置走平**：学习率偏大。1e-2 在后 500 步只降了 0.12，3e-3 降了 0.37。
- **一直在降，到最后还没走平**：步数不够，接着训练就是。T.1 的那次训练，最后 500 步还降了 0.14。
- **训练 loss 在降，留出 loss 却在升**：数据太少，模型在背书（[图 T-4](#fig-T-4)）。

```verify id=T.7-b title="四种曲线形状：读取录制的运行"
drop_3e3, drop_1e2 = (H[rid][500] - H[rid][1000] for rid in ("default", "lr1e-2"))
tail = held[1500] - held[2000]                            # T.1 那次训练的最后 500 步
apart = [(s, round(mean10(s - 10), 3), x) for s, x in of["held"] if s in (100, 200, 300)]
print(f"第 25 步：3e-4 {H['lr3e-4'][25]:.2f}，3e-3 {H['default'][25]:.2f}")
print(f"第 500 → 1,000 步：1e-2 降了 {drop_1e2:.2f}，3e-3 降了 {drop_3e3:.2f}；默认训练最后 500 步降了 {tail:.2f}")
print("过拟合：（步数，前 10 步的训练 loss，留出 loss）", apart)
assert H["lr3e-4"][25] - H["default"][25] > 0.6 and drop_1e2 < 0.13 < 0.36 < drop_3e3
assert abs(tail - 0.14) < 0.005
assert all(a[1] < 0.05 for a in apart) and [a[2] for a in apart] == sorted(a[2] for a in apart)
```

这个检查只要一两百步。在一次要跑几小时的训练开始之前，花这一两百步很值得。
:::

::: confusion 训练 loss 很低，就说明模型学会了吗？
不一定。过拟合的那条曲线就是反例：训练 loss 0.003，留出 loss 6.76。过拟合模式里，batch 永远是同一个，训练 loss 量的是模型对它记得多牢。正常训练时，每一步的训练 loss 是在参数更新之前、用新取来的窗口算的，模型还没学过这些窗口，所以它和留出 loss 靠得很近（2,000 步时 1.87 对 1.94）。不过，训练窗口都来自训练文档，与学过的窗口还有重叠；真正的泛化，只能用从不参加训练的留出文档来量。`ch03/01_pretrain.py` 没有留出集，只记录训练 loss；[第 6 章](ch06.html)的 StoryBot 才加上验证集。本页的训练器从一开始就把两条曲线画在一起。
:::

## T.8 从迷你版回到 CodeBot

迷你 GPT 与 CodeBot 用的是同一张图纸、同一个训练循环，不同的只是数字。把这些数字放大，就回到了第 3 章：

| | 迷你版（2,000 步） | CodeBot（`01_pretrain.py`） |
|---|---|---|
| 参数量 N | 93,408 | 11,121,640（约 119 倍） |
| 每步的 token 数 | 8 × 64 = 512 | 32 × 256 = 8,192（16 倍） |
| 训练的 token 总数 D | 1,024,000 | 163,840,000（160 倍） |
| 训练计算量 6ND | 5.7 × 10¹¹ FLOPs | 1.1 × 10¹⁶ FLOPs（约 19,000 倍） |

$6ND$ 是估算训练计算量的经验公式：每个参数、每个 token，前向传播约 2 次浮点运算，反向传播约 4 次。

```verify id=T.8-a title="训练计算量：6ND；同一份计算交给本页的训练器、CPU 上的 PyTorch 和 GPU" files=data/ch00/cpu_step.json,data/runs/codebot-3060ti/report.json
mini_tokens = 2000 * 8 * 64                      # 2,000 步 × 8 条 × 64 个字符
codebot_tokens = 20_000 * 32 * 256               # ch03/01_pretrain.py: max_iters × batch_size × context_len
mini_flops, codebot_flops = 6 * mini * mini_tokens, 6 * codebot * codebot_tokens
print(f"迷你版 : {mini_tokens:>13,} token，6ND = {mini_flops:.2e} FLOPs")
print(f"CodeBot: {codebot_tokens:>13,} token，6ND = {codebot_flops:.2e} FLOPs（{codebot_flops / mini_flops:,.0f} 倍）")
js = 6 * mini * 8 * 64 / (rec["timing"]["stepMs"]["simd"] / 1000)            # 本页的训练器：每秒的浮点运算次数
cpu = json.load(open("data/ch00/cpu_step.json", encoding="utf-8"))           # 第 0 章：PyTorch 在同一台机器的 CPU 上
torch_cpu = 6 * codebot * cpu["tokens_per_step"] / min(e["s_per_step"] for e in cpu["earlier"])
report = json.load(open("data/runs/codebot-3060ti/report.json", encoding="utf-8"))
gpu_s = next(s["seconds"] for s in report["steps"] if s["script"] == "ch03/01_pretrain.py")
for name, flops in [("本页的训练器（JavaScript，单线程）", js), ("PyTorch，同一台机器的 CPU", torch_cpu),
                    ("PyTorch，RTX 3060 Ti（实测）", codebot_flops / gpu_s)]:
    print(f"{name}：每秒 {flops / 1e9:,.1f} GFLOP，CodeBot 的预训练要 {codebot_flops / flops / 3600:,.1f} 小时")
assert codebot_tokens == 163_840_000 and 1.09e16 < codebot_flops < 1.10e16
assert 8.4e9 < js < 8.7e9 and round(codebot_flops / js / 86400) == 15
assert round(codebot_flops / torch_cpu / 3600) == 23 and round(torch_cpu / js) == 15
assert round(gpu_s) == 2513 and 450 < codebot_flops / gpu_s / js < 550
assert round(6_487_033 / len(train)) == 29                                   # 小结：训练文本差 29 倍
```

计算量差了约 19,000 倍。这个标签页的训练器是单线程的 JavaScript：在录制用的机器上，它每秒约完成 86 亿次浮点运算；照这个速度，CodeBot 的预训练要 15 天左右。这量的是我们手写的代码，不是 CPU 的上限：在同一台机器上，PyTorch 用 8 个线程跑 CodeBot，最快的一次每步 4.1 秒，2 万步约 23 小时（[第 0 章的时间账](ch00.html#v-0.7-b)），算力约是本页训练器的 15 倍。GPU 又快得多：同一个 `01_pretrain.py` 在 RTX 3060 Ti 上原样运行，实测 2,513 秒（约 42 分钟，运行记录 codebot-3060ti），约是本页训练器的 500 倍。这正是 CodeBot 要交给 GPU 的原因。

除了尺寸，放大时还有几处要换掉：分词器从字符级换成第 1 章的 BPE，序列长度缩短到字符级的约 1/2.36，同样 256 个 token 能装下更多代码；dropout 恢复为 0.1；矩阵乘法交给 GPU 上的 cuBLAS，不再需要自己写 SIMD。

::: cs336 先在 CPU 上调通，再上 GPU
[CS336](https://cs336.stanford.edu/) 给学生的建议是先在 CPU 上调试，再用 GPU 跑作业 1、4、5 的训练和作业 2 的基准测试。作业 1 的 TinyStories 语言模型也有一个“低资源”版本：训练 4,000 万个 token，目标是验证集 loss 不高于 2.00；参考实现在 M4 Max 的 CPU 上跑了 1 小时 22 分钟（用 MPS 是 36 分钟），达到 1.80（[A1 讲义 v26.0.3](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712)；均查证于 2026-10）。本页的过拟合检查和对拍，正是这种“先在小尺寸上把代码调对”的做法。
:::

## 小结

这一页，我们在浏览器里把第 3 章的预训练完整地走了一遍，只是计算量缩小到原来的约 1/19,000。数据是 TinyCodes 的一小片，分词器是字符级的；模型照搬 `codebot/model.py`，参数从 1,112 万缩到 9 万；前向传播、反向传播和 AdamW 都用 JavaScript 手写，先逐层做梯度确认，再与 DLFS6 的 GPT 对拍。训练时，loss 从 ln V 附近出发，先越过 unigram 线，再越过 bigram 线，生成的文字也从符号乱码慢慢变得像代码。

本页所学的内容：

- 字符级分词器的词表是 95 种字符加 `<|endoftext|>`，共 96 个；loss 的三个参照点是瞎猜 ln 96 = 4.564、unigram 3.387、bigram 2.693（[验证 T.2-b](#v-T.2-b)）。只看前一个字符的模型，在留出文本上不可能低于 2.562；低于它，才能断定模型用上了更远的上下文（[验证 T.2-c](#v-T.2-c)）。
- 迷你 GPT 与 CodeBot 是同一个类、不同的尺寸，参数量 93,408 对 11,121,640（[验证 T.3-a](#v-T.3-a)）；随机初始化时 loss 约为 ln V + σ²/2（[验证 T.3-b](#v-T.3-b)），在代码文本上还会因权重共享略低于 ln V（[验证 T.3-c](#v-T.3-c)）。
- 交叉熵对 logits 的梯度是 (softmax − one-hot)/N（[验证 T.4-a](#v-T.4-a)）；线性层 dx = dy·W、dW = dyᵀ·x（[验证 T.4-b](#v-T.4-b)）。
- Attention 的反向传播多出的只有 softmax 的反向 dS = P ⊙ (G − rowsum(P ⊙ G))，被遮住的位置梯度恰好为 0（[验证 T.4-e](#v-T.4-e)）；权重共享时，两条路径的梯度相加（[验证 T.4-f](#v-T.4-f)）。
- 梯度确认只证明反向与前向一致；前向是不是 CodeBot 的那个函数，要靠与 PyTorch 对拍（[验证 T.5-a](#v-T.5-a)）。
- AdamW 的第一步，每个参数都移动约 lr，与梯度的大小无关（[验证 T.6-b](#v-T.6-b)）；学习率要量出来，不能照抄，还要多换几个随机种子：3e-4 太慢，1e-2 太激进，1e-3 与 3e-3 的差别比换一个种子还小（[验证 T.6-e](#v-T.6-e)）。
- 能把一个 batch 背得一字不差，说明代码没有大错；训练 loss 低不等于学会了，要看留出 loss（[验证 T.7-a](#v-T.7-a)）。

迷你 GPT 已经能写出一些像代码的零件，离 CodeBot 还差得远：差在 19,000 倍的计算量，也差在分词器和 29 倍的训练文本（[验证 T.8-a](#v-T.8-a)；训练好的 CodeBot 能写出什么，可以在 [CodeBot 工作台](playground.html) 里试）。下一页的 [GPU 实战手册](gpu.html)，就把 CodeBot 的训练真正搬上 GPU。
