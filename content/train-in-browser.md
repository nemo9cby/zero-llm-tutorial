---
chapter: T
slug: train-in-browser
order: 165
kind: extra
title: 在浏览器里从零训练一个迷你 GPT
nav_title: 浏览器里训练迷你 GPT
lede: 不安装任何东西，就在这个标签页里训练一个 9 万参数的字符级 GPT。前向、反向传播和 AdamW 全部手写，每一个梯度都与 DLFS6 的 GPT 对拍过；一分钟左右，就能看到 loss 从 ln V 降到 bigram 线以下。
pyodide_packages: [numpy]
---

[第 3 章](ch03.html)用 `ch03/01_pretrain.py` 训练 CodeBot：1,112 万个参数，每一步 32 条长 256 个 token 的序列，一共 2 万步。这样的训练要交给 GPU，而跑起来以后，我们能看到的只有终端里不停刷新的 loss。参数是怎样一步步被推到“会写代码”的位置上的？隔着一行 `loss.backward()`，这些都看不见。

这一页把同一件事缩小到浏览器的一个标签页里。模型与 CodeBot 用的是同一张图纸，只是小得多：字符级分词，2 层，9 万多个参数，训练数据是 CodeBot 的语料 TinyCodes 的一小片。不同的是，这里不借助任何框架：前向传播、反向传播和 AdamW 都用 JavaScript 手写，在后台线程里运行。为了确认手写的代码没有算错，每一层的梯度都做了数值确认，整个模型的 loss 和梯度也都与 DLFS6 的 `codebot.model.GPT` 逐个对拍过。

我们按这样的顺序走：先看一次录好的训练（T.1）；然后准备数据和分词器（T.2），搭模型（T.3），逐层手写反向传播（T.4），与 PyTorch 对拍（T.5）；接着亲手训练（T.6），并练习一个调试的好习惯，先让模型过拟合一个 batch（T.7）；最后把这个玩具放大回 CodeBot（T.8）。

## T.1 先来看结果

话不多说，先看一次已经录好的训练。下图是这个迷你 GPT 在 TinyCodes 上训练 2,000 步的过程，横轴是参数更新的次数，纵轴是 loss。拖动图下方的滑块，可以看到它在不同阶段从 `def ` 开始续写的文字。

```widget trainer-replay caption="训练 2,000 步，留出 loss 从 4.54 降到 1.95，在第 475 步越过 bigram 线。这次录制的样例里，第 0 步是符号乱码；第 1,000 步出现了 # 注释、= 和 for；到第 2,000 步，已经能写出 self.row_string(string, …) 和 import 这样的片段。"
{"views": ["default"]}
```

图中有两条曲线。**训练 loss** 是模型在它正在学习的那 8 段文字上的成绩，每 10 步取一次平均；**留出 loss**（held-out loss）是它在 86 篇从不参加训练的文档上的成绩。三条虚线是三个参照点，从上到下依次是瞎猜（ln 96）、只看字符频率（unigram）、只看前一个字符（bigram）。它们为什么是这几个数，T.2 会亲手算出来。

先记住这条曲线的样子：开头几十步掉得飞快，之后越来越慢。下面的验证单元直接读取录下来的运行记录，把几个关键时刻列出来：

```verify id=T.1-a title="录制的训练：几个关键时刻" files=data/train-in-browser/runs.json
import json

rec = json.load(open("data/train-in-browser/runs.json", encoding="utf-8"))
run, base = rec["runs"]["default"], rec["baselines"]
held = dict((s, x) for s, x in run["held"])            # 每 25 步测一次的留出 loss
first_below = lambda y: next(s for s, x in run["held"] if x < y)
for s in (0, 100, 500, 1000, 2000):
    print(f"第 {s:>5,} 步  留出 loss {held[s]:.3f}")
print(f"第一次低于 unigram 线（{base['unigram']:.3f}）：第 {first_below(base['unigram'])} 步")
print(f"第一次低于 bigram 线（{base['bigram']:.3f}）：第 {first_below(base['bigram'])} 步")
secs = run["seconds"]
print(f"录制用时 {secs:.0f} 秒，平均每秒 {2000 / secs:.1f} 步；第 475 步约在第 {475 / 2000 * secs:.0f} 秒")
assert abs(held[0] - 4.5406) < 1e-3 and abs(held[1000] - 2.2784) < 1e-3 and abs(held[2000] - 1.9511) < 1e-3
assert first_below(base["unigram"]) == 25 and first_below(base["bigram"]) == 475
assert round(secs) == 205 and round(475 / 2000 * secs) == 49
```

在录制用的机器上，2,000 步一共用了 205 秒（包括每 25 步一次的留出评估和每 100 步一次的采样），平均每秒近 10 步。按这个速度算，越过 bigram 线的第 475 步出现在第 49 秒前后。录制时这台机器还在忙别的任务，你的电脑多半更快；T.6 的训练器会实时显示每秒的步数。

这条曲线可以原样复现。录制脚本 `record_runs.mjs`（在 `tests/js/fixtures/trainer/` 目录下）用的就是本页的同一份代码，随机种子固定为 0。下面的测试会把前 20 步重新跑一遍，与录下的 loss 逐步对照，并检查本页引用的每一个里程碑：

```localcheck verify/test_trainer.py::test_recorded_runs_replay title="录制的运行可以逐步复现"
```

## T.2 数据与分词器

### 从 TinyCodes 里切一小片

CodeBot 的预训练数据是 `codebot/tiny_codes.txt`：一份 6,487,033 字节的 Python 代码语料，21,836 篇短文档之间用 `<|endoftext|>` 隔开。整份搬进浏览器太大，我们只取一小片：每 25 篇取 1 篇，跳过含有制表符等非打印字符的文档；取出来的文档中，每 10 篇留 1 篇不参加训练，专门用来测留出 loss。

这样得到 778 篇训练文档（220,437 个字符）和 86 篇留出文档（26,171 个字符），两个文件加起来不到 250 KB。切片的规则写在 `verify/_trainer_data.py` 里，下面的测试会从原书语料重新切一遍，确认与网页上的文件逐字节相同。

```localcheck verify/test_trainer.py::test_data_slice_is_reproducible title="网页上的数据就是原书语料的切片"
```

### 字符级分词器

分词器用最朴素的一种：一个字符就是一个 token。这正是 [1.1 节](ch01.html#s1-1)的字符级分词器，只改了一处：1.1 节直接拿 Unicode 码位当 id，词表有 111 万多行；这里只给训练文本里真正出现过的字符编号，按码位排好，从 0 开始。CodeBot 的分词器把 `<|endoftext|>` 放在词表的最后（id 999），我们也照做。

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

再用 NumPy 写一个同样的分词器，在网页上的数据文件上数一数：

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
```

TinyCodes 是纯 ASCII 文本，出现过的字符有 95 种：换行符，加上 94 个可打印字符（只有 `~` 一次也没出现）。再加上 `<|endoftext|>`，词表大小 V = 96。

字符级的代价，[1.2 节](ch01.html#s1-2)已经见过：序列变长。同一段留出文本，CodeBot 的 BPE 分词器切成 10,659 个 token，字符级切成 25,151 个，长了 2.36 倍。上下文同样是 64 个 token，字符级模型一次只能看到 64 个字符。在浏览器里，我们接受这个代价，换来一张只有 96 行的嵌入表。

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

这里，$p(x_i \mid x_{i-1})$ 是训练文本里字符 $x_{i-1}$ 后面紧跟着 $x_i$ 的比例。两个计数模型都在计数上加了 0.1。这叫平滑：留出文本里可能出现训练时没见过的组合，不加平滑，这些组合的概率就是 0，loss 会变成无穷大。

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
assert abs(ln_v - 4.5643) < 1e-4 and abs(h1 - 3.3867) < 1e-4 and abs(h2 - 2.6931) < 1e-4
assert abs(ln_v / math.log(2) - math.log2(96)) < 1e-12   # 1 nat = 1/ln 2 = 1.4427 bits
```

这样我们就有了一把尺子：瞎猜 4.564，unigram 3.387，bigram 2.693。loss 降到 unigram 线以下，说明模型至少学会了字符的频率；降到 bigram 线以下，说明它用上了比“前一个字符”更远的上下文，而把远处的字符搬过来，正是 Attention 的工作。

::: note nats 与 bits
本页的 loss 都以 e 为底，单位是 **nats**；以 2 为底时，单位是 bits，1 nat = 1/ln 2 ≈ 1.4427 bits。所以 ln 96 nats 正好等于 log₂ 96 ≈ 6.585 bits，也就是从 96 个符号里均匀地挑一个所需的信息量。PyTorch 的 `F.cross_entropy` 用自然对数，CodeBot 训练日志里的 loss 也是 nats（T.5 的对拍会确认两边的 loss 一致到 1e-12）。
:::

## T.3 模型：同一张图纸

迷你 GPT 的结构与 CodeBot 完全相同。下面是 `codebot/model.py` 的 GPT 类：token 嵌入加上可学习的位置嵌入，`n_layer` 个 Transformer 块，一个 LayerNorm，最后由 `unembed` 映射回词表。注意倒数第二行的 `self.embed.weight = self.unembed.weight`：`unembed` 与 `embed` 共享同一个权重矩阵（**权重共享**，weight tying），但 `unembed` 保留了自己的 bias。

```listing codebot/model.py#L102-L126
```

不同的只是尺寸。下表把两者并排：

| | CodeBot（`ch03/01_pretrain.py`） | 迷你版 | 为什么这样改 |
|---|---|---|---|
| 词表 `vocab_size` | 1,000（BPE） | 96（字符） | 字符级分词，见 T.2 |
| 上下文 `context_len` | 256 | 64 | Attention 的打分矩阵是 64 × 64，计算量随长度的平方增长 |
| 嵌入维度 `embed_dim` | 384 | 64 | |
| 头数 `n_head` | 6 | 4 | 每个头 16 维 |
| 层数 `n_layer` | 6 | 2 | |
| FFN 宽度 `ff_dim` | 1,536（4 倍） | 192（3 倍） | 把参数量控制在 10 万以内 |
| `dropout_rate` | 0.1 | 0 | 训练只有几千步，见下文 |
| 参数量 | 11,121,640 | 93,408 | 约为 1/119 |

`embed_dim` 是残差流的宽度。同一个量，CS336 叫 `d_model`，Hugging Face 的模型配置一般叫 `hidden_size`，GPT-2 的配置里叫 `n_embd`。dropout 设为 0 有两个原因：一是训练很短，2,000 步时训练 loss（1.87）和留出 loss（1.95）还很接近，用不着正则化；二是 dropout 会让每次前向传播都带随机性，梯度确认和对拍都会变得麻烦。

参数量可以逐项加出来。以迷你版为例：嵌入表 96 × 64 = 6,144，位置嵌入 64 × 64 = 4,096；每个块有两个 LayerNorm（各 64 + 64）、Q/K/V/O 四个 64 × 64 的矩阵（没有 bias），以及 FFN 的 64 × 192 + 192 和 192 × 64 + 64，合计 41,472；两个块共 82,944；最后的 LayerNorm 128，`unembed` 的 bias 96。总数 6,144 + 4,096 + 82,944 + 128 + 96 = 93,408。权重共享的矩阵只算一次。

```verify id=T.3-a title="参数量：迷你版 93,408，CodeBot 11,121,640"
def gpt_params(V, C, E, L, F):
    block = 2 * E + 4 * E * E + 2 * E + (E * F + F) + (F * E + E)  # norm1 + Q/K/V/O（无 bias）+ norm2 + FFN
    return V * E + C * E + L * block + 2 * E + V                   # embed（与 unembed 共享）+ 位置 + 各块 + norm + unembed 的 bias

mini = gpt_params(V=96, C=64, E=64, L=2, F=192)
codebot = gpt_params(V=1000, C=256, E=384, L=6, F=4 * 384)
print(f"迷你版 : {mini:>10,}")
print(f"CodeBot: {codebot:>10,}  （约 {codebot / mini:.0f} 倍）")
assert mini == 93_408 and codebot == 11_121_640
assert 96 * 64 + 64 * 64 + 2 * 41_472 + 128 + 96 == mini   # 正文里逐项相加的结果
```

公式是一回事，原书的类是另一回事。下面的测试用这组超参数构造 `codebot.model.GPT`，确认 `sum(p.numel() for p in model.parameters())` 也是 93,408：

```localcheck verify/test_trainer.py::test_param_counts title="DLFS6 的 GPT 在迷你尺寸下有 93,408 个参数"
```

JavaScript 这边，每个参数都按 PyTorch 的 `state_dict` 取名，形状也与 PyTorch 相同。这样，两边的权重可以直接互相搬运，T.5 的对拍就靠这一点：

```js
  m.tok = add('embed.weight', [V, E], 'normal'); // also unembed.weight (tied)
  m.pos = add('pos_embed.weight', [C, E], 'normal');
```

### 初始 loss

初始化同样照搬 `_init_weights`：所有线性层和嵌入的权重取自 $\mathcal{N}(0, 0.02^2)$，bias 为 0，LayerNorm 的 γ 为 1、β 为 0。

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

::: think T.1 的曲线在第 0 步的留出 loss 是 4.54，比瞎猜的 ln 96 = 4.564 还低。一个随机初始化的模型，怎么会比瞎猜还准？
:::
::: answer
原因在权重共享。初始时，残差流里带着当前字符自己的嵌入向量 $e_c$，而 `unembed` 用的恰好是同一张表，所以“下一个字符还是 c”这一项的 logit 会比别的字符高出一截。代码里，下一个字符与当前字符相同的情况并不少见（缩进里连续的空格、`==`、`''`），占了 8.58%。模型一出生就带着这个偏好，而这个偏好碰巧对了一部分，loss 就略低于 ln V。

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
:::

## T.4 反向传播：每一层都自己写

在 PyTorch 里，`loss.backward()` 一行就算完所有参数的梯度。浏览器里没有 PyTorch，梯度要我们自己算。好在 GPT 只由几种零件搭成，每种零件只需要写一次 forward 和一次 backward。

如果你读过《深度学习入门：基于Python的理论与实现》第 5 章，对这种写法一定不陌生：每一层都是一个带 forward 和 backward 的小零件，正向传播按顺序调用，反向传播按逆序调用。迷你 GPT 一共只用到下面 6 种零件（残差连接是加法，梯度原样流过，就不单独列了）。表中 dy 是上游传来的梯度 ∂L/∂y。

| 层 | 前向 | 对输入的梯度 | 对参数的梯度 | PyTorch 里的对应物 |
|---|---|---|---|---|
| 嵌入 | x = E[id] + P[t] | — | 按 id 累加 dy | `nn.Embedding` |
| 线性 | y = xWᵀ + b | dx = dy·W | dW = dyᵀx，db = Σdy | `nn.Linear` |
| LayerNorm | y = γ⊙x̂ + β | 见下文 | dγ = Σ dy⊙x̂，dβ = Σ dy | `nn.LayerNorm` |
| GELU | y = x·Φ(x) | dx = dy⊙(Φ(x) + x·φ(x)) | — | `nn.GELU()` |
| 因果 Attention | O = softmax(QKᵀ/√D + M)·V | 见下文 | — | `MultiHeadAttention` |
| softmax + 交叉熵 | L = −(1/N)·Σ ln p(答案) | (p − onehot)/N | — | `F.cross_entropy` |

下面逐个来看。每个零件都配一个 NumPy 的验证单元：先按公式算出梯度，再用**数值微分**（中心差分）独立算一遍，两者应当一致。

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

线性层 $y = xW^\top + b$ 的反向传播有三个公式：$dx = dy\,W$，$dW = dy^\top x$，$db$ 是 $dy$ 沿 batch 方向的和。这里，$x$ 的形状是 $[N, \text{in}]$，$W$ 是 $[\text{out}, \text{in}]$（PyTorch 的存法），$dy$ 是 $[N, \text{out}]$。公式不必死记：$dW$ 的形状必须与 $W$ 相同，是 $[\text{out}, \text{in}]$，能凑出这个形状的乘法只有 $dy^\top x$。如果你读过《深度学习入门：基于Python的理论与实现》5.6.2 节或《深度学习入门2：自制框架》步骤 41，用的正是这个“按形状反推”的技巧。

```verify id=T.4-b title="线性层：dx = dy·W，dW = dyᵀ·x，db = Σdy"
rng = np.random.default_rng(1)
x, Wl, b = rng.normal(size=(3, 4)), rng.normal(size=(2, 4)), rng.normal(size=2)
R = rng.normal(size=(3, 2))                      # 上游传来的梯度 dL/dy（L = sum(y * R)）
L = lambda: ((x @ Wl.T + b) * R).sum()
dx, dW, db = R @ Wl, R.T @ x, R.sum(axis=0)      # 反向传播的三个公式
for name, a, ref in [("dx", dx, x), ("dW", dW, Wl), ("db", db, b)]:
    e = rel_err(a, num_grad(L, ref))
    assert e < 1e-9 and a.shape == ref.shape
    print(f"{name}: 形状 {a.shape}，与数值微分一致（相对误差 < 1e-9）")
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
```

### LayerNorm

LayerNorm 先把每个 token 的向量标准化，再缩放和平移：

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \odot \hat{x} + \beta$$

这里，$\mu$ 和 $\sigma^2$ 是这个向量 64 个分量的均值和（有偏）方差，$\epsilon = 10^{-5}$。反向传播的难点在于，$\mu$ 和 $\sigma$ 本身也依赖 $x$。记 $g = dy \odot \gamma$，推导之后得到

$$dx = \frac{1}{\sqrt{\sigma^2 + \epsilon}}\left(g - \overline{g} - \hat{x} \odot \overline{g \odot \hat{x}}\right)$$

其中上划线表示对这 64 个分量取平均。减去的两项分别来自 $\mu$ 和 $\sigma$。它有一个可以心算的性质：给 $x$ 的所有分量加上同一个数，输出不变，所以 $dx$ 各分量之和必须为 0。

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
```

如果你读过《深度学习入门：基于Python的理论与实现》6.3 节，那里的 Batch Norm 没有展开反向传播的推导。LayerNorm 的推导与它是同一回事，只是均值和方差沿特征方向而不是 batch 方向计算。JavaScript 版逐行对应上面的公式：

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

JavaScript 有 `Math.exp`，却没有 `Math.erf`。训练器里的 `erf` 移植自 fdlibm 的有理逼近（[Go 语言的 `math.Erf`](https://go.dev/src/math/erf.go) 用的也是这一份），在 −7 到 7 之间与 Python 的 `math.erf` 相差不超过 2.3e-16，在双精度浮点数的末位量级。下一节会看到，这里偷懒用近似公式，对拍就过不去。

### 因果 Attention

Attention 的前向传播与 `codebot/model.py` 一一对应：打分、遮住未来、softmax、加权求和。

```listing codebot/model.py#L33-L41
```

反向传播从后往前走。记 $P$ 为 softmax 之后的注意力权重，$dO$ 为输出的梯度：

- $O = PV$ 是乘法，所以 $dV = P^\top dO$，而 $P$ 的梯度是 $G = dO\,V^\top$；
- softmax 的反向传播：$dS = P \odot \left(G - \text{rowsum}(P \odot G)\right)$，再乘上缩放 $1/\sqrt{D}$；
- $S = QK^\top$ 又是乘法：$dQ = dS\,K$，$dK = dS^\top Q$。

被遮住的位置，$P$ 恰好为 0，于是 $dS$ 也恰好为 0：未来的 token 拿不到梯度，因果性在反向传播中同样成立。

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
```

JavaScript 版对每个 batch、每个头做同样的几次乘法：

```js
      matmulBT(dO, V, G, T, D, T); // G[i, j] = dout_i . v_j
      for (let i = 0; i < T; i++) {
        const r = i * T;
        let dot = 0;
        for (let j = 0; j <= i; j++) dot += Pb[r + j] * G[r + j];
        for (let j = 0; j < T; j++) dS[r + j] = j <= i ? Pb[r + j] * (G[r + j] - dot) * scale : 0;
      }
```

### 嵌入与权重共享

嵌入层是查表：第 t 个位置取出词表的第 id 行。反向传播就是把梯度按 id 加回去。同一个 token 出现几次，它那一行就收到几份梯度，所以是累加，不是覆盖。如果你读过《深度学习入门2：自制框架》附录 B，`np.add.at` 做的正是这件事。

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

如果你读过《深度学习入门：基于Python的理论与实现》5.7.3 节，会记得这叫**梯度确认**（gradient check），而且误差不会恰好为 0：数值微分本身有截断误差和舍入误差。我们用的判据是 $\|a - b\| / (\|a\| + \|b\|)$，它不会因为某几个梯度分量接近 0 而失真。

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

所以还需要第二道检查：与原书的模型本身对拍。做法是这样的。`verify/_trainer_export.py` 用 DLFS6 的 `codebot.model.GPT` 构造一个很小的模型（V = 23，上下文 16，嵌入 16 维，4 个头，2 层，FFN 64），关掉 dropout，用 float64 计算；把随机权重、一批 token、loss、全部梯度，以及接下来 3 步 `torch.optim.AdamW` 之后的参数都存成文件。JavaScript 按 `state_dict` 的名字读入同样的权重，再算一遍。

| 对拍的内容 | 容差 |
|---|---|
| 一批数据上的 loss | 绝对误差 < 1e-12 |
| 全部 29 个参数张量的梯度 | 相对误差 < 1e-12 |
| 3 步 AdamW 之后的 loss 和参数 | < 1e-11 |
| 同样的权重，换成 float32 计算（浏览器训练时用的精度） | 相对误差 < 1e-5 |
| 实际尺寸（V = 96，93,408 个参数）的 loss 与梯度校验和 | < 1e-10 |

全部通过。下面第一个测试确认这些文件确实出自 DLFS6 的 GPT（重新导出一遍，结果相同），第二个测试运行 JavaScript 一侧的对拍：

```localcheck verify/test_trainer.py::test_parity_fixtures_come_from_dlfs6_gpt title="对拍文件可以从 DLFS6 的 GPT 重新生成"
```

```localcheck verify/test_trainer.py::test_js_matches_dlfs6_gpt title="JavaScript 训练器与 DLFS6 的 GPT 对拍"
```

::: cs336 A1 的测试也是“与 PyTorch 对拍”
[CS336 作业 1](https://github.com/stanford-cs336/assignment1-basics) 规定，除了 `nn.Parameter`、容器类和 `optim.Optimizer` 基类，不许使用 `torch.nn`、`torch.nn.functional` 和 `torch.optim` 里现成的定义；交叉熵要与 `F.cross_entropy` 比对（atol 1e-4），手写的 AdamW 要在 `Linear(3, 2)` 上跑 1,000 步后与 `torch.optim.AdamW` 比对（atol 1e-4），模型的各个零件则与固定权重下的快照比对（精确的运算 atol 1e-5，整模型 1e-4）。思路与本节相同。不同的是，CS336 的学生仍然可以用 autograd 求梯度；我们连反向传播也是手写的，所以多了 T.4 的梯度确认这一道。
:::

::: note WebAssembly SIMD
训练时，矩阵乘法占了大部分时间。浏览器支持 WebAssembly 的 SIMD 指令时，训练器会把 `matmulBT` 换成一个 670 字节的 WebAssembly 版本：一条指令同时做 4 个 float32 的乘加。它与 JavaScript 版算的是同一个乘积，只是累加用 float32 而不是 float64，两者相差约 1e-7（相对）；上表里 float32 那一行的对拍，走的正是这条路径。在录制用的机器上，它让每一步训练快了 3 倍多：

```verify id=T.5-b title="WebAssembly SIMD 对训练速度的影响（录制时测得）"
sps = rec["stepsPerSecond"]
print(f"每秒训练步数：SIMD {sps['simd']}，纯 JavaScript {sps['js']}，快 {sps['simd'] / sps['js']:.1f} 倍")
assert sps["simd"] / sps["js"] > 3
```

T.6 的训练器上有一个开关，可以在你自己的浏览器里关掉它比较。
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

### 取窗口

训练数据的组织方式也与 CodeBot 相同。`TokenDataset` 从 token 序列里切出长度为 `context_len` 的窗口，目标 `y` 是输入 `x` 错开一位；`DataLoader(shuffle=True)` 把所有窗口打乱后依次取用。

```listing ch03/01_pretrain.py#L33-L50
```

训练器的 `makeBatcher` 做同样的事：每个 epoch 把全部窗口的起点打乱一次，依次取 8 个组成一个 batch，取完再重新打乱。我们的训练文本有 211,113 个 token，可以切出 211,049 个窗口。2,000 步只用掉 16,000 个，还不到一个 epoch 的 8%，所以模型几乎不会两次看到同一个窗口。

```verify id=T.6-a title="窗口数与 2,000 步用掉的比例"
n_windows = len(t) - 64                          # 起点 0 .. N-C-1，与 TokenDataset.__len__ 相同
used = 2000 * 8
print(f"窗口数 {n_windows:,}，2,000 步用掉 {used:,} 个，占 {used / n_windows:.1%}")
assert n_windows == 211_049 and used / n_windows < 0.08
```

### AdamW

优化器是 `torch.optim.AdamW`，超参数全部用默认值：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$，weight decay $\lambda = 0.01$，并且作用于所有参数（包括 LayerNorm 和 bias），与 `01_pretrain.py` 第 62 行一致。

```listing ch03/01_pretrain.py#L62-L62
```

一步 AdamW 是这样的：

$$m \leftarrow \beta_1 m + (1-\beta_1) g,\quad v \leftarrow \beta_2 v + (1-\beta_2) g^2,\quad w \leftarrow w(1 - \eta\lambda) - \eta\,\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)} + \epsilon}$$

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

训练器里的 AdamW 逐元素地做同样的计算（T.5 已经确认它与 `torch.optim.AdamW` 一致到 1e-11）：

```js
      for (let i = 0; i < w.length; i++) {
        w[i] *= decay; // decoupled weight decay
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        w[i] -= (stepSize * m[i]) / (Math.sqrt(v[i]) / bc2Sqrt + eps);
      }
```

### 这些数字从哪来

训练器的每个超参数，要么照搬原书，要么是我们量出来的：

| 超参数 | 取值 | 出处 |
|---|---|---|
| 学习率 | 3e-3 | 在 3e-4、3e-3、1e-2 上各训练 1,000 步，3e-3 最好（见本节末尾的“你可能会困惑”） |
| batch | 8 条 × 64 个字符 | 录制的机器上每秒近 10 步，看得到曲线一路往下走 |
| AdamW 的 β、ε、weight decay | 0.9 / 0.999、1e-8、0.01 | `torch.optim.AdamW` 的默认值，与 `01_pretrain.py` 相同 |
| 初始化 | 正态分布 N(0, 0.02²) | `codebot/model.py` 的 `_init_weights` |
| 留出 loss | 每 25 步测一次，16 段 × 64 个字符 | 均匀地取自 86 篇留出文档 |
| 样例 | 每 100 步一次，温度 0.8，从 `def ` 开始写 120 个字符 | 遇到 `<|endoftext|>` 不停下，以标记显示后接着写（原书的 `generate` 会停） |
| 随机种子 | 0 | 每次“重置”之后都能从头复现 |

### 为什么要放进 Web Worker

浏览器的页面只有一个主线程，它既要跑 JavaScript，又要响应点击、重绘页面。如果训练循环直接在主线程里跑，在它跑完之前，页面就会卡住不动。**Web Worker** 是浏览器提供的后台线程：训练放在 Worker 里（`web/widgets/trainer-worker.js`），每训练约 0.1 秒就把这段时间的 loss 发回页面一次，页面只负责画图。暂停、换学习率这些操作，也是作为消息发给 Worker 的。

### 亲手训练

现在轮到你了。点“开始训练”，看 loss 往下走；每 100 步，下方会多出一段模型写的文字。训练中可以随时暂停、换学习率，或者切换到“过拟合一个 batch”模式（T.7 会用到）。

```widget trainer-live caption="浏览器里的迷你 GPT。曲线是实时的：训练 loss 每 10 步取平均，留出 loss 每 25 步测一次；三条虚线依次是瞎猜、unigram 和 bigram 的水平。"
{"batch": 8, "lr": 0.003, "maxSteps": 1000, "evalEvery": 25, "sampleEvery": 100, "sampleChars": 120, "prompt": "def ", "temperature": 0.8, "seed": 0}
```

你应该看到这样几个阶段（数字来自 T.1 录制的同一份设置）：

1. **第 0 步**：留出 loss 约 4.54，样例是各种符号的乱码。
2. **前 25 步**：loss 掉到 unigram 线以下。模型最先学会的是字符的频率（空格最多，其次是 e 和 t），录制中第 100 步的样例，四成多是空格。
3. **第 475 步前后**：越过 bigram 线。从这里开始，模型用上了比前一个字符更远的上下文。
4. **第 1,000 步**：留出 loss 约 2.28。样例里能认出 `#` 注释、`=` 和 `for` 这样的零件，但还拼不成像样的代码。

在 Chrome、Edge 这类以 V8 为引擎的浏览器里，不改设置、只按“开始训练”，曲线会与 T.1 的录制逐步相同（我们在无头 Chromium 里核对过前 120 步）。其他浏览器没有逐位核对过：数学函数的实现若有细微差别，数字会略有出入，但走势相同。

::: confusion 学习率为什么不用 CodeBot 的 3e-4？
直觉上，原书的超参数应该最可靠，照搬总没错。可是 3e-4 是 CodeBot 用的值，它面对的是另一个问题：1,112 万个参数，训练 2 万步。迷你 GPT 小了 119 倍，训练只有一两千步，最合适的学习率没有理由一样。下图是同一个模型、同一个随机种子，只换学习率，各训练 1,000 步的留出 loss：

```widget trainer-replay caption="学习率 3e-3 的留出 loss 最低（1,000 步时 2.28）；3e-4 太保守（2.65，第 900 步才越过 bigram 线）；1e-2 又太激进（2.82），1,000 步内一次也没有越过 bigram 线。"
{"view": "lr", "views": ["lr"]}
```

```verify id=T.6-c title="三种学习率在第 1,000 步的留出 loss"
at1000 = {rid: dict((s, x) for s, x in rec["runs"][rid]["held"])[1000] for rid in ("lr3e-4", "default", "lr1e-2")}
for rid, x in at1000.items():
    print(f"学习率 {rec['runs'][rid]['lr']:.0e}: 第 1,000 步留出 loss {x:.3f}")
assert at1000["default"] < at1000["lr3e-4"] - 0.3 and at1000["default"] < at1000["lr1e-2"] - 0.3
cross = {rid: next((s for s, x in rec["runs"][rid]["held"] if x < base["bigram"]), None) for rid in at1000}
print("第一次低于 bigram 线的步数:", cross)
assert cross == {"lr3e-4": 900, "default": 475, "lr1e-2": None}
```

直观上，学习率太小，每一步都走得太近；太大，又容易在谷底附近来回晃荡，下不去。可以带走的结论是：**学习率是量出来的，不是抄来的**。换了模型的尺寸或训练的长度，就重新扫一遍。[第 6 章](ch06.html)还会在学习率上加 warmup 和衰减。
:::

## T.7 先让它过拟合一个 batch

开始一次很长的训练之前，有一个花不了几秒钟的检查：只拿一个 batch 反复训练。如果模型和训练代码都没有错，它应该能把这一个 batch 背下来，loss 一路降到接近 0。做不到，多半说明代码里有 bug，或者学习率错得离谱；这时候加数据、加步数都无济于事。

把训练器的模式切换到“过拟合一个 batch”再开始训练，就是这个检查。下图是录好的结果：

```widget trainer-replay caption="反复训练同一个 batch：训练 loss 在 100 步内降到 0.03，300 步时降到 0.003；留出 loss 在第 25 步降到 3.42 之后便一路上升，第 100 步已超过瞎猜的水平，300 步时升到 6.6。"
{"view": "overfit", "views": ["overfit"]}
```

```verify id=T.7-a title="录制的过拟合运行"
of = rec["runs"]["overfit"]
mean10 = lambda a: sum(of["losses"][a:a + 10]) / 10      # 第 a..a+9 步的平均训练 loss
held_of = dict((s, x) for s, x in of["held"])
for a in (0, 40, 90, 290):
    print(f"第 {a:>3}–{a + 9:<3} 步  训练 loss {mean10(a):.4f}")
print(f"留出 loss：第 0 步 {held_of[0]:.2f}，第 100 步 {held_of[100]:.2f}，第 300 步 {held_of[300]:.2f}（ln 96 = {ln_v:.2f}）")
assert mean10(90) < 0.035 and mean10(290) < 0.003
assert held_of[100] > ln_v and abs(held_of[300] - 6.60) < 0.01
```

300 步之后，这 512 个位置的平均 loss 是 0.0027，换算成概率，模型给正确答案的概率（几何平均）是 $e^{-0.0027} \approx 99.7\%$：它把这一页背得滚瓜烂熟。可是在没见过的文本上，它比瞎猜还差。

同样的检查在 PyTorch 里也成立：用 DLFS6 的 GPT 以同样的尺寸、同样的学习率反复训练一个 batch，150 步内训练 loss 就降到 0.05 以下。

```localcheck verify/test_trainer.py::test_dlfs6_gpt_overfits_one_batch title="DLFS6 的 GPT 同样能过拟合一个 batch"
```

::: think 过拟合之后，留出 loss 为什么会比瞎猜（4.56）还高？
:::
::: answer
瞎猜对每个字符都给 1/96 的概率，错也错得不多。背熟了一个 batch 的模型却非常“自信”：它把几乎全部概率押在“在那个 batch 里出现过的下一个字符”上。到了新文本，这种自信的猜测经常押错，而交叉熵对“自信地猜错”的惩罚是 $-\ln p$：给正确答案的概率只剩 0.1% 时，这一个位置的 loss 就是 $-\ln 0.001 \approx 6.9$。
:::

::: confusion loss 不降：该怪数据、学习率，还是步数不够？
这三种怀疑都很自然，也都可能对。麻烦在于，它们表现出来是同一个样子：一条平的曲线。过拟合一个 batch 能先把它们分开。

- 连一个 batch 都背不下来：问题出在模型或训练代码（或者学习率错了几个数量级），与数据量、步数无关。先查代码。
- 一个 batch 能背下来，正常训练却停滞不前：代码大概率没问题，再去看学习率（本页 T.6 的比较）、数据和训练长度。

这个检查只要一两百步。在一次要跑几小时的训练开始之前，花这一两百步很值得。
:::

::: confusion 训练 loss 很低，就说明模型学会了吗？
不一定。过拟合的那条曲线就是反例：训练 loss 0.003，留出 loss 6.6。训练 loss 衡量的是模型对见过的数据记得多牢，留出 loss 衡量的才是它在没见过的数据上表现如何，也就是我们真正关心的泛化。`ch03/01_pretrain.py` 没有留出集，只记录训练 loss；[第 6 章](ch06.html)的 StoryBot 才加上验证集。本页的训练器从一开始就把两条曲线画在一起，正常训练时它们靠得很近（2,000 步时 1.87 对 1.95），过拟合时则迅速分开。
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

```verify id=T.8-a title="训练计算量：6ND"
mini_tokens = 2000 * 8 * 64                      # 2,000 步 × 8 条 × 64 个字符
codebot_tokens = 20_000 * 32 * 256               # ch03/01_pretrain.py: max_iters × batch_size × context_len
mini_flops, codebot_flops = 6 * mini * mini_tokens, 6 * codebot * codebot_tokens
print(f"迷你版 : {mini_tokens:>13,} token，6ND = {mini_flops:.2e} FLOPs")
print(f"CodeBot: {codebot_tokens:>13,} token，6ND = {codebot_flops:.2e} FLOPs（{codebot_flops / mini_flops:,.0f} 倍）")
days = codebot_flops / mini_flops * rec["runs"]["default"]["seconds"] / 86400
print(f"按迷你版的速度（2,000 步 {rec['runs']['default']['seconds']:.0f} 秒）粗算，CodeBot 要 {days:.0f} 天")
assert codebot_tokens == 163_840_000 and 1.09e16 < codebot_flops < 1.10e16
assert round(days) == 45
```

计算量差了约 19,000 倍。迷你版在录制用的机器上跑 2,000 步用了 205 秒；按同样的计算速度粗算，CodeBot 要 45 天左右。这正是 CodeBot 必须交给 GPU 的原因。除了尺寸，放大时还有几处要换掉：分词器从字符级换成第 1 章的 BPE，序列短了约 2.36 倍，同样 256 个 token 能装下更多代码；dropout 恢复为 0.1；矩阵乘法交给 GPU 上的 cuBLAS，不再需要自己写 SIMD。

::: cs336 先在 CPU 上调通，再上 GPU
[CS336](https://cs336.stanford.edu/) 给学生的建议是先在 CPU 上调试，再用 GPU 跑作业 1、4、5 的训练和作业 2 的基准测试。作业 1 的 TinyStories 语言模型也有一个“低资源”版本：训练 4,000 万个 token，目标是验证集 loss 不高于 2.00；参考实现在 M4 Max 的 CPU 上跑了 1 小时 22 分钟（用 MPS 是 36 分钟），达到 1.80。本页的过拟合检查和对拍，正是这种“先在小尺寸上把代码调对”的做法。
:::

## 小结

这一页，我们在浏览器里把第 3 章的预训练完整地走了一遍，只是计算量缩小到原来的约 1/19,000。数据是 TinyCodes 的一小片，分词器是字符级的；模型照搬 `codebot/model.py`，参数从 1,112 万缩到 9 万；前向传播、反向传播和 AdamW 都用 JavaScript 手写，先逐层做梯度确认，再与 DLFS6 的 GPT 对拍。训练时，loss 从 ln V 附近出发，先越过 unigram 线，再越过 bigram 线，生成的文字也从符号乱码慢慢变得像代码。

本章所学的内容：

- 字符级分词器的词表是 95 种字符加 `<|endoftext|>`，共 96 个；loss 的三个参照点是瞎猜 ln 96 = 4.564、unigram 3.387、bigram 2.693（[验证 T.2-b](#v-T.2-b)）。
- 迷你 GPT 与 CodeBot 是同一个类、不同的尺寸，参数量 93,408 对 11,121,640（[验证 T.3-a](#v-T.3-a)）；随机初始化时 loss 约为 ln V + σ²/2（[验证 T.3-b](#v-T.3-b)），在代码文本上还会因权重共享略低于 ln V（[验证 T.3-c](#v-T.3-c)）。
- 交叉熵对 logits 的梯度是 (softmax − one-hot)/N（[验证 T.4-a](#v-T.4-a)）；线性层 dx = dy·W、dW = dyᵀ·x（[验证 T.4-b](#v-T.4-b)）。
- Attention 的反向传播多出的只有 softmax 的反向 dS = P ⊙ (G − rowsum(P ⊙ G))，被遮住的位置梯度恰好为 0（[验证 T.4-e](#v-T.4-e)）；权重共享时，两条路径的梯度相加（[验证 T.4-f](#v-T.4-f)）。
- 梯度确认只证明反向与前向一致；前向是不是 CodeBot 的那个函数，要靠与 PyTorch 对拍（[验证 T.5-a](#v-T.5-a)）。
- AdamW 的第一步，每个参数都移动约 lr，与梯度的大小无关（[验证 T.6-b](#v-T.6-b)）；学习率要量出来，不能照抄（[验证 T.6-c](#v-T.6-c)）。
- 能把一个 batch 背下来，说明代码没有大错；训练 loss 低不等于学会了，要看留出 loss（[验证 T.7-a](#v-T.7-a)）。

迷你 GPT 已经能写出一些像代码的零件，离 CodeBot 还差得远，差距就在那 19,000 倍的计算量上（训练好的 CodeBot 能写出什么，可以在 [CodeBot 工作台](playground.html) 里试）。下一页的 [GPU 实战手册](gpu.html)，就把 CodeBot 的训练真正搬上 GPU。
