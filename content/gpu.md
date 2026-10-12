---
chapter: G
slug: gpu
order: 170
kind: extra
title: GPU 实战手册：本机、Modal、RunPod 与 Lambda
nav_title: GPU 实战手册
lede: 三台机器人对算力的胃口相差近千倍。这一页先把账算清楚，再分别讲在本机、Modal、RunPod 和 Lambda 上怎样跑原书脚本，最后讲跑出来的结果怎样回到这本书里。每个数字都有出处，每条命令都可以直接复制。
pyodide_packages: [numpy]
---

前面各章，我们把三台机器人的分词器、模型和训练脚本一行一行地读了下来。读懂了，自然想亲手跑一遍。可是三台机器人对算力的要求很不一样：原书前言说，第 I 部只用 CPU 也能运行，第 II 部推荐用 GPU，第 III 部则必须有 GPU，最好是 A100 或 H100 级别的卡[^preface]。

[^preface]: 前言原文转引自 [note.com 上一篇读书笔记](https://note.com/_heartyfluid/n/nc73f03d4b18c)（2026-10 查证）。前言还提到，没有 GPU 的读者可以用 Google Colab、Kaggle 的免费 GPU（有限制），或者在 RunPod、Lambda、Vast.ai 上“比较便宜地”租用实例。

那么，具体要花多少时间、多少钱？自己的 RTX 3060 Ti 能跑到哪一步，什么时候该租云端的卡？跑完以后，训练曲线和运行记录又怎样回到本书的页面上？这一页回答这三个问题。

我们先看一张总账（G.1），再把它背后的公式一步一步算出来，并做成一个计算器（G.2）。接着专门讲多卡训练的一个陷阱，它能让账单翻上 8 倍（G.3）。然后依次是本机（G.4）、Modal（G.5）、RunPod 与 Lambda（G.6）的具体做法。最后一节讲运行报告怎样回到书里（G.7）。

## G.1 先看结果：三台机器人的账单

先把结论放在前面。下表的数字，后面都会一一算出来或者给出出处。

| | CodeBot | StoryBot | WebBot |
|---|---|---|---|
| 参数量 | 1,112 万 | 2,270 万 | 1.52 亿 |
| 训练 token | 1.64 亿 | 3.28 亿 | 131 亿 |
| 计算量（训练 + 评估） | 1.14 × 10¹⁶ | 5.16 × 10¹⁶ | 1.13 × 10¹⁹ |
| 本机 RTX 3060 Ti | 42 分钟（实测） | 约 1.8 小时（估算） | 放不下 |
| 云端 | 不必上云 | Modal H100 实测：预训练 16 分钟，连同生成与 DPO 约 1.17 美元 | 单卡 H100 约 10.4 小时、28–45 美元（按探针实测外推） |

表中的数字依次出自：参数量 [验证 G.2-b](#v-G.2-b)，训练 token [验证 G.2-c](#v-G.2-c)，计算量 [验证 G.2-e](#v-G.2-e)，本机 [验证 G.2-g](#v-G.2-g) 与 [验证 G.4-a](#v-G.4-a)，云端 [验证 G.2-h](#v-G.2-h)、[验证 G.2-j](#v-G.2-j) 与 [验证 G.5-a](#v-G.5-a)。

这张表说了两件事。第一，CodeBot 在本机跑起来毫无压力；StoryBot 估计也能在本机跑完，不过我们还没有完整跑过（[G.4](#sG-4)）。第二，WebBot 的计算量是 StoryBot 的两百多倍，这一级必须上云，而且值得先花几毛钱做一次“探针”：只跑真实训练的前 200 步，读出速度（[G.2](#sG-2) 的最后一部分细说）。我们的探针花了约 0.27 美元，读出的速度比事前的估算慢 13%，全程因此从 9.1 小时变成了约 10.4 小时。

真正跑过的运行，都记在下面这块“运行记录”里。每份报告回答三个问题：在哪里跑的（机器、GPU、PyTorch 版本），跑的是哪份代码（教程和原书的两个提交号），结果如何（每个脚本的用时、最后的指标和费用）。其中 `codebot-grpo-evalfix-3060ti` 是一次勘误实验：起点是 `codebot-3060ti` 的 SFT 检查点，只在 GRPO 的更新循环前加了一行 `model.eval()`（[勘误 E01](errata.html#e01)）。计划了但还没跑的运行，也连同命令一起列在后面。

```widget gpu-runboard caption="运行记录：每次运行都带着自己的出处。原书提交与子模块一致，说明跑的正是本书引用的那份代码。"
{}
```

## G.2 算账：从 6ND 到美元

估算训练成本，业内最常用的是一个很短的公式：

$$C \approx 6ND$$

这里，$N$ 是参数量，$D$ 是训练看过的 token 数，$C$ 是浮点运算次数（FLOPs）。第 8 章挑选模型规模时用过它（[8.1 节](ch08.html#s8-1)）。

系数 6 是这样来的。一个 token 经过一个权重，要做一次乘法、一次加法，所以前向是每个参数 2 次运算。反向要算两样东西：传给前一层的输入梯度 $dX = dY\,W^\top$，和交给优化器的权重梯度 $dW = X^\top dY$。两者各是一次和前向一样大的矩阵乘法，合计 4 次。前向加反向，$2 + 4 = 6$[^palm]。

[^palm]: PaLM 论文的附录 B 就是这样数的：前向 $2N$、反向 $4N$（[arXiv:2204.02311](https://arxiv.org/abs/2204.02311)，2026-10 查证）。

手算一个例子：一个 $d_{\text{in}} = 3$、$d_{\text{out}} = 2$ 的线性层，一次喂进 4 个 token。前向 $Y = XW$ 是 $2 \times 4 \times 3 \times 2 = 48$ 次运算；反向的 $dX$ 和 $dW$ 各 48 次，合计 96 次，正好是前向的 2 倍。下面按形状把这三次矩阵乘法的运算次数数一遍，再用数值微分确认 $dW$ 确实是这样算的。

```verify id=G.2-a title="系数 6：前向 2 次，反向的 dX 与 dW 各 2 次"
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))          # 4 个 token，d_in = 3
W = rng.normal(size=(3, 2))          # d_out = 2
dY = rng.normal(size=(4, 2))         # 从上游传回来的梯度

def mm_flops(a, b):
    """(m×k) @ (k×n)：每个输出元素 k 次乘法、k 次加法"""
    return 2 * a.shape[0] * a.shape[1] * b.shape[1]

fwd = mm_flops(X, W)                       # Y = X W
bwd = mm_flops(dY, W.T) + mm_flops(X.T, dY)  # dX 与 dW
per = (fwd + bwd) // (X.shape[0] * W.size)
print(f"前向 {fwd} 次，反向 {bwd} 次")
print(f"每个 token、每个参数：{per} 次")
assert (fwd, bwd, per) == (48, 96, 6)

# loss = sum(Y * dY) 对 W 的梯度，应当正好是 X^T dY
dW = X.T @ dY
eps, num = 1e-6, np.zeros_like(W)
for i in range(3):
    for j in range(2):
        Wp = W.copy()
        Wp[i, j] += eps
        num[i, j] = (np.sum(X @ Wp * dY) - np.sum(X @ W * dY)) / eps
err = np.abs(num - dW).max()
# loss 对 W 是线性的，前向差分只剩舍入误差（约 1e-15 / 1e-6 = 1e-9）
assert err < 1e-6
print("dW = X^T dY 与数值微分一致（最大差小于 1e-6）")
```

简单起见，激活函数、归一化和 softmax 这些逐元素的运算都不计入。下面我们把 $N$、$D$ 和 $C$ 依次算出来，再换算成小时和美元。

### 参数量 N

严格地讲，6ND 里的 $N$ 应该是“每个 token 都要乘一遍的矩阵权重”，我们把它记作 $n_{\text{mm}}$。embedding 只是查表，不做乘法，不算在内；RMSNorm、LayerNorm 和 bias 的参数很少，也可以忽略。剩下的就是每一层的 Attention 与 FFN 的线性层，再加上最后把向量变回词表大小的 unembed。

以 CodeBot 为例：每层 Attention 有 $4d^2$ 个权重，GELU 的 FFN 有 $2d \cdot d_{\text{ff}}$ 个，$d = 384$、$d_{\text{ff}} = 1536$，6 层合计 $6 \times (589{,}824 + 1{,}179{,}648) = 10{,}616{,}832$；再加 unembed 的 $1000 \times 384 = 384{,}000$，得到 $n_{\text{mm}} = 11{,}000{,}832$。

下面的函数按同样的思路处理三台机器人。它读的 `bots.json` 是我们从原书脚本抄出的结构与超参数，本地测试会实例化真实模型、逐个核对。

```verify id=G.2-b title="三台机器人的参数量与每 token 的矩阵权重" files=data/gpu/bots.json
import json
import math

bots = json.load(open("data/gpu/bots.json", encoding="utf-8"))["bots"]

def param_counts(b):
    E, L, ff, V, T = b["d"], b["layers"], b["ff"], b["vocab"], b["ctx"]
    # GQA：K、V 只投影到 kv_heads 个头
    kv = b["kv_heads"] * (E // b["heads"])
    # W_q、W_o，加上 W_k、W_v（都没有 bias）
    attn = 2 * E * E + 2 * E * kv
    # CodeBot 是 GPT-2 式，FFN 用 GELU，两个矩阵；
    # 另外两台是现代式，SwiGLU 三个矩阵
    gpt2 = b["ffn"] == "gelu"
    ffn = (2 if gpt2 else 3) * E * ff
    # 加上 FFN 的 bias 与两个 LayerNorm，或者两个 RMSNorm
    block = attn + ffn + (ff + E + 4 * E if gpt2 else 2 * E)
    pos = T * E if b["learned_pos"] else 0
    params = V * E + pos + L * block + (2 * E if gpt2 else E)
    # unembed：CodeBot 与 embedding 共享，但有 bias
    params += 0 if b["tied"] else V * E
    params += V if gpt2 else 0
    # 第二项就是 n_mm
    return params, L * (attn + ffn) + V * E

for name, b in bots.items():
    p, n = param_counts(b)
    print(f"{name:9} 全部参数 {p:>12,}   n_mm {n:>12,}")
    assert (p, n) == (b["params"], b["n_mm"])
assert param_counts(bots["codebot"]) == (11_121_640, 11_000_832)

w = bots["webbot"]
# embedding 与 unembed 各 V × d
emb_share = 2 * w["vocab"] * w["d"] / w["params"]
print(f"WebBot 的 embedding 与 unembed 占全部参数的 {emb_share:.1%}")
assert abs(emb_share - 0.504) < 0.001
```

```localcheck verify/test_gpu.py::test_param_counts_match_the_real_models title="实例化原书的三个 GPT 类，参数量与 n_mm 和上面完全一致" needs="PyTorch"
```

WebBot 的 1.52 亿参数里，embedding 和 unembed 就占了一半（各 3,840 万），这是 5 万词表的代价。

### token 数 D

$D$ 由训练循环决定：每一步取多少条序列、每条多长、一共多少步。CodeBot 每步 32 条 × 256 个 token，跑 20,000 步，$D = 1.64$ 亿。WebBot 每步取 4 次 micro-batch（梯度累积），每次 32 条 × 1,024 个 token，跑 100,000 步。

评估也要算进去。StoryBot 每 500 步、WebBot 每 1,000 步，把整个验证集完整走一遍，最后一步再走一遍。下面的函数照着原书的循环条件 `i % eval_iters == 0 or i == max_iters - 1` 数评估次数，照着 `get_batch` 的窗口条件数每次评估的 token。

```verify id=G.2-c title="训练 token 数与评估次数" files=data/gpu/bots.json
def n_evals(max_iters, every, steps=None):
    """for i in range(steps):
    若 i % every == 0 或 i == max_iters - 1 就评估一次"""
    if not every:
        return 0
    steps = steps or max_iters
    last = 1 if max_iters - 1 < steps and (max_iters - 1) % every else 0
    return math.ceil(steps / every) + last

def eval_windows(valid_tokens, ctx):
    """起点 0, ctx, 2·ctx, …，且 起点 + ctx + 1 < 验证集长度"""
    return max(0, math.ceil((valid_tokens - ctx - 1) / ctx))

def run_shape(b, world=1, max_iters=None, batch=None, accum=None,
              steps=None):
    it = max_iters or b["max_iters"]
    # steps：只跑前几步（探针）
    run = min(steps, it) if steps else it
    mb, acc = batch or b["batch"], accum or b["accum"]
    per_rank = mb * acc * b["ctx"]
    evals = n_evals(it, b["eval_every"], run)
    windows = eval_windows(b["valid_tokens"], b["ctx"])
    return {"world": world, "steps": run,
            "tokens_per_step_per_rank": per_rank,
            "tokens_per_step": per_rank * world,
            "train_tokens": per_rank * world * run,
            "evals": evals, "eval_tokens": windows * b["ctx"] * evals}

for name, b in bots.items():
    s = run_shape(b)
    print(f"{name:9} 训练 {s['train_tokens']:>15,} token   "
          f"评估 {s['evals']:>3} 次，共 {s['eval_tokens']:>14,} token")
cb, sb, wb = bots["codebot"], bots["storybot"], bots["webbot"]
assert run_shape(cb)["train_tokens"] == 20_000 * 32 * 256 == 163_840_000
# 0, 500, …, 39500，再加最后一步 39999
assert run_shape(sb)["evals"] == 81
assert run_shape(wb)["train_tokens"] == 100_000 * 32 * 4 * 1024
# StoryBot：评估 4.4 亿 token，比训练的 3.3 亿还多
s = run_shape(sb)
assert s["eval_tokens"] > s["train_tokens"]
assert round(s["eval_tokens"] / 1e8, 1) == 4.4
```

```localcheck verify/test_gpu.py::test_bots_json_matches_the_book_scripts title="bots.json 的每个超参数都和原书脚本里写的一样（用 ast 读脚本）"
```

StoryBot 的评估一共要过 4.4 亿个 token，比训练本身（3.3 亿）还多。这笔开销后面会再出现。

::: confusion 上下文只有 1,024，怎么一次训练就要 131 亿个 token？
这个疑问很自然：我们习惯把上下文当成“模型手里的全部材料”，于是会以为总量也被它卡住。可是 WebBot 的上下文长度是 1,024，这里的 $D$ 却是一百多亿，两者差了七个数量级。其实它们量的是两件事。上下文长度限制的是**单条序列**，也就是模型做一次预测时最多能往回看多远；$D$ 是整个训练过程**累计**处理过的 token。两者之间隔着两层乘法：一步有很多条序列，一次训练有很多步。

```verify id=G.2-d title="一条序列、一步、全程：三个不同的量" files=data/gpu/bots.json
w = bots["webbot"]
# 一条序列：一次预测最多看见的 token
per_seq = w["ctx"]
# 一步：32 条 × 累积 4 次
per_step = w["batch"] * w["accum"] * w["ctx"]
# 全程：10 万步的累计
per_run = per_step * w["max_iters"]
print(f"一条序列 {per_seq:>14,} token")
print(f"一步     {per_step:>14,} token")
print(f"全程     {per_run:>14,} token")
assert per_step == 128 * per_seq and per_run == 13_107_200_000
```

智能体的账单也是这样记的，只是多了一层重复：每一轮请求都把完整的历史重发一遍。第 5 章算过（[验证 5.5-k](ch05.html#v-5.5-k)）：开头 3,000 个 token、每轮新增 800、走 300 轮，单次请求最多 24.3 万个 token，没有超过 256K 的窗口；累计输入却是 3,702 万，随轮数平方增长。训练的 $D$ 是许多个不同窗口之和，智能体的累计则是同一段历史被反复计数。可以带走的一句话是：**上下文长度是一次能看多远，token 数是一共看了多少。**
:::

### 计算量 C

有了 $n_{\text{mm}}$ 和 $D$，先手算一下 CodeBot：$6 \times 1{,}100 \text{ 万} \times 1.64 \text{ 亿} \approx 1.08 \times 10^{16}$。

6ND 漏掉了一项：Attention 里 $QK^\top$ 和“权重乘 $V$”这两次矩阵乘法。它们的计算量随上下文长度增长，与参数无关。一个 query 要和 $T$ 个 key 各做一次 $d$ 维点积，是 $2Td$ 次运算；再用 $T$ 个权重把 $T$ 个 $d$ 维的 value 加起来，又是 $2Td$，合计 $4Td$。因果掩码下，第 $t$ 个位置只看前 $t$ 个位置，平均只看一半，所以我们按 $2Td$ 计。每层前向 $2Td$，训练时乘 3，每个 token 就是 $6LTd$。评估只有前向，每个 token 是 $2n_{\text{mm}} + 2LTd$。于是

$$C = (6n_{\text{mm}} + 6LTd)\,D_{\text{train}} + (2n_{\text{mm}} + 2LTd)\,D_{\text{eval}}$$

这里，$L$ 是层数，$T$ 是上下文长度，$d$ 是 embedding 维度。以 WebBot 为例：$6 \times 12 \times 1024 \times 768 \approx 5{,}660$ 万，而 $6n_{\text{mm}} \approx 6.83$ 亿，Attention 只占每个 token 计算量的 7.7%。

```verify id=G.2-e title="每个 token 的计算量与全程 FLOPs" files=data/gpu/bots.json
def flops(b, s):
    attn = b["layers"] * b["ctx"] * b["d"]
    train = (6 * b["n_mm"] + 6 * attn) * s["train_tokens"]
    evals = (2 * b["n_mm"] + 2 * attn) * s["eval_tokens"]
    return train, evals

for name, b in bots.items():
    per_tok = 6 * b["n_mm"] + 6 * b["layers"] * b["ctx"] * b["d"]
    train, evals = flops(b, run_shape(b))
    print(f"{name:9} 每 token {per_tok / 1e6:6.1f} M FLOPs   "
          f"训练 {train:.3e}   评估 {evals:.3e}")

six_nd = 6 * cb["n_mm"] * run_shape(cb)["train_tokens"]
print(f"CodeBot 只算 6ND：{six_nd:.3e}")
# 和手算一致
assert abs(six_nd / 1.08e16 - 1) < 0.01
sb_train, sb_eval = flops(sb, run_shape(sb))
print(f"StoryBot 的评估相当于训练的 {sb_eval / sb_train:.0%}")
assert abs(sb_eval / sb_train - 0.45) < 0.01
attn = 6 * 12 * 1024 * 768
attn_share = attn / (6 * wb["n_mm"] + attn)
print(f"WebBot：6LTd = {attn:,}，6·n_mm = {6 * wb['n_mm']:,}，"
      f"Attention 占 {attn_share:.1%}")
assert abs(attn_share - 0.077) < 0.0005
total = {name: sum(flops(b, run_shape(b))) for name, b in bots.items()}
ratio = total["webbot"] / total["storybot"]
span = total["webbot"] / total["codebot"]
print(f"WebBot 的总计算量是 StoryBot 的 {ratio:.0f} 倍、"
      f"CodeBot 的 {span:.0f} 倍")
assert 200 < ratio < 300 and 900 < span < 1100
# G.1 表里的三个总数
g1 = [f"{total[n]:.2e}" for n in bots]
assert g1 == ["1.14e+16", "5.16e+16", "1.13e+19"]
```

注意这是一个口径。PaLM 论文附录 B 和 nanoGPT 的 `estimate_mfu` 都不打这个对折，Attention 一项按 $12LTd$ 计[^attn12]。引用别处的 MFU 之前，先对一下分子是怎么数的。

[^attn12]: [PaLM，arXiv:2204.02311](https://arxiv.org/abs/2204.02311) 附录 B 写作 $6N + 12LHQT$，其中 $HQ = d$；nanoGPT 的 [`model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) 里是 `6*N + 12*L*H*Q*T`（均为 2026-10 查证）。

StoryBot 每 500 步就完整评估一次，多出了 45% 的计算；WebBot 的上下文长到 1,024，Attention 那一项也只占 7.7%。这两个数都会直接反映在账单上。

### 换算成小时与美元

计算量除以 GPU 实际做到的算力，就是时间：

$$\text{小时} = \frac{C}{\text{峰值} \times \text{MFU} \times 3600}$$

这里，峰值是 GPU 数据表上的理论算力，单位是 FLOP/s（数据表上常写成 TFLOP/s，代入时要乘上 $10^{12}$）；**MFU**（Model FLOPs Utilization）是实际用上的比例。以 CodeBot 为例：它用 FP32 训练，3060 Ti 的 FP32 峰值是 16.2 TFLOP/s，取 MFU 30%，$1.14 \times 10^{16} \div (16.2 \times 10^{12} \times 0.30 \times 3600) \approx 0.65$ 小时，约 39 分钟。

MFU 像工厂的开工率：铭牌上写着一小时能冲 1,000 个零件，实际冲了 310 个，开工率就是 31%。严格地讲，MFU 的分子只数 $6n_{\text{mm}} + 6LTd$ 这些“有用的运算”，重算、通信和等待都不算在内。

数据表上的数字都要读仔细：NVIDIA 给 H100 标的 BF16 算力是 1,979 TFLOP/s，旁边有一行小字“with sparsity”，那是结构化稀疏下的数字，普通训练要除以 2，按 989.5 计算。本页用到的每块 GPU 的峰值，都在 `web/data/gpu/gpus.json` 里写明了来源。

消费级显卡的数据表通常不写 Tensor Core 算力，我们可以从白皮书推出来。下面以本机的 RTX 3060 Ti 为例，看看 32.5 这个数从哪来。

```verify id=G.2-f title="RTX 3060 Ti 的峰值算力是怎样推出来的" files=data/gpu/gpus.json
# GA102 白皮书表 2：RTX 3080 有 68 个 SM，加速频率 1.71 GHz，
# BF16（FP32 累加）59.5 TFLOPS
per_sm_clk = 59.5e12 / (68 * 1.71e9)
# RTX 3060 Ti：4864 个 CUDA 核，每个 SM 128 个；加速频率 1.67 GHz
sm = 4864 // 128
bf16 = sm * 512 * 1.67e9 / 1e12
# 每个核每个时钟一次乘加 = 2 次运算
fp32 = 4864 * 2 * 1.67e9 / 1e12
print(f"RTX 3080 每个 SM 每个时钟 {per_sm_clk:.1f} 次运算；"
      f"3060 Ti 有 {sm} 个 SM")
print(f"3060 Ti 峰值：BF16 {bf16:.1f} TFLOP/s，FP32 {fp32:.2f} TFLOP/s")

gpus = json.load(open("data/gpu/gpus.json", encoding="utf-8"))
G = {g["id"]: g for g in gpus["gpus"]}
assert round(per_sm_clk) == 512 and sm == 38
assert abs(bf16 - G["rtx3060ti"]["bf16"]) < 0.05
# gpus.json 照数据表取 16.2；按核数算出的是 16.25
assert abs(fp32 - G["rtx3060ti"]["fp32"]) < 0.06
```

严格地讲，这是按数据表上的加速频率（1.67 GHz）算出的“标称峰值”。GPU Boost 会在功耗和温度还有余量时调高频率[^boost]，实际频率并不总是 1.67 GHz，所以 MFU 只是相对这个标称值的比例。

[^boost]: [NVIDIA：GPU Boost](https://www.nvidia.com/en-us/geforce/technologies/gpu-boost/)（2026-10 查证）。

MFU 没法查表，只能估计或者实测。WebBot 的矩阵大，又开了 `torch.compile` 和 Flash Attention，事前我们按 35% 估计（范围 25%–45%）。探针跑出来以后，实测是 31%：每步 0.317 秒，比估计的 0.280 秒慢 13%，全程因此从 9.1 小时变成约 10.4 小时（[验证 G.2-j](#v-G.2-j)）。有了 MFU 就能算出小时数，再乘上单价就是美元。下面先看事前的估算。

```verify id=G.2-g title="事前估算：WebBot 单卡 H100 要多久、多少钱；CodeBot 在本机的估算与实测" files=data/gpu/gpus.json,data/runs/codebot-3060ti/report.json
O = {o["id"]: o for o in gpus["offers"]}

def price(offer):
    """Modal 按秒标价，其余按小时"""
    if "usd_per_gpu_second" in offer:
        return offer["usd_per_gpu_second"] * 3600
    return offer["usd_per_gpu_hour"]

def estimate(b, gpu, usd_per_hour, s, mfu, floor_ms=0.0):
    peak = gpu["bf16"] if b["precision"] == "bf16" else gpu["fp32"]
    # 实际算力，FLOP/s
    eff = peak * 1e12 * mfu
    train, evals = flops(b, s)
    per_tok = 6 * b["n_mm"] + 6 * b["layers"] * b["ctx"] * b["d"]
    step = s["tokens_per_step_per_rank"] * per_tok / eff
    # 评估只在 rank 0 上跑
    wall = s["steps"] * max(step, floor_ms / 1e3) + evals / eff
    return {"step_ms": step * 1e3, "wall_h": wall / 3600,
            "usd": wall * s["world"] / 3600 * usd_per_hour}

h100 = price(O["modal:h100"])
e = estimate(wb, G["h100"], h100, run_shape(wb), 0.35)
c = estimate(cb, G["rtx3060ti"], 0, run_shape(cb), 0.30)
st = estimate(sb, G["rtx3060ti"], 0, run_shape(sb), 0.25)
per_s = O["modal:h100"]["usd_per_gpu_second"]
print(f"Modal H100：{per_s} 美元/秒 = {h100:.2f} 美元/小时")
print(f"WebBot，单卡 H100，MFU 35%：{e['wall_h']:.2f} 小时，"
      f"{e['usd']:.2f} 美元，每步计算 {e['step_ms']:.0f} 毫秒")
print(f"CodeBot，本机 3060 Ti，MFU 30%：{c['wall_h'] * 60:.0f} 分钟")
print(f"StoryBot，本机 3060 Ti，MFU 25%：{st['wall_h']:.2f} 小时")
assert abs(h100 - 3.95) < 0.01
assert 9.0 < e["wall_h"] < 9.2 and 35 < e["usd"] < 37
assert round(c["wall_h"] * 60) == 39
assert abs(st["wall_h"] - 1.76) < 0.005

rep = json.load(open("data/runs/codebot-3060ti/report.json",
                     encoding="utf-8"))
measured_s = next(x["seconds"] for x in rep["steps"]
                  if x["script"] == "ch03/01_pretrain.py")
peak32 = G["rtx3060ti"]["fp32"] * 1e12
mfu = sum(flops(cb, run_shape(cb))) / measured_s / peak32
print(f"CodeBot 实测 {measured_s / 60:.0f} 分钟，折合 MFU {mfu:.0%}")
assert 0.25 < mfu < 0.30

# 第 8 章的粗算：N 取全部参数，不算 Attention 与评估
ch8 = 6 * wb["params"] * run_shape(wb)["train_tokens"]
ch8_h = ch8 / (G["h100"]["bf16"] * 1e12 * 0.35) / 3600
print(f"同样按 MFU 35%，第 8 章的 6·N·D：{ch8_h:.1f} 小时")
# 全部参数比 n_mm 多出的：只查表的 embedding，加 25 个 RMSNorm
emb = wb["vocab"] * wb["d"]
norms = (2 * wb["layers"] + 1) * wb["d"]
print(f"全部参数比 n_mm 多 {wb['params'] - wb['n_mm']:,} 个："
      f"embedding {emb:,}，RMSNorm {norms:,}")
assert round(ch8_h, 1) == 9.6
assert wb["params"] - wb["n_mm"] == emb + norms
```

CodeBot 在本机的估算是 39 分钟，运行记录 `codebot-3060ti` 里的实测是 42 分钟，倒推出的 MFU 是 28%，和我们假设的 30% 很接近。这说明对这台机器、这个模型，30% 是个靠得住的数。换一台机器、换一个模型，就得重新量。

WebBot 要跑多久，第 8 章也事前估过一次，数字和这里不一样，差别出在口径上。第 8 章用 $6ND$ 粗算，$N$ 取全部 1.52 亿参数，不算 Attention 和评估，按 MFU 35% 得到约 9.6 小时（[验证 8.1-f](ch08.html#v-8.1-f)）。这里的 $n_{\text{mm}}$ 少了 3,840 万个只查表的 embedding 参数和 1.92 万个 RMSNorm 权重，又补上了 Attention 与评估两项，得到 9.07 小时。两个估算后来都拿同一次探针校准过，第 8 章、第 9 章和这里推得的都是约 10.4 小时（[验证 8.1-g](ch08.html#v-8.1-g)、[验证 9.7-d](ch09.html#v-9.7-d)、[验证 G.2-j](#v-G.2-j)）。

### 小模型遇上快 GPU

这个公式有一个盲区。StoryBot 每一步只有 8,192 个 token，放到 H100 上，纯计算只要几毫秒。可是每一步还要启动一长串 CUDA kernel、跑一遍 Python 循环、从 memmap 里切出 32 段数据，这些开销和模型大小几乎无关。计算越快，它们占的比例越大。这就像用重型卡车送一封信：路上的时间几乎都花在起步和等红灯上，换一辆更快的卡车也快不了多少。

严格地讲，我们可以给每一步定一个最短耗时，计算再快也不会低于它。下面先只按 FLOPs 估一次，再假设每步至少 15 毫秒估一次。StoryBot 已经在 Modal 的 H100 上原样跑过（运行记录 `storybot-h100`），最后拿实测来对答案。

```verify id=G.2-h title="StoryBot 在 H100 上：固定开销比计算还长，实测是纯 FLOPs 估算的 4 倍多" files=data/runs/storybot-h100/report.json
import re

s = run_shape(sb)
plain = estimate(sb, G["h100"], h100, s, 0.25)
floored = estimate(sb, G["h100"], h100, s, 0.25, floor_ms=15)
print(f"每步的计算只要 {plain['step_ms']:.1f} 毫秒")
print(f"只按 FLOPs 估：{plain['wall_h'] * 60:.1f} 分钟；"
      f"假设每步至少 15 毫秒：{floored['wall_h'] * 60:.1f} 分钟")
assert plain["step_ms"] < 15 and floored["wall_h"] > 3 * plain["wall_h"]

rep = json.load(open("data/runs/storybot-h100/report.json",
                     encoding="utf-8"))
pre = next(x for x in rep["steps"]
           if x["script"] == "ch06/05_pretrain.py")
# 日志最后几行里有一条完整评估的进度条：668 个 batch，平均速度
bar = [ln for ln in pre["log_tail"] if "Validation" in ln][-1]
n, rate = re.search(r"(\d+)/\d+ \[[^\]]*, ([\d.]+)it/s\]", bar).groups()
sb_eval_s = int(n) / float(rate)
eff25 = G["h100"]["bf16"] * 1e12 * 0.25
eval_model = flops(sb, s)[1] / s["evals"] / eff25
step_ms = (pre["seconds"] - s["evals"] * sb_eval_s) / s["steps"] * 1e3
print(f"实测 {pre['seconds'] / 60:.1f} 分钟："
      f"评估每次 {sb_eval_s:.1f} 秒"
      f"（按 FLOPs 只要 {eval_model:.1f} 秒）")
print(f"训练每步约 {step_ms:.0f} 毫秒")
# 固定开销比计算长得多
assert pre["seconds"] > 4 * plain["wall_h"] * 3600
# 训练步和假设的 15 毫秒相当；评估也被开销卡住，所以下限估算仍偏乐观
assert 14 < step_ms < 17 and sb_eval_s > 4 * eval_model
assert pre["seconds"] > floored["wall_h"] * 3600
# CS336 作业 1 的及格线是验证集 loss ≤ 1.45
val = pre["final"]["val_loss"]
print(f"最后的验证集 loss：{val:.3f}")
assert val < 1.45 and round(val, 3) == 1.304
```

实测是 16.2 分钟，是只按 FLOPs 估算的 4 倍多。拆开来看，训练每步约 16 毫秒，和我们假设的 15 毫秒下限几乎一样；81 次评估各花了约 4 秒，而按 FLOPs 算只要 0.8 秒，因为评估的 668 个 batch 同样被固定开销卡住。我们的估算只给训练步加了下限，没有给评估加，所以 11.1 分钟仍然偏乐观。

“每步至少 15 毫秒”这个数，我们把它做成了下面计算器里的一个参数。它对 WebBot（每步计算 280 毫秒）和本机的 3060 Ti 都不起作用，只在小模型遇上快 GPU 时才起作用。这种情况下，与其猜 MFU 和固定开销，不如直接测。

下面就是本节公式做成的计算器。预设里有三台机器人和 G.3 要讲的三种多卡跑法，还有一个 WebBot 精简版：按第 8 章 Chinchilla 的经验，让 token 数约为参数量的 20 倍，只跑 23,000 步（[8.1 节](ch08.html#s8-1)）。你可以换 GPU、换卡数、拖动 MFU，或者改 `max_iters`、batch、累积步数和 $n_{\text{mm}}$。它的计算代码与本节的验证单元、`gpu/costs.py` 逐行对应，有自动测试保证三者算出同样的数。

```widget gpu-calc caption="默认是 WebBot 单卡 H100：MFU 35% 时约 9.1 小时、36 美元。切到“× 8 卡，脚本原样”预设，对照表里原样启动的那一行要花 8 倍的钱。"
{"scenario": "webbot"}
```

```localcheck verify/test_gpu.py::test_page_cells_agree_with_costs_module title="本页验证单元里的函数、gpu/costs.py 与计算器的 JS 代码，对所有预设算出同样的数"
```

### 别信估算，信探针

估算总有误差，最可靠的办法是先跑一小段。原书的进度条（tqdm）会实时显示速度，跑上 200 步就能读出稳定的值，再乘以总步数。评估要单独计入：WebBot 一次完整评估的计算量，大约相当于 164 个训练步。

::: confusion “探针”到底指什么？
“探针”（probe）这个词在机器学习里被用得很多。读论文时最常见的，是在冻结的隐藏状态上训练一个线性分类器，看某种信息是否“编码”在里面，这叫线性探针。还有人把“不训练，先拿基础模型跑一遍看看通过率”也叫作探针。

本页的 `webbot-probe` 是第三种意思：**计时探针**。它只把 `ch09/01_pretrain.py` 里的 `range(max_iters)` 换成 `range(200)`，其余一字不改，学习率调度、随机种子都和全程一样，所以它就是真实训练的前 200 步。我们要的不是它的 loss（loss 曲线在 [9.7 节](ch09.html#s9-7)），而是它的速度。按下面的[验证 G.2-i](#v-G.2-i)，它的计算量只有全程的 0.3%；在 Modal 的 H100 上实际用了 218 秒、约 0.27 美元（运行记录 `webbot-probe`），读到的是每步 0.317 秒。

可以带走的一句话是：**计时探针 = 真实训练的开头一小段，只用来读速度。**
:::

```verify id=G.2-i title="200 步探针：计算量只有全程的 0.3%，却能定下全程的账"
probe = run_shape(wb, steps=200)
one_step = flops(wb, {"train_tokens": probe["tokens_per_step"],
                      "eval_tokens": 0})[0]
one_eval = flops(wb, {"train_tokens": 0,
                      "eval_tokens": probe["eval_tokens"]})[1]
full = run_shape(wb)
share = sum(flops(wb, probe)) / sum(flops(wb, full))
print(f"探针：{probe['steps']} 步，{probe['train_tokens']:,} token，"
      f"完整评估 {probe['evals']} 次")
print(f"一次完整评估 ≈ {one_eval / one_step:.0f} 个训练步的计算量")
print(f"探针的计算量占全程的 {share:.2%}")
assert probe["evals"] == 1 and round(one_eval / one_step) == 164
assert share < 0.005
```

读数要读中段，不要读最后一行。tqdm 收尾时打印的是全程平均：`webbot-probe` 的第一步花了 141 秒（`torch.compile` 编译，加上第 0 步的完整评估），所以最后一行显示 1.02 s/it，照它推算，全程要 28 小时。中段稳定的读数是 3.17 it/s，也就是每步约 0.32 秒。运行报告里的 `s_per_it`（0.32）也是这样算的：取后半程的用时差，除以步数差。

下面的 `timing.json` 是我们从探针的原始日志里摘出来的：第一步的耗时，和第 20 步到第 200 步的平均每步秒数（tqdm 的计时精度是 1 秒）。评估的速度直接读日志节选里 `Evaluating` 那条进度条的中段。本地测试会拿原始日志逐行核对这两份摘录。

```verify id=G.2-j title="探针实测：每步 0.317 秒、MFU 31%，全程约 10.4 小时" files=data/runs/webbot-probe/timing.json,data/runs/webbot-probe/01_pretrain.log.txt,data/runs/webbot-probe/report.json
import re

base = "data/runs/webbot-probe/"
timing = json.load(open(base + "timing.json", encoding="utf-8"))
# 第 20 步到第 200 步：57 秒 ÷ 180 步
s_step = timing["steady_s_per_step_20_200"]

# 评估：Evaluating 进度条中段（> 30 it/s，跳过开头编译的几行）的读数
with open(base + "01_pretrain.log.txt", encoding="utf-8") as f:
    log = f.read().splitlines()
EVAL_RE = re.compile(r"(\d+)/(\d+) \[[^\]]*, ([\d.]+)it/s\]")
seen = [EVAL_RE.search(ln) for ln in log if ln.startswith("Evaluating")]
steady = [m for m in seen if m and float(m.group(3)) > 30]
rate = sorted(float(m.group(3)) for m in steady)[len(steady) // 2]
n_batches = int(steady[0].group(2))
eval_s = n_batches / rate

# 第一步（编译 + 第 0 步的评估）照实计入，其余按稳定速度
rest = (full["steps"] - 1) * s_step + (full["evals"] - 1) * eval_s
wall_h = (timing["first_step_s"] + rest) / 3600
before = estimate(wb, G["h100"], h100, full, 0.35)

per_tok = 6 * wb["n_mm"] + 6 * 12 * 1024 * 768
tok_s = timing["tokens_per_step"] / s_step
peak = G["h100"]["bf16"] * 1e12
mfu_probe = tok_s * per_tok / peak
print(f"训练：每步 {s_step} 秒，{tok_s:,.0f} token/s，"
      f"MFU {mfu_probe:.1%}")
print(f"评估：{n_batches} 个 batch ÷ {rate} it/s = {eval_s:.0f} 秒")
print(f"全程：{wall_h:.2f} 小时"
      f"（事前按 MFU 35% 估 {before['wall_h']:.2f} 小时）")
usd = {}
for oid in ["runpod-community:h100", "runpod-secure:h100",
            "modal:h100", "lambda:h100"]:
    usd[oid] = wall_h * price(O[oid])
    print(f"  {usd[oid]:5.1f} 美元   {O[oid]['label']}")
assert 0.30 < mfu_probe < 0.32 and 10.2 < wall_h < 10.6
# 每步比事前估的 0.280 秒慢 13%
assert round(s_step / (before["step_ms"] / 1e3) - 1, 2) == 0.13
assert (round(min(usd.values())), round(max(usd.values()))) == (28, 45)

rep = json.load(open(base + "report.json", encoding="utf-8"))
probe_run = rep["steps"][0]
print(f"探针本身：{probe_run['seconds']:.0f} 秒，"
      f"约 {rep['cost_usd']} 美元")
assert probe_run["seconds"] < 300 and rep["cost_usd"] < 0.5
# 误读最后一行：那是全程平均，含第一步的 141 秒
last = probe_run["rate"]
naive_h = full["steps"] * float(last.removesuffix("s/it")) / 3600
print(f"照最后一行的 {last} 推算：{naive_h:.0f} 小时")
assert naive_h > 2.5 * wall_h

# 同样的速度，换一种数 FLOPs 的口径，MFU 就不同
conventions = {"本页：6·n_mm + 6LTd": per_tok,
               "PaLM、nanoGPT：6·n_mm + 12LTd":
                   per_tok + 6 * 12 * 1024 * 768,
               "6 × 全部参数": 6 * wb["params"]}
mfus = [tok_s * f / peak for f in conventions.values()]
for name, m in zip(conventions, mfus):
    print(f"  MFU {m:.0%}   {name}")
assert [round(m, 2) for m in mfus] == [0.31, 0.33, 0.38]
```

实测的 MFU 是 31%，落在事前估计的 25%–45% 之内，但低于 35%。全程因此约 10.4 小时：在 RunPod Community 上约 28 美元，在 Lambda 上约 45 美元。口径也要留意：同样的速度，按 PaLM 和 nanoGPT 的 Attention 口径算是 33%，若把全部 1.52 亿参数都算进 $6N$，就成了 38%。

上面计算器里的“用实测数字估算”（第二个折叠项）就是做这件事的：填入每步秒数和每次评估的秒数，它会按同样的步数和评估次数推出全程。

::: cs336 A1 资源核算与 TinyStories 基线
CS336 作业 1 有一组“资源核算”题：数 GPT-2 XL 形状的参数量和 FLOPs，算 AdamW 的显存，再按 H100 的 TF32 算力 495 TFLOP/s、MFU 50% 估训练时间，思路和本节完全一样。作业里 TinyStories 那个实验的配置（词表 1 万、上下文 256、$d$ 512、4 层 16 头、$d_{\text{ff}}$ 1344、3.2768 亿 token）与 StoryBot 逐项相同，handout 给的参考是“在 1 张 B200 上约 20–30 分钟”。课程网站还列过 2026-03-28 的单张 B200 时价：Modal 6.25、Lambda 6.69、RunPod 4.99 美元/小时。来源：[A1 仓库与 handout](https://github.com/stanford-cs336/assignment1-basics)、[课程网站](https://cs336.stanford.edu/)。

我们在 Modal 的 1 张 H100 上原样跑了 StoryBot：预训练 16.2 分钟，验证集 loss 1.304，达到了 A1 的 ≤ 1.45（[验证 G.2-h](#v-G.2-h)）。
:::

## G.3 多卡的陷阱：8 张卡，8 倍的账单

WebBot 单卡要 10 个小时左右。很自然的想法是：租 8 张卡，用 `torchrun` 一起跑，1 个多小时就完了。原书第 9 章的脚本也确实支持多卡（[9.3 节](ch09.html#s9-3)），只要用 `torchrun` 启动，它就会进入分布式数据并行（DDP）模式：

```listing ch09/01_pretrain.py#L18-L32
```

最后一行本身没有错：每个进程用 `42 + rank` 作随机种子，各自从训练集里随机取自己的 batch；如果种子相同，8 张卡就会把同样的数据算 8 遍。问题出在下面这几行：总步数 `max_iters` 写死成 100,000，每张卡的 batch 也写死了，都不随卡数缩小：

```listing ch09/01_pretrain.py#L107-L113
```

于是 8 张卡并不是把同一份工作分成 8 份，而是每张卡都完整地跑 100,000 步、每步都处理自己的 13 万个 token。总 token 变成单卡的 8 倍，墙钟时间却和单卡一样。

```verify id=G.3-a title="四种跑法各看了多少 token" files=data/gpu/bots.json
ways = {"单卡，原样": {},
        "8 卡，原样": {"world": 8},
        "8 卡，max_iters ÷ 8": {"world": 8, "max_iters": 12_500},
        "8 卡，每卡 batch ÷ 8": {"world": 8, "batch": 16, "accum": 1}}
shapes = {name: run_shape(wb, **kw) for name, kw in ways.items()}
for name, s in shapes.items():
    print(f"{s['steps']:>7,} 步 × {s['tokens_per_step']:>9,} = "
          f"{s['train_tokens']:>16,} token，"
          f"评估 {s['evals']:>3} 次   {name}")
one = shapes["单卡，原样"]["train_tokens"]
by_iters = shapes["8 卡，max_iters ÷ 8"]
by_batch = shapes["8 卡，每卡 batch ÷ 8"]
assert shapes["8 卡，原样"]["train_tokens"] == 8 * one
assert by_iters["train_tokens"] == by_batch["train_tokens"] == one
# max_iters ÷ 8：每步 1,024 条长 1,024 的序列，只更新 12,500 次
assert by_iters["tokens_per_step"] == 1024 * 1024
assert by_iters["steps"] == 12_500
# 每卡 batch ÷ 8：每步 128 条，和单卡一样更新 100,000 次
assert by_batch["tokens_per_step"] == 128 * 1024
assert by_batch["steps"] == 100_000
```

```verify id=G.3-b title="四种跑法的时间与费用（Modal H100，MFU 35%）"
costs = {name: estimate(wb, G["h100"], h100, s, 0.35)
         for name, s in shapes.items()}
for name, e in costs.items():
    print(f"{e['wall_h']:5.2f} 小时  {e['usd']:7.2f} 美元   {name}")
base = costs["单卡，原样"]["usd"]
# 一样久，8 倍的钱
assert abs(costs["8 卡，原样"]["usd"] / base - 8) < 1e-9
# 快 8 倍，钱几乎一样
assert abs(costs["8 卡，max_iters ÷ 8"]["usd"] / base - 1) < 0.02
# 101 次评估让 7 张卡空等
assert costs["8 卡，每卡 batch ÷ 8"]["usd"] > 1.9 * base
```

原样启动 8 张卡，花 286 美元，换来的是 8 倍的数据量，而不是 8 倍的速度。要让 8 张卡只看单卡那么多的 token，有两种改法，代价各不相同。

第一种是把 `max_iters` 除以 8。每一步 8 张卡合起来处理 104.9 万个 token，是单卡的 8 倍，步数相应地减到 12,500，总 token 不变。学习率调度是按步数写的，所以我们把 warmup 的 500 步也除以 8（取 63 步），让“学习率随 token 的变化”和单卡保持一致。代价是优化过程变了：全局 batch 变成 1,024 条序列，参数只更新 12,500 次，峰值学习率却还是 6e-4，最终的 loss 不一定和单卡一样。

第二种是把每张卡的 batch 除以 8：单卡每步是 32 × 4 = 128 条序列，8 张卡各出 16 条，步数和学习率调度都不用动。它和单卡的优化过程完全相同，但账单反而翻倍：评估只在 rank 0 上跑，而它的次数是按步数定的，仍然是 101 次，每次评估时另外 7 张卡都在空等，照样计费。

::: confusion 全局 batch 到底是多少？
DDP、梯度累积（[9.1 节](ch09.html#s9-1)）、micro-batch 叠在一起，很容易数不清一步到底用了多少数据。规则其实只有一条：**全局 batch = 每卡 micro-batch × 累积步数 × 卡数**。原书单卡是 32 × 4 × 1 = 128 条序列；原样开 8 卡，就变成 32 × 4 × 8 = 1,024 条。

为什么可以这样乘？每张卡先把 4 个 micro-batch 的梯度加起来（每个 loss 都除以 4，所以是平均），DDP 再在 8 张卡之间求平均。平均的平均，就是 1,024 条序列一起算出的梯度。下面用一个线性模型直接验证这件事。

```verify id=G.3-c title="8 卡 × 累积 4 次 × 每次 32 条 = 一次看 1,024 条"
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(8 * 4 * 32, 5))
y = rng.normal(size=8 * 4 * 32)
w0 = rng.normal(size=5)

def grad(Xb, yb):
    """均方误差 mean((Xw - y)²) 对 w 的梯度"""
    return 2 * Xb.T @ (Xb @ w0 - yb) / len(yb)

ranks = []
# 8 张卡，每张卡累积 4 个 micro-batch，每个的 loss 除以 4
for r in range(8):
    acc = np.zeros(5)
    for k in range(4):
        rows = slice((r * 4 + k) * 32, (r * 4 + k + 1) * 32)
        acc += grad(X[rows], y[rows]) / 4
    ranks.append(acc)
# all-reduce：8 张卡的梯度求平均
ddp = np.mean(ranks, axis=0)
# 一次看完全部 1,024 条
full = grad(X, y)
# 两种求和顺序只差浮点舍入
assert np.allclose(ddp, full, rtol=0, atol=1e-12)
print(f"全局 batch {len(y)} 条：DDP 的结果与整批梯度一致"
      "（逐元素差小于 1e-12）")
print("梯度：", np.array2string(full, precision=4))
```

可以带走的一句话是：**加卡等于加大 batch；步数不变，就等于加 token。**
:::

第一种改法里，warmup 用 63 步代替 500 步，学习率随 token 的变化几乎不变：

```verify id=G.3-d title="warmup 按 token 计：500 步单卡 ≈ 63 步 8 卡"
one_gpu = 500 * run_shape(wb)["tokens_per_step"]
eight = 63 * run_shape(wb, world=8)["tokens_per_step"]
print(f"单卡 500 步 = {one_gpu:,} token；8 卡 63 步 = {eight:,} token")
assert abs(eight / one_gpu - 1) < 0.01
```

这两种改法都写在 `gpu/jobs.py` 里，Modal、RunPod、Lambda 用的是同一份。它只改脚本工作区里的副本，每处替换必须恰好匹配一次，改了什么、为什么改，都会记进运行报告：

```python gpu/jobs.py
    if scale == "iters":
        if base % nproc:
            raise ValueError(f"max_iters {base} is not divisible by nproc {nproc}")
        new = {"max_iters": base // nproc, "warmup_iters": max(1, int(500 / nproc + 0.5))}
    elif scale == "batch":
        if 128 % nproc:
            raise ValueError(f"global batch 128 is not divisible by nproc {nproc}")
        seqs = 128 // nproc                      # sequences each GPU contributes per optimizer step
        micro, accum = (32, seqs // 32) if seqs >= 32 else (seqs, 1)
        new = {"max_iters": base, "micro_batch_size": micro, "accumulation_steps": accum}
```

```localcheck verify/test_gpu.py::test_ddp_patches_keep_the_token_budget title="把补丁打到真实的 ch09/01_pretrain.py 上，读回超参数，总 token 与单卡相同"
```

::: cs336 A2 分布式数据并行
CS336 作业 2 要求自己实现 DDP：先写朴素版，再写“反向传播时就开始通信”的重叠版，然后是优化器状态分片（简化的 ZeRO-1）和 FSDP。它的测试用 `mp.spawn` 在 CPU 的 gloo 后端上起 2 个进程，所以没有多卡也能做。原书直接用 PyTorch 自带的 `DistributedDataParallel`，并用 `no_sync()` 让梯度累积期间不通信。来源：[A2 仓库与 handout](https://github.com/stanford-cs336/assignment2-systems)。
:::

## G.4 本机：RTX 3060 Ti 与 Windows

先说在自己电脑上能跑什么。CodeBot 的全部四个脚本都在本机跑过（运行记录 `codebot-3060ti`）。StoryBot 的显存下界只有 0.64 GiB（[验证 G.4-a](#v-G.4-a)），预训练与 DPO 估计也能跑，不过我们还没有在本机完整跑过。WebBot 不行，原因后面讲。

### 让 PyTorch 用上 GPU

PyPI 上的 PyTorch，Windows 版只有 CPU 构建，Linux 版则自带 CUDA 13.0[^uvtorch]。原书仓库的 `uv.lock` 从 PyPI 取 torch，所以在 Windows 上照原书的方法装好以后，`torch.cuda.is_available()` 是 False。本教程的 `pyproject.toml` 加了一个只对 Windows 生效的索引，让 uv 从 PyTorch 官方的 CUDA 13.0 索引取 torch：

[^uvtorch]: [uv 的 PyTorch 集成指南](https://docs.astral.sh/uv/guides/integration/pytorch/)（2026-10 查证）。

```toml pyproject.toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'win32'" }]
```

`explicit = true` 表示这个索引只给明确指定的包用，其他包照常从 PyPI 取。装好以后，用一行命令确认：

```shell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

运行记录 `codebot-3060ti` 的主机信息里，记着这台机器的组合：PyTorch 2.14.1+cu130、CUDA 13.0、RTX 3060 Ti（计算能力 8.6）、BF16 可用。3060 Ti 是 Ampere 架构，StoryBot 的 BF16 混合精度可以原样使用。

```localcheck verify/test_gpu.py::test_windows_torch_comes_from_the_cu130_index title="pyproject 的索引配置；原书的锁文件从 PyPI 取 torch；本机装上的 torch 是 +cu130 版" needs="PyTorch"
```

### 记录一次运行

直接 `python ch03/01_pretrain.py` 当然可以，但本书的运行都通过 `gpu/run_local.py` 来跑。它先把子模块里的原书代码复制到 `runs/<运行名>/work/`，在副本里原样运行脚本，所以检查点不会写进 `third_party/dlfs6`；同时把日志、tqdm 里的 loss、生成的图和一份运行报告记下来：

```shell
uv run python gpu/run_local.py --run codebot-3060ti --job codebot      # 预训练 → 生成 → SFT → GRPO
uv run python gpu/run_local.py --run storybot-3060ti --job storybot    # 预训练 → 生成 → DPO
```

`--job` 的定义在 `gpu/jobs.py`，与 Modal 上跑的作业完全相同。StoryBot 需要先下载 1.1 GB 的训练数据：`uv run python gpu/fetch.py storybot`。下面是本机 CodeBot 预训练的 loss 曲线，直接读自这次运行的记录：

```widget runcurve caption="CodeBot 在本机 RTX 3060 Ti 上的预训练：loss 从 ln 1000 ≈ 6.9 附近开始下降。数据来自运行记录 codebot-3060ti。"
{"run": "codebot-3060ti", "step": "01_pretrain", "metrics": ["loss"], "yLabel": "loss（nats）", "refLines": [{"y": 6.9078, "label": "ln 1000"}], "command": "uv run python gpu/run_local.py --run codebot-3060ti --job codebot", "maxPoints": 500}
```

### 显存不够时，Windows 不会报错

在 Linux 上，显存用完会立刻报 `CUDA out of memory`。Windows 的 NVIDIA 驱动从 536.40 起换了做法：显存不够时，悄悄借用系统内存继续跑[^sysmem]。程序不会崩，只是慢了下来，而且没有任何报错。对训练来说，这比报错更难察觉：进度条照走，只是速度掉了，你可能要等很久才意识到出了问题。

[^sysmem]: [NVIDIA 支持文档 5490](https://nvidia.custhelp.com/app/answers/detail/a_id/5490)（2026-10 查证）。关闭这一行为的设置从驱动 546.01 开始提供。

我们宁可让它报错。打开 NVIDIA 控制面板，进入 Manage 3D settings（管理 3D 设置）的 Program Settings（程序设置）页，添加 Python 解释器，把它的 CUDA - Sysmem Fallback Policy（系统内存回退策略）设为 Prefer No Sysmem Fallback（不回退）。这里有个小坑：`.venv\Scripts\python.exe` 只是 uv 的启动器，真正占用 GPU 的是它拉起的那个解释器，要添加的是后者。它的路径用下面这行命令查：

```shell
uv run python -c "import sys; print(sys._base_executable)"
```

设置好以后，显存一超就报 OOM，我们马上知道该减 batch。

```localcheck verify/test_gpu.py::test_venv_python_is_only_a_launcher title="venv 里的 python.exe 会另起一个进程运行真正的解释器"
```

那么 WebBot 为什么放不下？不用测，算一算就知道。

```verify id=G.4-a title="WebBot 原样训练，光是参数和 logits 就超过 8 GiB" files=data/gpu/bots.json
def lower_bound_gib(b):
    # fp32 参数 4 + 梯度 4 + AdamW 的 m、v 各 4 = 每参数 16 字节
    static = 16 * b["params"] / 2**30
    # 一个 micro-batch 的 logits 元素数；bf16 每个 2 字节
    logits = b["batch"] * b["ctx"] * b["vocab"]
    return static, logits, logits * 2 / 2**30

static, logits, logits_gib = lower_bound_gib(wb)
print(f"WebBot 参数 + 梯度 + AdamW 的 m、v：{static:.2f} GiB")
print(f"一个 micro-batch 的 logits：{logits:,} 个数，"
      f"bf16 存一份 {logits_gib:.2f} GiB，反向时它的梯度还要一份")
web_gib = static + 2 * logits_gib
print(f"合计至少 {web_gib:.2f} GiB，还没算 12 层的激活")
s_static, _, s_logits = lower_bound_gib(sb)
story_gib = s_static + 2 * s_logits
print(f"同样两项，StoryBot 只有 {story_gib:.2f} GiB")
assert web_gib > 8 and round(web_gib, 2) == 8.37
assert story_gib < 1 and round(story_gib, 2) == 0.64

eff30 = G["rtx3060ti"]["bf16"] * 1e12 * 0.30
days = sum(flops(wb, run_shape(wb))) / eff30 / 86400
print(f"就算放得下：WebBot 全程在 3060 Ti 上"
      f"按 MFU 30% 也要 {days:.0f} 天")
assert 12 < days < 14
```

5 万的词表乘上 32 × 1,024 个位置，logits 本身就是一个 16 亿元素的张量。把 micro-batch 降到 4、累积步数加到 32，或许能挤进 8 GB，但计算量摆在那里，3060 Ti 要算上将近两个星期，这条路不值得走。

### Windows 的其他坑

- **第 9 章在原生 Windows 上跑不起来。** 多卡通信要用 NCCL，而 Windows 版 PyTorch 没有 NCCL。`torch.compile` 编译 CUDA 上的模型要用 Triton，Windows 版 PyTorch 不带它（本机 `import triton` 会失败）；编译 CPU 上的代码还要 MSVC 的 `cl`，本机直接报 `Compiler: cl is not found`。WebBot 请用 WSL2 或者云端的 Linux。
- **控制台编码。** Windows 控制台的默认编码往往不是 UTF-8（中文系统是代码页 936，也就是 GBK），而原书脚本会打印日文。`run_local.py` 已经替子进程设了 `PYTHONUTF8=1`；自己直接运行脚本时，也请先设上它。
- **换行符。** `.sh` 文件如果带着 Windows 的 CRLF 换行符上传到 Linux，会报 `$'\r': command not found`。本仓库的 `.gitattributes` 把所有文本文件固定为 LF。
- **Python 版本。** `.python-version` 固定为 3.12，与 Modal 镜像一致。

```localcheck verify/test_gpu.py::test_windows_cannot_run_ch09_natively title="本机确认：没有 NCCL，也没有 Triton（inductor 遇到 CUDA 设备会报 TritonMissing）；编译 CPU 代码找不到 cl" needs="PyTorch"
```

## G.5 Modal：一条命令，按秒计费

Modal 是一个“把 Python 函数搬到云端 GPU 上执行”的平台。我们在本地写一个普通的 Python 文件，用装饰器声明“这个函数要一块 H100、这个镜像、这个存储卷”，执行 `modal run`，函数就在云端的容器里跑起来，日志实时传回本地。它按秒计费，不用的时候不花钱。Starter 计划每月有 30 美元的免费额度[^modalprice]，而本页的 StoryBot 全套（预训练、生成、DPO）按容器计时约 1.17 美元，WebBot 探针约 0.27 美元（[验证 G.5-a](#v-G.5-a)）。所以手头没有 GPU 的读者，也能把这两个跑完。

[^modalprice]: [Modal 价格页](https://modal.com/pricing)（2026-10 查证）。

几个概念先说清楚。**Image** 是容器镜像，我们用 Debian 加上与本教程 `uv.lock` 相同版本的 PyTorch 等包。**Volume** 是持久存储，数据集、检查点和运行报告都放在一个名为 `zero-llm` 的 Volume 里。**Function** 是被装饰的函数，`gpu="H100:8"` 就是一台机器上的 8 张 H100。

```python gpu/modal_app.py
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")               # ch09: torch.compile -> Triton compiles kernels with gcc
    .uv_pip_install(*PINS)                        # the Linux torch wheel bundles CUDA 13.0 (Modal: driver 580, CUDA 13.0)
    .env({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "MPLBACKEND": "Agg", "WANDB_MODE": "disabled"})
    .add_local_dir(DLFS6, BOOK, ignore=_not_tracked)
    .add_local_file(HERE / "run_local.py", "/root/run_local.py")
    .add_local_file(HERE / "jobs.py", "/root/jobs.py")
    .add_local_file(HERE / "costs.py", "/root/costs.py")
    .add_local_file(ROOT / "tools" / "fetch_data.py", "/root/fetch_data.py")
)
```

原书代码只复制子模块里受 git 管理的文件；大数据集不进镜像，由 Volume 提供。`build-essential` 是给第 9 章的 `torch.compile` 准备的，原因和上一节 Windows 的那个报错一样：它要现场编译 kernel。Modal 主机的驱动是 580.95.05，CUDA 驱动 API 为 13.0[^modalcuda]；而 Linux 版 torch 2.14.1 依赖的正是 CUDA 13 的一组库（`nvidia-cudnn-cu13`、`nvidia-nccl-cu13` 等，见下面的测试），两边对得上。

[^modalcuda]: [Modal 文档：CUDA](https://modal.com/docs/guide/cuda)（2026-10 查证）。

```localcheck verify/test_gpu.py::test_modal_image_matches_the_lockfile title="镜像里钉的版本与 uv.lock 一致；Linux 版 torch 2.14.1 依赖 CUDA 13 的库与 NCCL"
```

### 四条命令

```shell
uv run --group gpu modal setup                                              # 浏览器登录，一次就好
uv run --group gpu modal run gpu/modal_app.py::prepare --bots storybot       # Hugging Face -> Volume（CPU，几美分）
uv run --group gpu modal run --detach gpu/modal_app.py::launch --job storybot --gpu H100 --run storybot-h100 --budget 4
uv run --group gpu python gpu/modal_app.py pull --run storybot-h100           # 结果 -> web/data/runs/
```

`--detach` 让任务在你关掉终端后继续运行。`launch` 在启动任何东西之前，先用 G.2 的模型打印一份估算：每个脚本多少小时、合计多少美元。估算的上限超过 `--budget`，它就拒绝启动：

```python gpu/modal_app.py
def budget_error(est: dict, budget: float) -> str | None:
    """Why `launch` refuses to start, or None. The upper end of the estimate must fit the budget."""
    if est["usd"][1] > budget:
        return (f"the estimate goes up to ${est['usd'][1]:.2f}, above --budget {budget:.2f}. "
                "Rerun with a larger --budget if you mean it (or try the job on a cheaper GPU).")
    return None
```

`--budget` 默认是 5 美元，所以误敲一条 WebBot 全程的命令也启动不了。想先看估算、不花一分钱，就运行 `uv run --group gpu python gpu/modal_app.py plan --job storybot --gpu H100`：它只打印估算，不连接 Modal。

Modal 的价格按秒标，换算成小时就能和别家比较。每张卡我们还另外申请了 2 核 CPU 和 8 GiB 内存，它们单独计费。运行报告里的费用就是“容器里计的秒数 × 这个时价”：它不含容器的冷启动、镜像构建、`prepare` 和 Volume 的存储费，所以是一个下限。

```verify id=G.5-a title="Modal 的秒价换成时价、CPU 与内存的附加费，以及两次运行记下的费用" files=data/gpu/gpus.json,data/runs/storybot-h100/report.json,data/runs/webbot-probe/report.json
for oid in ["modal:t4", "modal:l4", "modal:a100-80", "modal:h100",
            "modal:b200"]:
    per_s = O[oid]["usd_per_gpu_second"]
    print(f"{O[oid]['label']:16} {per_s:.6f} 美元/秒 = "
          f"{price(O[oid]):.2f} 美元/小时")
modal = gpus["providers"]["modal"]
cpu_mem = (2 * modal["usd_per_core_second"]
           + 8 * modal["usd_per_gib_second"])
extra = 3600 * cpu_mem
print(f"每张卡另配 2 核 CPU、8 GiB 内存：再加 {extra:.3f} 美元/小时")
assert abs(price(O["modal:b200"]) - 6.25) < 0.01
assert abs(extra - 0.158) < 0.001

spent = {}
for run in ["storybot-h100", "webbot-probe"]:
    path = f"data/runs/{run}/report.json"
    rep = json.load(open(path, encoding="utf-8"))
    bill = rep["billing"][0]
    print(f"{run}：容器计时 {bill['seconds']:.0f} 秒 × "
          f"{bill['usd_per_hour']} 美元/小时 = {rep['cost_usd']} 美元")
    # 记账用的时价 = GPU 的时价 + CPU 与内存的附加费
    assert abs(bill["usd_per_hour"] - (h100 + extra)) < 1e-3
    # 费用 = 容器计时 × 时价
    assert abs(bill["usd"] - bill["seconds"] / 3600
               * bill["usd_per_hour"]) < 1e-4
    spent[run] = rep["cost_usd"]
assert spent == {"storybot-h100": 1.17, "webbot-probe": 0.27}
# 两次加起来，不到 Starter 每月免费额度（30 美元）的 5%
assert sum(spent.values()) < 0.05 * 30
```

还有两处细节。`gpu="H100"` 有时会被免费升级到 H200，做计时对比时可以写 `"H100!"` 固定型号；`"A100"` 也可能被升级到 80 GB 版，价格按 40 GB 算[^modalgpu]。

[^modalgpu]: [Modal 文档：GPU](https://modal.com/docs/guide/gpu)（2026-10 查证）。

### 被抢占了怎么办

Modal 的文档写得很直白：所有函数默认都可能被抢占，而用来关闭抢占的 `nonpreemptible` 选项不支持 GPU 函数[^preempt]。被抢占时，Modal 会在同一份输入上重新启动函数。单次执行的超时上限是 24 小时，每次重试重新计时[^timeout]。

[^preempt]: [Modal 文档：抢占](https://modal.com/docs/guide/preemption)（2026-10 查证）。
[^timeout]: [Modal 文档：超时](https://modal.com/docs/guide/timeouts)（2026-10 查证）。

原书脚本没有断点续训，被打断就只能从头再来。我们的 `run_job` 做了两件事来减少损失。第一，一个作业里的几个脚本是分开记录的，每跑完一个就把检查点和报告存进 Volume；重启后，已经完成的脚本（同一个脚本、同样的补丁）直接跳过。第二，脚本本身出错时只记录、不抛异常，免得重试机制把同一个 bug 重跑两遍、多付两次钱。

```python gpu/modal_app.py
@app.function(gpu="L4", volumes={VOL_MOUNT: vol}, timeout=24 * 3600, single_use_containers=True,
              retries=modal.Retries(initial_delay=0.0, max_retries=2))
def run_job(plan: dict, run: str, meta: dict, force: bool = False) -> dict:
```

装饰器里的 `gpu="L4"` 只是默认值，`launch` 会用 `run_job.with_options(gpu=…)` 换成你在命令行里选的卡。

即便如此，一个 10 小时的单卡 WebBot 在第 9 小时被抢占，仍然要从头再跑。所以 WebBot 全程更适合 G.6 的整机租用，或者在 Modal 上用 8 卡把墙钟缩到一个多小时。

```localcheck verify/test_gpu.py::test_modal_run_job_with_a_fake_volume title="在本机用假的 Volume 执行 run_job 的函数体：运行、记录、续跑时跳过已完成的脚本、pull 回 web/data/runs" needs="modal"
```

### StoryBot 的 DPO：检查点的键名要改

`storybot-dpo` 作业可以不先预训练，直接从原书发布在 Hugging Face 上的 StoryBot 检查点出发。不过这份检查点有个坑：它保存时，SwiGLU 那一层叫 `mlp`，而仓库里的 `storybot/model.py` 后来把它改名成了 `ffn`。原样加载会在 `load_state_dict` 报 Missing key（[勘误 E02](errata.html#e02)）。我们的作业在把检查点放进工作区时，把键名里的 `.mlp.` 换成 `.ffn.`，并把这件事写进报告。如果同一次运行里已经跑过自己的预训练，就直接用自己的检查点，不需要改名。

```localcheck verify/test_gpu.py::test_hf_storybot_checkpoint_needs_the_key_rename title="原样加载 HF 检查点会失败；改名后加载成功，验证集 loss 远低于随机初始化的 ln 10000" needs="PyTorch"
```

::: think GRPO 只有 500 步，为什么比 2 万步的预训练还慢？
:::
::: answer
先看运行记录 `codebot-3060ti`：预训练 2 万步用了 41.9 分钟，GRPO 500 步用了 89.7 分钟。GRPO 的时间几乎都花在“生成”上，而生成是一条一条、一个 token 一个 token 地来的。`ch03/09_grpo.py` 每一步取一批题，每题采样 8 个回答。81 道题按 32、32、17 循环（见[勘误 E07](errata.html#e07)），所以 RL 里一步的 batch 是 256 条轨迹，每三步有一步只有 136 条。每 10 步还要用贪心解码把 81 道题全部答一遍（[3.9 节](ch03.html#s3-9)）。

```verify id=G.5-b title="GRPO 的 500 步里，一共要生成多少个回答" files=data/runs/codebot-3060ti/report.json
# 抄自 ch03/09_grpo.py
batch_size, group_size, max_iters = 32, 8, 500
eval_interval, n_problems = 10, 81
# DataLoader 不丢掉最后那个不满的 batch，每轮是 32、32、17；
# cycle 只迭代一次 DataLoader，之后按第一轮的顺序重放
per_epoch = [min(batch_size, n_problems - k)
             for k in range(0, n_problems, batch_size)]
sizes = (per_epoch * max_iters)[:max_iters]
sampled = sum(sizes) * group_size
greedy = math.ceil(max_iters / eval_interval) * n_problems
print(f"每轮的 batch：{per_epoch}；500 步共取 {sum(sizes):,} 道题")
print(f"训练时采样 {sampled:,} 个回答，评估时再贪心生成 {greedy:,} 个，"
      "每次只生成一条")
assert per_epoch == [32, 32, 17]
assert sampled == 108_080 and greedy == 4_050

rep = json.load(open("data/runs/codebot-3060ti/report.json",
                     encoding="utf-8"))
secs = {x["script"]: x["seconds"] for x in rep["steps"]}
pre, grpo = secs["ch03/01_pretrain.py"], secs["ch03/09_grpo.py"]
print(f"运行记录：预训练 20,000 步 {pre / 60:.1f} 分钟，"
      f"GRPO 500 步 {grpo / 60:.1f} 分钟")
assert (round(pre / 60, 1), round(grpo / 60, 1)) == (41.9, 89.7)
assert grpo > 2 * pre and round(grpo) == 5_379
```

```localcheck verify/test_gpu.py::test_grpo_constants_match_the_book title="五个常数与 ch03/09_grpo.py 一致；真实的 DataLoader 加 cycle，500 步取到 13,510 道题；generate 每次只喂一条序列、每步重算整段上下文" needs="PyTorch"
```

CodeBot 的 `generate` 一次只处理一条序列，也没有 KV cache，每生成一个 token 都要把整段上下文重算一遍。每次只算一条序列的下一个 token，GPU 再大也没有可以并行的活。换到 H100 上，每个 token 的那一串小 kernel 照样要一个个启动，所以我们估计也快不了多少（这一点没有在 H100 上测过）。GRPO 最划算的地方就是本机：在 3060 Ti 上约一个半小时（5,379 秒），不花钱。
:::

## G.6 RunPod 与 Lambda：租一整台机器

Modal 适合几十分钟的任务。WebBot 单卡全程要 10 个小时左右，这时租一整台按需（on-demand）实例更合适。RunPod 的按需 Pod 按秒计费，文档写明资源专属于你的 Pod，不会被其他用户挤占[^runpodbill]；Lambda 的实例按分钟计费，从实例通过健康检查算起，一直算到你终止为止[^lambdabill]。原书前言推荐的云服务里就有这两家。先看同一件事在三家各花多少钱，事前的估算和按探针实测的外推并排列出：

[^runpodbill]: [RunPod 文档：Pod 计费](https://docs.runpod.io/pods/pricing)（2026-10 查证）。

```verify id=G.6-a title="WebBot 全程在三家的费用：MFU 35% 的估算与按探针实测的外推（2026-10-11 的价格）" files=data/gpu/gpus.json
s1, s8 = run_shape(wb), run_shape(wb, world=8, max_iters=12_500)

def measured_h(s):
    """按探针的读数：第一步照实计入，其余的步与评估按稳定速度"""
    rest = (s["steps"] - 1) * s_step + (s["evals"] - 1) * eval_s
    return (timing["first_step_s"] + rest) / 3600

singles = ["modal:h100", "runpod-community:h100",
           "runpod-secure:h100", "lambda:h100"]
eights = ["modal:h100", "runpod-community:h100", "lambda:h100x8"]
plans = [("单卡", s1, oid) for oid in singles]
plans += [("8 卡", s8, oid) for oid in eights]
print("      MFU 35% 的估算       按探针实测外推")
rows = []
for kind, s, oid in plans:
    p = price(O[oid])
    est = estimate(wb, G["h100"], p, s, 0.35)
    m_h = measured_h(s)
    m_usd = m_h * s["world"] * p
    rows.append((kind, oid, est["usd"], m_usd))
    print(f"{kind}  {est['wall_h']:5.2f} 小时 {est['usd']:6.2f} 美元   "
          f"{m_h:5.2f} 小时 {m_usd:6.2f} 美元   {O[oid]['label']}")

def usd_range(kind, col):
    vals = [r[col] for r in rows if r[0] == kind]
    return round(min(vals)), round(max(vals))

# 两种算法下，最便宜的都是 RunPod Community 单卡
for col in (2, 3):
    assert min(rows, key=lambda r: r[col])[1] == "runpod-community:h100"
assert usd_range("单卡", 2) == (24, 39)
assert usd_range("单卡", 3) == (28, 45)
assert usd_range("8 卡", 3) == (29, 43) and 1.3 < measured_h(s8) < 1.4
assert abs(measured_h(s1) - wall_h) < 1e-9

# Modal 的 CPU 与内存另算（G.5-a）
modal_all_in = measured_h(s1) * (h100 + extra)
print(f"Modal 加上 CPU 与内存：单卡全程按实测"
      f"约 {modal_all_in:.1f} 美元")
assert round(modal_all_in, 1) == 42.7
```

RunPod 的 Community 档最便宜：它把个人和小机构的机器经过审核后租给你。Secure 档的机器放在 T3/T4 级的数据中心里，也就是 Uptime Institute 分级中的 Tier III（可以不停机检修）和 Tier IV（再加上容错），这里的 T4 和那块 T4 显卡无关[^runpodtiers]。按每张卡每小时算，Lambda 的 8 卡实例（3.99 美元）比 1 卡实例（4.29 美元）便宜。Modal 的那一行只算了 GPU：每张卡另配的 CPU 与内存再加 0.158 美元/小时（[验证 G.5-a](#v-G.5-a)），单卡全程按实测约 42.7 美元。

[^runpodtiers]: [RunPod 文档：Pod 概览](https://docs.runpod.io/pods/overview)、[Uptime Institute：Tier 分级](https://uptimeinstitute.com/tiers)（2026-10 查证）。价格来自各家的价格页（[RunPod](https://www.runpod.io/pricing)、[Lambda](https://lambda.ai/pricing)，2026-10-11 查证）。RunPod 的 Community 与 Secure 两档价格取自价格页里的结构化数据（JSON-LD）；页面卡片上显示的是另一组更高的价格，例如 H100 SXM 标 3.99 美元/小时。下单前请以控制台显示的为准。

事前估算的区间是 24–39 美元，按探针实测外推是 28–45 美元。想省钱，就选 RunPod Community 单卡：约 10.4 小时、28 美元。想当天拿到结果，就选 8 卡：按同样的速度外推约 1.3 小时、29–43 美元（这里没有算卡与卡之间的通信）。

### 从零到开跑

两家的流程大同小异：在控制台建一块持久存储，启动一台带 NVIDIA 驱动的 Ubuntu 机器并挂上它，用 SSH 登录，运行我们的 `gpu/setup_gpu_box.sh`。持久存储很重要：RunPod 的 Network Volume 挂在 `/workspace`，Pod 删了数据还在；Lambda 的 Filesystem 必须和实例在同一区域，而且只能在启动时挂上[^storage]。

[^storage]: [RunPod 文档：存储类型](https://docs.runpod.io/pods/storage/types)、[Lambda 文档：文件系统](https://docs.lambda.ai/public-cloud/filesystems/)（2026-10 查证）。

教程仓库没有放到网上也没关系。在自己电脑上把当前提交打成一个 bundle，连同脚本一起复制过去：

```shell
git bundle create zllm.bundle HEAD
scp -P <端口> zllm.bundle gpu/setup_gpu_box.sh root@<IP>:/workspace/
ssh root@<IP> -p <端口>
WORK=/workspace TUTORIAL_SRC=/workspace/zllm.bundle BOTS="webbot" bash /workspace/setup_gpu_box.sh
```

RunPod 要用控制台里 “SSH over exposed TCP” 那一行的地址，只有它支持 scp。Lambda 的登录用户是 `ubuntu`[^lambdassh]，目录换成 `/lambda/nfs/<文件系统名>`。

[^lambdassh]: [Lambda 文档：连接实例](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/)（2026-10 查证）。

这个脚本先读驱动版本。`uv.lock` 在 Linux 上锁定的 torch 2.14.1 用 CUDA 13.0 构建，要求驱动不低于 580；租来的机器驱动不一定这么新，所以驱动在 525 到 579 之间时，它改装同一版本的 CUDA 12.6 构建：

```shell gpu/setup_gpu_box.sh
if [ "$DRIVER_MAJOR" -ge 580 ]; then
  TORCH_INDEX=""
elif [ "$DRIVER_MAJOR" -ge 525 ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
  echo "driver $DRIVER_MAJOR is older than 525: pick another host" >&2
  exit 1
fi
```

接下来，它先克隆 bundle，再从 GitHub 取回原书子模块并核对提交号。然后用 `uv sync --frozen` 装好环境；驱动较旧时，再把 torch 换成 CUDA 12.6 版：

```shell gpu/setup_gpu_box.sh
  run uv pip install --reinstall-package torch "torch==2.14.1+cu126" --index "$TORCH_INDEX"
  # From now on `uv run` must not "repair" torch back to the CUDA 13 wheel.
  run sh -c "echo 'export UV_NO_SYNC=1' > '$ENV_FILE'"
```

这里有个坑。如果写成 `--index-url <CUDA 12.6 的索引> --extra-index-url <PyPI>`，uv 会让 `--extra-index-url` 排在 `--index-url` 前面，结果从 PyPI 装回的还是 CUDA 13.0 版[^uvindex]。所以我们用 `--index` 指定 PyTorch 的索引，它排在默认的 PyPI 前面；版本也钉成 `+cu126`，选错了就当场报错。写进 `zllm.env` 的 `UV_NO_SYNC=1`，则让之后的 `uv run` 不再把 torch “修”回 CUDA 13.0 版。

[^uvindex]: [uv 的命令行文档](https://docs.astral.sh/uv/reference/cli/#uv-pip-install)：`--extra-index-url` 给出的索引优先于 `--index-url`，`--index` 给出的索引优先于默认的 PyPI（2026-10 查证）。下面第二个本地验证用 `uv pip compile` 实际解析过两种写法。

最后，它用一行 Python 确认 GPU 可用，并打印是否支持 BF16，再用 `gpu/fetch.py` 只下载这台机器人要用的数据（WebBot 约 5.4 GB）。

```localcheck verify/test_gpu.py::test_setup_script_picks_torch_by_driver title="用假的 nvidia-smi 演练 setup 脚本：驱动 575 用 --index 改装 +cu126 版，驱动 610 直接用 PyPI 的 CUDA 13.0 版"
```

```localcheck verify/test_gpu.py::test_setup_script_torch_flags_resolve_to_cu126 title="把脚本里的参数交给 uv pip compile 实际解析：得到 torch==2.14.1+cu126；旧写法会从 PyPI 取回不带 +cu126 的版本" needs="联网"
```

### 在 tmux 里跑，跑完带回来

训练要跑几个小时，SSH 一断进程就没了，所以一律在 tmux 里跑。命令和本机完全一样。`--nproc 8` 会用 torchrun 启动 8 个进程，并按 G.3 的第一种改法保持总 token 不变；想要和单卡完全相同的优化过程，就再加上 `--scale batch`，代价是评估时 7 张卡空等。

```shell
tmux new -s train
cd /workspace/zero-llm-tutorial
source /workspace/zllm.env    # 旧驱动时它设了 UV_NO_SYNC=1，免得 uv run 把 torch 换回 CUDA 13 版
uv run python gpu/run_local.py --run webbot-probe-h100 --job webbot-probe     # 先测 200 步
uv run python gpu/run_local.py --run webbot-8xh100 --job webbot --nproc 8     # 8 卡全程
```

`Ctrl-b d` 离开 tmux，训练继续；`tmux attach -t train` 回来查看。跑完以后，把运行目录复制回自己的电脑，登记进索引：

```shell
scp -r -P <端口> root@<IP>:/workspace/zero-llm-tutorial/web/data/runs/webbot-8xh100 web/data/runs/
uv run python gpu/register_run.py webbot-8xh100
```

最后，**一定记得停掉或终止实例**。闲着的 GPU 照样计费，Lambda 按分钟算，从实例通过健康检查一直算到终止；不再需要的持久存储也要删掉，它们按容量另外计费。完整的步骤和排错表在仓库的 `gpu/runpod_lambda.md` 里。

[^lambdabill]: [Lambda 文档：计费](https://docs.lambda.ai/public-cloud/billing/)（2026-10 查证）。

## G.7 运行如何回到书里

不管在哪里跑，结果都汇总到同一个地方：`web/data/runs/<运行名>/`。本机和租来的机器由 `run_local.py` 直接写在这里；Modal 先写进 Volume，再由 `pull` 下载过来。构建网站时，这个目录原样复制到站点里，各章的训练曲线和上面的运行记录都只从这里读。

每次运行的目录里有这样几种文件，都由 `run_local.py`（或 Modal 上的同一套代码）生成：

- **`report.json`**：教程与原书的提交号、主机（平台、GPU、PyTorch 与 CUDA 版本）、费用，以及每个脚本的开始时间、用时、退出码、最后的指标、吞吐、改动（patches）和起点检查点的来历。
- **`<脚本名>.metrics.jsonl`**：从 tqdm 进度条解析出的指标，每行一个 `{"it": …, "loss": …}`；`it` 是 tqdm 的计数，下面细说。
- **脚本自己画的图**：文件名沿用脚本里写的，例如 `loss_pretrain.png`、`webbot_val_loss.png`。
- **`../index.json`**：登记过的运行，以及每次运行里哪些脚本有指标。

另有几份是我们从原始日志 `runs/<运行名>/<脚本名>.log` 里手工摘出来的：探针的 `timing.json`（第一步的耗时、稳定后的每步秒数）和日志节选 `01_pretrain.log.txt`，以及两个生成脚本的样例 `02_generate.output.txt`、`06_generate.output.txt`。原始日志太大，不进仓库；本地测试在日志存在时逐行核对这些摘录，并从日志重新算出 `timing.json`。

```localcheck verify/test_gpu.py::test_hand_extracts_match_the_raw_logs title="手工摘录与原始日志一致：节选与样例是日志里按顺序出现的原文，timing.json 能从日志重新算出来"
```

指标从哪来？原书的脚本不写日志文件（第 9 章的 wandb 也被我们默认关掉了），只有 tqdm 进度条上的 `loss=…`。`run_local.py` 用两个正则表达式把它们从日志里读出来。

下面拿真实日志里相邻的三行试一试。先要知道，tqdm 的计数是已经做完的步数，同一个计数会打印两次。循环变量 `i = 9` 的这一步做完，计数变成 10，tqdm 刷新一次，这时的 postfix 还是 `i = 9` 写的；接着 `i = 10` 这一步做完，`set_postfix` 写入它的值，而计数要等下一步开始才加 1，所以仍是 10。

GRPO 在 `i = 10` 这一步更新之后做了评估（`i % 10 == 0`），于是第二行的 acc 从 53.1% 跳到了 61.7%。第三行是计数变成 11 时的刷新，带的还是 `i = 10` 的值。

```verify id=G.7-a title="从 tqdm 的进度条里读出指标：正则与 run_local.py 相同，结果与运行记录一致" files=data/runs/codebot-3060ti/09_grpo.metrics.jsonl
import re

TQDM_RE = re.compile(r"(\d+)/(\d+) \[([^\]]*)\]")
POSTFIX_RE = re.compile(r"(\w+)=([-+\d.eE]+)")

def parse(line):
    m = TQDM_RE.search(line)
    values = POSTFIX_RE.findall(m.group(3))
    return {"it": int(m.group(1)), **{k: float(v) for k, v in values}}

# 运行记录 codebot-3060ti 的原始日志：计数 10 的两行，计数 11 的第一行
lines = [
    "  2%|▏         | 10/500 [04:01<3:06:14, 22.81s/it, "
    "loss=0.1076, acc=53.1%]",
    "  2%|▏         | 10/500 [04:32<3:06:14, 22.81s/it, "
    "loss=0.0910, acc=61.7%]",
    "  2%|▏         | 11/500 [04:32<3:27:55, 25.51s/it, "
    "loss=0.0910, acc=61.7%]",
]
first, second, third = (parse(line) for line in lines)
for p in (first, second, third):
    print(p)
# 百分号不是数字，正则自然停在它前面
assert first == {"it": 10, "loss": 0.1076, "acc": 53.1}
# 第二行和第三行都是 i = 10 写下的值
assert second == {"it": 10, "loss": 0.091, "acc": 61.7}
assert third == {**second, "it": 11}

path = "data/runs/codebot-3060ti/09_grpo.metrics.jsonl"
with open(path, encoding="utf-8") as f:
    rows = {r["it"]: r for r in map(json.loads, f)}
print("运行记录里的 it = 10 与 it = 11：")
print(rows[10])
print(rows[11])
# run_local.py 每个计数只留第一行
assert (rows[10], rows[11]) == (first, third)
# 开头是例外：计数 0 没有刷新的那一行，记下的就是第 1 步的值
assert rows[0] == {**rows[1], "it": 0}
```

`run_local.py` 每个计数只留第一行，所以运行记录里 it = k 的那一行，是第 k 步（`i = k − 1`）写下的读数：`i = 10` 测得的 61.7%，记在 it = 11。唯一的例外在开头：计数 0 没有刷新的那一行，记下的就是第 1 步的值，所以 it = 0 和 it = 1 两行相同。原书的准确率图以 `i` 为横轴（0、10、20……），同一次评估在那张图上落在 10。拿运行记录和原书的图对照时，要记得这一步之差。

```localcheck verify/test_gpu.py::test_parse_tqdm_keeps_the_first_line_of_each_counter title="run_local.py 每个计数只留第一行，与运行记录一致，每份记录的 it = 0 与 it = 1 相同；原书在 i 这一步更新之后评估，准确率图的横轴就是 i"
```

```localcheck verify/test_gpu.py::test_run_local_records_a_run_end_to_end title="在临时目录里用一个假的原书脚本跑通 run_local.py：报告、指标、索引，以及补丁和起点检查点的记录"
```

在章节里引用一次运行，只要一个 `runcurve` 组件，指明运行名和脚本名即可（上面 G.4 的曲线就是这样嵌进来的）。还没跑的运行不会让页面出错，组件会显示应该执行的命令。

最后是三条规矩。第一，报告由脚本生成，不手改；手工摘录另放，并由测试对照原始日志。第二，每份报告都带着两个提交号，但教程的提交号记的是开始运行时的 `HEAD`，当时没提交的改动不在里面。前两次本机运行就是这样：`codebot-3060ti` 开跑时运行器 `run_local.py` 还没有提交，`codebot-grpo-evalfix-3060ti` 开跑时它的作业定义也还没有提交。上面的运行记录给这两次标出了第一个包含那份代码的提交，重跑时检出它即可。第三，花了钱的运行都记下费用，下一个人在动手之前就知道要花多少；这个数是按容器计时乘以标价算的，是一个下限。

```localcheck verify/test_gpu.py::test_run_reports_point_at_commits_that_contain_their_code title="每份报告的教程提交里都有跑它的代码；两处例外在 plan.json 里标了真正的提交，用 git 核对它是第一个包含那份代码的提交"
```

## 小结

这一页我们把三台机器人的训练从“能跑”推进到了“知道要花多少、在哪跑最划算、跑完怎样留下证据”。估算的骨架是 $C = 6ND$ 加上 Attention 与评估两项，再除以峰值乘 MFU；但小模型在快 GPU 上会被固定开销卡住，事前的 MFU 也只是猜测，所以真正可靠的永远是 200 步的计时探针。多卡训练最容易出的错，是原样启动脚本：每张卡都跑满全部步数，账单乘以卡数。

本机的 3060 Ti 跑完了 CodeBot。StoryBot 估计也能在本机跑完，只是我们还没有完整跑过；它在 Modal 的 H100 上实测 16 分钟。WebBot 则交给 Modal 或者整机租用。无论在哪，结果都以同一种格式回到 `web/data/runs/`，各章的曲线就从这里读取。

**本页所学的内容**

- 系数 6 是前向 2 次加反向 4 次；6ND 里的 $N$ 应该用每个 token 都要乘一遍的矩阵权重 $n_{\text{mm}}$，CodeBot、StoryBot、WebBot 分别是 1,100 万、1,757 万和 1.14 亿（[验证 G.2-a](#v-G.2-a)、[验证 G.2-b](#v-G.2-b)）。
- 上下文长度是一次能看多远，训练 token 是一共看了多少：WebBot 一条序列 1,024 个 token，一步 13 万，全程 131 亿（[验证 G.2-d](#v-G.2-d)）。
- StoryBot 每 500 步完整评估一次，评估的计算量相当于训练的 45%；在 H100 上它被固定开销卡住，实测 16.2 分钟，是只按 FLOPs 估算的 4 倍多（[验证 G.2-e](#v-G.2-e)、[验证 G.2-h](#v-G.2-h)）。
- WebBot 单卡 H100 事前按 MFU 35% 估 9.1 小时、36 美元；200 步探针（约 0.27 美元）实测每步 0.317 秒、每次评估 56 秒、MFU 31%，推得全程约 10.4 小时、28–45 美元（[验证 G.2-g](#v-G.2-g)、[验证 G.2-j](#v-G.2-j)）。
- 原样用 torchrun 启动 8 卡，总 token 和费用都是单卡的 8 倍。保持总 token 不变有两种改法：`max_iters` 除以 8 最省钱，但全局 batch 变成 8 倍、更新次数变成 1/8；每卡 batch 除以 8 与单卡的优化过程完全相同，但评估让 7 张卡空等，费用翻倍（[验证 G.3-a](#v-G.3-a)、[验证 G.3-b](#v-G.3-b)）。
- Windows 驱动在显存不够时不报错，而是悄悄借用系统内存，所以要把这个回退关掉；WebBot 原样训练至少要 8.37 GiB，8 GB 的 3060 Ti 放不下（[验证 G.4-a](#v-G.4-a)）。
- GRPO 的 500 步要一条一条地生成 108,080 个回答，所以在本机比 2 万步的预训练还慢一倍多（[验证 G.5-b](#v-G.5-b)）。

有了这本账，就可以放心地去跑了。先从本机的 CodeBot 开始，跑完回到这一页，看看你的那一行是否出现在运行记录里。
