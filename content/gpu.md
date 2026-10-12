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

先把结论放在前面。下表的数字，后面都会一一算出来或者给出出处，链接指向对应的验证。

| | CodeBot | StoryBot | WebBot |
|---|---|---|---|
| 参数量（[验证 G.2-a](#v-G.2-a)） | 1,112 万 | 2,270 万 | 1.52 亿 |
| 训练 token（[验证 G.2-b](#v-G.2-b)） | 1.64 亿 | 3.28 亿 | 131 亿 |
| 计算量 FLOPs，训练 + 评估（[验证 G.2-d](#v-G.2-d)） | 1.14 × 10¹⁶ | 3.56 × 10¹⁶ + 1.60 × 10¹⁶ | 9.70 × 10¹⁸ + 1.61 × 10¹⁸ |
| 本机 RTX 3060 Ti（[验证 G.2-f](#v-G.2-f)） | 42 分钟（实测） | 约 1.8 小时（估算） | 放不下（[验证 G.4-a](#v-G.4-a)） |
| 云端参考 | 不必上云 | Modal H100 约 0.8–3.2 美元，含生成与 DPO（见下方计划） | 单卡 H100 约 9 小时、24–39 美元（[验证 G.6-a](#v-G.6-a)） |

这张表说了两件事。第一，CodeBot 和 StoryBot 完全可以在本机跑。第二，WebBot 的计算量是 StoryBot 的两百多倍，这一级必须上云，而且值得先花几毛钱做一次“探针”实测，再决定花几十美元跑全程。

真正跑过的运行，都记在下面这块“运行记录”里。它读取 `web/data/runs/` 下的报告：在哪台机器、哪块 GPU、哪个版本的 PyTorch、哪两个代码提交、每个脚本用了多久、花了多少钱。计划了但还没跑的运行，也连同命令一起列在后面。

```widget gpu-runboard caption="运行记录：每次运行都带着自己的出处。原书提交与子模块一致，说明跑的正是本书引用的那份代码。"
{}
```

## G.2 算账：从 6ND 到美元

估算训练成本，业内最常用的是一个很短的公式：

$$C \approx 6ND$$

这里，$N$ 是参数量，$D$ 是训练看过的 token 数，$C$ 是浮点运算次数（FLOPs）。系数 6 来自两处：一次乘加算 2 次运算，而反向传播的计算量大约是前向的 2 倍，于是“前向 + 反向”约为前向的 3 倍，$2 \times 3 = 6$。下面我们把 $N$、$D$ 和 $C$ 依次算出来，再换算成小时和美元。

### 参数量 N

严格地讲，6ND 里的 $N$ 应该是“每个 token 都要乘一遍的矩阵权重”，我们把它记作 $n_{\text{mm}}$。embedding 只是查表，不做乘法，不算在内；RMSNorm、LayerNorm 和 bias 的参数很少，也可以忽略。剩下的就是每一层的 Attention 与 FFN 的线性层，再加上最后把向量变回词表大小的 unembed。

以 CodeBot 为例：每层 Attention 有 $4d^2$ 个权重，GELU 的 FFN 有 $2d \cdot d_{\text{ff}}$ 个，$d = 384$、$d_{\text{ff}} = 1536$，6 层合计 $6 \times (589{,}824 + 1{,}179{,}648) = 10{,}616{,}832$；再加 unembed 的 $1000 \times 384 = 384{,}000$，得到 $n_{\text{mm}} = 11{,}000{,}832$。

下面的函数按同样的思路处理三台机器人。它读的 `bots.json` 是我们从原书脚本抄出的结构与超参数，本地测试会实例化真实模型、逐个核对。

```verify id=G.2-a title="三台机器人的参数量与每 token 的矩阵权重" files=data/gpu/bots.json
import json
import math

bots = json.load(open("data/gpu/bots.json", encoding="utf-8"))["bots"]

def param_counts(b):
    E, L, ff, V, T = b["d"], b["layers"], b["ff"], b["vocab"], b["ctx"]
    kv = b["kv_heads"] * (E // b["heads"])      # GQA：K、V 只投影到 kv_heads 个头
    attn = 2 * E * E + 2 * E * kv                # W_q、W_o，加上 W_k、W_v（都没有 bias）
    gpt2 = b["ffn"] == "gelu"                    # CodeBot 是 GPT-2 式，另外两台是现代式
    ffn = (2 if gpt2 else 3) * E * ff            # GELU 的 FFN 两个矩阵，SwiGLU 三个
    block = attn + ffn + (ff + E + 4 * E if gpt2 else 2 * E)   # 加 bias 与两个 LayerNorm，或两个 RMSNorm
    params = V * E + (T * E if b["learned_pos"] else 0) + L * block + (2 * E if gpt2 else E)
    params += 0 if b["tied"] else V * E          # unembed（CodeBot 与 embedding 共享）
    params += V if gpt2 else 0                   # CodeBot 的 unembed 有 bias
    return params, L * (attn + ffn) + V * E      # 第二项就是 n_mm

for name, b in bots.items():
    p, n = param_counts(b)
    print(f"{name:9} 全部参数 {p:>12,}   n_mm {n:>12,}")
    assert (p, n) == (b["params"], b["n_mm"])
assert param_counts(bots["codebot"]) == (11_121_640, 11_000_832)

w = bots["webbot"]
emb_share = 2 * w["vocab"] * w["d"] / w["params"]          # embedding 与 unembed 各 V × d
print(f"WebBot 的 embedding 与 unembed 占全部参数的 {emb_share:.1%}")
assert abs(emb_share - 0.504) < 0.001
```

```localcheck verify/test_gpu.py::test_param_counts_match_the_real_models title="实例化原书的三个 GPT 类，参数量与 n_mm 和上面完全一致"
```

WebBot 的 1.52 亿参数里，embedding 和 unembed 就占了一半（各 3,840 万），这是 5 万词表的代价。

### token 数 D

$D$ 由训练循环决定：每一步取多少条序列、每条多长、一共多少步。CodeBot 每步 32 条 × 256 个 token，跑 20,000 步，$D = 1.64$ 亿。WebBot 每步取 4 次 micro-batch（梯度累积），每次 32 条 × 1,024 个 token，跑 100,000 步。

评估也要算进去。StoryBot 每 500 步、WebBot 每 1,000 步，把整个验证集完整走一遍，最后一步再走一遍。下面的函数照着原书的循环条件 `i % eval_iters == 0 or i == max_iters - 1` 数评估次数，照着 `get_batch` 的窗口条件数每次评估的 token。

```verify id=G.2-b title="训练 token 数与评估次数" files=data/gpu/bots.json
def n_evals(max_iters, every, steps=None):
    """for i in range(steps): 若 i % every == 0 或 i == max_iters - 1 就评估一次"""
    if not every:
        return 0
    steps = steps or max_iters
    last = 1 if max_iters - 1 < steps and (max_iters - 1) % every else 0
    return math.ceil(steps / every) + last

def eval_windows(valid_tokens, ctx):
    """起点 0, ctx, 2·ctx, …，且 起点 + ctx + 1 < 验证集长度"""
    return max(0, math.ceil((valid_tokens - ctx - 1) / ctx))

def run_shape(b, world=1, max_iters=None, batch=None, accum=None, steps=None):
    it = max_iters or b["max_iters"]
    run = min(steps, it) if steps else it                       # steps：只跑前几步（探针）
    per_rank = (batch or b["batch"]) * (accum or b["accum"]) * b["ctx"]
    evals = n_evals(it, b["eval_every"], run)
    return {"world": world, "steps": run, "tokens_per_step_per_rank": per_rank,
            "tokens_per_step": per_rank * world, "train_tokens": per_rank * world * run,
            "evals": evals, "eval_tokens": eval_windows(b["valid_tokens"], b["ctx"]) * b["ctx"] * evals}

for name, b in bots.items():
    s = run_shape(b)
    print(f"{name:9} 训练 {s['train_tokens']:>15,} token   评估 {s['evals']:>3} 次，共 {s['eval_tokens']:>14,} token")
assert run_shape(bots["codebot"])["train_tokens"] == 20_000 * 32 * 256 == 163_840_000
assert run_shape(bots["storybot"])["evals"] == 81       # 0, 500, …, 39500，再加最后一步 39999
assert run_shape(bots["webbot"])["train_tokens"] == 100_000 * 32 * 4 * 1024
```

```localcheck verify/test_gpu.py::test_bots_json_matches_the_book_scripts title="bots.json 的每个超参数都和原书脚本里写的一样（用 ast 读脚本）"
```

StoryBot 的评估一共要过 4.4 亿个 token，比训练本身（3.3 亿）还多。这笔开销后面会再出现。

::: confusion 上下文只有 1,024，怎么一次训练就要 131 亿个 token？
这个疑问很自然：我们反复说 WebBot 的上下文长度是 1,024，而这里的 $D$ 是一百多亿，两者差了七个数量级。其实它们量的是两件事。上下文长度限制的是**单条序列**，也就是模型做一次预测时最多能往回看多远；$D$ 是整个训练过程**累计**处理过的 token。两者之间隔着两层乘法：一步有很多条序列，一次训练有很多步。

```verify id=G.2-c title="一条序列、一步、全程：三个不同的量" files=data/gpu/bots.json
w = bots["webbot"]
per_seq = w["ctx"]                               # 一条序列：一次预测最多看见的 token
per_step = w["batch"] * w["accum"] * w["ctx"]    # 一步：32 条 × 累积 4 次
per_run = per_step * w["max_iters"]              # 全程：10 万步的累计
print(f"一条序列 {per_seq:>14,} token")
print(f"一步     {per_step:>14,} token")
print(f"全程     {per_run:>14,} token")
assert per_step == 128 * per_seq and per_run == 13_107_200_000
```

和智能体对话时“上下文窗口 256K，累计却消耗了 80 万 token”是同一个道理：窗口管的是一次请求，账单记的是所有请求之和。可以带走的一句话是：**上下文长度是一次能看多远，token 数是一共看了多少。**
:::

### 计算量 C

有了 $n_{\text{mm}}$ 和 $D$，先手算一下 CodeBot：$6 \times 1{,}100 \text{ 万} \times 1.64 \text{ 亿} \approx 1.08 \times 10^{16}$。

6ND 漏掉了一项：Attention 里 $QK^\top$ 和“权重乘 $V$”这两次矩阵乘法，它们的计算量随上下文长度增长，与参数无关。按因果掩码只算一半的口径，每个 token 每层前向约 $2Td$ 次运算，训练时乘 3，得到 $6LTd$。评估只有前向，每个 token 是 $2n_{\text{mm}} + 2LTd$。于是

$$C = (6n_{\text{mm}} + 6LTd)\,D_{\text{train}} + (2n_{\text{mm}} + 2LTd)\,D_{\text{eval}}$$

这里，$L$ 是层数，$T$ 是上下文长度，$d$ 是 embedding 维度。

```verify id=G.2-d title="每个 token 的计算量与全程 FLOPs" files=data/gpu/bots.json
def flops(b, s):
    attn = b["layers"] * b["ctx"] * b["d"]
    return (6 * b["n_mm"] + 6 * attn) * s["train_tokens"], (2 * b["n_mm"] + 2 * attn) * s["eval_tokens"]

for name, b in bots.items():
    per_tok = 6 * b["n_mm"] + 6 * b["layers"] * b["ctx"] * b["d"]
    train, evals = flops(b, run_shape(b))
    print(f"{name:9} 每 token {per_tok / 1e6:6.1f} M FLOPs   训练 {train:.3e}   评估 {evals:.3e}")

cb, sb, wb = bots["codebot"], bots["storybot"], bots["webbot"]
six_nd = 6 * cb["n_mm"] * run_shape(cb)["train_tokens"]
print(f"CodeBot 只算 6ND：{six_nd:.3e}")
assert abs(six_nd / 1.08e16 - 1) < 0.01                           # 和手算一致
sb_train, sb_eval = flops(sb, run_shape(sb))
print(f"StoryBot 的评估相当于训练的 {sb_eval / sb_train:.0%}")
assert abs(sb_eval / sb_train - 0.45) < 0.01
attn_share = 6 * 12 * 1024 * 768 / (6 * wb["n_mm"] + 6 * 12 * 1024 * 768)
print(f"WebBot 每 token 的计算里，Attention 分数占 {attn_share:.1%}")
ratio = sum(flops(wb, run_shape(wb))) / sum(flops(sb, run_shape(sb)))
span = sum(flops(wb, run_shape(wb))) / sum(flops(cb, run_shape(cb)))
print(f"WebBot 的总计算量是 StoryBot 的 {ratio:.0f} 倍、CodeBot 的 {span:.0f} 倍")
assert 200 < ratio < 300 and 900 < span < 1100
```

StoryBot 每 500 步就完整评估一次，多出了 45% 的计算；WebBot 的上下文长到 1,024，Attention 那一项才占到 7.7%。这两个数都会直接反映在账单上。

### 换算成小时与美元

计算量除以 GPU 实际做到的算力，就是时间：

$$\text{小时} = \frac{C}{\text{峰值} \times \text{MFU} \times 3600}$$

这里，峰值是 GPU 数据表上的理论算力（TFLOP/s），**MFU**（Model FLOPs Utilization）是实际用上的比例。数据表上的数字都要读仔细：NVIDIA 给 H100 标的 BF16 算力是 1,979 TFLOP/s，旁边有一行小字“with sparsity”，那是结构化稀疏下的数字，普通训练要除以 2，按 989.5 计算。本页用到的每块 GPU 的峰值，都在 `web/data/gpu/gpus.json` 里写明了来源。

消费级显卡的数据表通常不写 Tensor Core 算力，我们可以从白皮书推出来。下面以本机的 RTX 3060 Ti 为例，看看 32.5 这个数从哪来。

```verify id=G.2-e title="RTX 3060 Ti 的峰值算力是怎样推出来的" files=data/gpu/gpus.json
# GA102 白皮书表 2：RTX 3080 有 68 个 SM，加速频率 1.71 GHz，BF16（FP32 累加）59.5 TFLOPS
per_sm_clk = 59.5e12 / (68 * 1.71e9)
# RTX 3060 Ti：4864 个 CUDA 核，每个 SM 128 个；加速频率 1.67 GHz
sm = 4864 // 128
bf16 = sm * 512 * 1.67e9 / 1e12
fp32 = 4864 * 2 * 1.67e9 / 1e12                 # 每个核每个时钟一次乘加 = 2 次运算
print(f"RTX 3080 每个 SM 每个时钟 {per_sm_clk:.1f} 次运算；3060 Ti 有 {sm} 个 SM")
print(f"3060 Ti 峰值：BF16 {bf16:.1f} TFLOP/s，FP32 {fp32:.1f} TFLOP/s")

gpus = json.load(open("data/gpu/gpus.json", encoding="utf-8"))
G = {g["id"]: g for g in gpus["gpus"]}
assert round(per_sm_clk) == 512 and sm == 38
assert abs(bf16 - G["rtx3060ti"]["bf16"]) < 0.05 and abs(fp32 - G["rtx3060ti"]["fp32"]) < 0.06
```

MFU 没法查表，只能估计或者实测。WebBot 的矩阵大，又开了 `torch.compile` 和 Flash Attention，我们先按 35% 估计，并取 25%–45% 作为范围，等探针跑出来再换成实测值。有了 MFU，再乘上单价，就是美元。

```verify id=G.2-f title="WebBot 单卡 H100 要多久、多少钱；CodeBot 在本机的估算与实测" files=data/gpu/gpus.json
O = {o["id"]: o for o in gpus["offers"]}

def price(offer):                                 # Modal 按秒标价，其余按小时
    return offer["usd_per_gpu_second"] * 3600 if "usd_per_gpu_second" in offer else offer["usd_per_gpu_hour"]

def estimate(b, gpu, usd_per_hour, s, mfu, floor_ms=0.0):
    peak = gpu["bf16"] if b["precision"] == "bf16" else gpu["fp32"]
    eff = peak * 1e12 * mfu                       # 实际算力，FLOP/s
    train, evals = flops(b, s)
    step = s["tokens_per_step_per_rank"] * (6 * b["n_mm"] + 6 * b["layers"] * b["ctx"] * b["d"]) / eff
    wall = s["steps"] * max(step, floor_ms / 1e3) + evals / eff      # 评估只在 rank 0 上跑
    return {"step_ms": step * 1e3, "wall_h": wall / 3600, "usd": wall * s["world"] / 3600 * usd_per_hour}

h100 = price(O["modal:h100"])
e = estimate(wb, G["h100"], h100, run_shape(wb), 0.35)
c = estimate(cb, G["rtx3060ti"], 0, run_shape(cb), 0.30)
st = estimate(sb, G["rtx3060ti"], 0, run_shape(sb), 0.25)
print(f"Modal H100：{O['modal:h100']['usd_per_gpu_second']} 美元/秒 = {h100:.2f} 美元/小时")
print(f"WebBot，单卡 H100，MFU 35%：{e['wall_h']:.2f} 小时，{e['usd']:.2f} 美元，每步计算 {e['step_ms']:.0f} 毫秒")
print(f"CodeBot，本机 3060 Ti，MFU 30%：{c['wall_h'] * 60:.0f} 分钟")
print(f"StoryBot，本机 3060 Ti，MFU 25%：{st['wall_h']:.2f} 小时")
assert abs(h100 - 3.95) < 0.01 and 9.0 < e["wall_h"] < 9.2 and 35 < e["usd"] < 37

measured_s = 2513.3                               # 运行记录 codebot-3060ti 中 ch03/01_pretrain.py 的用时
mfu = sum(flops(cb, run_shape(cb))) / measured_s / (G["rtx3060ti"]["fp32"] * 1e12)
print(f"CodeBot 实测 {measured_s / 60:.0f} 分钟，折合 MFU {mfu:.0%}")
assert 0.25 < mfu < 0.30
```

CodeBot 在本机的估算是 39 分钟，运行记录 `codebot-3060ti` 里的实测是 42 分钟，倒推出的 MFU 是 28%，和我们假设的 30% 很接近。这说明对这台机器、这个模型，30% 是个靠得住的数。换一台机器、换一个模型，就得重新量。

### 小模型遇上快 GPU

这个公式有一个盲区。StoryBot 每一步只有 8,192 个 token，放到 H100 上，纯计算只要几毫秒。可是每一步还要启动一长串 CUDA kernel、跑一遍 Python 循环、从 memmap 里切出 32 段数据，这些开销和模型大小几乎无关。计算越快，它们占的比例越大。

```verify id=G.2-g title="StoryBot 在 H100 上：每步的固定开销比计算还长"
s = run_shape(sb)
plain = estimate(sb, G["h100"], h100, s, 0.25)
floored = estimate(sb, G["h100"], h100, s, 0.25, floor_ms=15)
print(f"StoryBot 在 H100 上，每步的计算只要 {plain['step_ms']:.1f} 毫秒")
print(f"只按 FLOPs 估：{plain['wall_h'] * 60:.1f} 分钟；假设每步至少 15 毫秒：{floored['wall_h'] * 60:.1f} 分钟")
assert plain["step_ms"] < 15 and floored["wall_h"] > 3 * plain["wall_h"]
```

“每步至少 15 毫秒”是一个假设，不是测量。我们把它做成了下面计算器里的一个参数，默认 15 毫秒；它对 WebBot（每步计算 280 毫秒）和本机的 3060 Ti 都不起作用，只在小模型遇上快 GPU 时才起作用。这种情况下，与其猜 MFU 和固定开销，不如直接测。

下面就是本节公式做成的计算器。预设里有三台机器人，也有 G.3 要讲的三种多卡跑法；你可以换 GPU、换卡数、拖动 MFU，或者改 `max_iters`、batch、累积步数和 $n_{	ext{mm}}$。它的计算代码与本节的验证单元、`gpu/costs.py` 逐行对应，有自动测试保证三者算出同样的数。

```widget gpu-calc caption="同样的 token 数，换一块 GPU，时间和费用可以差上百倍；而在多卡预设里，原样启动的那一行花了 8 倍的钱。"
{"scenario": "webbot"}
```

```localcheck verify/test_gpu.py::test_page_cells_agree_with_costs_module title="本页验证单元里的函数、gpu/costs.py 与计算器的 JS 代码，对所有预设算出同样的数"
```

### 别信估算，信探针

估算总有误差，最可靠的办法是先跑一小段。原书的进度条（tqdm）会实时显示每秒的迭代数，跑上 200 步就能读出稳定的速度，再乘以总步数。评估要单独计入：WebBot 一次完整评估的计算量，大约相当于 164 个训练步。

```verify id=G.2-h title="200 步探针：计算量只有全程的 0.3%，却能定下全程的账"
probe = run_shape(wb, steps=200)
one_step = flops(wb, {"train_tokens": probe["tokens_per_step"], "eval_tokens": 0})[0]
one_eval = flops(wb, {"train_tokens": 0, "eval_tokens": probe["eval_tokens"]})[1]
full = run_shape(wb)
print(f"探针：{probe['steps']} 步，{probe['train_tokens']:,} token，完整评估 {probe['evals']} 次")
print(f"一次完整评估 ≈ {one_eval / one_step:.0f} 个训练步的计算量")
print(f"探针的计算量占全程的 {sum(flops(wb, probe)) / sum(flops(wb, full)):.2%}")
# 举例：假设探针读到 0.30 s/it，一次评估 90 秒（真实数字以运行记录为准）
eta = (full["steps"] * 0.30 + full["evals"] * 90) / 3600
print(f"若读到 0.30 s/it、评估一次 90 秒：全程 {eta:.1f} 小时，约 {eta * h100:.0f} 美元（Modal H100）")
assert probe["evals"] == 1 and round(one_eval / one_step) == 164
assert sum(flops(wb, probe)) / sum(flops(wb, full)) < 0.005
```

计算器下方的“用实测数字估算”就是做这件事的：把探针读到的秒数填进去，它会按同样的步数和评估次数推出全程。

::: cs336 A1 资源核算与 TinyStories 基线
CS336 作业 1 有一组“资源核算”题：数 GPT-2 XL 形状的参数量和 FLOPs，算 AdamW 的显存，再按 H100 的 TF32 算力 495 TFLOP/s、MFU 50% 估训练时间，思路和本节完全一样。作业里 TinyStories 那个实验的配置（词表 1 万、上下文 256、$d$ 512、4 层 16 头、$d_{\text{ff}}$ 1344、3.2768 亿 token）与 StoryBot 逐项相同，handout 给的参考是“在 1 张 B200 上约 20–30 分钟”。课程网站还列过 2026-03-28 的单张 B200 时价：Modal 6.25、Lambda 6.69、RunPod 4.99 美元/小时。来源：[A1 仓库与 handout](https://github.com/stanford-cs336/assignment1-basics)、[课程网站](https://cs336.stanford.edu/)。
:::

## G.3 多卡的陷阱：8 张卡，8 倍的账单

WebBot 单卡要 9 个小时。很自然的想法是：租 8 张卡，用 `torchrun` 一起跑，1 个多小时就完了。原书第 9 章的脚本也确实支持多卡，只要用 `torchrun` 启动，它就会进入分布式数据并行（DDP）模式：

```listing ch09/01_pretrain.py#L18-L32
```

问题藏在最后一行。每个进程用 `42 + rank` 作随机种子，各自从训练集里随机取自己的 batch；而总步数 `max_iters` 是写死的 100,000：

```listing ch09/01_pretrain.py#L107-L113
```

于是 8 张卡并不是把同一份工作分成 8 份，而是每张卡都完整地跑 100,000 步、每步都处理自己的 13 万个 token。总 token 变成单卡的 8 倍，墙钟时间却和单卡一样。

```verify id=G.3-a title="四种跑法各看了多少 token" files=data/gpu/bots.json
ways = {"单卡，原样": {}, "8 卡，原样": {"world": 8}, "8 卡，max_iters ÷ 8": {"world": 8, "max_iters": 12_500},
        "8 卡，每卡 batch ÷ 8": {"world": 8, "batch": 16, "accum": 1}}
shapes = {name: run_shape(wb, **kw) for name, kw in ways.items()}
for name, s in shapes.items():
    print(f"{s['steps']:>7,} 步 × {s['tokens_per_step']:>9,} = {s['train_tokens']:>16,} token，评估 {s['evals']:>3} 次   {name}")
one = shapes["单卡，原样"]["train_tokens"]
assert shapes["8 卡，原样"]["train_tokens"] == 8 * one
assert shapes["8 卡，max_iters ÷ 8"]["train_tokens"] == shapes["8 卡，每卡 batch ÷ 8"]["train_tokens"] == one
```

```verify id=G.3-b title="四种跑法的时间与费用（Modal H100，MFU 35%）"
costs = {name: estimate(wb, G["h100"], h100, s, 0.35) for name, s in shapes.items()}
for name, e in costs.items():
    print(f"{e['wall_h']:5.2f} 小时  {e['usd']:7.2f} 美元   {name}")
base = costs["单卡，原样"]["usd"]
assert abs(costs["8 卡，原样"]["usd"] / base - 8) < 1e-9                 # 一样久，8 倍的钱
assert abs(costs["8 卡，max_iters ÷ 8"]["usd"] / base - 1) < 0.02        # 快 8 倍，钱几乎一样
assert costs["8 卡，每卡 batch ÷ 8"]["usd"] > 1.9 * base                 # 101 次评估让 7 张卡空等
```

原样启动 8 张卡，花 286 美元，换来的是 8 倍的数据量，而不是 8 倍的速度。要让 8 张卡真正“分担”单卡的工作，有两种改法。

第一种是把 `max_iters` 除以 8。每一步 8 张卡合起来处理 104.9 万个 token，是单卡的 8 倍，步数相应地减到 12,500，总 token 不变。学习率调度是按步数写的，所以我们把 warmup 的 500 步也除以 8（取 63 步），让“学习率随 token 的变化”和单卡保持一致。

第二种是把每张卡的 batch 除以 8：单卡每步是 32 × 4 = 128 条序列，8 张卡各出 16 条，步数和学习率调度都不用动。这种改法更忠实于原书的优化过程，但账单反而翻倍：评估只在 rank 0 上跑，而它的次数是按步数定的，仍然是 101 次，每次评估时另外 7 张卡都在空等，照样计费。

::: confusion 全局 batch 到底是多少？
DDP、梯度累积、micro-batch 叠在一起，很容易数不清一步到底用了多少数据。规则其实只有一条：**全局 batch = 每卡 micro-batch × 累积步数 × 卡数**。原书单卡是 32 × 4 × 1 = 128 条序列；原样开 8 卡，就变成 32 × 4 × 8 = 1,024 条。

为什么可以这样乘？每张卡先把 4 个 micro-batch 的梯度加起来（每个 loss 都除以 4，所以是平均），DDP 再在 8 张卡之间求平均。平均的平均，就是 1,024 条序列一起算出的梯度。下面用一个线性模型直接验证这件事。

```verify id=G.3-c title="8 卡 × 累积 4 次 × 每次 32 条 = 一次看 1,024 条"
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(8 * 4 * 32, 5))
y = rng.normal(size=8 * 4 * 32)
w0 = rng.normal(size=5)

def grad(Xb, yb):                                 # 均方误差 mean((Xw - y)²) 对 w 的梯度
    return 2 * Xb.T @ (Xb @ w0 - yb) / len(yb)

ranks = []
for r in range(8):                                # 8 张卡
    acc = np.zeros(5)
    for k in range(4):                            # 每张卡累积 4 个 micro-batch，loss 除以 4
        rows = slice((r * 4 + k) * 32, (r * 4 + k + 1) * 32)
        acc += grad(X[rows], y[rows]) / 4
    ranks.append(acc)
ddp = np.mean(ranks, axis=0)                      # all-reduce：8 张卡的梯度求平均
full = grad(X, y)                                 # 一次看完全部 1,024 条
assert np.allclose(ddp, full, rtol=0, atol=1e-12)
print(f"全局 batch {len(y)} 条：DDP 的结果与整批梯度一致（逐元素差小于 1e-12）")
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

先说在自己电脑上能跑什么。CodeBot 的全部四个脚本、StoryBot 的预训练与 DPO 都可以；WebBot 不行，原因后面讲。

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

```localcheck verify/test_gpu.py::test_windows_torch_comes_from_the_cu130_index title="pyproject 的索引配置；原书的锁文件从 PyPI 取 torch；本机装上的 torch 是 +cu130 版"
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
    static = 16 * b["params"] / 2**30            # fp32 参数 4 + 梯度 4 + AdamW 的 m、v 各 4 = 每参数 16 字节
    logits = b["batch"] * b["ctx"] * b["vocab"]  # 一个 micro-batch 的 logits 元素数
    return static, logits, logits * 2 / 2**30    # bf16 每个 2 字节

static, logits, logits_gib = lower_bound_gib(wb)
print(f"WebBot 参数 + 梯度 + AdamW 的 m、v：{static:.2f} GiB")
print(f"一个 micro-batch 的 logits：{logits:,} 个数，bf16 存一份 {logits_gib:.2f} GiB，反向时它的梯度还要一份")
print(f"合计至少 {static + 2 * logits_gib:.2f} GiB，还没算 12 层的激活")
s_static, _, s_logits = lower_bound_gib(sb)
print(f"同样两项，StoryBot 只有 {s_static + 2 * s_logits:.2f} GiB")
assert static + 2 * logits_gib > 8 and s_static + 2 * s_logits < 1

days = sum(flops(wb, run_shape(wb))) / (G["rtx3060ti"]["bf16"] * 1e12 * 0.30) / 86400
print(f"就算放得下：WebBot 全程在 3060 Ti 上按 MFU 30% 也要 {days:.0f} 天")
assert days > 10
```

5 万的词表乘上 32 × 1,024 个位置，logits 本身就是一个 16 亿元素的张量。把 micro-batch 降到 4、累积步数加到 32，或许能挤进 8 GB，但计算量摆在那里，3060 Ti 要算上将近两个星期，这条路不值得走。

### Windows 的其他坑

- **第 9 章在原生 Windows 上跑不起来。** 多卡通信要用 NCCL，而 Windows 版 PyTorch 没有 NCCL；`torch.compile` 还要一个 C++ 编译器，在本机直接报 `Compiler: cl is not found`。WebBot 请用 WSL2 或者云端的 Linux。
- **控制台编码。** Windows 控制台的默认编码往往不是 UTF-8（中文系统是代码页 936，也就是 GBK），而原书脚本会打印日文。`run_local.py` 已经替子进程设了 `PYTHONUTF8=1`；自己直接运行脚本时，也请先设上它。
- **换行符。** `.sh` 文件如果带着 Windows 的 CRLF 换行符上传到 Linux，会报 `$'\r': command not found`。本仓库的 `.gitattributes` 把所有文本文件固定为 LF。
- **Python 版本。** `.python-version` 固定为 3.12，与 Modal 镜像一致。

```localcheck verify/test_gpu.py::test_windows_cannot_run_ch09_natively title="本机确认：没有 NCCL，torch.compile 找不到编译器"
```

## G.5 Modal：一条命令，按秒计费

Modal 是一个“把 Python 函数搬到云端 GPU 上执行”的平台。我们在本地写一个普通的 Python 文件，用装饰器声明“这个函数要一块 H100、这个镜像、这个存储卷”，执行 `modal run`，函数就在云端的容器里跑起来，日志实时传回本地。它按秒计费，不用的时候不花钱，Starter 计划每月还有 30 美元的免费额度。

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

`launch` 在启动任何东西之前，先用 G.2 的模型打印一份估算：每个脚本多少小时、合计多少美元。估算的上限超过 `--budget`，它就拒绝启动：

```python gpu/modal_app.py
def budget_error(est: dict, budget: float) -> str | None:
    """Why `launch` refuses to start, or None. The upper end of the estimate must fit the budget."""
    if est["usd"][1] > budget:
        return (f"the estimate goes up to ${est['usd'][1]:.2f}, above --budget {budget:.2f}. "
                "Rerun with a larger --budget if you mean it (or try the job on a cheaper GPU).")
    return None
```

`--budget` 默认是 5 美元，所以误敲一条 WebBot 全程的命令也启动不了。想先看估算、不花一分钱，就把 `launch` 换成 `python gpu/modal_app.py plan --job storybot --gpu H100`，它连 Modal 都不连接。`--detach` 让任务在你关掉终端后继续运行。

Modal 的价格按秒标，换算成小时就能和别家比较。每张卡我们还另外申请了 2 核 CPU 和 8 GiB 内存，它们单独计费：

```verify id=G.5-a title="Modal 的秒价换成时价，以及 CPU 与内存的附加费" files=data/gpu/gpus.json
for oid in ["modal:t4", "modal:l4", "modal:a100-80", "modal:h100", "modal:b200"]:
    print(f"{O[oid]['label']:16} {O[oid]['usd_per_gpu_second']:.6f} 美元/秒 = {price(O[oid]):.2f} 美元/小时")
modal = gpus["providers"]["modal"]
extra = 3600 * (2 * modal["usd_per_core_second"] + 8 * modal["usd_per_gib_second"])
print(f"每张卡另配 2 核 CPU、8 GiB 内存：再加 {extra:.3f} 美元/小时")
assert abs(price(O["modal:b200"]) - 6.25) < 0.01 and abs(extra - 0.158) < 0.001
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

即便如此，一个 9 小时的单卡 WebBot 在第 8 小时被抢占，仍然要从头再跑。所以 WebBot 全程更适合 G.6 的整机租用，或者在 Modal 上用 8 卡把墙钟缩到一个多小时。

```localcheck verify/test_gpu.py::test_modal_run_job_with_a_fake_volume title="在本机用假的 Volume 执行 run_job 的函数体：运行、记录、续跑时跳过已完成的脚本、pull 回 web/data/runs"
```

### StoryBot 的 DPO：检查点的键名要改

`storybot-dpo` 作业可以不先预训练，直接从原书发布在 Hugging Face 上的 StoryBot 检查点出发。不过这份检查点有个坑：它保存时，SwiGLU 那一层叫 `mlp`，而仓库里的 `storybot/model.py` 后来把它改名成了 `ffn`。原样加载会在 `load_state_dict` 报 Missing key。我们的作业在把检查点放进工作区时，把键名里的 `.mlp.` 换成 `.ffn.`，并把这件事写进报告。如果同一次运行里已经跑过自己的预训练，就直接用自己的检查点，不需要改名。

```localcheck verify/test_gpu.py::test_hf_storybot_checkpoint_needs_the_key_rename title="原样加载 HF 检查点会失败；改名后加载成功，验证集 loss 远低于随机初始化的 ln 10000"
```

::: confusion “探针”到底指什么？
“探针”（probe）这个词在机器学习里被用得很多。读论文时最常见的，是在冻结的隐藏状态上训练一个线性分类器，看某种信息是否“编码”在里面，这叫线性探针。还有人把“不训练，先拿基础模型跑一遍看看通过率”也叫作探针。

本页的 `webbot-probe` 是第三种意思：**计时探针**。它只把 `ch09/01_pretrain.py` 里的 `range(max_iters)` 换成 `range(200)`，其余一字不改，学习率调度、随机种子都和全程一样，所以它就是真实训练的前 200 步。我们要的不是它的 loss，而是它的速度。按 [验证 G.2-h](#v-G.2-h)，它的计算量只有全程的 0.3%，算上启动和编译，Modal H100 上也就几毛钱到一美元。

可以带走的一句话是：**计时探针 = 真实训练的开头一小段，只用来读速度。**
:::

::: think 为什么 CodeBot 的 GRPO 换到 H100 上也快不了多少？
:::
::: answer
因为它的时间几乎都花在“生成”上，而生成是一条一条、一个 token 一个 token 地来的。`ch03/09_grpo.py` 每一步给 32 道题、每题采样 8 个回答，再每 10 步用贪心解码把 81 道题全部答一遍。CodeBot 的 `generate` 一次只处理一条序列，也没有 KV cache，每生成一个 token 都要把整段上下文重算一遍。每次只算一条序列的下一个 token，GPU 再大也没有可以并行的活。

```verify id=G.5-b title="GRPO 的 500 步里，一共要生成多少个回答"
batch_size, group_size, max_iters, eval_interval, n_problems = 32, 8, 500, 10, 81   # 抄自 ch03/09_grpo.py
sampled = max_iters * batch_size * group_size
greedy = math.ceil(max_iters / eval_interval) * n_problems
print(f"训练时采样 {sampled:,} 个回答，评估时再贪心生成 {greedy:,} 个，每次只生成一条")
assert sampled == 128_000 and greedy == 4_050
```

所以 GRPO 最划算的地方就是本机：它在 3060 Ti 上要两个小时上下（准确的数字见运行记录），但不花钱。
:::

```localcheck verify/test_gpu.py::test_grpo_constants_match_the_book title="上面的五个常数与 ch03/09_grpo.py 一致"
```

## G.6 RunPod 与 Lambda：租一整台机器

Modal 适合几十分钟的任务。WebBot 单卡全程要 9 个小时，这时租一整台按需（on-demand）实例更合适。RunPod 的按需 Pod 按秒计费，文档写明资源专属于你的 Pod，不会被其他用户挤占[^runpodbill]；Lambda 的实例按分钟计费，从启动一直算到你终止为止。原书前言推荐的云服务里就有这两家。先看同一件事在三家各花多少钱：

[^runpodbill]: [RunPod 文档：Pod 计费](https://docs.runpod.io/pods/pricing)（2026-10 查证）。

```verify id=G.6-a title="WebBot 全程在三家的费用（MFU 35%，2026-10-11 的价格）" files=data/gpu/gpus.json
rows = []
for oid in ["modal:h100", "runpod-community:h100", "runpod-secure:h100", "lambda:h100"]:
    rows.append(("单卡", oid, estimate(wb, G["h100"], price(O[oid]), run_shape(wb), 0.35)))
for oid in ["modal:h100", "runpod-community:h100", "lambda:h100x8"]:
    rows.append(("8 卡", oid, estimate(wb, G["h100"], price(O[oid]), run_shape(wb, world=8, max_iters=12_500), 0.35)))
for kind, oid, e in rows:
    print(f"{kind}  {e['wall_h']:5.2f} 小时  {e['usd']:6.2f} 美元   {O[oid]['label']}（{price(O[oid]):.2f} 美元/卡·时）")
assert min(rows, key=lambda r: r[2]["usd"])[1] == "runpod-community:h100"
```

RunPod 的 Community 档最便宜：它把个人和小机构的机器经过审核后租给你，Secure 档则运行在 T3/T4 级数据中心里[^runpodtiers]。Lambda 的 8 卡实例单价比 1 卡便宜。

[^runpodtiers]: [RunPod 文档：Pod 概览](https://docs.runpod.io/pods/overview)（2026-10 查证）。价格来自各家的价格页（[RunPod](https://www.runpod.io/pricing)、[Lambda](https://lambda.ai/pricing)，2026-10-11 查证），下单前请以控制台显示的为准。

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

之后它克隆 bundle，从 GitHub 取回原书子模块并核对提交号，装好环境，在 GPU 上自检一行，再用 `gpu/fetch.py` 只下载这台机器人要用的数据（WebBot 约 5.4 GB）。

```localcheck verify/test_gpu.py::test_setup_script_picks_torch_by_driver title="用假的 nvidia-smi 演练 setup 脚本：驱动 575 选 CUDA 12.6 构建，驱动 610 用 PyPI 的 CUDA 13.0 构建"
```

### 在 tmux 里跑，跑完带回来

训练要跑几个小时，SSH 一断进程就没了，所以一律在 tmux 里跑。命令和本机完全一样，`--nproc 8` 会用 torchrun 启动 8 个进程，并按 G.3 的第一种改法保持总 token 不变：

```shell
tmux new -s train
cd /workspace/zero-llm-tutorial && source /workspace/zllm.env
uv run python gpu/run_local.py --run webbot-probe-h100 --job webbot-probe     # 先测 200 步
uv run python gpu/run_local.py --run webbot-8xh100 --job webbot --nproc 8     # 8 卡全程
```

`Ctrl-b d` 离开 tmux，训练继续；`tmux attach -t train` 回来查看。跑完以后，把运行目录复制回自己的电脑，登记进索引：

```shell
scp -r -P <端口> root@<IP>:/workspace/zero-llm-tutorial/web/data/runs/webbot-8xh100 web/data/runs/
uv run python gpu/register_run.py webbot-8xh100
```

最后，**一定记得停掉或终止实例**。闲着的 GPU 照样计费，Lambda 按分钟算，从实例通过健康检查一直算到终止[^lambdabill]；不再需要的持久存储也要删掉，它们按容量另外计费。完整的步骤和排错表在仓库的 `gpu/runpod_lambda.md` 里。

[^lambdabill]: [Lambda 文档：计费](https://docs.lambda.ai/public-cloud/billing/)（2026-10 查证）。

## G.7 运行如何回到书里

不管在哪里跑，结果都汇总到同一个地方：`web/data/runs/<运行名>/`。本机和租来的机器由 `run_local.py` 直接写在这里；Modal 先写进 Volume，再由 `pull` 下载过来。构建网站时，这个目录原样复制到站点里，各章的训练曲线和上面的运行记录都只从这里读。

| 文件 | 内容 |
|---|---|
| `report.json` | 教程与原书的提交号、主机（平台、GPU、PyTorch 与 CUDA 版本）、费用，以及每个脚本的开始时间、用时、退出码、最后的指标、吞吐、改动（patches）和起点检查点的来历 |
| `<脚本名>.metrics.jsonl` | 从 tqdm 进度条解析出的指标，每行一个 `{"it": …, "loss": …}` |
| `<脚本名>.png` | 脚本自己画的图（loss 曲线等） |
| `../index.json` | 登记过的运行，以及每次运行里哪些脚本有指标 |

指标从哪来？原书的脚本不写日志文件（第 9 章的 wandb 也被我们默认关掉了），只有 tqdm 进度条上的 `loss=…`。`run_local.py` 用两个正则表达式把它们从日志里读出来。下面直接拿真实日志里的两行试一试：

```verify id=G.7-a title="从 tqdm 的进度条里读出指标，和 run_local.py 用的是同一对正则"
import re

TQDM_RE = re.compile(r"(\d+)/(\d+) \[([^\]]*)\]")
POSTFIX_RE = re.compile(r"(\w+)=([-+\d.eE]+)")

def parse(line):
    m = TQDM_RE.search(line)
    return {"it": int(m.group(1)), **{k: float(v) for k, v in POSTFIX_RE.findall(m.group(3))}}

pretrain = parse(" 15%|█▌        | 3087/20000 [05:52<31:27,  8.96it/s, loss=1.4478]")
grpo = parse("  2%|▏         | 10/500 [02:31<2:03:20, 15.11s/it, loss=0.0213, acc=70.4%]")
print(pretrain)
print(grpo)
assert pretrain == {"it": 3087, "loss": 1.4478}
assert grpo == {"it": 10, "loss": 0.0213, "acc": 70.4}     # 百分号不是数字，正则自然停在它前面
```

```localcheck verify/test_gpu.py::test_run_local_records_a_run_end_to_end title="在临时目录里用一个假的原书脚本跑通 run_local.py：报告、指标、索引，以及补丁和起点检查点的记录"
```

在章节里引用一次运行，只要一个 `runcurve` 组件，指明运行名和脚本名即可（上面 G.4 的曲线就是这样嵌进来的）。还没跑的运行不会让页面出错，组件会显示应该执行的命令。

最后是三条规矩。报告由脚本生成，不手改；每份报告都带着两个提交号，读者可以检出同样的代码重跑；花了钱的运行都记下费用，下一个人在动手之前就知道要花多少。

## 小结

这一页我们把三台机器人的训练从“能跑”推进到了“知道要花多少、在哪跑最划算、跑完怎样留下证据”。估算的骨架是 $C = 6ND$ 加上 Attention 与评估两项，再除以峰值乘 MFU；但小模型在快 GPU 上会被固定开销卡住，所以真正可靠的永远是 200 步的计时探针。多卡训练最容易出的错，是原样启动脚本：每张卡都跑满全部步数，账单乘以卡数。

本机的 3060 Ti 足够跑完 CodeBot 和 StoryBot；WebBot 交给 Modal 或者整机租用。无论在哪，结果都以同一种格式回到 `web/data/runs/`，各章的曲线就从这里读取。

**本章所学的内容**

- 6ND 里的 $N$ 应该用每个 token 都要乘一遍的矩阵权重 $n_{\text{mm}}$；CodeBot、StoryBot、WebBot 分别是 1,100 万、1,757 万和 1.14 亿（[验证 G.2-a](#v-G.2-a)）。
- 上下文长度是一次能看多远，训练 token 是一共看了多少：WebBot 一条序列 1,024 个 token，一步 13 万，全程 131 亿（[验证 G.2-c](#v-G.2-c)）。
- StoryBot 每 500 步完整评估一次，评估的计算量相当于训练的 45%（[验证 G.2-d](#v-G.2-d)）。
- 数据表上标“with sparsity”的算力要除以 2；消费级显卡的峰值可以从白皮书推算，3060 Ti 的 BF16 是 32.5 TFLOP/s（[验证 G.2-e](#v-G.2-e)）。
- WebBot 单卡 H100 按 MFU 35% 估要 9 小时、约 36 美元；200 步探针只花全程 0.3% 的计算量，就能把这个估算换成实测（[验证 G.2-f](#v-G.2-f)、[验证 G.2-h](#v-G.2-h)）。
- 原样用 torchrun 启动 8 卡，总 token 和费用都是单卡的 8 倍；把 `max_iters` 除以 8 才是“8 张卡分担同一份工作”（[验证 G.3-a](#v-G.3-a)、[验证 G.3-b](#v-G.3-b)）。
- 全局 batch = 每卡 micro-batch × 累积步数 × 卡数；DDP 的梯度平均与一次看完整个全局 batch 完全相同（[验证 G.3-c](#v-G.3-c)）。

有了这本账，就可以放心地去跑了。先从本机的 CodeBot 开始，跑完回到这一页，看看你的那一行是否出现在运行记录里。
