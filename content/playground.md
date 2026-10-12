---
chapter: P
slug: playground
order: 160
kind: extra
title: CodeBot 工作台：一个 prompt 的一生
nav_title: CodeBot 工作台
lede: 第 I 部训练出来的 CodeBot，现在就在这个标签页里运行。输入一段文字，看它被切成 token、穿过 6 个 Transformer 块、变成 1,000 个 logits，再被抽成下一个 token；把预训练、SFT、GRPO 三个检查点换来换去，比一比同一个问题的三种回答。每一个数字都可以与 PyTorch 对拍。
pyodide_packages: [numpy]
---

第 I 部用三章做出了 CodeBot。第 1 章的分词器把代码切成 token，第 2 章的 GPT 根据前面的 token 算出下一个 token 的概率，第 3 章先预训练、再做 SFT、最后用 GRPO 强化，得到了三个检查点。可是一路走来，模型一直待在终端里：我们写下 `model(ids)`，拿到一个形状为 (1, C, 1000) 的张量，中间发生的一切都看不见。

这一页把 CodeBot 搬进浏览器，把中间的每一步摊开来看。权重是本书在一张 RTX 3060 Ti 上原样运行原书 ch03 脚本训练出来的；推理代码是一份不依赖任何框架的 JavaScript，在后台线程里运行，生成的 token 与 PyTorch 逐个相同。

路线是这样的。先看结果（P.1）。然后跟着一个 prompt 走完它的一生：权重怎样装进浏览器（P.2），文字怎样变成 token（P.3），token 怎样穿过 6 个 Transformer 块（P.4），Attention 在看哪里（P.5），1,000 个 logits 怎样变成下一个 token（P.6），一个接一个地生成时 KV cache 省下了什么、256 个位置的上限又意味着什么（P.7）。最后，我们把三个检查点放在一起比较（P.8），并亲手确认这份 JavaScript 与 PyTorch 算得一样（P.9）。

## P.1 先来看结果

下面就是工作台。它进入视野时会下载 SFT 检查点（11.3 MB），然后按图中的设置回答“read a file line by line”。你可以改动指令、切换检查点、拖动采样参数，再点“生成”。

```widget playground-bench caption="CodeBot 工作台。勾选模板时，输入会被套进 ch03 的 Alpaca 格式；生成的 token 底色越深，说明它被抽中时的概率越低。把鼠标移到某个 token 上，能看到当时的前 5 个候选。"
{"checkpoint": "sft", "prompt": "read a file line by line", "template": true, "temperature": 0.8, "topK": 40, "topP": 0.95, "maxNewTokens": 128, "seed": 1}
```

工作台做的事情，与原书的 `ch03/04_chat.py` 相同：把输入套进模板，让模型一个接一个地生成 token，遇到 `<|endoftext|>` 就停下（[3.4 节](ch03.html#s3-4)）。不同的是，这里的全部计算都在你的浏览器里进行，没有任何服务器参与。

不妨先切换到“预训练”，点一下“生成”。预训练模型只会续写，它在模板后面写起了一行行以 `#` 开头的注释。换回“SFT”或“GRPO”，模型就会照着模板作答，答完输出 `<|endoftext|>`。再试试示例里的 `3+4=`：把温度拖到 0（greedy），分别问 SFT 和 GRPO，两者的回答不一样。P.8 节会细看这件事。

那么，这三个检查点是从哪里来的呢？它们不是原书作者公开在 Hugging Face 上的官方权重，而是本书自己训练的：在本机的 RTX 3060 Ti 上，用 `gpu/run_local.py` 依次原样运行 `ch03/01_pretrain.py`、`03_sft.py` 和 `09_grpo.py`（[G.4 节](gpu.html#sG-4)）。运行报告记下了每个脚本的用时和最终的指标，导出权重时，这些记录被原样抄进了每个检查点的清单文件（`manifest.json`）。下面的验证单元把两边对一对：

```verify id=P.1-a title="三个检查点都出自本书自己的训练记录" files=data/runs/codebot-3060ti/report.json,data/playground/pretrain/manifest.json,data/playground/sft/manifest.json,data/playground/grpo/manifest.json
import json
import numpy as np

report = json.load(open("data/runs/codebot-3060ti/report.json", encoding="utf-8"))
steps = {s["script"]: s for s in report["steps"]}
for name in ["pretrain", "sft", "grpo"]:
    m = json.load(open(f"data/playground/{name}/manifest.json", encoding="utf-8"))
    st = m["provenance"]["run_step"]
    final = "，".join(f"{k} = {v}" for k, v in st["final"].items() if k != "it")
    print(f"{name:8} ← {st['script']:19} {st['seconds'] / 60:5.1f} 分钟，{final}")
    assert st["final"] == steps[st["script"]]["final"] and st["exit_code"] == 0
    assert m["provenance"]["checkpoint"] == f"runs/codebot-3060ti/work/codebot/model_{name}.pt"
assert steps["ch03/09_grpo.py"]["final"]["acc"] == 86.4
```

预训练用了 42 分钟，最终 loss 0.619；SFT 只用了 2 分钟；GRPO 最慢，将近一个半小时，进度条上最后的准确率是 86.4%。第 3 章的三条训练曲线，读的也是这次运行的记录（[3.1.5 节](ch03.html#s3-1-5)、[3.3.6 节](ch03.html#s3-3-6)、[3.9.3 节](ch03.html#s3-9-3)）。原书的官方权重同样可以装进工作台，P.2.4 节会给出命令。

## P.2 把权重装进浏览器

### P.2.1 1,112 万个参数有多重

CodeBot 有 11,121,640 个参数（[2.10 节](ch02.html#s2-10)）。用 float32 保存时，每个参数占 4 个字节，合计 44.5 MB，原书保存的 `model_sft.pt` 就是这么大。保存的内容由 `GPT.save()` 决定：

```listing codebot/model.py#L152-L163
```

除了参数本身（`state_dict`），文件里还有重建模型所需的 7 个超参数。导出工具从中读出决定形状的 6 个，`dropout_rate` 在推理时用不到。

44.5 MB 放在本地硬盘上不算什么，放进网页就太重了：三个检查点合计 133 MB。于是，我们把每个参数压成 1 个字节的整数，也就是 **int8 量化**（quantization）。先算一笔账：

```verify id=P.2-a title="参数量与字节数：float32 是 44.5 MB，int8 按行量化是 11.3 MB"
V, C, E, L, FF = 1000, 256, 384, 6, 1536            # vocab_size, max_context_len, embed_dim, n_layer, ff_dim
mats = [(V, E), (C, E)] + L * 4 * [(E, E)] + L * [(FF, E), (E, FF)]   # 2 维权重的（行, 列）
vecs = L * [E, E, E, E, FF, E] + [E, E, V]           # LayerNorm 的 γ 与 β、FFN 的偏置、最后的 LN、unembed 的偏置
n_params = sum(r * c for r, c in mats) + sum(vecs)
f32_bytes = 4 * n_params
q8_bytes = sum(r * c + 4 * r for r, c in mats) + 4 * sum(vecs)   # 每个元素 1 字节，每行再加 4 字节的 scale
print(f"参数 {n_params:,} 个")
print(f"float32：{f32_bytes:,} 字节（{f32_bytes / 1e6:.1f} MB）")
print(f"int8 按行量化：{q8_bytes:,} 字节（{q8_bytes / 1e6:.1f} MB），约为 float32 的 1/{f32_bytes / q8_bytes:.2f}")
assert n_params == 11_121_640 and q8_bytes == 11_277_120
```

这里只量化 2 维的权重矩阵：token embedding、位置 embedding、Attention 的 4 个投影矩阵和 FFN 的 2 个矩阵，它们占了参数的 99.8%。LayerNorm 的 γ、β 和各处的偏置一共只有 22,504 个数，照旧用 float32 保存。`unembed.weight` 则根本不用存：CodeBot 的输出层与 embedding 共用同一个矩阵（权重共享，[2.10.2 节](ch02.html#s2-10-2)）。

### P.2.2 按行量化成 int8

一个 int8 只能表示 −128 到 127 之间的整数，而权重是 0.0123 这样的小数。办法是给每一行配一个缩放系数 **scale**：

$$
s_r = \frac{\max_j |w_{rj}|}{127}, \qquad q_{rj} = \mathrm{round}\!\left(\frac{w_{rj}}{s_r}\right), \qquad \hat w_{rj} = q_{rj}\, s_r
$$

这里，$w_{rj}$ 是第 $r$ 行第 $j$ 列的权重，$s_r$ 是这一行的 scale（一个 float32），$q_{rj}$ 是 −127 到 127 之间的整数，只占 1 个字节。浏览器加载时，用 $\hat w_{rj}$ 把权重还原出来。四舍五入最多偏 0.5，所以还原的误差不超过半个 scale：$|\hat w_{rj} - w_{rj}| \le s_r / 2$。

手算一个例子。一行权重是 $[0.5, -1.27, 0.003, 1.0]$，最大的绝对值是 1.27，所以 $s = 1.27 / 127 = 0.01$。每个数除以 0.01 再四舍五入，得到 $[50, -127, 0, 100]$；乘回 0.01，还原成 $[0.5, -1.27, 0, 1.0]$。唯一的损失是 0.003 变成了 0，它比半个 scale（0.005）还小。

```verify id=P.2-b title="按行量化：手算的例子，以及误差不超过半个 scale"
def quantize_rows(w):
    """每行一个 scale：scale = max|w| / 127，q = round(w / scale)，限制在 -127…127。"""
    scale = (np.abs(w).max(axis=1) / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0                       # 全零的行：scale 取多少都能还原
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale

def dequantize_rows(q, scale):
    return q.astype(np.float32) * scale[:, None]

w = np.array([[0.5, -1.27, 0.003, 1.0]], dtype=np.float32)
q, s = quantize_rows(w)
print(f"scale = {s[0]:.4f}，q = {q[0].tolist()}，还原 = {[round(float(v), 4) for v in dequantize_rows(q, s)[0]]}")
assert q[0].tolist() == [50, -127, 0, 100]

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(384, 384)).astype(np.float32)
q, s = quantize_rows(W)
err = np.abs(dequantize_rows(q, s) - W)
print(f"384×384 的随机矩阵：误差最大是半个 scale 的 {(err / (s[:, None] / 2)).max():.3f} 倍")
assert np.all(err <= s[:, None] / 2 * (1 + 1e-5))
```

为什么要“每行一个 scale”，而不是整个矩阵共用一个呢？因为 scale 由最大的那个数决定。只要矩阵里有一行特别大，共用的 scale 就会被拉大，其余各行的小数值都挤到 0、±1 这几个整数上，精度就没了。按行量化把这种影响关在了那一行里：

```verify id=P.2-c title="有一行离群时，整个矩阵共用一个 scale 的误差大得多"
W2 = W.copy()
W2[7] *= 30                                       # 制造一个离群的行
others = np.arange(384) != 7
per_row = np.abs(dequantize_rows(*quantize_rows(W2)) - W2)[others].mean()
t = np.abs(W2).max() / 127                        # 整个矩阵只用一个 scale
per_tensor = np.abs(np.clip(np.rint(W2 / t), -127, 127) * t - W2)[others].mean()
print(f"其余 383 行的平均误差：每行一个 scale {per_row:.1e}，整个矩阵一个 scale {per_tensor:.1e}（{per_tensor / per_row:.0f} 倍）")
assert per_tensor > 10 * per_row
```

导出工具 `tools/export_codebot.py` 里的量化函数，与上面的 `quantize_rows` 是同一段代码（只有注释不同）。本地测试会拿两者在同一个矩阵上对比，并检查导出文件里每一行的误差都不超过半个 scale：

```localcheck verify/test_playground.py::test_quantize_cell_matches_exporter title="验证单元里的 quantize_rows 与导出工具逐位相同"
```

```localcheck verify/test_playground.py::test_exports_match_checkpoints title="三个导出文件：分片的 SHA-256、参数量、每行误差 ≤ scale/2、偏置与 LayerNorm 原样保存"
```

### P.2.3 量化改变了多少

权重变了一点点，模型的输出会变多少呢？导出时，我们用原书的 `codebot.model.GPT` 在 PyTorch 里跑了 4 个 prompt：续写 `def add(a, b):`、对话模板下的 `print hello` 和 `3+4=`，以及一段 251 个 token 的 TinyCodes 代码。每个 prompt 跑两遍，一遍用原始的 float32 权重，一遍用 int8 还原出来的权重，把最后一个位置的 logits 和 greedy 生成的 token 都记了下来（`ref.json`）：

```verify id=P.2-d title="int8 量化让 logits 最多偏移 0.29；greedy 序列有时会在中途分岔" files=data/playground/pretrain/ref.json,data/playground/sft/ref.json,data/playground/grpo/ref.json
refs = {n: json.load(open(f"data/playground/{n}/ref.json", encoding="utf-8")) for n in ["pretrain", "sft", "grpo"]}
worst = 0.0
for n, ref in refs.items():
    for p in ref["prompts"]:
        d = np.abs(np.array(p["q8"]["logits"]) - np.array(p["f32"]["logits"])).max()
        a, b = p["q8"]["greedy"], p["f32"]["greedy"]
        same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
        worst = max(worst, d)
        print(f"{n:8} {p['name']:5} max|Δlogit| = {d:.3f}，greedy 的前 {same:2d}/{len(a)} 个 token 相同")
assert worst < 0.3
```

logits 最多偏移了 0.29。对大多数位置来说，这点偏移改变不了谁排第一，所以 greedy 生成的 token 往往完全相同。可一旦遇到两个候选几乎打平的位置，第一名就可能易主，此后的序列跟着分岔：预训练模型在对话模板后面的续写，第 14 个 token 起就不一样了。两边写出来的都是像样的续写，只是不再是同一段。

所以，浏览器里的 CodeBot 是训练好的模型的一个“四舍五入版”。P.8 节会看到，在 81 道加法题上，它与原始权重的回答逐题相同。P.9 节的对拍，则是拿同一份“四舍五入”后的权重，检查 JavaScript 与 PyTorch 算得是否一样。

### P.2.4 分片、按需下载与校验

导出工具把一个检查点写成一个文件夹：`manifest.json` 记录模型的形状、每个张量的位置与类型、每个分片的 SHA-256，以及上面看到的训练记录；`weights-00.bin` 到 `weights-05.bin` 是 6 个不超过 2 MB 的分片；`ref.json` 是 PyTorch 的参考结果。工作台只在某张图第一次用到某个检查点时才下载它，6 个分片并行下载，进度显示在图里。下载完成后，先核对每个分片的大小和 SHA-256，再交给模型（页面不是通过 https 或本机打开时，浏览器不提供哈希函数，就只核对大小）。

我们自己的三个检查点，就是用下面的命令导出的。把 `--ckpt` 换成原书的官方权重（`tools/fetch_data.py` 会把它们下载到 `third_party/dlfs6/codebot/`），再换一个 `--name`，工作台的检查点开关里就会多出一项：

```shell
uv run python tools/export_codebot.py --ckpt runs/codebot-3060ti/work/codebot/model_sft.pt --name sft \
    --label SFT --template alpaca --order 1
uv run python tools/export_codebot.py --ckpt third_party/dlfs6/codebot/model_sft.pt --name official-sft \
    --label "SFT（官方）" --template alpaca --order 11 \
    --source https://huggingface.co/datasets/koki0702/zero-llm-data/blob/main/codebot/model_sft.pt
```

```localcheck verify/test_playground.py::test_exporter_handles_official_checkpoint title="同一条命令导出官方 GRPO 权重：清单、分片与参考结果都齐全"
```

## P.3 文字变成 token

权重装好了，接下来跟着一个 prompt 走。第一站是分词器。工作台用的是第 1 章训练出来的那个：`codebot/merge_rules.pkl` 被转换成了一个 JSON 文件，由 JavaScript 版的 BPE 读取，编码结果与原书的 `codebot.tokenizer` 逐个 id 相同（[1.7 节](ch01.html#s1-7)；P.9 节的测试也会拿 PyTorch 一侧记录的 id 再对一遍）。

SFT 和 GRPO 模型见过的 prompt，都套着同一个模板。原书在 `04_chat.py` 里是这样写的：

```listing ch03/04_chat.py#L17-L18
```

下面这张图把模板和你的指令一起切成 token。颜色较淡的是模板，颜色较深的是指令：

```widget playground-tokens caption="套上模板之后，模型真正读到的 token。模板本身占 18 个，3+4= 只占 4 个；### 被切成了 3 个 #。"
{"instruction": "3+4=", "template": true}
```

模板有多少个 token 呢？我们可以不借助分词器，直接从 PyTorch 记录的两个 prompt 里读出来：`print hello` 与 `3+4=` 套着同一个模板，两串 id 开头相同的部分和结尾相同的部分，就是模板本身。

```verify id=P.3-a title="模板占 18 个 token：两个 prompt 开头与结尾相同的部分" files=data/tokenizers/codebot.json
spec = json.load(open("data/tokenizers/codebot.json", encoding="utf-8"))
vocab = {i: bytes([i]) for i in range(256)}
for k in range(spec["num_merges"]):                # merges 依次记录 (a, b)：第 k 条合并出 id 256 + k
    a, b = spec["merges"][2 * k], spec["merges"][2 * k + 1]
    vocab[256 + k] = vocab[a] + vocab[b]
for s, i in spec["special_tokens"].items():
    vocab[i] = s.encode()

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

chat, math = (next(p for p in refs["sft"]["prompts"] if p["name"] == k) for k in ["chat", "math"])
assert decode(chat["ids"]) == chat["text"] and decode(math["ids"]) == math["text"]
a, b = chat["ids"], math["ids"]                   # 两个 prompt 的开头和结尾都相同，那就是模板
head = next(i for i in range(len(a)) if a[i] != b[i])
tail = next(i for i in range(1, len(a)) if a[-i] != b[-i]) - 1
print("模板开头：", [decode([i]) for i in a[:head]])
print("模板结尾：", [decode([i]) for i in a[len(a) - tail:]])
print(f"模板 {head + tail} 个 token；'print hello' 占 {len(a) - head - tail} 个，'3+4=' 占 {len(b) - head - tail} 个")
pairs = set(zip(spec["merges"][0::2], spec["merges"][1::2]))
assert head + tail == 18 and (35, 35) not in pairs          # 35 是 "#"：没有一条规则把两个 # 合并
assert spec["special_tokens"]["<|endoftext|>"] == 999
```

`###` 被切成了 3 个 `#`：743 条合并规则里，没有一条把两个 `#` 合在一起。对模型来说，模板没有任何特殊的地位，它只是 18 个普通 token。SFT 让模型学会了一个规律：在 `### Response:` 和换行之后写回答，写完输出 `<|endoftext|>`（id 999）。预训练模型没学过这个规律，所以只会把模板当成普通文本续写下去。

## P.4 穿过 6 个块

### P.4.1 残差流

token id 有了，接下来进入模型。`GPT.forward` 只有十几行：

```listing codebot/model.py#L136-L150
```

每个 token 先查表得到一个 384 维的向量，再加上它所在位置的位置 embedding（[2.5.4 节](ch02.html#s2-5-4)），然后依次穿过 6 个 Transformer 块，最后经过 LayerNorm 和 `unembed`，变成 1,000 个 logits。每个块做的事情是这样的：

```listing codebot/model.py#L96-L99
```

注意这两行的写法：`x = x + ...`。每个块都不是把 `x` 换成新的东西，而是在 `x` 上**加**一点东西。这条从 embedding 一路加到输出的向量，常被称为**残差流**（residual stream）。既然每一层都只是往上“加”，我们就可以在任意一层把它截下来，看看模型此时“心里想的”是什么。

### P.4.2 logit lens：让模型提前交卷

怎样看呢？最直接的办法是：把第 $l$ 个块之后的残差流，直接送进模型自己的最后一个 LayerNorm 和 `unembed`，就好像后面的块都不存在一样。这样得到的分布，可以理解为“如果模型在第 $l$ 层就停下，它会预测什么”。这种看法叫 **logit lens**，最早由 nostalgebraist 在 2020 年提出（[原文](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)）。第 6 层的 logit lens 就是模型的输出本身。

```widget playground-lens caption="logit lens：在 embedding 之后和每个块之后，把残差流直接送进最后的 LayerNorm 与 unembed。第 0 层的第一名总是输入的最后一个 token 自己，正确的下一个 token 要到后面几层才浮上来。"
{"checkpoint": "pretrain", "prompt": "def add(a, b):\n", "template": false}
```

以预训练模型续写 `def add(a, b):` 加换行为例。第 0 层，也就是刚做完 embedding、一个块都没经过的时候，第一名是换行符本身，概率几乎是 1。第 1 层仍是换行符，第 2 层换成 `#`，第 3 层换成制表符；到第 4 层，` return` 排到了第一，并一直保持到输出。

第 0 层为什么会“预测自己”？这正是权重共享的结果。第 0 层的残差流基本就是这个 token 自己的 embedding，`unembed` 又用同一张表给每个 token 打分，分数是两个向量的点积；在 384 维的空间里，一个向量与自己的点积，远大于它与任何别的向量的点积。如果你读过《深度学习进阶：自然语言处理》6.5.3 节，那里的 BetterRnnlm 也让输入和输出共用同一张表。在 PyTorch 记录的 24 个例子里，这件事无一例外；用随机初始化的嵌入矩阵，也能看到同样的现象：

```verify id=P.4-a title="第 0 层的 logit lens 预测的是“当前 token 自己”"
hits = total = 0
for n, ref in refs.items():
    for p in ref["prompts"]:
        for v in ["f32", "q8"]:
            lens = p[v]["lens"]                   # 第 0 层（只有 embedding）到第 6 层，各自的前 5 名
            total += 1
            hits += lens[0]["ids"][0] == p["ids"][-1] and lens[0]["probs"][0] > 0.999
            assert lens[-1]["ids"][0] == p[v]["greedy"][0]    # 最后一层的 lens 就是模型的输出
print(f"第 0 层的第一名就是输入的最后一个 token，且概率 > 0.999：{hits}/{total}")
assert hits == total == 24

rng = np.random.default_rng(1)
emb = rng.normal(0, 0.02, size=(1000, 384))       # 与 CodeBot 的初始化相同：N(0, 0.02²)
pos = rng.normal(0, 0.02, size=384)
ln = lambda x: (x - x.mean()) / np.sqrt(x.var() + 1e-5)
self_top = np.mean([np.argmax(emb @ ln(emb[t] + pos)) == t for t in range(1000)])
print(f"随机初始化的嵌入矩阵：{self_top:.1%} 的 token 在第 0 层把自己排在第一")
assert self_top > 0.99
```

换句话说，第 0 层的“预测”只是把输入原样照了一遍镜子。6 个块要做的事，就是把这个向量一步步推向“下一个 token”的方向。

```localcheck verify/test_playground.py::test_reference_rederived_with_torch title="用 PyTorch 重算 ref.json：logits、greedy 生成、logit lens 与注意力权重"
```

## P.5 Attention 在看哪里

每个块里，真正让不同位置互相“交流”的是 Attention。原书的实现里，算权重的是这几行：

```listing codebot/model.py#L33-L41
```

`weights` 的形状是（batch, 头, 查询位置, 被看的位置），每一行是一个 softmax 分布，因果掩码让每个位置只能看见它自己和前面的 token（[2.6 节](ch02.html#s2-6)）。CodeBot 有 6 层、每层 6 个头（[2.8 节](ch02.html#s2-8)），所以对一段输入，一共有 36 张这样的图：

```widget playground-attn caption="当前输入在 6 层 × 6 个头里的注意力权重。点缩略图可以放大。第 1 层第 1 个头的权重集中在紧贴对角线下方的那条斜线上：每个 token 大约一半的注意力给了前一个 token。"
{"checkpoint": "pretrain", "prompt": "def add(a, b):\n    return a + b", "template": false, "layer": 1, "head": 1}
```

不同的头看的地方很不一样。有的头把注意力分散在很多位置上，有的头盯着某几个关键的 token。第 1 层第 1 个头最好认：它的权重集中在对角线正下方那一条，也就是说，每个 token 都把相当大的一部分注意力给了紧挨着它的前一个 token。这种头常被称为“前一个 token 头”（previous-token head）。用 PyTorch 记录的权重确认一下：

```verify id=P.5-a title="注意力权重：每行和为 1，对角线以上全是 0；第 1 层第 1 个头偏爱前一个 token"
code = next(p for p in refs["pretrain"]["prompts"] if p["name"] == "code")
att = np.array(code["q8"]["attention"])           # （层, 头, 查询位置, 被看的位置）
L_, H_, n, _ = att.shape
print(f"形状 {att.shape}；每行之和与 1 最多差 {np.abs(att.sum(-1) - 1).max():.0e}；对角线以上最大 {np.triu(att, 1).max()}")
assert np.allclose(att.sum(-1), 1, atol=1e-5) and np.all(np.triu(att, 1) == 0)
t = np.arange(1, n)
prev = att[:, :, t, t - 1].mean(-1)               # 每个头：平均有多少注意力给了前一个 token
uniform = np.mean(1 / (t + 1))                    # 平均分配时的期望值
print(f"给前一个 token 的平均权重：均匀分配 {uniform:.3f}，第 1 层第 1 个头 {prev[0, 0]:.3f}")
assert prev[0, 0] > 2 * uniform
```

8 个 token 的例子还太短，均匀分配也有 0.245。换成 251 个 token 的长 prompt，均匀分配只剩 0.02，而这个头仍然把 46% 到 48% 的注意力给了前一个 token，三个检查点都是如此：

```localcheck verify/test_playground.py::test_previous_token_head title="251 个 token 的长 prompt：第 1 层第 1 个头给前一个 token 的平均权重超过 0.4，均匀分配只有 0.02"
```

## P.6 从 logits 到下一个 token

6 个块之后，最后一个位置的向量经过 LayerNorm 和 `unembed`，变成 1,000 个 logits。从这里到“下一个 token”，还要经过第 3 章讲过的几道工序（[3.2 节](ch03.html#s3-2)）：logits 除以温度 $T$ 后做 softmax，

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

这里，$z_i$ 是第 $i$ 个 token 的 logit。然后 Top-k 只留概率最大的 $k$ 个，Top-p 再留下累计概率首次达到 $p$ 的最少几个，每截一次都重新归一化，最后按剩下的分布抽一个。温度为 0 时不抽样，直接取 argmax（greedy）。

```widget playground-next caption="预训练模型在 for i in 之后的下一个 token。range 一个就占了大半；调低温度，分布变尖，调高温度，长尾被抬起来。灰色的候选是被 Top-k / Top-p 截掉的。"
{"checkpoint": "pretrain", "prompt": "for i in", "template": false, "temperature": 1, "topK": 0, "topP": 1, "seed": 0}
```

```localcheck verify/test_playground.py::test_for_i_in_prefers_range title="预训练模型在 for i in 之后：range 排第一，概率超过一半（float32 与 int8 权重都是）"
```

工作台按“温度 → softmax → Top-k → Top-p”的顺序处理。顺序是有影响的：Top-p 量的是 Top-k 剩下的那几个 token 的累计概率。用第 3 章的手算例子（logits 为 $[2, 1, 0.5, 0, -1]$）来看：单独用 Top-p = 0.7，要留 2 个 token（0.563 还不够，加上 0.207 才超过 0.7）；先用 Top-k = 2，剩下的两个重新归一化成 0.731 和 0.269，第一个就已经超过 0.7，Top-p 之后只剩 1 个。

```verify id=P.6-a title="温度、Top-k、Top-p 的顺序：先 Top-k，Top-p 量的是剩下的部分"
def softmax(x, T=1.0):
    z = np.asarray(x, dtype=np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()

def filter_dist(logits, T=1.0, top_k=0, top_p=1.0):
    """工作台的顺序：温度 → softmax → Top-k → 重新归一化 → Top-p → 重新归一化。"""
    if T == 0:                                    # greedy：概率全给 argmax
        return np.eye(len(logits))[int(np.argmax(logits))]
    p = softmax(logits, T)
    keep = np.argsort(-p, kind="stable")
    if 0 < top_k < len(p):
        keep = keep[:top_k]
    if top_p < 1:
        cum = np.cumsum(p[keep]) / p[keep].sum()
        keep = keep[: int(np.searchsorted(cum, top_p)) + 1]   # 累计概率首次达到 top_p 的最少 token
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum()

logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
for kw in [{}, {"top_k": 2}, {"top_p": 0.9}, {"top_p": 0.7}, {"top_k": 2, "top_p": 0.7}, {"T": 0}]:
    print(f"{str(kw):27} → {np.round(filter_dist(logits, **kw), 3).tolist()}")
assert np.count_nonzero(filter_dist(logits, top_p=0.9)) == 4
assert np.count_nonzero(filter_dist(logits, top_p=0.7)) == 2
assert np.count_nonzero(filter_dist(logits, top_k=2, top_p=0.7)) == 1   # 先 Top-k，Top-p 量的是剩下的 2 个
```

工作台的抽样用了一个带种子的随机数生成器，同样的种子、同样的设置，抽出的 token 序列总是相同的。原书的 `generate()` 用的是 `torch.multinomial`，没有固定种子，所以随机抽样的结果无法与 PyTorch 逐个对比；P.9 节对拍的是不涉及随机性的 greedy 生成。

图中还显示了分布的“熵”。熵衡量分布有多分散：所有概率集中在一个 token 上时，熵是 0；1,000 个 token 平分概率时，熵最大。

::: confusion 熵的单位“nats”是什么？
看到“熵 2.31 nats”这样的说法，很自然会问：nats 是什么单位，和常听说的 bits 有什么关系？其实两者只差在对数的底。用自然对数 ln 算出来的熵，单位叫 nats；用以 2 为底的对数算，单位叫 bits。两者之间只差一个常数倍：1 nat = 1/ln 2 ≈ 1.443 bits。

PyTorch 的 `cross_entropy` 用的是自然对数，所以第 3 章训练曲线上的 loss，单位也是 nats。CodeBot 的词表有 1,000 个 token，完全瞎猜时，每个 token 的熵是 ln 1000 = 6.908 nats，也就是 log₂ 1000 ≈ 9.97 bits，差不多要 10 个“是或否”的问题才能猜中。
:::

```verify id=P.6-b title="熵的单位：ln 得 nats，log₂ 得 bits；温度越高，熵越大"
def entropy(q):
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())          # 自然对数：单位是 nats

H_max = np.log(1000)                              # 1,000 个 token 均匀分布：熵的上限
print(f"均匀分布：{H_max:.3f} nats = {H_max / np.log(2):.2f} bits")
for T in [0.5, 1.0, 2.0]:
    print(f"温度 {T}：{entropy(filter_dist(logits, T=T)):.3f} nats")
assert entropy(filter_dist(logits, T=0.5)) < entropy(filter_dist(logits)) < entropy(filter_dist(logits, T=2.0))
assert abs(H_max / np.log(2) - np.log2(1000)) < 1e-12 and entropy(filter_dist(logits, T=0)) == 0
```

## P.7 一个接一个：KV cache 与 256 的墙

### P.7.1 原书的生成循环

抽出一个 token 之后，把它接到序列末尾，再预测下一个。原书 `codebot/utils.py` 的循环是这样的：

```listing codebot/utils.py#L18-L37
```

每一轮，`model(ids)` 都把整个序列从头算一遍，只为了取最后一个位置的 logits。前面那些位置的 K 和 V，上一轮明明已经算过了。第 5 章为 StoryBot 引入的 **KV cache**，就是把它们存起来，每一轮只算新来的那一个 token（[5.5 节](ch05.html#s5-5)）。

工作台也是这样做的：先用一次 **prefill** 把整个 prompt 算完，之后每生成一个 token，只做一步 **decode**。这样做必须保证一件事：结果与从头重算完全相同。下面用一个 2 层的迷你 GPT 来确认。它保留了 CodeBot 结构上的要点：可学习的位置 embedding、先 LayerNorm 再 Attention、输出层与嵌入表共享权重；简单起见，省去了偏置和 LayerNorm 的参数。

```verify id=P.7-a title="KV cache 与从头重算的 logits 相同；超出位置 embedding 的行数就无法继续"
rng = np.random.default_rng(0)
V_, T_, E_, H_, L_ = 24, 8, 16, 2, 2               # 迷你版：词表 24，最多 8 个位置
D_ = E_ // H_
Pm = {"tok": rng.normal(0, 0.3, (V_, E_)), "pos": rng.normal(0, 0.3, (T_, E_))}
for k in ["Wq", "Wk", "Wv", "Wo"]:
    Pm[k] = rng.normal(0, 0.3, (L_, E_, E_))
Pm["W1"], Pm["W2"] = rng.normal(0, 0.3, (L_, E_, 4 * E_)), rng.normal(0, 0.3, (L_, 4 * E_, E_))
norm = lambda x: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
gelu = lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def attend(q, K, Vs):                             # 一个查询位置看 K、V 的全部行
    s = np.einsum("hd,thd->ht", q.reshape(H_, D_), K.reshape(-1, H_, D_)) / np.sqrt(D_)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return np.einsum("ht,thd->hd", w, Vs.reshape(-1, H_, D_)).reshape(E_)

def full(ids):                                    # 不用 cache：每次从头算整个序列（原书 generate 的做法）
    x = Pm["tok"][ids] + Pm["pos"][: len(ids)]
    for l in range(L_):
        h = norm(x)
        Q, K, Vs = h @ Pm["Wq"][l], h @ Pm["Wk"][l], h @ Pm["Wv"][l]
        x = x + np.stack([attend(Q[t], K[: t + 1], Vs[: t + 1]) for t in range(len(ids))]) @ Pm["Wo"][l]
        x = x + gelu(norm(x) @ Pm["W1"][l]) @ Pm["W2"][l]
    return norm(x[-1]) @ Pm["tok"].T              # 权重共享：unembed 用的是同一张嵌入表

def step(i, t, cache):                            # 有 cache：只算位置 t 的这一个 token
    x = Pm["tok"][i] + Pm["pos"][t]
    for l in range(L_):
        h = norm(x)
        cache[l]["K"].append(h @ Pm["Wk"][l])
        cache[l]["V"].append(h @ Pm["Wv"][l])
        x = x + attend(h @ Pm["Wq"][l], np.array(cache[l]["K"]), np.array(cache[l]["V"])) @ Pm["Wo"][l]
        x = x + gelu(norm(x) @ Pm["W1"][l]) @ Pm["W2"][l]
    return norm(x) @ Pm["tok"].T

ids = [3, 1, 4, 1, 5, 9, 2, 6]
cache = [{"K": [], "V": []} for _ in range(L_)]
diff = max(np.abs(step(i, t, cache) - full(ids[: t + 1])).max() for t, i in enumerate(ids))
assert diff < 1e-12                               # 只差浮点舍入（具体数值随 NumPy 的矩阵库而变）
print("8 个位置：有 cache 与从头重算，logits 之差都小于 1e-12")
try:
    step(7, T_, cache)                            # 第 9 个位置
except IndexError as e:
    print(f"位置 {T_} 没有位置 embedding：{e}")
```

工作台的 JavaScript 走得更远一步：prefill 用的矩阵乘法和 decode 用的矩阵向量乘法，按完全相同的顺序累加，所以两条路算出的 logits 逐位相同，不只是“差别很小”。因此，用不用 KV cache，生成的 token 都一模一样（`tests/js/playground-gpt.test.mjs` 在合成的小模型上、`playground-parity.test.mjs` 在三个真实检查点上检查了这一点）。

### P.7.2 256 的墙

可是，KV cache 在 CodeBot 身上有一个限度。CodeBot 的位置 embedding 只有 256 行，第 257 个位置没有向量可查。原书的循环在这里把序列从左边截掉，只保留最后 256 个 token（[3.2.4 节](ch03.html#s3-2-4)）。截掉一个 token 之后，剩下的每个 token 都往前挪了一个位置：原来在位置 1 的 token，现在在位置 0。而缓存里的 K、V 是按旧的位置算出来的，全都作废了。所以，序列一旦超过 256 个 token，每生成一个 token，都只能把最近的 256 个 token 从头重算一遍。

下面的图分两步测一测。先测“生成 1 个新 token 要多久”：有 KV cache 时只算 1 个位置，没有时要把上下文从头算一遍。再从一段 251 个 token 的代码开始生成 12 个 token，看看越过 256 之后发生了什么。

```widget playground-kv caption="先测生成 1 个新 token 要多久（对数刻度）：没有 KV cache 时，时间大致随上下文长度成正比地增长；有 KV cache 时只是缓慢增加。再从 251 个 token 开始生成 12 个：从第 7 个新 token 起，每一个都要重算整个 256 窗口。"
{"checkpoint": "pretrain", "lengths": [8, 16, 32, 64, 128, 192, 256], "cross": 12}
```

计时取决于你的设备，但规律是确定的。我们只数“前向传播一共算了多少个位置”，这个数与设备无关：

```verify id=P.7-b title="一共要算多少个位置：KV cache 省下的，以及越过 256 之后失去的"
def positions_computed(n_prompt, n_new, T=256, cache=True):
    """生成 n_new 个 token，前向传播一共要算多少个位置。"""
    total = min(n_prompt, T)                      # 第 1 个新 token：prefill 整个 prompt（最多 T 个）
    for seq in range(n_prompt + 1, n_prompt + n_new):   # 生成后面每一个 token 时，序列已有 seq 个
        total += 1 if cache and seq <= T else min(seq, T)
    return total

for n_prompt, n_new in [(22, 100), (251, 12)]:
    a, b = positions_computed(n_prompt, n_new), positions_computed(n_prompt, n_new, cache=False)
    print(f"prompt {n_prompt:3d} 个、生成 {n_new:3d} 个：有 KV cache 算 {a:,} 个位置，原书的写法算 {b:,} 个")
assert positions_computed(22, 100) == 121 and positions_computed(22, 100, cache=False) == 7150
assert positions_computed(251, 12) == 1792 and positions_computed(251, 12, cache=False) == 3057
```

从 22 个 token 的指令生成 100 个 token，有 KV cache 只算 121 个位置，原书的写法要算 7,150 个。而从 251 个 token 开始生成 12 个，只有前 6 个新 token 还在 256 个位置之内（第 1 个来自 prefill，后 5 个各算 1 个位置）；之后的 6 个，每个都要算 256 个位置。图 P-6 下方表格里“累计计算的位置”一栏，最后一行正是 1,792。

::: confusion 上下文只有 256，为什么能生成上千个 token？
“上下文 256”很容易被理解成“一共只能处理 256 个 token”。可是原书的 `generate()` 默认 `max_new_tokens=1000`，生成上千个 token 也不会报错。这两个数说的不是同一件事。

上下文长度限制的是**一次前向传播能看见多少个 token**；生成的总长度则是**所有前向传播加起来处理了多少个 token**。用 P.7-b 的数字来说：从 251 个 token 生成 12 个，模型每一步都只看见最近的 256 个 token，但前向传播累计算了 1,792 个位置。窗口是一次能看多远，累计量是一共看了多少；超出窗口的部分不会报错，只是被悄悄忘掉了。
:::

::: cs336 第 10 讲：推理
CS336 的第 10 讲专门讲推理：prefill 与 decode 的算术强度、KV cache 的大小以及如何压缩它（GQA、MQA、MLA）、量化、剪枝与蒸馏、投机采样、continuous batching 与 PagedAttention（[讲义](https://cs336.stanford.edu/lectures/?trace=lecture_10)）。这部分没有对应的作业，作业 5 直接使用 vLLM。DLFS6 里，CodeBot 的 `utils.generate` 每一步重算整个上下文，StoryBot 和 WebBot 的版本才用上了 KV cache。本页的 P.2 与 P.7 分别是量化和 prefill/decode 的最小演示。
:::

## P.8 三个检查点，三种回答

一个 prompt 的一生走完了。现在把三个检查点放在一起：同一串 token、同样的 greedy 解码，只有权重不同。

```widget playground-compare caption="三个检查点回答同一个 prompt。3+4= 这一题，预训练模型写起了注释，SFT 模型答 6，GRPO 模型答 7。下方逐题测试 81 道个位数加法，以及它没见过的两组题。"
{"prompt": "3+4=", "template": true, "maxNewTokens": 48, "set": "train"}
```

预训练模型不认识模板，在后面接着写起了注释。SFT 模型学会了模板的格式，回答简短，答完就停，可是答错了。GRPO 模型答对了。第 3 章的 GRPO 训练用的正是这 81 道题：

```listing ch03/09_grpo.py#L18-L26
```

判断对错的是奖励函数：取回答里的最后一个整数，与正确答案比较。

```listing ch03/09_grpo.py#L64-L72
```

我们用同样的规则，在 PyTorch 里把 81 道题全部问了一遍，greedy 解码，直到模型输出 `<|endoftext|>`（每个回答都在 64 个 token 之内停下，所以与原书默认的上限 1000 等价）。除了本书自己的两个检查点，还问了原书的官方权重，并加了两组 GRPO 没练过的题：含 0 的 19 道，以及两位数加一位数的 81 道。

```verify id=P.8-a title="81 道加法：SFT 答对 44 道，GRPO 答对 70 道；没练过的两组题，一道也没答对" files=data/playground/addition.json
add = json.load(open("data/playground/addition.json", encoding="utf-8"))
res = lambda name, s="train": add["models"][name]["results"][s]
print(f"{'':18}" + "".join(f"{s:>10}" for s in add["sets"]))
for name in add["models"]:
    print(f"{name:18}" + "".join(f"{res(name, s)['n_correct']:>6}/{len(add['sets'][s]):<3}" for s in add["sets"]))
assert res("ours-sft-f32")["n_correct"] == 44 and res("ours-grpo-f32")["n_correct"] == 70
assert res("official-sft-f32")["n_correct"] == 44 and res("official-grpo-f32")["n_correct"] == 67
same = {f"{n}/{s}": sum(a == b for a, b in zip(res(f"ours-{n}-q8", s)["responses"], res(f"ours-{n}-f32", s)["responses"]))
        for n in ["sft", "grpo"] for s in add["sets"]}
print("int8 与 float32 回答相同的题数：", same)
assert same["sft/train"] == same["grpo/train"] == 81 and sum(same.values()) == 2 * 181 - 1
assert all(res(name, s)["n_correct"] == 0 for name in add["models"] for s in ["zero", "teens"])
pick = lambda s, p: res("ours-grpo-f32", s)["responses"][add["sets"][s].index(p)]
print("GRPO 的回答：", {p: pick("teens", p) for p in ["11+3=", "12+5=", "15+4="]}, {"0+5=": pick("zero", "0+5=")})
teens = [(p, int(p[:-1].split("+")[1])) for p in add["sets"]["teens"]]
one_plus_b = sum(pick("teens", p) == str(1 + b) for p, b in teens)       # 回答恰好是 1 + b 的题数
print(f"两位数的题里，GRPO 回答恰好是 1 + b 的：{one_plus_b}/81；70/81 = {70 / 81:.1%}")
assert pick("teens", "11+3=") == "4" and pick("zero", "0+5=") == "10" and one_plus_b == 48

i34 = add["sets"]["train"].index("3+4=")              # 图 P-7 的例子：三个检查点对 3+4= 的 greedy 回答
math_pretrain = decode(next(p for p in refs["pretrain"]["prompts"] if p["name"] == "math")["q8"]["greedy"])
print("3+4= →", {"预训练": math_pretrain[:22], "SFT": res("ours-sft-q8")["responses"][i34], "GRPO": res("ours-grpo-q8")["responses"][i34]})
assert math_pretrain.startswith("#") and res("ours-sft-q8")["responses"][i34] == "6" and res("ours-grpo-q8")["responses"][i34] == "7"
```

先看第一列。本书的 SFT 模型答对 44 道，GRPO 模型答对 70 道，正是进度条上的 86.4%。官方权重是 44 道和 67 道，与第 3 章测得的一样（[3.4.2 节](ch03.html#s3-4-2)）。int8 的两行（`-q8`）在 81 道练过的题上，与 float32 的回答逐题相同；全部 362 个回答里只有一个不同（SFT 模型对 `14+6=` 的回答，int8 是 7，float32 是 9，都不对）。在浏览器里点“逐题测试”，得到的也是 44 和 70，而且每一道题的回答都与 PyTorch 用同一份 int8 权重算出的相同。

::: think GRPO 模型对 `0+5=` 会答什么？先猜一猜，再看上面验证单元的最后一行。
:::
::: answer
它答 `10`。含 0 的 19 道题，它一道也没答对。0 从来没有出现在训练的题目里，模型并没有学会“加法”，只学会了 81 道题的答案。两位数的题更说明问题：`11+3=` 它答 `4`，`12+5=` 答 `6`，`15+4=` 答 `5`。81 道两位数的题里，有 48 道的回答恰好是 1 + b，就好像它只读了第一个数的第一位。
:::

```localcheck verify/test_playground.py::test_addition_record_is_current title="重新问一遍 6 个模型的全部 181 道题，与 addition.json 逐题相同"
```

::: confusion 进度条上的 86.4%，能说明模型学会了加法吗？
GRPO 训练结束时，进度条上的 `acc` 是 86.4%，很容易把它当成“模型的加法能力”。可是 `09_grpo.py` 评估用的，正是训练时反复练习的那 81 道题，而且这 81 道题连同答案，早就出现在 SFT 的数据里了。这个数字说明的是“练过的题答对了多少”，相当于把作业题原样拿来当考试。

要知道模型学到了什么，得拿它没见过的题来考，也就是留出评估（held-out evaluation）。上面加的两组题就是最简单的留出集：结果是 0 分。训练中看到的准确率是训练信号，告诉我们优化在推进；它不能代替留出评估，告诉我们模型能做什么。
:::

::: cs336 作业 5：GRPO 的评估方式
CS336 的作业 5 也做 GRPO，但评估方式不同：在留出的验证集上评估，并且用 4 个随机种子重复实验（[作业 5 的仓库](https://github.com/stanford-cs336/assignment5-alignment)）。DLFS6 的 `09_grpo.py` 则直接在训练用的 81 道题上评估。对一个只为演示 GRPO 机制的玩具任务来说，这样做无可厚非，只是读数时要记住它的含义。
:::

## P.9 与 PyTorch 对拍

最后，我们来确认工作台里的这份 JavaScript，算的确实是原书的 CodeBot。点下面的按钮，工作台会把 P.2.3 节那 4 个 prompt 在你的浏览器里重算一遍，与 PyTorch 记录的结果逐项比较：最后一个位置的 1,000 个 logits、greedy 生成的 16 到 32 个 token、`def add(a, b):` 的 36 张注意力图，以及每一层的 logit lens。

```widget playground-parity caption="与 PyTorch 对拍。int8 一栏与使用同一份 int8 权重的 PyTorch 比较，检验的是实现；float32 一栏与原始权重比较，差别来自量化。"
{"checkpoint": "grpo", "tolerance": 0.0001}
```

参考结果有两份，各回答一个问题。与“同一份 int8 权重的 PyTorch”比较，检验 JavaScript 的实现对不对：这时 logits 最多差 1.5×10⁻⁵ 左右，greedy 的 token 全部相同。与“原始 float32 权重的 PyTorch”比较，看到的是量化带来的差别：logits 差到 0.1 到 0.3，greedy 序列有时会在中途分岔（P.2.3 节）。

那么，用的是同一份权重，为什么还会差 10⁻⁵ 呢？原因在于浮点数的加法不满足结合律。384 或 1,536 个乘积相加，先加哪个、后加哪个，float32 的舍入就不一样。PyTorch 调用的矩阵库有它自己的加法顺序，工作台的 SIMD 内核用的是 4 条通道各自累加、最后两两相加的顺序：

```verify id=P.9-a title="同样的点积，两种 float32 加法顺序给出不同的结果；参考 prompt 的第一名都领先得足够多"
rng = np.random.default_rng(0)
x = rng.normal(size=1536).astype(np.float32)
w = rng.normal(size=1536).astype(np.float32)
exact = float(np.dot(x.astype(np.float64), w.astype(np.float64)))
left = np.float32(0)
for a, b in zip(x, w):                            # 从左到右，一个一个加
    left = np.float32(left + a * b)
lanes = np.zeros(4, dtype=np.float32)
for i in range(0, 1536, 4):                       # 4 条通道各加各的，最后两两相加（工作台的 SIMD 内核）
    lanes += x[i:i + 4] * w[i:i + 4]
simd = np.float32((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
print(f"float64：{exact:.7f}；float32 从左到右：{left:.7f}；float32 四条通道：{simd:.7f}")
print(f"两种 float32 加法顺序相差 {abs(float(left) - float(simd)):.1e}")
assert left != simd and abs(float(left) - exact) < 1e-3 and abs(float(simd) - exact) < 1e-3

gaps = []
for n, ref in refs.items():
    for p in ref["prompts"]:
        top2 = np.sort(np.array(p["q8"]["logits"]))[-2:]
        gaps.append(top2[1] - top2[0])
print(f"12 个参考 prompt 里，第一名与第二名 logit 的最小差距：{min(gaps):.3f}")
assert min(gaps) > 1e-3
```

只要第一名与第二名之间的差距比这种舍入误差大，greedy 选出的 token 就不会变。12 个参考 prompt 的最后一个位置上，这个差距最小是 0.004，是 10⁻⁵ 的几百倍。当然，这只是 12 个位置；生成过程中每一步是否都有这样的余量，要靠上面的对拍来回答：每个检查点的 112 个 greedy token 全部相同。

浏览器里的按钮只能在你的设备上检查一次。为了让每次修改代码之后都能复查，同样的比较也写成了测试：JavaScript 一侧用 Node 运行，PyTorch 一侧由 pytest 重新生成参考结果。

```shell
node --test tests/js/playground-gpt.test.mjs tests/js/playground-parity.test.mjs tests/js/playground-engine.test.mjs
uv run pytest verify/test_playground.py
```

## 小结

这一页，我们跟着一个 prompt 把 CodeBot 从头到尾走了一遍。文字先被第 1 章的分词器切成 token，套上第 3 章的模板；每个 token 查表得到向量，加上位置 embedding，再穿过 6 个块，每个块都只往残差流上“加”一点东西；最后一个位置的向量变成 1,000 个 logits，经过温度、Top-k 和 Top-p，抽出下一个 token。为了让这一切在浏览器里跑起来，权重被按行量化成 int8，推理用 KV cache 一步一步地做，而 CodeBot 的可学习位置 embedding，给它立了一堵 256 个位置的墙。

三个检查点的比较说明了第 3 章三个阶段各自的作用：预训练学会续写，SFT 学会格式，GRPO 把练过的题答对。但只是练过的题。

本章所学的内容：

- 本书自己训练的三个检查点，用时与最终指标都能追溯到运行报告（[验证 P.1-a](#v-P.1-a)）。
- CodeBot 的 11,121,640 个参数，按行量化成 int8 后是 11.3 MB，约为 float32 的四分之一（[验证 P.2-a](#v-P.2-a)）；每个权重的误差不超过半个 scale，一行离群也不会拖累别的行，但 logits 仍会偏移最多 0.29，greedy 序列可能在近乎打平的位置分岔（[验证 P.2-b](#v-P.2-b)、[验证 P.2-c](#v-P.2-c)、[验证 P.2-d](#v-P.2-d)）。
- logit lens 的第 0 层总是“预测”当前 token 自己，这是权重共享的结果（[验证 P.4-a](#v-P.4-a)）；第 1 层第 1 个头是一个“前一个 token 头”（[验证 P.5-a](#v-P.5-a)）。
- Top-p 量的是 Top-k 剩下的部分，所以两者的先后顺序会改变结果；熵用自然对数算，单位是 nats（[验证 P.6-a](#v-P.6-a)、[验证 P.6-b](#v-P.6-b)）。
- KV cache 与从头重算的结果相同；可学习的位置 embedding 让缓存无法滑动，越过 256 之后每个 token 都要重算整个窗口（[验证 P.7-a](#v-P.7-a)、[验证 P.7-b](#v-P.7-b)）。
- GRPO 把 81 道练过的题从 44 道提高到 70 道，没练过的题一道也答不对；训练中的准确率不能代替留出评估（[验证 P.8-a](#v-P.8-a)）。
- 同一份权重下，JavaScript 与 PyTorch 只差浮点加法顺序带来的 10⁻⁵ 量级的舍入，远小于第一名领先的幅度，所以 greedy 生成的 token 逐个相同（[验证 P.9-a](#v-P.9-a)）。

工作台里的 CodeBot 是在 GPU 上训练好的，我们只是看着它推理。训练本身能不能也搬进浏览器呢？下一页，我们就在这个标签页里，从零训练一个迷你 GPT。
