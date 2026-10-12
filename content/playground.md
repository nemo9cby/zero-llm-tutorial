---
chapter: P
slug: playground
order: 160
kind: extra
title: CodeBot 工作台：一个 prompt 的一生
nav_title: CodeBot 工作台
lede: 第 I 部训练出来的 CodeBot，现在就在这个标签页里运行。输入一段文字，看它被切成 token、穿过 6 个 Transformer 块、变成 1,000 个 logits，再从中抽出下一个 token；把预训练、SFT、GRPO 三个检查点换来换去，比一比同一个问题的三种回答。在 4 个参考 prompt 上，浏览器算出的数字可以与 PyTorch 记录的结果逐项对拍。
pyodide_packages: [numpy]
---

第 I 部用三章做出了 CodeBot。第 1 章的分词器把代码切成 token，第 2 章的 GPT 根据前面的 token 算出下一个 token 的概率，第 3 章先预训练、再做 SFT、最后用 GRPO 强化，得到了三个检查点。可是一路走来，模型一直待在终端里：我们写下 `model(ids)`，拿到一个形状为 (1, C, 1000) 的张量，中间发生的一切都看不见。

这一页把 CodeBot 搬进浏览器，把中间的每一步摊开来看。权重是本书在一张 RTX 3060 Ti 上原样运行原书 ch03 脚本训练出来的。推理代码是一份不依赖任何框架的 JavaScript（矩阵乘法交给一个很小的 WebAssembly SIMD 内核），在后台线程里运行。用同一份 int8 权重做 greedy 生成，它得到的 token 与 PyTorch 逐个相同（P.9 节）。

路线是这样的。先看结果（P.1），再把权重装进浏览器（P.2）。然后跟着一个 prompt 走完它的一生：文字变成 token（P.3），穿过 6 个 Transformer 块（P.4），在 Attention 里互相“交流”（P.5），最后 1,000 个 logits 变成下一个 token（P.6）。接着看一个接一个地生成时，KV cache 省下了什么，256 个位置的上限又意味着什么（P.7）。最后，我们把三个检查点放在一起比较（P.8），并亲手确认这份 JavaScript 与 PyTorch 算得一样（P.9）。

## P.1 先来看结果

下面就是工作台。它进入视野时会下载 SFT 检查点（11.3 MB），然后按图中的设置回答“read a file line by line”。你可以改动指令、切换检查点、拖动采样参数，再点“生成”。

```widget playground-bench caption="CodeBot 工作台。勾选模板时，输入会被套进 ch03 的 Alpaca 格式；生成的 token 底色越深，说明它被抽中时的概率越低。把鼠标移到某个 token 上，能看到当时的前 5 个候选。"
{"checkpoint": "sft", "prompt": "read a file line by line", "template": true, "temperature": 0.8, "topK": 40, "topP": 0.95, "maxNewTokens": 128, "seed": 1}
```

工作台做的事情，与原书的 `ch03/04_chat.py` 基本相同：把输入套进模板，让模型一个接一个地生成 token，遇到 `<|endoftext|>` 就停下（[3.4 节](ch03.html#s3-4)）。不同的有两点。第一，全部计算都在你的浏览器里进行，没有任何服务器参与。第二，采样多了 Top-k 与 Top-p 两个旋钮。原书的 `generate()` 只有温度这一个旋钮，`04_chat.py` 的设置是这样的：

```listing ch03/04_chat.py#L9-L15
```

Top-k 与 Top-p 是工作台按第 3 章的写法补上的（[3.2.3 节](ch03.html#s3-2-3)）。工作台默认用温度 0.8、Top-k 40、Top-p 0.95，最多生成 128 个 token。想要与 `04_chat.py` 相同的抽样分布，把温度拖到 1、Top-k 拖到“不限”、Top-p 拖到 1 即可（这几道工序 P.6 节再细看）。

不妨先切换到“预训练”，点一下“生成”。预训练模型只会续写，它多半会在模板后面写起以 `#` 开头的注释（P.2.3 节的验证单元里有两段这样的续写）。换回“SFT”或“GRPO”，模型就会照着模板作答，答完输出 `<|endoftext|>`。再试试示例里的 `3+4=`：把温度拖到 0（greedy），分别问 SFT 和 GRPO，两者的回答不一样。P.8 节会细看这件事。

那么，这三个检查点是从哪里来的呢？它们不是原书作者公开在 Hugging Face 上的官方权重，而是本书自己训练的：在本机的 RTX 3060 Ti 上，用 `gpu/run_local.py` 依次原样运行 `ch03/01_pretrain.py`、`03_sft.py` 和 `09_grpo.py`（[G.4 节](gpu.html#sG-4)）。运行报告记下了每个脚本的用时和最终的指标，导出权重时，这些记录被原样抄进了每个检查点的清单文件（`manifest.json`）。下面的验证单元把两边对一对：

```verify id=P.1-a title="清单里的训练记录与运行报告一致" files=data/runs/codebot-3060ti/report.json,data/playground/pretrain/manifest.json,data/playground/sft/manifest.json,data/playground/grpo/manifest.json
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

这个单元核对的只是记录本身。分片里的权重确实来自这几个 checkpoint，要靠 P.2.2 节的本地验证：它重新计算 checkpoint 和每个分片的 SHA-256，并逐行比较还原出的权重。

预训练用了 42 分钟，最终 loss 0.619；SFT 只用了 2 分钟；GRPO 最慢，将近一个半小时，进度条上最后的准确率是 86.4%。第 3 章的三条训练曲线，读的也是这次运行的记录（[3.1.5 节](ch03.html#s3-1-5)、[3.3.6 节](ch03.html#s3-3-6)、[3.9.3 节](ch03.html#s3-9-3)）。原书的官方权重同样可以装进工作台，P.2.4 节会给出命令。

## P.2 把权重装进浏览器

### P.2.1 1,112 万个参数有多重

CodeBot 有 11,121,640 个参数（[2.10 节](ch02.html#s2-10)）。用 float32 保存时，每个参数占 4 个字节，合计 44.5 MB，原书保存的 `model_sft.pt` 就是这么大。保存的内容由 `GPT.save()` 决定：

```listing codebot/model.py#L152-L163
```

除了参数本身（`state_dict`），文件里还有重建模型所需的 7 个超参数。导出工具从中读出决定形状的 6 个，`dropout_rate` 在推理时用不到。

44.5 MB 放在本地硬盘上不算什么，放进网页就太重了：三个检查点合计 133 MB。于是，我们把绝大多数参数压成 1 个字节的整数，也就是 **int8 量化**（quantization）。先算一笔账：

```verify id=P.2-a title="参数量与字节数：float32 是 44.5 MB，int8 按行量化是 11.3 MB"
V, C, E, L, FF = 1000, 256, 384, 6, 1536            # vocab_size, max_context_len, embed_dim, n_layer, ff_dim
mats = [(V, E), (C, E)] + L * 4 * [(E, E)] + L * [(FF, E), (E, FF)]   # 2 维权重的（行, 列）
vecs = L * [E, E, E, E, FF, E] + [E, E, V]           # LayerNorm 的 γ 与 β、FFN 的偏置、最后的 LN、unembed 的偏置
n_params = sum(r * c for r, c in mats) + sum(vecs)
f32_bytes = 4 * n_params
q8_bytes = sum(r * c + 4 * r for r, c in mats) + 4 * sum(vecs)   # 每个元素 1 字节，每行再加 4 字节的 scale
print(f"参数 {n_params:,} 个，其中 2 维权重占 {1 - sum(vecs) / n_params:.1%}，一维参数（LayerNorm 与偏置）{sum(vecs):,} 个")
print(f"float32：{f32_bytes:,} 字节（{f32_bytes / 1e6:.1f} MB），三个检查点合计 {3 * f32_bytes / 1e6:.0f} MB")
print(f"int8 按行量化：{q8_bytes:,} 字节（{q8_bytes / 1e6:.1f} MB），约为 float32 的 1/{f32_bytes / q8_bytes:.2f}")
assert n_params == 11_121_640 and q8_bytes == 11_277_120
assert sum(vecs) == 22_504 and round(1 - sum(vecs) / n_params, 3) == 0.998 and round(3 * f32_bytes / 1e6) == 133
```

这里只量化 2 维的权重矩阵：token embedding、位置 embedding、Attention 的 4 个投影矩阵和 FFN 的 2 个矩阵，它们占了参数的 99.8%。LayerNorm 的 γ、β 和各处的偏置一共只有 22,504 个数，照旧用 float32 保存。`unembed.weight` 则根本不用存：CodeBot 的输出层与 embedding 共用同一个矩阵（权重共享，[2.10.2 节](ch02.html#s2-10-2)）。

### P.2.2 按行量化成 int8

一个 int8 只能表示 −128 到 127 之间的整数，而权重是 0.0123 这样的小数。办法是给每一行配一个缩放系数 **scale**：

$$
s_r = \frac{\max_j |w_{rj}|}{127}, \qquad q_{rj} = \mathrm{round}\!\left(\frac{w_{rj}}{s_r}\right), \qquad \hat w_{rj} = q_{rj}\, s_r
$$

这里，$w_{rj}$ 是第 $r$ 行第 $j$ 列的权重，$s_r$ 是这一行的 scale（一个 float32），$q_{rj}$ 是 −127 到 127 之间的整数，只占 1 个字节（−128 用不上：scale 按这一行最大的绝对值来定，$|w_{rj} / s_r|$ 最大正好是 127）。浏览器加载时，按 $\hat w_{rj} = q_{rj}\, s_r$ 把权重还原成 float32。四舍五入最多偏 0.5，所以还原的误差不超过半个 scale：$|\hat w_{rj} - w_{rj}| \le s_r / 2$。

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
ulp = np.spacing(np.abs(W).max(axis=1, keepdims=True))   # 这一行最大的数在 float32 里相邻两个数的间隔
assert np.all(err <= s[:, None] / 2 + 2 * ulp)    # 容差：w / s 与 q·s 在 float32 里各舍入一次，误差可能多出一两个 ulp
```

为什么要“每行一个 scale”，而不是整个矩阵共用一个呢？因为 scale 由最大的那个数决定。只要矩阵里有一行特别大，共用的 scale 就会被拉大，其余各行的数，九成以上被挤到 −2 到 2 这五个整数上（七成落在 0 和 ±1），精度就没了。按行量化把这种影响关在了那一行里：

```verify id=P.2-c title="有一行离群时，整个矩阵共用一个 scale 的误差大得多"
W2 = W.copy()
W2[7] *= 30                                       # 制造一个离群的行
others = np.arange(384) != 7
per_row = np.abs(dequantize_rows(*quantize_rows(W2)) - W2)[others].mean()
t = np.abs(W2).max() / 127                        # 整个矩阵只用一个 scale
q_shared = np.clip(np.rint(W2 / t), -127, 127)
per_tensor = np.abs(q_shared * t - W2)[others].mean()
print(f"其余 383 行的平均误差：每行一个 scale {per_row:.1e}，整个矩阵一个 scale {per_tensor:.1e}（{per_tensor / per_row:.0f} 倍）")
within2, within1 = (np.mean(np.abs(q_shared[others]) <= k) for k in (2, 1))
print(f"整个矩阵一个 scale 时，其余各行的数有 {within2:.0%} 落在 −2…2，{within1:.0%} 落在 0 和 ±1")
assert per_tensor > 10 * per_row and within2 > 0.9 and within1 > 0.7
```

导出工具 `tools/export_codebot.py` 里的量化函数，与上面的 `quantize_rows` 是同一段代码（只有注释不同）。本地测试会拿两者在同一个矩阵上对比，并检查导出文件里每一行的误差都不超过半个 scale：

```localcheck verify/test_playground.py::test_quantize_cell_matches_exporter title="验证单元里的 quantize_rows 与导出工具逐位相同"
```

```localcheck verify/test_playground.py::test_exports_match_checkpoints title="三个导出文件：分片的 SHA-256、参数量、每行误差 ≤ scale/2、偏置与 LayerNorm 原样保存"
```

### P.2.3 量化改变了多少

权重变了一点点，模型的输出会变多少呢？导出时，我们用原书的 `codebot.model.GPT` 在 PyTorch 里跑了 4 个 prompt：续写 `def add(a, b):`、对话模板下的 `print hello` 和 `3+4=`，以及一段 251 个 token 的 TinyCodes 代码。每个 prompt 跑两遍，一遍用原始的 float32 权重，一遍用 int8 还原出来的权重，把最后一个位置的 logits 和 greedy 生成的 token 都记了下来（`ref.json`）：

```verify id=P.2-d title="int8 量化：12 个参考 prompt 的最后一个位置上，logits 最多偏移 0.29；greedy 序列有时会在中途分岔" files=data/playground/pretrain/ref.json,data/playground/sft/ref.json,data/playground/grpo/ref.json,data/tokenizers/codebot.json
refs = {n: json.load(open(f"data/playground/{n}/ref.json", encoding="utf-8")) for n in ["pretrain", "sft", "grpo"]}
spec = json.load(open("data/tokenizers/codebot.json", encoding="utf-8"))
vocab = {i: bytes([i]) for i in range(256)}        # 把 id 还原成文字：第 1 章的合并规则，倒过来查表
for k in range(spec["num_merges"]):                # merges 依次记录 (a, b)：第 k 条合并出 id 256 + k
    a, b = spec["merges"][2 * k], spec["merges"][2 * k + 1]
    vocab[256 + k] = vocab[a] + vocab[b]
for s, i in spec["special_tokens"].items():
    vocab[i] = s.encode()

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

EOS = spec["special_tokens"]["<|endoftext|>"]
worst, forks = 0.0, {}
for n, ref in refs.items():
    for p in ref["prompts"]:
        d = np.abs(np.array(p["q8"]["logits"]) - np.array(p["f32"]["logits"])).max()
        a, b = p["q8"]["greedy"], p["f32"]["greedy"]
        same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
        eos = b.index(EOS) if EOS in b else None   # 参考序列遇到 <|endoftext|> 也不停，这里记下它第一次出现的位置
        worst = max(worst, d)
        forks[n, p["name"]] = (same, len(a), eos)
        print(f"{n:8} {p['name']:5} max|Δlogit| = {d:.3f}，greedy 的前 {same:2d}/{len(a)} 个 token 相同"
              + (f"（第 {eos + 1} 个是 <|endoftext|>）" if eos is not None else ""))
chat_p = next(p for p in refs["pretrain"]["prompts"] if p["name"] == "chat")
print("预训练模型续写对话模板，float32：", repr(decode(chat_p["f32"]["greedy"])))
print("                          int8：", repr(decode(chat_p["q8"]["greedy"])))
assert worst < 0.3 and forks["pretrain", "chat"][0] == 13          # 从第 14 个 token 起不同
assert decode(chat_p["f32"]["greedy"]).startswith("#") and decode(chat_p["q8"]["greedy"]).startswith("#")
assert all(same == total or (eos is not None and same > eos)       # SFT 与 GRPO：分岔只发生在 <|endoftext|> 之后
           for (n, _), (same, total, eos) in forks.items() if n != "pretrain")
```

在这 12 个 prompt 的最后一个位置上，logits 最多偏移了 0.29。对大多数位置来说，这点偏移改变不了谁排第一，所以 greedy 生成的 token 往往完全相同。可一旦遇到两个候选几乎打平的位置，第一名就可能易主，此后的序列跟着分岔。预训练模型在对话模板后面的续写，第 14 个 token 起就不一样了：两边写的都还是以 `#` 开头的注释行，只是内容不同了（上面打印的两段）。

SFT 和 GRPO 续写 `def add(a, b):` 时也分了岔，不过都在 `<|endoftext|>` 之后。参考序列遇到 `<|endoftext|>` 也不停，好让每个检查点比较同样多的 token；按原书 `generate()` 的规则，生成在那里就结束了，两边的回答其实相同。

0.29 只是这 12 个位置上的最大值。换成 TinyCodes 里的任意位置，偏移可以到 0.4 左右，大约 1% 的位置第一名易主：

```localcheck verify/test_playground.py::test_int8_logit_shift_over_windows title="TinyCodes 里取 8 个 256 token 的窗口：int8 让 logits 最多偏移不到 0.5，第一名易主的位置不到 2%"
```

所以，浏览器里的 CodeBot 是训练好的模型的一个“四舍五入版”。P.8 节会看到，在 81 道加法题上，它与原始权重的回答逐题相同。P.9 节的对拍，则是拿同一份“四舍五入”后的权重，检查 JavaScript 与 PyTorch 算得是否一样。

### P.2.4 分片、按需下载与校验

导出工具把一个检查点写成一个文件夹，里面有这几样东西：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 模型的形状、每个张量的位置与类型、每个分片的 SHA-256、训练记录 |
| `weights-00.bin` … `weights-05.bin` | 6 个分片，每个不超过 2 MB |
| `ref.json` | PyTorch 的参考结果 |

工作台只在某张图第一次用到某个检查点时才下载它，6 个分片并行下载，进度显示在图里。下载完成后，先核对每个分片的大小和 SHA-256，再交给模型。（浏览器只在“安全上下文”（secure context）里提供计算 SHA-256 的接口，例如 https 页面或本机的 localhost；其他情况下只核对大小。见 [MDN 的说明](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto)，查证于 2026-10。）

我们自己的三个检查点，就是用这样的命令导出的（下面是 SFT 的那一条）。把 `--ckpt` 换成原书的官方权重（`tools/fetch_data.py` 会把它们下载到 `third_party/dlfs6/codebot/`），再换一个 `--name`，工作台的检查点开关里就会多出一项：

```shell
uv run python tools/export_codebot.py --ckpt runs/codebot-3060ti/work/codebot/model_sft.pt --name sft \
    --label SFT --template alpaca --order 1 --source "gpu/run_local.py --run codebot-3060ti" \
    --description "在预训练权重的基础上原样运行 ch03/03_sft.py：学会了按 Alpaca 模板作答。"
uv run python tools/export_codebot.py --ckpt third_party/dlfs6/codebot/model_sft.pt --name official-sft \
    --label "SFT（官方）" --template alpaca --order 11 \
    --source https://huggingface.co/datasets/koki0702/zero-llm-data/blob/main/codebot/model_sft.pt
uv run python tools/build.py
```

导出工具写的是 `web/data/playground/`，页面读的是构建出来的站点，所以导出之后要重新构建一次（最后一条命令），新的检查点才会出现在开关里。

```localcheck verify/test_playground.py::test_exporter_handles_official_checkpoint title="同一条命令（把 sft 换成 grpo）导出官方 GRPO 权重：清单、分片与参考结果都齐全"
```

推理的大头是矩阵乘法。工作台把它交给一段 1 KB 多的 WebAssembly 内核（源码是 `web/data/playground/kernels.wat`），用 SIMD 指令一次处理 4 个 float32。浏览器不支持 WebAssembly SIMD 时，自动换成纯 JavaScript 写的同一套计算。图 P-1 的状态行会写明用的是哪个后端。

## P.3 文字变成 token

权重装好了，接下来跟着一个 prompt 走。第一站是分词器。工作台用的是第 1 章训练出来的那个：`codebot/merge_rules.pkl` 被转换成了一个 JSON 文件，由 JavaScript 版的 BPE 读取，编码结果与原书的 `codebot.tokenizer` 逐个 id 相同（[1.7 节](ch01.html#s1-7)；P.9 节的测试也会拿 PyTorch 一侧记录的 id 再对一遍）。

SFT 和 GRPO 模型见过的 prompt，都套着同一个模板。原书在 `04_chat.py` 里是这样写的：

```listing ch03/04_chat.py#L17-L18
```

下面这张图把模板和你的指令一起切成 token。虚线框的是模板，实心的是指令：

```widget playground-tokens caption="套上模板之后，模型真正读到的 token。模板本身占 18 个，3+4= 只占 4 个；### 被切成了 3 个 #。"
{"instruction": "3+4=", "template": true}
```

模板有多少个 token 呢？我们可以不借助分词器，直接从 PyTorch 记录的两个 prompt 里读出来：`print hello` 与 `3+4=` 套着同一个模板，两串 id 开头相同的部分和结尾相同的部分，就是模板本身。（两条指令的第一个 token 不同，最后一个 token 也不同，所以“相同的部分”不会多吃进指令里的 token；下面的单元也确认了，夹在中间的恰好是两条指令。）

```verify id=P.3-a title="模板占 18 个 token：两个 prompt 开头与结尾相同的部分"
chat, math = (next(p for p in refs["sft"]["prompts"] if p["name"] == k) for k in ["chat", "math"])
assert decode(chat["ids"]) == chat["text"] and decode(math["ids"]) == math["text"]   # decode 定义在验证 P.2-d 里
a, b = chat["ids"], math["ids"]                   # 两个 prompt 的开头和结尾都相同，那就是模板
head = next(i for i in range(len(a)) if a[i] != b[i])
tail = next(i for i in range(1, len(a)) if a[-i] != b[-i]) - 1
print("模板开头：", [decode([i]) for i in a[:head]])
print("模板结尾：", [decode([i]) for i in a[len(a) - tail:]])
print(f"模板 {head + tail} 个 token；'print hello' 占 {len(a) - head - tail} 个，'3+4=' 占 {len(b) - head - tail} 个")
pairs = set(zip(spec["merges"][0::2], spec["merges"][1::2]))
print(f"{spec['num_merges']} 条合并规则里，把两个 # 合在一起的规则有 {int((35, 35) in pairs)} 条")
assert decode(a[head:len(a) - tail]) == "print hello" and decode(b[head:len(b) - tail]) == "3+4="   # 中间恰好是指令
assert head + tail == 18 and len(b) - head - tail == 4 and spec["num_merges"] == 743
assert (35, 35) not in pairs                      # 35 是 "#"：没有一条规则把两个 # 合并
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

注意这两行的写法：`x = x + ...`。每个块都不是把 `x` 换成新的东西，而是在 `x` 上**加**一点东西。这条从 embedding 一路加到输出的向量，常被称为**残差流**（residual stream）。每个块读的、写的，都是同一个 384 维空间里的向量，用的是同一套坐标。所以，我们可以在任意一层把残差流截下来，用最后那套“读法”（LayerNorm 加 `unembed`）读一读，看看模型此时“心里想的”是什么。

### P.4.2 logit lens：让模型提前交卷

怎样看呢？最直接的办法是：把第 $l$ 个块之后的残差流，直接送进模型自己的最后一个 LayerNorm 和 `unembed`，就好像后面的块都不存在一样。这样得到的分布，可以理解为“如果模型在第 $l$ 层就停下，它会预测什么”。这种做法叫 **logit lens**，名字出自 nostalgebraist 2020 年的一篇文章（[原文](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)，查证于 2026-10）。第 6 层的 logit lens 就是模型的输出本身。

::: think 第 0 层还没经过任何块。在这里提前交卷，第一名会是哪个 token？
:::
::: answer
多半是输入的最后一个 token 自己。下面的例子里，它的概率接近 1；后文会解释原因，也会看到例外。
:::

```widget playground-lens caption="logit lens：在 embedding 之后和每个块之后，把残差流直接送进最后的 LayerNorm 与 unembed。这个例子里，模型最终预测的 return 要到第 4 层才浮上来。"
{"checkpoint": "pretrain", "prompt": "def add(a, b):\n", "template": false}
```

以预训练模型续写 `def add(a, b):` 加换行为例。第 0 层，也就是刚做完 embedding、一个块都没经过的时候，第一名是换行符本身，概率几乎是 1。第 1 层仍是换行符。第 2 层换成 `#`，不过它与制表符几乎打平（20.5% 对 20.0%）；第 3 层制表符领先。到第 4 层，` return` 排到了第一，并一直保持到输出。

第 0 层为什么会“预测自己”？这与权重共享有关。第 0 层的残差流基本就是这个 token 自己的 embedding，`unembed` 又用同一张表给每个 token 打分，分数是两个向量的点积。直观上，在 384 维的空间里，一个向量与自己的点积，往往远大于它与别的向量的点积（严格地讲，中间还隔着最后一个 LayerNorm 的 γ、β 和 `unembed` 的偏置；下面的随机实验把它们省掉了）。

如果你读过《深度学习进阶：自然语言处理》6.5.3 节，会认出这就是那里讲的权重共享，同一章的 BetterRnnlm 也让输入和输出共用同一张表。在 PyTorch 记录的 24 个例子（12 个 prompt，各有 float32、int8 两份权重）里，第 0 层“预测自己”无一例外。随机初始化的嵌入矩阵也有这个倾向，只是自己的概率没那么高；换成不共享的输出层，这个现象就消失了：

```verify id=P.4-a title="第 0 层的 logit lens 预测“当前 token 自己”：24 个例子无一例外；随机初始化时，只有共享权重才这样"
hits = total = 0
for n, ref in refs.items():
    for p in ref["prompts"]:
        for v in ["f32", "q8"]:
            lens = p[v]["lens"]                   # 第 0 层（只有 embedding）到第 6 层，各自的前 5 名
            total += 1
            hits += lens[0]["ids"][0] == p["ids"][-1] and lens[0]["probs"][0] > 0.999
            assert lens[-1]["ids"][0] == p[v]["greedy"][0]    # 最后一层的 lens 就是模型的输出
lasts = sorted({decode([p["ids"][-1]]) for ref in refs.values() for p in ref["prompts"]})
print(f"第 0 层的第一名就是输入的最后一个 token，且概率 > 0.999：{hits}/{total}（最后一个 token 只有 {lasts} 这 {len(lasts)} 种）")
assert hits == total == 24 and len(lasts) == 2

code_p = next(p for p in refs["pretrain"]["prompts"] if p["name"] == "code")
top1 = [decode([L["ids"][0]]) for L in code_p["q8"]["lens"]]
L2 = code_p["q8"]["lens"][2]
print("def add(a, b):↵ 每一层的第一名：", top1)
print(f"第 2 层的前两名：{decode([L2['ids'][0]])!r} {L2['probs'][0]:.1%}，{decode([L2['ids'][1]])!r} {L2['probs'][1]:.1%}")
assert top1 == ["\n", "\n", "#", "\t", " return", " return", " return"]

rng = np.random.default_rng(1)
emb = rng.normal(0, 0.02, size=(1000, 384))       # 与 CodeBot 的初始化相同：N(0, 0.02²)
pos = rng.normal(0, 0.02, size=384)
X = emb + pos                                     # 每个 token 都放在同一个位置上
H = (X - X.mean(1, keepdims=True)) / np.sqrt(X.var(1, keepdims=True) + 1e-5)   # LayerNorm（省去 γ、β）
S = H @ emb.T                                     # 共享权重：unembed 用的就是嵌入表
P = np.exp(S - S.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)
self_top = np.mean(S.argmax(1) == np.arange(1000))
W_out = np.random.default_rng(2).normal(0, 0.02, size=(1000, 384))   # 不共享：输出层另用一张随机表
untied = np.mean((H @ W_out.T).argmax(1) == np.arange(1000))
print(f"随机初始化，共享权重：{self_top:.1%} 的 token 在第 0 层把自己排在第一，自己的概率中位数 {np.median(np.diag(P)):.2f}")
print(f"随机初始化，不共享权重：{untied:.1%} 的 token 把自己排在第一")
assert self_top > 0.99 and np.median(np.diag(P)) < 0.5 and untied < 0.01
```

上面这 24 个例子，最后一个 token 只有换行和冒号两种。别的 token 也会“预测自己”吗？我们把 1,000 个 token 逐一放到 256 个位置上，用浏览器里的 int8 权重检查了一遍（三个检查点都是）。预训练数据 TinyCodes 里出现过的 839 个 token，在每个位置上都把自己排在第一，概率最低的也超过 0.5。从没出现过的 161 个 token 则不然，每一个都在某些位置上被别的 token 抢走了第一。它们大多是非 ASCII 的字节（128 个）：TinyCodes 全是 ASCII，而汉字的 UTF-8 编码用的正是这些字节。在图 P-3 里输入“不”试试：它被切成 3 个字节 token，第 0 层的第一名并不是最后那个字节。

```localcheck verify/test_playground.py::test_layer0_lens_self_prediction title="1,000 个 token × 256 个位置：TinyCodes 里出现过的 839 个 token 在第 0 层都把自己排在第一，从没出现过的 161 个都有例外"
```

换句话说，对见过的 token，第 0 层的“预测”只是把输入原样照了一遍镜子。6 个块要做的事，就是把这个向量一步步推向“下一个 token”的方向。

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

不同的头看的地方很不一样。有的头把注意力分散在很多位置上，有的头盯着某几个关键的 token。第 1 层第 1 个头最好认：它的权重集中在对角线正下方那一条，也就是说，每个 token 都把相当大的一部分注意力给了紧挨着它的前一个 token。这种头常被称为“前一个 token 头”（previous-token head，见 [Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)，查证于 2026-10）。用 PyTorch 记录的权重确认一下：

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

```localcheck verify/test_playground.py::test_previous_token_head title="251 个 token 的长 prompt：第 1 层第 1 个头给前一个 token 的平均权重在 0.45 到 0.49 之间，均匀分配只有 0.02"
```

## P.6 从 logits 到下一个 token

6 个块之后，最后一个位置的向量经过 LayerNorm 和 `unembed`，变成 1,000 个 logits。从这里到“下一个 token”，还要经过几道工序（[3.2 节](ch03.html#s3-2)）。第一道是温度：logits 除以温度 $T$ 后做 softmax，

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

这里，$z_i$ 是第 $i$ 个 token 的 logit。用第 3 章的手算例子来说，logits 为 $[2, 1, 0.5, 0, -1]$ 时，第一名的概率在 $T = 1$ 时是 0.563，$T = 0.5$ 时升到 0.829，$T = 2$ 时降到 0.375（验证 P.6-b 会算一遍）。原书的 `generate()` 只有这一道工序。工作台按第 3 章的写法又加了两道（[3.2.3 节](ch03.html#s3-2-3)）：Top-k 只留概率最大的 $k$ 个，Top-p 再留下累计概率首次达到 $p$ 的最少几个，每截一次都重新归一化，最后按剩下的分布抽一个。温度为 0 时不抽样，直接取 argmax（greedy）。

```widget playground-next caption="预训练模型在 for i in 之后的下一个 token。range 一个就占了八成（第 3 章用官方权重测得 86%，这里是本书自己训练的权重）；调低温度，分布变尖，调高温度，长尾被抬起来。灰色的候选是被 Top-k / Top-p 截掉的。"
{"checkpoint": "pretrain", "prompt": "for i in", "template": false, "temperature": 1, "topK": 0, "topP": 1, "seed": 0}
```

```localcheck verify/test_playground.py::test_for_i_in_prefers_range title="预训练模型在 for i in 之后：range 排第一，概率约八成（float32 与 int8 权重都是）；int8 权重下分布的熵是 1.23 nats"
```

工作台按“温度 → softmax → Top-k → Top-p”的顺序处理。顺序是有影响的：Top-p 量的是 Top-k 剩下的那几个 token 的累计概率。用第 3 章的手算例子（logits 为 $[2, 1, 0.5, 0, -1]$）来看：单独用 Top-p = 0.7，要留 2 个 token（0.563 还不够，加上 0.207 才超过 0.7）；先用 Top-k = 2，剩下的两个重新归一化成 0.731 和 0.269，第一个就已经超过 0.7，Top-p 之后只剩 1 个。反过来，先做 Top-p = 0.7（留下 2 个），再做 Top-k = 2，还是 2 个。同样两个参数，换个顺序，结果就从 1 个变成了 2 个。

```verify id=P.6-a title="温度、Top-k、Top-p 的顺序：先 Top-k，Top-p 量的是剩下的部分；换个顺序，结果就不同"
def softmax(x, T=1.0):
    z = np.asarray(x, dtype=np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()

def filter_dist(logits, T=1.0, top_k=0, top_p=1.0, order="kp"):
    """温度 → softmax → 按 order 依次做 Top-k（k）与 Top-p（p），每一步都在剩下的 token 里量。
    工作台的顺序是 "kp"：Top-k → 重新归一化 → Top-p → 重新归一化。"""
    if T == 0:                                    # greedy：概率全给 argmax
        return np.eye(len(logits))[int(np.argmax(logits))]
    p = softmax(logits, T)
    keep = np.argsort(-p, kind="stable")
    for step in order:
        if step == "k" and 0 < top_k < len(keep):
            keep = keep[:top_k]
        if step == "p" and top_p < 1:
            cum = np.cumsum(p[keep]) / p[keep].sum()
            keep = keep[: int(np.searchsorted(cum, top_p)) + 1]   # 累计概率首次达到 top_p 的最少 token
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum()

logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
cases = [({}, "不过滤"), ({"top_k": 2}, "Top-k = 2"), ({"top_p": 0.9}, "Top-p = 0.9"), ({"top_p": 0.7}, "Top-p = 0.7"),
         ({"top_k": 2, "top_p": 0.7}, "先 Top-k = 2，再 Top-p = 0.7"),
         ({"top_k": 2, "top_p": 0.7, "order": "pk"}, "先 Top-p = 0.7，再 Top-k = 2"), ({"T": 0}, "温度 0（greedy）")]
for kw, label in cases:
    print(f"{label} → {np.round(filter_dist(logits, **kw), 3).tolist()}")
assert np.count_nonzero(filter_dist(logits, top_p=0.9)) == 4
assert np.count_nonzero(filter_dist(logits, top_p=0.7)) == 2
assert np.count_nonzero(filter_dist(logits, top_k=2, top_p=0.7)) == 1   # 先 Top-k，Top-p 量的是剩下的 2 个
assert np.count_nonzero(filter_dist(logits, top_k=2, top_p=0.7, order="pk")) == 2   # 反过来：还是 2 个
```

工作台的抽样用了一个带种子的随机数生成器（mulberry32），同样的种子、同样的设置，抽出的 token 序列总是相同的。原书的 `generate()` 用 `torch.multinomial` 抽样，用的是另一个随机数生成器。即使两边都固定种子，抽出的随机数也不一样，所以随机抽样的结果无法与 PyTorch 逐个对比；P.9 节对拍的是不涉及随机性的 greedy 生成。

图中还显示了分布的熵：

$$
H(p) = -\sum_i p_i \ln p_i
$$

这里，$p_i$ 是第 $i$ 个 token 的概率。熵衡量分布有多分散：两个 token 各占一半时，$H = \ln 2 \approx 0.693$；全部概率在一个 token 上时，$H = 0$；1,000 个 token 平分时最大，等于 $\ln 1000 \approx 6.908$。

::: confusion 熵的单位“nats”是什么？
看到图 P-5 下方“熵 1.23 nats”这样的说法，很自然会问：nats 是什么单位，和常听说的 bits 有什么关系？其实两者只差在对数的底。用自然对数 ln 算出来的熵，单位叫 nats；用以 2 为底的对数算，单位叫 bits。两者之间只差一个常数倍：1 nat = 1/ln 2 ≈ 1.443 bits。

PyTorch 的 `cross_entropy` 用的是自然对数，所以第 3 章训练曲线上的 loss，单位也是 nats。CodeBot 的词表有 1,000 个 token，完全瞎猜时，每个 token 的熵是 ln 1000 = 6.908 nats，也就是 log₂ 1000 ≈ 9.97 bits，差不多要 10 个“是或否”的问题才能猜中。
:::

```verify id=P.6-b title="温度越高，第一名的概率越低、熵越大；熵的单位：ln 得 nats，log₂ 得 bits"
def entropy(q):
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())          # 自然对数：单位是 nats

H_max = np.log(1000)                              # 1,000 个 token 均匀分布：熵的上限
print(f"两个 token 各占一半：{entropy(np.array([0.5, 0.5])):.3f} nats；1,000 个均匀分布：{H_max:.3f} nats = {H_max / np.log(2):.2f} bits")
for T in [0.5, 1.0, 2.0]:
    q = filter_dist(logits, T=T)
    print(f"温度 {T}：第一名的概率 {q[0]:.3f}，熵 {entropy(q):.3f} nats")
assert [round(filter_dist(logits, T=T)[0], 3) for T in [1.0, 0.5, 2.0]] == [0.563, 0.829, 0.375]
assert entropy(filter_dist(logits, T=0.5)) < entropy(filter_dist(logits)) < entropy(filter_dist(logits, T=2.0))
assert abs(entropy(np.array([0.5, 0.5])) - np.log(2)) < 1e-12 and entropy(filter_dist(logits, T=0)) == 0
assert abs(H_max / np.log(2) - np.log2(1000)) < 1e-12 and round(H_max, 3) == 6.908
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
else:
    raise AssertionError(f"位置 {T_} 不该有位置 embedding")
```

工作台的 JavaScript 更进一步：prefill 用的矩阵乘法和 decode 用的矩阵向量乘法，按完全相同的顺序累加，所以两条路算出的 logits 逐位相同，不只是“差别很小”。因此，用不用 KV cache，生成的 token 都一模一样。Node 运行的测试在合成的小模型上和三个真实检查点上都检查了这一点，P.9 节的本地验证会把这些测试跑一遍。

### P.7.2 256 的墙

可是，KV cache 在 CodeBot 身上有一个限度。CodeBot 的位置 embedding 只有 256 行，第 257 个位置没有向量可查。原书的循环在这里把序列从左边截掉，只保留最后 256 个 token（[3.2.4 节](ch03.html#s3-2-4)）。

截掉之后，缓存还能接着用吗？第 5 章说过，KV cache 靠的是一个性质：往序列后面追加 token 时，前面位置的输出不变，所以它们的 K、V 算一次就能一直用（[5.5.2 节](ch05.html#s5-5-2)）。从左边截掉 token，恰好破坏了这个性质，而且破坏了两处。

第一，剩下的 token 都往前挪了一个位置：原来在位置 1 的 token，现在在位置 0。CodeBot 的位置 embedding 按位置查表，所以第 1 层的 K、V 全都变了。第二，被截掉的 token 曾被后面每个 token 看见过，它的信息已经混进了第 2 层起的每一个 K、V。第二点与位置编码无关，换成 RoPE、甚至干脆没有位置编码，也是一样。

用 P.7-a 的迷你 GPT 试一试：9 个 token，比 8 个位置多出 1 个。截掉第 1 个 token 之前缓存的 K，与截掉之后重算的 K，逐层比一比；再把位置 embedding 整个拿掉，重做一遍：

```verify id=P.7-b title="从左边截掉 1 个 token：拿掉位置 embedding，第 1 层的 K 不变了，第 2 层的 K 照样变"
def keys(ids, pos):                               # 从头算一遍，返回每一层的 K（每行一个 token）
    x = Pm["tok"][ids] + pos[: len(ids)]
    Ks = []
    for l in range(L_):
        h = norm(x)
        Q, K, Vs = h @ Pm["Wq"][l], h @ Pm["Wk"][l], h @ Pm["Wv"][l]
        Ks.append(K)
        x = x + np.stack([attend(Q[t], K[: t + 1], Vs[: t + 1]) for t in range(len(ids))]) @ Pm["Wo"][l]
        x = x + gelu(norm(x) @ Pm["W1"][l]) @ Pm["W2"][l]
    return Ks

seq = ids + [7]                                   # 9 个 token，比 8 个位置多 1 个
diff = {}
for name, pos in [("可学习的位置 embedding", Pm["pos"]), ("没有位置 embedding", np.zeros_like(Pm["pos"]))]:
    cached, fresh = keys(seq[:8], pos), keys(seq[1:], pos)    # 截掉之前缓存的 K；截掉第 1 个 token 后重算的 K
    diff[name] = [np.abs(cached[l][1:] - fresh[l][:-1]).max() for l in range(L_)]   # 同样 7 个 token 的 K
    print(f"{name}：第 1 层的 K 最多差 {diff[name][0]:.3f}，第 2 层最多差 {diff[name][1]:.3f}")
assert diff["可学习的位置 embedding"][0] > 0.1                          # 位置挪了一格：第 1 层就变了
assert diff["没有位置 embedding"][0] < 1e-12 and diff["没有位置 embedding"][1] > 0.1   # 第 1 层不变，第 2 层照样作废
```

所以，要与原书“截掉再从头算”的结果逐个相同，序列一旦超过 256 个 token，每生成一个 token，就只能把最近的 256 个 token 从头重算一遍。第 5 章的 StoryBot 用的是 RoPE，越过 256 之后它没有重算，而是接着用缓存，生成的结果就与“每一步只看最近 256 个、从头重算”分岔了（[5.5.5 节](ch05.html#s5-5-5)）。

下面的图分两步测一测。先测“生成 1 个新 token 要多久”：有 KV cache 时只算 1 个位置，没有时要把上下文从头算一遍。再从一段 251 个 token 的代码开始生成 12 个 token，看看越过 256 之后发生了什么。

```widget playground-kv caption="先测生成 1 个新 token 要多久（双对数刻度，成正比的关系在图上是一条直线）：没有 KV cache 时，要重算的位置随上下文长度成正比地增加，注意力那部分还按长度的平方增长，上下文长了以后，时间涨得比正比还快；有 KV cache 时只是缓慢增加。再从 251 个 token 开始生成 12 个：从第 7 个新 token 起，每一个都要重算整个 256 窗口。"
{"checkpoint": "pretrain", "lengths": [8, 16, 32, 64, 128, 192, 256], "cross": 12}
```

注意表里的“倍数”。n = 8 时，没有 KV cache 要算 8 个位置，却远没有慢到 8 倍；n = 256 时，也不到 256 倍。原因第 5 章讲过：decode 一步只算 1 个位置，却要把全部权重读一遍；prefill 读一遍权重，就能同时服务好几个位置。所以实测的加速远低于计算量之比（[验证 5.5-g](ch05.html#v-5.5-g)）。

计时取决于你的设备。与设备无关的是计算量：我们只数“前向传播一共算了多少个位置”：

```verify id=P.7-c title="一共要算多少个位置：KV cache 省下的，以及越过 256 之后失去的"
def positions_computed(n_prompt, n_new, T=256, cache=True):
    """生成 n_new 个 token，前向传播一共要算多少个位置。"""
    total = min(n_prompt, T)                      # 第 1 个新 token：prefill 整个 prompt（最多 T 个）
    for seq in range(n_prompt + 1, n_prompt + n_new):   # 生成后面每一个 token 时，序列已有 seq 个
        total += 1 if cache and seq <= T else min(seq, T)
    return total

for n_prompt, n_new in [(22, 100), (251, 12), (22, 1000)]:
    a, b = positions_computed(n_prompt, n_new), positions_computed(n_prompt, n_new, cache=False)
    print(f"prompt {n_prompt:3d} 个、生成 {n_new:4d} 个，序列长 {n_prompt + n_new:5,}：有 KV cache 算 {a:7,} 个位置，原书的写法算 {b:7,} 个")
assert positions_computed(22, 100) == 121 and positions_computed(22, 100, cache=False) == 7150
assert positions_computed(251, 12) == 1792 and positions_computed(251, 12, cache=False) == 3057
assert positions_computed(22, 1000) == 196_096 and positions_computed(22, 1000, cache=False) == 228_505
```

从 22 个 token 的 prompt（模板 18 个，加上 `3+4=` 的 4 个，P.3 节）生成 100 个 token，有 KV cache 只算 121 个位置，原书的写法要算 7,150 个。而从 251 个 token 开始生成 12 个，只有前 6 个新 token 还在 256 个位置之内（第 1 个来自 prefill，后 5 个各算 1 个位置）；之后的 6 个，每个都要算 256 个位置。图 P-6 下方表格里“累计位置数”一栏，最后一行正是 1,792。

::: confusion 上下文只有 256，为什么能生成上千个 token？
“上下文 256”很容易被理解成“一共只能处理 256 个 token”。可是原书的 `generate()` 默认 `max_new_tokens=1000`，生成上千个 token 也不会报错。这是怎么回事呢？

这里其实有三个数，说的是三件事。用上面的例子来数：从 251 个 token 的 prompt 生成 12 个。第一个数是窗口：每一次前向传播最多看见 256 个 token，“上下文 256”限制的只是它。第二个数是序列的总长：251 + 12 = 263 个 token。它可以一直长下去，只是最早的 token 会被挤出窗口，悄悄忘掉。第三个数是累计计算量：所有前向传播加起来，一共算了 1,792 个位置。

生成上千个 token 时，三个数差得更远。从 22 个 token 的 prompt 生成 1,000 个，窗口还是 256，序列长到 1,022，累计却要算约 19.6 万个位置。有 KV cache 也只省下一小部分（不用 KV cache 约 22.9 万），因为越过 256 之后，每一步都要重算整个窗口。第 5 章的智能体计量表讲的是同一件事（[5.5.5 节](ch05.html#s5-5-5)末尾）。带走一句话：窗口管的是一次，累计算的是总和。
:::

::: cs336 第 10 讲：推理
CS336 的第 10 讲专门讲推理。它从算术强度讲起：prefill 一次处理很多 token，decode 一次只处理一个。接着讲怎样把 KV cache 变小（GQA、MLA 等），怎样把模型变小（量化、剪枝、蒸馏）。最后讲怎样让推理服务更快：投机采样、continuous batching 与 PagedAttention（[讲义](https://cs336.stanford.edu/lectures/?trace=lecture_10)，源码 [lecture_10.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)，查证于 2026-10）。这部分没有对应的作业，作业 5 直接使用 vLLM。DLFS6 里，CodeBot 的 `utils.generate` 每一步重算整个上下文，StoryBot 和 WebBot 的版本才用上了 KV cache。本页的 P.2 与 P.7 分别是量化和 prefill/decode 的最小演示。
:::

## P.8 三个检查点，三种回答

一个 prompt 的一生走完了。现在把三个检查点放在一起：同一串 token、同样的 greedy 解码，只有权重不同。

```widget playground-compare caption="三个检查点回答同一个 prompt。3+4= 这一题，预训练模型写起了注释，SFT 模型答 6，GRPO 模型答 7。下方逐题比较 SFT 与 GRPO：81 道练过的个位数加法，以及 GRPO 没练过的两组题（切换题组即可）。"
{"prompt": "3+4=", "template": true, "maxNewTokens": 48, "set": "train"}
```

预训练模型不认识模板，在后面接着写起了注释。SFT 模型学会了模板的格式，回答简短，答完就停，可是答错了。GRPO 模型答对了。第 3 章的 GRPO 训练用的正是这 81 道题：

```listing ch03/09_grpo.py#L18-L26
```

判断对错的是奖励函数：取文字里的最后一个整数，与正确答案比较。

```listing ch03/09_grpo.py#L64-L72
```

训练时，奖励函数拿到的只是模型的回答。进度条上的准确率，则来自训练循环末尾的这段评估：

```listing ch03/09_grpo.py#L188-L205
```

评估每 10 轮做一次（第 189 行），问的正是那 81 道题。注意第 194～195 行：`generate()` 返回的是“提示 + 回答”的整段文字，奖励函数拿到的也是这一整段（[勘误 E32](errata.html#e32)）。另外，最后一次评估在第 490 轮，模型却是在第 499 轮之后才保存的（第 205 行）。

我们在 PyTorch 里把 81 道题全部问了一遍，greedy 解码，直到模型输出 `<|endoftext|>`（每个回答都在 64 个 token 之内停下，所以与原书默认的上限 1000 等价）。判分与训练时的奖励相同：只把回答交给 `calculate_reward`。除了本书自己的两个检查点，还问了原书的官方权重，并加了两组 GRPO 没练过的题：含 0 的 19 道，以及两位数加一位数的 81 道。

::: think GRPO 没练过的题，它会怎么回答？
GRPO 只练过 1 到 9 的个位数加法。它没见过含 0 的题（比如 0+5=），也没见过两位数（比如 11+3=）。先猜一猜，再看下面验证单元输出里以“GRPO 的回答：”开头的那一行。
:::
::: answer
`0+5=` 它答 `10`。含 0 的 19 道题，它一道也没答对，0 从来没有出现在训练的题目里。两位数的题更说明问题：`11+3=` 它答 `4`，`12+5=` 答 `6`，`15+4=` 答 `5`。81 道两位数的题里，有 48 道的回答恰好是 1 + b，就好像它只读了第一个数的第一位。模型并没有学会“加法”，它学会的是这 81 道题里的 70 道。
:::

```verify id=P.8-a title="81 道练过的加法与两组没练过的题：各个检查点答对多少" files=data/playground/addition.json
import re

add = json.load(open("data/playground/addition.json", encoding="utf-8"))
res = lambda name, s="train": add["models"][name]["results"][s]
print("train = 练过的 81 道，zero = 含 0 的 19 道，teens = 两位数加一位数的 81 道；ours = 本书训练，official = 官方权重，q8 = 浏览器里的 int8")
print(f"{'':18}" + "".join(f"{s:>10}" for s in add["sets"]))
for name in add["models"]:
    print(f"{name:18}" + "".join(f"{res(name, s)['n_correct']:>6}/{len(add['sets'][s]):<3}" for s in add["sets"]))
assert res("ours-sft-f32")["n_correct"] == 44 and res("ours-grpo-f32")["n_correct"] == 70
assert res("official-sft-f32")["n_correct"] == 44 and res("official-grpo-f32")["n_correct"] == 67
diff = [(n, s, p, a, b) for n in ["sft", "grpo"] for s in add["sets"]
        for p, a, b in zip(add["sets"][s], res(f"ours-{n}-q8", s)["responses"], res(f"ours-{n}-f32", s)["responses"]) if a != b]
print(f"int8 与 float32 的 {2 * sum(len(v) for v in add['sets'].values())} 对回答里，不同的（检查点, 题组, 题, int8, float32）：", diff)
assert diff == [("sft", "teens", "14+6=", "7", "9")]
assert all(res(name, s)["n_correct"] == 0 for name in add["models"] for s in ["zero", "teens"])
pick = lambda s, p: res("ours-grpo-f32", s)["responses"][add["sets"][s].index(p)]
print("GRPO 的回答：", {p: pick("teens", p) for p in ["11+3=", "12+5=", "15+4="]}, {"0+5=": pick("zero", "0+5=")})
teens = [(p, int(p[:-1].split("+")[1])) for p in add["sets"]["teens"]]
one_plus_b = sum(pick("teens", p) == str(1 + b) for p, b in teens)       # 回答恰好是 1 + b 的题数
print(f"两位数的题里，GRPO 回答恰好是 1 + b 的：{one_plus_b}/81；70/81 = {70 / 81:.1%}")
assert [pick("teens", p) for p in ["11+3=", "12+5=", "15+4="]] == ["4", "6", "5"]
assert pick("zero", "0+5=") == "10" and one_plus_b == 48

i34 = add["sets"]["train"].index("3+4=")              # 图 P-7 的例子：三个检查点对 3+4= 的 greedy 回答
math_pretrain = decode(next(p for p in refs["pretrain"]["prompts"] if p["name"] == "math")["q8"]["greedy"])
print("3+4= →", {"预训练": math_pretrain[:22], "SFT": res("ours-sft-q8")["responses"][i34], "GRPO": res("ours-grpo-q8")["responses"][i34]})
assert math_pretrain.startswith("#") and res("ours-sft-q8")["responses"][i34] == "6" and res("ours-grpo-q8")["responses"][i34] == "7"

def calculate_reward(ground_truth, response):     # 09_grpo.py 的奖励函数（上面的代码清单），去掉了 try/except
    matches = re.findall(r"(-?\d+)", response)
    return 1.0 if matches and int(matches[-1]) == ground_truth else 0.0

def n_right(name, s, whole):                      # whole=True：像 09_grpo.py 的评估那样，连提示一起交给奖励函数
    return sum(calculate_reward(gt, (f"### Instruction:\n{p}\n\n### Response:\n" if whole else "") + r) > 0
               for p, gt, r in zip(add["sets"][s], add["answers"][s], res(name, s)["responses"]))

assert all(n_right(name, s, False) == res(name, s)["n_correct"] for name in add["models"] for s in add["sets"])
assert all(n_right(name, "train", True) == res(name)["n_correct"] for name in add["models"])   # 练过的题：两种判法相同
print(f"连提示一起交给奖励函数：GRPO 在含 0 的题上答对 {n_right('ours-grpo-f32', 'zero', True)}/19（它对 0+3= 的回答是 {pick('zero', '0+3=')!r}）")
assert n_right("ours-grpo-f32", "zero", True) == 1 and pick("zero", "0+3=") == ""
```

先看“练过的 81 道”一列（train）。本书的 SFT 模型答对 44 道，GRPO 模型答对 70 道（86.4%），与进度条最后显示的数字相同。不过，这是两次测量恰好一致：进度条上的 86.4% 来自第 490 轮的评估，之后又更新了 9 轮才保存模型，我们重测的是保存下来的模型。

官方权重是 44 道和 67 道，与第 3 章测得的一样（[3.4.2 节](ch03.html#s3-4-2)）。int8 的两行（`-q8`）在 81 道练过的题上，与 float32 的回答逐题相同；全部 362 对回答里只有一对不同（SFT 模型对 `14+6=` 的回答，int8 是 7，float32 是 9，都不对）。在浏览器里点“逐题测试”，得到的也是 44 和 70，而且每一道题的回答都与 PyTorch 用同一份 int8 权重算出的相同（P.9 节的本地验证会在 Node 里把 181 道题重算一遍）。

判分的写法，会影响没练过的题。如果照搬 `09_grpo.py` 评估时的写法，把“提示 + 回答”整段交给奖励函数，GRPO 对 `0+3=` 什么也没答，整段文字里的最后一个整数却恰好是题目里的 3，于是被判为答对：含 0 的题从 0 道变成了 1 道（单元输出的最后一行）。在 81 道练过的题上，两种判法的结果相同。在图 P-7 下方切换到“含 0 的 19 道”，可以看到每一格的回答。

```localcheck verify/test_playground.py::test_addition_record_is_current title="重新问一遍 6 个模型的全部 181 道题，与 addition.json 逐题相同"
```

::: confusion 进度条上的 86.4%，能说明模型学会了加法吗？
GRPO 训练结束时，进度条上的 `acc` 是 86.4%，很容易把它当成“模型的加法能力”。可是 `09_grpo.py` 评估用的，正是训练时反复练习的那 81 道题，而且这 81 道题连同答案，早就出现在 SFT 的数据里了（[验证 3.3-a](ch03.html#v-3.3-a)）。这个数字说明的是“练过的题答对了多少”，相当于把作业题原样拿来当考试。

要知道模型学到了什么，得拿它没见过的题来考，也就是留出评估（held-out evaluation）。上面加的两组题就是最简单的留出集：结果是 0 分。

严格地讲，这两组题不只是“没见过的题”，也是“没见过的形式”：多了 0，多了一位数。所以这个 0 分同时说明了两件事，没法分开。更干净的做法是从 81 道里留出一部分，SFT 和 GRPO 都不让它见到，再拿它来考。CodeBot 做不到这一点，因为这 81 道题全都在 SFT 数据里。

训练中看到的准确率是训练信号，告诉我们优化在推进；它不能代替留出评估，告诉我们模型能做什么。
:::

::: cs336 作业 5：GRPO 的评估方式
CS336 的作业 5 也做 GRPO，但评估方式不同：在留出的验证集上评估，并且用 4 个随机种子重复实验（[作业 5 的仓库](https://github.com/stanford-cs336/assignment5-alignment)，查证于 2026-10）。DLFS6 的 `09_grpo.py` 则直接在训练用的 81 道题上评估。对一个只为演示 GRPO 机制的玩具任务来说，这样做无可厚非，只是读数时要记住它的含义。
:::

## P.9 与 PyTorch 对拍

最后，我们来确认工作台里的这份 JavaScript，算的确实是原书的 CodeBot。点下面的按钮，工作台会把 P.2.3 节那 4 个 prompt 在你的浏览器里重算一遍，与 PyTorch 记录的结果逐项比较：最后一个位置的 1,000 个 logits、greedy 生成的 16 到 32 个 token、`def add(a, b):` 的 36 张注意力图，以及每一层的 logit lens。

```widget playground-parity caption="与 PyTorch 对拍。int8 一栏与使用同一份 int8 权重的 PyTorch 比较，检验的是实现；float32 一栏与原始权重比较，差别来自量化。"
{"checkpoint": "grpo", "tolerance": 0.0001}
```

参考结果有两份，各回答一个问题。与“同一份 int8 权重的 PyTorch”比较，检验 JavaScript 的实现对不对：这时 logits 最多差 1.5×10⁻⁵ 左右，greedy 的 token 全部相同。与“原始 float32 权重的 PyTorch”比较，看到的是量化带来的差别：logits 差到 0.1 到 0.3，greedy 序列有时会在中途分岔（P.2.3 节）。

浏览器里的按钮，检查的只是你这台设备上的这一次运行。为了让每次修改代码之后都能复查，同样的比较也写成了测试：JavaScript 一侧用 Node 运行，PyTorch 一侧由 pytest 重新生成参考结果（P.4 节的本地验证）。下面这个本地验证把 Node 的测试跑一遍，再核对它们报告的数字：三个检查点、12 个参考 prompt 的 logits，greedy 生成的 token，以及 P.8 节的 181 道加法题。

```localcheck verify/test_playground.py::test_javascript_matches_pytorch needs="Node.js" title="用 Node 运行工作台的 JavaScript：logits 与同一份 int8 权重的 PyTorch 相差不到 2×10⁻⁵，每个检查点 112 个 greedy token、181 道加法的回答全部相同"
```

那么，用的是同一份权重，为什么还会差 10⁻⁵ 呢？原因是浮点舍入。同样的运算，换一种顺序、换一种精度来做，舍入的结果就不一样，而两边舍入的地方并不相同：PyTorch 调用的矩阵库有它自己的加法顺序；工作台的 LayerNorm 和 softmax 先用 float64 算，再存回 float32；GELU 里的 erf 用的是一个近似公式（Abramowitz–Stegun 7.1.26）。

下面拿其中最容易看清的一种，加法的顺序，举个例子。浮点数的加法不满足结合律：384 或 1,536 个乘积相加，先加哪个、后加哪个，float32 的结果就会差一点。默认的 WASM SIMD 内核给每个输出配 4 条通道，各自累加，最后按 (l0 + l1) + (l2 + l3) 合起来：

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

纯 JavaScript 后端也是 4 路累加，只是 JavaScript 的数是 float64，舍入又发生在别的地方。在 9 个较短的参考 prompt 上，它与 PyTorch 同样只差 10⁻⁵ 量级，greedy token 同样全部相同（上面的本地验证；图 P-8 也可以切换后端试试）。

只要第一名与第二名之间的差距比这种舍入误差大，greedy 选出的 token 就不会变。12 个参考 prompt 的最后一个位置上，这个差距最小是 0.004，是 10⁻⁵ 的几百倍。当然，这只是 12 个位置；生成过程中每一步是否都有这样的余量，要靠上面的对拍来回答：每个检查点的 112 个 greedy token 全部相同。

想自己从头复查一遍，就运行这两条命令：

```shell
node --test tests/js/playground-gpt.test.mjs tests/js/playground-parity.test.mjs tests/js/playground-engine.test.mjs
uv run pytest verify/test_playground.py
```

## 小结

这一页，我们跟着一个 prompt 把 CodeBot 从头到尾走了一遍。文字先套上第 3 章的模板，再被第 1 章的分词器切成 token；每个 token 查表得到向量，加上位置 embedding，再穿过 6 个块，每个块都只往残差流上“加”一点东西；最后一个位置的向量变成 1,000 个 logits，经过温度、Top-k 和 Top-p，抽出下一个 token。为了让这一切在浏览器里跑起来，权重被按行量化成 int8，推理用 KV cache 一步一步地做，而 CodeBot 的可学习位置 embedding，给它立了一堵 256 个位置的墙。

三个检查点的比较说明了第 3 章三个阶段各自的作用：预训练学会续写，SFT 学会格式，GRPO 把练过的题答对，但也只限于练过的题。

本章所学的内容：

- 工作台的三个检查点出自本书自己的训练：清单里的训练记录与运行报告一致，分片与 checkpoint 也逐行核对过（[验证 P.1-a](#v-P.1-a)、[P.2.2 节的本地验证](#sP-2-2)）。
- CodeBot 的 11,121,640 个参数，按行量化成 int8 后是 11.3 MB，约为 float32 的四分之一（[验证 P.2-a](#v-P.2-a)）；每个权重的误差不超过半个 scale，一行离群也不会拖累别的行。logits 仍会偏移：12 个参考 prompt 的最后一个位置上最多 0.29，TinyCodes 的任意位置上可以到 0.4 左右；greedy 序列可能在近乎打平的位置分岔（[验证 P.2-b](#v-P.2-b)、[验证 P.2-c](#v-P.2-c)、[验证 P.2-d](#v-P.2-d)、[P.2.3 节的本地验证](#sP-2-3)）。
- logit lens 的第 0 层几乎总是“预测”当前 token 自己：TinyCodes 里出现过的 839 个 token 无一例外，从没出现过的 token（例如汉字的 UTF-8 字节）则不一定；这与权重共享有关（[验证 P.4-a](#v-P.4-a)、[P.4.2 节的本地验证](#sP-4-2)）。第 1 层第 1 个头是一个“前一个 token 头”（[验证 P.5-a](#v-P.5-a)）。
- Top-p 量的是 Top-k 剩下的部分，所以两者的先后顺序会改变结果；熵用自然对数算，单位是 nats（[验证 P.6-a](#v-P.6-a)、[验证 P.6-b](#v-P.6-b)）。
- KV cache 与从头重算的结果相同。越过 256 之后，从左边截断让缓存整体作废：位置平移了，被截掉的 token 也已混进第 2 层起的 K、V，所以每个 token 都要重算整个窗口（[验证 P.7-a](#v-P.7-a)、[验证 P.7-b](#v-P.7-b)、[验证 P.7-c](#v-P.7-c)）。
- GRPO 把 81 道练过的题从 44 道提高到 70 道，没练过的题一道也答不对（只看回答来判分）；训练中的准确率不能代替留出评估（[验证 P.8-a](#v-P.8-a)）。
- 同一份 int8 权重下，JavaScript 与 PyTorch 只差 10⁻⁵ 量级的浮点舍入，远小于第一名领先的幅度，所以 greedy 生成的 token 逐个相同（[验证 P.9-a](#v-P.9-a)、[P.9 节的本地验证](#sP-9)）。

工作台里的 CodeBot 是在 GPU 上训练好的，我们只是看着它推理。训练本身能不能也搬进浏览器呢？下一页，我们就在这个标签页里，从零训练一个迷你 GPT。
