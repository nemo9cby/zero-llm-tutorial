---
chapter: III
slug: part3
order: 75
kind: part
title: 第 III 部 挑战篇：制作 WebBot
nav_title: 第 III 部 挑战篇：制作 WebBot
lede: 第三台机器人 WebBot 读的是 12 GB 的真实网页文本，模型有 1.5 亿个参数，要用数据中心级的 GPU 训练。原书在这一部只给提示、不给完整步骤；本教程照样一节一节地写，并在每一节配上能在本地运行的最小验证。
pyodide_packages: [numpy]
---

最后一台机器人是 **WebBot**。它面对的不再是整理好的小程序或儿童故事，而是真实的网页文本：有噪声，文体五花八门，话题无所不包。前两台机器人用过的技术，在这里还管用吗？这是第 III 部要回答的问题。

原书在这一部换了写法：它明确表示不再给出详细的实现步骤，只提示需要思考的要点和方向，请读者调动前两部学到的知识，自己把 WebBot 做出来。所以第 7–9 章的代码很少：仓库里只有分词器的两个脚本、`webbot/` 下的模型和生成函数，以及训练和生成的两个脚本，第 8 章甚至没有代码目录。本教程照样按原书的小节来写，凡是原书只给提示的地方，都配上一段可以在本地运行的最小验证，并在 [GPU 实战手册](gpu.html) 里给出真正去训练的命令和预算。

## 先看起点

前两部的扉页都先展示了完成后的机器人。WebBot 做不到这一点：原书作者没有发布 WebBot 的权重（2026-10-11 查证时，Hugging Face 上的 `webbot/model_pretrain.pt` 返回 404），本书的正式训练也要等 GPU 实验完成。所以我们反过来，先看看训练开始之前的 WebBot。

下面用的是原书 `ch09/02_generate.py` 里的 5 个提示词和它的温度 0.5，模型按 `ch09/01_pretrain.py` 的超参数构造、随机初始化（种子 0），每个提示词生成 30 个 token（原书生成 100 个，这里截短是为了让本地验证跑得快）：

```verify id=III-a title="未训练的 WebBot：输出与提示词几乎无关" files=data/part3/preview.json
import json
import itertools
import math

rec = json.load(open("data/part3/preview.json", encoding="utf-8"))
for prompt, out in zip(rec["prompts"], rec["outputs"]):
    print(f"{prompt}｜{out[len(prompt):][:70]}…")
ids = [set(x) for x in rec["continuation_ids"]]
shared = [len(a & b) for a, b in itertools.combinations(ids, 2)]
n = sum(len(x) for x in ids) / len(ids)
print(f"两两之间平均有 {sum(shared) / len(shared):.1f} 个相同的 token（每段约 {n:.0f} 个）；"
      f"若从 50,000 个 token 里随机各抽 {n:.0f} 个，期望只有 {n * n / 50_000:.3f} 个相同")
print(f"初始 loss {rec['init_loss']:.4f}，ln 50000 = {math.log(50_000):.4f}")
assert min(shared) >= 9 and abs(rec["init_loss"] - math.log(50_000)) < 0.2
```

五段输出都是语法和意义都不成立的词串，这在意料之中。值得注意的是它们彼此很像：任意两段之间平均有 11 个相同的 token，而随机抽的话几乎不会有重合。随机初始化的模型几乎不看上下文，不管提示词是什么，冒出来的都是差不多的一串词。WebBot 训练的全部意义，就是让这个分布学会依赖前文。初始 loss 也和第 0 章说的一样，比 ln 50000 略高一点。

```localcheck verify/test_part3.py::test_webbot_untrained_preview_matches_record title="按原书超参数构造 WebBot，用原书的提示词和温度重新生成，与录制结果逐字比较"
```

## III.1 数据集

WebBot 的数据是 **OpenWebText**（OWT）。它的来历要从 GPT-2 说起：OpenAI 训练 GPT-2 时，收集了 Reddit 上至少获得 3 个赞（karma）的帖子所链接的网页，称为 WebText（见 [GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 2.1 节），但没有公开。OpenWebText 是 Gokaslan 与 Cohen 按同样的思路重新收集的开放版本（[OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/)）。原书用的是斯坦福 CS336 课程整理的样本 [stanford-cs336/owt-sample](https://huggingface.co/datasets/stanford-cs336/owt-sample)，[koki0702/zero-llm-data](https://huggingface.co/datasets/koki0702/zero-llm-data) 的说明标注它与 OpenWebText 都是 MIT 许可。

| | 训练集 | 验证集 |
|---|---|---|
| 文件大小 | 11,920,511,059 字节 | 289,998,753 字节 |
| token 数（WebBot 分词器） | 2,649,004,800 | 64,496,066 |
| 文档篇数 | — | 59,060 |

网页文本有各自的版权，所以本书不附带 OWT 的文本样本，只引用一句话。下面是验证集第 10 篇文档（一篇物理新闻）的第一句：

```verify id=III.1-a title="OWT 验证集里的一句话" files=data/part3/tokenizers.json
tk = json.load(open("data/part3/tokenizers.json", encoding="utf-8"))
print(tk["sentence"])
print("来源：", tk["sentence_source"])
assert tk["sentence"].startswith("Two separate teams of researchers") and "“Weyl fermion.”" in tk["sentence"]
```

一句话里就有弯引号这样的非 ASCII 字符，这在 TinyCodes 里一个都没有。网页的长度也和前两份数据完全不同。下图把三份数据的文档长度分布放在一起，竖线是 CodeBot、StoryBot 的上下文长度 256 和 WebBot 的 1,024：

```widget part1-doclen caption="网页比程序和故事长得多：OWT 验证集的中位数是 696 个 token，约三分之一的文档比上下文长度 1,024 还长。"
{"datasets": [{"name": "OpenWebText（验证集）", "file": "data/part3/doclen.json"}, {"name": "TinyStories（验证集）", "file": "data/part2/doclen.json"}, {"name": "TinyCodes", "file": "data/part1/doclen.json"}], "xMax": 2048}
```

```verify id=III.1-b title="32.0% 的网页文档超过上下文长度 1,024" files=data/part3/doclen.json
import numpy as np

owt = json.load(open("data/part3/doclen.json", encoding="utf-8"))
L = np.array(owt["lengths"])
print(f"{len(L):,} 篇文档，中位数 {np.median(L):.0f}，平均 {L.mean():.0f}，最长 {L.max():,} 个 token")
for ctx in (256, 1024, 2048):
    print(f"不超过 {ctx:>5,} 个 token 的文档：{(L <= ctx).mean():.1%}")
print(f"前 200 万个 token 解码后，平均每个 token {owt['bytes_per_token_first_2M']:.2f} 字节")
assert len(L) == 59_060 and np.median(L) == 696 and (L > 1024).sum() == 18_925
```

```localcheck verify/test_part3.py::test_owt_valid_stats title="在完整验证集上重算文档长度与压缩率"
```

数据变了，前两台机器人的分词器还能用吗？算法是能用的：第 7 章训练 WebBot 的分词器，直接调用第 4 章写好的 `storybot.tokenizer.train_bpe`。但学到的词表不能照搬。下图让三台机器人的分词器读同一句网页文本，再比较它们在三种数据上的压缩率：

```widget part3-tokenizers caption="CodeBot 的分词器把这句话切成 64 个 token，WebBot 的只要 26 个。每个分词器都在自己的数据上更省，而词表越大，整体越省。"
{}
```

```verify id=III.1-c title="压缩率（字节 / token）：在网页文本上，WebBot 的分词器是 CodeBot 的 2.3 倍"
names = ["codebot", "storybot", "webbot"]
for row in tk["matrix"]:
    print(f"{row['corpus']:<12}" + "".join(f"{n:>9} {row['bytes'] / row['tokens'][n]:.2f}" for n in names))
web = next(r for r in tk["matrix"] if r["corpus"] == "OpenWebText")
story = next(r for r in tk["matrix"] if r["corpus"] == "TinyStories")
code = next(r for r in tk["matrix"] if r["corpus"] == "TinyCodes")
print(f"同一份网页文本，CodeBot 的分词器切出的 token 是 WebBot 的 {web['tokens']['codebot'] / web['tokens']['webbot']:.2f} 倍")
assert {n: len(c) for n, c in tk["chips"].items()} == {"codebot": 64, "storybot": 34, "webbot": 26}
assert min(web["tokens"], key=web["tokens"].get) == "webbot"           # 网页上 WebBot 最省
assert min(story["tokens"], key=story["tokens"].get) == "storybot"     # 故事上 StoryBot 最省
assert code["tokens"]["webbot"] < code["tokens"]["codebot"]            # 连代码上，5 万词表也比 1,000 词表省
```

结果有两层意思。第一，**领域**很重要：在故事上，只有 1 万词表的 StoryBot 分词器反而比 5 万词表的 WebBot 略胜一筹；用 TinyStories 训练出的词表放到网页上，每个 token 只装 3.41 个字节。第二，**词表大小**也很重要：CodeBot 只有 1,000 个 token，哪怕在自己的代码数据上，也比不过 5 万词表的 WebBot 分词器。同一篇网页，用 CodeBot 的分词器要多出 1.3 倍的 token，模型读到的序列也就长了一倍多。WebBot 需要在网页文本上重新训练一个更大的词表，这是第 7 章的起点。

```localcheck verify/test_part3.py::test_tokenizer_matrix_matches_record title="用三个分词器重新编码，并核对第 1 章与第 4 章两种编码实现的结果相同"
```

::: confusion 上下文长度 1,024 是谁定的？
读训练脚本时，`context_len = 1024`、`n_layer = 12` 这样的数字看起来像是随手写下的“魔法数字”。想知道它们从哪来，是很自然的疑问。答案通常有三个出处：照搬一个成熟的配置、看数据的统计、算算力的账。WebBot 的上下文长度三样都能对上。

第一，它和 GPT-2 small 的“身材”一模一样。Hugging Face 上 [GPT-2 的配置文件](https://huggingface.co/openai-community/gpt2/blob/main/config.json) 写着 `n_ctx` 1024、`n_embd` 768、`n_layer` 12、`n_head` 12、词表 50,257（2026-10-11 查证），WebBot 只是把词表换成了自己训练的 50,000。第二，数据：上下文取 1,024 时，68% 的网页能被一眼看完；取 256 时只有 9%。第三，算力：每个头的 Attention 分数是一张上下文长度 × 上下文长度的表，1,024 比 256 大 16 倍。

```verify id=III.1-d title="1,024 的三个出处：GPT-2 small、数据、算力" files=data/ch00/bots.json
gpt2_small = {"n_ctx": 1024, "n_embd": 768, "n_layer": 12, "n_head": 12, "vocab_size": 50257}  # HF 配置文件
bots = {b["id"]: b for b in json.load(open("data/ch00/bots.json", encoding="utf-8"))["bots"]}
web_cfg = bots["webbot"]["model"]
same = {"n_ctx": "context_len", "n_embd": "embed_dim", "n_layer": "n_layer", "n_head": "n_head"}
for k, v in same.items():
    print(f"GPT-2 small {k:<8} = {gpt2_small[k]:>5}   WebBot {v:<11} = {web_cfg[v]:>5}")
    assert gpt2_small[k] == web_cfg[v]
for ctx in (256, 1024):
    print(f"上下文 {ctx:>5}：{(L <= ctx).mean():.0%} 的网页能完整放进去；每个头的 Attention 表有 {ctx * ctx:,} 个数")
assert round((L <= 1024).mean(), 2) == 0.68 and round((L <= 256).mean(), 2) == 0.09
assert (1024 * 1024) // (256 * 256) == 16
```

可以带走的一句话：**每个超参数背后都应该有一个出处**。读脚本时先问“这个数从哪来”：是抄自一个成熟的配置，还是由数据或算力决定的？问不出来的，就是值得做实验的地方。
:::

## III.2 开发流程

WebBot 的三章对应原书的 7.1–7.6、8.1–8.6、9.1–9.7 节。下表标出了每一节在原书仓库里有没有代码：

| 章 | 原书的小节 | 仓库里的代码 |
|---|---|---|
| [第 7 章](ch07.html) 分词器 | 7.1 WebBot 的分词器、7.2 定位瓶颈、7.3 用 Rust/C++ 加速、7.4 添加特殊 token、7.5 其他分词器、7.6 不用分词器的方法 | 7.1 有（`ch07/01_bpe_train.py`、`02_bpe_encode.py`），其余只有提示 |
| [第 8 章](ch08.html) 模型 | 8.1 选择模型规模、8.2 Flash Attention、8.3 GQA、8.4 MoE、8.5 torch.compile、8.6 扩展上下文长度 | 没有 `ch08/` 目录；8.2、8.3 写在 `webbot/model.py` 里，8.5 写在 `ch09/01_pretrain.py` 里；8.4、8.6 没有代码 |
| [第 9 章](ch09.html) 训练 | 9.1 梯度累积、9.2 梯度检查点、9.3 分布式训练、9.4 ZeRO、9.5 Muon 优化器、9.6 Weights & Biases、9.7 训练与生成结果 | 9.1、9.3、9.6、9.7 在 `ch09/01_pretrain.py`、`02_generate.py` 里；9.2、9.4、9.5 没有代码 |

```localcheck verify/test_part3.py::test_hint_only_topics title="在原书仓库里确认：哪些主题有代码，哪些没有"
```

模型是 GPT-2 small 的身材，配上 StoryBot 的现代零件（RoPE、SwiGLU、RMSNorm，无 bias，不共享权重），再加上 GQA：12 个查询头只配 4 组键和值。参数一共 152,316,672 个，其中整整一半在词嵌入和输出层这两张 50,000 × 768 的大表里：

```verify id=III.2-a title="WebBot 的参数有 50.4% 在词嵌入和输出层"
split = bots["webbot"]["params_split"]
table = split["embedding"] + split["output_extra"]
print(f"总参数 {bots['webbot']['params']:,}；词嵌入 + 输出层 {table:,} = 2 × 50,000 × 768，占 {table / bots['webbot']['params']:.1%}")
assert table == 2 * 50_000 * 768 and round(table / bots["webbot"]["params"], 3) == 0.504
```

训练的超参数写在 `ch09/01_pretrain.py` 里。每张 GPU 每一步处理 32 条序列 × 梯度累积 4 次 × 1,024 个 token：

```listing ch09/01_pretrain.py#L97-L113
```

::: think 用 8 张 GPU 跑这个脚本，会比 1 张快 8 倍吗？
:::
::: answer
不会。脚本从不读取 GPU 的张数，每张卡都要跑满 `max_iters = 100000` 步。多卡并行时，每张卡处理不同的数据，梯度取平均后一起更新，所以墙钟时间和单卡差不多，读过的 token 却变成了 8 倍：从 131 亿变成 1,049 亿。全局 batch 也一样变大：32 条 × 4 次累积 × 8 张卡 = 1,024 条序列。想在 8 张卡上保持单卡的 token 预算，要把 `max_iters` 除以 8，改成 12,500。第 9 章讲分布式训练时会回到这一点。
:::

```verify id=III.2-b title="加卡等于加 token：每张卡都跑满 10 万步"
per_gpu_step = 32 * 4 * 1024
for gpus in (1, 8):
    print(f"{gpus} 张 GPU：每步全局 {32 * 4 * gpus:,} 条序列，共 {100_000 * per_gpu_step * gpus:,} 个 token")
assert 100_000 * per_gpu_step == bots["webbot"]["pretrain"]["tokens"] == 13_107_200_000
assert (100_000 // 8) * per_gpu_step * 8 == 13_107_200_000      # max_iters 改成 12,500 才能保持预算
```

::: gpu 在一张 H100 上要多久？
按 $6ND$ 粗算，WebBot 的单卡预训练约需 1.2 × 10¹⁹ 次浮点运算。H100（SXM）的稠密 BF16 峰值约 989 TFLOP/s（[NVIDIA 数据手册](https://www.nvidia.com/en-us/data-center/h100/)上的 1,979 是含稀疏加速的数字）。实际训练达不到峰值，这里假设能用上 35%，这是估算里最关键、也最不确定的一个数。

```verify id=III.2-c title="估算：一张 H100 约 10 小时（假设 35% 的峰值利用率）"
flops = 6 * bots["webbot"]["params"] * bots["webbot"]["pretrain"]["tokens"]
peak, mfu, price = 989e12, 0.35, 3.95          # FLOP/s；假设的利用率；Modal H100 每小时美元（2026-10-11）
hours = flops / (peak * mfu) / 3600
print(f"6ND = {flops:.2e} FLOPs → {hours:.1f} 小时 → 约 {hours * price:.0f} 美元（不含验证集评估）")
assert 9 < hours < 10.5
```

价格取自 [Modal 的价目表](https://modal.com/pricing)（2026-10-11）。这只是开工前的估算：先在云上跑 200 步，从进度条读出真实的速度，再决定要不要跑完整的 10 万步。计时探针的结果和完整的运行命令都在 [GPU 实战手册](gpu.html) 里。
:::

::: cs336 CS336 用的是同一份网页数据
CS336 的 A1 也在这份 OWT 样本上训练分词器和语言模型，验证集文件与原书的 `owt_valid.txt` 逐字节相同（研究阶段核对过 SHA-256）；不过 A1 在 OWT 上训练的 BPE 词表是 32,000，而不是 WebBot 的 50,000。原书在第 8、9 章只给提示的主题，大多能在 CS336 里找到动手的版本：A2 要用 Triton 写 FlashAttention2、实现 DDP 和优化器状态切分，A3 要用实验拟合 scaling law。资料来源：[CS336 课程主页](https://cs336.stanford.edu/)。
:::

## 本部要点

第 III 部没有现成的终点可看，但起点和路线都能验证：未训练的 WebBot 不看上下文，只会吐出差不多的词串；它要学的网页文本又长又杂，三分之一的文档比上下文还长；前两台机器人的分词器算法可以复用，词表却必须在网页上重新训练；而整个预训练需要约 10¹⁹ 次浮点运算，是数据中心 GPU 上以小时计的工作量。

**本部所学的内容**

- 未训练的 WebBot 对 5 个提示词给出几乎相同的词串，初始 loss 比 ln 50000 略高（[验证 III-a](#v-III-a)）。
- OWT 验证集有 59,060 篇文档，中位数 696 个 token，32.0% 超过上下文长度 1,024（[验证 III.1-a](#v-III.1-a)、[验证 III.1-b](#v-III.1-b)）。
- 在网页文本上，WebBot 的分词器每个 token 装 4.67 字节，CodeBot 的只装 2.00 字节；领域和词表大小都有影响（[验证 III.1-c](#v-III.1-c)）。
- 上下文长度 1,024 照搬自 GPT-2 small，也能从数据和算力上讲通（[验证 III.1-d](#v-III.1-d)）。
- WebBot 的参数一半在词嵌入和输出层；加 GPU 不会缩短训练时间，而是增加训练的 token 数（[验证 III.2-a](#v-III.2-a)、[验证 III.2-b](#v-III.2-b)）。
- 一张 H100 上的单卡预训练估计约 10 小时，前提是 35% 的峰值利用率（[验证 III.2-c](#v-III.2-c)）。

第 7 章的第一步，就是在 12 GB 的网页文本上训练一个 5 万词表的分词器。光是这一步，就会让第 4 章的加速成果接受一次真正的考验。
