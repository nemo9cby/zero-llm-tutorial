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

原书在这一部换了写法：它明确表示不再给出详细的实现步骤，只提示需要思考的要点和方向，请读者调动前两部学到的知识，自己把 WebBot 做出来（[原书第 III 部扉页的试读](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9784814401611/files/part3.xhtml)，2026-10-12 查证）。所以第 7–9 章的代码很少：仓库里只有分词器的两个脚本、`webbot/` 下的模型和生成函数，以及训练和生成的两个脚本，第 8 章甚至没有代码目录。本教程照样按原书的小节来写，凡是原书只给提示的地方，都配上一段可以在本地运行的最小验证，并在 [GPU 实战手册](gpu.html) 里给出真正去训练的命令和预算。

前两部的扉页都先展示了完成后的机器人。WebBot 做不到这一点：原书作者没有发布 WebBot 的权重（2026-10-12 查证时，Hugging Face 的[文件列表](https://huggingface.co/api/datasets/koki0702/zero-llm-data/tree/main?recursive=true)里，`webbot/` 下只有数据和分词器，没有 `model_pretrain.pt`）；本书也只跑了正式训练的前 200 步，完整的预训练还没有跑。所以我们反过来，先看看训练开始之前的 WebBot。

下面用的是原书 `ch09/02_generate.py` 里的 5 个提示词和它的温度 0.5，模型按 `ch09/01_pretrain.py` 的超参数构造、随机初始化（种子 0），每个提示词生成 30 个 token（原书生成 100 个，这里截短是为了让本地验证跑得快）。有一点要先说明：每个提示词生成之前，我们都把随机种子重置为 0，所以五次抽样用的是同一串随机数。为了对照，我们又给每个提示词换一个种子（100–104），各生成一次：

```verify id=III-a title="未训练的 WebBot：分布近乎平坦，五段输出的重合来自同一个随机种子" files=data/part3/preview.json
import json
import itertools
import math

rec = json.load(open("data/part3/preview.json", encoding="utf-8"))
for prompt, out in zip(rec["prompts"], rec["outputs"]):
    print(f"{prompt}｜{out[len(prompt):][:70]}…")

def shared(ids):                       # 两两之间有几个相同的 token（按集合计）
    return [len(set(a) & set(b)) for a, b in itertools.combinations(ids, 2)]

def tv(p, q):                          # 总变差距离：½ Σ |p − q|
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))

same, indep = shared(rec["continuation_ids"]), shared(rec["continuation_ids_indep"])
n = sum(len(x) for x in rec["continuation_ids"]) / 5
print(f"五段都用种子 0：两两之间平均有 {sum(same) / len(same):.1f} 个相同的 token（每段约 {n:.0f} 个）")
print(f"每段换一个种子：两两之间平均有 {sum(indep) / len(indep):.1f} 个相同的 token")
fs = rec["first_step"]
print(f"第一个新 token 的分布：最大的概率只有 {min(fs['max_prob']):.3%}–{max(fs['max_prob']):.3%}"
      f"（均匀分布是 {1 / fs['vocab_size']:.3%}）")
print(f"五个提示词的分布，两两之间的总变差距离 {min(fs['tv']):.2f}–{max(fs['tv']):.2f}")
print(f"初始 loss {rec['init_loss']:.4f}，ln 50000 = {math.log(50_000):.4f}")
assert tv([0.5, 0.5], [1.0, 0.0]) == 0.5                 # 手算：要把 0.5 的概率从第 2 项挪到第 1 项
assert min(same) >= 9 and max(indep) == 0                # 重合只在共用同一串随机数时出现
assert max(fs["max_prob"]) < 0.002 and min(fs["tv"]) > 0.5
assert abs(rec["init_loss"] - math.log(50_000)) < 0.2
```

五段输出都是语法和意义都不成立的词串，这在意料之中：未训练的模型给出的分布几乎是平的，最可能的那个 token 也不到 0.2% 的概率，抽样就像在 5 万个词里随手抓。

那么，五段输出为什么有三分之一的 token 相同呢？原因不在模型，而在随机数：五次生成用的是同一串随机数。直观上，分布越平，抽中哪个 token 越是由随机数决定，同一串随机数也就常常抽中同一个 token；第 II 部扉页的 StoryBot 也有这个现象。每段换一个种子，两两之间相同的 token 就降到了 0。严格地讲，模型并没有无视提示词。我们用**总变差距离**（total variation distance）$\tfrac12\sum_i |p_i - q_i|$ 衡量两个分布差多远：它是把一个分布改成另一个时要挪动的概率，0 表示完全相同，1 表示毫不重叠。比如 $(0.5, 0.5)$ 和 $(1, 0)$ 之间要挪 0.5。五个提示词给出的下一个 token 分布，两两之间的距离在 0.53–0.59。训练要做的，是把这些又平又乱的分布变尖，而且尖在合适的词上。初始 loss 也和第 0 章说的一样，比 ln 50000 略高一点。

```localcheck verify/test_part3.py::test_webbot_untrained_preview_matches_record needs="PyTorch" title="按原书超参数构造 WebBot，两种种子各生成一遍，与录制结果逐字比较，并重算第一步的分布"
```

训练开始以后会怎样呢？本书在 Modal 的一张 H100 上原样运行了 `ch09/01_pretrain.py`，只把循环次数从 10 万改成 200，作为计时探针（运行记录 webbot-probe）。下图是这 200 步的训练 loss：

```widget runcurve caption="WebBot 真实训练的前 200 步：loss 从 ln 50000 附近出发，降到 6.1 左右。这时学习率还在 warmup 阶段，只升到最大值的四成。"
{"run": "webbot-probe", "step": "01_pretrain", "metrics": ["loss"], "yLabel": "loss（nats）", "xLabel": "iteration", "refLines": [{"y": 10.8198, "label": "ln 50000"}], "command": "uv run --group gpu modal run --detach gpu/modal_app.py::launch --job webbot-probe --gpu H100 --run webbot-probe"}
```

```verify id=III-b title="运行记录：前 200 步把训练 loss 从 10.98 降到 6.14" files=data/runs/webbot-probe/report.json,data/runs/webbot-probe/01_pretrain.metrics.jsonl
probe = json.load(open("data/runs/webbot-probe/report.json", encoding="utf-8"))
step = probe["steps"][0]
with open("data/runs/webbot-probe/01_pretrain.metrics.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
print(f"{probe['host']['gpu']}，PyTorch {probe['host']['torch']}；改动：{step['patches'][0]['new']}")
print(f"第 0 步 loss {rows[0]['loss']:.4f}（完整验证集 {rows[0]['val_loss']:.4f}），第 {rows[-1]['it']} 步 {rows[-1]['loss']:.4f}")
print(f"用时 {step['seconds']:.0f} 秒（含 torch.compile 和第 0 步的完整验证），费用 {probe['cost_usd']} 美元")
assert len(step["patches"]) == 1 and "range(200)" in step["patches"][0]["new"]
assert abs(rows[0]["loss"] - math.log(50_000)) < 0.2 and rows[-1]["it"] == 200 and rows[-1]["loss"] < 6.2
assert 0.35 < rows[-1]["lr"] / 6e-4 < 0.45            # warmup 500 步，200 步时学习率约为最大值 6e-4 的四成
```

200 步只是 10 万步的 0.2%，模型还远远谈不上会写东西。但这条曲线说明：原书的脚本在真实的硬件上跑得通，loss 也在按预期下降。这次探针更重要的产物是速度，III.2 节会用它来算全程要多久。

## III.1 数据集

WebBot 的数据是 **OpenWebText**（OWT）。它的来历要从 GPT-2 说起：OpenAI 训练 GPT-2 时，收集了 Reddit 上至少获得 3 个赞（karma）的帖子所链接的网页，称为 WebText（见 [GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 2.1 节），但没有公开。OpenWebText 是 Gokaslan 与 Cohen 按同样的思路重新收集的开放版本（[OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/)）。原书用的是斯坦福 CS336 课程整理的样本 [stanford-cs336/owt-sample](https://huggingface.co/datasets/stanford-cs336/owt-sample)，[koki0702/zero-llm-data](https://huggingface.co/datasets/koki0702/zero-llm-data) 的说明标注它与 OpenWebText 都是 MIT 许可。

| | 训练集 | 验证集 |
|---|---|---|
| 文件大小 | 11,920,511,059 字节 | 289,998,753 字节 |
| token 数（WebBot 分词器） | 2,649,004,800 | 64,496,066 |
| 文档篇数 | — | 59,059 |

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
assert len(L) == 59_059 and L.min() > 0 and np.median(L) == 696 and (L > 1024).sum() == 18_925
```

```localcheck verify/test_part3.py::test_owt_valid_stats title="在完整验证集上重算文档长度与压缩率"
```

数据变了，前两台机器人的分词器还能用吗？算法是能用的：第 7 章训练 WebBot 的分词器，直接调用第 4 章写好的 `storybot.tokenizer.train_bpe`。但学到的词表不能照搬。下图让三台机器人的分词器读同一句网页文本，再比较它们在三种数据上的压缩率：

```widget part3-tokenizers caption="CodeBot 的分词器把这句话切成 64 个 token，WebBot 的只要 26 个。每个分词器在自己的训练数据上，都比在另外两种数据上更省；词表越大，三种数据上平均越省。"
{}
```

```verify id=III.1-c title="压缩率（字节 / token）：在网页文本上，CodeBot 的分词器切出的 token 是 WebBot 的 2.3 倍" files=data/ch00/bots.json
names = ["codebot", "storybot", "webbot"]
own = {"codebot": "TinyCodes", "storybot": "TinyStories", "webbot": "OpenWebText"}   # 各自的训练数据
bpt = {r["corpus"]: {n: r["bytes"] / r["tokens"][n] for n in names} for r in tk["matrix"]}
for r in tk["matrix"]:
    print(f"{r['corpus']:<12}{r['bytes']:>10,} 字节 {r['docs']:>5,} 篇" + "".join(f"{n:>9} {bpt[r['corpus']][n]:.2f}" for n in names))
web = next(r for r in tk["matrix"] if r["corpus"] == "OpenWebText")
story = next(r for r in tk["matrix"] if r["corpus"] == "TinyStories")
code = next(r for r in tk["matrix"] if r["corpus"] == "TinyCodes")
means = [sum(bpt[c][n] for c in bpt) / len(bpt) for n in names]
print(f"同一份网页文本，CodeBot 的分词器切出的 token 是 WebBot 的 {web['tokens']['codebot'] / web['tokens']['webbot']:.2f} 倍")
print("三种数据上的平均：" + "  ".join(f"{n} {m:.2f}" for n, m in zip(names, means)))
corpus = {b["id"]: b["corpus"] for b in json.load(open("data/ch00/bots.json", encoding="utf-8"))["bots"]}["codebot"]
full_code, web_2m = corpus["text_bytes"] / corpus["tokens"], owt["bytes_per_token_first_2M"]
gap_code, gap_web = full_code / bpt["TinyCodes"]["codebot"] - 1, web_2m / bpt["OpenWebText"]["webbot"] - 1
print(f"对照：CodeBot 在完整的 TinyCodes 上是 {corpus['text_bytes']:,} ÷ {corpus['tokens']:,} = {full_code:.2f}（比表里 {gap_code:+.0%}）；"
      f"WebBot 在 OWT 前 200 万个 token 上是 {web_2m:.2f}（比表里 {gap_web:+.0%}）")
assert {n: len(c) for n, c in tk["chips"].items()} == {"codebot": 64, "storybot": 34, "webbot": 26}
assert all(max(bpt, key=lambda c: bpt[c][n]) == own[n] for n in names)   # 每个分词器都在自己的训练数据上最省
assert min(web["tokens"], key=web["tokens"].get) == "webbot"           # 网页上 WebBot 最省
assert min(story["tokens"], key=story["tokens"].get) == "storybot"     # 故事上 StoryBot 最省
assert code["tokens"]["webbot"] < code["tokens"]["codebot"]            # 连代码上，5 万词表也比 1,000 词表省
assert means[0] < means[1] < means[2]                                  # 词表 1,000 < 10,000 < 50,000
assert (round(gap_code, 2), round(gap_web, 2)) == (0.09, -0.03)         # 换一份样本，数值能差到一成
```

结果有两层意思。第一，**领域**很重要：每个分词器都在自己的训练数据上最省。在故事上，只有 1 万词表的 StoryBot 分词器甚至比 5 万词表的 WebBot 略胜一筹；反过来，用 TinyStories 训练出的词表放到网页上，每个 token 只装 3.22 个字节。第二，**词表大小**也很重要：CodeBot 只有 1,000 个 token，哪怕在自己的代码数据上，也比不过 5 万词表的 WebBot 分词器。同一份网页文本，CodeBot 的分词器切出的 token 是 WebBot 的 2.30 倍，模型要读的序列也就长了一倍多。WebBot 需要在网页文本上重新训练一个更大的词表，这是第 7 章的起点。

这张表每格只用了约 100 万字节的样本（网页约 200 篇），具体数值会随样本而变，差别可达一成：CodeBot 在完整的 TinyCodes 上是 2.42，比表里的 2.21 高 9%；WebBot 在 OWT 验证集前 200 万个 token（约 890 万字节）上是 4.46，比表里的 4.61 低 3%（[验证 III.1-b](#v-III.1-b)、[验证 III.1-c](#v-III.1-c)）。比较谁大谁小没有问题；引用具体数值时，要说明是在哪份样本上测的。

```localcheck verify/test_part3.py::test_tokenizer_matrix_matches_record title="用三个分词器重新编码，并核对第 1 章与第 4 章两种编码实现的结果相同"
```

::: confusion 上下文长度 1,024 是谁定的？
读训练脚本时，`context_len = 1024`、`n_layer = 12` 这样的数字看起来像是随手写下的“魔法数字”。想知道它们从哪来，是很自然的疑问。答案通常有三个出处：照搬一个成熟的配置、看数据的统计、算算力的账。对 WebBot 的 1,024 来说，能直接给出这个数的只有第一样；数据和算力给的是取舍，正好拿来检验它。

第一，它和 GPT-2 small 的“身材”一样。Hugging Face 上 [GPT-2 的配置文件](https://huggingface.co/openai-community/gpt2/blob/main/config.json) 写着 `n_ctx` 1024、`n_embd` 768、`n_layer` 12、`n_head` 12、词表 50,257（2026-10-11 查证）。在这五项配置里，WebBot 只改了词表，换成自己训练的 50,000；五项之外的结构并不相同（III.2 节）。上下文长度与 GPT-2 small 相同，至于原书是不是有意照搬，我们没有看到说明。

第二，数据：上下文从 256 加到 1,024，能被一眼看完的网页从 9% 升到 68%；再加到 2,048 是 90%。哪一档都不是“刚好够用”，这是取舍。

第三，算力：每个头的 Attention 分数是一张上下文长度 × 上下文长度的表，1,024 比 256 大 16 倍。但摊到每个 token 上，这部分在总计算量里占比很小。按 Kaplan 等人的估算（[arXiv:2001.08361](https://arxiv.org/abs/2001.08361) 表 1），训练时每个 token 约要

$$
6N' + 6 L \, n_\text{ctx} \, d
$$

次浮点运算。这里，$N'$ 是去掉 token embedding 后的参数量（embedding 只是查表，不做乘法），$L$ 是层数，$n_\text{ctx}$ 是上下文长度，$d$ 是向量宽度；第二项就是 Attention 分数那部分。拿 WebBot 算一算：$6N' \approx 6 \times 1.14\text{ 亿} \approx 6.8$ 亿，第二项在 $n_\text{ctx} = 1{,}024$ 时是 $6 \times 12 \times 1{,}024 \times 768 \approx 0.57$ 亿，在 256 时是 0.14 亿。所以 1,024 只比 256 多约 6% 的计算量。

```verify id=III.1-d title="1,024：与 GPT-2 small 相同；数据和算力给的是取舍" files=data/ch00/bots.json
gpt2_small = {"n_ctx": 1024, "n_embd": 768, "n_layer": 12, "n_head": 12, "vocab_size": 50257}  # HF 配置文件
bots = {b["id"]: b for b in json.load(open("data/ch00/bots.json", encoding="utf-8"))["bots"]}
web_cfg = bots["webbot"]["model"]
same = {"n_ctx": "context_len", "n_embd": "embed_dim", "n_layer": "n_layer", "n_head": "n_head"}
for k, v in same.items():
    print(f"GPT-2 small {k:<8} = {gpt2_small[k]:>5}   WebBot {v:<11} = {web_cfg[v]:>5}")
    assert gpt2_small[k] == web_cfg[v]
assert web_cfg["vocab_size"] == 50_000 != gpt2_small["vocab_size"]
n_prime = bots["webbot"]["params"] - bots["webbot"]["params_split"]["embedding"]   # 去掉 token embedding
d, layers = web_cfg["embed_dim"], web_cfg["n_layer"]
flops = {}
for ctx in (256, 1024, 2048):
    flops[ctx] = 6 * n_prime + 6 * layers * ctx * d
    print(f"上下文 {ctx:>5}：{(L <= ctx).mean():5.1%} 的网页能完整放进去；Attention 表 {ctx * ctx:>9,} 个数；"
          f"每个 token 约 {flops[ctx] / 1e8:.2f} 亿次运算（Attention 占 {6 * layers * ctx * d / flops[ctx]:.1%}）")
print(f"1,024 比 256：Attention 表大 {1024 ** 2 // 256 ** 2} 倍，每个 token 的计算量只多 {flops[1024] / flops[256] - 1:.1%}")
assert round((L <= 1024).mean(), 2) == 0.68 and round((L <= 256).mean(), 2) == 0.09 and round((L <= 2048).mean(), 2) == 0.90
assert 1.05 < flops[1024] / flops[256] < 1.07
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

模型是 GPT-2 small 的身材，配上 StoryBot 的现代零件（RoPE、SwiGLU、RMSNorm，无 bias，不共享权重），再加上 GQA：12 个查询头只配 4 组键和值。参数一共 152,316,672 个，其中略多于一半在 token embedding 和输出层这两张 50,000 × 768 的大表里：

```verify id=III.2-a title="WebBot 的参数有 50.4% 在 token embedding 和输出层"
split = bots["webbot"]["params_split"]
table = split["embedding"] + split["output_extra"]
print(f"总参数 {bots['webbot']['params']:,}；token embedding + 输出层 {table:,} = 2 × 50,000 × 768，占 {table / bots['webbot']['params']:.1%}")
assert table == 2 * 50_000 * 768 and round(table / bots["webbot"]["params"], 3) == 0.504
```

训练的超参数写在 `ch09/01_pretrain.py` 里。每张 GPU 每一步处理 32 条序列 × 梯度累积 4 次 × 1,024 个 token：

```listing ch09/01_pretrain.py#L97-L113
```

::: think 用 8 张 GPU 跑这个脚本，会比 1 张快 8 倍吗？
:::
::: answer
不会。脚本从不读取 GPU 的张数，每张卡都要跑满 `max_iters = 100000` 步。多卡并行时，每张卡用不同的随机种子（`torch.manual_seed(42 + RANK)`）抽取训练窗口，处理的是不同的数据，梯度取平均后一起更新。所以墙钟时间和单卡差不多，读过的 token 却变成了 8 倍：从 131 亿变成 1,049 亿。全局 batch 也一样变大：32 条 × 4 次累积 × 8 张卡 = 1,024 条序列。想在 8 张卡上保持单卡的 token 预算，要把 `max_iters` 除以 8，改成 12,500。第 9 章讲分布式训练时会回到这一点。
:::

```verify id=III.2-b title="加卡等于加 token：每张卡都跑满 10 万步"
per_gpu_step = 32 * 4 * 1024
for gpus in (1, 8):
    print(f"{gpus} 张 GPU：每步全局 {32 * 4 * gpus:,} 条序列，共 {100_000 * per_gpu_step * gpus:,} 个 token")
assert 100_000 * per_gpu_step == bots["webbot"]["pretrain"]["tokens"] == 13_107_200_000
assert (100_000 // 8) * per_gpu_step * 8 == 13_107_200_000      # max_iters 改成 12,500 才能保持预算
```

::: gpu 在一张 H100 上要多久？
按 $6ND$ 粗算，WebBot 的单卡预训练约需 1.2 × 10¹⁹ 次浮点运算。H100（SXM）的稠密 BF16 峰值约 989 TFLOP/s（[NVIDIA 数据手册](https://www.nvidia.com/en-us/data-center/h100/)上的 1,979 是含稀疏加速的数字）。实际训练达不到峰值，开工之前只能假设一个峰值利用率（MFU，Model FLOPs Utilization），这里取 35%。这是估算里最关键、也最不确定的一个数。

```verify id=III.2-c title="开工前的估算：一张 H100 约 10 小时（假设 35% 的峰值利用率）"
flops = 6 * bots["webbot"]["params"] * bots["webbot"]["pretrain"]["tokens"]
peak, mfu, price = 989e12, 0.35, 3.95          # FLOP/s；假设的利用率；Modal H100 每小时美元（只算 GPU，2026-10-11）
hours = flops / (peak * mfu) / 3600
print(f"6ND = {flops:.2e} FLOPs → {hours:.1f} 小时 → 约 {hours * price:.0f} 美元（只算 GPU 的标价，不含验证集评估）")
assert 9 < hours < 10.5
```

价格取自 [Modal 的价目表](https://modal.com/pricing)（2026-10-11）。这只是开工前的估算。稳妥的做法是先在云上跑 200 步，从进度条读出真实的速度，再决定要不要跑完整的 10 万步。

本书做了这个计时探针（本页开头的运行记录 webbot-probe）。稳定之后每步 0.317 秒，按 6ND（全部参数）的口径，MFU 约 38%，比假设的 35% 略好，10 万步的训练本身约 8.8 小时。MFU 的分子还有另一种数法：只数参与矩阵乘的参数，再加上 Attention 分数那部分运算，同一个速度算出来约 31%（[G.2 节](gpu.html#sG-2)、[验证 9.7-e](ch09.html#v-9.7-e)）。两个数说的是同一个速度，比较 MFU 之前，先确认分子是怎么数的。别忘了验证：脚本每 1,000 步（以及最后一步）在全部 6,450 万个验证 token 上评估一次，从探针的日志看，每次约 56 秒，101 次合计约 1.6 小时。所以全程约 10.4 小时。费用按本书两次 Modal 运行的实际账单算（GPU 加上 CPU 和内存，折合每小时 4.11 美元），约 43 美元：

```verify id=III.2-d title="计时探针：每步 0.317 秒（MFU 按 6ND 的口径约 38%，按矩阵乘加 Attention 的口径约 31%），加上验证全程约 10.4 小时" files=data/runs/webbot-probe/timing.json,data/runs/webbot-probe/report.json,data/runs/webbot-probe/01_pretrain.log.txt
import re

timing = json.load(open("data/runs/webbot-probe/timing.json", encoding="utf-8"))
probe = json.load(open("data/runs/webbot-probe/report.json", encoding="utf-8"))
log = open("data/runs/webbot-probe/01_pretrain.log.txt", encoding="utf-8").read()
s = timing["steady_s_per_step_20_200"]                       # 第 20–200 步的平均秒数
mfu = 6 * bots["webbot"]["params"] * timing["tokens_per_step"] / s / peak   # 6ND：全部参数
n_mm = bots["webbot"]["params"] - split["embedding"] - (2 * layers + 1) * d  # 只数矩阵乘：去掉查表的 embedding 和 25 个 RMSNorm 的 γ
mfu_mm = (6 * n_mm + 6 * layers * web_cfg["context_len"] * d) * timing["tokens_per_step"] / s / peak   # 加上 Attention 分数那一项
train_h = 100_000 * s / 3600
# 第 0 步的完整验证：进度条“已完成/总批数 [已用时间<剩余, 速度]”，跳过开头编译的那几行
bars = [(int(a), int(b), float(r)) for a, b, r in re.findall(r"Evaluating:.*? (\d+)/(\d+) \[[^,]*, ([\d.]+)it/s", log)]
rate = sum(r for a, b, r in bars if a >= 100) / sum(1 for a, b, r in bars if a >= 100)
eval_s = bars[0][1] / rate
n_eval = 100_000 // 1000 + 1                                 # step % 1000 == 0 共 100 次，再加最后一步
eval_h = n_eval * eval_s / 3600
usd_h = probe["billing"][0]["usd_per_hour"]
print(f"训练：每步 {s} 秒 × 10 万步 = {train_h:.1f} 小时；MFU：6ND 口径 {mfu:.1%}，矩阵乘加 Attention 口径 {mfu_mm:.1%}")
print(f"验证：{bars[0][1]:,} 批 ÷ {rate:.1f} 批/秒 = {eval_s:.0f} 秒/次，{n_eval} 次 = {eval_h:.1f} 小时")
print(f"全程约 {train_h + eval_h:.1f} 小时 × {usd_h} 美元/小时 ≈ {(train_h + eval_h) * usd_h:.0f} 美元")
assert 0.35 < mfu < 0.40 and 8.5 < train_h < 9.0 and 1.4 < eval_h < 1.8
assert n_mm == 113_897_472 and 0.30 < mfu_mm < 0.32                 # 与 9.7-e 的记账相同
assert bars[0][1] * 32 * 1024 >= 64_000_000                  # 每次都把 6,450 万个验证 token 过一遍
assert 10 < train_h + eval_h < 11 and 40 < (train_h + eval_h) * usd_h < 46
```

完整的运行命令在 [GPU 实战手册](gpu.html) 里。
:::

::: cs336 CS336 用的是同一份网页数据
CS336 的 A1 也在这份 OWT 样本上训练分词器和语言模型。验证集文件与原书的 `owt_valid.txt` 逐字节相同：把 CS336 的 [owt_valid.txt.gz](https://huggingface.co/datasets/stanford-cs336/owt-sample/tree/main) 解压后算 SHA-256，开头是 `2406f278e71829d2`，与原书数据集[文件列表](https://huggingface.co/api/datasets/koki0702/zero-llm-data/tree/main?recursive=true)里 `webbot/owt_valid.txt` 的哈希相同，大小也都是 289,998,753 字节（2026-10-12 查证）。不过 A1 在 OWT 上训练的 BPE 词表是 32,000，而不是 WebBot 的 50,000。原书在第 8、9 章只给提示的主题，大多能在 CS336 里找到动手的版本：A2 要用 Triton 写 FlashAttention-2、实现 DDP 和优化器状态切分，A3 要用实验拟合 scaling law。资料来源：[CS336 课程主页](https://cs336.stanford.edu/)。
:::

## 本部要点

第 III 部没有现成的终点可看，但起点和路线都能验证：未训练的 WebBot 只会从近乎平坦的分布里随手抓词，原书脚本在 H100 上的前 200 步也确实让 loss 降了下来；它要学的网页文本又长又杂，三分之一的文档比上下文还长；前两台机器人的分词器算法可以复用，词表却必须在网页上重新训练；而整个预训练需要约 10¹⁹ 次浮点运算，在一张 H100 上要跑十个多小时。

**本部所学的内容**

- 未训练的 WebBot 输出近乎随机的词串；五段输出的重合来自同一个随机种子，换种子就消失；初始 loss 比 ln 50000 略高（[验证 III-a](#v-III-a)、[验证 III-b](#v-III-b)）。
- OWT 验证集有 59,059 篇文档，中位数 696 个 token，32.0% 超过上下文长度 1,024（[验证 III.1-a](#v-III.1-a)、[验证 III.1-b](#v-III.1-b)）。
- 在约 100 万字节的网页样本上，WebBot 的分词器每个 token 装 4.61 字节，CodeBot 的只装 2.01 字节；领域和词表大小都有影响（[验证 III.1-c](#v-III.1-c)）。
- 上下文长度 1,024 与 GPT-2 small 相同；数据和算力给的是取舍：从 256 加到 1,024，看得完的网页从 9% 升到 68%，每个 token 的计算量只多约 6%（[验证 III.1-d](#v-III.1-d)）。
- WebBot 的参数略多于一半在 token embedding 和输出层；加 GPU 不会缩短训练时间，而是增加训练的 token 数（[验证 III.2-a](#v-III.2-a)、[验证 III.2-b](#v-III.2-b)）。
- 一张 H100 上的单卡预训练：计时探针实测每步 0.317 秒（MFU 按 6ND 的口径约 38%，按矩阵乘加 Attention 的口径约 31%），训练约 8.8 小时，加上验证约 10.4 小时（[验证 III.2-c](#v-III.2-c)、[验证 III.2-d](#v-III.2-d)）。

第 7 章的第一步，就是在 12 GB 的网页文本上训练一个 5 万词表的分词器。光是这一步，就会让第 4 章的加速成果接受一次真正的考验。
