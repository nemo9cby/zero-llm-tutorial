---
chapter: II
slug: part2
order: 45
kind: part
title: 第 II 部 应用篇：制作 StoryBot
nav_title: 第 II 部 应用篇：制作 StoryBot
lede: 第二台机器人 StoryBot 用 2.2 GB 的儿童故事训练，会自己编出一个完整的小故事。分词器、模型、训练还是那三步，但数据大了三百多倍，每一步都要换上更工程化、更现代的做法。
pyodide_packages: [numpy]
---

CodeBot 写的是代码，StoryBot 写的是故事。从这一部开始，我们从生成程序转向用自然语言创作。数据从 6.5 MB 增加到 2.2 GB，模型换成了今天的 LLM 普遍采用的结构，训练也用上了学习率调度、混合精度和验证集。节奏还是“分词器 → 模型 → 训练”，只是每一步都要面对规模带来的新问题。

还是先来看结果。原书作者在 Hugging Face 上发布了 StoryBot 预训练过程中的三个 checkpoint：第 500 步、第 5,000 步和训练结束时的第 40,000 步。我们让它们用同一个随机种子各写一个故事（在 CPU 上生成，提示词是 `<|endoftext|>`，意思是“从头开始写一个新故事”）。

```widget part2-stories caption="同一个随机种子，三个训练阶段。第 500 步只是像英语，句子常常不通；第 5,000 步句子大多通顺，情节还会跑偏；训练结束时，故事大体连贯，细看仍有破绽。"
{"iter": 40000, "seed": 0}
```

第 500 步的故事只是“像英语”：猫“liked to help other feels”，这已经不成句子了；人物也凭空出现又消失。到第 5,000 步，句子大多通顺了，但情节还会跑偏，刚出场的“小动物”隔了一句就成了“拿着红旗的小学生”（a pupil with a red flag）。训练结束时，Lucy 的故事有开头、有转折，最后落到“要听妈妈的话”这个儿童故事式的道理上。遗憾的是，细看仍有破绽：厨房里有“a small door on the door”；Lucy 刚刚记着妈妈的话没碰冷饮，下一句两人又一起分着吃了。

切换到种子 1 会看到一个有趣的现象：三个阶段的故事都以同一句 “Once upon a time, there was a fat cat named Tom.” 开头。这是因为同一个种子让每一步抽样用的随机数完全相同。直观上，只要开头几步的概率分布相差不大，同一个随机数就会落在同一个词上；分布差得远了，故事才各自分开。

```verify id=II-a title="三个 checkpoint：看过的 token 越多，验证集上的 loss 越低" files=data/part2/preview.json
import json

rec = json.load(open("data/part2/preview.json", encoding="utf-8"))
for c in rec["checkpoints"]:
    firsts = [s["text"].strip().split(". ")[0] for s in c["samples"]]
    print(f"第 {c['iters']:>6,} 步 看过 {c['tokens_seen']:>11,} token  loss {c['val_loss_subset']:.4f}  种子 1 的开头：{firsts[1]!r}")
losses = [c["val_loss_subset"] for c in rec["checkpoints"]]
assert losses[0] > losses[1] > losses[2]
assert [c["tokens_seen"] for c in rec["checkpoints"]] == [501 * 8192, 5_001 * 8192, 40_000 * 8192]   # 更新次数见 II.2 节
assert len({c["samples"][1]["text"].strip().split(". ")[0] for c in rec["checkpoints"]}) == 1
```

这里的 loss 是在验证集里随机取 32 个 256-token 窗口算出的平均交叉熵，只用来看趋势。在完整验证集上的评估，第 6 章会做；本书在 H100 上原样重跑时也做了，II.2 节会拿来对照。

```localcheck verify/test_part2.py::test_storybot_preview_matches_record needs="PyTorch" title="用官方的三个 checkpoint 在 CPU 上重新生成，与录制结果逐字比较；更新次数按原书脚本的保存位置核对"
```

::: warn 下载的 StoryBot 权重要改个名字才能加载
Hugging Face 上的 StoryBot checkpoint，把 SwiGLU 那一层的参数存成了 `blocks.N.mlp.*`，而仓库里的 `storybot/model.py` 把这一层叫作 `ffn`。直接用 `GPT.load_from` 加载会报 `Missing key(s) ... "blocks.0.ffn.W.weight"`。改名之后就能加载，本书的预览就是这样生成的：

```python
ck = torch.load("storybot/model_pretrain.pt", map_location="cpu")
state = {k.replace(".mlp.", ".ffn."): v for k, v in ck["model_state_dict"].items()}
model = GPT(ck["vocab_size"], ck["max_context_len"], ck["embed_dim"], ck["n_head"],
            ck["n_layer"], ck["ff_dim"], ck["theta"])
model.load_state_dict(state)
```

另外，`storybot/download.sh` 的注释里列出了 DPO 之后的模型 `model_dpo.pt`，但 Hugging Face 上并没有这个文件（2026-10-12 查证时，[文件列表](https://huggingface.co/api/datasets/koki0702/zero-llm-data/tree/main?recursive=true)里 `storybot/` 下的权重只有上面三个），要在第 6 章自己训练。
:::

```localcheck verify/test_part2.py::test_storybot_checkpoints_need_key_rename needs="PyTorch" title="原样加载会失败，改名后可以加载"
```

## II.1 数据集

StoryBot 的训练数据是 **TinyStories**，出自 Eldan 与 Li 2023 年的论文《TinyStories: How Small Can Language Models Be and Still Speak Coherent English?》（[arXiv:2305.07759](https://arxiv.org/abs/2305.07759)）。这是一批由 GPT-3.5 和 GPT-4 生成的短篇故事，用词限制在三四岁孩子通常能听懂的范围内。论文的出发点是：像 GPT-2 small 这样约 1.25 亿参数的模型，在一般文本上训练后也很少能写出连贯的英文；如果把语言变简单，小得多的模型能不能做到？StoryBot 只有 2,270 万个参数，本页开头那个连贯的故事就是答案。

原书用的是其中的 **V2-GPT4** 版本（`TinyStoriesV2-GPT4-{train,valid}.txt`）。按 [Hugging Face 上的数据集说明](https://huggingface.co/datasets/roneneldan/TinyStories)，这个版本只收录 GPT-4 生成的故事（说明里说，原始版本中由 GPT-3.5 生成的那部分质量较差）。原始版本里由 GPT-4 生成的故事，它全都收录了，规模却比这一部分大得多。许可证是 CDLA-Sharing-1.0。

原书把两个文件改名为 `storybot/tiny_stories_train.txt` 和 `tiny_stories_valid.txt`，内容与斯坦福 CS336 作业所用的文件逐字节相同：原书数据集的[文件列表](https://huggingface.co/api/datasets/koki0702/zero-llm-data/tree/main?recursive=true)和 [roneneldan/TinyStories 的文件列表](https://huggingface.co/api/datasets/roneneldan/TinyStories/tree/main)给出的 SHA-256 一样，训练集开头是 `6418d412de72888f`，验证集开头是 `6874bae9a4c1a4e7`（2026-10-12 查证）。验证集已经下载到本地，可以自己算一遍：

```localcheck verify/test_part2.py::test_tinystories_valid_is_the_cs336_file title="本地验证集的 SHA-256 与 CS336 所用文件的哈希相同"
```

| | 训练集 | 验证集 |
|---|---|---|
| 文件大小 | 2,227,753,162 字节 | 22,502,601 字节 |
| token 数（StoryBot 分词器） | 541,229,223 | 5,465,873 |
| 故事篇数 | 约 274 万（推算） | 27,630 篇完整的故事 + 开头的半篇 |

原书第 II 部扉页（[试读](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9784814401611/files/part2.xhtml)，2026-10-12 查证）说，TinyStories 收录了 GPT-4 生成的约 200 万篇故事。这个数字与原始版本的规模一致：Hugging Face 的数据集服务（[datasets-server](https://datasets-server.huggingface.co/size?dataset=roneneldan/TinyStories)，2026-10-12 查证）给出的原始训练集是 2,119,719 篇，其中也有 GPT-3.5 生成的。原书实际用的 V2-GPT4 训练集，篇数没有现成的数字，我们按验证集每篇的平均字节数推算：

```verify id=II.1-a title="V2-GPT4 训练集约有 274 万篇故事（按验证集的平均长度推算）" files=data/part2/doclen.json,data/part1/doclen.json
stats = json.load(open("data/part2/doclen.json", encoding="utf-8"))
codes = json.load(open("data/part1/doclen.json", encoding="utf-8"))
per_story = stats["text_bytes"] / stats["segments"]
train_bytes = 2_227_753_162
print(f"验证集：{stats['text_bytes']:,} 字节 ÷ {stats['segments']:,} 段 = 每篇约 {per_story:.0f} 字节")
print(f"训练集：{train_bytes:,} 字节 ÷ {per_story:.0f} ≈ {train_bytes / per_story / 1e4:,.0f} 万篇")
print(f"每个 token 平均 {stats['text_bytes'] / stats['tokens']:.2f} 字节"
      f"（CodeBot 在 TinyCodes 上是 {codes['text_bytes'] / codes['tokens']:.2f}）")
assert stats["separators"] == 27_630 and stats["segments"] == 27_631
assert round(train_bytes / per_story / 1e4) == 274
assert round(stats["text_bytes"] / stats["tokens"], 2) == 4.12 and round(codes["text_bytes"] / codes["tokens"], 2) == 2.42
```

验证集的文件是从一篇故事的中间开始的，开头的半句话是 “u don't have to be scared of the loud dog...”。下面看第一篇完整的故事。本书附带了验证集开头的 200 篇完整故事（`data/part2/stories_head.txt`），原样取自原文件：

```verify id=II.1-b title="验证集的第一篇完整故事" files=data/part2/stories_head.txt
EOT = "<|endoftext|>"
stories = open("data/part2/stories_head.txt", encoding="utf-8").read().split(EOT)
first = stories[0].strip()
print(first[:400], "…")
assert len(stories) == 200 and first.startswith("Once upon a time, in a warm and sunny place")
```

主角、一个小小的麻烦、朋友帮忙，这是 TinyStories 的典型结构。这 200 篇里，有一半以上以 “Once upon a time” 开头，用到的不同单词也不到两千个：

```verify id=II.1-c title="200 篇样本：57% 以 Once upon a time 开头，只用了 1,976 个不同的词"
import re
from collections import Counter

words = [w.lower() for s in stories for w in re.findall(r"[A-Za-z']+", s)]
opening = sum(s.strip().startswith("Once upon a time") for s in stories)
print(f"以 Once upon a time 开头：{opening} / {len(stories)}")
print(f"共 {len(words):,} 个词，不同的词 {len(set(words)):,} 个")
print("最常见的词：", ", ".join(w for w, _ in Counter(words).most_common(10)))
assert opening == 114 and len(set(words)) == 1_976
```

故事有多长？StoryBot 的上下文长度和 CodeBot 一样是 256 个 token。下图把验证集全部故事的长度分布和 TinyCodes 放在一起：

```widget part1-doclen caption="故事比程序长：TinyStories 验证集的中位数是 176 个 token，约 11% 的故事超过上下文长度 256。"
{"datasets": [{"name": "TinyStories（验证集）", "file": "data/part2/doclen.json"}, {"name": "TinyCodes", "file": "data/part1/doclen.json"}], "xMax": 800}
```

```verify id=II.1-d title="10.9% 的故事超过上下文长度 256"
import numpy as np

lengths = np.array(stats["lengths"])
print(f"中位数 {np.median(lengths):.0f}，90% 分位数 {np.percentile(lengths, 90):.0f}，最长 {lengths.max()}")
print(f"超过 256 个 token：{(lengths > 256).sum():,} 篇（{(lengths > 256).mean():.2%}）")
assert np.median(lengths) == 176 and (lengths > 256).sum() == 3_002
```

大约每 9 篇故事就有 1 篇比上下文窗口长，模型在训练时永远无法一眼看完它们。生成时情况更微妙。原书 StoryBot 的 `generate` 每一步只把新 token 送进模型，KV cache 不会丢掉开头，所以写到 256 个 token 以后，模型仍然看得见故事的开头；只是从这以后，每个新 token 的位置都被钳在 255（[5.5.5 节](ch05.html#s5-5-5)会看到这个坑）。本页开头的故事都限制在 255 个新 token 以内，就是为了避开这个问题；原书的 `06_generate.py` 写的是 300。

```localcheck verify/test_part2.py::test_storybot_cache_keeps_the_beginning needs="PyTorch" title="上下文上限设为 8、连续生成 20 步：StoryBot 的 KV cache 存着 20 个位置，WebBot 的只留 8 个"
```

```localcheck verify/test_part2.py::test_tinystories_valid_stats title="在完整验证集上重算：篇数、token 数、长度分布，并确认附带的样本原样取自原文件"
```

## II.2 开发流程

StoryBot 的开发同样分三章，每一章都是在第 I 部的基础上“升级”：

| 章 | 主题 | 相对第 I 部的变化 |
|---|---|---|
| [第 4 章](ch04.html) | 分词器 | 算法还是 BPE，但第 1 章的实现处理 2.2 GB 太慢。依次加上去重、增量更新、分块读取、多进程并行，再加速编码，最后用 memmap 写出 `.bin` |
| [第 5 章](ch05.html) | 模型 | 在 GPT-2 的骨架上换零件：RoPE 位置编码、SwiGLU、RMSNorm，去掉 dropout 和 bias，取消权重共享，再加上生成用的 KV cache |
| [第 6 章](ch06.html) | 训练 | AdamW 的原理与实现、warmup 加线性衰减到 0 的学习率调度、bf16 混合精度、带验证集的预训练、用 LLM 当裁判来评估，最后用 DPO 让它更爱写快乐的结局 |

模型有 22,696,448 个参数，约为 CodeBot 的 2 倍。预训练的超参数写在 `ch06/05_pretrain.py` 里：

```listing ch06/05_pretrain.py#L82-L96
```

40,000 步 × 32 × 256 一共是 3.28 亿个 token，只相当于把训练集看了 0.61 遍。脚本还在第 500 步和第 5,000 步各存了一个中间 checkpoint，再加上训练结束时的 `model_pretrain.pt`，就是本页开头三个故事的来源。严格地讲，文件名里的 500 是从 0 数起的循环变量 `i`，保存发生在这一次循环的 `optimizer.step()` 之后，所以 `model_iter_500.pt` 已经更新了 501 次；同理，`model_iter_5000.pt` 更新了 5,001 次。

```verify id=II.2-a title="预算：3.28 亿 token，训练集的 0.61 遍；model_iter_5000.pt 看过 4,097 万 token"
per_step = 32 * 256
for c in rec["checkpoints"]:
    print(f"{c['file']:<28} {c['updates']:>6,} 次更新 × {per_step:,} = {c['tokens_seen']:>11,} token"
          f"（训练集的 {c['tokens_seen'] / 541_229_223:.3f} 遍）")
print(f"训练集的原始文本是 TinyCodes 的 {2_227_753_162 / 6_487_033:.0f} 倍")
assert 40_000 * per_step == 327_680_000 and round(40_000 * per_step / 541_229_223, 2) == 0.61
assert [c["updates"] for c in rec["checkpoints"]] == [501, 5_001, 40_000]
assert all(c["tokens_seen"] == c["updates"] * per_step for c in rec["checkpoints"])
assert round(2_227_753_162 / 6_487_033) == 343
```

后训练只有一步 DPO：用 144 对故事（同一个开头，一个快乐结局、一个悲伤结局）教模型偏爱快乐的那个。和第 I 部的强化学习不同，DPO 不需要模型自己生成，也不需要奖励函数，直接从成对的偏好数据里学。

硬件方面，原书前言推荐第 II 部使用 GPU。本书在 Modal 的一张 H100 上原样跑完了第 6 章的预训练：3.28 亿个 token 用了 973.6 秒（约 16 分钟），连同生成和 DPO 一共花了 1.17 美元（运行记录 storybot-h100，命令见 [GPU 实战手册](gpu.html)）。这次运行每 500 步在完整验证集上评估一次，正好可以和本页开头的估算对照：

```verify id=II.2-b title="H100 上原样重跑：完整验证集 loss 第 5,000 步 1.662，训练结束 1.304" files=data/runs/storybot-h100/report.json,data/runs/storybot-h100/05_pretrain.metrics.jsonl
run = json.load(open("data/runs/storybot-h100/report.json", encoding="utf-8"))
pre = next(s for s in run["steps"] if s["script"] == "ch06/05_pretrain.py")
with open("data/runs/storybot-h100/05_pretrain.metrics.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
v5000 = next(r["val_loss"] for r in rows if r["it"] > 5_000)    # 第 5,000 步的评估结果一直显示到第 5,500 步
final = pre["final"]["val_loss"]
print(f"{run['host']['gpu']}：{pre['final']['it']:,} 步用了 {pre['seconds']:.1f} 秒；整个任务 {run['cost_usd']} 美元（含生成和 DPO）")
print(f"完整验证集：第 5,000 步 {v5000:.3f}，训练结束 {final:.3f}")
print(f"本页用 32 个窗口估算的官方 checkpoint：第 5,000 步 {rec['checkpoints'][1]['val_loss_subset']:.3f}，"
      f"第 40,000 步 {rec['checkpoints'][2]['val_loss_subset']:.3f}")
assert pre["exit_code"] == 0 and pre["final"]["it"] == 40_000
assert v5000 <= 2.00 and final <= 1.45                         # CS336 A1 的两个目标
```

::: cs336 StoryBot 就是 CS336 A1 的 TinyStories 基线
CS336 A1 的 TinyStories 实验配置是：词表 10,000、上下文 256、`d_model` 512、`d_ff` 1,344、RoPE θ = 10,000、4 层、16 个头，训练 327,680,000 个 token，与上面的超参数逐项相同。A1 还给了一个低资源版本：把预算降到 4,000 万个 token，目标放宽到 2.00；讲义里的参考运行用的是 32 × 5,000 × 256 = 4,096 万个 token，与 StoryBot 的 `model_iter_5000.pt`（4,097 万个 token）几乎相同。A1 的验收目标是完整验证集上的 loss：完整预算下 ≤ 1.45，低资源版本 ≤ 2.00。

本页用 32 个随机窗口估算的 loss，官方 checkpoint 第 5,000 步约 1.66、第 40,000 步约 1.31。本书在 H100 上原样重跑时，完整验证集上的 loss 是 1.662 和 1.304（[验证 II.2-b](#v-II.2-b)），两个目标都达到了。两次训练并不是同一次（原书脚本没有设随机种子），数字却几乎一样。资料来源：[CS336 A1 仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)（讲义 v26.0.3 第 7.2 节）。
:::

## 本部要点

StoryBot 的三个零件都比 CodeBot 更“工业化”：分词器要能在合理时间内吃下 2.2 GB 的文本，模型换上了今天的 LLM 普遍采用的零件，训练要靠学习率调度、混合精度和验证集来保驾护航。从官方 checkpoint 的三个故事可以看出，同一个模型看过的 token 越多，故事越连贯。这条训练在一张 H100 上只要 16 分钟，本书原样重跑的结果达到了 CS336 的目标。

**本部所学的内容**

- 官方 StoryBot checkpoint 看过的 token 越多，验证集上的 loss 越低，故事越连贯；同一个种子下，三个阶段的开头可以完全相同（[验证 II-a](#v-II-a)）。
- TinyStories V2-GPT4 只收录 GPT-4 生成的故事，训练集约 274 万篇，每个 token 平均 4.12 字节（[验证 II.1-a](#v-II.1-a)）。
- 200 篇样本里有 114 篇以 Once upon a time 开头，只用了 1,976 个不同的词（[验证 II.1-b](#v-II.1-b)、[验证 II.1-c](#v-II.1-c)）。
- 10.9% 的故事超过上下文长度 256（[验证 II.1-d](#v-II.1-d)）。
- StoryBot 的预训练是 3.28 亿 token、训练集的 0.61 遍；`model_iter_5000.pt` 更新了 5,001 次，看过约 4,097 万 token，接近 CS336 低资源参考运行的 4,096 万（[验证 II.2-a](#v-II.2-a)）。
- 本书在一张 H100 上原样重跑：预训练 16 分钟，连同 DPO 共 1.17 美元；完整验证集 loss 第 5,000 步 1.662、训练结束 1.304，达到 CS336 的两个目标（[验证 II.2-b](#v-II.2-b)）。

第 4 章的问题很具体：第 1 章的 BPE 每合并一次，就要把全部语料重新数一遍；数据变成 343 倍以后，该怎么办？
