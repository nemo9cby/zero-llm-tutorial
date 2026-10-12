---
chapter: I
slug: part1
order: 15
kind: part
title: 第 I 部 基础篇：制作 CodeBot
nav_title: 第 I 部 基础篇：制作 CodeBot
lede: 第一台机器人 CodeBot 是一个会写 Python 的小语言模型。它能打招呼、按指令写几行代码、做个位数加法。动手之前，我们先看它完成后的样子，再看要把它做出来，需要哪些步骤和数据。
pyodide_packages: [numpy]
---

从这里开始，我们制作第一台机器人 **CodeBot**。它只有 1,112 万个参数，训练数据是 6.5 MB 的 Python 小程序，但它已经具备了 LLM 的全部要素：能续写代码，能按对话模板回答问题，还用强化学习专门练过个位数加法。

老规矩，先来看结果。下面是原书作者在 Hugging Face 上发布的三个 CodeBot 检查点的真实回答，由本书在 CPU 上重新生成。我们用的是贪心解码，也就是每一步都取概率最大的 token，所以在同样的环境下重跑，会得到同样的回答。可以在三个训练阶段之间切换。

```widget part1-chat caption="CodeBot 在三个阶段的回答。只做了预训练的模型不会接话；SFT 之后会打招呼、会写代码；强化学习之后，寒暄和代码几乎不变。"
{"model": "grpo"}
```

完成后的 CodeBot 会打招呼，知道自己叫 CodeBot，能写出一个带 `else` 的 `for` 循环。可是 `7+8=` 它答成了 16。原书第 I 部开头展示的对话里，CodeBot 对这道题答出了 15；我们用官方发布的 SFT 和强化学习两个检查点贪心解码，得到的都是 16。原书的对话脚本 `ch03/04_chat.py` 用 temperature 1.0 采样，训练本身也有随机性，书中那段对话未必能用这份检查点、以贪心解码复现。这件事提醒我们：小模型的能力要用数据说话，不能只看一两段对话。I.4 节会把 81 道个位数加法全部测一遍。

```verify id=I-a title="官方检查点的回答（录制在 data/part1/preview.json 里）" files=data/part1/preview.json
import json

rec = json.load(open("data/part1/preview.json", encoding="utf-8"))
for name, m in rec["models"].items():
    print(f"[{name}] sha256 {m['sha256'][:12]}…")
    for prompt, reply in zip(rec["prompts"], m["greedy"]):
        first = reply.splitlines()[0] if reply else "（结束记号）"
        print(f"  You: {prompt:<16} Bot: {first}")
for name in ("sft", "grpo"):
    reply = dict(zip(rec["prompts"], rec["models"][name]["greedy"]))
    assert "CodeBot" in reply["Who are you?"] and reply["write loop"].startswith("for ")
    assert reply["7+8="] == "16"                      # 两个检查点都答错了这道题
```

```localcheck verify/test_part1.py::test_codebot_preview_matches_record title="用官方检查点在 CPU 上重新生成，与录制结果逐字比较（并核对检查点的 sha256）"
```

CodeBot 并不聪明，原书前言也坦白了这一点。但重要的不是结果，而是过程：从一个个字节到会回答问题的模型，中间的每一步我们都会亲手做一遍。

## I.1 开发流程

CodeBot 的开发分三步，每一步占一章：

| 章 | 主题 | 做什么 |
|---|---|---|
| [第 1 章](ch01.html) | 分词器 | 实现 BPE，把 Python 代码变成 token 的 id 序列，词表 1,000 |
| [第 2 章](ch02.html) | 模型 | 从 Attention 的原理出发，实现一个小型的 GPT-2 |
| [第 3 章](ch03.html) | 训练 | 依次做预训练、SFT、强化学习，看模型一步步变“聪明” |

每章的代码最后都整理进 `codebot/` 这个包：`tokenizer.py`、`model.py`、`utils.py`，后面的章节直接 `import` 它们。数据和产物在这三章之间这样流动：

```text
tiny_codes.txt ──第1章 BPE──► merge_rules.pkl ──编码──► tiny_codes.bin
                                                          │
                                                   第3章 预训练
                                                          ▼
tiny_codes_sft.json ──第3章 SFT──► model_sft.pt ◄── model_pretrain.pt
                                        │
                          第3章 强化学习（81 道加法）
                                        ▼
                                  model_grpo.pt
```

训练语料、编码好的 `tiny_codes.bin` 和分词器的合并规则 `merge_rules.pkl` 都已经放在原书仓库里，所以即使跳过第 1 章，也可以直接做第 3 章。三个检查点（每个 44.5 MB）不在仓库里，放在 Hugging Face 上，运行 `uv run python tools/fetch_data.py` 就会下载。

```localcheck verify/test_part1.py::test_codebot_ships_data_and_tokenizer_in_git title="仓库里有数据和分词器，没有权重文件"
```

## I.2 预训练

预训练用的数据集叫 **TinyCodes**，文件是 `codebot/tiny_codes.txt`。它由 Hugging Face 上的数据集 [flytech/python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k)（MIT 许可）修改而来，原书作者在数据集 [koki0702/zero-llm-data](https://huggingface.co/datasets/koki0702/zero-llm-data) 的说明里写明了“Modified and used in the book”。整份文件 6,487,033 字节，是 21,836 个 Python 小程序，程序之间用特殊记号 `<|endoftext|>` 隔开（共 21,835 个），全部是 ASCII 字符。

先看看第一个程序。本书附带了语料开头的 500 个程序（`data/part1/tiny_codes_head.txt`），原样取自原文件：

```verify id=I.2-a title="TinyCodes 的第一个程序" files=data/part1/tiny_codes_head.txt
EOT = "<|endoftext|>"
head = open("data/part1/tiny_codes_head.txt", encoding="utf-8").read()
programs = head.split(EOT)
print(programs[0])
assert len(programs) == 500
assert programs[0].startswith("tasks = []\nwhile True:")
```

这是一个记录待办事项的小程序，读起来很像 Python。但仔细看第 3 行：单引号字符串里又套了一对单引号 `'done'`，Python 根本解析不了。这样的程序在语料里并不少见，我们用 Python 自己的解析器 `ast.parse` 数一数：

```verify id=I.2-b title="样本里有 17% 的程序不是合法的 Python"
import ast
import warnings

def parses(src):
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False

with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # 语料里有不少无效的转义序列，会触发 SyntaxWarning
    bad = [k for k, p in enumerate(programs) if not parses(p)]
print(f"前 500 个程序里，{len(bad)} 个解析失败（{len(bad) / len(programs):.0%}），例如第 {bad[:5]} 个")
assert len(bad) == 85 and not parses(programs[0])
```

在整份语料上，21,836 个程序里有 2,426 个解析失败，占 11.1%（开头的 500 个程序恰好更“脏”一些；本地测试用 Python 3.12 数这 500 个，也是 85 个，与浏览器里的 Python 3.14 一致）。这是生成数据常见的质量问题。它对分词器没有影响，但会影响 CodeBot 学到的东西：模型学的是语料里出现过的样子，包括这些错误。第 3 章生成代码时，你会看到语法“像 Python”、却经常跑不通的程序。

```localcheck verify/test_part1.py::test_tinycodes_full_corpus_stats title="在完整语料上重算：程序数、字节数、token 数、解析失败的比例"
```

预训练时，模型每次看的是长度为 256 个 token 的窗口，这就是它的**上下文长度**（context length）。程序有多长？下图是全部 21,836 个程序的长度分布（用 CodeBot 的分词器计数），竖线是上下文长度：

```widget part1-doclen caption="TinyCodes 程序长度的累计分布。九成程序不超过 256 个 token，一个上下文窗口就放得下。"
{"datasets": [{"name": "TinyCodes", "file": "data/part1/doclen.json"}], "xMax": 800}
```

```verify id=I.2-c title="90.3% 的程序不超过上下文长度 256" files=data/part1/doclen.json
import numpy as np

stats = json.load(open("data/part1/doclen.json", encoding="utf-8"))
lengths = np.array(stats["lengths"])
print(f"{len(lengths):,} 个程序 + {stats['separators']:,} 个分隔记号 = {lengths.sum() + stats['separators']:,} 个 token")
print(f"中位数 {np.median(lengths):.0f}，90% 分位数 {np.percentile(lengths, 90):.0f}，最长 {lengths.max()}")
print(f"超过 256 个 token 的程序：{(lengths > 256).sum():,} 个（{(lengths > 256).mean():.2%}）")
assert len(lengths) == 21_836 and lengths.sum() + 21_835 == 2_677_398
assert np.median(lengths) == 92 and (lengths > 256).sum() == 2_113
```

接着看预训练的预算。超参数写在 `ch03/01_pretrain.py` 的开头：

```listing ch03/01_pretrain.py#L21-L31
```

每一步读入 32 × 256 = 8,192 个 token，2 万步一共是 1.64 亿个 token。而整份语料编码后只有 267.7 万个 token，相当于把同一份语料反复看 61 遍。这是有意为之的小规模实验，第 3 章会讨论它对模型意味着什么。

```verify id=I.2-d title="2 万步 × 8,192 token = 语料的 61 遍"
tokens_seen = 20_000 * 32 * 256
print(f"{tokens_seen:,} token ÷ {stats['tokens']:,} token = {tokens_seen / stats['tokens']:.1f} 遍")
assert tokens_seen == 163_840_000 and round(tokens_seen / stats["tokens"]) == 61
```

下图是本书在本机 RTX 3060 Ti 上原样运行这个脚本时记录的 loss 曲线，图下是这次运行的硬件、耗时等信息：

```widget runcurve caption="CodeBot 预训练的 loss。虚线是 ln 1000：曲线从它附近出发，2 万步后降到 0.6 左右。"
{"run": "codebot-3060ti", "step": "01_pretrain", "metrics": ["loss"], "yLabel": "loss（nats）", "xLabel": "iteration", "refLines": [{"y": 6.9078, "label": "ln 1000"}], "command": "uv run python gpu/run_local.py --run codebot-3060ti --script ch03/01_pretrain.py"}
```

```verify id=I.2-e title="运行记录：第一步的 loss 就在 ln 1000 附近" files=data/runs/codebot-3060ti/report.json,data/runs/codebot-3060ti/01_pretrain.metrics.jsonl
import math

report = json.load(open("data/runs/codebot-3060ti/report.json", encoding="utf-8"))
pre = next(s for s in report["steps"] if s["script"] == "ch03/01_pretrain.py")
with open("data/runs/codebot-3060ti/01_pretrain.metrics.jsonl", encoding="utf-8") as f:
    first = json.loads(f.readline())
print(f"{report['host']['gpu']}，{pre['seconds'] / 60:.1f} 分钟；"
      f"第一步 loss {first['loss']:.4f}（ln 1000 = {math.log(1000):.4f}），最后一步 {pre['final']['loss']:.4f}")
assert abs(first["loss"] - math.log(1000)) < 0.2
assert pre["final"]["it"] == 20_000 and pre["final"]["loss"] < 1.0
```

loss 从 ln 1000 ≈ 6.91 附近出发，这正是第 0 章说的“平均地猜”；最后降到 0.6 左右。不过 CodeBot 在这次预训练里没有验证集，这个数字是在训练数据上算的。把语料看了 61 遍之后，它在多大程度上是“背下来”了，第 3 章再谈。

## I.3 SFT（监督微调）

预训练之后的模型只会续写。给它 `def`，它接着写出一个排序函数；把问题套进对话模板交给它，它却多半一开口就输出结束记号，什么也不说。上面对话图里“只做了预训练”那一栏就是这个样子：6 个问题里有 4 个空着，剩下两个续写出了不相干的代码。

```verify id=I.3-a title="只做了预训练的模型：会续写代码，不会接话"
base = rec["models"]["pretrain"]
print(base["def_continuation"])
empty = [p for p, r in zip(rec["prompts"], base["greedy"]) if r == ""]
print("回答为空的问题：", empty)
assert base["def_continuation"].startswith("def sort_list(lst):")
assert empty == ["Hello", "Who are you?", "write loop", "7+8="]
```

要让它像助手一样作答，需要用“指令 → 回答”的成对数据再训练一次，这就是 **SFT**（Supervised Fine-Tuning，监督微调）。CodeBot 的 SFT 数据是 `codebot/tiny_codes_sft.json`，共 1,092 条（原书仓库没有注明来源）。内容大致有四类：寒暄和自我介绍，“写一个变量 / 循环 / 函数”这样的编程指令，“什么是左移”这样的概念问答，以及最后 81 条个位数加法。我们把整份文件附在 `data/part1/` 里，在浏览器里数一数：

```verify id=I.3-b title="1,092 条 SFT 数据，最后 81 条是 1+1= 到 9+9=" files=data/part1/tiny_codes_sft.json
sft = json.load(open("data/part1/tiny_codes_sft.json", encoding="utf-8"))
for k in (0, 360, 540, 1091):
    print(f"#{k:<5} {sft[k]['instruction']!r:30} → {sft[k]['response'].splitlines()[0]!r}")
print(f"共 {len(sft):,} 条；回答长度的中位数 {np.median([len(x['response']) for x in sft]):.0f} 个字符；"
      f"多行的回答（通常是代码）{sum(chr(10) in x['response'] for x in sft)} 条")
print(f"SFT 训练 500 步 × 每步 32 条 = 16,000 条，相当于把数据看了 {500 * 32 / len(sft):.1f} 遍")
assert len(sft) == 1092
assert [x["instruction"] for x in sft[-81:]] == [f"{i}+{j}=" for i in range(1, 10) for j in range(1, 10)]
assert all(x["response"] == str(int(x["instruction"][0]) + int(x["instruction"][2])) for x in sft[-81:])
assert round(500 * 32 / len(sft), 1) == 14.7
```

每条数据都要套进一个固定的模板（Alpaca 格式），模型从中学会“`### Response:` 之后该轮到我说话了”：

```listing ch03/03_alpaca.py#L20-L27
```

训练时只在回答部分计算 loss，指令部分不算，也就是第 0 章 [验证 0.4-a](ch00.html#v-0.4-a) 里权重为 0 的那几个位置。SFT 跑 500 步，每步 32 条，相当于把这 1,092 条数据看了 14.7 遍。对话图里“SFT 之后”的回答，就是这一步的产物。

## I.4 强化学习

有意思的是，那 81 道加法题明明都在 SFT 数据里，SFT 之后的 CodeBot 却只答对了其中 44 道（54.3%）。照着标准答案模仿还不够，第 3 章用**强化学习**（reinforcement learning）继续训练：每道题让模型自己回答 8 次，答对得 1 分、答错得 0 分；比同一道题的平均分高的回答被鼓励，低的被抑制。打分规则就是下面这个函数，它取回答里最后一个整数，和正确答案比较：

```listing ch03/09_grpo.py#L64-L72
```

这个算法叫 **GRPO**（Group Relative Policy Optimization），“组”指的就是同一道题的 8 个回答。每一步取 32 道题，共 256 个回答，训练 500 步。结果如下，格子里是模型对 $i + j$ 的回答（行是 $i$，列是 $j$）：

```widget part1-arith caption="强化学习把加法的正确数从 44 道提高到 67 道：修好了 29 道，也弄坏了 6 道。"
{"mode": "diff"}
```

```verify id=I.4-a title="用原书的打分规则，重算 81 道题的成绩"
import re

def calculate_reward(ground_truth, response):     # 与 ch03/09_grpo.py 第 64–72 行的判定相同
    matches = re.findall(r"(-?\d+)", response)
    return 1.0 if matches and int(matches[-1]) == ground_truth else 0.0

correct = {}
for name in ("sft", "grpo"):
    answers = rec["models"][name]["addition_greedy"]
    correct[name] = {q for q, a in answers.items() if calculate_reward(int(q[0]) + int(q[2]), a) > 0}
    print(f"{name:4} 答对 {len(correct[name])} / 81（{len(correct[name]) / 81:.1%}）")
fixed, broke = correct["grpo"] - correct["sft"], correct["sft"] - correct["grpo"]
print(f"强化学习修好了 {len(fixed)} 道，弄坏了 {len(broke)} 道：{sorted(broke)}")
assert (len(correct["sft"]), len(correct["grpo"])) == (44, 67)
assert (len(fixed), len(broke)) == (29, 6) and "7+8" not in correct["grpo"]
```

强化学习之后，正确率从 54.3% 升到了 82.7%。不过这个进步不是单向的：有 6 道原来答对的题反而答错了，`7+8` 则自始至终没有答对。组内 8 个回答全对或全错的题为什么学不到东西，训练时显示的 loss 为什么不能当进度条看，第 3 章会细说。

::: warn 这 81 道题既是训练题，也是考试题
原书的 GRPO 脚本每 10 步测一次准确率，用的正是训练时的这 81 道题（上图也是）。所以 82.7% 是训练集上的成绩，不能说明模型学会了“加法”这件事本身。对于一个只有 81 道题、可以穷举的玩具任务，这样做可以理解；但读论文或自己做实验时，要分清“训练中看到的分数”和“在留出数据上的评测”。
:::

::: cs336 CS336 A5 的 GRPO 是同一个形状
CS336 的作业 A5（2026 年版）用 GRPO 训练 OLMo-2 1B 做 GSM8K 数学题，默认每步 32 道题、每题 8 个回答，共 256 个回答，与原书 CodeBot 的设置形状相同。原书的 advantage 只减组内平均、不除以标准差，loss 按回答条数归一化，也没有 KL 惩罚项，这几点与 A5 里称为 Dr. GRPO 的变体一致；此外它还用了 PPO 式的裁剪（ε = 0.2），每批回答更新两次。A5 则要求在标准做法、Dr. GRPO 等几种变体之间做对比实验。资料来源：[CS336 A5 仓库](https://github.com/stanford-cs336/assignment5-alignment)。
:::

## 本部要点

CodeBot 是全书的基本功。它的数据只有 6.5 MB，模型只有 1,112 万个参数，却完整地走过了分词器、模型、预训练、SFT、强化学习这条路。完成后的 CodeBot 会打招呼、会写简单的代码，个位数加法也练到了八成以上的正确率，但那是在训练题上测出来的，而且有几道题始终学不会。

**本部所学的内容**

- 官方 CodeBot 检查点会自我介绍、会写循环，但两个检查点都把 `7+8=` 答成 16（[验证 I-a](#v-I-a)）。
- TinyCodes 有 21,836 个程序；开头 500 个里有 85 个不是合法的 Python，整份语料是 11.1%（[验证 I.2-a](#v-I.2-a)、[验证 I.2-b](#v-I.2-b)）。
- 九成程序不超过上下文长度 256 个 token；预训练把 267.7 万 token 的语料看了 61 遍（[验证 I.2-c](#v-I.2-c)、[验证 I.2-d](#v-I.2-d)）。
- 只做预训练的模型会续写代码，却不会接话；SFT 数据有 1,092 条，最后 81 条正是 1+1= 到 9+9=（[验证 I.3-a](#v-I.3-a)、[验证 I.3-b](#v-I.3-b)）。
- 强化学习把加法的正确数从 44 提高到 67，修好 29 道、弄坏 6 道，而且是在训练题上测的（[验证 I.4-a](#v-I.4-a)）。

下一章，我们从最朴素的问题出发：一段文字，怎样变成一串整数？
