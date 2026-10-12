---
chapter: Z
slug: errata
order: 200
kind: extra
title: 原书代码的已知问题与修复
nav_title: 已知问题与修复
lede: 把原书仓库的脚本逐个跑过、拆开来测之后，我们整理出了 50 个问题。每一条都写明严重度、能重跑的最小复现、修法，以及它会不会改变书中的结果。
pyodide_packages: [numpy, regex]
---

写这个教程的过程中，我们把 DLFS6 仓库（固定在 commit `c9b6e2e`）里的脚本逐个跑了一遍，把关键函数一个个拿出来单独测试，还拿斯坦福 CS336 的公开测试去对照了一遍。绝大多数代码都经得起这样的折腾，但也碰到了一些问题：有的会让第 3 章 GRPO 的准确率在前 200 次迭代里低二三十个百分点，有的会让第 6 章的脚本在某种用法下根本跑不起来，更多的是注释过期、边界情况没有处理之类的小毛病。

“逐个跑了一遍”要说清楚范围。规模大的、或者要花钱调用 API 的脚本，我们没有全部原样跑完。下表是每一类脚本实际的运行方式：

| 脚本 | 我们怎么跑的 |
|---|---|
| 第 3 章的预训练、SFT、GRPO | 本机的 RTX 3060 Ti，原样跑完（运行记录 codebot-3060ti） |
| 第 6 章的预训练、生成、DPO | Modal 的一张 H100，原样跑完（运行记录 storybot-h100） |
| 第 4 章的 BPE 训练与编码 | 在 TinyStories 的切片和完整的验证集上运行，产物与官方文件逐字节相同；2.2 GB 的完整训练集没有跑 |
| 第 7 章的 BPE 训练与编码 | 换成 OWT 验证集的前 30 MiB |
| 第 6 章的两个 LLM 评审脚本 | 用离线的模拟评审打通流程，没有调用付费的 API |
| `ch09/01_pretrain.py` | 在 H100 上只跑了 200 步的计时探针（运行记录 webbot-probe） |
| `ch09/02_generate.py` | 没有发布的 WebBot 权重（[E48](#e48)），只用一个训练了 30 步的小模型跑通代码 |
| 其余脚本 | 本机 CPU，原样运行（交互式的 `ch03/04_chat.py` 改成从标准输入读） |

原书 README 里写着正误表的地址（[`…/wiki/errata`](https://github.com/oreilly-japan/deep-learning-from-scratch-6/wiki/errata)），可是这个页面并不存在：2026-10-12 访问时，它先被重定向到 `/wiki`，再回到仓库首页。GitHub 上的 issue 也只有寥寥几条。于是有了这一页。这里只收**代码本身的问题**。DLFS6 与 CS336 在设计上的取舍不同（比如学习率按线性衰减到 0，而不是 cosine），那不是错误，放在 [CS336 对照总表](cs336.html)里讨论。

每一条都配了复现：能在浏览器里演示的，写成带“验”字印章的验证单元；需要 PyTorch 或真实权重的，写成本地测试（`verify/test_reference.py` 里名为 `test_eNN_…` 的函数），页面上可以展开看到测试源码。先看一览表，再按严重度逐条展开。

## Z.1 一览

严重度分三档。**高**：照着书原样运行，结果会变差，或者某一章的主线脚本直接报错。**中**：只在某些输入或某些用法下出错，而且往往是静默地出错；或者会删掉不该删的东西。**低**：数值细节、接口不一致、注释和文档的问题。“影响结果”一栏只回答一个问题：照着书原样运行，得到的数字会不会因此不同。

下面的筛选器可以按严重度、所在的章、是否影响结果来过滤表格。

```widget errata-board caption="按严重度、章节和“是否影响书中结果”筛选下面的一览表。计数随筛选实时更新。"
{"table": "#errata-index table"}
```

<div id="errata-index">

| 编号 | 严重度 | 影响结果 | 问题 | 位置 |
|---|---|---|---|---|
| [E01](#e01) | 高 | 是 | GRPO 更新时策略模型还开着 dropout | ch03/09_grpo.py |
| [E02](#e02) | 中 | 否（只在使用下载的权重时报错） | HF 上的 StoryBot 权重用 `mlp`，代码用 `ffn` | storybot/model.py；ch06/06–09 |
| [E03](#e03) | 中 | 否（书中流程不触发） | 把 git 检出的文本交给第 4 章的分词器，CRLF 会改变词表 | storybot/tokenizer.py（输入是经 git 检出的文本文件时） |
| [E04](#e04) | 中 | 仅长生成 | 生成超过 256 个 token 后，RoPE 的位置被冻结 | storybot/model.py；storybot/utils.py |
| [E05](#e05) | 中 | 否 | 输入比上限恰好多 1 个 token 时，位置信息被静默丢掉 | storybot、webbot 的 model.py |
| [E06](#e06) | 中 | 否 | 带缓存时一次喂多个 token，没有因果掩码 | storybot、webbot 的 model.py |
| [E07](#e07) | 中 | 部分 | `cycle(DataLoader)` 只洗一次牌，还留住每个 batch | ch03/01、03_sft、09；ch06/08 |
| [E08](#e08) | 中 | 否 | `encode_file` 会删掉整个 `cache_dir` | storybot/tokenizer.py |
| [E09](#e09) | 中 | 否 | 奖励函数的裸 `except` 吞掉一切错误 | ch03/09_grpo.py |
| [E10](#e10) | 中 | 未测量 | DPO 把 prompt 和回答分开编码 | ch06/08_dpo.py |
| [E11](#e11) | 中 | 否 | 用户文本里的 `<\|endoftext\|>` 被当成特殊 token | codebot、storybot 的 tokenizer.py |
| [E12](#e12) | 中 | 否（无法运行） | 第 4.5 节用到的 ch04/05_eval.py 不在仓库里 | ch04/ |
| [E13](#e13) | 低 | 否 | `get_batch` 永远采不到最后一个窗口 | ch06/05、ch09/01 |
| [E14](#e14) | 低 | 几乎没有 | `evaluate` 少算最后一个完整窗口 | ch06/05、ch09/01 |
| [E15](#e15) | 低 | 否（曲线缺了起点） | GRPO 的第 0 次准确率是两次更新之后测的 | ch03/09_grpo.py |
| [E16](#e16) | 低 | 否 | ratio 在概率空间计算，极小概率时有偏 | ch03/09_grpo.py |
| [E17](#e17) | 低 | 否 | 两套 RMSNorm 的 eps 不一致 | ch05/03；storybot、webbot |
| [E18](#e18) | 低 | 否 | CodeBot 实际用的是 erf 版 GELU 和 `nn.LayerNorm` | codebot/model.py |
| [E19](#e19) | 低 | 否 | `head_dim` 不检查能否整除 | storybot、webbot 的 Block |
| [E20](#e20) | 低 | 否 | EOS 不在 GRPO 的学习目标里 | ch03/09_grpo.py |
| [E21](#e21) | 低 | 否 | GRPO 生成不设长度上限，CodeBot 只有 256 个位置 | ch03/09_grpo.py；codebot |
| [E22](#e22) | 低 | 否 | `generate()` 会把模型留在 eval 模式 | codebot、storybot 的 utils.py |
| [E23](#e23) | 低 | 否 | 温度极小（低于约 1e-38）时 softmax 得到 NaN | codebot、storybot 的 utils.py |
| [E24](#e24) | 低 | 否 | 逐个 token 解码会冒出 U+FFFD | codebot、storybot 的 tokenizer.py |
| [E25](#e25) | 低 | 否 | 用 uint16 存 id，词表不能超过 65,536 | ch01/09；storybot/tokenizer.py |
| [E26](#e26) | 低 | 否 | 两个 `count_pairs` 的参数不一样 | codebot、storybot 的 tokenizer.py |
| [E27](#e27) | 低 | 否 | `merge_rules` 的顺序本身就是数据 | codebot、storybot 的 tokenizer.py |
| [E28](#e28) | 低 | 否 | `measure_generation_time` 清缓存的条件写反了 | ch05/05_kvcache.py |
| [E29](#e29) | 低 | 仅中文 Windows | 读文本不指定编码，中文 Windows 上静默乱码 | ch01/07–09；ch03/03_alpaca、03_sft；ch04/01、02、06；ch06/08 |
| [E30](#e30) | 低 | 否 | 软字典的 softmax 没有减去最大值 | ch02/01_soft_dict.py |
| [E31](#e31) | 低 | 否 | RoPE 的 cos/sin 表存进了每一层的检查点 | storybot、webbot 的 model.py |
| [E32](#e32) | 低 | 否 | GRPO 评估时把“提示 + 回答”整段交给奖励函数 | ch03/09_grpo.py |
| [E33](#e33) | 低 | 否 | 用 pickle 保存与读取分词规则 | codebot、storybot 的 tokenizer.py |
| [E34](#e34) | 低 | 否 | 注释里的 id 930 应为 932 | ch03/03_alpaca.py |
| [E35](#e35) | 低 | 否 | 三处注释与代码不符 | ch05/05_kvcache.py |
| [E36](#e36) | 低 | 否 | README 写的是 ch02/10_gpt.py | README.md |
| [E37](#e37) | 低 | 否 | README 的正误表链接打不开 | README.md |
| [E38](#e38) | 低 | 否 | README 写 PyTorch 2.x，实际需要 2.5 以上 | README.md；requirements.txt |
| [E39](#e39) | 低 | 否 | `RMSNorm.__init__(self, x)` 的参数名 | ch05/03_rmsnorm.py |
| [E40](#e40) | 低 | 否 | 注释把头数叫作“ヘッドサイズ” | ch02/08_multi_head.py |
| [E41](#e41) | 低 | 否 | `loss_grpo.png` 画的是准确率 | ch03/09_grpo.py |
| [E42](#e42) | 低 | 否 | 字体写死成 macOS 的 Hiragino Sans | ch06/lr_graph.py |
| [E43](#e43) | 低 | 否 | 训练脚本都没有设随机种子 | ch03、ch06 的训练脚本 |
| [E44](#e44) | 低 | 否 | `autocast` 写死了 `device_type='cuda'` | ch09/01_pretrain.py |
| [E45](#e45) | 低 | 否 | 多卡训练的几处日志细节 | ch09/01_pretrain.py |
| [E46](#e46) | 低 | 否 | 一份与 `pretoken_chunk` 几乎相同的死代码 | storybot/tokenizer.py |
| [E47](#e47) | 低 | 否 | 压缩率只在前 10,000 个字符上测 | ch01/08_eval.py |
| [E48](#e48) | 低 | 否 | `model_dpo.pt` 与 WebBot 权重没有发布 | storybot/download.sh；HF |
| [E49](#e49) | 低 | 否（梯度相同，只是多通信） | `no_sync()` 只包住 backward，每个 micro-step 都在通信 | ch09/01_pretrain.py |
| [E50](#e50) | 中 | 仅多卡 | `max_iters` 写死，开 8 张卡就读 8 倍的 token | ch09/01_pretrain.py |

</div>

## Z.2 原书注释即断言

先从一个整体的体检说起。DLFS6 的脚本有个好习惯：很多 `print` 的后面直接用注释写出了期望的输出，比如 `print(ids)  # [259, 108, ...]`。这些注释其实就是一条条断言，只是没有人去运行它们。我们写了一个测试，把它们全部找出来，改写成断言再运行。

具体做法是：用 Python 的 `ast` 与 `tokenize` 模块找出所有“只有一个参数、并且带期望输出注释”的 `print`，一共 38 处（35 处写在同一行，3 处写在下方的注释块里）；把每个 `print(x)` 改写成记录 `x` 的函数调用，在仓库的临时副本里运行脚本（不往 `third_party/dlfs6` 写任何东西），最后逐条比较。比较分几档：打印出来的文字与注释逐字相同，算“逐字一致”；注释写的是 `repr` 而 `print` 显示的是 `str`（如 `'h'` 与 `h`），或者注释省掉了 `defaultdict(...)`、`torch.Size(...)` 这层外壳，算“表示形式不同”；其余都算不一致。

```verify id=Z.2-a title="38 处注释：30 处逐字一致，7 处只是写法不同，1 处过期" files=data/errata/book_comments.json
import json

s = json.load(open("data/errata/book_comments.json", encoding="utf-8"))
print(f"带期望输出注释的 print：{s['total']} 处，分布在 {s['scripts']} 个脚本里")
names = {"exact": "逐字一致", "repr": "repr 与 str", "defaultdict": "省略 defaultdict",
         "torch.Size": "省略 torch.Size", "stale": "注释过期"}
for k, label in names.items():
    print(f"  {label}：{s['totals'][k]}")
st = s["stale"][0]
print(f"过期的一处：{st['script']} 第 {st['line']} 行的 print（期望值写在第 {st['line'] + 1} 行的注释里），"
      f"第 {st['index'] + 1} 个 id（下标 {st['index']}）注释写 {st['comment']}（{st['comment_token']!r}），"
      f"实际是 {st['actual']}（{st['actual_token']!r}）")
assert s["total"] == 38
assert s["totals"] == {"exact": 30, "repr": 3, "defaultdict": 1, "torch.Size": 3, "stale": 1}
```

38 处里只有 1 处真的对不上，它就是下面的 [E34](#e34)。另外 7 处只是写法上的差别，值并没有错。

```localcheck verify/test_book_comments.py::test_book_comments_hold title="逐个脚本运行，注释与输出一致（过期的一处必须恰好差在第 31 个 id，也就是下标 30）"
```

## Z.3 高：会改变结果，或让主线脚本跑不起来

<div id="e01"></div>

### E01 GRPO 更新时，策略模型还开着 dropout

在我们找到的问题里，这一个对结果的影响最大。第 3 章的 GRPO 脚本里有两份模型：用来采样的 `old_model`，和被更新的 `model`。前者显式调用了 `eval()`，后者没有：

```listing ch03/09_grpo.py#L148-L153
```

`GPT.load_from` 返回的模型处于 PyTorch 的默认状态，也就是训练模式，CodeBot 的 dropout（比率 0.1）是开着的。于是在计算 loss 时，分子 `probs` 来自一个“随机丢掉一部分神经元”的模型，分母 `old_probs` 来自完整的模型：

```listing ch03/09_grpo.py#L110-L118
```

按 PPO 的设计，每批数据的第一次更新时两份模型的权重完全相同，比值 ratio 应该处处等于 1，裁剪（clip）还不起作用。可是有了 dropout，ratio 变成了一个随机数。我们拿作者在 Hugging Face 上发布的 SFT 权重 `model_sft.pt`（`codebot/download.sh` 里这一行被注释掉了，要手动下载），把 81 道加法题的正确答案当作回答喂进去（共 117 个回答 token），实测在训练模式下，约一半的 token 的 ratio 落在裁剪区间 [0.8, 1.2] 之外（换 20 个随机种子，这个比例在 42% 到 58% 之间，平均 48%）；切到 eval 模式后，这个比例是 0，ratio 与 1 的最大偏差不到 $10^{-5}$。

```localcheck verify/test_reference.py::test_e01_grpo_policy_updates_in_train_mode needs="PyTorch" title="作者发布的 SFT 权重：训练模式下约一半的 ratio 超出裁剪区间，eval 模式下全部等于 1"
```

那么，这对训练结果有多大影响呢？我们在本机的 RTX 3060 Ti 上把原书的 `09_grpo.py` 原样跑满 500 次迭代（运行记录 codebot-3060ti，用时 1 小时 29 分）；再从这次运行训练出的 SFT 检查点出发，只在更新循环之前加一行 `model.eval()`，又跑了一遍（运行记录 codebot-grpo-evalfix-3060ti，用时 1 小时 16 分）。下图默认显示这两次运行的 greedy 准确率（81 道题，每 10 次迭代测一次）。

```widget errata-grpo caption="同一个 SFT 起点，唯一的差别是更新前是否调用 model.eval()。关掉 dropout 后，51 次迭代就到了 98.8%，第 131 次起一直是 100%；原样运行跑满 500 次，停在 86.4%。另外两个选项是研究阶段在 CPU 上的对照，方向相同。"
{"runs": {"base": "data/runs/codebot-3060ti/09_grpo.metrics.jsonl", "fix": "data/runs/codebot-grpo-evalfix-3060ti/09_grpo.metrics.jsonl"}, "data": "data/errata/grpo_dropout_runs.json"}
```

```verify id=Z.3-a title="书中脚本跑满 500 次：第 51 次迭代 60.5% 对 98.8%，最终 86.4% 对 100%" files=data/runs/codebot-3060ti/09_grpo.metrics.jsonl,data/runs/codebot-3060ti/report.json,data/runs/codebot-grpo-evalfix-3060ti/09_grpo.metrics.jsonl,data/runs/codebot-grpo-evalfix-3060ti/report.json
import json

def grpo_run(run_id):
    with open(f"data/runs/{run_id}/09_grpo.metrics.jsonl", encoding="utf-8") as f:
        acc = {r["it"]: r["acc"] for r in map(json.loads, f)}       # 第 it 次迭代之后显示的准确率（%）
    with open(f"data/runs/{run_id}/report.json", encoding="utf-8") as f:
        step = next(s for s in json.load(f)["steps"] if s["script"] == "ch03/09_grpo.py")
    return acc, step

base, step_b = grpo_run("codebot-3060ti")
fix, step_f = grpo_run("codebot-grpo-evalfix-3060ti")
seeded = json.load(open("data/runs/codebot-grpo-evalfix-3060ti/report.json", encoding="utf-8"))["seeded"][0]
print("唯一的改动：", step_f["patches"][0]["new"].strip().splitlines()[0])
print("出发点：", seeded["path"], "来自", seeded["from"], "，sha256", seeded["sha256"][:12] + "…")
for it in (1, 21, 51, 101, 201, 500):
    print(f"第 {it:3d} 次迭代后：原样 {base[it]:5.1f}%    加 model.eval() {fix[it]:5.1f}%")
evals = range(1, 500, 10)                                   # 评估在第 1、11、21……次迭代的两次更新之后
gaps = [fix[i] - base[i] for i in evals if 21 <= i <= 201]
first100 = min(i for i in evals if fix[i] == 100.0)
print(f"第 21–201 次迭代之间，差距在 {min(gaps):.1f} 到 {max(gaps):.1f} 个百分点之间；"
      f"加 eval() 的从第 {first100} 次起一直是 100%")
for name, st in (("原样", step_b), ("加 eval()", step_f)):
    print(f"用时（{name}）：{int(st['seconds']) // 3600} 小时 {int(st['seconds']) % 3600 // 60} 分")
assert len(step_f["patches"]) == 1 and step_f["patches"][0]["new"].lstrip().startswith("model.eval()")
assert seeded["from"] == "tutorial run codebot-3060ti"
assert (base[51], fix[51], base[500], fix[500]) == (60.5, 98.8, 86.4, 100.0)
assert first100 == 131 and all(fix[i] == 100.0 for i in fix if i >= first100)
assert 20 < min(gaps) and max(gaps) < 40 and round(fix[500] - base[500], 1) == 13.6
```

第 51 次迭代后的评估，原样运行是 60.5%，加了 `eval()` 的是 98.8%。加了 `eval()` 的从第 131 次迭代起一直是 100%；原样运行跑满 500 次，停在 86.4%。在第 21 到第 201 次迭代之间，两者的差距是 21 到 38 个百分点，到最后还差 13.6 个百分点。把两次运行最后保存的 `model_grpo.pt` 拿到 CPU 上重测，得到的也是 86.4% 与 100%：

```localcheck verify/test_reference.py::test_e01_recorded_grpo_checkpoints_on_cpu needs="PyTorch" title="本机的两份 GRPO 检查点在 CPU 上重测：70/81 与 81/81，回答里都没有两个以上的整数；共同的 SFT 起点是 44/81，sha256 与记录一致"
```

需要说明的是，两种设置各只跑了一次，而且脚本本身没有设随机种子（[E43](#e43)），具体的数字会随运行而变。研究阶段我们在 CPU 上做过同样的对照：批量采样版的 GRPO（算法和超参数都与原脚本相同），两个 seed 各跑 40–60 次迭代，从作者发布的 SFT 权重出发。方向是一致的，上图可以切换查看：

```verify id=Z.3-b title="研究阶段的 CPU 对照：两个 seed，关掉 dropout 都提高了 20 个百分点以上" files=data/errata/grpo_dropout_runs.json
runs = json.load(open("data/errata/grpo_dropout_runs.json", encoding="utf-8"))["runs"]
for seed, (a, b) in {0: ("base", "nodrop"), 1: ("base_s1", "nodrop_s1")}.items():
    ra, rb = runs[a], runs[b]
    print(f"seed {seed}，第 {ra['acc_it'][-1]} 次迭代：原样 {ra['greedy_acc'][-1]:.1f}%  →  "
          f"更新前 eval() {rb['greedy_acc'][-1]:.1f}%")
    assert rb["greedy_acc"][-1] - ra["greedy_acc"][-1] > 20
    assert all(y >= x for x, y in zip(ra["greedy_acc"][1:], rb["greedy_acc"][1:]))
```

机制本身是确定的：dropout 让第一次更新的 ratio 偏离 1，裁剪和梯度方向都跟着失真。

**修法**：在更新循环之前切到 eval 模式。`eval()` 只关掉 dropout，并不影响梯度的计算；评估之后脚本会调用 `model.train()`，因此每轮都要重新调用一次。上面那次 GPU 运行用的正是这一行：

```python
    # 生成データに対して複数回更新
    model.eval()                      # 新增：更新时也关掉 dropout
    for _ in range(n_update_per_generation):
        ...
```

另一种做法是构造模型时直接把 `dropout_rate` 设为 0。CS336 的模型从一开始就没有 dropout，所以不会遇到这个问题。

## Z.4 中：在某些用法下静默出错

<div id="e02"></div>

### E02 HF 上的 StoryBot 权重用 `mlp`，代码用 `ffn`

先说清楚什么时候会遇到它。第 6 章的生成、评估和 DPO 默认加载的是 `ch06/05_pretrain.py` 自己训练、自己保存的检查点，它们用的是现在的键名，不受影响。我们在 H100 上原样运行 05 → 06 → 08，三个脚本都顺利结束：

```verify id=Z.4-a title="按书自己训练：05 存下三个检查点，06、08 原样读取，都正常结束" files=data/runs/storybot-h100/report.json
import json

story = json.load(open("data/runs/storybot-h100/report.json", encoding="utf-8"))
for s in story["steps"]:
    saved = [p for p in s["produced"] if p.endswith(".pt")]
    print(f"{s['script']}：退出码 {s['exit_code']}，补丁 {len(s.get('patches', []))} 处"
          + (f"，保存了 {'、'.join(saved)}" if saved else ""))
assert [s["script"] for s in story["steps"]] == ["ch06/05_pretrain.py", "ch06/06_generate.py", "ch06/08_dpo.py"]
assert all(s["exit_code"] == 0 and not s.get("patches") for s in story["steps"])
assert {"storybot/model_pretrain.pt", "storybot/model_iter_500.pt",
        "storybot/model_iter_5000.pt"} <= set(story["steps"][0]["produced"])
```

问题出在另一条路上。`storybot/download.sh` 里有几行被注释掉了，取消注释就能直接下载作者在 Hugging Face 上发布的权重（`model_pretrain.pt`、`model_iter_500.pt`、`model_iter_5000.pt`），省掉 6.5 节的预训练。这三个文件保存时，SwiGLU 子模块叫 `mlp`；后来仓库在 commit `234a9c6`（2026-03-27，“原稿との整合性修正”）里把 `storybot/model.py` 的 `self.mlp` 改名为 `self.ffn`，权重却没有重新上传：

```listing storybot/model.py#L158-L170
```

结果是，用下载来的权重运行 `ch06/06_generate.py`、`07_llm_judge.py`、`08_dpo.py`、`09_llm_judge.py`，都会在 `load_state_dict` 报出 `Missing key(s) "blocks.0.ffn.W.weight" … Unexpected key(s) "blocks.0.mlp.W.weight"`。WebBot 的 `webbot/model.py` 仍然叫 `mlp`，两份代码之间也不一致。

```localcheck verify/test_reference.py::test_e02_hf_storybot_checkpoints_use_mlp_keys needs="PyTorch" title="三个发布的检查点都用 mlp，download.sh 里下载它们的几行都被注释掉了；改了 load_from 的那一行之后可以严格加载"
```

**修法**：加载时把键名改回来。`GPT.load_from` 的最后几行是这样的：

```listing storybot/model.py#L232-L249
```

只改其中的 `model.load_state_dict(checkpoint['model_state_dict'])` 这一行（第 246 行），换成：

```python
        model.load_state_dict({k.replace('.mlp.', '.ffn.'): v for k, v in checkpoint['model_state_dict'].items()})
```

<div id="e03"></div>

### E03 把 git 检出的文本交给第 4 章的分词器，CRLF 会改变词表

这一条只在一种组合下出现：在 `core.autocrlf=true` 的 Windows 上，用第 4 章之后以二进制模式读文件的分词器，去处理一个从 git 仓库检出的文本文件。`core.autocrlf=true` 是 Git for Windows 常见的配置（本机的全局和系统级配置都是这样），它会在检出时把文本文件的换行改成 `\r\n`。

先说书中的流程为什么恰好避开了它。DLFS6 仓库里经 git 检出的文本语料只有 `codebot/tiny_codes.txt` 一个，用到它的脚本（第 1 章的 07–09，第 4 章的 01、02、06）都以文本模式读取，Python 会把 `\r\n` 自动转回 `\n`。以二进制模式读语料的是第 4 章的 03、04、07 和第 7 章的 01、02，它们读的 TinyStories 和 OWT 由 `download.sh` 用 `wget` 下载，不经过 git，始终是 LF。

```localcheck verify/test_reference.py::test_e03_binary_mode_readers_only_read_downloaded_corpora title="扫描全部脚本的 open()：以二进制模式读语料的只有 TinyStories 和 OWT；tiny_codes.txt 只以文本模式读取"
```

可是只要把一个检出的文本文件交给 `storybot.tokenizer` 的 `train_bpe` 或 `encode_file`，问题就出现了。DLFS6 仓库没有 `.gitattributes`，在上面那样配置的机器上，`tiny_codes.txt` 检出后会从 6,487,033 字节变成 6,743,534 字节（多出的正好是 256,501 个换行前的 `\r`）。`storybot/tokenizer.py` 为了分块，用二进制模式读文件，`\r` 原封不动地留了下来：

```listing storybot/tokenizer.py#L99-L117
```

`\r\n` 在代码语料里每行都出现一次，自然是最常见的字节对之一，于是 BPE 很早就会把它学成一个 token。我们先用一个十几行的小 BPE（与第 4 章同样的预分词正则、同样的平局规则）在一段代码上演示：

```verify id=Z.4-b title="同一段代码，换行写成 \r\n 后，前几条 merge 里就出现了 b'\r\n'"
import regex
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe_toy(text, n_merges):
    words = Counter(tuple(t.encode("utf-8")) for t in regex.findall(PAT, text))
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, 256 + n_merges):
        pairs = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += c
        best = max(pairs, key=lambda p: (pairs[p], p[0], p[1]))   # DLFS6 的平局规则
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merged = Counter()
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(new_id); i += 2
                else:
                    out.append(w[i]); i += 1
            merged[tuple(out)] += c
        words = merged
    return [vocab[i] for i in range(256, 256 + n_merges)]

code = "def add(a, b):\n    return a + b\n\nfor i in range(3):\n    print(add(i, i))\n" * 30
lf = train_bpe_toy(code, 6)
crlf = train_bpe_toy(code.replace("\n", "\r\n"), 6)
print("LF  ：", lf)
print("CRLF：", crlf)
assert b"\r\n" not in lf and b"\r\n" in crlf
```

真实语料上也是一样。用书中的 `storybot.tokenizer.train_bpe` 在 tiny_codes 的前 200,000 个字符上训练，LF 版本里没有 `\r`；CRLF 版本的第 2 条 merge 就是 `(13, 10)`，也就是 `\r\n`。从这里开始，后面的词表全都对不上了。

```localcheck verify/test_reference.py::test_e03_crlf_checkout_changes_binary_mode_merges needs="PyTorch" title="书中的 train_bpe：CRLF 文件的第 2 条 merge 是 \r\n；文本模式读取则看不到 \r"
```

**修法**：克隆之前执行 `git config --global core.autocrlf false`，或者在仓库里加一个 `.gitattributes`（写上 `* text=auto eol=lf`）。本教程的项目就是这样做的，所以 `third_party/dlfs6` 里的 `tiny_codes.txt` 是 LF。如果已经检出了 CRLF 文件，也可以在读入后执行 `data.replace(b"\r\n", b"\n")`。

<div id="e04"></div>

### E04 生成超过 256 个 token 后，RoPE 的位置被冻结

StoryBot 的 `generate()` 用 KV cache 逐个生成 token：每一步只把新 token 喂给模型，缓存则一直增长，从不截断。函数开头那段截断 `ids` 的代码，对缓存没有任何作用，因为送进模型的只有 `next_id`：

```listing storybot/utils.py#L17-L21
```

问题出在 RoPE。位置一旦超过 `max_context_len`，`forward` 就把偏移量夹回最后一个合法位置：

```listing storybot/model.py#L31-L37
```

于是第 256 个 token 之后，每个新 token 都被当成“位置 255”。我们把这两行逻辑原样搬过来，看看生成 300 个 token 时实际用到的位置：

```verify id=Z.4-c title="300 个 token 只用到了 256 个位置：从第 255 个起全部停在 255"
max_ctx = 256

def rope_offset(offset, C, max_len=max_ctx):     # storybot/model.py 第 33–34 行
    if offset + C > max_len:
        offset = max_len - C
    return offset

used = [rope_offset(t, 1) for t in range(300)]    # 每步喂 1 个 token，cache_offset 每步加 1
print("第 254、255、256、299 个 token 的位置：", used[254], used[255], used[256], used[299])
print("300 个 token 只用到了", len(set(used)), "个不同的位置")
assert used[:256] == list(range(256)) and set(used[255:]) == {255}
```

在一个缩小的 StoryBot 上（`max_context_len=16`），我们把“带缓存生成”与“每步只看最近 16 个 token、重新计算”的结果逐位比较：前 16 个位置完全相同，从第 16 个位置开始就分叉了。

```localcheck verify/test_reference.py::test_e04_storybot_cache_grows_and_rope_position_freezes needs="PyTorch" title="缓存长到 30（上限 16）；与滑窗重算在第 16 个位置开始分叉"
```

这会影响书中的结果吗？`ch06/06_generate.py` 设的是 `max_new_tokens=300`，足以越过这条线。长故事并不罕见：TinyStories 验证集的 27,630 个故事里，有 10.9% 超过 256 个 token（中位数是 176 个）。模型写到这么长时，后半段的位置信息就失真了；在这之前结束的故事则不受影响。WebBot 的 `webbot/model.py` 会把缓存截断到最近 `max_context_len` 个，可是位置同样被夹在最后，也只是近似。

**修法**：缓存满了以后，清空缓存，用最近 `max_context_len - 1` 个 token 重新做一次 prefill（滑窗重算）；最简单的做法是限制 `max_new_tokens`，让“提示 + 生成”不超过 256。

<div id="e05"></div>

### E05 输入比上限恰好多 1 个 token 时，位置信息被静默丢掉

还是上面那两行。不用缓存时（训练、评估、prefill），偏移量从 0 开始；如果输入的长度 C 超过了 `max_context_len`，偏移量就会变成负数。负数切片在 Python 里有特殊含义，结果因长度而异：

```verify id=Z.4-d title="C = max+1 时只取到 1 行 cos/sin，被广播到所有位置；C = max+2 时形状对不上"
import numpy as np

max_len, d = 8, 4
cos_cache = np.arange(max_len).reshape(-1, 1) * np.ones((1, d))   # 每一行记下它的位置
for C in (8, 9, 10):
    offset = 0
    if offset + C > max_len:                    # 同一段夹取逻辑
        offset = max_len - C
    rows = cos_cache[offset:offset + C]
    x = np.ones((C, d))
    try:
        used = (x * rows)[:, 0].astype(int).tolist()
        print(f"C={C:2d}  offset={offset:2d}  取到 {rows.shape[0]} 行  各 token 的位置 {used}")
    except ValueError as e:
        print(f"C={C:2d}  offset={offset:2d}  取到 {rows.shape[0]} 行  广播失败：{e}")
assert cos_cache[-1:8].shape[0] == 1 and cos_cache[-2:8].shape[0] == 2
```

C 等于 `max_context_len + 1` 时，切片 `[-1:max_len]` 只取到最后一行，它被广播到所有位置：每个 token 都按“位置 7”旋转，模型失去了全部位置信息，却不会报任何错误。长度再多 1，形状就对不上了，这时才会报错。WebBot 的 `webbot/model.py` 里有完全相同的两行，上限是 1,024，所以 1,025 个 token 的输入同样会悄悄丢掉位置信息。

```localcheck verify/test_reference.py::test_e05_rope_length_max_plus_one_is_silently_wrong needs="PyTorch" title="StoryBot 与 WebBot 的 RoPE：9 个 token 全部按位置 7 旋转；10 个 token 才报错"
```

在书中的用法里，训练窗口恰好等于上限，不会触发；但 `generate()` 不截断过长的提示，一段 257 个 token 的提示就会悄悄地丢掉位置信息。**修法**：在 `forward` 开头断言 `offset + C <= max_context_len`，或者在 `generate()` 里先把提示截断到最近的 `max_context_len` 个 token。

<div id="e06"></div>

### E06 带缓存时一次喂多个 token，没有因果掩码

StoryBot 的注意力只在“第一次调用”时加因果掩码：

```listing storybot/model.py#L115-L121
```

注释说得很清楚：解码阶段一次只来一个 token，它本来就该看见全部缓存，不需要掩码。这个前提在书中成立，因为 `generate()` 每步只喂一个 token。可是一旦有人一次喂进多个新 token（比如分块 prefill、推测解码），这几个 token 之间就会互相看见，前面的 token 看见了后面的 token。实测：先 prefill 3 个 token，再一次喂入 2 个，第一个新 token 的 logits 与整段重新计算的结果相差超过 $10^{-3}$；逐个喂入则完全一致。WebBot 用的是 `F.scaled_dot_product_attention`，写法是 `is_causal = is_first_call`，问题相同。

```localcheck verify/test_reference.py::test_e06_multi_token_call_after_prefill_is_unmasked needs="PyTorch" title="StoryBot 与 WebBot：一次喂两个新 token，第一个看见了第二个；逐个喂入则与整段计算一致"
```

**修法**：非首次调用时如果 C > 1，要用“右下角对齐”的掩码：新 token 的第 i 个可以看见全部缓存加上前 i 个新 token，可以写成 `torch.ones(C, K).tril(diagonal=K - C)`，K 是缓存后的总长度。WebBot 不能直接改成 `is_causal=True`：查询与键的长度不同时，PyTorch 的 `is_causal` 按左上角对齐，同样不对。要把上面的掩码转成布尔型，通过 `attn_mask` 传进去。

<div id="e07"></div>

### E07 `cycle(DataLoader)` 只洗一次牌，还留住每个 batch

第 3 章的预训练、SFT、GRPO 和第 6 章的 DPO 都用同一种写法得到一个“无限的”数据迭代器：

```listing ch03/09_grpo.py#L155-L157
```

问题在于 `itertools.cycle` 的工作方式：它把第一遍迭代吐出的每个元素都**存下来**，第一遍结束后，不再重新调用 `iter()`，而是把存下来的元素按原来的顺序一遍遍重放。`DataLoader` 的 `shuffle=True` 只在每次 `iter()` 时生效，于是只有第一轮是洗过牌的，此后每一轮的 batch 都与第一轮一模一样：

```verify id=Z.4-e title="itertools.cycle 重放第一轮：第二轮的 batch 就是第一轮的那些对象"
import random
from itertools import cycle

class ShuffledLoader:                      # 与 DataLoader(shuffle=True) 一样：每次 iter() 重新洗牌
    def __init__(self, data, batch_size, seed=0):
        self.data, self.bs, self.rng = list(data), batch_size, random.Random(seed)

    def __iter__(self):
        idx = self.data[:]
        self.rng.shuffle(idx)
        return iter([idx[i:i + self.bs] for i in range(0, len(idx), self.bs)])

it = cycle(ShuffledLoader(range(10), 3))   # 每轮 4 个 batch
epoch1 = [next(it) for _ in range(4)]
epoch2 = [next(it) for _ in range(4)]
print("第 1 轮：", epoch1)
print("第 2 轮：", epoch2)
assert epoch1 == epoch2 and all(a is b for a, b in zip(epoch1, epoch2))

batch_bytes = 2 * 32 * 256 * 8             # 预训练一个 batch 的 x 与 y（int64）
print(f"预训练 20,000 步，cycle 存下的 batch 共 {20_000 * batch_bytes / 2**30:.2f} GiB")
```

影响因脚本而异。SFT 的 1,092 条样本每轮分成 35 个 batch，500 步约是 14.3 轮；DPO 的 144 对每轮 18 个 batch，1,000 步约是 55.6 轮；GRPO 的 81 道题每轮分成 32、32、17 三批，500 次迭代约是 167 轮。这三个脚本从第二轮起，每个 batch 的组成和顺序都固定不变。预训练反而不会重放：20,000 步只取了 640,000 个窗口，还不到一轮（共 2,677,142 个窗口）；但 `cycle` 会把这 20,000 个 batch 都留在内存里，多占约 2.4 GiB。

```localcheck verify/test_reference.py::test_e07_cycle_dataloader_replays_first_epoch_and_keeps_batches needs="PyTorch" title="真实的 DataLoader：第二轮的 batch 与第一轮是同一批对象"
```

**修法**：用一个每轮重新迭代的生成器代替 `cycle`：

```python
def infinite(dataloader):
    while True:
        yield from dataloader          # 每一轮都重新调用 iter()，重新洗牌，也不缓存
data_iter = infinite(dataloader)
```

<div id="e08"></div>

### E08 `encode_file` 会删掉整个 `cache_dir`

`BPETokenizer.encode_file` 把每个分块的编码结果先缓存到磁盘，最后在 `finally` 里删除缓存目录：

```listing storybot/tokenizer.py#L272-L278
```

```listing storybot/tokenizer.py#L314-L321
```

`shutil.rmtree(cache_dir)` 删的是**整个目录**，不管里面原来有什么。如果把一个已经存在的目录当作 `cache_dir` 传进去，里面原有的文件会一并消失。我们在临时目录里建了一个装着 `notes.txt` 的文件夹，把它当作 `cache_dir`，运行之后整个文件夹都没了。另外，默认值 `"bpe_cache"` 是相对路径，同一个工作目录里同时跑两个 `encode_file`，会互相删掉对方的缓存。

```localcheck verify/test_reference.py::test_e08_encode_file_deletes_the_whole_cache_dir needs="PyTorch" title="把装着 notes.txt 的目录当 cache_dir：运行后连目录带文件一起被删"
```

**修法**：只删自己创建的东西。用 `tempfile.mkdtemp()` 创建一个新的临时目录作缓存，结束时只删它；或者只删除自己写下的 `chunk_*.npy` 文件。顺带一提，这些缓存文件的扩展名虽然是 `.npy`，内容却是 `tofile` 写的裸 uint16，`np.load` 读不了，只能用 `np.fromfile`。

<div id="e09"></div>

### E09 奖励函数的裸 `except` 吞掉一切错误

GRPO 的奖励函数取回答里的最后一个整数，与正确答案比较：

```listing ch03/09_grpo.py#L63-L72
```

`except:` 后面没有写异常类型，它会捕获**一切**异常，包括编程错误。设想你改写这个函数时，不小心删掉了文件开头的 `import re`：每次调用都会抛出 `NameError`，被吞掉后返回 0.0。训练照常进行，只是所有回答都被判成“答错”，所有组的 advantage 全是 0，模型一点也学不到，而日志里看不出任何异常。下面把这个函数原样拿过来，先在没有 `re` 的环境里调用：

```verify id=Z.4-f title="没有 import re 时，奖励函数不报错，只是永远返回 0.0"
globals().pop("re", None)                  # 模拟忘了 import re

def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])  # 最後の数値を取得
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0

print("没有 import re：", calculate_reward(7, "7"))
import re
print("import re 之后：", calculate_reward(7, "7"), calculate_reward(7, "7 8"), calculate_reward(7, "8 7"))
assert calculate_reward(7, "7") == 1.0 and calculate_reward(7, "7 8") == 0.0
```

```localcheck verify/test_reference.py::test_e09_calculate_reward_swallows_every_error needs="PyTorch" title="书中的函数：缺了 re 时返回 0.0；页面上的副本与原函数的语法树完全相同"
```

**修法**：只捕获预期的异常，比如 `except ValueError:`（`int()` 解析失败时抛出的就是它），让编程错误照常报出来。

顺便说一句，“最后一个整数”这个规则本身也可以被钻空子：回答 `'8 7'` 会得 1 分，`'7.0'` 却得 0 分。本书两次 GPU 运行训练出的模型没有学会这一招：对 81 道题的 greedy 回答，没有一个含两个以上的整数（原样运行的模型对 `3+3=` 答了一个空串，其余都是一个数；E01 里那个本地验证同时检查了这一点，温度为 1 的采样我们没有统计）。设计奖励函数时，还是值得留心。

<div id="e10"></div>

### E10 DPO 把 prompt 和回答分开编码

DPO 的数据集把 prompt 和回答分别编码，再拼接起来：

```listing ch06/08_dpo.py#L59-L69
```

`tiny_stories_dpo.json` 里的 144 条 prompt 全都以一个空格结尾，回答则以一个单词开头。GPT-2 式的预分词会把空格粘在后面的单词上，所以一起编码时是 `' Tim'`；分开编码时，prompt 的结尾单独剩下一个 `' '`，回答的开头是 `'Tim'`。下面只用预分词正则就能看出差别：

```verify id=Z.4-g title="分开编码得到 ' ' + 'Tim'，一起编码得到 ' Tim'"
prompt = "Once upon a time, there was a boy named Tim who lost his toy. "   # 第 1 条样本的结尾
answer = "Tim looked everywhere for his toy."
apart = regex.findall(PAT, prompt) + regex.findall(PAT, answer)   # PAT 是验证 Z.4-b 里的预分词正则
joint = regex.findall(PAT, prompt + answer)
k = apart.index(" ")
print("分开编码：", apart[k - 2:k + 3])
print("一起编码：", joint[k - 2:k + 2])
assert apart[k:k + 2] == [" ", "Tim"] and " Tim" in joint and " " not in joint
```

用真实的 StoryBot 分词器检查全部 144 对数据：288 条序列（每对的 chosen 与 rejected）的 token 序列都与“一起编码”不同，其中 271 条多出 1 个 token，15 条多出 2 个，另有 2 条长度相同但切法不同。预训练时模型从没见过“单独一个空格后面跟着 Tim”这样的切法，所以回答第一个 token 的对数概率是在一个不自然的上下文里算的。chosen 和 rejected 都受到同样的影响，有一部分会相互抵消。它对最终 DPO 结果的影响有多大，我们没有测量。

```localcheck verify/test_reference.py::test_e10_dpo_prompt_and_answer_are_tokenized_apart needs="PyTorch" title="真实分词器：288 条序列全部与一起编码不同"
```

**修法**：先编码完整的文本 `prompt + answer`，再用 prompt 的长度划分掩码；或者把 prompt 结尾的空格移到回答的开头。

<div id="e11"></div>

### E11 用户文本里的 `<|endoftext|>` 被当成特殊 token

CodeBot 和 StoryBot 的 `encode` 第一步就是按 `<|endoftext|>` 切开文本，不区分这串字符来自哪里：

```listing codebot/tokenizer.py#L87-L91
```

这意味着用户在对话里打出 `<|endoftext|>` 这几个字符，模型收到的就是“文档结束”的 id（CodeBot 是 999）。

```verify id=Z.4-h title="切分这一步就把用户文本里的 <|endoftext|> 认作了特殊 token"
text = "print('<|endoftext|>')"
pieces = regex.split("(" + regex.escape("<|endoftext|>") + ")", text)   # tokenizer.encode 的第一步
print(pieces)
assert pieces == ["print('", "<|endoftext|>", "')"]
```

```localcheck verify/test_reference.py::test_e11_special_token_text_is_injected needs="PyTorch" title="CodeBot：print('<|endoftext|>') 编码为 [350, 338, 999, 343]"
```

在 `ch03/04_chat.py` 这样的对话脚本里，这相当于让用户可以“注入”控制 token。tiktoken 默认会拒绝文本中的特殊 token，必须显式允许才行。**修法**：给 `encode` 加一个参数（比如 `allow_special=False`），只有在拼接训练数据时才允许；对用户输入，把特殊 token 当普通文本编码。

<div id="e12"></div>

### E12 第 4.5 节用到的 ch04/05_eval.py 不在仓库里

书中第 4.5 节“评估分词器”引用了 `ch04/05_eval.py`，但仓库的 `ch04/` 里只有 01、02、03、04、06、07 六个脚本。读者在 GitHub 上提了 [issue #7](https://github.com/oreilly-japan/deep-learning-from-scratch-6/issues/7)（2026-07-28），并贴出了根据正文整理的代码；截至 2026-10-12，这个 issue 仍然开着。

```localcheck verify/test_reference.py::test_e12_ch04_05_eval_is_missing title="ch04 的脚本编号是 01、02、03、04、06、07"
```

**修法**：参考 issue #7 的代码，或者照第 1 章 `08_eval.py` 的思路自己写：压缩率 = 字节数 ÷ token 数。

<div id="e50"></div>

### E50 `max_iters` 写死，开 8 张卡就读 8 倍的 token

第 9 章的 WebBot 用 `torchrun` 做多卡训练。可是 `max_iters = 100000` 是写死的，脚本里没有任何地方按卡数去调整它：

```listing ch09/01_pretrain.py#L107-L113
```

每个进程每一步照样处理自己的 32 × 4 条序列。于是开 8 张卡原样运行，每一步读的 token 是 1 张卡的 8 倍；跑完 100,000 步的墙钟时间和 1 张卡差不多，读过的 token 和用掉的 GPU 时间却都变成了 8 倍。下面用 WebBot 在 H100 上实测的每步用时来算（运行记录 webbot-probe，不计多卡之间的通信）：

```verify id=Z.4-i title="max_iters 写死：8 张卡原样运行，墙钟不变，token 与 GPU 小时都是 8 倍" files=data/runs/webbot-probe/timing.json
t = json.load(open("data/runs/webbot-probe/timing.json", encoding="utf-8"))   # 1 张 H100，原书脚本的前 200 步
sec, tok = t["steady_s_per_step_20_200"], t["tokens_per_step"]
max_iters = 100_000                                   # ch09/01_pretrain.py 第 111 行
for world, iters in ((1, max_iters), (8, max_iters), (8, max_iters // 8)):
    print(f"{world} 张卡 × {iters:>7,} 步：读 {world * tok * iters / 1e9:5.1f}B 个 token，"
          f"墙钟约 {sec * iters / 3600:4.1f} 小时，共 {world * sec * iters / 3600:5.1f} GPU 小时")
assert tok == 32 * 4 * 1024 and 8.7 < sec * max_iters / 3600 < 8.9 and max_iters // 8 == 12_500
```

想用 8 张卡在约 1/8 的时间里读完同样多的 token，就要把 `max_iters` 改成 12,500（[9.3 节](ch09.html#s9-3)的[验证 9.3-d](ch09.html#v-9.3-d) 做了同样的计算）。在租来的 8 卡机器上原样运行，花费是预想的 8 倍。

```localcheck verify/test_reference.py::test_e50_max_iters_ignores_the_world_size title="语法树：max_iters 是写死的 100000，脚本里没有读 WORLD_SIZE，每个进程每步取 32 × 4 条序列"
```

**修法**：按卡数换算步数，例如在初始化之后写 `max_iters = 100000 // int(os.environ.get('WORLD_SIZE', 1))`；或者干脆以要读的 token 总数为目标来定步数。学习率调度用的也是 `max_iters`，所以 warmup 和衰减会跟着一起缩短。

## Z.5 低：数值、接口与一致性

这一节的问题都不会改变书中的结果，但在你改写代码、换数据、或者拿书中的实现去对接别的东西时，很可能会踩到。

<div id="e13"></div>

### E13 `get_batch` 永远采不到最后一个窗口

第 6 章和第 9 章的 `get_batch` 这样选起点：

```listing ch06/05_pretrain.py#L29-L31
```

`torch.randint(n, ...)` 的取值范围是 0 到 n−1，不含 n。长度为 `len(data)` 的数据，最后一个合法的起点是 `len(data) - context_len - 1`（此时 y 恰好取到最后一个 token），而这里的上界把它排除了。

```verify id=Z.5-a title="100 个 token、窗口 7：合法起点有 93 个，只抽得到 92 个"
import numpy as np

n, C = 100, 7
rng = np.random.default_rng(0)
starts = rng.integers(0, n - C - 1, size=32_000)        # torch.randint(len(data) - C - 1)：不含上界
print(f"抽到的起点：最小 {starts.min()}，最大 {starts.max()}，共 {len(np.unique(starts))} 种")
print(f"最后一个合法起点：{n - C - 1}（x = data[92:99]，y = data[93:100]）")
assert starts.max() == n - C - 2 and len(np.unique(starts)) == n - C - 1
```

StoryBot 的训练集有 5.4 亿个 token，在这上面少一个起点，对训练没有可察觉的影响；但它会让 CS336 的 `test_get_batch` 失败（详见 [CS336 对照总表](cs336.html)）。**修法**：上界改为 `len(data) - context_len`。

```localcheck verify/test_reference.py::test_e13_get_batch_never_samples_the_last_window needs="PyTorch" title="书中的 get_batch：1,000 次抽样，起点最大只到 91；HF 上的训练集是 541,229,223 个 token"
```

<div id="e14"></div>

### E14 `evaluate` 少算最后一个完整窗口

验证时按顺序取不重叠的窗口，过滤条件是：

```listing ch06/05_pretrain.py#L33-L35
```

窗口起点 i 合法的条件是 `i + context_len + 1 <= len(data)`，这里写成了严格小于，于是恰好放得下的最后一个窗口被丢掉了。

```verify id=Z.5-b title="513 个 token 放得下两个 256 的窗口，过滤后只剩一个"
import numpy as np

def kept(n, C=256, B=2):
    ix = np.arange(0, B * C, C)
    return ix[ix + C + 1 < n]                  # ch06/05 第 35 行

print("n=513：", kept(513).tolist(), "  n=514：", kept(514).tolist())
assert kept(513).tolist() == [0] and kept(514).tolist() == [0, 256]
```

验证集有 546 万个 token，少算的至多是最后一个窗口，对验证 loss 的影响可以忽略。**修法**：把 `<` 改成 `<=`。

```localcheck verify/test_reference.py::test_e14_evaluate_drops_the_last_full_window needs="PyTorch" title="书中的 get_batch(random=False)：n=513 时只返回 1 个窗口；验证集是 5,465,873 个 token"
```

<div id="e15"></div>

### E15 GRPO 的第 0 次准确率是两次更新之后测的

GRPO 的训练循环里，评估写在两次更新之后（见 [E01](#e01) 中的代码），所以 `accuracies` 的第一个值并不是 SFT 模型的起点，而是已经更新过两次的模型。本书的两次 GPU 运行恰好能说明这一点：它们从同一个 SFT 检查点出发（sha256 相同），记下的“第 0 次”准确率却一个是 53.1%，一个是 56.8%。如果这真是起点，两个数应该相同。

```verify id=Z.5-c title="同一个 SFT 起点，两次运行记下的“第 0 次”却不同：53.1% 与 56.8%" files=data/runs/codebot-3060ti/09_grpo.metrics.jsonl,data/runs/codebot-grpo-evalfix-3060ti/09_grpo.metrics.jsonl
import json

first = {}
for run_id in ("codebot-3060ti", "codebot-grpo-evalfix-3060ti"):
    with open(f"data/runs/{run_id}/09_grpo.metrics.jsonl", encoding="utf-8") as f:
        first[run_id] = json.loads(f.readline())["acc"]
print("记下的“第 0 次”准确率：", first)
print(f"SFT 起点（本地验证）：44 / 81 = {100 * 44 / 81:.1f}%")
assert first == {"codebot-3060ti": 53.1, "codebot-grpo-evalfix-3060ti": 56.8}
```

这个 SFT 检查点本身在 CPU 上 greedy 答对 81 道题里的 44 道，即 54.3%（作者发布的 `model_sft.pt` 恰好也是 44 道）。研究阶段批量采样版的记录也一样：seed 0 记下的“第 0 次”是 55.6%，而它的起点是作者发布的 SFT 权重，54.3%。

```localcheck verify/test_reference.py::test_e15_sft_greedy_baseline_is_44_of_81 needs="PyTorch" title="作者发布的 SFT 权重：81 道题 greedy 答对 44 道"
```

```localcheck verify/test_reference.py::test_e15_grpo_logs_accuracy_after_the_updates title="语法树：评估块排在更新循环之后；画图时第 k 个点放在 10k 处"
```

作图时，脚本把第 k 个准确率画在横坐标 10k 处（`steps = list(range(0, len(accuracies) * eval_interval, eval_interval))`），而它其实是第 10k+1 次迭代之后测的。也就是说，每个点都比它的横坐标晚一次迭代：整条曲线向左偏了 1 次迭代（不是一个评估间隔），真正的起点没有画出来。本页 E01 的图把点画在“已完成的迭代次数”处，并用虚线标出 SFT 的起点。

**修法**：在训练循环之前先测一次准确率作为第 0 个点，循环里改成在 `i + 1` 是 `eval_interval` 的倍数时评估。

<div id="e16"></div>

### E16 ratio 在概率空间计算，极小概率时有偏

GRPO 的 ratio 写成 `probs / (old_probs + 1e-8)`（见 E01 的代码）。分母上的 `1e-8` 是为了防止除以 0，但当 token 的概率本身很小时，它就不再“可以忽略”了：

```verify id=Z.5-d title="新旧概率相同时，ratio 应该是 1；p = 1e-8 时只有 0.5"
for p in (1e-4, 1e-6, 1e-7, 1e-8):
    print(f"p = {p:.0e}：ratio = {p / (p + 1e-8):.6f}")
assert abs(1e-8 / (1e-8 + 1e-8) - 0.5) < 1e-12
```

温度为 1 的采样偶尔会抽到概率很低的 token，这时 ratio 会系统性地偏小。**修法**：在对数空间计算，`ratio = torch.exp(logp - old_logp)`，其中 `logp` 由 `log_softmax` 加 `gather` 得到。CS336 的 A5 也是这样写的。

```localcheck verify/test_reference.py::test_e16_ratio_is_taken_in_probability_space needs="PyTorch" title="源码中的写法，以及 PyTorch 下的数值"
```

<div id="e17"></div>

### E17 两套 RMSNorm 的 eps 不一致

第 5 章手写的 RMSNorm 用 `eps = 1e-5`，而 StoryBot 和 WebBot 的模型直接用了 `nn.RMSNorm(embed_dim)`。不传 `eps` 时，PyTorch 用的是该数据类型的机器精度，float32 下约为 $1.19\times10^{-7}$：

```listing storybot/model.py#L158-L165
```

在正常尺度的激活上，两者只差 $10^{-5}$ 量级；但激活很小的时候，差别就大了：

```verify id=Z.5-e title="RMS 为 1e-3 的输入：机器精度 eps 给出 0.945，eps=1e-5 给出 0.302"
eps32 = float(np.finfo(np.float32).eps)

def rmsnorm(x, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)

for scale in (1.0, 1e-2, 1e-3):
    x = np.full(8, scale, dtype=np.float64)
    print(f"RMS = {scale:<6g} eps={eps32:.2e} → {rmsnorm(x, eps32)[0]:.4f}    eps=1e-5 → {rmsnorm(x, 1e-5)[0]:.4f}")
assert abs(rmsnorm(np.full(8, 1e-3), eps32)[0] - 0.9452) < 1e-4
assert abs(rmsnorm(np.full(8, 1e-3), 1e-5)[0] - 0.3015) < 1e-4
```

它不影响书中的训练，但正是这个差别让 StoryBot 在 CS336 的 `test_transformer_lm` 上差了 $4.6\times10^{-4}$，没能通过；把 eps 改成 `1e-5` 之后误差降到 $1.05\times10^{-5}$（见 [CS336 对照总表](cs336.html)）。**修法**：显式写 `nn.RMSNorm(embed_dim, eps=1e-5)`，与第 5 章的手写版一致。

```localcheck verify/test_reference.py::test_e17_storybot_rmsnorm_uses_float32_machine_eps needs="PyTorch" title="按书中配置构造的 StoryBot：9 个 RMSNorm（每层 2 个，最后 1 个）的 eps 都是 None"
```

<div id="e18"></div>

### E18 CodeBot 实际用的是 erf 版 GELU 和 `nn.LayerNorm`

第 2 章手写了 LayerNorm 和 tanh 近似的 GELU，`codebot/model.py` 也定义了这两个类，但模型里实际用的是 PyTorch 的版本，手写的类一次也没有被实例化：

```listing codebot/model.py#L73-L94
```

`nn.LayerNorm` 与手写版在数值上一致；`nn.GELU()` 却是 erf 精确版，与书中的 tanh 近似版最多相差 $4.7\times10^{-4}$：

```verify id=Z.5-f title="erf 版与 tanh 版 GELU 最多相差约 4.7e-4"
import math

x = np.linspace(-6, 6, 12001)
g_erf = 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
g_tanh = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
d = np.abs(g_erf - g_tanh)
print(f"最大差 {d.max():.2e}，出现在 x = {x[d.argmax()]:+.3f}")
assert 4.7e-4 < d.max() < 4.8e-4
```

这不是错误，只是读者对着书中的公式手算 CodeBot 的输出时，会多出这点误差。如果要与 Hugging Face 的 GPT-2 对拍，则要换成 `nn.GELU(approximate='tanh')`。

```localcheck verify/test_reference.py::test_e18_codebot_runs_erf_gelu_and_torch_layernorm needs="PyTorch" title="CodeBot 的 GELU 是 approximate='none'；手写的 LayerNorm、GELU 没有被用到"
```

<div id="e19"></div>

### E19 `head_dim` 不检查能否整除

`Block` 用 `head_dim = embed_dim // n_head` 计算每个头的维数，没有检查能否整除。`embed_dim=100, n_head=3` 时，`head_dim` 是 33，三个头合起来只有 99 维，代码照样能跑，不报任何错误。nanoGPT 在同样的位置写了 `assert config.n_embd % config.n_head == 0`。**修法**：在 `__init__` 里加上同样的断言。

```localcheck verify/test_reference.py::test_e19_head_dim_is_not_checked needs="PyTorch" title="Block(100, 3, ...)：W_q 是 100→99，前向照常运行"
```

<div id="e20"></div>

### E20 EOS 不在 GRPO 的学习目标里

`generate()` 遇到结束 token 就停止，而且不把它放进返回的文本；GRPO 再把回答文本重新编码，得到的 token 序列里没有 EOS（见 E01 前后的代码）。所以 EOS 本身不在目标里，“答完就停”从来不被直接奖励。

这并不等于“多写”不受惩罚。回答里所有的 token 都在掩码之内（`mask = [0] * len(prompt_ids) + [1] * len(response_ids)`），包括答案后面多写的那些。比如正确答案是 7，模型回答 `'7 8'`，奖励按最后一个整数算是 0；只要这一组里有答对的回答，它就拿到负的 advantage，多写的 `' 8'` 的概率会被压低，结束 token 的概率也就间接抬高了。缺的只是对 EOS 这个 token 的直接奖励。

```localcheck verify/test_reference.py::test_e20_eos_is_not_part_of_the_grpo_objective needs="PyTorch" title="回答 '7' 和 '12' 的批次里没有 999；回答 '7 8' 时，掩码盖住了多写的 ' 8'"
```

这是一个设计选择，在加法任务上问题不大。**修法**：如果希望直接强化“答完就停”，可以在回答末尾补上 EOS，并把它纳入掩码。

<div id="e21"></div>

### E21 GRPO 生成不设长度上限，而 CodeBot 只有 256 个位置

`generate_group` 调用 `generate()` 时没有传 `max_new_tokens`，用的是默认值 1000。生成本身不会出错，因为 `generate()` 每步只看最近的 256 个 token；但更新时 `compute_probs` 要把“提示 + 回答”整段送进模型，而 CodeBot 的位置嵌入只有 256 个。提示 22 个 token，回答只要超过 234 个 token，就会触发 `IndexError`。

```localcheck verify/test_reference.py::test_e21_grpo_has_no_length_cap_but_codebot_has_256_positions needs="PyTorch" title="默认 max_new_tokens=1000；超过 256 个 token 的批次让 compute_probs 报 IndexError；SFT 模型答一个数就结束的概率最低 97.6%"
```

书中没有触发，是因为 SFT 之后的模型几乎总是只答一个数就结束。在温度 1 下，81 道题里的每一道，模型回答“0 到 18 中的某个数（1 到 2 个 token），随即输出结束 token”的概率最低也有 97.6%，平均 99.7%。换成需要长回答的任务，就会出问题。**修法**：调用时显式传 `max_new_tokens`，让“提示 + 回答”不超过 256。

<div id="e22"></div>

### E22 `generate()` 会把模型留在 eval 模式

`generate()` 一开始就调用 `model.eval()`，结束时没有恢复原来的模式。GRPO 脚本评估时对正在训练的 `model` 调用了 `generate()`（第 194 行），紧接着在第 198 行手动调用 `model.train()`，所以恰好没有出问题；但如果在训练循环的别处对正在训练的模型调用 `generate()`，又忘了切回来，dropout 就会被悄悄关掉。

```localcheck verify/test_reference.py::test_e22_generate_leaves_the_model_in_eval_mode needs="PyTorch" title="train() 之后调用 generate()，模型变成了 eval 模式；GRPO 脚本在评估后手动切回"
```

**修法**：记下原来的 `model.training`，返回前恢复。

<div id="e23"></div>

### E23 温度极小（低于约 1e-38）时 softmax 得到 NaN

`generate()` 只在 `temperature == 0` 时走 argmax 分支，温度是正数时就计算 `softmax(logits / temperature)`。常见的小温度没有问题：PyTorch 的 softmax 会先减去最大值再取指数，温度越小，结果越接近 argmax 的 one-hot。麻烦在 `logits / temperature` 这一步本身。它是 float32，超过 float32 的上限（约 $3.4\times10^{38}$）就成了 inf，inf 减去 inf 得到 NaN。对几到几十的 logits，温度要小于约 $10^{-38}$ 才会发生：

```verify id=Z.5-g title="float32 下 T = 1e-38 仍是 argmax，T = 1e-39 时 softmax 全是 NaN"
z = np.array([2.0, 1.0, 0.5], dtype=np.float32)

def softmax_t(z, T):                       # 与 torch.softmax(z / T) 相同：float32，先减去最大值
    with np.errstate(over="ignore", invalid="ignore"):
        s = z / np.float32(T)
        e = np.exp(s - s.max())
        return e / e.sum()

for T in (1e-3, 1e-30, 1e-38, 1e-39, 1e-45):
    print(f"T = {T:.0e}：softmax(z / T) = {np.array2string(softmax_t(z, T), precision=3)}")
assert np.allclose(softmax_t(z, 1e-38), [1, 0, 0]) and np.isnan(softmax_t(z, 1e-39)).all()
```

PyTorch 里的 `torch.multinomial` 遇到 NaN 会直接报错。实际使用时很少有人把温度设得这么小，所以这只是一个边界情况。**修法**：把条件改成 `temperature < 1e-5` 之类的阈值，直接走 argmax。

```localcheck verify/test_reference.py::test_e23_tiny_temperature_gives_nan needs="PyTorch" title="PyTorch：T = 1e-38 仍是 one-hot，T = 1e-39 得到 NaN，multinomial 报错"
```

<div id="e24"></div>

### E24 逐个 token 解码会冒出 U+FFFD

`decode` 用的是 `errors="replace"`，非法的 UTF-8 字节会变成替换字符 U+FFFD。整段解码没有问题，可是如果流式输出时一个 token 一个 token 地解码，一个汉字的 3 个字节就会变成 3 个替换字符：

```verify id=Z.5-h title="“世”的 3 个字节分开解码，得到 3 个 U+FFFD"
b = "世".encode("utf-8")
parts = [bytes([x]).decode("utf-8", errors="replace") for x in b]
print(list(b), parts, b.decode("utf-8"))
assert parts == ["�"] * 3
```

WebBot 的 5 万词表里，除了 128 个高位单字节，还有 157 个合并出来的 token 本身不是合法的 UTF-8。第一个是 id 290，它是 `b'\xe2\x80'`，也就是“’”“—”这类 3 字节字符的前两个字节。`ch01/08_eval.py` 只严格解码 CodeBot 的 20 个 id（256–265 与 990–999），恰好都合法；如果把这个循环改成遍历 WebBot 的整个词表，就会抛出 `UnicodeDecodeError`。**修法**：流式输出时先累积字节，再增量解码（Python 用 `codecs.getincrementaldecoder("utf-8")`）。

```localcheck verify/test_reference.py::test_e24_single_token_decode_and_webbot_partial_utf8 needs="PyTorch" title="CodeBot 解码单个字节得到 U+FFFD；WebBot 有 157 个合并 token 单独不是合法 UTF-8，第一个是 id 290"
```

<div id="e25"></div>

### E25 用 uint16 存 id，词表不能超过 65,536

编码后的数据用 `np.uint16` 存盘，2 个字节一个 token，只占 `int64` 的四分之一。书中的词表最大是 5 万，没有问题；但如果把词表调到 65,536 以上，id 就放不下了。NumPy 1.x 会把它悄悄回绕成除以 65,536 的余数：65536 变成 0，70000 变成 4464；从 1.24 版起才多了一个 DeprecationWarning（[NumPy 1.24 的发行说明](https://numpy.org/doc/stable/release/1.24.0-notes.html#conversion-of-out-of-bound-python-integers)，我们在 NumPy 1.26.4 上确认过）。NumPy 2 则直接报错：

```verify id=Z.5-i title="70000 放不进 uint16：回绕的话是 4464，NumPy 2 直接报错"
print("uint16 的最大值：", np.iinfo(np.uint16).max)
print("70000 回绕后是：", 70000 % 2**16)                 # NumPy 1.x 存进去的值
try:
    np.array([70000], dtype=np.uint16)
    raised = False
except OverflowError as e:
    raised = True
    print("NumPy 2：OverflowError:", e)
assert raised and 70000 % 2**16 == 4464
```

**修法**：词表超过 65,536 时改用 `np.uint32`，读取时的 `dtype` 也要同步修改。

```localcheck verify/test_reference.py::test_e25_uint16_cannot_hold_ids_above_65535 title="encode_file 写死了 np.uint16"
```

<div id="e26"></div>

### E26 两个 `count_pairs` 的参数不一样

`codebot/tokenizer.py` 的 `count_pairs(ids, counts=None)` 与 `storybot/tokenizer.py` 的 `count_pairs(ids, weight=1, counts=None)` 同名，但第二个位置参数的含义不同。把第 1 章的调用方式 `count_pairs(ids, counts)` 用在第 4 章的函数上，`counts` 会被当成 `weight`，结果是 `TypeError`。**修法**：调用时一律写成关键字参数 `counts=...`。

```localcheck verify/test_reference.py::test_e26_count_pairs_signatures_differ needs="PyTorch" title="把字典按位置传给 StoryBot 的 count_pairs：TypeError"
```

<div id="e27"></div>

### E27 `merge_rules` 的顺序本身就是数据

`BPETokenizer.__init__` 按字典的插入顺序构造词表，每条规则都要用到前面规则产生的 token。顺序一乱，就会遇到还没定义的 id：

```verify id=Z.5-j title="把第 1 章的 4 条规则倒过来，构造词表时 KeyError: 256"
rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}   # ch01/04 的规则

def build(rules):
    id_to_bytes = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in rules.items():          # BPETokenizer.__init__ 的写法
        id_to_bytes[new_id] = id_to_bytes[a] + id_to_bytes[b]
    return id_to_bytes

print("原顺序：", build(rules)[257])
try:
    build(dict(reversed(list(rules.items()))))
    err = None
except KeyError as e:
    err = e
print("倒过来：KeyError", err)
assert str(err) == "256"
```

pickle 会保存字典的顺序，所以书中的流程没有问题。存成 JSON 也可以：要先把元组键改成字符串（比如 `"105,115"`），`json.dump` 按插入顺序写出键，`json.load` 也按文件里的顺序重建字典。会出错的是保存时排过序（例如 `json.dump(..., sort_keys=True)`），或者交给不保证对象键顺序的工具处理。**修法**：构造之前按新 id 排序，`sorted(merge_rules.items(), key=lambda kv: kv[1])`；存成 JSON 时存成列表。

```localcheck verify/test_reference.py::test_e27_merge_rules_order_is_part_of_the_data needs="PyTorch" title="真实的 merge_rules.pkl：倒序后构造 BPETokenizer 报 KeyError；JSON 往返保住了顺序，sort_keys=True 则不行"
```

<div id="e28"></div>

### E28 `measure_generation_time` 清缓存的条件写反了

第 5 章的测速函数只在**不用**缓存时清空缓存，可是需要先清空的恰恰是用缓存的那次测量：

```listing ch05/05_kvcache.py#L282-L294
```

脚本先测“无缓存”，再测“有缓存”，而无缓存的生成不会写入缓存，所以原来的调用顺序恰好不受影响。如果连续两次测量有缓存的情况，第二次会接着第一次的缓存继续生成：位置从 200 开始，很快就超过了 256（见 [E04](#e04)）。**修法**：把条件改成 `if use_cache: model.clear_cache()`，或者干脆每次都清空。

```localcheck verify/test_reference.py::test_e28_measure_generation_time_clears_the_wrong_cache needs="PyTorch" title="连续两次有缓存的测量：cache_offset 从 5 变成 10"
```

<div id="e29"></div>

### E29 读文本不指定编码，中文 Windows 上会静默乱码

不少脚本用 `open(path)` 读文本文件，没有写 `encoding=`。仓库里一共有 9 处这样的调用，分布在 9 个脚本里：第 1 章的 07、08、09，第 3 章的 `03_alpaca.py` 和 `03_sft.py`，第 4 章的 01、02、06，以及第 6 章的 `08_dpo.py`。这时 Python 用操作系统的区域编码；在简体中文的 Windows 上，它是 cp936，也就是 GBK（本机 `locale.getencoding()` 返回的正是 `'cp936'`）。只有开启 Python 的 UTF-8 模式（`PYTHONUTF8=1` 或 `-X utf8`）时，才会按 UTF-8 读取。

后果分三种。`tiny_codes.txt` 是纯 ASCII，读法不影响结果。`ch03/03_sft.py` 读的 `tiny_codes_sft.json` 里有两条样本含有 “²”（`O(n²)`），按 GBK 读进来不会报错，而是悄悄变成 “虏”：SFT 就在这两条乱码的样本上训练。TinyStories 的文本含有大量非 ASCII 字符，按 GBK 读会直接抛出 `UnicodeDecodeError`（`ch04/02_bpe_cache.py` 末尾被注释掉的那几行正是这样读的）。

```verify id=Z.5-k title="UTF-8 的 “²” 按 GBK 解码，不报错，变成了 “虏”"
utf8 = "O(n²) time".encode("utf-8")
print("字节：", list(utf8))
print("按 UTF-8 读：", utf8.decode("utf-8"))
print("按 GBK 读  ：", utf8.decode("gbk"))
assert utf8.decode("gbk") == "O(n虏) time"
```

```localcheck verify/test_reference.py::test_e29_text_files_are_opened_with_the_locale_encoding title="9 个脚本各有 1 处用区域编码读文本；按 GBK 读 SFT 数据有 2 条变成乱码，读 TinyStories 则报错"
```

**修法**：一律写 `open(path, encoding="utf-8")`。对 `ch03/03_sft.py` 来说，就是把第 35 行改成 `open(data_path, encoding="utf-8")`。或者运行时设置 `PYTHONUTF8=1`，本教程给出的所有命令都设置了它。

<div id="e30"></div>

### E30 软字典的 softmax 没有减去最大值

第 2 章开头的软字典直接对相似度取指数：

```listing ch02/01_soft_dict.py#L38-L40
```

float64 能表示的最大数约为 $1.8\times10^{308}$，它的自然对数约是 709.8。相似度一旦超过这个数，`np.exp` 就溢出成 inf，inf 除以 inf 得到 NaN。书中电影例子的相似度只有几十，所以没有触发；把查询向量放大到 (100, 100, 100) 就会看到：

```verify id=Z.5-l title="相似度 1300：直接取指数得到 NaN；先减去最大值则没有问题"
scores = np.array([1300.0, 1300.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(scores) / np.exp(scores).sum()
stable = np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()
print(f"exp 的上限：ln(float64 最大值) = {np.log(np.finfo(np.float64).max):.2f}")
print("直接取指数：", naive, "  先减最大值：", stable)
assert np.isnan(naive).all() and np.allclose(stable, [0.5, 0.5])
```

第 2 章后面的代码改用 PyTorch 的 `F.softmax`，它在内部做了减去最大值这一步。**修法**：`np.exp(similarity - np.max(similarity))`，结果不变（softmax 对平移不变），也不会溢出。

```localcheck verify/test_reference.py::test_e30_soft_dictionary_softmax_overflows title="书中的 soft_dictionary：查询 (100, 100, 100) 时结果是 NaN"
```

<div id="e31"></div>

### E31 RoPE 的 cos/sin 表存进了每一层的检查点

RoPE 预先算好的 cos、sin 表是用 `register_buffer` 注册的，默认会写进 `state_dict`。所有 Transformer 块（StoryBot 4 个、WebBot 12 个）共用一个 RoPE 模块，表却在每一层的键下各存一份：StoryBot 的 `state_dict` 里有 8 张这样的表，WebBot 有 24 张。这些表完全由 θ 和 `max_context_len` 算出，本不需要保存；保存下来反而带来一个限制：想换一个更长的 `max_context_len`（比如第 8 章的上下文扩展），用原来的检查点加载会因为形状不同而报错。

```localcheck verify/test_reference.py::test_e31_rope_tables_are_saved_into_every_layer needs="PyTorch" title="StoryBot 的 state_dict 里有 8 张 cos/sin 表，12 层的 WebBot 有 24 张；换一个上下文长度加载就报 size mismatch"
```

**修法**：注册时写 `self.register_buffer("cos_cache", cos, persistent=False)`，sin 表同理。

<div id="e32"></div>

### E32 GRPO 评估时把“提示 + 回答”整段交给奖励函数

训练时，`generate_group` 会先把提示切掉，只把回答交给奖励函数（第 85 行 `response = full_text[len(prompt):]`）；评估时却直接用了 `generate()` 的返回值，而它包含提示：

```listing ch03/09_grpo.py#L192-L196
```

奖励函数取的是全文的最后一个整数。如果模型的回答里没有数字，取到的就是提示里的数字。在 DLFS6 的 81 道题里，提示的最后一个数是 j，而答案是 i + j（i ≥ 1），两者永远不相等，所以恰好无害；换成 `0+5=` 这样的题，一个空回答也会被判为答对：

```verify id=Z.5-m title="空回答：81 道原题都判为 0 分，但 0+5= 会被判为答对"
template = "### Instruction:\n{}+{}=\n\n### Response:\n"         # GRPODataset 的提示
empty = [calculate_reward(i + j, template.format(i, j)) for i in range(1, 10) for j in range(1, 10)]
print("81 道原题、回答为空：得分之和 =", sum(empty))
print("0+5=、回答为空：得分 =", calculate_reward(5, template.format(0, 5)))   # calculate_reward 来自验证 Z.4-f
assert sum(empty) == 0 and calculate_reward(5, template.format(0, 5)) == 1.0
```

```localcheck verify/test_reference.py::test_e32_grpo_evaluation_scores_prompt_plus_response title="源码中训练与评估的两种写法；书中的奖励函数在 0+5= 上给出假阳性"
```

**修法**：评估时也写 `response = generate(...)[len(prompt):]`。

<div id="e33"></div>

### E33 用 pickle 保存与读取分词规则

`BPETokenizer.load_from` 用 `pickle.load` 读取 `merge_rules.pkl`，而这些文件要从网上下载。Python 的文档明确提醒：pickle 不安全，只能读取你信任的数据，因为反序列化时可以执行任意代码（[pickle 模块的文档](https://docs.python.org/3/library/pickle.html)）。我们检查了三个随书发布的文件：里面只有构造字典与整数元组的操作码，没有导入或调用任何函数，所以它们本身是安全的。

```localcheck verify/test_reference.py::test_e33_shipped_merge_rules_pickles_hold_only_data title="三个 merge_rules.pkl 只含数据操作码"
```

**修法**：把规则存成 JSON 列表 `[[id1, id2, new_id], ...]`。这样既不会执行代码，也保留了顺序（[E27](#e27)）。

<div id="e49"></div>

### E49 `no_sync()` 只包住 backward，每个 micro-step 都在通信

第 9 章把梯度累积和 DDP 合在一起用。原书的想法是：前 3 个 micro-step 先不通信，只在最后一个 micro-step 做一次 all-reduce。`model.no_sync()` 就是“这一次先别通信”的意思：

```listing ch09/01_pretrain.py#L172-L188
```

问题在于 `no_sync()` 只包住了 `loss.backward()`，前向在它外面，而 DDP 在前向时就决定了这一轮要不要通信。于是每个 micro-step 都照样做了 all-reduce。结果并没有错：每张卡最后得到的梯度相同，也等于单卡把全部 micro-batch 当成一个大 batch 算出的梯度；只是通信多了 3 倍，每步 4 次而不是 1 次。我们用 gloo 后端在 CPU 上起了 2 个真正的进程，用一个通信钩子数 all-reduce 的次数：原书的写法是 `[1, 1, 1, 1]`，把前向也放进 `no_sync()` 是 `[0, 0, 0, 1]`（细节见 [9.3 节](ch09.html#s9-3)）。

```localcheck verify/test_ch09.py::test_ddp_two_processes_gloo needs="PyTorch" title="2 个 gloo 进程：初值被广播；梯度等于单进程的大 batch；原书写法每个 micro-step 都在通信"
```

**修法**：把前向也放进 `no_sync()`；或者像 nanoGPT 那样，在前向之前设置 `model.require_backward_grad_sync = (micro_step == accumulation_steps - 1)`。

## Z.6 低：注释、文档与可移植性

这一组都是小问题，修法也都只有一两行。它们由同一个本地测试逐条核对源码，那个测试的源码可以在本节末尾展开。外部链接都查证于 2026-10-12。

| 编号 | 位置 | 问题 | 修法 |
|---|---|---|---|
| <span id="e34"></span>E34 | ch03/03_alpaca.py 第 31–32 行 | 第 32 行注释里的第 31 个 id（下标 30）写的是 930（`'alize'`），实际输出是 932（`' you'`）。这行注释来自 2025-12-18 的首次提交，`merge_rules.pkl` 则是 2026-01-07 才提交的（[验证 Z.2-a](#v-Z.2-a)） | 改成 932 |
| <span id="e35"></span>E35 | ch05/05_kvcache.py 第 116、301 行 | 三处注释与代码不符。注释“KVキャッシュ使用時は不要”说用缓存时不需要掩码，但 prefill 需要，代码也加了；“固定シード”只固定了起始 token，文件里没有设置随机种子，模型权重每次运行都不同；文件末尾被注释掉的一致性检查用到了已经删除的变量 `max_new_tokens`，取消注释就会报 `NameError` | 注释改成“decode 時は不要”；加 `torch.manual_seed`；补上 `max_new_tokens` |
| <span id="e36"></span>E36 | README.md | 运行示例写的是 `python ch02/10_gpt.py`，实际文件名是 `10_gpt2.py` | 改文件名 |
| <span id="e37"></span>E37 | README.md | 正误表链接 [`…/wiki/errata`](https://github.com/oreilly-japan/deep-learning-from-scratch-6/wiki/errata) 不存在，访问时先被重定向到 `/wiki`，再回到仓库首页 | 本页 |
| <span id="e38"></span>E38 | README.md 第 26 行 | 写的是“PyTorch 2.x”，但 `webbot/model.py` 用了 `enable_gqa`，需要 2.5 以上；commit `f85907f`（2026-04-10）只改了 requirements.txt 与 pyproject.toml | README 写明 2.5 以上 |
| <span id="e39"></span>E39 | ch05/03_rmsnorm.py 第 6 行 | `__init__(self, x)` 的参数其实是维数，[PR #6](https://github.com/oreilly-japan/deep-learning-from-scratch-6/pull/6)（2026-06-12，尚未合并）建议改名为 `embed_dim` | 改名 |
| <span id="e40"></span>E40 | ch02/08_multi_head.py 第 66 行 | 注释把 H（头数）叫作“ヘッドサイズ”，而 nanoGPT 的 `head_size` 指每个头的维数 D（见[术语对照](glossary.html)） | 改成“ヘッド数” |
| <span id="e41"></span>E41 | ch03/09_grpo.py 第 215 行 | 文件名 `loss_grpo.png`，画的却是准确率 | 改成 `acc_grpo.png` |
| <span id="e42"></span>E42 | ch06/lr_graph.py 第 5 行 | 字体写死成 macOS 的 `'Hiragino Sans'`。没有这款字体的系统（比如本机的 Windows）会退回 DejaVu Sans，它没有日文字形，图上的日文显示成方块 | 按平台选择，或用注释里的 `'Yu Gothic'` |
| <span id="e43"></span>E43 | ch03/01、03_sft、09；ch06/05、08 | 训练脚本都没有设随机种子，结果不可复现（只有 ch09/01 设了） | 开头加 `torch.manual_seed(0)` |
| <span id="e44"></span>E44 | ch09/01_pretrain.py 第 86、177 行 | `autocast(device_type='cuda')` 写死，在没有 CUDA 的机器上不会生效，只留下一条警告，计算悄悄地回到 float32 | 用 `device.type` |
| <span id="e45"></span>E45 | ch09/01_pretrain.py 第 31、123、194 行 | `is_main` 用的是 `LOCAL_RANK`，多机时每台机器都有一个“主进程”，wandb、评估、保存会重复（[验证 9.3-c](ch09.html#v-9.3-c)）；记进 wandb 的 `effective_batch_size` 没有乘卡数（[验证 9.3-e](ch09.html#v-9.3-e)）；`train/loss` 只是 rank 0 最后一个 micro-batch 的值（[9.6 节](ch09.html#s9-6)） | 用全局 `RANK`；乘 `world_size`；对 micro-batch 求平均再 all-reduce |
| <span id="e46"></span>E46 | storybot/tokenizer.py 第 79–97 行 | 除了参数的写法（四个参数，还是一个元组再解包），`process_single_chunk` 与 `pretoken_chunk` 的函数体完全相同；前者从未被调用 | 删除 |
| <span id="e47"></span>E47 | ch01/08_eval.py 第 24、38–49 行 | 压缩率只在前 10,000 个字符上测（2.12），全语料是 2.42；与 tiktoken 比较的代码写在字符串里，永远不会执行 | 用全语料，或注明样本 |
| <span id="e48"></span>E48 | storybot/download.sh；HF | `download.sh` 里被注释掉的一行列出了 `storybot/model_dpo.pt`，可是 HF 上没有这个文件（[下载地址](https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_dpo.pt)返回 404）；WebBot 也没有发布权重，[webbot 目录](https://huggingface.co/datasets/koki0702/zero-llm-data/tree/main/webbot)里只有数据和分词规则。`ch06/09_llm_judge.py` 因此必须先跑 `08_dpo.py` | 先训练再评估 |

```localcheck verify/test_reference.py::test_z6_comment_doc_and_portability_facts title="逐条核对源码中的这些写法"
```

```localcheck verify/test_reference.py::test_errata_external_facts_match_the_local_files title="HF 的文件列表与 GitHub 的记录（2026-10-12）：本地检查点的 sha256 与 HF 相同；model_dpo.pt 与 WebBot 权重不存在"
```

```localcheck verify/test_ch09.py::test_script_ddp_and_logging_details needs="PyTorch" title="E45、E49：读原书脚本的语法树，no_sync 只包住 backward、is_main 用 local_rank、effective_batch_size 没乘卡数"
```

```localcheck verify/test_reference.py::test_e42_hiragino_falls_back_to_a_font_without_japanese title="E42：找不到 Hiragino Sans 时，matplotlib 退回的字体里没有日文字形"
```

```localcheck verify/test_reference.py::test_e44_cuda_autocast_does_nothing_on_cpu needs="PyTorch" title="E44：在 CPU 上，autocast('cuda') 里的矩阵乘法仍是 float32，并给出警告"
```

```localcheck verify/test_reference.py::test_e47_compression_on_the_first_10k_chars_vs_whole_corpus needs="PyTorch" title="E47：前 10,000 个字符 2.1236，全语料 2.4229"
```

```localcheck verify/test_reference.py::test_e48_download_script_lists_unpublished_dpo_model title="E48：download.sh 的注释行与 09_llm_judge.py 都引用了 model_dpo.pt"
```

## Z.7 已经修好的问题

仓库的提交历史里还能看到几处已经修好的问题，这里记下来，免得读者在旧的副本或旧的讨论里看到时困惑：

| commit | 修了什么 |
|---|---|
| `8b3911f` | KV cache 的 prefill 阶段没有加因果掩码。这个错误用只有 1 个 token 的提示测不出来，因为 1 个 token 不需要掩码 |
| `1760066` | 第 6 章评估时的 token 计数（`y.numel()`）与 autocast 的 dtype |
| `6855e70` | RoPE 改为在 float32 下计算 |
| `c240183` | 简化了因果掩码的注释和条件式（E35 的注释就是这次留下的） |

[issue #4](https://github.com/oreilly-japan/deep-learning-from-scratch-6/issues/4)（2025-12-28）报告的 `ch06/07_llm_judge.py` 第 21 行的非法字符 `ß`，在现在的代码里已经不存在了，虽然这个 issue 本身还开着。

```localcheck verify/test_reference.py::test_fixed_upstream_history_is_in_the_pinned_submodule title="这些 commit 都在固定的子模块历史里；ß 已不在文件中"
```

## 小结

50 个问题里，真正影响书中结果的只有少数几个，影响最大的是 E01：GRPO 更新时开着 dropout。在本书的 GPU 运行里，原样跑满 500 次迭代停在 86.4%；从同一个 SFT 检查点出发、只加一行 `model.eval()`，第 51 次迭代就到了 98.8%，之后一直是 100%。其次是几个只在特定用法下出现、后果却不小的问题：用 HF 上下载的 StoryBot 权重，第 6 章的脚本一加载就报错（E02）；把 git 检出的文本交给第 4 章的分词器，CRLF 会悄悄改变词表（E03）；用 8 张卡原样运行 WebBot 的预训练，读的 token 和用掉的 GPU 时间都变成 8 倍（E50）。

其余大多是在边界情况下静默出错的问题。它们的共同特点是“不报错”：位置被冻结（E04）、位置被广播（E05）、掩码缺失（E06）、洗牌失效（E07）、错误被吞掉（E09）。这也是我们坚持给每个结论配一个可以重跑的检查的原因：不报错的问题，只有去测才会发现。

这一页的问题，大多来自把代码放到书中没有设想过的场景里：更长的输入、不同的平台、更多的卡、拿去对接别的测试。[CS336 对照总表](cs336.html)记录的就是最后一种：我们把 DLFS6 的实现接到斯坦福 CS336 的测试上，看哪些能通过、哪些不能，以及为什么。

本章所学的内容：

- 原书 38 处“写在注释里的期望输出”中，30 处逐字一致，7 处只是写法不同，1 处已经过期（[验证 Z.2-a](#v-Z.2-a)）。
- GRPO 更新时开着 dropout：原样跑满 500 次迭代停在 86.4%，只加一行 `model.eval()`，第 51 次迭代就到 98.8%，之后一直是 100%（[验证 Z.3-a](#v-Z.3-a)）。
- 二进制模式读取会保留 CRLF 里的 `\r`，BPE 会把 `\r\n` 学成一个 token；书中的流程恰好不会把检出的文件交给这样的读取（[验证 Z.4-b](#v-Z.4-b)）。
- StoryBot 的 RoPE 把超出范围的位置夹在最后：生成时位置被冻结，输入恰好多 1 个 token 时位置被广播（[验证 Z.4-c](#v-Z.4-c)、[验证 Z.4-d](#v-Z.4-d)）。
- `itertools.cycle` 只迭代一次并缓存所有元素，所以 `shuffle=True` 只在第一轮生效（[验证 Z.4-e](#v-Z.4-e)）。
- `max_iters` 写死时，加卡不会更快，只会读更多的 token（[验证 Z.4-i](#v-Z.4-i)）。
- 概率空间的 ratio 在 p = 1e-8 时只有 0.5；对数空间没有这个问题（[验证 Z.5-d](#v-Z.5-d)）。

这是本书的最后一页。如果你在原书代码里发现了新的问题，不妨照这一页的格式记下来：一个最小复现、一个修法，再加一个谁都能重跑的检查。
