---
chapter: Y
slug: glossary
order: 190
kind: extra
title: 术语与命名对照
nav_title: 术语与命名对照
lede: 同一个量，在不同的代码库里有不同的名字；同一个词，在不同的场合又指不同的东西。这一页把本书的术语、各家代码的参数名、强化学习训练的指标名对齐在一起，并拆开几个最容易混淆的词。
pyodide_packages: [numpy]
---

读论文、读代码、看训练日志的时候，最耽误时间的往往不是概念本身，而是名字。DLFS6 叫 `embed_dim` 的那个数，Hugging Face 叫 `hidden_size`，CS336 叫 `d_model`，GPT-2 的代码叫 `n_embd`；概念完全相同，第一次见到时却很容易以为是四样东西。反过来，“epoch”“step”“batch”这些词，在不同的场合又各有所指。

这一页分四部分。Y.1 是本书的术语表，每个术语都指向它第一次出现的小节。Y.2 是参数名的对照，Y.3 是强化学习训练指标的对照。Y.4 拆开几个一词多义的词。正文里也有两张小一些的对照表：[2.10.1 节](ch02.html#s2-10-1)的方框对齐了 CodeBot 的参数名，[3.9.3 节](ch03.html#s3-9-3)的专栏对齐了最常用的几个强化学习指标。这一页把它们扩展到更多的代码库，并逐项核对出处。

表格很多，可以先用下面的搜索框：输入中文或英文都行，大小写、空格和下划线都不计较。

```widget glossary-search caption="在本页的表格里查找，同时列出标题里含这个词的小节和方框。试试 hidden_size、critic、probe、gbs 或“上下文”。"
{"scope": "article.chapter"}
```

## Y.1 术语表

术语按主题分组。“首次出现”指向本书里第一次正式讲解它的小节；中文一列沿用正文的写法：正文里直接用英文的术语，写成“英文（中文译名）”，如 Attention（注意力）、KV cache（KV 缓存）；token、logits 这样正文从不翻译、也没有统一译名的词，只写英文。最后一组是一词多义的词，它们在 Y.4 里逐个拆开。

### 分词

| 中文 | English | 一句话 | 首次出现 |
|---|---|---|---|
| 分词器 | tokenizer | 把文字变成整数序列、再变回文字的组件 | [1.1](ch01.html#s1-1) |
| token | token | 分词器切出的单位，模型的输入和输出都以它计 | [1.1](ch01.html#s1-1) |
| 词表、词表大小 | vocabulary, vocabulary size (V) | 全部 token 的集合与个数：CodeBot 1,000、StoryBot 10,000、WebBot 50,000 | [1.1](ch01.html#s1-1) |
| 码位 | code point | Unicode 给每个字符的编号，从 0 到 0x10FFFF | [1.1](ch01.html#s1-1) |
| UTF-8 | UTF-8 | 把一个码位写成 1–4 个字节的变长编码 | [1.2](ch01.html#s1-2)、[附录 A](appendix-a.html) |
| BPE（字节对编码） | byte pair encoding (BPE) | 反复把最常见的相邻两个 token 合并成一个新 token | [1.3](ch01.html#s1-3) |
| 合并规则 | merge rules | BPE 学到的合并表；DLFS6 存成 `{(id1, id2): new_id}` | [1.3](ch01.html#s1-3) |
| 平票规则 | tie-break | 两对出现次数相同时选哪一对 | [1.3](ch01.html#s1-3)（专栏） |
| 特殊 token | special token | 不由 BPE 学出、表示文档边界等含义的 token，如 `<\|endoftext\|>` | [1.5](ch01.html#s1-5) |
| 预分词 | pre-tokenization | 先用正则把文本切成“词”，再在每个词内部做 BPE | [1.6](ch01.html#s1-6) |
| 压缩率 | compression ratio | 字节数 ÷ token 数，越大说明一个 token 装的文字越多 | [1.8](ch01.html#s1-8) |
| 去重、增量更新、分块、并行 | deduplication, incremental update, chunking, parallelism | 第 4 章让 BPE 训练提速的四步 | [4.1](ch04.html#s4-1)–[4.4](ch04.html#s4-4) |
| 内存映射 | memory-mapped file (`np.memmap`) | 把磁盘上的文件当作数组来读写，用到哪一段才读入哪一段 | [4.7](ch04.html#s4-7)、[6.5](ch06.html#s6-5) |

### 模型

| 中文 | English | 一句话 | 首次出现 |
|---|---|---|---|
| Attention（注意力） | Attention | 按相似度给各个位置加权，混合它们的值 | [2.1](ch02.html#s2-1) |
| 查询、键、值 | query, key, value (Q, K, V) | “拿什么去查”“和什么比”“取回什么” | [2.1](ch02.html#s2-1)–[2.2](ch02.html#s2-2) |
| 缩放 | scaling (1/√d) | 点积除以 √d，让 softmax 的尖锐程度不随维数变化 | [2.3](ch02.html#s2-3)、[附录 B](appendix-b.html) |
| logits | logits | softmax 之前的分数，每个 token 一个 | [2.4](ch02.html#s2-4) |
| 位置编码 | positional encoding | 把 token 的先后顺序告诉模型 | [2.5](ch02.html#s2-5) |
| 因果掩码 | causal mask | 让每个位置只能看见它自己和它前面的位置 | [2.6](ch02.html#s2-6) |
| 多头 Attention（多头注意力） | multi-head attention | 把注意力拆成 H 个头并行计算，再拼接起来 | [2.8](ch02.html#s2-8) |
| 残差流 | residual stream | 每一层都往上“加”自己的结果的那条主干；它的宽度就是 `embed_dim` | [2.9](ch02.html#s2-9) |
| LayerNorm（层归一化） | LayerNorm | 对每个 token 的向量减去均值、除以标准差 | [2.9](ch02.html#s2-9) |
| FFN（前馈网络） | feed-forward network (FFN) | 每个位置各自通过的全连接层；它中间那层的宽度是 `ff_dim` | [2.9](ch02.html#s2-9) |
| GELU | GELU | CodeBot 的 FFN 用的激活函数，平滑版的 ReLU | [2.9](ch02.html#s2-9) |
| dropout | dropout | 训练时随机把一部分值置 0、其余按比例放大；eval 模式下关闭 | [2.10](ch02.html#s2-10)、[5.3](ch05.html#s5-3) |
| 权重共享 | weight tying | 输入 embedding 与输出层共用一个矩阵（CodeBot） | [2.10](ch02.html#s2-10) |
| RoPE（旋转位置编码） | rotary position embedding (RoPE) | 按位置旋转 Q 与 K，让点积只与相对位置有关 | [5.1](ch05.html#s5-1) |
| Swish、SiLU | Swish, SiLU | $x\,\sigma(x)$：SwiGLU 的门控用它 | [5.2](ch05.html#s5-2) |
| SwiGLU | SwiGLU | 用 SiLU 作门控的 FFN，有三个权重矩阵 | [5.2](ch05.html#s5-2) |
| RMSNorm | RMSNorm | 只除以均方根、不减均值的归一化 | [5.3](ch05.html#s5-3) |
| KV cache（KV 缓存） | KV cache | 生成时把算过的 K、V 存起来，每步只算新 token | [5.5](ch05.html#s5-5) |
| Prefill、Decode | prefill, decode | 生成的两个阶段：先把整段提示一次算完，之后每步只送进一个新 token | [5.5](ch05.html#s5-5) |
| scaling law | scaling law | loss 随参数量、数据量变化的经验规律 | [8.1](ch08.html#s8-1) |
| Flash Attention | FlashAttention | 分块计算注意力，不显式存下 C×C 的分数矩阵 | [8.2](ch08.html#s8-2) |
| GQA（分组查询注意力） | grouped-query attention (GQA) | 几个查询头共用一组 K、V，KV cache 因此变小 | [8.3](ch08.html#s8-3) |
| MoE（混合专家） | mixture of experts (MoE) | 每个 token 只经过少数几个“专家”FFN | [8.4](ch08.html#s8-4) |
| torch.compile | torch.compile | 把模型捕获成计算图再编译，相邻的运算融合成一个 kernel，少读写显存 | [8.5](ch08.html#s8-5) |

### 训练

| 中文 | English | 一句话 | 首次出现 |
|---|---|---|---|
| 预训练 | pretraining | 在大量文本上学习“预测下一个 token” | [3.1](ch03.html#s3-1) |
| 交叉熵 | cross-entropy | 正确 token 的负对数概率的平均；取自然对数时单位是 nats | [0.2](ch00.html#s0-2)、[2.4](ch02.html#s2-4)（专栏）、[3.1](ch03.html#s3-1) |
| nats、bits | nat, bit | 交叉熵、熵与 KL 的单位：用 ln 算是 nats，用 log₂ 算是 bits，1 nat ≈ 1.4427 bits | [0.2](ch00.html#s0-2)、[3.1](ch03.html#s3-1) |
| 困惑度 | perplexity | 交叉熵取指数，可以理解为“模型平均在几个候选之间犹豫” | [2.4](ch02.html#s2-4) |
| batch | batch | 一次参数更新用到的样本。多卡时是 micro-batch × 累积次数 × 卡数；强化学习里还要分清题数和回答条数 | [3.1](ch03.html#s3-1)、[9.3](ch09.html#s9-3)、[Y.4](#y4-batch) |
| epoch、iteration | epoch, iteration | 把数据过一遍；一次参数更新。“一遍”有不同的数法 | [3.1](ch03.html#s3-1)、[Y.4](#y4-epoch) |
| padding、−100 标签 | padding, ignore index (−100) | 把一个 batch 的序列补到同样长；标签为 −100 的位置不计入 loss | [3.3](ch03.html#s3-3) |
| AdamW | AdamW | 把权重衰减与梯度更新分开的 Adam | [6.1](ch06.html#s6-1)–[6.2](ch06.html#s6-2)、[附录 D](appendix-d.html) |
| 学习率调度 | learning-rate schedule | 学习率随步数的变化。第 3 章的三个训练脚本和第 6 章的 DPO 用恒定的学习率；StoryBot、WebBot 的预训练先 warmup，再线性衰减到 0 | [6.3](ch06.html#s6-3) |
| warmup（预热）、D2Z | warmup, decay to zero (D2Z) | 学习率先从 0 线性升到峰值，再线性降到 0 | [6.3](ch06.html#s6-3) |
| 混合精度 | mixed precision | 用 bf16 或 fp16 计算，用 fp32 保存参数 | [6.4](ch06.html#s6-4) |
| bf16、fp16 | bfloat16, float16 | 两种 16 位浮点：bf16 保留 fp32 的 8 位指数，范围大、精度低；fp16 精度较高，范围只到 65,504 | [6.4](ch06.html#s6-4) |
| 梯度裁剪 | gradient clipping | 梯度的总范数超过阈值时整体缩小 | [3.9](ch03.html#s3-9)、[6.5](ch06.html#s6-5) |
| 梯度累积 | gradient accumulation | 累积几个 micro-batch 的梯度后再更新一次 | [9.1](ch09.html#s9-1) |
| 梯度检查点 | gradient checkpointing | 前向时少存激活，反向时重新计算 | [9.2](ch09.html#s9-2) |
| DDP（分布式数据并行） | distributed data parallel (DDP) | 每张卡一份模型，各算一部分数据，再平均梯度 | [9.3](ch09.html#s9-3) |
| 全局 batch | global batch size (gbs) | 一次参数更新在所有卡上用到的样本数：micro-batch × 累积次数 × 卡数 | [9.3](ch09.html#s9-3)、[Y.4](#y4-batch) |
| 进程编号 | `RANK`、`LOCAL_RANK`、`WORLD_SIZE` | `torchrun` 给每个进程设的环境变量：全局编号、本机内的编号、进程总数 | [9.3](ch09.html#s9-3) |
| ZeRO | ZeRO | 把优化器状态（以及梯度、参数）分片存到各张卡上 | [9.4](ch09.html#s9-4) |
| Muon | Muon | 先把更新量正交化再应用的优化器 | [9.5](ch09.html#s9-5) |
| Weights & Biases | Weights & Biases (wandb) | 把训练曲线记到网上的服务；第 9 章的脚本用它记录 | [9.6](ch09.html#s9-6) |
| 对拍 | parity check | 把同一个输入交给两份实现（比如本书的 NumPy 版与原书的 PyTorch 版），逐项比较输出，差别应当只在浮点舍入的范围内 | [T.5](train-in-browser.html#sT-5) |

```localcheck verify/test_reference.py::test_cs336_clipping_and_lr_schedules title="“学习率调度”一行：从原书的 6 个训练脚本读出，只有 StoryBot、WebBot 的预训练按步调整学习率"
```

### 生成、评估与后训练

| 中文 | English | 一句话 | 首次出现 |
|---|---|---|---|
| 温度 | temperature | logits 除以 T 再做 softmax；T = 0 就是贪心解码 | [3.2](ch03.html#s3-2) |
| Top-k、Top-p | top-k, top-p sampling | 只在概率最高的 k 个、或累计概率 p 以内的 token 里采样 | [3.2](ch03.html#s3-2) |
| SFT（监督微调） | supervised fine-tuning (SFT) | 在“指令 → 回答”的数据上继续训练 | [3.3](ch03.html#s3-3) |
| 策略 | policy | 给定上下文、输出下一个 token 的分布，也就是语言模型本身 | [3.5](ch03.html#s3-5) |
| 轨迹 | trajectory | 状态与动作交替组成的序列；对语言模型来说，就是一个完整的回答 | [3.5](ch03.html#s3-5) |
| 奖励 | reward | 给一次完整回答的分数 | [3.5](ch03.html#s3-5) |
| 回报 | return | 从某一步开始往后累计的奖励 | [3.5](ch03.html#s3-5) |
| 策略梯度 | policy gradient | 用奖励给 log π 的梯度加权 | [3.6](ch03.html#s3-6)、[附录 C](appendix-c.html) |
| 基线、优势 | baseline, advantage | 从奖励里减去的参考值；减完剩下的就是优势 | [3.7](ch03.html#s3-7) |
| 价值网络、critic | value network, critic | 另外训练一个网络来估计“从这里往后还能拿多少奖励”，用作基线；GRPO 没有它 | [3.7](ch03.html#s3-7) |
| rollout | rollout | 用当前策略为一道题采出的一条完整回答，连同它的奖励；GRPO 每道题采 G 条 | [3.7](ch03.html#s3-7)、[3.9](ch03.html#s3-9) |
| 同策略、离策略 | on-policy, off-policy | 用当前策略采的样本更新它自己；或者用旧策略采的样本去更新新策略 | [3.8](ch03.html#s3-8) |
| 重要性采样、裁剪 | importance sampling, clipping | 离策略时给每个样本乘上概率比 π_new / π_old；再把这个比限制在 [1−ε, 1+ε] 之内 | [3.8](ch03.html#s3-8) |
| GRPO（组相对策略优化） | group relative policy optimization | 以同一道题的一组回答的平均奖励作基线 | [3.7](ch03.html#s3-7)、[3.9](ch03.html#s3-9) |
| Dr. GRPO | Dr. GRPO | 不除以标准差、不按长度归一化的 GRPO | [3.7](ch03.html#s3-7)（专栏） |
| LLM-as-a-Judge（LLM 评审） | LLM-as-a-Judge | 用另一个 LLM 给生成结果打分 | [6.6](ch06.html#s6-6) |
| RLHF（基于人类反馈的强化学习） | reinforcement learning from human feedback | 先用人类偏好训练奖励模型，再做强化学习 | [6.7](ch06.html#s6-7) |
| KL 散度 | KL divergence | 两个分布相差多少：非负，只在两者相同时为 0，而且不对称 | [6.7](ch06.html#s6-7)、[附录 E](appendix-e.html) |
| Bradley-Terry 模型 | Bradley-Terry model | 把“A 比 B 好”的概率写成两者奖励之差的 sigmoid | [6.7](ch06.html#s6-7)、[附录 E](appendix-e.html) |
| DPO（直接偏好优化） | direct preference optimization | 直接用偏好对优化策略，不需要显式的奖励模型 | [6.7](ch06.html#s6-7)、[附录 E](appendix-e.html) |

### 一词多义

| 词 | English | 常见的几种意思 | 详见 |
|---|---|---|---|
| 探针 | probe | 可解释性研究里的线性探针；训练前先跑一遍的“第 0 步”评估 | [Y.4](#y4-probe) |
| epoch | epoch | 按 token 数、或按 `DataLoader` 的样本数过一遍；强化学习里还指同一批 rollout 拿来更新几遍 | [Y.4](#y4-epoch) |
| step | step | 一次参数更新、梯度累积里的一个 micro-step、GRPO 的一轮、生成时的一个 decode step | [Y.4](#y4-step) |
| 上下文 | context | 一次前向计算能看见的 token 数；也常指整段对话历史和累计消耗的 token | [Y.4](#y4-context) |
| batch | batch | micro-batch、全局 batch；强化学习里的题数与回答条数 | [Y.4](#y4-batch) |
| critic | critic | PPO 里的价值网络；也是 veRL 指标名的一个分组前缀，与有没有价值网络无关 | [Y.3](#sY-3) |

## Y.2 命名对照：同一个量的五个名字

先看一张总表。每一行是同一个量，每一列是一个代码库里的写法：

| 含义 | DLFS6 | Hugging Face | CS336 A1 | GPT-2（HF 的 `GPT2Config`） | nanoGPT |
|---|---|---|---|---|---|
| 词表大小 | `vocab_size` | `vocab_size` | `vocab_size` | `vocab_size` | `vocab_size` |
| 上下文长度 | `max_context_len` | `max_position_embeddings` | `context_length`（RoPE 里叫 `max_seq_len`） | `n_positions` | `block_size` |
| 残差流宽度 | `embed_dim` | `hidden_size` | `d_model` | `n_embd` | `n_embd` |
| 注意力头数 | `n_head` | `num_attention_heads` | `num_heads` | `n_head` | `n_head` |
| KV 头数 | `n_kv_head`（WebBot） | `num_key_value_heads` | A1 没有 | 没有 | 没有 |
| 层数 | `n_layer` | `num_hidden_layers` | `num_layers` | `n_layer` | `n_layer` |
| FFN 中间层宽度 | `ff_dim` | `intermediate_size` | `d_ff` | `n_inner`（缺省为 4 × `n_embd`） | 固定为 4 × `n_embd` |
| 每头维数 | `head_dim`（= `embed_dim // n_head`） | OLMo-2、Qwen2.5 的配置里没有单列，由 `hidden_size ÷ num_attention_heads` 算出；Qwen3 等较新的配置单列 `head_dim`，而且可以不等于这个商（Qwen3-0.6B：128 ≠ 1024 ÷ 16） | `d_k`（RoPE 的参数） | `n_embd // n_head` | `C // n_head`，注释里叫 `hs` |
| RoPE θ | `theta` | `rope_theta` | `rope_theta`（RoPE 里叫 `theta`） | 没有 RoPE | 没有 RoPE |
| 归一化的 eps | CodeBot：`nn.LayerNorm` 的默认值 1e-5；StoryBot、WebBot：`nn.RMSNorm` 不传 eps，float32 下约 1.19e-7 | `rms_norm_eps` 或 `layer_norm_epsilon` | `eps` | `layer_norm_epsilon` | 写死为 `1e-5` |
| 权重共享 | CodeBot 共享，其余不共享 | `tie_word_embeddings` | 不共享 | 共享 | 共享 |

这些名字都来自源码本身。DLFS6 一列取自三个 `GPT.__init__` 的参数，CS336 一列取自 A1 的 `tests/adapters.py`，Hugging Face 一列取自真实模型的 `config.json` 与 `GPT2Config` 的源码，nanoGPT 一列取自 `model.py` 的 `GPTConfig`。[2.10.1 节](ch02.html#s2-10-1)的方框里有一张只含 CodeBot 的同类表，这里多了 Hugging Face 上较新的模型和归一化、权重共享等几行。我们把这些来源的原文、链接与 commit 存成了一个数据文件，下面的验证单元逐项核对：

```verify id=Y.2-a title="名字取自源码：GPT2Config 把 hidden_size 直接映射到 n_embd；Qwen3-0.6B 单列的 head_dim 不等于 hidden_size ÷ 头数" files=data/glossary/sources.json
import json, re

src = json.load(open("data/glossary/sources.json", encoding="utf-8"))
amap = dict(re.findall(r'"(\w+)": "(\w+)"', " ".join(src["hf"]["gpt2_attribute_map"]["lines"])))
print("GPT2Config.attribute_map：", amap)
assert amap == {"hidden_size": "n_embd", "max_position_embeddings": "n_positions",
                "num_attention_heads": "n_head", "num_hidden_layers": "n_layer"}

dl = src["dlfs6"]["signatures"]
print("DLFS6 StoryBot GPT(...)：", dl["storybot/model.py GPT.__init__"])
print("DLFS6 WebBot   GPT(...)：", dl["webbot/model.py GPT.__init__"])
a1 = src["cs336_a1"]["signatures"]
print("CS336 run_transformer_lm(...)：", a1["run_transformer_lm"][:7])
nano = " ".join(src["nanogpt"]["gptconfig_lines"])
print("nanoGPT GPTConfig：", re.findall(r"(\w+): \w+ =", nano))
assert dl["storybot/model.py GPT.__init__"] == ["vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim", "theta"]
assert "n_kv_head" in dl["webbot/model.py GPT.__init__"]
assert a1["run_transformer_lm"][:7] == ["vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta"]
assert set(re.findall(r"(\w+): \w+ =", nano)) == {"block_size", "vocab_size", "n_layer", "n_head", "n_embd", "dropout", "bias"}
olmo = src["hf"]["olmo2_0425_1b"]["config"]
assert {"hidden_size", "intermediate_size", "num_key_value_heads", "rope_theta", "rms_norm_eps", "tie_word_embeddings"} <= set(olmo)
assert "head_dim" not in olmo and "head_dim" not in src["hf"]["qwen25_math_1p5b"]["config"]
q3 = src["hf"]["qwen3_0p6b"]["config"]
print(f"Qwen3-0.6B：head_dim = {q3['head_dim']}，hidden_size ÷ 头数 = {q3['hidden_size'] // q3['num_attention_heads']}")
assert q3["head_dim"] == 128 != q3["hidden_size"] // q3["num_attention_heads"] == 64
```

```localcheck verify/test_reference.py::test_glossary_norm_eps_defaults needs="PyTorch" title="CodeBot 的 nn.LayerNorm 用默认的 eps = 1e-5；StoryBot、WebBot 的 nn.RMSNorm 不传 eps，等于 float32 的机器精度 1.19e-7"
```

::: confusion hidden_size 到底是哪个数？
这个困惑很自然。在经典的多层感知机里，“hidden size”说的是隐藏层的宽度；Transformer 的 FFN 正好也有一个隐藏层，StoryBot 里它有 1,344 个神经元。于是看到 Hugging Face 配置里的 `hidden_size`，很容易以为它指的是 FFN 中间那一层。

其实不是。Hugging Face 的 `hidden_size` 是**残差流的宽度**，也就是每个 token 的向量在层与层之间传递时的维数：DLFS6 的 `embed_dim`、CS336 的 `d_model`、GPT-2 的 `n_embd`。FFN 中间那一层另有名字：`intermediate_size`，对应 DLFS6 的 `ff_dim`、CS336 的 `d_ff`。上面验证单元里的 `GPT2Config.attribute_map` 写得最直白：`hidden_size` 就是 `n_embd` 的别名。

还可以用参数量来判断。残差流的宽度 d 会出现在几乎每一个权重矩阵里，所以一层的参数量大约是 $12d^2$（注意力 4 个 $d\times d$，FFN 约 $8d^2$）。拿 StoryBot 手算一下：

```verify id=Y.2-b title="按 d = 512 算，12·L·d² 与真实的块参数量只差 1%；按 1344 算差了 7 倍"
L, d, ff = 4, 512, 1344                       # StoryBot：embed_dim = 512，ff_dim = 1344
per_layer = 4 * d * d + 3 * d * ff + 2 * d    # 注意力 4 个 d×d，SwiGLU 3 个 d×ff，两个 RMSNorm
actual = L * per_layer
print(f"真实的块参数量 L × (4d² + 3·d·ff + 2d) = {actual:,}")
for guess in (d, ff):
    est = 12 * L * guess * guess
    print(f"把 {guess} 当作 d：12·L·d² = {est:,}，是真实值的 {est / actual:.2f} 倍")
print(f"SwiGLU 的 3 个矩阵 3·d·ff = {3 * d * ff:,}，粗估里 FFN 的 8d² = {8 * d * d:,}")
assert actual == 12_455_936
assert abs(12 * L * d * d / actual - 1) < 0.02 and 12 * L * ff * ff / actual > 6
assert abs(3 * d * ff / (8 * d * d) - 1) < 0.02            # 1344 ≈ 8 × 512 ÷ 3，所以粗估这么准
```

```localcheck verify/test_reference.py::test_cs336_storybot_parameter_count title="实例化真实的 StoryBot：4 个块共 12,455,936 个参数，与上面的手算相同"
```

StoryBot 的粗估之所以这么准，是因为它的 `ff_dim` 恰好让 SwiGLU 的 3 个矩阵合起来约等于 $8d^2$（1,344 ≈ 8 × 512 ÷ 3）。FFN 更宽的模型，粗估就会偏低，下面的配置翻译器会把差在哪里算出来。

带走一句话：**`hidden_size` = 残差流宽度 = `embed_dim` = `d_model` = `n_embd`；FFN 中间层的宽度是 `intermediate_size` = `ff_dim` = `d_ff`。**
:::

`n_head` 与“head size”也容易混。DLFS6 第 2 章的一行注释把头数 H 叫作“ヘッドサイズ”（[勘误 E40](errata.html#e40)），而 nanoGPT 注释里的 `hs`（head size）指的是每个头的维数 D。读代码时，要以形状为准。

能不能整除，两边的处理也不同。nanoGPT 在构造模型时断言 `n_embd % n_head == 0`；DLFS6 直接用整除计算 `head_dim`，不检查能否整除，`embed_dim=100, n_head=3` 时三个头合起来只有 99 维（[勘误 E19](errata.html#e19)）。

最后是 DLFS6 代码里的单字母记号。它们写在注意力的 `forward` 里：

```listing storybot/model.py#L69-L71
```

| 记号 | 含义 | 本书的取值（CodeBot / StoryBot / WebBot） |
|---|---|---|
| B | batch 里的序列数 | 32（预训练） |
| C | 上下文长度（序列里的 token 数） | 256 / 256 / 1,024 |
| E | 残差流宽度 `embed_dim` | 384 / 512 / 768 |
| H | 头数 `n_head` | 6 / 16 / 12 |
| D | 每头维数 `head_dim` | 64 / 32 / 64 |
| V | 词表大小 | 1,000 / 10,000 / 50,000 |
| G | GRPO 的组大小 | 8 |

想看一个真实模型的配置换成本书的名字是什么样子，可以用下面的翻译器：选一个预设，或者把任意一个 Hugging Face 模型的 `config.json` 粘贴进去。

```widget glossary-config caption="配置翻译器：同一组数在 DLFS6、CS336 和 nanoGPT 里叫什么。下面列出由此算出的块参数量（与粗估差得多时说明差在哪里），最后给出用 DLFS6 的类写出来的样子；DLFS6 写不出的配置（比如 Qwen3-0.6B 的 head_dim）会直接说明。"
{"data": "data/glossary/sources.json", "preset": "olmo2"}
```

默认的 OLMo-2-0425-1B 是一个粗估失灵的例子。它的 `intermediate_size` 是 `hidden_size` 的 4 倍，又是 3 个矩阵的 SwiGLU，所以 FFN 每层有 $12d^2$，加上注意力的 $4d^2$ 一共 $16d^2$，比粗估假设的 $12d^2$ 多出三分之一：

```verify id=Y.2-c title="OLMo-2-0425-1B：按矩阵形状数是 1,073,807,360，粗估 12·L·d² 只有 805,306,368，低了 25%" files=data/glossary/sources.json
c = src["hf"]["olmo2_0425_1b"]["config"]                    # src 来自验证 Y.2-a
d, L, ff, H, kv = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"], c["num_attention_heads"], c["num_key_value_heads"]
D = d // H
attn = 2 * d * H * D + 2 * d * kv * D                        # Q、O 各 d×(H·D)，K、V 各 d×(n_kv·D)
ffn = 3 * d * ff                                             # SwiGLU：3 个 d×ff
blocks = L * (attn + ffn + 2 * d)                            # 再加两个 RMSNorm
rough = 12 * L * d * d
print(f"每层：注意力 {attn / d**2:.0f}d²，FFN {ffn / d**2:.0f}d²；ff / d = {ff / d:.0f}")
print(f"按矩阵形状：{blocks:,}；粗估 12·L·d²：{rough:,}，低了 {1 - rough / blocks:.0%}")
assert (attn, ffn) == (4 * d * d, 12 * d * d) and blocks == 1_073_807_360 and rough == 805_306_368
assert round(1 - rough / blocks, 2) == 0.25
```

## Y.3 强化学习训练指标对照

做 GRPO 这类强化学习训练时，各个框架在日志里给出的指标名差别更大。下表把 DLFS6 的 `09_grpo.py`、CS336 2026 年 A5 要求记录的指标、veRL 与 tinker-cookbook 的指标名放在一起。DLFS6 一列写的是“脚本里有没有、在哪里”；其余三列的名字都取自它们的源码或讲义原文（见下面的验证单元），“—”表示我们核对的源码和讲义里没有找到同名的指标。

| 想看什么 | 怎么读 | DLFS6 `09_grpo.py` | CS336 A5（2026） | veRL | tinker-cookbook |
|---|---|---|---|---|---|
| 平均奖励 | 按回答平均；应该缓慢上升 | 没有记录 | Train rewards (total, format) | `critic/rewards/mean`、`critic/score/mean` | `env/all/reward/total` |
| 有没有学习信号 | 全对或全错的组 advantage 全是 0，没有梯度 | 没有记录 | 建议记录 pass@k（k 从 1 到 G） | — | `env/all/by_group/frac_mixed`、`frac_all_good`、`frac_all_bad` |
| 策略 loss | 第一次更新时等于 −Σ Aᵢ·Lᵢ / N，只有各回答一样长才是 0；高低与学得好坏无关（[3.9 节](ch03.html#s3-9)） | 进度条上的 `loss`（每轮第 2 次更新的值） | The loss | `actor/pg_loss` | — |
| 梯度范数 | 突然变大说明更新不稳 | `clip_grad_norm_` 的返回值（裁剪前的总范数），没有记录 | Gradient norm | `actor/grad_norm` | — |
| 熵 | 降得太快，说明探索在塌缩 | 没有记录 | Token entropy | `actor/entropy` | `optim/entropy` |
| 被裁剪的比例 | 只有同一批样本更新多次时才有意义 | 没有记录 | clip fraction（离策略的实验） | `actor/pg_clipfrac` | — |
| 新旧策略的 KL | 同一批数据上，PPO 的几次更新让策略移动了多少 | 没有记录 | — | `actor/ppo_kl` | — |
| 采样与训练之差 | 同一批 token 在采样时与训练时的对数概率之差，健康时接近 0；原书 GRPO 的 dropout 会让它偏离 0（[3.9 节](ch03.html#s3-9)、[勘误 E01](errata.html#e01)） | 没有记录 | — | — | `optim/kl_sample_train_v1` |
| advantage 的均值 | 按 token 平均：组内按回答平均是 0，但长回答的 token 多、权重大，所以一般不是 0；答错的回答更长时偏负。它不是学习信号 | 没有记录 | — | `critic/advantages/mean` | — |
| 回答长度 | 长度漂移、被截断的比例 | 没有记录 | Val average response length | `response_length/mean`、`response_length/clip_ratio` | `env/all/ac_tokens_per_turn` |
| 学习率 | 确认调度按预期在走 | 固定 7e-6 | — | `actor/lr` | `optim/lr` |
| 评估准确率 | 训练集上的准确率不等于泛化 | 每 10 轮在 81 道**训练题**上贪心解码 | Val rewards (total, format)，在留出集上 | `val-core/<数据源>/<acc 或 reward>/mean@N` | — |

```verify id=Y.3-a title="表里的每个指标名都能在源码或讲义原文里找到" files=data/glossary/sources.json
verl = " ".join(l for lines in src["verl"]["lines"].values() for l in lines)   # src 来自验证 Y.2-a
tinker = " ".join(l for lines in src["tinker_cookbook"]["lines"].values() for l in lines)
a5 = " ".join(src["cs336_a5_2026"]["required_metrics_excerpt"] + src["cs336_a5_2026"]["plotted_metrics_excerpt"]
              + src["cs336_a5_2026"]["clip_fraction_excerpt"])
for name in ("critic/rewards/mean", "critic/score/mean", "critic/advantages/mean", "actor/pg_loss",
             "actor/pg_clipfrac", "actor/ppo_kl", "actor/entropy", "actor/grad_norm",
             "response_length/mean", "response_length/clip_ratio"):
    assert name in verl, name
assert 'rename_dict(actor_output, "actor/")' in verl            # grad_norm 等指标统一加上 actor/ 前缀
assert 'final_metrics["lr"] = lr' in verl                       # lr 也在其中，所以是 actor/lr
assert 'pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"' in verl and 'metric_sec = "val-core"' in verl
assert 'core_var = "acc" if "acc" in var2metric2val else "reward"' in verl
assert 'f"env/all/{k}"' in tinker and '"reward/total"' in tinker # env/all/ + reward/total
for name in ('"by_group/frac_mixed"', '"by_group/frac_all_good"', '"by_group/frac_all_bad"',
             '"optim/entropy"', '"optim/kl_sample_train_v1"', '"optim/lr"', '"ac_tokens_per_turn"'):
    assert name in tinker, name
assert "logprob_diff = sampling_logprobs_actions - training_logprobs_actions" in tinker   # 采样时减训练时
for phrase in ("The loss.", "Gradient norm.", "Token entropy.", "Train rewards (total, format).",
               "Val rewards (total, format).", "Val average response length.", "log pass@k", "log the clip fraction"):
    assert phrase in a5, phrase
print("veRL @", src["verl"]["commit"][:7], " tinker-cookbook @", src["tinker_cookbook"]["commit"][:7],
      " CS336 A5 讲义 @", src["cs336_a5_2026"]["url"].rsplit("/", 1)[-1][:7])
print("全部指标名核对通过")

mu = dict(l.split(": ", 1) for l in src["verl"]["lines"]["verl/trainer/ppo/metric_utils.py"])
line = lambda text: next(int(n) for n, code in mu.items() if code.startswith(text))
order = [line(t) for t in ("if use_critic:", '"critic/values/mean"', "critic_value_metrics = {}", '"critic/rewards/mean"')]
print("veRL metric_utils.py 的行号：if use_critic: {}，values {}，空字典 {}，rewards {}".format(*order))
assert order == sorted(order)          # values 只在 use_critic 分支里；rewards 在分支之外，总会记录
print("奖励与 advantage 的平均方式：")
for t in ("sequence_reward = ", "reward_mean = ", "valid_adv = ", "adv_mean = "):
    print(f"  第 {line(t)} 行  {mu[str(line(t))]}")
assert "masked_select(advantages, response_mask)" in mu[str(line("valid_adv = "))]   # 每个回答 token 一个值
```

DLFS6 一列里，梯度范数和熵都只差一两行代码。把脚本里的梯度裁剪那一行改成记下返回值，再在每轮生成之后、更新之前，用处于 eval 模式的 `old_model`（此时它的权重与 `model` 相同）算一次回答 token 上的平均熵：

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 返回裁剪前的总范数
with torch.no_grad():
    logp = F.log_softmax(old_model(ids)[:, :-1], dim=-1)                 # 与 compute_probs 一样错开一位
    entropy = -(logp.exp() * logp).sum(-1)[mask[:, 1:].bool()].mean()  # 只在回答的 token 上平均
```

```localcheck verify/test_reference.py::test_glossary_grpo_logging_lines needs="PyTorch" title="这几行接进 09_grpo.py 的更新跑 2 轮：返回值等于裁剪前的总范数；logits 全相同时熵恰好是 ln 1000"
```

::: confusion 指标名以 critic/ 开头，说明有 critic 吗？
在 PPO 里，critic 是一个学习出来的价值网络，它估计“从这里往后能拿多少奖励”，用作基线。所以看到 veRL 日志里的 `critic/rewards/mean`，自然会以为训练里有一个 critic。

前缀本身说明不了有没有 critic。veRL 在计算数据指标的函数里，**不论用什么算法**，都把奖励、得分、advantage 和回报的统计放在 `critic/` 这个分组下面。真正需要价值网络的只有 `critic/values/*` 和 `critic/vf_explained_var`：它们写在 `if use_critic:` 分支里，没有价值网络时，日志里根本不会出现这几项（上面的验证单元核对了这几行源码的位置）。需要价值网络的算法（比如 PPO）在 veRL 里确实有 critic，这时日志里才多出这几项；GRPO 则没有：它用同一道题的一组回答的平均奖励作基线，DLFS6 的 `advantages = rewards - rewards.mean()` 就是这一行。

手算一个小例子。三道题，每题采 4 个回答，奖励分别是 [1, 1, 0, 0]、[1, 0, 0, 0] 和 [1, 1, 1, 1]。第 1 组的 advantage 是 [+0.5, +0.5, −0.5, −0.5]，按回答平均是 0。可是 veRL 先把每条回答的 advantage 复制到它的每个 token 上，再在全部回答 token 上平均。假设第 1 组答对的两条回答各 5 个 token，答错的两条各 20 个 token，按 token 平均就是 $(0.5\times5\times2 - 0.5\times20\times2)\div50 = -0.3$：

```verify id=Y.3-b title="critic/rewards/mean 是按回答平均的奖励；advantage 按回答平均是 0，按 token 平均不一定"
import numpy as np

groups = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=float)   # 第 3 组全对
adv = groups - groups.mean(axis=1, keepdims=True)                           # DLFS6 / GRPO 的基线
print(f"critic/rewards/mean = {groups.mean():.4f}   （全部 12 个奖励的平均）")
print(f"advantage 按回答平均 = {adv.mean():.4f}   （每组都减去了自己的均值）")
same = (groups == groups[:, :1]).all(axis=1)
print(f"frac_mixed = {np.mean(~same):.3f}，frac_all_good = {np.mean(same & (groups[:, 0] == 1)):.3f}，"
      f"frac_all_bad = {np.mean(same & (groups[:, 0] == 0)):.3f}")
print("第 3 组的 advantage：", adv[2].tolist(), "← 全对的组没有学习信号")
assert abs(groups.mean() - 7 / 12) < 1e-12 and abs(adv.mean()) < 1e-12 and not adv[2].any()

L = np.array([5, 5, 20, 20])                     # 第 1 组：答对的两条短，答错的两条长
tok = (adv[0] * L).sum() / L.sum()               # veRL：每个回答 token 都带着所在回答的 advantage
print(f"第 1 组：按回答平均 {adv[0].mean():+.2f}，按 token 平均 {tok:+.2f}")
assert adv[0].mean() == 0 and abs(tok + 0.3) < 1e-12
```

所以 `critic/rewards/mean` 是按回答平均的奖励，`critic/advantages/mean` 却是按 token 平均的。组内 advantage 的和是 0，只有各回答一样长时，按 token 的平均才是 0；答错的回答更长时，它就偏负。它反映的是长度和对错之间的关系，不是学习信号。真正告诉你“还有没有学习信号”的，是 tinker-cookbook 的 `frac_all_good` 与 `frac_all_bad`：全对或全错的组，advantage 全是 0。

带走一句话：**`critic/` 只是 veRL 的分组名，不是“有 critic”的证据；GRPO 的基线是组均值，不是一个网络。**
:::

## Y.4 一词多义

最后是几个在不同场合指不同东西的词。

<div id="y4"></div>
<div id="y4-probe"></div>

### probe

“probe”至少有两种常见用法。在可解释性研究里，**探针**（probe）指在模型的隐藏状态上训练一个简单的分类器（通常是线性的），看某种信息能不能从中读出来。在做实验的日常对话里，“先 probe 一下”往往指训练开始前，先用现成的模型把任务跑一遍，看看它的基础通过率，也就是“第 0 步”的评估。

::: confusion “先 probe 一下”到底指什么？
同一个词指两件完全不同的事，听的人自然会困惑：是要训练一个分类器，还是只跑一遍评估？在训练强化学习之前说“probe”，通常指后一种：不改动任何参数，用基础模型做一遍题，记下通过率作为起点。

这一步看起来简单，但小样本的通过率很不稳定。8 道题对了 5 道，看起来是 62.5%，可它的 95% 置信区间是 [0.31, 0.86]，从三成到八成多都有可能。这个区间是 **Wilson 区间**：

$$\frac{\hat p + \dfrac{z^2}{2n} \pm z\sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}$$

这里，$\hat p = k/n$ 是观测到的通过率，$n$ 是题数，$z = 1.96$ 对应 95%。手算 5/8：$\hat p = 0.625$，$z^2/n = 0.480$，中心是 $(0.625 + 0.240) \div 1.480 = 0.584$，半宽是 $1.96 \times \sqrt{0.0293 + 0.0150} \div 1.480 = 0.279$，区间就是 $[0.31, 0.86]$。

作者发布的 SFT 权重在 81 道加法题上贪心解码答对 44 道（54.3%），我们按书自己训练的 SFT checkpoint 也是 44 道（[勘误 E15](errata.html#e15) 的本地验证）。这个通过率的区间是 [0.44, 0.65]：

```verify id=Y.4-a title="5/8 的 95% Wilson 区间是 [0.31, 0.86]（中心 0.584，半宽 0.279）；44/81 是 [0.44, 0.65]"
def wilson(k, n, z=1.96):
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half

for k, n in ((5, 8), (44, 81), (440, 810)):
    lo, hi = wilson(k, n)
    print(f"{k}/{n} = {k / n:.3f}，95% 区间 [{lo:.3f}, {hi:.3f}]，宽 {hi - lo:.3f}")
lo, hi = wilson(5, 8)
print(f"5/8 的中心 {(lo + hi) / 2:.3f}，半宽 {(hi - lo) / 2:.3f}")
assert abs((lo + hi) / 2 - 0.584) < 1e-3 and abs((hi - lo) / 2 - 0.279) < 1e-3    # 上面的手算
assert abs(lo - 0.306) < 1e-3 and abs(hi - 0.863) < 1e-3
lo, hi = wilson(44, 81)
assert abs(lo - 0.435) < 1e-3 and abs(hi - 0.647) < 1e-3
width = lambda k, n: wilson(k, n)[1] - wilson(k, n)[0]
assert 0.30 < width(440, 810) / width(44, 81) < 0.34       # 样本多 10 倍，宽度约乘 1/√10 ≈ 0.316
```

样本多 10 倍，区间才缩窄到原来的三分之一左右（宽度大约按 $1/\sqrt{n}$ 缩小）。所以第 0 步的评估要么多做些题，要么把区间一起报出来。

那么，训练之后再测一次，怎样判断提升是真的呢？两次都用同一组题，两次都对或者都错的题不提供信息，要看的是结果变了的题。假设有 21 道题的结果变了：13 道是训练后才做对的，8 道反而从对变成了错。如果训练其实没有效果，每道题往哪边变就像抛一次硬币；抛 21 次，出现 13 比 8 或者更悬殊的结果，概率是 0.38。这就是**符号检验**（sign test）。13 比 8 还说明不了训练有效，要到 16 比 5，p 值才低于 0.05：

```verify id=Y.4-b title="结果变了的 21 道题：13 比 8 的双侧 p = 0.38；16 比 5 才低于 0.05"
from math import comb

def sign_test(better, worse):            # 双侧符号检验：只看结果变了的题
    n, k = better + worse, max(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

for better, worse in ((13, 8), (15, 6), (16, 5)):
    print(f"变好 {better}、变差 {worse}：双侧 p = {sign_test(better, worse):.3f}")
assert abs(sign_test(13, 8) - 0.383) < 1e-3
assert sign_test(15, 6) > 0.05 > sign_test(16, 5)
```

带走一句话：**说“probe”时说清是哪一种；训练前的“第 0 步”评估要带上置信区间，前后对比要看结果变了的那些题。**
:::

<div id="y4-epoch"></div>

### epoch

**epoch** 指把数据完整过一遍，但“一遍”可以有不同的数法。第 3 章的 CodeBot 预训练常被说成“过了 61 个 epoch”：20,000 步 × 32 个窗口 × 256 个 token，约是语料 2,677,398 个 token 的 61 倍。可是从 `DataLoader` 的角度看，数据集里的每个样本是一个起点不同的窗口，共有 2,677,142 个，20,000 步只取了其中 640,000 个，连一遍都没有过完。两种说法都对，只是“一遍”的单位不同：一个按 token 算，一个按窗口算。

微调的数据少，epoch 就多了起来。数的时候要以 `DataLoader` 实际给出的 batch 为准：SFT 的 1,092 条样本每轮分成 35 个 batch，最后一个只有 4 条，所以 500 步是 14.3 轮，而不是粗算的 500 × 32 ÷ 1,092 ≈ 14.7 轮。

```verify id=Y.4-c title="同一次预训练：按 token 是 61.2 个 epoch，按 DataLoader 的样本只有 0.24 个；SFT 的 500 步是 14.3 轮"
import math

tokens, ctx, iters, batch = 2_677_398, 256, 20_000, 32          # ch03/01_pretrain.py
windows = tokens - ctx                                            # len(TokenDataset)
print(f"按 token 算：{iters * batch * ctx:,} ÷ {tokens:,} = {iters * batch * ctx / tokens:.1f} 个 epoch")
print(f"按窗口算：{iters * batch:,} ÷ {windows:,} = {iters * batch / windows:.3f} 个 epoch")
assert round(iters * batch * ctx / tokens, 1) == 61.2 and round(iters * batch / windows, 3) == 0.239

for name, n, bs, steps in (("SFT", 1_092, 32, 500), ("DPO", 144, 8, 1_000)):    # ch03/03_sft.py、ch06/08_dpo.py
    per_pass = math.ceil(n / bs)                  # DataLoader 默认保留最后一个不满的 batch
    print(f"{name}：每轮 {per_pass} 个 batch，{steps:,} 步 = {steps / per_pass:.1f} 轮"
          f"（粗算 {steps:,} × {bs} ÷ {n:,} = {steps * bs / n:.1f}）")
assert round(500 / math.ceil(1092 / 32), 1) == 14.3 and round(1000 / math.ceil(144 / 8), 1) == 55.6
```

强化学习里还有第三种用法：CS336 2025 年 A5 的 `epochs_per_rollout_batch` 指“同一批 rollout 拿来更新几遍”，相当于 DLFS6 的 `n_update_per_generation = 2`。

```localcheck verify/test_reference.py::test_glossary_epoch_two_ways title="从原书脚本读出超参数：预训练的两种 epoch 是 61.2 与 0.239；用真实的 DataLoader 数，SFT 14.3 轮、DPO 55.6 轮"
```

<div id="y4-step"></div>

### step

**step** 至少有四种意思。在 DLFS6 的训练循环里，一次 **iteration** 通常就是一次参数更新（一个 optimizer step）。但有了梯度累积，一次更新要先跑好几个 **micro-step**：WebBot 每次更新累积 4 个 micro-batch，每张卡每步处理 32 × 4 × 1,024 = 131,072 个 token。GRPO 的一次 iteration（本书叫“一轮”）包含一次生成和两次更新，也就是两个 optimizer step。生成时的一个 **decode step**，则是让模型多吐出一个 token 的那一次前向计算。

```verify id=Y.4-d title="一个 step 里有多少东西：WebBot 每次更新 131,072 个 token；GRPO 每轮最多 32 道题 × 8 = 256 条回答、2 次更新"
micro_batch, accum, ctx = 32, 4, 1024                      # ch09/01_pretrain.py
print(f"WebBot：每次更新 = {accum} 个 micro-step，每张卡 {micro_batch * accum * ctx:,} 个 token")
prompts, G, updates = 32, 8, 2                            # ch03/09_grpo.py
print(f"GRPO：每次迭代最多 {prompts} 道题 × {G} 个回答 = {prompts * G} 条 rollout，更新 {updates} 次"
      "（81 道题分成 32、32、17，见下面的 batch）")
assert micro_batch * accum * ctx == 131_072 and prompts * G == 256
```

<div id="y4-context"></div>

### context

**context** 最常见的意思是**上下文长度**：模型一次最多能看见多少个 token，CodeBot 和 StoryBot 是 256，WebBot 是 1,024。这是一次前向计算的上限。可是在聊天和智能体的场合，“上下文”还常指整段对话历史，以及“这次任务一共消耗了多少 token”。两者经常被混为一谈。

窗口限制的是**单次请求**，消耗算的是**所有请求之和**。智能体每一轮都把完整的历史重新发一遍，单次请求的长度随轮数线性增长，累计消耗却按平方增长。举个例子：系统提示 3,000 个 token，每轮新增 800 个，跑 300 轮：

```verify id=Y.4-e title="单次请求最多 243,000 个 token，没超过 256K 的窗口；300 轮累计却有 3,700 万"
rounds = np.arange(1, 301)
request = 3_000 + 800 * rounds                  # 第 k 轮发出的请求长度
print(f"单次请求的最大长度：{request.max():,}")
print(f"300 轮的累计输入：{request.sum():,}")
assert request.max() == 243_000 <= 256_000 < request.sum() == 37_020_000
```

DLFS6 的 `generate()` 也有类似的区分：生成的总长度可以超过 256，但每一步只看最近的 256 个 token（[5.5 节](ch05.html#s5-5)；StoryBot 的 KV cache 在这里有一个问题，见[勘误 E04](errata.html#e04)）。

<div id="y4-batch"></div>

### batch

**batch** 听上去是一个数，可是先要问清楚它数的是什么。预训练里，一次参数更新背后有三层：每张卡一次前向算多少条序列（micro-batch），每张卡累积几次，一共几张卡。三者之积才是**全局 batch**（global batch size，常缩写成 gbs）。WebBot 在 8 张卡上是 32 × 4 × 8 = 1,024 条序列，约 105 万个 token（推导见 [9.3 节](ch09.html#s9-3)的[验证 9.3-e](ch09.html#v-9.3-e)）。原书记进 wandb 的 `effective_batch_size` 只乘了前两项，不论几张卡都是 128（[勘误 E45](errata.html#e45)）。

卡数来自 `torchrun` 给每个进程设的环境变量：`WORLD_SIZE` 是进程总数，`RANK` 是全局编号（0 到 `WORLD_SIZE − 1`），`LOCAL_RANK` 是本机内的编号，用来选 GPU。原书用 `LOCAL_RANK == 0` 判断谁来记日志、存模型，多台机器时每台都有一个这样的进程（[验证 9.3-c](ch09.html#v-9.3-c)）。

强化学习还要再多数一层：每一轮取几道题，每道题采几个回答。两者之积才是这一轮要打分、要更新的回答条数，也就是 rollout 的条数。CS336 的讲义特意提醒，它的 `rollout_batch_size` 和 `train_batch_size` 数的是回答，不是题。三家的名字对照如下：

| 数的是什么 | DLFS6 `09_grpo.py` | CS336 A5（2026） | veRL |
|---|---|---|---|
| 每一轮取几道题 | `batch_size` = 32 | `rollout_batch_size` ÷ `group_size` = 32 | `data.train_batch_size` |
| 每道题采几个回答 | `group_size` = 8 | `group_size` = 8 | `actor_rollout_ref.rollout.n` |
| 这一轮采样、打分的回答条数 | 32 × 8 = 256；81 道题按 32、32、17 分批，每三轮有一轮只有 17 × 8 = 136 条 | `rollout_batch_size` = 256 | 题数 × `rollout.n` |
| 一次参数更新用多少条回答 | 全部，同一批数据更新 2 次（`n_update_per_generation`） | `train_batch_size` = 256，分 32 次累积（`gradient_accumulation_steps`）；离策略的实验每次只用 8 条 | `ppo_mini_batch_size` × `rollout.n` |

```verify id=Y.4-f title="GRPO 的一批：81 道题分成 32、32、17，即 256、256、136 条回答；CS336 与 veRL 都按“题数 × 每题的回答数”来数" files=data/glossary/sources.json
sizes = [min(32, 81 - i) for i in range(0, 81, 32)]       # DataLoader(81 道题, batch_size=32) 保留最后不满的一批
rollouts = [s * 8 for s in sizes]                          # group_size = 8
print(f"DLFS6：每轮 {sizes} 道题 → {rollouts} 条回答")
a5 = src["cs336_a5_2026"]["batch_excerpt"]                 # src 来自验证 Y.2-a
for l in a5[:6]:
    print("  A5 讲义：", l)
verl = " ".join(src["verl"]["lines"]["verl/trainer/ppo/ray_trainer.py"])
assert sizes == [32, 32, 17] and rollouts == [256, 256, 136]
assert "rollout_batch_size = n_prompts_per_rollout_batch * group_size." in a5
assert "rollout_batch_size = train_batch_size = 256 means 32 prompts with 8 rollouts each." in a5
assert "gradient_accumulation_steps = 32" in a5 and "train_batch_size = 8" in a5
assert "self.config.data.train_batch_size" in verl and "batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n" in verl
assert "ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n" in verl
```

读一个陌生的配置时，先把这几层对上号，再去比较数字：同样写着 256，可能是 256 道题，也可能是 256 条回答。

## 小结

名字的问题，说到底是“同一个东西，不同的叫法”和“同一个叫法，不同的东西”。前者靠对照表解决：遇到陌生的参数名，先找它在 DLFS6 里叫什么，再看它出现在哪些矩阵的形状里；后者靠追问一句“这里指的是哪一种”。判断一个参数到底是什么，最可靠的办法往往是算一算参数量：残差流的宽度会以平方的形式出现。

正文里遇到不认识的词或参数名，可以随时回来查；想把本书和 CS336 对照着读，去看 [CS336 对照总表](cs336.html)。

本章所学的内容：

- Hugging Face 的 `GPT2Config` 把 `hidden_size` 直接映射到 `n_embd`：它是残差流的宽度，不是 FFN 中间层（[验证 Y.2-a](#v-Y.2-a)）。
- 一层 Transformer 的参数量约为 $12d^2$，所以平方出现的那个数就是残差流宽度（[验证 Y.2-b](#v-Y.2-b)）。
- veRL、tinker-cookbook 与 CS336 A5 的指标名都能在源码或讲义原文里找到，名字不同，看的东西相通（[验证 Y.3-a](#v-Y.3-a)）。
- `critic/rewards/mean` 是按回答平均的奖励；`critic/advantages/mean` 按 token 平均，各回答不一样长时一般不是 0，也不是学习信号（[验证 Y.3-b](#v-Y.3-b)）。
- 8 道题对 5 道的 95% 区间是 [0.31, 0.86]；前后对比时 13 比 8 的符号检验 p = 0.38：小样本的评估要带上区间和检验（[验证 Y.4-a](#v-Y.4-a)、[验证 Y.4-b](#v-Y.4-b)）。
- “一遍”和“一批”都要问数的是什么：同一次预训练，按 token 是 61.2 个 epoch，按样本只有 0.24 个；强化学习的 batch 是“题数 × 每题的回答数”，DLFS6 的 81 道题分成 256、256、136 条回答（[验证 Y.4-c](#v-Y.4-c)、[验证 Y.4-f](#v-Y.4-f)）。
- 上下文窗口限制单次请求，累计消耗随轮数按平方增长（[验证 Y.4-e](#v-Y.4-e)）。

名字对齐了，还剩下原书代码本身。下一页把我们在原书代码里找到的问题逐条列出来，每一条都配有能重跑的复现。
