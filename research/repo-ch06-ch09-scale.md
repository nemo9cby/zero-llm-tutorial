# DLFS6 ch06–ch09 dossier：训练技巧、StoryBot 预训练、生成与评测、DPO、WebBot 规模化

> 范围：`ch06/*.py`（含 `lr_graph.py`）、`ch09/*.py`、`storybot/utils.py`、`webbot/utils.py`、`storybot/tiny_stories_dpo.json`、`storybot/download.sh`、`webbot/download.sh`。为了算参数量和解释生成/DPO，也通读了 `storybot/model.py`、`webbot/model.py`、`storybot/tokenizer.py`、`codebot/{model,utils}.py`、`ch03/01_pretrain.py`、`ch05/0{4,5}_*.py`、`ch07/*.py`。
>
> 运行环境：Windows 11，i7‑11700F（8C/16T，AVX‑512，**没有原生 BF16 指令**），47.7 GB RAM，Python 3.12.9，torch 2.9.1+cpu，numpy 1.26.4，wandb 0.30.0（没装 openai、tiktoken）。整个过程中其他 agent 也在用 CPU（负载 57–100%），所以下面的墙钟时间都是“有竞争时”的上限。
>
> 所有验证脚本和打过补丁的副本都在 scratch：`C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch06-ch09-scale\`（下文记作 `$SCR`）。仓库里的文件一个都没改。按 download.sh 的清单，下载放进了 `storybot/` 和 `webbot/`（见 §4.2）。

---

## 0. 速览（先看这一节）

**已经确认的硬伤（教程必须提前处理）**

1. **HF 上的 StoryBot checkpoint 与仓库代码不兼容。** `model_pretrain.pt`、`model_iter_500.pt`、`model_iter_5000.pt` 里 SwiGLU 的 key 叫 `blocks.N.mlp.*`，而仓库的 `storybot/model.py` 把这个子模块叫 `self.ffn`。所以 `06_generate.py`、`07`、`08`、`09` 拿下载来的模型直接跑，都会在 `load_state_dict` 报 `Missing key(s) "blocks.0.ffn.W.weight"… Unexpected key(s) "blocks.0.mlp.W.weight"…`。修法是加载时把 `.mlp.` 改名成 `.ffn.`（`$SCR/run_keyfix.py` 就这样做，原脚本一字不改）。
2. **`storybot/model_dpo.pt`（download.sh 里列了）和 `webbot/model_pretrain.pt` 在 HF 上都是 404。** DPO 模型和 WebBot 模型都只能自己训练。
3. **代码里的学习率调度是“线性 warmup + 线性衰减到 0”（D2Z），不是 cosine。** `lr_graph.py` 只是画了 cosine 和 D2Z 的对比图。
4. **ch09 在 Windows 上没法原样跑。** `torch.compile` 报 `InductorError: Compiler: cl is not found`；没有 NCCL；torch 2.9.1+cpu 的 Windows wheel 不带 libuv，torchrun 起不来，gloo 也建不了 device。DDP 只能在 Linux、WSL2 或云上跑。
5. **StoryBot 生成超过 256 个 token 时 RoPE 位置会“卡住”。** KV cache 不截断，位置被钳在 255。拿真实模型对比“带 cache 生成”和“滑窗重算”，两者从第 273 个位置开始分叉。TinyStories 里有 11% 的故事超过 256 token，所以这不是罕见情况。

**关键数字（全部实测或由实测推出）**

| 项目 | StoryBot（ch06） | WebBot（ch09） |
|---|---|---|
| 配置 | V=10000, ctx=256, d=512, H=16, L=4, ff=1344, RoPE θ=1e4，不共享 embedding | V=50000, ctx=1024, d=768, H=12, **KV=4（GQA）**, L=12, ff=2048 |
| 参数量 | **22,696,448**（embedding 5.12M + unembed 5.12M + blocks 12,455,936 + 512） | **152,316,672**（embedding 和 unembed 各 38.4M，占 50.4%） |
| 训练 token 数 | 40000×32×256 = **327,680,000**（和 CS336 A1 TinyStories 一模一样） | 100000×32×4×1024 = **13,107,200,000 / 每个 rank** |
| D/N | 14.4 | 86（单卡） |
| 训练 FLOPs（6·N_mm·D） | 3.46e16（全量验证每次 +45%，合计约 5.2e16） | 8.96e18（+attention 8%，+eval 17%，合计约 1.13e19） |
| 数据 | train.bin 541.2M token（1.08 GB），valid 5.47M | train.bin 2.649B token（5.30 GB），valid 64.5M |
| 估计显存 | 约 2.1 GiB（batch 32） | micro‑batch 32 约 30 GiB；mb 16 约 16 GiB；mb 4 约 6 GiB |
| 估计时间 | H100 约 10–20 min，A100 约 15–30 min，3060 Ti 约 1–2 h | 1×H100 约 8–10 h；1×A100 约 25–35 h；3060 Ti 不现实（约 10+ 天） |
| 发布 checkpoint 在 held‑out 上的 loss | iter 0: 9.318 → 500: **2.416** → 5000: **1.652** → 40000: **1.302**（ppl 3.68） | 无（404） |
| 本次 CPU 实测的训练 | 冒烟（全尺寸，40 步）：val 9.314→5.467，329 s；迷你版（3.4M，4.9M token）：val 9.237→**2.651**，28 min；**DPO 全程 1000 步**：loss 0.693→**0.0075**，reward acc 1.0（训练集和 held‑out 都是），LM loss 1.302→1.315，68 min | 缩小版（13.2M，30 步）：val 10.835→9.646，166 s |

---

## 1. 推断的章节叙事

### 1.1 编号与缺口

| 章 | 有代码的编号 | 缺口 | 推断 |
|---|---|---|---|
| ch06 | 02,03,04,05,06,07,08,09 + `lr_graph.py`（画图脚本，不编号） | **01** | 6.1 应该是纯文字：介绍 StoryBot 和 TinyStories（数据来自 ch04 的 10K BPE 与 `.bin`，模型沿用 ch05 的 RoPE/SwiGLU/RMSNorm/KV‑cache），并说明“模型和数据都变大以后需要哪些训练技巧”。 |
| ch07 | 01_bpe_train, 02_bpe_encode | 无 | 用 `storybot/tokenizer.py` 里 ch04 写好的并行 `train_bpe`/`encode_file`，在 OWT（11.9 GB）上训练 50K 词表，产出 `webbot/merge_rules.pkl` 和 `owt_*.bin`。 |
| ch08 | **整章没有代码** | — | 见 §1.3。 |
| ch09 | 01_pretrain, 02_generate | 无 | WebBot：DDP、梯度累积、`torch.compile`、wandb、GQA + SDPA(Flash)，在 OWT 上预训练 152M 模型。 |

（范围外附带记一笔：ch02 缺 04/05，ch03 缺 05–08 且有两个 03，ch04 缺 05。说明斋藤的写法是“文字小节不配脚本”。）

### 1.2 ch06 的推断目录

| 小节（推断） | 脚本 | 引入的概念 |
|---|---|---|
| 6.1 StoryBot 与 TinyStories | （无） | 数据规模、10K 词表、沿用 ch05 的模型 |
| 6.2 优化器：SGD → Adam → AdamW | `02_adamw.py` | 继承 `torch.optim.Optimizer`；一阶、二阶矩；偏差修正；**decoupled weight decay**；最后换成 `torch.optim.AdamW` |
| 6.3 学习率调度 | `03_lr_scheduler.py`, `lr_graph.py` | warmup；cosine 与 D2Z（linear decay‑to‑zero）的对比图；书中代码采用 D2Z |
| 6.4 混合精度 | `04_mixed_precision.py` | FP32/FP16/BF16 的位宽与范围；FP16 的上溢、下溢和吞掉小数；BF16 的范围优势；`torch.autocast` |
| 6.5 预训练 StoryBot | `05_pretrain.py` | memmap 随机取 batch；bf16 autocast；梯度裁剪（`clip_grad_norm_`，没有单独脚本，应该在正文里讲）；周期性在全验证集上评估；按迭代保存 |
| 6.6 文本生成 | `06_generate.py` + `storybot/utils.py` | 带 KV cache 的 `generate`；temperature；`<|endoftext|>` 作起点和终止 |
| 6.7 用 LLM 评测 | `07_llm_judge.py` | LLM‑as‑a‑Judge 绝对打分（coherence 和 grammar 各 1–5 分），比较 iter 500/5000/40000 |
| 6.8 偏好学习：DPO | `08_dpo.py` + `tiny_stories_dpo.json` | chosen=happy ending，rejected=sad ending；参考模型；β；只对回答部分计 log‑prob |
| 6.9 评测 DPO 效果 | `09_llm_judge.py` | pairwise judge，随机交换 A/B 防位置偏差，计算 win rate（不计平局） |

**斋藤式的组织规律（从代码可见）。** 每个概念先给一个几行就能跑的小实验（04 打印三个“出乎意料”的数；02 用 `Linear(2,1)` 跑 5 步），再把它塞进完整的训练脚本。“先自己实现，再换成框架”：02 写了自制的 SGD 和 AdamW，但最后一行用的是 `torch.optim.AdamW`，自制版被注释掉了。三个 bot 是一条主线：CodeBot（ch01–03）→ StoryBot（ch04–06）→ WebBot（ch07–09），每一级都多一个“规模化”的问题。

### 1.3 ch08（无代码）应该讲了什么

ch09 用到了不少前文从没出现过的东西，代码注释里也几乎不解释，所以它们很可能在 ch08 用文字讲过：

1. **为什么要变大**：scaling law、计算量估算 6ND、token/参数比（Chinchilla）。
2. **GPU 与显存**：HBM 和 SRAM 的层次、显存账本（参数、梯度、Adam 状态共 16 B/param，加上激活）。
3. **FlashAttention**：`F.scaled_dot_product_attention`，是 IO‑aware 的融合 kernel。
4. **GQA/MQA**：`n_kv_head`，KV cache 显存减为 1/3（WebBot 每 token 6144 个值，同规模 MHA 是 18432）。
5. **`torch.compile`**：kernel fusion，以及图捕获的限制。
6. **数据并行**：all‑reduce、`DistributedDataParallel`、`torchrun` 和 RANK/LOCAL_RANK、NCCL、`no_sync`。
7. **梯度累积**：effective batch = micro_batch × accum × world。
8. **实验管理**：Weights & Biases。
9. （可能）云 GPU 环境、checkpoint、长时间任务管理。

这些是推断，教程可以把这部分当“缺失章节”来补写，CS336 A2 和第 2/5/7/8/10 讲正好对应（见 §7）。

---

## 2. 逐脚本详解

### 2.1 `ch06/02_adamw.py`：自制 SGD 与 AdamW

**目的**：演示怎样继承 `torch.optim.Optimizer` 写优化器，并实现 AdamW。最后用的是 `torch.optim.AdamW`（自制版那一行被注释掉）。

**签名**
- `class SGD(Optimizer): __init__(self, params, lr=0.01)`；`step(self)`
- `class AdamW(Optimizer): __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`；`step(self)`。这些默认值和 `torch.optim.AdamW` 一样。

**核心代码**
```python
state['t'] += 1; t = state['t']
m = beta1 * m + (1 - beta1) * grad              # 一阶矩
v = beta2 * v + (1 - beta2) * grad**2           # 二阶矩
m_hat = m / (1 - beta1**t); v_hat = v / (1 - beta2**t)   # 偏差修正
p.data = p.data - lr * m_hat / (v_hat.sqrt() + eps) - lr * wd * p.data   # 更新 + decoupled weight decay
```
**日文注释翻译**：1次モーメントと2次モーメントの更新 = 更新一阶矩和二阶矩；バイアス補正 = 偏差修正；パラメータ更新 = 参数更新；シンプルな線形モデル = 简单线性模型；ダミーデータで学習 = 用假数据训练；数ステップ学習してlossの減少を確認 = 训练几步，确认 loss 在降。

**实际输出**（原样运行，墙钟 5.2 s，其中大部分是 import torch）
```
Step 0: loss = 8.0011
Step 1: loss = 5.8991
Step 2: loss = 4.1294
Step 3: loss = 2.6933
Step 4: loss = 1.5855
```
**等价性验证**（`$SCR/v_adamw.py`：只 exec 文件里 `torch.manual_seed(0)` 之前的类定义，保证测的就是书里的代码）
```
[A] Linear(2,1), lr=0.1, wd=0.01, 5 steps: mine 8.001141/5.899070/4.129449/2.693277/1.585504
                                          torch 8.001141/5.899070/4.129449/2.693277/1.585503
    max |param diff| = 5.96e-08
[B] Linear(32,8), lr=1e-3, wd=0.1, 500 steps: max|dparam| = 1.580e-06 (default/foreach=False/foreach=True 都一样)
    Adam(weight_decay=0.1)（耦合 L2）vs AdamW: max|dparam| = 2.123e-01   ← 不同的算法
[C] NumPy AdamW (float64) vs torch AdamW (float64), 100 steps: max diff = 2.22e-16
[D] t=1 时 update/lr = [0.990099, -1, 1]（g = 1e-6, -3, 50）→ 第一步约等于 lr·sign(g)
```
**要点**：torch 的实现是先 `p *= (1 - lr·wd)` 再做 Adam 更新，书里是一次写完 `- lr·wd·p_old`，两者在代数上完全相同，差异只来自 float32 运算顺序。

### 2.2 `ch06/03_lr_scheduler.py`：warmup + 线性衰减到 0（D2Z）

```python
def get_lr(it, max_lr, warmup_iters, max_iters):
    if it < warmup_iters:                       # ウォームアップ：0 -> max_lr（预热）
        return max_lr * (it / warmup_iters)
    if it < max_iters:                          # アニーリング：max_lr -> 0（退火）
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)
    return 0.0
```
原样运行没有输出（只定义函数），用时 0.15 s。`$SCR/v_lr.py` 检查了几件事：去掉注释后，这个函数和 `05_pretrain.py`、`ch09/01_pretrain.py` 里的 `get_lr` **逐字相同**（True）。ch09 的注释写的是“線形減衰”（线性衰减），比 ch06 写的“アニーリング”更准确。

```
StoryBot (1e-3, 200, 40000): get_lr(0)=0, (1)=5e-06, (100)=5e-4, (199)=9.95e-4, (200)=1e-3,
  (201)=9.99975e-4, (20100)=5e-4, (39999)=2.5126e-08, (40000)=0
  全程平均 lr = 5e-4 = max_lr/2
WebBot (6e-4, 500, 100000): (0)=0, (250)=3e-4, (500)=6e-4, (50250)=3e-4, (99999)=6.03e-09
```

### 2.3 `ch06/lr_graph.py`：cosine 与 D2Z 对比图

- `cosine_annealing(t, warmup_ratio=0.05, eta_min_ratio=0.1)`：warmup 占全程 5%，之后 `eta_min + 0.5(1-eta_min)(1+cos(π·progress))`，最低降到峰值的 10%。
- `d2z(t, warmup_ratio)`：warmup 之后线性降到 0。
- 注释翻译：日本語フォント設定 = 日文字体设置（`'Hiragino Sans'` 只有 macOS 有，Windows 那行被注释了）；ウォームアップ期間（全体の5%）= 预热占全程 5%；コサインアニーリングの最小学習率（最大の10%）= 余弦退火的最低学习率为峰值的 10%；ウォームアップ領域をグレーで塗りつぶし = 把预热区涂成灰色；学習の進行度 = 训练进度；学習率 = 学习率。
- **实测**：在 Windows 上 1.2 s 跑完，但会刷出 `UserWarning: Glyph … missing from font(s) DejaVu Sans`，图里的日文全是方块（`$SCR/lr_schedule.png` 已目检）。
- 数值（`$SCR/v_lr.py`）：

| t | 0.025 | 0.05 | 0.1 | 0.525 | 0.9 | 0.999 | 1.0 |
|---|---|---|---|---|---|---|---|
| cosine | 0.5 | 1.0 | 0.9939 | 0.55 | 0.1244 | 0.1000 | 0.1 |
| D2Z | 0.5 | 1.0 | 0.9474 | 0.5 | 0.1053 | 0.0011 | 0.0 |

  曲线下面积（平均 lr/峰值）：cosine 0.5475，D2Z 0.5000。

### 2.4 `ch06/04_mixed_precision.py`：数值格式与 autocast

**实际输出**（原样运行，3.4 s）
```
----- FP16 -----
tensor(1000., dtype=torch.float16)        # 1000 + 0.01：fp16 在 1000 附近的间距是 0.5，0.01 被吞掉
tensor(0., dtype=torch.float16)           # 1e-8 下溢（最小次正规数 5.96e-8）
tensor(inf, dtype=torch.float16)          # 70000 上溢（最大 65504）
----- BF16 -----
tensor(0., dtype=torch.float16)
tensor(1.0012e-08, dtype=torch.bfloat16)  # bf16 指数位和 fp32 一样
tensor(inf, dtype=torch.float16)
tensor(70144., dtype=torch.bfloat16)      # 能表示，但 70000 被舍入成 70144（只有 8 位有效位）
----- 自動混合精度 -----
torch.bfloat16                            # a @ a
torch.float32                             # a.sum()
```
注释翻译：FP16ではアンダーフロー = FP16 下溢；BF16では表現可能 = BF16 能表示；FP16ではオーバーフロー = FP16 上溢；自動混合精度 = 自动混合精度；行列積はBF16 = 矩阵乘用 BF16；累積はFP32 = 累加用 FP32。

**补充验证**（`$SCR/v_precision.py`）
- `torch.finfo`：fp16 max 65504，eps 9.77e-4，最小正规数 6.10e-5，最小次正规数 5.96e-8；bf16 max 3.39e38，eps 7.81e-3，最小正规数 1.18e-38，最小次正规数 9.18e-41。
- **用 NumPy 模拟 bf16**（对 fp32 的高 16 位做 round‑to‑nearest‑even），在横跨 26 个数量级的 20 万个随机数上和 torch 的 bf16 **完全一致**（True），所以可以在 Pyodide 里演示。
- fp16 下的梯度下溢：3e-8 存成 fp16 变成 **6e-8**（被舍入到最小次正规数，误差 2 倍）。先乘 2^16 再存，还原后是 3.0006e-8。这就是 GradScaler 存在的理由。bf16 直接存是 3.0035e-8，不需要 scaler。
- **autocast(cpu, bf16) 在 StoryBot 里的 dtype**（forward hook 实测）：embedding 输出 fp32；RMSNorm 输入输出都是 fp32；`W_q` 等 Linear 输出 bf16；MHA 和 SwiGLU 输出 bf16；**block 输出（残差流）是 fp32**，因为 fp32 加 bf16 会提升成 fp32；logits 是 bf16；loss 是 fp32；参数和梯度都是 fp32。
- **CPU 和 CUDA 的 autocast 策略不同**：CPU 上 `softmax(bf16)` 得 bf16，`bf16.sum()` 得 bf16，`fp32 @ fp32` 得 bf16，`fp32.sum()` 得 fp32。也就是说，书里“累積はFP32”在 CPU 上成立，只是因为 `a` 本来就是 fp32。按 PyTorch 文档的 CUDA autocast 列表，`sum` 和 `softmax` 会自动升到 fp32（**这条没在 GPU 上验证，待 GPU 跑**）。
- **速度**：在这台 CPU 上，StoryBot 全尺寸、batch 8×256 做一次 fwd+bwd，fp32 用 **1.58 s/step**，bf16 autocast 用 **3.69 s/step**，**bf16 反而慢 2.3 倍**（没有 AVX512‑BF16 或 AMX）。所以 CPU 小模型实验应该关掉 autocast。
- 整个 repo 都**没有 GradScaler、没有 fp16 训练**，只用 bf16。

### 2.5 `ch06/05_pretrain.py`：StoryBot 预训练

**签名**
- `get_lr(it, max_lr, warmup_iters, max_iters)`：同 03。
- `get_batch(data, context_len, batch_size, device, random=True, offset=0)`：`random=True` 时 `ix = torch.randint(len(data)-context_len-1, (B,))`；`random=False` 时 `ix = arange(offset, offset+B*ctx, ctx)`，并用 `ix[ix+ctx+1 < len(data)]` 过滤，取空时返回 `(None, None)`。
- `evaluate(model, val_data, context_len, batch_size, device)`：用不重叠的窗口**走完整个验证集**，`reduction='sum'` 累加后除以 token 数。

**超参**：`context_len=256, vocab_size=10000, batch_size=32, learning_rate=0.001（max_lr）, warmup_iters=200, max_iters=40000, embed_dim=512, n_head=16, n_layer=4, ff_dim=1344, theta=10000, eval_iters=500, grad_clip=1.0, save_iters=[500, 5000]`。优化器是 `torch.optim.AdamW(model.parameters(), lr)`，即 betas (0.9, 0.999)、wd 0.01，**没有 param groups**，所以 RMSNorm 的 weight 和 embedding 也做 weight decay。

**核心训练步**
```python
lr = get_lr(i, learning_rate, warmup_iters, max_iters)
for param_group in optimizer.param_groups: param_group['lr'] = lr
batch_x, batch_y = get_batch(train_data, context_len, batch_size, device)
optimizer.zero_grad()
with autocast(device_type=device.type, dtype=torch.bfloat16):      # 順伝播と損失計算(Mixed Precision)
    logits = model(batch_x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
```
注释翻译：バッチを作成 = 组 batch；全データを順番に処理 = 按顺序处理全部数据；データをmemmapで読み込み = 用 memmap 读数据；学習率を更新 = 更新学习率；勾配をリセット = 清零梯度；特定のイテレーションでモデルを保存 = 在指定迭代保存模型；定期的に評価 = 定期评估；Validation lossのグラフを描画 = 画验证 loss 曲线。

**数据**：`storybot/tiny_stories_train.bin` 有 1,082,458,446 B，即 **541,229,223 个 uint16 token**，超过 200 MB 所以没下载；`tiny_stories_valid.bin` 有 10,931,746 B，即 **5,465,873 个 token**。327.68M 训练 token 约等于 train.bin 的 **0.61 个 epoch**（有放回随机采样）。

**冒烟运行（忠实缩小版）** `$SCR/p05_pretrain_smoke.py`（打了 12 处补丁，全部 assert 只匹配一次，见 `$SCR/make_patched.py`）：
- 模型**保持全尺寸**（22.7M），**保留 bf16 autocast**。
- 训练数据换成 `tiny_stories_valid.bin` 的前 4,919,634 个 token（在 EOS 处切分），验证用剩下部分的前 33,068 个 token。
- `batch_size 32→8`，`max_iters 40000→40`，`warmup 200→10`，`eval_iters 500→20`，`save_iters→[20]`，输出路径全部改到 scratch，并设 `torch.manual_seed(0)`。

实际输出（去掉 tqdm 重绘）：
```
パラメータ数: 22,696,448 (22.7M)
  0%|  0/40 [00:31<?, loss=9.3452, val_loss=9.313624]        ← 初始 val 约 9.31 > ln(10000)=9.2103
  ... loss: 9.2989, 8.9401, 8.5860, 8.2087, 7.9265, 7.6259, 7.3775, 7.0247, 6.7222, 6.2652（第 10 步）...
モデルを保存しました（イテレーション 20）: …\out\smoke_model_iter_20.pt
 50%| 20/40 [03:01, loss=6.0122, val_loss=5.802967]
 98%| 39/40 [05:14, loss=5.4917, val_loss=5.467001]
exit=0 wall=329s
```
每步约 5–6 s（batch 8，bf16，有 CPU 竞争）；每次 17 个 batch 的验证约 45 s。

**发布 checkpoint 的真实学习曲线**（`$SCR/v_valloss.py`：用书里的 `evaluate()`，从 05 exec 出来，autocast 换成 fp32 以加速；在 `valid.bin` 前 131,072 个 token 上算，这部分对这些 checkpoint 是 held‑out 的）
```
fresh init: logit std = 0.4492 -> 预测 CE ≈ ln V + var/2 = 9.3112
fresh init (no training)   val_loss=9.3182  ppl=11138.78  bits/byte=3.2654
iter 500                   val_loss=2.4164  ppl=   11.21  bits/byte=0.8468
iter 5000                  val_loss=1.6524  ppl=    5.22  bits/byte=0.5790
iter 40000 (final)         val_loss=1.3023  ppl=    3.68  bits/byte=0.4564
bf16 autocast vs fp32（final，4 个 batch）：1.31153 vs 1.31137（差 0.00016）
```
最终模型的 1.302 已经低于 CS336 A1 对同一配置设的目标 1.45（目标值见 `cs336-crossref.md`）。每个 checkpoint 评估约 60–90 s（有竞争）。

**CPU 迷你版 StoryBot**（`$SCR/p05_pretrain_mini.py`，同一份代码，补丁 16 处：d=128、H=4、ff=384、ctx=128、fp32（`autocast(..., enabled=False)`）、1200 iters、warmup 50、每 200 步评估、4 线程）：【结果见 §2.5.1】

#### 2.5.1 迷你版结果

`パラメータ数: 3,413,120 (3.4M)`（其中 2.56M 是 embedding 和 unembed）。1200 步 × 32 × 128 = 4,915,200 个 token，正好约等于训练切片（4.92M token）的 1 个 epoch。墙钟 **1689 s（28 min，4 线程，同时在跑 DPO 和其他 agent 的任务）**，约 1.2–1.5 s/it。验证集是 `valid.bin` 后 10% 里的 33,068 个 token：

| iter | 0 | 200 | 400 | 600 | 800 | 1000 | 1199 |
|---|---|---|---|---|---|---|---|
| val loss | 9.2372 | 3.5576 | 3.0917 | 2.8857 | 2.7739 | 2.6923 | **2.6512** |

采样（`$SCR/v_mini_samples.py`，prompt “Once upon a time”，T=0.8）：
> seed 0：Once upon a time, there was a little boy named Tim. Tim had a yummy tummy. It was was a very important place to play. One day, Tim came to the park with his mom. He saw him and wanted to play with it. So, he tried to catch the gate, but he could not. He was very hurt. …
> seed 1：Once upon a time, there was a little bunny named Baby. He had a gift to go on a trip. Tim loved to play with his friends, but he was all very happy. …

可以拿来作为教程的“对照组”：笔记本 CPU 半小时能学会 TinyStories 的句式和词汇（loss 2.65，ppl 约 14），但人物和情节前后不一致。全尺寸模型训练 327M token 以后 loss 降到 1.30（ppl 3.7），故事就连贯了。按 CS336 的低资源版本（40M token），目标是 ≤ 2.00（见 `cs336-crossref.md`），在 CPU 上大约需要把这次运行再延长 8 倍。

### 2.6 `ch06/06_generate.py` + `storybot/utils.py::generate`

**签名**：`generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0)`，带 `@torch.no_grad()`。

**核心逻辑**（带 KV cache 的版本，codebot 版没有 cache）
```python
model.eval(); model.clear_cache()
ids = torch.tensor([tokenizer.encode(prompt)]); generated_ids = ids; next_id = ids
for _ in range(max_new_tokens):
    if ids.size(1) > model.max_context_len:          # ← 用了 cache 以后这里是死代码：ids 根本不进模型
        ids = ids[:, -model.max_context_len:]
    logits = model(next_id, use_cache=True)[:, -1, :]  # 第一次是 prefill（整段 prompt），之后每次 1 个 token
    next_id = logits.argmax(-1, keepdim=True) if temperature == 0 else torch.multinomial(F.softmax(logits / temperature, -1), 1)
    if next_id.item() == tokenizer.end_token_id: break
    ids = torch.cat((ids, next_id), 1); generated_ids = torch.cat((generated_ids, next_id), 1)
generated_ids = generated_ids[generated_ids != tokenizer.end_token_id]   # 去掉 EOS（包括作为 prompt 的那个）
```
06 的设置：`prompt = "<|endoftext|>"`（id 9999，表示“从一个新故事开头写起”），`max_new_tokens=300`，`temperature=1.0`，`num_samples=3`。注释：生成の開始プロンプト = 起始提示；温度パラメータ（高いほどランダム）= 温度（越高越随机）。

**原样运行会报错**（key 名不匹配，见 §0‑1）。改用 `run_keyfix.py`（seed 0）后 10.2 s 跑完，输出节选：
```
--- サンプル 1 ---
One day, a little girl named Sue went to the zoo with her mom. Sue was very happy because she loved animals. …
--- サンプル 3 ---
Once upon a time, there was a grumpy alligator. He lived in a big pond with many fish. The alligator did not like to swim. …
From that day on, the alligator never had a friend who would just look down at him.
```
输出以 `\n` 开头：EOS 后面紧跟的第一个 token 常常是换行。

**KV cache 相关验证**（`$SCR/v_attn.py`）：
- 随机小模型上，40 个 greedy token 与“每步全量重算”完全一致（storybot 和 webbot 都是 True）。
- 真实模型，prompt 16 个 token、生成 320 个：带 cache 用 **5.7 s**，滑窗重算用 **26.0 s**（快 4.6 倍）。两者**第一次分叉在第 273 个位置**。生成结束时 `k_cache` 长度是 335，超过 256。
- `RoPE(x, offset=300)` 和 `offset=255` 的结果逐元素相同：位置被钳住了。
- 温度表（logits = [2,1,0,−1]）：T=0.1 → [1,0,0,0]；T=1 → [0.6439, 0.2369, 0.0871, 0.0321]；T=10 → [0.2887, 0.2612, 0.2363, 0.2138]；熵从 0 升到 1.380。

**在浏览器里跑 StoryBot 是可行的**（`$SCR/v_numpy_forward.py`）：纯 NumPy 前向（RMSNorm 的 eps 取 `finfo(fp32).eps`，交错式 RoPE，因果 MHA，SwiGLU）和 torch 的 logits 最大差 **8.6e‑6**，20 个 greedy token 完全相同（“…a little cat named Kitty. Kitty loved to play with her ball. …”）。CPython 下不带 cache 每 token 157 ms。权重存成 fp16 的压缩 npz 只有 **40.0 MiB**，logits 差 0.0045，argmax 不变。更进一步，**纯 JS 版**（`$SCR/v_js_forward.js`，带 KV cache，不需要 Pyodide）用 fp32 或 fp16 权重，greedy 都和 torch 在全部 51 个 token 上一致，Node 单线程约 58–74 ms/token（F54）。分词器同样可以用纯 JS 实现（F52、F53）。所以“在浏览器里训练好的模型上直接生成故事”这个端到端演示，只需要下载约 45 MB 的 fp16 权重和约 190 KB 的 merges。

### 2.7 `ch06/07_llm_judge.py`：LLM‑as‑a‑Judge（绝对分）

用 `generate` 从 iter 500、5000、40000 三个 checkpoint 各生成 10 个故事（prompt `<|endoftext|>`，200 token，T=1.0），再调用 `gpt-4o-mini`（`response_format={"type":"json_object"}`，`max_tokens=300`），最后打印每个维度的“均值 ± stdev”。

**Judge prompt 中文翻译**（原文是日文，被评的故事是英文）：
> 请从两个维度给下面这个儿童故事打 1–5 分。
> 故事：{story}
> 评分维度：1. Coherence（一致性）：逻辑是否连贯，作为故事是否讲得通；2. Grammar（语法）：英语语法是否正确。
> 请按以下 JSON 格式回答：{"coherence": <1-5 的整数>, "grammar": <1-5 的整数>, "comment": "<简短理由>"}

注释翻译：response_formatを使えば、パース処理がシンプルになる = 用了 response_format，解析就简单了。

**离线 dry‑run**（**没有调用任何 API**）：`$SCR/fakeopenai/openai/__init__.py` 是一个 mock 客户端，绝对分固定返回 `{"coherence":3,"grammar":4}`，只用来测通管线。`$SCR/p07_judge_mock.py` 把 `num_samples` 改成 2。实测 48.6 s 跑通，并抓到了实际发出的 prompt 全文。三个 checkpoint 都需要 key 修复。生成样本节选：
```
iter 500 : "One day, Tom and his sister, Sue, were playing a kite. They were very high and hurt. …"
iter 5000: "Once upon a time, there was an elderly lady. She lived in a big house with her cat. wheat was in a house. …"
iter 40000: "One day, a little girl named Lily found a flag. It was a pretty flag with a big red heart on it. …"
Summary 打印格式：  coherence: 3.00 ± 0.00 / grammar: 4.00 ± 0.00（mock 值）
```

### 2.8 `ch06/08_dpo.py`：DPO

**数据** `storybot/tiny_stories_dpo.json`：144 条 `{"prompt","chosen","rejected"}`，prompt 两两不同，都是 “Once upon a time, there was … . ” 的形式（**末尾带空格**）。chosen 是 happy ending，rejected 是 sad ending。字符数：prompt 53–83；chosen 160–280（均值 217.4）；rejected 114–207（均值 146.4）。第 0 条：
> prompt: "Once upon a time, there was a boy named Tim who lost his toy. "
> chosen: "Tim looked everywhere for his toy. He asked his mom for help. … They found it under the bed! Tim was so happy. …"
> rejected: "Tim looked for his toy but couldn't find it. He cried and cried. His toy was gone forever. …"

**签名与核心代码**
```python
class DPODataset(Dataset):  # __init__(data_path, tokenizer, context_len); _pad_and_mask(ids, prompt_len); _create_sample(prompt, chosen, rejected)
    mask = [0]*prompt_len + [1]*(len(ids)-prompt_len)       # 只有回答部分是 1
    ids = ids + [0]*pad_len; mask = mask + [0]*pad_len       # 右填充，pad id 0（字节 '\x00'）

def get_sequence_logprobs(model, ids, mask):
    log_probs = F.log_softmax(model(ids)[:, :-1, :], dim=-1)            # (B, C-1, V)
    per_token = torch.gather(log_probs, -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (per_token * mask[:, 1:]).sum(-1)                            # 只对回答 token 求和（不按长度归一化）

def compute_dpo_loss(model, ref_model, c_ids, c_mask, r_ids, r_mask, beta):
    logits = beta * ((lp(model,c) - lp(model,r)) - (lp(ref,c) - lp(ref,r)))   # ref 部分在 no_grad 下算
    return -F.logsigmoid(logits).mean()
```
超参：`context_len=256, batch_size=8, learning_rate=5e-6, beta=0.1, max_iters=1000`。`model` 和 `ref_model` 都从 `model_pretrain.pt` 加载，`ref_model.eval()`。数据迭代用 `cycle(DataLoader(shuffle=True))`。注释翻译：パディングとマスクの作成 = 构造 padding 和 mask；マスクを適用（応答部分のみ）= 只对回答部分应用 mask；現在のモデル／参照モデルのlog-prob = 当前模型／参考模型的 log‑prob。

**手算与等价性验证**（`$SCR/v_dpo.py`，函数直接从 08 exec 出来）
```
[1] β=0.1，logπ_c=-10，logπ_r=-12，logref_c=-11，logref_r=-11
    z = 0.1·((−10+12) − (−11+11)) = 0.2；手算 log(1+e^{-0.2}) = 0.598139；torch 0.598139
    ∂L/∂logπ_c = −0.045017 = −β·σ(−z)；∂L/∂logπ_r = +0.045017
[2] get_sequence_logprobs == 手写循环：[-6.729046, -12.019203] 两边相同
[3] model == ref_model 时 loss = 0.6931471824645996 = ln 2
[4] 真实分词：vocab 10000，EOS 9999，encode("Once upon a time") = [429, 438, 258, 397]
    prompt 15–21 token；prompt+chosen 49–81（均值 64.0）；prompt+rejected 40–64（均值 49.7）→ 256 内从不截断，128 也放得下
    sample0 mask[:20] = 17 个 0 加 3 个 1；sum(mask) = 53
    分开编码 ≠ 一起编码（70 vs 69 token）：分开是 [' toy','.',' ','Tim',' looked']，一起是 [' toy','.',' Tim',' looked']
[5] cycle(DataLoader(shuffle=True))：每个 epoch 18 个 batch；batch 0 == batch 18：True；batch 5 == batch 23：True
    → 只有第一个 epoch 是随机的，之后按同样顺序重放
    1000 × 8 / 144 = 55.6 个 epoch
```
`$SCR/v_dpo_mask.py`（policy 用 iter_5000，ref 用 final，真实数据 8 条）：
```
context_len=256: loss(只 mask 回答)=1.066696  loss(prompt+回答都计)=1.066695
context_len=128: 同上（右填充对 causal 模型没有影响）
```
可见“只对回答部分计 log‑prob”在数学上不影响 DPO loss：prompt 的 log‑prob 在 chosen 和 rejected 里一样，相减就抵消了。mask 真正有用的是去掉 padding。

**CPU 运行（缩小版）** `$SCR/p08_dpo_shrunk.py`：只改了 5 处。`context_len 256→128`（上面已验证不改变数值）；加了 `torch.manual_seed(0)` 和 `set_num_threads(4)`；保存路径改到 scratch；另外把 losses 写成 JSON。**其他保持原样（1000 iters、batch 8、lr 5e‑6、β 0.1）**。通过 `run_keyfix.py` 加载。结果：【见 §2.8.1】

#### 2.8.1 DPO 运行结果

**实测**：`exit=0 wall=4065s`（1000 步，68 min，4 线程，同时有 mini 训练和其他 agent 在抢 CPU，2.9–8 s/it）。tqdm 最后一行：`100%| 1000/1000 [1:07:31, 4.05s/it, loss=0.0075]`。模型存为 `$SCR/out/model_dpo_cpu.pt`（90.8 MB，key 名是 `ffn`），曲线图在 `$SCR/out/loss_dpo.png`。

| iter | 0 | 1 | 5 | 10 | 18（第 1 个 epoch 结束） | 36 | 50 | 100 | 200 | 300 | 500 | 750 | 999 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | **0.6931** | 0.6886 | 0.6673 | 0.6304 | 0.4855 | 0.3894 | 0.3622 | 0.2069 | 0.1016 | 0.0535 | 0.0154 | 0.0104 | **0.0075** |

最后 100 步的平均是 0.0066，最小值 0.0038。第 0 步正好是 ln 2，和 F28 一致。

**训练后分析**（`$SCR/v_dpo_post.py`，日志在 `$SCR/out/dpo_post.log`；参考模型就是 `model_pretrain.pt`）：

| 指标 | 144 条训练 pair | 6 条 held‑out pair（我另写的同风格样本，`$SCR/heldout_pairs.json`） |
|---|---|---|
| implicit reward accuracy（r_c > r_r） | **1.000** | **1.000** |
| 平均 reward margin β·Δ | 5.254 | 2.349 |
| DPO loss | 0.0059 | 0.1010 |
| 平均 logπ − logπ_ref，chosen | **+13.11 nats** | +0.72 nats |
| 平均 logπ − logπ_ref，rejected | **−39.44 nats** | −22.78 nats |
| 参考模型按“整句 log‑prob 之和”已经更偏好 chosen 的比例 | 6% | 50% |

几点解读：
- 在这份数据上**没有出现 likelihood displacement**（chosen 的 log‑prob 不降反升），margin 主要靠把 rejected 压下去：−39 nats 对 +13 nats。
- 只看“整句 log‑prob 之和”，参考模型在 94% 的训练 pair 上“更喜欢” rejected。原因只是 chosen 更长（求和会更负），这正是长度混淆。DPO 用的是相对参考模型的差，所以这部分抵消了。
- held‑out 上 margin 小一半，但准确率仍然是 100%。说明学到的“happy ending 偏好”能泛化到没见过的 prompt（只有 6 条，属于示意，不算统计证据）。
- **通用语言建模能力几乎没受影响**：131,072 个 held‑out TinyStories token 上，loss 从 1.3023 变成 1.3149（+0.0126，约 1% 的 alignment tax）。

**同 seed 采样对比**（prompt “Once upon a time”，T=1.0，120 token，seed 1/2/3）：pretrain 的 seed 2 是一个陌生人送棒棒糖的故事；DPO 的 seed 2 是 “Tim … wanted to be the best superhero … decided to help Sam find his toy … Sam was happy now. Tim felt good that he helped …”。DPO 的样本更多是“帮助、开心”的情节，但这只是轶事。系统的比较见 §2.9.1。

### 2.9 `ch06/09_llm_judge.py`：pairwise judge，比较 pretrain 与 DPO

两个模型用同一个 prompt “Once upon a time” 各生成一个故事（150 token，T=1.0），共比较 100 次。每次**用 `random.random()<0.5` 交换 A/B 顺序**（位置バイアスを避けるため = 为了避免位置偏差），然后问 judge 哪个更 happy ending。最后打印各自胜场和平局数，以及**不计平局的 DPO 胜率**。

**Judge prompt 中文翻译**：
> 请比较下面两个儿童故事，判断哪一个更像 happy ending。【Story A】{a}【Story B】{b}
> 请判断哪一个结局更明亮幸福，或者内容更充满希望。用 JSON 回答：{"winner": "A" or "B" or "tie", "reason": "简短理由"}

离线 dry‑run 用 `$SCR/p09_judge_mock.py`：DPO 模型路径指向我在 CPU 上训练的模型，比较 40 次。mock judge 是**粗糙的词表打分**：只看最后 40 个词，正面词数减负面词数，**明确标注不是 LLM 判断**。【结果见 §2.9.1】

#### 2.9.1 结果

`exit=0 wall=339s`（40 次比较，即 80 次生成，每次 150 token）。抓到的 judge prompt 和 §2.9 的译文一致。书里的汇总代码原样输出：
```
📊 PAIRWISE COMPARISON RESULTS
  Pretrain wins:   9 ( 22.5%)
  DPO wins:       23 ( 57.5%)
  Ties:            8 ( 20.0%)
  DPO win rate (excluding ties): 71.9%
```
23/32 的单侧二项检验 p=0.010（双侧 0.020），Wilson 95% CI 是 [0.546, 0.844]。**注意这只是词表 mock 的结果，不是 LLM 的判断**：20% 的平局来自“结尾 40 个词里一个情感词都没有”；而且 mock 是对称的，A/B 的随机交换对它没有影响。要得到书中那种结论，需要真的调用 judge API（或者用本地开源模型当 judge）。这次 dry‑run 证明的是：整条流水线（两个模型生成、随机交换、解析、计数、胜率）在离线环境下能跑通。

另外，09 用 `random.random()` 做交换，没设 seed（torch 的 seed 管不到 Python 的 `random`），所以每次运行的 A/B 顺序都不一样；judge 的结果本身也有随机性。教程里应该固定 seed，并报告置信区间。

### 2.10 `ch09/01_pretrain.py`：WebBot（DDP + 梯度累积 + compile + wandb）

**结构**
```python
ddp = int(os.environ.get('RANK', -1)) != -1                         # 由 torchrun 设置
if ddp:
    dist.init_process_group(backend='nccl'); local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}'); torch.cuda.set_device(device); seed_offset = int(os.environ['RANK'])
is_main = (not ddp) or (local_rank == 0)                            # ← 用的是 local_rank（单机够用，多机有问题）
torch.manual_seed(42 + seed_offset)                                 # 每个 rank 随机采不同的 batch
if is_main: wandb.init(project='webbot-pretrain', config={})
...
model = GPT(vocab_size, context_len, embed_dim, n_head, n_kv_head, n_layer, ff_dim, theta).to(device)
model = torch.compile(model)
if ddp: model = DDP(model, device_ids=[local_rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for step in pbar:
    lr = get_lr(step, ...); optimizer.zero_grad()
    for micro_step in range(accumulation_steps):                    # 勾配蓄積ループ（梯度累积循环）
        x, y = get_batch(train_data, context_len, micro_batch_size, device)
        with autocast(device_type='cuda', dtype=torch.bfloat16):    # ← 写死成 cuda
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1)) / accumulation_steps
        if ddp and micro_step < accumulation_steps - 1:
            with model.no_sync(): loss.backward()                   # 除最后一个 micro_step 外不同步梯度
        else:
            loss.backward()                                         # 最后一次触发 all-reduce（求均值）
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip); optimizer.step()
    train_loss = loss.item() * accumulation_steps                   # ← 只是 rank0 最后一个 micro-batch 的 loss
    if is_main: wandb.log({'train/loss': train_loss, 'train/lr': lr}, step=step)
    if is_main and (step % eval_interval == 0 or step == max_iters-1):
        raw_model = model.module if ddp else model; val_loss = evaluate(raw_model, val_data, context_len, micro_batch_size, device)
```
**超参**：`vocab 50000, ctx 1024, d 768, H 12, KV 4, L 12, ff 2048, θ 1e4, micro_batch 32, accum 4, lr 6e-4, warmup 500, max_iters 100000, grad_clip 1.0, eval_interval 1000`。wandb config 里记的 `effective_batch_size = 32×4 = 128`，**没有乘 world size**。

**模型差异**（webbot vs storybot）：多了 GQA（`W_k`/`W_v` 输出维度是 KV·D）；attention 换成 `F.scaled_dot_product_attention(..., enable_gqa=True)`，prefill 时 `is_causal=True`，decode 时 `is_causal=False`；KV cache 超过 `cos_cache` 长度就截断，并把 `cache_offset` 设成 1024；FFN 属性名叫 `mlp`；`if __name__=="__main__"` 会打印参数量。

注释翻译：DDP初期化 = DDP 初始化；wandb初期化（rank 0のみ）= 只在 rank 0 初始化 wandb；configは後で更新 = config 之后再更新；線形減衰 = 线性衰减；勾配蓄積ループ = 梯度累积循环；DDP: 最後のmicro_step以外は勾配同期を無効化 = DDP：除最后一个 micro_step 外关闭梯度同步；毎ステップのログ = 每步记录；学習曲線の保存 = 保存学习曲线；モデル保存 = 保存模型。

**CPU 缩小版运行** `$SCR/p09_pretrain_shrunk.py`（补丁 11 处）：
- 模型改小：ctx 128，d 128，H 4，KV 2，L 2，ff 384（vocab 50000 不能改，词表决定了它）。
- `micro_batch 4`（accum 保持 4），`max_iters 30`，`warmup 5`，`eval_interval 10`。
- 训练数据用 `owt_valid.bin` 前 90%，验证用之后的 66,736 个 token。
- **注释掉 `torch.compile`**（Windows 缺 cl.exe），环境变量 `WANDB_MODE=disabled`。

实际输出：
```
学習データ: 58,046,459 tokens
検証データ: 66,736 tokens
パラメータ数: 13,193,856 (13.2M)          ← 其中 12.8M 是 embedding/unembed（词表 5 万）
UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling   ← autocast 被静默关闭
step 0: val_loss = 10.8349      （ln 50000 = 10.8198；第 0 步 train loss = 10.8536）
step 10: val_loss = 10.2431
step 20: val_loss = 9.7905
step 29: val_loss = 9.6461
lr 序列：0 → 1.2e-4 → … → 6.0e-4（第 5 步）→ 线性降到 2.4e-5（第 29 步）
モデル保存: …\out\webbot_tiny.pt
exit=0 wall=166s
```
**`torch.compile` 实测**：`InductorError: RuntimeError: Compiler: cl is not found.`（编译时报错，等了 32.9 s）。
**DDP 实测**：`torchrun` 报 `DistStoreError: use_libuv was requested but PyTorch was built without libuv support`，而且设 `USE_LIBUV=0` 也不管用，因为 torchrun 的 static rendezvous 不读这个变量。手动设 RANK 等环境变量绕过 torchrun 后，gloo 报 `RuntimeError: makeDeviceForHostname(): unsupported gloo device`（指定 `GLOO_SOCKET_IFNAME` 后报 `makeDeviceForInterface()`，错误相同）。`dist.is_nccl_available()` 返回 False。**结论：这台 Windows 机器上 DDP 不可行。** 梯度累积和 DDP 均值的数学性质改用单进程模拟来验证（§3）。

### 2.11 `ch09/02_generate.py` + `webbot/utils.py`

`webbot/utils.py` 和 `storybot/utils.py` 只差三处：一行 `# kv cache` 注释、一个空行、文件末尾的换行（已用 diff 确认）。WebBot 截断 cache 的行为也验证过：在 ctx=16 的小模型上生成 30 个 token，第 14 次调用以后 `k_cache` 长度一直是 16、`cache_offset` 一直是 16，而 `rope(x,16)==rope(x,15)`，说明位置被钳在 15。02 用 **`storybot.tokenizer.BPETokenizer`** 加载 `webbot/merge_rules.pkl`（webbot 没有自己的 tokenizer.py），`temperature=0.5`，`max_new_tokens=100`，5 个事实型 prompt。分词结果：
```
'In 1991, Linus Torvalds created' -> [787, 10049, 44, 5894, 384, 3694, 118, 1784, 115, 2588]  ['In',' 1991',',',' Lin','us',' Tor','v','ald','s',' created']
'Monday, Tuesday, Wednesday,'     -> [24219, 44, 2711, 44, 2881, 44]
'Python was created by'           -> [35154, 358, 2588, 417]
'Machine learning is defined as'  -> [47145, 4259, 320, 6354, 352]
'The capital of Japan is'         -> [445, 2786, 287, 2742, 320]
```
用 30 步的 tiny 模型跑 `$SCR/p09_generate_tiny.py`（20.7 s）：输出 “モデルとトークナイザを読み込み中... / 読み込み完了！” 和 5 段乱码续写（例：“The capital of Japan is News Albertaargs Messages Conservatives …”）。代码路径是通的；HF 上**没有**训练好的 WebBot。

### 2.12 `storybot/download.sh`、`webbot/download.sh`

默认只下载 `.txt`（`wget -nc`），`.bin`、`merge_rules.pkl` 和模型那几行**都被注释掉**了。所以照默认跑，你得自己用 ch04/ch07 的 BPE 训练和编码。OWT 11.9 GB 用纯 Python BPE 会非常慢，**教程应该建议把 `.bin` 和 `merge_rules.pkl` 那几行取消注释**。实测大小见 §4.2。

---

## 3. 可最小化验证的事实清单

“浏览器”一列：✅ 表示只用 NumPy、纯 Python（Pyodide）或纯 JS 就能复现；◐ 表示需要预先导出一小份数据或参考值；❌ 表示需要 torch。所有“观测输出”都是这次实际跑出来的。

需要说明的是：标 ✅ 的行里，**F2、F4–F9、F11、F15、F16、F18、F19、F23、F24、F27、F30、F38、F42、F44、F45、F50、F52–F54 的那个检查本身就是用 NumPy、纯 Python 或 JS 跑的**；而 F3、F26、F31、F34–F36、F39–F41、F51 这次是用 torch 跑的，标 ✅ 只是因为改写成 NumPy 很直接（几行矩阵运算），本次没有单独跑 NumPy 版。

| # | 事实 | 最小检查（脚本） | 观测输出 | 浏览器 |
|---|---|---|---|---|
| F1 | 书里的 AdamW 和 `torch.optim.AdamW` 等价 | `v_adamw.py [A]` | 5 步后 max\|Δθ\|=5.96e‑8；loss 8.001141→1.585504 vs 1.585503 | ❌ |
| F2 | NumPy AdamW（f64）和 torch（f64）一致 | `v_adamw.py [C]` | 2.22e‑16 | ◐（预存 torch 参考值） |
| F3 | Adam+L2（耦合）≠ AdamW（解耦） | `v_adamw.py [B]` | 500 步后差 0.212 | ✅（NumPy 两种实现对比） |
| F4 | Adam 的第一步更新约等于 lr·sign(g) | `m̂=g, v̂=g²` | update/lr = [0.990099, −1, 1]（g=1e‑6, −3, 50） | ✅ |
| F5 | `get_lr` 关键点 | `v_lr.py` | (0)=0，(1)=5e‑6，(200)=1e‑3，(20100)=5e‑4，(39999)=2.51e‑8，(40000)=0 | ✅ |
| F6 | D2Z 全程平均 lr = max_lr/2；cosine（5% warmup，10% 底）的面积是 0.5475 | `v_lr.py` | 0.0005；0.5475 vs 0.5000 | ✅ |
| F7 | 三处 `get_lr` 逐字相同 | `v_lr.py` | True | ✅（比较字符串） |
| F8 | fp16：1000+0.01=1000；1e‑8→0；70000→inf；65520→inf；1000 附近的间距是 0.5 | `np.float16(...)` | 如左 | ✅ |
| F9 | bf16：1e‑8→1.0011718e‑08；70000→70144；π→3.140625；1/3→0.33398438 | NumPy RNE 模拟 | 和 torch 在 20 万个值上完全相同 | ✅ |
| F10 | finfo 表（fp16 max 65504，min subnormal 5.96e‑8；bf16 eps 7.81e‑3） | `torch.finfo` | 见 §2.4 | ✅（`np.finfo(np.float16)` 加查表） |
| F11 | fp16 下梯度 3e‑8 存成 6e‑8；×2^16 后再还原是 3.0006e‑8 | `v_precision.py [4]` | 如左 | ✅ |
| F12 | autocast(bf16) 下：残差流 fp32、Linear 输出 bf16、logits bf16、loss fp32、参数和梯度 fp32 | forward hook | 见 §2.4 | ❌（可以做成示意图） |
| F13 | CPU autocast：softmax(bf16)、bf16.sum() 都留在 bf16 | `v_precision.py [5]` | bf16、bf16 | ❌ |
| F14 | 这台 CPU 上 bf16 autocast 比 fp32 慢 2.3 倍 | `v_precision.py [6]` | 3.69 vs 1.58 s/step | ❌ |
| F15 | StoryBot 参数 = 22,696,448（闭式公式 = 实例化结果） | `v_params.py` | 每个 block 3,113,984 | ✅（公式） |
| F16 | WebBot 参数 = 152,316,672；每个 block 6,292,992 | `v_params.py` | embedding 加 unembed 占 50.4% | ✅ |
| F17 | CodeBot 共享权重（`embed.weight is unembed.weight`），StoryBot 和 WebBot 不共享 | `v_params.py` | True/False/False；CodeBot 11,121,640 | ❌ |
| F18 | 训练 token 数 327,680,000（StoryBot）和 13,107,200,000/rank（WebBot） | 乘法 | 如左 | ✅ |
| F19 | 6·N_mm·D：StoryBot 3.456e16；WebBot 8.959e18；StoryBot 全量 eval 再加 45% | `v_params.py` | 如左 | ✅ |
| F20 | 初始 loss ≈ ln V + Var(logit)/2 | `v_valloss.py` | 预测 9.311，实测 9.318（ln V=9.210）；WebBot tiny 10.854 vs 10.820 | ◐（NumPy 随机初始化也能演示） |
| F21 | 发布 checkpoint 的 held‑out loss：9.318→2.416→1.652→1.302 | `v_valloss.py` | ppl 3.68；0.456 bits/byte | ◐（NumPy 前向加 40 MB 权重） |
| F22 | bf16 和 fp32 的 eval loss 只差 1.6e‑4 | `v_valloss.py` | 1.31153 vs 1.31137 | ❌ |
| F23 | `encode("Once upon a time")=[429,438,258,397]`；EOS=9999；WebBot EOS=49999 | `v_dpo.py [4]` | 如左 | ✅（Pyodide 有 `regex`，把 merges 转成 JSON） |
| F24 | valid.bin：5,465,873 token，27,630 个 EOS；故事长度均值 197.8、中位数 177、p90 266；11.0% 超过 256；4.117 B/token；前 3 个故事重新编码后和 .bin 完全相同（747 token） | 数据分析脚本 | 如左 | ◐ |
| F25 | owt_valid.bin：64,496,066 token，59,059 篇文档，最大 id 49999，4.496 B/token | 同上 | 如左 | ◐ |
| F26 | `get_sequence_logprobs` 等于手写循环 | `v_dpo.py [2]` | [−6.729046, −12.019203] | ✅（移植到 NumPy） |
| F27 | DPO 手算：z=0.2 → L=0.598139；∂L/∂logπ_c=−β·σ(−z)=−0.045017 | `v_dpo.py [1]` | 如左 | ✅ |
| F28 | π=π_ref 时 DPO loss = ln 2 | `v_dpo.py [3]` | 0.6931471824645996 | ✅ |
| F29 | 是否 mask prompt 不影响 DPO loss；右填充长度不影响 | `v_dpo_mask.py` | 1.066696 vs 1.066695；256 和 128 相同 | ❌（原理可以用 NumPy 讲） |
| F30 | DPO 数据长度与边界问题：' '+'Tim' ≠ ' Tim' | `v_dpo.py [4]` | 70 vs 69 token | ✅ |
| F31 | `cycle(DataLoader(shuffle))` 会重放第一个 epoch | `v_dpo.py [5]` | batch0 == batch18；共 55.6 epoch | ✅（纯 Python `itertools.cycle` 演示） |
| F32 | 朴素 masked attention == SDPA(is_causal) | `v_attn.py [1]` | 2.4e‑7 | ❌（NumPy 只能演示朴素版） |
| F33 | SDPA(enable_gqa) 按 repeat_interleave 分组，不是 tile | `v_attn.py [2]` | 0.0 vs 4.48 | ◐（NumPy 能演示分组） |
| F34 | Lq=1、Lk=5 时用 `is_causal=True`，只看得到第 0 个 key | `v_attn.py [3]` | True | ✅（NumPy 演示左上对齐的 mask） |
| F35 | KV cache 和全量重算的 greedy 结果相同 | `v_attn.py [4]` | True / True | ✅（NumPy 小模型） |
| F36 | RoPE 的 offset 被钳住：rope(x,300)==rope(x,255) | `v_attn.py [5]` | True | ✅ |
| F37 | 真实模型生成过 256 后会分叉（第 273 位）；cache 长 335；cache 带来 4.6 倍加速 | `v_attn.py [6]` | 如左 | ◐ |
| F38 | 温度表 | `v_attn.py [7]` | 见 §2.6 | ✅ |
| F39 | “DDP 均值 + loss/K”等于全 batch 梯度；忘了 /K 梯度变成 K 倍 | `v_gradaccum.py` | 2.24e‑8；4.0000 | ✅（NumPy 线性模型） |
| F40 | micro‑batch 有效 token 数不等时，各 batch 均值的均值 ≠ 全局 token 均值 | `v_gradaccum.py [3]` | 1.9215 vs 2.0930 | ✅ |
| F41 | `clip_grad_norm_` 返回裁剪前的范数，裁剪后范数为 1，方向不变 | `v_gradaccum.py [4]` | 86.3687 → 1.000000 | ✅ |
| F42 | AdamW 静态显存 16 B/param：StoryBot 0.34 GiB，WebBot 2.27 GiB；WebBot mb32 的 logits 有 1.64e9 个元素（bf16 3.05 GiB，fp32 6.10 GiB） | `v_params.py` | 如左 | ✅ |
| F43 | 激活（为 backward 保存的张量）：StoryBot 每条序列 46.5 MiB；WebBot 每条 695.5 MiB，另有每次前向 217.6 MiB 的 bf16 权重副本 | `v_actmem.py` + 增量法 | 如左 | ❌ |
| F44 | 每 token 的 KV cache：StoryBot 4096 个值；WebBot GQA 6144，MHA 是 18432 | 乘法 | 如左 | ✅ |
| F45 | NumPy 版 StoryBot 前向等于 torch；fp16 npz 40 MiB | `v_numpy_forward.py` | 8.6e‑6，greedy 相同 | ✅（这就是浏览器 demo 的基础） |
| F46 | Windows 上 `torch.compile` 失败 | 小函数编译 | InductorError: cl not found | ❌ |
| F47 | Windows 上 torchrun 和 gloo 失败，没有 NCCL | `v_ddp.py` | 见 §2.10 | ❌ |
| F48 | HF checkpoint 的 key 是 `mlp`，仓库代码是 `ffn` | 直接加载 | RuntimeError | ❌ |
| F49 | `model_dpo.pt` 和 `webbot/model_pretrain.pt` 返回 404 | `curl -sIL` | 404 | ✅（浏览器 fetch） |
| F50 | ch06/02 的输出是 8.0011, 5.8991, 4.1294, 2.6933, 1.5855 | 原样运行；再用纯 NumPy（float64）重放：初始化 `w=[[-0.0052939653396606445, 0.37932294607162476]]`, `b=[-0.5819807648658752]`（即 `torch.manual_seed(0); nn.Linear(2,1)`），x=[1,2]，y=3 | 5 个 loss 和书里逐位相同 | ✅（把初始权重写死进页面） |
| F51 | `get_batch` 的 y 就是 x 右移一位；`evaluate` 的窗口过滤有 off‑by‑one | 用 `np.arange(513)` 当数据调用书里的 `get_batch` | `y[:, :-1]==x[:, 1:]`：True；len=513 时只返回 1 个窗口（本来放得下 2 个），len=514 时返回 2 个 | ✅ |
| F52 | GPT‑2 预分词正则在 JavaScript（`/u` + `\p{L}`、`\p{N}`）和 Python `regex` 下结果完全相同，所以分词器可以用纯 JS 在浏览器里实现 | `$SCR/v_pretok_py.py` 和 `v_pretok_js.js`（node 24），6 个刁钻用例：缩写、数字、连续空格和换行、中日文、emoji、重音字母 | 全部相同。顺带看到两个细节：`"DON'T"` 会被切成 `['DON', "'", 'T']`（缩写规则只认小写）；DPO prompt 结尾单独成了一个 `' '` 预分词 token，这正是 F30 分词边界问题的来源 | ✅（纯 JS） |
| F53 | StoryBot 的 BPE 编码器整个可以用约 40 行纯 JS 实现，结果与 Python 逐 token 相同 | `$SCR/v_bpe_export.py` 把 9,743 条 merge 导成 JSON（连同测试文本约 190 KB），`v_bpe_js.js` 移植 encode/decode；测试文本是 valid.txt 的前 3 个故事再加上缩写、数字、日文、emoji、重音字母 | JS ids == Python ids：true（778 个 token，11 ms）；前 747 个 == `valid.bin` 前缀：true；decode(encode(x)) == x：true；`encode("Once upon a time")=[429,438,258,397]` | ✅（纯 JS，不需要 Pyodide） |
| F54 | **StoryBot 可以完全用 JS 在浏览器里跑**：纯 JS 前向（带 KV cache，逐 token），greedy 结果与 torch 完全相同 | `$SCR/v_js_export.py` 把权重导成扁平 fp32/fp16 二进制加 JSON 索引（fp16 是 45,392,896 B）；`v_js_forward.js`（node 24，单线程）跑 11 个 prompt token 加 40 个生成 token | fp32 和 fp16 权重都与 torch greedy 在全部 51 个 token 上一致；decode 58 ms/token（fp32）和 74 ms/token（fp16，含 half→float 转换），prefill 11 个 token 约 0.7 s。文本：“…a little cat named Kitty. Kitty loved to play with her ball. One day, she saw a big tree and wanted to climb it. …” | ✅（纯 JS，约 15 token/s） |
| F55 | DPO（书中超参，ctx 128）在 CPU 上跑完 1000 步：loss 0.6931→0.4855（第 18 步）→0.2069（第 100 步）→0.0075（第 999 步）；训练 pair 和 held‑out pair 的 reward accuracy 都是 1.0；LM loss 只从 1.3023 升到 1.3149 | `$SCR/p08_dpo_shrunk.py` + `v_dpo_post.py` | 见 §2.8.1（68 min；chosen +13.1、rejected −39.4 nats） | ◐（曲线和指标可导出成 JSON 回放） |
| F56 | CPU 迷你版 StoryBot（3.4M 参数，4.9M token，28 min）：val 9.237→2.651 | `$SCR/p05_pretrain_mini.py` | 见 §2.5.1 | ◐（曲线可回放；在浏览器里训练只适合更小的模型） |
| F57 | 09 的整条 pairwise 流水线可以离线跑通；用词表 mock 时 DPO 23 胜、pretrain 9 胜、平局 8（不计平局 71.9%，p=0.020）。这不是 LLM 判断 | `$SCR/p09_judge_mock.py` | 见 §2.9.1 | ❌ |

---

## 4. GPU 与数据需求

### 4.1 哪些脚本需要 GPU

| 脚本 | 是否需要 GPU | 说明 |
|---|---|---|
| ch06/02, 03, 04, lr_graph | 不需要 | 秒级 |
| ch06/05_pretrain（全量） | **需要** | CPU 上全尺寸 batch 32 约 6 s/step（fp32），4 万步要 60+ 小时 |
| ch06/06_generate | 不需要 | CPU 上 3 个样本 10 s |
| ch06/07, 09 judge | 不需要 GPU，但需要 OpenAI API（要花钱） | 可以用 mock 或本地开源 judge 替代 |
| ch06/08_dpo | 可选 | CPU 上 1000 步实测 68 min（ctx 128，4 线程，有竞争时 2.9–8 s/it）；GPU 上估计 1–3 分钟（每步 4 次前向、2 次反向，batch 8×≤81 token；显存 2–3 GB：两份 22.7M 模型 + AdamW 状态 + 很小的激活） |
| ch09/01_pretrain | **需要（Linux + CUDA）** | 100K 步在单张 H100 上约 8–10 h |
| ch09/02_generate | 不需要 | 但要先有训练好的 WebBot |

### 4.2 数据与模型文件（HF `koki0702/zero-llm-data`，`curl -sIL` 实测）

| 文件 | 大小（B） | token 数 | 本次下载？ |
|---|---|---|---|
| storybot/tiny_stories_train.txt | 2,227,753,162 | — | 否（> 200 MB） |
| storybot/tiny_stories_valid.txt | 22,502,601 | — | 已存在（其他 agent 下的） |
| storybot/tiny_stories_train.bin | 1,082,458,446 | 541,229,223 | 否 |
| storybot/tiny_stories_valid.bin | 10,931,746 | 5,465,873 | 已存在 |
| storybot/merge_rules.pkl | 103,877 | — | 已存在 |
| storybot/model_pretrain.pt | 90,835,030 | — | 是 |
| storybot/model_iter_500.pt | 90,835,030 | — | 是 |
| storybot/model_iter_5000.pt | 90,835,075 | — | 是 |
| storybot/model_dpo.pt | **404** | — | — |
| webbot/owt_train.txt | 11,920,511,059 | — | 否 |
| webbot/owt_valid.txt | 289,998,753 | — | 否 |
| webbot/owt_train.bin | 5,298,009,600 | 2,649,004,800 | 否 |
| webbot/owt_valid.bin | 128,992,132 | 64,496,066 | 是 |
| webbot/merge_rules.pkl | 533,514 | — | 是 |
| webbot/model_pretrain.pt | **404**（download.sh 里也没列） | — | — |
| codebot/model_pretrain.pt | 44,516,353 | — | 否 |

TinyStories 的 txt/bin 和 CS336 用的 `TinyStoriesV2-GPT4-*.txt` 是同一份文件（`cs336-crossref.md` 用 SHA‑256 核对过）；`owt_train.txt` 也和 CS336 的 OWT 样本同为 11.9 GB。

### 4.3 计算量与时间估算

计算口径：N_mm 是参与矩阵乘的参数（总参数减去输入 embedding，unembed 要算）；FLOPs = 6·N_mm·D + attention 6·L·ctx·d·D（因果、减半的口径），再加 eval 的 2·N_mm 乘 eval token 数。

| | StoryBot | WebBot（单卡，100K 步） |
|---|---|---|
| N_mm | 17,576,448 | 113,916,672 |
| D | 3.2768e8 | 1.31072e10 |
| 6·N_mm·D | 3.456e16 | 8.959e18 |
| attention | 1.03e15（+3%；朴素实现会算满 C×C，实际约 +6%） | 7.42e17（+8.3%） |
| eval | 1.60e16（81 次 × 5.47M token，**+45%**） | 1.61e18（101 次 × 64.5M token，+17%） |
| 合计 | **5.16e16** | **1.13e19** |

**理想时间** = FLOPs ÷（bf16 稠密峰值 × MFU）。峰值取 H100 SXM 989、A100 312、L40S 362、L4 121、RTX 3060 Ti 约 32 TFLOPS（GeForce 的 bf16 走 fp32 累加，按半速估）：

| GPU | StoryBot（MFU 40% / 20%） | WebBot（MFU 40% / 20%） |
|---|---|---|
| H100 SXM | 0.04 h / 0.07 h | **7.9 h / 15.9 h** |
| A100 | 0.11 / 0.23 h | 25.2 / 50.3 h |
| L40S | 0.10 / 0.20 h | 21.7 / 43.4 h |
| L4 | 0.30 / 0.59 h | 64.9 / 129.8 h |
| RTX 3060 Ti | 1.1 / 2.2 h | 242 / 485 h |

**按经验修正**：StoryBot 太小，GPU 喂不饱，时间主要花在 eager 模式的 kernel launch（**估计**每步几百次，没有实测）、Python 版 `get_batch` 和朴素 attention 上。所以实际大约是 **H100 10–20 min、A100 15–30 min、L4 30–60 min、3060 Ti 1–2 h**；全量 eval 在其中占不小比例，可以改成 eval 子集来省时间。WebBot 是算力主导，有 compile 和 flash 时 MFU 在 35–45% 比较现实。**建议的做法**：先跑前 100 步，读 tqdm 的 it/s，再乘以 max_iters。教程里可以放一个 ETA 计算器（§6）。

**DDP 的坑（影响时间）**：`max_iters` 是固定的，所以 W 张卡并不会更快，而是每步处理 W 倍的 token（总量 13.1B×W）。想用 8 卡在约 1 h 内跑完同样的 13.1B token，应该把 `max_iters` 改成 12,500，或者像 nanoGPT 那样用 `accumulation_steps //= world_size`。另外 eval 只在 rank 0 上跑，它不会因为加卡而变快。

### 4.4 显存估算（参数 + 梯度 + Adam 是 16 B/param，加上实测的激活）

用 `saved_tensors_hooks` 统计 backward 需要保存的张量，取“B=2 减 B=1”的增量，得到：

| | 静态 | 每条序列的激活 | 每次前向固定开销 | logits 梯度（fp32） | 合计估计 |
|---|---|---|---|---|---|
| StoryBot，batch 32×256 | 0.34 GiB | 46.5 MiB | 34 MiB | 0.31 GiB | **约 2.1 GiB**（加 CUDA context 约 2.6 GiB）→ 3060 Ti 8 GB 足够 |
| WebBot，mb 32×1024 | 2.27 GiB | 695.5 MiB | 218 MiB | 6.10 GiB | **约 30 GiB** → 需要 A100‑40G（偏紧）或 80G 卡 |
| WebBot mb 16 | | | | | 约 16.4 GiB → L4 24G 能跑（把 accum 调到 8 保持等效 batch） |
| WebBot mb 8 | | | | | 约 9.4 GiB |
| WebBot mb 4 | | | | | 约 6.0 GiB → 3060 Ti 勉强（accum 32） |

（CPU 上的 autocast 和 SDPA kernel 跟 CUDA 不完全一样，误差按 ±30% 看。CUDA 上 softmax 和 log_softmax 是 fp32，`torch.compile` 的融合会减少激活。）

### 4.5 怎样缩小到 CPU（已验证可行的配方）

- **StoryBot 冒烟**：全尺寸、batch 8、40 步，约 5.5 min（bf16）。改成 fp32 能快约 2 倍。
- **StoryBot 迷你**：d128/H4/L4/ff384/ctx128，3.4M 参数，fp32，1200 步（约 4.9M token）。结果见 §2.5.1。
- **DPO**：ctx 改成 128，数值完全不变。1000 步在 4 线程、有竞争时实测 68 min。也可以只跑 200 步（约 1/5 的时间），loss 已经从 0.693 降到约 0.10（第 100 步约 0.21）。
- **WebBot**：d128/L2/KV2/ctx128，mb 4，30 步，166 s。要注释掉 compile，并设 `WANDB_MODE=disabled`。
- 通用：关掉 autocast（或改成 `enabled=device.type=='cuda'`）；验证集只取一小段；`torch.set_num_threads(4–8)`。

### 4.6 云端运行要点（Modal / RunPod / Lambda）

**通用准备**
1. 在 Linux x86_64 上执行 `git clone` 和 `uv sync` 就够了。`uv.lock` 锁的是 PyPI 上的 torch 2.9.1，在 Linux x86_64 上会连带装 `nvidia-*-cu12`（含 cuDNN 和 NCCL）和 `triton`（已查 lockfile 的 marker），所以得到的是带 NCCL 和 Triton 的 CUDA 12 版 torch，`torch.compile` 和 DDP 都能直接用。Windows 上同一个 lockfile 装出来的是 CPU 版 wheel，这就是本机 torch 显示 `+cpu` 的原因。
   最小命令序列（RunPod 或 Lambda 的 Ubuntu 加 CUDA 镜像）：`git clone https://github.com/oreilly-japan/deep-learning-from-scratch-6 dlfs6 && cd dlfs6 && curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync`，然后 `nvidia-smi` 和 `uv run python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name())"` 自检。
2. **取消 download.sh 里 `.bin` 和 `merge_rules.pkl` 那几行的注释**。StoryBot 约 1.1 GB；WebBot 约 5.4 GB；StoryBot 的 checkpoint 每个 91 MB。
3. 修 key（`.mlp.`→`.ffn.`），或者干脆不用 HF 模型，自己从头训练。
4. 设 `WANDB_API_KEY`，或者 `WANDB_MODE=offline`。无界面的机器上 `wandb.init` 会弹交互式登录。
5. 用 `nohup`/`tmux` 跑；checkpoint 和 loss JSON 存到持久盘。

**StoryBot**：1×A100 或 L4 就够了，约 0.5 h（L4 约 1 h）。DPO 在任意 GPU 上都只要几分钟。总花费低于 1–2 美元。

**WebBot**：
- 方案 A：1×H100（80 GB），micro_batch 32 × accum 4，约 8–10 h，约 25–40 美元。
- 方案 B：8×H100 跑 `torchrun --standalone --nproc_per_node=8 ch09/01_pretrain.py`，同时把 `max_iters` 改成 12,500。墙钟约 1–1.5 h（eval 另算，只在 rank 0 上跑），GPU·h 和方案 A 差不多。
- 方案 C（省钱或练手）：把 max_iters 改成 10,000，或者把模型降到 d512/L8，L4 或 A10 上 1–3 h。

**Modal 草图**（API 以 Modal 官方文档为准）：
```python
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy<2", "tqdm", "regex", "wandb", "matplotlib")
vol = modal.Volume.from_name("dlfs6", create_if_missing=True)
@app.function(gpu="H100", image=image, volumes={"/vol": vol}, timeout=24*3600)
def train():  # 若 /vol 中没有 .bin 就从 HF 下载；把仓库挂进来；subprocess 调 python 或 torchrun；输出写回 /vol
```
多卡用 `gpu="H100:8"`，在函数里用 subprocess 调 torchrun。

**本机 3060 Ti（Windows）**：StoryBot 全量可行，约 1–2 h，约 2–3 GiB 显存。ch09 要先注释掉 compile（Windows 上没有官方 Triton），DDP 不可行，WebBot 全量也不现实。建议本机只跑 StoryBot、DPO 和 WebBot 缩小版；或者装 WSL2。

---

## 5. 易错点、坑与潜在 bug（每条都实测过；“推断”会单独标出）

**加载与兼容性**
1. **[bug，已验证]** HF checkpoint 里是 `mlp`，仓库 `storybot/model.py` 里是 `ffn`，所以 06/07/08/09 原样跑会失败。可以在加载时改 key；或者把 storybot 的属性名改成 `mlp`，与 webbot 保持一致（更推荐）。
2. **[缺文件]** `model_dpo.pt` 在 HF 上是 404，但 download.sh 里列了它；WebBot 没有发布模型。
3. **[小坑]** RoPE 的 cos/sin buffer 在 state_dict 里每个 block 存了一份（共享模块存了 4 次，多 32,768 个元素），无害。

**训练配方**
4. **[事实]** 调度是 D2Z，不是 cosine。另外 `get_lr(0)=0`：第 0 步参数不变，但 AdamW 的 m、v、t 已经更新了。
5. **[设计取舍]** AdamW 没分 param group，RMSNorm 的 weight 和 embedding 也被 weight decay；betas 是 (0.9, 0.999)，wd 0.01（LLM 常用 (0.9, 0.95) 和 0.1）。
6. **[性能]** 每 500 步就在完整验证集（5.47M token）上评估一次，多出 45% 的 FLOPs。
7. **[CPU]** CPU 上 bf16 autocast 反而慢 2.3 倍；`lr_graph.py` 的字体在 Windows 上显示成方块。
8. **[语义]** autocast 下残差流仍是 fp32（embedding 是 fp32，加法会类型提升），这也是不需要 GradScaler 的原因之一。如果改成 fp16，就必须加 `torch.amp.GradScaler`。
9. **[断点续训]** `GPT.save` 只存权重和配置，不存优化器状态和步数，没法精确续训（CS336 要求的 checkpoint 会存这些）。
10. **[小 off‑by‑one]** `evaluate` 的过滤条件 `ix + ctx + 1 < len(data)` 用了严格小于，会多丢掉最后一个刚好放得下的窗口。
11. **[数据]** `get_batch` 是有放回的随机采样，窗口可以跨越 EOS（跨故事）。没有文档级的 attention mask，这是标准做法，但值得在教程里说一句。

**生成**
12. **[bug 级]** StoryBot 的 KV cache 不截断，RoPE 位置钳在 255。06 的 `max_new_tokens=300` 加上 1 个 prompt token 会超过 256，11% 的 TinyStories 故事本来就超过 256 token。实测在第 273 位分叉。`ids` 的截断在用 cache 时是死代码。WebBot 会截断 cache，但被截断后的 K 仍保留原来的绝对位置旋转，位置也钳在 1023，同样只是近似。修法：超长时清空 cache，用最后 ctx−1 个 token 重新 prefill（滑窗重算），或者干脆限制 `max_new_tokens`。
13. **[观察]** 以 `<|endoftext|>` 为 prompt 时，解码出来的文本常以 `\n` 开头；`generate` 会把 prompt 里的 EOS 一并去掉。

**评测（LLM judge）**
14. **[健壮性]** `json.loads` 没有 try/except；`max_tokens=150` 可能把 JSON 截断；`judgment["winner"]` 如果返回 `"Tie"` 或 `"Story A"` 会 KeyError；生成没设 seed。（推断，没调 API。）
15. **[方法论]** 07 是 1–5 分的绝对评分，区分度低，可能有天花板效应；09 做了位置随机化，这是好的做法。但只比较 happy ending，而 DPO 训练的就是 happy ending，所以这个胜率主要说明“目标达成了”，不代表通用质量。judge prompt 是日文，中文教程里要翻译。

**DPO**
16. **[分词边界]** prompt 末尾的空格和回答分开编码，得到的是 `' '`+`'Tim'`，而预训练时同样的文本会编成 `' Tim'`。chosen 和 rejected 都受影响，所以大部分会抵消，但第一个回答 token 的 log‑prob 偏低。
17. **[数据]** 回答后面没有加 EOS；pad id 0 是字节 `\x00`（被 mask 掉了，无害）；prompt 的 mask 在数学上不影响 loss（F29），真正必要的是 mask 掉 padding。
18. **[过拟合]** 1000×8/144 = 55.6 个 epoch。`cycle()` 让后续 epoch 按第一个 epoch 的 batch 顺序重放。loss 会降到接近 0（见 §2.8.1）。没有验证集，书里也没算 reward accuracy。
19. **[长度]** log‑prob 是求和、不按长度归一化，而 chosen 比 rejected 长（64.0 vs 49.7 token，含 prompt），存在长度混淆。
20. **[训练模式]** `model` 处于 train 模式，`ref_model` 处于 eval 模式。StoryBot 没有 dropout，所以没有影响；换成带 dropout 的模型就要注意（`cs336-crossref.md` 记录过 GRPO 里的类似问题）。
21. **[随机性]** 08 没设 seed，结果不可复现。

**WebBot / 分布式**
22. **[可移植性]** `autocast(device_type='cuda')` 写死了，在 CPU 或 MPS 上会静默关闭（只有一条警告）。
23. **[Windows]** compile、NCCL、gloo、torchrun 都不可用（F46、F47）。
24. **[DDP 语义]** 加卡等于加 token，不会缩短时间；wandb 里的 `effective_batch_size` 没乘 world size；`is_main` 用的是 LOCAL_RANK，多机时每台机器都会有一个“main”（wandb、eval、save 都会重复）；`train_loss` 只记了 rank 0 最后一个 micro‑batch，没在 rank 间平均；eval 期间其他 rank 卡在下一次 all‑reduce 上，慢卡上 eval 太久可能触发 NCCL 超时（推断）。
25. **[正确性]** 梯度累积和 DDP 的数学是对的（F39）：`loss/accum` 加 DDP 求均值，等于全局 batch 求均值。前提是每个 micro‑batch 的 token 数相同，这里成立（固定长度窗口）。
26. **[compile 与 eval]** 在 eval 和 train 之间切换会触发重新编译；最后一个 eval batch 可能更小，又会多编译一次（推断，没在 GPU 上验证）。
27. **[数据加载]** `get_batch` 每个 micro‑step 都在 Python 里切 32 段，没有 pin memory 和预取，GPU 会等 CPU。nanoGPT 会用 `pin_memory` 加 `non_blocking`。

**跨 bot 差异速查**：CodeBot 用 LayerNorm、GELU、可学习的位置 embedding、dropout，并且共享权重，没有 KV cache，生成时用滑窗重算。StoryBot 用 RMSNorm、SwiGLU、RoPE，没有 dropout，不共享权重，有 KV cache，FFN 叫 `ffn`。WebBot 在 StoryBot 的基础上加了 GQA、SDPA、cache 截断，FFN 叫 `mlp`。codebot 的 `generate` 返回 `generated_ids[0]`，storybot 和 webbot 用布尔 mask 把结果拉平成 1D。

---

## 6. 互动组件点子

1. **优化器竞技场**（NumPy/JS）：在二维病态二次面或 Rosenbrock 上同时跑 SGD、Adam、AdamW。滑块调 lr、β1、β2、wd、eps。开关“耦合 L2 / 解耦 wd”。逐步显示 m、v、m̂、v̂，突出 t=1 时的更新约等于 lr·sign(g)（F4）。
2. **学习率调度游乐场**：调 warmup 比例、总步数、cosine 的 η_min、WSD 的 decay 比例；叠加 D2Z、cosine、WSD 三条曲线，显示平均 lr（面积，F6）；标出 StoryBot 和 WebBot 的实际参数，以及 `get_lr(0)=0` 这一点。
3. **浮点格式显微镜**：输入一个数，显示 fp32、fp16、bf16 的位布局（符号、指数、尾数），舍入后的值和相对误差。拖动数量级滑块时标出 fp16 的上溢区（>65504）和下溢区（<5.96e‑8）。演示“1000+0.01”和 loss scaling（F8–F11，可用 NumPy 的 bf16 模拟）。
4. **autocast 地图**：StoryBot block 的数据流图，每个张量按 dtype 着色（实测，F12），可切换 CPU 和 CUDA 两套策略，标出“残差流为什么是 fp32”。
5. **训练预算计算器**：输入模型配置（V、d、L、ff、H、KV）和训练配置（batch、ctx、iters、accum、world），输出参数分解、6ND、D/N 和 Chinchilla 20 的对比、各型号 GPU 的时间（MFU 滑块）、显存（16 B/param 加 0.68 GiB/seq 等实测系数）、花费；还可以把实测的 it/s 填进去推 ETA。预设 CodeBot、StoryBot、WebBot 和 CS336 TinyStories。
6. **学习曲线加样本时间轴**：真实的 val loss 点（9.318、2.416、1.652、1.302），拖动滑块就切换到预先生成好的、对应 checkpoint 的故事样本，让读者直观看到“loss 1.65 和 1.30 写出来的东西差在哪里”。
7. **温度与采样**：用真实 StoryBot 在某个上下文下的 top‑20 logits（预先导出）。拖动 T，看概率条和熵的变化；点“采样”按钮；T=0 时就是 argmax（F38）。
8. **KV cache 逐步动画**：逐 token 显示哪些 K/V 是新算的、哪些是复用的，以及累计节省的 FLOPs（4.6 倍）。显示 RoPE 的位置编号，演示超过 256 后位置卡在 255；可以对比 cache 和滑窗两种方式在第 273 位分叉（F37）。
9. **浏览器里的 StoryBot（端到端例子的压轴）**：首选纯 JS 方案，45 MB fp16 权重、JS 分词器、带 KV cache 的 JS 前向，约 15 token/s（F52–F54）；备选是 Pyodide 加 NumPy 前向（F45）。可调温度，可切换 iter 500、5000、40000 和 DPO 四个模型（每个 45 MB，可以按需加载）。再加一个“位置钳制”开关，演示超过 256 token 以后会发生什么（F37）。
10. **DPO 手算器**：输入 4 个 log‑prob 和 β，显示 z、loss、梯度，以及 σ 曲线上当前的位置。初始时 loss=ln 2。演示拉大 margin 的几种方式：chosen 升、rejected 降（本次实测在训练 pair 上是 +13.1 和 −39.4 nats，见 §2.8.1）；或者两者一起降、但 rejected 降得更多（即文献里的 likelihood displacement，本次没出现）。两种情况下 loss 都会下降，这正说明 DPO 只约束“相对差”。
11. **DPO 数据显微镜**：显示一条样本的 token、mask（prompt 为 0、回答为 1、padding 为 0）、分开编码与一起编码的差异（F30）。开关“mask prompt”，loss 不变（F29）。
12. **DPO 训练回放**：用 CPU 实测的 loss 曲线和 reward margin；画出 epoch 边界（每 18 步）；演示 cycle 的重放（F31）。
13. **Judge 实验台**：中文版的两个 judge prompt；用位置交换讲位置偏差；计算含与不含平局的 win rate；把离线 mock judge 和真实 LLM judge 放在一起对比，说明“规则打分的局限”。
14. **梯度累积与 DDP 动画**：滑块调 micro_batch、accum、world。动画里 micro‑step 在 no_sync 下各自累积，最后一步做 all‑reduce。演示“忘了 /K，梯度就变成 K 倍”（F39），以及“固定 max_iters 时加卡等于加 token”。
15. **GQA 与 KV cache 显存**：12 个 Q 头和 4 个 KV 头的分组连线（repeat_interleave，F33）；KV cache 显存计算器，对比 MHA、GQA、MQA，以及上下文长度和 batch 的影响（F44）。
16. **WebBot 显存预算**：调 micro‑batch 大小，看估计显存和各张卡（8、24、40、80 GB）的对比，自动给出保持等效 batch 128 所需的 accum（§4.4）。

---

## 7. CS336 对照（凭课程知识；与 `cs336-crossref.md` 保持一致）

| DLFS6 部件 | CS336 对应 | 关系 |
|---|---|---|
| `02_adamw.py` 自制 AdamW（`Optimizer` 子类） | **A1** `adamw`（同样要求继承 `torch.optim.Optimizer`），`adamw_accounting`（16 B/param、FLOPs、显存账） | 几乎是同一道题；书里最后换成了 torch 实现 |
| `03_lr_scheduler.py` / `lr_graph.py` | **A1** `learning_rate_schedule`（cosine 加 warmup，降到 min_lr）；Lecture 3（超参、WSD、cosine） | **调度形状不同**：书是 D2Z，CS336 是 cosine |
| `04_mixed_precision.py` | **A2** `mixed_precision_accumulation`、`benchmarking_mixed_precision`（问 autocast 下各张量的 dtype）；Lecture 2（fp32/fp16/bf16/fp8） | F12、F13 正好回答了 A2 那道题的 CPU 版 |
| `05_pretrain.py` 的 `get_batch` 和 memmap | **A1** `data_loading`（随机起点、x/y 错一位、np.memmap） | 同构 |
| `clip_grad_norm_` | **A1** `gradient_clipping`（自己实现，eps 1e‑6） | 书里直接用 torch 的 |
| `05_pretrain.py` 的整体配置 | **A1** TinyStories 实验：同一组超参，同样的 327,680,000 token，目标 val ≤ 1.45（40M token 版本 ≤ 2.00） | **完全一致**；书的最终模型实测 1.302 |
| `GPT.save` | **A1** `checkpointing`（要求连优化器状态和步数一起存） | 书只存权重 |
| `06_generate.py` | **A1** `decoding`（temperature 加 top‑p）；Lecture 10（推理、KV cache） | 书里没有 top‑p |
| KV cache、GQA | Lecture 10（KV cache 显存、GQA/MQA）；Lecture 3（架构） | F44 |
| `07`/`09` LLM judge | Lecture 12（评测）；A5 补充作业的评测 | 书用 gpt‑4o‑mini |
| `08_dpo.py` | **A5 补充作业** `compute_per_instance_dpo_loss` 和 DPO 训练；Lecture 15（SFT/RLHF/DPO） | 同一个 loss。CS336 用 HH 数据、参考模型是 SFT 模型、RMSprop lr 1e‑6、等效 batch 64、报告 implicit reward accuracy；书用自造的 happy/sad 数据、参考模型是预训练模型、AdamW 5e‑6 |
| ch07 的 50K BPE | **A1** OWT BPE（32K） | 同一份 owt_train.txt |
| `ch09/01_pretrain.py` 的 DDP 和 `no_sync` | **A2** DDP 系列：naive DDP、flat、overlap individual、bucketed | 书用 PyTorch 自带的 DDP；A2 要求自己写 |
| `torch.compile` | **A2** benchmark compiled 与 eager | — |
| SDPA（flash） | **A2** FlashAttention‑2（先写 PyTorch 版再写 Triton 版） | 书只调 API |
| 梯度累积 | A1 和 A2 讨论 batch size 时会涉及 | — |
| wandb | **A1** `experiment_log` | — |
| 6ND、D/N、GPU 时间 | Lecture 2（resource accounting）；**A3** scaling laws（IsoFLOPs） | §4.3 |
| OWT、TinyStories 数据来源 | **A4**（数据过滤、去重）提供背景 | — |
| WebBot 训练本身 | **A1** OWT 实验和 leaderboard（1.5 H100‑h 预算） | WebBot 的预算大得多（约 8 H100‑h） |

**教学上的取舍**：CS336 要求“所有零件都自己写”（Linear、Embedding、softmax、交叉熵、AdamW、clip、DDP、FlashAttention）；DLFS6 先手写一次，再换成框架实现。教程可以跟着 DLFS6 的节奏，把 CS336 的对应作业当成每节末尾的“挑战题”，例如：把 D2Z 换成 cosine，再拿 CS336 测试里的 warmup 7、cycle 21 去验数值。

---

## 附录 A：复现命令（全部在仓库根目录执行）

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
SCR=C:/Users/chenf/AppData/Local/Temp/claude/C--Users-chenf/2766cf54-7521-48a1-be58-647c0a0601b0/scratchpad/ch06-ch09-scale
uv run python ch06/02_adamw.py                     # 原样
uv run python ch06/04_mixed_precision.py           # 原样
uv run python $SCR/run_keyfix.py ch06/06_generate.py 0      # 原脚本 + 加载时改 key
for v in v_adamw v_lr v_precision v_params v_dpo v_attn v_gradaccum v_valloss v_actmem v_dpo_mask v_numpy_forward; do
  uv run python $SCR/$v.py                         # 各自几秒到几分钟（v_valloss 约 5 min）
done
uv run python $SCR/v_pretok_py.py && node $SCR/v_pretok_js.js          # F52：正则在 Python 和 JS 下一致
uv run python $SCR/v_bpe_export.py && node $SCR/v_bpe_js.js            # F53：纯 JS 版 BPE
uv run python $SCR/v_js_export.py && node $SCR/v_js_forward.js         # F54：纯 JS 前向
uv run python $SCR/make_patched.py && uv run python $SCR/make_judge_mock.py   # 生成所有补丁副本（每处替换都 assert 只匹配一次）
uv run python $SCR/p05_pretrain_smoke.py           # 约 5.5 min
uv run python $SCR/p05_pretrain_mini.py && uv run python $SCR/v_mini_samples.py     # 约 28 min
uv run python $SCR/run_keyfix.py $SCR/p08_dpo_shrunk.py && uv run python $SCR/v_dpo_post.py   # 约 68 min + 2 min
WANDB_MODE=disabled uv run python $SCR/p09_pretrain_shrunk.py && uv run python $SCR/p09_generate_tiny.py
uv run python $SCR/run_keyfix.py $SCR/p07_judge_mock.py      # 离线 mock，不调 API
uv run python $SCR/run_keyfix.py $SCR/p09_judge_mock.py      # 离线 mock（词表打分），需要先跑完 DPO
```
主要日志在 `$SCR/out/`：`smoke_run.log`、`mini_run.log`、`dpo_run.log`、`dpo_post.log`、`webbot_run.log`、`judge07_mock.log`、`judge09_mock.log`、`ddp_rank*.log`。产物有 `model_dpo_cpu.pt`、`mini_model_pretrain.pt`、`webbot_tiny.pt`、`loss_dpo.png`、`mini_loss_val.png`、`smoke_loss_val.png`、`webbot_val_loss.png`；`$SCR/js/` 里是导出的 fp16/fp32 权重和索引。
