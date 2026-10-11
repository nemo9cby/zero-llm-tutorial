# DLFS6 第 2 章「模型【基本编】」研究档案：从软字典到 GPT-2

> 档案编号：repo-ch02-attention-gpt　·　日期：2026-10-11
> 范围：`ch02/*.py`（01、02、03、06、07、08、09、10、graph.py）与 `codebot/model.py`，并对照 `storybot/model.py`、`webbot/model.py` 做变体比较。
> 仓库：oreilly-japan/deep-learning-from-scratch-6（本地 `C:\Users\chenf\Projects\_refs\dlfs6`，未改动任何文件）。
> 运行环境：Windows 11，i7-11700F（8 核 16 线程），Python 3.12.9，torch 2.9.1+cpu，numpy 1.26.4，uv 0.9.13；浏览器侧用 Pyodide 314.0.7（Python 3.14.2，numpy 2.4.6，经 Node 24.13 运行）。
> 验证工具包（本档案所有数字都能用它复现）：`C:\Users\chenf\Projects\zero-llm-tutorial\research\assets\ch02-attention-gpt\`

---

## 0. 一页摘要

- **文件编号就是小节编号**（已用 O'Reilly Japan 官方目录核对）：01=2.1 软字典，02=2.2 公式与实现，03=2.3 缩放，**04/05 空缺**（2.4 Transformer 基本结构、2.5 位置编码，均无独立脚本），06=2.6 Attention mask，07=2.7 Value 矩阵的低秩化，08=2.8 多头，09=2.9 Transformer 块（LayerNorm/GELU/FFN），10=2.10 GPT-2；`graph.py` 画 ReLU 与 GELU 的对比图。
- **全章只用 CPU**：每个脚本 0.3–4 s（大部分时间花在 `import torch`）。GPU 只有在 ch03 训练这个模型时才需要：20,000 步 × 32 × 256 token，约 1.2×10¹⁶ FLOPs，fp32 下激活约 2.2 GiB，8 GB 显存足够。
- **数值等价性全部实测**：手写 attention 与 `F.scaled_dot_product_attention(is_causal=True)`、`nn.MultiheadAttention` 一致（误差约 1e-7）；书中 LayerNorm 与 `nn.LayerNorm` 一致（9.5e-7）；书中 GELU 与 `nn.GELU(approximate='tanh')` 一致（4.8e-7），**但 `codebot/model.py` 实际用的是 `nn.GELU()`（erf 版），最多相差 4.7e-4**。
- **codebot 的 GPT 就是 GPT-2 结构**：把随机初始化的 HF GPT-2 权重拷进来，logits 相差 1.9e-6（前提是把 GELU 换成 tanh 版、把 HF 的 qkv/proj bias 置零）。与 GPT-2 的差别只有四处：没有 qkv/proj bias、unembed 多一个 bias、GELU 用 erf、初始化时不缩放残差投影。
- **参数量**：ch02/ch03 配置为 11,121,640；用 GPT-2 small 超参时为 124,453,201（HF GPT-2 small 为 124,439,808，差值 +13,393 正好等于 +50,257 − 36,864）。
- 写教程时最值得放大的发现：
  1. 点积相似度不等于“最近”，并列时“硬查找”也会失败（F02、F03）；
  2. 01 的朴素 softmax 会溢出（F04）；
  3. “不缩放会导致梯度消失”只在 d 很大时成立，不缩放时雅可比范数随 d 先升后降（§3.2 表）；初始化时注意力本来就近乎均匀（F10）；
  4. 初始 loss 不是 ln V，而是约等于 ln V + σ²/2（F45）；
  5. 去掉 mask 的“作弊实验”能把随机 token 的 loss 从 ln V 压到 ln(V)/C（F20）；
  6. 拆头时漏掉 transpose 会悄悄泄露未来信息（F27）；
  7. SDPA 的 `is_causal` 是左上角对齐（F21）；
  8. 训练后的模型有 previous-token head 和 attention sink（F51）。

### 0.1 如何复现（Git Bash）

```bash
KIT=/c/Users/chenf/Projects/zero-llm-tutorial/research/assets/ch02-attention-gpt
REPO=/c/Users/chenf/Projects/_refs/dlfs6
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1

# 书中脚本（原样运行；graph.py 会往当前目录写 relu_gelu.png，所以在临时目录里运行）
cd $REPO && MPLBACKEND=Agg uv run python ch02/01_soft_dict.py      # 02/03/06/07/08/09/10 同理

# 53 条最小可验证事实（CPython，约 80 s，结果写入 $KIT/facts_cpython.json）
cd $KIT/verify && uv run --project $REPO --with transformers python run_facts.py
# 其中 23 条“浏览器可跑”的事实放到 Pyodide 里再跑一遍（结果写入 $KIT/facts_pyodide.json）
cd $KIT/pyodide_test && npm i && node run_facts.mjs "$KIT"
# 需要训练好的 checkpoint 的脚本（F13/F50、v_trained.py 等）：先在仓库里执行 codebot/download.sh（44,516,353 字节）
```

工具包内容：`verify/`（facts.py、run_facts.py、render_facts.py、np_gpt.py、v01…v_variants.py 等专项脚本）、`pyodide_test/`（run_facts.mjs、run.mjs、run_trained.mjs、js_matvec_bench.mjs、tiny_ref.npz、package.json，pyodide 版本锁定为 314.0.7）、`ch02_attn_maps.json`（训练好的模型在一段代码上的全部注意力权重，6 层 × 6 头 × 13 × 13）。

---

## 1. 推断的章节叙事

### 1.1 官方目录与文件的对应

依据：O'Reilly Japan 书页（ISBN 978-4-8144-0161-1）的目录。下表列出节标题的中文译文和对应文件：

| 节 | 节标题（中译） | 小节（中译） | 对应文件 |
|---|---|---|---|
| 2.1 | Attention 的基本原理 | Python 的字典 / 软字典 / 软字典的实现示例 | `01_soft_dict.py` |
| 2.2 | Attention 的公式与实现 | 用公式表示 Attention / 支持多个 query / Attention 的实现 | `02_attn_math.py` |
| 2.3 | Attention 缩放 | softmax 的饱和问题 / 引入缩放 / 缩放的效果 | `03_attn_scaling.py` |
| 2.4 | Transformer 的基本结构 | Transformer 的处理流程 / Attention 在 Transformer 中的作用 | 无（04 空缺） |
| 2.5 | 位置编码 | 为什么需要位置信息 / 位置编码的原理 / 绝对位置编码 / 可学习的位置编码 | 无（05 空缺）；可学习位置编码以 `pos_embed` 形式出现在 10 中 |
| 2.6 | Attention mask | 高效的数据格式 / Attention 的问题 / mask 的原理 / mask 的实现 | `06_attn_mask.py` |
| 2.7 | Value 矩阵的高效化 | 大模型中 Attention 的问题 / 用矩阵分解做低秩近似 / 从矩阵分解看 Attention 的新解释 / 实现示例 | `07_attn_value.py` |
| 2.8 | 多头 Attention | 从多个视角看 / 多头 Attention 的公式 / 两个等价的公式 / 实现 / 实现（汇总） | `08_multi_head.py`（前半函数式代码对应 2.8.4，类对应 2.8.5） |
| 2.9 | Transformer 块 | LayerNorm 的原理 / LayerNorm 的实现 / FFN 的实现 / Transformer 块的实现 | `09_norm_gelu.py`；`graph.py`（GELU 插图） |
| 2.10 | GPT-2 | GPT-2 的实现 / 模型的改进 / 模型的保存与加载 / 运行确认 | `10_gpt2.py`；最终沉淀为 `codebot/model.py` |

本章前后的上下文：第 1 章是 tokenizer（BPE，产出 `codebot/merge_rules.pkl` 与 `tiny_codes.bin`），第 3 章是训练（预训练、文本生成、SFT、GRPO），用的就是本章的 `codebot.model.GPT`。

### 1.2 叙事主线（推断）

斋藤的节奏是“提出问题 → 做最小改动 → 给一段能跑的代码”，每一步只引入一个新概念：

1. **从已知出发（2.1）**：Python `dict` 是“硬”查找，query 必须与 key 完全相等。把“相等”放宽成“相似度”（点积），把“取出一个 value”放宽成“按 softmax 权重混合所有 value”，就得到软字典。电影例子（key = 动作/剧情/喜剧三维特征，value = 评分）给后面的矩阵公式提供了直观锚点。
2. **写成矩阵（2.2）**：多个 query 一起算就是 softmax(QKᵀ)V。02 沿用 01 的数据，第一行结果同为 78.66，读者可以直接对照。
3. **发现数值问题并修补（2.3）**：分数一大 softmax 就饱和（[100,200,300] → [0, 3.8e-44, 1]）；随机向量点积的方差等于 d，所以除以 √d。
4. **先俯瞰全局（2.4，纯文字）**：Transformer 怎样处理一串 token，Attention 在其中负责什么。
5. **补上缺失的信息（2.5，纯文字或零散代码）**：Attention 对输入顺序不敏感（F11），所以需要位置编码。先讲绝对（大概率是正弦式）位置编码，再讲 GPT-2 用的可学习位置嵌入。
6. **训练时的作弊问题（2.6）**：“高效的数据格式”应该是指一条长度为 C+1 的序列一次给出 C 个（输入, 目标）对，所有位置并行训练（即 ch03 `TokenDataset` 的 `x=tokens[i:i+C]`、`y=tokens[i+1:i+C+1]`）；“Attention 的问题”是在这种格式下每个位置都能看到答案。解决办法是下三角 mask。F20 把“作弊”量化成了一个可复现实验。
7. **精简参数（2.7）**：这一节解释了 06 → 07 的代码差异。06 的 `W_v` 是 E×E（256×256），07 把它分解成 `W_v`（E→D）和 `W_o`（D→E），相当于用秩为 D 的 `W_o·W_v` 近似 E×E 矩阵（value 部分参数 65,536 → 32,768）。“新解释”大概率是：`W_o` 是把注意力结果写回残差流的输出变换，即 Transformer Circuits 里的 OV 电路 W_OV = W_O·W_V（F22 验证了等价性和秩）。
8. **多个视角并行（2.8）**：一次线性层算出所有头的 Q/K/V，再用 view+transpose 拆头。“两个等价的公式”应是“concat 后乘 W_o”等于“每个头乘 W_o 的对应列块再求和”（F29）。
9. **搭积木（2.9）**：自己实现 LayerNorm、GELU（tanh 近似）、4 倍宽的 FFN，以及 Pre-LN 残差块。graph.py 生成的 ReLU vs GELU 图应是 2.9.3 的插图。
10. **组装与工程化（2.10）**：token 嵌入 + 位置嵌入 → n_layer 个 Block → 最终 LayerNorm → unembed。“模型的改进”推断包括 weight tying（注释 `# 重み共有`）、N(0, 0.02) 初始化（注释 `# 重みの初期化`）和 dropout，也可能包括改用 PyTorch 内置的 `nn.LayerNorm` / `nn.GELU`：`codebot/model.py` 里被注释掉的 `# LayerNorm(embed_dim)`、`# GELU()` 暗示了这次切换。2.10.3 对应 `save()` / `load_from()`，2.10.4 对应脚本末尾的形状检查。
11. **代码沉淀**：从 09 开始脚本改为 `from codebot.model import ...`，即前面写过的类已经收进 `codebot` 包。这也是 09、10 要 `os.chdir` 到仓库根目录的原因。

---

## 2. 逐脚本详解

> 下面的“实际输出”都是原样运行仓库脚本得到的（`MPLBACKEND=Agg`）。用时是在本机上用 `time uv run python ...` 测的整体墙钟时间，其中约 2–2.5 s 是 Python 启动加 `import torch`。

### 2.1 `ch02/01_soft_dict.py`（§2.1，NumPy）

- **目的**：先用 `d['banana']` 演示硬查找，再实现软字典：用 query 和每个 key 的点积做相似度，softmax 得到权重，再对 value 加权求和。
- **函数**：`soft_dictionary(query, dictionary) -> (result: float, weights: np.ndarray)`
- **核心代码**：

```python
for key in dictionary:
    s = np.dot(query, key)                 # 類似度 → 相似度
    similarity.append(s)
exp_similarity = np.exp(similarity)        # ソフトマックス → softmax（注意：没有减最大值）
weights = exp_similarity / np.sum(exp_similarity)
for weight, value in zip(weights, dictionary.values()):
    result += weight * value               # 重み付き和 → 加权和
```

- **日文注释翻译**：ディクショナリの例 → 字典的例子；キーを指定して値を取得 → 指定 key 取 value；キー: (アクション性, ドラマ性, コメディ性) 各0〜10で表現 → key 为（动作性, 剧情性, 喜剧性），各用 0–10 表示；バリュー: ユーザーの評価点 (0〜100点) → value 为用户评分（0–100 分）；アクション重視の映画 / ドラマ重視 / コメディ重視 / バランスの取れた映画 / アクションドラマ / コメディドラマ / 純粋なアクション → 偏动作 / 偏剧情 / 偏喜剧 / 均衡型 / 动作剧情片 / 喜剧剧情片 / 纯动作片；新しい映画 → 新电影；（输出）の予測評価 → 的预测评分，各映画の重み → 各电影的权重。
- **实际输出**（0.33 s）：

```text
200
新しい映画 (6, 4, 5) の予測評価: 78.66 点

各映画の重み:
映画 (8, 2, 3): 0.49%
映画 (3, 9, 1): 0.00%
映画 (1, 2, 9): 0.00%
映画 (5, 5, 5): 26.71%
映画 (7, 6, 2): 72.62%
映画 (2, 7, 6): 0.18%
映画 (9, 1, 1): 0.00%
```

- **值得补充的点**：分数是 [71, 59, 59, 75, 76, 70, 63]，最大的是 (7,6,2)，而不是距离最近的 (5,5,5)（F02）；未减最大值的 exp 在特征放大 10 倍时会溢出（F04）。

### 2.2 `ch02/02_attn_math.py`（§2.2，torch）

- **目的**：把 01 写成矩阵形式，并支持多个 query：K (7×3)、V (7×1)、Q (3×3)。
- **函数**：`attention(Q, K, V) -> (output (3×1), weights (3×7))`
- **核心代码**：

```python
similarity = torch.matmul(Q, K.t())     # QK^Tを計算 → 计算 QKᵀ
weights = F.softmax(similarity, dim=1)  # ソフトマックス関数 → softmax 函数
output = torch.matmul(weights, V)       # 重み付き和 → 加权和
```

- **注释翻译**：キー（映画のジャンル特性）→ key（电影的类型特征）；バリュー（ユーザーの評価）→ value（用户评分）；新しい映画のジャンル特性（複数のクエリ）→ 新电影的类型特征（多个 query）；バランスの取れたアクション寄りの映画 → 偏动作的均衡型电影；コメディ寄りの映画 → 偏喜剧的电影；結果の表示 → 显示结果。
- **实际输出**（2.92 s）：

```text
映画 [6. 4. 5.] の予測評価: 78.66
映画 [2. 8. 3.] の予測評価: 69.76
映画 [4. 3. 7.] の予測評価: 61.20
```

- **注意**：`K.t()` 只适用于二维张量，批量版本要改成 `K.transpose(-2, -1)`；`dim=1` 在二维时等于最后一维，后面的代码统一写成 `dim=-1`。这里还没有除以 √d。

### 2.3 `ch02/03_attn_scaling.py`（§2.3，torch + NumPy + matplotlib）

- **目的**：先展示 softmax 饱和，再用 10,000 次采样说明：q、k 各分量独立服从 N(0,1) 时，q·k 的方差约为 d，除以 √d 后约为 1（d=10），并画直方图。
- **核心代码**：

```python
x = torch.tensor([100.0, 200.0, 300.0]); y = F.softmax(x, dim=0)
...
dot_product = np.dot(q, k)
scaled_dot_product = dot_product / np.sqrt(d)
...
print("Variances without scaling:", np.var(dot_products))
print("Variances with scaling:", np.var(scaled_dot_products))
```

- **注释翻译**：先ほどの内積の計算を10000回行う → 把刚才的内积计算重复 10000 次；結果をプロット → 绘制结果。
- **实际输出**（3.98 s；**脚本没有设随机种子**，两次运行的数值不同）：

```text
tensor([0.0000e+00, 3.7835e-44, 1.0000e+00])
dot product: 1.4283991204436783
scaled dot product: 0.4516994628383206
Variances without scaling: 9.916118784195705
Variances with scaling: 0.9916118784195703
```

第二次运行：`dot product: -1.4553…`，`Variances without scaling: 10.0104…`，`with scaling: 1.00104…`。两个方差之比恒为 10，这是代数恒等式，与采样无关。在 Agg 后端下，`plt.show()` 只给出一条 `UserWarning: FigureCanvasAgg is non-interactive`。

- **补充**：缩放到底解决了什么，见 §3.2 的饱和与雅可比表（v03b）。

### 2.4 `ch02/06_attn_mask.py`（§2.6，torch）

- **目的**：self-attention 类（Q、K、V 都由同一个 x 线性变换得到），加上下三角因果 mask。
- **类**：`Attention(embed_dim, key_dim)`，其中 `W_q`、`W_k` 为 E→D，**`W_v` 为 E→E**，没有 `W_o`；`forward(x: (B,C,E)) -> (B,C,E)`。
- **核心代码**：

```python
K_t = K.transpose(-2, -1)                              # (B, D, C)
scores = torch.matmul(Q, K_t) / (self.key_dim ** 0.5)   # (B, C, C)
mask = torch.tril(torch.ones(C, C, device=scores.device))
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)                      # (B, C, E)
```

- **注释翻译**：Q, K, Vの変換行列 → Q、K、V 的变换矩阵；Attentionマップの計算 → 计算注意力图（attention map）；マスクの適用 → 应用 mask；入力形状 / 出力形状 → 输入形状 / 输出形状。
- **实际输出**（2.62 s）：

```text
入力形状: torch.Size([2, 5, 256])
出力形状: torch.Size([2, 5, 256])
```

### 2.5 `ch02/07_attn_value.py`（§2.7，torch）

- **目的**：Value 矩阵低秩化。`W_v` 改为 E→D，再加输出变换 `W_o`（D→E）。
- **类**：`Attention(embed_dim, key_dim)`，含 `W_q`/`W_k`/`W_v`: E→D，`W_o`: D→E（`# 出力変換行列` → 输出变换矩阵）。
- **与 06 的差别**：06 的参数为 2·E·D + E² = 98,304，07 为 4·E·D = 65,536（E=256, D=64）；把 06 的 `W_v` 设成 07 的 `W_o·W_v` 后两者输出一致，且该矩阵的秩恰为 64（F22）。
- **实际输出**（2.68 s）：与 06 相同（`[2, 5, 256]` → `[2, 5, 256]`）。

### 2.6 `ch02/08_multi_head.py`（§2.8，torch）

- **前半（函数式，B=2, C=4, E=16, H=3, D=8）**：一次线性层算出所有头的 Q/K/V（注释：効率的な実装：全ヘッド分の重みを一つの行列にまとめる → 高效实现：把所有头的权重合并成一个矩阵），用 view+transpose 拆成 (B,H,C,D)，算 (B,H,C,C) 分数，加 mask，softmax，合并各头，最后乘 `W_o`。注意这里 **H·D = 24 ≠ E = 16**，说明 H·D 不必等于 E，`W_o` 负责映射回 E。
- **后半（类）**：`MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate=0.1)`，比前半多了 `attention_dropout`（作用于注意力权重）和 `output_dropout`（作用于输出）。**与 `codebot/model.py` 中的同名类除注释外完全相同**（AST 比对一致；拷贝 state_dict 后输出差为 0.0）。
- **核心 reshape**：

```python
Q = Q.view(B, C, H, D).transpose(1, 2)            # (B, H, C, D)
scores = torch.matmul(Q, K.transpose(-2, -1))     # (B, H, C, C)
scores = scores / (D ** 0.5)
...
hidden = hidden.transpose(1, 2).contiguous()      # (B, C, H, D)
hidden = hidden.view(B, C, H * D)                 # (B, C, H*D)
output = self.W_o(hidden)                         # (B, C, E)
```

- **注释翻译**：バッチサイズ / コンテキスト長 / 埋め込みの次元数 / ヘッド数 / ヘッドの次元数 → 批大小 / 上下文长度 / 嵌入维数 / 头数 / 每头维数；形状の変換 → 形状变换；マスク処理 → mask 处理；Attention重み → 注意力权重；出力変換 → 输出变换；Dropoutを追加 → 加入 Dropout；各ヘッドに分割して並べ替え → 拆分到各个头并重排；weightsにDropout → 对 weights 做 Dropout；ヘッドの結合と出力変換 → 拼接各头并做输出变换；最終出力にDropout → 对最终输出做 Dropout；使用例 / テスト用データ / 実行 → 使用示例 / 测试数据 / 执行。**术语坑**：类里的注释 `H, D = self.n_head, self.head_dim  # ヘッドサイズ、ヘッドの次元数` 把 H（头数）叫作“ヘッドサイズ”（head size），而 nanoGPT 里的 `head_size` 指的是 D。
- **实际输出**（2.78 s）：

```text
入力形状: torch.Size([2, 10, 512])
出力形状: torch.Size([2, 10, 512])
```

注意：演示是在 train 模式下调用 `mha(x)` 的，dropout 处于开启状态，所以输出带随机性（只打印形状，看不出来，见 F30）。

### 2.7 `ch02/09_norm_gelu.py`（§2.9，torch）

- **目的**：自己实现 `LayerNorm`、`GELU`（tanh 近似）、`FFN`、`Block`（Pre-LN 残差）。`MultiHeadAttention` 从 `codebot.model` 导入，所以脚本开头有 `os.chdir(...)` 与 `sys.path.append('.')`。
- **签名**：`LayerNorm(embed_dim)`（参数为 `gamma`/`beta`，`eps=1e-5`）；`GELU()`；`FFN(x_dim, hidden_dim=None, dropout_rate=0.1)`（hidden 默认 4·x_dim）；`Block(embed_dim, n_head, ff_dim=None, dropout_rate=0.1)`（`head_dim = embed_dim // n_head`）。
- **核心代码**：

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)       # 有偏方差，与 nn.LayerNorm 一致
norm_x = (x - mean) / torch.sqrt(var + self.eps)
return self.gamma * norm_x + self.beta
...
return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
...
x = x + self.attn(self.norm1(x))                         # Pre-LN：先归一化再进子层，再加残差
x = x + self.ffn(self.norm2(x))
```

- **实际输出**：无（只定义类），2.79 s。
- **与 codebot 的差别**：codebot 的 `Block` 用 `nn.LayerNorm`（数值相同，但 state_dict 的键名是 weight/bias 而不是 gamma/beta，F38），`FFN` 用 `nn.GELU()`（erf 版，与这里的 tanh 版最多相差 4.7e-4，在 Block 输出上相差 1.7e-4，F37）。

### 2.8 `ch02/graph.py`（§2.9 插图，NumPy + matplotlib）

- **目的**：在 [−3, 3] 上画 ReLU 和 GELU（tanh 近似）并保存为 `relu_gelu.png`。**它没有 chdir，所以图片会写到当前工作目录**；本研究在 scratch 目录里运行，没有污染仓库。
- **注释翻译**：データ生成 → 生成数据；ReLU関数 / GELU関数（近似式）→ ReLU 函数 / GELU 函数（近似式）；プロット作成 → 绘图；軸の設定 → 设置坐标轴；グリッド → 网格；凡例 → 图例；余白調整 → 调整边距；PNGで保存 → 保存为 PNG。
- **实际结果**（1.05 s）：无标准输出，生成 41,983 字节的 PNG。图中 GELU 在 x≈−0.75 处有最小值约 −0.17，x>2 后贴近 ReLU（数值见 F36）。

### 2.9 `ch02/10_gpt2.py`（§2.10，torch）

- **目的**：组装 GPT-2。流程为 token 嵌入 + 可学习位置嵌入 → dropout → n_layer 个 Block → 最终 LayerNorm → unembed（`nn.Linear`，**带 bias**）。另有 weight tying、N(0, 0.02) 初始化，以及 `save` / `load_from`。`Block` 从 `codebot.model` 导入，所以 10 实际用的是 `nn.LayerNorm` 和 `nn.GELU()`。
- **签名**：`GPT(vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, dropout_rate)`；`forward(ids: LongTensor(B,C)) -> logits (B,C,V)`；`save(file_path)`；`@classmethod load_from(file_path, device='cpu')`。
- **核心代码**：

```python
self.embed = nn.Embedding(vocab_size, embed_dim)
self.pos_embed = nn.Embedding(max_context_len, embed_dim)       # 可学习的绝对位置
...
self.norm = nn.LayerNorm(embed_dim)
self.unembed = nn.Linear(embed_dim, vocab_size)                 # 有 bias
self.embed.weight = self.unembed.weight                         # 重み共有 → 权重共享（同一个 Parameter）
self.apply(self._init_weights)                                  # 重みの初期化 → N(0, 0.02)，bias 置 0
...
pos = torch.arange(0, C, dtype=torch.long, device=device)
x = self.dropout(self.embed(ids) + self.pos_embed(pos))
for block in self.blocks: x = block(x)
logits = self.unembed(self.norm(x))                             # (B, C, vocab_size)
```

- **注释翻译**：語彙サイズ → 词表大小；最大コンテキスト長 → 最大上下文长度；埋め込み次元数 → 嵌入维数；Attentionのヘッド数 → Attention 头数；Transformerブロックの数 → Transformer 块的数量；FFNの隠れ層サイズ → FFN 隐藏层大小；ドロップアウト率 → dropout 率；埋め込み層 / 出力層 → 嵌入层 / 输出层；重み共有 → 权重共享；重みの初期化 → 权重初始化；重みの読み込み → 读取权重；モデルを作成 → 创建模型；動作テスト → 运行测试。
- **实际输出**（3.05 s）：

```text
出力形状: torch.Size([1, 256, 1000])
```

- **参数量**：11,121,640，拆解为 tied 嵌入 384,000 + unembed bias 1,000 + 位置嵌入 98,304 + 6 × 1,772,928 + 最终 LN 768（F41/F42）。
- `10_gpt2.GPT` 与 `codebot.model.GPT` 除注释外完全相同（AST 比对一致；拷贝 state_dict 后输出差为 0.0）。

### 2.10 `codebot/model.py`（本章代码的“定稿”）

- **内容**：`MultiHeadAttention`（与 08 的 AST 一致）、`LayerNorm` / `GELU`（与 09 的 AST 一致，**但在本文件中没有被使用**）、`FFN(embed_dim, hidden_dim, dropout_rate)`（无默认值，激活为 `nn.GELU()  # GELU()`）、`Block(embed_dim, n_head, ff_dim, dropout_rate=0.1)`（`nn.LayerNorm  # LayerNorm(embed_dim)`）、`GPT`（与 10 的 AST 一致）。FFN 和 Block 与 09 不同（见 2.7）。
- **使用者**：ch02/09（MHA）、ch02/10（Block）、ch03/01_pretrain、02_generate、03_sft、04_chat、09_grpo（GPT）。
- **checkpoint 格式**：一个 dict，包含 `model_state_dict` 与 7 个超参（vocab_size、max_context_len、embed_dim、n_head、n_layer、ff_dim、dropout_rate）。state_dict 里同时有 `embed.weight` 和 `unembed.weight` 两个键，但它们指向同一块存储，`torch.save` 只写一份（F43）。官方 `model_pretrain.pt` 的配置为 `{'vocab_size': 1000, 'max_context_len': 256, 'embed_dim': 384, 'n_head': 6, 'n_layer': 6, 'ff_dim': 1536, 'dropout_rate': 0.1}`。torch ≥ 2.6 的 `torch.load` 默认 `weights_only=True`，可以正常加载（已实测）。

### 2.11 变体对照：codebot / storybot / webbot

| | codebot（ch02→ch03） | storybot（ch05→ch06） | webbot（ch09） |
|---|---|---|---|
| 位置信息 | 可学习的绝对位置 `pos_embed` | RoPE（旋转 Q/K，θ=10000） | RoPE |
| 归一化 | LayerNorm（Pre-LN）+ 最终 LN | RMSNorm | RMSNorm |
| FFN | Linear → GELU(erf) → Linear，4E，带 bias，后接 Dropout | SwiGLU，无 bias，默认 8/3·E | SwiGLU（ff=2048） |
| Attention | 手写，tril mask，dropout | 手写，mask，KV cache | SDPA（Flash），GQA（n_kv_head=4），KV cache（解码时 `is_causal=False`，过长时截断 cache） |
| Dropout | 0.1（嵌入、注意力权重、注意力输出、FFN 输出） | 无 | 无 |
| 权重共享 | 是 | 否 | 否 |
| unembed bias | 有 | 无 | 无 |
| 训练配置 | V=1000, ctx=256, E=384, H=6, L=6, ff=1536 | V=10000, ctx=256, E=512, H=16, L=4, ff=1344 | V=50000, ctx=1024, E=768, H=12, KV=4, L=12, ff=2048 |
| 参数量（实测） | 11,121,640 | 22,696,448 | 152,316,672 |
| 嵌入与 unembed 占比 | 3.5% | 45.1% | 50.4% |
| 与 codebot 注意力核心的等价性 | — | `rope=None` 且不用 cache 时，与 codebot MHA 逐位相同（差 0.0） | SDPA(enable_gqa) 与“用 repeat_interleave 复制 KV 头”的手写版一致（1.2e-7） |

（数据来源：`verify/v_variants.py`。RoPE、KV cache、GQA 的细节归 ch05/ch09 档案。）

---

## 3. 可最小化验证的事实清单

### 3.1 事实表（53 条，全部实测）

> 下面每条事实的代码都由 `verify/run_facts.py` 原样执行（CPython 3.12.9，torch 2.9.1+cpu，numpy 1.26.4，Windows，i7-11700F）。标为“浏览器可跑”的条目还由 `pyodide_test/run_facts.mjs` 在 Pyodide 314.0.7（Python 3.14.2，numpy 2.4.6）里实际跑过，输出与 CPython 一致（只有 F27/F29/F52 末位浮点数不同）。用时只计 snippet 本身；新开进程还要加上 `import torch` 约 2–3 s，Pyodide 冷启动约 8.5 s（含从 CDN 下载 numpy）。快照来自运行时的 JSON（`facts_cpython.json` / `facts_pyodide.json`）。

| ID | 节 | 断言 | 依赖 | 浏览器 | 用时 |
|---|---|---|---|---|---|
| F01 | 2.1 | 软字典：新电影 (6,4,5) 的预测评分为 78.66；72.62% 的权重落在 (7,6,2)，26.71% 落在 (5,5,5) | 仅 NumPy | 可（Pyodide 实测） | 0.73 s |
| F02 | 2.1 | 点积相似度对向量长度敏感：点积最大的是 (7,6,2)，但余弦最相似、欧氏距离最近的都是 (5,5,5) | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F03 | 2.1 | “硬查询是软字典在 T→0 时的极限”只在没有并列时成立：用每个键去查，(8,2,3) 得 87.5（应为 85），(5,5,5) 得 73.33（应为 75） | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F04 | 2.1 | 01 里的朴素 softmax（不减最大值）在分数约 710 时溢出成 NaN；减去最大值的稳定写法不会 | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F05 | 2.1 | softmax 对平移不变，对缩放敏感 | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F06 | 2.2 | 矩阵形式 softmax(QKᵀ)V 复现 02 的三个预测 78.66 / 69.76 / 61.20，且每行权重之和为 1 | 仅 NumPy | 可（Pyodide 实测） | 0.01 s |
| F07 | 2.3 | softmax([100,200,300]) 饱和（float32 下为 [0, 3.8e-44, 1]）；先除以 100 则得 [0.090, 0.245, 0.665] | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F08 | 2.3 | q、k ~ N(0, I_d) 时 Var(q·k) ≈ d，除以 √d 后 ≈ 1（两者之比恰为 d） | 仅 NumPy | 可（Pyodide 实测） | 0.05 s |
| F09 | 2.3 | 不缩放时 d 越大，256 个 key 上的注意力越尖锐（饱和行占比上升）；除以 √d 后尖锐程度与 d 无关 | 仅 NumPy | 可（Pyodide 实测） | 2.07 s |
| F10 | 2.3 | 初始化（std 0.02）时，真实模型第 0 层的分数很小（std≈0.15），注意力几乎均匀，此时缩放与否影响不大 | torch | 否 | 9.12 s |
| F11 | 2.5 | 没有 mask 也没有位置信息时，attention 对输入顺序是置换等变的；因果 mask 打破了这一点 | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F12 | 2.5 | 正弦位置编码：PE(p)·PE(p+k) 只取决于 k；\|PE(p)\|² = d/2 | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F13 | 2.5 | 训练后模型的可学习位置嵌入是平滑的：间隔 1/8/64 的平均余弦为 0.949 / 0.562 / 0.037（随机初始化约为 0） | torch + 训练好的 checkpoint | 否 | 0.47 s |
| F14 | 2.5 | 可学习的绝对位置嵌入限定了上下文长度：向 max_context_len=256 的模型输入 257 个 token 会触发 IndexError | torch | 否 | 0.33 s |
| F15 | 2.6 | 因果 softmax：上三角恰为 0，每行之和为 1，第 0 行为 [1,0,...] | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F16 | 2.6 | 对书中 06 的 Attention 做因果性测试：改动第 3、4 个 token 后，位置 0–2 的输出逐位不变 | torch | 否 | 0.02 s |
| F17 | 2.6 | 06 的 Attention 与 F.scaled_dot_product_attention(is_causal=True) 一致（value 维度 256 ≠ key 维度 64 也可以） | torch | 否 | 0.02 s |
| F18 | 2.6 | 一整行都被 mask 时 softmax 输出 NaN（tril 不会出现，padding mask 可能出现） | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F19 | 2.6 | softmax 做在错误的轴上时，即使被遮住的元素为 0，也会泄露未来信息 | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F20 | 2.6 | 为什么需要 mask（2.6.1–2.6.2）：在 i.i.d. 随机 token 上，因果模型只能停在 ln V，去掉 mask 的模型能“作弊”降到 ln(V)/C | torch | 否 | 13.69 s |
| F21 | 2.6 | SDPA 的 is_causal=True 采用左上角对齐：1 个 query 对 5 个缓存 key 时只能看到 key 0，所以 KV-cache 解码必须用 is_causal=False | torch | 否 | 0.0 s |
| F22 | 2.7 | 07 等价于把 06 的 E×E value 矩阵换成 W_o·W_v（秩 64），参数从 98,304 降到 65,536 | torch | 否 | 0.04 s |
| F23 | 2.7 | 位置 0 只能看到自己，因此它的输出恰好是自己的 value 向量 | torch | 否 | 0.01 s |
| F24 | 2.8 | 08 的形状变化：(2,4,24) → view → (2,4,3,8) → transpose → (2,3,4,8)，此时张量已不连续（non-contiguous） | torch | 否 | 0.0 s |
| F25 | 2.8 | transpose 之后直接 .view 会报 RuntimeError；.contiguous().view 与 .reshape 结果相同 | torch | 否 | 0.0 s |
| F26 | 2.8 | 批量多头（reshape+transpose）与按权重行切片逐头计算的结果完全一致 | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F27 | 2.8 | 拆头时用 reshape(H,C,D) 而不先 transpose，会悄悄破坏因果性 | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F28 | 2.8 | codebot 的 MultiHeadAttention、SDPA(is_causal=True) 路径、拷贝权重后的 torch.nn.MultiheadAttention 三者一致 | torch | 否 | 0.08 s |
| F29 | 2.8 | 2.8.3 的“两个等价公式”：concat(heads)·W_oᵀ = Σ_h head_h·W_o[:, hD:(h+1)D]ᵀ | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F30 | 2.8 | 模块创建后默认处于 train 模式：dropout 让两次前向结果不同，被 dropout 的注意力行之和也不再是 1 | torch | 否 | 0.05 s |
| F31 | 2.8 | MHA 参数量 = 4·E·(H·D) = 1,048,576（E=512, H=8, D=64，无 bias） | torch | 否 | 0.01 s |
| F32 | 2.9 | 书中的 LayerNorm 与 nn.LayerNorm 一致（eps 1e-5，有偏方差） | torch | 否 | 0.01 s |
| F33 | 2.9 | 若误用无偏方差，LN 输出会整体缩放 √((E−1)/E) = 0.998697（E=384） | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F34 | 2.9 | LN（γ=1, β=0）之后，每个 token 向量的均值为 0、方差为 1（扣除 eps 的影响） | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F35 | 2.9 | LayerNorm 按 token 归一化，样本的输出与 batch 里其他样本无关（BatchNorm 则有关） | torch | 否 | 0.0 s |
| F36 | 2.9 | tanh 近似 GELU 与精确（erf）GELU 最多相差 4.73e-4；GELU 最小值为 −0.16997，位于 x = −0.752 | 仅 NumPy | 可（Pyodide 实测） | 0.08 s |
| F37 | 2.9 | 书中的 GELU = nn.GELU(approximate='tanh')；而 codebot 的 FFN 用的是 nn.GELU()，也就是 erf 版本 | torch | 否 | 0.01 s |
| F38 | 2.9 | 09 的 Block（gamma/beta）与 codebot 的 Block（weight/bias）的 state_dict 不改名就不能互相加载 | torch | 否 | 0.08 s |
| F39 | 2.9 | 残差流：把 W_o 和 FFN 输出层置零后，Block 严格等于恒等映射 | torch | 否 | 0.04 s |
| F40 | 2.9 | 一个 Block（E=384, H=6, ff=4E）有 1,772,928 个参数 = 2 个 LN + 4E² + FFN（约等于 12E² = 1,769,472） | torch | 否 | 0.03 s |
| F41 | 2.10 | ch02/ch03 所用 GPT 的参数量闭式解：11,121,640（不共享权重则为 11,505,640） | 仅 NumPy | 可（Pyodide 实测） | 0.0 s |
| F42 | 2.10 | 对 parameters() 求 numel 之和得 11,121,640（共享矩阵只算一次）；对 state_dict 求和会重复计算；embed.weight 与 unembed.weight 是同一个对象 | torch | 否 | 0.31 s |
| F43 | 2.10 | 保存的 checkpoint 约 44.5 MB ≈ 4 字节 × 11.12M 参数，说明共享矩阵只存了一份（官方 model_pretrain.pt 为 44,516,353 字节，HTTP HEAD 得到） | torch | 否 | 0.44 s |
| F44 | 2.10 | 初始化：Linear/Embedding ~ N(0, 0.02)，bias 为 0，LayerNorm 为 1/0；对共享矩阵而言，Linear 的初始化最后执行，因此最终生效 | torch | 否 | 0.71 s |
| F45 | 2.10 | 初始化时的 loss 约为 6.98 = ln V + σ²/2（σ 是 logits 的标准差，约 0.39），略高于 ln(1000) = 6.9078 | torch | 否 | 0.92 s |
| F46 | 2.10 | 整模型因果性：改动第 100 个 token，位置 0–99 的 logits 逐位不变 | torch | 否 | 0.57 s |
| F47 | 2.10 | eval() 下模型输出确定且与 batch 无关；train()（有 dropout）下不确定 | torch | 否 | 1.22 s |
| F48 | 2.10 | HF GPT-2 small 有 124,439,808 个参数；同样超参的 codebot GPT 有 124,453,201 个（多 50,257 个 lm_head bias，少 36,864 个 qkv/proj bias） | torch + transformers | 否 | 11.04 s |
| F49 | 2.10 | 把 HF GPT-2 的权重拷进 codebot GPT（qkv/proj bias 置零）后，logits 相差 3.5e-4（erf GELU）；换成 tanh GELU 后相差 1.9e-6 | torch + transformers | 否 | 1.51 s |
| F50 | 2.10 | 训练好的 codebot：代码片段的精确 token id；在 32 个训练窗口上 CE 为 0.40（新初始化约 6.94–6.98）；从 'def' 开始的贪心生成 | torch + 训练好的 checkpoint | 否 | 3.29 s |
| F51 | 2.10 | 训练后的注意力头（ch02_attn_maps.json）：L4H3 是 previous-token head（' b'→' +' 权重 0.99）；第 1 层约一半权重落在第一个 token 上（attention sink） | 仅 NumPy | 可（Pyodide 实测） | 0.01 s |
| F52 | 2.10 | 纯 NumPy 的 GPT 前向（np_gpt.py）能复现小型 codebot GPT 的 torch logits，误差约 1e-7 | 仅 NumPy | 可（Pyodide 实测） | 0.05 s |
| F53 | 2.10 | 端到端自检：11.1M 参数的 GPT 能在 CPU 上过拟合一个固定 batch 的真实 TinyCodes token（50 步内从 7.0 降到约 0.02） | torch | 否 | 28.0 s |

#### F01（§2.1）软字典：新电影 (6,4,5) 的预测评分为 78.66；72.62% 的权重落在 (7,6,2)，26.71% 落在 (5,5,5)

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.73 s

```python
import numpy as np
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float)
V = np.array([85,70,60,75,80,65,90], float)
s = K @ np.array([6., 4., 5.])
w = np.exp(s - s.max()); w /= w.sum()
print((w * 100).round(2), round(float(w @ V), 2))
```

实测输出：

```text
[ 0.49  0.    0.   26.71 72.62  0.18  0.  ] 78.66
```

Pyodide：输出逐字相同（2.02 s）。

#### F02（§2.1）点积相似度对向量长度敏感：点积最大的是 (7,6,2)，但余弦最相似、欧氏距离最近的都是 (5,5,5)

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float); q = np.array([6., 4., 5.])
print("dot:", K[np.argmax(K @ q)], "cos:", K[np.argmax(K @ q / np.linalg.norm(K, axis=1))],
      "nearest:", K[np.argmin(np.linalg.norm(K - q, axis=1))])
```

实测输出：

```text
dot: [7. 6. 2.] cos: [5. 5. 5.] nearest: [5. 5. 5.]
```

Pyodide：输出逐字相同（0.01 s）。

#### F03（§2.1）“硬查询是软字典在 T→0 时的极限”只在没有并列时成立：用每个键去查，(8,2,3) 得 87.5（应为 85），(5,5,5) 得 73.33（应为 75）

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float); V = np.array([85,70,60,75,80,65,90.])
def soft(q, T):
    s = K @ q / T; w = np.exp(s - s.max()); return float((w / w.sum()) @ V)
print([round(soft(k, 0.001), 2) for k in K], "true:", V.tolist())
```

实测输出：

```text
[87.5, 70.0, 60.0, 73.33, 80.0, 65.0, 90.0] true: [85.0, 70.0, 60.0, 75.0, 80.0, 65.0, 90.0]
```

Pyodide：输出逐字相同（0.01 s）。

#### F04（§2.1）01 里的朴素 softmax（不减最大值）在分数约 710 时溢出成 NaN；减去最大值的稳定写法不会

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
s = np.array([71., 59, 59, 75, 76, 70, 63]) * 10        # 01's scores with features on a 0..100 scale
with np.errstate(over="ignore", invalid="ignore"):
    print("naive :", np.exp(s) / np.exp(s).sum())
e = np.exp(s - s.max()); print("stable:", (e / e.sum()).round(4))
```

实测输出：

```text
naive : [nan  0.  0. nan nan  0.  0.]
stable: [0. 0. 0. 0. 1. 0. 0.]
```

Pyodide：输出逐字相同（0.01 s）。

#### F05（§2.1）softmax 对平移不变，对缩放敏感

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
sm = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
x = np.array([1., 2., 3.])
print(sm(x).round(4), sm(x + 100).round(4), sm(100 * x).round(4))
```

实测输出：

```text
[0.09   0.2447 0.6652] [0.09   0.2447 0.6652] [0. 0. 1.]
```

Pyodide：输出逐字相同（0.01 s）。

#### F06（§2.2）矩阵形式 softmax(QKᵀ)V 复现 02 的三个预测 78.66 / 69.76 / 61.20，且每行权重之和为 1

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.01 s

```python
import numpy as np
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float); V = np.array([[85],[70],[60],[75],[80],[65],[90.]])
Q = np.array([[6,4,5],[2,8,3],[4,3,7]], float)
S = Q @ K.T; W = np.exp(S - S.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
print(S.shape, W.sum(1), (W @ V).ravel().round(2))
```

实测输出：

```text
(3, 7) [1. 1. 1.] [78.66 69.76 61.2 ]
```

Pyodide：输出逐字相同（0 s）。

#### F07（§2.3）softmax([100,200,300]) 饱和（float32 下为 [0, 3.8e-44, 1]）；先除以 100 则得 [0.090, 0.245, 0.665]

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
for x in (np.array([100., 200., 300.], dtype=np.float32), np.array([1., 2., 3.])):
    e = np.exp(x - x.max()); print(e / e.sum())
```

实测输出：

```text
[0.0e+00 3.8e-44 1.0e+00]
[0.09003057 0.24472847 0.66524096]
```

Pyodide：输出逐字相同（0 s）。

#### F08（§2.3）q、k ~ N(0, I_d) 时 Var(q·k) ≈ d，除以 √d 后 ≈ 1（两者之比恰为 d）

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.05 s

```python
import numpy as np
rng = np.random.default_rng(0); d = 64
q, k = rng.standard_normal((10000, d)), rng.standard_normal((10000, d))
dots = (q * k).sum(1)
print(round(dots.var(), 2), round((dots / np.sqrt(d)).var(), 4), round(dots.var() / (dots / np.sqrt(d)).var(), 6))
```

实测输出：

```text
65.29 1.0201 64.0
```

Pyodide：输出逐字相同（0.17 s）。

#### F09（§2.3）不缩放时 d 越大，256 个 key 上的注意力越尖锐（饱和行占比上升）；除以 √d 后尖锐程度与 d 无关

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 2.07 s

```python
import numpy as np
rng = np.random.default_rng(0)
for d in (16, 64, 256, 1024):
    raw = np.array([rng.standard_normal((256, d)) @ rng.standard_normal(d) for _ in range(200)])
    out = []
    for z in (raw, raw / np.sqrt(d)):
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        out.append(f"maxw={p.max(1).mean():.3f} sat={(p.max(1) > .99).mean():.2f}")
    print(f"d={d:4d}  unscaled: {out[0]}  |  scaled: {out[1]}")
```

实测输出：

```text
d=  16  unscaled: maxw=0.514 sat=0.01  |  scaled: maxw=0.045 sat=0.00
d=  64  unscaled: maxw=0.759 sat=0.14  |  scaled: maxw=0.045 sat=0.00
d= 256  unscaled: maxw=0.892 sat=0.44  |  scaled: maxw=0.044 sat=0.00
d=1024  unscaled: maxw=0.946 sat=0.70  |  scaled: maxw=0.043 sat=0.00
```

Pyodide：输出逐字相同（3.74 s）。

#### F10（§2.3）初始化（std 0.02）时，真实模型第 0 层的分数很小（std≈0.15），注意力几乎均匀，此时缩放与否影响不大

依赖：torch；浏览器：否；CPython 用时 9.12 s

```python
import torch, torch.nn.functional as F
from codebot.model import GPT
torch.manual_seed(0); m = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval(); a = m.blocks[0].attn
ids = torch.randint(0, 1000, (1, 256))
with torch.no_grad():
    h = m.blocks[0].norm1(m.embed(ids) + m.pos_embed(torch.arange(256)))
    q, k = (l(h).view(1, 256, 6, 64).transpose(1, 2) for l in (a.W_q, a.W_k))
    s = q @ k.transpose(-2, -1) / 8
print("score std:", round(s.std().item(), 4), "| last-row max weight:", round(F.softmax(s[0, 0, -1], -1).max().item(), 5), "vs 1/256 =", round(1 / 256, 5))
```

实测输出：

```text
score std: 0.1513 | last-row max weight: 0.00663 vs 1/256 = 0.00391
```

#### F11（§2.5）没有 mask 也没有位置信息时，attention 对输入顺序是置换等变的；因果 mask 打破了这一点

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
rng = np.random.default_rng(0); C, E = 5, 8
Wq, Wk, Wv = (rng.standard_normal((E, E)) for _ in range(3)); x = rng.standard_normal((C, E))
def attn(x, causal):
    s = (x @ Wq) @ (x @ Wk).T / np.sqrt(E)
    if causal: s = np.where(np.tril(np.ones((C, C), bool)), s, -np.inf)
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); return p @ (x @ Wv)
perm = [4, 2, 0, 1, 3]
for causal in (False, True):
    print("causal" if causal else "no mask", np.abs(attn(x[perm], causal) - attn(x, causal)[perm]).max())
```

实测输出：

```text
no mask 8.881784197001252e-16
causal 7.868527045300903
```

Pyodide：输出逐字相同（0.01 s）。

#### F12（§2.5）正弦位置编码：PE(p)·PE(p+k) 只取决于 k；|PE(p)|² = d/2

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
pos, i = np.arange(256)[:, None], np.arange(32)[None, :]
pe = np.zeros((256, 64)); ang = pos / 10000 ** (2 * i / 64); pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang)
print([round(float(pe[p] @ pe[p + 5]), 6) for p in (0, 37, 150)], (pe ** 2).sum(1)[:3])
```

实测输出：

```text
[23.503971, 23.503971, 23.503971] [32. 32. 32.]
```

Pyodide：输出逐字相同（0.01 s）。

#### F13（§2.5）训练后模型的可学习位置嵌入是平滑的：间隔 1/8/64 的平均余弦为 0.949 / 0.562 / 0.037（随机初始化约为 0）

依赖：torch + 训练好的 checkpoint；浏览器：否；CPython 用时 0.47 s

```python
import torch, torch.nn.functional as F
from codebot.model import GPT
from bookload import checkpoint
P = F.normalize(GPT.load_from(checkpoint()).pos_embed.weight.detach(), dim=-1)
print({k: round((P[:-k] * P[k:]).sum(-1).mean().item(), 3) for k in (1, 8, 64)})
```

实测输出：

```text
{1: 0.949, 8: 0.562, 64: 0.037}
```

#### F14（§2.5）可学习的绝对位置嵌入限定了上下文长度：向 max_context_len=256 的模型输入 257 个 token 会触发 IndexError

依赖：torch；浏览器：否；CPython 用时 0.33 s

```python
import torch
from codebot.model import GPT
m = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
try: m(torch.randint(0, 1000, (1, 257)))
except IndexError as e: print("IndexError:", e)
```

实测输出：

```text
IndexError: index out of range in self
```

#### F15（§2.6）因果 softmax：上三角恰为 0，每行之和为 1，第 0 行为 [1,0,...]

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
C = 5; s = np.random.default_rng(0).standard_normal((C, C))
s = np.where(np.tril(np.ones((C, C), bool)), s, -np.inf)
w = np.exp(s - s.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
print(w.round(3)); print("upper==0:", bool((np.triu(w, 1) == 0).all()), "row sums:", w.sum(1))
```

实测输出：

```text
[[1.    0.    0.    0.    0.   ]
 [0.28  0.72  0.    0.    0.   ]
 [0.32  0.622 0.058 0.    0.   ]
 [0.146 0.176 0.221 0.457 0.   ]
 [0.096 0.426 0.056 0.154 0.268]]
upper==0: True row sums: [1. 1. 1. 1. 1.]
```

Pyodide：输出逐字相同（0.01 s）。

#### F16（§2.6）对书中 06 的 Attention 做因果性测试：改动第 3、4 个 token 后，位置 0–2 的输出逐位不变

依赖：torch；浏览器：否；CPython 用时 0.02 s

```python
import torch
from bookload import load
torch.manual_seed(0); att = load("ch02/06_attn_mask.py").Attention(256, 64)
x = torch.randn(2, 5, 256); x2 = x.clone(); x2[:, 3:] = torch.randn(2, 2, 256)
d = (att(x) - att(x2)).abs()
print("positions 0-2:", d[:, :3].max().item(), "| positions 3-4:", round(d[:, 3:].max().item(), 3))
```

实测输出：

```text
positions 0-2: 0.0 | positions 3-4: 0.857
```

#### F17（§2.6）06 的 Attention 与 F.scaled_dot_product_attention(is_causal=True) 一致（value 维度 256 ≠ key 维度 64 也可以）

依赖：torch；浏览器：否；CPython 用时 0.02 s

```python
import torch, torch.nn.functional as F
from bookload import load
torch.manual_seed(0); att = load("ch02/06_attn_mask.py").Attention(256, 64); x = torch.randn(2, 5, 256)
with torch.no_grad():
    ref = F.scaled_dot_product_attention(att.W_q(x), att.W_k(x), att.W_v(x), is_causal=True)
    print(tuple(ref.shape), (att(x) - ref).abs().max().item())
```

实测输出：

```text
(2, 5, 256) 1.1920928955078125e-07
```

#### F18（§2.6）一整行都被 mask 时 softmax 输出 NaN（tril 不会出现，padding mask 可能出现）

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
s = np.full(3, -np.inf)
with np.errstate(invalid="ignore"): print(np.exp(s - s.max()) / np.exp(s - s.max()).sum())
```

实测输出：

```text
[nan nan nan]
```

Pyodide：输出逐字相同（0 s）。

#### F19（§2.6）softmax 做在错误的轴上时，即使被遮住的元素为 0，也会泄露未来信息

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
rng = np.random.default_rng(0); C, E = 5, 8
Wq, Wk, Wv = (rng.standard_normal((E, E)) for _ in range(3))
x = rng.standard_normal((C, E)); x2 = x.copy(); x2[3:] = rng.standard_normal((2, E))
def attn(x, axis):
    s = np.where(np.tril(np.ones((C, C), bool)), (x @ Wq) @ (x @ Wk).T / np.sqrt(E), -np.inf)
    p = np.exp(s - s.max(axis, keepdims=True)); p /= p.sum(axis, keepdims=True); return p @ (x @ Wv)
for axis in (-1, -2): print("softmax axis", axis, "-> max change at positions 0-2:", np.abs(attn(x, axis) - attn(x2, axis))[:3].max())
```

实测输出：

```text
softmax axis -1 -> max change at positions 0-2: 0.0
softmax axis -2 -> max change at positions 0-2: 5.1186863891717005
```

Pyodide：输出逐字相同（0.01 s）。

#### F20（§2.6）为什么需要 mask（2.6.1–2.6.2）：在 i.i.d. 随机 token 上，因果模型只能停在 ln V，去掉 mask 的模型能“作弊”降到 ln(V)/C

依赖：torch；浏览器：否；CPython 用时 13.69 s

```python
import math
from v_cheat import run
print("causal:", round(run(True).mean().item(), 4), "| no mask:", round(run(False).mean().item(), 4),
      "| ln 8 =", round(math.log(8), 4), "| ln(8)/8 =", round(math.log(8) / 8, 4))
```

实测输出：

```text
causal: 2.0803 | no mask: 0.2637 | ln 8 = 2.0794 | ln(8)/8 = 0.2599
```

#### F21（§2.6）SDPA 的 is_causal=True 采用左上角对齐：1 个 query 对 5 个缓存 key 时只能看到 key 0，所以 KV-cache 解码必须用 is_causal=False

依赖：torch；浏览器：否；CPython 用时 0.0 s

```python
import torch, torch.nn.functional as F
torch.manual_seed(0); q, K, V = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8)
print(torch.allclose(F.scaled_dot_product_attention(q, K, V, is_causal=True), V[:, :, :1]))
```

实测输出：

```text
True
```

#### F22（§2.7）07 等价于把 06 的 E×E value 矩阵换成 W_o·W_v（秩 64），参数从 98,304 降到 65,536

依赖：torch；浏览器：否；CPython 用时 0.04 s

```python
import torch
from bookload import load
torch.manual_seed(0)
a7, a6 = load("ch02/07_attn_value.py").Attention(256, 64), load("ch02/06_attn_mask.py").Attention(256, 64)
with torch.no_grad():
    a6.W_q.weight.copy_(a7.W_q.weight); a6.W_k.weight.copy_(a7.W_k.weight)
    a6.W_v.weight.copy_(a7.W_o.weight @ a7.W_v.weight)
    x = torch.randn(2, 5, 256)
    print((a7(x) - a6(x)).abs().max().item(), torch.linalg.matrix_rank(a6.W_v.weight).item(),
          sum(p.numel() for p in a6.parameters()), sum(p.numel() for p in a7.parameters()))
```

实测输出：

```text
2.980232238769531e-07 64 98304 65536
```

#### F23（§2.7）位置 0 只能看到自己，因此它的输出恰好是自己的 value 向量

依赖：torch；浏览器：否；CPython 用时 0.01 s

```python
import torch
from bookload import load
torch.manual_seed(0); att = load("ch02/06_attn_mask.py").Attention(256, 64); x = torch.randn(2, 5, 256)
with torch.no_grad(): print((att(x)[:, 0] - att.W_v(x)[:, 0]).abs().max().item())
```

实测输出：

```text
0.0
```

#### F24（§2.8）08 的形状变化：(2,4,24) → view → (2,4,3,8) → transpose → (2,3,4,8)，此时张量已不连续（non-contiguous）

依赖：torch；浏览器：否；CPython 用时 0.0 s

```python
import torch, torch.nn as nn
B, C, E, H, D = 2, 4, 16, 3, 8
Q = nn.Linear(E, H * D, bias=False)(torch.randn(B, C, E)); Qh = Q.view(B, C, H, D).transpose(1, 2)
print(tuple(Q.shape), tuple(Q.view(B, C, H, D).shape), tuple(Qh.shape), Qh.is_contiguous())
```

实测输出：

```text
(2, 4, 24) (2, 4, 3, 8) (2, 3, 4, 8) False
```

#### F25（§2.8）transpose 之后直接 .view 会报 RuntimeError；.contiguous().view 与 .reshape 结果相同

依赖：torch；浏览器：否；CPython 用时 0.0 s

```python
import torch
h = torch.randn(2, 8, 10, 64).transpose(1, 2)
try: h.view(2, 10, 512)
except RuntimeError as e: print("RuntimeError:", str(e)[:58])
print(torch.equal(h.reshape(2, 10, 512), h.contiguous().view(2, 10, 512)))
```

实测输出：

```text
RuntimeError: view size is not compatible with input tensor's size and s
True
```

#### F26（§2.8）批量多头（reshape+transpose）与按权重行切片逐头计算的结果完全一致

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
rng = np.random.default_rng(0); C, E, H = 6, 16, 4; D = E // H
Wq, Wk, Wv = (rng.standard_normal((H * D, E)) for _ in range(3)); x = rng.standard_normal((C, E))
mask = np.tril(np.ones((C, C), bool))
def attn(q, k, v):
    s = np.where(mask, q @ np.swapaxes(k, -1, -2) / np.sqrt(D), -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); return p @ v
split = lambda t: t.reshape(C, H, D).transpose(1, 0, 2)                      # (C, H*D) -> (H, C, D)
batched = attn(split(x @ Wq.T), split(x @ Wk.T), split(x @ Wv.T))
loop = np.stack([attn(*(x @ W[h*D:(h+1)*D].T for W in (Wq, Wk, Wv))) for h in range(H)])
print(batched.shape, np.abs(batched - loop).max())
```

实测输出：

```text
(4, 6, 4) 0.0
```

Pyodide：输出逐字相同（0.01 s）。

#### F27（§2.8）拆头时用 reshape(H,C,D) 而不先 transpose，会悄悄破坏因果性

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
rng = np.random.default_rng(0); C, E, H = 6, 16, 4; D = E // H
Wq, Wk, Wv = (rng.standard_normal((H * D, E)) for _ in range(3))
x = rng.standard_normal((C, E)); x2 = x.copy(); x2[4:] = rng.standard_normal((2, E))
def mha(x, split):
    q, k, v = (split(x @ W.T) for W in (Wq, Wk, Wv))
    s = np.where(np.tril(np.ones((C, C), bool)), q @ k.transpose(0, 2, 1) / np.sqrt(D), -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); return (p @ v).transpose(1, 0, 2).reshape(C, E)
right = lambda t: t.reshape(C, H, D).transpose(1, 0, 2)
wrong = lambda t: t.reshape(H, C, D)
for name, sp in (("right", right), ("wrong", wrong)): print(name, "max change at positions 0-3:", np.abs(mha(x, sp) - mha(x2, sp))[:4].max())
```

实测输出：

```text
right max change at positions 0-3: 0.0
wrong max change at positions 0-3: 7.945126243720273
```

Pyodide 输出（0.01 s，仅末位浮点差异）：

```text
right max change at positions 0-3: 0.0
wrong max change at positions 0-3: 7.945126243720274
```

#### F28（§2.8）codebot 的 MultiHeadAttention、SDPA(is_causal=True) 路径、拷贝权重后的 torch.nn.MultiheadAttention 三者一致

依赖：torch；浏览器：否；CPython 用时 0.08 s

```python
import torch, torch.nn as nn, torch.nn.functional as F
from codebot.model import MultiHeadAttention
torch.manual_seed(0); m = MultiHeadAttention(512, 8, 64).eval(); x = torch.randn(2, 10, 512)
with torch.no_grad():
    Q, K, V = (l(x).view(2, 10, 8, 64).transpose(1, 2) for l in (m.W_q, m.W_k, m.W_v))
    sdpa = m.W_o(F.scaled_dot_product_attention(Q, K, V, is_causal=True).transpose(1, 2).reshape(2, 10, 512))
    t = nn.MultiheadAttention(512, 8, bias=False, batch_first=True).eval()
    t.in_proj_weight.copy_(torch.cat([m.W_q.weight, m.W_k.weight, m.W_v.weight])); t.out_proj.weight.copy_(m.W_o.weight)
    ref, _ = t(x, x, x, attn_mask=torch.triu(torch.ones(10, 10, dtype=torch.bool), 1), need_weights=False)
    print((m(x) - sdpa).abs().max().item(), (m(x) - ref).abs().max().item())
```

实测输出：

```text
1.341104507446289e-07 1.341104507446289e-07
```

#### F29（§2.8）2.8.3 的“两个等价公式”：concat(heads)·W_oᵀ = Σ_h head_h·W_o[:, hD:(h+1)D]ᵀ

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
rng = np.random.default_rng(0); C, H, D, E = 6, 4, 8, 32
heads = rng.standard_normal((H, C, D)); Wo = rng.standard_normal((E, H * D))
concat = heads.transpose(1, 0, 2).reshape(C, H * D) @ Wo.T
summed = sum(heads[h] @ Wo[:, h*D:(h+1)*D].T for h in range(H))
print(np.abs(concat - summed).max())
```

实测输出：

```text
3.552713678800501e-15
```

Pyodide 输出（0 s，仅末位浮点差异）：

```text
7.105427357601002e-15
```

#### F30（§2.8）模块创建后默认处于 train 模式：dropout 让两次前向结果不同，被 dropout 的注意力行之和也不再是 1

依赖：torch；浏览器：否；CPython 用时 0.05 s

```python
import torch, torch.nn.functional as F
from codebot.model import MultiHeadAttention
torch.manual_seed(0); m = MultiHeadAttention(512, 8, 64); x = torch.randn(2, 10, 512)
print("training:", m.training, "| equal:", torch.equal(m(x), m(x)), end=" ")
m.eval(); print("| after eval():", torch.equal(m(x), m(x)))
print(torch.nn.Dropout(0.1)(F.softmax(torch.randn(4, 10), -1)).sum(-1))
```

实测输出：

```text
training: True | equal: False | after eval(): True
tensor([0.9292, 0.9914, 1.0367, 1.1111])
```

#### F31（§2.8）MHA 参数量 = 4·E·(H·D) = 1,048,576（E=512, H=8, D=64，无 bias）

依赖：torch；浏览器：否；CPython 用时 0.01 s

```python
from codebot.model import MultiHeadAttention
print(sum(p.numel() for p in MultiHeadAttention(512, 8, 64).parameters()), 4 * 512 * 8 * 64)
```

实测输出：

```text
1048576 1048576
```

#### F32（§2.9）书中的 LayerNorm 与 nn.LayerNorm 一致（eps 1e-5，有偏方差）

依赖：torch；浏览器：否；CPython 用时 0.01 s

```python
import torch, torch.nn as nn
from bookload import load
torch.manual_seed(0); ln = load("ch02/09_norm_gelu.py").LayerNorm(384); ref = nn.LayerNorm(384)
with torch.no_grad():
    g, b = torch.randn(384), torch.randn(384); ln.gamma.copy_(g); ln.beta.copy_(b); ref.weight.copy_(g); ref.bias.copy_(b)
    x = torch.randn(4, 16, 384) * 3 + 1.5
    print((ln(x) - ref(x)).abs().max().item())
```

实测输出：

```text
9.5367431640625e-07
```

#### F33（§2.9）若误用无偏方差，LN 输出会整体缩放 √((E−1)/E) = 0.998697（E=384）

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
x = np.random.default_rng(0).standard_normal((4, 384)) * 3 + 1.5
z = lambda ddof: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, ddof=ddof, keepdims=True) + 1e-5)
print(round(float((z(1) / z(0)).mean()), 6), round(float(np.sqrt(383 / 384)), 6))
```

实测输出：

```text
0.998697 0.998697
```

Pyodide：输出逐字相同（0.01 s）。

#### F34（§2.9）LN（γ=1, β=0）之后，每个 token 向量的均值为 0、方差为 1（扣除 eps 的影响）

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
import numpy as np
x = np.random.default_rng(0).standard_normal((5, 384)) * 3 + 2
y = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
print(np.abs(y.mean(-1)).max(), y.var(-1).round(6))
```

实测输出：

```text
1.0639637319324417e-16 [0.999999 0.999999 0.999999 0.999999 0.999999]
```

Pyodide：输出逐字相同（0.01 s）。

#### F35（§2.9）LayerNorm 按 token 归一化，样本的输出与 batch 里其他样本无关（BatchNorm 则有关）

依赖：torch；浏览器：否；CPython 用时 0.0 s

```python
import torch, torch.nn as nn
torch.manual_seed(0); x = torch.randn(4, 384) * 3 + 1.5
ln, bn = nn.LayerNorm(384), nn.BatchNorm1d(384).train()
with torch.no_grad():
    other = torch.cat([x[:1], torch.randn(3, 384)])
    print("LN:", (ln(x)[0] - ln(other)[0]).abs().max().item(), "| BN:", round((bn(x)[0] - bn(other)[0]).abs().max().item(), 3))
```

实测输出：

```text
LN: 0.0 | BN: 3.033
```

#### F36（§2.9）tanh 近似 GELU 与精确（erf）GELU 最多相差 4.73e-4；GELU 最小值为 −0.16997，位于 x = −0.752

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.08 s

```python
import math, numpy as np
t = np.linspace(-6, 6, 120001)
exact = 0.5 * t * (1 + np.vectorize(math.erf)(t / math.sqrt(2)))
tanh = 0.5 * t * (1 + np.tanh(math.sqrt(2 / math.pi) * (t + 0.044715 * t ** 3)))
i = exact.argmin()
print(f"max|tanh-erf|={np.abs(tanh - exact).max():.3e}  min={exact[i]:.5f} at x={t[i]:.3f}  GELU(0)={exact[60000]:.1f}")
```

实测输出：

```text
max|tanh-erf|=4.732e-04  min=-0.16997 at x=-0.752  GELU(0)=0.0
```

Pyodide：输出逐字相同（0.15 s）。

#### F37（§2.9）书中的 GELU = nn.GELU(approximate='tanh')；而 codebot 的 FFN 用的是 nn.GELU()，也就是 erf 版本

依赖：torch；浏览器：否；CPython 用时 0.01 s

```python
import torch, torch.nn as nn
from bookload import load
import codebot.model as cb
t = torch.linspace(-6, 6, 120001); g = load("ch02/09_norm_gelu.py").GELU()(t)
print((g - nn.GELU(approximate="tanh")(t)).abs().max().item(), (g - nn.GELU()(t)).abs().max().item(), cb.FFN(8, 32, 0.1).layers[1])
```

实测输出：

```text
4.76837158203125e-07 0.0004737377166748047 GELU(approximate='none')
```

#### F38（§2.9）09 的 Block（gamma/beta）与 codebot 的 Block（weight/bias）的 state_dict 不改名就不能互相加载

依赖：torch；浏览器：否；CPython 用时 0.08 s

```python
from bookload import load
import codebot.model as cb
b09, bcb = load("ch02/09_norm_gelu.py").Block(384, 6), cb.Block(384, 6, 1536)
print([k for k in b09.state_dict() if "norm1" in k], [k for k in bcb.state_dict() if "norm1" in k])
try: b09.load_state_dict(bcb.state_dict())
except RuntimeError as e: print("load_state_dict fails:", " ".join(str(e).split())[:90])
```

实测输出：

```text
['norm1.gamma', 'norm1.beta'] ['norm1.weight', 'norm1.bias']
load_state_dict fails: Error(s) in loading state_dict for Block: Missing key(s) in state_dict: "norm1.gamma", "no
```

#### F39（§2.9）残差流：把 W_o 和 FFN 输出层置零后，Block 严格等于恒等映射

依赖：torch；浏览器：否；CPython 用时 0.04 s

```python
import torch
import codebot.model as cb
torch.manual_seed(0); blk = cb.Block(384, 6, 1536).eval(); x = torch.randn(2, 16, 384)
with torch.no_grad():
    blk.attn.W_o.weight.zero_(); blk.ffn.layers[2].weight.zero_(); blk.ffn.layers[2].bias.zero_()
    print((blk(x) - x).abs().max().item())
```

实测输出：

```text
0.0
```

#### F40（§2.9）一个 Block（E=384, H=6, ff=4E）有 1,772,928 个参数 = 2 个 LN + 4E² + FFN（约等于 12E² = 1,769,472）

依赖：torch；浏览器：否；CPython 用时 0.03 s

```python
import codebot.model as cb
E = 384; n = sum(p.numel() for p in cb.Block(E, 6, 4 * E).parameters())
print(n, 2 * 2 * E + 4 * E * E + (E * 4 * E + 4 * E) + (4 * E * E + E), 12 * E * E)
```

实测输出：

```text
1772928 1772928 1769472
```

#### F41（§2.10）ch02/ch03 所用 GPT 的参数量闭式解：11,121,640（不共享权重则为 11,505,640）

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.0 s

```python
V, CTX, E, L, FF = 1000, 256, 384, 6, 1536
block = 2 * 2 * E + 4 * E * E + (E * FF + FF) + (FF * E + E)
print(f"{V*E + V + CTX*E + L*block + 2*E:,}  untied: {2*V*E + V + CTX*E + L*block + 2*E:,}")
```

实测输出：

```text
11,121,640  untied: 11,505,640
```

Pyodide：输出逐字相同（0 s）。

#### F42（§2.10）对 parameters() 求 numel 之和得 11,121,640（共享矩阵只算一次）；对 state_dict 求和会重复计算；embed.weight 与 unembed.weight 是同一个对象

依赖：torch；浏览器：否；CPython 用时 0.31 s

```python
from codebot.model import GPT
m = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
print(f"{sum(p.numel() for p in m.parameters()):,}", f"{sum(t.numel() for t in m.state_dict().values()):,}", m.embed.weight is m.unembed.weight)
```

实测输出：

```text
11,121,640 11,505,640 True
```

#### F43（§2.10）保存的 checkpoint 约 44.5 MB ≈ 4 字节 × 11.12M 参数，说明共享矩阵只存了一份（官方 model_pretrain.pt 为 44,516,353 字节，HTTP HEAD 得到）

依赖：torch；浏览器：否；CPython 用时 0.44 s

```python
import os, tempfile
from codebot.model import GPT
p = os.path.join(tempfile.mkdtemp(), "m.pt"); GPT(1000, 256, 384, 6, 6, 1536, 0.1).save(p)
print(os.path.getsize(p), 11_121_640 * 4); os.remove(p)
```

实测输出：

```text
44509194 44486560
```

#### F44（§2.10）初始化：Linear/Embedding ~ N(0, 0.02)，bias 为 0，LayerNorm 为 1/0；对共享矩阵而言，Linear 的初始化最后执行，因此最终生效

依赖：torch；浏览器：否；CPython 用时 0.71 s

```python
import torch.nn as nn
from codebot.model import GPT
class Probe(GPT):
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.constant_(m.weight, 2.0)
        elif isinstance(m, nn.Embedding): nn.init.constant_(m.weight, 1.0)
print("tied value:", Probe(50, 8, 16, 2, 1, 64, 0.0).embed.weight[0, 0].item())
m = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
print(round(m.blocks[0].attn.W_q.weight.std().item(), 4), m.blocks[0].norm1.weight.mean().item(), m.unembed.bias.abs().max().item())
```

实测输出：

```text
tied value: 2.0
0.02 1.0 0.0
```

#### F45（§2.10）初始化时的 loss 约为 6.98 = ln V + σ²/2（σ 是 logits 的标准差，约 0.39），略高于 ln(1000) = 6.9078

依赖：torch；浏览器：否；CPython 用时 0.92 s

```python
import math, torch, torch.nn.functional as F
from codebot.model import GPT
torch.manual_seed(0); m = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval(); ids = torch.randint(0, 1000, (8, 257))
with torch.no_grad(): lg = m(ids[:, :-1])
ce = F.cross_entropy(lg.reshape(-1, 1000), ids[:, 1:].reshape(-1)).item(); s = lg.std().item()
print(f"CE={ce:.4f}  lnV={math.log(1000):.4f}  lnV+s^2/2={math.log(1000) + s * s / 2:.4f}  s={s:.3f}")
```

实测输出：

```text
CE=6.9846  lnV=6.9078  lnV+s^2/2=6.9858  s=0.395
```

#### F46（§2.10）整模型因果性：改动第 100 个 token，位置 0–99 的 logits 逐位不变

依赖：torch；浏览器：否；CPython 用时 0.57 s

```python
import torch
from codebot.model import GPT
torch.manual_seed(0); m = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval()
a = torch.randint(0, 1000, (1, 256)); b = a.clone(); b[0, 100] = (b[0, 100] + 1) % 1000
with torch.no_grad(): d = (m(a) - m(b)).abs()
print(d[:, :100].max().item(), round(d[:, 100:].max().item(), 3))
```

实测输出：

```text
0.0 0.768
```

#### F47（§2.10）eval() 下模型输出确定且与 batch 无关；train()（有 dropout）下不确定

依赖：torch；浏览器：否；CPython 用时 1.22 s

```python
import torch
from codebot.model import GPT
torch.manual_seed(0); m = GPT(1000, 256, 384, 6, 6, 1536, 0.1); ids = torch.randint(0, 1000, (4, 64))
print("train equal:", torch.equal(m(ids), m(ids)), end=" | ")
m.eval()
with torch.no_grad(): print("eval equal:", torch.equal(m(ids), m(ids)), "| batch vs alone:", (m(ids)[0] - m(ids[:1])[0]).abs().max().item())
```

实测输出：

```text
train equal: False | eval equal: True | batch vs alone: 1.8477439880371094e-06
```

#### F48（§2.10）HF GPT-2 small 有 124,439,808 个参数；同样超参的 codebot GPT 有 124,453,201 个（多 50,257 个 lm_head bias，少 36,864 个 qkv/proj bias）

依赖：torch + transformers；浏览器：否；CPython 用时 11.04 s

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from codebot.model import GPT
with torch.device("meta"):
    hf, cb = GPT2LMHeadModel(GPT2Config()), GPT(50257, 1024, 768, 12, 12, 3072, 0.1)
n = lambda m: sum(p.numel() for p in m.parameters())
print(f"{n(hf):,} {n(cb):,} {n(cb) - n(hf):+,} = {50257 - 12 * 4 * 768:+,}")
```

实测输出：

```text
124,439,808 124,453,201 +13,393 = +13,393
```

#### F49（§2.10）把 HF GPT-2 的权重拷进 codebot GPT（qkv/proj bias 置零）后，logits 相差 3.5e-4（erf GELU）；换成 tanh GELU 后相差 1.9e-6

依赖：torch + transformers；浏览器：否；CPython 用时 1.51 s

```python
import runpy, contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf): runpy.run_path("verify/v_hf_gpt2.py")
print("\n".join(l for l in buf.getvalue().splitlines() if "max|logits diff|" in l))
```

实测输出：

```text
max|logits diff| with codebot's nn.GELU() (erf): 3.498e-04
max|logits diff| after switching to tanh-GELU  : 1.907e-06
```

#### F50（§2.10）训练好的 codebot：代码片段的精确 token id；在 32 个训练窗口上 CE 为 0.40（新初始化约 6.94–6.98）；从 'def' 开始的贪心生成

依赖：torch + 训练好的 checkpoint；浏览器：否；CPython 用时 3.29 s

```python
import os, numpy as np, torch, torch.nn.functional as F
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate
from bookload import checkpoint, REPO
tok = BPETokenizer.load_from(os.path.join(REPO, "codebot", "merge_rules.pkl")); m = GPT.load_from(checkpoint()).eval()
print(tok.encode("def add(a, b):\n    return a + b\n"))
data = np.fromfile(os.path.join(REPO, "codebot", "tiny_codes.bin"), dtype=np.uint16)
st = np.random.default_rng(0).integers(0, len(data) - 257, size=32)
x = torch.tensor(np.stack([data[s:s + 257] for s in st]).astype(np.int64))
with torch.no_grad(): print("CE on 32 train windows:", round(F.cross_entropy(m(x[:, :-1]).reshape(-1, 1000), x[:, 1:].reshape(-1)).item(), 4))
print(repr(generate(m, tok, "def", max_new_tokens=24, temperature=0)))
```

实测输出：

```text
[304, 882, 40, 97, 44, 345, 285, 265, 307, 273, 347, 345, 10]
CE on 32 train windows: 0.4014
'def sort_list(lst):\n    for i in range(len(lst) - 1):\n        for j in'
```

#### F51（§2.10）训练后的注意力头（ch02_attn_maps.json）：L4H3 是 previous-token head（' b'→' +' 权重 0.99）；第 1 层约一半权重落在第一个 token 上（attention sink）

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.01 s

```python
import json, numpy as np
A = np.array(json.load(open("ch02_attn_maps.json", encoding="utf-8"))["attn"])     # (layer, head, query, key)
prev = np.mean([A[:, :, t, t - 1] for t in range(1, A.shape[-1])], axis=0)
print("best prev-token head (layer, head):", tuple(int(i) for i in np.unravel_index(prev.argmax(), prev.shape)), round(float(prev.max()), 2),
      "| L4H3 ' b'->' +':", A[4, 3, 11, 10], "| layer-1 mean weight on token 0:", round(float(A[1, :, 1:, 0].mean()), 2))
```

实测输出：

```text
best prev-token head (layer, head): (4, 3) 0.57 | L4H3 ' b'->' +': 0.9882 | layer-1 mean weight on token 0: 0.49
```

Pyodide：输出逐字相同（0.01 s）。

#### F52（§2.10）纯 NumPy 的 GPT 前向（np_gpt.py）能复现小型 codebot GPT 的 torch logits，误差约 1e-7

依赖：仅 NumPy；浏览器：可（Pyodide 实测通过）；CPython 用时 0.05 s

```python
import numpy as np
from np_gpt import gpt_forward
Z = np.load("pyodide_test/tiny_ref.npz"); P = {k: Z[k].astype(np.float64) for k in Z.files if k not in ("ids", "ref_logits")}
out = gpt_forward([int(i) for i in Z["ids"]], P, n_layer=2, n_head=4)
print(out.shape, np.abs(out - Z["ref_logits"]).max())
```

实测输出：

```text
(12, 50) 8.33490231277878e-08
```

Pyodide 输出（0.05 s，仅末位浮点差异）：

```text
(12, 50) 8.33490230167655e-08
```

#### F53（§2.10）端到端自检：11.1M 参数的 GPT 能在 CPU 上过拟合一个固定 batch 的真实 TinyCodes token（50 步内从 7.0 降到约 0.02）

依赖：torch；浏览器：否；CPython 用时 28.0 s

```python
import numpy as np, torch, torch.nn.functional as F, os
from codebot.model import GPT
from bookload import REPO
torch.manual_seed(0)
data = np.fromfile(os.path.join(REPO, "codebot", "tiny_codes.bin"), dtype=np.uint16)
st = np.random.default_rng(0).integers(0, len(data) - 65, size=4)
x = torch.tensor(np.stack([data[s:s + 65] for s in st]).astype(np.int64))
m = GPT(1000, 256, 384, 6, 6, 1536, 0.0); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
for step in range(51):
    loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 1000), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0: print(step, round(loss.item(), 4))
```

实测输出：

```text
0 7.0115
25 0.5956
50 0.0163
```

### 3.2 补充测量（不适合写成单行 snippet 的结果，脚本均在工具包内）

**(a) 缩放到底解决了什么（`verify/v03b_saturation_table.py`，仅 NumPy，约 10 s）**：256 个 key，1000 次试验，随机种子 0。“maxw”是每行最大权重的均值，“sat”是最大权重大于 0.99 的行所占比例，‖J‖_F 是 softmax 雅可比矩阵 diag(p) − ppᵀ 的 Frobenius 范数。

```text
 d    | unscaled: maxw  sat>0.99  ||J||_F | scaled: maxw  sat>0.99  ||J||_F
    4 |        0.178   0.000   0.1909  |        0.045   0.000   0.0998
   16 |        0.490   0.012   0.2893  |        0.044   0.000   0.0997
   64 |        0.740   0.135   0.2357  |        0.043   0.000   0.0987
  256 |        0.878   0.418   0.1407  |        0.043   0.000   0.0991
 1024 |        0.935   0.631   0.0843  |        0.042   0.000   0.0984
 4096 |        0.969   0.794   0.0418  |        0.043   0.000   0.0991
```

解读：缩放之后，注意力的尖锐程度和梯度大小都与 d 无关，这正是“温度不随 d 变化”。不缩放时，饱和行的比例随 d 单调上升，而雅可比范数**先升后降**（d=16 时最大）：中等程度的尖锐分布梯度反而更大，只有接近 one-hot 时梯度才消失。所以“不缩放 → 梯度消失”这一说法只在 d 很大时成立。教程里应该写“不缩放时，softmax 的有效温度随 √d 变化”，再用这张表说明后果。

**(b) 一个 Block 为反向传播保存了什么（`verify/v_block_saved.py`）**：B=1，C=256，fp32，CPU，train 模式。每个样本每个 Block 实测 11.32 MiB（不含参数）。按“8CE + 2C·ff + 2HC² 个 float”估算的下界是 9.00 MiB，另有约 0.6 MiB 的 mask。多出来的部分来自三处：CPU 上 dropout mask 以 float32 保存（注意力权重处 1.5 MiB，两处残差 dropout 各 0.375 MiB）；为做 batched matmul，注意力权重被 reshape 复制了一份；Q、K 也各复制了一份。6 个 Block 加上 logits 的 log-softmax，每个样本约 70 MiB；`v_compute.py` 实测为 B=32 时 2,270.5 MiB，即 71 MiB/样本。CUDA 上 dropout mask 应以 bool 保存，所以 GPU 上的数字会略小（推测，未在 GPU 上实测）。

**(c) 训练好的模型（`verify/v_trained.py`，需要 checkpoint）**：
- tokenizer 有 743 次 merge，词表大小 1000，`end_token_id`=999。代码片段 `"def add(a, b):\n    return a + b\n"` 编码为 `[304, 882, 40, 97, 44, 345, 285, 265, 307, 273, 347, 345, 10]`，对应 `['def', ' add', '(', 'a', ',', ' b', '):', '\n   ', ' return', ' a', ' +', ' b', '\n']`。
- 在 32 个训练窗口上，训练好的模型 CE 为 0.4014（perplexity 1.49），新初始化的模型为 6.9432。这些窗口来自训练集，ch03 本身也没有划分验证集，所以 0.40 不能当作泛化能力的指标。
- 从 `def` 开始贪心生成：`'def sort_list(lst):\n    for i in range(len(lst) - 1):\n        for j in range(i + 1, len(lst)):\n            if lst[i] < lst[j]:\n                lst[i], lst[j] = l'`。这段 4 行前缀在语料中逐字出现 0 次（语料共 6,487,033 字符、21,835 篇文档，见 `verify/v_misc.py`），说明不是逐字背诵。
- 注意力头（`ch02_attn_maps.json`）：L4H3 是最明显的 previous-token head，在 t−1 上的平均权重为 0.57，' b'→' +' 为 0.99；L3H1 在两个示例 query 上把 0.94 和 0.85 的权重放在前一个 token；L5H2 从 ' a' 指向函数名 ' add'，权重 0.92；第 1 层六个头平均约 0.49 的权重落在第一个 token 'def' 上（attention sink），L1H2 达到 0.71。
- 位置嵌入：位置 0 的向量范数为 4.37，而位置 128、255 分别只有 0.71、0.99（token 嵌入的平均范数为 1.57），可能与 attention sink 有关（推测）。

**(d) 与 HF GPT-2 的结构等价性（`verify/v_hf_gpt2.py`，需要 `--with transformers`，用随机初始化，不下载权重）**：
- GPT-2 small：HF 为 124,439,808，codebot 为 124,453,201。GPT-2 XL：HF 为 1,557,611,200，codebot 为 1,557,354,257；同样尺寸下的 LLaMA 式结构（RMSNorm + SwiGLU + RoPE + 不共享嵌入，即 CS336 A1 的结构）为 2,127,057,600 = 2VE + L(4E² + 3E·ff + 2E) + E（`verify/v_xl_counts.py`）。
- 在 V=1000、E=384、L=6 的尺寸下拷贝权重：使用 erf GELU 时 logits 最大差 3.498e-4，换成 tanh GELU 后为 1.907e-6。HF 的 `c_proj` 初始化标准差为 0.0058 = 0.02/√(2L)，说明 GPT-2 会缩放残差投影的初始化，codebot 不会。

**(e) 浏览器可行性**：
- Pyodide 314.0.7 冷启动约 4.3 s，首次从 CDN 加载 numpy 约 7.6 s（之后有缓存）。
- 小模型（V=50, E=32, L=2）的 NumPy 前向需 3–11 ms，与 torch 的差为 8.3e-8。
- 训练好的 11.1M 模型（fp16 权重 22.3 MB，float32 计算）在 Pyodide 里前向一次：C=13 时 0.92 s，C=64 时 3.4 s，C=256 时 14.7 s；与 torch 的最大差 4.7e-3，argmax 100% 一致。因此在 Pyodide 里做逐 token 生成太慢。
- 用纯 JS（Float32Array，单线程）做带 KV cache 的解码，每个 token 约 1100 万次乘加，实测 62.5 ms/token（约 16 token/s），可以做交互式生成（`pyodide_test/js_matvec_bench.mjs`）。
- CPython 下手写 attention 与 SDPA 的耗时（B=8, H=8, D=64）：C=256 时 13.7 ms 对 5.9 ms，C=1024 时 209.6 ms 对 54.3 ms；后者的分数张量为 256 MiB（`verify/v08_mha.py` H10）。

---

## 4. GPU / 数据需求

### 4.1 ch02 本身

全部脚本都只用 CPU，单个脚本 0.3–4 s；53 条验证事实合计约 80 s（含 transformers 加载）。**本章没有必须用 GPU 的部分。**

### 4.2 本章模型的训练开销（训练在 `ch03/01_pretrain.py` 中进行）

| 项目 | 数值 | 来源 |
|---|---|---|
| 配置 | V=1000, ctx=256, E=384, H=6, L=6, ff=1536, dropout 0.1, AdamW lr 3e-4（无调度、无梯度裁剪），batch 32，20,000 步 | ch03/01_pretrain.py |
| 参数 N | 11,121,640（参与矩阵乘法的约 11,022,336） | F41 / v_compute |
| 训练 token 数 D | 20,000 × 32 × 256 = 163,840,000，相当于 tiny_codes.bin（2,677,398 token）的 **61.2 个 epoch** | v_compute / v_misc |
| FLOPs | 6·N·D = 1.084×10¹⁶，加上注意力（按代码实际计算完整 C×C：每 token 3 × L × 4·C·E = 7.1 MFLOP）共约 **1.20×10¹⁶** | v_compute |
| 显存（fp32） | 参数 + 梯度 + AdamW(m, v) = 170 MiB；B=32、C=256 时保存的激活为 2,270.5 MiB（CPU 实测，GPU 上 dropout mask 为 bool，会略小）；加上 CUDA context，峰值估计约 3 GiB | v_compute / v_block_saved |
| CPU 耗时 | B=32、C=256 时 5.24 s/步，20k 步约 29 h（不可行）；B=8、C=128 时 0.62 s/步 | v_compute（机器有负载，±30%） |

**GPU 耗时估算（假设的 MFU，未实测）**：RTX 3060 Ti fp32（16.2 TFLOPS 峰值，35% MFU）约 35 分钟；A100 fp32 不开 TF32（50% MFU）约 20 分钟；A100 bf16（25% MFU）约 3 分钟的纯计算时间。但这个模型很小，20k 步 × 每步 5–15 ms 的 kernel launch 与 Python 开销可能占主导，A100/H100 上实际大概 5–15 分钟（估计）。建议看 tqdm 前 200 步的 it/s 再外推。书中代码没有开启 TF32 或 AMP；在 Ampere 及更新的 GPU 上加一行 `torch.set_float32_matmul_precision('high')` 通常能明显提速（未实测）。

### 4.3 数据与模型文件

| 文件 | 大小（字节） | 位置 | 说明 |
|---|---|---|---|
| codebot/tiny_codes.bin | 5,354,796 | 仓库内 | uint16，2,677,398 个 token，最大 id 999 |
| codebot/tiny_codes.txt | 6,743,534 | 仓库内 | 6,487,033 字符，21,835 篇文档 |
| codebot/merge_rules.pkl | 7,421 | 仓库内 | 743 次 merge → 词表 1000 |
| codebot/tiny_codes_sft.json | 129,717 | 仓库内 | ch03 SFT 用 |
| codebot/model_pretrain.pt | 44,516,353 | HF `koki0702/zero-llm-data`（download.sh 中未注释） | 已下载到 scratch 并实测；另一个 agent 也把同一文件放进了 `codebot/` |
| codebot/model_sft.pt | 44,515,938 | HF（download.sh 中已注释） | 只用 HTTP HEAD 查了大小 |
| codebot/model_grpo.pt | 44,516,021 | HF（download.sh 中已注释） | 只用 HTTP HEAD 查了大小 |

### 4.4 在 CPU 上缩小

- 做概念验证（tracer bullet）：过拟合一个 batch（F53：B=4、C=64、dropout=0，50 步内 loss 从 7.0 降到 0.016，CPU 上 28–45 s），或者 B=8、C=128 跑几百步（约 0.6 s/步）。
- 想要更小的模型：E=128、L=2、H=4 的参数量和 FLOPs 大约降一个数量级，参数量可以用 F41 的公式现算。
- 教程里的“端到端例子”不必在 CPU 上真训练 20k 步，直接加载官方 `model_pretrain.pt`（44.5 MB）演示推理即可。

### 4.5 Modal / RunPod / Lambda 运行方案

- `uv.lock` 锁定 PyPI 上的 torch 2.9.1，并在 linux x86_64 上带 `nvidia-*-cu12` 依赖，**所以在 Linux GPU 机器上执行 `uv sync` 会直接得到 CUDA 版 torch**；Windows 上得到的是 CPU 版。本机的 RTX 3060 Ti 需要另建环境，从 PyTorch 官方 index 安装 CUDA wheel（例如 cu128；未实测，也不要改仓库）。
- 选卡：任何 ≥ 8 GB 显存的卡都够，比如 T4、L4、A10G、A100、本机 3060 Ti。运行时间不到 1 小时，按 2025–26 年常见报价，成本约为 1–3 美元以内（请以平台实时价格为准）。
- Modal：镜像为 `debian_slim` + uv + 仓库代码；挂载一个 Volume 用来保存 `model_pretrain.pt`（44.5 MB）和 `loss_pretrain.png`；运行 `uv run python ch03/01_pretrain.py`（脚本会把结果写进 `codebot/`，正好落在 Volume 上），或者运行一个把保存路径改到 Volume 的副本。
- RunPod / Lambda：`git clone` → `uv sync` → `uv run python ch03/01_pretrain.py`，wandb 在本脚本中没有用到。
- 更重的 GPU 任务（ch06 的 storybot 22.7M、ch09 的 webbot 152M + DDP + torch.compile）见对应章节的档案。

---

## 5. 易错点、坑与潜在 bug（均已实测；标“推测”的除外）

1. **01 的 softmax 会溢出**：`np.exp(similarity)` 没有减最大值，分数超过约 709 就会溢出，得到 inf/inf = NaN（F04）。书中的数据分数只有 59–76，所以没有触发。
2. **点积相似度受向量长度影响**：(6,4,5) 的点积最大值落在 (7,6,2)，而不是最近的 (5,5,5)（F02）。T→0 时，用 (8,2,3) 和 (5,5,5) 去查，结果是 87.5 和 73.33（F03）。这是讲“软字典是硬字典的推广”时最容易被追问的点。
3. **02 只适用于二维**：`K.t()` 和 `dim=1` 换到批量张量上就错了，要用 `transpose(-2,-1)` 和 `dim=-1`。
4. **03 没有设种子**：每次运行数值都不同；方差之比恒为 d，这是代数结果。
5. **缩放的理由要讲准确**：雅可比范数不是单调的（§3.2a）；初始化时分数本来就很小，注意力近乎均匀（F10）；缩放的核心作用是让 softmax 的温度不随 d 变化。
6. **mask 相关**：
   - 整行都是 −inf 时得到 NaN（F18），tril 不会出现这种情况，但加了 padding mask 或 KV cache 写错时可能出现；
   - softmax 做在错误的轴上会泄露未来信息（F19）；
   - `masked_fill(mask == 0, -inf)` 依赖的是 float 型 mask，`mask == 0` 会转成 bool；
   - mask 每次前向都会重新分配，没有 `register_buffer`，只是效率问题。
7. **SDPA 的 `is_causal=True` 采用左上角对齐**：L=1、S=5 时只能看到 key 0（F21）。webbot 解码时用 `is_causal=False`，是正确的。
8. **06 与 07 的 `W_v` 形状不同**：06 是 E×E，07 是 E×D 再接 `W_o`。只看代码容易以为 06 漏了 `W_o`，其实它是 2.7 低秩化之前的版本（F22）。
9. **拆头的形状问题**：
   - `transpose` 之后直接 `.view` 会报错，要先 `.contiguous()` 或改用 `.reshape`（F25）；
   - `view(B,H,C,D)` 而不先 transpose **不会报错，但会把位置和头混在一起，从而泄露未来信息**（F27）；
   - NumPy 的 reshape 会自动复制，没有 contiguous 的问题，但错误拆分同样会泄露。
10. **术语混淆**：08 的注释把 H 叫作“ヘッドサイズ”，nanoGPT 的 `head_size` 指 D，读代码时要注意。
11. **`head_dim = embed_dim // n_head` 不检查整除**：E=100、H=3 时 head_dim=33，H·D=99≠100，照样能运行，没有任何报错（`verify/v_misc.py`）。
12. **dropout 默认开启**：模块创建后处于 train 模式，两次前向结果不同，被 dropout 的注意力行之和也不再是 1（F30）。生成前必须调用 `model.eval()`（`codebot/utils.generate` 已经调用）。
13. **LayerNorm 要用有偏方差**：`x.var()` 默认无偏，必须写 `unbiased=False`，否则输出整体缩放 √((E−1)/E)（F33）。NumPy 的 `np.var` 默认 ddof=0，刚好一致。
14. **GELU 有两个版本**：`nn.GELU()` 默认是 erf 版，tanh 版要写 `approximate='tanh'`。codebot 用的是 erf 版，与书中 09 和 GPT-2 的 tanh 版不同（F37、F49）。另外，09 的 GELU 每次调用都会新建 `torch.tensor(2/π)`（CPU 上的 0 维张量）；与 CUDA 张量相乘在 PyTorch 中是允许的，但本研究没有在 GPU 上实测。
15. **键名不兼容**：09 的 Block 用 gamma/beta，codebot 用 weight/bias，state_dict 不改名就不能互相加载（F38）。codebot/model.py 里的 `LayerNorm`、`GELU` 两个类定义了却没有被使用。
16. **weight tying 的计数和初始化**：
   - 对 state_dict 求和会把共享矩阵算两次（F42）；
   - `apply()` 先访问 embed、后访问 unembed，所以共享矩阵最终由 Linear 的初始化决定，目前两种初始化相同，未来改动时要注意（F44）；
   - 保存时共享矩阵只写一份（F43）；
   - unembed 有 bias，而 GPT-2 的 lm_head 没有。
17. **初始化与 GPT-2 不同**：没有对残差投影做 0.02/√(2L) 的缩放（HF 的 c_proj 标准差为 0.0058，见 §3.2d）。
18. **可学习位置嵌入限定了上下文长度**：超过 max_context_len 会触发 IndexError（F14），`generate()` 通过截断到最近 256 个 token 来规避。
19. **初始 loss 的期望值**：讲成“初始 loss ≈ ln V”可以，但实测是 ln V + σ²/2，在 E=384、std=0.02 时约高 0.08（F45）。
20. **工作目录**：09、10 在 import 时会 `os.chdir` 到仓库根目录，学习者在自己的 notebook 里 import 这两个文件时，相对路径会被改掉。graph.py 不 chdir，图片会写到当前目录。
21. **README 的小错**：README 示例写的是 `python ch02/10_gpt.py`，实际文件名是 `10_gpt2.py`。
22. **训练 loss 的含义**：ch03 在 268 万个 token 上训练了 61 个 epoch，0.40 是训练集上的 loss，不代表泛化能力。不过贪心生成的样本并不是语料的逐字拷贝（§3.2c）。
23. **手写 attention 的代价**：显式生成 (B,H,C,C) 张量，C=1024 时比 SDPA 慢约 4 倍、多占 256 MiB（§3.2e）。
24. **学习者常见困惑**（结合教学经验整理，不来自历史 session 数据）：
   - 为什么 Q/K 的维度可以和 V 不同（F17）；
   - mask 加在 softmax 之前，而不是之后；
   - 为什么第 0 个位置的输出就是它自己的 V（F23）；
   - LayerNorm 和 BatchNorm 归一化的维度不同（F35）；
   - 共享权重后参数量怎么算（F41、F42）；
   - 为什么 H·D 可以不等于 E（08 的前半段）。

---

## 6. 互动组件点子

“可行性”一栏的依据见 §3.2e：标“Pyodide”的用 NumPy 在页面里实时计算；标“JS”的用纯 JS 实现；标“预计算”的读取工具包里的 JSON。

| # | 组件 | 学习者操作 | 学习者观察 | 对应事实 | 可行性 |
|---|---|---|---|---|---|
| W1 | 软字典游乐场（2.1） | 用三个滑块调新电影的动作/剧情/喜剧值；调温度 T；切换相似度（点积/余弦/负距离）；切换“特征 ×10” | 7 部电影的权重柱状图和预测评分；点积选 (7,6,2) 而余弦选 (5,5,5)；T→0 时 (5,5,5) 和 (8,2,3) 并列导致 73.33、87.5；×10 时朴素 softmax 变成 NaN | F01–F05 | JS 或 Pyodide |
| W2 | 矩阵透视（2.2） | 悬停在权重矩阵（3×7）的某一格 | 高亮对应的 Q 行和 K 行、点积数值、该行的 softmax，以及 ×V 之后的评分；每个张量旁标注形状 | F06 | JS |
| W3 | 缩放温度计（2.3） | 拖动 d（4–4096，对数刻度）；切换是否除以 √d；重新采样 | q·k 直方图（复现 03 的图）；256 个 key 上一行注意力的形状；maxw、饱和比例、雅可比范数随 d 的曲线（§3.2a 的表）；初始化时的分数尺度（F10） | F07–F10、§3.2a | Pyodide（d≤1024 时约 2 s） |
| W4 | 位置编码实验台（2.5） | 打乱输入 token 顺序；切换有无位置编码、有无 mask；选择正弦式或训练好的可学习位置嵌入 | 输出只是跟着重排（置换等变）；PE 热图与“点积只取决于偏移量”的曲线；训练后可学习位置嵌入的余弦相似度呈带状对角结构（0.949 / 0.562 / 0.037） | F11–F14 | Pyodide；带状矩阵预计算（64×64 下采样） |
| W5 | 因果 mask 泄漏探测器（2.6） | 点击 C×C 网格解除某个未来格的遮挡；切换 softmax 轴；切换拆头方式 | “改动未来 token 后，过去位置的输出是否变化”的红绿灯；加上作弊实验的预计算 loss 曲线（有 mask 时 ln V，无 mask 时 ln(V)/C，最后一个位置单独无法作弊） | F15–F20、F27 | Pyodide（前向）+ 预计算（训练曲线） |
| W6 | Value 低秩滑块（2.7） | 拖动秩 D（1…E） | 参数量 E² 对 2ED 的对比；W_o·W_v 的奇异值在第 D 个之后掉到 0 | F22 | Pyodide |
| W7 | 多头 reshape 动画（2.8） | 一步步执行 view → transpose → 计算 → transpose → view；按“错误拆分”按钮；按“不调 contiguous”按钮 | 按头和位置着色的内存块动画；错误拆分时泄漏探测器变红；不调 contiguous 时报 RuntimeError；“两个等价公式”的切换 | F24–F29 | JS（动画）+ Pyodide |
| W8 | LayerNorm / GELU 拨盘（2.9） | 拖动一个向量的柱子；调 γ、β；切换有偏/无偏方差；切换 GELU 的 erf/tanh 版本 | 归一化前后的均值和方差；√((E−1)/E) 的缩放；把 tanh−erf 的差放大 1000 倍后显示，并标出最小值 −0.170 | F32–F37 | Pyodide（erf 用 math.erf） |
| W9 | GPT 参数、算力、显存计算器（2.10） | 输入 V/ctx/E/H/L/ff，开关 tie 和 bias，选预设（ch02 / GPT-2 small / XL / storybot / webbot / LLaMA 式） | 各部分参数的堆叠条；6ND + 注意力 FLOPs；显存（参数、梯度、Adam、激活 71 MiB/样本）；“能放进我的 3060 Ti 吗”“在 A100 上要多久” | F40–F43、F48、§3.2b/d、§4.2 | JS |
| W10 | 真实注意力图浏览器（训练好的 codebot） | 选层和头；悬停 token；以后还可以输入自己的代码 | L4H3 的 previous-token 模式（0.99）、L5H2 的 ' a'→' add'、第 1 层的 attention sink | F51、§3.2c | 预计算 `ch02_attn_maps.json`（41 KB）；输入自定义代码需要 JS 前向和 22 MB 权重 |
| W11 | 初始 loss 预言机（2.10） | 调初始化 std、V、E；开关权重共享 | 预测值 ln V + σ²/2 与实测 CE 对比 | F45 | Pyodide（用小模型） |
| W12 | 端到端流水线（tokens → logits） | 输入一行代码；逐层前进 | 每一步的张量形状、logits 前 5 名与概率；最后贪心选出下一个 token；配合 F46 的因果性按钮 | F50、F52、F53 | 小模型用 Pyodide；真实模型用 JS 前向（62.5 ms/token） |

---

## 7. CS336 对照

以下凭记忆整理，未做网络检索，problem 名称和细节可能与当年的 handout 有出入。

| DLFS6 ch02 内容 | CS336（Stanford Language Modeling from Scratch）对应 | 差异与备注 |
|---|---|---|
| 2.1–2.2 softmax(QKᵀ)V | A1 实现 `softmax`（要求减最大值保证数值稳定）与 `scaled_dot_product_attention` | A1 的 mask 是布尔型（True 表示可见），被遮位置填 −inf，与 `torch.tril` 的写法一致；DLFS6 的 01 没有减最大值（F04） |
| 2.3 1/√d_k | A1 的 SDPA 公式；Lecture 3（架构与超参）讨论 attention logits 的稳定性（QK-norm、logit soft-capping、z-loss） | §3.2a 的表可作为数值注脚 |
| 2.4 Transformer 结构 | A1 的 Transformer LM：pre-norm block、最终 norm、LM head | 一致 |
| 2.5 位置编码 | A1 实现 RoPE，并有 NoPE 消融（`no_pos_emb`）；Lecture 3 比较正弦、可学习绝对位置、相对位置、RoPE | codebot 用 GPT-2 式可学习绝对位置；storybot/webbot（ch05/ch09）才与 A1 一致 |
| 2.6 causal mask | A1 的 `multihead_self_attention` 必须是因果的 | 测试时可以用 F16、F46 这种“改未来、看过去”的方法 |
| 2.7 W_o 与低秩 value | A1 MHA 的 W_O ∈ R^{d_model × h·d_v}，d_k = d_v = d_model / h | DLFS6 单独用一节讲“为什么有 W_o”，CS336 直接给出 |
| 2.8 多头 reshape | A1 要求把所有头的投影合并成一次矩阵乘，把头当作 batch 维；推荐用 einops 的 `rearrange(x, "... s (h d) -> ... h s d")` | einops 写法可以从结构上避免 F27 那样的错误拆分 |
| 2.9 LayerNorm / GELU / FFN | A1 改用 `rmsnorm` 和 SwiGLU 版 `positionwise_feedforward`（d_ff ≈ 8/3·d_model）；消融实验有 layer_norm_ablation、pre_norm_ablation（改成 post-norm）、swiglu_ablation（SiLU FFN，d_ff=4·d_model） | codebot 用 GPT-2 原版组件；A1 的版本对应 DLFS6 ch05 的 storybot |
| 2.10 GPT 组装、初始化、权重共享 | A1 的 `transformer_block` 与 `transformer_lm`；Linear 无 bias，初始化为截断正态 σ²=2/(d_in+d_out)，Embedding 为截断 N(0,1)；lm_head 不与嵌入共享 | codebot：N(0,0.02)、共享嵌入、unembed 有 bias。可以把两种初始化下的初始 loss（F45）做成对照 |
| 参数量与 FLOPs | A1 的 `transformer_accounting`（GPT-2 small 到 XL 的参数量与前向 FLOPs 分解）；Lecture 2（资源核算：6ND、MFU、内存） | F41、F48，`v_xl_counts.py`（XL 尺寸下 LLaMA 式结构为 2,127,057,600），§4.2 |
| 激活与显存 | A1 的 `adamwAccounting`（参数、激活、梯度、优化器状态的峰值内存，以及给定 MFU 下的训练时间） | §3.2b 的实测可以和 A1 的公式对表 |
| 初始 loss 与交叉熵 | A1 的 `cross_entropy`（用 log-sum-exp 保证稳定） | F45 |
| save / load | A1 的 `checkpointing`（save_checkpoint / load_checkpoint） | GPT.save / load_from |
| 手写 attention 与 SDPA | A2（systems）：先给朴素 attention 做基准测试（长序列会 OOM），再 torch.compile，再用 Triton 写 FlashAttention-2 的前向和反向 | §3.2e：C=1024 时 210 ms 对 54 ms，分数张量 256 MiB |
| 6ND 与训练成本 | A3（scaling laws）：C ≈ 6ND，IsoFLOP 分析 | §4.2 |
| GQA / KV cache（变体） | Lecture 3 的 MQA/GQA；推理讲座中的 KV cache | F21 与 webbot 的 is_causal 处理 |
| （下一章）SFT / GRPO | A5（alignment）：SFT、expert iteration、GRPO | ch03 的 03_sft、09_grpo |

**教学建议**：把 DLFS6 ch02 当作“GPT-2 原版”，把 CS336 A1 当作“现代 LLaMA 式”。最好的桥梁是 storybot（ch05），它逐项替换了 RoPE、RMSNorm 和 SwiGLU。教程可以在每一节末尾加一个“CS336 对照”边栏，再配一组可以直接对照 A1 测试的数值，比如参数量公式、因果性测试和 SDPA 等价性。

---

## 附录 A：脚本运行记录

| 脚本 | 墙钟时间 | 输出 / 产物 |
|---|---|---|
| ch02/01_soft_dict.py | 0.33 s | 78.66 分与 7 个权重 |
| ch02/02_attn_math.py | 2.92 s | 78.66 / 69.76 / 61.20 |
| ch02/03_attn_scaling.py | 3.98 s | 饱和示例；方差 ≈ 10 与 ≈ 1（无种子） |
| ch02/06_attn_mask.py | 2.62 s | 形状 [2,5,256] → [2,5,256] |
| ch02/07_attn_value.py | 2.68 s | 同上 |
| ch02/08_multi_head.py | 2.78 s | [2,10,512] → [2,10,512] |
| ch02/09_norm_gelu.py | 2.79 s | 无输出 |
| ch02/10_gpt2.py | 3.05 s | 出力形状: torch.Size([1, 256, 1000]) |
| ch02/graph.py | 1.05 s | relu_gelu.png（41,983 字节，写到当前目录） |

## 附录 B：工具包文件索引（`research/assets/ch02-attention-gpt/`）

- `verify/facts.py`：53 条事实的代码与断言（英文）；`run_facts.py`：执行器；`render_facts.py`：生成 §3.1（含中文断言）。
- `verify/np_gpt.py`：纯 NumPy 的 GPT 前向（softmax、layer_norm、gelu、causal_attention、gpt_forward），可以直接给浏览器组件用。
- `verify/v01_softdict_np.py`、`v03_scaling_np.py`、`v03b_saturation_table.py`：2.1–2.3 的纯 NumPy 演示。
- `verify/v06_mask_value.py`、`v08_mha.py`、`v09_norm_gelu.py`、`v10_gpt.py`：逐节的 torch 验证（M1–M8、H1–H10、N1–N8、G1–G12）。
- `verify/v_sections.py`：2.5、2.7、2.8.3 的验证；`v_cheat.py`：作弊实验；`v_overfit.py`：过拟合单个 batch。
- `verify/v_hf_gpt2.py`、`v_xl_counts.py`：与 HF GPT-2 的对照（需要 `--with transformers`）；`v_variants.py`：三个变体的对照。
- `verify/v_compute.py`、`v_block_saved.py`：算力与激活显存；`v_trained.py`：训练好的模型（生成 `../ch02_attn_maps.json`）；`export_fp16.py`：为 `pyodide_test/run_trained.mjs` 导出权重；`v_misc.py`：整除陷阱与语料事实；`bookload.py`：加载书中脚本和定位 checkpoint 的辅助函数。
- `pyodide_test/`：`run_facts.mjs`（浏览器事实）、`run.mjs`（v_browser_suite + 演示）、`run_trained.mjs`（真实模型计时）、`js_matvec_bench.mjs`（JS 解码计时）、`tiny_ref.npz`（小模型的权重和 torch 参考 logits）、`package.json`（pyodide 314.0.7）。
- `facts_cpython.json`、`facts_pyodide.json`：本档案 §3.1 的原始运行记录。
