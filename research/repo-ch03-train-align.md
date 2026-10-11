# 第 3 章 训练【基础篇】（原题「3章 学習【基本編】」）—— CodeBot 的预训练 → 生成 → SFT → 对话 → GRPO（仓库逆向 dossier）

> **范围**：`ch03/*.py`（01_pretrain, 02_generate, 03_alpaca, 03_sft, 04_chat, 09_grpo）全部逐行读过；`codebot/utils.py`、`codebot/model.py`（参考）、`codebot/tiny_codes_sft.json`；为对照也读了 `codebot/tokenizer.py`、`ch01/07,09`、`ch02/10_gpt2.py`、`ch06/05_pretrain.py`、`ch06/08_dpo.py`（开头）、`storybot/utils.py`、`webbot/utils.py`。
>
> **环境**：Windows 11，i7-11700F（8C/16T），47.7 GB RAM，Python 3.12.9（uv），torch 2.9.1+cpu（8 threads），numpy 1.26.4。
> **计时警告**：本次所有 wall time 都是在同机其它 agent 的作业把 CPU 打满（`LoadPercentage=100`、十几个 python 进程）时测得，未争用时应快约 1.5–3 倍。凡写“实测”的数字都来自我真正跑过的脚本；脚本与日志路径见附录 B。
>
> **下载**：`codebot/model_pretrain.pt`（44,516,353 B，sha256 `3e32e410…a3ecb`）按 `download.sh` 放进了 `codebot/`；`model_sft.pt`（44,515,938 B，sha256 `8d3f09b5…14f54`）与 `model_grpo.pt`（44,516,021 B，sha256 `e89848d8…9832d`）在 download.sh 里是注释行，我放在 scratch，没有放进仓库。
>
> **给教程作者的附带数据**（都在 `research/data/`，均由真 checkpoint 算出）：
> * `ch03_answer_dists.json`（60 KB）——6 个 checkpoint（键名 `sft`、`grpo` 为下载的模型；`model_sft_mine` 为我复现的 SFT；`model_grpo_fast_{base,nodrop,stdnorm}` 为 60 iter 消融的终点）在 81 道加法题上对答案 `"0".."18"` 的**精确**概率（含 EOS，最后一列 `other`=其余概率），驱动 W10/W11/W15；
> * `ch03_next_token_logits.json`（30 KB）——`model_pretrain.pt` 在 `def`、`def add(a, b):\n    return`、`for i in` 之后的全部 1000 个 logits 与 token 字符串，驱动 W3（实测：`def` 后熵 4.45 nats，top-1 `' s'` 0.071；`return` 后 `' a'` 0.996；`for i in` 后 `' range'` 0.864）；
> * `ch03_ratio_first_update.json`（3 KB）——第一次 GRPO 更新时 283 个 response token 的 ratio（train/eval 两种模式），驱动 W12；
> * `ch03_grpo_metrics_*.jsonl`——插桩版 GRPO 每 iter 的指标，驱动 W13；
> * `ch03_numpy/`——已在本机跑通、**不依赖 torch** 的验证脚本（可直接放进 Pyodide）：`facts_basic.py`（温度、采样频率、ln V、奖励函数、advantage、零方差概率、clip 导数、ratio 偏差）、`facts_pg_baselines_topk.py`（loss≈0 但梯度≠0、baseline 无偏、(G−1)/G 穷举、top-k/top-p）、`tabular_grpo_passk.py <answer_dists.json>`（pass@k + 表格 GRPO 及消融）。

---

## 0. TL;DR（最重要的 15 条）

1. **文件编号 = 小节编号**（已用 O'Reilly Japan 官方目录核对）：01→3.1 事前学习，02→3.2 テキスト生成，03_alpaca/03_sft→3.3 SFT，04→3.4 SFTモデルとの対話，**05–08 缺号 = 3.5 强化学习与语言模型、3.6 策略梯度、3.7 baseline、3.8 off-policy 与 clipping，纯理论无代码**，09→3.9 GRPO。3.2.3 讲 Top-k/Top-p，但仓库代码只实现了 temperature。
2. CodeBot 模型是 GPT-2 式：V=1000、C=256、E=384、6 层 6 头、FFN 1536、dropout 0.1、learned 位置编码、pre-LN、**embedding 与 unembedding 权重共享**；参数 **11,121,640**（重复计算共享权重则为 11,505,640）。
3. 随机初始化时 loss **7.0025**（eval，seed 0）/ 首步训练 loss **6.9895**，≈ ln(1000)=6.9078 再多约 0.08–0.09——多出的部分来自 logit 的方差（std 实测 0.3955 ≈ 0.02·√384）。
4. 预训练 20000 iter × 32 × 256 = 1.638 亿 token-position，**相当于语料（2,677,398 token）过了 61 遍**；下载的 `model_pretrain.pt` 在训练语料上的 loss 只有 **0.424（ppl 1.53）**，几乎是背下来了；本章没有验证集（第 6 章才加）。
5. SFT 数据 1092 条，其中 **81 条正是 GRPO 用的全部 `i+j=`（i,j∈1..9）**；最长样本 138 token，所以 `context_len=256` 下 **91.5% 的位置是 padding**，只有 8.455% 的位置带 label。
6. SFT 用 `labels=-100` 屏蔽 prompt，`F.cross_entropy(ignore_index=-100)` 是**整个 batch 的 token 级平均**（不是逐样本平均）；SFT 后 response 上的 loss 从 4.04 降到 **0.028**，但 prompt token 上的 loss 从 3.40 **升到 6.18**（模型不再会“续写用户的话”）。
7. SFT 模型在 81 道加法上 greedy 只有 **54.3%**（44/81）——训练集里就有这些题，它也没背下来；平均 P(正确答案)=0.450。这正是 3.4.2「モデルの計算能力」引出强化学习的动机。
8. 09_grpo.py 的 GRPO = **Dr. GRPO 风格**：advantage 只减组均值、**不除 std**；loss 是对 token 求和后除以序列数 B·G（**没有 1/|o_i| 长度归一化**）；**没有 KL、没有 reference model**；`old_model` 每轮从当前模型拷贝；每批 rollout 更新 2 次，所以只有第 2 次更新时 clip 才真正起作用。官方目录里恰好有专栏「GRPOとDr.GRPO」。
9. **潜在 bug（已实测）**：更新时 policy 处于 `train()` 模式、dropout=0.1 生效，而 `old_model` 是 `eval()`。权重完全相同的第一次更新，ratio 的 std 竟有 **0.73**、范围 0.0036–12.3，**53.9% 的 response token 落在 [0.8, 1.2] 之外**；eval 模式下 ratio 全在 [0.99999, 1.0]（只剩 +1e-8 的微小偏差）。带 dropout 的梯度与“干净的” REINFORCE 梯度的余弦只有 **0.64–0.73**。**只把更新前改成 `model.eval()`**：seed 0 跑 60 iter，greedy 准确率 **95.1% vs 70.4%**；seed 1 跑 40 iter，**95.1% vs 74.1%**（都超过下载的、用原脚本训了 500 iter 的模型的 82.7%）。
10. **ratio=1 时 GRPO-clip 的梯度与 REINFORCE-with-baseline 的梯度完全相同**（float64 余弦 1.0000000000，max|diff| 8.2e-8）；而此时 loss 的**数值**等于 −Σᵢ Aᵢ|oᵢ|/N（实测 0.007324 两边一致），跟学没学到无关——这正好回答了用户过去“为什么 policy loss 永远≈0”的困惑。
11. 组内奖励全相同 → advantage 全 0 → 这一组**零梯度**；group_size=1 → 永远学不动。SFT 起点有 **18.5%** 的组零方差，下载的 GRPO 终点有 **96.3%**（65 组全对、13 组全错）。
12. 下载的 GRPO 模型 greedy **82.7%**、pass@1 0.821，但有 **12 道题 P(正确)<0.01**（SFT 时最低也有 0.056）——这些题被 GRPO“压死”了，全错组没有梯度，再也救不回来。于是 **pass@k 曲线交叉**：k=8 时 SFT 0.915 > GRPO 0.834。关掉 dropout 也会发生：60 iter 后被压死的恰好是答案为 15 的全部 4 道题——“15”在 SFT 分布下被当错误答案采到的次数是当正确答案的 2.52 倍（全表最高），共享的输出被负 advantage 整体压低（F49）。
13. 期望意义下，组均值 baseline 的梯度 = 真梯度 × **(G−1)/G**（穷举验证：G=2→0.5，G=4→0.75），RLOO 才严格无偏；G=1 时恰好为 0。
14. CPU 可行性：预训练 4.1–6.4 s/iter（≈23–35 小时，不可行；缩到 0.9M 参数的小配置可在 10 分钟内跑完，见 §2.1）；SFT 7.35 s/iter（ctx 256）或 4.27 s/iter（ctx 137，与 256 数学等价）；GRPO 约 19 s/iter（逐条 batch=1 采样、无 KV cache，**是生成瓶颈**）。GPU 上整章大约 1 GPU 小时以内。
15. **CPU 端到端已跑通**（§4.4）：把模型缩到 0.9M 参数，预训练 9 分钟 + SFT 3 分钟 + GRPO（关 dropout、lr 1e-4）5.4 分钟，加法准确率 12.3% → 72.8%；开 dropout 的同样设置只到 19.8%。

---

## 1. 推断的章节叙事（用官方目录核对过）

### 1.1 证据

* 我抓取了 `https://www.oreilly.co.jp/books/9784814401611/` 的目录（页面 UTF-8），第 3 章标题是「3章 学習【基本編】」，属于「第I部 基本編――CodeBotを作る」。第 0 章「LLMができるまで」与 I.1–I.4（開発の流れ / 事前学習 / SFT / 強化学習）先做全景预告。
* 第 3 章小节（括号内为中文意译）：

| 小节 | 子小节 | 对应文件 |
|---|---|---|
| 3.1 事前学習（预训练） | 3.1.1 データセットの確認（查看数据集）/ 3.1.2 デバイスの設定（设备设置，即 `get_device`）/ 3.1.3 事前学習のコード（前半部分）/ 3.1.4 （後半部分）/ 3.1.5 学習結果の確認（看训练结果） | `01_pretrain.py`，`codebot/utils.py:get_device` |
| 3.2 テキスト生成（文本生成） | 3.2.1 生成方法 / 3.2.2 温度パラメータ / **3.2.3 Top-kとTop-pサンプリング** / 3.2.4 実装 / 3.2.5 実行 | `02_generate.py`（**无 top-k/top-p 代码**） |
| 3.3 SFT | 3.3.1 SFT 数据格式 / 3.3.2 Alpaca形式でトークン化 / 3.3.3 マスク処理 / 3.3.4 パディング / 3.3.5 SFTの実装 / 3.3.6 学習結果の確認 | `03_alpaca.py`（3.3.2），`03_sft.py`（3.3.3–3.3.6） |
| 3.4 SFTモデルとの対話 | 3.4.1 インタラクティブな対話 / **3.4.2 モデルの計算能力**（模型的计算能力） | `04_chat.py` |
| 3.5 強化学習と言語モデル | 3.5.1 对应关系 / 3.5.2 軌跡（轨迹）/ 3.5.3 報酬と収益（奖励与回报）/ 3.5.4 语言模型中的奖励与回报 | **无代码（05 缺号）** |
| 3.6 方策勾配法 | 3.6.1 直接改进策略 / 3.6.2 策略梯度推导 / 3.6.3 策略梯度的近似 | **无代码（06 缺号）** |
| 3.7 ベースライン | 3.7.1 baseline 的想法 / 3.7.2 带 baseline 的策略梯度 / **3.7.3 GRPO：グループ平均をベースラインにする** | **无代码（07 缺号）** |
| 3.8 オフポリシーとクリッピング | 3.8.1 on-policy 与 off-policy / 3.8.2 重要度采样推梯度 / 3.8.3 重要度权重的近似 / 3.8.4 clipping 稳定化 | **无代码（08 缺号）** |
| 3.9 GRPO | 3.9.1 算法 / 3.9.2 実装 / 3.9.3 実験結果 | `09_grpo.py` |

* 与本章相关的专栏与附录：专栏「Transformerと文章の確率計算」「**GRPOとDr.GRPO**」「BPEのタイブレーク」；附录 C「**ベースライン付き方策勾配の不偏性の証明**」（带 baseline 的策略梯度无偏性证明）。
* 仓库里 `ch02/` 也同样“按小节编号、缺号即纯理论小节”（ch02 缺 04、05），说明这是全书一致的命名约定。

### 1.2 概念出场顺序（教程可照此编排）

1. **数据就是一长串 token id**（3.1.1）：`tiny_codes.bin` 是第 1 章用 BPE 把 `tiny_codes.txt` 编码后的 uint16 数组。
2. **设备选择**（3.1.2）：`cuda → mps → cpu`。
3. **窗口与错位一位**（3.1.3）：`TokenDataset` 用滑动窗口取 `x=tokens[i:i+C]`、`y=tokens[i+1:i+C+1]`——“预测下一个 token”在数据层面就是错开一位。
4. **训练循环**（3.1.4）：GPT 前向 → 把 (B, C, V) 摊平成 (B·C, V) 做 cross-entropy → AdamW。
5. **看 loss 曲线**（3.1.5）。
6. **自回归生成**（3.2.1）：只取最后一个位置的 logits；超出上下文就截断最旧的 token；遇到 `<|endoftext|>` 停。
7. **温度**（3.2.2）与 **Top-k/Top-p**（3.2.3，仅文字）。
8. **指令数据与模板**（3.3.1–3.3.2）：Alpaca 格式 `### Instruction:\n…\n\n### Response:\n…<|endoftext|>`。
9. **只在回答上算 loss**（3.3.3 掩码）与 **补齐长度**（3.3.4 padding）。
10. **SFT 训练 = 预训练同一套循环，换数据、换 loss 掩码、换初始化**（3.3.5）。
11. **聊天**（3.4.1）→ **发现它不会算术**（3.4.2）→ 引出强化学习。
12. **把 LM 看成 RL 的策略**（3.5）：状态 = 已生成前缀，动作 = 下一个 token，轨迹 = 整个回答，奖励只在末尾。
13. **策略梯度**（3.6）→ **baseline 降方差、且无偏**（3.7 + 附录 C）→ **GRPO 用组均值当 baseline**（3.7.3）。
14. **重复使用样本需要重要度采样，ratio 用逐 token 近似，再用 clip 稳住**（3.8）。
15. **把以上拼成 GRPO 并实验**（3.9）。

> 斋藤的组织手法在这一章很明显：每段代码前先有“确认”小节（数据集の確認、学習結果の確認、実行），理论（3.5–3.8）集中放在“发现问题（不会算术）”之后、实现（3.9）之前；`codebot/utils.py` 里的 ❶❷❸ 是书中“代码标号 + 正文逐条解释”的痕迹；`generate` 先在 02 里内联讲解，后被收进 `codebot/utils.py` 供 04、09 复用（两者 AST 完全相同，见 F12）。

---

## 2. 逐脚本解析

### 2.1 `ch03/01_pretrain.py`（§3.1 预训练）

**目的**：在 TinyCodes（Python 代码小语料）上从零预训练 CodeBot。

**关键结构**

```python
class TokenDataset(Dataset):
    def __init__(self, tokens, context_len)        # tokens: np.uint16 数组 → torch.long
    def __len__(self):  return len(self.tokens) - self.context_len
    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.context_len]
        y = self.tokens[idx+1:idx+self.context_len+1]
        return x, y
```

超参：`context_len=256, vocab_size=1000, batch_size=32, learning_rate=3e-4, max_iters=20000, embed_dim=384, n_head=6, n_layer=6, ff_dim=4*384, dropout_rate=0.1`；`torch.optim.AdamW(model.parameters(), lr=3e-4)`（默认 betas=(0.9,0.999)、weight_decay=0.01）。**没有** LR schedule、梯度裁剪、验证集、中途 checkpoint、随机种子。

**核心算法（本质只有这几行）**

```python
ids = np.fromfile(data_path, dtype=np.uint16)
dataloader = DataLoader(TokenDataset(ids, 256), batch_size=32, shuffle=True)
data_iter = cycle(dataloader)                        # 無限ループ化
for i in pbar:
    batch_x, batch_y = next(data_iter)
    logits = model(batch_x)                          # (B, C, V)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**日文注释翻译**：設定=设置；ハイパーパラメータ=超参数；データセットクラス=数据集类；データ準備=数据准备；モデル、オプティマイザ=模型与优化器；`パラメータ数`=参数量；無限ループ化=变成无限循环；結果を保存=保存结果。

**实测输出**（补丁副本：只改了 `os.chdir`、输出路径、`max_iters`，加了 `torch.manual_seed(0)` 和每 25 步一行打印；超参不变）：

```
パラメータ数: 11,121,640 (11.1M)
iter     0 loss 6.9895  t=5.3s
iter    25 loss 5.7115  t=107.4s
iter    29 loss 5.7377  t=124.1s
losses[0]=6.9895 last=5.7377 min=5.6855; wall 124.3s = 4.142 s/iter
```

（`max_iters=30`；第一步 loss 6.9895 ≈ ln 1000 + 0.08。）先前 10 iter 的计时运行（更重的争用下）：`losses[0]=6.9895 last=6.0081 min=6.0081; wall 63.9s = 6.389 s/iter`。

**墙钟**：4.1–6.4 s/iter（视争用而定）→ 完整 20000 iter ≈ **23–35 小时**，CPU 不可行。

**CPU 小配置**（另一个补丁副本，代码路径不变，只缩模型：E=128, H=4, L=4, ff=512, C=128, lr=1e-3, 1500 iter；参数量按同一公式为 128,000+1,000+16,384+4×197,760+256 = 936,680）：

```
パラメータ数: 936,680 (0.9M)
iter     0 loss 6.9300  t=2.6s
iter   100 loss 4.7412  t=43.2s
iter   200 loss 4.1514  t=79.9s
iter   500 loss 3.3422  t=184.8s
iter  1000 loss 2.5000  t=360.7s
iter  1499 loss 2.5181  t=533.3s
losses[0]=6.9300 last=2.5181 min=2.3793; wall 533.4s = 0.356 s/iter
```

用它从 `def` 采样（T=1，seed 0，80 token）——已经是“Python 的样子”，但比 11M 模型更乱：

```
def generate_length(str1, lc1):
    long2 = [1,n1]
    num2 = 12

    if num % i > lst_numbers:
        return max = list2
    if num % i == 0:
        return 1
    return num1
```

这个小配置是 §4.4「CPU 端到端」的起点。

### 2.2 `ch03/02_generate.py`（§3.2 文本生成）

**目的**：用预训练模型从 `"def"` 开始采样 5 段代码（`max_new_tokens=200, temperature=1.0`）。

**签名**：`@torch.no_grad() def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0) -> str`

**核心**

```python
model.eval()
ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
generated_ids = ids.clone()
for _ in range(max_new_tokens):
    if ids.size(1) > model.max_context_len:
        ids = ids[:, -model.max_context_len:]           # 超长就丢最旧的 token
    logits = model(ids)[:, -1, :]                       # 只用最后一个位置
    if temperature == 0: next_id = logits.argmax(dim=-1, keepdim=True)
    else: next_id = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1)
    if next_id.item() == tokenizer.end_token_id: break  # EOS 不会被追加
    ids = torch.cat((ids, next_id), dim=1); generated_ids = torch.cat((generated_ids, next_id), dim=1)
return tokenizer.decode(generated_ids[0].tolist())      # 返回的是 prompt + 续写
```

**注释翻译**：生成の開始プロンプト=生成的起始 prompt；生成するトークン数の上限=生成 token 数上限；温度パラメータ（高いほどランダム）=温度（越高越随机）；評価モード=评估模式；パラメータのデバイスを取得=取参数所在设备；コンテキスト長を超えた場合、古いトークンを切り捨てる=超出上下文长度时丢弃旧 token；最後の位置のロジットを取得（次トークンの予測）=取最后位置的 logits（预测下一个 token）；終了トークンが生成されたら停止=生成结束 token 即停止；デコードして返す=解码后返回；サンプル=样本。

**实测输出**（原样运行，未 seed，18.1 s）：

```
--- サンプル 1 ---
def subElement(arr, target):
    for i, len(arr) in enumerate(arr):
        if arr[i] != target:
            return i
    return -1

arr = [1, 3, 8, 10]
target = 4
print(subElement(arr, target))
# Output: 4

--- サンプル 2 ---
def constraints(n):
    score = 1
    result = 0
    for i in range(1, n + 1):
        score += (n - i) * ((s - score))
    return result
...
--- サンプル 4 ---
def get_words(sentence):
    words = sentence.split(' ')
    word_count = {}
    for word in words:
        if word not in word_count:
            word_dict[word] = 1
...
--- サンプル 5 ---
def generate_prime_numbers(n):
    primes = []
    i = 2
    while len(primes) < n and is_prime(i):
        if prime.append(i)
        i += 1
    return primes
```

语法“像 Python”，语义经常不通（变量名前后不一致）——这正是 11M 参数、2.7M token 小模型的样子。完整输出见 `out/out_02_generate.txt`。

**注意**：没有 KV cache，每生成一个 token 都把整段上下文重算一遍（第 5 章 StoryBot 才加 KV cache）。

### 2.3 `ch03/03_alpaca.py`（§3.3.2 Alpaca 格式）

**目的**：把 `{'instruction','response'}` 套进模板并 tokenize。

```python
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
ids = tokenizer.encode(text)
```

**注释翻译**：トークナイザの読み込み=加载分词器；JSONデータの読み込み=读 JSON；1つ目のサンプルを取り出す=取第一个样本；Alpaca形式に変換=转成 Alpaca 格式；トークン化=分词。

**实测输出**（2.1 s）：

```
{'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}
### Instruction:
Hello

### Response:
Hello. What can I help you with?<|endoftext|>
[35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 932, 657, 63, 999]
```

**与源码注释不一致**：注释里写的是 `…, 108, 112, 930, 657, 63, 999]`，实际是 **932**。用仓库自带的 `merge_rules.pkl` 解码：`932 = b' you'`（由 `(398 ' y', 334 'ou')` 合并），`930 = b'alize'`。推测注释写于 BPE 平局规则修改之前（`tokenizer.py` 里 `# best_pair = max(counts, key=counts.get)` 被替换为带 `(count, pair[0], pair[1])` 的确定性平局规则；书中有专栏「BPEのタイブレーク」）。逐 token 字节：`'#','#','#',' In','str','u','ction',':','\n','Hello','\n','\n','#','#','#',' Re','sponse',':','\n','Hello','.',' W','h','at',' c','an',' I',' he','l','p',' you',' with','?','<|endoftext|>'`——34 个 token，其中模板占 18 个（`### Instruction:\n` 9 个 + `\n\n### Response:\n` 9 个），再加 1 个 EOS；因此任何 GRPO 加法 prompt 都是 9 + 4（`3`,`+`,`4`,`=`）+ 9 = 22 个 token。

### 2.4 `ch03/03_sft.py`（§3.3.3–3.3.6 SFT）

**目的**：从 `model_pretrain.pt` 出发，在 1092 条指令数据上做监督微调。

**关键结构**

```python
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len)
    def _create_sample(self, instruction, response) -> (ids, labels)
```

**核心（掩码 + 错位 + 补齐）**

```python
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
response = f"{response}<|endoftext|>"
prompt_ids, response_ids = tokenizer.encode(prompt), tokenizer.encode(response)
ids    = prompt_ids + response_ids
labels = [-100] * len(prompt_ids) + response_ids     # プロンプト部分は-100でマスク
ids, labels = ids[:-1], labels[1:]                   # 入力と正解を1つずらす
pad_len = context_len - len(ids)
if pad_len > 0:  ids += [0] * pad_len; labels += [-100] * pad_len   # パディングIDとして0を使用
elif pad_len < 0: ids, labels = ids[:context_len], labels[:context_len]
...
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1), ignore_index=-100)
```

超参：`context_len=256, batch_size=32, learning_rate=3e-4, max_iters=500`（35 iter/epoch → **14.65 epoch**）。

**注释翻译**：プロンプトとレスポンスをフォーマット=格式化 prompt 与 response；入力系列とラベルの作成（プロンプト部分は-100でマスク）=构造输入与标签（prompt 部分用 −100 屏蔽）；言語モデル用にシフト（入力と正解を1つずらす）=为语言模型错位（输入与答案错开一位）；context_lenに合わせてパディングまたは切り詰め=按 context_len 补齐或截断；パディングIDとして0を使用=用 0 作 padding id；学習ループ=训练循环；モデルの保存=保存模型。

**实测输出**
* 原配置 4 iter 计时：`losses[0]=4.8989 last=2.9073 min=2.8191; wall 29.4s = 7.350 s/iter`。
* **复现运行**（补丁：`context_len=137`、`max_iters=300`、seed 0、输出到 scratch。最长样本 138 token ⇒ 输入长 137，故 137 与 256 在数学上等价——causal mask 下尾部 padding 不影响前面位置，且 padding 位 label 全是 −100，见 F16；唯一差别是 dropout 随机数流不同）：

```
iter    0 loss 4.8791  t=44.0s
iter   25 loss 1.3936  t=173.0s
iter   50 loss 1.4206  t=257.8s
iter  100 loss 0.8280  t=449.5s
iter  150 loss 0.3775  t=668.3s
iter  200 loss 0.2566  t=871.4s
iter  250 loss 0.3006  t=1066.1s
iter  299 loss 0.2140  t=1280.0s
losses[0]=4.8791 last=0.2140 min=0.0498; wall 1280.4s = 4.268 s/iter
```

* 与下载的 `model_sft.pt`（500 iter）对比（eval 模式）：

| 模型 | SFT 集 response loss | 81 道加法 greedy 准确率 | greedy 回复（Hello / print hello / make variable x / 3+4= / Who are you?） |
|---|---|---|---|
| 下载的 model_sft.pt | 0.0276 | 54.3% | `Hello. What can I help you with?` / `print('hello')` / `x = 'hello'` / `8` / `I'm CodeBot, a Python coding assistant. How can I help you?` |
| 我的复现（300 iter，ctx 137） | 0.0590 | 48.1% | `Hello. What do you need?` / `print('hello')` / `x = 'test'` / `7` / `I'm an AI assistant for Python programming. What do you need?` |

### 2.5 `ch03/04_chat.py`（§3.4 对话）

**目的**：交互式聊天。`format_prompt(user_message)` 返回 `"### Instruction:\n{user_message}\n\n### Response:\n"`；`generate(..., max_new_tokens=200, temperature=1.0)`；取 `"### Response:"` 之后的部分并 `strip()`；含换行就换行显示。注释掉的一行 `# model_path = 'codebot/model_grpo.pt'` 表示同一脚本也用来和 GRPO 模型聊。

**注释翻译**：モデルとトークナイザの読み込み=加载模型与分词器；プロンプトのフォーマットと生成=格式化 prompt 并生成；アシスタントの応答部分のみ抽出=只取助手回答部分；改行が含まれているかで出力形式を切り替える=按是否含换行切换输出格式。

**实测输出**（补丁：模型路径由环境变量给、`torch.manual_seed(0)`、stdin 读到 EOF 就退出并回显输入；温度仍是 1.0）：

| 输入 | model_sft.pt | model_grpo.pt |
|---|---|---|
| Hello | Hello. What can I help you with? | Hello. What can I help you with? |
| Who are you? | I'm CodeBot, a Python coding assistant. How can I help you? | 同左 |
| print hello | print('hello') | print('hello') |
| make variable x | x = 2 | x = 2 |
| plus 9 and 6 | value = 9 + 6 | value = 9 + 6 |
| **3+4=** | **6** ✗ | **7** ✓ |
| 5+7= | 11 ✗ | 11 ✗（GRPO 后被“压死”的题之一，见 §2.6） |
| write a function that returns the square of a number | `def square__num(n):\n    return n**2` | `def square_of_numbers(n):\n    return sum(16)` |
| What is the capital of France? | First Frab launcular expression … | France allows capital. Useful for parent. … |

（原脚本 `while True: input()` 遇到管道 EOF 会抛 `EOFError`，所以需要上面的补丁才能非交互运行。注意它是**单轮**模板：每次输入都重新套模板，不保留对话历史，也没有 system prompt。）

### 2.6 `ch03/09_grpo.py`（§3.5–3.9 GRPO）——重点

#### 2.6.1 结构与签名

```python
class GRPODataset(Dataset):
    def __init__(self, tokenizer)                    # 81 条 (prompt, i+j)，i,j ∈ 1..9
    def get_batch(self, prompts, responses, device)  # -> ids (N, Lmax) long, mask (N, Lmax) float
def calculate_reward(ground_truth, response)         # 最后一个整数 == 正确答案 → 1.0，否则 0.0
def generate_group(model, tokenizer, prompts, gts, group_size)   # -> prompts×G, responses, advantages (N,)
def compute_probs(model, ids)                        # -> (N, L-1)：实际下一个 token 的概率
def grpo_loss(model, old_model, ids, mask, advantages, epsilon=0.2)
```

超参：`learning_rate=7e-6, max_iters=500, n_update_per_generation=2, eval_interval=10, epsilon=0.2, group_size=8, batch_size=32`。

#### 2.6.2 核心代码

```python
# 组内 advantage（只减均值）
rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
advantages = rewards - rewards.mean()

# 逐 token 概率与比值
probs = compute_probs(model, ids)                       # 当前模型（train 模式，dropout 开）
with torch.no_grad(): old_probs = compute_probs(old_model, ids)   # 旧模型（eval 模式）
ratio = probs / (old_probs + 1e-8)                      # 概率空间，不是 log 空间
unclipped = ratio * advantages.unsqueeze(-1)
clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages.unsqueeze(-1)
token_objective = torch.min(unclipped, clipped) * mask[:, 1:]
return -token_objective.sum() / ids.size(0)             # 除以序列数 B·G，不除 token 数

# 训练循环
for i in pbar:
    prompts, gts = next(data_iter)
    old_model.load_state_dict(model.state_dict())       # π_old ← π_θ
    all_prompts, all_responses, all_advantages = generate_group(old_model, tokenizer, prompts, gts, group_size)
    ids, mask = dataset.get_batch(all_prompts, all_responses, device)
    for _ in range(n_update_per_generation):            # 同一批 rollout 更新 2 次
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, all_advantages, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    if i % eval_interval == 0: ...                      # 81 题 greedy 准确率（在本轮更新之后）
```

写成公式（教程可直接用）：对 batch 中每个 prompt q 采 G 个回答 oᵢ ~ π_old(·|q)（T=1），rᵢ ∈ {0,1}，

  Aᵢ = rᵢ − (1/G)Σⱼ rⱼ，ρᵢ,ₜ = π_θ(oᵢ,ₜ | q, oᵢ,<ₜ) / (π_old(oᵢ,ₜ | q, oᵢ,<ₜ) + 10⁻⁸)

  L(θ) = −(1/(|B|·G)) Σ_q Σᵢ Σₜ min(ρᵢ,ₜ Aᵢ, clip(ρᵢ,ₜ, 0.8, 1.2) Aᵢ)

对照 DeepSeekMath 原版 GRPO：(1/G)Σᵢ(1/|oᵢ|)Σₜ[min(…) − β·KL(π_θ‖π_ref)]，且 Aᵢ=(rᵢ−mean)/std。09_grpo.py 去掉了 std、1/|oᵢ| 与 KL——即 Dr. GRPO 的做法（Liu et al. 2025），书中专栏「GRPOとDr.GRPO」应当在讲这件事。

#### 2.6.3 逐项回答任务里的问题

| 问题 | 09_grpo.py 的做法 | 实测/验证 |
|---|---|---|
| reward | 正则 `(-?\d+)` 取**最后一个**整数，等于 i+j 得 1.0；否则 0.0；裸 `except:` 吞掉任何异常返回 0.0 | `'8 7'`→1，`'7 8'`→0，`'7.0'`→0，`'07'`→1，`'-7'`→0，`'1 2'`(gt=12)→0；`re` 未导入时一律 0.0（F20, F21） |
| group size | G=8；batch 32 prompt → 每步 256 条序列（第 3 个 batch 只有 17 prompt → 136 条） | DataLoader 批大小 [32, 32, 17]（F34） |
| advantage | `r − mean(r)`，**不除 std** | 组内和恰为 0；k/8 正确时 A₊=1−k/8、A₋=−k/8（F22） |
| 全部奖励相同 | A 全 0 → 该组对 loss 和梯度贡献 0（不会 NaN，因为没有除 std）；但它们**仍计入分母 B·G**，会稀释其余组的梯度（Adam 会部分抵消整体缩放），也照样花算力 | 若改成除 std 且不加 eps → NaN（F22）；SFT 起点 18.5% 组零方差，GRPO 终点 96.3%（F36） |
| group size = 1 | `r − r = 0` → 永远零梯度，只剩 AdamW 的 decoupled weight decay（每步 p←p·(1−lr·wd)=p·(1−7e-8)，F47） | 表格 GRPO：G=1 跑 150 步准确率纹丝不动（F39）；期望梯度 = 真梯度 × (G−1)/G = 0（F24） |
| ratio / clip | 概率空间比值 + 1e-8；ε=0.2；`torch.min(unclipped, clipped)` | ratio=1 时梯度 = REINFORCE-with-baseline 梯度（F27）；clip 导数表（F25）；同一批 rollout 两次更新后 clipfrac 12.7%→24.4%（F46）。clip 只在“重复使用样本”（n_update>1）时才有意义 |
| KL | **没有**（无 reference model、无 KL 项） | — |
| reference / old policy | `old_model` 每轮 `load_state_dict(model.state_dict())`，用它采样、用它算 old_probs；没有独立的 π_ref | 第 1 次更新本应 ratio≡1，但 dropout 让它≠1（F30） |
| 归一化 | Σ_tokens / (B·G)：长回答权重更大，loss 数值 = −Σ Aᵢ·len(oᵢ)/N（ratio=1 时） | 0.007324 两种算法一致（F28） |
| EOS | 生成时遇到 EOS 就停且不追加 → response 字符串不含 EOS → **EOS 的概率不在目标里** | batch 中不存在 id 999，mask 只覆盖数字 token（F33） |
| 记录了什么 | 只有 tqdm 上的最后一次 `loss` 和每 10 步的 greedy 准确率；画图存成 `loss_grpo.png`（名字叫 loss，画的却是 accuracy） | — |
| 应该记录什么 | 见下表（实测值） | — |

**建议记录的指标（实测值来自我的插桩版 `grpo_fast.py`；只用 eval 模式额外前向一次，不改变训练；“base”= 脚本原样，“nodrop”= 更新时关 dropout）**

| 指标 | 含义 | 实测（base 前 10 iter → 后 10 iter；nodrop 后 10 iter） |
|---|---|---|
| reward/mean | rollout 的平均奖励 = T=1 下的采样准确率 | 0.458 → 0.679；nodrop 0.929 |
| reward/std、zero_var_frac | 组内零方差（全对/全错）占比 → 没有学习信号的比例 | 0.21 → 0.45；nodrop 0.80 |
| pg_loss | 即脚本里的 loss；**不是进度指标** | +0.100 → +0.155（dropout 偏差，F48）；nodrop ≈ 0 |
| grad_norm（裁剪前） | `clip_grad_norm_` 的返回值，免费可得 | 4.0 → 15.2（始终 >1，每步都被裁剪）；nodrop 2.6 |
| clipfrac | 被 clip 截断梯度的 token 占比（同一批 rollout 重复更新时才有意义） | 3.9% → 0.4%；nodrop 2.0%；单批两次更新的例子 12.7% → 24.4%（F46） |
| approx KL(π_old‖π_θ)（k3 = r−1−log r，r=π_θ/π_old，样本来自 π_old） | 每轮策略移动了多少 | 4.1e-3 → 8.0e-4；nodrop 2.2e-3 |
| entropy（response token） | 探索还剩多少；塌缩预警 | 0.76 → 0.42 nats；nodrop 0.13；下载的 GRPO 终点首 token 熵 0.066 |
| response length | 长度漂移 | 1.45–1.51 token，基本不变 |
| ratio min/max | 数值健康度 | 单批两次更新后 [0.22, 2.70]（F46） |
| greedy acc（eval） | 真正的进度 | 54.3% → 70.4%（base，60 iter）/ 95.1%（nodrop，60 iter） |

#### 2.6.4 实测运行

**(a) 原脚本，只改路径/迭代数/seed**（`max_iters=12`，其余超参不变；保存到 scratch）：

```
iter    0 loss +0.1141 acc 55.6% t=59.7s
iter    1 loss +0.1100 acc 55.6% t=116.7s
iter    2 loss +0.1031 acc 55.6% t=155.1s
iter    3 loss +0.1019 acc 55.6% t=212.1s
iter    4 loss +0.1979 acc 55.6% t=256.6s
iter    5 loss +0.1333 acc 55.6% t=289.6s
iter    6 loss +0.1578 acc 55.6% t=358.9s
iter    7 loss +0.2264 acc 55.6% t=413.4s
iter    8 loss +0.1356 acc 55.6% t=453.3s
iter    9 loss +0.0482 acc 55.6% t=513.1s
iter   10 loss +0.1047 acc 58.0% t=577.0s
iter   11 loss +0.0647 acc 58.0% t=608.9s
accuracies: [55.55555555555556, 58.0246913580247]
wall 609.6s = 50.80 s/iter
```

注意：iter 0 打印的 55.6% 是**本轮 2 次更新之后**的准确率；真正的 SFT 起点是 54.3%（F38）。负载较轻时第一次计时为 24.4 s（含评估）/ 18.9 s / 10.7 s（第 3 个 batch 只有 17 prompt）。

**(b) 下载的 checkpoint 对比**（81 题；T=1 每题采 8 个）：

| | SFT（model_sft.pt） | GRPO（model_grpo.pt） |
|---|---|---|
| greedy 准确率 | 54.3%（44/81） | 82.7%（67/81） |
| T=1 平均奖励 | 0.440 | 0.824 |
| 零方差组（全对+全错） | 15/81 = 18.5%（7+8） | 78/81 = 96.3%（65+13） |
| k-直方图（8 个里对 k 个） | {0:8, 1:13, 2:13, 3:5, 4:17, 5:7, 6:5, 7:6, 8:7} | {0:13, 1:1, 6:1, 7:1, 8:65} |
| 平均 P(正确答案) | 0.450 | 0.821 |
| P(正确)<0.01 的题数 | 0 | 12（3+5, 3+9, 5+3, 5+5, 5+7, 6+6, 6+8, 7+5, 7+7, 7+8, 8+6, 8+7） |
| 首个答案 token 的平均熵 | 0.592 nats | 0.066 nats |
| 答案 NLL（−log P(正确)） | 0.994 | **1.121**（更差！被压死的题拉高了） |
| response 长度 | 平均 1.47 token，最长 2；648/648 以 EOS 结束 | 平均 1.46，最长 8 |
| 重 tokenize 不一致 | 0/648 | 0/648 |
| pass@1/2/4/8/16/64/256 | 0.450/0.631/0.799/**0.915**/0.973/0.999/1.000 | 0.821/0.829/0.831/**0.834**/0.839/0.861/0.900 |

示例：`3+4=` 的 8 个采样，SFT 为 `['6','6','6','8','8','8','7','8']`，GRPO 为 8 个 `'7'`；`9+9=` SFT 为 `['18','18','17','17','16','18','13','16']`。

**(c) 插桩版 GRPO（批量采样，算法与原脚本一致）**：`scripts/grpo_fast.py` 用 `exec` 直接从 `09_grpo.py` 取出 `GRPODataset / calculate_reward / compute_probs / grpo_loss` 原函数；唯一改动是把 256 次逐条采样换成一次批量采样（81 个 prompt 都是 22 个 token，无需 padding），统计意义上等价，随机数流不同。超参与原脚本相同（lr 7e-6、G=8、batch 32、ε=0.2、每批 2 次更新、clip_grad_norm 1.0），每 iter 额外用 eval 模式前向一次记录指标（不改变训练）。CPU 上约 17–21 s/iter（争用下）。

greedy 准确率（%，81 题，每 10 iter 测一次；起点都是 SFT 的 54.3%）：

| 运行（seed 0，60 iter） | it0 | it10 | it20 | it30 | it40 | it50 | it59 |
|---|---|---|---|---|---|---|---|
| **base**：脚本原样（更新时 dropout 开） | 55.6 | 70.4 | 74.1 | 71.6 | 74.1 | 74.1 | 70.4 |
| **nodrop**：更新前 `model.eval()`（唯一差别） | 58.0 | 81.5 | 90.1 | 92.6 | 93.8 | 93.8 | **95.1** |
| **stdnorm**：A 除以 (std+1e-4)（dropout 仍开） | 54.3 | 70.4 | 69.1 | 70.4 | 71.6 | 67.9 | 70.4 |

stdnorm 末段：reward 0.663、零方差 0.488、entropy 0.404、裁剪前 grad_norm **37.1**（advantage 被放大，比 base 的 15.2 更大；因为每步都被裁到 1.0、又有 Adam 归一化，所以最终更新幅度与 base 相近）、loss +0.228。单 seed 下与 base 无明显差别——在 dropout 开着时，std 归一化不是主要矛盾。

窗口均值（前 10 iter → 最后 10 iter）：

| 指标 | base it0–9 | base it50–59 | nodrop it0–9 | nodrop it50–59 |
|---|---|---|---|---|
| reward/mean（T=1 采样准确率） | 0.458 | 0.679 | 0.537 | **0.929** |
| zero_var_frac（零方差组占比） | 0.212 | 0.447 | 0.173 | **0.804** |
| loss（第 2 次更新） | +0.100 | +0.155 | **−0.011** | +0.011 |
| grad_norm（裁剪前，第 1 次更新） | 4.04 | **15.15** | 3.20 | 2.58 |
| clipfrac（两次更新后） | 3.9% | 0.4% | 7.9% | 2.0% |
| KL_k3(π_old‖π_θ) | 4.1e-3 | 8.0e-4 | 7.3e-3 | 2.2e-3 |
| entropy（response token，nats） | 0.762 | 0.417 | 0.705 | **0.133** |
| response 长度（token） | 1.49 | 1.46 | 1.45 | 1.49 |

读法：(1) 关掉 dropout 是本章 GRPO **最大的单点改进**：60 iter 就到 95.1%，超过下载的、用原脚本训了 500 iter 的 `model_grpo.pt`（82.7%）；(2) 开 dropout 时 loss 一直为正、裁剪前梯度范数越训越大（4 → 15），与 F48 的“dropout 偏差”一致；关掉后 loss 在 0 附近、梯度范数稳定；(3) 学得越好，零方差组越多（nodrop 末段 80%），有效 batch 越来越小——这就是用户过去问到的“reward 饱和、失去方差”。以上为单 seed 结果，第二个 seed 见下。

终点模型的精确 pass@k（由答案分布算出，F37 的同一方法）：

| 模型 | pass@1 | pass@8 | pass@64 | P(正确)<0.01 的题数 |
|---|---|---|---|---|
| SFT（下载） | 0.450 | 0.915 | 0.999 | 0 |
| 我复现的 SFT（300 iter） | 0.305 | 0.835 | 0.996 | 0 |
| GRPO（下载，500 iter，dropout 开） | 0.821 | 0.834 | 0.861 | 12 |
| base（60 iter） | 0.678 | 0.872 | 0.964 | 2（7+8, 8+7） |
| stdnorm（60 iter） | 0.662 | 0.864 | 0.968 | 2 |
| nodrop（60 iter） | **0.925** | **0.951** | 0.957 | 4（答案都是 15） |

nodrop 在 k≤8 时全面胜出，但 pass@64 已低于 SFT（0.957 < 0.999）——曲线照样交叉，只是交叉点后移。

**seed 1 复核**（40 iter，其余完全相同）：

| 运行（seed 1，40 iter） | it0 | it10 | it20 | it30 | it39 | 末 10 iter：reward / 零方差 / entropy / 裁剪前 grad_norm |
|---|---|---|---|---|---|---|
| base（dropout 开） | 59.3 | 65.4 | 72.8 | 79.0 | 74.1 | 0.629 / 0.303 / 0.537 / 6.39 |
| nodrop（更新前 `model.eval()`） | 59.3 | 81.5 | 86.4 | 91.4 | **95.1** | 0.860 / 0.557 / 0.281 / 3.56 |

两个 seed 结论一致：关 dropout 后 30 iter 左右就到 91–93%，开 dropout 时在 70%–79% 徘徊。由于两种设置在第 0 轮用的是同一批 prompt 与同一批 rollout（同 seed、dropout 开始消耗随机数之前），it0 的差别（seed 0：55.6 vs 58.0；seed 1：两者都是 59.3）只可能来自更新本身；之后两条轨迹的随机数流才分叉。

### 2.7 `codebot/utils.py`

* `generate(...)`：与 02 的版本 **AST 完全相同**（F12），只是注释多了 ❶ プロンプトをトークン化（把 prompt 分词）、❷ トークン生成ループ（token 生成循环）、❸ デコードして返す（解码返回）；`コンテキスト長を超えた場合は末尾のみ使用`=超出上下文时只用末尾；`次のトークンの予測`=预测下一个 token。
* `get_device()`：`cuda` → `mps` → `cpu`。
* 副作用：`generate()` 会把模型切到 `eval()` 且不还原（F11）。

### 2.8 `codebot/model.py`（参考）

* `MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate=0.1)`：Q/K/V/O 四个无 bias 的 Linear；`scores = QKᵀ/√D`，`torch.tril` 造 causal mask 后 `masked_fill(-inf)`；对 attention 权重和输出各一个 dropout。显式物化 (B,H,C,C)。
* `LayerNorm`、`GELU`（tanh 近似）两个自写类**没有被使用**：`Block` 用的是 `nn.LayerNorm`，`FFN` 用的是 `nn.GELU()`（erf 精确版），自写版本作为注释保留（`# LayerNorm(embed_dim)`、`# GELU()`）。二者差异：LayerNorm 最大 4.8e-7；GELU tanh 近似 vs erf 最大 4.74e-4（x≈2.70），与 `nn.GELU(approximate='tanh')` 相差 2.4e-7（F41）。
* `Block`：pre-LN 残差 `x = x + attn(norm1(x)); x = x + ffn(norm2(x))`。
* `GPT`：token embedding + learned 位置 embedding（只有 256 个位置，第 257 个位置直接 `IndexError`，F10）→ dropout → 6 个 Block → LayerNorm → `unembed`（Linear，**带 bias**）。`self.embed.weight = self.unembed.weight` 共享权重；初始化全部 N(0, 0.02²)、bias 置 0（没有 GPT-2 对残差投影的 1/√(2L) 缩放）。`save/load_from` 把超参和 state_dict 一起存，`torch.load` 默认 `weights_only=True`（torch 2.9），所以加载下载文件是安全的。
* 参数量公式：V·E + V（unembed bias）+ C·E + L·(4E² + 2·E·F + F + E + 4E) + 2E = 384,000 + 1,000 + 98,304 + 6×1,772,928 + 768 = **11,121,640**。

### 2.9 `codebot/tiny_codes_sft.json`

* 1092 条 `{"instruction","response"}`；1011 个不同 instruction；3 对完全重复。
* 粗分类（启发式）：闲聊与编程概念问答 502（`How are you feeling?` → `I don't have feelings. What do you need help with?`；`What's REST?` → `REST uses HTTP methods (GET, POST, PUT, DELETE) for CRUD operations on resources.`）、单行赋值 257（`set amount` → `amount = 50`、`convert string to uppercase` → `upper = my_string.upper()`）、多行代码 188（`create recursive fibonacci`、`read JSON from file`、最长的是 `merge two sorted lists`）、`i+j=` 算术 **81（恰好 9×9 全集且答案全对）**、`print(...)` 64（`output name` → `print('name')`）。另有 `plus 9 and 6 → value = 9 + 6`、`find difference of 15 and 5 → diff = 15 - 5` 这类“自然语言→代码”的算术。3 对重复是 `display end`、`Good afternoon`、`print start`；2 条非 ASCII 来自 `O(n²)`。
* 长度（模板+回答+EOS）：min 24 / 中位 47 / 均值 48.3 / max 138 token；prompt 19–72；response（含 EOS）2–112，均值 21.6。全部 ≤ 257，所以 `pad_len < 0` 的截断分支在这份数据上**从未触发**。
* 每 epoch 23,635 个监督 token / 279,552 个位置（8.455%）。

---

## 3. 可最小化验证的事实清单

> 约定：所有片段都在仓库根目录 `C:\Users\chenf\Projects\_refs\dlfs6` 下用 `PYTHONUTF8=1 uv run python -c "…"` 或脚本运行；“浏览器”列：**是** = 只需 NumPy/纯 Python（Pyodide 可跑；纯 Python 分词器需要 `regex` 包，Pyodide 自带），**否** = 需要 torch 或 checkpoint。所有“实测输出”都是我实际运行得到的。

| # | 断言 | 最小代码 | 实测输出 | 浏览器 |
|---|---|---|---|---|
| F01 | 语料 2,677,398 个 uint16 token，21,835 个 EOS（=文档数），id 范围 9–999（padding id 0 从未出现） | `ids=np.fromfile('codebot/tiny_codes.bin',np.uint16); print(len(ids),ids.min(),ids.max(),(ids==999).sum(),(ids==0).sum())` | `2677398 9 999 21835 0` | 是（需加载 5.35 MB .bin，或用切片） |
| F02 | 窗口错位：`y[:-1]==x[1:]`；`len(TokenDataset)=N−256` | `x=ids[i:i+256]; y=ids[i+1:i+257]; print((y[:-1]==x[1:]).all(), len(ids)-256)` | `True 2677142` | 是 |
| F03 | 参数量 11,121,640 = 公式值 | `m=GPT(1000,256,384,6,6,1536,0.1); sum(p.numel() for p in m.parameters())` | `11,121,640`（公式同值；每 block 1,772,928；非 embedding 10,639,336） | 公式：是；模型：否 |
| F04 | 权重共享：`parameters()` 去重；不去重是 11,505,640 | `m.embed.weight is m.unembed.weight; sum(p.numel() for _,p in m.named_parameters(remove_duplicate=False))` | `True 11505640` | 否 |
| F05 | 初始 loss ≈ ln V + σ²/2，σ≈0.02·√384 | 随机初始化 GPT，seed 0，32 个随机窗口，eval 模式 | `ln(V)=6.9078 init loss=7.0025 logit std=0.3955 (理论 0.3919)`；首个训练步 6.9895 | NumPy 类比：是（均匀 logits 恰为 6.907755；N(0,0.392²) logits 为 6.9694） |
| F06 | `ignore_index=-100` 的 CE = 有效 token 的平均；≠ 逐样本平均 | 随机 logits(4,10,1000)，前 3 位 −100 | torch `7.147117` = NumPy `7.147117`；逐样本平均 `7.183113` | NumPy 部分：是 |
| F07 | 温度：T→0 变 one-hot，T→∞ 变均匀，熵随 T 单调增 | `softmax(z/T)`，z=[2,1,.5,0,−1] | T=0.01 → [1,0,0,0,0] H=0；T=1 H=1.2065；T=100 H=1.6094=ln5 | 是 |
| F08 | `multinomial` 频率收敛到概率 | 抽 200k 次 | 经验 [0.5645,0.2076,0.1252,0.0750,0.0277] vs p [0.5630,0.2071,0.1256,0.0762,0.0280] | 是 |
| F09 | top-k=1 等于 greedy；top-p 截断后重新归一化 | 见 `f12_numpy_more.py (D)` | top-k=2 → [0.731,0.269,0,0,0]；top-p=0.9 → [0.579,0.213,0.129,0.078,0] | 是 |
| F10 | 位置编码只有 256 个：257 个 token 直接报错 | `m(torch.zeros(1,257,dtype=torch.long))` | `IndexError index out of range in self` | 否 |
| F11 | `generate()` 把模型切到 eval 且不还原 | `m.train(); generate(m,tok,'def',3); m.training` | `False` | 否 |
| F12 | 02 里的 `generate` 与 `codebot/utils.py` 的 AST 完全相同 | `ast.dump` 比较 | `True` | 是（纯 Python） |
| F13 | Alpaca 示例实际 id 第 31 个是 932（`' you'`），注释写的 930 是 `'alize'` | `tok.encode(text)`；`tok.id_to_bytes[930/932]` | `932 b' you'`，`930 b'alize'`，共 34 token | 是（纯 Python 分词器） |
| F14 | `3+4=` 样本：prompt 22 token，labels = 21×(−100) + [55,999]；第一个监督位置 = len(prompt)−1 | `_create_sample('3+4=','7')` | `labels[:23]=[-100]*21+[55,999]`；`first supervised t = 21` | 是 |
| F15 | SFT 数据：1092 条、81 条算术、最长 138、监督位置 8.455%、500 iter = 14.65 epoch | `f02_sft_data.py` | 同左 | 是 |
| F16 | causal mask ⇒ 尾部 padding 不影响前面位置的 logits | item 5 分别补到 256 / 137，比较前 137 个位置 | `max abs diff = 1.62e-05`（float32 舍入），labels 前 137 位相同，137 之后全 −100 | 否 |
| F17 | SFT 前后 loss：response 4.0415 → 0.0276；prompt token 3.3968 → 6.1821（GRPO 6.4977） | `f03_sft_checks.py`、`f03c_prompt_vs_resp.py` | 同左 | 否 |
| F18 | 数字 token：'2'..'9' = 50..57（单 token）；'10' = 743（单 token）；'11'..'18' = ['1', d] 两个 token | `{s: tok.encode(str(s)) for s in range(2,19)}` | `{2:[50],…,9:[57],10:[743],11:[49,49],…,18:[49,56]}` | 是 |
| F19 | 81 个 GRPO prompt 全是 22 个 token；`decode(encode(p))==p` | `{len(tok.encode(p)) for p in P}` | `{22}`、`True` | 是 |
| F20 | 奖励 = “最后一个整数”精确匹配 | `calculate_reward(7, s)` | `'7'`1、`' 7'`1、`'The answer is 7'`1、`'3+4=7'`1、`'7 8'`0、`'8 7'`1、`'07'`1、`'7.0'`0、`'-7'`0、`'seven'`0、`''`0、`'7<\|endoftext\|>'`1 | 是 |
| F21 | 裸 `except:` 吞掉编程错误：`re` 没导入时所有奖励都是 0.0 | 在不含 `re` 的命名空间里 exec 该函数 | `reward(7,"7") = 0.0`（NameError 被吞）；导入后 `1.0` | 是 |
| F22 | advantage：组内和为 0；全同 → 全 0；G=1 → 0；除 std 不加 eps → NaN；torch 默认无偏 std | NumPy | `[1,0×7] → A=[0.875,−0.125×7]`；std 归一化 `[2.6458,−0.378…]`；`[1]*8 → /std = nan`；`/(std(ddof=1)+1e-4) → 0` | 是 |
| F23 | 组零方差概率 = p^G+(1−p)^G | NumPy | p=0.44：G=1 1.000 / 2 0.507 / 4 0.136 / 8 0.011 / 16 0.000；p=0.95：G=8 0.663 | 是 |
| F24 | 期望梯度：GRPO 组均值 = 真梯度×(G−1)/G；RLOO = 真梯度；常数 baseline 期望为 0 | 3 个动作的 softmax 策略，穷举所有 G 元组 | G=1 → 0；G=2 → 0.5；G=3 → 0.66667；G=4 → 0.75；RLOO 全为 1.0；`E[b·score]=[0,0,0]` | 是 |
| F25 | PPO-clip 导数：A>0 且 ratio>1+ε、A<0 且 ratio<1−ε 时梯度为 0；反方向不截断 | 数值微分 | A=+1：ratio 0.5→+1、1.5→0；A=−1：ratio 0.5→0、1.5→−1 | 是 |
| F26 | θ=θ_old 时 ∇ratio = ∇log π = onehot(a) − π | softmax 玩具，有限差分 | 三者一致 `[-0.1341,-0.1248,0.8244,-0.501,-0.0645]` | 是 |
| F27 | ratio=1（eval 模式）时，`grpo_loss` 的梯度与 −Σ A·log π / N 的梯度相同 | `f10b_grad_cos.py`（真 SFT 模型，64 条 rollout） | `‖g‖=8.403800 vs 8.403801, cos(float64)=1.0000000000, max abs diff=8.2e-08` | 否（NumPy 类比见 F26） |
| F28 | ratio=1 时 loss 数值 = −Σᵢ Aᵢ·len(oᵢ) / N | 真 SFT 模型 256 条 rollout | `0.007324` vs `0.007324` | 否（NumPy 类比：是） |
| F29 | loss≈0 ≠ 梯度≈0 | 2 个样本 A=[+.5,−.5]，ratio=1 | `loss=0, dloss/dθ=[-0.25,0.25,0]`；一步 SGD 后 E[r] 0.5065→0.6045 | 是 |
| F30 | dropout 让“权重相同”的第一次更新 ratio≠1 | 真 SFT 模型，同一批 rollout（283 个 response token），各前向 5 次 | train：mean 0.9822 std 0.7274 min 0.0036 max 12.28，**53.9%** 在 [0.8,1.2] 外（单次前向复测：54.1%）；eval：全部落在 [0.99999, 1.0]——偏离 1 的那一点正是 `+1e-8` 造成的（F32） | 否 |
| F31 | dropout 改变更新方向 | 3 次 dropout 抽样 vs 干净梯度 | 余弦 0.6457 / 0.7283 / 0.6393；两次抽样之间 0.5208 | 否 |
| F32 | 概率空间 ratio 的偏差：p_new=p_old 时 ratio = p/(p+1e-8) | NumPy | p=1e-4 → 0.9999；1e-6 → 0.9901；1e-7 → 0.9091；1e-8 → 0.5 | 是 |
| F33 | GRPO batch 里没有 EOS；mask 只覆盖数字 token | `get_batch([p,p],['7','12'])` | row0 `[…,10,55,0]` mask `[0…0,1,0]`；row1 `[…,10,49,50]` mask `[…,1,1]`；`999 in ids: False` | 是（纯 Python 复刻） |
| F34 | `cycle(DataLoader(shuffle=True))` 第二轮起重放**同样顺序**的同样 batch；GRPO 每轮 3 个 batch [32,32,17] | 81 个下标的 Dataset | `pass2 == pass1: True`；重新迭代 DataLoader：`False` | NumPy 类比：是 |
| F35 | `cycle` 会保存它吐出的每个 batch | 1000 个预训练 batch 前后的 RSS | `cycle(dl): +228.2 MiB` vs 普通循环 `+68.8 MiB`；每 batch x+y = 128 KiB ⇒ 20000 iter 约 2.44 GiB | 否 |
| F36 | SFT/GRPO 在 81 题上的 greedy、采样奖励、零方差组 | `f09_grpo_probe.py` | 见 §2.6.4(b) 表 | 否 |
| F37 | GRPO 让 12 题 P(正确)<0.01；pass@k 曲线交叉 | 读 `research/data/ch03_answer_dists.json` | SFT pass@8 0.915 > GRPO 0.834；pass@1 0.450 < 0.821 | **是**（JSON + NumPy） |
| F38 | 日志里 iter 0 的准确率是 2 次更新后的 | 对比 `generate(sft, temperature=0)` 的准确率 | 起点 54.3%，脚本打印 55.6% | 否 |
| F39 | 表格 GRPO（从真 SFT 分布初始化）：G=1 不学；G=8 约 13–16 步到 95% | `f08_tabular_grpo.py` | 见 §6 的 W10；G=1：150 步后 greedy 0.556 不变 | **是** |
| F40 | 预训练 checkpoint 在训练语料上 loss 0.424（ppl 1.53） | 64 个随机窗口，eval | `loss 0.4240 (ppl 1.53)` | 否 |
| F41 | 自写 LayerNorm/GELU 未被使用；GELU tanh 近似 vs erf 最大差 4.74e-4 | `type(m.blocks[0].norm1)`，`linspace(-6,6)` 比较 | `torch.nn.modules.normalization.LayerNorm`；`4.74e-04 at x=2.697`；与 `approximate='tanh'` 差 `2.38e-07` | 公式：是 |
| F42 | storybot/webbot 的 `generate`：截断 `ids` 管不住 KV cache | 随机初始化 storybot GPT，max_context_len=8，生成 20 个 | `k_cache length 22, cache_offset 22`（> 8） | 否 |
| F43 | 训练模式激活显存：B=32,C=256 约 2.22 GiB（fp32） | `saved_tensors_hooks` 统计反向需要保存的张量 | `2270.6 MiB`；B=256,C=24：`1125.9 MiB`；B=32,C=137：`1019.6 MiB`；权重+梯度+Adam：`170 MiB` | 否 |
| F44 | 温度必须恰为 0 才走 greedy；极小的 T 会溢出 | `softmax(lg/1e-45)` | `[[nan,nan,nan]]`，`multinomial → RuntimeError: probability tensor contains either inf, nan or element < 0` | 否 |
| F45 | 下载的 SFT 模型在 81 题上的答案分布 argmax 正确 45/81；GRPO 67/81 | `f07_answer_dists.py` | 同左（greedy 解码 44/81 与之略有差别：两位数答案是逐 token 取 argmax） | 是（JSON） |
| F46 | 同一批 rollout 上按脚本做 2 次更新（train 模式）后，用 eval 模式测：clipfrac 12.7% → 24.4%，KL_k3 1.08e-2 → 3.56e-2，entropy 0.839 → 0.821 nats，裁剪前 grad_norm 2.60 / 2.23 | `f10_grpo_ratio.py (4)` | 同左；ratio 范围 [0.42,1.80] → [0.22,2.70] | 否 |
| F48 | 日志里 GRPO loss 为正，主要来自 dropout 与 clip 的偏差，而非学习信号 | `f16_loss_bias.py`：同一批 256 条 rollout，eval 下算 1 次，train 下各抽 20 次 | clean `+0.007324`（= −Σ A·len/N；错误回答平均 1.153 token vs 正确 1.140）；dropout + ε=0.2：均值 `+0.0991`（sd 0.026，20 次全为正）；dropout、不 clip：`+0.0582`（sd 0.031） | 否 |
| F49 | “为什么偏偏是 15 被压死”：SFT 分布下答案 15 的“错用/正用”采样比最高 | 对每个答案 v：正确质量 C=8·Σ_{sum=v}P(v)，错误质量 W=8·Σ_{sum≠v}P(v)（读 `ch03_answer_dists.json`） | v=15：C=2.24、W=5.65、W/C=**2.52**（全表最高，C 也最小）；60 iter 后 P(15\|sum=15)：nodrop 0.002（其余答案全 ≥0.958）、base 0.015；全体 prompt 上 P(答 15)：SFT 0.0122 → nodrop 0.0002 | **是** |
| F47 | 零梯度时 AdamW 仍做 weight decay | `p=ones(3); (p*0).sum().backward(); AdamW([p],lr=7e-6).step()` | `p = 0.99999994`（= 1 − 7e-6×0.01） | 否 |

---

## 4. GPU / 数据需求

### 4.1 数据与文件

| 文件 | 大小 | 来源 | 用途 |
|---|---|---|---|
| `codebot/tiny_codes.txt` | 6,743,534 B | 仓库自带 | 原文语料（第 1 章训练 BPE） |
| `codebot/tiny_codes.bin` | 5,354,796 B = 2,677,398 × uint16 | 仓库自带（ch01/09 生成） | 预训练数据 |
| `codebot/merge_rules.pkl` | 7,421 B（743 条 merge） | 仓库自带 | 分词器 |
| `codebot/tiny_codes_sft.json` | 129,717 B（1092 条） | 仓库自带 | SFT |
| `codebot/model_pretrain.pt` | 44,516,353 B | HF `koki0702/zero-llm-data/codebot/`（download.sh 有效行） | 02、03_sft 的输入 |
| `codebot/model_sft.pt` | 44,515,938 B | 同上（download.sh 注释行） | 04、09 的输入 |
| `codebot/model_grpo.pt` | 44,516,021 B | 同上（注释行） | 04 的可选输入 |

### 4.2 算力估算

* **预训练**：D = 20000 × 32 × 256 = 163,840,000 token-position（语料的 61.2 倍）；N = 11.12M；**6ND = 1.093e16 FLOPs**，加上注意力（6·L·C·E/token）约 **1.14e16**；每 iter ≈ 5.7e11 FLOPs。
  * CPU（本机，争用下）4.14–6.39 s/iter → 23–35 h（≈ 89–138 GFLOPS 有效）。
  * GPU 估算（未实测，需跑前 200 iter 校准）：默认 fp32 且 PyTorch 默认不开 TF32 matmul，注意力是显式 (32,6,256,256) 张量（每个约 50 MB，访存密集）。RTX 3060 Ti 8GB：约 60–120 ms/iter → **20–40 min**；A10G/L4：约 30–60 ms/iter → 10–20 min；A100（fp32 非 TF32 只有 19.5 TFLOPS）：约 30–50 ms/iter；开 `torch.set_float32_matmul_precision('high')`（TF32）或 bf16 autocast（第 6 章内容）可再快 2–3 倍。
  * 显存：激活 2.22 GiB + 权重/梯度/AdamW 170 MiB + CUDA context ≈ **3 GB**，8 GB 卡足够。
  * 主机内存：`itertools.cycle` 会把 20000 个 batch 都留在内存里（≈ 2.4 GiB），Modal 上请把 `memory` 设到 ≥ 8 GB。
* **SFT**：500 iter × 32 × 256（其中 91.5% 是 padding）≈ 6ND = 2.7e14 FLOPs；GPU 上 1–2 分钟；CPU 上 ctx 256 约 60 min、ctx 137 约 21–36 min（实测 4.27 s/iter）。
* **GRPO**：瓶颈在**采样**——每 iter 256 次 `generate()`，batch=1、无 KV cache、每个回答约 2.5 次前向（1–2 个数字 + EOS 那一步）、每步 `.item()` 同步一次 → 约 640 次小前向；评估每 10 iter 再加 81 次 greedy 生成。
  * CPU 实测：轻争用 ~19 s/iter（其中 32×8 采样实测 10.8 s，其余约 8 s 为推算的 2 次更新）→ 500 iter ≈ **2.6 h**；重争用 51 s/iter。
  * GPU 估算：每次小前向受 kernel launch 限制约 2–3 ms → 采样 ~1.5–2.5 s/iter，更新 (256×24 token) 毫秒级 → 500 iter ≈ **15–25 min**；GPU 型号几乎无关（T4 足够）。
  * 批量采样（我的 `grpo_fast.py` 思路）可以把 256 次逐条生成变成 ~3 次批量前向，GPU 上整轮 GRPO 可降到几分钟。
  * 显存：两份模型 + Adam + (256×24) 激活约 1.1 GiB → ≈ 2 GB。
* **整章 GPU 预算**：本地 3060 Ti 全部能跑（需要装 CUDA 版 torch，例如 `uv pip install torch --index-url https://download.pytorch.org/whl/cu126`，当前 uv 环境是 CPU 版）；云上 1 张 A10G/L4 约 **1 GPU 小时**（预训练 ≤40 min + SFT 2 min + GRPO ≤25 min + 余量）。

### 4.3 Modal / RunPod 运行要点

* 镜像：`torch==2.9.1`（CUDA 版）、`numpy<2`、`matplotlib`、`tqdm`、`regex`；把仓库挂进去（脚本自己 `os.chdir` 到仓库根目录，所以保持目录结构即可）；输出 `codebot/model_*.pt`、`loss_*.png` 写在仓库根目录，用 Volume 带出来。
* `gpu="A10G"`（或 `"L4"`/`"T4"`），`timeout ≥ 2h`，`memory ≥ 8192`。
* 顺序：`01_pretrain.py` →（可选）`02_generate.py` → `03_sft.py` → `09_grpo.py` → `04_chat.py`（交互式，云上需改成读文件）。
* 建议在 GPU 上补做的消融（本机 CPU 只做了 60 iter、1–2 个 seed）：`09_grpo` 原版（逐条采样）vs **更新时 `model.eval()`（关 dropout）** vs **除 std** vs **G∈{2,4,8,16}**，各 500 iter、≥3 个 seed，记录 §2.6.3 的指标。CPU 上已经看到的现象（关 dropout：60 iter 95.1% vs 70.4%）需要在完整 500 iter 上确认，尤其要看关 dropout 后是否仍会出现“被压死的题”（§5 第 8 条）。
* RunPod/Lambda：开一台 RTX 4090/A10/A100 按小时计费的机器，`git clone` + `uv sync` + 装 CUDA torch，1 小时内跑完整章。

### 4.4 CPU 端到端（笔记本可跑，约 20 分钟；全部实测）

把整章流水线缩到 0.9M 参数，代码路径不变（补丁副本见附录 A），在本机 CPU 上一口气跑完：

| 阶段 | 配置 | 实测耗时 | 结果 |
|---|---|---|---|
| 预训练 `p01_pretrain_tiny.py` | E=128、H=4、L=4、ff=512、C=128、batch 32、lr 1e-3、1500 iter | 533 s（0.356 s/iter） | loss 6.93 → 2.52；能写出“像 Python”的代码（§2.1） |
| SFT `p03_sft_tiny.py` | 原超参（lr 3e-4、batch 32、500 iter），`context_len=128`（小模型只有 128 个位置；1092 条里只有 2 条被截断，长度 133/138，丢了 EOS 标签） | 168 s（0.336 s/iter） | loss 4.06 → 1.57（最低 0.48）；`Hello` → `Hello. What would you like to work on?`，`print hello` → `print('hello')`，`make variable x` → `x = 'hello'`，`Who are you?` → `I'm assistant. What do you need?`；**加法 greedy 只有 12.3%**（`3+4=` → `11`） |
| GRPO `grpo_fast.py --init model_sft_tiny.pt` | G=8、batch 32、100 iter（4 种设置见下表） | 280–360 s/次 | 见下表 |

GRPO 阶段的 greedy 加法准确率（%，起点都是 12.3%）：

| 设置（tiny 模型，100 iter） | it0 | it20 | it40 | it60 | it80 | it99 | 末 10 iter：reward / 零方差 / entropy |
|---|---|---|---|---|---|---|---|
| dropout 开（原脚本行为），lr 7e-6（原值） | 12.3 | 11.1 | 9.9 | 13.6 | 14.8 | 13.6 | 0.148 / 0.758 / 0.571 |
| dropout 开，lr 1e-4 | 11.1 | 16.0 | 21.0 | 14.8 | 23.5 | 19.8 | 0.245 / 0.876 / 0.384 |
| **dropout 关**，lr 7e-6 | 12.3 | 12.3 | 19.8 | 24.7 | 32.1 | 35.8 | 0.236 / 0.237 / 1.457 |
| **dropout 关，lr 1e-4** | 13.6 | 33.3 | 43.2 | 55.6 | 63.0 | **72.8** | 0.573 / 0.379 / 0.656 |

这一节给教程的结论：
1. **整条流水线在笔记本 CPU 上约 18 分钟跑完**（预训练 9 分钟 + SFT 3 分钟 + GRPO 5.4 分钟），而且每一步都有肉眼可见的效果：loss 6.93→2.52 并能写出代码样子的文本；SFT 后学会模板式回答；GRPO（关 dropout、lr 1e-4）把加法从 12.3% 提到 72.8%。这就是教程可用的“CPU 端到端例子”。
2. **起点很弱时，零方差组很多**：SFT 准确率 12% 时，组大小 8 的全错概率约 0.9⁸≈0.43，与实测首轮 0.44 吻合；但剩下一半多的组仍有信号，只要更新质量够好就学得动。
3. **dropout 的影响在小模型上更极端**：开 dropout、lr 7e-6 时熵从 1.50 掉到 0.57，准确率却没有任何提升（策略“变尖”但没变对）；关 dropout 同样的 lr，熵基本保持（1.66→1.46），准确率升到 35.8%。小模型需要更大的 lr（7e-6 是为 11M 模型挑的）。
4. 以上每组都是单 seed；指标逐 iter 记录在 `research/data/ch03_grpo_metrics_tiny_*.jsonl`。

---

## 5. 易错点 / 坑 / 潜在 bug（按影响排序；“已验证”都有对应的 F 编号）

1. **GRPO 更新时 dropout 开着（已验证 F30/F31）**：`model` 从 `load_from` 出来默认 `train()`，每次评估后又被 `model.train()`；`old_model` 一直是 `eval()`。于是 ratio 里混入了 dropout 噪声：第一次更新本该 ratio≡1，实测 53.9% 的 token 超出 clip 范围，梯度方向余弦只有 0.64–0.73，loss 被系统性抬高（F48）。**训练曲线上的代价很大**：批量采样版、同 seed、60 iter，唯一差别是更新前 `model.eval()`，greedy 准确率 95.1% vs 70.4%（§2.6.4(c)）。许多 RLHF/GRPO 实现会在 RL 阶段关闭 dropout（例如 TRL 提供 `disable_dropout` 选项）。修法：更新前 `model.eval()`（不影响反向传播），或构造模型时 `dropout_rate=0`。
2. **评估日志错位（已验证 F38）**：`if i % eval_interval == 0` 在本轮更新之后执行，所以 iter 0 记录的 55.6% 不是 SFT 基线（54.3%）。画图的横轴也因此整体偏了一轮。
3. **`itertools.cycle(DataLoader)` 的两个副作用（已验证 F34/F35）**：(a) `shuffle=True` 只在第一轮生效，之后按同样顺序重放同样的 batch——SFT（35 个 batch，14.65 轮）与 GRPO（固定的 3 个 prompt 分组）都受影响；(b) cycle 保存每个吐出的 batch，预训练 20000 iter 约多占 2.4 GiB 主机内存。替代写法：`def infinite(dl): while True: yield from dl`。
4. **EOS 不在 GRPO 目标里（已验证 F33）**：“何时停”从不被强化学习奖励或惩罚，只靠 SFT 学到的习惯；若模型在答案后多说一个数字（`'7 8'`），奖励按最后一个整数算就会错，却无法通过 EOS 的梯度纠正。
5. **长度没有上限检查（机制已验证 F10；实际未触发）**：`generate_group` 用默认 `max_new_tokens=1000`，`get_batch` 不截断；prompt 22 token，只要某个回答 > 234 token，`compute_probs` 就会因位置编码越界报 `IndexError`。SFT/GRPO 模型的回答最长 8 token，未触发；用预训练模型当策略采 6 次最长 107 token，也未触发——但这是潜在崩溃点。
6. **loss 的数值不是进度指标（已验证 F28/F29/F48）**：ratio=1 时它等于 −Σ Aᵢ·len(oᵢ)/N，组内 advantage 和为 0，只剩“长度与 advantage 的相关性”（实测这批里错误回答平均 1.153 token、正确回答 1.140 token，于是 clean loss 只有 +0.0073）。日志里看到的 +0.05~+0.23 主要不是这个：同一批数据上抽 20 次 dropout，loss 平均 **+0.058**（不 clip）/ **+0.099**（ε=0.2）——dropout 让“自信且正确”的 token ratio 偏到 1 以下、让“不太可能的错误” token ratio 偏到 1 以上，两者都把 loss 往上推；clip 的 `min` 是悲观下界，又把偏差放大。看 reward/mean、greedy acc、零方差比例、entropy 才知道是否在学。
7. **零方差组（已验证 F22/F23/F36）**：组内全对或全错 → 零梯度。训练越成功，零方差组越多（SFT 18.5% → GRPO 96.3%），有效 batch 越小；全错的组（13 组）则永远学不到。DAPO 的 dynamic sampling 就是把这类组过滤后补采。
8. **GRPO 把一部分题“压死”（已验证 F37/F45/F49）**：下载的 GRPO 模型把 12 道题的 P(正确) 从 ≥0.056 压到 <0.01，全错组没有梯度，回不来；pass@8 从 0.915 降到 0.834。这**不是 dropout 造成的**：关 dropout 的 60 iter 模型同样有 4 道题 P(正确)<0.01，而且恰好是答案为 15 的全部 4 道（6+9, 7+8, 8+7, 9+6）。在 SFT 分布下，“15”被当作**错误答案**采到的期望次数是被当作正确答案采到的 2.52 倍（全部答案中最高），正确样本的期望数也最少（每 epoch 2.24 个）；于是共享的“15”这个输出被负 advantage 整体压低（全体 prompt 上 P(答 15) 从 0.0122 降到 0.0002）。这与“参数共享下的答案级干扰”一致（不是严格证明）。表格玩具没有参数共享，所以最后 100% 正确——两者对照正好说明问题。
9. **长度加权（设计选择，F28）**：Σ_tokens/(B·G) 让两位数答案（2 token）的权重是一位数答案的 2 倍；Dr. GRPO 有意如此（去掉 1/|oᵢ| 的长度偏置），但教程应说明与 DeepSeekMath 原版不同。
10. **std 归一化被去掉（设计选择，F22）**：与 DeepSeekMath 不同；若读者自己加 std，必须加 eps，否则零方差组 NaN；torch 的 `std()` 默认无偏（ddof=1），与 NumPy 默认（ddof=0）不同。
11. **概率空间 ratio + 1e-8（已验证 F32）**：p 很小时 ratio 系统性偏小（p=1e-7 时 0.909）；更稳的写法是 `exp(logp − logp_old)`，`logp` 用 `log_softmax` + `gather`。
12. **裸 `except:`（已验证 F21）**：任何编程错误都被当成“答错”，训练静默地什么都不学。至少改成 `except ValueError:`。
13. **奖励可被“最后一个数”规则利用（F20）**：`'8 7'` 得 1，`'7.0'` 得 0，`'1 2'` ≠ 12。小模型在本任务里没出现这种 hack，但讲 reward design 时是绝佳例子。
14. **评估用的是“prompt+回答”的全文（已验证无害）**：`generate()` 返回包含 prompt，`calculate_reward` 取最后一个整数；回答没有数字时会取到 prompt 里的 j，而 j ≠ i+j（i≥1），所以本任务恰好无害（全文/仅回答两种算法的准确率一致）；换成 `0+5=` 这样的题就会误判。
15. **解码→再编码（实测 0/648 不一致，F36）**：`response = full_text[len(prompt):]` 后又 `encode(response)`，如果模型采样出非规范切分（如 '1','0' 而不是 '10'），训练的就不是实际采样的 token 序列；本任务里没发生，但大模型/中文里很常见。
16. **`generate()` 的副作用（已验证 F11）**：调用后模型停在 eval；在训练循环里对 policy 调用 `generate` 会悄悄关掉 dropout。09 里只对 `old_model` 调用、评估后又 `model.train()`，所以恰好没问题。
17. **SFT padding 浪费（F15/F16）**：91.5% 的计算花在 padding 上；按 batch 内最大长度动态 padding 或把 `context_len` 设成 137 结果完全等价（F16）。
18. **SFT 的 loss 是 token 平均（F06）**：长回答（多行代码，最长 112 token）主导梯度；与“逐样本平均”数值不同。
19. **掩码后的遗忘（F17）**：SFT 后 prompt token 上的 loss 从 3.40 升到 6.18，模型几乎不会再“续写用户输入”；这是 loss 掩码的直接后果，不是 bug，但值得在教程里点明。
20. **SFT 没能背下算术（F36/F45）**：81 道题都在训练集里、训练了 14.65 轮，平均 P(正确) 仍只有 0.45；这些 prompt 只差两个 token，像 9×9 查表，而每题平均只被看到 ~15 次。
21. **预训练没有验证集，且明显过拟合语料（F40）**：训练语料上 ppl 1.53；第 6 章才引入验证集与 LR schedule、梯度裁剪。
22. **03_alpaca 注释里的 id 过时（F13）**：930 → 932。
23. **`model.py` 里的 LayerNorm/GELU 自写类未被使用（F41）**：读者若以为模型用的是 tanh-GELU 会产生 ~5e-4 的误差预期偏差。
24. **温度边界（F44）**：只有 `temperature == 0` 走 argmax；极小的正温度会溢出成 NaN。
25. **变体差异**：codebot 的 `generate` 每步重算全部上下文；storybot/webbot 用 KV cache（`model(next_id, use_cache=True)`），并在输出前过滤 EOS（`generated_ids[generated_ids != end_token_id]`）。但它们的 `ids` 截断对 KV cache 没有作用：cache 会长过 `max_context_len`（已验证 F42：上限 8，cache 长到 22），RoPE offset 被夹在最后一个位置——属于第 5 章的问题，这里只做标注。
26. **没有任何随机种子**：所有脚本都不可复现；教程里给出 `torch.manual_seed`。
27. **`plt.savefig("loss_grpo.png")` 画的是 accuracy**；所有 png 都写到仓库根目录。

---

## 6. 互动组件点子（每个都标出学习者“动什么、看什么”，以及能否纯浏览器）

| # | 组件 | 学习者操作 | 观察到什么 | 数据/实现 |
|---|---|---|---|---|
| W1 | **滑动窗口与错位一位** | 拖动起点 i、改 context 长度 | x 与 y 两行 token 对齐、y 恰好左移一位；`len = N − C` | 取 `tiny_codes.bin` 前 5 万个 id + 解码表（纯 JS） |
| W2 | **初始 loss 计算器** | 改 V、改 logit 标准差 σ | ln V 与 ln V + σ²/2，再和 Monte-Carlo 结果比；“看到 6.9 说明初始化正常” | NumPy（Pyodide） |
| W3 | **温度 / Top-k / Top-p 采样台** | 三个滑块 + “采样 100 次” | 柱状图被拉尖/拉平、熵数值、被截掉的长尾、重归一化后的概率 | 预先用真模型算好 “def” 后的下一 token 分布（JSON），其余 NumPy |
| W4 | **自回归逐步生成** | 每步从 top-10 里点一个 token（或按温度自动采样） | 文本如何一步步长出来；EOS 出现即停；超过 256 时最旧 token 被丢 | 预先缓存一棵浅层分布树 |
| W5 | **Alpaca 模板 + 掩码可视化** | 输入 instruction / response | 实时 BPE 分词；prompt token 灰色（label −100）、response 绿色、padding 浅色；箭头显示“输入第 t 个 → 预测第 t+1 个” | 浏览器内 BPE：`merge_rules.pkl` 转 JSON（743 条），正则用 JS `\p{L}`/`\p{N}` 或 Pyodide `regex` |
| W6 | **token 平均 vs 样本平均** | 改三条样本的回答长度 | 两种平均的 loss 与权重分配不同 | NumPy |
| W7 | **GRPO 组模拟器** | 设成功率 p、组大小 G、是否除 std | 采样 8 个奖励 → advantage 条形图；零方差组闪烁提示；P(零方差)=p^G+(1−p)^G 曲线 | NumPy |
| W8 | **PPO-clip 曲面** | 拖 ratio、切换 A 的正负、调 ε | 目标函数折线与“梯度为 0”的阴影区；“只截断继续往好方向走的那一边” | NumPy |
| W9 | **“loss≈0 但在学习”演示** | 点“更新一步” | loss 一直在 0 附近，P(正确) 却上升；梯度箭头非零 | NumPy（F29） |
| W10 | **端到端：浏览器里的 GRPO**（推荐作为全章收束） | G、ε、lr、每批更新次数、是否除 std、是否给 ratio 加乘性噪声（模拟 dropout） | 9×9 热力图显示每道题的 P(正确) 随迭代变化；曲线：greedy 准确率、采样奖励、零方差比例、熵；G=1 时完全不动 | 初始化用 `research/data/ch03_answer_dists.json` 的真 SFT 分布；`f08_tabular_grpo.py` 已验证：G=8 约 13–16 步到 95%，G=2 约 41–53 步，G=1 不学，除 std 约 7–10 步 |
| W11 | **真模型对照面板** | 切换 SFT / GRPO / 我的 SFT | 81 题热力图、被压死的 12 题高亮、pass@k 曲线在 k≈3–8 处交叉 | JSON + NumPy（F37） |
| W12 | **dropout 让 ratio 失真** | 切换 train/eval | ratio 直方图（eval 时一根柱子在 1.0；train 时散到 0.004–12），clip 区间 [0.8,1.2] 叠加 | 预先导出 F30 的 ratio 样本 |
| W13 | **训练仪表盘解读** | 悬停每条曲线 | reward/mean、zero_var、pg_loss、grad_norm、clipfrac、KL、entropy 的真实曲线与一句话解释（对应 verl 的 `critic/rewards/mean`、`actor/pg_loss`、`actor/grad_norm`、`actor/entropy`） | 用 `out/grpo_fast_*.jsonl` |
| W14 | **期望梯度的 (G−1)/G** | 选 G | 穷举得到的期望梯度与真梯度的比值条；RLOO 恒为 1 | NumPy（F24） |
| W15 | **“为什么偏偏是 15”：答案级干扰** | 点选某个答案 v | 柱状图：v 被当正确答案 / 错误答案采到的期望次数（SFT 分布）；旁边是训练后 P(v\|sum=v)；15 的“错/对”比最高、最终被压到 0.002 | JSON + NumPy（F49） |
| W16 | **dropout 开/关的训练曲线对照** | 切换 base / nodrop | 同 seed 的 greedy 准确率、reward、零方差比例、entropy、grad_norm 曲线并排（95.1% vs 70.4%） | `ch03_grpo_metrics_{base,nodrop}.jsonl` |

---

## 7. CS336 对照（Stanford CS336 Language Modeling from Scratch，凭记忆，以 2025 春季版为准）

| DLFS6 第 3 章 | CS336 对应 | 异同要点 |
|---|---|---|
| `TokenDataset` + `DataLoader(shuffle)` + `cycle` | A1 `get_batch(x, batch_size, context_length, device)`：从 `np.memmap` 的 token 数组里随机取起点 | CS336 是有放回随机起点；DLFS6 第 6 章 `get_batch` 改成了与 CS336 同款 |
| `F.cross_entropy(logits.view(-1,V), y.view(-1))`；初始 loss ≈ ln V | A1 自己实现数值稳定的 `cross_entropy`（减 max、logsumexp），并以初始 loss≈ln V 作为 sanity check | CS336 不允许直接用 `F.cross_entropy` |
| AdamW（torch 自带）、无 schedule、无裁剪 | A1 自己实现 `AdamW`、cosine + warmup 调度、`gradient_clipping` | DLFS6 第 6 章补齐这三样 |
| `model.save/load_from` | A1 `save_checkpoint/load_checkpoint`（含 optimizer 与 iteration） | DLFS6 不存 optimizer 状态 |
| `generate()`：温度、截断上下文、遇 EOS 停 | A1 “decoding”：温度 + **top-p**、最大 token 数、遇 `<\|endoftext\|>` 停 | DLFS6 的 3.2.3 讲了 top-k/top-p 但仓库没实现 |
| GPT-2 式模型（LayerNorm、GELU、learned PE、dropout、权重共享） | A1 Llama 式（RMSNorm、SwiGLU、RoPE、无 dropout、无 bias） | 对应 DLFS6 第 5 章 StoryBot；CS336 无 dropout，因此不会遇到本章的 ratio/dropout 问题 |
| 6ND 估算、激活显存 | A2 benchmarking / memory profiling；A3 scaling（IsoFLOPs、6ND） | — |
| 显式 (B,H,C,C) attention | A2 用 Triton 实现 FlashAttention-2 | DLFS6 第 8 章 |
| `SFTDataset._create_sample`（−100 掩码、错位） | A5 `tokenize_prompt_and_output` 产出 `input_ids / labels / response_mask` | CS336 用 `response_mask` 而非 −100 |
| `F.cross_entropy(ignore_index=-100)`（token 平均） | A5 `masked_normalize`（除以常数）/ `sft_microbatch_train_step`（梯度累积） | 归一化方式不同，正是 F06 的差别 |
| `compute_probs`（softmax → gather，概率空间） | A5 `get_response_log_probs`（`log_softmax` → gather，可选返回 `compute_entropy`） | CS336 在 log 空间，且顺便算 entropy |
| `generate_group` 的 `r − mean` | A5 `compute_group_normalized_rewards(..., normalize_by_std, advantage_eps)` | 09 相当于 `normalize_by_std=False` |
| `grpo_loss` | A5 `compute_grpo_clip_loss`（ratio = exp(logp − old_logp)）+ `masked_mean` 或 `masked_normalize` | 09 的“token 求和 / 序列数”接近 `masked_normalize`（Dr. GRPO 式）；CS336 让学生做长度归一化与 std 归一化的消融 |
| `n_update_per_generation=2` | A5 `epochs_per_rollout_batch`（>1 即 off-policy，需要 clip） | 一致 |
| 无 KL、无 π_ref | A5 的 GRPO 同样不加 KL（沿用近期做法） | 一致 |
| `clip_grad_norm_(1.0)` | A5 同样 1.0 | 一致 |
| `calculate_reward`（最后一个整数） | A5 `r1_zero_reward_fn`（格式奖励 + 答案奖励，`</think> <answer>` 标签） | CS336 区分 format reward 与 answer reward |
| 81 题 greedy 评估 | A5 用 vLLM 在 MATH 验证集上评估、`log_generations` 记录熵/长度/奖励 | CS336 要求记录的指标正是 §2.6.3 表里那些 |
| 逐条 batch=1 采样 | A5 用 vLLM 批量 rollout | 本章 GRPO 慢的根源 |
| GRPO 的 (G−1)/G、零方差组 | A5 讨论 baseline、std 归一化的偏差（Dr. GRPO）与组大小 | 可直接引用 F22–F24 |
| SFT + 对话 | A5 SFT（Qwen2.5-Math-1.5B 在推理轨迹上）、expert iteration | DLFS6 规模小几个数量级，但流程同构 |
| DPO | 2024 版 A5 / 2025 版补充作业（指令微调 + DPO 安全对齐） | 对应 DLFS6 第 6 章 6.7–6.8 |

---

## 8. 用户过往困惑点 → 本章应加强的解释

在用户过去的会话记录（`research/private/wsl_msgs.jsonl`）里，与本章直接相关的提问有：

1. **“policy loss ≈ −mean(ratio·advantage)……期望永远≈0，不管学没学——没懂”**。本章能给出完全可复现的回答：(a) ratio=1 时 loss 的数值只剩 −Σ Aᵢ|oᵢ|/N（F28 实测 0.007324），组内 advantage 和为 0，所以它≈0；(b) 但梯度 = Σ Aᵢ∇log π(oᵢ)/N 不为 0（F27：范数 8.40，余弦 1.0）；(c) 下一轮重新采样、重新减均值，loss 又回到 0 附近；(d) 09_grpo 实际日志里 +0.05~+0.23 的正值，也不是“学得不好”，而是 dropout 噪声经 clip 的悲观 `min` 放大后的系统偏差（F48：clean +0.0073，dropout 下 +0.058，再加 clip +0.099）。配合 W9（NumPy 两样本演示）讲清楚。
2. **“actor/entropy、critic/rewards/mean、actor/grad_norm、actor/pg_loss 是什么”**。用 §2.6.3 的指标表，把 verl 的名字对到 09_grpo.py 的变量，并给出本章的真实数值（reward 0.43–0.57、grad_norm 1.9–5.5 且每步被裁到 1、entropy 0.76–0.84 nats、下载的 GRPO 终点熵 0.066）。顺带说明 GRPO 没有 critic，但 verl 仍把奖励/advantage 统计记在 `critic/` 前缀下。
3. **“reward 饱和后两边都失去方差、零方差告警是什么意思”**。直接用 F23 的 p^G+(1−p)^G 与 F36 的 18.5% → 96.3%，再用 W7/W10 动态展示。
4. **“为什么 2 个 rollout 的 GRPO 没意义、gbs 是多少”**。09 的“全局 batch”是 32 prompt × 8 rollout = 256 条序列；G=2 时期望梯度只剩真梯度的一半（F24），零方差概率 ≥ 0.5（F23），表格玩具里达到 95% 需要 41–53 步，而 G=8 只需 13–16 步（F39）。

---

## 附录 A：补丁副本做了什么（全部放在 scratch，仓库文件未改动）

* `p01_pretrain.py`：`os.chdir` → 仓库绝对路径；`max_iters` 由命令行给（10 / 30）；`torch.manual_seed(0)`；模型与 png 存到 scratch；每 25 iter 打印一行。超参不变。
* `p01_pretrain_tiny.py`：在上面基础上 E=128、H=4、L=4、ff=512、C=128、lr=1e-3、1500 iter。
* `p03_sft.py`：路径、seed、`max_iters`（4 / 300）、`context_len` 由环境变量给（256 / 137）。
* `p04_chat.py`：模型路径由环境变量给、`torch.manual_seed(0)`、stdin 读到 EOF 退出并回显输入。
* `p09_grpo.py`：`sft_model_path` 指向下载的 `model_sft.pt`、`max_iters`（3 / 12）、seed、每 iter 打印一行、输出到 scratch。
* `p03_sft_tiny.py`：预训练权重改为 `model_pretrain_tiny.pt`、`context_len=128`、seed、输出到 scratch；其余超参（lr 3e-4、batch 32、500 iter）不变。
* `grpo_fast.py`：见 §2.6.4(c)；参数 `--iters --seed --dropout-off --std-norm --G --lr（默认 7e-6）--init --tag`。

复现命令模板（Git Bash，仓库根目录）：`export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg CH03_OUT=<scratch>\out; uv run python <scratch>/scripts/p09_grpo.py 12`；`uv run python <scratch>/scripts/grpo_fast.py --iters 60 --tag nodrop --dropout-off`。

## 附录 B：产物路径

scratch 根目录：`C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch03\`

* `scripts/`：`f01_model_data.py`、`f02_sft_data.py`、`f03_sft_checks.py`、`f03c_prompt_vs_resp.py`、`f04_arith.py`、`f05_numpy_facts.py`、`f06_torch_facts.py`、`f07_answer_dists.py`、`f08_tabular_grpo.py`、`f09_grpo_probe.py`、`f10_grpo_ratio.py`、`f10b_grad_cos.py`、`f11_memory_flops.py`、`f12_numpy_more.py`、`f13_eval_mine.py`、`f14_more_checks.py`、`f15_export_widget_data.py`、`f16_loss_bias.py`、`f17_tiny_eval.py`、`grpo_fast.py`、`summarize_fast.py`、`patch.py` 与各补丁副本（`p01_pretrain.py`、`p01_pretrain_tiny.py`、`p03_sft.py`、`p03_sft_tiny.py`、`p04_chat.py`、`p09_grpo.py`）、`queue.sh`–`queue4.sh`
* `out/`：各脚本的 `.txt` 输出、`q1`–`q14` 运行日志、`grpo_fast_*.jsonl` 指标、模型 `model_sft_mine.pt`、`model_pretrain_tiny.pt`、`model_sft_tiny.pt`、`model_grpo_fast_*.pt`
* 持久化副本（不会随 scratch 消失）：`research/data/` 下的 JSON / JSONL 与 `ch03_numpy/` 脚本
* `models/`：下载的 `model_sft.pt`、`model_grpo.pt`
* `web/toc.txt`：官方目录抓取结果
