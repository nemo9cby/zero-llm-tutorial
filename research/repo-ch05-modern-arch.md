# ch05 dossier：现代 LLM 架构（RoPE / SwiGLU / RMSNorm / 改良 GPT / KV Cache / GQA）

> 逆向来源：`C:\Users\chenf\Projects\_refs\dlfs6`（oreilly-japan/deep-learning-from-scratch-6，HEAD `c9b6e2e`，2026-04-15）
> 本 dossier 覆盖（逐行读过）：`ch05/01_rope.py`、`02_swiglu.py`、`03_rmsnorm.py`、`04_improved_gpt.py`、`05_kvcache.py`、`graph.py`、`storybot/model.py`、`webbot/model.py`。
> 为了说明上下游，还读了 `codebot/model.py`、`storybot/utils.py`、`webbot/utils.py`，以及 `ch06/05_pretrain.py`、`ch09/01_pretrain.py` 的超参数部分。
> 写作约定：正文中文，术语保留英文；所有"事实"都附有**实际运行过的**片段与原样输出（§3）。

---

## 0. 元信息与运行环境

- 机器：Intel i7-11700F（8C/16T），48 GB RAM，Windows 11。Python 3.12.9，torch 2.9.1+cpu（默认 8 线程），numpy 1.26.4。全部用 `uv run` 执行，并设置 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg`。
- **测时警告（重要）**：运行期间有其它并行任务把 CPU 占满（`Win32_Processor.LoadPercentage` 多次采样都在 88–100%）。校准用的微基准：一次 1-token 的 `Linear(384,384)` 要 106 µs，一个极小的 `add` 要 11 µs，比空闲机器慢好几倍。所以本文所有 wall time 都偏大、噪声也大。**可信的是相等性、形状、参数量这类确定性结论，以及"加速比随长度增长"的趋势**；绝对秒数不要原样写进教程。
- 验证脚本放在 `<scratch>/ch05/checks/*.py`，其中 `<scratch>` = `C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad`。scratch 可能被清理，所以**全部片段和输出都原样内嵌在 §3**。除特别说明外，片段都以 repo 根目录为 cwd 运行：`uv run python <file>`。
- 下载（只读，没有放进 repo，避免与其它 agent 并发写同一文件）：HF `koki0702/zero-llm-data` 的 `storybot/model_pretrain.pt`（90,835,030 B）和 `storybot/merge_rules.pkl`（103,877 B），存到 `<scratch>/ch05/dl_storybot/`。pickle 先用 `pickletools` 做了静态 opcode 检查（见 §3-J0），`.pt` 走 torch 2.9 默认的 `weights_only=True`；凡是读取它们的 Python 都用 `-I` 运行。
- repo 内文件一个都没改。`graph.py` 的 png 输出写到了 scratch。

---

## 1. 推断的章节叙事

ch05 的脚本编号 01–05 连续，**没有缺号**。另有一个未编号的作图脚本 `graph.py`：它和 `ch02/graph.py`（ReLU vs GELU）用的是同一个模板，只是多画了一条 Swish 曲线，显然是 5.2 节的配图。

| 顺序 | 文件 | 概念 | 推断的正文内容 |
|---|---|---|---|
| 5.0（纯文字） | — | 动机 | 对比 GPT-2 式的 CodeBot（ch02–03）与 LLaMA 式现代 LLM，列出要替换的部件：learned absolute position embedding、LayerNorm、GELU-FFN、dropout、bias，以及推理时的重复计算 |
| 5.1 | `01_rope.py` | RoPE | 绝对位置 vs 相对位置；二维旋转；把 head_dim 两两配成 d/2 个平面，第 i 个平面以频率 θ_i = θ^(−2i/d) 旋转；cos/sin 预计算后 `register_buffer`；只作用于 Q、K（不作用于 V），且在拆 head 之后；fp32 计算（commit `6855e70` 加入） |
| 5.2 | `graph.py` → `02_swiglu.py` | 激活函数 → GLU 门控 → SwiGLU | 回顾 ReLU/GELU（ch02），引出 Swish/SiLU = x·σ(x)；GLU 的"门"；SwiGLU = O(SiLU(Wx) ⊙ Vx)；hidden = 8/3·d，使参数量与 4d 的 FFN 相等 |
| 5.3 | `03_rmsnorm.py` | RMSNorm | 回顾 `ch02/09_norm_gelu.py` 里手写的 LayerNorm；去掉减均值和 β，只除以 RMS；之后改用 `nn.RMSNorm` |
| 5.4 | `04_improved_gpt.py` | 组装"改良版 GPT" | Pre-norm Block：RMSNorm→MHA(RoPE)→残差，RMSNorm→SwiGLU→残差；去掉 pos_embed、dropout、bias；unembed 不再和 embed 共享权重；checkpoint 里多存一个 theta。演示用 vocab=10000，正好是 StoryBot tokenizer 的词表大小 |
| 5.5 | `05_kvcache.py` | KV Cache | 自回归生成中的重复计算；逐层缓存 K/V；Prefill（首轮处理整段 prompt）vs Decode（之后每次 1 个 token）；RoPE 需要 offset；causal mask 只在训练和 prefill 时需要；`clear_cache`；CPU 测速（打印"高速化率"）；文件末尾有一段被注释掉的"输出一致性确认" |
| 推断（纯文字或后续章节） | `webbot/model.py` | GQA、`F.scaled_dot_product_attention`（Flash Attention）、滑动窗口 cache | ch05 里没有对应脚本，只出现在 ch09 用的 webbot 模型中。可能放在没有代码的 ch08 或 ch09 正文里讲（**推测，未证实**） |

模型谱系（全部用 `diff` 核对过）：

```
codebot/model.py   (ch02–03，GPT-2 式：pos_embed + LayerNorm + GELU-FFN + dropout + bias + weight tying)
 └─ ch05/04_improved_gpt.py   (RoPE + RMSNorm + SwiGLU；无 bias/dropout；embed 与 unembed 不共享)
     └─ ch05/05_kvcache.py    (+ KV cache，RoPE 支持 offset)
         ≡ storybot/model.py  (逻辑完全相同，只有注释和 mask 条件的写法不同) ← ch06 预训练/DPO 使用
             └─ webbot/model.py (+ GQA n_kv_head，+ SDPA(enable_gqa=True)，+ cache 截断为滑动窗口，ffn 改名 mlp) ← ch09 使用
```

细节：05 的测速模型（vocab 1000、E=384、H=6、L=6、ff=1024）和 `storybot/model.py` 的 `__main__` 配置，都等于 CodeBot 在 ch03 的尺寸（`ch03/01_pretrain.py`：vocab 1000、384、6 头、6 层、ff=4·384），只是 FFN 换成了 SwiGLU(1024 = 8/3·384)。正文很可能写的是"与 CodeBot 同尺寸"。

---

## 2. 逐脚本解析

### 2.1 `ch05/01_rope.py` — RoPE + 带 RoPE 的 MHA

**目的**：实现 RoPE，并演示把它插进 Multi-Head Attention。

**关键类与签名**
- `class RoPE(nn.Module)`: `__init__(self, theta, key_dim, max_context_len)`；`forward(self, x)`，其中 x 形状为 `(batch, num_head, context_len, key_dim)`。
- `class MultiHeadAttention(nn.Module)`: `__init__(self, embed_dim, n_head, head_dim, rope=None)`；`forward(self, x)`，x 形状为 `(B, C, E)`。

**核心算法（原文摘录）**

```python
half_ids = torch.arange(0, half)
inv_freq = 1.0 / (theta ** ( (2.0 * half_ids) / key_dim ))  # (half,)
positions = torch.arange(max_context_len)
angles = positions[:, None] * inv_freq[None, :]              # (max_context_len, half)
self.register_buffer("cos_cache", torch.cos(angles))
self.register_buffer("sin_cache", torch.sin(angles))
...
x = x.float()                                  # 用 fp32 计算
cos = self.cos_cache[:context_len]; sin = self.sin_cache[:context_len]
x_even = x[..., 0::2]; x_odd = x[..., 1::2]    # 相邻两维配成一对（interleaved）
x_rot_even = x_even * cos - x_odd * sin
x_rot_odd  = x_even * sin + x_odd * cos
out = torch.stack([x_rot_even, x_rot_odd], dim=-1).reshape(B, H, C, D)
return out.to(input_dtype)
```

MHA 里的用法是先拆 head，再对 Q、K（不对 V）做 RoPE：`Q = self.rope(Q); K = self.rope(K)`。之后是常规的 `QKᵀ/√D` → `tril` mask → softmax → `·V` → `W_o`。四个投影都是 `bias=False`。

数学：第 i 对 `(x_{2i}, x_{2i+1})` 在位置 m 上旋转角度 m·θ_i，θ_i = 10000^(−2i/d)。等价写法是把这对数看成复数 `x_{2i} + i·x_{2i+1}`，再乘以 `e^{i m θ_i}`。于是 ⟨R_m q, R_n k⟩ = qᵀ R_mᵀ R_n k = qᵀ R_{n−m} k，只依赖相对位置。§3-A 对这几种写法做了数值验证。

**日文注释翻译**

| 原文 | 中文 |
|---|---|
| `key_dimは偶数` | key_dim 必须是偶数 |
| `入力の型を保存し、float32で計算` | 记下输入的 dtype，用 float32 计算 |
| `偶数・奇数インデックスに分割` | 按偶数/奇数下标拆开 |
| `回転を適用` | 施加旋转 |
| `偶数・奇数インデックスを元に戻す` | 把偶数/奇数下标交错还原 |
| `元の型に戻す` | 转回原来的 dtype |
| `RoPEの適用` | 应用 RoPE |
| `ハイパーパラメータ / 初期化 / テスト用データ / 順伝播` | 超参数 / 初始化 / 测试数据 / 前向传播 |

**实际输出**（E=512、H=8、D=64、θ=10000、max_ctx=1024，输入 (2,10,512)）：

```
torch.Size([2, 10, 512])
wall 2.902 s   （绝大部分是 import torch 的时间）
```

**git 历史**：初版（`933052b`）的 MHA `__init__` 用一个全局变量 `max_context_len`（并不是参数）注册了 mask buffer，`fdba7c9` 删掉了；`e3c46d0` 把 `self.head_dim**0.5` 改成 `D**0.5`；`6855e70`（"rope -> fp32"）加上了"转 fp32 计算、再转回原 dtype"。

---

### 2.2 `ch05/02_swiglu.py` — SwiGLU

**关键定义**：`def silu(x)`（教学用的辅助函数，`forward` 里实际调用的是 `F.silu`）；`class SwiGLU(nn.Module)`: `__init__(self, x_dim, hidden_dim=None)`，默认 `hidden_dim = int(x_dim * 8 / 3)`；三个 `nn.Linear(..., bias=False)`：`W`、`V`（x_dim→hidden）和 `O`（hidden→x_dim）。

**核心**

```python
a = self.W(x); b = self.V(x)
gated = F.silu(a) * b        # SiLU(a) 当"门"，逐元素乘 b
out = self.O(gated)
```

**参数匹配**：普通 FFN（d→4d→d，无 bias）有 2·d·4d = 8d² 个参数；SwiGLU 有 3·d·h，令 h = 8d/3 就得到 8d²。d=384 时 h=1024，两者都是 1,179,648（§3-C1、D1）。d=512 时 `int(512*8/3)=1365`，两者不再严格相等（2,096,640 vs 2,097,152）；ch06 的 StoryBot 则直接用了 `ff_dim=1344`（=21×64，是 64 的倍数）。

日文注释：无。**实际输出**：无打印（只定义类）。wall 2.606 s。

---

### 2.3 `ch05/graph.py` — ReLU / GELU / Swish 对比图

用 NumPy 画三条曲线，x∈[−3,3]：ReLU、GELU（tanh 近似式 `0.5x(1+tanh(√(2/π)(x+0.044715x³)))`）、Swish=x·σ(x)。图保存为当前目录下的 `activation_comparison.png`（PNG，`c9b6e2e` 把 svg 改成了 png）。

**日文注释翻译**：データ生成=生成数据；ReLU関数=ReLU 函数；GELU関数（近似式）=GELU 函数（近似式）；Swish関数（SiLU）=Swish 函数（SiLU）；プロット作成=作图；軸の設定=坐标轴设置；グリッド=网格；凡例=图例；余白調整=调整边距；PNGで保存=保存为 PNG。

**实际运行**（cwd 设为 scratch）：wall 1.890 s，生成 `activation_comparison.png`（54,529 B）。图上能看到：Swish 在负半轴有一个明显的下凹（最小值约 −0.278，位于 x≈−1.278），GELU 的下凹更浅（约 −0.170，位于 x≈−0.752）；正半轴三条线都趋近 y=x，其中 Swish 最低（§3-C3）。

---

### 2.4 `ch05/03_rmsnorm.py` — 手写 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, x):                       # 注意：参数名叫 x，其实是维度
        self.gamma = nn.Parameter(torch.ones(x)); self.eps = 1e-5
    def forward(self, x):
        ms = (x**2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        return self.gamma * x / rms
```

和 LayerNorm 相比：不减均值，没有 β，只有 d 个参数（LayerNorm 是 2d 个）。当输入均值为 0 时两者完全相同（§3-E2、D6）。**实际输出**：无打印。wall 2.959 s。

**注意**：04/05/storybot/webbot 用的都是 `nn.RMSNorm(embed_dim)`，它的默认 `eps=None`，表示使用 `torch.finfo(dtype).eps`（fp32 下约 1.19e-7），**和这里手写版的 1e-5 不同**（§3-D4、F6）。

---

### 2.5 `ch05/04_improved_gpt.py` — 改良版 GPT

**关键类与签名**：`RoPE`、`MultiHeadAttention(embed_dim, n_head, head_dim, rope=None)`、`SwiGLU(x_dim, hidden_dim=None)`、`Block(embed_dim, n_head, ff_dim, rope=None)`、`GPT(vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, theta=10000)`，外加 `save(file_path)` 与 `@classmethod load_from(file_path, device='cpu')`。

**核心**

```python
class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, rope=None):
        head_dim = embed_dim // n_head
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, rope)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ff_dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))      # pre-norm + 残差
        x = x + self.ffn(self.norm2(x))
        return x

class GPT(nn.Module):
    ...
        self.embed = nn.Embedding(vocab_size, embed_dim)          # 没有 pos_embed
        rope = RoPE(theta, head_dim, max_context_len)              # 一个 RoPE 实例被所有层共享
        self.blocks = nn.ModuleList([Block(embed_dim, n_head, ff_dim, rope) for _ in range(n_layer)])
        self.norm = nn.RMSNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False) # 不和 embed 共享
        self.apply(self._init_weights)                              # Linear/Embedding ~ N(0, 0.02)
    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks: x = block(x)
        return self.unembed(self.norm(x))
```

**相对 codebot 的改动（逐项核对 `codebot/model.py`）**

| 部件 | codebot（ch02–03） | 改良 GPT（ch05） |
|---|---|---|
| 位置信息 | `pos_embed = nn.Embedding(max_context_len, E)`，加在 token embedding 上 | RoPE，在每层 attention 内作用于 Q/K，**0 个参数** |
| Norm | `nn.LayerNorm`（γ、β） | `nn.RMSNorm`（只有 γ） |
| FFN | `Linear(E,4E)` + `nn.GELU()` + `Linear(4E,E)` + Dropout，带 bias | `SwiGLU(E, 8E/3)`，三个矩阵，无 bias |
| Dropout | 0.1（embedding、attention 权重、attention 输出、FFN 各一处） | 无 |
| bias | FFN 和 unembed 有 bias（attention 投影本来就没有） | 全部没有（§3-F1：bias 参数列表为空） |
| weight tying | `self.embed.weight = self.unembed.weight` | 不共享 |
| 结构 | pre-norm，末尾 LayerNorm | pre-norm，末尾 RMSNorm |
| 初始化 | N(0, 0.02) | 相同 |

顺带一提：codebot 自己定义了 `LayerNorm` 和 tanh 近似的 `GELU` 类，但实际用的是 `nn.LayerNorm` 和 `nn.GELU()`（后者默认是精确的 erf 版本）。两种 GELU 的最大差约 4.7e-4（§3-C4/D3）。

同尺寸比较（§3-F2，vocab 1000、E=384、H=6、L=6）：codebot 是 11,121,640 个参数，改良版是 11,389,824（多出的主要是不再共享的 unembed，+384,000；少掉的是 pos_embed −98,304，以及 norm 和 bias）。

**日文注释**：`ハイパーパラメータ`=超参数；`モデルの初期化`=初始化模型；`動作テスト`=运行测试。

**实际输出**（vocab 10000、ctx 256、E=384、H=6、L=6、ff=1024，输入 (8,256)）：

```
torch.Size([8, 256, 10000])
wall 3.331 s
```

参数量 18,301,824（§3-F1，按组：embed 3,840,000 / attn 3,538,944 / ffn 7,077,888 / norm1+norm2 4,608 / 末尾 norm 384 / unembed 3,840,000）。

**git 历史**：`85fb33a` 把 context_len 改名为 max_context_len；`838ef97` 加入 RoPE 的 fp32 计算，并把 `mlp` 改名为 `ffn`；`5ff851d` 把输出注释从 `[8,256,1000]` 改成 `[8,256,10000]`。

---

### 2.6 `ch05/05_kvcache.py` — KV Cache 与测速

**关键类与签名**
- `RoPE.forward(self, x, offset=0)`：取 `cos_cache[offset:offset+C]`；当 `offset + C > max_context_len` 时**把 offset 夹到 `max_context_len − C`**。
- `MultiHeadAttention`：新增 `self.k_cache / self.v_cache / self.cache_offset`；`forward(self, x, use_cache=False)`；`clear_cache()`。
- `Block.forward(x, use_cache=False)`、`Block.clear_cache()`；`GPT.forward(ids, use_cache=False)`、`GPT.clear_cache()`。
- 模块级函数：`generate_without_cache(model, start_ids, max_new_tokens)`、`generate_with_cache(model, start_ids, max_new_tokens)`、`measure_generation_time(model, start_ids, use_cache, num_tokens=200)`。

**核心算法（原文摘录）**

```python
if self.rope is not None:
    if use_cache:
        Q = self.rope(Q, self.cache_offset)       # 新 token 的绝对位置从 cache_offset 开始
        K = self.rope(K, self.cache_offset)
    else:
        Q = self.rope(Q); K = self.rope(K)

if use_cache:
    is_first_call = (self.k_cache is None)        # Prefill 还是 Decode
    if is_first_call:
        self.k_cache = K; self.v_cache = V
    else:
        self.k_cache = torch.cat([self.k_cache, K], dim=2)
        self.v_cache = torch.cat([self.v_cache, V], dim=2)
    self.cache_offset += C
    K = self.k_cache; V = self.v_cache            # 用缓存里全部的 K/V

scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
if not use_cache or is_first_call:                # 训练 / Prefill 才加 mask
    mask = torch.tril(torch.ones(C, C, device=scores.device))
    scores = scores.masked_fill(mask == 0, float('-inf'))
```

```python
def generate_with_cache(model, start_ids, max_new_tokens):
    model.eval(); ids = start_ids; next_id = start_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(next_id, use_cache=True)   # 第一次是整段 prompt（prefill），之后每次 1 个 token（decode）
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    return ids
```

要点：缓存的是**旋转之后的 K**（以及原样的 V），每层一份（§3-K1）。Decode 时 Q 的形状是 (B,H,1,D)，K 是 (B,H,T,D)，缓存里的 key 全都在过去，所以不需要 mask。

**日文注释翻译**

| 原文 | 中文 |
|---|---|
| `offsetを考慮して位置エンコーディングを取得` | 考虑 offset 取出位置编码 |
| `KV-Cache用の変数を追加` / `Keyのキャッシュ` / `Valueのキャッシュ` / `現在のキャッシュ位置を追跡` | 添加 KV-Cache 用的变量 / Key 的缓存 / Value 的缓存 / 记录当前缓存位置 |
| `RoPEにoffsetを渡す` | 把 offset 传给 RoPE |
| `KV-Cacheの処理` | KV-Cache 的处理 |
| `Prefill（初回）かDecode（2回目以降）かを判定` | 判断是 Prefill（第一次调用）还是 Decode（第二次及以后） |
| `初回:キャッシュを初期化` | 第一次：初始化缓存 |
| `2回目以降:新しいKeyとValueをキャッシュに追加` | 之后：把新的 Key/Value 追加到缓存 |
| `オフセットを更新(次のトークンの位置へ)` | 更新 offset（移到下一个 token 的位置） |
| `キャッシュされた全てのKeyとValueを使用` | 使用缓存中的全部 Key 与 Value |
| `通常のAttention計算` | 常规的 Attention 计算 |
| `Causal Mask（KVキャッシュ使用時は不要）` | Causal Mask（使用 KV 缓存时不需要）← **这句注释不准确**：prefill 时需要，代码也确实在第一次调用时加了 mask |
| `キャッシュをクリアする` / `全てのブロックのキャッシュをクリアする` | 清空缓存 / 清空所有 block 的缓存 |
| `最初のトークン` / `毎回、全てのトークンを渡す` / `新しいトークンを連結` / `1トークンずつ生成` | 最初的 token / 每次都传入全部 token / 拼接新 token / 每次生成一个 token |
| `固定シード` | "固定种子"← **名不副实**：这里固定的只是起始 token 42，并没有设随机种子 |
| `KV-Cacheなし / あり / 高速化率 …倍` | 无 / 有 KV-Cache / 加速比 …倍 |
| （注释块）`出力の一致確認` / `同じ開始トークンで生成` / `出力が一致しました` / `出力が一致しません` / `差分の数` | 输出一致性确认 / 用相同起始 token 生成 / 输出一致 / 输出不一致 / 不一致的个数 |

**实际输出**（原脚本，跑了 3 次；CPU 被占满时测的）

```
KV-Cacheなし: 5.09秒   KV-Cacheあり: 2.17秒   高速化率: 2.3倍   (wall 10.804 s)
KV-Cacheなし: 5.26秒   KV-Cacheあり: 1.90秒   高速化率: 2.8倍   (wall 10.512 s)
KV-Cacheなし: 6.40秒   KV-Cacheあり: 2.19秒   高速化率: 2.9倍   (wall 11.936 s)
```

被注释掉的一致性检查：原样执行会 `NameError: name 'max_new_tokens' is not defined`（这个变量在 `23046f6` 中被删了）；补上 `max_new_tokens = 200` 后输出 `✓ 出力が一致しました!`，两边都是 `torch.Size([1, 201])`（§3-G0）。

**git 历史（`git log -p -- ch05/05_kvcache.py`）**

| commit | 日期 | 改动 |
|---|---|---|
| `933052b` | 2025-12-18 | 初版。mask 条件是 `if not use_cache:`，**prefill 时也不加 causal mask**（bug） |
| `85fb33a` | 2026-01-08 | context_len→max_context_len；把误放在 MHA 里的 save/load 移到 GPT |
| `e3c46d0` | 2026-01-20 | `self.head_dim**0.5` → `D**0.5` |
| `838ef97` | 2026-01-24 | RoPE 用 fp32 计算并转回原 dtype |
| `8b3911f` | 2026-02-18 | **"Fix KV cache bug: Apply causal mask during Prefill phase"**：引入 `is_first_call`，条件改成 `not use_cache or (use_cache and is_first_call)`。同一个 commit 也改了 storybot/model.py，并新增了 webbot/model.py（SDPA 的 `is_causal=is_first_call`） |
| `5ff851d` | 2026-03-27 | `mlp`→`ffn`；**测速公平性修正**：之前无缓存测 200 个 token，有缓存却用默认的 100 个，加速比被高估 |
| `23046f6` | 2026-03-28 | `num_tokens` 默认值改为 200，删掉 `max_new_tokens` 变量，于是注释块里引用了一个不存在的名字 |
| `c240183` | 2026-04-07 | 把条件简化成 `if not use_cache or is_first_call:`，注释改成"KV缓存使用时不需要"（表述不准确） |

这个 bug 的特点：**脚本自带的测试用的是 1-token prompt（`[[42]]`），根本测不出来**。只有 prompt ≥ 2 个 token、而且层数 ≥ 2 时才会出现差异（§3-G4；在真实训练好的 StoryBot 上也复现了，§3-J4）。

---

### 2.7 `storybot/model.py`（ch06 使用）

和 `ch05/05_kvcache.py` 的模型部分逐行 diff，**逻辑完全相同**，只有三处不同：
1. mask 条件写成 `if not use_cache or (use_cache and is_first_call):`（语义等价），并附三行注释：
   - `学習時（use_cache=False）: 常に適用` → 训练时（use_cache=False）：总是应用
   - `Prefill（初回・プロンプト全体処理）: 適用（各トークンは前方のみ）` → Prefill（第一次调用，处理整段 prompt）：应用（每个 token 只能看前面）
   - `Decode（2回目以降・1トークン生成）: 不要（新トークンは全キャッシュにattend）` → Decode（之后每次生成 1 个 token）：不需要（新 token 对整个缓存做 attention）
2. 没有 generate/measure 等测速代码；有一个 `if __name__ == "__main__":` 演示（vocab 1000、ctx 256、E=384、H=6、L=6、ff=int(384·8/3)=1024）。
3. 部分注释措辞不同。

git：`8b3911f` 修 prefill mask；`234a9c6`（2026-03-27，"原稿との整合性修正"= 与书稿保持一致）把 `self.mlp` 改名为 `self.ffn`。**这次改名导致无法加载 HF 上发布的 checkpoint**（checkpoint 的最后上传时间是 2026-01-23，里面的 key 还是 `mlp`），§3-J1 有实测。

**实际输出**：`出力形状: torch.Size([1, 256, 1000])`（出力形状 = 输出形状），wall 3.931 s。

相关的 `storybot/utils.py::generate`（带 cache 的生成循环）：开头先 `model.clear_cache()`，第一次把整段 prompt 送进去（prefill），之后每次只送 `next_id`；temperature=0 时用 argmax，否则做 multinomial 采样；遇到 end token 就停。`5ff851d` 删掉了一行多余的 `next_id = torch.multinomial(probs, ...)`：它会覆盖 greedy 的结果，而且在 temperature=0 时 `probs` 根本没有定义。

---

### 2.8 `webbot/model.py`（ch09 使用）

**和 storybot 的 diff（完整核对）**

| 位置 | storybot | webbot |
|---|---|---|
| MHA 签名 | `(embed_dim, n_head, head_dim, rope)` | `(embed_dim, n_head, n_kv_head, head_dim, rope)` |
| K/V 投影 | `Linear(E, H*D)` | `Linear(E, KV*D)`（**GQA**） |
| attention | 手写 matmul + `tril` mask + softmax | `F.scaled_dot_product_attention(Q, K, V, is_causal=..., enable_gqa=True)` |
| mask | prefill/训练时用 `tril` | 训练用 `is_causal=True`；cache 模式下 `is_causal = is_first_call` |
| cache 长度 | 无限增长 | 超过 `max_cache_len = rope.cos_cache.size(0)` 时只保留最后 max 个，并把 `cache_offset = max_cache_len` |
| FFN 属性名 | `self.ffn` | `self.mlp`（state_dict key 不兼容，§3-I8） |
| `silu` 辅助函数 | 有（未使用） | 删除 |
| GPT 签名 | `(V, ctx, E, H, L, ff, theta)` | `(V, ctx, E, H, KV, L, ff, theta)`；save/load 多一个 `n_kv_head` |
| `__main__` | 1000/256/384/6/6 | 50000/1024/768/12 头/4 KV 头/12 层/ff 2048，并打印参数量 |

**核心（原文摘录）**

```python
self.W_k = nn.Linear(E, KV * D, bias=False)
self.W_v = nn.Linear(E, KV * D, bias=False)
...
max_cache_len = self.rope.cos_cache.size(0) if self.rope else 2048
if self.k_cache.size(2) > max_cache_len:
    self.k_cache = self.k_cache[:, :, -max_cache_len:]
    self.v_cache = self.v_cache[:, :, -max_cache_len:]
    self.cache_offset = max_cache_len
...
if use_cache:
    is_causal = is_first_call
    hidden = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True)
else:
    hidden = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
```

**日文注释**：`KV-Cache用の変数`=KV-Cache 用的变量；`初回かどうかを判定（キャッシュ更新前に）`=判断是否第一次调用（必须在更新缓存之前判断）；`コンテキスト長制限: キャッシュが長すぎる場合は古い部分を切り捨て`=上下文长度限制：缓存过长时丢弃最旧的部分；`Flash Attention（GQA対応）`=Flash Attention（支持 GQA）；`学習時: is_causal=Trueでcausal mask適用`=训练时用 is_causal=True 施加 causal mask；`パラメータ数を表示`=显示参数量。

**实际输出**

```
パラメータ数: 152,316,672 (152.3M)
出力形状: torch.Size([1, 1024, 50000])
wall 7.468 s
```

（パラメータ数 = 参数量。）`enable_gqa` 需要 torch ≥ 2.5，`f85907f` 因此把依赖改成了 `torch>=2.5.0`。

---

## 3. 可最小化验证的事实清单（全部实际运行过）

"浏览器"一列的含义：✅ 表示只用 NumPy/纯 Python，可以在 Pyodide 中运行；❌ 表示需要 torch 或 repo 代码。

### 3-A RoPE（纯 NumPy）—— `a_rope_numpy.py`（0.68 s）✅

```python
import numpy as np

def rope(x, pos, theta=10000.0):
    D = x.shape[-1]
    inv_freq = 1.0 / theta ** (np.arange(D // 2) * 2.0 / D)
    ang = np.asarray(pos, dtype=np.float64)[..., None] * inv_freq
    c, s = np.cos(ang), np.sin(ang)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty(np.broadcast_shapes(x.shape, c.shape[:-1] + (D,)))
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out

rng = np.random.default_rng(0); D = 64
q, k = rng.standard_normal(D), rng.standard_normal(D)
print("|q|, |rope(q,37)|:", np.linalg.norm(q), np.linalg.norm(rope(q, 37)))
print("rope(q,0)==q:", np.allclose(rope(q, 0), q))
for m, n in [(5, 2), (105, 102), (1005, 1002), (3, 0)]:
    print(f"m={m:5d} n={n:5d} m-n={m-n}:", rope(q, m) @ rope(k, n))
P = np.arange(128)
S = rope(q[None, :], P) @ rope(k[None, :], P).T
print("Toeplitz max |S[m,n]-S[m+1,n+1]|:", np.abs(S[1:, 1:] - S[:-1, :-1]).max())
print("but S[5,2] vs S[2,5] (m-n=+3 vs -3):", S[5, 2], S[2, 5])
m = 7; inv_freq = 1.0 / 10000 ** (np.arange(D // 2) * 2.0 / D)
z = (q[0::2] + 1j * q[1::2]) * np.exp(1j * m * inv_freq); r = rope(q, m)
print("complex == rope:", np.allclose(z.real, r[0::2]) and np.allclose(z.imag, r[1::2]))
def R(m):
    M = np.zeros((D, D))
    for i, w in enumerate(inv_freq):
        c, s = np.cos(m * w), np.sin(m * w); M[2*i:2*i+2, 2*i:2*i+2] = [[c, -s], [s, c]]
    return M
print("R(9)q == rope(q,9):", np.allclose(R(9) @ q, rope(q, 9)))
print("R(5).T@R(12) == R(7):", np.allclose(R(5).T @ R(12), R(7)))
print("inv_freq[0], [1], [-1]:", inv_freq[0], inv_freq[1], inv_freq[-1])
print("wavelength 2pi/inv_freq min, max (tokens):", 2*np.pi/inv_freq[0], 2*np.pi/inv_freq[-1])
def rope_half(x, pos, theta=10000.0):                     # HF 的 rotate_half 约定
    D = x.shape[-1]; h = D // 2
    inv = 1.0 / theta ** (np.arange(h) * 2.0 / D); a = pos * inv; c, s = np.cos(a), np.sin(a)
    x1, x2 = x[:h], x[h:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c])
perm = np.concatenate([np.arange(0, D, 2), np.arange(1, D, 2)])
print("rope(q)[perm] == rope_half(q[perm]):", np.allclose(rope(q, 11)[perm], rope_half(q[perm], 11)))
print("but rope(q) == rope_half(q)?", np.allclose(rope(q, 11), rope_half(q, 11)))
ones = np.ones(D)
for d in [0, 1, 2, 5, 10, 50, 100, 500, 1000]:
    print(f"delta={d:4d}: {rope(ones, d) @ rope(ones, 0) / D:+.4f}")
```

输出（原样）：

```
|q|, |rope(q,37)|: 7.315343549442273 7.315343549442273
rope(q,0)==q: True
m=    5 n=    2 m-n=3: -8.44615245571497
m=  105 n=  102 m-n=3: -8.44615245571497
m= 1005 n= 1002 m-n=3: -8.446152455714971
m=    3 n=    0 m-n=3: -8.446152455714971
Toeplitz max |S[m,n]-S[m+1,n+1]|: 5.3290705182007514e-14
but S[5,2] vs S[2,5] (m-n=+3 vs -3): -8.446152455714971 -13.049657349106212
complex == rope: True
R(9)q == rope(q,9): True
R(5).T@R(12) == R(7): True
inv_freq[0], [1], [-1]: 1.0 0.7498942093324559 0.0001333521432163324
wavelength 2pi/inv_freq min, max (tokens): 6.283185307179586 47117.2427801674
rope(q)[perm] == rope_half(q[perm]): True
but rope(q) == rope_half(q)? False
delta=   0: +1.0000
delta=   1: +0.9662
delta=   2: +0.8845
delta=   5: +0.7345
delta=  10: +0.6579
delta=  50: +0.4898
delta= 100: +0.5586
delta= 500: +0.2006
delta=1000: +0.2789
```

| ID | 事实 | 观测 |
|---|---|---|
| A1 | RoPE 是旋转，范数不变；位置 0 时是恒等变换 | 7.315343549442273 = 7.315343549442273；True |
| A2 | q·k 只依赖 m−n（Toeplitz） | 四组 m−n=3 的分数都是 −8.44615…；对角线最大偏差 5.3e-14 |
| A2' | 依赖的是**有符号**距离，不是 \|m−n\| | S[5,2]=−8.446，S[2,5]=−13.050 |
| A3 | 等价于复数乘法 e^{imθ} | True |
| A4 | 矩阵形式满足 R(m)ᵀR(n)=R(n−m) | True |
| A5 | d=64 时频率范围：θ_0=1，θ_31=1.3335e-4；波长 6.28 到 47,117 个 token | 见输出 |
| A6 | repo 用 interleaved 配对，HF 用 rotate_half：两者只差一个维度置换，但**不能直接互换** | True / False |
| A7 | 长程衰减：整体在下降，但**不是单调的** | 1.0 → 0.49(Δ50) → 0.56(Δ100) → 0.20(Δ500) → 0.28(Δ1000) |

### 3-B RoPE 对照 repo 代码（torch）—— `b_rope_torch.py`（3.17 s）❌

```python
import numpy as np, torch
ns = {}; exec(open('ch05/01_rope.py', encoding='utf-8').read().split('# ハイパーパラメータ')[0], ns)
RoPE, MHA = ns['RoPE'], ns['MultiHeadAttention']
ns5 = {}; exec(open('ch05/05_kvcache.py', encoding='utf-8').read().split('# テスト')[0], ns5)
torch.manual_seed(0)
rope = RoPE(10000, 64, 1024)
print("params:", sum(p.numel() for p in rope.parameters()),
      "| buffers:", {k: tuple(v.shape) for k, v in rope.named_buffers()})
x = torch.randn(1, 1, 10, 64)
D = 64; inv = 1.0 / 10000 ** (np.arange(D // 2) * 2.0 / D); ang = np.arange(10)[:, None] * inv
xe, xo = x[0, 0, :, 0::2].double().numpy(), x[0, 0, :, 1::2].double().numpy()
ref = np.empty((10, D)); ref[:, 0::2] = xe*np.cos(ang) - xo*np.sin(ang); ref[:, 1::2] = xe*np.sin(ang) + xo*np.cos(ang)
print("max |repo - numpy|:", np.abs(rope(x)[0, 0].numpy() - ref).max())
xb = x.bfloat16()
print("out dtype:", rope(xb).dtype)
naive = (xb[..., 0::2] * rope.cos_cache[:10].bfloat16() - xb[..., 1::2] * rope.sin_cache[:10].bfloat16())
print("max err even-part, repo(bf16 io) vs fp32 truth:",
      (rope(xb).float()[..., 0::2] - rope(xb.float())[..., 0::2]).abs().max().item(),
      "| naive all-bf16:", (naive.float() - rope(xb.float())[..., 0::2]).abs().max().item())
for L in [256, 1024, 32768]:
    r = RoPE(10000, 64, L); a64 = np.arange(L)[:, None] * inv
    print(f"max_context_len={L:6d}: max |cos_cache - cos64| = {np.abs(r.cos_cache.numpy() - np.cos(a64)).max():.2e}")
r5 = ns5['RoPE'](10000, 64, 256)
Q, K = torch.randn(1, 2, 8, 64), torch.randn(1, 2, 8, 64)
s0 = r5(Q, 0) @ r5(K, 0).transpose(-2, -1); s100 = r5(Q, 100) @ r5(K, 100).transpose(-2, -1)
print("max |scores(offset=0) - scores(offset=100)|:", (s0 - s100).abs().max().item(),
      "| scale of scores:", s0.abs().mean().item())
mha = MHA(512, 8, 64, rope=RoPE(10000, 64, 1024))
print("out:", tuple(mha(torch.randn(2, 10, 512)).shape))
```

```
params: 0 | buffers: {'cos_cache': (1024, 32), 'sin_cache': (1024, 32)}
max |repo - numpy|: 3.7039912648140216e-07
out dtype: torch.bfloat16
max err even-part, repo(bf16 io) vs fp32 truth: 0.007399797439575195 | naive all-bf16: 0.011633157730102539
max_context_len=   256: max |cos_cache - cos64| = 8.78e-06
max_context_len=  1024: max |cos_cache - cos64| = 3.57e-05
max_context_len= 32768: max |cos_cache - cos64| = 1.20e-03
max |scores(offset=0) - scores(offset=100)|: 2.8133392333984375e-05 | scale of scores: 6.596142768859863
out: (2, 10, 512)
```

| ID | 事实 | 观测 |
|---|---|---|
| B1 | RoPE 没有可学习参数；buffer 形状是 (max_ctx, head_dim/2) | params 0；(1024, 32) |
| B2 | repo 的 RoPE 和 NumPy 参考实现一致 | 最大差 3.7e-7 |
| B3 | 输入 bf16 时，内部用 fp32 算，输出还是 bf16，而且比全程 bf16 更准 | 0.0074 vs 0.0116 |
| B4 | cos/sin 表是用 fp32 算角度的，误差随位置变大 | 256：8.8e-6；1024：3.6e-5；32768：1.2e-3 |
| B5 | 用 repo 的 offset 参数把 Q、K 一起平移 100 个位置，分数不变 | 2.8e-5（分数本身的量级约 6.6） |
| B6 | 01 脚本的形状 | (2,10,512) |

### 3-C SwiGLU 与激活函数（纯 NumPy）—— `c_swiglu_numpy.py`（0.81 s）✅

```python
import math
import numpy as np
for d in [384, 512, 768]:
    h = int(d * 8 / 3)
    print(f"d={d}: hidden={h}  FFN 2*d*4d={2*d*4*d:,}  SwiGLU 3*d*h={3*d*h:,}")
for d in [4096]:
    h = int(2 * 4 * d / 3); h256 = 256 * ((h + 255) // 256)
    print(f"d={d}: int(8d/3)={h}, rounded to 256 -> {h256}")
x = np.linspace(-6, 6, 1_200_001)
silu = x / (1 + np.exp(-x))
gelu_tanh = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
erf = np.vectorize(math.erf); gelu_erf = 0.5 * x * (1 + erf(x / np.sqrt(2)))
i = silu.argmin(); j = gelu_erf.argmin()
print(f"SiLU min at x={x[i]:.4f}, value={silu[i]:.4f}")
print(f"GELU min at x={x[j]:.4f}, value={gelu_erf[j]:.4f}")
print(f"max |tanh - erf| on [-6,6] = {np.abs(gelu_tanh - gelu_erf).max():.2e}")
for v in [-3, -1, 0, 1, 3]:
    print(v, round(v / (1 + math.exp(-v)), 4))
```

```
d=384: hidden=1024  FFN 2*d*4d=1,179,648  SwiGLU 3*d*h=1,179,648
d=512: hidden=1365  FFN 2*d*4d=2,097,152  SwiGLU 3*d*h=2,096,640
d=768: hidden=2048  FFN 2*d*4d=4,718,592  SwiGLU 3*d*h=4,718,592
d=4096: int(8d/3)=10922, rounded to 256 -> 11008
SiLU min at x=-1.2785, value=-0.2785
GELU min at x=-0.7518, value=-0.1700
max |tanh - erf| on [-6,6] = 4.73e-04
-3 -0.1423
-1 -0.2689
0 0.0
1 0.7311
3 2.8577
```

| ID | 事实 | 观测 |
|---|---|---|
| C1 | h=8d/3 时 SwiGLU 与 4d FFN 参数量相等（d 是 3 的倍数时严格相等） | 384/768 严格相等；512 时差 512 |
| C2 | LLaMA 把 8d/3 向上取整到 256 的倍数（d=4096 → 11008）；repo 不取整 | 10922 → 11008 |
| C3 | SiLU 不单调，最小值约 −0.2785，位于 x*≈−1.2785。由 f′(x)=0 可得 σ(x*)=1+1/x*，所以最小值 f(x*)=x*+1，与数值一致。GELU 最小值约 −0.170，位于 x≈−0.752 | 见输出 |
| C4 | GELU 的 tanh 近似与精确 erf 版最大相差 4.7e-4 | 4.73e-04 |

### 3-D SwiGLU / RMSNorm 对照 repo 与 PyTorch —— `d_swiglu_rmsnorm_torch.py`（4.66 s）❌

```python
import torch, torch.nn as nn, torch.nn.functional as F
ns = {}; exec(open('ch05/02_swiglu.py', encoding='utf-8').read(), ns)
ns3 = {}; exec(open('ch05/03_rmsnorm.py', encoding='utf-8').read(), ns3)
SwiGLU, silu, RMSNorm = ns['SwiGLU'], ns['silu'], ns3['RMSNorm']
torch.manual_seed(0)
m = SwiGLU(384)
print("hidden:", m.W.out_features, "params:", sum(p.numel() for p in m.parameters()),
      "| FFN(384,1536) no-bias params:", 2 * 384 * 1536)
print("shape:", tuple(m(torch.randn(2, 5, 384)).shape))
x = torch.randn(1000)
print((silu(x) - F.silu(x)).abs().max().item(), (F.silu(x) - nn.SiLU()(x)).abs().max().item())
print(nn.GELU().approximate, (nn.GELU()(x) - nn.GELU(approximate='tanh')(x)).abs().max().item())
d = 384; x = torch.randn(4, 7, d); mine = RMSNorm(d)
print("nn.RMSNorm(d).eps =", nn.RMSNorm(d).eps, "| torch.finfo(float32).eps =", torch.finfo(torch.float32).eps)
print("vs nn.RMSNorm(d, eps=1e-5):", (mine(x) - nn.RMSNorm(d, eps=1e-5)(x)).abs().max().item())
print("vs nn.RMSNorm(d) [eps=None]:", (mine(x) - nn.RMSNorm(d)(x)).abs().max().item())
xs = x * 1e-3
print("tiny input (x*1e-3): custom vs default:", (mine(xs) - nn.RMSNorm(d)(xs)).abs().max().item(),
      "| rms(custom out)=", mine(xs).pow(2).mean().sqrt().item())
print(sum(p.numel() for p in nn.RMSNorm(d).parameters()), sum(p.numel() for p in nn.LayerNorm(d).parameters()))
ln = nn.LayerNorm(d, eps=1e-5, elementwise_affine=False); rms = nn.RMSNorm(d, eps=1e-5, elementwise_affine=False)
print((ln(x) - rms(x - x.mean(-1, keepdim=True))).abs().max().item())
```

```
hidden: 1024 params: 1179648 | FFN(384,1536) no-bias params: 1179648
shape: (2, 5, 384)
2.384185791015625e-07 0.0
none 0.00047319196164608
nn.RMSNorm(d).eps = None | torch.finfo(float32).eps = 1.1920928955078125e-07
vs nn.RMSNorm(d, eps=1e-5): 2.384185791015625e-07
vs nn.RMSNorm(d) [eps=None]: 2.288818359375e-05
tiny input (x*1e-3): custom vs default: 2.5984747409820557 | rms(custom out)= 0.3003879189491272
384 768
4.76837158203125e-07
```

| ID | 事实 | 观测 |
|---|---|---|
| D1 | `SwiGLU(384)` 默认 hidden=1024，参数 1,179,648，与 FFN(384,1536) 相同；形状不变 | ✓ |
| D2 | 辅助函数 `silu` == `F.silu` == `nn.SiLU` | 2.4e-7 / 0 |
| D3 | `nn.GELU()` 默认 `approximate='none'`（erf 版） | 'none'；与 tanh 版差 4.7e-4 |
| D4 | 手写 RMSNorm 的 eps=1e-5，而 `nn.RMSNorm` 默认 eps=None（=finfo.eps≈1.19e-7）。输入量级正常时差 2.3e-5；输入很小时差别巨大（手写版输出 RMS 只有 0.30） | 见输出 |
| D5 | 参数量：RMSNorm 为 d，LayerNorm 为 2d | 384 / 768 |
| D6 | LayerNorm(x) == RMSNorm(x − mean(x))（eps 相同、无仿射） | 4.8e-7 |

### 3-E RMSNorm vs LayerNorm（纯 NumPy）—— `e_norm_numpy.py`（0.54 s）✅

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.standard_normal((3, 384)) * 2.0 + 0.7          # 故意让均值不为 0
def rmsnorm(x, eps=1e-5): return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)
def layernorm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)
y = rmsnorm(x)
print("E1 RMS of output:", np.sqrt((y**2).mean(-1)))
print("E2 LayerNorm(x) == RMSNorm(x - mean):", np.abs(layernorm(x) - rmsnorm(x - x.mean(-1, keepdims=True))).max())
print("E3 RMSNorm(x) vs LayerNorm(x) for non-zero-mean x:", np.abs(layernorm(x) - y).max())
print("E4 scale invariance RMSNorm(10x) vs RMSNorm(x):", np.abs(rmsnorm(10 * x) - y).max())
print("E5 shift: LayerNorm(x+5) vs LayerNorm(x):", np.abs(layernorm(x + 5) - layernorm(x)).max(),
      "| RMSNorm(x+5) vs RMSNorm(x):", np.abs(rmsnorm(x + 5) - y).max())
print("E6 mean of RMSNorm output (not centered):", y.mean(-1))
```

```
E1 RMS of output: [0.99999887 0.99999887 0.99999869]
E2 LayerNorm(x) == RMSNorm(x - mean): 0.0
E3 RMSNorm(x) vs LayerNorm(x) for non-zero-mean x: 0.5466916413510341
E4 scale invariance RMSNorm(10x) vs RMSNorm(x): 4.261898278734577e-06
E5 shift: LayerNorm(x+5) vs LayerNorm(x): 8.881784197001252e-16 | RMSNorm(x+5) vs RMSNorm(x): 3.0259512271749074
E6 mean of RMSNorm output (not centered): [0.3012197  0.3287551  0.29327298]
```

结论：两者都对缩放不变；LayerNorm 还对平移不变，RMSNorm 不是。RMSNorm 的输出 RMS≈1，但均值不为 0。

### 3-F 改良 GPT（torch）—— `f_gpt_torch.py`（7.73 s）❌

```python
import math, sys, torch, torch.nn.functional as F
sys.path.insert(0, '.')
ns = {}; exec(open('ch05/04_improved_gpt.py', encoding='utf-8').read().split('# ハイパーパラメータ')[0], ns)
GPT = ns['GPT']
from codebot.model import GPT as CodeGPT
import storybot.model as sb
torch.manual_seed(0)
def count(m): return sum(p.numel() for p in m.parameters())
def groups(m):
    g = {}
    for n, p in m.named_parameters():
        k = n.split('.')[0] if not n.startswith('blocks') else 'blocks.*.' + n.split('.')[2]
        g[k] = g.get(k, 0) + p.numel()
    return g
m = GPT(10000, 256, 384, 6, 6, int(384 * 8 / 3), 10000)
print(f"{count(m):,}", groups(m))
print("bias params:", [n for n, _ in m.named_parameters() if n.endswith('bias')])
cb = CodeGPT(1000, 256, 384, 6, 6, 4 * 384, 0.1); im = GPT(1000, 256, 384, 6, 6, 1024, 10000)
print(f"codebot : {count(cb):,}", groups(cb)); print(f"improved: {count(im):,}", groups(im))
print("codebot tied embed/unembed:", cb.embed.weight is cb.unembed.weight, "| improved tied:", im.embed.weight is im.unembed.weight)
ids = torch.randint(0, 10000, (8, 256)); tgt = torch.randint(0, 10000, (8, 256))
with torch.no_grad(): logits = m(ids)
print("loss:", F.cross_entropy(logits.view(-1, 10000), tgt.view(-1)).item(), "ln(10000):", math.log(10000))
print("logit std at init:", logits.std().item())
ids2 = ids.clone(); ids2[:, 100] = (ids2[:, 100] + 1) % 10000
with torch.no_grad(): l2 = m(ids2)
print("max diff pos<100:", (logits[:, :100] - l2[:, :100]).abs().max().item(),
      "| max diff pos>=100:", (logits[:, 100:] - l2[:, 100:]).abs().max().item())
sd = m.state_dict(); keys = [k for k in sd if 'rope' in k]
print(len(keys), keys[:2], "| same storage:", len({sd[k].data_ptr() for k in keys if k.endswith('cos_cache')}) == 1)
with torch.no_grad(): e = m.embed(ids); rms = e.pow(2).mean(-1).sqrt().mean().item()
print("embed RMS:", rms, "| out RMS with eps=1e-5:", (rms**2 / (rms**2 + 1e-5)) ** 0.5,
      "| with eps=1.19e-7:", (rms**2 / (rms**2 + 1.19e-7)) ** 0.5)
for name, model in [('04 improved', GPT(1000, 256, 384, 6, 2, 1024)), ('storybot/05', sb.GPT(1000, 256, 384, 6, 2, 1024))]:
    for T in [257, 258]:
        try:
            with torch.no_grad(): out = model(torch.randint(0, 1000, (1, T)))
            print(f"{name} T={T}: ran silently, out {tuple(out.shape)}")
        except Exception as ex:
            print(f"{name} T={T}: {type(ex).__name__}: {str(ex)[:90]}")
```

```
18,301,824 {'embed': 3840000, 'blocks.*.norm1': 2304, 'blocks.*.attn': 3538944, 'blocks.*.norm2': 2304, 'blocks.*.ffn': 7077888, 'norm': 384, 'unembed': 3840000}
bias params: []
codebot : 11,121,640 {'embed': 384000, 'pos_embed': 98304, 'blocks.*.norm1': 4608, 'blocks.*.attn': 3538944, 'blocks.*.norm2': 4608, 'blocks.*.ffn': 7089408, 'norm': 768, 'unembed': 1000}
improved: 11,389,824 {'embed': 384000, 'blocks.*.norm1': 2304, 'blocks.*.attn': 3538944, 'blocks.*.norm2': 2304, 'blocks.*.ffn': 7077888, 'norm': 384, 'unembed': 384000}
codebot tied embed/unembed: True | improved tied: False
loss: 9.286330223083496 ln(10000): 9.210340371976184
logit std at init: 0.3909436762332916
max diff pos<100: 0.0 | max diff pos>=100: 0.6691974401473999
12 ['blocks.0.attn.rope.cos_cache', 'blocks.0.attn.rope.sin_cache'] | same storage: True
embed RMS: 0.019995808601379395 | out RMS with eps=1e-5: 0.9877245463587476 | with eps=1.19e-7: 0.9998512208427852
04 improved T=257: RuntimeError: The size of tensor a (257) must match the size of tensor b (256) at non-singleton dimensio
04 improved T=258: RuntimeError: The size of tensor a (258) must match the size of tensor b (256) at non-singleton dimensio
storybot/05 T=257: ran silently, out (1, 257, 1000)
storybot/05 T=258: RuntimeError: The size of tensor a (258) must match the size of tensor b (2) at non-singleton dimension 
```

（codebot 的 `unembed: 1000` 只是 bias，它的权重和 embed 共享，`named_parameters` 去重后只算一次。）

| ID | 事实 | 观测 |
|---|---|---|
| F1 | 04 配置的参数量是 18,301,824，没有任何 bias | ✓ |
| F2 | 同尺寸下 codebot 11,121,640，改良版 11,389,824；codebot 共享权重，改良版不共享 | ✓ |
| F3 | 初始 loss ≈ ln V + σ²/2，其中 σ≈√E·0.02≈0.392（末尾 RMSNorm 把 RMS 拉到 1，再乘 N(0,0.02²) 的 unembed） | 9.2863 vs 9.2103+0.0764=9.2867 |
| F4 | 因果性：改动位置 100 的 token，之前位置的 logits 完全不变 | 0.0 / 0.669 |
| F5 | 一个 RoPE 实例被 6 层共享，但 state_dict 里出现 12 个 key，都指向同一块存储 | 12，True |
| F6 | 初始化时 embedding 的 RMS≈0.02，这时 eps=1e-5 和 1.19e-7 会让输出尺度差约 1.2% | 0.9877 vs 0.99985 |
| F7 | 长度超过 max_ctx：04 的简单 RoPE 在 257 时直接报错；05/storybot 带 offset 夹取的 RoPE 在 **257（=max+1）时静默运行**（所有位置都用了 255 的旋转），258 时才报错 | 见输出 |

补充（纯 NumPy，✅）：初始 loss 公式的独立验证

```python
import numpy as np
rng = np.random.default_rng(0)
V, E, n = 10000, 384, 2048
h = rng.standard_normal((n, E)); h /= np.sqrt((h**2).mean(-1, keepdims=True))   # RMSNorm 输出，RMS=1
W = rng.normal(0, 0.02, (E, V))                                                  # unembed ~ N(0, 0.02^2)
z = h @ W; y = rng.integers(0, V, n)
lse = np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)
ce = (lse - z[np.arange(n), y]).mean()
print('logit std:', z.std(), '~ sqrt(E)*0.02 =', np.sqrt(E) * 0.02)
print('CE at init:', ce, '| ln V + std^2/2 =', np.log(V) + z.std()**2 / 2, '| ln V =', np.log(V))
```

```
logit std: 0.3917962859584487 ~ sqrt(E)*0.02 = 0.39191835884530846
CE at init: 9.276495713046058 | ln V + std^2/2 = 9.2870925368216 | ln V = 9.210340371976184
```

### 3-G KV Cache 正确性（torch）—— `g_kvcache_torch.py`（34.8 s）❌

**G0：把被注释掉的一致性检查原样执行**

```python
import torch
src = open('ch05/05_kvcache.py', encoding='utf-8').read()
defs, rest = src.split('# テスト')
block = rest.split('"""')[1]                    # 注释块原文
ns = {}; exec(defs, ns)
torch.manual_seed(0)
ns['model'] = ns['GPT'](vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6, ff_dim=1024)
ns['start_ids'] = torch.tensor([[42]])
try:
    exec(block, ns)
except NameError as e:
    print('verbatim block ->', type(e).__name__ + ':', e)
ns['max_new_tokens'] = 200
exec(block, ns)
```

```
=== 出力の一致確認 ===
verbatim block -> NameError: name 'max_new_tokens' is not defined

=== 出力の一致確認 ===
KV-Cacheなし: [42, 406, 406, 878, 414, 85, 878, 85, 256, 85, 256]
KV-Cacheあり: [42, 406, 406, 878, 414, 85, 878, 85, 256, 85, 256]
torch.Size([1, 201]) torch.Size([1, 201])
✓ 出力が一致しました!
```

**G1–G5**

```python
import torch
SRC = open('ch05/05_kvcache.py', encoding='utf-8').read().split('# テスト')[0]
ns = {}; exec(SRC, ns)
GPT, gen_nc, gen_c = ns['GPT'], ns['generate_without_cache'], ns['generate_with_cache']
torch.manual_seed(0)
model = GPT(vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6, ff_dim=1024)
start = torch.tensor([[42]])
model.clear_cache(); out_nc = gen_nc(model, start, 200)
model.clear_cache(); out_c = gen_c(model, start, 200)
print("no-cache:", out_nc[0, :11].tolist()); print("cache   :", out_c[0, :11].tolist())
print("shapes:", tuple(out_nc.shape), tuple(out_c.shape), "| torch.equal(full):", torch.equal(out_nc, out_c))
model.clear_cache(); worst = 0.0
with torch.no_grad():
    for t in range(1, 201):
        full = model(out_nc[:, :t], use_cache=False)[:, -1]
        inc = model(out_nc[:, :t] if t == 1 else out_nc[:, t-1:t], use_cache=True)[:, -1]
        worst = max(worst, (full - inc).abs().max().item())
print("max |logits_nocache - logits_cache| over 200 steps:", worst)
model.clear_cache()
with torch.no_grad():
    model(torch.randint(0, 1000, (1, 8)), use_cache=True)
    for _ in range(5): model(torch.randint(0, 1000, (1, 1)), use_cache=True)
a = model.blocks[0].attn
print("k_cache:", tuple(a.k_cache.shape), "v_cache:", tuple(a.v_cache.shape), "cache_offset:", a.cache_offset)
per_tok = 2 * len(model.blocks) * a.n_head * a.head_dim * 4
print("bytes/token (fp32, all layers, K+V):", per_tok, "| x256 tokens = %.2f MB" % (per_tok * 256 / 2**20))
# G4：重现 2026-02 修掉的 bug（prefill 不加 mask）
assert 'if not use_cache or is_first_call:' in SRC
nsb = {}; exec(SRC.replace('if not use_cache or is_first_call:', 'if not use_cache:'), nsb)
buggy = nsb['GPT'](vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6, ff_dim=1024)
buggy.load_state_dict(model.state_dict())
prompt8 = torch.tensor([[42, 7, 99, 3, 500, 17, 256, 8]])
for name, p in [("1-token prompt", start), ("8-token prompt", prompt8)]:
    model.clear_cache(); ref = gen_nc(model, p, 30)
    buggy.clear_cache(); bad = nsb['generate_with_cache'](buggy, p, 30)
    model.clear_cache(); good = gen_c(model, p, 30)
    print(f"{name}: fixed==nocache {torch.equal(good, ref)} | buggy==nocache {torch.equal(bad, ref)}"
          f" | mismatching tokens (buggy) {(bad != ref).sum().item()}/30")
# G5：不清空 cache 连续调用两次
model.clear_cache(); first = gen_c(model, start, 20)
second = gen_c(model, start, 20)
print("first :", first[0, :12].tolist())
print("second:", second[0, :12].tolist(), "| equal:", torch.equal(first, second),
      "| cache_offset now:", model.blocks[0].attn.cache_offset)
```

```
--- G1 the script's commented-out check, actually run (200 new tokens, greedy)
no-cache: [42, 406, 406, 878, 414, 85, 878, 85, 256, 85, 256]
cache   : [42, 406, 406, 878, 414, 85, 878, 85, 256, 85, 256]
shapes: (1, 201) (1, 201) | torch.equal(full): True
--- G2 logits, step by step (teacher-forced on out_nc)
max |logits_nocache - logits_cache| over 200 steps: 1.7881393432617188e-06
--- G3 cache contents after prefill(8) + 5 decode steps
k_cache: (1, 6, 13, 64) v_cache: (1, 6, 13, 64) cache_offset: 13
bytes/token (fp32, all layers, K+V): 18432 | x256 tokens = 4.50 MB
--- G4 the Feb-2026 bug: no causal mask during prefill (old line `if not use_cache:`)
1-token prompt: fixed==nocache True | buggy==nocache True | mismatching tokens (buggy) 0/30
8-token prompt: fixed==nocache True | buggy==nocache False | mismatching tokens (buggy) 30/30
--- G5 stale cache: generate_with_cache twice without clear_cache()
first : [42, 406, 406, 878, 414, 85, 878, 85, 256, 85, 256, 85]
second: [42, 733, 950, 733, 950, 85, 950, 85, 950, 85, 950, 85] | equal: False | cache_offset now: 40
```

| ID | 事实 | 观测 |
|---|---|---|
| G0 | 注释块原样执行会 NameError；补上变量后输出一致 | ✓ |
| G1 | 有无 cache 的 greedy 输出完全相同（201 个 token） | True |
| G2 | 每一步 logits 的差都在 fp32 舍入误差以内 | 1.79e-6 |
| G3 | cache 形状是 (B,H,T,D)，T = prefill 长度 + decode 次数；fp32 下每个 token 占 2·L·H·D·4 = 18,432 B | (1,6,13,64)；256 个 token 共 4.5 MB |
| G4 | 旧 bug 在 1-token prompt 下看不出来，8-token prompt 下 30/30 个 token 全错 | ✓ |
| G5 | 不调 `clear_cache` 就再生成一次，会接着旧 cache 继续（offset=20→40），结果完全不同 | False |

### 3-H KV Cache 测速（CPU，**有其它负载干扰**）❌

`h_kv_timing.py <threads>`（max_ctx 改成 1024 以便测到 400 个 token；先各自 warm-up，再取 2 次中的最小值）：

```python
import sys, time, torch
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else torch.get_num_threads())
ns = {}; exec(open('ch05/05_kvcache.py', encoding='utf-8').read().split('# テスト')[0], ns)
GPT, gen_nc, gen_c = ns['GPT'], ns['generate_without_cache'], ns['generate_with_cache']
torch.manual_seed(0)
model = GPT(vocab_size=1000, max_context_len=1024, embed_dim=384, n_head=6, n_layer=6, ff_dim=1024)
start = torch.tensor([[42]])
def t_nc(n):
    t0 = time.perf_counter(); gen_nc(model, start, n); return time.perf_counter() - t0
def t_c(n):
    model.clear_cache(); t0 = time.perf_counter(); gen_c(model, start, n); return time.perf_counter() - t0
t_nc(5); t_c(5)
for n in [50, 100, 200, 400]:
    a = min(t_nc(n) for _ in range(2)); b = min(t_c(n) for _ in range(2))
    print(f"N={n:4d}: no-cache {a:6.2f}s  cache {b:5.2f}s  speedup {a/b:4.1f}x  | per-token cache {1000*b/n:5.2f} ms")
# （后半部分逐步测单步延迟，略）
```

```
threads: 8
N=  50: no-cache   1.04s  cache  0.75s  speedup  1.4x  | per-token cache 15.06 ms
N= 100: no-cache   2.30s  cache  1.28s  speedup  1.8x  | per-token cache 12.78 ms
N= 200: no-cache   7.26s  cache  2.64s  speedup  2.8x  | per-token cache 13.19 ms
N= 400: no-cache  18.83s  cache  4.59s  speedup  4.1x  | per-token cache 11.47 ms
(wall 138 s)

threads: 1
N=  50: no-cache   1.84s  cache  1.06s  speedup  1.7x
N= 100: no-cache   5.45s  cache  1.07s  speedup  5.1x
N= 200: no-cache  14.01s  cache  3.80s  speedup  3.7x
N= 400: no-cache  65.50s  cache  7.05s  speedup  9.3x
(wall 286 s)
```

为减少负载干扰，`h2_kv_cputime.py` 改成 1 线程，并用 `time.process_time()`（本进程消耗的 CPU 秒）计时。注意：Windows 上这个时钟的粒度是 15.6 ms，单步数值是 5 次平均，所以都是 3.125 ms 的倍数。

```python
import time, torch
torch.set_num_threads(1)
ns = {}; exec(open('ch05/05_kvcache.py', encoding='utf-8').read().split('# テスト')[0], ns)
GPT, gen_nc, gen_c = ns['GPT'], ns['generate_without_cache'], ns['generate_with_cache']
torch.manual_seed(0)
model = GPT(vocab_size=1000, max_context_len=1024, embed_dim=384, n_head=6, n_layer=6, ff_dim=1024)
start = torch.tensor([[42]]); clk = time.process_time
def t_nc(n):
    t0 = clk(); gen_nc(model, start, n); return clk() - t0
def t_c(n):
    model.clear_cache(); t0 = clk(); gen_c(model, start, n); return clk() - t0
t_nc(5); t_c(5)
for n in [50, 100, 200, 400]:
    a, b = t_nc(n), t_c(n)
    print(f"N={n:4d}: no-cache {a:6.2f} cpu-s  cache {b:5.2f} cpu-s  speedup {a/b:5.1f}x | cache per-token {1000*b/n:.2f} ms")
with torch.no_grad():
    for t in [1, 50, 100, 200, 400, 800]:
        ids = torch.randint(0, 1000, (1, t))
        t0 = clk()
        for _ in range(5): model(ids, use_cache=False)
        a = (clk() - t0) / 5
        model.clear_cache(); model(ids[:, :-1], use_cache=True) if t > 1 else None
        t0 = clk()
        for _ in range(5):
            saved = [(b.attn.k_cache, b.attn.v_cache, b.attn.cache_offset) for b in model.blocks]
            model(ids[:, -1:], use_cache=True)
            for blk, (k, v, o) in zip(model.blocks, saved):
                blk.attn.k_cache, blk.attn.v_cache, blk.attn.cache_offset = k, v, o
        b = (clk() - t0) / 5
        print(f"  t={t:4d}: full-recompute {1000*a:7.2f} ms   cached-decode {1000*b:5.2f} ms   ratio {a/b:5.1f}x")
```

```
N=  50: no-cache   1.41 cpu-s  cache  0.73 cpu-s  speedup   1.9x | cache per-token 14.69 ms
N= 100: no-cache   4.20 cpu-s  cache  1.36 cpu-s  speedup   3.1x | cache per-token 13.59 ms
N= 200: no-cache  13.14 cpu-s  cache  2.83 cpu-s  speedup   4.6x | cache per-token 14.14 ms
N= 400: no-cache  53.64 cpu-s  cache  7.78 cpu-s  speedup   6.9x | cache per-token 19.45 ms
per-step CPU ms at length t (mean of 5):
  t=   1: full-recompute   15.62 ms   cached-decode 21.88 ms   ratio   0.7x
  t=  50: full-recompute   46.88 ms   cached-decode 18.75 ms   ratio   2.5x
  t= 100: full-recompute   78.12 ms   cached-decode 18.75 ms   ratio   4.2x
  t= 200: full-recompute  137.50 ms   cached-decode 18.75 ms   ratio   7.3x
  t= 400: full-recompute  281.25 ms   cached-decode 21.88 ms   ratio  12.9x
  t= 800: full-recompute  837.50 ms   cached-decode 34.38 ms   ratio  24.4x
(wall 154 s)
```

`k_misc_torch.py` 里的 K6（纯 Python FLOP 估算，✅）：

```python
N = 6 * (4 * 384**2 + 3 * 384 * 1024) + 384 * 1000          # 每做一次 token 前向要用到的权重数
for n in [50, 200, 1000]:
    nocache = 2 * N * sum(range(1, n + 1)); cache = 2 * N * n
    print(f"n={n:5d}: no-cache {nocache/1e9:8.1f} GFLOP  cache {cache/1e9:6.2f} GFLOP  ratio {nocache/cache:6.1f}x")
```

```
n=   50: no-cache     28.1 GFLOP  cache   1.10 GFLOP  ratio   25.5x
n=  200: no-cache    442.2 GFLOP  cache   4.40 GFLOP  ratio  100.5x
n= 1000: no-cache  11011.8 GFLOP  cache  22.00 GFLOP  ratio  500.5x
```

单步 decode 的 profiler（`torch.profiler`，20 步，1 线程）：`aten::mm` 占 34%，其余是 `to/matmul/cat/transpose/mul/view` 等大量小算子；一个 1-token 的 `Linear(384,384)` 106 µs，一个小 `add` 11 µs。

| ID | 事实 | 观测 |
|---|---|---|
| H1 | 原脚本在本机测得 2.3–2.9× 的加速 | 3 次运行 |
| H2 | 加速比随生成长度增长（单步比值约与 t 成正比） | 单步：t=200 时 7.3×，t=800 时 24.4× |
| H3 | 理论 FLOP 比约 n/2，但实测远低于它：CPU、batch=1、小模型时每步固定开销占主导 | n=200：理论 100×，实测 2.8–4.6× |
| H4 | 有 cache 时单步延迟几乎是常数（随 t 只缓慢增长，那部分来自 attention 与 `torch.cat`） | 18.75 ms → 34 ms（t=800） |
| H5 | 原脚本先测无 cache（冷启动），后测有 cache，顺序本身对 cache 有利；做 warm-up 后结论不变，只是 N 小时比值更低 | 见上 |

### 3-I GQA / SDPA / storybot vs webbot（torch）—— `i_gqa_sdpa_torch.py`（36.3 s）❌

```python
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '.')
import webbot.model as wb, storybot.model as sb
torch.manual_seed(0)
count = lambda m: sum(p.numel() for p in m.parameters())
for kv in [12, 4, 1]:
    m = wb.GPT(50000, 1024, 768, 12, kv, 12, 2048); a = m.blocks[0].attn
    print(f"KV={kv:2d}: total {count(m):,} | W_k {tuple(a.W_k.weight.shape)} | non-embedding {count(m) - 2*50000*768:,}")
B, H, KV, T, D = 2, 12, 4, 7, 64
Q, K, V = torch.randn(B, H, T, D), torch.randn(B, KV, T, D), torch.randn(B, KV, T, D)
g = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
ri = F.scaled_dot_product_attention(Q, K.repeat_interleave(H // KV, 1), V.repeat_interleave(H // KV, 1), is_causal=True)
tl = F.scaled_dot_product_attention(Q, K.repeat(1, H // KV, 1, 1), V.repeat(1, H // KV, 1, 1), is_causal=True)
print("vs repeat_interleave:", (g - ri).abs().max().item(), "| vs repeat (tile):", (g - tl).abs().max().item())
print("query head h uses kv head h // (H/KV):", [h // (H // KV) for h in range(H)])
Kf, Vf = torch.randn(B, H, T, D), torch.randn(B, H, T, D)
s = (Q @ Kf.transpose(-2, -1)) / D ** 0.5
s = s.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
manual = torch.softmax(s, -1) @ Vf
print("max diff:", (F.scaled_dot_product_attention(Q, Kf, Vf, is_causal=True) - manual).abs().max().item())
q1, k5 = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 5, 8)
eye = torch.eye(5).view(1, 1, 5, 5)                 # value 取成 key 位置的 one-hot，输出就等于注意力权重
print("is_causal=True :", F.scaled_dot_product_attention(q1, k5, eye, is_causal=True)[0, 0, 0].tolist())
print("is_causal=False:", [round(v, 3) for v in F.scaled_dot_product_attention(q1, k5, eye, is_causal=False)[0, 0, 0].tolist()])
m = wb.GPT(1000, 64, 128, 4, 2, 2, 344).eval(); p = torch.randint(0, 1000, (1, 6))
with torch.no_grad():
    ids = p.clone()
    for _ in range(20): ids = torch.cat([ids, m(ids)[:, -1:].argmax(-1)], 1)
    m.clear_cache(); ids2 = p.clone(); nxt = p
    for _ in range(20):
        nxt = m(nxt, use_cache=True)[:, -1:].argmax(-1); ids2 = torch.cat([ids2, nxt], 1)
print("equal:", torch.equal(ids, ids2))
for name, mod, args in [("storybot", sb, (1000, 16, 64, 4, 2, 172)), ("webbot", wb, (1000, 16, 64, 4, 2, 2, 172))]:
    m = mod.GPT(*args).eval(); m.clear_cache(); nxt = torch.randint(0, 1000, (1, 4))
    with torch.no_grad():
        for step in range(30):
            a = m.blocks[0].attn
            eff = min(a.cache_offset, 16 - nxt.shape[1])   # RoPE.forward 实际会用的 offset
            nxt = m(nxt, use_cache=True)[:, -1:].argmax(-1)
            if step in (0, 11, 12, 13, 29):
                print(f"  {name} step {step:2d}: rope offset used={eff:2d}  k_cache len={a.k_cache.shape[2]:2d}  cache_offset={a.cache_offset}")
for kv, nm in [(12, 'MHA'), (4, 'GQA'), (1, 'MQA')]:
    b16 = 2 * 12 * kv * 64 * 2
    print(f"{nm} KV={kv:2d}: {b16:,} B/token bf16 -> {b16 * 1024 / 2**20:.1f} MiB @1024 tokens")
s_keys = set(sb.GPT(1000, 16, 64, 4, 1, 172).state_dict()); w_keys = set(wb.GPT(1000, 16, 64, 4, 4, 1, 172).state_dict())
print("only storybot:", sorted(s_keys - w_keys)[:3], "| only webbot:", sorted(w_keys - s_keys)[:3])
```

```
--- I1 webbot params: GQA(KV=4) vs MHA(KV=12) vs MQA(KV=1), V=50000 C=1024 E=768 H=12 L=12 ff=2048
KV=12: total 161,753,856 | W_k (768, 768) | non-embedding 84,953,856
KV= 4: total 152,316,672 | W_k (256, 768) | non-embedding 75,516,672
KV= 1: total 148,777,728 | W_k (64, 768) | non-embedding 71,977,728
--- I2 enable_gqa=True == repeat_interleave (NOT repeat/tile)
vs repeat_interleave: 0.0 | vs repeat (tile): 5.02939510345459
query head h uses kv head h // (H/KV): [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
--- I3 SDPA(is_causal=True) == manual tril-mask softmax (storybot/ch05 path)
max diff: 3.5762786865234375e-07
--- I4 pitfall: is_causal=True with Lq=1, Lk=5 is TOP-LEFT aligned -> new token sees only key 0
is_causal=True : [1.0, 0.0, 0.0, 0.0, 0.0]
is_causal=False: [0.138, 0.093, 0.326, 0.056, 0.387]
--- I5 webbot: cached generation == full recompute (small config, KV=2)
equal: True
--- I6 beyond max_context_len (16): storybot cache grows, webbot cache slides; RoPE offset is clamped
  storybot step  0: rope offset used= 0  k_cache len= 4  cache_offset=4
  storybot step 11: rope offset used=14  k_cache len=15  cache_offset=15
  storybot step 12: rope offset used=15  k_cache len=16  cache_offset=16
  storybot step 13: rope offset used=15  k_cache len=17  cache_offset=17
  storybot step 29: rope offset used=15  k_cache len=33  cache_offset=33
  webbot step  0: rope offset used= 0  k_cache len= 4  cache_offset=4
  webbot step 11: rope offset used=14  k_cache len=15  cache_offset=15
  webbot step 12: rope offset used=15  k_cache len=16  cache_offset=16
  webbot step 13: rope offset used=15  k_cache len=16  cache_offset=16
  webbot step 29: rope offset used=15  k_cache len=16  cache_offset=16
--- I7 KV-cache bytes per token (all layers, K+V) for the webbot config
MHA KV=12: 36,864 B/token bf16 -> 36.0 MiB @1024 tokens
GQA KV= 4: 12,288 B/token bf16 -> 12.0 MiB @1024 tokens
MQA KV= 1: 3,072 B/token bf16 -> 3.0 MiB @1024 tokens
--- I8 storybot ('ffn') vs webbot ('mlp') state_dict keys are not interchangeable
only storybot: ['blocks.0.ffn.O.weight', 'blocks.0.ffn.V.weight', 'blocks.0.ffn.W.weight'] | only webbot: ['blocks.0.mlp.O.weight', 'blocks.0.mlp.V.weight', 'blocks.0.mlp.W.weight']
```

补充：prompt 长度超过 max_ctx 时的 prefill（max_ctx=16）

```python
import sys, torch
sys.path.insert(0, '.')
import webbot.model as wb, storybot.model as sb
torch.manual_seed(0)
for name, m in [('webbot', wb.GPT(100, 16, 32, 4, 2, 1, 85)), ('storybot', sb.GPT(100, 16, 32, 4, 1, 85))]:
    for T in [16, 17, 18]:
        m.clear_cache()
        try:
            with torch.no_grad(): out = m(torch.randint(0, 100, (1, T)), use_cache=True)
            print(f'{name} prompt T={T} (max 16), use_cache=True: ran, out {tuple(out.shape)}, k_cache len {m.blocks[0].attn.k_cache.shape[2]}')
        except Exception as e:
            print(f'{name} prompt T={T} (max 16), use_cache=True: {type(e).__name__}: {str(e)[:70]}')
```

```
webbot prompt T=16 (max 16), use_cache=True: ran, out (1, 16, 100), k_cache len 16
webbot prompt T=17 (max 16), use_cache=True: ran, out (1, 17, 100), k_cache len 16
webbot prompt T=18 (max 16), use_cache=True: RuntimeError: The size of tensor a (18) must match the size of tensor b (2) at non-s
storybot prompt T=16 (max 16), use_cache=True: ran, out (1, 16, 100), k_cache len 16
storybot prompt T=17 (max 16), use_cache=True: ran, out (1, 17, 100), k_cache len 17
storybot prompt T=18 (max 16), use_cache=True: RuntimeError: The size of tensor a (18) must match the size of tensor b (2) at non-s
```

| ID | 事实 | 观测 |
|---|---|---|
| I1 | webbot 参数量：GQA(4) 152,316,672；MHA 161,753,856；MQA 148,777,728。差值 = 12 层 × 2 × 768 × (768−256) = 9,437,184 | ✓ |
| I2 | `enable_gqa=True` 等价于 `repeat_interleave`（第 h 个 query head 用第 h//3 个 KV head），**不等价**于 `repeat`/tile | 0.0 vs 5.03 |
| I3 | SDPA(is_causal) 与手写的 mask+softmax 一致 | 3.6e-7 |
| I4 | 陷阱：Lq≠Lk 时 `is_causal=True` 按**左上角**对齐，1 个新 token 只能看到第 0 个 key；所以 decode 时必须用 `is_causal=False`（webbot 的做法是对的） | [1,0,0,0,0] |
| I5 | webbot 有无 cache 的输出一致 | True |
| I6 | 超过 max_ctx 后：storybot 的 cache 一直变长；webbot 的 cache 保持 16，像滑动窗口。两者的 RoPE offset 都被夹在 15，**之后每个新 token 都编码在同一个位置** | 见输出 |
| I7 | KV cache 显存：bf16 下 MHA 36 KiB/token，GQA 12 KiB/token（省 3×），MQA 3 KiB/token | ✓ |
| I8 | storybot 用 `ffn`、webbot 用 `mlp`，state_dict 不能互相加载 | ✓ |
| I9 | prompt 长度 = max+1 时静默运行（位置全错），≥ max+2 时报错 | 见补充 |

### 3-J 真实训练好的 StoryBot checkpoint（torch）—— `j_storybot_trained.py`（101.5 s）❌

**J0 下载文件的 pickle 静态检查**（不会执行文件里的代码）

```python
import pickletools, collections, zipfile, sys
p = sys.argv[1]; ops = collections.Counter()
with open(p, 'rb') as f:
    for op, arg, pos in pickletools.genops(f): ops[op.name] += 1
print('opcodes:', dict(ops))
dangerous = {'GLOBAL','STACK_GLOBAL','REDUCE','INST','OBJ','NEWOBJ','NEWOBJ_EX','BUILD','EXT1','EXT2','EXT4'}
print('dangerous opcodes present:', sorted(set(ops) & dangerous))
z = zipfile.ZipFile(sys.argv[2]); names = z.namelist(); print('pt zip entries:', len(names), names[:3])
```

```
opcodes: {'PROTO': 1, 'FRAME': 2, 'EMPTY_DICT': 1, 'MEMOIZE': 9744, 'MARK': 10, 'BININT1': 3339, 'TUPLE2': 9743, 'BININT2': 25890, 'SETITEMS': 10, 'STOP': 1}
dangerous opcodes present: []
pt zip entries: 45 ['model_pretrain/data.pkl', 'model_pretrain/byteorder', 'model_pretrain/data/0']
```

（merge_rules 里有 9743 条合并规则，256 + 9743 + 1 个 end token = 10000。）

**J1–J5**（脚本原文）

```python
import sys, time, math, json, torch, torch.nn.functional as F
REPO = r'C:\Users\chenf\Projects\_refs\dlfs6'
DL = r'<scratch>\ch05\dl_storybot'
sys.path.insert(0, REPO)
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import generate
ck = torch.load(DL + r'\model_pretrain.pt', map_location='cpu')
print({k: v for k, v in ck.items() if k != 'model_state_dict'})
print("ffn keys:", sum('ffn' in k for k in ck['model_state_dict']), "| mlp keys:", sum('mlp' in k for k in ck['model_state_dict']))
try:
    GPT.load_from(DL + r'\model_pretrain.pt')
except RuntimeError as e:
    print("GPT.load_from FAILS:", str(e).splitlines()[1][:80], '...')
def load_renamed(cls, path):
    ck = torch.load(path, map_location='cpu')
    cfg = {k: ck[k] for k in ['vocab_size', 'max_context_len', 'embed_dim', 'n_head', 'n_layer', 'ff_dim', 'theta']}
    m = cls(**cfg)
    m.load_state_dict({k.replace('.mlp.', '.ffn.'): v for k, v in ck['model_state_dict'].items()})
    return m.eval()
model = load_renamed(GPT, DL + r'\model_pretrain.pt')
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
tok = BPETokenizer.load_from(DL + r'\merge_rules.pkl')
print("vocab:", tok.vocab_size, "end_token_id:", tok.end_token_id)
ex = json.load(open(REPO + r'\storybot\tiny_stories_dpo.json', encoding='utf-8'))[0]
ids = torch.tensor([tok.encode(ex['prompt'] + ex['chosen'])])
with torch.no_grad(): lg = model(ids)
loss = F.cross_entropy(lg[0, :-1], ids[0, 1:]).item()
print(f"tokens={ids.shape[1]}  loss={loss:.3f}  ppl={math.exp(loss):.1f}  | ln(10000)={math.log(10000):.3f}")
def gen_nocache(prompt_ids, n):
    ids = prompt_ids
    with torch.no_grad():
        for _ in range(n):
            nxt = model(ids[:, -model.max_context_len:])[:, -1:].argmax(-1); ids = torch.cat([ids, nxt], 1)
    return ids
def gen_cache(m, prompt_ids, n):
    m.clear_cache(); ids = prompt_ids; nxt = prompt_ids
    with torch.no_grad():
        for _ in range(n):
            nxt = m(nxt, use_cache=True)[:, -1:].argmax(-1); ids = torch.cat([ids, nxt], 1)
    return ids
prompt = "Once upon a time, there was a little girl named Lily. She"
p = torch.tensor([tok.encode(prompt)])
t0 = time.perf_counter(); a = gen_nocache(p, 60); t1 = time.perf_counter(); b = gen_cache(model, p, 60); t2 = time.perf_counter()
print("identical:", torch.equal(a, b), f"| no-cache {t1-t0:.2f}s  cache {t2-t1:.2f}s")
print("TEXT:", repr(tok.decode(b[0].tolist())))
print("utils.generate(T=0) same text:", generate(model, tok, prompt, 60, temperature=0) == tok.decode(b[0].tolist()))
src = open(REPO + r'\storybot\model.py', encoding='utf-8').read()
line = 'if not use_cache or (use_cache and is_first_call):'; assert line in src
nsb = {'__name__': 'buggy'}; exec(src.replace(line, 'if not use_cache:'), nsb)
buggy = load_renamed(nsb['GPT'], DL + r'\model_pretrain.pt')
c = gen_cache(buggy, p, 60)
print("buggy == correct:", torch.equal(b, c)); print("BUGGY TEXT:", repr(tok.decode(c[0].tolist())))
n = 300 - p.shape[1]
a = gen_nocache(p, n); b = gen_cache(model, p, n)
diff = (a != b).nonzero(); first = diff[0, 1].item() if len(diff) else None
print("total length:", a.shape[1], "| first differing position:", first)
```

```
--- J1 load checkpoint (weights_only default) + config
{'vocab_size': 10000, 'max_context_len': 256, 'embed_dim': 512, 'n_head': 16, 'n_layer': 4, 'ff_dim': 1344, 'theta': 10000}
ffn keys: 0 | mlp keys: 12
GPT.load_from FAILS: 	Missing key(s) in state_dict: "blocks.0.ffn.W.weight", "blocks.0.ffn.V.weight", ...
params: 22,696,448
vocab: 10000 end_token_id: 9999
--- J2 loss on one held-in-repo story (tiny_stories_dpo.json[0], chosen) vs ln(V)
tokens=69  loss=1.424  ppl=4.2  | ln(10000)=9.210
--- J3 greedy 60 tokens: KV cache vs full recompute (prompt has 14 tokens)
identical: True | no-cache 4.97s  cache 2.06s
TEXT: "Once upon a time, there was a little girl named Lily. She had a big, soft pillow that she loved very much. One day, Lily's mom told her that they were going to have a party. Lily was very excited and wanted to help her mom.\nAt the party, Lily saw her friend, Tom. Tom was sad because he did not have"
utils.generate(T=0) same text: True
--- J4 the old prefill bug on the trained model (mask skipped in prefill)
buggy == correct: False
BUGGY TEXT: 'Once upon a time, there was a little girl named Lily. She had a big, soft pillow that she loved very much. Every night, she would hug her pillow tight and close her eyes.\nOne day, Lily\'s pillow was gone. She looked everywhere for it, but she could not find it. She was very sad. She asked her mom, "'
--- J5 beyond max_context_len=256: cache (RoPE offset clamped) vs sliding-window recompute
total length: 300 | first differing position: 262
```

补充：三个已发布的 StoryBot checkpoint 都用 `mlp` key。这几个文件已由其它 agent 下载到 repo 的 `storybot/`；其中 `model_pretrain.pt` 的 sha256 与我下载到 scratch 的副本以及 HF 的 `x-linked-etag` 一致，都是 `b32f3f0e6c45a7a00eb6d6fea8d8ec70c12781e6154ccc4b6bb7e9f5eff7b027`。

```python
import os, torch
REPO = r'C:\Users\chenf\Projects\_refs\dlfs6'
for f in ['storybot/model_iter_500.pt', 'storybot/model_iter_5000.pt', 'storybot/model_pretrain.pt']:
    ck = torch.load(os.path.join(REPO, *f.split('/')), map_location='cpu')
    sd = ck['model_state_dict']
    print(f, '| mlp keys:', sum('.mlp.' in k for k in sd), '| ffn keys:', sum('.ffn.' in k for k in sd),
          '| cfg:', {k: v for k, v in ck.items() if k != 'model_state_dict'})
```

```
storybot/model_iter_500.pt | mlp keys: 12 | ffn keys: 0 | cfg: {'vocab_size': 10000, 'max_context_len': 256, 'embed_dim': 512, 'n_head': 16, 'n_layer': 4, 'ff_dim': 1344, 'theta': 10000}
storybot/model_iter_5000.pt | mlp keys: 12 | ffn keys: 0 | cfg: {'vocab_size': 10000, 'max_context_len': 256, 'embed_dim': 512, 'n_head': 16, 'n_layer': 4, 'ff_dim': 1344, 'theta': 10000}
storybot/model_pretrain.pt | mlp keys: 12 | ffn keys: 0 | cfg: {'vocab_size': 10000, 'max_context_len': 256, 'embed_dim': 512, 'n_head': 16, 'n_layer': 4, 'ff_dim': 1344, 'theta': 10000}
```

补充（同一个训练好的模型）：整个模型对平移不变，并核对 prompt 的 token id

```python
import sys, torch
sys.path.insert(0, r'C:\Users\chenf\Projects\_refs\dlfs6')
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
DL = r'<scratch>\ch05\dl_storybot'
ck = torch.load(DL + r'\model_pretrain.pt', map_location='cpu')
m = GPT(**{k: ck[k] for k in ['vocab_size','max_context_len','embed_dim','n_head','n_layer','ff_dim','theta']}).eval()
m.load_state_dict({k.replace('.mlp.', '.ffn.'): v for k, v in ck['model_state_dict'].items()})
tok = BPETokenizer.load_from(DL + r'\merge_rules.pkl')
ids = torch.tensor([tok.encode('Once upon a time, there was a little girl named Lily. She')])
print('prompt ids:', ids[0].tolist())
with torch.no_grad():
    m.clear_cache(); l0 = m(ids, use_cache=True)
    for off in [100, 200]:
        m.clear_cache()
        for b in m.blocks: b.attn.cache_offset = off       # 让 prompt 从绝对位置 off 开始
        l1 = m(ids, use_cache=True)
        print(f'offset {off}: max |logits diff| = {(l0 - l1).abs().max().item():.2e}  (logit scale {l0.abs().max().item():.1f}), same argmax: {torch.equal(l0.argmax(-1), l1.argmax(-1))}')
print('next token:', repr(tok.decode([l0[0, -1].argmax().item()])))
```

```
prompt ids: [429, 438, 258, 397, 44, 400, 282, 258, 389, 476, 401, 405, 46, 338]
offset 100: max |logits diff| = 5.72e-06  (logit scale 21.0), same argmax: True
offset 200: max |logits diff| = 4.77e-06  (logit scale 21.0), same argmax: True
next token: ' had'
```

| ID | 事实 | 观测 |
|---|---|---|
| J1 | **HF 上发布的 storybot checkpoint 用当前代码加载会失败**：checkpoint 的 key 是 `mlp`，代码已经改成了 `ffn`。把 key 改名后可以正常加载 | 0 个 ffn key、12 个 mlp key；`load_from` 抛 RuntimeError |
| J1' | StoryBot 的配置与参数量 | V=10000、ctx=256、E=512、H=16、L=4、ff=1344；22,696,448 |
| J2 | 训练好的模型在一个故事上的 loss 是 1.424（ppl 4.2），初始化时约为 ln V = 9.21 | ✓ |
| J3 | 真实模型上，有无 cache 的 greedy 生成完全一致，而且与 `storybot.utils.generate(temperature=0)` 的文本相同 | True / True |
| J4 | 旧的 prefill bug 会改变生成的故事，**但生成的文本依然通顺**，肉眼看不出问题 | 见两段文本 |
| J5 | 超过 256 之后，cache 版（位置被夹）与"滑动窗口重算"版开始分叉（这里在第 262 个位置） | 262 |
| J6 | RoPE-only 模型对整体平移不变：prompt 从位置 100 或 200 开始，logits 只差 ~5e-6 | ✓ |
| J7 | "Once upon a time, there was a little girl named Lily. She" 的 token id（14 个），以及下一个 token `' had'` | 见输出 |

### 3-K 杂项（torch）—— `k_misc_torch.py`（17.3 s）❌

```python
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import storybot.model as sb, webbot.model as wb
torch.manual_seed(0)
m = sb.GPT(1000, 64, 128, 4, 2, 344).eval(); ids = torch.randint(0, 1000, (1, 8))
with torch.no_grad():
    m.clear_cache(); m(ids, use_cache=True)
    a = m.blocks[0].attn; h = m.blocks[0].norm1(m.embed(ids))
    K = a.W_k(h).view(1, 8, 4, 32).transpose(1, 2); V = a.W_v(h).view(1, 8, 4, 32).transpose(1, 2)
print("k_cache == rope(K):", torch.allclose(a.k_cache, a.rope(K), atol=1e-6),
      "| k_cache == K (unrotated):", torch.allclose(a.k_cache, K, atol=1e-6), "| v_cache == V:", torch.allclose(a.v_cache, V, atol=1e-6))
print("layer caches are distinct objects:", m.blocks[0].attn.k_cache is not m.blocks[1].attn.k_cache)
with torch.no_grad():
    m.clear_cache(); one = m(ids, use_cache=True)[:, -1]
    m.clear_cache(); m(ids[:, :4], use_cache=True); two = m(ids[:, 4:], use_cache=True)[:, -1]
    full = m(ids)[:, -1]
    m.clear_cache(); m(ids[:, :4], use_cache=True); two_all = m(ids[:, 4:], use_cache=True); ref_all = m(ids)[:, 4:]
print("last-token logits: one-shot prefill vs full:", (one - full).abs().max().item(), "| two chunks vs full:", (two - full).abs().max().item())
print("chunk-2 positions 4..6 (should see only the past) max diff:", (two_all[:, :3] - ref_all[:, :3]).abs().max().item())
w = wb.GPT(1000, 64, 768, 12, 4, 1, 2048).eval()
with torch.no_grad(): w.clear_cache(); w(torch.randint(0, 1000, (1, 10)), use_cache=True)
print("webbot k_cache shape (B, KV, T, D):", tuple(w.blocks[0].attn.k_cache.shape))
x = torch.randn(2, 5, 64)
with torch.autocast('cpu', dtype=torch.bfloat16):
    lin = nn.Linear(64, 64)(x); rn = nn.RMSNorm(64)(lin); ln = nn.LayerNorm(64)(lin)
    sd = F.scaled_dot_product_attention(lin[:, None], lin[:, None], lin[:, None], is_causal=True)
print("linear:", lin.dtype, "| RMSNorm(bf16 in):", rn.dtype, "| LayerNorm(bf16 in):", ln.dtype, "| sdpa:", sd.dtype)
ns = {}; exec(open('ch05/05_kvcache.py', encoding='utf-8').read().split('# テスト')[0], ns)
g1 = ns['GPT'](1000, 256, 384, 6, 6, 1024); g2 = ns['GPT'](1000, 256, 384, 6, 6, 1024)
print("two unseeded inits equal:", torch.equal(g1.embed.weight, g2.embed.weight))
# （K6 的 FLOP 估算见 §3-H）
```

```
UserWarning: Mismatch dtype between input and weight: input dtype = struct c10::BFloat16, weight dtype = float, Cannot dispatch to fused implementation. (… layer_norm.cpp:347)
--- K1 the cache holds POST-RoPE keys (and raw values), one cache per layer
k_cache == rope(K): True | k_cache == K (unrotated): False | v_cache == V: True
layer caches are distinct objects: True
--- K2 limitation: a 2nd call with C>1 gets NO causal mask (chunked prefill breaks)
last-token logits: one-shot prefill vs full: 0.0 | two chunks vs full: 0.09695491194725037
chunk-2 positions 4..6 (should see only the past) max diff: 0.39722734689712524
--- K3 webbot GQA cache stores KV heads only
webbot k_cache shape (B, KV, T, D): (1, 4, 10, 64)
--- K4 dtypes under CPU autocast(bf16)
linear: torch.bfloat16 | RMSNorm(bf16 in): torch.bfloat16 | LayerNorm(bf16 in): torch.bfloat16 | sdpa: torch.bfloat16
--- K5 '# 固定シード' in 05_kvcache.py fixes the START TOKEN, not the RNG: weights differ per run
two unseeded inits equal: False
```

| ID | 事实 | 观测 |
|---|---|---|
| K1 | cache 里存的是**旋转之后的** K，以及原样的 V；每层各有一份 | True/False/True |
| K2 | 分块 prefill（第二次调用时 C>1）没有 mask，结果是错的 | 0.097 / 0.397 |
| K3 | webbot 的 cache 只存 KV 个 head，形状 (1,4,T,64) | ✓ |
| K4 | CPU autocast(bf16) 下，`nn.RMSNorm` 收到 bf16 输入和 fp32 权重时会警告"无法走 fused 实现"，输出是 bf16 | ✓（GPU 上的行为待验证） |
| K5 | `# 固定シード` 并没有固定随机数：两次不设 seed 的初始化权重不同，所以每次运行打印的 token id 都会变 | False |

### 3-L 计算预算（纯 Python）—— `l_budget.py` ✅

峰值算力取规格表上的 dense bf16 值；MFU=35% 是**假设**，不是实测。

```python
GPUS = {"RTX3060Ti(8GB)": 32.4, "L4(24GB)": 121, "RTX4090(24GB)": 165, "A100(80GB)": 312, "H100-SXM(80GB)": 989}
MFU = 0.35
def run(name, N, L, T, E, steps, tokens_per_step):
    D = steps * tokens_per_step; dense = 6 * N * D; attn = 6 * L * T * E * D; C = dense + attn
    print(f"{name}: N={N/1e6:.1f}M  D={D/1e9:.3f}B tok  6ND={dense:.2e}  +attn={attn:.2e}  total={C:.2e} FLOP  | D/N={D/N:.0f} tok/param")
    print("   static mem (fp32 w + grad + AdamW m,v = 16 B/param): %.2f GB" % (16 * N / 1e9))
    for g, peak in GPUS.items():
        print(f"   {g:16s}: {C / (peak * 1e12 * MFU) / 3600:7.2f} h @ {int(MFU*100)}% MFU")
run("storybot ch06", 22_696_448, 4, 256, 512, 40_000, 32 * 256)
run("webbot   ch09", 152_316_672, 12, 1024, 768, 100_000, 32 * 4 * 1024)
run("ch05 ablation (x2 variants, 5k steps each)", 2 * 22_696_448, 4, 256, 512, 5_000, 32 * 256)
for name, B, T, V in [("storybot", 32, 256, 10_000), ("webbot", 32, 1024, 50_000)]:
    n = B * T * V
    print(f"{name} logits per micro-batch: {n/1e6:.0f}M elements = {2*n/1e9:.2f} GB bf16 / {4*n/1e9:.2f} GB fp32")
```

```
storybot ch06: N=22.7M  D=0.328B tok  6ND=4.46e+16  +attn=1.03e+15  total=4.57e+16 FLOP  | D/N=14 tok/param
   static mem (fp32 w + grad + AdamW m,v = 16 B/param): 0.36 GB
   RTX3060Ti(8GB)  :    1.12 h @ 35% MFU
   L4(24GB)        :    0.30 h @ 35% MFU
   RTX4090(24GB)   :    0.22 h @ 35% MFU
   A100(80GB)      :    0.12 h @ 35% MFU
   H100-SXM(80GB)  :    0.04 h @ 35% MFU
webbot   ch09: N=152.3M  D=13.107B tok  6ND=1.20e+19  +attn=7.42e+17  total=1.27e+19 FLOP  | D/N=86 tok/param
   static mem (fp32 w + grad + AdamW m,v = 16 B/param): 2.44 GB
   RTX3060Ti(8GB)  :  311.60 h @ 35% MFU
   L4(24GB)        :   83.44 h @ 35% MFU
   RTX4090(24GB)   :   61.19 h @ 35% MFU
   A100(80GB)      :   32.36 h @ 35% MFU
   H100-SXM(80GB)  :   10.21 h @ 35% MFU
ch05 ablation (x2 variants, 5k steps each): N=45.4M  D=0.041B tok  6ND=1.12e+16  +attn=1.29e+14  total=1.13e+16 FLOP  | D/N=1 tok/param
   …（RTX3060Ti 0.28 h；L4 0.07 h；A100 0.03 h）
storybot logits per micro-batch: 82M elements = 0.16 GB bf16 / 0.33 GB fp32
webbot logits per micro-batch: 1638M elements = 3.28 GB bf16 / 6.55 GB fp32
```

（ablation 那一行的 D/N 没有意义，因为 N 是两个模型加起来的。每个模型实际是 41M tokens / 22.7M ≈ 1.8 tokens/param。）

### 3-M/N/O NumPy-only 的现代 GPT + KV Cache（教程端到端示例的核心）✅

`m_numpy_minigpt.py`：完全对应 storybot/model.py 的 NumPy 实现（RMSNorm 的 eps 也取 finfo(fp32).eps），可以在 Pyodide 中运行。

```python
import numpy as np
EPS = float(np.finfo(np.float32).eps)          # nn.RMSNorm(eps=None) 用的就是 finfo(dtype).eps

def init_params(V=64, C=32, E=32, H=4, L=2, F=None, theta=10000.0, seed=0, std=0.02):
    F = F or int(E * 8 / 3)
    r = np.random.default_rng(seed); n = lambda *s: r.normal(0, std, s)
    D = E // H
    inv = 1.0 / theta ** (np.arange(D // 2) * 2.0 / D)
    ang = np.arange(C)[:, None] * inv
    blocks = [dict(g1=np.ones(E), Wq=n(E, E), Wk=n(E, E), Wv=n(E, E), Wo=n(E, E),
                   g2=np.ones(E), W=n(E, F), Vg=n(E, F), O=n(F, E)) for _ in range(L)]
    return dict(H=H, C=C, emb=n(V, E), blocks=blocks, g=np.ones(E), unemb=n(E, V),
                cos=np.cos(ang), sin=np.sin(ang))

def rmsnorm(x, g): return g * x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
def silu(x): return x / (1.0 + np.exp(-x))

def rope(x, P, off):                            # x: (H, T, D)，位置为 off..off+T-1
    T = x.shape[1]; c, s = P['cos'][off:off + T], P['sin'][off:off + T]
    e, o = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x); out[..., 0::2] = e * c - o * s; out[..., 1::2] = e * s + o * c
    return out

def attention(x, b, P, cache, causal_prefill_mask=True):
    T, E = x.shape; H = P['H']; D = E // H
    split = lambda m: m.reshape(T, H, D).transpose(1, 0, 2)
    q, k, v = split(x @ b['Wq']), split(x @ b['Wk']), split(x @ b['Wv'])
    off = 0 if cache is None else cache['off']
    q, k = rope(q, P, off), rope(k, P, off)
    first = cache is None or cache['k'] is None
    if cache is not None:
        cache['k'] = k if first else np.concatenate([cache['k'], k], 1)
        cache['v'] = v if first else np.concatenate([cache['v'], v], 1)
        cache['off'] += T; k, v = cache['k'], cache['v']
    s = q @ k.transpose(0, 2, 1) / np.sqrt(D)
    if first and (cache is None or causal_prefill_mask):                  # 训练 / prefill
        s = np.where(np.tril(np.ones((T, T))) == 0, -np.inf, s)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w @ v).transpose(1, 0, 2).reshape(T, E) @ b['Wo']

def forward(ids, P, caches=None, **kw):
    x = P['emb'][ids]
    for i, b in enumerate(P['blocks']):
        x = x + attention(rmsnorm(x, b['g1']), b, P, None if caches is None else caches[i], **kw)
        h = rmsnorm(x, b['g2'])
        x = x + (silu(h @ b['W']) * (h @ b['Vg'])) @ b['O']
    return rmsnorm(x, P['g']) @ P['unemb']

def generate(P, prompt, n, use_cache, **kw):
    ids = list(prompt)
    caches = [dict(k=None, v=None, off=0) for _ in P['blocks']] if use_cache else None
    feed = ids
    for _ in range(n):
        logits = forward(np.array(feed), P, caches, **kw) if use_cache else forward(np.array(ids), P)
        nxt = int(logits[-1].argmax()); ids.append(nxt); feed = [nxt]
    return ids
```

自测（`__main__`）输出：

```
[std=0.02] no-cache: [63, 14, 37, 29, 17, 20, 28, 19, 39, 17, 20, 28, 19, 39, 7, 7, 7, 7, 7, 7]
[std=0.02] cache == no-cache: True | 24.9 ms vs 24.9 ms
[std=0.02] buggy prefill (no mask): mismatches 0/20 | 1-token prompt buggy == correct: True
[std=0.2] no-cache: [50, 31, 31, 31, 19, 20, 20, 31, 37, 19, 24, 62, 16, 50, 19, 20, 20, 20, 20, 12]
[std=0.2] cache == no-cache: True | 24.8 ms vs 36.7 ms
[std=0.2] buggy prefill (no mask): mismatches 17/20 | 1-token prompt buggy == correct: True
```

bug 能不能看出来，取决于初始化尺度（同样是 6-token prompt）：

```
std=0.02: max|last-token logits diff| = 1.83e-02  token mismatches 0/20
std=0.1: max|last-token logits diff| = 3.43e-01  token mismatches 9/20
std=0.3: max|last-token logits diff| = 1.69e+00  token mismatches 19/20
```

（用 std=0.02 的小随机模型演示这个 bug 时，应该展示 logits 的差，而不是 token 是否一致；也可以直接用训练好的模型，见 J4。）

`n_crosscheck_torch.py`：把 `storybot.model.GPT(64,32,32,4,2,85)` 的权重拷进 NumPy 版后对比 ❌（需要 torch）

```
max |numpy - torch| logits: 1.0165919683835511e-07
numpy greedy (cache) == torch greedy (no cache): True
```

`o_numpy_storybot.py`：把**训练好的 StoryBot 权重**放进 NumPy 版推理（torch 只用来读 checkpoint；如果预先导出成 npz，就完全可以在浏览器里跑）

```python
ck = torch.load(os.path.join(DL, 'model_pretrain.pt'), map_location='cpu')
sd = {k: v.float().numpy() for k, v in ck['model_state_dict'].items()}
V, C, E, H, L, F = (ck[k] for k in ['vocab_size', 'max_context_len', 'embed_dim', 'n_head', 'n_layer', 'ff_dim'])
P = M.init_params(V, C, E, H, L, F)
P['emb'], P['unemb'], P['g'] = sd['embed.weight'], sd['unembed.weight'].T, sd['norm.weight']
for i, b in enumerate(P['blocks']):
    g = lambda n: sd[f'blocks.{i}.{n}']
    b.update(g1=g('norm1.weight'), g2=g('norm2.weight'), Wq=g('attn.W_q.weight').T, Wk=g('attn.W_k.weight').T,
             Wv=g('attn.W_v.weight').T, Wo=g('attn.W_o.weight').T, W=g('mlp.W.weight').T,
             Vg=g('mlp.V.weight').T, O=g('mlp.O.weight').T)
tok = BPETokenizer.load_from(os.path.join(DL, 'merge_rules.pkl'))
prompt = tok.encode("Once upon a time, there was a little girl named Lily. She")
for use_cache in [True, False]:
    t0 = time.perf_counter(); out = M.generate(P, prompt, 60, use_cache); dt = time.perf_counter() - t0
    print(f"use_cache={use_cache}: {dt:.2f}s  ({1e3*dt/60:.1f} ms/token)")
print("TEXT:", repr(tok.decode(out)))
```

```
use_cache=True: 0.83s  (13.8 ms/token)
use_cache=False: 8.51s  (141.9 ms/token)
TEXT: "Once upon a time, there was a little girl named Lily. She had a big, soft pillow that she loved very much. One day, Lily's mom told her that they were going to have a party. Lily was very excited and wanted to help her mom.\nAt the party, Lily saw her friend, Tom. Tom was sad because he did not have"
weights as float16: 43.4 MiB
```

→ NumPy 版逐字复现了 torch 版的 greedy 故事（与 J3 相同）。在同一台机器上，NumPy 版的 KV cache 加速达到 10×（60 个 token，14-token prompt），比 torch 版（2.4×）明显得多，因为 NumPy 每个算子的固定开销更小。权重转成 fp16 后是 43.4 MiB，**可以在浏览器里演示**（前提是先确认权重的再分发许可，或者让页面在运行时直接从 HF 下载）。

### 3-Z 事实速查总表

| # | 断言 | 怎么查 | 观测 | 浏览器 |
|---|---|---|---|---|
| 1 | RoPE 保持范数，位置 0 时为恒等 | A1 | 7.3153 = 7.3153；True | ✅ |
| 2 | RoPE 下 q·k 只依赖 m−n（而且是有符号的） | A2 | 偏差 5.3e-14；S[5,2]≠S[2,5] | ✅ |
| 3 | RoPE = 复数乘 e^{imθ} = 分块旋转矩阵 | A3/A4 | True | ✅ |
| 4 | d=64、θ=1e4 时波长范围 2π 到约 47k tokens | A5 | 6.28 / 47117 | ✅ |
| 5 | interleaved 与 rotate_half 只差一个置换 | A6 | True/False | ✅ |
| 6 | 长程衰减不单调 | A7 | 0.49(Δ50) < 0.56(Δ100) | ✅ |
| 7 | repo RoPE 无参数，buffer 形状 (1024,32) | B1 | ✓ | ❌ |
| 8 | fp32 角度误差随位置增长 | B4 | 3.6e-5@1024，1.2e-3@32768 | ✅（NumPy 可重写） |
| 9 | 把 Q、K 一起平移 100 位，分数不变 | B5 | 2.8e-5 | ❌ |
| 10 | SwiGLU(8d/3) 与 FFN(4d) 参数量相等 | C1/D1 | 1,179,648 | ✅ |
| 11 | SiLU 最小值 −0.2785 @ −1.2785 | C3 | ✓ | ✅ |
| 12 | GELU tanh 版与 erf 版相差 ≤4.7e-4 | C4/D3 | ✓ | ✅ |
| 13 | LayerNorm(x)=RMSNorm(x−mean) | E2/D6 | 0.0 / 4.8e-7 | ✅ |
| 14 | RMSNorm 对缩放不变，对平移会变 | E4/E5 | 4e-6 / 3.03 | ✅ |
| 15 | 手写 RMSNorm 的 eps=1e-5，而 nn.RMSNorm 默认 eps≈1.19e-7 | D4 | 2.3e-5；小输入时 2.6 | ❌ |
| 16 | 04 模型参数 18,301,824，0 个 bias | F1 | ✓ | ❌（可手算 ✅） |
| 17 | 初始 loss ≈ ln V + σ²/2，σ≈√E·0.02 | F3 + NumPy 版 | 9.2863 / 9.2765 | ✅ |
| 18 | 因果性 | F4 | 0.0 | ❌ |
| 19 | 长度 = max+1 时静默出错，max+2 时报错 | F7/I9 | ✓ | ❌ |
| 20 | 有无 cache 输出完全一致 | G1/G2/I5/J3/M | True；1.8e-6 | M 版 ✅ |
| 21 | 旧 bug 在 1-token prompt 下看不出来 | G4/M | 0/30 vs 30/30 | M 版 ✅ |
| 22 | 忘记 clear_cache 会污染下一次生成 | G5 | False | ❌ |
| 23 | 每 token 的 cache 字节数：2·L·H_kv·D·bytes | G3/I7 | 18,432 B（05 fp32）；12,288 B（webbot GQA bf16） | ✅ 公式 |
| 24 | 加速比随长度增长，但远低于 FLOP 比 | H/K6 | t=800 单步 24×；n=200 理论 100×，实测 2.8–4.6× | FLOP 部分 ✅ |
| 25 | enable_gqa 等价于 repeat_interleave | I2 | 0.0 | ❌ |
| 26 | SDPA is_causal 在 Lq≠Lk 时左上对齐 | I4 | [1,0,0,0,0] | ❌（NumPy 可模拟 ✅） |
| 27 | webbot 参数 152,316,672；MHA 版 161,753,856 | I1 | ✓ | ✅（手算） |
| 28 | 当前代码加载 HF storybot checkpoint 会因 mlp/ffn 改名失败 | J1 | RuntimeError | ❌ |
| 29 | cache 存的是旋转后的 K | K1 | True | ✅（M 版） |
| 30 | 分块 prefill 会出错 | K2 | 0.097 | ✅（M 版可演示） |
| 31 | RoPE-only 模型对平移不变（训练好的模型上验证） | J6 | 5.7e-6 | ❌（导出权重后 ✅） |

---

## 4. GPU / 数据需求

### 4.1 ch05 本身
- **ch05 的脚本都不需要 GPU**。本机 CPU 上每个都在 12 s 内跑完（其中 05 测速约 10–12 s，其余 2–4 s，多数时间花在 import torch）。
- 05_kvcache.py 在 GPU 机器上**也会跑在 CPU 上**，因为它没有 `.to(device)`。想测 GPU 需要打补丁：`model.to('cuda')`、`start_ids.to('cuda')`，计时前后加 `torch.cuda.synchronize()`。

### 4.2 可选的 GPU 实验（与 ch05 主题直接相关）

| 实验 | 目的 | 规模 / 估算 | 硬件建议 |
|---|---|---|---|
| (a) KV-cache GPU 测速 | 观察 prefill（compute-bound）与 decode（memory/launch-bound）的差别；看 GPU 上 batch=1 时加速比为什么也不大，以及 batch 变大、上下文变长时会怎样 | 05 模型 11M 参数，几分钟 | 本地 RTX 3060 Ti 足够。**未验证的预测**：batch=1 时每步约为 kernel launch 的毫秒级开销，无 cache 在 t≈几百时仍然很便宜，所以加速比也不会大；需要实测 |
| (b) 架构消融（仿 CS336 A1） | GPT-2 式（pos_embed+LayerNorm+GELU）vs 现代式（RoPE+RMSNorm+SwiGLU），以及逐项替换 | 每个变体约 22.7M 参数、5k–10k 步 × 8192 tokens（41–82M tokens），每个约 1.1e16–2.3e16 FLOP；3060 Ti 上每个约 0.3–0.6 h（假设 MFU 35%） | 本地 3060 Ti（8 GB 够用，静态内存 0.36 GB + 激活约 1–3 GB，估算）或 Modal L4（每个 < 0.15 h） |
| (c) StoryBot 完整预训练（属于 ch06，模型定义在本章范围） | 复现 `storybot/model_pretrain.pt` | 22,696,448 参数 × 327,680,000 tokens（40k 步 × 32 × 256）= 4.57e16 FLOP；3060 Ti 约 1.1 h，L4 约 0.3 h，A100 约 0.12 h（均假设 MFU 35%） | 本地 3060 Ti 可以 |
| (d) WebBot 预训练（属于 ch09） | 152.3M 参数、GQA、SDPA、torch.compile、DDP | 13,107,200,000 tokens（100k 步 × 32×4×1024），约是 2,649,004,800-token 训练集的 4.95 个 epoch；总计 1.27e19 FLOP → 1×A100 约 32 h，1×H100 约 10 h，8×H100 约 1.3 h 以上（MFU 35% 假设） | A100-80GB 或 H100。micro-batch 32×1024 时：静态 2.44 GB + logits 3.3 GB（bf16，CE 还会 upcast 到 fp32 6.6 GB）+ 激活约 10 GB 级 → 估计 20–30 GB（**未实测**）；24 GB 卡需要把 micro-batch 降到 8–16 并相应增加 accumulation |

FLOP 估算公式：训练约为 6·N·D，再加 attention 的 6·L·T·E·D（不打 causal 折扣）。§3-L 的脚本是实际跑过的。

### 4.3 数据 / 权重文件（HF `koki0702/zero-llm-data`，用 `curl -sIL` 测得的大小）

| 文件 | 字节 | 备注 |
|---|---|---|
| storybot/tiny_stories_train.txt | 2,227,753,162 | download.sh 默认下载 |
| storybot/tiny_stories_valid.txt | 22,502,601 | 同上 |
| storybot/tiny_stories_train.bin | 1,082,458,446 | uint16 → 541,229,223 tokens（注释掉了，没有默认下载） |
| storybot/tiny_stories_valid.bin | 10,931,746 | 5,465,873 tokens |
| storybot/merge_rules.pkl | 103,877 | **已下载到 scratch**（9743 条 merge） |
| storybot/model_pretrain.pt | 90,835,030 | **已下载到 scratch**；key 是 `mlp`，与当前代码不兼容 |
| storybot/model_iter_500.pt / model_iter_5000.pt | 90,835,030 / 90,835,075 | 我没有下载，但其它 agent 已经下载到了 `storybot/` 下。我用 weights_only 读取后确认：**两者也都是 12 个 `mlp` key、0 个 `ffn` key**，配置与 model_pretrain 相同（见 §3-J 补充） |
| storybot/model_dpo.pt | 404 | 未发布 |
| webbot/owt_train.txt / owt_valid.txt | 11,920,511,059 / 289,998,753 | 未下载 |
| webbot/owt_train.bin / owt_valid.bin | 5,298,009,600 / 128,992,132 | 2,649,004,800 / 64,496,066 tokens |
| webbot/merge_rules.pkl | 533,514 | 未下载 |
| webbot/model_pretrain.pt | 404 | **未发布**，要自己训练 |

HF 数据集最后一次上传是 2026-01-23（`/api/datasets/koki0702/zero-llm-data/commits/main`），早于 repo 里 `234a9c6`（2026-03-27）的 `mlp→ffn` 改名，这就是 J1 加载失败的原因。

### 4.4 Modal / RunPod / Lambda 建议
- ch05 相关的实验（a）（b）：本地 RTX 3060 Ti 就够；想更快可以用 Modal 的 L4 或 A10G，按小时计，一次消融总共不到 1 GPU-hour。
- ch09 WebBot：单张 A100-80GB 约 1.5 天，单张 H100 约 10 h，或者 8×H100 约 1.5–2 h（含 eval 与编译开销的粗估）。价格（凭记忆，**可能已经过时，用前请查**）：Modal L4 约 $0.8/h、A100-80GB 约 $2.5/h、H100 约 $4/h；RunPod 社区云 4090 约 $0.3–0.7/h、A100-80GB 约 $1.2–1.7/h、H100 约 $2–3/h；Lambda H100 约 $2.5–3.3/h。
- 缩到 CPU 能跑的规模：ch05 已经是 CPU 规模。若要在 CPU 上演示训练改良 GPT：E=128、L=2、H=4、ctx=128、batch 16、几百步（本 dossier **没有实测**收敛到多少，留给 ch06 的 dossier）。
- 浏览器：§3-M 的 NumPy 版（随机小模型）可以直接跑；训练好的 StoryBot 权重转成 fp16 后是 43.4 MiB，在 Pyodide 里每个 token 的推理成本，在本机原生 NumPy 上测得 13.8 ms（有 cache）/ 142 ms（无 cache，序列长度从 14 增到 74 期间的平均值），WASM 估计会慢几倍（未测）。

---

## 5. 易错点 / 坑 / 潜在 bug（标注了"已验证"的都实际跑过）

1. **[已验证，影响 ch06 读者] HF 上发布的 StoryBot checkpoint 用当前 `storybot/model.py` 加载会失败**：checkpoint 里是 `blocks.*.mlp.*`，代码在 `234a9c6` 改成了 `self.ffn`，`GPT.load_from` 会抛 Missing/Unexpected keys（J1）。`model_pretrain.pt`、`model_iter_500.pt`、`model_iter_5000.pt` 三个文件都是如此（§3-J 补充）。ch06 的 06_generate / 07_llm_judge / 08_dpo / 09_llm_judge 都是对这些路径直接调用 `GPT.load_from`（用 grep 确认过）。我实际运行了 `uv run python ch06/06_generate.py`，它确实在 `model.load_state_dict` 处抛出同样的 `RuntimeError: ... Missing key(s) "blocks.0.ffn.W.weight" ... Unexpected key(s) "blocks.0.mlp.W.weight" ...`。另外，09_llm_judge 还需要 `model_dpo.pt`，而 HF 上没有这个文件（404），必须先跑 08_dpo，而 08_dpo 加载预训练权重时也会遇到同样的问题。临时解决：`sd = {k.replace('.mlp.', '.ffn.'): v for k, v in sd.items()}`。webbot 仍然用 `mlp`，两个类之间同样不兼容（I8）。
2. **[已验证] prefill 不加 causal mask 的 bug（`8b3911f` 修复）用 1-token prompt 测不出来**。脚本自带的测速和一致性检查都只用了 `[[42]]`（G4、M）。在真实模型上，这个 bug 会让故事走向完全不同，但文本依然通顺，肉眼发现不了（J4）。教训：一致性测试的 prompt 至少要 2 个 token，而且模型至少 2 层。
3. **[已验证] 注释与实际行为不符**：`# Causal Mask（KVキャッシュ使用時は不要）` 说"使用 KV 缓存时不需要 mask"，但 prefill 时需要，代码也确实加了（`c240183` 的简化让注释变得不准确）。`# 固定シード` 说"固定种子"，实际只固定了起始 token；模型权重每次运行都不同（K5）。
4. **[已验证] 第二次调用时若 C>1（分块 prefill、一次追加多个 token、speculative decoding）没有 mask**，结果是错的（K2）。webbot 也一样：非首次调用时 `is_causal=False`。如果要支持，需要一个"右下角对齐"的 mask（I4 演示了 SDPA 的 `is_causal` 是左上角对齐的）。
5. **[已验证] `measure_generation_time` 的条件写反了**：只在 `use_cache=False` 时 `clear_cache()`，可是有 cache 的那次测量恰恰需要先清空。照脚本原来的顺序运行没有问题；如果连续测两次有 cache 的情况，第二次会接着上一次的 cache 继续（G5）。
6. **[已验证] 注释掉的一致性检查原样执行会 NameError**（`max_new_tokens` 在 `23046f6` 被删了），而且它只比较了前 200 个 token（总共 201 个）（G0）。
7. **[历史，来自 git diff] 测速公平性**：`5ff851d` 之前，无 cache 测 200 个 token，有 cache 只测默认的 100 个。有 cache 时耗时大致与 token 数成正比，所以加速比被高估约 2 倍。
8. **[已验证] 超过 max_context_len 后 RoPE offset 会被夹住**：所有新 token 都编码在位置 max−1。storybot 的 cache 无限增长；webbot 的 cache 像滑动窗口一样保留最近 max 个，但位置照样被冻结（I6）。在真实模型上，与"滑动窗口重算"的结果从第 262 个位置开始分叉（J5）。`utils.generate` 里对 `ids` 的截断是**死代码**，因为送进模型的只有 `next_id`。
9. **[已验证] 差一位的静默错误**：序列或 prompt 长度恰好等于 max+1 时，`cos_cache[-1:max]` 只取到 1 行，被广播到所有位置，模型就"静默"地失去了位置信息；长度 ≥ max+2 时报错（F7、I9）。04 的简单 RoPE 在 max+1 时就直接报错。`utils.generate` 不会截断过长的 prompt。
10. **[已验证] eps 不一致**：03 手写版是 1e-5，04 及之后用的 `nn.RMSNorm(eps=None)` 是 finfo.eps≈1.19e-7（D4）。初始化时 embedding 的 RMS≈0.02，输出尺度因此差约 1.2%（F6）。CS336 规定 eps=1e-5。
11. **[已验证] interleaved RoPE 与 HF 的 rotate_half 不能混用**（A6）。从 HF 导入 LLaMA 权重时，要么换成 rotate_half，要么对 W_q、W_k 的行做置换。
12. **[已验证] 用 `repeat` 而不是 `repeat_interleave` 来模拟 GQA 会算错**（I2）。PyTorch 的约定是第 h 个 query head 用第 h // (H/KV) 个 KV head。
13. **[已验证] SDPA 的 `is_causal=True` 在 Lq≠Lk 时是左上角对齐**（I4）。webbot 在 decode 时正确地用了 `is_causal=False`，但这一点很容易被学习者"顺手改坏"。
14. **[已验证] RoPE 的 cos/sin 表用 fp32 计算角度**，误差随位置增长（B4：1024 时 3.6e-5，32768 时 1.2e-3）。本书的长度没有问题，但扩展到长上下文时需要用 fp64 算角度，或者调整 θ（如 LLaMA-3 用 500000）。
15. **[已验证] RoPE buffer 是 persistent 的，而且每层在 state_dict 里各出现一次**（共享同一块存储，6 层就是 12 个 key，F5）。所以 checkpoint 自带 cos/sin 表，加载时如果改了 max_context_len 或 θ 就会形状冲突。CS336 的建议是 `persistent=False`。
16. **[已验证] RoPE 内部转 fp32、输出再转回 bf16，比全程 bf16 更准**（B3）。`6855e70` 加入了这一步；CS336 对 RMSNorm 也要求先 upcast 再计算。
17. **[已验证，CPU] autocast(bf16) 下 `nn.RMSNorm` 收到 bf16 输入和 fp32 权重时，会警告无法走 fused 实现**（K4）。GPU 上的行为和性能影响需要实测。
18. **[已验证] init loss 比 ln V 略高**：9.286 vs 9.210，可以用 ln V + σ²/2 精确解释（F3）。教程可以把它当作"先验算、后实验"的小练习。
19. **[已验证] KV cache 在 CPU 上实测的加速远低于 FLOP 比**（H/K6）。batch=1 的小模型被每个算子的固定开销主导；原脚本又是先测无 cache（冷启动），后测有 cache。教程里不要承诺"N 倍"，应该画"单步延迟随 t 变化"的曲线。
20. **默认 hidden=int(8d/3) 不取整**：d=512 时得到 1365，不是 64 的倍数。ch06 手动指定了 1344（=21×64）。CS336 和 LLaMA 都会取整到 64 或 256 的倍数（C2）。
21. **改良 GPT 不再共享 embed/unembed 权重**：参数多了 V·E（F2）。webbot 的 V=50000、E=768 时，unembed 一项就是 38.4M 参数（占 152.3M 的 25%）。
22. **`is_first_call` 只在 `use_cache=True` 分支里定义**，mask 条件 `not use_cache or is_first_call` 依赖短路求值。能用，但很脆弱：调整代码顺序就会 NameError。
23. **生成函数只支持 batch=1**：`next_id.item()` 要求只有一个元素，也没有 padding mask。
24. **`enable_gqa` 需要 torch ≥ 2.5**（`f85907f`）；`nn.RMSNorm` 需要 torch ≥ 2.4。
25. **ch09 训练用了 `torch.compile`**：`raw_model.save` 拿到的应该是原始模块的 state_dict（`OptimizedModule.__getattr__` 会转发到 `_orig_mod`），需要在 GPU 上确认保存的 key 没有 `_orig_mod.` 前缀（未验证，属于 ch09 dossier 的范围）。

---

## 6. 互动组件点子（每个都写明学习者操作什么、观察什么）

1. **RoPE 多指针时钟**：滑块拖动位置 m（0–1024），屏幕上有 32 个"指针"（每个频率平面一个），快的指针转得飞快，慢的几乎不动；悬停显示波长 2π/θ_i。要建立的直觉：位置被编码成一组不同速度的角度，快指针负责局部顺序，慢指针负责远距离。
2. **"一起平移"演示**：在 2D 平面上画 q（位于 m）和 k（位于 n）；按"同时 +1"按钮时两个箭头一起转，但 q·k 的读数不变；单独移动其中一个，读数才变。数据来自 A2/B5。
3. **Toeplitz 热力图**：显示 S[m,n] = RoPE(q,m)·RoPE(k,n) 的 64×64 热图，可以看到沿对角线的条纹；切换到"learned absolute embedding"（每个位置加一个随机向量）后条纹消失。再加一个按钮显示 S[5,2] ≠ S[2,5]（有符号距离）。
4. **长程衰减曲线**：q=k=全 1，画 score(Δ)/D；滑块调 θ（1e4 / 5e5）和 head_dim；阴影标出超出 max_ctx 的区域。让学习者自己发现：总体下降，但不单调（A7）。
5. **interleaved vs rotate_half 配对图**：用彩色弧线连接 (0,1),(2,3)… 与 (0,32),(1,33)…，展示两者只差一个置换（A6），并给出把 HF 权重转成 interleaved 的置换代码。
6. **激活函数操场**：ReLU/GELU/SiLU 曲线及其导数；标出 SiLU 的最小值 (−1.2785, −0.2785)。再画 SwiGLU 的门：在 (a,b) 平面上用热力图显示 SiLU(a)·b，体会"门关上"的区域。
7. **参数量计算器**：滑块调 d_model、V、L、H、KV、ff 倍率，实时显示 FFN(4d) 与 SwiGLU(8d/3) 的参数量、取整到 64 或 256 后的 hidden（d=4096 时是 11008）、整体参数构成的堆叠条形图；可切换"weight tying"和"GQA"，对照 F1/F2/I1 的精确数字。
8. **RMSNorm vs LayerNorm 拖拽条**：8 根可以拖动的柱子组成一个向量；左边显示 LayerNorm 输出，右边显示 RMSNorm 输出。按钮"+常数"（平移）和"×10"（缩放）；eps 滑块配合"缩小 1000 倍"按钮，演示 eps 陷阱（1e-5 vs 1.19e-7，D4）。
9. **CodeBot → 现代 GPT 的架构对照图**：左右两个 block 并排，逐个部件"换掉"（pos_embed→RoPE，LayerNorm→RMSNorm，GELU-FFN→SwiGLU，去掉 dropout/bias，取消共享权重）；每换一步，参数量即时更新（11,121,640 → 11,389,824）。
10. **KV Cache 单步动画（核心组件）**：token 一个个出现。"无 cache"模式下每步整个三角形闪烁重算；"有 cache"模式下只算新的一行，K/V 矩阵向右追加一列。计数器显示 FLOPs、cache 字节数（2·L·H_kv·D·bytes），清楚区分 prefill 阶段（三角 mask）和 decode 阶段（单行、无 mask）。
11. **prefill bug 开关**：在上面的动画里打开"buggy（prefill 不加 mask）"，prompt 内部出现指向未来的红色注意力格子；把 prompt 长度调成 1，bug 就"消失"了（G4 的结论）。还可以并排显示真实 StoryBot 的两段故事（J4 的文本），说明 bug 产物依然通顺。
12. **浏览器内的 NumPy mini-GPT（Pyodide）**：用 §3-M 的代码，学习者点击生成，同时看到"有/无 cache 的 token 序列完全相同"、两者的耗时和 FLOP 计数；附加开关"分块 prefill"（K2 的局限）和"init std"（std 太小时 bug 不显形）。
13. **浏览器内 StoryBot（进阶）**：加载 fp16 权重（43.4 MiB），输入 prompt，实时生成故事，可切换 KV cache，感受每个 token 约 14 ms 与约 140 ms 的差别（O 的数据；先确认权重许可）。
14. **GQA 分组可视化 + KV cache 显存计算器**：12 个 query head 按所属 KV 组着色；对比 `repeat_interleave`（正确）与 `repeat`（错误）的映射（I2）。显存计算器带滑块（层数、KV 头数、head_dim、序列长度、batch、dtype），预设 webbot / LLaMA-2-7B（MHA）/ LLaMA-3-8B（GQA 8）。
15. **SDPA 对齐陷阱**：Lq、Lk 两个滑块，并排显示 `is_causal=True` 的左上对齐 mask 与 decode 真正需要的右下对齐 mask；Lq=1、Lk=5 时，左上对齐只允许看第 0 个 key（I4）。
16. **上下文溢出演示**：生成超过 max_ctx 时，显示每个 token 实际使用的 RoPE 位置（在 max−1 处"卡住"）；storybot（cache 无限增长）、webbot（滑动窗口）与"重新 prefill"三种策略并排，标出开始分叉的位置（J5：262）。
17. **加速比曲线**：用 §3-H 的实测数据画"单步延迟 vs t"，再叠加理论 FLOP 曲线；加一个"固定开销"滑块，让学习者自己解释为什么实测只有 2–5×，而 FLOP 比是 100×。

---

## 7. CS336 对照（凭记忆，未查网页；A1 = Assignment 1 Basics，A2 = Systems）

| DLFS6 ch05 部件 | CS336 对应 | 异同 |
|---|---|---|
| `RoPE`（01） | A1 Transformer LM 部分的 RotaryPositionalEmbedding（adapter 大约是 `run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions)`）；Lecture 3（Architectures） | **配对约定相同**：都是相邻两维 (2k−1,2k) 的 interleaved 配对，θ_{i,k}=i/Θ^{(2k−2)/d}。区别：CS336 显式传入 `token_positions`（天然支持 KV cache 和任意位置），DLFS6 用 `offset` 加夹取；CS336 建议 buffer 用 `persistent=False` |
| `SwiGLU`（02） | A1 position-wise FFN（`run_swiglu`，以及 `run_silu`） | 公式相同：W2(SiLU(W1x)⊙W3x)，对应关系是 W→W1、V→W3、O→W2。CS336 要求 d_ff≈8/3·d_model 且取整到 64 的倍数；A1 的消融实验之一是"SwiGLU vs 参数量匹配的 SiLU-FFN（d_ff=4d）" |
| `RMSNorm`（03） | A1 `run_rmsnorm` | 公式相同，eps=1e-5；CS336 要求**先 upcast 到 fp32 再平方**，03 没有 upcast，04 之后用 `nn.RMSNorm`（eps 默认不同，见 §5-10） |
| `Block` / `GPT`（04） | A1 `run_multihead_self_attention_with_rope`、`run_transformer_block`、`run_transformer_lm`；`run_scaled_dot_product_attention` | 结构一致：pre-norm、无 bias、RoPE、SwiGLU、不共享权重。区别：CS336 要求从零实现 Linear/Embedding（不用 nn.Linear），初始化用截断正态（σ²=2/(d_in+d_out)），DLFS6 用 N(0,0.02)。A1 的其它消融：去掉 RMSNorm、post-norm vs pre-norm、NoPE vs RoPE，可以直接变成 §4 实验 (b) |
| 参数量 / FLOPs 计算（F1、I1、L） | A1 "Transformer LM resource accounting"（GPT-2 XL 的参数量与 FLOPs 拆解） | 本 dossier 的计数脚本可以当作"自动核对器" |
| `generate_*` / `utils.generate` | A1 decoding（temperature、top-p） | DLFS6 只有 temperature；CS336 A1 **不要求** KV cache |
| KV Cache（05）、prefill/decode | Lecture 10 "Inference"（KV cache、prefill 是 compute-bound、decode 是 memory-bound、arithmetic intensity、GQA/MQA/MLA、speculative decoding、PagedAttention/vLLM）；A5 用 vLLM 做 rollout | ch05 在这里**超出了 A1**，可以作为 A1 之后的扩展练习 |
| GQA（webbot `n_kv_head`） | Lecture 3（MQA/GQA）与推理 lecture | CS336 作业不要求实现；webbot 的 I1/I7 数字可以直接用在讲义里 |
| `F.scaled_dot_product_attention`（webbot） | A2：在 PyTorch 和 Triton 中手写 FlashAttention-2 的 forward（和 backward），并做 benchmark | DLFS6 直接调库；可以把 A2 的 Triton kernel 作为"拆开黑盒"的进阶 |
| bf16 autocast（ch06/ch09 训练） | A2 mixed precision；A1 RMSNorm 的 upcast | 与 §5-16、§5-17 呼应 |

---

## 附：建议教程采用的"端到端可验证"主线（来自本章证据）

1. 先用 NumPy 写 RoPE，立刻验证三件事：范数不变、Toeplitz、平移不变（A 组，浏览器内运行）。
2. 写 SwiGLU 和 RMSNorm，用参数量计算器与 LayerNorm 对照（C/E 组）。
3. 拼成 mini-GPT（§3-M），验证因果性和初始 loss（F3/F4 的 NumPy 版）。
4. 加 KV cache，用"有无 cache 输出相同"作为单元测试；再打开 buggy 开关，看清**为什么 1-token prompt 测不出 bug**（G4/M）。
5. 最后加载真实 StoryBot（处理 mlp→ffn 改名），在浏览器里生成故事，切换 cache 感受速度差（J3/O）。GPU 部分作为选做：本地 3060 Ti 跑架构消融，WebBot 的 ch09 训练放到云上的 A100/H100。
