# 研究档案 · DLFS6 第 4 章：把 BPE 做成"能跑 2 GB 语料"的工程版（StoryBot 分词器）

> **范围**（逐行读完）：`ch04/01_bpe_optimize.py`、`ch04/02_bpe_cache.py`、`ch04/03_bpe_chunk.py`、`ch04/04_bpe_parallel.py`、`ch04/06_encode_optimize.py`、`ch04/07_encode_parallel.py`、`storybot/tokenizer.py`、`storybot/download.sh`。为了对照，还读了 ch01 的 BPE 脚本、`codebot/tokenizer.py`、ch06/ch07/ch09 中使用 `storybot.tokenizer` 的地方。
>
> **机器**：Intel i7-11700F（8 核 16 线程）、47.7 GB 内存、Windows 11；Python 3.12.9（uv）、regex 2026.1.15、numpy 1.26.4、tqdm 4.67.1、torch 2.9.1 CPU、node 24.13。
>
> **计时警告（重要）**：测量期间同一台机器上有其它 agent 在跑 torch 训练/推理，CPU 负载在 52%–100% 之间波动（多数时间 84–100%）。所以**绝对时间普遍偏慢 1.5–2.5 倍**；同一次运行里的**比值**、以及 `time.process_time()` 记录的 **CPU 时间**更可信。每张表都注明了是 wall 还是 CPU。所有"输出相同"的结论与负载无关。
>
> **产物位置**：所有临时脚本、补丁副本、日志都在
> `C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch04-bpe\`（下文简称 `SCR/`）。本档案只往仓库里加了 `storybot/download.sh` 列出的三个文件（`tiny_stories_valid.txt`、`merge_rules.pkl`、`tiny_stories_valid.bin`），以及 Python 自动生成的 `__pycache__/`（已被 `.gitignore` 忽略），没有改动任何源码。`git status` 里的 `uv.lock` 改动发生在 16:05:54（uv 0.9 把锁文件格式从 revision 1 升到 3），早于本档案的第一条命令；其它未跟踪文件（模型 `.pt`、`webbot/*`）来自并行的其它 agent。

---

## 0. TL;DR（写教程的人先看这里）

1. **这一章在讲什么**：第 1 章的 BPE（CodeBot 用，6.7 MB 语料、词表 1000）在 TinyStories（2.2 GB、词表 10000）上完全跑不动。第 4 章用 4 步把**训练**提速，再用 2 步把**编码**提速，并且每一步的产物都与朴素版**逐项相同**。
2. **训练侧四步**：
   ① 预分词去重 + 频次加权（`01`）→ ② 增量维护 pair 计数 + 倒排索引 `pair_to_ids`（`02`，书中叫"キャッシュ"）→ ③ 按 `<|endoftext|>` 对齐切块、二进制分块读文件、生成器式预分词（`03`，主要省**内存**）→ ④ `multiprocessing.Pool` 并行预分词（`04`）。
3. **编码侧两步**：⑤ 把"对每个预分词依次尝试全部 9743 条合并规则"改成"反复合并当前**优先级（rank）最小**的相邻对"（`06`）→ ⑥ 文件分块 + 多进程 + 每块缓存为二进制 + `np.memmap` 拼成 `uint16` 的 `.bin`（`07`），供第 6 章预训练直接 memmap。
4. **实测（本机，同条件对比）**：
   - 训练 merge 循环：tiny_codes 全量（词表 1000）：朴素（第 1 章代码）**3,423.8 s CPU（77 分钟 wall）** → `01` 122.3 s → `02` 5.8 s（**约 590×**）；as-is 运行时 `01` 89 s → `02` 3.7 s。TinyStories 500 KB 切片、词表 1000：朴素 198.8 s → `01` 15.7 s → `02` 1.66 s（120×）。
   - 预分词并行（valid，22.5 MB，64 块，含进程启动）：串行 12.4 s → 4 进程 3.7 s（3.4×）→ 8 进程 3.0 s（**4.1×**）；16 进程反而 4.5 s（与同机其它任务争用）。`encode_file`：1 → 8 → 16 进程 163.6 → 30.4 → 22.1 s（**5.4×、7.4×**）。Windows spawn 启动 Pool 本身要 0.4–2.0 s。
   - 编码：同一 29.6 KB 切片，ch01 编码器 40.3 s CPU vs `06` 0.23 s（约 **175×**）；`06` 的词表越大优势越大（743 条规则时约 29×）。
5. **端到端验证**：用官方 `storybot/merge_rules.pkl` + 本章 `encode_file` 重新编码 `tiny_stories_valid.txt`，产出与官方 `tiny_stories_valid.bin` **逐字节相同**（`cmp` 通过，5,465,873 个 uint16）。训练集方向也验证了：把训练集前 100 MiB（截到最后一个 `<|endoftext|>`）用同样流程编码，得到 25,473,389 个 token，与官方 `tiny_stories_train.bin` 的前 50,946,778 字节（HTTP Range 下载）**逐字节相同**，而官方文件的下一个 token 恰好是 9999（`<|endoftext|>`）。`02` 在 LF 版 tiny_codes 上训练出的 743 条合并规则与仓库自带的 `codebot/merge_rules.pkl` **完全一致（含顺序）**，第 1 章的朴素训练器跑出来也完全一样。
6. **下一个瓶颈（书里没做）**：`02` 之后，merge 循环 **97%** 的时间花在 `max(pair_counts, key=…)` 这一次全表扫描上（每步平均只动 7 个唯一序列，中位数 1 个）。用"惰性删除的最大堆"替换后再快 **23–25×**，merge 结果逐项相同——适合做成教程里的"扩展练习"，也正好对应 CS336 的提示。
7. **与 CS336 A1 高度同构**：数据是同一份（TinyStoriesV2-GPT4 与 OWT 样本，字节数一致），预分词正则一字不差，`find_chunk_boundaries` 与 CS336 `pretokenization_example.py` 几乎逐行相同；最长 token 同为 `' accomplishment'`。**唯一实质差别**：平票（tie）时 DLFS6 比较 **token id**，CS336 规定比较 **bytes 字典序**——在 tiny_codes 上从第 289 条合并开始分叉。
8. **Windows 专属坑**：`core.autocrlf=true`（本机全局配置）让 `codebot/tiny_codes.txt` 检出为 CRLF。文本模式读（`01/02/06`）看不到 `\r`；二进制读（`03/04/07`）会把 `b'\r\n'` 学成第 2 条合并（id 257），官方结果从这一条开始就对不上了。
9. **内存坑**：CPython 的字符串按"全串最宽字符"决定每字符 1/2/4 字节（PEP 393）。100 MiB 训练前缀里只要有一个 🍌（U+1F34C），整段 `str` 就是 400 MiB。整文件读入（`02` 的写法）峰值 656 MiB，分块（`03`）只要 42 MiB。全量 2.2 GB 整读估计要 9 GB 以上，这才是 `03` 存在的真正理由。

---

## 1. 推断的章节叙事

### 1.1 全书位置：三个 bot、三档规模

| 部分 | Bot | 语料（本仓库数据） | 词表 | 分词器代码 |
|---|---|---|---|---|
| 第 1–3 章 | CodeBot | `codebot/tiny_codes.txt` 6,743,534 B（git 中为 LF 6,487,033 B） | 1,000 | `codebot/tokenizer.py`（第 1 章朴素版） |
| **第 4–6 章** | **StoryBot** | `storybot/tiny_stories_train.txt` **2,227,753,162 B**、`valid` 22,502,601 B | **10,000** | **`storybot/tokenizer.py`（本章产物）** |
| 第 7–9 章 | WebBot | `webbot/owt_train.txt` 11,920,511,059 B、`valid` 289,998,753 B | 50,000 | **没有** `webbot/tokenizer.py`，ch07/ch09 直接 `from storybot.tokenizer import …` |

第 4 章是"规模逼出工程"的转折点：语料从 6.7 MB 涨到 2.2 GB（约 330 倍），词表从 1000 涨到 10000（合并次数从 743 涨到 9743），第 1 章的实现在两个维度上同时失效。第 7 章训练 50k 词表的 OWT 分词器时，直接复用本章的 `train_bpe` / `encode_file`（`ch07/01_bpe_train.py` 显式传 `num_processes=8, num_chunks=64`）。

### 1.2 文件编号 → 小节顺序（推断）

| 编号 | 文件 | 推断小节 | 引入的概念 |
|---|---|---|---|
| — | （正文） | 4.0 引言：第 1 章的 BPE 为什么慢 | 复杂度 = 合并次数 × 语料总字节；本机在 6.5 MB 的 tiny_codes 上就要约 1 小时 |
| 01 | `01_bpe_optimize.py` | 4.1 去重：只处理"不同的"预分词 | `pretoken_counts`、`ids_counts`、`count_pairs(..., weight)`、确定性 tie-break |
| 02 | `02_bpe_cache.py` | 4.2 缓存：只更新受影响的序列 | 增量 `pair_counts`、倒排索引 `pair_to_ids`（"キャッシュ"） |
| 03 | `03_bpe_chunk.py` | 4.3 分块：别把 2 GB 一次读进内存 | `find_chunk_boundaries`（按 `<\|endoftext\|>` 对齐）、二进制 seek/read、`re.finditer` 生成器 |
| 04 | `04_bpe_parallel.py` | 4.4 并行：多进程预分词 | `Pool.imap`、结果归并、`if __name__ == '__main__'`、保存 `storybot/merge_rules.pkl` |
| **05** | **（缺号）** | **4.5 见下文推断** | — |
| 06 | `06_encode_optimize.py` | 4.6 编码提速：按优先级合并 | rank 字典查找、❶取相邻对 ❷找最小 rank ❸能否合并 ❹执行合并 |
| 07 | `07_encode_parallel.py` | 4.7 大文件编码：并行 + 磁盘缓存 + memmap | `_encode_chunk`、`encode_file`、`np.uint16`、`np.memmap`、`shutil.rmtree` |
| — | `storybot/tokenizer.py` | 本章代码的"定稿模块" | 上面所有函数 + `BPETokenizer` 完整版 |

**缺号 05 的推断**（按可能性排序）：

1. **"把训练代码整理进 `storybot/tokenizer.py`"的那一节**。证据：`06` 一开头就 `from storybot.tokenizer import pretokenize, count_pairs, merge`，说明在 06 之前模块已存在；而模块里的 `train_bpe` 与 `04` 几乎逐行相同（`diff` 只差注释、多一个没人调用的 `process_single_chunk`、以及 `num_chunks` 默认值 64→8）。这种"代码放在 `storybot/` 而不是 `ch04/`"的小节自然不会占用 `ch04/05_*.py` 这个编号。
2. **"在完整 TinyStories 上跑一遍并看看学到了什么"**。`04` 的 `__main__` 已经把结果存成 `storybot/merge_rules.pkl`，正文很可能在这里展示训练时长、前几个/最后几个 token、最长 token（对比 ch01/08 对 CodeBot 做过的同样分析）——这部分可以只用文字和输出截图。
3. **"编码为什么也慢"的动机小节**：`06` 的 `__main__` 故意用回 CodeBot 的 `merge_rules.pkl` 和 `tiny_codes.txt`，正好可以与 ch01/09 的编码耗时直接对比；这一节若只是复述 ch01 代码的计时，也不需要新文件。

> 编号缺口在全书很常见：ch02 缺 04/05，ch03 缺 05–08，ch06 缺 01。它们多半对应纯文字小节或只引用旧代码的小节。

### 1.3 叙事主线（教程可直接沿用的"问题 → 改进 → 验证"节奏）

1. **问题**：第 1 章 `train_bpe` 每步都把**全部预分词出现**（tiny_codes 有 1,826,876 个、TinyStories valid 有 5,419,001 个）重新数一遍、合并一遍。
2. **观察 1（Zipf）**：不同的预分词极少——tiny_codes 只有 30,535 个（59.8× 重复），TinyStories valid 只有 13,111 个（**413×** 重复）。→ 去重 + 加权（`01`）。
3. **观察 2（局部性）**：一次合并只影响含有该 pair 的序列——TinyStories valid 上平均每步只动 **7.0** 个唯一序列（中位数 **1**），占 13,111 的 0.054%。→ 增量更新 + 倒排索引（`02`）。
4. **观察 3（内存）**：2.2 GB 文件整读进 Python，再 `split` 一次，内存要翻好几倍（本档案实测：整读 100 MiB 峰值 656 MiB）。`<|endoftext|>` 天然是安全切点。→ 分块（`03`）。
5. **观察 4（可并行）**：各块的预分词互不相干，只需最后把计数相加。→ 多进程（`04`）。merge 循环本身是串行的，所以整体加速受 Amdahl 定律限制。
6. **编码侧观察**：第 1 章编码器对每个预分词都跑 9743 遍 `merge()`；但 99.74% 的预分词出现最终只变成 1 个 token，`' accomplishment'` 也只需 14 次合并。→ 按 rank 合并（`06`）。
7. **工程收尾**：541,229,223 个 token 若放进一个 Python list，实测约 8.6 B/token（int 对象本身是共享的，开销主要是指针），光指针就要约 4.3 GiB，编码时峰值约 13.4 B/token；存成 `uint16` 只要 2 B/token（1.08 GB）。所以每块写磁盘缓存，最后 memmap 拼接成 `uint16`（`07`）。第 6 章用 `np.memmap(..., dtype=np.uint16, mode='r')` 直接读。


---

## 2. 逐脚本解析

> 记号：`N_occ` = 预分词出现次数，`U` = 不同预分词个数，`B_occ`/`B_U` = 对应的总字节数，`M` = 合并次数（`vocab_size − 257`），`P_t` = 第 t 步时不同相邻对的个数，`A_t` = 第 t 步受影响的唯一序列数。
>
> 共用的三个小函数在 01–04 与 `storybot/tokenizer.py` 中逐字相同（只有 `pretokenize` 从 `findall` 改成了 `finditer` 生成器）：
>
> ```python
> def pretokenize(text):                      # 01/02: return re.findall(...)；03/04/storybot: 生成器
>     pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
>     for m in re.finditer(pattern, text):
>         yield m.group(0)
>
> def count_pairs(ids, weight=1, counts=None):   # ch04 新增 weight 参数（ch01 版没有）
>     if counts is None:
>         counts = defaultdict(int)
>     for pair in zip(ids, ids[1:]):
>         counts[pair] += weight
>     return counts
>
> def merge(ids, pair, new_id):               # 贪心、从左到右、不重叠
>     merged_ids = []; i = 0
>     while i < len(ids):
>         if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
>             merged_ids.append(new_id); i += 2
>         else:
>             merged_ids.append(ids[i]); i += 1
>     return merged_ids
> ```
>
> 所有脚本开头都有 `os.chdir(<仓库根>)` 与 `sys.path.append('.')`，所以从任何目录运行都以仓库根为相对路径基准。

### 2.0 基线：第 1 章的朴素实现（`codebot/tokenizer.py`，供对照）

```python
def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    texts = input_text.split(end_token)
    ids_list = []
    for text in texts:
        for pretoken in pretokenize(text):
            ids_list.append(list(pretoken.encode("utf-8")))   # 每一次"出现"都存一份
    num_merges = vocab_size - 256 - 1
    for step in tqdm(range(num_merges), desc="Training BPE"):
        counts = defaultdict(int)
        for ids in ids_list:                                   # 每步：全量重新计数
            counts = count_pairs(ids, counts)
        if not counts: break
        best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))
        merge_rules[best_pair] = 256 + step
        for i in range(len(ids_list)):                         # 每步：全量重新合并
            ids_list[i] = merge(ids_list[i], best_pair, 256 + step)
```

- 复杂度 ≈ `O(M · B_occ)`：每步把语料全部字节扫两遍。
- 编码器 `_encode_text` 对每个预分词**按训练顺序依次执行全部 M 条合并**：`O(M · L)`/预分词。
- 注意：ch01 的 `03_bpe_train.py` / `06_pretokenize.py` 用的是 `max(counts, key=counts.get)`（平票时取字典里**先插入**的那个），`codebot/tokenizer.py` 已经换成了确定性 key `(count, pair[0], pair[1])`，第 4 章沿用。这个 key 是后面所有优化"结果不变"的前提（§3.4）。

### 2.1 `01_bpe_optimize.py` —— 去重 + 加权计数

**目的**：同一个预分词（比如 `' the'` 在 valid 里出现 211,031 次）没必要存 21 万份。改为"不同预分词 → 出现次数"，计数时乘上次数。

**签名**：`pretokenize(text) -> list[str]`；`count_pairs(ids, weight=1, counts=None) -> defaultdict`；`merge(ids, pair, new_id) -> list[int]`；`train_bpe(input_text, vocab_size, end_token="<|endoftext|>") -> dict[tuple[int,int], int]`。

**核心代码**：

```python
pretoken_counts = defaultdict(int)
for text in tqdm(texts, desc="Pretokenizing"):
    for pretoken in pretokenize(text):
        pretoken_counts[pretoken] += 1
ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}

for step in tqdm(range(num_merges), desc="Training BPE"):
    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)          # 以出现次数为权重
    if not pair_counts: break
    best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
    merge_rules[best_pair] = 256 + step
    new_ids_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        new_ids_counts[tuple(merge(ids, best_pair, 256 + step))] += count
    ids_counts = new_ids_counts
```

**为什么结果不变**：pair 频次对"出现"是线性可加的；`merge` 是序列的确定性函数，所以相同的预分词永远同步演化。把 `n` 次相同出现合成一条权重为 `n` 的记录，计数与合并结果都完全一样（F03、F06、§3 E1/E2 实测）。复杂度降到 `O(M · B_U)`。

| 语料 | `N_occ` | `U` | `N_occ/U` | `B_occ` | `B_U` | 每步工作量缩小 |
|---|---:|---:|---:|---:|---:|---:|
| tiny_codes（LF） | 1,826,876 | 30,535 | 59.8× | 6,203,178 | 229,212 | **27.1×** |
| TinyStories valid | 5,419,001 | 13,111 | 413.3× | 22,143,411 | 90,844 | **243.8×** |

（`SCR/corpus_stats.py` 实测；valid 前 10 个预分词覆盖 31.9%、前 1000 个覆盖 92.5%。）

**日文注释中译**：特殊トークンで分割 → 按特殊 token 切分；各テキスト片を事前トークン化 → 对每个文本片段做预分词；進捗表示のためtqdmを使用 → 用 tqdm 显示进度；事前トークンをID列に変換 → 把预分词转成 ID 序列；各ids列について、その出現回数を考慮してペア頻度を集計 → 对每个 ids 序列按其出现次数加权统计相邻对频次；ペアが存在しない場合の処理 → 没有任何相邻对时的处理；最頻出ペアを選択 → 选出最频繁的相邻对；各ids列をマージして更新 → 对每个 ids 序列执行合并并更新；マージを適用 → 应用合并；語彙サイズの設定 → 设置词表大小。

**实际运行（as-is，tiny_codes，词表 1000，负载约 84%）**：

```text
Pretokenizing: 100%|██████████| 21836/21836 [00:01<00:00, 12544.79it/s]
Training BPE: 100%|██████████| 743/743 [01:29<00:00,  8.32it/s]
```

脚本本身不打印别的。21836 = 21,835 个 `<|endoftext|>` 切出的片段数。merge 循环 89 s。

### 2.2 `02_bpe_cache.py` —— 增量计数 + 倒排索引（"キャッシュ"）

**目的**：`01` 每一步仍然把 3 万个唯一序列全部重新计数、全部重新合并。其实只有**含 best_pair 的序列**会变。

**新数据结构**：`pair_to_ids: defaultdict(set)`，pair → 含有该 pair 的 ids 元组集合；`pair_counts` 只在初始化时全量计算一次。

**核心代码**：

```python
pair_to_ids = defaultdict(set)  # キャッシュ
pair_counts = defaultdict(int)
for ids, count in ids_counts.items():
    count_pairs(ids, count, pair_counts)
    for pair in zip(ids, ids[1:]):
        pair_to_ids[pair].add(ids)

for step in tqdm(range(num_merges), desc="Training BPE"):
    if not pair_counts: break
    best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
    new_id = 256 + step
    merge_rules[best_pair] = new_id

    affected_ids = pair_to_ids[best_pair]
    del pair_to_ids[best_pair]
    for ids in affected_ids:                       # 只遍历受影响的序列
        ids_count = ids_counts[tuple(ids)]
        new_ids = merge(ids, best_pair, new_id)
        del ids_counts[tuple(ids)]
        ids_counts[tuple(new_ids)] = ids_count
        for pair, count in count_pairs(ids).items():      # 减去旧序列的全部贡献
            pair_counts[pair] -= count * ids_count
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            pair_to_ids[pair].discard(tuple(ids))
        for pair, count in count_pairs(new_ids).items():  # 加上新序列的全部贡献
            pair_counts[pair] += count * ids_count
            pair_to_ids[pair].add(tuple(new_ids))
```

**为什么结果不变**（教程里值得写成一段"证明"）：
`pair_counts = Σ_序列 w·pairs(序列)`。不含 best_pair 的序列经过 `merge` 不变，贡献不变；含它的序列从 `pairs(old)` 变成 `pairs(new)`，差值正好是代码里"先整体减、再整体加"。索引 `pair_to_ids` 同样逐序列维护，所以始终精确（没有陈旧条目）。F15 在 300 步里每一步都与全量重算比对，全部相等。

**几个细节**：
- `ids_counts[tuple(new_ids)] = ids_count` 用的是赋值而非 `+=`：不会冲突，因为两个不同的预分词字节串不同，而合并不改变字节串，所以合并后的序列也不会撞车。
- `del pair_to_ids[best_pair]` 之后，循环里 `pair_to_ids[pair].discard(...)` 遇到 `pair == best_pair` 时，defaultdict 会**悄悄新建一个空集合**（F21）。无害，只是留下一个空条目。
- 合并后序列中不会再残留 best_pair（贪心从左到右的 `merge` 保证），所以那个空集合不会被再次填充。

**每步到底动了多少？**（`SCR/analyze_v02.py`，书中循环的逐行复刻，输出与参考 merges 相同）

| 语料 / 词表 | `U` | 平均受影响序列/步 | 中位数 | 占比 | 全程累计 vs `01` 的 `U×M` | `max()` 扫描的 pair 数（首/峰/末） |
|---|---:|---:|---:|---:|---:|---|
| tiny_codes / 1000 | 30,535 | 135.1 | 49 | 0.44% | 100,346 vs 22,687,505（226× 少） | 2,859 / 18,358 / 18,358 |
| TinyStories valid / 10000 | 13,111 | **7.0** | **1** | 0.054% | 68,657 vs 127,740,473（1861× 少） | 932 / 12,025 / 7,821 |
| 训练集前 100 MiB / 10000 | 21,808 | 11.1 | 2 | 0.051% | 108,554 vs 212,475,344（1957× 少） | 1,276 / 18,637 / 17,229 |

**日文注释中译**：キャッシュ → 缓存；キャッシュに登録 → 登记到缓存；best_pairを含むids列をキャッシュから取得 → 从缓存取出含 best_pair 的 ids 序列；もう使わないので削除 → 以后用不到了，删掉；影響のあるID列だけを更新 → 只更新受影响的 ID 序列；古いID列を削除 → 删除旧 ID 序列；新しいID列を追加 → 加入新 ID 序列；古いペア頻度を減少 → 减去旧的相邻对频次；新しいペア頻度を増加 → 加上新的相邻对频次。

**实际运行（as-is，tiny_codes，词表 1000）**：

```text
Pretokenizing: 100%|██████████| 21836/21836 [00:01<00:00, 16352.49it/s]
Training BPE: 100%|██████████| 743/743 [00:03<00:00, 202.71it/s]
```

整个进程 wall 5.5 s（含 uv 启动）；merge 循环 1:29 → 0:03.7。文件末尾注释掉的 TinyStories 训练（`open("storybot/tiny_stories_train.txt").read()`）在本机没有跑：整读 2.2 GB 的内存问题见 §2.3、§5。

### 2.3 `03_bpe_chunk.py` —— 按 `<|endoftext|>` 对齐的分块读取

**目的**：不再 `open(...).read()` 整个 2.2 GB 文件。用二进制 seek/read 一块一块处理；块的边界必须落在 `<|endoftext|>` 上，否则会把一个单词切成两半（F13：在 `' lit|tle'` 处切开，编码结果就变了）。

**签名**：`find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>") -> list[int]`；`train_bpe(file_path, vocab_size, end_token=...)`（内部写死 `num_chunks=64`）。`pretokenize` 改成生成器（`re.finditer`）。

**核心代码**：

```python
def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")
    with open(file_path, "rb") as file:
        file.seek(0, os.SEEK_END); file_size = file.tell(); file.seek(0)
        chunk_size = file_size // num_chunks
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)
        buffer_size = 4096
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)
            while True:
                buffer = file.read(buffer_size)
                if buffer == b"":                       # 读到文件尾也没找到
                    chunk_boundaries[bi] = file_size; break
                end_position = buffer.find(byte_end_token)
                if end_position != -1:                  # 边界移到下一个 <|endoftext|> 的起点
                    chunk_boundaries[bi] = chunk_position + end_position; break
                chunk_position += buffer_size
    return sorted(set(chunk_boundaries))           # 去重（块数 > 文档数时会重复）

# train_bpe 中：
for i in range(total_chunks):
    f.seek(start); chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
    for text in chunk_text.split(end_token):
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1
```

**要点**：
- 内部边界都恰好落在 `<|endoftext|>` 的第一个字节上（F11：valid 切 64 块，65 个边界，内部 63 个全部指向 `b'<|endoftext|>'`）。所以除第一块外，每块都以 `<|endoftext|>` 开头，`split` 后多出一个空串，对计数无影响。
- 因为边界是 ASCII 的 `<`，块永远在 UTF-8 字符边界上，`errors="ignore"` 实际上不会丢字节（前提是文件本身是合法 UTF-8；valid 已验证）。
- 若 `<|endoftext|>` 恰好跨两个 4096 字节缓冲区，这一次查找会错过它，边界落到下一个 `<|endoftext|>`——依然正确，只是块大小不均。
- 块数大于文档数时边界会重复，`sorted(set(...))` 去重：valid 要 100,000 块，实际只得到 27,629 个边界（F11）。
- 分块后的预分词计数与整文件 `split` 完全相同（F12：5,419,001 次出现、13,111 个不同预分词，`Counter` 相等），所以 merges 也相同（§3 E2）。
- **这一步主要省内存，不省时间**：单进程分块的预分词反而略慢（多了 seek/decode 与生成器开销）。内存实测见 §3.5：整读 100 MiB 峰值 656 MiB，分块 42 MiB。

**日文注释中译**：ファイルをバイナリモードで開く → 以二进制模式打开文件；ファイルサイズを取得 → 取得文件大小；チャンクの開始位置を計算（等間隔）→ 计算各块起点（等间距）；最後にファイル終端を追加 → 末尾加上文件结尾位置；境界から先読みするバイト数 → 从边界处往后预读的字节数；境界位置の調整（終了トークンを探す）→ 调整边界（寻找结束 token）；境界の推定位置から開始 → 从估计的边界位置开始；バッファサイズ分を読む → 读一个缓冲区；ファイル終端に達した場合 → 到达文件尾时；読み取ったチャンクで終了トークンを検索 → 在读到的数据里找结束 token；見つかった場合、その位置を新しい境界とする → 找到则以该位置为新边界；見つからなかった場合、次のバッファ位置に移動 → 找不到就移到下一个缓冲区；重複を除去し、ソートして返す → 去重并排序后返回。

**实际运行（补丁副本：训练集 → valid，其余不变；`SCR/patched/03_bpe_chunk.py`）**：

```text
Pretokenizing: 100%|██████████| 64/64 [00:14<00:00,  4.29it/s]
Training BPE: 100%|██████████| 9743/9743 [00:42<00:00, 228.03it/s]
ADDED: num merges = 9743 | first 5 = [((32, 116), 256), ((104, 101), 257), ((32, 97), 258), ((32, 115), 259), ((32, 119), 260)]
```

wall 59.1 s（负载约 84%）。`ADDED:` 一行是补丁加的打印。前 5 个 token：`' t'`、`'he'`、`' a'`、`' s'`、`' w'`。

### 2.4 `04_bpe_parallel.py` —— 多进程预分词

**目的**：各块的预分词互不依赖，丢给进程池；主进程把各块的 `Counter` 相加。

**签名**：`pretoken_chunk(args: tuple[file_path, start, end, end_token]) -> defaultdict[str, int]`；`train_bpe(file_path, vocab_size, end_token="<|endoftext|>", num_processes=8, num_chunks=64)`。

**核心代码**：

```python
def pretoken_chunk(args):
    file_path, start, end, end_token = args          # imap 只能传一个参数，所以打包成元组
    pretoken_counts = defaultdict(int)
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
        for text in chunk_text.split(end_token):
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1
    return pretoken_counts

# ステップ1〜3
chunk_info_list = [(file_path, s, e, end_token) for s, e in zip(b[:-1], b[1:])]
with Pool(processes=num_processes) as pool:
    all_results = list(tqdm(pool.imap(pretoken_chunk, chunk_info_list), total=len(chunk_info_list)))
pretoken_counts = defaultdict(int)
for chunk_result in all_results:
    for pretoken, count in chunk_result.items():
        pretoken_counts[pretoken] += count
# 之后与 02 完全相同的增量 merge 循环
...
if __name__ == '__main__':      # Windows/macOS 的 spawn 模式下必须有
    merge_rules = train_bpe(file_path, vocab_size, num_processes=8)
    with open("storybot/merge_rules.pkl", "wb") as f:
        pickle.dump(merge_rules, f)
```

**要点**：
- 只传 `(路径, 起点, 终点)` 给子进程，由子进程自己打开文件读——避免把几十 MB 文本 pickle 过管道。返回的是每块的计数字典（几千到几万个键），很小。
- `imap` 默认 `chunksize=1`，64 个任务按完成顺序动态分配给 8 个进程，负载均衡好。`storybot/tokenizer.py` 把默认 `num_chunks` 改成了 **8**：每个进程正好一块，既没有动态均衡，单块又大（全量时约 278 MB/块），内存见 §5。
- **merge 循环仍是串行**，所以并行只加速预分词。valid 上预分词约 13.5 s、merge 循环约 27–78 s（视负载），Amdahl 上限很明显。
- Windows 是 spawn：子进程会重新 import 主脚本，因此 `if __name__ == '__main__':` 必不可少；`os.chdir`/`sys.path.append` 写在模块顶层，子进程也会执行，这正好让子进程能找到数据文件。

**日文注释中译**：ステップ1: チャンクの準備 → 第 1 步：准备分块；ステップ2: 並列処理 → 第 2 步：并行处理；ステップ3: 結果を統合 → 第 3 步：合并结果；ファイルを開いてチャンクを読み込む → 打开文件读入分块；各テキストを事前トークン化 → 对每段文本做预分词；8つのプロセスで並列処理を実行 → 用 8 个进程并行执行。

**实际运行（补丁副本：训练集 → valid，pkl 存到 `SCR/out/`）**：

```text
Pretokenizing: 100%|██████████| 64/64 [00:01<00:00, 45.88it/s]
Training BPE: 100%|██████████| 9743/9743 [00:27<00:00, 359.85it/s]
ADDED: num merges = 9743 | first 5 = [((32, 116), 256), ((104, 101), 257), ((32, 97), 258), ((32, 115), 259), ((32, 119), 260)]
```

wall 29.2 s（tqdm 计时不含 Pool 启动；8 进程启动约 1.45 s）。保存的 pkl 与 `02`/`03` 在 valid 上的结果逐项相同。


### 2.5 `05`（缺号）

见 §1.2。没有代码文件。教程里建议把这里写成"小结 + 把函数收进 `storybot/tokenizer.py` + 在全量 TinyStories 上训练的结果展示"，并顺手引出"编码也要提速"。

### 2.6 `06_encode_optimize.py` —— 按优先级（rank）合并的编码器

**目的**：第 1 章的 `_encode_text` 对每个预分词把全部合并规则按训练顺序试一遍（`for merge_pair, new_id in self.merge_rules.items(): ids = merge(ids, ...)`），代价 `O(M · L)`。词表 10000 时每个预分词都要 9743 次 `merge()`。

**新算法**：`merge_rules` 的值 `new_id = 256 + 训练步数`，**本身就是优先级**（越小越早学到）。只看当前序列里实际存在的相邻对，挑 rank 最小的那个合并，重复直到没有可合并的对。

**签名**：`class BPETokenizer(merge_rules, end_token="<|endoftext|>")`；`load_from(filepath)`（staticmethod）；`_encode_text(text) -> list[int]`；`encode(input_text, show_progress=False) -> list[int]`；`decode(ids) -> str`。

**核心代码**（书中用 ❶–❹ 标注，说明正文逐步讲解）：

```python
def _encode_text(self, text):
    ids = list(text.encode("utf-8"))
    def get_merge_priority(pair):
        return self.merge_rules.get(pair, float('inf'))   # 不在规则里的对 = 最低优先级
    while len(ids) > 1:
        counts = count_pairs(ids)                          # ❶ 取得当前所有相邻对
        best_pair = min(counts, key=get_merge_priority)    # ❷ rank 最小的那个
        if best_pair not in self.merge_rules:              # ❸ 一个都不能合并 → 结束
            break
        new_id = self.merge_rules[best_pair]
        ids = merge(ids, best_pair, new_id)                # ❹ 执行合并
    return ids

def encode(self, input_text, show_progress=False):
    pattern = '(' + re.escape(self.end_token) + ')'       # 带捕获组：分隔符保留在结果里
    texts = re.split(pattern, input_text)
    for text in texts:
        if text == self.end_token: all_ids.append(self.end_token_id)
        else:
            for pretoken in pretokenize(text):
                all_ids.extend(self._encode_text(pretoken))
```

**为什么与第 1 章结果相同**（可以写进教程的简短证明）：
1. 规则 r 是 `(a, b) → 256+r`，a、b 必然在第 r 步之前就存在，所以 `a, b < 256 + r`；
2. 执行规则 r 之后，新出现的相邻对都含有 `256+r`，它们的 rank 都大于 r；贪心 `merge` 又保证不会残留 `(a, b)`；
3. 因此"当前存在的最小 rank"一定是"按顺序执行时下一个真正起作用的规则"。两种做法每一步都执行同一条规则、调用同一个 `merge`，结果逐 token 相同。

实测：valid 全量 5,465,873 个 token，ch01 编码器（切片上）、`06`、`storybot`、`07`、memo 扩展全部一致（§3 E6–E8）。`' accomplishment'` 一个词：ch01 做 9743 次 `merge()` 用 13.7–15.9 ms，`06` 只做 14 次、0.27–0.29 ms（F09，两次运行）。

**复杂度**：每个预分词 `O(L²)`（最多 L−1 轮，每轮 `count_pairs` + `merge` 都是 `O(L)`），与 M 无关（只是字典查找）。TinyStories 的预分词平均约 4 字节、最长 23 字节，所以极快。

**日文注释中译**：存在しないペアは最低優先度 → 不存在的相邻对优先级最低；現在のペアを取得（❶）→ 取得当前所有相邻对；最優先ペアを特定（❷）→ 找出优先级最高（rank 最小）的相邻对；マージ可能性の確認（❸）→ 确认能否合并；マージの実行（❹）→ 执行合并。

**实际运行（as-is：CodeBot 的 merge_rules.pkl 编码整个 tiny_codes.txt）**：

```text
100%|██████████| 43671/43671 [00:20<00:00, 2137.29it/s]
2677398
```

wall 21.1 s。43671 = 2 × 21,835 + 1（`re.split` 带捕获组时，n 个分隔符切出 2n+1 段）。**2,677,398 恰好等于仓库自带 `codebot/tiny_codes.bin` 的 5,354,796 字节 / 2**；另外逐个比对内容也**完全相同**（`np.array_equal` 为 True）——`06` 与 ch01/09 生成该 bin 时的编码一致（本机 CRLF 文件在文本模式下被还原成 LF，见 §6.1）。

### 2.7 `07_encode_parallel.py` —— 大文件并行编码 → `uint16` 二进制

**目的**：把 2.2 GB 文本编码成 541M 个 token 并落盘，供第 6 章 `np.memmap` 读取。

**签名**：`_encode_chunk(self, args: tuple[file_path, start, end, cache_dir, chunk_idx]) -> tuple[str, int]`；`encode_file(self, file_path, output_file, num_processes=8, num_chunks=64, cache_dir="bpe_cache") -> int`（返回 token 总数）。

**核心代码**：

```python
def _encode_chunk(self, args):
    file_path, start, end, cache_dir, chunk_idx = args
    with open(file_path, "rb") as f:
        f.seek(start)
        ids = self.encode(f.read(end - start).decode("utf-8", errors="ignore"))
    cache_file = os.path.join(cache_dir, f"chunk_{chunk_idx:05d}.npy")
    np.array(ids, dtype=np.uint16).tofile(cache_file)      # 原始二进制，不是 .npy 格式
    return cache_file, len(ids)

def encode_file(self, file_path, output_file, num_processes=8, num_chunks=64, cache_dir="bpe_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    try:
        chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
        ...
        with Pool(processes=num_processes) as pool:
            cache_results = list(tqdm(pool.imap(self._encode_chunk, chunk_info_list), ...))
        total_tokens = sum(r[1] for r in cache_results)
        arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(total_tokens,))
        idx = 0
        for cache_file in cache_files:                      # 按块顺序拼接
            chunk_data = np.fromfile(cache_file, dtype=np.uint16)
            arr[idx : idx + len(chunk_data)] = chunk_data
            idx += len(chunk_data)
        arr.flush(); del arr
    finally:
        shutil.rmtree(cache_dir)                            # 无论成败都删缓存目录
    return total_tokens
```

**设计要点**：
- 子进程只把"缓存文件名 + 长度"传回主进程，而不是把几百万个 id 的 list pickle 回来；两遍扫描（先求总长、再 memmap 写入）让最终文件一次分配到位。
- `imap` 保证结果按任务顺序返回，所以拼接顺序就是文件顺序；块边界在 `<|endoftext|>` 上，而 `encode` 先按 `<|endoftext|>` 切开，所以"分块编码再拼接 == 整体编码"（F13）。
- `uint16` 能装下 0–65535：StoryBot 10,000、WebBot 50,000 都够；词表超过 65,536 时 NumPy 1.26 会**静默回绕**（只给 DeprecationWarning），NumPy 2 会直接报错（F18）。
- `pool.imap(self._encode_chunk, ...)` 传的是绑定方法，每个任务都会把整个 tokenizer 对象 pickle 一次：实测 221,754 字节/任务，64 个任务共约 13.5 MiB（F24）。不算大，但这是"把大对象挂在 self 上再并行"的典型隐性开销。
- `cache_dir` 默认是相对路径 `bpe_cache`（相对于脚本 chdir 后的仓库根），`finally` 中 `shutil.rmtree(cache_dir)` 会删掉整个目录——**不要传一个已有内容的目录**。

**日文注释中译**：チャンクを処理してディスクにキャッシュ → 处理分块并缓存到磁盘；チャンクをエンコード → 对分块编码；キャッシュファイルに保存 → 保存为缓存文件；キャッシュディレクトリの準備 → 准备缓存目录；チャンクを並列処理でトークナイズしてキャッシュ → 并行地对各块分词并缓存；総トークン数を計算 → 计算 token 总数；memmapファイルを作成 → 创建 memmap 文件；バッチ処理でキャッシュからmemmapへ書き込み → 分批把缓存写入 memmap；キャッシュの削除 → 删除缓存；show_progressがTrueならtqdmで進捗表示 → show_progress 为 True 时用 tqdm 显示进度；各事前トークンをBPEエンコード → 对每个预分词做 BPE 编码。

**实际运行（补丁副本：只编码 valid，输出与缓存目录改到 `SCR/out/`）**：

```text
Encoding chunks: 100%|██████████| 64/64 [00:32<00:00,  2.00it/s]
ADDED: total tokens written = 5465873
```

wall 35.9 s（8 进程，负载高）。`cmp SCR/out/tiny_stories_valid_v07.bin storybot/tiny_stories_valid.bin` → **逐字节相同**；缓存目录已被 `finally` 删除。

### 2.8 `storybot/tokenizer.py` —— 本章定稿模块

内容 = `03/04` 的训练函数 + `07` 的完整 `BPETokenizer`，被 ch06（StoryBot 预训练/生成/DPO/评测）、ch07（WebBot 的 BPE 训练与编码）、ch09（WebBot 生成）复用。与 ch04 脚本的**差异**（`diff` 实测）：

| 项目 | ch04 脚本 | `storybot/tokenizer.py` | 影响 |
|---|---|---|---|
| `train_bpe(..., num_chunks=)` 默认值 | `04`: **64** | **8** | 8 块 = 每进程一块，无动态均衡；全量时每块约 278 MB，子进程内存可达 1+ GB（§5） |
| `encode_file(..., num_processes=)` 默认值 | `07`: **8** | **4** | ch07/02 显式传了 8，ch04 的 `07` 也传了 8 |
| `process_single_chunk(file_path, start, end, end_token)` | 无 | 有，但**全仓库无人调用** | 死代码（`pretoken_chunk` 的非元组参数版本） |
| 注释 | — | 多了"ステップ1–4"、"OpenWebTextの例のようにバッチ化" | 后者指向 nanoGPT `data/openwebtext/prepare.py` 的 memmap 分批写法（推断） |

模块无顶层副作用（没有 `os.chdir`），可以直接 `import`。函数签名与 `codebot/tokenizer.py` **不兼容**：`count_pairs(ids, weight=1, counts=None)` vs ch01 的 `count_pairs(ids, counts=None)`，按位置传 `counts` 会被当成 `weight`，抛 `TypeError`（F25）。另外 `counts` 必须是 `defaultdict(int)`，传普通 `{}` 会 `KeyError`。

追加的日文注释中译：1つのチャンクを処理する関数 → 处理单个分块的函数；ステップ2: 総トークン数を計算 → 第 2 步：计算 token 总数；ステップ3: memmapファイルを作成 → 第 3 步：创建 memmap 文件；ステップ4: バッチ処理でキャッシュからmemmapへ書き込み → 第 4 步：分批把缓存写入 memmap；OpenWebTextの例のようにバッチ化 → 像 OpenWebText 的例子那样分批处理；キャッシュは削除 → 缓存删除。

### 2.9 `storybot/download.sh`

```bash
cd "$(dirname "$0")"
wget -nc .../storybot/tiny_stories_train.txt      # 2,227,753,162 B（本档案未下载）
wget -nc .../storybot/tiny_stories_valid.txt      # 22,502,601 B（已下载）
# wget -nc .../storybot/tiny_stories_train.bin    # 1,082,458,446 B = 541,229,223 个 uint16
# wget -nc .../storybot/tiny_stories_valid.bin    # 10,931,746 B   = 5,465,873 个 uint16（已下载，用于校验）
# wget -nc .../storybot/merge_rules.pkl           # 103,877 B，9743 条合并（已下载）
# wget -nc .../storybot/model_pretrain.pt / model_iter_500.pt / model_iter_5000.pt / model_dpo.pt
```

（前缀均为 `https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main`；大小来自 `curl -sIL` 的 `x-linked-size`。）

- 默认只下载两份文本，**分词器与 .bin 默认被注释掉**：作者的意图是读者自己跑第 4 章生成它们；跑不动的读者可以取消注释直接下载。
- `wget -nc` 不覆盖已有文件。Windows 没有 wget 时可用 `curl -L -o <名> <URL>`。
- `.pkl` 是 pickle，加载会执行代码的风险理论上存在；本档案先用 `pickletools.genops` 检查过：只有 `EMPTY_DICT/BININT/TUPLE2/SETITEMS` 等纯数据操作码，无 `GLOBAL/REDUCE`（`SCR/check_pickle.py`）。教程可以顺带教这个习惯。
- 与 CS336：这两个 TinyStories 文件的字节数与 CS336 A1 使用的 `TinyStoriesV2-GPT4-{train,valid}.txt` 一致（valid 也是从半句 `"u don't have to be scared of the loud dog, I'll protect you"` 开始）；`webbot/owt_{train,valid}.txt` 也与 CS336 的 OWT 样本同大小。


---

## 3. 实测：速度与"结果完全相同"

### 3.0 方法

- **不改仓库代码**。`SCR/mkmods.py` 用 `ast` 把每个 ch04 脚本里的 import/函数/类**按原始行**抽成可导入模块 `SCR/mods/v01_dedup.py … v07_encode_parallel.py`（保留注释，丢掉 `os.chdir` 和顶层训练代码）；多进程在 Windows spawn 下需要可导入的真实模块，所以必须落成文件。
- `SCR/bench_train.py` 把各模块里的 `tqdm` 换成一个计时包装器，从而**用书中原代码路径**拿到"预分词"和"merge 循环"两段的 wall 与 CPU 时间，并把 merges 逐项比较（同一 pair、同一 id、同一顺序）。
- `SCR/bench_encode.py` 对编码器做同样的事，并与官方 `.bin` 比较，最后检查 `decode(encode(text)) == text`。
- "朴素"= `codebot/tokenizer.py`（第 1 章版本）。"LF tiny_codes"= `git show HEAD:codebot/tiny_codes.txt`（6,487,033 B），用来规避本机 CRLF 检出（§5）。

### 3.1 训练：merge 循环的加速（单进程，CPU 秒）

| 实验 | 语料 / 词表 | 朴素（ch01） | `01` 去重 | `02` 增量 | `03` 分块 | `04` 并行 | storybot | 结果 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E1 | TinyStories valid 前 499,577 B / 1000 | **198.80** | 15.67 | **1.66** | 1.61 | 1.48 | 1.55 | 六者**完全相同** |
| E2 | LF tiny_codes 全量 / 1000 | **3,423.75**（77 分钟 wall） | 122.30 | **5.77** | 7.19 | 6.67 | 4.47 | 六者完全相同，且都 = 仓库自带 `codebot/merge_rules.pkl` |
| E3 | TinyStories valid 全量 / 10000 | （估计数小时，未跑） | （未跑完整） | 62.62 | 64.73 | 63.91 | 67.14 | 四者相同 |
| E4 | 训练集前 100 MiB / 10000 | — | — | — | — | 136.38 | — | 与官方 merges 前 31 条相同 |

- E1（同一进程顺序执行）：**朴素 → `01` 12.7×，`01` → `02` 9.4×，朴素 → `02` 120×**。
- E2（朴素单独在后台跑、其余在前台跑，负载都很高）：**朴素 → `01` 28.0×，`01` → `02` 21.2×，朴素 → `02` 约 590×**。朴素版的 merge 循环用了 57 分钟 CPU / 77 分钟 wall。
- "去重后每步要处理的字节数"给出的预测：tiny_codes 27.1×（实测 28.0×），500 KB 切片 21.6×（实测 12.7×）。见下方复测说明：字节比只是粗略上界，不能当精确预测。
- E2 as-is 运行（负载约 84%）：`01` 的 merge 循环 89 s（tqdm 1:29），`02` 3.7 s，约 24×。
- E3/E4 的 merge 循环在负载较低时（`04` 补丁副本单独运行）为 27 s wall；负载 100% 时 62–67 s CPU。
- `03`/`04`/storybot 的 merge 循环与 `02` 是同一段代码，差异纯属测量噪声。

**复测说明（500 KB 切片，负载约 50%，同一进程交错执行 `01 → 朴素 → 01 → 02`）**：

```text
v01    merge_loop=  11.05s(cpu   9.92)
naive  merge_loop= 129.67s(cpu 121.42)
v01    merge_loop=   9.01s(cpu   8.73)
v02    merge_loop=   1.14s(cpu   1.03)
IDENTICAL (same pairs, same ids, same order): naive == v01
IDENTICAL (same pairs, same ids, same order): v02 == v01
```

朴素 → `01` 为 13.0×（第一次 12.7×），`01` → `02` 为 9.1×（第一次 9.4×），朴素 → `02` 118×（第一次 120×）——比值可复现。按字节比预测的 21.6× 偏高，原因是去重版对每个唯一序列每步还要多做 `tuple(new_ids)` 复制和字典哈希（都是逐元素的开销），而这 500 KB 切片里唯一序列只有 3,523 个、平均 6.5 字节，这些固定开销占比大。tiny_codes 上的 28.0× 是朴素版单独在后台、跨越 77 分钟不同负载时测得的，可靠性不如这组交错测量。教程里建议说"加速比与字节比同一量级"，不要说"等于字节比"。

### 3.2 训练：预分词阶段与并行（wall 秒）

| 实验 | 语料 | 串行（`02` 整读） | `03` 分块串行 | `04` Pool(8) | 备注 |
|---|---|---:|---:|---:|---|
| as-is/补丁运行 | valid 22.5 MB | — | 14.5（tqdm 0:14） | 1.4（tqdm 0:01，另有约 1.45 s 进程启动） | 负载约 84% |
| bench（E3） | valid 22.5 MB | 9.05 | 14.23 | 8.66 | 负载 100%，并行几乎被抵消 |
| bench（E4） | 训练前 100 MiB | — | — | 14.48 | 单进程吞吐约 1.36 MB/s（`heaps.py` 实测 77 s） |

`SCR/scaling.py`（书中 `pretoken_chunk` + `Pool.imap`，64 块，计入进程启动；每次都校验计数与串行相同）：

| 进程数 | Pool 启动+关闭 | 预分词（含启动） | 加速 | `encode_file`（valid 全量，官方 merges） | 加速 | 输出 |
|---:|---:|---:|---:|---:|---:|---|
| 串行（无 Pool） | — | 12.38 s（1.82 MB/s） | 1.0× | — | — | — |
| 1 | 0.44 s | 11.12 s | 1.1× | 163.64 s | 1.0× | 相同 |
| 2 | 0.46 s | 6.75 s | 1.8× | 89.20 s | 1.8× | 相同 |
| 4 | 0.55 s | 3.69 s | 3.4× | 47.20 s | 3.5× | 相同 |
| 8 | 0.70 s | 3.01 s | **4.1×** | 30.44 s | **5.4×** | 相同 |
| 16 | 1.97 s | 4.51 s | 2.7× | 22.14 s | **7.4×** | 相同 |

（`SCR/out/scaling2.log`；开始时负载 97%、结束时 77%。另一次负载约 90% 的预分词测量：串行 13.51 s，8 进程 4.13 s（3.3×），16 进程 3.69 s（3.7×）。）

**解读**：Windows 用 spawn 启动子进程（重新 import `regex`、`tqdm` 等），Pool 本身就要 0.4–2.3 s；valid 只有 22.5 MB，启动开销占比很大，加上同机其它任务争用，16 进程只有 2.7–3.7 倍（两次测量）。编码比预分词更"重"（每字节 CPU 更多），所以 `encode_file` 的并行效率更高（16 进程 7.4×）。语料越大，这部分越可忽略。小语料时并行反而更慢：E1 的 500 KB 切片上，`04`（4 进程）预分词 0.74 s，串行 `02` 只要 0.23 s。

### 3.3 `02` 之后的瓶颈 + 扩展：惰性最大堆（书中没有）

`SCR/analyze_v02.py` 逐行复刻 `02` 的循环并分段计时，再用一个"惰性删除"的最大堆替换 `max()`（堆元素 `(-count, -id0, -id1)`，与书中 key `(count, id0, id1)` 的全序一致；每步把计数变化过的 pair 各推一次，弹出时若计数对不上就丢弃）：

| 语料 / 词表 | 书中循环 CPU | 其中 `max()` | 堆版本 CPU | 加速 | merges |
|---|---:|---:|---:|---:|---|
| tiny_codes / 1000 | 11.38 s | 5.97 s（52%） | 6.89 s | 1.7× | 完全相同 |
| valid / 10000 | 70.78 s | 68.45 s（**97%**） | 3.03 s | **23.4×** | 完全相同 |
| 训练前 100 MiB / 10000 | 135.80 s | 131.64 s（**97%**） | 5.55 s | **24.5×** | 完全相同 |

注意：最初的写法"每次计数变化都 push 一次"反而更慢（tiny_codes 上 0.6×）；改成"每步每个变化的 pair 只 push 一次"后才变快。这个反例本身就是很好的教学材料：渐进复杂度好 ≠ 常数小。

### 3.4 平票（tie-break）实验：为什么那个 key 是"优化安全"的前提

`SCR/tiebreak.py`，LF tiny_codes，词表 1000（743 步）：

```text
A book key: identical to reference: True; steps whose max count was tied by >=2 pairs: 88/743 (11.8%)
  A(book: ids) vs B(CS336: bytes): first divergence at merge #289: b'ctor' vs b'pre'; learned-token sets differ by 2 tokens; same token set: False
  C(incremental, no tie-break) vs D(full recount, no tie-break): first divergence at merge #301: b'nte' vs b'class'; learned-token sets differ by 8 tokens; same token set: False
  D(full recount, no tie-break) vs A(book): first divergence at merge #122: b'))' vs b've'; learned-token sets differ by 0 tokens; same token set: True
```

- 11.8% 的步骤里最大频次是**并列**的。
- 若像 ch01 早期脚本那样用 `max(counts, key=counts.get)`，平票取决于字典的插入顺序；增量版（`02`）里 pair 不断被删除再插入，顺序与全量重算版不同 → **同一个算法、两种实现，第 301 步开始结果不同**。
- 书中的 key `(count, pair[0], pair[1])` 是一个**全序**，`max` 的结果与字典顺序无关 → 所有实现逐项相同。
- CS336 规定平票时比较 **bytes 的字典序**（`(count, vocab[a], vocab[b])`），与 DLFS6 比较 **id** 不同：单字节 token 两者等价，多字节 token 不等价 → 第 289 步起分叉。若教程要与 CS336 的测试对齐，需要改这一行 key。

### 3.5 内存：为什么要分块（`SCR/mem_peak.py`、`SCR/mem_chunk.py`，psutil `peak_wset`）

| 方式 | 21.5 MiB（valid） | 100 MiB（训练前缀） |
|---|---:|---:|
| `02` 整读 + `split` + 预分词（`train_bpe(..., 257)` 即 0 次合并） | 98.7 MiB | **656.0 MiB** |
| `03` 分块（64 块）串行 | 31.8 MiB | 42.4 MiB |
| `04` 并行（父进程） | 47.6 MiB | 65.9 MiB |

单个 `pretoken_chunk` 子进程的峰值随块大小线性增长：100 MiB 块 670.9 MiB、13 MiB 块 89.8 MiB、1.6 MiB 块 43.3 MiB（进程基线约 36 MiB）。

**原因是 PEP 393**：CPython 的 `str` 按全串最宽字符决定 1/2/4 字节/字符。valid 最宽字符是 `'…'`（U+2026）→ 整串 2 B/字符（42.9 MiB）；训练前 100 MiB 里有 `'🍌'`（U+1F34C）→ 整串 **4 B/字符（399.8 MiB）**。而按 `<|endoftext|>` 切开后只有 2.7% 的故事需要宽存储，切片总和 106.2 MiB。`"a"*10**6` 占 1,000,041 B，末尾加一个 emoji 就变成 4,000,064 B。

外推到全量 2.2 GB：整读至少约 8.9 GB 的 `str` + 解码时的 bytes + `split` 副本 → **十几 GB**；`04`（64 块，每块约 35 MB）每个子进程约 0.2 GB；`storybot.train_bpe` 默认 `num_chunks=8`（每块约 278 MB）每个子进程约 1.1–1.8 GB，8 个进程合计可达 9–14 GB。

### 3.6 编码：速度与一致性

| 实验 | 文本 / 规则 | ch01 朴素 | `06` | storybot | `07`（Pool） | memo（扩展） | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| E6 | valid 前 29,642 B / 9743 条 | **40.31 s CPU** | 0.23 s | 0.20 s | 2.36 s wall（4 进程，启动开销为主） | 0.06 s | 五者相同；7,294 tokens；往返相等 |
| E7 | LF tiny_codes 前 329,925 B / 743 条 | **59.42 s CPU** | 2.05 s | — | — | 0.70 s | 三者相同；154,190 tokens |
| E8 | valid 全量 / 9743 条 | （按 E6 外推约 8.6 小时） | 144.9 s CPU（227 s wall） | — | **45.4 s wall（8 进程）** | 12.7 s CPU | 全部相同，且 = 官方 `tiny_stories_valid.bin` |
| as-is | `06` 主程序：tiny_codes 全量 / 743 条 | （ch01/09，按 E7 外推约 20 分钟） | 20.4 s（tqdm） | — | — | — | 2,677,398 个，与 `codebot/tiny_codes.bin` 逐个相同 |

- 加速比随词表增大：743 条规则时 `06` 比 ch01 快约 29×，9743 条时约 175×（CPU 时间比）。
- **memo 扩展**（`dict` 缓存"预分词 → ids"，书中没有）：valid 只有 13,111 个不同预分词，而且 99.74% 的出现最终就是 1 个 token（F16），所以缓存命中率极高，单进程就比 `06` 快约 11×，甚至快过 `07` 的 8 进程。教程可以把它当作"下一步优化"的练习。
- 对照 tiktoken（Rust 实现，GPT-2 词表）在同一份 valid 上单次 `encode` 约 8.5 s（负载高）。

### 3.7 与官方产物的端到端对照

| 对照 | 结果 |
|---|---|
| `07` 补丁副本输出 vs 官方 `storybot/tiny_stories_valid.bin` | `cmp` 逐字节相同（10,931,746 B） |
| `bench_encode` 中 `06`/`07`/memo vs 官方 valid.bin | 5,465,873 个 id 完全相同 |
| `07`（8 进程）编码训练集前 104,856,833 B vs 官方 `tiny_stories_train.bin` 的前 50,946,778 B（Range 下载） | `cmp` 逐字节相同（25,473,389 个 token，114 s）；官方文件第 25,473,389 个 token 是 9999，正是截断处的 `<\|endoftext\|>` |
| `02` 在 LF tiny_codes 上的 743 条 merges vs 仓库 `codebot/merge_rules.pkl` | 完全相同（含顺序） |
| `06` 编码整个 tiny_codes（文本模式）vs `codebot/tiny_codes.bin` | 2,677,398 = 5,354,796 / 2，且 `np.array_equal` 为 True |
| 自己在 valid（22.5 MB）上训练的 merges vs 官方（在 2.2 GB 上训练） | 前 5 条相同；学到的 token 集合重叠 8,541/9,743（87.7%）；前 1000 个 token 重叠 977 |
| 自己在训练前 100 MiB 上训练 vs 官方 | 前 31 条相同；重叠 9,160/9,743（94.0%）；前 1000 个重叠 990 |
| 最长 token | 自己（valid）与官方都是 `' accomplishment'`（官方 id 7159，15 字节） |
| 压缩率（valid） | StoryBot 10k：4.117 B/token；GPT-2（50,257）：4.067；cl100k（100,277）：4.179 |


---

## 4. 可最小化验证的事实清单

**怎么跑**：所有片段由 `SCR/facts.py` 在**全新命名空间**中逐个 `exec`（工作目录 = 仓库根），记录原样代码、stdout 与耗时；下面"附 4.B"是该脚本输出的原文。命令：

```bash
cd C:/Users/chenf/Projects/_refs/dlfs6
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python <SCR>/facts.py          # 全部 27 条
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python <SCR>/facts.py F08 F09  # 只跑指定几条
```

**浏览器可行性是实测过的**：Pyodide 0.29.4（Node 中加载；Python 3.13.2、`regex` 可用、NumPy 2.2.5）里跑了 F01、F04、F26 和一个 `02` 风格的小训练，输出与 CPython 完全相同（`SCR/pyodide_test/`）。另外 GPT-2 预分词正则的 **JS 版本**（`/…/gu`）在整份 valid 上与 Python `regex` 逐个相同（5,419,001 个预分词）以及 12 个刁钻输入（NEL、BOM、NBSP、全角空格、emoji、中文、CRLF……）也相同（`SCR/pretok.mjs`、`SCR/js_parity.py`）。所以交互组件既可以用 Pyodide 跑书中原代码，也可以纯 JS 实现。
"需数据"表示需要先把对应文件放进 Pyodide 的虚拟文件系统（`merge_rules.pkl` 104 KB；`tiny_stories_valid.bin` 10.9 MB；valid 文本 22.5 MB，建议在浏览器里用 1–2 MB 的切片）。

### 4.A 一览表

| ID | 断言 | 观测（本机实跑） | 耗时 | 浏览器 |
|---|---|---|---:|---|
| F01 | GPT-2 正则：前导空格贴在词上；多个空格时 `\s+(?!\S)` 留一个给下一个词；弯引号 `’` 不算缩写 | `['Hello', ' world', '!', ' It', "'s", ' 2024', '.', '  ', ' Go']`；`['don', '’', 't']` | 0.8 s | ✅（Pyodide 实测相同；或 JS 正则） |
| F02 | 标准库 `re` 不支持 `\p{L}`，所以书里用第三方 `regex` | `re.error: bad escape \p at position 0`；`['naïve', '世界']` | <0.01 s | ✅ |
| F03 | 按次数加权计数 == 把出现展开后计数 | `True`，最高对 `(101, 115)`=`'es'` 计 9 次 | <0.01 s | ✅ |
| F04 | `merge` 是贪心、从左到右、不重叠 | `[256, 97]`、`[256, 256]`、`[4, 3, 4]` | <0.01 s | ✅（实测） |
| F05 | key `(count, id0, id1)` 与字典顺序无关；`max(key=get)` 取先插入者 | 两种插入顺序下书中 key 都选 `(99, 100)`；朴素 `max` 分别选 `(99,100)` 和 `(97,98)` | <0.01 s | ✅ |
| F06 | 玩具语料上 ch01 朴素 == `01` == `02` | 15 条合并、三者相同；`['st','est','ow','low','west','ne','newest',' newest',…]` | 0.01 s | ✅ |
| F07 | 词表账本：256 + 合并数 + 1 = vocab；`end_token_id = 256 + len(merges)` | `9743 9999 10000 9743`；最长 token `(7159, b' accomplishment')` | 0.02 s | ✅ 需数据 |
| F08 | 一句话的确切 id，往返无损 | `[429, 438, 258, 397, 44, 400, 282, 258, 389, 476, 401, 405, 46, 9999, 409, 869, 46]`；74 字节 → 17 token；`True` | 0.02 s | ✅ 需数据 |
| F09 | rank 编码器 == 逐条规则编码器，且轮数少得多 | `' accomplishment'` → `[7159]`；ch01 9743 次 merge 15.9 ms vs 14 次 0.271 ms（另一次运行 13.7 / 0.289） | 0.03 s | ✅ 需数据 |
| F10 | 特殊 token：训练时剥离、编码时映射为一个 id；学到的 token 里不含它 | `['a', '<\|endoftext\|>', 'b']`；`[1095, 46, 9999, 3868, 46]`；`False`；残缺的 `<\|endoftext` 只是 6 个普通 token | 0.02 s | ✅ 需数据 |
| F11 | 块边界恰好落在 `<\|endoftext\|>` 上；块数 > 文档数时去重 | 65 个边界 `[0, 351752, 703494, 1055567]…`，内部全部命中；要 100,000 块只得到 27,629 个边界 | 0.7 s | ✅ 需数据（用切片） |
| F12 | 分块计数 == 整文件计数 | `True 5419001 13111` | 22.6 s | ⚠️ 浏览器请用 1–2 MB 切片 |
| F13 | 在 `<\|endoftext\|>` 处切开再编码 == 整体编码；在词中间切开则不等 | `True`；`False ' lit' \| 'tle'` | 5.3 s | ✅ 需数据 |
| F14 | 语料复制一遍（用 `<\|endoftext\|>` 相连）merges 不变：只有相对频次重要 | `True 743` | 4.0 s | ✅（建议更小切片） |
| F15 | `02` 的不变式：增量计数每一步都等于全量重算 | 300 步全部相等：`True` | 3.4 s | ✅ |
| F16 | valid 中 99.74% 的预分词出现编码后恰好 1 个 token；按唯一预分词求和可精确复原官方 token 总数 | `single-token share=0.9974`；`5,438,243 + 27,630 = 5,465,873` | 13.7 s | ⚠️ 用切片 |
| F17 | 官方 valid.bin = 5,465,873 个 uint16，最大 id 9999，EOT 27,630 个，4.117 B/token | `(5465873,) uint16 9999 27630`；`4.117` | 0.02 s | ✅ 需数据（10.9 MB） |
| F18 | 词表 > 65,536 时 uint16 的陷阱 | NumPy 1.26.4：`[65535 0 4464]` + DeprecationWarning（静默回绕）；**Pyodide 的 NumPy 2.2.5：`OverflowError`** | <0.01 s | ✅（两种行为都实测） |
| F19 | TinyStories 分词器处理中文 ≈ 每字节 1 个 token | `'你好，世界'` 5 字 15 字节 → 15 token；`'Hello, world'` → 3 token | 0.01 s | ✅ 需数据 |
| F20 | 文本模式隐藏 CRLF，二进制模式不隐藏 | 原始字节 256,501 个 `\r\n`；文本模式 0 个 `\r` | 0.04 s | ❌（依赖 Windows git 检出；浏览器可用字符串模拟） |
| F21 | `del pair_to_ids[best]` 后再访问会悄悄新建空集合 | `{(1, 2, 3)} {(1, 2): set()}` | <0.01 s | ✅ |
| F22 | `07` 的缓存文件名叫 `.npy` 但其实是 `tofile` 原始字节 | `np.fromfile` 得 `[1 2 9999]`；`np.load` → `ValueError … allow_pickle=False` | 0.01 s | ✅ |
| F23 | 没有相邻对时训练提前结束，词表会比要求的小 | `"aaaa aaaa"` 要 1000 只得到 3 条合并；`end_token_id = 259`、`vocab_size = 260` | <0.01 s | ✅ |
| F24 | `07` 每个 Pool 任务都要 pickle 整个 tokenizer | 221,754 B/任务；64 任务约 13.5 MiB | 0.02 s | ✅ 需数据 |
| F25 | `count_pairs` 签名不兼容：codebot `(ids, counts)` vs storybot `(ids, weight, counts)` | 按位置传 → `TypeError: unsupported operand type(s) for +=: 'int' and 'collections.defaultdict'` | <0.01 s | ✅ |
| F26 | CS336 讲义的玩具语料，用书中训练器得到同样的前 6 条合并 | `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']` | <0.01 s | ✅（Pyodide 实测相同） |
| F27 | 扩展：预分词 → ids 的 memo 字典，结果不变 | `True 2925 memo entries`（300 KB 切片） | 1.8 s | ✅ 需数据 |

**超过 30 秒、但值得在教程里给出命令的"大验证"**（全部已实跑，输出见 §3）：

| ID | 验证 | 命令（在仓库根执行，前缀 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python`） | 本机耗时 |
|---|---|---|---|
| E1 | 朴素 = 01 = 02 = 03 = 04 = storybot（TinyStories 500 KB，词表 1000） | `SCR/bench_train.py SCR/out/valid_500k.txt 1000 naive,v01,v02,v03,v04,story --procs 4` | 约 5 分钟（朴素占 4 分多） |
| E2 | 02 在 LF tiny_codes 上 = 仓库 `codebot/merge_rules.pkl` | `SCR/bench_train.py SCR/tiny_codes_lf.txt 1000 v02,v03,v04,story` + 比较 pkl | 约 1 分钟 |
| E3 | valid 全量词表 10000：02 = 03 = 04 = storybot | `SCR/bench_train.py storybot/tiny_stories_valid.txt 10000 v02,v03,v04,story` | 约 6 分钟（负载 100%） |
| E5 | 堆扩展与书中循环 merges 相同、快 23× | `SCR/analyze_v02.py storybot/tiny_stories_valid.txt 10000 <ref.pkl>` | 约 2 分钟 |
| E8 | 编码 valid 全量 = 官方 valid.bin | `SCR/patched/07_encode_parallel.py` 后 `cmp` | 约 40 s（8 进程） |
| E9 | 编码训练集前 100 MiB = 官方 train.bin 的对应前缀 | `SCR/encode_prefix.py`，再 `curl -r 0-50946777` 官方 bin 后 `cmp` | 约 2 分钟（8 进程）+ 51 MB 下载 |
| T1 | 平票实验 | `SCR/tiebreak.py SCR/tiny_codes_lf.txt 1000 codebot/merge_rules.pkl` | 约 5 分钟 |
| H1 | Heaps 定律外推 | `SCR/heaps.py <train 前缀> <输出> 2227753162` | 约 80 s |


### 4.B 附：27 条事实的原样代码与输出（`SCR/facts.py` 生成，未经手工编辑）

> 每条标题末尾括号里是本机 wall 时间。路径 `C:\Users\chenf\AppData\…\mods` 指 `SCR/mods`（由 `SCR/mkmods.py` 从 ch04 脚本原样抽出的函数）。

#### F01 GPT-2 pretokenizer: leading space sticks to the word; \s+(?!\S) leaves one space for the next word  (0.77s)
```python
from storybot.tokenizer import pretokenize
print(list(pretokenize("Hello world! It's 2024.   Go")))
print(list(pretokenize("don’t")), "<- curly apostrophe is NOT a contraction")
```
```text
['Hello', ' world', '!', ' It', "'s", ' 2024', '.', '  ', ' Go']
['don', '’', 't'] <- curly apostrophe is NOT a contraction
```

#### F02 stdlib re cannot do \p{L}; that is why the book uses the third-party `regex`  (0.00s)
```python
import re
try:
    re.compile(r"\p{L}")
except re.error as e:
    print("re.error:", e)
import regex
print(regex.findall(r"\p{L}+", "naïve 世界 2024"))
```
```text
re.error: bad escape \p at position 0
['naïve', '世界']
```

#### F03 count_pairs with weight == counting the expanded occurrence list  (0.00s)
```python
from collections import Counter, defaultdict
from storybot.tokenizer import count_pairs
words = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
weighted = defaultdict(int)
for w, c in words.items():
    count_pairs(tuple(w.encode()), c, weighted)
expanded = Counter()
for w, c in words.items():
    for _ in range(c):
        expanded.update(zip(w.encode(), w.encode()[1:]))
print(dict(weighted) == dict(expanded), "| top:", max(weighted, key=weighted.get), weighted[max(weighted, key=weighted.get)])
```
```text
True | top: (101, 115) 9
```

#### F04 merge() is greedy, left-to-right, non-overlapping  (0.00s)
```python
from storybot.tokenizer import merge
print(merge([97, 97, 97], (97, 97), 256))
print(merge([97, 97, 97, 97], (97, 97), 256))
print(merge([1, 2, 3, 1, 2], (1, 2), 4))
```
```text
[256, 97]
[256, 256]
[4, 3, 4]
```

#### F05 Tie-break key (count, id0, id1): equal counts -> the pair with larger ids wins; plain max(key=get) depends on insertion order  (0.00s)
```python
pc = {(99, 100): 3, (97, 98): 3, (120, 121): 2}
print("book key :", max(pc, key=lambda p: (pc[p], p[0], p[1])))
print("plain max:", max(pc, key=pc.get), "<- first inserted among ties")
pc2 = {(97, 98): 3, (99, 100): 3, (120, 121): 2}   # same counts, different insertion order
print("plain max:", max(pc2, key=pc2.get), "| book key:", max(pc2, key=lambda p: (pc2[p], p[0], p[1])))
```
```text
book key : (99, 100)
plain max: (99, 100) <- first inserted among ties
plain max: (97, 98) | book key: (99, 100)
```

#### F06 Naive (ch01 list) == ch04/01 dedup == ch04/02 incremental on a toy corpus  (0.01s)
```python
import sys; sys.path.insert(0, r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch04-bpe\mods")
import codebot.tokenizer as naive, v01_dedup as v01, v02_cache as v02
for m in (naive, v01, v02): m.tqdm = lambda it, **k: it      # silence progress bars
text = ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3 + "<|endoftext|>") * 2
r0, r1, r2 = naive.train_bpe(text, 300), v01.train_bpe(text, 300), v02.train_bpe(text, 300)
print(len(r0), list(r0.items()) == list(r1.items()) == list(r2.items()))
inv = {v: k for k, v in r2.items()}
tb = lambda i: bytes([i]) if i < 256 else tb(inv[i][0]) + tb(inv[i][1])
print([tb(i).decode() for i in sorted(inv)][:12])
```
```text
15 True
['st', 'est', 'ow', 'low', 'west', 'ne', 'newest', ' newest', ' low', 'wi', 'wid', 'widest']
```

#### F07 Vocabulary bookkeeping: 256 bytes + merges + 1 special = vocab_size; end_token_id = 256 + len(merges)  (0.02s)
```python
import pickle
from storybot.tokenizer import BPETokenizer
rules = pickle.load(open("storybot/merge_rules.pkl", "rb"))
tok = BPETokenizer(rules)
print(len(rules), tok.end_token_id, tok.vocab_size, 10000 - 256 - 1)
print([tok.id_to_bytes[i] for i in range(256, 266)])
L = max(tok.id_to_bytes.items(), key=lambda kv: len(kv[1]) if kv[0] != tok.end_token_id else 0)
print("longest learned token:", L)
```
```text
9743 9999 10000 9743
[b' t', b'he', b' a', b' s', b' w', b'nd', b' the', b'ed', b' b', b' to']
longest learned token: (7159, b' accomplishment')
```

#### F08 Exact ids for a sentence with the official StoryBot merges + exact round trip  (0.02s)
```python
import pickle
from storybot.tokenizer import BPETokenizer
tok = BPETokenizer(pickle.load(open("storybot/merge_rules.pkl", "rb")))
s = "Once upon a time, there was a little girl named Lily.<|endoftext|>The end."
ids = tok.encode(s)
print(ids)
print([tok.decode([i]) for i in ids])
print(tok.decode(ids) == s, len(s.encode()), "bytes ->", len(ids), "tokens")
```
```text
[429, 438, 258, 397, 44, 400, 282, 258, 389, 476, 401, 405, 46, 9999, 409, 869, 46]
['Once', ' upon', ' a', ' time', ',', ' there', ' was', ' a', ' little', ' girl', ' named', ' Lily', '.', '<|endoftext|>', 'The', ' end', '.']
True 74 bytes -> 17 tokens
```

#### F09 Rank-based encoder (ch04/06) == apply-every-merge encoder (ch01), and it does far fewer passes  (0.03s)
```python
import pickle, time
import codebot.tokenizer as c1
from storybot.tokenizer import BPETokenizer, count_pairs, merge
rules = pickle.load(open("storybot/merge_rules.pkl", "rb"))
old, new = c1.BPETokenizer(rules), BPETokenizer(rules)
w = " accomplishment"
t = time.perf_counter(); a = old._encode_text(w); t_old = time.perf_counter() - t
t = time.perf_counter(); b = new._encode_text(w); t_new = time.perf_counter() - t
ids, passes = list(w.encode()), 0
while len(ids) > 1:
    best = min(count_pairs(ids), key=lambda p: rules.get(p, float("inf")))
    if best not in rules: break
    ids, passes = merge(ids, best, rules[best]), passes + 1
print(a, b, a == b)
print(f"ch01: {len(rules)} merge() passes, {t_old*1e3:.1f} ms | ch04/06: {passes} passes, {t_new*1e3:.3f} ms")
```
```text
[7159] [7159] True
ch01: 9743 merge() passes, 15.9 ms | ch04/06: 14 passes, 0.271 ms
```

#### F10 Special token: training strips it, encoding maps it to one id; no learned token contains it  (0.02s)
```python
import pickle, re
from storybot.tokenizer import BPETokenizer
tok = BPETokenizer(pickle.load(open("storybot/merge_rules.pkl", "rb")))
print(re.split("(" + re.escape("<|endoftext|>") + ")", "a<|endoftext|>b"))
print(tok.encode("Hi.<|endoftext|>Bye."))
print(any(b"<|" in v for k, v in tok.id_to_bytes.items() if k != tok.end_token_id))
print(tok.encode("<|endoftext"), "<- a broken marker is just text")
```
```text
['a', '<|endoftext|>', 'b']
[1095, 46, 9999, 3868, 46]
False
[60, 124, 352, 111, 790, 3272] <- a broken marker is just text
```

#### F11 find_chunk_boundaries: every internal boundary sits exactly on b'<|endoftext|>'  (0.71s)
```python
from storybot.tokenizer import find_chunk_boundaries
p = "storybot/tiny_stories_valid.txt"
b = find_chunk_boundaries(p, 64)
data = open(p, "rb").read()
print(len(b), b[:4], b[-1] == len(data))
print(all(data[x:x + 13] == b"<|endoftext|>" for x in b[1:-1]))
print(find_chunk_boundaries(p, 100000)[:3], len(find_chunk_boundaries(p, 100000)), "<- duplicates removed when chunks > stories")
```
```text
65 [0, 351752, 703494, 1055567] True
True
[0, 242, 1066] 27629 <- duplicates removed when chunks > stories
```

#### F12 Chunked pretoken counts == whole-file pretoken counts (so merges are identical)  (22.60s)
```python
from collections import Counter
from storybot.tokenizer import find_chunk_boundaries, pretoken_chunk, pretokenize
p, eot = "storybot/tiny_stories_valid.txt", "<|endoftext|>"
whole = Counter()
for t in open(p, encoding="utf-8").read().split(eot):
    whole.update(pretokenize(t))
b = find_chunk_boundaries(p, 64)
chunked = Counter()
for s, e in zip(b[:-1], b[1:]):
    chunked.update(pretoken_chunk((p, s, e, eot)))
print(whole == chunked, sum(whole.values()), len(whole))
```
```text
True 5419001 13111
```

#### F13 Encoding chunk-by-chunk (split at <|endoftext|>) == encoding the whole text  (5.29s)
```python
import pickle
from storybot.tokenizer import BPETokenizer
tok = BPETokenizer(pickle.load(open("storybot/merge_rules.pkl", "rb")))
text = open("storybot/tiny_stories_valid.txt", encoding="utf-8").read()[:200000]
cut = text.rfind("<|endoftext|>", 0, 100000)
print(tok.encode(text) == tok.encode(text[:cut]) + tok.encode(text[cut:]))
mid = text.find(" little", 100000) + 4   # a cut inside a word (" lit|tle") is NOT safe:
print(tok.encode(text) == tok.encode(text[:mid]) + tok.encode(text[mid:]), repr(text[mid-4:mid]), "|", repr(text[mid:mid+3]))
```
```text
True
False ' lit' | 'tle'
```

#### F14 Doubling the corpus (joined by <|endoftext|>) gives identical merges: only relative counts matter  (3.96s)
```python
import sys; sys.path.insert(0, r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch04-bpe\mods")
import v02_cache as v02
v02.tqdm = lambda it, **k: it
t = open("storybot/tiny_stories_valid.txt", encoding="utf-8").read()[:300000]
a = v02.train_bpe(t, 1000)
b = v02.train_bpe(t + "<|endoftext|>" + t, 1000)
print(list(a.items()) == list(b.items()), len(a))
```
```text
True 743
```

#### F15 Incremental pair counts stay exactly equal to a full recount after every merge (ch04/02's invariant)  (3.35s)
```python
from collections import defaultdict
from storybot.tokenizer import pretokenize, count_pairs, merge
text = open("storybot/tiny_stories_valid.txt", encoding="utf-8").read()[:200000]
pc0 = defaultdict(int)
for t in text.split("<|endoftext|>"):
    for p in pretokenize(t): pc0[p] += 1
ids_counts = {tuple(p.encode()): c for p, c in pc0.items()}
pair_counts, pair_to_ids = defaultdict(int), defaultdict(set)
for ids, c in ids_counts.items():
    count_pairs(ids, c, pair_counts)
    for pr in zip(ids, ids[1:]): pair_to_ids[pr].add(ids)
ok = True
for step in range(300):
    best = max(pair_counts, key=lambda p: (pair_counts[p], p[0], p[1])); new_id = 256 + step
    affected = pair_to_ids.pop(best)
    for ids in affected:
        w = ids_counts.pop(ids); new = tuple(merge(ids, best, new_id)); ids_counts[new] = w
        for pr, c in count_pairs(ids).items():
            pair_counts[pr] -= c * w
            if pair_counts[pr] <= 0: del pair_counts[pr]
            pair_to_ids[pr].discard(ids)
        for pr, c in count_pairs(new).items():
            pair_counts[pr] += c * w; pair_to_ids[pr].add(new)
    full = defaultdict(int)
    for ids, c in ids_counts.items(): count_pairs(ids, c, full)
    ok &= dict(full) == dict(pair_counts)
print("300 steps, incremental == recount every step:", ok)
```
```text
300 steps, incremental == recount every step: True
```

#### F16 99.7% of pretoken occurrences in valid become exactly ONE token (why a memo cache is so effective)  (13.65s)
```python
import pickle
from collections import Counter
from storybot.tokenizer import BPETokenizer, pretokenize
tok = BPETokenizer(pickle.load(open("storybot/merge_rules.pkl", "rb")))
c = Counter()
for t in open("storybot/tiny_stories_valid.txt", encoding="utf-8").read().split("<|endoftext|>"):
    c.update(pretokenize(t))
lens = {p: len(tok._encode_text(p)) for p in c}
occ = sum(c.values()); single = sum(n for p, n in c.items() if lens[p] == 1)
print(f"unique={len(c):,} occurrences={occ:,} single-token share={single/occ:.4f}")
print(f"tokens = {sum(n*lens[p] for p, n in c.items()):,} + EOT 27,630 = {sum(n*lens[p] for p, n in c.items()) + 27630:,}")
```
```text
unique=13,111 occurrences=5,419,001 single-token share=0.9974
tokens = 5,438,243 + EOT 27,630 = 5,465,873
```

#### F17 Official valid.bin: 10,931,746 bytes / 2 = 5,465,873 uint16 ids; ratio 4.117 bytes/token; max id < 10000  (0.02s)
```python
import numpy as np, os
ids = np.fromfile("storybot/tiny_stories_valid.bin", dtype=np.uint16)
print(ids.shape, ids.dtype, int(ids.max()), int((ids == 9999).sum()))
print(round(os.path.getsize("storybot/tiny_stories_valid.txt") / len(ids), 3))
print(np.iinfo(np.uint16).max, "<- uint16 holds ids up to 65535 (vocab 10k and 50k both fit)")
```
```text
(5465873,) uint16 9999 27630
4.117
65535 <- uint16 holds ids up to 65535 (vocab 10k and 50k both fit)
```

#### F18 uint16 overflow trap for vocab > 65,536 (NumPy 1.26 wraps with a DeprecationWarning; NumPy 2 raises)  (0.00s)
```python
import numpy as np, warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    a = np.array([65535, 65536, 70000], dtype=np.uint16)
print(np.__version__, a, [str(x.category.__name__) for x in w][:1])
```
```text
1.26.4 [65535     0  4464] ['DeprecationWarning']
```

#### F19 Chinese text with the TinyStories tokenizer is ~1 token per UTF-8 byte (3 per CJK char)  (0.01s)
```python
import pickle
from storybot.tokenizer import BPETokenizer
tok = BPETokenizer(pickle.load(open("storybot/merge_rules.pkl", "rb")))
for s in ["你好，世界", "Hello, world", "“Hi!” she said."]:
    ids = tok.encode(s)
    print(repr(s), len(s), "chars", len(s.encode()), "bytes ->", len(ids), "tokens", [tok.decode([i]) for i in ids][:8])
```
```text
'你好，世界' 5 chars 15 bytes -> 15 tokens ['�', '�', '�', '�', '�', '�', '�', '�']
'Hello, world' 12 chars 12 bytes -> 3 tokens ['Hello', ',', ' world']
'“Hi!” she said.' 15 chars 19 bytes -> 6 tokens ['“', 'Hi', '!”', ' she', ' said', '.']
```

#### F20 Text mode hides CRLF; binary mode (ch04/03,04,07) does not (git core.autocrlf=true on Windows)  (0.04s)
```python
raw = open("codebot/tiny_codes.txt", "rb").read()
txt = open("codebot/tiny_codes.txt", encoding="utf-8").read()
print(raw.count(b"\r\n"), txt.count("\r"), len(raw) - len(txt.encode()))
```
```text
256501 0 256501
```

#### F21 Removing pair_to_ids[best] then calling pair_to_ids[best] again silently re-creates an empty set (defaultdict)  (0.00s)
```python
from collections import defaultdict
d = defaultdict(set); d[(1, 2)].add((1, 2, 3))
s = d[(1, 2)]; del d[(1, 2)]
d[(1, 2)].discard((1, 2, 3))      # what the loop does for the merged pair
print(s, dict(d))
```
```text
{(1, 2, 3)} {(1, 2): set()}
```

#### F22 ch04/07 cache files are raw .tofile() dumps despite the .npy name: np.load fails, np.fromfile works  (0.01s)
```python
import numpy as np, os, tempfile
p = os.path.join(tempfile.mkdtemp(), "chunk_00000.npy")
np.array([1, 2, 9999], dtype=np.uint16).tofile(p)
print(np.fromfile(p, dtype=np.uint16))
try:
    np.load(p)
except Exception as e:
    print(type(e).__name__, str(e)[:70])
```
```text
[   1    2 9999]
ValueError Cannot load file containing pickled data when allow_pickle=False
```

#### F23 Training stops early when no pairs remain: vocab can be smaller than requested  (0.00s)
```python
import sys; sys.path.insert(0, r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch04-bpe\mods")
import v02_cache as v02
from storybot.tokenizer import BPETokenizer
v02.tqdm = lambda it, **k: it
r = v02.train_bpe("aaaa aaaa", 1000)
t = BPETokenizer(r)
print(r, "end_token_id =", t.end_token_id, "vocab_size =", t.vocab_size)
```
```text
{(97, 97): 256, (256, 256): 257, (32, 257): 258} end_token_id = 259 vocab_size = 260
```

#### F24 Pickled tokenizer size shipped to every Pool task in ch04/07 (bound method pickles self)  (0.02s)
```python
import pickle
from storybot.tokenizer import BPETokenizer
tok = BPETokenizer(pickle.load(open("storybot/merge_rules.pkl", "rb")))
print(len(pickle.dumps(tok._encode_chunk)), "bytes per task; x64 tasks =", 64 * len(pickle.dumps(tok._encode_chunk)) // 1024, "KiB")
```
```text
221754 bytes per task; x64 tasks = 13859 KiB
```

#### F25 count_pairs signature differs: codebot(ids, counts) vs storybot(ids, weight, counts) -> positional call breaks  (0.00s)
```python
from collections import defaultdict
import codebot.tokenizer as cb, storybot.tokenizer as sb
print(dict(cb.count_pairs([1, 2, 3], defaultdict(int))))
try:
    sb.count_pairs([1, 2, 3], defaultdict(int))
except TypeError as e:
    print("TypeError:", e)
print(dict(sb.count_pairs([1, 2, 3], counts=defaultdict(int))))
```
```text
{(1, 2): 1, (2, 3): 1}
TypeError: unsupported operand type(s) for +=: 'int' and 'collections.defaultdict'
{(1, 2): 1, (2, 3): 1}
```

#### F26 CS336 handout toy corpus through the book's trainer -> the same first 6 merges as the handout  (0.00s)
```python
import sys; sys.path.insert(0, r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch04-bpe\mods")
import v02_cache as v02
v02.tqdm = lambda it, **k: it
corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
r = v02.train_bpe(corpus, 256 + 6 + 1)
inv = {v: k for k, v in r.items()}
tb = lambda i: bytes([i]) if i < 256 else tb(inv[i][0]) + tb(inv[i][1])
print([f"{tb(a).decode()} {tb(b).decode()}" for (a, b) in r])
```
```text
['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']
```

#### F27 EXTENSION: a pretoken->ids memo dict gives identical ids (only 13,111 distinct pretokens in valid)  (1.83s)
```python
import pickle
from storybot.tokenizer import BPETokenizer, pretokenize
tok = BPETokenizer(pickle.load(open("storybot/merge_rules.pkl", "rb")))
text = open("storybot/tiny_stories_valid.txt", encoding="utf-8").read()[:300000]
memo = {}
def enc(t):
    out = []
    for piece in t.split("<|endoftext|>"):
        for p in pretokenize(piece):
            if p not in memo: memo[p] = tok._encode_text(p)
            out += memo[p]
        out.append(tok.end_token_id)
    return out[:-1]
print(enc(text) == tok.encode(text), len(memo), "memo entries")
```
```text
True 2925 memo entries
```


---

## 5. GPU / 数据需求

### 5.1 结论先行

- **第 4 章没有任何 GPU 代码**，全部是纯 Python（`regex`、`collections`、`multiprocessing`、`numpy` 只用于写 `.bin`）。在 GPU 实例上跑它是浪费 GPU 时间。
- 全量（2.2 GB）流程在本机（16 线程、47.7 GB 内存）可以直接跑：BPE 训练约 5–10 分钟，编码约 30–60 分钟（书中代码、8 进程，下面有推算依据）。
- 不想跑的读者：取消 `storybot/download.sh` 中 `merge_rules.pkl` 与两个 `.bin` 的注释直接下载；本档案已证明这些官方文件与书中代码的产出**逐字节一致**（valid 全量 + 训练集前 100 MiB）。

### 5.2 数据文件（大小均为 `curl -sIL` 的 `x-linked-size` 或本地 `stat`）

| 文件 | 字节 | 状态 | 说明 |
|---|---:|---|---|
| `storybot/tiny_stories_train.txt` | 2,227,753,162 | **未下载**（> 200 MB） | 约 2.72M 个故事（由前 100 MiB 的 127,993 个外推）；另用 HTTP Range 下载了前 104,857,600 B 到 `SCR/data/` 做实验 |
| `storybot/tiny_stories_valid.txt` | 22,502,601 | 已下载到 `storybot/` | 27,630 个 `<\|endoftext\|>`；纯 LF；合法 UTF-8；非 ASCII 字节 0.061%；以半句话开头 |
| `storybot/tiny_stories_train.bin` | 1,082,458,446 | 未下载 | **541,229,223** 个 uint16（4.116 B/token） |
| `storybot/tiny_stories_valid.bin` | 10,931,746 | 已下载（用于校验） | 5,465,873 个 uint16（4.117 B/token） |
| `storybot/merge_rules.pkl` | 103,877 | 已下载 | 9,743 条合并；pickle 只含纯数据操作码 |
| `codebot/tiny_codes.txt` | 6,743,534（本机 CRLF）/ 6,487,033（git 中 LF） | 仓库自带 | 21,835 个 `<\|endoftext\|>` |
| `codebot/tiny_codes.bin` | 5,354,796 | 仓库自带 | 2,677,398 个 token |
| `codebot/merge_rules.pkl` | 7,421 | 仓库自带 | 743 条合并 |
| `webbot/merge_rules.pkl` | 533,514 | 只下载到 `SCR/data/webbot/` | 49,743 条合并（ch07 的范围，仅用于对比） |
| `webbot/owt_train.txt` / `owt_valid.txt` | 11,920,511,059 / 289,998,753 | 未下载 | ch07 |
| `webbot/owt_train.bin` / `owt_valid.bin` | 5,298,009,600 / 128,992,132 | 未下载 | 2,649,004,800 / 64,496,066 个 token（4.50 B/token） |

**一个漂亮的自洽检验**：Heaps 定律（下节）外推全量训练集约有 536,549,840 个预分词出现；valid 上每个预分词平均 1.003551 个 token（不含 EOT）；再加约 2,719,296 个 `<|endoftext|>`，预测 541,174,339 个 token，与官方 `train.bin` 的 541,229,223 相差 **−0.01%**。

### 5.3 全量规模推算（依据都是本机实测）

**(a) 唯一预分词数（决定 merge 循环成本）**——`SCR/heaps.py` 在训练集前缀上逐段统计：

| 前缀 | 出现次数 | 唯一预分词 |
|---:|---:|---:|
| 1 MiB | 253,495 | 4,742 |
| 10 MiB | 2,526,467 | 10,235 |
| 50 MiB | 12,635,049 | 17,378 |
| 100 MiB | 25,254,556 | 21,808 |

拟合（≥5 MiB 的点）：`U ≈ 83.46 · N^0.327`。外推到 2.2 GB：约 5.37 亿次出现、**约 5.9 万个唯一预分词**（注意：Heaps 外推 20 倍有不确定性，量级可信）。

**(b) 训练（书中 `04` 代码，`num_chunks=64`）**：

| 阶段 | 依据 | 本机估计 |
|---|---|---|
| 预分词（串行） | 实测单进程 1.36–1.67 MB/s（负载高） | 约 22–27 分钟 |
| 预分词（8 / 16 进程） | 实测 valid 上 8 进程 3.3–4.1×、16 进程 2.7–3.7×（两次测量，含启动开销、负载 77–97%）；大文件上启动开销可忽略 | 约 4–7 分钟 |
| merge 循环（书中 `max()` 扫描） | 每扫描一个 pair 约 0.6–0.8 µs；平均 pair 数：valid 10,224、100 MiB 16,910 → 全量约 4.6 万（按 pair/唯一预分词 ≈ 0.78 估） | 约 3–6 分钟（安静时可能 2–3 分钟） |
| merge 循环（扩展：惰性堆） | valid 3.0 s、100 MiB 5.6 s | 约 10–20 秒 |
| 内存 | 64 块时每个子进程约 0.2 GB；父进程的计数表只有几万项 | 合计约 2 GB；**若用 `storybot.train_bpe` 默认 `num_chunks=8`：约 9–14 GB** |

**(c) 编码（书中 `07` 代码）**：`06` 的单进程速度在 valid 上是 22.5 MB / 144.9 s CPU ≈ 0.155 MB/s（负载高）。全量 2.2 GB ≈ 4 小时 CPU；8 进程约 30–60 分钟。memo 扩展单进程约 1.78 MB/s（0.56 s/MB）≈ 21 分钟 CPU，8 进程约 3–5 分钟。实测 `encode_file` 在 valid 上 1/8/16 进程分别 163.6/30.4/22.1 s，按 99 倍外推全量约 4.5 h / 50 min / 37 min。磁盘：输入 2.2 GB + 缓存块约 1.08 GB（结束时删除）+ 输出 1.08 GB。

### 5.4 在 Modal / RunPod / Lambda 上怎么跑

- **不需要 GPU**。若想把数据准备放在云上（比如紧挨着第 6 章的 GPU 训练），用纯 CPU 容器即可：≥ 16 vCPU、≥ 16 GB 内存（用 `num_chunks=64`）、≥ 10 GB 磁盘，约 45–75 分钟。
- Modal 写法要点：`@app.function(cpu=16, memory=16384, timeout=2*3600, volumes={"/data": vol})`；在函数里 `wget` 两份文本到 Volume，`os.chdir` 到仓库根，调用 `train_bpe(..., num_processes=16, num_chunks=64)` 与 `encode_file(..., num_processes=16)`；`multiprocessing` 在 Linux 上、Python 3.13 及以前默认 fork（子进程直接继承父进程内存，几乎没有 Windows spawn 那样的启动开销）；Python 3.14 起 Linux 默认改为 forkserver，工作函数同样必须放在可导入的模块里。
- RunPod / Lambda 主要卖 GPU；若已经开了一台 GPU 机器做第 6 章，可以先在同一台机器的 CPU 上跑本章（几十分钟），但更省钱的做法是在本机跑完再上传 `.bin`（1.08 GB），或直接在云端 `wget` 官方 `.bin`。
- **下游参考（第 6 章，供衔接）**：StoryBot 模型 22,696,448 个参数（`storybot.model.GPT(10000, 256, 512, 16, 4, 1344, 10000)` 实数）；第 6 章训练 40,000 步 × 32 × 256 = 327,680,000 个 token（官方 train.bin 的 0.605 个 epoch）；`6·N·D ≈ 4.46×10¹⁶` FLOPs。GPU 时长请以第 6 章档案为准。

### 5.5 CPU 缩小方案（给教程的"能在笔记本上 1–5 分钟跑完"版本）

| 方案 | 做法 | 本机耗时 | 与官方的差距 |
|---|---|---|---|
| A. 只用 valid 训练 | `04` 补丁副本（`file_path` 改为 valid） | 约 30–60 s | merges 前 5 条相同；token 集合重叠 87.7% |
| B. 训练集前 100 MiB | `curl -r 0-104857599 -L -o prefix.raw <train URL>`，在最后一个 `<\|endoftext\|>` 处截断 | 约 3–4 分钟（负载高） | 前 31 条相同；重叠 94.0% |
| C. 只编码 valid | `07` 补丁副本 + 官方 merges | 约 40 s（8 进程） | 与官方 valid.bin 逐字节相同 |
| D. 词表缩小 | 词表 1000 + 500 KB 切片 | 朴素版约 3–4 分钟，`02` 约 2 s | 用来演示加速比 |


---

## 6. 易错点 / 坑 / 潜在 bug（每条都实际跑过）

### 6.1 Windows 专属

1. **`core.autocrlf=true` 把语料改成 CRLF**（严重程度：高，静默）。本机全局 git 配置为 `true`；`git ls-files --eol` 显示 `i/lf w/crlf`：仓库里是 LF，检出后变 CRLF。`codebot/tiny_codes.txt` 因此从 6,487,033 B 变成 6,743,534 B（多 256,501 个 `\r`）。
   - 文本模式读取（`01`、`02`、`06` 的 `open(path).read()`）自动把 `\r\n` 变回 `\n`，所以结果与官方一致（`06` 得到 2,677,398 = 官方 `tiny_codes.bin` 长度）。
   - 二进制读取（`03`、`04`、`07` 的 `find_chunk_boundaries` + `decode`）保留 `\r`：在 tiny_codes 上训练，**第 2 条合并就是 `b'\r\n'`**，与官方结果从第 1 条起分叉，最终 16 个 token 含 `\r`（§3、F20）。
   - 预分词正则会把行尾的 `'\r\n'` 拆成 `'\r'`、`'\n'` 两个预分词（`'x\r\ny'` → `['x', '\r', '\n', 'y']`），而缩进前的 `'\r\n   '` 会整体成为一个预分词。
   - 规避：克隆前 `git config --global core.autocrlf false`（或 `input`），或在仓库加 `.gitattributes`：`*.txt -text`；TinyStories/OWT 是 `wget/curl` 下载的，不受影响。
2. **默认编码是 cp936（GBK）**。不开 UTF-8 模式时，`open('storybot/tiny_stories_valid.txt').read()` 抛 `UnicodeDecodeError: 'gbk' codec can't decode byte 0x9d in position 30846`；纯 ASCII 的 tiny_codes 则能正常读，所以 `01/02/06` 的 as-is 运行不会暴露问题，但 `02` 末尾注释掉的 TinyStories 代码会崩。规避：`PYTHONUTF8=1`（或 `python -X utf8`），或显式 `encoding="utf-8"`。控制台输出中文/日文也需要 `PYTHONIOENCODING=utf-8`。
3. **spawn 启动开销**：Windows 的 `multiprocessing` 只能 spawn，每个子进程重新 import 模块；本机 Pool(8) 启动+关闭约 0.7–1.45 s、Pool(16) 约 2 s。小数据上并行反而更慢（500 KB 切片：4 进程 0.74 s vs 串行 0.23 s）。并且脚本必须有 `if __name__ == '__main__':`（`04`/`07` 有；`03` 不用多进程所以没有）。tqdm 的计时从第一个结果返回才开始，**不含**进程启动时间，看进度条会高估加速比。
   - 本档案亲身踩到的变体：用 `python - <<EOF … EOF`（从 stdin 读脚本）调用 `encode_file`，spawn 出的子进程要重新执行 `__main__`，而 `<stdin>` 不是文件 → `OSError: [Errno 22] Invalid argument: '…\<stdin>'`，子进程全部失败，**父进程的 `pool.imap` 永远等下去（挂死，不报错）**。用 `python -c` 定义工作函数再 `Pool.imap` 也复现了：4 个子进程都报 `AttributeError: Can't get attribute 'square' on <module '__main__' …>`，父进程挂死到 40 s 超时被杀（exit 124）。在 Jupyter notebook 单元格里定义工作函数是同一机制（推断，未在 notebook 中实测）。规避：工作函数放进 `.py` 模块并从文件运行（书中做法）。

### 6.2 算法 / 正确性

4. **平票规则决定"优化后结果是否不变"**。没有显式 tie-break 时，增量版与全量版从第 301 步分叉（tiny_codes，11.8% 的步骤存在平票）。书中 key `(count, id0, id1)` 是全序，所以所有实现逐项一致。与 CS336 的 bytes 字典序规则不同（第 289 步分叉）——如果读者拿 CS336 的测试去测书中实现，会在 merges 上失败（推断：测试比较的正是 merges 列表）。
5. **`BPETokenizer.__init__` 依赖 `merge_rules` 的插入顺序**。构建 `id_to_bytes` 时按字典顺序遍历，若顺序不是训练顺序（例如从 JSON 读回并排序过），直接 `KeyError`（打乱顺序后实测 `KeyError: 362`）。ch01 的逐条编码器同样依赖顺序：换成打乱顺序的同一映射，输出从 19 个 token 变成 45 个，**不报错但结果错**。`06` 的 rank 编码器只用字典的值做优先级，对顺序不敏感（实测相同）。稳妥写法：`for (a, b), i in sorted(merge_rules.items(), key=lambda kv: kv[1])`。
6. **训练可能提前结束**：没有相邻对时 `break`，合并数少于 `vocab_size − 257`，此时 `end_token_id = 256 + len(merge_rules)` 也随之变小（`"aaaa aaaa"` 要 1000 只得到 3 条，`end_token_id = 259`）。模型的 `vocab_size` 超参数必须 ≥ `tokenizer.vocab_size`。另外 ch01/03 用 `vocab_size − 256`（当时还没有特殊 token），ch04 用 `− 257`，章节之间差 1。
7. **切块只能切在 `<|endoftext|>` 上**：在词中间切（`' lit' | 'tle'`）编码结果就变了（F13）。`find_chunk_boundaries` 的 4096 字节缓冲区若恰好把 `<|endoftext|>` 切在两次读取之间，会错过它、落到下一个——结果仍正确，只是块不均。
8. **`02` 里的 defaultdict 小陷阱**：`del pair_to_ids[best_pair]` 之后循环又对 `pair_to_ids[best_pair]` 调 `.discard`，defaultdict 会新建一个空集合并留在字典里（F21）。无害；也正因为 `affected_ids` 是已经摘下来的那个集合对象，循环中不会出现"边迭代边修改"的错误。
9. **`count_pairs` 签名在模块间不兼容**：codebot 是 `(ids, counts=None)`，storybot 是 `(ids, weight=1, counts=None)`。混用时按位置传 `counts` 会被当作 `weight` → `TypeError`；传普通 `{}` 而不是 `defaultdict(int)` → `KeyError`（F25）。
10. **预分词正则的细节**：弯引号 `’` 不触发缩写规则（`don’t` → `don`、`’`、`t`）；缩写区分大小写（`you'LL` → `' you'`、`"'"`、`'LL'`）；数字与字母分开（`' 2024'`）；空格串 `\s+(?!\S)` 会留一个空格给下一个词（`'.   Go'` → `'.'`、`'  '`、`' Go'`）。TinyStories 里弯引号很常见（`'“'` 是单独的 token，`'!”'` 也是）。
11. **中文**：TinyStories 分词器几乎没学过中文，`'你好，世界'` 15 字节 → 15 个 token（每个都是单字节，单独 decode 显示 `'�'`）。流式生成时要对**累积的 id 序列**整体 decode，不能逐 token decode 再拼接。教程面向中文读者时，这个例子很直观。
12. **编码最坏情况是超线性的**：`_encode_text` 每轮 `O(L)`、最多 `L−1` 轮。重复串（全空格、`abab…`）每轮能合并很多处，4000 字节只要几 ms；长的随机字母串（单个预分词）1000→8000 字母耗时 162 ms→3.2 s。TinyStories 预分词最长 23 字节，不受影响；OWT 里的长 URL/哈希串会慢一些。

### 6.3 工程 / 资源

13. **PEP 393 内存放大**：一个 emoji 让整段 `str` 变成 4 B/字符。整读 100 MiB 峰值 656 MiB；全量 2.2 GB 整读估计十几 GB（§3.5）。这才是"分块"的主要动机——教程里值得用一个"内存计"组件演示。
14. **`storybot/tokenizer.py` 默认参数与 ch04 不同**：`train_bpe(num_chunks=8)`（`04` 是 64）、`encode_file(num_processes=4)`（`07` 是 8）。全量训练时 8 块 = 每块约 278 MB，按实测的"峰值 ≈ 4–6.4 × 块大小"外推，每个子进程 1.1–1.8 GB，合计 9–14 GB；`ch07/01`（11.9 GB 的 OWT）显式传了 `num_chunks=64`，大概也是这个原因。教程建议统一显式传参。
15. **`encode_file` 的 `cache_dir`**：默认相对路径 `bpe_cache`，结束时 `shutil.rmtree(cache_dir)` 删除整个目录（包括里面原有的文件）。实测：在 scratch 里建 `precious/notes.txt`，把 `precious` 当 `cache_dir` 传给 `encode_file`，运行后整个目录连同 `notes.txt` 都没了。缓存文件名是 `chunk_00000.npy`，但内容是 `tofile` 写的原始字节，没有 `.npy` 头，`np.load` 会失败（F22）。
16. **`uint16` 上限 65,535**：10k / 50k 词表都没问题；若有人把词表调到 65,536 以上，NumPy 1.26 静默回绕（`65536 → 0`、`70000 → 4464`，只有 DeprecationWarning），NumPy 2 直接 `OverflowError`（F18，两种都实测）。`.bin` 没有文件头，读的时候必须知道 dtype（`np.memmap(path, dtype=np.uint16)`），字节序是本机序（x86/ARM 都是小端）。
17. **绑定方法的 pickle 开销**：`pool.imap(self._encode_chunk, …)` 每个任务都 pickle 整个 tokenizer（221,754 B）。数据大时可忽略，但"把大对象挂在 `self` 上再并行"在别处可能很贵；可以用 `Pool(initializer=…)` 在每个进程里只加载一次。
18. **merge 循环的串行瓶颈**：`02` 之后 97% 时间在 `max()` 全表扫描；并行只加速预分词，Amdahl 定律下整体加速有上限（valid：预分词 12–14 s 可以降到 3 s，但 merge 循环 27–63 s 不变）。
19. **pickle 安全**：`merge_rules.pkl` 来自网络；`pickle.load` 理论上能执行任意代码。可以先 `pickletools.genops` 检查只有纯数据操作码（本档案已检查三份 pkl）。
20. **valid 文件从半句话开始**（`"u don't have to be scared…"`），第一段不是完整故事。对统计无影响，但教程展示"第一个故事"时别被吓到。

### 6.4 codebot / storybot / webbot 三个变体的差异

| 方面 | codebot（第 1 章） | storybot（第 4 章） | webbot（第 7 章） |
|---|---|---|---|
| 分词器模块 | `codebot/tokenizer.py` | `storybot/tokenizer.py` | **无**，直接用 storybot 的 |
| `pretokenize` | `re.findall` 返回 list | `re.finditer` 生成器 | 同 storybot |
| `count_pairs` | `(ids, counts=None)` | `(ids, weight=1, counts=None)` | 同 storybot |
| 训练 | 每个出现一份 list，全量重算 | 去重 + 增量 + 分块 + 多进程 | 同 storybot（`num_chunks=64` 显式传入） |
| 编码 | 逐条执行全部合并 | rank 最小优先 + `encode_file` | 同 storybot |
| 词表 / 合并数 | 1,000 / 743 | 10,000 / 9,743 | 50,000 / 49,743 |
| `end_token_id` | 999 | 9,999 | 49,999 |
| 前 10 个 token | `'  '`、`'    '`、`'in'`、`'   '`、`'re'`、`'or'`、`' ='`、`'st'`… | `' t'`、`'he'`、`' a'`、`' s'`、`' w'`、`'nd'`、`' the'`、`'ed'`、`' b'`、`' to'` | `' t'`、`' a'`、`'he'`、`'in'`、`'re'`、`' the'`、`'on'`、`'er'`、`' s'`、`' w'` |
| 最长 token | — | `' accomplishment'`（15 B） | 64 个 `-`、64 个 `_`、以及一长串 `ÃÂÃÂ…`（双重 UTF-8 编码的乱码，网页语料的典型产物） |
| 同一句话 | — | 13 个 token：`[429, 438, 258, 397, 44, 400, 282, 258, 389, 476, 401, 405, 46]` | 13 个 token：`[7228, 3065, 257, 630, 44, 592, 358, 257, 1311, 2772, 3319, 32082, 46]` |
| 压缩率 | tiny_codes 全量 2.42 B/token（LF 6,487,033 B / 2,677,398；前 5% 切片为 2.14） | valid 4.117 B/token | OWT train 4.50 B/token |

（"同一句话" = `"Once upon a time, there was a little girl named Lily."`；GPT-2 的 tiktoken 结果也是 13 个：`[7454, 2402, 257, 640, 11, 612, 373, 257, 1310, 2576, 3706, 20037, 13]`。）


---

## 7. 互动组件点子（按章节顺序；括号里是可直接嵌入的实测数据）

> 实现建议：预分词用 JS 正则 `/'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu`（已验证与 Python `regex` 逐个相同）；训练/编码可以用 JS 重写，也可以在 Pyodide 里直接跑书中原函数（已验证 `regex`、`numpy` 可用）。三份 `merge_rules.pkl` 可预先转成 JSON（`[[id1, id2, new_id], …]`）供前端使用。

| # | 组件 | 学习者操作 | 学习者看到什么 | 对应概念 / 数据 |
|---|---|---|---|---|
| W1 | **预分词显微镜** | 输入任意文本；点预设例子（缩写、弯引号、多空格、数字、CRLF、中文、emoji） | 每个预分词一个色块，空格显示为 `·`、换行为 `↵`；悬停显示命中的是正则的哪个分支 | F01、F10、§6.2-10 |
| W2 | **Zipf 去重计数器** | 拖动"语料大小"滑块（内置 2 MB TinyStories 样本 + 实测点） | 出现次数 N、唯一数 U、两者之比实时变化；双对数图上的 Heaps 拟合线 `U = 83.46·N^0.327`，外推到 2.2 GB（约 5.9 万唯一、去重倍数约 9,000×） | `01`；§2.1 表、§5.3 表 |
| W3 | **BPE 单步调试器** | 编辑一个玩具语料（默认 CS336 讲义的 low/lower/widest/newest），点"下一步" | 唯一词表 + 次数、pair 计数表（排序）、高亮 best pair；三个计数器：朴素/去重/增量各自这一步"摸了多少字节" | `01`、`02`；F06、F26 |
| W4 | **增量账本** | 在 W3 中打开"显示索引" | `pair_to_ids` 倒排索引；被影响的词闪烁；pair 表里出现 `−旧 / +新` 的增量；右上角常驻"增量 == 重算 ✔"校验灯 | `02`；F15；每步平均只动 7 个词（中位数 1） |
| W5 | **平票实验室** | 选择 tie-break：字典顺序 / 书中 `(count, id0, id1)` / CS336 `(count, bytes0, bytes1)`；切换"全量重算"与"增量"实现 | 两列 merges 并排，第一个分叉处标红；显示"本语料 X% 的步骤有平票" | §3.4（tiny_codes 11.8%、第 301 / 289 步分叉） |
| W6 | **下一个瓶颈（扩展）** | 开关"用堆代替 max()" | 每步耗时堆叠条：`max()` 占 97% → 堆版本；pair 数随步数变化的曲线（932 → 峰值 12,025 → 7,821） | §3.3（70.8 s → 3.0 s） |
| W7 | **切块器** | 拖动 `num_chunks`（1–100,000）；打开"在词中间切" | 文件条上等距的估计切点"吸附"到下一个 `<\|endoftext\|>`，显示 4096 字节预读窗；唯一边界数（64→65 个，100,000→27,629 个）；在词中间切时 token 序列出现差异 | `03`；F11、F13 |
| W8 | **内存计（PEP 393）** | 在文本框里粘贴文本，点"加一个 🍌" | `len`、UTF-8 字节数、CPython `str` 大小（按最大码点 1/2/4 B/字符）；加一个 emoji 内存 ×4；旁边是实测：整读 656 MiB vs 分块 42 MiB | §3.5 |
| W9 | **Amdahl 计算器** | 调进程数、Pool 启动开销、每进程吞吐、merge 循环时间 | 总时间堆叠条与加速曲线；叠加本机实测点（预分词 8 进程 4.1×、16 进程回落到 2.7×；编码 16 进程 7.4×） | `04`、`07`；§3.2 表 |
| W10 | **编码器赛跑** | 输入一个词（默认 `' accomplishment'`） | ch01 编码器扫过 9,743 条规则的进度条（只有 14 条真正生效，高亮出来）vs rank 编码器逐轮显示相邻对及其 rank 徽章（最小者高亮）vs memo 一次查表；实测约 14–16 ms / 0.27–0.29 ms | `06`；F09、§3.6 |
| W11 | **Token 着色器** | 输入文本，切换 CodeBot(1k) / StoryBot(10k) / WebBot(50k) 合并表 | token 着色、悬停看 id 与字节；B/token；中文一字三 token；`<\|endoftext\|>` → 9999 | F08、F19、§6.4 表 |
| W12 | **.bin 检查器** | 用 HTTP Range 读 `tiny_stories_valid.bin` 的前 N 字节；切换 dtype | 十六进制（小端）→ id → 解码文本，9999 处断开故事；dtype 选错立即变乱码；`文件大小 / 2 = token 数` | `07`；F17 |
| W13 | **CRLF 陷阱** | 在 LF / CRLF 之间切换一段代码样本 | 二进制模式训练时 `b'\r\n'` 立刻成为第 2 条合并；文本模式下消失 | §6.1-1、F20 |
| W14 | **端到端流水线（贯穿例子）** | 点击每个阶段 | 22,502,601 B 文本 → 5,419,001 个预分词 → 13,111 个唯一 → 9,743 条合并 → 5,465,873 个 token → 10,931,746 B 的 `.bin` → 第 6 章 `get_batch` 的 `(32, 256)` 批次；每一步给出"如何验证"的一行代码 | 全章 + §3.7 |

**端到端例子的建议脚本**（浏览器内 1–2 分钟可完成）：取 valid 的前 1 MB（在 `<|endoftext|>` 处截断）→ 用 `02` 训练 1000 词表 → 用 `06` 编码同一段 → 写成 `Uint16Array` → 解码回来与原文逐字节比较 → 再用官方 10k 合并表编码同一段，与官方 `valid.bin` 的相应前缀比较（应完全相同）。

### 7.1 预判的困惑点与建议讲法

| 读者常见的疑问 | 建议的讲法（附可当场验证的证据） |
|---|---|
| "去重之后，结果还和原来一样吗？" | pair 频次对"出现"线性可加，相同的预分词永远同步演化 → 用权重代替复制。当场跑 F03（加权 == 展开）、F06（三种实现相同）。 |
| "只更新受影响的词，会不会漏掉什么？" | 写出不变式 `pair_counts = Σ w·pairs(seq)`，不受影响的序列 `merge` 后不变。W4 的"增量 == 重算 ✔"灯 + F15（300 步逐步核对）。 |
| "为什么要写那个奇怪的 `(count, pair[0], pair[1])`？" | 不写的话，平票由字典顺序决定，两种实现在第 301 步分叉（§3.4）。这是"优化不改变结果"的前提，不是装饰。 |
| "并行了 8 个进程，为什么总时间没快 8 倍？" | 只有预分词能并行；merge 循环是串行的（Amdahl）；Windows spawn 还要 0.5–2 s 启动。W9 用本机数据画出来。 |
| "rank 编码为什么等价于按顺序应用所有规则？" | §2.6 的三行证明：规则 r 的两个成分 id 都 < 256+r，执行后只会产生 rank 更大的新对。再用 F09 / 全量 valid 的逐 token 比对收尾。 |
| "vocab_size 10000，合并为什么是 9743？" | 256 个字节 + 9743 次合并 + 1 个 `<\|endoftext\|>` = 10000；`end_token_id` = 9999（F07）。顺带提醒第 1 章早期版本是 `− 256`。 |
| "字节、字符、token 到底差多少？" | valid：22,502,601 字节、22,493,387 字符、5,419,001 个预分词、5,465,873 个 token；中文一字 3 字节 = 3 个 token（F19）。 |
| "为什么非得在 `<\|endoftext\|>` 处切块？" | 在词中间切，编码结果就变（F13：`' lit' \| 'tle'`）；在特殊 token 处切，训练计数与编码结果都与整体处理相同（F12、F13）。 |
| "为什么要存成 uint16 / memmap？" | 541M 个 token 的 Python list 约 4.3 GiB（实测 8.6 B/token），uint16 只要 1.08 GB 且可 memmap 随机读取；上限 65,535（F18）。 |


---

## 8. CS336 对照（Assignment 1 "Basics" 的 BPE 部分；凭记忆，未联网核对）

| DLFS6 第 4 章 | CS336 A1 对应 | 异同 |
|---|---|---|
| GPT-2 预分词正则 | §2.4 "Pre-tokenization" 给出的 `PAT` | **字符串完全相同**；CS336 同样建议用 `regex` 和 `re.finditer` 避免一次性存下所有预分词 |
| 先按 `<\|endoftext\|>` 切开再预分词 | §2.4 "Special tokens" / §2.5 "Removing special tokens before pre-tokenization" | 相同思想；CS336 支持**多个**特殊 token（`re.split` + `"\|".join(map(re.escape, …))`），DLFS6 只有一个 `end_token` |
| `01` 预分词计数表 + 加权 | 讲义的 BPE 示例本身就用"词 → 频次"表（`{low: 5, lower: 2, widest: 3, newest: 6}`） | 相同；用书中训练器跑讲义的玩具语料，前 6 条合并与讲义一致（F26） |
| `02` 增量 pair 计数 + 倒排索引 | §2.5 "Optimizing the merging step"：只有与被合并 pair 重叠的计数会变，建立索引并增量更新 | **几乎就是同一段描述的实现**；CS336 也明确说 merge 步骤在 Python 里无法并行 |
| `03` `find_chunk_boundaries` | 起始代码 `cs336_basics/pretokenization_example.py` 的 `find_chunk_boundaries(file, desired_num_chunks, split_special_token)` | 结构几乎逐行相同（4096 字节小块向后找特殊 token、`sorted(set(...))` 去重）；DLFS6 版接收路径、CS336 版接收文件对象 |
| `04` 多进程预分词 | §2.5 "Parallelizing pre-tokenization" | 相同 |
| tie-break `(count, id0, id1)` | "preferring the lexicographically greater pair"，比较的是 **bytes**（如 `("BA","A") > ("A","B")`） | **不同**：tiny_codes 上第 289 步起分叉（§3.4）。若要通过 CS336 的 `tests/test_train_bpe.py`，需把 key 改为 `(count, vocab[a], vocab[b])` |
| `train_bpe(...) -> {(id1, id2): new_id}` | `train_bpe(input_path, vocab_size, special_tokens) -> (vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])` | 表示法不同：DLFS6 用 id 对，特殊 token 隐式放在最后（id = vocab_size − 1）；CS336 显式返回 vocab 与 bytes 形式的 merges |
| StoryBot：TinyStories、词表 10,000 | 问题 `train_bpe_tinystories`：同一份 TinyStoriesV2-GPT4、词表 10,000；问训练耗时/内存、最长 token、做 profiling | **数据一致**（字节数相同）；最长 token 同为 `' accomplishment'`；本档案 profiling 结论：`02` 之后 97% 时间在 `max()` |
| WebBot：OWT、词表 50,000（ch07） | 问题 `train_bpe_expts_owt`：OWT、词表 32,000 | 数据一致、词表不同；最长 token 都是网页乱码/分隔线一类 |
| `06` `BPETokenizer.encode/decode` | 问题 `tokenizer`：`Tokenizer(vocab, merges, special_tokens)`、`from_files`、`encode`、`encode_iterable`、`decode` | DLFS6 没有 `encode_iterable`（用 `encode_file` 分块 + 并行替代），不处理重叠的多个特殊 token；`decode` 同样用 `errors="replace"` |
| `06` 的 rank 编码 | CS336 描述的是"按创建顺序依次应用 merges" | 结果等价（§2.6 的证明 + 全量 valid 实测）；rank 版更快，且对字典顺序不敏感 |
| `07` `encode_file` → `uint16` `.bin` | 问题 `tokenizer_experiments`：压缩率、吞吐量、估算 Pile（825 GB）要多久、把数据集编码成 `uint16` NumPy 数组并说明为何用 uint16 | 本档案数据：valid 4.117 B/token；`06` 单进程约 0.155 MB/s（负载高）→ 825 GB 单进程约 62 天；memo 版约 1.78 MB/s → 约 5.4 天；tiktoken 约 2.66 MB/s → 约 3.6 天；uint16 因为 10k/50k < 65,536 |
| 与 GPT-2 对比（ch01/08 用 tiktoken） | 测试 `tests/test_tokenizer.py` 用 tiktoken 的 GPT-2 作为参考 | 本档案：valid 上 GPT-2 4.067 B/token，StoryBot 10k 反而更好（4.117） |

**教学上的建议**：把第 4 章写成"CS336 A1 的 BPE 部分 + 一份可以逐行对照的参考实现"。在 tie-break 处明确告诉读者两种规则都"正确"，区别只在约定；并给出一行改动让它与 CS336 对齐。

---

## 9. 附录：复现清单

所有命令在仓库根 `C:\Users\chenf\Projects\_refs\dlfs6` 执行，前缀 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`；`SCR` = `C:/Users/chenf/AppData/Local/Temp/claude/C--Users-chenf/2766cf54-7521-48a1-be58-647c0a0601b0/scratchpad/ch04-bpe`。

| 文件 | 作用 |
|---|---|
| `SCR/mkmods.py` → `SCR/mods/*.py` | 从 ch04 脚本按原始行抽出定义，生成可导入模块 |
| `SCR/mkpatched.py` → `SCR/patched/0{3,4,7}_*.py` | 训练集 → valid、输出改到 `SCR/out/` 的补丁副本（每处替换都有 assert） |
| `SCR/bench_train.py` | 训练变体的分段计时 + merges 逐项比较 |
| `SCR/bench_encode.py` | 编码变体计时 + 与官方 `.bin` 比较 + 往返检查 |
| `SCR/analyze_v02.py` | `02` 循环的逐行复刻（受影响序列数、pair 数、`max()` 占比）+ 惰性堆扩展 |
| `SCR/tiebreak.py` | 三种 tie-break × 两种实现 |
| `SCR/corpus_stats.py`、`SCR/heaps.py` | 去重统计、Heaps 定律拟合 |
| `SCR/mem_peak.py`、`SCR/mem_chunk.py` | 峰值内存（需 `uv run --with psutil`） |
| `SCR/scaling.py` | Pool 启动、预分词与 `encode_file` 的进程数扫描 |
| `SCR/encode_prefix.py` | 用 `07` 编码训练集前 100 MiB（再与官方 train.bin 前缀 `cmp`） |
| `SCR/facts.py` → `SCR/out/facts_all.md` | 27 条事实的代码与输出 |
| `SCR/pretok.mjs`、`SCR/js_parity.py` | JS 正则与 Python `regex` 的一致性检查 |
| `SCR/pyodide_test/` | Pyodide 0.29.4 中的浏览器可行性检查 |
| `SCR/check_pickle.py` | pickle 操作码检查 |
| `SCR/out/*.log`、`SCR/out/merges_*.pkl` | 所有运行日志与产出的 merges |

常用命令：

```bash
# as-is
uv run python ch04/01_bpe_optimize.py
uv run python ch04/02_bpe_cache.py
uv run python ch04/06_encode_optimize.py          # 打印 2677398
# 补丁副本（需要 storybot/tiny_stories_valid.txt）
uv run --project . python $SCR/patched/03_bpe_chunk.py
uv run --project . python $SCR/patched/04_bpe_parallel.py
uv run --project . python $SCR/patched/07_encode_parallel.py && cmp $SCR/out/tiny_stories_valid_v07.bin storybot/tiny_stories_valid.bin
# 等价性
uv run python $SCR/bench_train.py $SCR/out/valid_500k.txt 1000 naive,v01,v02,v03,v04,story --procs 4
uv run python $SCR/bench_encode.py storybot/merge_rules.pkl storybot/tiny_stories_valid.txt v06,v07,memo --procs 8 --bin storybot/tiny_stories_valid.bin
```
