# 研究档案：ch01 Tokenizer（附 ch07 WebBot tokenizer）

> 来源仓库：`C:\Users\chenf\Projects\_refs\dlfs6`（oreilly-japan/deep-learning-from-scratch-6，最新 commit `c9b6e2e`）
> 范围：`ch01/*.py`（9 个脚本，逐行读过）、`codebot/tokenizer.py`、`ch07/*.py`（2 个脚本）+ 它们依赖的 `storybot/tokenizer.py`；数据 `codebot/tiny_codes.txt`、`codebot/tiny_codes_sft.json`、`codebot/merge_rules.pkl`、`codebot/tiny_codes.bin`。
> 运行环境：Windows 11 · i7-11700F（8C/16T）· 48 GB RAM · uv 0.9.13 · 仓库 venv（Python 3.12.9，torch 2.9.1+cpu，numpy 1.26.4，regex 2026.1.15，tqdm 4.67.1）· node 24.13 · Pyodide 0.29.4（Node 内运行，Python 3.13.2 / numpy 2.2.5 / regex 2024.11.6）。
> 所有输出均为本机实际运行结果。草稿脚本、补丁副本、下载样本都在 scratch：`C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer\`（下文简称 `SC\`）。仓库内没有改动任何文件（`git status` 里的 `M uv.lock` 在本任务开始前就已存在）。

---

## 0. 一页速览（TL;DR）

| 项目 | 数值（均已实测） |
|---|---|
| CodeBot tokenizer | byte-level BPE + GPT-2 pre-tokenize 正则 + 1 个 special token `<|endoftext|>` |
| merge 数 / vocab | 743 条 merge（id 256–998）+ end token id **999** → `vocab_size = 1000` |
| 训练语料 `tiny_codes.txt` | 21,836 个 Python 小程序，由 21,835 个 `<|endoftext|>` 分隔（文件末尾没有 end token）；纯 ASCII；LF 版 6,487,033 字节（Windows checkout 因 autocrlf 是 CRLF，6,743,534 字节） |
| `tiny_codes.bin` | `np.uint16`，**2,677,398** 个 token（5,354,796 字节），id 范围 9–999，999 出现 21,835 次 |
| 压缩率 | 前 10,000 字符：10,000 B / 4,709 tok = **2.12** B/token；全语料 2.42 B/token |
| 对照（同 10,000 字符） | GPT-2 3.09 · cl100k_base 3.97 · o200k_base 3.98 B/token |
| 可复现性 | `ch01/07` 原算法完整重训（64 min）→ 与仓库 `merge_rules.pkl` **pickle 逐字节相同**；`ch01/09` 重跑（10.5 min）→ 与仓库 `tiny_codes.bin` **逐字节相同**（sha256 `d1e082b1…b495`）；另外 ch04 的快速训练器 19 s、纯 JS 训练器 28.7 s 也都逐条复现了这 743 条 merge |
| WebBot（ch07） | 50,000 vocab（49,743 merge），训练数据 OWT `owt_train.txt` 11,920,511,059 B；产物 `owt_train.bin` 2,649,004,800 tok、`owt_valid.bin` 64,496,066 tok（≈4.50 B/token） |
| StoryBot（ch04，供对照） | 10,000 vocab（9,743 merge）；`tiny_stories_train.bin` 541,229,223 tok |
| 浏览器可行性 | 35 个不依赖 tiktoken 的事实片段在 Pyodide 0.29.4 中 34 个输出与 CPython 逐字相同（唯一差异是 numpy 2 对 uint16 溢出改为抛异常）；纯 JS 移植的 encode 在 29 个 Unicode 用例 + 200k 字符样本上与 Python 逐 id 一致 |

---

## 1. 推断的章节叙事

### 1.1 全书骨架（从仓库结构推断）

全书用三个“bot”串起来，每个 bot 对应一个 tokenizer 规模与一套越来越工程化的实现：

| bot | 章节 | 语料 | vocab | tokenizer 实现 |
|---|---|---|---|---|
| CodeBot | ch01–ch03 | `codebot/tiny_codes.txt`（6.5 MB Python 小程序） | 1,000 | `codebot/tokenizer.py`：朴素实现（每步全量重数 pair；encode 按训练顺序逐条套 merge） |
| StoryBot | ch04–ch06 | TinyStories（2.2 GB） | 10,000 | `storybot/tokenizer.py`：ch04 的优化成果（pre-token 去重计数、pair→pretoken 倒排缓存、文件分块 + multiprocessing、按 rank 优先合并的 encode、memmap 输出） |
| WebBot | ch07–ch09 | OpenWebText 样本（11.9 GB） | 50,000 | **直接复用** `storybot/tokenizer.py`（ch07 只有两个调用脚本） |

所以 ch01 是“概念 + 能跑通的最小实现”，ch04 是“同一算法的工程化”，ch07 是“把工程化实现用到 Web 规模数据”。每个部分都按“tokenizer → 模型 → 训练”三章推进。

> **与官方目次对照**：本节先由文件编号和代码推断，写完后与兄弟档案 `research/dlfs6-toc.md`（找到了 O'Reilly Japan 页面上的官方完整目次）逐条核对。ch01 的推断与官方 1.1–1.9 **完全对应**（文件 `NN_*.py` ↔ 第 1.N 节）；ch07/ch08 的推断有一处需要修正，见 1.3。

### 1.2 ch01 的叙事（文件编号 01–09 连续，无缺号）

| 官方节名（中译，据 `dlfs6-toc.md`） | 文件 | 引入的概念 | 衔接逻辑（推断） |
|---|---|---|---|
| 1.1 字符级分词（1.1.1 把字符转换为 ID / 1.1.2 实现 / 1.1.3 问题） | `01_char_tokenizer.py` | `ord`/`chr`、Unicode code point、encode/decode 接口 | 最直观的方案；示例串 `"hello世界😁"` 贯穿 01/02/04 |
| 1.2 字节级分词（1.2.1 UTF-8 编码 / 1.2.2 实现 / 1.2.3 问题） | `02_byte_tokenizer.py` | UTF-8 变长编码（`'あ'` → 3 字节）、256 个基础 id、`bytes(ids).decode()` | 字符级词表太大（110 万码位）→ 改用 256 个字节；代价是序列变长（8 字符 → 15 字节） |
| 1.3 BPE 的训练算法（1.3.1 处理步骤 / 1.3.2 实现 / **专栏：BPE 的平票处理** / 1.3.3 运行验证） | `03_bpe_train.py` | `count_pairs`、`merge`、`train_bpe`、`num_merges = vocab_size - 256` | 字节序列太长 → 把常见相邻对合并成新 token；专栏正好对应代码里被注释掉的 tie-break 行 |
| 1.4 用 BPE 编码与解码（1.4.1 tokenizer / 1.4.2 编码 / 1.4.3 解码 / 1.4.4 验证） | `04_bpe_tokenizer.py` | `id_to_bytes` 递推构造词表；encode = 按训练顺序套用 merge；decode = 拼字节 + `errors="replace"` | 学到的 merge 规则怎么用 |
| 1.5 支持特殊 token（1.5.1 作用 / 1.5.2 训练 / 1.5.3 `train_bpe` / 1.5.4 tokenizer） | `05_special_token.py` | `<|endoftext|>`；训练前按它切开（不跨文档合并）；encode 时 `re.split` 带捕获组保留它；`vocab = 256 + merges + 1` | 训练数据由很多独立程序拼成，需要“文档边界” |
| 1.6 预分词（1.6.1 必要性 / 1.6.2 GPT-2 的正则 / 1.6.3 训练 / 1.6.4 tokenizer） | `06_pretokenize.py` | GPT-2 正则（`regex` 模块的 `\p{L}`/`\p{N}`）、先切词再在词内做 BPE、tqdm | 不切词时会学出 `'y hello? Just hello'` 这种跨词 token（见 F18） |
| 1.7 用 TinyCodes 训练（1.7.1 加载功能 / 1.7.2 整理代码 / 1.7.3 训练） | `07_tiny_codes.py` + `codebot/tokenizer.py` | `load_from`（pickle）；把 06 的代码整理成模块；tie-break 改为 `(count, id0, id1)` | 产出 `codebot/merge_rules.pkl` |
| 1.8 评估分词器（1.8.1 查看学到的 token / 1.8.2 压缩效率 / 1.8.3 与其他分词器比较） | `08_eval.py` | 最早/最晚学到的 token；压缩率 = bytes/token；与 tiktoken 的 gpt2/cl100k_base 对比（代码在字符串字面量里） | “好的 tokenizer”怎么量化 |
| 1.9 编码 TinyCodes | `09_bpe_encode.py` | 全语料 encode → `np.uint16` → `.bin`；为什么 uint16 够用（vocab < 65,536） | 衔接 ch02（模型）/ ch03（`np.fromfile(..., dtype=np.uint16)` 预训练）；`dlfs6-toc.md` 记录书中 3.1.1 印出的 token 数正是 2,677,398 |

书中可能只有文字、没有代码的内容（推断）：
- 1.1.3 / 1.2.3 “问题”小节：字符级词表要覆盖全部 Unicode 码位；字节级序列变长。可能顺带讨论 word-level tokenizer 与 OOV（未登录词）。
- 1.2.1：UTF-8 的 1–4 字节前缀位模式表。
- 0 章或 1.3 之前：BPE 的来历（1994 年的压缩算法；Sennrich et al. 2016 引入 NMT；GPT-2 的 byte-level BPE）。`dlfs6-toc.md` 记录第 0 章引用了 Sennrich 2016。
- 1.8：词表大小与压缩率、embedding 参数量之间的权衡。
- 章末：朴素实现很慢（本机：训练 64 分钟，encode 10.5 分钟），为第 4 章（4.1 去重 / 4.2 增量更新 / 4.3 分块 / 4.4 并行化 / 4.6–4.7 编码加速）埋下伏笔。

### 1.3 ch07 的叙事（只有 01、02 两个脚本；ch08 没有目录）

| 文件 | 作用 | 上下文 |
|---|---|---|
| `ch07/01_bpe_train.py` | 用 ch04 的并行版 `storybot.tokenizer.train_bpe` 在 `webbot/owt_train.txt`（11.9 GB）上训练 vocab=50,000（`num_processes=8, num_chunks=64`），保存 `webbot/merge_rules.pkl` | 官方 7.1 “WebBot 的分词器” |
| `ch07/02_bpe_encode.py` | `BPETokenizer.encode_file` 把 train/valid 编码成 `owt_train.bin` / `owt_valid.bin`（memmap，uint16） | 很可能也在 7.1（`dlfs6-toc.md` 的推断）；衔接 ch09（`np.memmap(..., dtype=np.uint16)` + DDP 预训练 152M 参数的 GPT） |

据 `dlfs6-toc.md`，第 III 部是“只给提示、不给详细实现步骤”的挑战篇，ch07 的其余小节都是文字：7.2 定位瓶颈、7.3 用 Rust/C++ 加速、7.4 添加特殊 token、7.5 其他分词器、7.6 不用分词器的方法。本档案第 4 节的实测数字（朴素 BPE 训练 64 分钟、OWT 编码 0.51 MB/s）可以直接给 7.2/7.3 当素材。

**修正一处推断**：我最初根据 ch09 的 DDP/`torch.compile`/wandb 猜 ch08 是“大规模训练理论”的纯文字章。官方目次显示 ch08 是“模型【挑战篇】”（scaling law、Flash Attention、GQA、MoE、`torch.compile`、上下文长度扩展），代码都在 `webbot/model.py` 和 `ch09/01_pretrain.py` 里，所以没有单独的 `ch08/` 目录。

---

## 2. 逐脚本解析

> 计时说明：`[wall]` 是 `uv run python …` 的端到端时间（含解释器启动约 0.12 s，`import regex, tqdm` 约 0.21 s）。长任务与其它后台任务并行运行，已注明。

### 2.1 `ch01/01_char_tokenizer.py` — 字符级 tokenizer

- **目的**：演示最朴素的 tokenizer：一个字符 = 一个 token，id = Unicode code point。
- **关键类**：`CharTokenizer.encode(text) -> list[int]`、`CharTokenizer.decode(ids) -> str`
- **核心代码**：
  ```python
  class CharTokenizer:
      def encode(self, text):
          return [ord(char) for char in text]
      def decode(self, ids):
          return ''.join([chr(i) for i in ids])
  ```
- **注释翻译**：`# エンコード` = 编码；`# デコード` = 解码。
- **实际输出**（与源码注释完全一致）：
  ```text
  ['h', 'e', 'l', 'l', 'o', '世', '界', '😁']
  104
  128513
  h
  😁
  [104, 101, 108, 108, 111, 19990, 30028, 128513]
  [104, 101, 108, 108, 111, 19990, 30028, 128513]
  hello世界😁
  ```
- **[wall] 0.126 s**
- 要点：Python 的 `len("😁") == 1`（按 code point 计），但 JS 的 `"😁".length == 2`（UTF-16 code unit）——做网页 widget 时必须用 `[...str]` 或 `codePointAt`（已在 node 中验证，见 J1）。

### 2.2 `ch01/02_byte_tokenizer.py` — 字节级 tokenizer

- **目的**：用 UTF-8 字节作为 token，词表固定 256，任何文本都能编码（没有 OOV）。
- **关键类**：`ByteTokenizer.encode(text) = list(text.encode("utf-8"))`，`decode(ids) = bytes(ids).decode("utf-8")`
- **注释翻译**：`# 'A' の場合` = 以 'A' 为例；`# 'あ' の場合` = 以 'あ' 为例；`# 使用例` = 使用示例。
- **实际输出**：
  ```text
  b'A'
  [65]
  b'\xe3\x81\x82'
  [227, 129, 130]
  A
  [104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]
  hello世界😁
  ```
- **[wall] 0.120 s**
- 要点：8 个字符 → 15 个 token（世/界各 3 字节、😁 4 字节）。注意这里的 `decode` 是 **strict** 的：把一个汉字的 3 个字节拆开单独解码会抛 `UnicodeDecodeError`（F05）——这正是 04 起 `decode` 改用 `errors="replace"` 的原因。

### 2.3 `ch01/03_bpe_train.py` — BPE 学习

- **目的**：实现 BPE 训练的三个零件。
- **签名**：
  - `count_pairs(ids) -> defaultdict[(int,int), int]`
  - `merge(ids, pair, new_id) -> list[int]`（从左到右、不重叠地替换）
  - `train_bpe(text, vocab_size) -> dict[(int,int), int]`（**有序** dict：插入顺序 = 学习顺序 = rank）
- **核心代码**：
  ```python
  def train_bpe(text, vocab_size):
      ids = list(text.encode("utf-8"))
      num_merges = vocab_size - 256          # 256は初期の語彙サイズ
      merge_rules = {}
      for step in range(num_merges):
          counts = count_pairs(ids)
          if not counts:
              break
          best_pair = max(counts, key=counts.get)
          # best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))
          new_id = 256 + step
          merge_rules[best_pair] = new_id
          ids = merge(ids, best_pair, new_id)
      return merge_rules
  ```
- **注释翻译**：テキストを0~255のID列に変換 = 把文本转成 0~255 的 ID 序列；マージ回数を決定 = 确定合并次数；256は初期の語彙サイズ = 256 是初始词表大小；隣接ペアの統計を取得 = 统计相邻 pair；ペアが存在しない場合の処理 = 没有 pair 时的处理（提前结束）；最頻出ペアを選択 = 选出现次数最多的 pair；新しいトークンIDを割り当て = 分配新 token ID；マージを実行 = 执行合并。
- **实际输出**：
  ```text
  defaultdict(<class 'int'>, {(1, 2): 2, (2, 3): 1, (3, 1): 1})
  [4, 3, 4]
  {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}
  ```
  4 条 merge 依次是 `'is'`、`'is '`、`'in'`、`'He'`（34 字节 → 27 个 token，见 F09）。
- **[wall] 0.116 s**
- 要点：
  1. `max(counts, key=counts.get)` 在并列时返回**最先插入**（文本中最先出现）的 pair；被注释掉的那行是后来 `codebot/tokenizer.py` 实际采用的规则。两条规则在这段文本的第 1 步就分道扬镳（`'is'` vs `'s '`，F10）。
  2. 第 4 步所有 pair 都只出现 1 次，`'He'` 被选中纯粹因为它是文本里的第一个 pair——小语料上 BPE 的后期 merge 几乎全由 tie-break 决定。
  3. `if not counts: break` 是 2026-01-07 的 commit “add checker for vocab size” 加上的（文本太短时提前结束，最终 vocab 小于请求值）。

### 2.4 `ch01/04_bpe_tokenizer.py` — BPE tokenizer

- **目的**：用学到的 merge 规则做 encode/decode。
- **签名**：`BPETokenizer(merge_rules)`，属性 `merge_rules`、`id_to_bytes: dict[int, bytes]`、`vocab_size`；方法 `encode(text)`、`decode(ids)`。
- **核心代码**：
  ```python
  self.id_to_bytes = {i: bytes([i]) for i in range(256)}
  for (id1, id2), new_id in merge_rules.items():
      self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

  def encode(self, text):
      ids = list(text.encode("utf-8"))
      for merge_pair, new_id in self.merge_rules.items():   # 学習時の順序で
          ids = merge(ids, merge_pair, new_id)
      return ids

  def decode(self, ids):
      return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")
  ```
- **注释翻译**：IDからバイト列への対応表（0~255を登録）= ID→字节串对照表（先登记 0~255）；マージされたトークンは元のトークンのバイト列を連結 = 合并 token 的字节串 = 两个原 token 字节串相连；語彙サイズを設定 = 设置词表大小；学習時の順序でマージルールを適用 = 按训练时的顺序应用 merge 规则；各トークンIDを対応するバイト列に変換 = 每个 id 转成对应字节串；すべてのバイト列を連結 = 拼接全部字节串；バイト列をUTF-8テキストに変換 = 把字节串按 UTF-8 解码；学習済みのマージルール = 已训练好的 merge 规则；トークナイザーを作成 = 创建 tokenizer；テキストをエンコード = 编码文本。
- **实际输出**：
  ```text
  [259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]
  Hello世界😁
  ```
- **[wall] 0.120 s**
- 要点：encode 的正确性依赖 dict 的**插入顺序**（Python 3.7+ 保证）。复杂度 O(#merges × len)，743 条 merge 时对整份 tiny_codes 要 10.5 分钟（见 2.9）。

### 2.5 `ch01/05_special_token.py` — 特殊 token

- **目的**：引入 `<|endoftext|>`：训练时按它切开（merge 不跨文档），encode 时把它映射成单个 id。
- **签名**：`count_pairs(ids, counts=None)`（可累加）、`train_bpe(input_text, vocab_size, end_token="<|endoftext|>")`、`BPETokenizer(merge_rules, end_token=...)`，新增属性 `end_token`、`end_token_id = 256 + len(merge_rules)`，私有方法 `_encode_text(text)`。
- **核心代码**：
  ```python
  texts = input_text.split(end_token)
  ids_list = [list(text.encode("utf-8")) for text in texts]
  num_merges = vocab_size - 256 - 1          # 基本語彙（0-255）+ 終了トークン用（1個）を除いた分
  ...
  def encode(self, input_text):
      pattern = '(' + re.escape(self.end_token) + ')'   # 捕获组 → split 结果里保留分隔符
      for text in re.split(pattern, input_text):
          if text == self.end_token: all_ids.append(self.end_token_id)
          else: all_ids.extend(self._encode_text(text))
  ```
- **注释翻译**：特殊トークンでテキストを分割 = 用特殊 token 切分文本；基本語彙（0-255）+ 終了トークン用（1個）を除いた分がマージ回数 = 去掉基础词表（0–255）和终止 token（1 个）后剩下的就是 merge 次数；隣接ペアの頻度を集計 = 汇总相邻 pair 的频次。
- **实际输出**：
  ```text
  {(105, 115): 256, (256, 32): 257, (105, 110): 258}
  [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 259]
  Hello world!<|endoftext|>
  ```
- **[wall] 0.134 s**
- 要点：同样 `vocab_size=260`，03 学 4 条 merge 而 05 只学 3 条——因为 vocab 的语义从“256 + merges”变成了“256 + merges + 1 个 special token”。**同一个 id 259 在 04 里是 `'He'`，在 05 里是 `<|endoftext|>`**。注意 05 用的是标准库 `re`，06/codebot 换成了第三方 `regex`（`import regex as re`），两者的 `escape/split` 在这里行为相同。

### 2.6 `ch01/06_pretokenize.py` — 事前分词（pre-tokenization）

- **目的**：先用 GPT-2 正则把文本切成“词”，BPE 只在词内部合并。
- **签名**：`pretokenize(text) -> list[str]`；`train_bpe` 的 `ids_list` 变成“每个 pre-token 一个 id 列表”；`encode(input_text, show_progress=False)`。
- **核心代码**：
  ```python
  pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  return re.findall(pattern, text)            # re = regex 模块（标准库 re 不支持 \p{L}）
  ```
  正则 6 个分支的含义：英文缩约（'s 't 'd 'm 'll 've 're，只认小写）｜可带 1 个前导空格的字母串｜可带前导空格的数字串｜可带前导空格的“其它符号”串｜后面不跟非空白的空白串（把最后一个空格留给下一个词）｜剩余空白。
- **注释翻译**：GPT-2で使用されている正規表現パターン = GPT-2 使用的正则模式；ステップ1: 特殊トークンで分割 = 第 1 步：按特殊 token 切分；ステップ2: 各テキスト片を事前トークン化 = 第 2 步：对每段文本做预分词；ID列に変換 = 转成 ID 序列；残りは元のコードと同じ（ただしtqdmを追加）= 其余同前（只加了 tqdm）；tqdmで進捗表示 = 用 tqdm 显示进度；show_progressがTrueならtqdmで進捗表示 = show_progress 为 True 时显示进度条；各事前トークンをBPEエンコード = 对每个 pre-token 做 BPE 编码；事前トークン化対応のBPE学習 = 支持预分词的 BPE 训练；各トークンIDをデコードして確認 = 逐个解码 token ID 来确认。
- **实际输出**（tqdm 进度条写到 stderr）：
  ```text
  [262, 260, 33]
  Say hello!
  262 -> 'Say'
  260 -> ' hello'
  33 -> '!'
  ```
  vocab_size=270 → 13 条 merge：`' h',' he',' hel',' hell',' hello','Sa','Say',' W',' Wh',' Why',' J',' Ju',' Jus'`，end token id = 269。
- **[wall] 0.255 s**
- 要点：pre-token 的空格是**前导**的（`' hello'` 而不是 `'hello '`）。同一句话换成 codebot 的 tie-break 会学出完全不同的 13 条 merge 顺序（`'lo','llo','he','hello',' hello',…`，F18）。

### 2.7 `ch01/07_tiny_codes.py` + `codebot/tokenizer.py` — 训练 CodeBot tokenizer

- **目的**：把 06 的实现整理成模块 `codebot/tokenizer.py`，在 `codebot/tiny_codes.txt` 上训练 vocab=1000 并 `pickle.dump` 到 `codebot/merge_rules.pkl`。
- **`codebot/tokenizer.py` 的 API**（ch03 全部 CodeBot 脚本都 import 它）：
  - `pretokenize(text) -> list[str]`
  - `count_pairs(ids, counts=None)`、`merge(ids, pair, new_id)`
  - `train_bpe(input_text: str, vocab_size, end_token="<|endoftext|>") -> dict`
  - `class BPETokenizer(merge_rules, end_token="<|endoftext|>")`：`load_from(filepath)`（staticmethod，读 pickle）、`_encode_text`、`encode(input_text, show_progress=False)`、`decode(ids)`；属性 `merge_rules / end_token / end_token_id / id_to_bytes / vocab_size`
- **与 ch01/06 的唯一算法差异**：tie-break
  ```python
  # best_pair = max(counts, key=counts.get)
  best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))
  ```
  并列时取 **id 元组最大**者，结果与 dict 迭代顺序无关（可复现）。
- **注释翻译**：語彙サイズ = 词表大小；学習済みマージルールをファイルに保存 = 把训练好的 merge 规则存到文件。
- **运行**：原脚本会**覆盖仓库里的** `codebot/merge_rules.pkl`，所以跑的是补丁副本 `SC\07_tiny_codes_patched.py`（只改了 `os.chdir` 与输出路径，训练参数原样：完整语料、vocab 1000）。
- **实际输出**（完整语料、vocab 1000，原算法未改）：
  ```text
  Training BPE: 100%|██████████| 743/743 [1:03:54<00:00,  5.16s/it]
  train_bpe wall: 3837.1s, merges: 743
  ```
  与仓库的 `codebot/merge_rules.pkl` 比较：dict 内容与顺序完全相同，**pickle 文件逐字节相同**（7,421 B，sha256 前缀 `7b1667406b2128a4`）。
- **[wall] 3,837.5 s（64 分钟，单核）**。注意：运行期间机器上同时有其它 agent 的多个重负载 Python 进程，前 200 步约 2.6 s/步、后期 6–7 s/步；独占机器时应明显更快。早期每步慢是因为要在 1,826,876 个 pre-token 列表上全量重数 pair。
- **独立复现**：`SC\js\bpe_train.mjs`（纯 JS、按 pre-token 去重计数、同样的 `(count, id0, id1)` tie-break）在完整语料（先把 CRLF 归一为 LF）上训练 → **743 条 merge 与 `merge_rules.pkl` 逐条相同**，node 用时 28.7 s（其中 pre-tokenize 1.6 s；与其它任务并行；负载较低时复测 22.1 s）。这与上面的 Python 重训一起证明了 `merge_rules.pkl` 的生成方式：codebot 的 `train_bpe` + `(count,id0,id1)` tie-break + LF 文本 + vocab 1000。
- **merge_rules.pkl 内容摘要**（F19/F20）：`dict`，743 项，pickle protocol 4，7,421 字节；值严格为 256…998；每条 merge 只引用比自己小的 id。最早 10 个 token：`'  '`(256)、`'    '`(257)、`'in'`、`'   '`、`'re'`、`'or'`、`' ='`、`'st'`、`'te'`、`'\n   '`(265)；最长 token：`'\n'+15 空格`(612，16 B)、`' BeautifulSoup'`(777，14 B)。合并 token 全是 ASCII（语料本身 0 个非 ASCII 字符）。token 字节长度分布：2B×237、3B×214、4B×129、5B×77、6B×34、7B×22、8B×14、9B×9、10B×2、12B×2、13B/14B/16B 各 1。
- **数据速览**：`tiny_codes.txt` 开头是这样的（每个程序之间是 `\n<|endoftext|>\n`）：
  ```text
  tasks = []
  while True:
      task = input('Enter a task or type 'done' to finish: ')
      if task == 'done': break
      tasks.append(task)
  print(f'Your to-do list for today: {tasks}')
  <|endoftext|>
  shopping_list = {}
  ...
  ```
  注意第 3 行的嵌套单引号——这不是合法 Python。用 Python 3.12 的 `ast.parse` 逐个检查：21,836 个程序里 **2,426 个（11.1%）解析失败**；程序长度最短 1 字符、中位 216、最长 1,727 字符。这是生成数据的质量问题，对 tokenizer 无影响，但值得在教程里点一句（CodeBot 学到的“语法”本身就带噪声）。
  `tiny_codes_sft.json`（ch03 的 SFT 用）是 1,092 条 `{"instruction", "response"}`，混合了闲聊（`"Hello"` → `"Hello. What can I help you with?"`）、代码问答（`"How do I remove whitespace from string?"` → `"trimmed = my_string.strip()"`）和算术（`"9+8="` → `"17"`）。
- **语料统计**（F47 等）：`split("<|endoftext|>")` 得 21,836 段（文件以最后一个程序结尾，没有尾随 end token；除第一段外每段都以 `'\n'` 开头），1,826,876 个 pre-token 出现、只有 **30,535** 种；初始 distinct byte pair 2,859 种、出现总数 4,376,302；最频繁 pair 是两个空格（335,027 次）——这就是第一条 merge 是 `'  '` 的原因。pre-token 频次 Top6：`'\n'` 160,460、`'_'` 74,293、`'.'` 69,082、`'('` 63,425、`' ='` 53,183、`','` 48,355。

### 2.8 `ch01/08_eval.py` — 评估

- **目的**：(1) 看最早/最晚学到的 10 个 token；(2) 在前 10,000 字符上算压缩率；(3)（字符串字面量里、不会执行）用 tiktoken 比较 GPT-2 与 cl100k_base。
- **核心代码**：
  ```python
  sample_text = open("codebot/tiny_codes.txt").read()[:10000]
  byte_count = len(sample_text.encode("utf-8"))
  compression_ratio = byte_count / len(tokenizer.encode(sample_text))
  ```
- **注释/字符串翻译**：最初に学習された10個 = 最先学到的 10 个；最後に学習された10個 = 最后学到的 10 个；圧縮率を測定 = 测量压缩率；最初の10000文字 = 前 10000 个字符；圧縮効率 = 压缩效率；バイト数 = 字节数；トークン数 = token 数；圧縮率: X倍（平均 X バイト/トークン）= 压缩率 X 倍（平均每 token X 字节）；以下はGPT系モデルのエンコードと圧縮率比較（tiktoken使用）= 以下是与 GPT 系模型的编码/压缩率比较（使用 tiktoken）；語彙サイズ = 词表大小。
- **实际输出**：
  ```text
  最初に学習された10個:
    ID 256: '  '
    ID 257: '    '
    ID 258: 'in'
    ID 259: '   '
    ID 260: 're'
    ID 261: 'or'
    ID 262: ' ='
    ID 263: 'st'
    ID 264: 'te'
    ID 265: '
     '

  最後に学習された10個:
    ID 990: ' '''
    ID 991: 'Error'
    ID 992: 'atter'
    ID 993: 'St'
    ID 994: 'thod'
    ID 995: 'cle'
    ID 996: ' on'
    ID 997: ' R'
    ID 998: 'ery'
    ID 999: '<|endoftext|>'

  === 圧縮効率 ===
  バイト数: 10,000
  トークン数: 4,709
  圧縮率: 2.12倍（平均 2.12 バイト/トークン）
  ```
- **[wall] 1.743 s**
- **tiktoken 注释块实际运行结果**（`uv run --with tiktoken`，tiktoken 0.12.0，首次运行需下载编码文件，18.8 s）：
  ```text
  GPT-2: 語彙サイズ 50,257, tokens 3,232, 圧縮率 3.09倍
  cl100k_base: 語彙サイズ 100,277, tokens 2,516, 圧縮率 3.97倍
  o200k_base: 語彙サイズ 200,019, tokens 2,512, 圧縮率 3.98倍     ← o200k 是我额外加的
  ```
- 要点：ID 265 的输出跨行，因为 token 本身是 `'\n   '`（换行+3 空格）。`byte_seq.decode("utf-8")` 在这里是 strict 的——对 CodeBot（全 ASCII）安全，但同样的循环用在 WebBot 词表上会在 157 个“半个字符”的 token 上抛异常（F35）。

### 2.9 `ch01/09_bpe_encode.py` — 生成 `tiny_codes.bin`

- **目的**：用 `merge_rules.pkl` 编码整份语料，存成 `np.uint16` 原始二进制（无 header）。
- **核心代码**：
  ```python
  ids = tokenizer.encode(text, show_progress=True)
  np.array(ids, dtype=np.uint16).tofile("codebot/tiny_codes.bin")
  ```
  读回：`np.fromfile(path, dtype=np.uint16)`（ch03/01_pretrain.py 第 48 行）。
- **注释翻译**：トークナイザを読み込み = 加载 tokenizer；テキストをトークンIDに変換（進捗バーを表示）= 把文本转成 token ID（显示进度条）；numpy配列に変換して保存 = 转成 numpy 数组并保存；トークンID数 = token ID 数；最初の20個のトークンID = 前 20 个 token ID。
- **运行**：补丁副本 `SC\09_bpe_encode_patched.py`（输出改写到 scratch，其余不变），完整语料。
- **实际输出**：
  ```text
  Encoding: 100%|██████████| 43671/43671 [10:28<00:00, ...]
  encode wall: 628.7s
  トークンID数: 2677398
  最初の20個のトークンID: [116 528 821 262 544  10 119 717 660  58 265 268 528 107 262 595 338  69
   670 273]
  ```
  （43,671 = 21,835 个 end token + 21,836 段文本——`re.split` 带捕获组的结果。）前 20 个 id 解码为 `"tasks = []\nwhile True:\n    task = input('Enter a"`。
- **[wall] 629.6 s（10.5 分钟）**；输出与仓库 `codebot/tiny_codes.bin` **`cmp` 逐字节一致**，sha256 均为 `d1e082b13158191bbe990a3b926fd91002a18ddecab082fc8380bb9f4f58b495`。
- 加速对照（同一语料、结果同样逐字节一致）：`storybot` 的按 rank 优先合并 encode 53.9 s；再加一个 pre-token 级缓存（只有 30,535 种 pre-token）9.9 s。

### 2.10 `ch07/01_bpe_train.py` — 训练 WebBot tokenizer（vocab 50,000）

- **目的**：在 `webbot/owt_train.txt`（11,920,511,059 B）上训练 50K BPE，存 `webbot/merge_rules.pkl`。
- **代码**（全部逻辑在 `storybot/tokenizer.py`）：
  ```python
  if __name__ == '__main__':            # Windows 的 spawn 式 multiprocessing 必须有这个守卫
      merge_rules = train_bpe("webbot/owt_train.txt", 50000, num_processes=8, num_chunks=64)
      pickle.dump(merge_rules, open("webbot/merge_rules.pkl", "wb"))
  ```
- **`storybot.tokenizer.train_bpe(file_path, vocab_size, end_token, num_processes=8, num_chunks=8)` 的算法要点**（ch04 的成果，ch07 直接复用）：
  1. `find_chunk_boundaries`：按文件大小等分 `num_chunks` 份，每个边界向后以 4096 B 为步长搜索 `b"<|endoftext|>"`，把边界挪到 end token 的起点；去重排序。
  2. `Pool.imap(pretoken_chunk, …)`：每个进程读自己的字节区间，`decode("utf-8", errors="ignore")`，按 end token 切开、pre-tokenize，返回 `{pretoken: count}`；主进程汇总。
  3. `ids_counts = {tuple(pretoken.encode()): count}`（**去重 + 计数**），`pair_counts` 按 `count` 加权，`pair_to_ids[pair]` = 含该 pair 的 pre-token 集合（倒排缓存）。
  4. 每步：`best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p[0], p[1]))`；只更新 `pair_to_ids[best_pair]` 里的 pre-token：先减旧 pair 频次、再加新 pair 频次。
- **注释翻译（storybot/tokenizer.py）**：ファイルをバイナリモードで開く = 以二进制模式打开；チャンクの開始位置を計算（等間隔）= 等间隔计算 chunk 起点；境界から先読みするバイト数 = 从边界往后读的字节数；境界位置の調整（終了トークンを探す）= 调整边界（寻找终止 token）；見つかった場合、その位置を新しい境界とする = 找到就把该位置作为新边界；重複を除去し、ソートして返す = 去重排序后返回；ステップ1: チャンクの準備 / ステップ2: 並列処理 / ステップ3: 結果を統合 = 准备 chunk / 并行处理 / 合并结果；事前トークンをID列に変換 = 把 pre-token 转成 ID 序列；キャッシュ（に登録）= 缓存（登记）；best_pairを含むids列をキャッシュから取得 = 从缓存取出含 best_pair 的序列；もう使わないので削除 = 不再使用故删除；影響のあるID列だけを更新 = 只更新受影响的序列；古いペア頻度を減少 / 新しいペア頻度を増加 = 减旧 pair 频次 / 加新 pair 频次。
- **运行**：完整版需要 11.9 GB 数据（未下载）。缩小版 `SC\ch07_01_bpe_train_shrunk.py`：**数据换成 `owt_valid.txt` 的前 30 MiB（HTTP Range 下载），vocab 50,000→2,000**，进程数/chunk 数不变，输出写 scratch。
- **实际输出**（与另一个单线程训练任务并行）：
  ```text
  Pretokenizing: 100%|██████████| 64/64 [00:05<00:00, 12.66it/s]
  Training BPE: 100%|██████████| 1743/1743 [02:24<00:00, 12.10it/s]
  train_bpe wall: 153.6s, merges: 1743
  identical-prefix merges vs shipped webbot rules: 15; same pair anywhere in shipped rules: 854/1743
  ```
  （“same pair” 按 id 对比会低估；按字节串对比：30 MB 学到的 1,743 个 token 中 1,741 个在官方 50K 词表里，1,673 个落在官方最先学到的 1,743 个里——F45。）
- **[wall] 155.6 s**

### 2.11 `ch07/02_bpe_encode.py` — 编码 OWT

- **目的**：`BPETokenizer.encode_file(src, dst, num_processes=8)` 把 train/valid 编码为 `.bin`。
- **`encode_file(file_path, output_file, num_processes=4, num_chunks=64, cache_dir="bpe_cache")`**：同样的边界切分 → 每个进程 `encode` 一个 chunk 并写 `bpe_cache/chunk_XXXXX.npy`（实为 raw uint16）→ 汇总总 token 数 → `np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(total,))` 逐块拷贝 → `finally: shutil.rmtree(cache_dir)`。
- **`_encode_text`（按 rank 优先）**：
  ```python
  while len(ids) > 1:
      counts = count_pairs(ids)
      best_pair = min(counts, key=lambda p: self.merge_rules.get(p, float('inf')))
      if best_pair not in self.merge_rules: break
      ids = merge(ids, best_pair, self.merge_rules[best_pair])
  ```
  与 codebot 的“按训练顺序逐条套用”**结果完全相同**（F30 + 全语料验证），但只对实际出现的 pair 工作。
- **注释翻译**：現在のペアを取得（❶）= 取当前全部 pair；最優先ペアを特定（❷）= 找优先级最高（rank 最小）的 pair；マージ可能性の確認（❸）= 确认能否合并；マージの実行（❹）= 执行合并；存在しないペアは最低優先度 = 不存在的 pair 优先级最低；チャンクを処理してディスクにキャッシュ = 处理 chunk 并缓存到磁盘；ステップ3: memmapファイルを作成 = 创建 memmap 文件；ステップ4: バッチ処理でキャッシュからmemmapへ書き込み = 分批把缓存写入 memmap；キャッシュは削除 = 删除缓存。
- **运行**：缩小版 `SC\ch07_02_bpe_encode_shrunk.py`：**只编码 `owt_valid.txt` 前 30 MiB**，使用 HF 上的官方 `webbot/merge_rules.pkl`（533,514 B，放在 `SC\dl\webbot\`），`cache_dir` 与输出改到 scratch。
- **实际输出**：
  ```text
  tokens: 7,019,785 · 31,457,280 B → 4.481 B/token · ≈61.7 s（throughput 0.51 MB/s, 8 procs）
  max id 49999 · end-token count 6315
  first 20 ids: [76 4091 2147 36662 44 11969 46 745 330 1127 45808 19138 8074 381 283 308 2583 351 625 6691]
  decoded head: 'LOUISVILLE, Ky. — A few unflattering reviews are to be expected with any hotel, ...'
  ```
  并用 HTTP Range 取了官方 `owt_valid.bin` 的前 4,000 字节：**前 2,000 个 id 与本机重编码完全一致**——官方 .bin 就是这份 merge_rules + 这个 encoder 的产物。同一 30 MiB 上 GPT-2（tiktoken）为 7,166,544 tok = 4.39 B/token，WebBot 50K 比 GPT-2 50K 少约 2% token。
- **[wall] 63.6 s**


---

## 3. 可最小化验证的事实清单

### 3.0 说明与索引

- 每条事实都给出：一个**真实运行过**的最小片段 + 本机观测输出 + 是否能在浏览器（Pyodide，只用 stdlib/numpy/regex）里跑。
- 片段默认在仓库根目录运行（`cd C:\Users\chenf\Projects\_refs\dlfs6`，`PYTHONUTF8=1 uv run python`）；需要 tiktoken 的用 `uv run --with tiktoken python`。
- 片段由 `SC\facts_runner.py`、`facts_runner2.py`、`facts_runner3.py` 批量执行并自动生成下面的 Markdown（代码与输出一字未改）。
- **Pyodide 实测**：把所有不依赖 tiktoken 的片段（35 个）连同 `codebot/tokenizer.py`、`storybot/tokenizer.py`、pkl/txt/bin 文件装进 Pyodide 0.29.4（Node 内，`loadPackage(["numpy","regex","tqdm"])`）执行：**34/35 输出与 CPython 逐字相同**；唯一不同的是 F29（Pyodide 带的是 numpy 2.2.5，uint16 溢出直接抛 `OverflowError`，而仓库环境 numpy 1.26.4 是静默回绕+DeprecationWarning）。脚本：`SC\export_facts.py` + `SC\pyodide\run_facts.mjs`。Pyodide 0.29.4 的 lock 里也有 `tiktoken 0.9.0`，但它要在运行时联网下载 BPE 文件（浏览器里有 CORS 问题，未测）。
- **纯 JS 实测**（见 3.4）：encode 完全移植（29/29 Unicode 用例 + 200k 字符样本逐 id 一致）；训练器移植后在完整语料上**逐条复现** `merge_rules.pkl`。

| ID | 结论（一句话） | 浏览器 |
|---|---|---|
| F01 | `"hello世界😁"` 的 code point 是 `[104,101,108,108,111,19990,30028,128513]`，`len==8` | ✅ |
| F02 | `sys.maxunicode == 0x10FFFF`，`chr(0x110000)` 抛 ValueError | ✅ |
| F03 | UTF-8 长度：A=1、é=2、あ/世=3、😁=4 字节 | ✅ |
| F04 | 字节级：8 字符 → 15 个 id，可逆 | ✅ |
| F05 | 截断的 `'あ'` 字节：strict 抛错，replace→`'\ufffd'`，ignore→`''` | ✅ |
| F06 | NFC `é`=[233]，NFD=[101,769]；UTF-8 分别 [195,169] / [101,204,129] | ✅ |
| F07 | `count_pairs([1,2,3,1,2])`、`merge(...,(1,2),4)==[4,3,4]` | ✅ |
| F08 | `[7,7,7]`：pair 计数 2，但 merge 只替换 1 次 | ✅ |
| F09 | ch01/03 的 4 条 merge = `is`,`is `,`in`,`He`；34→27 token | ✅ |
| F10 | 并列时 ch01 规则选 `'is'`，codebot 规则选 `'s '` | ✅ |
| F11 | `"Hello世界😁"` → 14 个 id（`He`=259，其余退回字节） | ✅ |
| F12 | ch01/05：end id 259、vocab 260、`<|endoftext|>` 本是 13 字节 | ✅ |
| F13 | `re.split` 必须带捕获组才保留 special token | ✅ |
| F14 | GPT-2 正则切分示例（前导空格、缩进、数字） | ✅ |
| F15 | 缩约规则只认小写（`DON'T` 不拆成 `'T`） | ✅ |
| F16 | `f'done'` → `f`,`'d`,`one`,`')` | ✅ |
| F17 | pre-tokenize 覆盖全部字符（10 个刁钻 Unicode 用例） | ✅ |
| F18 | 不做 pre-tokenize 会学出 `'y hello? Just hello'` | ✅ |
| F19 | merge_rules.pkl：743 条，id 256..998 连续，vocab 1000 | ✅ |
| F20 | CodeBot 词表：最长 token、长度分布、全 ASCII | ✅ |
| F21 | CodeBot 对 emoji/CJK/组合符/ZWJ/RTL round-trip 无损；CJK 每字 3 token | ✅ |
| F22 | `"世"` → 3 个 id，单独解码各是 `'\ufffd'` | ✅ |
| F23 | 文本里的 `<|endoftext|>` 被编码为 999（注入） | ✅ |
| F24 | 缩进：`'\n   '`(265)、`'\n       '`(293) 各 1 个 token | ✅ |
| F25 | tiny_codes.bin：5,354,796 B / uint16 / 2,677,398 tok；999 的个数 == 21,835 | ✅（拉取 5.4 MB） |
| F26 | `decode(tiny_codes.bin) == tiny_codes.txt`（LF） | ✅（拉取 12 MB） |
| F27 | 压缩率：前 10k 字符 2.1236，全语料 2.4229 | ✅ |
| F28 | ln(1000)=6.9078；tiny_codes unigram 熵 5.7626；只用到 839 个 id | ✅ |
| F29 | uint16 溢出：numpy 1.26 回绕（65536→0，70000→4464），numpy 2 抛错 | ✅（结果因 numpy 版本而异） |
| F30 | 两种 encode（逐条套用 vs rank 优先）结果相同 | ✅（≈85 s） |
| F31 | DLFS6 正则 ≠ tiktoken gpt2 的 pat_str 字符串，但切分结果相同 | ❌ tiktoken |
| F32 | 10k 字符压缩率：CodeBot 2.12 / GPT-2 3.09 / cl100k 3.97 / o200k 3.98 | ❌ tiktoken |
| F33 | tiktoken 默认拒绝文本中的 special token | ❌ tiktoken |
| F34 | 8 空格缩进：GPT-2 14 tok，cl100k 8，CodeBot 7 | ❌ tiktoken |
| F35 | WebBot pkl：49,743 merge；157 个 token 单独不是合法 UTF-8；最长是 mojibake `ÃÂ…` | ✅（拉取 533 KB） |
| F36 | WebBot 与 GPT-2 词表重合 88.2%；前 10 个 merge 几乎相同 | ❌ tiktoken |
| F37 | StoryBot pkl：9,743 merge；前 10 个 token | ✅ |
| F38 | 领域错配表（code/story/web × 4 个 tokenizer） | ⚠️ 去掉 GPT-2 列即可 |
| F39 | CodeBot 有 3 个从未出现的 token（`eauti`,`eautifulS`,`bonac`） | ✅ |
| F40 | Windows CRLF：文本模式读自动转 LF；`newline=""` 不转，5,000 字符多出 120 个 token | ✅ |
| F41 | loss→bits/byte 换算 | ✅ |
| F42 | ch03/03_alpaca.py 注释第 30 个 id 写的是 930（`alize`），实际是 932（`' you'`） | ✅ |
| F43 | ch04/ch07 的快速训练器与 ch01 朴素训练器 merge 完全相同；CRLF 文件会让第 2 条 merge 变成 `\r\n` | ❌ multiprocessing（逻辑可移植） |
| F44 | DLFS6（比 id）vs CS336（比 bytes）tie-break 学出不同 merge | ✅ |
| F45 | 30 MB OWT 训练的前 15 条 merge 与官方 50K 规则相同；1,673/1,743 落在官方前 1,743 个 token 里 | ⚠️ 需样本 |
| F46 | snake_case：GPT-2 正则在 `_` 处切开，cl100k 保留 `_time`；数字切法不同 | ⚠️ cl100k 部分需 tiktoken |
| F47 | tiny_codes 有 1,826,876 个 pre-token 出现，只有 30,535 种；0 个非 ASCII 字符 | ✅ |
| F48 | CodeBot GPT 初始化时 loss ≈ ln(1000)（实测 7.02 vs 6.91） | ❌ torch |
| F49 | 用书中原算法完整重训（64 min）→ 与 merge_rules.pkl 的 pickle **逐字节相同** | ❌（≈1 h 单核） |
| F50 | 压缩率曲线：vocab 257→1.04、357→1.62、457→1.86、657→2.14、1000→2.42 B/token（边际递减） | ✅（≈15–30 s） |
| J1 | JS 移植 encode 与 Python 逐 id 一致（需显式空白字符类） | ✅ JS |
| J2 | JS 训练器在完整语料上逐条复现 merge_rules.pkl（28.7 s） | ✅ JS（Web Worker） |

### 3.1 第一批（字符/字节/BPE 基础，CodeBot 产物）

#### F01 字符 tokenizer = Unicode code point；len 按 code point 计

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
text = "hello世界😁"
print([ord(c) for c in text], len(text))
print(''.join(chr(i) for i in [104, 101, 108, 108, 111, 19990, 30028, 128513]))
```

输出:

```text
[104, 101, 108, 108, 111, 19990, 30028, 128513] 8
hello世界😁
```

#### F02 code point 空间上限 0x10FFFF：字符级词表理论上要 1,114,112 个槽位

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
import sys
print(sys.maxunicode, hex(sys.maxunicode), sys.maxunicode + 1)
try:
    chr(0x110000)
except ValueError as e:
    print("ValueError:", e)
```

输出:

```text
1114111 0x10ffff 1114112
ValueError: chr() arg not in range(0x110000)
```

#### F03 UTF-8 变长：ASCII 1 字节、é 2、あ/世 3、emoji 4

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
for ch in ["A", "é", "あ", "世", "😁"]:
    b = ch.encode("utf-8")
    print(repr(ch), hex(ord(ch)), len(b), list(b))
```

输出:

```text
'A' 0x41 1 [65]
'é' 0xe9 2 [195, 169]
'あ' 0x3042 3 [227, 129, 130]
'世' 0x4e16 3 [228, 184, 150]
'😁' 0x1f601 4 [240, 159, 152, 129]
```

#### F04 ByteTokenizer：8 个字符 → 15 个 id，且可逆

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
text = "hello世界😁"
ids = list(text.encode("utf-8"))
print(len(text), len(ids), ids)
print(bytes(ids).decode("utf-8") == text)
```

输出:

```text
8 15 [104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]
True
```

#### F05 截断的多字节序列无法 strict 解码；errors='replace' 得到 U+FFFD

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
b = "あ".encode("utf-8")[:2]          # 只取 3 字节中的前 2 个
try:
    b.decode("utf-8")
except UnicodeDecodeError as e:
    print("UnicodeDecodeError:", e.reason)
print(repr(b.decode("utf-8", errors="replace")), repr(b.decode("utf-8", errors="ignore")))
```

输出:

```text
UnicodeDecodeError: unexpected end of data
'�' ''
```

#### F06 NFC 与 NFD：看起来一样的 'é' 是两种不同的 id 序列

浏览器(Pyodide)可跑: yes · 本机耗时 0.02s

```python
import unicodedata
nfc = "\u00e9"
nfd = unicodedata.normalize("NFD", nfc)
print(nfc == nfd, len(nfc), len(nfd))
print([ord(c) for c in nfc], [ord(c) for c in nfd])
print(list(nfc.encode()), list(nfd.encode()))
print(unicodedata.normalize("NFC", nfd) == nfc)
```

输出:

```text
False 1 2
[233] [101, 769]
[195, 169] [101, 204, 129]
True
```

#### F07 count_pairs / merge 基本行为（书中示例）

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from collections import defaultdict
def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts
def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out
print(dict(count_pairs([1, 2, 3, 1, 2])))
print(merge([1, 2, 3, 1, 2], (1, 2), 4))
```

输出:

```text
{(1, 2): 2, (2, 3): 1, (3, 1): 1}
[4, 3, 4]
```

#### F08 重叠 pair：count_pairs 数到 2 次，但 merge 只能替换 1 次（从左到右、不重叠）

浏览器(Pyodide)可跑: yes · 本机耗时 0.07s

```python
from codebot.tokenizer import count_pairs, merge
ids = [7, 7, 7]
print(dict(count_pairs(ids)), merge(ids, (7, 7), 8))
ids = [7, 7, 7, 7]
print(dict(count_pairs(ids)), merge(ids, (7, 7), 8))
```

输出:

```text
{(7, 7): 2} [8, 7]
{(7, 7): 3} [8, 8]
```

#### F09 ch01/03 的 train_bpe 复现：vocab 260 → 4 条 merge

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import count_pairs, merge
def train_bpe(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merge_rules = {}
    for step in range(vocab_size - 256):
        counts = count_pairs(ids)
        if not counts:
            break
        best_pair = max(counts, key=counts.get)
        merge_rules[best_pair] = 256 + step
        ids = merge(ids, best_pair, 256 + step)
    return merge_rules, ids
text = "Hello world! This is BPE training."
rules, ids = train_bpe(text, 260)
print(rules)
print(len(text.encode()), "->", len(ids))
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), n in rules.items():
    vocab[n] = vocab[a] + vocab[b]
print({n: vocab[n].decode() for n in rules.values()})
```

输出:

```text
{(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}
34 -> 27
{256: 'is', 257: 'is ', 258: 'in', 259: 'He'}
```

#### F10 平局打破（tie-break）规则不同 → 第一条 merge 就不同

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import count_pairs
ids = list("Hello world! This is BPE training.".encode())
counts = count_pairs(ids)
top = max(counts.values())
print("count==%d 的 pair:" % top, [(p, bytes(p).decode()) for p, c in counts.items() if c == top])
a = max(counts, key=counts.get)                                  # ch01/03~06：先出现者胜
b = max(counts, key=lambda p: (counts[p], p[0], p[1]))           # codebot：id 字典序最大者胜
print("ch01 规则:", a, repr(bytes(a).decode()), "| codebot 规则:", b, repr(bytes(b).decode()))
```

输出:

```text
count==2 的 pair: [((105, 115), 'is'), ((115, 32), 's '), ((105, 110), 'in')]
ch01 规则: (105, 115) 'is' | codebot 规则: (115, 32) 's '
```

#### F11 BPETokenizer(ch01/04)：'He' 被合并为 259，非 ASCII 退回原始字节

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
merge_rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}
from codebot.tokenizer import merge
def encode(text):
    ids = list(text.encode("utf-8"))
    for pair, new_id in merge_rules.items():
        ids = merge(ids, pair, new_id)
    return ids
ids = encode("Hello世界😁")
print(ids, len(ids), len("Hello世界😁".encode()))
```

输出:

```text
[259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129] 14 15
```

#### F12 special token：<|endoftext|> 作为 1 个 id（而不是 13 个字节）

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import BPETokenizer
rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258}      # ch01/05 学到的 3 条
tok = BPETokenizer(rules)
print(tok.end_token_id, tok.vocab_size, len("<|endoftext|>".encode()))
ids = tok.encode("Hello world!<|endoftext|>")
print(ids, repr(tok.decode(ids)))
```

输出:

```text
259 260 13
[72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 259] 'Hello world!<|endoftext|>'
```

#### F13 split 不加括号会丢掉分隔符；加了捕获组才会保留（encode 依赖这一点）

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
import re
s = "A<|endoftext|>B<|endoftext|>"
print(re.split(re.escape("<|endoftext|>"), s))
print(re.split("(" + re.escape("<|endoftext|>") + ")", s))
```

输出:

```text
['A', 'B', '']
['A', '<|endoftext|>', 'B', '<|endoftext|>', '']
```

#### F14 GPT-2 pre-tokenize 正则的切分结果（空格黏在后一个词上）

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import pretokenize
print(pretokenize("Say hello! Why hello? Just hello."))
print(pretokenize("x  =   1 + 23456"))
print(pretokenize("    return x\n\n"))
```

输出:

```text
['Say', ' hello', '!', ' Why', ' hello', '?', ' Just', ' hello', '.']
['x', ' ', ' =', '  ', ' 1', ' +', ' 23456']
['   ', ' return', ' x', '\n\n']
```

#### F15 缩约词规则只认小写：I'm/don't 拆出 'm/'t，大写 DON'T 不行

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import pretokenize
print(pretokenize("I'm sure they'll say don't"))
print(pretokenize("DON'T WE'LL"))
```

输出:

```text
['I', "'m", ' sure', ' they', "'ll", ' say', ' don', "'t"]
['DON', "'", 'T', ' WE', "'", 'LL']
```

#### F16 代码里的怪现象：f'done' 被切成 f / 'd / one / '（缩约词规则误伤）

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import pretokenize
print(pretokenize("print(f'done')"))
print(pretokenize("x = 'stop'"))
print(pretokenize("print(f'total: {t}')"))
```

输出:

```text
['print', '(', 'f', "'d", 'one', "')"]
['x', ' =', " '", 'stop', "'"]
['print', '(', 'f', "'t", 'otal', ':', ' {', 't', "}')"]
```

#### F17 pre-tokenize 覆盖每一个字符：''.join(pretokens) == text（含 emoji/CJK/组合符/ZWJ）

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import pretokenize
import unicodedata
tests = ["hello世界😁", "e\u0301 café", "👨\u200d👩\u200d👧 family", "🇯🇵🇨🇳", "مرحبا بالعالم",
         "naïve\tTab\r\nCRLF", "\ufeffBOM", "\x00NUL", "日本語のテキスト。句読点、", "  \n\n  x"]
for t in tests:
    p = pretokenize(t)
    assert "".join(p) == t
    print(len(p), p)
```

输出:

```text
2 ['hello世界', '😁']
3 ['e', '́', ' café']
2 ['👨\u200d👩\u200d👧', ' family']
1 ['🇯🇵🇨🇳']
2 ['مرحبا', ' بالعالم']
6 ['naïve', '\t', 'Tab', '\r', '\n', 'CRLF']
2 ['\ufeff', 'BOM']
2 ['\x00', 'NUL']
4 ['日本語のテキスト', '。', '句読点', '、']
2 ['  \n\n ', ' x']
```

#### F18 不做 pre-tokenize 时 BPE 会学到跨词的 token（如 'o! '），做了之后不会

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import pretokenize, count_pairs, merge
def train(chunks, n):
    rules = {}
    for step in range(n):
        counts = {}
        for ids in chunks:
            counts = count_pairs(ids, counts) if counts else count_pairs(ids)
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p[0], p[1]))
        rules[best] = 256 + step
        chunks = [merge(ids, best, 256 + step) for ids in chunks]
    return rules
def show(rules):
    v = {i: bytes([i]) for i in range(256)}
    for (a, b), n in rules.items():
        v[n] = v[a] + v[b]
    return [v[n].decode() for n in rules.values()]
text = "Say hello! Why hello? Just hello."
print("no pretok :", show(train([list(text.encode())], 13)))
print("with pretok:", show(train([list(p.encode()) for p in pretokenize(text)], 13)))
```

输出:

```text
no pretok : ['lo', 'llo', 'he', 'hello', ' hello', 'y hello', 'y hello?', 'y hello? ', 'y hello? J', 'y hello? Ju', 'y hello? Jus', 'y hello? Just', 'y hello? Just hello']
with pretok: ['lo', 'llo', 'he', 'hello', ' hello', 'us', 'ust', 'hy', 'ay', 'Why', 'Say', 'Just', ' Just']
```

#### F19 CodeBot merge_rules.pkl：743 条 merge + 1 个 end token = vocab 1000

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
import pickle
rules = pickle.load(open("codebot/merge_rules.pkl", "rb"))
print(type(rules).__name__, len(rules), 256 + len(rules) + 1)
print(list(rules.values()) == list(range(256, 999)))
print(all(a < n and b < n for (a, b), n in rules.items()))
```

输出:

```text
dict 743 1000
True
True
```

#### F20 CodeBot 词表：最长 token、token 字节长度分布、全部是 ASCII

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import BPETokenizer
from collections import Counter
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
merged = {i: tok.id_to_bytes[i] for i in range(256, 999)}
print(tok.vocab_size, tok.end_token_id, repr(tok.id_to_bytes[999]))
print(sorted(Counter(len(b) for b in merged.values()).items()))
longest = sorted(merged.items(), key=lambda kv: -len(kv[1]))[:6]
print([(i, b.decode()) for i, b in longest])
print("all ascii:", all(max(b) < 128 for b in merged.values()))
```

输出:

```text
1000 999 b'<|endoftext|>'
[(2, 237), (3, 214), (4, 129), (5, 77), (6, 34), (7, 22), (8, 14), (9, 9), (10, 2), (12, 2), (13, 1), (14, 1), (16, 1)]
[(612, '\n               '), (777, ' BeautifulSoup'), (511, '\n            '), (359, '\n           '), (762, 'eautifulSoup'), (806, ' Executing')]
all ascii: True
```

#### F21 CodeBot 对 Unicode 的 round-trip（emoji/CJK/组合符/ZWJ/国旗/RTL）全部无损

浏览器(Pyodide)可跑: yes · 本机耗时 0.01s

```python
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
tests = ["hello世界😁", "e\u0301 café", "👨\u200d👩\u200d👧", "🇯🇵", "مرحبا",
         "日本語のテキスト。", "def f(x):\n    return x  # 注释", "\ufeff\x00\r\n"]
for t in tests:
    ids = tok.encode(t)
    assert tok.decode(ids) == t
    print(f"{len(t.encode()):3d} bytes -> {len(ids):3d} tokens  {t!r}")
```

输出:

```text
 15 bytes ->  13 tokens  'hello世界😁'
  9 bytes ->   8 tokens  'é café'
 18 bytes ->  18 tokens  '👨\u200d👩\u200d👧'
  8 bytes ->   8 tokens  '🇯🇵'
 10 bytes ->  10 tokens  'مرحبا'
 27 bytes ->  27 tokens  '日本語のテキスト。'
 32 bytes ->  17 tokens  'def f(x):\n    return x  # 注释'
  6 bytes ->   6 tokens  '\ufeff\x00\r\n'
```

#### F22 单独解码一个“半个字符”的 token 会得到 U+FFFD（流式输出时要注意）

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
ids = tok.encode("世")
print(ids, [tok.decode([i]) for i in ids], tok.decode(ids))
```

输出:

```text
[228, 184, 150] ['�', '�', '�'] 世
```

#### F23 special token 注入：用户文本里的 '<|endoftext|>' 也会变成 id 999

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
print(tok.encode("print('<|endoftext|>')"))
```

输出:

```text
[350, 338, 999, 343]
```

#### F24 缩进被压成很少的 token（256='  ', 257='    ', 259='   '）

浏览器(Pyodide)可跑: yes · 本机耗时 0.01s

```python
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
for s in ["x = 1", "\n    x = 1", "\n        x = 1"]:
    ids = tok.encode(s)
    print(repr(s), ids, [tok.decode([i]) for i in ids])
```

输出:

```text
'x = 1' [120, 262, 314] ['x', ' =', ' 1']
'\n    x = 1' [265, 454, 262, 314] ['\n   ', ' x', ' =', ' 1']
'\n        x = 1' [293, 454, 262, 314] ['\n       ', ' x', ' =', ' 1']
```

#### F25 tiny_codes.bin：uint16、2,677,398 个 token、999 的个数 == 文本中 <|endoftext|> 的个数

浏览器(Pyodide)可跑: yes (6.7MB fetch) · 本机耗时 0.18s

```python
import numpy as np, os
arr = np.fromfile("codebot/tiny_codes.bin", dtype=np.uint16)
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()
print(os.path.getsize("codebot/tiny_codes.bin"), arr.size, arr.dtype, arr.min(), arr.max())
print(int((arr == 999).sum()), text.count("<|endoftext|>"))
```

输出:

```text
5354796 2677398 uint16 9 999
21835 21835
```

#### F26 decode(tiny_codes.bin) == tiny_codes.txt（LF 归一化后）——全语料无损

浏览器(Pyodide)可跑: yes (6.7MB fetch) · 本机耗时 0.50s

```python
import numpy as np
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
arr = np.fromfile("codebot/tiny_codes.bin", dtype=np.uint16)
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()   # 文本模式：CRLF→LF
print(tok.decode(arr.tolist()) == text, len(text.encode()), len(text))
```

输出:

```text
True 6487033 6487033
```

#### F27 压缩率：前 10,000 字符 2.12 B/token；全语料 bytes/token

浏览器(Pyodide)可跑: yes · 本机耗时 1.08s

```python
import numpy as np
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()
s = text[:10000]
n = len(tok.encode(s))
print(len(s.encode()), n, round(len(s.encode()) / n, 4))
arr = np.fromfile("codebot/tiny_codes.bin", dtype=np.uint16)
print(len(text.encode()), arr.size, round(len(text.encode()) / arr.size, 4))
```

输出:

```text
10000 4709 2.1236
6487033 2677398 2.4229
```

#### F28 初始 loss ≈ ln(vocab)，以及 tiny_codes 的 unigram 熵（“只看词频”能做到的 loss）

浏览器(Pyodide)可跑: yes · 本机耗时 0.02s

```python
import numpy as np
arr = np.fromfile("codebot/tiny_codes.bin", dtype=np.uint16)
p = np.bincount(arr, minlength=1000) / arr.size
H = -(p[p > 0] * np.log(p[p > 0])).sum()
print(round(np.log(1000), 4), round(float(H), 4), int((p > 0).sum()))
```

输出:

```text
6.9078 5.7626 839
```

#### F29 uint16 的上限：vocab 超过 65,536 时 np.uint16 会出问题（numpy 1.x 静默回绕/警告）

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
import numpy as np, warnings
print(np.__version__, np.iinfo(np.uint16).max)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        a = np.array([65535, 65536, 70000], dtype=np.uint16)
        print(a, [str(x.category.__name__) for x in w])
    except OverflowError as e:
        print("OverflowError:", e)
```

输出:

```text
1.26.4 65535
[65535     0  4464] ['DeprecationWarning', 'DeprecationWarning']
```

#### F30 两种 encode 等价：按训练顺序逐条套用 merge（codebot） == 每次挑 rank 最小的 pair（storybot）

浏览器(Pyodide)可跑: yes · 本机耗时 17.73s

```python
from codebot.tokenizer import BPETokenizer as Naive
from storybot.tokenizer import BPETokenizer as Fast
naive, fast = Naive.load_from("codebot/merge_rules.pkl"), Fast.load_from("codebot/merge_rules.pkl")
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()[:200000]
a, b = naive.encode(text), fast.encode(text)
print(len(a), a == b)
for t in ["hello世界😁", "    return f'{x}'\n", "e\u0301"]:
    assert naive.encode(t) == fast.encode(t)
print("ok")
```

输出:

```text
92436 True
ok
```


### 3.2 第二批（对照 tiktoken、WebBot/StoryBot 词表、环境陷阱）

#### F31 DLFS6 的 pre-tokenize 正则与 tiktoken 'gpt2' 的 pat_str 字符串不同（后者用了 possessive 量词优化），但切分结果完全一致

浏览器(Pyodide)可跑: no (tiktoken) · 本机耗时 1.35s

```python
import tiktoken, regex
DLFS6_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
tk_pat = tiktoken.get_encoding("gpt2")._pat_str
print(tk_pat == DLFS6_PAT)
print(tk_pat)
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()[:300000] + " a  \n\n  b   \t\n x\u3000y\u00a0z "
a, b = regex.findall(tk_pat, text), regex.findall(DLFS6_PAT, text)
print(len(a), a == b)
```

输出:

```text
False
'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
86932 True
```

#### F32 压缩率对比（tiny_codes 前 10,000 字符）：CodeBot-1000 vs GPT-2 vs cl100k vs o200k

浏览器(Pyodide)可跑: no (tiktoken) · 本机耗时 4.40s

```python
import tiktoken
from codebot.tokenizer import BPETokenizer
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()[:10000]
nbytes = len(text.encode("utf-8"))
n = len(BPETokenizer.load_from("codebot/merge_rules.pkl").encode(text))
print(f"{'codebot':<11} vocab {1000:>7,}  tokens {n:>5,}  ratio {nbytes / n:.2f}")
for name in ["gpt2", "cl100k_base", "o200k_base"]:
    enc = tiktoken.get_encoding(name)
    n = len(enc.encode(text, allowed_special={"<|endoftext|>"}))
    print(f"{name:<11} vocab {enc.n_vocab:>7,}  tokens {n:>5,}  ratio {nbytes / n:.2f}")
```

输出:

```text
codebot     vocab   1,000  tokens 4,709  ratio 2.12
gpt2        vocab  50,257  tokens 3,232  ratio 3.09
cl100k_base vocab 100,277  tokens 2,516  ratio 3.97
o200k_base  vocab 200,019  tokens 2,512  ratio 3.98
```

#### F33 tiktoken 默认拒绝文本中的 special token；DLFS6 则总是把它当 special

浏览器(Pyodide)可跑: no (tiktoken) · 本机耗时 0.00s

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
try:
    enc.encode("<|endoftext|>")
except ValueError as e:
    print("ValueError:", str(e).splitlines()[0][:80])
print(enc.encode("<|endoftext|>", allowed_special="all"), enc.encode_ordinary("<|endoftext|>"))
```

输出:

```text
ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'.
[50256] [27, 91, 437, 1659, 5239, 91, 29]
```

#### F34 缩进的代价：GPT-2 vs cl100k vs CodeBot 对 8 空格缩进的切法

浏览器(Pyodide)可跑: no (tiktoken) · 本机耗时 0.01s

```python
import tiktoken
from codebot.tokenizer import BPETokenizer
cb = BPETokenizer.load_from("codebot/merge_rules.pkl")
s = "def f():\n        return 1\n"
for name in ["gpt2", "cl100k_base"]:
    enc = tiktoken.get_encoding(name)
    ids = enc.encode(s)
    print(name, len(ids), [enc.decode([i]) for i in ids])
ids = cb.encode(s)
print("codebot", len(ids), [cb.decode([i]) for i in ids])
```

输出:

```text
gpt2 14 ['def', ' f', '():', '\n', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' return', ' 1', '\n']
cl100k_base 8 ['def', ' f', '():\n', '       ', ' return', ' ', '1', '\n']
codebot 7 ['def', ' f', '():', '\n       ', ' return', ' 1', '\n']
```

#### F35 WebBot merge_rules.pkl（HF 下载，533,514 B）：49,743 条 merge，vocab 50,000

浏览器(Pyodide)可跑: yes (533KB fetch) · 本机耗时 0.30s

```python
import pickle
rules = pickle.load(open(r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer\dl\webbot\merge_rules.pkl", "rb"))
print(len(rules), 256 + len(rules) + 1, list(rules.values()) == list(range(256, 256 + len(rules))))
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), n in rules.items():
    vocab[n] = vocab[a] + vocab[b]
print([vocab[n].decode("utf-8", "replace") for n in range(256, 276)])
bad = [n for n in range(256, 256 + len(rules)) if not vocab[n].decode("utf-8", "ignore").encode() == vocab[n]]
print("单独不能合法 UTF-8 解码的 token 数:", len(bad), [vocab[n] for n in bad[:5]])
longest = sorted(vocab.items(), key=lambda kv: -len(kv[1]))[:5]
print([(i, b.decode("utf-8", "replace")) for i, b in longest])
```

输出:

```text
49743 50000 True
[' t', ' a', 'he', 'in', 're', ' the', 'on', 'er', ' s', ' w', 'at', ' o', 'en', ' c', 'it', 'is', 'an', 'or', ' b', 'es']
单独不能合法 UTF-8 解码的 token 数: 157 [b'\xe2\x80', b' \xe2\x80', b' \xc2', b'\xe3\x81', b' \xe2']
[(39095, 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'), (25821, 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'), (25835, '----------------------------------------------------------------'), (41197, '________________________________________________________________'), (31273, '————————————————')]
```

#### F36 WebBot(OWT, 50K) 与 GPT-2(WebText, 50,257) 的词表重合度与前 10 条 merge 对比

浏览器(Pyodide)可跑: no (tiktoken) · 本机耗时 0.42s

```python
import pickle, tiktoken
rules = pickle.load(open(r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer\dl\webbot\merge_rules.pkl", "rb"))
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), n in rules.items():
    vocab[n] = vocab[a] + vocab[b]
gpt2 = tiktoken.get_encoding("gpt2")._mergeable_ranks          # bytes -> rank
ours = set(vocab.values())
print(len(ours), len(gpt2), len(ours & set(gpt2)), round(len(ours & set(gpt2)) / len(ours), 3))
g_first = sorted((r, b) for b, r in gpt2.items() if r >= 256)[:10]
print("GPT-2 :", [b.decode() for r, b in g_first])
print("WebBot:", [vocab[n].decode() for n in range(256, 266)])
```

输出:

```text
49999 50256 44110 0.882
GPT-2 : [' t', ' a', 'he', 'in', 're', 'on', ' the', 'er', ' s', 'at']
WebBot: [' t', ' a', 'he', 'in', 're', ' the', 'on', 'er', ' s', ' w']
```

#### F37 StoryBot merge_rules.pkl：9,743 条 merge，vocab 10,000；前 10 个 token

浏览器(Pyodide)可跑: yes (104KB fetch) · 本机耗时 0.02s

```python
import pickle
rules = pickle.load(open(r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer\dl\storybot\merge_rules.pkl", "rb"))
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), n in rules.items():
    vocab[n] = vocab[a] + vocab[b]
print(len(rules), 256 + len(rules) + 1)
print([vocab[n].decode("utf-8", "replace") for n in range(256, 266)])
```

输出:

```text
9743 10000
[' t', 'he', ' a', ' s', ' w', 'nd', ' the', 'ed', ' b', ' to']
```

#### F38 领域错配：同一段文本，用哪个领域训练的 tokenizer 压缩得最好（bytes/token）

浏览器(Pyodide)可跑: partly (no tiktoken column) · 本机耗时 10.88s

```python
import tiktoken
from storybot.tokenizer import BPETokenizer
toks = {"codebot-1K": BPETokenizer.load_from("codebot/merge_rules.pkl"),
        "storybot-10K": BPETokenizer.load_from(r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer\dl\storybot\merge_rules.pkl"),
        "webbot-50K": BPETokenizer.load_from(r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer\dl\webbot\merge_rules.pkl")}
gpt2 = tiktoken.get_encoding("gpt2")
code = open("codebot/tiny_codes.txt", encoding="utf-8").read()[:100000]
story = open(r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer\dl\storybot\tiny_stories_valid_head10MB.txt", encoding="utf-8", errors="ignore").read()[:100000]
web = open(r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer\dl\webbot\owt_valid_head30MB.txt", encoding="utf-8", errors="ignore").read()[:100000]
print(f"{'':14s}{'code':>8s}{'story':>8s}{'web':>8s}")
for name, tok in list(toks.items()) + [("gpt2-50K", None)]:
    row = []
    for t in (code, story, web):
        n = len(tok.encode(t)) if tok else len(gpt2.encode(t, allowed_special="all"))
        row.append(len(t.encode()) / n)
    print(f"{name:14s}" + "".join(f"{r:8.2f}" for r in row))
```

输出:

```text
                  code   story     web
codebot-1K        2.16    2.03    2.02
storybot-10K      1.98    4.09    3.35
webbot-50K        2.79    4.05    4.68
gpt2-50K          2.84    4.05    4.59
```

#### F39 CodeBot 词表里有“训练后从未出现”的 token（只作为中间产物，如 'eautifulSoup'）

浏览器(Pyodide)可跑: yes (6.7MB fetch) · 本机耗时 0.03s

```python
import numpy as np
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
arr = np.fromfile("codebot/tiny_codes.bin", dtype=np.uint16)
cnt = np.bincount(arr, minlength=1000)
dead = [i for i in range(256, 999) if cnt[i] == 0]
print(len(dead), [tok.id_to_bytes[i].decode() for i in dead[:12]])
rare = sorted(range(256, 999), key=lambda i: cnt[i])[len(dead):len(dead) + 5]
print([(tok.id_to_bytes[i].decode(), int(cnt[i])) for i in rare])
```

输出:

```text
3 ['eauti', 'eautifulS', 'bonac']
[('ntence', 1), ('eau', 1), ('bon', 7), ('Hel', 13), (' !', 13)]
```

#### F40 Windows + git autocrlf 陷阱：工作区里的 tiny_codes.txt 是 CRLF；文本模式读取会自动转 LF，二进制读取不会

浏览器(Pyodide)可跑: yes · 本机耗时 3.28s

```python
raw = open("codebot/tiny_codes.txt", "rb").read()
txt = open("codebot/tiny_codes.txt", encoding="utf-8").read()                 # universal newlines
txt_nl = open("codebot/tiny_codes.txt", encoding="utf-8", newline="").read()  # 不转换
print(len(raw), raw.count(b"\r\n"), len(txt), len(txt_nl))
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
a, b = txt[:5000], txt_nl[:5000 + txt[:5000].count("\n")]
print(len(tok.encode(a)), len(tok.encode(b)), b.replace("\r\n", "\n") == a)
```

输出:

```text
6743534 256501 6487033 6743534
2291 2411 True
```

#### F41 compression ratio 与 bits-per-byte 换算：同一 loss 在不同 tokenizer 下不可直接比较

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
import math
loss_nats_per_token = 1.0            # 假设某模型在 tiny_codes 上的 loss
bytes_per_token = 2.4229             # CodeBot 全语料压缩率 (F27)
bpb = loss_nats_per_token / math.log(2) / bytes_per_token
print(round(bpb, 4), "bits/byte")
```

输出:

```text
0.5954 bits/byte
```


### 3.3 第三批（仓库历史、CS336 tie-break、OWT 稳定性、统计、模型初始 loss、压缩率曲线）

#### F42 ch03/03_alpaca.py 注释里的 id 列表与实际输出有 1 处不同（930 vs 932），注释是旧 merge_rules 的产物

浏览器(Pyodide)可跑: yes · 本机耗时 0.29s

```python
import json
from codebot.tokenizer import BPETokenizer
tok = BPETokenizer.load_from("codebot/merge_rules.pkl")
item = json.load(open("codebot/tiny_codes_sft.json", encoding="utf-8"))[0]
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
ids = tok.encode(text)
doc = [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 930, 657, 63, 999]
diff = [(i, doc[i], ids[i]) for i in range(len(ids)) if ids[i] != doc[i]]
print(len(ids) == len(doc), diff, repr(tok.decode([930])), repr(tok.decode([932])))
```

输出:

```text
True [(30, 930, 932)] 'alize' ' you'
```

#### F44 tie-break：DLFS6 比较 id（创建顺序），CS336 要求比较 bytes 的字典序——同一语料学出不同 merge

浏览器(Pyodide)可跑: yes · 本机耗时 0.00s

```python
from codebot.tokenizer import count_pairs, merge
def train(docs, n_merges, rule):
    seqs = [list(d.encode()) for d in docs]
    vocab = {i: bytes([i]) for i in range(256)}
    rules = {}
    for step in range(n_merges):
        counts = {}
        for s in seqs:
            counts = count_pairs(s, counts) if counts else count_pairs(s)
        if rule == "dlfs6":
            best = max(counts, key=lambda p: (counts[p], p[0], p[1]))
        else:  # cs336: lexicographically greater pair of *bytes*
            best = max(counts, key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]]))
        new = 256 + step
        rules[best] = new
        vocab[new] = vocab[best[0]] + vocab[best[1]]
        seqs = [merge(s, best, new) for s in seqs]
    return [vocab[n].decode() for n in rules.values()]
docs = ["zzq"] * 3 + ["zz"] * 3 + ["aaq"] * 3 + ["aa"] * 2      # 文档间不跨越（相当于按 <|endoftext|> 切开）
print("DLFS6 :", train(docs, 3, "dlfs6"))
print("CS336 :", train(docs, 3, "cs336"))
```

输出:

```text
DLFS6 : ['zz', 'aa', 'aaq']
CS336 : ['zz', 'aa', 'zzq']
```

#### F45 OWT 上训练的前 15 条 merge 与官方 50K 规则完全一致；30MB 样本学到的 1,743 个 token 有 1,673 个落在官方前 1,743 个里

浏览器(Pyodide)可跑: yes (needs the 30MB sample + training ≈2.5 min) · 本机耗时 0.92s

```python
import pickle
SC = r"C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\ch01-tokenizer"
def vocab_of(rules):
    v = {i: bytes([i]) for i in range(256)}
    for (a, b), n in rules.items():
        v[n] = v[a] + v[b]
    return v
small = pickle.load(open(SC + r"\owt30MB_merge_rules_v2000.pkl", "rb"))      # ch07/01 缩小版的产物
full = pickle.load(open(SC + r"\dl\webbot\merge_rules.pkl", "rb"))         # 官方 webbot/merge_rules.pkl
vs, vf = vocab_of(small), vocab_of(full)
prefix = next(i for i, (p, q) in enumerate(zip(small, full)) if p != q)
learned = [vs[n] for n in small.values()]
print(prefix, sum(b in set(vf[n] for n in range(256, 256 + len(small))) for b in learned), len(learned))
```

输出:

```text
15 1673 1743
```

#### F46 snake_case 被 GPT-2 正则在 '_' 处切开（BPE 永远学不到 '_time'）；cl100k 的正则允许 '_time'

浏览器(Pyodide)可跑: partly (cl100k 行需 tiktoken) · 本机耗时 0.66s

```python
import tiktoken, regex
from codebot.tokenizer import pretokenize
s = "total_time = 0; x = 1234567"
print("gpt2  :", pretokenize(s))
print("cl100k:", regex.findall(tiktoken.get_encoding("cl100k_base")._pat_str, s))
```

输出:

```text
gpt2  : ['total', '_', 'time', ' =', ' 0', ';', ' x', ' =', ' 1234567']
cl100k: ['total', '_time', ' =', ' ', '0', ';', ' x', ' =', ' ', '123', '456', '7']
```

#### F47 tiny_codes 去重统计：182 万个 pre-token 出现，只有 30,535 种（ch04 按“种类×次数”训练的依据）

浏览器(Pyodide)可跑: yes (6.7MB fetch) · 本机耗时 5.59s

```python
from collections import Counter
from codebot.tokenizer import pretokenize
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()
c = Counter(p for doc in text.split("<|endoftext|>") for p in pretokenize(doc))
print(sum(c.values()), len(c), c.most_common(6))
print(sum(ord(ch) > 127 for ch in text))      # 非 ASCII 字符数
```

输出:

```text
1826876 30535 [('\n', 160460), ('_', 74293), ('.', 69082), ('(', 63425), (' =', 53183), (',', 48355)]
0
```

#### F48 tokenizer 决定了初始 loss：CodeBot GPT（ch03 的超参）随机初始化时 loss ≈ ln(1000)

浏览器(Pyodide)可跑: no (torch) · 本机耗时 11.56s

```python
import math, numpy as np, torch, torch.nn.functional as F
from codebot.model import GPT
torch.manual_seed(0)
m = GPT(vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6, ff_dim=1536, dropout_rate=0.1).eval()
arr = np.fromfile("codebot/tiny_codes.bin", dtype=np.uint16)
ix = np.random.default_rng(0).integers(0, len(arr) - 257, 8)
x = torch.tensor(np.stack([arr[i:i + 256] for i in ix]).astype(np.int64))
y = torch.tensor(np.stack([arr[i + 1:i + 257] for i in ix]).astype(np.int64))
with torch.no_grad():
    loss = F.cross_entropy(m(x).view(-1, 1000), y.view(-1))
print(sum(p.numel() for p in m.parameters()), round(loss.item(), 4), round(math.log(1000), 4))
```

输出:

```text
11121640 7.0242 6.9078
```

#### F50 压缩率随 vocab 增长、边际递减：只用前 k 条 merge 就是一个 vocab=257+k 的合法 tokenizer

浏览器(Pyodide)可跑: yes (≈15–30 s) · 本机耗时 15.72s

```python
import pickle
from collections import Counter
from storybot.tokenizer import BPETokenizer
from codebot.tokenizer import pretokenize
rules = list(pickle.load(open("codebot/merge_rules.pkl", "rb")).items())
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()
docs = text.split("<|endoftext|>")
c = Counter(p for d in docs for p in pretokenize(d))              # 按 pre-token 去重后再编码
nbytes = len(text.encode())
for k in [0, 25, 50, 100, 200, 400, 600, 743]:
    tok = BPETokenizer(dict(rules[:k]))
    ntok = sum(len(tok._encode_text(p)) * n for p, n in c.items()) + (len(docs) - 1)
    print(f"vocab {256 + k + 1:5d}: {ntok:>9,} tokens  {nbytes / ntok:.3f} B/token")
```

输出:

```text
vocab   257: 6,225,013 tokens  1.042 B/token
vocab   282: 5,043,737 tokens  1.286 B/token
vocab   307: 4,535,838 tokens  1.430 B/token
vocab   357: 4,003,581 tokens  1.620 B/token
vocab   457: 3,483,198 tokens  1.862 B/token
vocab   657: 3,034,942 tokens  2.137 B/token
vocab   857: 2,794,703 tokens  2.321 B/token
vocab  1000: 2,677,398 tokens  2.423 B/token
```


### 3.4 脚本级验证（多进程 / 长时间 / JS）

#### F43 ch04/ch07 的快速训练器 == ch01 的朴素训练器；CRLF 文件会改变 merge

浏览器(Pyodide)可跑: ❌（`storybot.train_bpe` 用 `multiprocessing.Pool`；算法本身可移植，见 J2）

实际运行的脚本：`SC\trainer_equivalence.py` 与 `SC\fast_train_full.py`，核心几行：

```python
from codebot.tokenizer import train_bpe as train_naive       # ch01：全量重数 pair
from storybot.tokenizer import train_bpe as train_fast       # ch04/ch07：去重 + 倒排缓存 + 多进程
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()                 # 文本模式：CRLF→LF
sample = text[:300_000]
open(lf_path,   "w", encoding="utf-8", newline="\n").write(sample)
open(crlf_path, "w", encoding="utf-8", newline="\r\n").write(sample)
naive = train_naive(sample, 600)
fast  = train_fast(lf_path, 600, num_processes=4, num_chunks=8)
print(list(naive.items()) == list(fast.items()))
crlf  = train_fast(crlf_path, 600, num_processes=4, num_chunks=8)
...
rules = train_fast(full_lf_path, 1000, num_processes=8, num_chunks=64)        # 完整语料
print(list(rules.items()) == list(pickle.load(open("codebot/merge_rules.pkl","rb")).items()))
rules_crlf = train_fast("codebot/tiny_codes.txt", 1000, 8, ...)                # Windows 工作区里的 CRLF 原文件
```

输出（与其它后台任务并行，计时偏慢）：

```text
(a) vocab 600: naive 113.4s, fast 3.7s, identical merge dicts (incl. order): True
(b) CRLF file: identical to LF? False; prefix agreement 1; merges involving b'\r': [((13, 10), 257)]
(c) first 200,000 chars, vocab 1000: 166.2s; prefix agreement with shipped 1; same pairs anywhere 193/743
(c) first 1,000,000 chars, vocab 1000: 748.6s; prefix agreement with shipped 1; same pairs anywhere 213/743
storybot.train_bpe(full LF tiny_codes, 1000): 19.2s, 743 merges, identical to shipped merge_rules.pkl (incl. order): True
same trainer on the CRLF checkout: 19.7s, identical: False, first 3 merges: [((32, 32), 256), ((13, 10), 257), ((256, 256), 258)]
```

补充（`SC\subset_overlap.py`，按 token 的**字节串**而不是 id 比较，避免 id 错位造成的低估）：

```text
first 200,000 chars: learned 743 tokens; same byte-strings as shipped: 404/743; first 5 learned ['  ', 'in', 'or', 'er', ' ='] vs shipped ['  ', '    ', 'in', '   ', 're']
first 1,000,000 chars: learned 743 tokens; same byte-strings as shipped: 468/743; first 5 learned ['  ', 'in', 'or', 'te', 'im'] vs shipped ['  ', '    ', 'in', '   ', 're']
```

解读：快速训练器在完整（LF）语料上 19 s 就复现了官方 `merge_rules.pkl`；而直接喂 Windows 工作区里的 CRLF 文件（二进制读取，没有换行归一化），第 2 条 merge 就变成 `\r\n`。(c) 说明：只用语料前缀训练 1000 vocab 时，与官方规则只有第 1 条相同、token 只重合 54%（200k）/ 63%（1M）——原因是语料没打乱，前 1M 字符里深缩进少、`import` 多（官方第 2 条 merge `'    '` 来自 8 空格缩进）。“缩小版”只能用来体验流程，不能用来复现官方产物。

#### F49 用书中原脚本（补丁副本）完整重训 == 仓库里的 merge_rules.pkl

浏览器(Pyodide)可跑: ❌（单核约 1 小时）

```python
# 第 1 步 SC\07_tiny_codes_patched.py：与 ch01/07_tiny_codes.py 相同，只改 os.chdir 和输出路径（外加计时）
text = open("codebot/tiny_codes.txt").read()
merge_rules = train_bpe(text, 1000)                  # codebot.tokenizer.train_bpe
with open(os.path.join(SC, 'merge_rules_retrained.pkl'), "wb") as f:
    pickle.dump(merge_rules, f)

# 第 2 步：比较
import pickle, hashlib
ra = open(SC + r"\merge_rules_retrained.pkl", "rb").read()
rb = open("codebot/merge_rules.pkl", "rb").read()
a, b = pickle.loads(ra), pickle.loads(rb)
print("retrained == shipped (dict items incl. order):", list(a.items()) == list(b.items()))
print("pickle bytes identical:", ra == rb, "| sizes:", len(ra), len(rb))
print("sha256:", hashlib.sha256(ra).hexdigest()[:16], hashlib.sha256(rb).hexdigest()[:16])
```

输出：

```text
train_bpe wall: 3837.1s, merges: 743   (07_tiny_codes_patched.py, 与其它 agent 的任务并行)
retrained == shipped (dict items incl. order): True
pickle bytes identical: True | sizes: 7421 7421
sha256: 7b1667406b2128a4 7b1667406b2128a4
```

#### J1 纯 JS 的 CodeBot encode 与 Python 逐 id 一致（需要显式空白字符类）

浏览器可跑: ✅（纯 JS，无依赖）。脚本：`SC\js\bpe_parity.mjs`、`SC\js\ws_fix.mjs`；参考数据由 Python 生成（`parity_cases.json`：29 个刁钻 Unicode 用例 + tiny_codes 前 200,000 字符的 Python 编码结果）。

```js
// 直接把 Python 正则搬到 JS（u 标志支持 \p{L}/\p{N} 与 lookahead）
const PAT_NAIVE = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
// 与 Python `regex` 的 \s 完全一致的写法：JS 的 \s 多了 U+FEFF、少了 U+0085
const WS = String.raw`\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000`;
const PAT_FIXED = new RegExp(String.raw`'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^${WS}\p{L}\p{N}]+|[${WS}]+(?![^${WS}])|[${WS}]+`, "gu");
// encode：TextEncoder 取 UTF-8 字节 → 对每个 pre-token 反复合并 rank 最小的 pair（与 storybot._encode_text 相同）
```

输出：

```text
naive \s: 27/29 pretokenization cases identical to Python; sample pretokens 57801
fixed WS: 29/29 pretokenization cases identical to Python; sample pretokens 57801
unicode cases: 27/29 match            ← 用 naive \s 时，'a \x85b' 和 'a \ufeffb' 的 pre-token 边界不同（最终 id 碰巧仍相同）
tiny_codes[:200000]: 92436 ids, equal to Python: true, 177 ms
JS "😁".length = 2, [..."😁"].length = 1, Python len = 1
```

另测：Python `regex` 的 `\s` 恰好 25 个码位（= Unicode White_Space），JS `\s` 也是 25 个，但集合不同（Python 有 U+0085、JS 有 U+FEFF）；Python 的 `str.isspace()` 还额外包含 U+001C–U+001F（共 29 个）。

#### J2 纯 JS 训练器在完整语料上逐条复现 merge_rules.pkl

浏览器可跑: ✅（建议放进 Web Worker）。脚本：`SC\js\bpe_train.mjs`（约 60 行：按 pre-token 去重计数 → 每步加权数 pair → `(count, key)` 最大者 → 只改含该 pair 的词）。

```js
const text = readFileSync(`${REPO}/codebot/tiny_codes.txt`, "utf8").replace(/\r\n/g, "\n"); // 模拟 Python 文本模式
const r = trainBPE(text, 1000);
```

输出（node 24，与 Python 后台任务并行）：

```text
JS trainBPE(full tiny_codes, 1000): 743 merges, 30535 unique pretokens, pretokenize 1626 ms, total 28.7 s, identical to shipped merge_rules.pkl: true
JS trainBPE(first 200k chars, 1000): 4.70 s, identical-prefix with shipped: 1
# 机器负载较低时复测：
JS trainBPE(full tiny_codes, 1000): 743 merges, 30535 unique pretokens, pretokenize 522 ms, total 22.1 s, identical to shipped merge_rules.pkl: true
JS trainBPE(first 200k chars, 1000): 2.27 s, identical-prefix with shipped: 1
```

tie-break 的实现：pair 编码成 `key = a * 2**20 + b`，并列时取 key 最大者——与 Python 的 `max(..., key=(count, a, b))` 等价。网页里若想更快，可以加 ch04 的倒排缓存（只更新受影响的词），28.7 s 可以降到几秒。


---

## 4. GPU / 数据需求

### 4.1 结论先行

- **ch01 与 ch07 全部是 CPU 任务，不需要 GPU。** ch01 的瓶颈是纯 Python 的朴素 BPE（单核）；ch07 的瓶颈是 11.9 GB 文本的 pre-tokenize / encode（多进程）以及 5 万步 merge 循环（单核）。
- 云上要跑 ch07，应租**多核 CPU + 大内存**的机器（Modal 的 CPU container、RunPod/Lambda 的 CPU 实例），而不是 GPU 实例；更省事的是直接下载作者在 HF 上放好的产物。
- 它们的产物决定了下游 GPU 章节的 D（token 数）：CodeBot 2.68M、StoryBot 541M、WebBot 2.649B。

### 4.2 数据文件（HF：`https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/…`，`curl -sIL` 实测大小）

| 文件 | 字节 | 备注 |
|---|---:|---|
| `codebot/tiny_codes.txt` | 6,743,534（CRLF 工作区）/ 6,487,033（LF） | 仓库自带；纯 ASCII |
| `codebot/merge_rules.pkl` | 7,421 | 仓库自带（HF 上同大小） |
| `codebot/tiny_codes.bin` | 5,354,796 | 仓库自带；= 2,677,398 × uint16 |
| `codebot/tiny_codes_sft.json` | 129,717 | 1,092 条 `{instruction, response}`；编码后长度 23–137（中位 46），全部 ≤ ch03 的 `context_len=256` |
| `codebot/model_pretrain.pt` | 44,516,353 | （ch03 用，未下载） |
| `storybot/tiny_stories_train.txt` | 2,227,753,162 | 未下载 |
| `storybot/tiny_stories_valid.txt` | 22,502,601 | 只下载了前 10 MiB 到 `SC\dl\storybot\` |
| `storybot/merge_rules.pkl` | 103,877 | 已下载到 `SC\dl\storybot\`；9,743 merge |
| `storybot/tiny_stories_train.bin` | 1,082,458,446 | = 541,229,223 tok（2,227,753,162 B / 541M = 4.12 B/tok） |
| `storybot/tiny_stories_valid.bin` | 10,931,746 | = 5,465,873 tok |
| `webbot/owt_train.txt` | 11,920,511,059 | **未下载** |
| `webbot/owt_valid.txt` | 289,998,753 | 只 Range 下载了前 30 MiB 到 `SC\dl\webbot\` |
| `webbot/merge_rules.pkl` | 533,514 | 已下载到 `SC\dl\webbot\`；49,743 merge |
| `webbot/owt_train.bin` | 5,298,009,600 | = 2,649,004,800 tok（4.500 B/tok） |
| `webbot/owt_valid.bin` | 128,992,132 | = 64,496,066 tok（4.496 B/tok）；前 2,000 个 id 已与本机重编码比对一致 |

（我下载的文件都放在 scratch，**没有**放进仓库的 bot 目录，避免与其它 agent 并发写同一路径。写档案时其它 agent 已把 `webbot/merge_rules.pkl`、`storybot/merge_rules.pkl` 等放进了仓库的 bot 目录；两份 pkl 与我在 scratch 下载的副本 sha256 相同：webbot `f83bb48e…`、storybot `8fb285d9…`。）

### 4.3 实测时间与外推

| 任务 | 规模 | 本机实测（i7-11700F） | 外推 / 备注 |
|---|---|---|---|
| `ch01/01–06` | 玩具文本 | 0.12–0.26 s | — |
| `ch01/07` 朴素训练（原脚本） | 6.5 MB，743 merge | **3,837 s（64 min）**，单核；与其它 agent 的重负载进程并行，偏慢（F49） | 用 `storybot.train_bpe` 只要 **19.2 s**（8 进程），结果完全相同 |
| `ch01/08` | 10k 字符 | 1.74 s | — |
| `ch01/09` 朴素 encode（原脚本） | 6.5 MB → 2.68M tok | **628.7 s** | rank 优先 encode 53.9 s；再加 pre-token 缓存 9.9 s（结果都逐字节一致） |
| `ch07/01` 缩小版 | 30 MiB OWT，vocab 2,000 | 153.6 s（pre-tokenize 5 s + 1,743 merge 144 s） | 见下 |
| `ch07/02` 缩小版 | 30 MiB OWT → 7.02M tok | ≈61.7 s（0.51 MB/s，8 进程） | 全量 train 11.92 GB ≈ **6.5 h**（8 进程，线性外推）；valid 290 MB ≈ 9.5 min |

`ch07/01` 全量（vocab 50,000 on 11.92 GB）**未实测**，估算依据：
- 单进程 pre-tokenize 实测 1.5–1.8 MB/s → 8 进程约 12–14 MB/s → **约 15–20 分钟**。
- 唯一 pre-token 数随数据量按 Heaps 定律增长：3.1 MB → 43,748、9.4 MB → 83,980、31.5 MB → 168,942（指数 β≈0.58）→ 外推到 11.92 GB 约 **5×10⁶** 种。`ids_counts` + `pair_to_ids` 倒排缓存的内存估计在 **10–30 GB** 量级（另加 64 个 chunk 的 Counter 回传主进程时的峰值）。
- merge 循环是单核的：每步 `max()` 扫描全部 pair（后期可达百万级）+ 更新受影响的 pre-token；按 30 MB 实测的 12 it/s 与 pair 数增长粗估，**全量需要数小时（3–10 h 量级）**。这一数字只是估算。
- 结论：要么下载 `webbot/merge_rules.pkl`（533 KB），要么在 ≥16 核、≥64 GB 内存的 CPU 机器上跑一晚。

### 4.4 Modal / RunPod / Lambda 方案

| 目的 | 建议 | 估计 |
|---|---|---|
| 复现 ch01 全部产物 | 本地 CPU 就够；`storybot.train_bpe` 19 s + 缓存版 encode 10 s | 免费 |
| 复现 ch07 encode（得到 `owt_*.bin`） | Modal CPU container（如 32 vCPU / 64 GB）挂 Volume；`num_processes=32` | 按 8 进程 0.51 MB/s 线性外推，32 核约 1.6 h；或直接 `wget` 5.3 GB 的 `owt_train.bin` |
| 复现 ch07 训练（50K merge） | 同上的 CPU 机器；merge 循环是单核瓶颈，核数多只加速 pre-tokenize | 数小时；不推荐，除非要做 tie-break/数据实验 |
| 下游 GPU（供其它档案参考） | CodeBot（ch03）：11,121,640 参数 × 163,840,000 token（= 20,000 iter × 32 × 256，约 61 个 epoch）→ 6ND ≈ **1.09×10¹⁶ FLOPs**；RTX 3060 Ti 8 GB 即可 | 分钟到几十分钟级 |
| | WebBot（ch09）：152,316,672 参数 × 13,107,200,000 token（100k iter × 128 × 1024，≈4.95 epoch of `owt_train.bin`）→ 6ND ≈ **1.20×10¹⁹ FLOPs** | 1×H100（bf16，按 40% MFU≈400 TFLOP/s）≈ 8.4 h；1×A100（≈125 TFLOP/s）≈ 27 h；8 卡 DDP 近似线性缩短 |

### 4.5 CPU 上的缩小方法（给学习者）

- **训练**：用 `storybot.train_bpe`（ch04 的成果）代替 `codebot.train_bpe`；必须喂 **LF** 文件（见 5.3）。完整 tiny_codes / vocab 1000 只需约 20 s，且与官方 `merge_rules.pkl` 完全一致。
- 如果坚持用 ch01 的朴素版：`text[:200_000]` + vocab 1000 要 166 s，`text[:1_000_000]` 要 749 s（与其它任务并行时测得）。注意前缀样本学出的 token 与官方只重合 54%（200k）/ 63%（1M），因为语料不是打乱的：前 1M 字符里 8 空格缩进行每千行 78 行、`import` 行 147 行，而后面分别约 130 行和 41 行。
- **encode**：给 `_encode_text` 加 pre-token 级 `lru_cache`（全语料只有 30,535 种 pre-token）→ 10 s。
- **ch07**：Range 下载 `owt_valid.txt` 的前 N MB（`curl -r 0-31457279 …`），vocab 降到 1–2K；或者只做 encode（用官方 merge_rules）。

---

## 5. 易错点 / 坑 / 潜在 bug（全部经运行验证）

### 5.1 算法语义

1. **tie-break 有三种，结果都不同**
   - `ch01/03–06`：`max(counts, key=counts.get)` → 并列时取文本中**最先出现**的 pair（依赖 dict 插入顺序）。
   - `codebot/storybot`：`(count, id0, id1)` 最大 → 并列时取 **id 最大**的 pair。
   - CS336：并列时取 **bytes 字典序最大**的 pair。
   - F10：同一句话第 1 步就分叉（`'is'` vs `'s '`）；F44：构造的语料上 DLFS6 选 `'aaq'`、CS336 选 `'zzq'`。id 序 ≠ bytes 序，因为 merge 产生的 id 按“创建时间”排序（`'zz'`=256 < `'aa'`=257，但 `b'zz' > b'aa'`）。ch01/06 与 codebot 在 06 的示例上学出完全不同的 13 条 merge。
2. **vocab_size 的定义在 05 改变了**：03/04 是 `256 + merges`；05 起是 `256 + merges + 1`。同为 `vocab_size=260`，id 259 在 04 是 `'He'`，在 05 是 `<|endoftext|>`（F12）。
3. **pair 计数会数重叠**：`[7,7,7]` 里 `(7,7)` 计 2 次但只能合并 1 次（F08）。长串空格/`=====` 的频次被高估——这是 CS336 和多数实现的共同约定，但讲课时值得点破。
4. **merge 规则的顺序就是语义**：`merge_rules` 是有序 dict，pickle 保留顺序。打乱顺序后构造 `BPETokenizer` 直接 `KeyError: 294`（`id_to_bytes` 递推需要先有子 token）；按逆序套用 merge，`'def f(x):\n    return x'` 编成 14 个 token 而不是 8 个（decode 仍可逆，但与模型训练时见到的 id 分布不一致）。存成 JSON 时要存成**列表**而不是以字符串为 key 的对象。
5. **两种 encode 等价**：“按训练顺序逐条套用全部 merge”（codebot，O(#merges·len)）与“反复合并 rank 最小的现存 pair”（storybot）结果完全相同（F30 + 全语料逐字节验证）。理由：合并只会产生含新 id 的新 pair，而新 id 参与的规则 rank 一定更大，所以处理顺序一致。
6. **special token 只有一个、且没有 PAD**：ch03 的 SFT 用 id 0（字节 `\x00`）做 padding 并把 label 设为 -100；chat 模板 `### Instruction:` 是普通文本（`#` 都没被合并：`[35, 35, 35, 962, …]`）。
7. **special token 注入**：codebot/storybot 的 `encode` 会把用户文本里的 `<|endoftext|>` 无条件变成 end id（F23：`print('<|endoftext|>')` → `[350, 338, 999, 343]`）。ch03 的 chat 里用户一输入它，模型就会收到“文档结束”。tiktoken 默认直接报错（F33）。
8. **decode 用 `errors="replace"` 会掩盖问题**：逐 token 流式输出时，一个汉字的 3 个字节 token 各自解码成 `'\ufffd'`（F22）。WebBot 词表里有 157 个“半个字符”的 token（如 `b'\xe2\x80'`），`08_eval.py` 那样逐个 strict `decode("utf-8")` 的循环放到 WebBot 词表上会抛 `UnicodeDecodeError`（本机验证：157 个全部失败）。流式输出应先累积字节再增量解码（JS 里用 `TextDecoder` 的 `{stream: true}`）。

### 5.2 pre-tokenize 正则（GPT-2 版）在代码语料上的怪现象

9. **缩约规则误伤代码**：`print(f'done')` → `'print','(','f',"'d",'one',"')"`；`f'total: {t}'` 里出现 `"'t"`（F16）。只认小写：`DON'T` → `'DON',"'",'T'`（F15）。
10. **snake_case 永远在 `_` 处断开**：`total_time` → `'total','_','time'`，BPE 学不到 `_time`；`'_'` 是 tiny_codes 里第二常见的 pre-token（74,293 次）。cl100k 的正则允许一个非字母前缀，所以有 `'_time'`（F46）。
11. **数字整段保留**：`' 1234567'` 是一个 pre-token；cl100k 按 1–3 位切（`'123','456','7'`）。
12. **组合符号被拆出去**：NFD 的 `'e\u0301'` → `'e'`、`'\u0301'` 两个 pre-token；NFC/NFD 的 `é` 得到不同 id（F06/F17）。DLFS6 不做 Unicode 正规化。
13. **CJK 会整段成为一个 pre-token**（`'日本語のテキスト'`），但 CodeBot 的 merge 全是 ASCII，所以每个 CJK 字符 3 个 token、emoji 4 个（F21）。
14. **必须用第三方 `regex` 模块**：标准库 `re` 编译这条正则直接报 `re.error: bad escape \p at position 23`（本机验证）。05 用 `re` 没问题，是因为它只 `re.escape`/`re.split` special token；06 起 `import regex as re` 是有意为之。Pyodide 自带 `regex`。
15. **tiktoken 的 `gpt2` pat_str 与书中字符串不同**（possessive 量词 + `\s++$` + 末尾 `\s`），但在 30 万字符代码 + 200 万字符网页文本上切分结果完全相同（F31）。比较正则要比较行为，不要比较字符串。

### 5.3 Windows / 环境

16. **git `core.autocrlf=true` 让 `tiny_codes.txt` 变成 CRLF**（本机 `git ls-files --eol`：`i/lf w/crlf`；25.65 万行）。
    - `open(path).read()`（文本模式）会自动转回 LF → ch01/07–09 不受影响，重新生成的 `tiny_codes.bin` 与仓库逐字节相同。
    - 但 ch04/ch07 的 `storybot` 代码以**二进制**读文件（`find_chunk_boundaries`、`pretoken_chunk`、`encode_file`）→ 喂 CRLF 文件会学出 `\r\n` 作为第 2 条 merge（F43），encode 时每行多 1 个 `\r` token（F40：5,000 字符多 120 个 token）。
    - 修复：克隆前 `git config core.autocrlf false`，或加 `.gitattributes`（`*.txt -text`），或读入后 `replace(b"\r\n", b"\n")`。
17. **不设 `PYTHONUTF8=1` 时，Windows 的默认编码是 cp936**：`open(owt_sample).read()` 抛 `UnicodeDecodeError: 'gbk' … illegal multibyte sequence`；tiny_codes 是纯 ASCII 所以碰巧没事。ch04 里被注释掉的 `open("storybot/tiny_stories_train.txt").read()` 在中文 Windows 上会失败。
18. **multiprocessing 必须有 `if __name__ == '__main__':`**（Windows 是 spawn）。ch07 的两个脚本都有；自己写缩小版时别丢。
19. **`encode_file` 的 `cache_dir="bpe_cache"` 是相对路径**：脚本先 `chdir` 到仓库根，所以会在仓库里建 `bpe_cache/` 并在 `finally` 里 `rmtree`。同一 cwd 并发跑两个 `encode_file` 会互相删缓存；缓存文件扩展名是 `.npy` 但内容是 `tofile` 的裸 uint16（`np.load` 读不了，只能 `np.fromfile`）。
20. **uint16 上限 65,535**：vocab 超过 65,536 时，numpy 1.26（仓库环境）把 65536 静默回绕成 0、70000 变 4464（只有 DeprecationWarning）；numpy 2（Pyodide）直接 `OverflowError`（F29）。

### 5.4 仓库里的小问题

21. **`ch03/03_alpaca.py` 的注释过期**：第 30 个 id 注释写 930（`'alize'`），实际是 932（`' you'`）。该注释来自最初的 commit（2025-12-18），而 `merge_rules.pkl` 是 2026-01-07 才提交的（F42）。
22. `storybot/tokenizer.py` 里 `process_single_chunk` 与 `pretoken_chunk` 完全重复，前者是死代码。
23. `08_eval.py` 只在前 10,000 字符（前 46 个程序，含 45 个 end token）上测压缩率：2.12，而全语料是 2.42——样本偏差（F27）。tiktoken 对比被包在字符串字面量里，脚本永远不会执行它。
24. CodeBot 词表里有 3 个 token 在 `tiny_codes.bin` 中出现 0 次（`'eauti'`、`'eautifulS'`、`'bonac'`），它们只是合并 `' BeautifulSoup'`/`'Fibonacci'` 的中间产物；`'ntence'`、`'eau'` 只出现 1 次（F39）。对应的 embedding 几乎得不到梯度——这是“glitch token”（GPT-2 的 SolidGoldMagikarp）的迷你版。WebBot 词表最长的 token 是 mojibake `'ÃÂÃÂ…'`（F35），同类现象。
25. 语料没有打乱：开头是 `input()` 小程序，深缩进代码集中在后面，前缀样本训练出的 tokenizer 偏差大（4.5 节）。
26. 语料里 11.1% 的程序（2,426/21,836）连 `ast.parse` 都过不了（典型是 `input('… type 'done' to …')` 这种嵌套单引号），另有长度只有 1 个字符的“程序”。对 tokenizer 无害，但 CodeBot 生成不合法代码时不全是模型的锅。

---

## 6. 互动组件点子

| # | 组件 | 学习者操作 | 观察到的现象 / 要建立的直觉 | 实现备注 |
|---|---|---|---|---|
| 1 | **Unicode 显微镜** | 输入任意文本；切换 NFC/NFD | 每个字符的 code point、UTF-8 字节（按 `0xxxxxxx/110xxxxx/1110xxxx/11110xxx` 前缀上色）、Python `len` vs JS `.length` vs 字节数；`é` 在 NFC/NFD 下的 id 不同；👨‍👩‍👧 = 5 个 code point / 18 字节 | 纯 JS（`TextEncoder`、`normalize`） |
| 2 | **剪刀实验：UTF-8 截断** | 在字节条上拖动剪刀位置 | strict 抛错 / replace → `�` / ignore → 丢字；联系 `decode(errors="replace")` 与流式输出 | `TextDecoder({fatal})` |
| 3 | **BPE 单步动画** | 编辑小语料，点“下一步”；下拉切换 tie-break（先出现 / DLFS6 id / CS336 bytes） | pair 计数表（并列行高亮）、被选中的 pair、新 id、序列变短曲线；看到三种规则从第 1 步就分叉（F10/F44） | 纯 JS，数据量小 |
| 4 | **重叠陷阱** | 输入 `aaaaaaa` | `count_pairs` 说 6 次，`merge` 只替换 3 次 | 纯 JS |
| 5 | **pre-tokenizer 实验台** | 输入/选择预设（缩约词、`f'done'`、snake_case、数字、缩进、CJK、组合符）；逐个开关 6 个正则分支；切换 GPT-2 / cl100k 规则 | 彩色切片；看到 `'d` 误伤、`_` 断词、`' 1234567'` vs `'123','456','7'` | JS 正则要用显式空白类（J1）；cl100k 的 possessive 量词 JS 不支持，需改写 |
| 6 | **pre-tokenize 开/关对照训练** | 一键对比 | 关掉时学出 `'y hello? Just hello'`，打开后只在词内合并（F18） | 纯 JS |
| 7 | **CodeBot tokenizer 游乐场** | 粘贴代码/中文/emoji | token 胶囊（hover 看 id 与字节）、bytes/token；中文每字 3 个 token、emoji 4 个；`print('<|endoftext|>')` 冒出 999 | 纯 JS 移植（与 Python 逐 id 一致）+ `codebot_merges.json`（743 行） |
| 8 | **词表浏览器 + merge 树** | 按 id（学习顺序）/长度/语料频次排序，点击 token | 二叉 merge 树（`' BeautifulSoup'` 怎么一步步拼出来）；3 个死 token 与稀有 token 标红 | 频次来自 `tiny_codes.bin` 的 bincount |
| 9 | **在浏览器里训练 CodeBot tokenizer（端到端）** | 选语料比例、vocab 滑块（300–1000），点 Train | 进度条、实时 merge 列表、压缩率随 vocab 上升的曲线（参考值 F50：257→1.04、457→1.86、1000→2.42 B/token）；全语料 + 1000 时出现“与书中 merge_rules.pkl 完全一致 ✓” | Web Worker 跑 J2 的训练器（node 28.7 s；加倒排缓存可到几秒）；先把 CRLF 归一为 LF |
| 10 | **从语料到 `.bin`（端到端）** | 训练/加载 merges → encode 全语料 → 下载 | 生成 `Uint16Array`，校验前 20 个 id = `[116, 528, 821, 262, 544, 10, 119, …]`，文件 5,354,796 B；解释 uint16 的理由 | Blob 下载；与 ch03 的 `np.fromfile(..., np.uint16)` 对接 |
| 11 | **领域错配热力图** | 选 tokenizer × 语料 | F38 的 4×3 表：CodeBot-1K 在故事/网页上只有 ~2.0 B/token；StoryBot-10K 在代码上比 CodeBot-1K 还差 | 预计算数字 + 可选在线重算 |
| 12 | **special token 注入开关** | 在“chat 输入框”里键入 `<|endoftext|>`；切换“DLFS6 行为 / tiktoken 行为” | DLFS6 静默变 999；tiktoken 报错；讨论 chat 安全 | 纯 JS |
| 13 | **流式解码** | 逐个吐出 `"世界😁"` 的 token | 朴素逐 token decode 出 `���`；字节缓冲 + `TextDecoder(stream)` 正确 | 纯 JS |
| 14 | **uint16 预算** | 拖动 vocab 大小 | 需要的位数、dtype、文件大小（2.68M / 2.65B token）；越过 65,536 时演示回绕（numpy 1）与报错（numpy 2） | 纯 JS |
| 15 | **loss 基线阶梯** | 无 | ln(1000)=6.91（均匀）→ unigram 熵 5.76 → 训练后的 loss；用 2.42 B/token 换算成 bits/byte（F28/F41/F48） | 预计算 |
| 16 | **两种 encode 赛跑** | 输入一个词 | 左边逐条套 743 条 merge（大多数是空操作），右边只合并现存的最小 rank pair；结果相同、步数差两个数量级（F30） | 纯 JS |
| 17 | **chunk 边界可视化（ch07）** | 拖动 chunk 数 | 文件条等分 → 每个边界滑到下一个 `<|endoftext|>` → 并行 pre-tokenize → Counter 合并（map-reduce） | SVG 动画 |
| 18 | **“BPE 重新发现 GPT-2”** | 无 | WebBot 与 GPT-2 前 20 个 merge 并排、88.2% 词表重合；30 MB 样本已复现前 15 条（F36/F45） | 预计算 |
| 19 | **CRLF 陷阱（Windows 读者专属）** | 切换 LF/CRLF | 第 2 条 merge 变 `\r\n`、每行多 1 个 token | 预计算 + 小样本在线重算 |

---

## 7. CS336 对照（Stanford CS336: Language Modeling from Scratch）

> 本节主体凭记忆写成。标 **[crossref]** 的条目已由兄弟档案 `research/cs336-crossref.md` 联网或实测核对过，我在这里只引用其结论。

| DLFS6 | CS336 对应 | 异同 |
|---|---|---|
| `ch01/01` 字符级、`ch01/02` 字节级 | Lecture 1（Overview & Tokenization：character / byte / word / BPE tokenizer 与 compression ratio）；Assignment 1 §2.1–2.2 的 `unicode1`、`unicode2` 小题（`chr(0)`、UTF-8 vs UTF-16/32、错误的逐字节解码函数） | 思路一致；DLFS6 用 `"hello世界😁"` 一个例子贯穿 |
| `ch01/03` `count_pairs`/`merge`/`train_bpe` | A1 §2.4 “BPE Tokenizer Training”（含 `low lower widest newest` 手算例） | CS336 的 merges 是 `list[tuple[bytes, bytes]]`、vocab 是 `dict[int, bytes]`；DLFS6 用 `dict[(int,int)] → int`，词表靠 `id_to_bytes` 递推 |
| tie-break `(count, id0, id1)` | A1 规定 “prefer the lexicographically greater pair”（比较 bytes） | **不同**（本档案 F44 用构造的小语料演示）。**[crossref]** 在 CS336 的 `corpus.en`（vocab 500）上，DLFS6 ch04 训练器在第 92 条 merge 处与参考答案分叉（243 条中 170 条位置相同）；只把 key 换成 bytes 字典序，243/243 全部一致，耗时 1.04 s（低于 CS336 的 1.5 s 速度线） |
| `ch01/04` encode/decode | A1 §2.6 `Tokenizer.encode/decode`（按创建顺序应用 merges；decode 用 `errors='replace'`） | 一致 |
| `ch01/05` `<|endoftext|>` | A1：special token 加入词表；训练前按 special token 切开、禁止跨文档合并；encode 支持多个、可重叠的 special token（按长度降序匹配） | DLFS6 只支持 1 个、固定放在最后一个 id（与 GPT-2 的 50256 同理）；不处理重叠 |
| `ch01/06` GPT-2 正则 | A1 给出的 `PAT` 与 DLFS6 **逐字相同** | 一致 |
| `ch01/07` + `codebot/tokenizer.py` | A1 deliverable `train_bpe`（`run_train_bpe(input_path, vocab_size, special_tokens)`） | CS336 输入是文件路径（大文件），DLFS6 ch01 版输入是整段字符串；ch04/storybot 版改为文件路径 |
| `ch01/08` 压缩率 + tiktoken 对比 | A1 §2.7 `tokenizer_experiments`：(a) 压缩率 bytes/token，(b) 跨领域（TinyStories tokenizer 编 OWT），(c) 吞吐（并估算编码 Pile 825 GB 的时间） | 一致；F38 的领域错配表正好是 (b) 的扩展版；ch07 缩小版测到 0.51 MB/s（8 进程）可直接做 (c) |
| `ch01/09` / `ch07/02` → `uint16` `.bin` | A1 §2.7 (d)：把 train/dev 编码成 `uint16` NumPy 数组，并回答 “为什么 uint16 合适” | 一致；DLFS6 用 `tofile`/`memmap`，CS336 常用 `np.save`/`np.memmap` |
| `ch04`（storybot/tokenizer.py：chunk 边界、多进程、去重计数、倒排缓存、rank 优先 encode） | A1 §2.5 性能优化：并行 pre-tokenization（讲义附带的 `pretokenization_example.py` 里的 `find_chunk_boundaries` 与 DLFS6 版本结构几乎相同：等分 → 每 4096 字节找 special token）、增量更新 pair 计数；`train_bpe_tinystories`（10K vocab，时间/内存上限）| 一致；CS336 鼓励用堆/增量结构，DLFS6 每步仍 `max()` 全扫 |
| `ch07/01`（OWT，50K） | A1 `train_bpe_expts_owt`（OWT，32K；讲义给的上限是 12 h / 100 GB RAM） | **同一份数据**。**[crossref]** 作者的 HF 数据集卡片注明 OWT 样本来自 `stanford-cs336/owt-sample`，且 `webbot/owt_valid.txt` 与 CS336 的 `owt_valid.txt.gz` 解压后完全相同。CS336 vocab 32K，DLFS6 50K |
| `ch07/02` `encode_file` | A1 `encode_iterable`（流式、内存受限地编码大文件） | DLFS6 用“chunk + 多进程 + 磁盘缓存 + memmap”，CS336 要求 generator 式流式接口 |
| StoryBot/TinyStories 10K | A1 `train_bpe_tinystories`（TinyStoriesV2-GPT4，10K vocab；30 min / 30 GB 上限） | **同一份数据、同一 vocab**。**[crossref]** `storybot/tiny_stories_{train,valid}.txt` 与 CS336 的 `TinyStoriesV2-GPT4-{train,valid}.txt` 大小和 SHA-256 都相同；StoryBot 的预训练超参也与 A1 的 TinyStories 配置一致 |
| 初始 loss ≈ ln(V)、bits/byte | A1 §4–5 训练/评估部分；Lecture 中关于 perplexity 与 tokenizer 无关指标的讨论 | DLFS6 书中未见 bits/byte；建议教程补上（F41） |

一句话：DLFS6 ch01 ≈ CS336 A1 §2.1–2.4 + §2.6 的“讲解版”，ch04 ≈ §2.5 的“优化版”，ch07 ≈ `train_bpe_expts_owt` + §2.7(d) 的“规模版”；最大的实现差异是 tie-break 与 special token 的通用性。

---

## 附录 A：复现命令

```bash
cd C:/Users/chenf/Projects/_refs/dlfs6
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
uv run python ch01/01_char_tokenizer.py          # 01–06、08 直接跑（不写文件）
uv run python ch01/08_eval.py
SC="C:/Users/chenf/AppData/Local/Temp/claude/C--Users-chenf/2766cf54-7521-48a1-be58-647c0a0601b0/scratchpad/ch01-tokenizer"
uv run python "$SC/07_tiny_codes_patched.py"      # ≈1 h，写 $SC/merge_rules_retrained.pkl
uv run python "$SC/09_bpe_encode_patched.py"      # ≈10.5 min，写 $SC/tiny_codes_reencoded.bin
uv run python "$SC/fast_train_full.py"            # storybot 训练器 19 s 复现 merge_rules.pkl + CRLF 对照
uv run python "$SC/ch07_01_bpe_train_shrunk.py" 2000
uv run python "$SC/ch07_02_bpe_encode_shrunk.py"
uv run python "$SC/facts_runner.py" "$SC/facts_part1.md"
uv run --with tiktoken python "$SC/facts_runner2.py" "$SC/facts_part2.md"
uv run --with tiktoken python "$SC/facts_runner3.py" "$SC/facts_part3.md"
node "$SC/js/bpe_parity.mjs"; node "$SC/js/ws_fix.mjs"; node "$SC/js/bpe_train.mjs"
uv run python "$SC/export_facts.py" && (cd "$SC/pyodide" && node run_facts.mjs)   # Pyodide 0.29.4 对照
```

## 附录 B：scratch 中可直接给前端用的资产

- `SC\js\codebot_merges.json`：743 条 `[a, b, new_id]`（按 rank 顺序），网页里的 CodeBot tokenizer 只需要它。
- `SC\js\parity_cases.json`：29 个 Unicode 用例 + 200k 字符样本的 Python 参考输出（前端单元测试用）。
- `SC\js\bpe_parity.mjs` / `bpe_train.mjs`：可直接改成浏览器模块的 encode / train 实现。
- `SC\dl\webbot\merge_rules.pkl`、`SC\dl\storybot\merge_rules.pkl`：官方 50K / 10K 规则。
- `SC\dl\webbot\owt_valid_head30MB.txt`、`SC\dl\storybot\tiny_stories_valid_head10MB.txt`：领域对比样本。
