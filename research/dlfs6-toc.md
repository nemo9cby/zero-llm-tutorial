# 研究档案：DLFS6《ゼロから作るDeep Learning ❻ —LLM編》官方目次、叙事结构与配套资源

> 调研日期：2026-10-11。本地仓库：`C:\Users\chenf\Projects\_refs\dlfs6`（oreilly-japan/deep-learning-from-scratch-6，HEAD `c9b6e2e`，2026-04-15，本地为 depth=50 的浅克隆）。
> 草稿脚本与原始网页抓取都在 scratch：`C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\toc-research\`（下文简称 `SC\`）。仓库内没有改动任何文件。
> 引用书中原文时，每处最多 1 到 2 句，其余一律转述。

**证据标签**（每条结论都带一个）：

| 标签 | 含义 |
|---|---|
| 【源】 | 一手来源：出版社页面、书的官方预览片段、官方仓库代码或 issue、HF 数据集卡片、作者或出版社的 X 帖子。后面给出 `[Sx]` 编号，URL 见 §13 |
| 【二手】 | 第三方转述书中内容，例如 note.com 引用的「まえがき」或 Amazon 书评。没有拿到书本身来核对 |
| 【验】 | 本机实际跑过命令或脚本。命令见 §14 |
| 【推】 | 推断：由代码、文件名或目录结构推出，未经书面证实 |

---

## 0. 一页速览（TL;DR）

1. **找到了官方完整目次** 【源 S1】。O'Reilly Japan 书页列出全部章、节、小节，外加「コラム目次」（6 篇专栏）。O'Reilly Learning 的公开 API 里还有电子版目录，标出了每篇专栏所在的位置 【源 S8】。两者一致。
2. **结构**：まえがき → 0章 → 三个部分，每部分都依次讲 **トークナイザ／モデル／学習** 三章：
   - 第 I 部「基本編」做 CodeBot，第 1–3 章。
   - 第 II 部「応用編」做 StoryBot，第 4–6 章。
   - 第 III 部「挑戦編」做 WebBot，第 7–9 章。
   - 之后是附录 A–E → おわりに → 参考文献 → 索引。

   前言把这种结构概括为「同じリズムを、異なるスケールで」（同一节奏，不同规模）【源 S10 ch00f】。
3. **书目信息**：
   - O'Reilly Japan 出版，**2026-06-03 发售**。版权页记为「2026年6月1日 初版第1刷発行」【源 S1、S2、S6、S10 colophon】。原定 6 月下旬，2026-04-23 宣布提前到 6/3 【源 S19】。
   - **384 页**，A5 开本（21×15×2 cm），定价 ¥4,400（本体 ¥4,000＋税）【源 S1、S2、S3、S5、S6】。
   - **ISBN 978-4-8144-0161-1**（ISBN-10 4814401612，校验位已核对【验】）。
   - 国立国会图书馆（NDL）著录为「341p ; 21cm」【源 S4】。差额的原因见 §1。
4. **第 III 部是"提示模式"**：书中明说这一部分不给出详细实现步骤，只提示要点和方向【源 S10 part3】。因此第 7–9 章没有三级小节，体量也很小：O'Reilly 虚拟页分别只有 2、4、6 页，而第 2 章有 50 页【源 S9】。
5. **ch08 没有代码目录**，原因【推】：第 8 章（模型・挑戦編）的代码实际都在 `webbot/model.py` 里，包括：
   - 用 `F.scaled_dot_product_attention` 做 Flash Attention；
   - 用 `enable_gqa=True` 加 `n_kv_head=4` 实现 GQA。

   `torch.compile` 写在 `ch09/01_pretrain.py` 里【验】。MoE 和上下文长度扩展在仓库里没有任何代码【验】。
6. **三份数据** 都由作者重新打包，发布在 HF 数据集 `koki0702/zero-llm-data`（许可证 CDLA-Sharing-1.0）【源 S14】：
   - **TinyCodes**：由 `flytech/python-codes-25k` 修改而来，约 6.5 MB。
   - **TinyStories V2-GPT4**：来自 `roneneldan/TinyStories`，约 2.2 GB。
   - **OpenWebText 样本**：来自 **`stanford-cs336/owt-sample`**，约 11.9 GB。
7. **GPU 的官方说法**，出自前言，经 note.com 引用【二手 S20】：
   - 第 I 部只用 CPU 也能跑；
   - 第 II 部推荐 GPU；
   - 第 III 部必须有 GPU，最好是 A100/H100 级。
   - 前言还点名了 Colab、Kaggle，以及 **RunPod、Lambda**、Vast.ai。
   - 据 Amazon 书评，书中给出的是 **RTX 4090** 上的处理时间【二手 S6】。具体数值没拿到。
   - 营销页和推文里都没有成本或时长的数字【源 S1、S18、S19】。
8. **与 CS336 的关系**：书里明确致谢并引用了 CS336【源 S10 ch27aft、ch05、ch28bib】。代码层面的证据有三条：
   - StoryBot 的超参数与 CS336 A1 的 TinyStories 配置完全相同，`find_chunk_boundaries` 的写法也与 CS336 样例同构【验 + 源 S27、S28】；
   - WebBot 直接使用 CS336 的 OWT 样本【源 S14】。

   详见兄弟档案 `cs336-crossref.md`。
9. **译本情况**：
   - 韩文版『밑바닥부터 시작하는 딥러닝 6』由 Hanbit Media 于 2026-08-31 出版，400 页，译者 개앞맵시（이복연）【源 S24】。配套仓库里有 Windows 用的 `download.ps1`【源 S25】。
   - 截至 2026-10-11 **没有找到中文版**。豆瓣条目 38372517 收录的是日文原版【源 S23】。
   - 英文版尚未见到，只有 issue #1 里读者的催问【源 S13】。

---

## 1. 书目信息（逐条带来源）

| 项目 | 值 | 来源 |
|---|---|---|
| 书名 | ゼロから作るDeep Learning ❻ ―LLM編 | 【源 S1】 |
| 作者 | 斎藤 康毅（Koki Saitoh；X 账号 @SaitohKoki；HF / GitHub 账号 koki0702） | 【源 S1、S15、S14】 |
| 出版／发行 | 株式会社オライリー・ジャパン。纸书经销为 Ohmsha。制作协力为アリエッタ株式会社 | 【源 S1、S2、S10 colophon】 |
| 发售日 | 2026-06-03。O'Reilly Learning 电子版同日上线 | 【源 S1、S2、S3、S6；二手 S20 转引 @oreilly_japan 2026-06-03 推文】 |
| 版权页日期 | 2026年6月1日 初版第1刷発行。O'Reilly Learning API 的 `publication_date` 字段为 2026-06-01 | 【源 S10 colophon、S7】 |
| 原定日期与提前 | 2025-12-22 预告"2026年6月発行予定"。2026-04-23 宣布"発売日が早まり…6/3(水)発売予定" | 【源 S18、S19】 |
| 预售期的旧数据 | 豆瓣仍显示 2026-6-29、336 页。当时乐天、TechPlay 也是这组数，现在页面已 404 或 403 | 【源 S23】。另外两家为【推】（来自搜索摘要） |
| 页数 | 384（出版社、Ohmsha、版元ドットコム、纪伊国屋、Amazon、O'Reilly API `page_count`） | 【源 S1–S3、S5–S7】 |
| NDL 著录 | 「341p ; 21cm」，内容細目「LLM編」，注記「文献あり 索引あり」「サンプルデータダウンロード」 | 【源 S4】。341 与 384 的差额，【推】是因为前付部分用罗马数字编页、卷末又有不编号的页，NDL 只计了正文的阿拉伯数字页码 |
| 开本 | A5，21 cm（Amazon 标 21×15×2 cm，软皮） | 【源 S2、S5、S6】 |
| 定价 | 4,400 円（本体 4,000 円＋税）。电子版 4,400 円，提供 PDF、EPUB | 【源 S1、S2】 |
| ISBN | 978-4-8144-0161-1 / 4814401612 | 【源 S1、S6】；校验位【验】§14 |
| 难度与读者 | O'Reilly 标为 Intermediate。Ohmsha 写「ディープラーニングの応用を知りたい技術者、Pythonプログラマー、教育関係者」 | 【源 S2；搜索摘要】 |
| 电子版体量 | 224 个"虚拟页"，阅读时长约 4 小时 17 分（224 × 1.15 分 ≈ 257.6 分） | 【源 S7、S9；验 §14】 |
| 公开审稿 | 2025-12-20 开始，2026-01-18 截止，在 Notion 上进行。审稿人名单页于 2026-01-27 公开。审稿页现已不公开（`publicAccessRole: "none"`） | 【源 S15、S16、S17、S31】 |
| 勘误 | 仓库 README 指向 `…/wiki/errata`，但 wiki 不存在，访问会 302 跳回仓库首页。Ohmsha 页写「正誤表はございません」。韩文版勘误文档写着"尚无已确认错误" | 【验 S11；源 S2、S25】 |
| 仓库时间线 | 仓库创建于 2025-09-10，共 52 个 commit。2025-12-18 推送代码（公开审稿开始前两天），2026-02-18 补上 ch09，最后一次提交在 2026-04-15 | 【源 S32】 |
| HF 数据集时间线 | 2025-12-09 创建，最后修改 2026-01-23 | 【源 S14 API】 |
| 韩文版 | 『밑바닥부터 시작하는 딥러닝 6：토크나이저부터 추론 모델까지 직접 만드는 LLM』，Hanbit Media，2026-08-31，400 页，译者 개앞맵시（이복연），31,500 원 | 【源 S24】 |
| 中文版 | 未找到。前几卷由人民邮电出版社 / 图灵出版，书名为「深度学习入门」「深度学习进阶」等 | 【源 S23；搜索结果，日期为 2026-10-11】 |

---

## 2. 官方完整目次（日文原题 ＋ 中文译名 ＋ 对应代码）

原题取自 O'Reilly Japan 书页【源 S1】，专栏位置取自电子版目录【源 S8】。

两份目录的写法有几处小差异，都只是格式问题：

| 项目 | O'Reilly Japan 书页 | 电子版目录 |
|---|---|---|
| 罗马数字 | 「第I部」 | 「第Ⅰ部」 |
| 专栏名 | 「Dr.GRPO」 | 「Dr. GRPO」 |
| 7.3 节题 | 「Rust/C++」 | 「Rust / C++」 |
| 卷末 | 列出「索引」 | 没有「索引」，多出「著者紹介」「奥付」 |

章节后方括号里写的是本地仓库中的对应文件。这个映射的依据是：
- 文件名编号 `NN_*.py` 对应第 N 节。书的预览片段证实了几处，例如 1.1.1 标题下印着 `ch01/01_char_tokenizer.py`，2.1.1 下印着 `ch02/01_soft_dict.py`，7.1 下印着 `ch07/01_bpe_train.py`【源 S10】；issue #7 证实 4.5 节用的是 `ch04/05_eval.py`【源 S12】。
- 其余文件为【推】，即由文件内容对照目次推出。

各章体量：节数与小节数【验 §14】，电子版虚拟页数【源 S9】。

| 章 | 节 | 小节 | 虚拟页 |
|---|---|---|---|
| 1 | 9 | 27 | 23 |
| 2 | 10 | 36 | 50 |
| 3 | 9 | 35 | 32 |
| 4 | 7 | 28 | 23 |
| 5 | 5 | 21 | 18 |
| 6 | 8 | 27 | 26 |
| 7 | 6 | 0 | 2 |
| 8 | 6 | 0 | 4 |
| 9 | 7 | 0 | 6 |

其余部分的虚拟页：前言 4、第 0 章 10、三个部扉 3/3/2、五个附录各 1、参考文献 7。

已知页码锚点：4.5 节在纸书第 197–199 页【源 S12】。

### まえがき（前言）
### 0章　LLMができるまで（第0章　LLM 是如何炼成的）［无代码］

### 第I部　基本編――CodeBotを作る（第 I 部　基础篇：制作 CodeBot）［`codebot/`］
- I.1 開発の流れ（开发流程）
- I.2 事前学習（预训练）
- I.3 SFT（教師ありファインチューニング）（SFT：监督微调）
- I.4 強化学習（强化学习）

#### 1章　トークナイザ【基本編】（第1章　分词器【基础篇】）［`ch01/`、`codebot/tokenizer.py`］
- **1.1 文字単位のトークン化（字符级分词）** → `01_char_tokenizer.py`
  - 1.1.1 文字をIDに変換する（把字符转换为 ID）
  - 1.1.2 文字単位トークナイザの実装（实现字符级分词器）
  - 1.1.3 文字単位トークナイザの課題（字符级分词器的问题）
- **1.2 バイト単位のトークン化（字节级分词）** → `02_byte_tokenizer.py`
  - 1.2.1 UTF-8エンコード（UTF-8 编码）
  - 1.2.2 バイト単位トークナイザの実装（实现字节级分词器）
  - 1.2.3 バイト単位トークナイザの課題（字节级分词器的问题）
- **1.3 BPEの学習アルゴリズム（BPE 的训练算法）** → `03_bpe_train.py`
  - 1.3.1 BPEの処理手順（BPE 的处理步骤）
  - 1.3.2 BPEの実装（实现 BPE）
  - 〔コラム〕BPEのタイブレーク（引き分け処理）（专栏：BPE 的平票处理）
  - 1.3.3 BPEの動作確認（运行验证 BPE）
- **1.4 BPEによるエンコードとデコード（用 BPE 编码与解码）** → `04_bpe_tokenizer.py`
  - 1.4.1 BPEトークナイザの実装（实现 BPE 分词器）
  - 1.4.2 エンコードの実装（实现编码）
  - 1.4.3 デコードの実装（实现解码）
  - 1.4.4 動作確認（运行验证）
- **1.5 特殊トークンへの対応（支持特殊 token）** → `05_special_token.py`
  - 1.5.1 特殊トークンの役割（特殊 token 的作用）
  - 1.5.2 特殊トークンに対応したBPEの学習（支持特殊 token 的 BPE 训练）
  - 1.5.3 train_bpe関数の実装（实现 train_bpe 函数）
  - 1.5.4 特殊トークンに対応したトークナイザの実装（实现支持特殊 token 的分词器）
- **1.6 事前トークン化（预分词 pre-tokenization）** → `06_pretokenize.py`
  - 1.6.1 事前トークン化の必要性（为什么需要预分词）
  - 1.6.2 GPT-2の事前トークン化パターン（GPT-2 的预分词正则）
  - 1.6.3 事前トークン化に対応したBPE学習の実装（实现支持预分词的 BPE 训练）
  - 1.6.4 事前トークン化に対応したトークナイザの実装（实现支持预分词的分词器）
- **1.7 TinyCodesを用いた学習（用 TinyCodes 训练）** → `07_tiny_codes.py`，以及整理出来的 `codebot/tokenizer.py`
  - 1.7.1 トークナイザの読み込み機能（分词器的加载功能）
  - 1.7.2 これまでのコードの整理（整理目前为止的代码）
  - 1.7.3 TinyCodesデータセットを用いた学習（用 TinyCodes 数据集训练）
- **1.8 トークナイザの評価（评估分词器）** → `08_eval.py`
  - 1.8.1 学習済みトークンの確認（查看学到的 token）
  - 1.8.2 圧縮効率の測定（测量压缩效率）
  - 1.8.3 他トークナイザとの比較（与其他分词器比较，tiktoken 的 gpt2 和 cl100k_base）
- **1.9 TinyCodesのエンコード（编码 TinyCodes）** → `09_bpe_encode.py`

#### 2章　モデル【基本編】（第2章　模型【基础篇】）［`ch02/`、`codebot/model.py`］
- **2.1 Attentionの基本原理（Attention 的基本原理）** → `01_soft_dict.py`
  - 2.1.1 Pythonのディクショナリ（Python 字典）
  - 2.1.2 ソフトなディクショナリ（"软"字典）
  - 2.1.3 ソフトなディクショナリの実装例（软字典的实现示例）
- **2.2 Attentionの数式と実装（Attention 的公式与实现）** → `02_attn_math.py`
  - 2.2.1 Attentionを数式で表す（用公式表示 Attention）
  - 2.2.2 複数クエリへの対応（支持多个查询）
  - 〔コラム〕Attentionの行単位の計算（专栏：Attention 的逐行计算）
  - 2.2.3 Attentionの実装（实现 Attention）
- **2.3 Attentionスケーリング（Attention 缩放）** → `03_attn_scaling.py`
  - 2.3.1 ソフトマックス関数の飽和問題（softmax 的饱和问题）
  - 2.3.2 スケーリングの導入（引入缩放）
  - 2.3.3 スケーリングの効果（缩放的效果）
- **2.4 Transformerの基本構造（Transformer 的基本结构）**［无代码文件］
  - 2.4.1 Transformerの動作プロセス（Transformer 的运作流程）
  - 〔コラム〕Transformerと文章の確率計算（专栏：Transformer 与句子概率的计算）
  - 2.4.2 TransformerにおけるAttentionの役割（Attention 在 Transformer 中的作用）
- **2.5 位置エンコーディング（位置编码）**［无代码文件］
  - 2.5.1 位置情報が必要な理由（为什么需要位置信息）
  - 2.5.2 位置エンコーディングの仕組み（位置编码的机制）
  - 2.5.3 絶対位置エンコーディング（绝对位置编码）
  - 2.5.4 学習可能な位置エンコーディング（可学习的位置编码）
- **2.6 Attentionマスク（Attention 掩码）** → `06_attn_mask.py`
  - 2.6.1 効率的なデータ形式（高效的数据格式）
  - 2.6.2 Attentionの問題点（Attention 的问题）
  - 2.6.3 Attentionマスクの仕組み（Attention 掩码的机制）
  - 2.6.4 Attentionマスクの実装（实现 Attention 掩码）
- **2.7 バリュー行列の効率化（Value 矩阵的高效化）** → `07_attn_value.py`
  - 2.7.1 大規模モデルにおけるAttentionの課題（大模型中 Attention 的难点）
  - 2.7.2 行列分解による低ランク近似（用矩阵分解做低秩近似）
  - 2.7.3 行列分解によるAttentionの新たな解釈（从矩阵分解重新理解 Attention）
  - 2.7.4 実装例（实现示例）
- **2.8 マルチヘッドAttention（多头 Attention）** → `08_multi_head.py`
  - 2.8.1 複数の視点で捉える（从多个视角捕捉）
  - 2.8.2 マルチヘッドAttentionの数式（多头 Attention 的公式）
  - 2.8.3 2つの等価な数式（两个等价的公式）
  - 2.8.4 マルチヘッドAttentionの実装（实现多头 Attention）
  - 2.8.5 マルチヘッドAttentionの実装（まとめ）（实现多头 Attention：汇总）
- **2.9 Transformerブロック（Transformer 块）** → `09_norm_gelu.py`；`graph.py` 画 ReLU 与 GELU 的对比图
  - 2.9.1 LayerNormの仕組み（LayerNorm 的机制）
  - 2.9.2 LayerNormの実装（实现 LayerNorm）
  - 2.9.3 FFNの実装（实现 FFN）
  - 2.9.4 Transformerブロックの実装（实现 Transformer 块）
- **2.10 GPT-2（GPT-2）** → `10_gpt2.py`
  - 2.10.1 GPT-2の実装（实现 GPT-2）
  - 2.10.2 モデルの改良（改进模型）
  - 2.10.3 モデルの保存と読み込み（模型的保存与加载）
  - 2.10.4 動作確認（运行验证）

#### 3章　学習【基本編】（第3章　训练【基础篇】）［`ch03/`、`codebot/`］
- **3.1 事前学習（预训练）** → `01_pretrain.py`
  - 3.1.1 データセットの確認（查看数据集）
  - 3.1.2 デバイスの設定（设置设备）
  - 3.1.3 事前学習のコード（前半部分）（预训练代码：前半）
  - 3.1.4 事前学習のコード（後半部分）（预训练代码：后半）
  - 3.1.5 学習結果の確認（查看训练结果）
- **3.2 テキスト生成（文本生成）** → `02_generate.py`
  - 3.2.1 テキスト生成の方法（文本生成的方法）
  - 3.2.2 温度パラメータ（温度参数）
  - 3.2.3 Top-kとTop-pサンプリング（Top-k 与 Top-p 采样）
  - 3.2.4 テキスト生成の実装（实现文本生成）
  - 3.2.5 テキスト生成の実行（运行文本生成）
- **3.3 SFT（教師ありファインチューニング）（SFT：监督微调）** → `03_alpaca.py`（3.3.2）、`03_sft.py`
  - 3.3.1 SFT用データセットの形式（SFT 数据集的格式）
  - 3.3.2 Alpaca形式でトークン化する（按 Alpaca 格式分词）
  - 3.3.3 マスク処理（掩码处理）
  - 3.3.4 パディング（填充 padding）
  - 3.3.5 SFTの実装（实现 SFT）
  - 3.3.6 学習結果の確認（查看训练结果）
- **3.4 SFTモデルとの対話（与 SFT 模型对话）** → `04_chat.py`
  - 3.4.1 インタラクティブな対話（交互式对话）
  - 3.4.2 モデルの計算能力（模型的计算能力）
- **3.5 強化学習と言語モデル（强化学习与语言模型）**［纯理论］
  - 3.5.1 強化学習と言語モデルの対応関係（强化学习与语言模型的对应关系）
  - 3.5.2 軌跡（轨迹）
  - 3.5.3 報酬と収益（奖励与回报）
  - 3.5.4 言語モデルにおける報酬と収益（语言模型中的奖励与回报）
- **3.6 方策勾配法（策略梯度法）**［纯理论］
  - 3.6.1 方策を直接改善する（直接改进策略）
  - 3.6.2 方策勾配の導出（推导策略梯度）
  - 3.6.3 方策勾配の近似（近似策略梯度）
- **3.7 ベースライン（基线 baseline）**［纯理论］
  - 3.7.1 ベースラインのアイデア（基线的想法）
  - 3.7.2 ベースライン付き方策勾配法（带基线的策略梯度法）
  - 3.7.3 GRPO：グループ平均をベースラインにする（GRPO：用组内平均作基线）
  - 〔コラム〕GRPOとDr. GRPO（专栏：GRPO 与 Dr. GRPO）
- **3.8 オフポリシーとクリッピング（离策略 off-policy 与裁剪 clipping）**［纯理论］
  - 3.8.1 オンポリシー学習とオフポリシー学習（同策略学习与离策略学习）
  - 3.8.2 重要度サンプリングによる勾配の導出（用重要性采样推导梯度）
  - 3.8.3 重要度重みの近似（近似重要性权重）
  - 3.8.4 クリッピングによる安定化（用裁剪稳定训练）
- **3.9 GRPO（GRPO）** → `09_grpo.py`
  - 3.9.1 GRPOのアルゴリズム（GRPO 算法）
  - 3.9.2 GRPOの実装（实现 GRPO）
  - 3.9.3 実験結果（实验结果）

### 第II部　応用編――StoryBotを作る（第 II 部　应用篇：制作 StoryBot）［`storybot/`］
- II.1 データセット（数据集）
- II.2 開発の流れ（开发流程）

#### 4章　トークナイザ【応用編】（第4章　分词器【应用篇】）［`ch04/`、`storybot/tokenizer.py`］
- **4.1 BPE学習の改善①：重複除去（改进 BPE 训练 ①：去重）** → `01_bpe_optimize.py`
  - 4.1.1 元の実装の問題点（原实现的问题）
  - 4.1.2 重複の除去（去除重复）
  - 4.1.3 改良版のBPE学習（改进版 BPE 训练）
  - 4.1.4 改良版の動作確認（运行验证改进版）
- **4.2 BPE学習の改善②：部分更新（改进 BPE 训练 ②：增量更新）** → `02_bpe_cache.py`
  - 4.2.1 現在の問題点（当前的问题）
  - 4.2.2 キャッシュによる部分更新（用缓存做增量更新）
  - 4.2.3 改良版の実装（实现改进版）
  - 4.2.4 動作確認（运行验证）
- **4.3 BPE学習の改善③：分割処理（改进 BPE 训练 ③：分块处理）** → `03_bpe_chunk.py`
  - 4.3.1 大規模データでの問題（大规模数据下的问题）
  - 4.3.2 メモリ消費の原因（内存消耗的原因）
  - 4.3.3 ファイルの部分読み取り（部分读取文件）
  - 4.3.4 改良版のtrain_bpe関数（改进版 train_bpe 函数）
  - 4.3.5 動作確認（运行验证）
  - 〔コラム〕事前トークン化のメモリ効率化（专栏：降低预分词的内存占用）
- **4.4 BPE学習の改善④：並列化（改进 BPE 训练 ④：并行化）** → `04_bpe_parallel.py`
  - 4.4.1 Pythonでの並列処理（Python 中的并行处理）
  - 4.4.2 事前トークン化の並列処理（预分词的并行处理）
  - 4.4.3 動作確認（运行验证）
- **4.5 トークナイザの評価（评估分词器）** → 书中引用 `ch04/05_eval.py`，**但仓库里没有这个文件**【源 S12】，内容见第 197–199 页
  - 4.5.1 学習済みトークンの確認（查看学到的 token）
  - 4.5.2 圧縮効率の測定（测量压缩效率）
  - 4.5.3 CodeBotトークナイザとの比較（与 CodeBot 分词器比较）
- **4.6 BPEエンコードの改善①：高速マージ（改进 BPE 编码 ①：快速合并）** → `06_encode_optimize.py`
  - 4.6.1 現状のエンコード実装（现有的编码实现）
  - 4.6.2 効率的なエンコード処理（高效的编码处理）
  - 4.6.3 改良版の実装（实现改进版）
  - 4.6.4 動作確認（运行验证）
- **4.7 BPEエンコードの改善②：大規模データ対応（改进 BPE 编码 ②：支持大规模数据）** → `07_encode_parallel.py`
  - 4.7.1 分割処理（分块处理）
  - 4.7.2 チャンク処理の実装（实现 chunk 处理）
  - 4.7.3 np.memmapによる効率的な書き出し（用 np.memmap 高效写出）
  - 4.7.4 エンコードの実装（实现编码）
  - 4.7.5 動作確認（运行验证）

#### 5章　モデル【応用編】（第5章　模型【应用篇】）［`ch05/`、`storybot/model.py`］
- **5.1 RoPE（RoPE 旋转位置编码）** → `01_rope.py`
  - 5.1.1 RoPEのアイデア（RoPE 的想法）
  - 5.1.2 高次元ベクトルへの拡張（推广到高维向量）
  - 5.1.3 RoPEの実装（实现 RoPE）
  - 5.1.4 MultiHeadAttentionクラスへの組み込み（集成进 MultiHeadAttention 类）
- **5.2 SwiGLU（SwiGLU）** → `02_swiglu.py`；`graph.py` 画 ReLU、GELU、Swish 的对比图
  - 5.2.1 Swish関数（Swish 函数）
  - 5.2.2 GLUとSwiGLU（GLU 与 SwiGLU）
  - 5.2.3 SwiGLUを使ったFFN（使用 SwiGLU 的 FFN）
  - 5.2.4 SwiGLUの実装（实现 SwiGLU）
- **5.3 RMSNormとその他の改良（RMSNorm 及其他改进）** → `03_rmsnorm.py`
  - 5.3.1 RMSNorm
  - 5.3.2 Dropoutの削除（去掉 Dropout）
  - 5.3.3 バイアスの削除（去掉 bias）
  - 5.3.4 重み共有の廃止（取消权重共享 weight tying）
- **5.4 改良版GPTの実装（实现改进版 GPT）** → `04_improved_gpt.py`
  - 5.4.1 Blockクラスの実装（实现 Block 类）
  - 5.4.2 GPTクラスの実装（实现 GPT 类）
  - 5.4.3 モデルの保存と読み込み（模型的保存与加载）
  - 5.4.4 動作確認（运行验证）
- **5.5 KVキャッシュ（KV 缓存）** → `05_kvcache.py`
  - 5.5.1 テキスト生成における無駄（文本生成中的浪费）
  - 5.5.2 AttentionとKVキャッシュ（Attention 与 KV 缓存）
  - 5.5.3 PrefillとDecode（Prefill 与 Decode）
  - 5.5.4 KVキャッシュの実装（实现 KV 缓存）
  - 5.5.5 テキスト生成での使用（在文本生成中使用）

#### 6章　学習【応用編】（第6章　训练【应用篇】）［`ch06/`、`storybot/`］
- **6.1 AdamWの仕組み（AdamW 的原理）**［理论；附录 D 有补充推导］
  - 6.1.1 SGD（確率的勾配降下法）（SGD：随机梯度下降）
  - 6.1.2 Momentum（モーメンタム）（动量）
  - 6.1.3 Adam（Adaptive moment estimation）（Adam：自适应矩估计）
  - 6.1.4 AdamW
- **6.2 AdamWの実装（实现 AdamW）** → `02_adamw.py`
  - 6.2.1 SGDの実装（实现 SGD）
  - 6.2.2 AdamWの実装（实现 AdamW）
  - 6.2.3 動作確認（运行验证）
- **6.3 学習率スケジュール（学习率调度）** → `03_lr_scheduler.py`；`lr_graph.py` 画余弦退火与 D2Z 的对比图
  - 6.3.1 なぜ学習率を変化させるのか（为什么要改变学习率）
  - 6.3.2 ウォームアップとアニーリング（预热 warmup 与退火 annealing）
  - 6.3.3 D2Zの実装（实现 D2Z：线性衰减到零）
- **6.4 混合精度（混合精度）** → `04_mixed_precision.py`
  - 6.4.1 低精度の問題点（低精度的问题）
  - 6.4.2 浮動小数点数の表現方法（浮点数的表示方法）
  - 6.4.3 混合精度の活用（利用混合精度）
  - 〔コラム〕GPU環境でBF16が使えない場合（专栏：GPU 不支持 BF16 时怎么办）
- **6.5 事前学習（改良版）（预训练：改进版）** → `05_pretrain.py`
  - 6.5.1 メモリマップによるデータ読み込み（用内存映射读取数据）
  - 6.5.2 勾配クリッピング（梯度裁剪）
  - 6.5.3 検証データによる評価（用验证集评估）
  - 6.5.4 事前学習の実装（实现预训练）
- **6.6 テキスト生成とモデル評価（文本生成与模型评估）** → `06_generate.py`（6.6.1）、`07_llm_judge.py`（6.6.3）
  - 6.6.1 テキスト生成（文本生成）
  - 6.6.2 モデルの評価（模型评估）
  - 6.6.3 LLM-as-a-Judge（以 LLM 作裁判）
- **6.7 RLHFとDPOの理論（RLHF 与 DPO 的理论）**［理论；附录 E 有推导］
  - 6.7.1 RLHFとDPOが解く問題（RLHF 与 DPO 要解决的问题）
  - 6.7.2 Bradley-Terryモデルと選好データ（Bradley-Terry 模型与偏好数据）
  - 6.7.3 RLHFのアプローチ（RLHF 的做法）
  - 6.7.4 DPOのアプローチ（DPO 的做法）
- **6.8 DPOの実装（实现 DPO）** → `08_dpo.py`、`09_llm_judge.py`（6.8.3）
  - 6.8.1 選好データとマスクの設計（偏好数据与掩码的设计）
  - 6.8.2 DPOの実装（实现 DPO）
  - 6.8.3 DPOモデルの評価（评估 DPO 模型）

### 第III部　挑戦編――WebBotを作る（第 III 部　挑战篇：制作 WebBot）［`webbot/`］
- III.1 データセット（数据集）
- III.2 開発の流れ（开发流程）

#### 7章　トークナイザ【挑戦編】（第7章　分词器【挑战篇】）［`ch07/`，复用 `storybot/tokenizer.py`］
- 7.1 WebBotのトークナイザ（WebBot 的分词器）→ `01_bpe_train.py`。`02_bpe_encode.py` 也在本节【推】
- 7.2 ボトルネックの検出（定位瓶颈）
- 7.3 Rust/C++による高速化（用 Rust/C++ 加速）
- 7.4 特殊トークンの追加（添加特殊 token）
- 7.5 他のトークナイザ（其他分词器）
- 7.6 トークナイザを使わないアプローチ（不用分词器的方法）

#### 8章　モデル【挑戦編】（第8章　模型【挑战篇】）［**没有 `ch08/` 目录**，代码见 `webbot/model.py`］
- 8.1 モデルサイズの選択（选择模型规模：从 scaling law 讲起【源 S10 ch08】）
- 8.2 Flash Attention → `webbot/model.py` 中的 `F.scaled_dot_product_attention`
- 8.3 Grouped-Query Attention（GQA）（分组查询注意力）→ `webbot/model.py`：`n_kv_head=4`、`enable_gqa=True`，需要 torch ≥ 2.5（见 commit `f85907f`）
- 8.4 Mixture of Experts（MoE）（混合专家）［仓库里没有代码【验】］
- 8.5 torch.compile → `ch09/01_pretrain.py` 中的 `model = torch.compile(model)`
- 8.6 コンテキスト長の拡張（扩展上下文长度）［仓库里没有代码【验】］

#### 9章　学習【挑戦編】（第9章　训练【挑战篇】）［`ch09/`、`webbot/`］
- 9.1 勾配蓄積（梯度累积）→ `01_pretrain.py`：`accumulation_steps=4`，用 `no_sync`
- 9.2 勾配チェックポイント（梯度检查点）［仓库里没有代码【验】］
- 9.3 分散学習（分布式训练）→ `01_pretrain.py`：DDP，`torchrun` 环境变量，NCCL
- 9.4 ZeRO（ZeRO）［仓库里没有代码【验】］
- 9.5 Muonオプティマイザ（Muon 优化器）［仓库里没有代码，用的仍是 `torch.optim.AdamW`【验】］
- 9.6 Weights & Biases → `01_pretrain.py`（`wandb` 是必需依赖）
- 9.7 WebBotの学習と生成結果（WebBot 的训练与生成结果）→ `01_pretrain.py`、`02_generate.py`

### 附录、后记与卷末
- 付録A　UTF-8エンコーディングの仕組み（附录A　UTF-8 编码的原理）
- 付録B　Attentionのスケーリング効果の証明（附录B　Attention 缩放效果的证明）
- 付録C　ベースライン付き方策勾配の不偏性の証明（附录C　带基线策略梯度无偏性的证明）
- 付録D　Adamのバイアス補正（附录D　Adam 的偏差校正）
- 付録E　DPOの最適解の導出（附录E　DPO 最优解的推导）
- おわりに（后记）／参考文献／索引

### コラム目次（6 篇专栏，位置见电子版目录【源 S8】）
| 专栏 | 位置 |
|---|---|
| BPEのタイブレーク（引き分け処理） | 1.3.2 之后 |
| Attentionの行単位の計算 | 2.2.2 之后 |
| Transformerと文章の確率計算 | 2.4.1 之后 |
| GRPOとDr. GRPO | 3.7.3 之后 |
| 事前トークン化のメモリ効率化 | 4.3 末尾 |
| GPU環境でBF16が使えない場合 | 6.4 末尾 |

---

## 3. 各章内容

依据有三类：O'Reilly Learning 的公开预览片段【源 S10】（每章只有开头约 2–4 KB 可读），目次【源 S1】，以及代码【验】。

### まえがき（前言）
【源 S10 ch00f】：
- 以 ChatGPT 的"魔法"开场，宣告本书要从零实现全部核心技术。
- 提出 LLM 开发的固定套路：トークナイザ → モデル → 学習，并用「同じリズムを、異なるスケールで」点题。
- 第 I 部：CodeBot 用约 6 MB 的小数据集，模型是 GPT-2。原文还说这个 bot "并不聪明"，但重要的是过程，不是结果。

【二手 S20，note.com 转引前言】：
- 第 II 部：StoryBot 建立在 GPT-2 之上，换用更现代的架构，并把分词器加速到能处理约 2 GB 的数据。
- 第 III 部：WebBot 使用 10 GB 以上的网络数据，只给提示，不给详细步骤。
- 前言坦言，凭个人之力做出 ChatGPT 级别的 LLM 几乎不可能，但在小规模上走一遍同样的流程完全可行。
- 各部分所需的算力见 §8。

### 0章　LLMができるまで
【源 S10 ch00g】：
- 先给出全景：语言模型就是给一段字符串打出"有多自然"的概率。例子是「A cat sat on the mat」对比打乱语序的版本。
- 再讲按下一个词预测来计算概率，例子是「A cat sat on the」之后接 mat、chair、floor 的概率（图 0-1）。
- 接着介绍 LLM 的构成要素，以及预训练和后训练两个阶段。

本章参考文献列有【源 S10 ch28bib】：
- Sennrich 2016（BPE）；
- Vaswani 2017；
- OpenAI GPT-OSS 2025；
- Karpathy 的 nanoGPT 2023 和 nanochat；
- Stanford CS336 2025。

注意：书中把 nanochat 的年份写成 2024，照录原文。

### 第 I 部扉页（CodeBot）
【源 S10 part1】：
- 先展示做好的 CodeBot 的对话样例：Hello 的寒暄；问"Who are you?"它回答"I'm CodeBot"；输入"write loop"它写出 for 循环；输入 `7+8=` 它回答 `15`。
- 再用一张表交代开发流程：第 1 章用 BPE，第 2 章做小型 GPT-2，第 3 章依次做预训练、SFT、强化学习。
- I.2 节说明预训练数据是 `codebot/tiny_codes.txt`。

### 第 1 章　分词器（基础篇）
【源 S10 ch01 + 验 代码】，演进路线是 字符 → 字节 → BPE：
- **字符级**：用 `ord` / `chr`，例句 `"hello世界😁"`。问题在于词表要覆盖全部 Unicode 码位。
- **字节级**：用 UTF-8，词表只有 256。问题在于序列变长。
- **BPE 训练**：`count_pairs`、`merge`、`train_bpe`，合并次数 = vocab_size − 256。
- **编码与解码**：用 `id_to_bytes` 表。
- **特殊 token**：`<|endoftext|>`，其 `end_token_id = 256 + len(merges)`，因此合并次数 = vocab_size − 257。
- **GPT-2 预分词正则**。
- **在 TinyCodes 上训练**：vocab 1000，即 743 次 merge，加 256 个字节 token，加 1 个特殊 token【验】。
- **评估**：压缩率按 bytes/token 计；1.8.3 与 tiktoken 的 gpt2、cl100k_base 对比（那段代码在 `08_eval.py` 里被包在 docstring 中，默认不执行）。
- **编码成 `uint16` 的 `.bin`**：共 2,677,398 个 token【验，与书中 3.1.1 印出的数一致：源 S10 ch03】。
- 全语料压缩率为 2.42 bytes/token【验】。
- 专栏讨论平票时如何选 pair：代码里用 `max(..., key=(count, pair[0], pair[1]))`，旧写法 `counts.get` 留在注释里。

### 第 2 章　模型（基础篇）
【源 S10 ch02 + 验 代码】：
- 本章目标是理解 Transformer 并亲手实现 GPT-2，参照 GPT-2 原论文和 nanoGPT【源】。
- **Attention 当作"软字典"**：先用 Python dict，再用一个电影偏好的玩具例子：key 是（动作, 剧情, 喜剧）三维评分，value 是用户打分，query 是一部新电影。
- **公式与实现**：QKᵀ → softmax → 加权和。
- **缩放**：用 softmax 饱和的例子 `[100,200,300]` 说明问题；做 1 万次采样实验，看点积方差在除以 √d 前后的变化；严格证明放在附录 B。
- **结构与位置**：Transformer 结构；正弦位置编码和可学习位置编码。
- **因果掩码**：`torch.tril` 加 `masked_fill(-inf)`。
- **Value 矩阵低秩化**：拆成 `W_v` 到 key_dim，再接 `W_o`。
- **多头**：用 view / transpose 实现，并说明两种公式等价。
- **Transformer 块**：LayerNorm、GELU（tanh 近似）、FFN，pre-norm 加残差。
- **GPT-2**：
  - 结构：token embedding + 可学习 pos embedding，dropout 0.1，最后一层 LayerNorm，unembed；
  - **权重共享**：`embed.weight = unembed.weight`；
  - 用 std 0.02 初始化；
  - 提供 `save` / `load_from`。

### 第 3 章　训练（基础篇）
【源 S10 ch03 + 验 代码】，三段式：预训练 → SFT → RL。最后一步以算术题答对与否作为奖励。

**预训练**
- 用 `TokenDataset` 滑窗取样，DataLoader 打乱顺序。
- 超参：ctx 256，batch 32，AdamW lr 3e-4，20,000 iter，共约 1.64 亿 token，约等于把 2.68M token 的语料**重复看 61 遍**【验】。

**生成**
- 正文讲了温度、Top-k、Top-p。
- 但代码 `codebot/utils.generate` 只实现了温度：`temperature=0` 时贪心解码，没有 Top-k / Top-p【验】。

**SFT**
- 格式是 Alpaca 风格：`### Instruction:\n…\n\n### Response:\n…<|endoftext|>`。
- prompt 部分的 label 设为 −100，padding 用 id 0，对应 label 也是 −100。
- 训练 500 iter。
- 数据 `tiny_codes_sft.json` 共 1,092 条：
  - 其中 81 条恰好是 1–9 加 1–9 的全部个位数加法【验】；
  - 176 条回答里含有代码【验】。

**RL 理论**：轨迹，奖励与回报，策略梯度及其近似，基线（无偏性证明在附录 C），GRPO 用组内均值作基线，off-policy 下的重要性采样加 clipping。

**GRPO 实现**
- prompt 就是那 81 道 `i+j=`；奖励规则是回答中最后一个整数等于答案得 1，否则得 0。
- 每组 8 个样本，每批 32 个 prompt，每轮生成后更新 2 次。
- ε = 0.2，lr 7e-6，训练 500 iter，梯度裁剪 1.0。
- 每 10 iter 用贪心解码评测一次准确率。
- 优势函数 = r − 组内均值，**不除以 std**；loss 按样本数归一，而不是按 token 数归一。这与专栏讨论的 Dr. GRPO 的取舍相符【推】。
- 详细数值核对见 `cs336-crossref.md` §0-7。

### 第 II 部扉页（StoryBot）
【源 S10 part2】：主题从生成代码转到用自然语言创作。数据集是 TinyStories（Eldan et al., 2023），约 200 万篇由 GPT-4 写成的短故事，用语简单到儿童也能懂。

### 第 4 章　分词器（应用篇）
【源 S10 ch04 + 验 代码】本章逐步加速第 1 章的 BPE，输出结果不变，只是快得多。依次做了四件事：

1. **去重**：先把预分词结果统计成计数表，相同的片段只算一次。
2. **增量更新**：建一个 `pair_to_ids` 缓存，每次合并只更新受影响的序列。
3. **分块读文件**：`find_chunk_boundaries` 在 `<|endoftext|>` 处对齐块边界，每次向前读 4096 字节。写法与 CS336 的 `pretokenization_example.py` 同构【验 + 源 S28】。专栏讲预分词的内存问题，代码随之从 `findall` 改为生成器 `finditer`【推】。
4. **并行**：`multiprocessing.Pool` 配 64 个 chunk。

训练结果：StoryBot 的 vocab 为 10,000。

评估一节（4.5）在仓库里缺少对应文件【源 S12】。

编码也做了两步加速：
- "快速合并"：每轮只合并当前序列中优先级最高（rank 最小）的 pair，不再按顺序遍历全部 merge 规则。
- 分块并行编码，结果先写进缓存，最后用 `np.memmap` 写成 `uint16` 的 `.bin`。

### 第 5 章　模型（应用篇）
【源 S10 ch05 + 验 代码】：
- 在保留 GPT-2 骨架的前提下引入 RoPE、SwiGLU、RMSNorm 等现代改进。
- 书中说实现时参考了 **gpt-oss、nanochat、Stanford CS336**【源】。

具体改动：
- **RoPE**：相邻的奇偶维两两配对做旋转，θ=10000，预先缓存 cos/sin，只作用于 Q 和 K，在 fp32 下计算。
- **SwiGLU**：hidden = ⌊8/3·d⌋。
- **RMSNorm**：正式模型用的是 `nn.RMSNorm`；`03_rmsnorm.py` 是手写的教学版。
- **其他精简**：去掉 dropout，去掉 bias，取消权重共享，也不再有 pos_embed。
- **KV cache**：分 prefill 和 decode 两个阶段，只在 prefill 时加掩码，并比较开关缓存时的生成速度。

### 第 6 章　训练（应用篇）
【源 S10 ch06 + 验 代码】：

**优化器与调度**
- 从 SGD → Momentum → Adam → AdamW 讲下来，说明为什么选 AdamW；手写一个 `Optimizer` 子类。
- 学习率：warmup 之后用 **D2Z**（线性衰减到 0），并与余弦退火比较。

**混合精度**
- 演示 FP16 的下溢、上溢与 BF16 的对比。
- 用 `torch.autocast` 跑 BF16。

**预训练（改进版）**
- 数据随机采样自 `np.memmap`。
- 梯度裁剪 1.0；每 500 iter 在完整验证集上算一次 loss；BF16 autocast。
- 训练 40,000 iter，共 327,680,000 token；在第 500 和 5000 iter 各存一个中间 checkpoint。
- 模型 22.7M 参数【验】。

**生成与评测**
- 生成时以 `<|endoftext|>` 作为 prompt，走带 KV cache 的生成。
- **LLM-as-a-Judge**：用 `gpt-4o-mini` 给 500、5000、40000 iter 的 checkpoint 打分，coherence 和 grammar 各 1–5 分，需要 OpenAI API key。

**RLHF 与 DPO 理论**：Bradley–Terry 模型；RLHF 带 KL 正则；DPO 的闭式解（推导在附录 E）。

**DPO 实现**
- 偏好数据 144 条：同一开头，一个结局快乐（chosen），一个结局悲伤（rejected）【验】。
- 参数：β=0.1，lr 5e-6，batch 8，1000 iter，参考模型冻结。
- 评测：成对比较"哪个结局更快乐"，共 100 次，A/B 顺序随机打乱以消除位置偏差。

### 第 III 部扉页（WebBot）
【源 S10 part3】：
- 开头抛出问题：面对真实网页文本的噪声、风格和话题多样性，前两个 bot 的技术还管不管用？
- 书中明说：「このステージでは、詳細な実装手順は示しません。」（本阶段不给出详细的实现步骤），改为提示要点和方向。
- 数据集 OpenWebText 是为复现 GPT-2 的训练数据而从网络文章构建的。

### 第 7 章　分词器（挑战篇）
【源 S10 ch07 + 验 代码】：
- 语料 12 GB。直接复用 StoryBot 的 BPE，vocab 设为 50,000，接近 GPT-2 的 50,257。
- 书中提醒读者权衡：词表越大，单个 token 携带的信息越多，但 embedding 参数也越多。
- 训练"相当耗时"，所以后面几节讨论瓶颈定位、Rust/C++ 加速、新增特殊 token、其他分词器以及无分词器的方案。这几节具体讲什么，没有可见文本【推】。

### 第 8 章　模型（挑战篇）
【源 S10 ch08】：
- 模型比 StoryBot 更大，从 scaling law 讲起：loss 随参数量 N 和数据量 D 呈幂律下降，于是可以拿小模型的实验去估计大模型。
- 其余各节为 Flash Attention、GQA、MoE、torch.compile、上下文扩展，代码映射见 §5。
- WebBot 的配置【验】：
  - vocab 50,000，ctx 1024，d=768；
  - 12 层；12 个 query head，4 个 KV head；ff 2048；
  - θ=10000，RoPE、SwiGLU、RMSNorm，不用 bias，不共享权重；
  - 共 **152.3M** 参数。

### 第 9 章　训练（挑战篇）
【源 S10 ch09】：
- 一上来就点明 GPU 显存的约束，以及模型放不进单卡时需要多卡。
- 9.1 用"batch 8 × 累积 4 = batch 32"的例子讲梯度累积。

代码 `ch09/01_pretrain.py`【验】：
- DDP：由 `RANK` 环境变量触发，使用 NCCL；
- `torch.compile`；
- micro-batch 32 × 累积 4；
- lr 6e-4，warmup 500，100k iter，D2Z；
- 梯度裁剪 1.0，BF16 autocast（`device_type` 写死为 `'cuda'`）；
- wandb 记录日志。

`02_generate.py` 用这些 prompt 测试，温度 0.5：
- "In 1991, Linus Torvalds created"
- "Monday, Tuesday, Wednesday,"
- "Python was created by"
- "Machine learning is defined as"
- "The capital of Japan is"

### 附录 A–E
【源 S10 ch21a–ch25e】：
- **A**：用对照表比较 UTF-32 / 16 / 8 下 'A'、'あ'、'😁' 的编码，解释为什么"任何字符都能写成 0–255 的数列"。
- **B**：设 q、k 各分量独立、均值 0、方差 1，证明 q·k 的方差为 d，因此除以 √d 后方差为 1。
- **C**：证明加入只依赖状态的基线后，策略梯度仍然无偏。
- **D**：从零初始化造成的偏差出发，解释为什么要除以 1−βᵗ。
- **E**：推导 KL 正则目标下最优策略与奖励的关系（引入配分函数 Z）。

### おわりに（后记）
【源 S10 ch27aft】：
- 系列始于 2016 年，至今约 10 年。
- 明确说执笔时从 **Stanford CS336** 和 Karpathy 的 nanoGPT、nanochat 中得到了很多启发。
- 结尾以披头士作比，谈 AI 的这段时代。

---

## 4. 部 ↔ 章 ↔ 仓库目录 ↔ bot 对照

| 部 | 章 | 仓库 | 规模（数据 / vocab / ctx / 参数） | 来源 |
|---|---|---|---|---|
| — | まえがき、0章 | 无 | — | 【源 S1】 |
| I 基本編 | 1 トークナイザ、2 モデル、3 学習 | `ch01/`、`ch02/`、`ch03/`，包 `codebot/`（`tokenizer.py`、`model.py`、`utils.py` 及数据） | TinyCodes 6.49 MB / 1,000 / 256 / **11.1M** | 【源 S1、S10；验】 |
| II 応用編 | 4、5、6 | `ch04/`、`ch05/`、`ch06/`，包 `storybot/` | TinyStories 2.23 GB / 10,000 / 256 / **22.7M** | 【源 S1、S14；验】 |
| III 挑戦編 | 7、8、9 | `ch07/`、（无 `ch08/`）、`ch09/`，包 `webbot/`（只有 `model.py`、`utils.py`，分词器复用 `storybot/tokenizer.py`） | OWT 样本 11.92 GB / 50,000 / 1,024 / **152.3M** | 【源 S1、S14；验】 |
| — | 附录 A–E、おわりに | 无 | — | 【源 S1】 |

几点说明：
- **包与章节的关系【推】**：`codebot/` 等包里放的是章节代码整理后的"定稿版"，后续章节都从包里 import。依据是 1.7.2「これまでのコードの整理」，以及 `codebot/model.py` 与 `ch02/08`、`ch02/09`、`ch02/10` 的类基本一致：手写的 GELU 换成了 `nn.GELU`，手写的 LayerNorm 换成了 `nn.LayerNorm`，FFN 的构造参数也略有不同。
- **StoryBot 包已含第 5 章的 KV cache**：`storybot/model.py` 等于 `ch05/05_kvcache.py` 的模型部分，外加 prefill 阶段的 causal mask 修正（commit `8b3911f`）【验】。

### 文件 ↔ 节 速查（【推】，除非另有标注）
| 文件 | 节 | 文件 | 节 |
|---|---|---|---|
| ch01/01–09 | 1.1–1.9（一一对应；01 已由预览证实） | ch04/01–04 | 4.1–4.4 |
| ch02/01、02、03 | 2.1、2.2、2.3（01 已证实） | ch04/05_eval.py（缺失） | 4.5【源 S12】 |
| ch02/06、07、08、09、10 | 2.6、2.7、2.8、2.9、2.10 | ch04/06、07 | 4.6、4.7 |
| ch02/graph.py | 2.9.3 的 ReLU/GELU 图 | ch05/01–05 | 5.1–5.5 |
| ch03/01、02 | 3.1、3.2 | ch05/graph.py | 5.2.1 的激活函数图 |
| ch03/03_alpaca、03_sft | 3.3.2、3.3.5 | ch06/02、03、04、05、06 | 6.2、6.3、6.4、6.5、6.6.1 |
| ch03/04_chat | 3.4 | ch06/07_llm_judge | 6.6.3（文件编号与节号不一致） |
| ch03/09_grpo | 3.9 | ch06/08_dpo、09_llm_judge | 6.8.2、6.8.3 |
| ch07/01、02 | 7.1（01 已证实） | ch06/lr_graph.py | 6.3.2 的调度曲线图 |
| ch09/01、02 | 9.1、9.3、9.6、9.7 / 9.7 | webbot/model.py | 8.2、8.3（以及 9.7） |

---

## 5. 第 8 章为什么没有代码目录，它讲什么

**它讲什么**【源 S1、S10】：模型规模选择（scaling law），Flash Attention，GQA，MoE，torch.compile，上下文长度扩展。

**为什么没有 `ch08/`**【推，依据为 S10 part3 的明确表态和代码事实】：
- 第 III 部的方针就是只给提示和要点；
- 第 8 章里能落成代码的部分都直接写进了最终模型 `webbot/model.py`；
- `torch.compile` 属于训练脚本，所以放在 `ch09/01_pretrain.py` 里。

**具体对照**【验，grep 和读代码】：
- SDPA：`F.scaled_dot_product_attention(Q, K, V, is_causal=…, enable_gqa=True)`。训练时 `is_causal=True`。用 KV cache 时，prefill 阶段为 True，decode 阶段为 False。
- GQA：`W_k`、`W_v` 的输出维度是 `n_kv_head*D`（4 × 64），`W_q` 是 12 × 64。
- 不存在的部分：MoE 层、长上下文技巧（如 RoPE 缩放、YaRN）、梯度检查点、ZeRO、Muon，全仓库都没有实现。
- **教程可以补的空白**：这些主题在书中大概率只有概念讲解（第 8、9 章合计约 10 个虚拟页【源 S9】），正好是教程用 CS336 A2 素材补足的地方（见 `cs336-crossref.md`）。

---

## 6. 数据集与产物（来源、许可、规模）

| 数据 | 上游来源（许可） | 仓库或 HF 中的文件 | 规模 | 来源 |
|---|---|---|---|---|
| TinyCodes（CodeBot 预训练） | `flytech/python-codes-25k`（MIT），书中注明"Modified and used in the book" | `codebot/tiny_codes.txt`（已在 git 中，同时也在 HF） | 6,487,033 B（LF）；21,836 个 Python 小程序，以 21,835 个 `<|endoftext|>` 分隔；编码后 2,677,398 token | 【源 S14；验】。Windows 下 git 的 autocrlf 会把它变成 CRLF，文件变为 6,743,534 B（兄弟档案 `repo-ch01-tokenizer.md` 实测） |
| CodeBot SFT | 没有注明来源，【推】为作者自制 | `codebot/tiny_codes_sft.json` | 1,092 条 instruction/response，含 81 道个位数加法 | 【验】 |
| TinyStories V2-GPT4 | `roneneldan/TinyStories`（CDLA-Sharing-1.0）；论文 Eldan et al. 2023 | HF `storybot/tiny_stories_{train,valid}.txt`（由 `download.sh` 下载） | 训练集 2,227,753,162 B；验证集 22,502,601 B；训练集编码后 **541,229,223** token，验证集 5,465,873 token；书中说约 200 万篇 | 【源 S14、S10 part2；验 HF API】 |
| StoryBot DPO | 没有注明来源，【推】为作者自制 | `storybot/tiny_stories_dpo.json` | 144 条 prompt / chosen（快乐结局）/ rejected（悲伤结局） | 【验】 |
| OpenWebText 样本 | `stanford-cs336/owt-sample`（MIT），源自 `Skylion007/openwebtext`（MIT） | HF `webbot/owt_{train,valid}.txt` | 训练集 11,920,511,059 B（书中说 12 GB）；编码后训练集 **2,649,004,800** token，验证集 64,496,066 token | 【源 S14、S10 ch07；验 HF API】 |

**整体打包与许可**：HF 上整个 `koki0702/zero-llm-data` 因为含有 TinyStories，统一采用 **CDLA-Sharing-1.0**【源 S14】。原始文件名（如 `TinyStoriesV2-GPT4-train.txt`）在 HF 上已改为规范名（如 `tiny_stories_train.txt`）【源 S14】。

**预训练权重（HF）**【验 HF API】：
- 有：`codebot/model_{pretrain,sft,grpo}.pt`（各 44.5 MB）；`storybot/model_{pretrain,iter_500,iter_5000}.pt`（各 90.8 MB）；三个 `merge_rules.pkl`；各 `.bin`。
- **没有** `storybot/model_dpo.pt`（虽然 `storybot/download.sh` 的注释里列了它）。
- **没有任何 WebBot 权重**。

**下载脚本**：
- `codebot/download.sh` 默认只下载 `model_pretrain.pt`。
- `storybot/download.sh` 和 `webbot/download.sh` 默认只下载 `.txt`；`.bin`、`.pkl`、`.pt` 的下载行都被注释掉了【验】。
- 韩文版仓库额外提供了 PowerShell 版的 `download.ps1`【源 S25】，对用户的 Windows 环境很有用。

---

## 7. 模型配置与训练预算（全部摘自代码【验】）

| | CodeBot（ch03） | StoryBot（ch06） | WebBot（ch09） |
|---|---|---|---|
| 架构 | GPT-2：可学习 pos emb、LayerNorm、GELU、dropout 0.1、有 bias、权重共享 | RoPE、SwiGLU、`nn.RMSNorm`、无 bias、无 dropout、不共享权重，带 KV cache | 在 StoryBot 基础上加 GQA（4 个 KV head）、SDPA/Flash、KV cache 截断 |
| vocab / ctx / d / 层 / 头 / ff | 1000 / 256 / 384 / 6 / 6 / 1536 | 10000 / 256 / 512 / 4 / 16 / 1344 | 50000 / 1024 / 768 / 12 / 12（KV 4） / 2048 |
| 参数量 | 11,121,640 | 22,696,448 | 152,316,672 |
| 优化器与学习率 | `torch.optim.AdamW`，lr 3e-4，恒定 | AdamW，峰值 1e-3，warmup 200，D2Z | AdamW，峰值 6e-4，warmup 500，D2Z |
| batch × iter | 32 × 20,000 | 32 × 40,000 | 32 × 累积 4 × 100,000（**每张卡**） |
| 看过的 token | 163,840,000（语料约 61 遍） | **327,680,000**（与 CS336 A1 完全相同） | 每张卡 13,107,200,000（单卡相当于 OWT 约 4.9 遍；DDP 下按卡数线性增加，因为 `max_iters` 是写死的） |
| 精度 | fp32 | BF16 autocast | BF16 autocast，加 `torch.compile` |
| 后训练 | SFT 500 iter；GRPO 500 iter（lr 7e-6，G=8） | DPO 1000 iter（β 0.1，lr 5e-6，batch 8） | 无 |

---

## 8. GPU 要求与成本：书中和外界怎么说

### 8.1 书中的官方说法（前言，经 note.com 引用）【二手 S20】
| 部分 | 要求 |
|---|---|
| 第 I 部 | 只用 CPU 也能运行；有 GPU 可以缩短训练时间 |
| 第 II 部 | 推荐用 GPU；CPU 也能跑，但训练耗时 |
| 第 III 部 | **必须有 GPU**，最好是 **A100 / H100 级** |

前言还给出了获取 GPU 的途径：
- 免费：Google Colab、Kaggle Notebooks（有限制）；
- 更高性能的云 GPU：**RunPod**、**Lambda**、Vast.ai，书中说可以"比较便宜地"租到实例。

前言**没有给出具体价格**。

### 8.2 书中与 GPU 相关的其他线索
- 第 6 章有一篇专栏「GPU環境でBF16が使えない場合」，只看得到标题【源 S1】。
- 第 9 章开篇讲 GPU 显存约束和多卡训练【源 S10 ch09】。
- **书中给出的处理时间是在 RTX 4090 上测的**。这一点来自 Amazon 书评人 Kyle 2026-06-10 的评论："ColabのデフォルトのGPUは本書に記載されているRTX4090の処理時間の大体６〜８倍"【二手 S6】。具体数值没有拿到。

### 8.3 营销页面与作者
O'Reilly Japan 书页、Ohmsha 页面、出版社和作者的推文里，都**没有** GPU、时长或成本的数字【源 S1、S2、S15–S19】。

### 8.4 读者反馈【二手】
- Amazon 书评人 Kyle（2026-06-10）【S6】：
  - 代码在 2026 年的 Colab 上能跑，但需要改写成 notebook，最好再接上 Google Drive；
  - CodeBot 在免费额度内就能全部训练完；
  - StoryBot 及以后需要付费，或者把训练拆成几段，靠 checkpoint 加学习率调度接续。
- note.com Ei-chan（2026-07-29）【S21】：用 Google Colab 的 GPU 环境就足够跑。
- 第三方仓库 manaty226 在 AWS SageMaker 的 `ml.g5.xlarge`（A10G 24 GB）上原样跑了 ch03 和 ch06 的预训练脚本，但没有公布耗时【源 S30】。

### 8.5 来自 CS336 的参照数据（同一配置，同一份数据）
- CS336 A1 handout（v26.0.3，Spring 2026）【源 S27】：
  - 在 1 张 B200 上，StoryBot 这套配置跑完 327.68M token 约需 20–30 分钟；
  - 低资源方案是在 M4 Max 上跑 40.96M token，用 MPS 需 36 分钟，用 CPU 需 1 小时 22 分。
- cs336.stanford.edu【源 S26】列出了 2026-03-28 的单张 B200 价格：

| 平台 | 价格 |
|---|---|
| Modal | $6.25/h（每月有 $30 免费额度） |
| Lambda | $6.69/h |
| RunPod | $4.99/h |
| Nebius | $5.50/h |
| Together | $7.49/h（至少 8 卡） |

### 8.6 本机核查【验】
- 仓库的 `uv.lock` 在 Windows 上解析出的是 **`torch 2.9.1+cpu`**，所以 `torch.cuda.is_available()` 返回 False。
- 本机其实有 **RTX 3060 Ti 8 GB**（驱动 610.60）。
- 想在本地用 GPU，必须另装 CUDA 版的 wheel。该 GPU 是 Ampere 架构，**支持 BF16**【推】。

### 8.7 粗略估算（全部为【推】，用于排期，必须用实测 tokens/s 取代）
**公式**：训练 FLOPs ≈ 6·N_非嵌入·D + 6·L·ctx·d·D。

**WebBot**：
- 每 token 约 7.4×10⁸ FLOPs，乘以 13.1B token，约 **1×10¹⁹ FLOPs**。
- 按有效算力估算的单卡耗时：

| GPU | 假设有效算力 | 估计耗时 |
|---|---|---|
| H100 | 约 300 TFLOPS | 约 9 小时 |
| A100 | 约 95 TFLOPS | 约 1.2 天 |
| RTX 4090 | 约 50 TFLOPS | 约 2.2 天 |
| RTX 3060 Ti | — | 不现实：显存放不下 micro-batch 32×1024，而且要算 10 天以上 |

- 用 DDP 时，代码会让总 token 数随卡数线性增加；要保持预算不变，必须按卡数等比例减小 `max_iters` 或 `accumulation_steps`。

**StoryBot**：约 3.6×10¹⁶ FLOPs。这么小的模型，瓶颈在数据加载和 kernel 启动开销，FLOPs 估算会严重偏乐观。应以 CS336 的 B200 参照（20–30 分钟）和本机实测为准。

**Windows 上的限制**：
- NCCL 不支持 Windows，所以 ch09 的 DDP 需要 Linux（WSL2 或云端）；
- `torch.compile` 在原生 Windows 上依赖 Triton，支持有限。

---

## 9. 与 CS336 的交叉证据（摘要；详见 `cs336-crossref.md`）

**书中的明示**【源】：
- 后记写明从 CS336 和 nanoGPT、nanochat 中获益良多（S10 ch27aft）；
- 第 5 章说实现时参考了 gpt-oss、nanochat、CS336（S10 ch05）；
- 第 0 章参考文献列有 [Stanford CS336, 2025]（S10 ch28bib）。

**数据**【源 S14】：WebBot 的 OWT 直接取自 `stanford-cs336/owt-sample`。

**超参数**【验 + 源 S27 §7.2.1】：StoryBot 的配置是 vocab 10000、ctx 256、d 512、d_ff 1344、4 层、16 头、θ 10000、共 327,680,000 token，与 CS336 A1 的 TinyStories 基线**逐项相同**。

**代码**【验 + 源 S28】：`find_chunk_boundaries` 的结构与 CS336 的 `pretokenization_example.py` 相同，包括：
- 4096 字节向前读取；
- 读到 EOF 时把边界设为文件末尾；
- 最后用 `sorted(set(...))` 去重。

**主题对应**【推，依据 S26 列出的作业和讲座名】：

| DLFS6 | CS336 |
|---|---|
| 第 1、4 章 | A1 §2（BPE） |
| 第 2、5 章 | A1 §3（Transformer LM，含 RoPE、SwiGLU、RMSNorm） |
| 第 6 章的 AdamW 与学习率调度 | A1 §4–5 |
| 第 8 章的 Flash Attention、GQA、MoE | A2（Triton 实现 FlashAttention2），Lecture 4（Attention alternatives & MoE） |
| 8.1 节 scaling law | A3，Lectures 9、11 |
| 第 9 章的 DDP 与 ZeRO | A2（分布式训练、优化器状态切分），Lectures 7–8 |
| 3.5–3.9 节 GRPO | A5（SFT 加 RL 做数学推理），Lecture 16（RLVR） |
| 6.7–6.8 节 DPO | A5 的可选部分，Lecture 15 |
| 无对应 | A4（Common Crawl 过滤与去重） |

**主要差异**【推】：
- CS336 禁止使用 `torch.nn` 和 `torch.optim` 中的现成组件。DLFS6 的正式脚本却直接用 `nn.Linear`、`nn.Embedding`、`nn.RMSNorm`、`torch.optim.AdamW`；手写版本只作为教学示例出现，例如 `ch06/02_adamw.py`、`ch05/03_rmsnorm.py`。
- 硬件：CS336 以 B200 为前提；DLFS6 的前言对第 III 部建议 A100/H100。

---

## 10. 斋藤的组织手法（供教程设计参考）

1. **同一节奏，规模递增**（前言【源 S10】）：三个部分都按 トークナイザ → モデル → 学習 推进，规模逐级放大：

   | 维度 | CodeBot | StoryBot | WebBot |
   |---|---|---|---|
   | 数据 | 6 MB | 2 GB | 12 GB |
   | 参数 | 11M | 23M | 152M |
   | vocab | 1k | 10k | 50k |
   | ctx | 256 | 256 | 1024 |

   书评人 Ei-chan 称之为螺旋式课程【二手 S21】。
2. **部扉三件套**【源 S10 part1、part2、part3】：先展示做好的 bot（对话或生成样例），再给一张开发流程表，最后介绍数据集。
3. **先朴素后改进**：
   - 分词：字符 → 字节 → BPE；
   - BPE 训练改进 ①→④，编码改进 ①→②；
   - 模型：GPT-2 → 现代 GPT。

   每做一步改进，都以「動作確認」收尾。第 1–6 章共有 **11 个**这样的标题【验 §14】。
4. **先类比，后数学**：例如先用电影评分的玩具字典讲"软字典"，再给 QKᵀ 公式【源 S10 ch02】。
5. **理论放进附录和专栏**：
   - 证明集中在附录 A–E；
   - 枝节话题放进 6 篇专栏，主线不被打断【源 S1、S8】。
6. **文件名编号对应节号**，正文以文件路径作代码标题；整理过的定稿代码收进各 bot 的包里【源 S10；推】。
7. **允许跳过昂贵步骤**：
   - 仓库预置了 `tiny_codes.bin`，跳过第 1 章也能做第 3 章【源 S10 ch03】；
   - HF 提供预训练权重和编码好的 `.bin`【源 S14】。
8. **第 III 部切换为"提示模式"**：不再有三级小节，只讲要点，留给读者自己动手【源 S1、S10 part3】。

---

## 11. 适合教程重点讲解的"坑"（可直接做成检查点）

1. **`ch04/05_eval.py` 缺失**（书中第 197–199 页引用了它）。issue #7 给出了从正文整理出的代码【源 S12】。
2. **README 里的文件名写错**：README 写 `python ch02/10_gpt.py`，实际文件是 `ch02/10_gpt2.py`【验】。韩文版 README 也照抄了这个错误【源 S25】。
3. **StoryBot 的 KV cache 不截断**【验 §14】：
   - 实测：`max_context_len=8` 时连续喂 12 个 token，StoryBot 的 `k_cache` 长到 12，WebBot 的截断回 8。
   - 后果【推】：StoryBot 一旦生成超过 256 个 token（`ch06/06_generate.py` 设的是 `max_new_tokens=300`），RoPE 的 offset 会被钳在末端位置，后面的 token 位置信息失真。
   - 缓解因素：TinyStories 的故事通常在 256 token 内就生成 `<|endoftext|>` 而停止，所以实际影响不大。
4. **GRPO 用训练题评测**【验】：GRPO 的 81 道题正是 SFT 数据里的那 81 道加法，而且训练和评测用同一组题。所以第 3 章报告的准确率是分布内的（同时也是训练集上的）结果。读者在 issue #2 里报告过约 71.6% 的准确率，并讨论了 loss 很小而准确率不升的问题【源 S13】。
5. **GRPO 的速度没有上限保护**【验 + 推】：`generate_group` 调用 `generate()` 时没有传 `max_new_tokens`，所以用的是默认值 1000，而 CodeBot 的生成又没有 KV cache。整体速度取决于 SFT 模型能否及时输出 `<|endoftext|>`。
6. **HF 上缺少两类权重**【验】：没有 `model_dpo.pt`，也没有 WebBot 权重。因此 `ch06/09_llm_judge.py` 之前必须先跑 `ch06/08_dpo.py`。
7. **LLM-as-a-Judge 的依赖**：需要 OpenAI API key（脚本里写的是 `api_key="your_api_key_here"`），还要装可选依赖 `openai`（`pip install .[full]` 中的 full 组）【验】。
8. **字体问题**：`ch06/lr_graph.py` 写死了 macOS 字体 `'Hiragino Sans'`，Windows 需要改用注释里的 `'Yu Gothic'`【验】。
9. **ch09 只能在 CUDA 上跑**：`autocast(device_type='cuda')` 写死，并且用 NCCL 通信【验】。
10. **两处已修好的旧问题**【验，现行代码里已经没有】：issue #4 提到的非法字符 `ß`；`ch06` 的 `y.numel()` 修正（commit `1760066`）。
11. **RMSNorm 的参数名**：`ch05/03_rmsnorm.py` 的 `__init__(self, x)` 参数名容易误导，PR #6 建议改成 `embed_dim`【源 S13】。

---

## 12. 待解决或无法核实的事项

- **书中各训练步骤的实测时间**（RTX 4090）以及「GPU環境でBF16が使えない場合」专栏的具体内容：需要书本身或 O'Reilly Learning 订阅才能看到。
- **第 7 章 7.2–7.6 节和第 8、9 章各节的具体论述**：预览只到每章开头。
- **公开审稿版原稿**：Notion 页面已不公开【源 S31】。
- **中文版的出版计划**：未见公告。

---

## 13. 来源（URL）

| 编号 | 来源 | URL |
|---|---|---|
| S1 | O'Reilly Japan 书页（完整目次、专栏目次、发行日、页数、ISBN、定价） | https://www.oreilly.co.jp/books/9784814401611/ |
| S2 | Ohmsha（384 頁、2026/06/03、A5、正誤表なし） | https://www.ohmsha.co.jp/book/9784814401611/ |
| S3 | 版元ドットコム | https://www.hanmoto.com/bd/isbn/9784814401611 |
| S4 | 国立国会图书馆（NDL Search） | https://ndlsearch.ndl.go.jp/books/R100000002-I034727050 |
| S5 | 纪伊国屋 | https://www.kinokuniya.co.jp/f/dsg-01-9784814401611 |
| S6 | Amazon.co.jp（商品详情；书评：Kyle 2026-06-10、他人 2026-06-04、おくぼん 2026-07-06） | https://www.amazon.co.jp/dp/4814401612 |
| S7 | O'Reilly Learning API：元数据 | https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9784814401611/ |
| S8 | O'Reilly Learning API：目录 | https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9784814401611/table-of-contents/ |
| S9 | O'Reilly Learning API：各章虚拟页 | https://learning.oreilly.com/api/v2/epub-chapters/?epub_identifier=urn:orm:book:9784814401611 |
| S10 | O'Reilly Learning 公开预览片段（每章开头约 2–4 KB） | 见表下说明 |
| S11 | 官方仓库 | https://github.com/oreilly-japan/deep-learning-from-scratch-6 |
| S12 | issue #7：`ch04/05_eval.py` 缺失，第 197–199 页 | https://github.com/oreilly-japan/deep-learning-from-scratch-6/issues/7 |
| S13 | 其他 issue 与 PR | 见表下说明 |
| S14 | HF 数据集卡片与文件树 | https://huggingface.co/datasets/koki0702/zero-llm-data |
| S15 | 公开审稿开始（2025-12-20） | https://x.com/SaitohKoki/status/2002192283840709012 |
| S16 | 审稿截止提醒（2026-01-17） | https://x.com/SaitohKoki/status/2012506044522762310 |
| S17 | 审稿人名单（2026-01-27） | https://x.com/SaitohKoki/status/2016073555340099678 |
| S18 | 预售信息公开（2025-12-22） | https://x.com/oreilly_japan/status/2002964345115083063 |
| S19 | 发售日提前至 6/3（2026-04-23） | https://x.com/oreilly_japan/status/2047244091780653495 |
| S20 | note.com ykagata，2026-10-08，引用了前言和后记 | https://note.com/_heartyfluid/n/nc73f03d4b18c |
| S21 | note.com Ei-chan，2026-07-29 | https://note.com/ef_english_diary/n/nf918a5470d4a |
| S22 | 北海道情報大学図書館书评 | https://hiulibrary.do-johodai.ac.jp/review/e-books/2305/ |
| S23 | 豆瓣「深度学习入门6」（日文原版条目） | https://book.douban.com/subject/38372517/ |
| S24 | 韩文版 | 见表下说明 |
| S25 | 韩文版仓库与勘误 | https://github.com/WegraLee/deep-learning-from-scratch-6 ；https://docs.google.com/document/d/1OmaYaKZPlcHpygzp61qmJUTe5tKRceMaMJRxx9H-k9s |
| S26 | CS336 课程主页（Spring 2026：作业、讲座、B200 价格） | https://cs336.stanford.edu/ |
| S27 | CS336 A1 仓库与 handout（v26.0.3） | https://github.com/stanford-cs336/assignment1-basics |
| S28 | CS336 `pretokenization_example.py` | https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py |
| S29 | 上游数据集 | 见表下说明 |
| S30 | 第三方运行记录 | https://github.com/manaty226/deep-learning-from-scratch-6-leaning-by-aws ；https://github.com/okaduki/deep-learning-from-scratch-6-rust |
| S31 | Notion 公开审稿页（经 `getPublicPageData` 查得 `publicAccessRole: "none"`） | https://www.notion.so/Deep-Learning-LLM-f5ebc6e38c6d4b8cbcec17a22e806faa |
| S32 | 官方仓库提交历史（共 52 个 commit） | https://api.github.com/repos/oreilly-japan/deep-learning-from-scratch-6/commits |

**S10 的 URL 规则**：`https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9784814401611/files/<name>.xhtml`，其中 `<name>` 取以下之一：

| 类别 | `<name>` |
|---|---|
| 前言与正文 | ch00f、ch00g、part1、ch01、ch02、ch03、part2、ch04、ch05、ch06、part3、ch07、ch08、ch09 |
| 附录 | ch21a、ch22b、ch23c、ch24d、ch25e |
| 卷末 | ch27aft、ch28bib、author、colophon |

**S13 的链接**（前缀均为 `https://github.com/oreilly-japan/deep-learning-from-scratch-6`）：

| 条目 | 路径 |
|---|---|
| issue #1（求英文版） | `/issues/1` |
| issue #2（GRPO 讨论） | `/issues/2` |
| issue #3（BPE 提速建议） | `/issues/3` |
| issue #4（`ß` 字符 bug） | `/issues/4` |
| PR #6（RMSNorm 参数名） | `/pull/6` |

**S24 韩文版**：

| 站点 | URL |
|---|---|
| Hanbit | https://www.hanbit.co.kr/books/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-6?code=B7716758239 |
| Kyobo | https://product.kyobobook.co.kr/detail/S000220996795 |
| 청년개발자신문（2026-08-21） | https://www.devtimes.co.kr/news/514959 |

**S29 上游数据集**：

| 数据集 | URL |
|---|---|
| python-codes-25k | https://huggingface.co/datasets/flytech/python-codes-25k |
| TinyStories | https://huggingface.co/datasets/roneneldan/TinyStories |
| owt-sample | https://huggingface.co/datasets/stanford-cs336/owt-sample |
| openwebtext | https://huggingface.co/datasets/Skylion007/openwebtext |

---

## 14. 最小验证命令（Git Bash；Python 一律经 uv 运行）

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
# (1) 从出版社页面抽出官方目次 → 应有 280 行，从「目次」到「関連書籍」
curl -sSL https://www.oreilly.co.jp/books/9784814401611/ | sed -e 's/<[^>]*>/\n/g' | sed 's/^[[:space:]]*//' | grep -v '^$' | sed -n '/^目次$/,/^関連書籍$/p' | head -20
# (2) 电子版各章虚拟页 → ch02=50、ch07=2、ch08=4、ch09=6；合计 224
curl -sSL "https://learning.oreilly.com/api/v2/epub-chapters/?epub_identifier=urn:orm:book:9784814401611&limit=100" | uv run --no-project python -c "import json,sys;d=json.load(sys.stdin);print([(c['title'].strip()[:12],c['virtual_pages']) for c in d['results']]);print(sum(c['virtual_pages'] for c in d['results']))"
# (3) 第 III 部"不给详细步骤"的原句
curl -sSL https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9784814401611/files/part3.xhtml | grep -o '詳細な実装手順は示しません'
# (4) HF 文件大小 → owt_train.bin 5298009600 B = 2,649,004,800 个 uint16 token
curl -sSL "https://huggingface.co/api/datasets/koki0702/zero-llm-data/tree/main?recursive=true" | uv run --no-project python -c "import json,sys;[print(e['path'],e.get('lfs',{}).get('size',e.get('size'))) for e in json.load(sys.stdin) if e['type']=='file']"
# (5) ISBN 校验位 → True 9784814401611
uv run --no-project python -c "i='4814401612';s=sum((10-k)*int(c) for k,c in enumerate(i[:9]));c='978'+i[:9];t=sum(int(x)*(1 if k%2==0 else 3) for k,x in enumerate(c));print((11-s%11)%11==int(i[-1]),c+str((10-t%10)%10))"
# (6) 三个 bot 的参数量 → 11,121,640 / 22,696,448 / 152,316,672（使用仓库自带 venv，不 sync）
cd /c/Users/chenf/Projects/_refs/dlfs6 && uv run --no-sync --frozen python -c "
import torch;from codebot.model import GPT as A;from storybot.model import GPT as B;from webbot.model import GPT as C
for m in [A(1000,256,384,6,6,1536,0.1),B(10000,256,512,16,4,1344),C(50000,1024,768,12,4,12,2048)]:print(sum(p.numel() for p in m.parameters()))"
# (7) StoryBot 的 KV cache 不截断，WebBot 截断 → storybot 12, webbot 8
uv run --no-sync --frozen python -c "
import torch;from storybot.model import GPT as S;from webbot.model import GPT as W
for m in [S(100,8,32,4,2,64),W(100,8,32,4,2,2,64)]:
  m.eval();m.clear_cache();x=torch.tensor([[1]])
  with torch.no_grad():
    for _ in range(12): x=m(x,use_cache=True)[:,-1:].argmax(-1)
  print(m.blocks[0].attn.k_cache.size(2))"
# (8) SFT 中的 81 道个位数加法 = GRPO 的全部题目 → 81
uv run --no-sync --frozen python -c "import json,re;d=json.load(open('codebot/tiny_codes_sft.json',encoding='utf-8'));print(sum(1 for x in d if re.fullmatch(r'\d\+\d=',x['instruction'])))"
# (9) 第 1–6 章的「動作確認」小节数 → 11（对 (1) 抽出的目次执行）
#     ... | grep -c '動作確認'
```
