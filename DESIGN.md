# 设计文档：《从零构建大语言模型》互动可验证教程

> 本文件是全书的总设计。写任何一章之前先读它和 `STYLE.md`。
> 研究材料都在 `research/`（`private/` 下是用户的过往会话，不入库）。

## 1. 目标与边界

- **来源**：O'Reilly Japan《ゼロから作る Deep Learning ❻ — LLM編》（斋藤康毅，2026-06）的全部源代码，
  以 git submodule 固定在 `third_party/dlfs6`（commit `c9b6e2e`）。原书正文我们看不到，
  章节结构取自官方目录（`research/dlfs6-toc.md`），内容由代码“反向生成”。
- **读者**：有工程背景、想把 LLM 从分词器到 RL 后训练亲手走一遍的人。用户本人是 ML/编译器工程师，
  中英双语，偏好证据、可运行的验证和“这个数字从哪来”（见 `research/private/confusions.md` 的学习风格）。
- **语言**：简体中文正文，技术术语保留英文（token、Attention、softmax、KV cache、GRPO…）。
- **三条硬要求**：
  1. 行文自然流畅，组织方式学斋藤康毅（`research/style-*.md`）。
  2. **每一个技术断言都可最小化验证**：没有验证的断言不发布（见 §4）。
  3. GPU 部分真的能跑：本机 RTX 3060 Ti、Modal、RunPod、Lambda 都给出命令、耗时和成本。
- **不做**：翻译或复述原书正文（我们没有，也不该有）；大段引用受版权保护的书籍文字。

## 2. 全书结构：同样的节奏，不同的规模

原书最重要的组织原则：**分词器 → 模型 → 训练**这个节奏重复三遍，规模逐级放大
（「同じリズムを、異なるスケールで」）。我们沿用它，并把这条主线放在首页和每部开头。

| 部 | 机器人 | 数据 | 词表 | 上下文 | 参数量 | 训练 token | 硬件与预训练用时 |
|---|---|---|---|---|---|---|---|
| I 基础篇 | CodeBot | TinyCodes 6.5 MB | 1,000 | 256 | 11.1M | 1.64 亿 | 3060 Ti 实测 42 分钟；CPU 上至少约 23 小时 |
| II 应用篇 | StoryBot | TinyStories 2.2 GB | 10,000 | 256 | 22.7M | 3.28 亿 | H100 实测 16 分钟；3060 Ti 估算约 1.8 小时 |
| III 挑战篇 | WebBot | OpenWebText 11.9 GB | 50,000 | 1,024 | 152.3M | 131 亿/卡 | 1 张 H100 约 10.4 小时（200 步探针外推），约 $41–43 |

（起初这些数字来自 `research/repo-*.md` 与 `research/tech-frontend-gpu.md` 的实测或推算；现在都以页面上的验证为准：
参数量与 token 数见第 0 章验证 0.5-b、0.5-c；用时来自 `web/data/runs/` 的运行记录 `codebot-3060ti`、`storybot-h100`、
`webbot-probe`，CPU 用时见验证 0.7-b，3060 Ti 上 StoryBot 的估算见 GPU 手册 G.1 与验证 G.2-g；WebBot 的外推见 §8。
写进正文的数字都必须给出验证或溯源。）

### 页面清单（`content/*.md`，按 `order` 排序）

| order | slug | 标题 | 对应代码 | 主要来源 |
|---|---|---|---|---|
| 0 | index | 封面（书名实时分词）、如何使用这本教程、目录 | — | — |
| 10 | ch00 | 第0章 LLM 是如何炼成的 | 全部 | dlfs6-toc |
| 15 | part1 | 第 I 部 基础篇：制作 CodeBot | codebot/ | dlfs6-toc |
| 20 | ch01 | 第1章 分词器【基础篇】：把文字变成数字 | ch01/, codebot/tokenizer.py | repo-ch01 |
| 30 | ch02 | 第2章 模型【基础篇】：从软字典到 GPT-2 | ch02/, codebot/model.py | repo-ch02 |
| 40 | ch03 | 第3章 训练【基础篇】：预训练、SFT 与 GRPO | ch03/, codebot/utils.py | repo-ch03 |
| 45 | part2 | 第 II 部 应用篇：制作 StoryBot | storybot/ | dlfs6-toc |
| 50 | ch04 | 第4章 分词器【应用篇】：让 BPE 快起来 | ch04/, storybot/tokenizer.py | repo-ch04 |
| 60 | ch05 | 第5章 模型【应用篇】：现代 Transformer 的零件 | ch05/, storybot/model.py | repo-ch05 |
| 70 | ch06 | 第6章 训练【应用篇】：AdamW、混合精度与 DPO | ch06/, storybot/utils.py | repo-ch06-ch09 |
| 75 | part3 | 第 III 部 挑战篇：制作 WebBot | webbot/ | dlfs6-toc |
| 80 | ch07 | 第7章 分词器【挑战篇】：WebBot 的分词器 | ch07/ | repo-ch01（ch07 部分） |
| 90 | ch08 | 第8章 模型【挑战篇】：规模、注意力加速与 MoE | webbot/model.py | repo-ch05, repo-ch06-ch09 |
| 100 | ch09 | 第9章 训练【挑战篇】：多卡训练 WebBot | ch09/, webbot/utils.py | repo-ch06-ch09 |
| 110–150 | appendix-a…e | 附录 A UTF-8 编码的原理 / B Attention 缩放效果的证明 / C 带基线策略梯度的无偏性证明 / D Adam 的偏差校正 / E DPO 最优解的推导 | — | dlfs6-toc |
| 160 | playground | CodeBot 工作台：一个 prompt 的一生 | codebot/ | tech-frontend-gpu |
| 165 | train-in-browser | 在浏览器里从零训练一个迷你 GPT | codebot/model.py（同一个类，缩小） | — |
| 170 | gpu | GPU 实战手册：本机、Modal、RunPod 与 Lambda | gpu/ | tech-frontend-gpu |
| 180 | cs336 | CS336 对照总表 | — | cs336-crossref |
| 190 | glossary | 术语与命名对照 | — | confusions, cs336-crossref |
| 200 | errata | 原书代码的已知问题与修复 | — | 全部 repo-* 的 gotchas |
| 999 | kitchen | 组件样板间（`hidden: true`，不进目录；给作者看的组件与提示框用法） | — | — |

一共 26 个页面。各章的小节**严格沿用官方目录的编号与顺序**（`research/dlfs6-toc.md`）。官方目录里的 6 篇专栏（コラム）
都已写成 `::: column`：BPE 的平票处理（1.3）、Attention 的逐行计算（2.2）、Transformer 与句子的概率计算（2.4）、
GRPO 与 Dr. GRPO（3.7）、预分词的内存效率（4.3）、GPU 不支持 BF16 时怎么办（6.4）。另有两篇是本书加的：
怎样读强化学习的训练曲线（3.9）、为什么 Muon 不用在 embedding 上（9.5）。
第 III 部原书只给提示不给实现；我们照样按小节写，但每一节都要配可运行的最小验证（见 §6）。

## 3. 站点架构

```
content/*.md          章节源文件（Markdown + 自定义指令，见 STYLE.md §3）
web/                  模板、样式、运行时、交互组件与数据（构建时复制到输出目录，默认 site/）
  template.html       页面骨架
  css/site.css        设计令牌与全部样式（共享，章节作者不要改）
  js/runtime.js       主题/目录/KaTeX/组件挂载/验证单元
  js/pyworker.js      Pyodide Web Worker
  cell_harness.py     验证单元的执行器（浏览器与构建时共用同一份）
  widgets/_lib.js     DOM 工具（共享）
  widgets/_plot.js    SVG 图表：折线、柱状、热力图（共享）
  widgets/_bpe.js     浏览器里的 BPE，与原书的 Python 分词器逐 id 一致（共享库，没有 mount）
  widgets/_gpt.js     CodeBot 的 GPT 前向，CodeBot 工作台用（与 PyTorch 对拍）
  widgets/_tinygpt.js 能在标签页里训练的字符级 GPT，浏览器内训练页用
  widgets/<slug>-<name>.js  各交互组件，导出 mount(root, config)
  data/<slug>/        验证单元与组件读取的数据；data/runs/<run>/ 是 GPU 运行记录
tools/build.py        构建：Markdown → HTML，执行全部验证单元（任何一条失败即构建失败；全量构建时站内链接或锚点失效也失败）
tools/run_cells.py    在 Python 3.14 + NumPy 2.4.6（Pyodide 镜像）中执行验证单元
tools/check_pages.mjs 无头浏览器检查（Playwright）：报错、横向溢出、组件越界、KaTeX 错误，全部验证单元在真 Pyodide 中盖章；可截图
tools/fetch_data.py   下载官方 checkpoint 与数据，放进 third_party/dlfs6/<bot>/
tools/export_*.py     导出工作台的 int8 权重（export_codebot）与浏览器用的分词器 JSON（export_tokenizers）
tests/js/             组件里的 JS 实现（BPE、前向计算、训练器等）的 node --test 测试，与 Python/PyTorch 的参考结果对拍
verify/               本地 pytest（需要 PyTorch 的断言），直接导入 third_party/dlfs6 的代码；以 _ 开头的是辅助模块和生成 web/data 记录的脚本
gpu/                  GPU 运行：run_local.py（本机、RunPod、Lambda）、modal_app.py（Modal）、jobs.py（作业与补丁）、costs.py（估算）；运行报告写到 web/data/runs/
site/ build/ runs/    构建产物、预览构建、本机运行的工作目录（都不入库，见 .gitignore；部署方式待用户决定）
```

- 浏览器内 Python：**Pyodide 314.0.7**（Python 3.14.2，numpy 2.4.6，regex，matplotlib）。
  必须从 `https://cdn.jsdelivr.net/pyodide/v314.0.7/full/` 加载（npm 路径取不到包）。
- 构建时的验证单元在同版本的镜像环境里执行，所以页面上嵌入的“构建时输出”和浏览器现场输出逐字一致。
  和浏览器一样，一页的单元共享一个命名空间、按顺序执行，工作目录里只有 `files=` 声明过的文件，漏声明的文件读不到。
- 本地 PyTorch：`uv sync`；Windows 上通过 cu130 索引装 CUDA 版 torch（见 `pyproject.toml`）。

## 4. 验证体系（全书的灵魂）

**规则：一个技术断言，要么有 A 层验证，要么有 B 层验证，否则不写。**

- **A 层（浏览器，毫秒级）**：` ```verify ` 单元，NumPy-only，用 `assert` 把结论写死。
  构建时执行，失败则整站构建失败（借鉴 CS336 的可执行讲义：发布出来的每条断言都成立过）。
- **B 层（本地或云端）**：
  - ` ```localcheck verify/test_chNN.py::test_xxx ` 指向 pytest 测试，页面展示测试源码和命令；
  - GPU 运行报告：`web/data/runs/<run-id>/report.json` 记下教程与原书的 commit、主机、GPU、Python/torch/CUDA 版本，
    每个脚本的开始时间、墙钟时间、退出码、最终指标、日志末尾和打过的补丁；从别的运行接过 checkpoint 时记下它的
    sha256；在 Modal 上还记下计费。同一目录里还有从 tqdm 解析的 `<脚本>.metrics.jsonl`、脚本保存的图、生成脚本的
    输出 `<脚本>.output.txt`。运行记录列在 `web/data/runs/index.json` 里，页面上的训练曲线和样例只从这里读。
- **外部事实**（CS336 的要求、论文结论、价格）：给 URL，并注明查证日期（2026-10）。
- **原书注释即断言**：DLFS6 有 38 处写在注释里的期望输出（35 行 `print(...)  # 期望值`，加上 `ch03/03_alpaca.py`
  里 3 个写在下方的注释块），`verify/test_book_comments.py` 把它们全部改写成断言运行：30 处逐字一致，7 处只是写法不同
  （repr、defaultdict、torch.Size），1 处是过期的注释（勘误页 Z.2，验证 Z.2-a）。研究阶段只数到 31 条，其中 29 条直接通过。
- **代码清单一律从源文件抽取**（` ```listing path#Lx-Ly `），不手抄，构建时校验行号范围。
- **“这个数字从哪来”**：每个训练脚本的超参数都要有出处表（值、来源、改了会怎样）。

## 5. 视觉与交互

- 设计令牌在 `web/css/site.css` 的 `:root`：冷白纸（`--paper`）、靛蓝墨（`--ink`/`--ai`）、
  朱色（`--shu`：**实心的朱色只用在验证通过的印章上**；报错文字和少数组件里的标记用朱色的字、线或淡底）、
  金茶色（`--kincha`，输出不一致）。
  字体：Source Serif 4 + Noto Serif SC 正文，JetBrains Mono 代码。
- **唯一的醒目元素是验证印章**：每个验证单元一枚“验”字印章（宽屏在右侧页边，1180 px 以下放在单元下方），
  点击即在浏览器里运行，输出与构建时一致就盖上朱印。其余一切保持安静、书卷气。
- 图编号“图 章-序号”由构建自动生成；交互图的说明文字写结论，不写“如图所示”。
- 组件契约：`export function mount(root, config)`，只依赖共享库（`_lib.js`、`_plot.js`，以及 `_bpe.js`、`_gpt.js`、
  `_tinygpt.js`）和本页的辅助模块（如 `ch03-common.js`），不引入外部库；颜色只用 CSS 变量，
  支持明暗主题与窄屏（375px），键盘可操作，`prefers-reduced-motion` 时不做动画。

## 6. 各章要点（写作时以官方目录为纲，以对应 dossier 为料）

每章都要覆盖：官方小节、该章脚本的全部关键代码、dossier 里的可验证事实、易错点（含已验证的 bug）、
CS336 对照、用户过往困惑（`research/private/confusions.md` 中标注到本章的条目）、跨书锚点
（“如果你读过《深度学习入门4》9.3 节……”，精确到节，见 `research/style-*.md` 的 concept anchors）。

- **ch00**：LLM 的三段流水线（预训练 → SFT → RL）；三台机器人与规模表（可复算的 6ND）；
  “不变的内核”（next-token 预测 + 交叉熵、Attention、策略梯度）；读者路线与硬件需求；
  命名对照的预告（embed_dim = hidden_size = d_model = n_embd，困惑 TRN-04）。
- **part1/2/3**：开发流程、数据集介绍（来源与许可）、本部终点预览（先给结果）。
- **ch01**：字符 → 字节 → BPE 训练（平票处理专栏，含“先出现者胜 / 比 id / 比字节串（CS336）”三种规则的分叉实验）
  → 编解码（`errors="replace"` 的由来）→ 特殊 token → 预分词（GPT-2 正则在代码上的怪现象）
  → 在 TinyCodes 上训练（vocab 1000 = 256 + 743 + 1）→ 评估（压缩率 2.12 vs 全语料 2.42，与 tiktoken 对比）
  → 编码成 uint16 .bin。组件：BPE 单步训练器、预分词查看器、压缩率曲线、真实 CodeBot 分词器。
- **ch02**：软字典 → 公式 → 缩放（Var(q·k)≈d，附录 B）→ Transformer 结构与句子概率（专栏）→ 位置编码
  → 因果掩码（tril 的半开区间、整行 -inf 得 NaN，困惑 TENSOR-01）→ Value 低秩化 → 多头（形状追踪，
  `transpose` 后不能直接 `view`）→ LayerNorm/GELU（erf 与 tanh 版差 4.7e-4）→ GPT-2 组装
  （参数量 11,121,640，权重共享的重复计数陷阱，初始 loss ≈ ln V + σ²/2）。
- **ch03**：预训练（数据窗口；每个 token 平均当 61 次预测目标，按 DataLoader 的 epoch 算还不到 1/4 个；没有验证集）
  → 生成（温度、Top-k、Top-p；T 极小得 NaN）
  → SFT（Alpaca 模板、-100 掩码；81.5% 的位置是 padding，91.5% 的位置不产生 loss）→ 对话
  → **强化学习理论（3.5–3.8）要写得最扎实**：
  从 REINFORCE 到基线、到 critic、到组均值（衔接《深度学习入门4》第9章）；离策略与裁剪；GRPO 实现。
  必须正面回应困惑 RL-01…RL-09、TRN-01、EVAL-01/02：G=1 与零方差组无梯度、P(无信号)=p^G+(1-p)^G、
  pg_loss≈0 不是进度条、指标 Rosetta 表、G=2 等于 RLOO 的一半、训练准确率不是留出评估。
  已验证的 bug：GRPO 更新时策略处于 train 模式（dropout）。只在更新前加一行 `model.eval()`：CPU 上 60 轮的对照里，
  贪心解码的准确率 70.4% → 95.1%（种子 0；三个种子是 70%–74% 对 94%–95%，验证 3.9-h）；完整规模在 3060 Ti 上跑满 500 轮，
  原样脚本是 86.4%，加了这一行是 100%，第 51 轮已到 98.8%，第 131 轮起一直是 100%（运行记录 `codebot-3060ti` 与
  `codebot-grpo-evalfix-3060ti`，验证 3.9-n）。
- **ch04**：去重 → 增量更新 → 分块（chunk 边界对齐特殊 token；Windows CRLF 陷阱）→ 并行（Amdahl）
  → 评估（原书引用的 05_eval.py 仓库缺失，issue #7）→ 编码加速（按 rank 合并）→ 大规模编码（memmap）。
  每一步都要证明“结果与朴素版逐 id 相同”，并给出本机实测加速比（注明测量条件）。
- **ch05**：RoPE（相对位置性质的数值验证；交错式与 rotate_half 不可互换）→ SwiGLU（参数量对齐，
  1344 = 21×64）→ RMSNorm 与其他改良（去 dropout、去 bias、解除权重共享；eps 的坑）→ 改良版 GPT
  → KV cache（prefill 的因果掩码 bug 与修复 8b3911f；超过 max_context_len 后 RoPE 位置被钳住；
  上下文窗口 ≠ 累计 token 消耗，困惑 CTX-01）。
- **ch06**：AdamW 原理与实现（与 torch 对拍 4.8e-6）→ 学习率：warmup + 线性衰减到 0（D2Z，不是 cosine）
  → 混合精度（fp16 上下溢、bf16 范围，位级浮点探索器）→ 改良版预训练（与 CS336 A1 TinyStories 配置完全相同，
  目标 val loss ≤ 1.45；本书在 H100 上原样跑完是 1.304）→ 生成与 LLM-as-a-Judge → RLHF 与 DPO 理论（附录 E）→ DPO 实现
  （初始 loss = ln 2；HF 上的 StoryBot 权重键名是 mlp，需要改名）。困惑 TRN-02（初始学习率从哪来）、
  TRN-03（nats 与 bits，KL 的“里程表”）、DIST-01/02（谁生成、谁打分）。
- **ch07**：WebBot 分词器（50K 词表，复用 storybot.train_bpe）→ 定位瓶颈 → Rust/C++ → 特殊 token
  → 其他分词器 → 无分词器方法。每节配最小验证（例如 profile 一段小语料、tiktoken 对比）。
- **ch08**：模型规模与 scaling law（Chinchilla 20 tok/param，6ND 计算器）→ Flash Attention（在线 softmax
  的分块计算，NumPy 验证与普通 softmax 相等）→ GQA（repeat_interleave，KV 显存 12 vs 36 KiB/token）
  → MoE（路由玩具）→ torch.compile → 上下文扩展（RoPE 插值）。
- **ch09**：梯度累积（与大 batch 等价的验证）→ 梯度检查点 → 分布式训练（DDP、no_sync；
  max_iters 固定时加卡等于加 token）→ ZeRO（显存分解计算器）→ Muon（Newton–Schulz 正交化的 NumPy 验证）
  → Weights & Biases → WebBot 的训练与生成（CPU 上把原书脚本的缩小版真的跑一遍；H100 上的 200 步计时探针
  `webbot-probe`，外推全程见 §8）。
- **附录 A–E**：推导逐行写成公式，每一步都配“数值验证这一步”的验证单元（例如 B-b、C-a、D-b、E-c 逐行验证），
  每篇另有交互图。原计划的“推导步进器”没有做：推导拆成逐行之后，五篇附录的 34 个独立公式在 375 px 宽度下
  都能完整显示、不需要横向滚动（2026-10-12 用 Playwright 在构建产物上逐个量过），逐步展开的组件已经没有必要。
- **playground**：浏览器里的 CodeBot（纯 JS 前向，按行 int8 量化，每个 checkpoint 11.3 MB，KV cache，WASM SIMD 内核），
  可切换预训练 / SFT / GRPO 三个 checkpoint（本书在 3060 Ti 上自己训练的 `codebot-3060ti`，不是官方权重），看 token、
  logit lens、各层各头的注意力、下一个 token 的分布，并有“与 PyTorch 对拍”按钮（与同一份 int8 权重的 PyTorch 相比，
  logits 差 10⁻⁵ 量级，贪心生成的 token 全部相同）。
- **train-in-browser**：在标签页里从零训练一个 93,408 个参数的字符级 GPT（与 CodeBot 同一个类，2 层，V = 96），
  前向、反向传播和 AdamW 都用 JS 手写，在 Web Worker 里运行；每一层做梯度确认，loss 与梯度和 DLFS6 的
  `codebot.model.GPT` 对拍。
- **gpu**：本机 / Modal / RunPod / Lambda 的完整操作手册、成本计算器、全部运行报告的成绩单。
- **cs336**：逐组件对照表、差异解释（平票规则、D2Z vs cosine、RMSNorm eps…）、把 DLFS6 接到
  CS336 A1 测试上的结果（除速度测试外原样通过 35 项，三处小修改后再多 5 项）。
- **glossary**：术语双语表、命名对照（DLFS6 / Hugging Face / CS336 A1 / GPT-2 / nanoGPT）、RL 指标对照（DLFS6 /
  CS336 A5 / veRL / tinker-cookbook）、一词多义（probe、epoch、step、context、batch）。
- **errata**：所有经过验证的 DLFS6 代码问题（50 条，按严重度分三组），每条都有最小复现和修复；另有“原书注释即断言”一节。

## 7. 端到端例子

1. **浏览器里的一生**（playground）：输入 → BPE → 前向 → 采样 → 输出，全程可视；贪心生成与使用同一份 int8 权重的
   PyTorch 逐 token 一致。
2. **标签页里训练迷你 GPT**（train-in-browser 页）：93,408 个参数的字符级 GPT，纯 JS 手写反向传播（逐层梯度确认，
   并与 DLFS6 的 GPT 对拍）。在一台普通台式机上约 30 秒越过 bigram 线（第 475 步），1,000 步约 1 分钟；
   录制的样例从第 0 步的乱码，到第 2,000 步写出像样的代码片段。
3. **真实流水线**（gpu）：CodeBot 预训练 → SFT → GRPO（本机 3060 Ti，另有加一行 `model.eval()` 的 GRPO 勘误对照），
   StoryBot 预训练 → 生成 → DPO（Modal H100），WebBot 的 200 步计时探针（Modal H100）；
   产物（曲线、样例、权重）回流到页面和 playground。

## 8. GPU 预算与运行记录（用户已批准 Modal 上限 $30）

除 WebBot 的正式预训练外，计划中的运行都已跑完，记录在 `web/data/runs/`。耗时是运行报告里每个脚本的墙钟时间；
Modal 的费用是容器计时 × 时价（H100 每小时 3.95 美元，加上每张卡配的 2 核 CPU 与 8 GiB 内存，合计 4.11 美元），
不含冷启动、镜像构建、`prepare` 与存储，所以是下限。

| 任务 | 位置 | 耗时 | 费用 | 运行记录 |
|---|---|---|---|---|
| CodeBot 预训练 / SFT / GRPO | 本机 RTX 3060 Ti | 实测 41.9 min / 2.1 min / 89.7 min | $0 | `codebot-3060ti` |
| CodeBot GRPO 勘误对照（更新前加一行 `model.eval()`） | 本机 RTX 3060 Ti | 实测 76.5 min | $0 | `codebot-grpo-evalfix-3060ti` |
| StoryBot 预训练 / 生成 / DPO | Modal H100 | 实测 16.2 min / 7 s / 34 s | $1.17（容器计时 1,028 s） | `storybot-h100` |
| WebBot 200 步计时探针 | Modal H100 | 实测 218 s（容器计时 240 s） | $0.27 | `webbot-probe` |
| WebBot 正式预训练 | 只给命令，等用户批准（超出上限） | 按探针外推：1 卡约 10.4 h（训练 8.8 h，加 101 次评估、每次约 56 s，共 1.6 h）；8 卡把 `max_iters` 改成 12,500、`warmup_iters` 改成 63，约 1.3 h | 1 卡按 GPU 单价约 $41，按探针的结算价（含 CPU 与内存）约 $43；8 卡按 GPU 单价约 $42 | — |

WebBot 的外推见第 9 章验证 9.7-d、第 8 章验证 8.1-g 与 GPU 手册验证 G.2-j；8 卡的数字假设每张卡的速度不变，
没有算卡之间的通信。8 卡不改 `max_iters`、原样运行，会读 8 倍的 token，花费也是 8 倍（约 $330）。
两次 Modal 运行的记录合计约 $1.45（下限）。规则不变：每次在 Modal 上启动前先打印成本估算，累计不超过上限。
`gpu/modal_app.py` 的 `launch` 就是这样做的：估算的上限超过 `--budget`，它就拒绝启动。

（开跑之前的计划估算，留作对照：CodeBot 37 min / 2 min / 约 2 h；StoryBot 预训练 20–30 min、约 $2，DPO 数分钟、< $0.2；
探针数分钟、< $0.5；WebBot 正式预训练 1 卡约 9 h、8 卡约 1.2 h、约 $36–40。WebBot 的计划估算按 MFU 35% 算，已经含 101 次
评估；探针实测每步 0.317 秒，比估算的 0.280 秒慢 13%，全程因此从约 9.1 h 变成约 10.4 h，见 GPU 手册 G.1 与验证 G.2-j。）

## 9. 工作流程

1. 基础设施与示范章节（ch01 的 1.1–1.2 已完成，作为语气和组件用法的样板）。
2. 各章并行撰写（每位作者只改自己的 `content/<slug>.md`、自己的组件、自己的 `verify/test_<slug>.py`）。
3. 对抗式审校：技术正确性（重跑每条断言）、斋藤风格与中文行文、困惑覆盖、CS336 事实核查。
4. GPU 运行，产物回填。
5. 整站构建、浏览器全量检查、提交。

这五步已经完整走过一轮（第一轮对抗式审校与之后的全量关卡见提交 `ac0b37f`）；现状与下一步见 `PROGRESS.md`。
