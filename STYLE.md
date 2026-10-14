# 写作与实现规范

本规范约束每一章的作者（包括 AI 代理）。样板见 `content/ch01.md` 的 1.1–1.2 节：语气、组件用法都以它为准。

## 1. 声音与行文（从斋藤康毅四本书的统计中提炼，详见 `research/style-*.md`）

- **主语是“我们”**：作者与读者同行（“我们来实现……”“我们来确认一下”）。“你”只用于读者的体验或选择
  （“你可以拖动……”“如果你对此感兴趣……”）。不用“笔者认为”，除非是在交代一个教学上的取舍。
- **一句一意，一段一步**：段落 3–5 句，只推进一件事；句子以短句和中句为主。不写配置清单式的长句
  （用户会把这种句子原样贴回来问“这是什么意思”）。
- **设问即答**：用“那么，……呢？”“为什么……呢？”开启小话题，紧接着回答。不留悬空的问题。
- **类比只用一次**：一个概念最多一个生活化类比（1–3 句），随后用“严格地讲，……”回到精确表述。
- **公式三件套**：公式之后立刻写“这里，$x$ 表示……”，再给一个能手算的数值例子，再给一个验证单元。
  只有公式、没有数字的段落，用户不会买账。
- **先给答案**：常用“先来看结果”“先说结论”，再展开推导或实现。章首和长训练前先给终点预览。
- **诚实**：不用“显然”“易证”“众所周知”。简化要明说（“简单起见……”）；局限要承认（“遗憾的是……”）；
  直观解释标明是直观（“直观上……，严格的证明见附录 B”）。
- **术语**：首次出现时 **中文（English）** 加粗，例如 **分词器**（tokenizer）；之后中英择一并保持一致。
  这些词正文直接用英文：token、BPE、Transformer、Attention、softmax、logits、embedding、LayerNorm、RMSNorm、
  RoPE、SwiGLU、GELU、KV cache、SFT、RLHF、GRPO、PPO、DPO、AdamW、batch、epoch、checkpoint、warmup。
  译名用常用译法（智能体、回报、折扣因子、策略梯度），首次出现时可注明《深度学习入门4》的译法。
- **统一用词**（用于正文、标题、图注、验证单元与 localcheck 的标题，也用于读者看得到的代码文字：验证单元里的注释和
  打印，localcheck 展示的测试函数里的注释、文档字符串和打印，组件界面上的文字。标识符、字典键、数据、单元 id、URL、
  代码清单，以及对原书、论文、讲义的引文保持原样；测试断言了打印的文字时，两边一起改）：
  - BPE 的并列取舍叫**平票**、**平票规则**（第 1 章专栏的定义），不叫“平局”。第 6 章偏好数据里的平局（打平、舍入）是另一回事。
  - 模型的存档叫 checkpoint（“三个 checkpoint，三种回答”），不叫“检查点”。第 9 章的梯度检查点（activation checkpointing）是另一个概念，沿用第 9 章的写法。
  - GRPO 的迭代按“轮”计数（“第 51 轮”）：一轮是一次生成加两次优化器更新，运行记录里的 it=k（tqdm 计数 k 的第一行，见 GPU 手册 G.7）就是第 k 轮。优化器的更新、预训练、SFT 和 DDP 训练的迭代仍然叫“步”，生成时每出一个 token 也叫一步。
  - “轮”只数 GRPO 的迭代。数据过一遍叫 epoch（名词：每个 epoch、第一个 epoch），数遍数时用“遍”（14.3 遍、过一遍）。“轮”的其他意思不受影响：基准测试的重复（第 r 轮）、对话的轮次、BPE 的合并轮次。
  - Attention、多头 Attention 是机制的名字（第 2 章引入，之后不再加粗）；注意力权重、注意力分数、注意力头这类复合词照常写“注意力”。
  - 词嵌入写作 token embedding（第 2 章首次出现时写 **词嵌入**（token embedding）），位置嵌入写作位置 embedding，嵌入层写作 embedding 层。输出端的投影叫输出层（代码里是 `unembed`），不叫“反嵌入”。
  - warmup 直接写英文（代码里是 `warmup_iters`）；第 6 章首次出现时写 **学习率预热**（warmup）。
  - Attention 首字母大写。Flash Attention 指这项技术和 8.2 节（官方目录的写法）；FlashAttention、FlashAttention-2/-3 只在指论文或库时使用。
  - Bradley-Terry 用 ASCII 连字符（官方目录 6.7.2 和 DPO 论文的写法），不用 en dash。
  - greedy decoding 译作**贪心解码**，首次出现时加粗并注明（greedy decoding）；“greedy 生成”写作“贪心生成”。标识符不变。
  - 术语表的“中文”一栏：正文里写英文的术语（上面的列表，以及 FFN、GQA、MoE、RoPE、KV cache 等）写成“English（中文）”，例如 Attention（注意力）、KV cache（KV 缓存）、LLM-as-a-Judge（LLM 评审）。
- **路标**：“下面”“接下来”“首先……然后……最后”串起步骤；小节以“以上就是……”或一句承上启下的话收尾。
- **节奏**：每 2–3 段至少有一个可运行或可看的东西（代码清单、验证单元、交互图、表格）。
- **幽默**：干而轻，点到即止；推导时不开玩笑。

## 2. 章的骨架

1. **章首**（2–3 段，不用列表）：上一章做到了什么、卡在哪里 → 本章要解决的问题 → 路线图。
2. **各小节**（编号与标题严格沿用官方目录，`## 2.3 Attention 缩放`）：
   问题 → 最小例子（能心算的数字）→ 原书代码（`listing`）→ 输出 → 验证单元 → 归纳与封装。
   原书一个脚本对应一节时，按“先看结果 → 逐段解释 → 验证”的顺序讲完整个脚本的关键代码。
3. **专栏**：官方目录里有的专栏必须写（`::: column 标题`）。
4. **小结**：先 1–2 段散文回顾并指向下一章，再写“本章所学的内容”列表（5–7 条可检验的陈述，
   每条末尾用链接指向支撑它的验证单元，如 `（[验证 2.3-a](#v-2.3-a)）`）。
5. 最后一句是下一章的钩子。

## 3. Markdown 指令

````markdown
---
chapter: 2            # 图编号前缀（图 2-1）；部用 I/II/III，附录用 A–E
slug: ch02
order: 30             # 全书排序
kind: chapter         # chapter | part | appendix | extra
title: 模型【基础篇】：从软字典到 GPT-2
lede: 一句话导语。
pyodide_packages: [numpy]   # 本页验证单元需要的 Pyodide 包（numpy、regex、matplotlib 可用；没有 torch）
---

```listing ch02/02_attn_math.py#L10-L25       # 原书代码，从 third_party/dlfs6 抽取；行号必须准确
```

```verify id=2.2-a title="注意力权重每行之和为 1"
# NumPy-only。所有验证单元按顺序共享一个命名空间（与浏览器一致）。
```

```localcheck verify/test_ch02.py::test_attention_matches_sdpa title="手写 Attention 与 SDPA 一致"
```

```widget attnmask caption="图注写结论，例如：去掉因果掩码后，第 1 个位置“看见”了未来的 token。"
{"n": 6}
```

::: column BPE 的并列取舍
专栏正文……
:::

::: note | warn | cs336 <标题> | confusion <标题> | gpu <标题> | think <问题>
……
:::
::: answer
（紧跟在 think 之后，折叠的答案）
:::
````

- 数学：行内 `$...$`，独立 `$$...$$`（KaTeX）。美元符号作货币时写“美元”。
- 交叉引用：`[2.3 节](ch02.html#s2-3)`、`[图 2-4](#fig-2-4)`、`[验证 2.3-a](#v-2.3-a)`。
- 跨书引用固定句式：“如果你读过《深度学习入门4：强化学习》9.3 节，……”（书名与节号取自
  `research/style-*.md` 的 concept anchors；中文版书名：《深度学习入门：基于Python的理论与实现》、
  《深度学习进阶：自然语言处理》、《深度学习入门2：自制框架》、《深度学习入门4：强化学习》）。
- 提示框标题（`::: kind 标题`）里可以写行内代码、`$…$` 公式和链接；强调、图片和 HTML 不生效，所以 `embed_dim`
  里的下划线、`<|endoftext|>` 都原样显示。
- 代码清单显示从引用的起始行开始的行号（行号不会被复制）。引用范围从代码行开始、在代码行结束，不要把首尾的空行带进来。
- 行内代码保留空格：开头的空格、连续的空格会和后一个字符粘在一起，不会单独断到行尾，所以 `' add'` 这样的 token
  可以照原样写，不必改写成 `␣add`。

## 4. 验证单元规则

- **只用 NumPy / regex / 标准库**（Pyodide 里没有 torch）。需要 torch 的结论写成 `verify/test_<slug>.py`
  里的 pytest，再用 `localcheck` 引用。
- **确定性**：随机数一律 `rng = np.random.default_rng(种子)`；不依赖时间、文件系统顺序或 dict 以外的哈希顺序。
- **输出可读且稳定**：打印格式化后的值（`f"{x:.4f}"`、`np.array2string(a, precision=4)`），
  不要直接打印 NumPy 标量的 repr（NumPy 2 会显示成 `np.float64(...)`）。
- **用 assert 写死结论**：每个单元至少一个 assert，断言正文中的那句话。容差要写明理由
  （例如 `assert abs(x - 6.9078) < 1e-4  # ln(1000)`）。
- **快**：单个单元在构建机上 < 1 秒（浏览器里的 NumPy 没有 BLAS，大约慢 3–10 倍）。矩阵 ≤ 几百维。
- **id**：`<章>.<节>-<字母>`，如 `3.9-c`；附录用 `B-a`；同页唯一。
- **翻译原书代码时**：PyTorch 代码改写成 NumPy 版本做 A 层验证时，必须再有一个 B 层测试证明
  “NumPy 版 == 原书 PyTorch 版”（`verify/test_<slug>.py` 里导入 third_party/dlfs6 的模块做对拍）。
- 数据文件：放在 `web/data/<slug>/`（构建时复制到站点的 `data/<slug>/`），验证单元用 `files=data/<slug>/xxx` 声明，
  代码里用相对路径 `data/<slug>/xxx` 读取（构建与浏览器路径一致）。单个文件 ≤ 2 MB。

## 5. 交互组件规则

- 文件：`web/widgets/<slug>-<name>.js`（共享组件不加前缀，放 `_` 开头的库文件里，章节作者不要改共享库，
  需要改就在返回报告里提出）。导出 `mount(root, config)`。
- 共享库只有 `_lib.js`（DOM）、`_plot.js`（SVG 图表与主题助手），以及 `_bpe.js`、`_gpt.js`、`_tinygpt.js`；同一页的
  组件可以共用一个辅助模块（如 `ch03-common.js`）。不引入外部库；颜色只用 CSS 变量（`var(--ai)` 等），
  明暗主题都要清楚；窄屏 375px 不出现横向滚动；所有控件可用键盘操作并有 `aria-label`。
- 跟随主题：调色板颜色用 `_plot.js` 的 `seriesVars(el)` 设好 `--series-k`，样式里写 `var(--series-k)`（`color-mix()`
  里也一样）；在 JS 里算出的颜色（画布、按十六进制填色的 SVG）通过 `onThemeChange(fn)` 重画。不要在挂载时把
  `seriesColor()` 的值写死进样式，也不要自己写 MutationObserver：它们看不到操作系统的明暗切换。
- 图表：对数数据把原值交给 `lineChart` 的 `logX`/`logY`，不要先取 log10 再画在线性轴上；`lineChart` 还有 `xMin`/`xMax`、
  `marginRight` 和 `markers`（`true`，或要标记的点的数组）。
- 按需加载的组件，加载时显示“进入视野时才……”“正在下载”“正在准备”“正在加载”“正在载入”“正在读取”这类提示，或在
  状态元素上设 `aria-busy="true"`：`check_pages.mjs` 截图前会等它加载完（最多 30 秒）。
- 组件里的计算如果与正文结论有关，**要有同一结论的验证单元**（组件给直觉，验证单元钉死结论）。
- 组件需要的样式写在组件里（`root.style` 或以 `w-<slug>-` 为前缀的 `<style>`），不要改 `site.css`。

## 6. 针对用户的强化点

`research/private/confusions.md` 记录了用户过去真实困惑过的 22 个点，每条标注了目标章节。
写到对应章节时，用 `::: confusion <标题>` 正面处理：先复述那个直觉为什么自然，再用一个能手算的例子
和一个验证单元把它拆开，最后给一句可以带走的结论。不要提及“用户”或“过去的会话”，写成任何读者都会有的疑问。

用户的学习偏好（务必照顾）：要证据和出处（“这个数从哪来”）；先预测再运行；公式必须配手算例子和类比；
测验要短（每节最多 1–2 个“想一想”）；关键概念在后续章节以新角度复现一次（间隔重复）。

## 7. CS336 对照

每章至少一个 `::: cs336` 框，内容取自 `research/cs336-crossref.md`：对应的作业组件或讲次、
CS336 的要求、DLFS6 的差异以及差异带来的后果。所有关于 CS336 的说法必须能在该文件或其引用的 URL 里找到。

## 8. 完成标准（每章）

- [ ] 官方目录的每一小节、每一个专栏都已写到；该章每个脚本的关键代码都用 `listing` 讲到。
- [ ] 每个技术断言都有 A 层或 B 层验证；`uv run python tools/build.py --only <slug> --out build/preview-<slug>` 通过。
- [ ] `uv run pytest verify/test_<slug>.py` 通过。
- [ ] `node tools/check_pages.mjs build/preview-<slug> <slug>` 通过（页面无报错，全部验证单元在真实 Pyodide 中盖章）。
- [ ] 组件在明暗主题、桌面与 375px 宽度下都可用（`check_pages.mjs --widget-shots` 会截浅色、切换后的深色和操作系统深色三种图）。
- [ ] 小结有“本章所学的内容”，每条都链接到验证单元。
- [ ] 中文通读一遍：没有翻译腔、没有配置清单式长句、没有未回答的设问。

## 9. Windows 上的工具陷阱（作者必读）

- Python 写文本文件一律 `Path.write_text(..., encoding="utf-8", newline="\n")`，否则会写出 CRLF，数据文件的哈希随之改变。
- 含反斜杠的代码不要用 bash heredoc 写入（Git Bash 会吞掉反斜杠，`\b` 甚至会变成退格符）；用 Edit/Write 工具或 Python 脚本文件。
- 隐藏 GPU 用 `CUDA_VISIBLE_DEVICES=-1`；Windows 会丢掉空字符串的环境变量，`""` 不起作用。
- `localcheck` 默认标签是“本地验证”；需要特定依赖时写 `needs=PyTorch`（或 `needs=tiktoken`），标签会显示“本地验证（需要 PyTorch）”。
- 全站构建会检查每一个站内链接（页面与 `#锚点` 都必须存在），改动节标题编号前先搜索引用它的地方。
  没有编号的标题，锚点由标题文字的哈希生成（`h-` 加 8 位），改字也会改锚点。
- 验证单元的 id 一旦被其他页面链接，就保持不变：新增的单元用新字母，不要给已有的单元重新编号。
- 子模块必须是 LF。已经检出成 CRLF 的，先 `git -C third_party/dlfs6 config core.autocrlf false`，再
  `git -C third_party/dlfs6 rm -r -q --cached .` 和 `git -C third_party/dlfs6 reset -q --hard`；只执行 `checkout -- .` 不可靠
  （见 README 第 2 步与 `verify/test_ch00.py::test_checkout_alone_leaves_crlf_files`）。
- 内存紧张的机器上，pytest 一个测试文件一个进程：整套测试在一个进程里会涨到约 8 GB（会话级 fixture 缓存着各章的模型），
  内存不够时后半段会出现分配失败造成的 F/E，单独重跑那个文件就能分辨。
- Edit/Write 工具会把参数里的 `\uXXXX` 转义变成字面字符。JS 里需要转义时写 `\u{XXXX}`，或者在代码里用 `String.fromCodePoint` 构造。
