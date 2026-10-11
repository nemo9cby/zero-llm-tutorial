# 同类教程调研 → 设计模式目录（tutorial-patterns.md）

> 用途：为「从 dlfs6（ゼロから作るDeep Learning ❻ LLM編）反向生成的中文交互式教程」提供设计依据。
> 调研日期：2026-10-11。调研方式：WebSearch/WebFetch + 本地参考仓库（build-nanogpt、nanochat、llama2.c、nano-vllm、dlfs6）+ 若干本地最小实验。
> 可信度标注：**【实测】** 本次在本机跑过（脚本见附录 A）；**【源码】** 直接读源码/包内文档核对；**【来源】** 来自网页/README（附链接）；**【待实测】** 合理推断但尚未验证，落地前必须先跑。

---

## 0. 一页结论（TL;DR）

1. **最接近「处处可验证」的先例是 CS336 的「可执行讲义」**：讲义本身就是 Python 程序，`text()` 写叙述，代码旁用 `# @inspect` 暴露变量，用 `assert` 把每句断言钉死，然后渲染成能对照代码查看变量值的 trace 页面。我们的教程应采用同一思路：**每个技术声明 = 一张「验证卡」**（声明 + 最小检查 + 期望值 + 出处），并标注运行层级（浏览器 / 本地 CPU / 本地 GPU / 云 GPU）。
2. **浏览器内可验证的面积比想象中大**：Pyodide 314.0.7 内置 `numpy 2.4.6`、`regex`、`tiktoken 0.12.0`，但没有 torch【来源】。因此 ch01/ch04/ch07 的 BPE 代码（纯 Python + `regex`）可以原样在浏览器跑；注意力、RoPE、RMSNorm、fp16/bf16 等可用 NumPy 移植后在浏览器跑。fp16/bf16 的金标输出用 NumPy 能逐一复现【实测】。
3. **dlfs6 自带「注释金标」**：脚本里大量 `print(x)  # 期望输出`。用 AST 把它们自动变成断言，首轮 31 条中 29 条直接通过，另 2 条只是表示形式不同（`defaultdict(...)`、`torch.Size`）【实测】。这是「处处可验证」最便宜的起点。
4. **CS336 × dlfs6 可以「互相对拍」**：CS336 A1 的 `tests/adapters.py` 接口（`run_rope`、`run_rmsnorm`、`run_swiglu`、`run_train_bpe`、`get_adamw_cls`……）几乎一一对应 dlfs6 的模块；dlfs6 `storybot/tokenizer.py::find_chunk_boundaries` 与 CS336 `pretokenization_example.py` 是同一算法（含同一个 4096 字节窗口的局限）【源码】。CS336 的快照测试（`.npz`，rtol 1e-4 / atol 1e-2）、BPE 速度测试（<1.5 s）、tiktoken 对拍都能直接借来验收 dlfs6 的实现。
5. **三台机器人的规模已经算出来了**【实测】：CodeBot 11.12M 参数、StoryBot 22.70M、WebBot 152.32M；初始化损失分别 6.99 / 9.30 / 10.96，与 ln V（6.91 / 9.21 / 10.82）吻合。这些数字适合做「先预测再揭晓」和训练健全性检查。
6. **等价性测试是最有说服力的验证**：dlfs6 的 KV cache 增量解码与整段前向的 logits 最大误差约 1e-7（StoryBot、WebBot 都成立）【实测】；GQA 的 cache 确实只存 `n_kv_head` 个头。「同一结果、多种实现」应该成为每章的固定栏目。
7. **「挪动这一行会怎样」的失败表有现成素材**：例如 ch09 只把 `loss.backward()` 包进 `no_sync()`，而 PyTorch 文档明确要求前向也要在上下文里，否则梯度仍会同步【源码】（all-reduce 是线性的，所以结果仍正确，只是每个 micro-step 都白付一次通信）；cache 解码超过 `max_context_len` 时 RoPE 位置被钳在 M−1，StoryBot 的 cache 会无限增长、WebBot 会截断【实测】。
8. **GPU 部分用「GPU 运行卡 + 产物回传 + 页面回放」**：每次真实训练输出 `report.json / loss.jsonl / samples.json`（含 git sha、硬件、种子、耗时、费用），网页只展示有产物背书的数字。参考 nanochat 的成绩单、llama2.c 的模型/损失表、picotron 的「多卡 loss 与单卡一致」验收。
9. **针对用户过往困惑的加强点**集中在 RL 指标与批量概念。例如按 dlfs6 GRPO 的写法，在 ratio=1 时 loss 恰好等于 −Σ Aᵢ·Lᵢ / n（奖励与回答长度的组内协方差），长度相同时为 0，但梯度不为 0【实测，玩具模型】。更值得讲的是：ch03/09 里 `model` 一直处于 train 模式（CodeBot 的 dropout=0.1），`old_model` 处于 eval 模式，**权重完全相同时 ratio 也不等于 1**：在 CodeBot 检查点上测得均值 0.91、标准差 0.58，首步就有 33% 的 token 落在截断区间 [0.8, 1.2] 之外；切到 eval 后 ratio 恰为 1【实测】。这两点应做成「指标解码器」+ 可拖动的玩具模拟 + 失败表的一行。
10. **风格上借 Welch Labs / 3Blue1Brown 的统一视觉语法与 distill 的「细节按需」，避开 3D 重型可视化与 600MB 级模型下载**；每个交互都必须绑定一句可验证的声明，否则不做。

---

## 1. 调研范围与方法

- **线上**：Karpathy 全系列（micrograd、makemore、Let's build GPT、tokenizer/minbpe、build-nanogpt、nanochat、llm.c、llama2.c、microgpt、LLM101n）、Raschka（LLMs-from-scratch、reasoning-from-scratch）、Annotated Transformer、Jay Alammar、Transformer Explainer、Bycroft LLM Visualization、Tiktokenizer、picoGPT、HF Ultra-Scale Playbook / picotron、How to Scale Your Model、Transformer Math 101、kipply 推理算术、d2l、UDL、distill、Bret Victor/Tangle、Ciechanowski、Welch Labs、Sasha Rush puzzles、tiny-llm、Deep-ML、HF LLM Course、CS336（2025/2026）、中文的动手学大模型、happy-llm、MiniMind、tiny-universe，以及浏览器运行基础设施（Pyodide、marimo、Quarto Live、gpt-tokenizer、WebGPU 训练玩具）。
- **本地**：`C:\Users\chenf\Projects\{build-nanogpt,nanochat,llama2.c,nano-vllm}` 只读浏览；`_refs\dlfs6` 只读导入（实验脚本全部放在 scratch，未改仓库文件）。
- **本地实验**（附录 A）：参数量/初始损失/KV 字节、注释金标批量断言、KV cache 等价性、超长上下文行为、NumPy 复现 fp16/bf16 金标、GRPO loss 玩具模拟、DDP `no_sync` 文档核对。

---

## 2. 参考作品总表

### 2.1 代码优先的「从零构建」系列

| 名称 | 链接 | 做得好的 | 借鉴什么 | 避免什么 |
|---|---|---|---|---|
| Karpathy · Neural Networks: Zero to Hero（micrograd / makemore 1–5 / Let's build GPT / GPT Tokenizer） | https://karpathy.ai/zero-to-hero.html · https://github.com/karpathy/nn-zero-to-hero | 从空文件逐行写，写完立刻跑、立刻看数。Backprop Ninja 用 `cmp()` 把手推梯度和 autograd 逐变量比对，打印 exact / approximate / maxdiff【来源】。Let's build GPT 用几种等价写法推出 masked attention，再用 `allclose` 验证 | 逐变量对拍；「同一结果多种写法」的阶梯；先心算初始损失再看实际值 | 两小时的视频不适合当主载体；视频和代码会漂移（ng-video-lecture README 自己承认初始化没讲，所以收敛慢） |
| minbpe + exercise.md | https://github.com/karpathy/minbpe | 5 步渐进练习，每一步都给出可对拍的目标：第 3 步要与 tiktoken `cl100k_base` 输出逐 id 一致，第 4 步加特殊 token。测试覆盖 Wikipedia 例子 `aaabdaaabac` → `[258,100,258,97,99]`、编码解码往返、保存再加载【来源】 | 练习给「目标输出」而不是「答案代码」；拿 tiktoken 当金标 | tiktoken 词表要联网下载，浏览器里需要预置 |
| build-nanogpt | https://github.com/karpathy/build-nanogpt（本地 clone 只有 `train_gpt2.py`） | 一个 commit 一个概念，可以照 git log 跟读；README 维护勘误（例如 `require_backward_grad_sync` 要覆盖前向）；124M 约 1 小时、约 $10【来源】 | 把提交历史当目录；公开维护勘误区 | 只有代码和视频，没有内嵌验证 |
| nanochat | https://github.com/karpathy/nanochat · https://github.com/karpathy/nanochat/discussions/1 | 一个复杂度旋钮 `--depth`；`runs/speedrun.sh` 一键跑完全流程；早期版本输出 `report.md` 成绩单（BASE/MID/SFT/RL × CORE/ARC/GSM8K/HumanEval/MMLU/ChatCORE，3h51m 约 $92）【来源】；Time-to-GPT-2 排行榜；`runs/runcpu.sh` 给出 CPU/MPS 路线。测试很「hermetic」：进程内训练一个小 tokenizer、自写 AdamW 与 `torch.optim.AdamW` 对拍（`assert_close`）、两次运行逐位一致、收敛性断言 `d1 < 0.5·d0`【源码】 | 成绩单、一键脚本、hermetic 测试 | 8×H100 是默认前提；FA3/FP8 等 Hopper 专属路径在 3060 Ti（Ampere）上不可用 |
| llm.c | https://github.com/karpathy/llm.c · https://github.com/karpathy/llm.c/discussions/481 | C/CUDA 实现与 PyTorch 导出的 debug state（logits、loss、梯度、10 步 Adam）对拍，通过时打印 `overall okay: 1`；GPT-2 124M 复现 90 分钟、约 $20（8×A100），HellaSwag 29.9 对 OpenAI 的 29.4【来源】 | 跨实现对拍用的「调试状态文件」 | C/CUDA 门槛高 |
| llama2.c | https://github.com/karpathy/llama2.c | PyTorch 训练 + 500 行 C 推理；TinyStories 15M/42M/110M 的 val loss 表（1.072/0.847/0.760）；`test_all.py` 用 temperature=0 比较 C 与 PyTorch 的 token 序列【源码】 | 「模型规模—损失」参考表；argmax 解码对拍 | 测试依赖本地 ckpt |
| microgpt（2026-02）+ MicroGPT Visualized | https://karpathy.github.io/2026/02/12/microgpt/ · https://microgpt.jtauber.com/ | 约 200 行纯 Python、零依赖（标量 autograd + GPT + Adam），约 4192 参数，MacBook 上约 1 分钟；loss 从 3.3（=ln 27）降到约 2.37；train0→train5 六个版本（计数 bigram → 手推梯度 → autograd → 单头注意力 → 多头 → Adam），用 gist revision 看 diff；可视化站把 6 步拆开讲【来源】 | 纯 Python，所以能直接在 Pyodide 里训练；「diff 就是课程」 | 标量 autograd 很慢，只适合几千参数 |
| LLM101n（已归档，未完成） | https://github.com/karpathy/LLM101n | 17 章大纲：bigram → … → Need for Speed I/II/III（设备/精度/分布式）→ KV cache → 量化 → SFT → RL → 部署，贯穿一个 Storyteller AI【来源】 | 与 dlfs6 StoryBot 同构的大纲参照 | 没有实际内容 |
| modded-nanogpt speedrun | https://github.com/KellerJordan/modded-nanogpt | 目标明确（8×H100 上 FineWeb val loss 到 3.28），有世界纪录史（从 45 分钟一路降到秒级，最新数字来自非官方来源）【来源】 | 「目标值 + 计时」的游戏化 | 大量 Hopper 专用优化；社区在 RTX 3090 上实测，最先挂的是 Hopper 内核【来源】 |
| Raschka · LLMs-from-scratch | https://github.com/rasbt/LLMs-from-scratch | 每章有 notebook、摘要脚本、习题和答案，另有 Manning 测验 PDF；CI 覆盖 Linux/macOS/Windows、旧版和 RC 版 PyTorch、pip/uv/pixi；KV cache 版本与无 cache 版本逐 token `torch.equal`；Qwen3/Llama/Gemma/Olmo 的从零实现在极小配置下与 HF transformers 做 `assert_close(1e-5)`；`test_context_overflow_bug`；3 个 notebook 走 nbval；主线代码「笔记本电脑可跑」【来源】 | 与 HF 参考实现对拍；多 OS CI；bonus 体系（KV cache、GQA/MLA、DPO、Qwen3） | 有的测试只查长度或形状（ch05 只断言有 39 个 loss） |
| Raschka · Build a Reasoning Model (From Scratch) | https://github.com/rasbt/reasoning-from-scratch | Qwen3 基座 + MATH-500 verifier + 推理时扩展 + 从零写 GRPO；按章说明 CPU/GPU 需求【来源】 | 可验证奖励（verifier）与 GRPO 的讲法，对应 dlfs6 ch03/09 | 模型更大，需要 GPU |
| picoGPT / GPT in 60 Lines of NumPy | https://github.com/jaymody/picoGPT · https://jaykmody.com/blog/gpt-from-scratch/ | NumPy 版 GPT-2 加载 OpenAI 权重；每个函数标形状 `[n_seq, n_embd]`；小检查（softmax 和为 1、LayerNorm 均值 0 方差 1）；端到端用一句固定 prompt 的输出与官方实现对照【来源】 | NumPy 实现可在 Pyodide 跑；形状注释 | 124M 权重 fp32 约 500MB，不适合浏览器 |
| nano-vllm | https://github.com/GeeeekExplorer/nano-vllm（本地有 clone） | 约 1200 行实现 prefix cache、TP、CUDA graph；在 RTX 4070 Laptop 8GB 上与 vLLM 做吞吐对比（1434 vs 1362 tok/s）【源码】 | 推理系统的进阶阅读；`bench.py` 的公平对比写法 | 需要 Qwen3-0.6B 权重和 Linux GPU |

### 2.2 交互可视化与解释型文章

| 名称 | 链接 | 做得好的 | 借鉴什么 | 避免什么 |
|---|---|---|---|---|
| Transformer Explainer（Georgia Tech Polo Club） | https://poloclub.github.io/transformer-explainer/ · 论文 https://arxiv.org/abs/2408.04619 | 浏览器里实时跑 GPT-2：输入文本、温度/top-k/top-p 旋钮、悬停看注意力、展开 embedding；从总览到细节的分段结构；论文报告了 90 人的用户研究，新版声称 49 万用户【来源】 | 采样参数旋钮；悬停看细节；总分结构 | 模型文件大（第三方称 ONNX 约 600MB），首屏很重。我们应换成自己的 11M CodeBot |
| LLM Visualization（Brendan Bycroft） | https://bbycroft.net/llm | 3D 展示每一次矩阵乘；先讲一个约 85k 参数、给 ABC 排序的 nano-gpt；侧边叙事面板，按 Continue/空格推进；悬停显示行列数和计算公式【来源】 | 「先玩具模型、后真模型」；逐步高亮 | WebGL 3D 的开发和维护成本高，手机上吃力 |
| Tiktokenizer | https://tiktokenizer.vercel.app/ · https://github.com/dqbd/tiktokenizer | 输入即着色，显示 token 数和 id，可选编码（如 o200k_base）；Karpathy 讲 tokenizer 时用它开场【来源】 | token 着色和 id 并排 | 基于 Next.js/tRPC，是否纯前端未确认；不解释「为什么这么切」 |
| Jay Alammar · Illustrated Transformer / GPT-2 | https://jalammar.github.io/illustrated-gpt2/ | 从整体到单个 token 的路径逐层放大；先 4 token 的玩具，再 1024；文件柜类比（query 是便签，key 是标签，value 是内容）；给出 768/1024/3072/50257、117M 与 124M 等具体数字【来源】 | 逐层放大；具体数字；类比 | 静态图，读者没法验证 |
| distill.pub（Hohman 等 2020《Communicating with Interactive Articles》；Goh 2017《Why Momentum Really Works》） | https://distill.pub/2020/communicating-with-interactive-articles/ · https://distill.pub/2017/momentum/ | 总结了互动文章的五种收益：连接人与数据、可玩的系统、促进自省（先预测）、个性化、降低认知负荷（细节按需），也列出代价：制作成本、可访问性、长期维护。Momentum 一文把 α、β 控件直接嵌进论证【来源】 | 设计原则清单；参数控件嵌入论证 | 成本高，distill 已停更。必须控制范围 |
| Bret Victor · Explorable Explanations / Tangle | http://worrydream.com/ExplorableExplanations/ · http://worrydream.com/Tangle/ | reactive document：拖动文中数字，周围的文字和图跟着变【来源】 | scrubbable numbers | 社区批评：难以发现可以拖、缺参照刻度 |
| Bartosz Ciechanowski · Exposing Floating Point + float.exposed | https://ciechanow.ski/exposing-floating-point/ · https://float.exposed/ | 位级可编辑的浮点讲解【来源】 | ch06 混合精度的位级探索器 | 没有 bf16，需要自己做 |
| Welch Labs（YouTube @WelchLabsVideo） | https://www.youtube.com/@WelchLabsVideo | 用一个具体问题贯穿的视觉叙事，有历史线索，视觉语法统一（与 3Blue1Brown 同属一类，作者学过 manim）【来源】 | 统一视觉语法；开场悬念；屏幕上出现真实数字 | 视频制作昂贵且不能交互，只借风格 |
| Sequence Toy（piston）/ TabGPT | https://sequence.toys/ · https://github.com/vinhowe/piston · https://github.com/jbingen/tabgpt | 用 WebGPU 在浏览器里训练序列模型；TabGPT 在一个标签页里预训练约 24M 的 GPT，用 IndexedDB 断点续训【来源】 | 「在你的标签页里训练」 | WebGPU 兼容性和学生设备差异 |
| Pope & Tedre 2026《Turning Language Model Training from Black Box into a Sandbox》 | https://arxiv.org/abs/2601.21631 | 162 名 CS1 学生在浏览器里短时间训练小模型后，解释「LLM 为什么会胡说」时，从拟人化说法转向数据和模型层面的推理【来源】 | 证据：亲手训练小模型有教学价值 | 4 页 WIP，证据还很初步 |

### 2.3 系统、规模与 GPU

| 名称 | 链接 | 做得好的 | 借鉴什么 | 避免什么 |
|---|---|---|---|---|
| HF Ultra-Scale Playbook | https://huggingface.co/spaces/nanotron/ultrascale-playbook | 4000+ 次实验、最多 512 GPU；显存分解 widget（可调 micro-batch、ZeRO 阶段、重算等，带 Llama 3 8B 预设）；先给公式再给实测表（N = h·v + L(12h²+13h) + 2h；混合精度下参数/梯度/fp32 副本/优化器为 2N/2N/4N/8N；激活 L·s·b·h·(34 + 5·a·s/h)）；有 profiler trace 和 cheatsheet【来源】 | 公式、计算器、实测三者并列 | 规模远超本教程；widget 不含 PP |
| picotron_tutorial | https://github.com/huggingface/picotron_tutorial | step1…step8 逐步加入 TP/DP/PP；在 Llama-1B、409.6 万 tokens 上检查各并行配置的 loss 与单卡基线一致【来源】 | 用「并行不改变 loss」做验收 | 需要多卡 |
| How to Scale Your Model（Google DeepMind，2025） | https://jax-ml.github.io/scaling-book/ | 从 roofline 讲起；每章末的 Worked Problems 答案默认折叠；有 Takeaway 框；画图的代码一并给出【来源】 | 隐藏答案的计算题；Takeaway 框 | 以 TPU/JAX 为中心 |
| EleutherAI · Transformer Math 101 | https://blog.eleuther.ai/transformer-math/ | C ≈ 6PD；推理显存约为权重的 1.2 倍；AdamW 12 字节/参数；三种重算策略下的激活公式；明说「当 sanity check 用」【来源】 | 用估算公式和实测对照 | 要标明是近似 |
| kipply · Transformer Inference Arithmetic | https://kipp.ly/transformer-inference-arithmetic/ | KV 每 token 2·2·L·H·d 字节；每 token 2P FLOPs；访存/算力瓶颈阈值（A100 约 208）；拿估算对照 FasterTransformer 实测，并逐项解释差距【来源】 | 「地板估算 → 实测 → 解释差距」 | A100 和大模型语境，要换算到我们的规模 |

### 2.4 课程、习题与判题平台

| 名称 | 链接 | 做得好的 | 借鉴什么 | 避免什么 |
|---|---|---|---|---|
| **Stanford CS336**（2025 / 2026 春） | https://cs336.stanford.edu/ · https://github.com/stanford-cs336/spring2025-lectures · https://github.com/stanford-cs336/assignment1-basics | **可执行讲义**：`text()` 叙述 + 代码 + `# @inspect` + `assert`，`execute.py` 生成 trace JSON，在 React 写的 trace viewer 里对照代码查看变量值【来源】。**A1 测试**：先在 `tests/adapters.py` 接线，初始全部 `NotImplementedError`；`NumpySnapshot`（`.npz`，rtol 1e-4，atol 1e-2，种子固定的 fixture）；BPE 速度 < 1.5 s（参考实现约 0.38 s）；merges 与参考完全一致；特殊 token 不被合并；与 tiktoken gpt2 对拍；Linux 下 `encode_iterable` 限 1MB 内存【来源】。**算力**：建议先在 CPU 上调对再上 GPU；2026 年 Modal 为赞助方（B200 $6.25/h，每月 $30 免费额度），RunPod B200 $4.99/h，Lambda $6.69/h【来源】 | 可执行讲义、adapter 式测试、快照、资源断言；直接拿来交叉验证 dlfs6 | 作业量大且全英文；讲义文本不能照搬 |
| The Annotated Transformer（Harvard NLP，2018/2022） | https://nlp.seas.harvard.edu/annotated-transformer/ · https://github.com/harvardnlp/annotated-transformer | 论文原文与代码逐段交织；jupytext 单一源（`.py` 是源，notebook 是构建产物），`make html`；mask、位置编码、学习率、label smoothing 都配小图；先做 copy task 的 First Example【来源】 | 文码交织；单一源构建 | CI 只查 PEP8，没有数值测试 |
| d2l.ai / 动手学深度学习 | https://d2l.ai/ · https://zh.d2l.ai/ | 每节都是可执行 notebook；多框架 tab；`#@save` 把函数沉淀进 d2l 包供后文复用；`d2lbook` 构建时重跑全部 notebook，输出不入源码；每节配讨论区【来源】 | 构建即测试；可复用函数库 | 多框架维护成本高 |
| Understanding Deep Learning（Prince）notebooks | https://udlbook.github.io/udlbook/ | 68 个 Colab notebook，读者补全代码；答案册只给教师【来源】 | 补全式练习 | 没有自动判分 |
| Sasha Rush · GPU / Tensor / LLM-Training Puzzles | https://github.com/srush/GPU-Puzzles · https://github.com/srush/Tensor-Puzzles · https://github.com/srush/LLM-Training-Puzzles | 每题有 spec 函数，`check()` 打印期望数组和你的数组；`show()` 把读写访问画成图；Tensor Puzzles 21 题配 hypothesis【来源】 | spec + check 判题；访问模式可视化 | 是练习集，不成教材 |
| tiny-llm（skyzh） | https://github.com/skyzh/tiny-llm | 按周/天解锁；学员实现 `tiny_llm`，测试与 `tiny_llm_ref` 参考包对拍；`--week/--day` 只跑某一天的测试【来源】 | 按章节过滤测试；参考实现包 | 基于 MLX/Apple 平台 |
| Deep-ML / TorchCode 类判题站 | https://www.deep-ml.com/ | 浏览器里写 softmax 等小函数，提交后立刻看测试结果【来源】 | 小粒度即时判题 | 题目与叙事脱节 |
| HF LLM Course | https://huggingface.co/learn/llm-course | 章末测验（代码输出题、判断题），可无限重做，答错指回对应章节【来源】 | 章末自测 | 偏 API 使用 |
| Karpathy · A Recipe for Training Neural Networks | https://karpathy.github.io/2019/04/25/recipe/ | 训练前的健全性清单：检查初始损失、单 batch 过拟合、与输入无关的基线、看喂进网络前的数据、用反传检查依赖关系等【来源】 | 直接转成验证卡 | 本身不可交互 |

### 2.5 中文生态

| 名称 | 链接 | 做得好的 | 借鉴什么 | 避免什么 |
|---|---|---|---|---|
| 动手学大模型（Dive into LLMs，上海交大） | https://github.com/Lordog/dive-into-llms | 11 章，每章「课件 PDF + 教程 README + notebook」三件套；2025 年起有昇腾版《大模型开发全流程》【来源】 | 三件套的组织方式 | 没有自动验证，偏应用 |
| happy-llm（Datawhale） | https://github.com/datawhalechina/happy-llm · https://datawhalechina.github.io/happy-llm/ | 从 NLP 基础讲到手写 LLaMA2、训练 tokenizer、预训练/SFT，并扩展到 Agentic RL；放出 215M Base/SFT 模型和在线 demo【来源】 | 中文叙述节奏；可下载检查点当金标 | 没有测试和习题 |
| MiniMind | https://github.com/jingyaogong/minimind | 纯 PyTorch 从零：tokenizer（6400 词）→ 预训练 → SFT → LoRA → DPO → PPO/GRPO/CISPO → Agentic RL → 蒸馏/MoE；minimind-3（64M，2026-04-01 发布，架构对齐 Qwen3）；作者估计单卡 3090 约 2 小时、约 3 元；坦白承认小模型在 C-Eval 等榜上接近随机【来源】 | 成本表、全流程脚本、诚实标注 | 数据集按 GB 计；成本是作者估计 |
| tiny-universe（Datawhale） | https://github.com/datawhalechina/tiny-universe | 「白盒」系列：TinyLLM（约 2GB 显存）、Qwen2 拆解、TinyRAG/Agent/Eval/Diffusion【来源】 | 白盒理念；显存门槛可控 | 没有测试 |
| CS336 中文社区笔记 | 例：https://juejin.cn/post/7643037374348296233 | 中文社区对 A1 BPE 的逐步实现讲解【来源】 | 中文术语对照 | 二手资料，细节须对照官方题面 |

### 2.6 在线运行与部署基础设施

| 名称 | 链接 | 做得好的 | 借鉴什么 | 避免什么 |
|---|---|---|---|---|
| Pyodide 314.0.7 | https://pyodide.org/en/stable/usage/packages-in-pyodide.html | 内置 numpy 2.4.6、regex 2026.3.32、tiktoken 0.12.0、scipy、matplotlib；没有 torch 和 tokenizers【来源】 | ch01/04/07 的 tokenizer 代码原样在浏览器跑；注意力等用 NumPy 移植 | 没有多进程（ch04/04、ch07 的 Pool 需要录制结果）；首屏加载；tiktoken 词表需联网或预置【待实测】 |
| marimo WASM 导出 | https://docs.marimo.io/guides/wasm/ | 反应式 notebook 导出成静态 HTML（Pyodide），拖滑块自动重算依赖单元；有 run/edit 两种模式【来源】 | 若要「notebook 即页面」可以直接用 | 不支持 torch 和多进程；样式受限 |
| Quarto Live | https://r-wasm.github.io/quarto-live/ | Pyodide 代码单元 + exercise/hint/solution 按钮 + check cell 自定义判分；`______` 占位填空【来源】 | 判分单元的数据格式 | 需要 Quarto 工具链 |
| gpt-tokenizer（JS） | https://github.com/niieani/gpt-tokenizer | 纯 JS 的 tiktoken 实现（r50k/p50k/cl100k/o200k），浏览器可同步加载【来源】 | 不依赖 Pyodide 的 GPT-2/4 金标 | 版本差异大，必须锁版本 |
| Modal 示例 hp_sweep_gpt | https://modal.com/docs/examples/hp_sweep_gpt | 从零训练小语言模型：8 组超参用 starmap 并行、早停、Volume 存数据/ckpt/TensorBoard【来源】 | 「GPU 运行卡」的云端模板；用 Volume 取回 ckpt | 示例会随版本变化，要锁版本 |

---

## 3. 先把 dlfs6 摆上桌：规模、层级与 CS336 对照

### 3.1 三台机器人的规模（数字可复算）

| Bot（章） | 配置（来自脚本） | 参数量 | 初始损失 vs ln V | KV/token（bf16） | AdamW 训练状态（fp32 参数+梯度+m+v） | 训练 tokens 与 6ND |
|---|---|---|---|---|---|---|
| CodeBot（ch01–03） | V=1000，ctx 256，384 维，6 头，6 层，权重共享 | **11.12M**【实测】 | 6.989 vs 6.908【实测】 | 9.0 KiB | 0.17 GiB | 20k 步 × 32 × 256 = 1.64 亿；1.09e16 FLOPs |
| StoryBot（ch04–06） | V=10000，ctx 256，512 维，16 头，4 层，ff 1344，RoPE/SwiGLU/RMSNorm | **22.70M**【实测】 | 9.299 vs 9.210【实测】 | 8.0 KiB | 0.34 GiB | 40k 步 × 32 × 256 = 3.28 亿；4.46e16 FLOPs |
| WebBot（ch07、ch09） | V=50000，ctx 1024，768 维，12 头 / 4 KV 头（GQA），12 层，ff 2048，SDPA，DDP，bf16 autocast，`torch.compile` | **152.32M**【实测】 | 10.959 vs 10.820【实测】 | 12.0 KiB（同配置 MHA 为 36 KiB） | 2.27 GiB | 每个 rank 每步 32×4×1024 = 131,072；100k 步 = 131 亿/rank；1.20e19 FLOPs/rank |

由此可以直接引出几道「先预测、再揭晓」：

- **CodeBot 会把数据看多少遍？** `tiny_codes.bin` 是 uint16，5,354,796 字节，即 2,677,398 个 token；预训练看 1.64 亿 token，约 61 遍【源码+算术】。顺带得出压缩率：6,743,534 字节 / 2,677,398 token ≈ 2.52 字节/token（词表 1000）。
- **WebBot 用 8 张卡会快 8 倍吗？** 不会。ch09 固定 `max_iters` 和每个 rank 的 batch，加卡只会让全局 batch 和总 token 数变成 W 倍，墙钟时间约等于单卡【源码+算术】。这正好对上过往会话里「gbs 到底是什么」的困惑（gbs = mbs × accum × world_size）。

### 3.2 运行层级（每个验证卡都要标）

| 层级 | 环境 | 时间量级 | 适合的 dlfs6 内容 |
|---|---|---|---|
| **L0 浏览器** | Pyodide（numpy、regex、tiktoken）或纯 JS | 秒级 | ch01 全部；ch02 注意力（NumPy 移植）；ch05 RoPE/RMSNorm/SwiGLU 的性质；ch06 LR 调度和 fp16/bf16；小模型推理/训练玩具 |
| **L1 本地 CPU** | `uv run`（PYTHONUTF8=1） | 分钟级 | torch 版的等价性和性质测试、KV cache 对拍、BPE 加速（多进程）、CS336 adapter 测试 |
| **L2 本地 GPU** | RTX 3060 Ti 8GB（Ampere：支持 bf16，不支持 FP8；FA3 是 Hopper 专属） | 分钟~小时 | CodeBot 预训练/SFT/GRPO、StoryBot 预训练/DPO、WebBot 缩小版 smoke |
| **L3 云 GPU** | Modal / RunPod / Lambda | 小时 | WebBot 正式预训练（含 DDP）、多卡 loss 一致性检查 |

> 环境提醒【源码】：dlfs6 的 `uv.lock` 解析的是 PyPI 上的 torch 2.9.1。PyPI 的 CUDA 依赖（`nvidia-*-cu12`）只有 Linux wheel，所以 Linux 云 GPU 上可以直接 `uv sync`；Windows 本机现有 `.venv` 里是 `torch 2.9.1+cpu`，3060 Ti 要用 PyTorch 官方 CUDA 索引另配环境【待实测：具体 index 与 uv 配置写法】。

### 3.3 dlfs6 × CS336 A1 组件对照（交叉验证的骨架）

| CS336 adapter（tests/adapters.py） | dlfs6 对应 | 已知差异 / 注意 |
|---|---|---|
| `run_train_bpe` / `get_tokenizer` | ch01/03–06、`codebot|storybot/tokenizer.py`、ch04 加速版、ch07 | CS336 有速度测试（<1.5 s）和 tiktoken 对拍；dlfs6 的 `find_chunk_boundaries` 与 CS336 `pretokenization_example.py` 算法相同（4096 字节窗口，`sorted(set(...))`），跨窗口的 special token 会被漏掉、改落到下一个位置【源码】。合并次数 = V − 256 − 1（codebot/storybot/webbot 的 merge_rules 分别有 743/9743/49743 条）【实测】。**频率并列时的取舍**：dlfs6 用 `max(..., key=(count, id1, id2))` 按 token id 比较，CS336 题面要求按字节串的字典序取较大者；两者在单字节阶段一致，出现合并 token 后可能分歧，CS336 的 `test_train_bpe`（merges 必须与参考完全一致）可能因此不过【待实测】 |
| `run_softmax`、`run_scaled_dot_product_attention` | ch02/02–07 | dlfs6 ch02 是逐步搭；WebBot 用 `F.scaled_dot_product_attention` |
| `run_multihead_self_attention(_with_rope)`、`run_rope` | ch02/08、ch05/01 | RoPE 实现采用偶/奇维交错旋转（`x[...,0::2]`、`x[...,1::2]`），接 CS336 测试前要确认两边的维度配对约定一致【待实测】 |
| `run_rmsnorm`、`run_silu`、`run_swiglu` | ch05/02–03 | WebBot 用 `nn.RMSNorm`，ch05 是手写 |
| `run_transformer_block`、`run_transformer_lm` | ch02/10、ch05/04、storybot/webbot model.py | 权重命名不同，adapter 负责搬运；CodeBot 是 GPT-2 式 LayerNorm + 学习式位置编码 |
| `run_get_batch` | ch06/05、ch09 `get_batch` | 一致思路：随机起点切片，y 右移一位 |
| `run_cross_entropy`、`run_gradient_clipping` | 各训练脚本、`clip_grad_norm_` | — |
| `get_adamw_cls` | ch06/02 手写 AdamW | 可加一条「与 `torch.optim.AdamW` 对拍」（nanochat 的写法） |
| `run_get_lr_cosine_schedule` | ch06/03 `get_lr` | **dlfs6 是 warmup + 线性衰减，CS336 是余弦**，测试必然不同，应在对照页明确标出 |
| `run_save_checkpoint` / `run_load_checkpoint` | `GPT.save` / `GPT.load_from` | — |

> 用法：写一个 `adapters_dlfs6.py`，把 dlfs6 的实现接到 CS336 A1 的测试上（L1 本地）。对照页逐条列出每个测试是通过、失败（附原因，比如余弦与线性）还是不适用。这比「我们也讲了 RoPE」有力得多。【待实测】

---

## 4. 交互模式目录（Pattern Catalog）

每条按「是什么 → 出处 → 用在 dlfs6 哪里 → 实现要点 → 坑」写。原则：**每个交互都必须绑定一句可验证的声明**，交互只是让读者亲手跑这次验证。

### P01 双轨运行单元（浏览器运行 / 复制本地命令）
- **是什么**：代码块下方两个按钮：「在浏览器运行」（L0）和「复制本地命令」（`uv run python ch01/01_char_tokenizer.py`），输出旁边并排显示期望值。
- **出处**：CS336 可执行讲义、marimo WASM、Quarto Live、d2l/Raschka/UDL 的 Colab 按钮。
- **落点**：ch01 全章（纯 Python + `regex`，Pyodide 内置）；ch02 的 soft dictionary 和电影类型 Q/K 例子（NumPy）；ch06/04 fp16/bf16（NumPy 能复现金标【实测】）。
- **实现要点**：Pyodide 放在 Web Worker 里；预加载 numpy/regex；单元设超时；本地命令统一带 `PYTHONUTF8=1`（Windows 控制台是 GBK）。
- **坑**：Pyodide 没有 torch，也没有多进程。ch04/04、ch07 的 `multiprocessing.Pool` 只能展示录制结果，或改成单进程小样本。

### P02 可拖动数字 / 参数旋钮（scrubbable numbers）
- **是什么**：正文里的数字可以拖，图和结论实时变化。
- **出处**：Bret Victor 的 Tangle、distill Momentum 的 α/β 控件、Transformer Explainer 的温度旋钮。
- **落点**：ch02/03 维度 d 与 q·k 方差（不缩放时方差约为 d，除以 √d 后约为 1）；ch03/02 温度；ch05/01 RoPE 的 θ 与位置差；ch06/03 warmup 与 max_iters；ch06/08 DPO β；ch03/09 GRPO 的 ε 截断。
- **实现要点**：数字加虚下划线和「可拖」提示；显示范围刻度；学习率、θ 用对数刻度；带复位按钮。计算放在 JS 里（公式很小，不用 Python）。
- **坑**：可发现性差、缺参照（社区对 Tangle 的批评）。每个旋钮旁边要写它在验证哪一句话。

### P03 先预测，再揭晓（predict → reveal）
- **是什么**：读者先填一个数、画一条线或选一个选项，揭晓的答案由运行单元现场算出，不写死在页面里。
- **出处**：distill 总结的「促进自省」、NYT You Draw It、How to Scale Your Model 的折叠答案、HF 测验。
- **落点（都有可算的答案）**：词表 1000 的初始损失是多少（ln 1000 ≈ 6.91，实测 6.99）；`aaabdaaabac` 做 3 次合并后的 id 是什么；`fp16(70000)` 等于多少（inf）、`bf16(70000)` 呢（70144）；WebBot 每个 token 的 KV 占多少（12 KiB）；CodeBot 把数据看了几遍（约 61 遍）；8 卡会快 8 倍吗（不会）；GRPO 首步 loss 是多少（见 P16）。
- **实现要点**：揭晓前锁定答案；猜测只存 localStorage；答错时给出「哪一步想岔了」的提示。
- **坑**：题太多会疲劳，每节最多 1–2 题。

### P04 单步执行器 / 步进动画（step-through）
- **是什么**：把一个算法的执行录成状态序列，用滑块或「下一步」回放，并同步高亮对应的代码行。
- **出处**：Bycroft 的 Continue/空格推进与悬停公式；Alammar 的逐层放大；CS336 trace viewer。
- **落点**：BPE 训练（count_pairs → 选最高频 → merge → 新 id 256、257……）；按 merge 优先级编码（ch04/06）；因果 mask 填充（ch02/06）；KV cache 的 prefill 与 decode（ch05/05）；GRPO 的组采样 → 奖励 → 优势；梯度累积的 micro-step 与 DDP all-reduce（ch09）。
- **实现要点**：状态序列由 Python 生成（浏览器内现跑，或构建期导出 JSON），前端用 2D SVG 回放；支持键盘操作。
- **坑**：不要做 3D/WebGL。动画只服务于「状态如何变化」。

### P05 Token 着色与 id 对照（tokenizer playground）
- **是什么**：输入任意文本，看它被切成什么、id 是多少、压缩率如何。
- **出处**：Tiktokenizer、Karpathy tokenizer 课、minbpe。
- **落点**：ch01 字符、字节、BPE 三种切法；CodeBot（V=1000）、StoryBot（V=10000）、WebBot（V=50000）与 GPT-2 并排；special token `<|endoftext|>` 不参与合并（对应 CS336 测试）；中文和 emoji 的多字节被切开的样子（半个字符解码成 `�`）。
- **实现要点**：dlfs6 的 tokenizer 在 Pyodide 里直接跑；GPT-2 金标用 Pyodide 的 tiktoken 或 JS 的 gpt-tokenizer；空白字符要显式画出来。
- **坑**：Pyodide 里的 tiktoken 首次要下载词表，可能被 CORS 拦住，需要预置文件【待实测】。

### P06 注意力 / 概率热力图 + 悬停细节
- **是什么**：小矩阵热力图，悬停显示具体数值和算式。
- **出处**：Transformer Explainer、Alammar 的 masked attention 百分比。
- **落点**：ch02/02 电影类型的 K/Q（数字小，读得懂）；ch02/06 mask；ch02/08 每个头一张图；ch03/02 下一个 token 的分布随温度变化；ch06/08 DPO 的 chosen/rejected 对数概率差。
- **实现要点**：矩阵最多 16×16；色板要对色弱友好；悬停时显示 `exp/Σexp` 的展开式。
- **坑**：注意力图容易被过度解读。要写明「这是权重，不是解释」。

### P07 代码 ⇄ 公式 ⇄ 图 三联对照
- **是什么**：公式、代码、示意图三栏并排，悬停任一符号，另外两处同时高亮。
- **出处**：Annotated Transformer、CS336 讲义、picoGPT 形状注释。
- **落点**：softmax(QKᵀ/√d)V ↔ ch02/07；RoPE 旋转 ↔ ch05/01（`x_even*cos − x_odd*sin`）；RMSNorm ↔ ch05/03；AdamW 更新 ↔ ch06/02；DPO loss ↔ ch06/08；GRPO 截断目标 ↔ ch03/09。
- **实现要点**：公式用 KaTeX；代码块在构建期从源文件按行号抽取，不手抄（单一源）；三处共享同一个 id。
- **坑**：代码一改就会错位，所以必须构建期抽取 + CI 检查。

### P08 形状追踪器（shape tracer）
- **是什么**：每执行一行，显示张量形状；维度可调，并实时检查约束（比如 `embed_dim % n_head == 0`）。
- **出处**：picoGPT 的 `[n_seq, n_embd]`、Karpathy 的 B/T/C、CS336 einops 讲义、dlfs6 ch02/08 自己就把 B/C/E/H/D 写成常量。
- **落点**：ch02/08 多头注意力的 view/transpose；ch09 GQA 的 `(B, KV, C, D)` 与 `enable_gqa=True`；KV cache 第三维随步数增长（`k_cache` 形状 `(1, 2, 12, 16)`【实测】）。
- **实现要点**：默认选互不相同的维度（CS336 fixture 就用 n_queries=12、n_keys=16），这样搞错转置会立刻报错或露馅。
- **坑**：无。属于低成本、高回报。

### P09 等价实现阶梯（同一结果，多种写法）
- **是什么**：一张表，每行一种实现，旁边是与参考实现的 `max|diff|` 徽章和耗时。
- **出处**：Karpathy 推 masked attention 的几种写法 + `allclose`；Raschka 的 cache/无 cache 逐 token 相等；llm.c 的 C 对 PyTorch；picoGPT 对 OpenAI；nanochat 的 AdamW 对 torch 参考。
- **落点**：循环版注意力 ↔ 矩阵版 ↔ `F.scaled_dot_product_attention`；手写 AdamW ↔ `torch.optim.AdamW`；KV cache 增量解码 ↔ 整段前向（最大误差约 1e-7【实测】）；GQA 在 KV=H 时 ↔ MHA；手写 LayerNorm/RMSNorm ↔ `nn.LayerNorm`/`nn.RMSNorm`；朴素 BPE 编码 ↔ ch04/06 优化版 ↔ ch04/07 并行版（id 序列必须完全相同）。
- **实现要点**：容差按 dtype 设（fp32 约 1e-6，bf16 约 1e-2）；GPU 上的非确定性要单独说明。
- **坑**：容差宽到什么都能过，就等于没测。要写明为什么选这个容差。

### P10 失败表：「挪动这一行会怎样」
- **是什么**：每行是一处改动，配一个让读者先猜后果的提示、观察到的结果（现跑或录制）和原因。可以用开关一键切换。
- **出处**：Karpathy 的训练菜谱、build-nanogpt 勘误、Raschka 的 `test_context_overflow_bug`、GPU Puzzles 的「期望 vs 你的」。
- **dlfs6 现成素材**（没有标注的行是教科书结论，落地时每一行都要配现跑或录制的证据）：
  - 去掉 `/√d`（ch02/03）→ softmax 饱和，梯度变小；
  - 去掉因果 mask → 训练 loss 低得可疑（看到了答案），生成却崩；
  - 梯度累积时忘了 `loss / accumulation_steps` → 等效学习率放大 N 倍；
  - fp16 不配 GradScaler → 小梯度下溢为 0（ch06/04 的 `1e-8 → 0`）；
  - **`no_sync()` 只包住 backward**（ch09 当前写法）→ PyTorch 文档写明「前向也要在上下文里，否则梯度仍会同步」，结果仍然正确，但每个 micro-step 都白付一次 all-reduce【源码】。多卡上可用 comm hook 计数验证【待实测】；
  - **cache 解码超过 `max_context_len`** → RoPE 位置被钳在 M−1（`rope(offset=40) == rope(offset=31)`），StoryBot 的 cache 无限增长（48 > 32），WebBot 截断到 32【实测】；
  - 推理时忘了 `model.eval()` → dropout 还开着，输出抖动（CodeBot 有 dropout 0.1）；
  - **GRPO 中 policy 处于 train 模式、old policy 处于 eval 模式**（ch03/09 当前写法）→ 权重相同，ratio 仍带 dropout 噪声：均值 0.91、标准差 0.58，33% 的 token 首步就被截断；`model.eval()` 后 ratio 恰为 1【实测，CodeBot 检查点】。对 GRPO 准确率曲线的实际影响需对照实验【待实测】；
  - 关掉梯度裁剪 → loss 尖峰（需要录制运行【待实测】）。
- **坑**：只能在 GPU 上看到的现象必须用录制产物，并清楚标明「录制」。

### P11 训练曲线回放（loss curve replay）
- **是什么**：真实训练日志的回放，可叠加多次运行和读者自己的运行，并标出关键时刻。
- **出处**：nanochat 的 wandb 图和 val_bpb、llama2.c 的损失表、Ultra-Scale 的实验图、Modal 示例的 TensorBoard、picotron 的 loss 对齐图。
- **落点**：ch03/01、ch06/05、ch09 的预训练；SFT、DPO、GRPO 的曲线；在 step 0 标出 ln V，标出 warmup 结束点和学习率衰减段。
- **实现要点**：日志统一成 JSONL 产物，附运行元数据（git sha、GPU、种子、torch 版本）；带平滑滑块；支持把自己的 `loss.jsonl` 拖进来，与参考带对比。
- **坑**：绝不伪造曲线；参考曲线要标出种子方差带。

### P12 样本时间轴（同一 prompt，不同检查点）
- **是什么**：固定 prompt 和种子，用滑块切换检查点，看生成质量怎么变。
- **出处**：build-nanogpt 展示 10B/40B tokens 时的样本；llama2.c 的样例。
- **落点**：StoryBot 现成有 `model_iter_500.pt`、`model_iter_5000.pt`、`model_pretrain.pt`、`model_dpo.pt`（download.sh 中列出，本地已有前三个）；CodeBot 的 pretrain → SFT → GRPO。
- **实现要点**：生成结果在构建期预算成 JSON（temperature 0 一条，再加若干固定种子的采样）。
- **坑**：挑样本会误导读者，要固定 prompt 集、每个检查点展示多条。

### P13 估算器 / 计算器（参数、FLOPs、显存、KV、费用）
- **是什么**：输入配置，算出参数量、6ND、训练显存、KV cache、在某张卡上的预计时间和费用；旁边一列是实测值和误差百分比。
- **出处**：Ultra-Scale 的显存 widget、Transformer Math 101、kipply、How to Scale Your Model。
- **落点**：ch05 KV cache（GQA 12 KiB 对 MHA 36 KiB）；ch06 混合精度显存；ch09 WebBot 在 3060 Ti / Modal / RunPod 上的时间和费用；gbs = mbs × accum × world_size。
- **实现要点**：公式放在 JS 里；实测列从运行产物读取（`torch.cuda.max_memory_allocated`、tokens/s）。
- **坑**：激活显存的公式依赖实现（用了 SDPA/flash 就没有 5·a·s/h 那一项），所以公式必须和实测并列，不能单独出现。

### P14 隐藏答案的计算题
- **是什么**：章末 2–4 道计算题，答案折叠；展开后附一个能复算的运行单元。
- **出处**：How to Scale Your Model 的 Worked Problems、UDL、HF 测验。
- **落点**：「WebBot 每步训练多少 token？」（每个 rank 131,072）；「BPE 要合并多少次？」（V − 256 − special token 数）；「GQA 省了多少 KV？」；「DPO 的 β 变大，隐式奖励的尺度怎么变？」。
- **坑**：答案要由代码算，不能手写。

### P15 渐进式代码 diff（diff 就是课程）
- **是什么**：相邻两个版本的统一 diff，配讲解和一条回归或等价测试。
- **出处**：microgpt 的 train0→train5、build-nanogpt 一 commit 一概念、Saito DLFS3 的 step 结构。
- **落点**：dlfs6 自己就是一条 diff 链：CodeBot（GPT-2 式）→ ch05/04（RoPE + SwiGLU + RMSNorm）→ ch05/05（+KV cache）→ WebBot（+GQA、SDPA、DDP、compile）。
- **坑**：diff 太大就失去意义，每一步只引入一个概念。

### P16 指标解码器（training-log metric decoder）
- **是什么**：点训练面板上的任何一个指标，弹出它在代码里由哪一行算出、理论上该怎么变、一个可拖动的玩具模拟。
- **出处**：文献里基本没有，是这次调研发现的空白。nanochat 只解释了 val_bpb、CORE、MFU。
- **落点**：
  - ch03/09 GRPO：奖励/准确率、ratio、截断比例、优势的方差、零方差组比例、loss；
  - ch06/08 DPO：chosen/rejected 的 logratio、margin、隐式奖励准确率；
  - 预训练：loss、val loss、lr、grad norm、tokens/s。
- **关键例子（已验证）**：dlfs6 的 `grpo_loss` 是 token 求和、除以样本数，优势 = r − mean(r)（不除以标准差）。
  - **理论情形**：每批生成后的第一次更新，新旧策略相同，ratio = 1，所以 **loss = −Σᵢ Aᵢ·Lᵢ / n**（Lᵢ 是回答长度），即奖励与回答长度的组内协方差取负。各回答等长时 loss 正好为 0，但梯度范数不为 0（玩具实验：loss 0.00000，|grad| 0.29；不等长时 loss −1.40625，与公式完全一致【实测】）。所以 loss 的数值不是学习信号，要看准确率、截断比例、零方差组比例。
  - **dlfs6 实际情形**：`model` 没有切到 eval，dropout=0.1 一直开着，而 `old_model` 在 eval 模式。于是即使权重相同，ratio 也带噪声（CodeBot 检查点上均值 0.91、标准差 0.58、33% 的 token 首步越出 [0.8, 1.2]），loss 和截断比例的读数都会被这层噪声污染【实测】。这正好说明为什么要先弄清「理论上该是多少」，再读日志。
  - 组内奖励全相同时 Aᵢ ≡ 0，这一组既不贡献 loss 也不贡献梯度。dlfs6 不除以标准差，所以不会出现 NaN。
- **坑**：每个解释都要连回代码行，不能泛泛而谈。

### P17 成绩单 / 速通榜
- **是什么**：每次运行生成一张卡：参数量、看过的 token 数、墙钟时间、费用、GPU、val loss、样本、任务指标。
- **出处**：nanochat 的 report.md、modded-nanogpt 的纪录史、CS336 排行榜、llama2.c 的模型表。
- **落点**：三台机器人各一张；CodeBot 加 GRPO 准确率；StoryBot 加 LLM-as-judge 分数；ch04/ch07 的 BPE 加速可以做个人速通榜。
- **实现要点**：运行脚本输出 `report.json`，页面只负责渲染；参考卡与「你的卡」并排。

### P18 位级浮点探索器
- **是什么**：显示 fp32/fp16/bf16 的符号、指数、尾数位；拖动数值，高亮上溢和下溢。
- **出处**：Ciechanowski《Exposing Floating Point》和 float.exposed。
- **落点**：ch06/04。NumPy 就能复现全部金标：`fp16 1000+0.01 = 1000`、`fp16 1e-8 = 0`、`fp16 70000 = inf`、`bf16 1e-8 ≈ 1.0012e-08`、`bf16 70000 = 70144`（bf16 用 float32 位运算做就近偶数舍入来模拟）【实测】。
- **坑**：bf16 舍入模式要写明（就近偶数），否则与 torch 对不上。

### P19 标签页里训练的小模型
- **是什么**：在浏览器里从零训练一个极小模型，看 loss 从 ln V 降下来，再采样。
- **出处**：microgpt（纯 Python，约 4k 参数，约 1 分钟）、Sequence Toy / TabGPT（WebGPU），以及 Pope & Tedre 2026 的教学证据。
- **落点**：ch03 预训练的预演：用 tiny_codes 的一小段训练一个迷你 GPT，然后说明「同一套代码放大规模就是 CodeBot（GPU 上跑）」。
- **实现要点**：推荐 NumPy 写的迷你自动微分。它和 Saito《ゼロから作る❸》的 DeZero 一脉相承，读者熟悉。参数控制在 10⁴–10⁵；用数值梯度检查验证反传；前 3 步的 loss 与 PyTorch 在同一初始化下导出的金标对拍。
- **坑**：Pyodide 计算慢，所以模型要小；要让读者知道这是玩具，不是 CodeBot 本身。

### P20 细节按需 / 双语术语卡
- **是什么**：术语悬停出卡片（英文原词 + 中文释义 + 首次出现的章节）；数学推导默认折叠，标「深挖」。
- **出处**：distill 的细节按需与 StretchText、Alammar 的脚注。
- **落点**：全书。例如 RoPE 的复数推导、DPO 由 Bradley–Terry 推导、GRPO 的优势估计。
- **坑**：主线必需的步骤不能折叠；折叠内容要支持键盘访问。

### P21 单一源构建（讲义即程序）
- **是什么**：每一课的源文件是 `lesson_xx.py`（或带代码块的 md），构建时执行一遍，生成 HTML 并顺便跑完全部断言。
- **出处**：CS336 的 `execute.py` + trace viewer、Annotated Transformer 的 jupytext、d2l 的 `#@save` + `d2lbook eval`、Raschka 的 nbval CI。
- **落点**：教程本身的构建流水线（见 V12）。

---

## 5. 验证机制目录（Verification Mechanisms）

### 5.0 两个约定

**验证卡（每个技术声明一张）**

```
✔ 验证卡 · <短标题>                                        [L0 浏览器 | L1 本地CPU | L2 本地GPU | L3 云GPU]
声明：  <一句话技术声明>
检查：  <运行单元 / 本地命令 / 测试名>
期望：  <金标值、性质或容差>
出处：  <dlfs6 文件:行 | CS336 测试名 | 外部参考链接>
若失败：<最常见的原因>
```

**数字必须有出处**：页面上的每个数字，要么由运行单元现场算出，要么来自带溯源信息的产物文件（V14）。正文里不出现手抄的实验数字。

### V01 断言单元（assert cells）
CS336 讲义的写法：叙述一句，紧跟一个 `assert`，例如 `assert decode(encode(s)) == s`、`assert chr(97) == "a"`。我们的每节至少 1 条，L0 优先。

### V02 注释金标 → 断言（dlfs6 专属，成本最低）
dlfs6 脚本中有 35 行 `print(...)  # 期望输出`（ch01 21 行、ch02 3 行、ch05 2 行、ch06 9 行）。用 AST 把 print 换成记录函数，执行后与注释比对：首轮 31 条中 29 条直接通过，另外 2 条只是表示形式不同（`defaultdict(<class 'int'>, {...})` 对 `{...}`，`torch.Size([2, 10, 512])` 对 `(2, 10, 512)`）【实测】。落地时加一层归一化（转成 `dict`/`tuple` 再比），或改写成显式断言。

### V03 快照测试（golden tensors）
照 CS336 的 `NumpySnapshot`：在 L1 用 PyTorch 跑参考实现，导出 `.npz`（固定种子、选互不相同的维度）；浏览器里的 NumPy 移植版与之比对（fp32 用 rtol 1e-4 / atol 1e-2 起步，按模块收紧）。这样 L0 的移植代码就有了「与 dlfs6 原版一致」的证据。

### V04 参考实现对拍
- tokenizer：与 tiktoken 对拍（minbpe、CS336 都这么做）。dlfs6 的 GPT-2 预分词正则与 GPT-2 相同，可以先对拍预分词结果，再对拍完整编码（dlfs6 自训的 merges 不同，不能直接比 id）。
- 优化器：手写 AdamW 与 `torch.optim.AdamW` 对拍（nanochat 用 `assert_close(rtol=1e-5, atol=1e-6)`）。
- 模型：Raschka 的做法是极小配置下与 HF transformers 比 logits；llm.c 的做法是导出 debug state 再比 logits、loss、梯度和若干步优化。
- 跨课程：dlfs6 接入 CS336 A1 测试（3.3 节）。

### V05 等价性与不变性测试
- 增量解码（KV cache）== 整段前向（StoryBot / WebBot 最大误差约 1e-7【实测】）；
- KV=H 的 GQA == MHA；
- 朴素 BPE == 优化版 == 并行版（id 序列完全相同）；
- 1 卡与 N 卡的 loss 曲线一致（picotron 的验收方式，L3）；
- 同种子两次运行逐位相同（借 nanochat 的 `test_determinism` 思路；它在 CUDA 上测，我们放到 CPU 上做，避开 GPU 非确定性）。

### V06 梯度检查
Karpathy 的 `cmp()`（exact / approximate / maxdiff）；Saito《ゼロから作る》1 和 3 里的数值梯度确认；`torch.autograd.gradcheck`（用 double）。适用于 P19 的迷你自动微分和 ch06/02 手写优化器。

### V07 性质测试（property-based）
- softmax 每行和为 1，平移不变（`softmax(x + c) == softmax(x)`）；
- RoPE 的相对位置性质：`⟨R_m q, R_n k⟩` 只依赖 m − n；
- RMSNorm 对输入缩放不变；
- 因果性：`∂ out_t / ∂ x_{t+1} == 0`（Karpathy 菜谱里的「用反传检查依赖」）；
- 往返：`decode(encode(s)) == s`，包括空串、单字符、中文、emoji；
- 可以用 hypothesis 生成输入（Tensor Puzzles 的做法）。

### V08 训练健全性
- 初始损失约等于 ln V（三台机器人都已验证【实测】）；
- 单 batch 过拟合到接近 0；
- 与输入无关的基线（输入置零后 loss 应变差）；
- 加容量后训练 loss 下降。

### V09 性能与资源断言
- BPE 训练速度（CS336：corpus.en、V=500，< 1.5 s，参考实现约 0.38 s），用于 ch04 加速前后对比；
- 内存上限（CS336 在 Linux 上用 `RLIMIT_AS` 限 1MB 跑 `encode_iterable`，Windows 不可用，需标注）；
- 峰值显存与估算器对比、tokens/s、MFU。

### V10 带容差的结果金标
不写「val loss = 1.234」，写「第 N 步的 val loss 落在 [a, b]」，区间由参考运行（多个种子）得出，并记录 GPU 和 torch 版本。参考：llama2.c 的损失表、nanochat 的 CORE 门槛（GPT-2 = 0.256525）。

### V11 自测题与隐藏答案
Quarto Live 的 check cell（exercise/hint/solution）、HF 章末测验、How to Scale Your Model 的折叠答案。答案由代码算。

### V12 CI 守护
- 构建时执行全部 lesson 源文件（d2lbook eval / CS336 execute.py 的做法），任何断言失败都让构建失败；
- 无头浏览器跑一遍所有 L0 单元（Playwright + Pyodide）；
- L1 测试在 Windows 与 Linux 上各跑一次（Raschka 的多 OS 矩阵）；
- 链接检查。

### V13 评测与裁判
- CodeBot 的 GRPO 奖励是可验证的（取最后一个整数与答案比对），本身就是 verifier；
- StoryBot 的 LLM-as-judge（ch06/07、09）要用已知优劣的样本对校准，并报告评委的一致率；
- 像 MiniMind 那样诚实标注小模型在通用榜单上接近随机。

### V14 运行产物溯源（provenance）
每次 L2/L3 运行输出：`report.json`（git sha、配置、种子、GPU 型号、驱动/CUDA/torch 版本、墙钟时间、按单价计的费用、最终指标）、`loss.jsonl`、`samples.json`，ckpt 可选。网页只读这些文件。Modal 上用 Volume 保存，`modal volume get` 取回（这也正好回答了过往会话里「训练完怎么拿回 ckpt」的问题）。

---

## 6. 端到端示例设计

### E1「标签页里的 CodeBot-mini」（只用浏览器，约 10 分钟）
- **目标**：不装任何东西，走完「文本 → BPE → GPT → 训练 → 采样」，每一步都有验证卡。
- **流程**：取 tiny_codes 的一小段（几百 KB）→ 在 Pyodide 里直接调用 dlfs6 `codebot/tokenizer.py` 的 `train_bpe`（纯 Python + regex）→ 用 NumPy 迷你自动微分搭一个约 5 万参数的 GPT → 先测初始损失约等于 ln V → 单 batch 过拟合 → 训练几百步 → 采样。
- **验证**：V01、V02（ch01 注释金标）、V06（数值梯度）、V08；前 3 步的 loss 与 PyTorch 同初始化导出的金标对拍（V03）。
- **模式**：P01、P03、P04、P05、P11、P19。

### E2「一个 prompt 的一生」（浏览器推理真模型）
- **目标**：用真正训练好的 CodeBot（`model_pretrain.pt` 约 44.5MB fp32，转成 fp16 `.npz` 约 22MB），在浏览器里用 NumPy 前向，所有中间量都能查看。
- **流程**：prompt → token 着色 → embedding + 位置编码 → 逐层逐头的注意力热力图 → MLP → logits → 温度/top-k 采样。每一层输出都可以展开查看形状和数值。
- **验证**：浏览器里的 logits 与本地 PyTorch 导出的快照比对（V03/V04）；temperature 0 下生成的 token 序列与本地完全一致（llama2.c 的 argmax 对拍）。
- **模式**：P04、P06、P07、P08、P02（温度）。

### E3「三台机器人的成绩单」（GPU 主线）
- **目标**：把 ch03（CodeBot）、ch06（StoryBot）、ch09（WebBot）的训练都变成能复现、有收据的运行。
- **GPU 运行卡**（每个训练脚本一张）：

```
⚙ GPU 运行卡 · ch06 StoryBot 预训练
规模（可复算）：22.70M 参数；40,000 步 × 32 × 256 = 3.28 亿 tokens；6ND ≈ 4.46e16 FLOPs
目标：val loss 落在 [参考值 ± 容差]（参考值由首次参考运行产生）       【待实测】
在哪跑：本地 RTX 3060 Ti 8GB ｜ Modal ｜ RunPod / Lambda（SSH + uv sync + 脚本）
带回：report.json、loss.jsonl、samples.json、（可选）ckpt
页面：成绩单 + 曲线回放 + 与参考带对比 + 时间/费用计算器（用实测吞吐换算）
```

- **WebBot 的特别处理**：3060 Ti 只跑缩小版 smoke（层数、上下文、步数都减小），验证管线能跑通、loss 在下降；正式预训练放 L3。预测题「8 卡会快 8 倍吗」配 P13 计算器。DDP 的 `no_sync` 写法做成 P10 的一行，用 comm hook 计数验证【待实测】。
- **验证**：V08、V10、V14；多卡加一条「1 卡与 2 卡前若干步 loss 一致」（V05）。
- **注意**：3060 Ti 是 Ampere，支持 bf16、不支持 FP8；涉及 Hopper 专属内核的参考代码（nanochat 的 FA3、modded-nanogpt）不能在本机跑。

### E4「对拍 CS336」（交叉验证页）
- **目标**：用 CS336 A1 的测试检验 dlfs6 的实现，并把差异讲清楚。
- **流程**：克隆 assignment1-basics → 写 `adapters_dlfs6.py` 把 dlfs6 的函数接上 → `uv run pytest` → 页面逐条列出每个测试的结果和说明（比如学习率调度：线性对余弦）。
- **价值**：「CS336 交叉」从口头对照变成可执行对照；也顺带检验 BPE 速度（V09）。
- **验证**：V03、V04、V09。【待实测：接口映射是否都能接上】

### E5「后训练三部曲 + 指标解码器」（直接回应过往困惑）
- **目标**：CodeBot 的 SFT（ch03/03）→ GRPO（ch03/09），StoryBot 的 DPO（ch06/08）。每个指标都能点开解码。
- **组件**：
  - 优势计算器：拖动组内 8 个奖励，看优势和「零方差组」；
  - 「为什么 loss 不是学习信号」玩具：拖动回答长度，看 loss = −Σ Aᵢ·Lᵢ / n，同时梯度范数不为 0（P16 已验证的例子）；
  - 「dropout 开关」对照：同一批数据，`model.train()` 与 `model.eval()` 下的 ratio 分布和截断比例（已测 33% 对 0%），再用 L2 跑两条 GRPO 准确率曲线比较【待实测】；
  - DPO 面板：β、chosen/rejected 的 logratio 和 margin。
- **验证**：玩具模拟与真实 `grpo_loss` 函数在小输入上数值一致（V05）；GRPO 准确率曲线来自产物（V14）。
- **模式**：P02、P03、P11、P12、P16。

### E6「推理实验室」
- **内容**：KV cache 等价性（已实测）→ 计时（ch05/05 自带对比）→ GQA 的 KV 显存（12 vs 36 KiB/token）→ 超长上下文的失败表行（已实测）→ 选读：对照 nano-vllm 的 continuous batching 和 prefix cache。
- **模式**：P08、P09、P10、P13。

### E7「BPE 加速赛」
- **内容**：ch04 的四步（optimize → cache → chunk → parallel）各自计时，id 序列必须完全一致（V05），速度用 CS336 的 1.5 s 当参照（V09）；可以做个人速通榜（P17）。边界用例：构造一个 special token 正好跨 4096 字节窗口的文件，验证切分仍然落在 special token 上，只是位置后移（V07）。

---

## 7. 落到 Saito 式的渐进结构

Saito 的写法（另有专人分析原书，这里只取与交互设计相关的几点）：一本书一条主线项目（dlfs6 是 CodeBot → StoryBot → WebBot）；自底向上，每次只加一个概念；代码短，输出写在注释里；类比先行（ch02/01 用「soft dictionary」引出注意力，ch02/02 用电影类型当 key）；章末一段「まとめ」；能在 CPU 上跑。《ゼロから作る》1 和 3 已经有数值梯度确认这种「验证」习惯，可以顺着延伸。

### 7.1 每节模板

1. **开场问题**：一个能用代码回答的问题，例如「为什么要除以 √d？」
2. **最小代码**（≤ 30 行）+ P01 双轨运行单元。
3. **验证卡**（1–3 张），L0 优先。
4. **一个交互图**。一节只放一个，选最能说明问题的。
5. **失败表**（2–4 行，P10）。
6. **CS336 对照框**：同一概念在 CS336 第几讲、哪个 adapter、哪个测试。
7. **GPU 运行卡**：只在需要时出现。
8. **まとめ**：3–6 条，每条链接回对应的验证卡。

### 7.2 dlfs6 各章的模式分配

| 章 | 内容 | 主层级 | 主要交互 | 主要验证 |
|---|---|---|---|---|
| ch01 | 字符/字节/BPE、special token、预分词、tiny_codes | L0 | P01 P03 P04 P05 P15 | V01 V02 V04 V07 |
| ch02 | soft dictionary → 注意力 → 缩放 → mask → 多头 → LN/GELU → GPT-2 | L0（NumPy）+ L1 | P02 P06 P07 P08 P09 P10 | V03 V05 V06 V07 |
| ch03 | CodeBot 预训练、生成、SFT、对话、GRPO | L2 | P11 P12 P16 P17 P19 | V08 V10 V13 V14 |
| ch04 | BPE 加速（cache/chunk/parallel、并行编码） | L0（单进程）+ L1 | P04 P09 P13 P17 | V05 V09 |
| ch05 | RoPE、SwiGLU、RMSNorm、改进版 GPT、KV cache | L0 + L1 | P02 P04 P07 P09 P10 P15 | V05 V07 |
| ch06 | AdamW、LR 调度、混合精度、StoryBot 预训练、生成、LLM 评委、DPO | L0 + L2 + API | P02 P11 P12 P13 P16 P18 | V04 V08 V10 V13 |
| ch07 | WebBot 的 BPE（V=50000，8 进程，64 块） | L1（多核 CPU） | P13 P17 | V09 |
| ch08 | 书中无代码（内容待对照原书目录） | — | 视内容做 P13/P14 | V11 |
| ch09 | WebBot 预训练：DDP、GQA、compile、bf16 | L3（3060 Ti 只做 smoke） | P10 P11 P13 P14 P17 | V05 V09 V10 V14 |

---

## 8. 针对过往困惑的加强点（只到主题层面）

从过往会话的提问中，能看到几类反复出现的困惑，本教程恰好有对应的章节可以加强：

| 过往困惑（主题） | 教程中的落点 | 处理方式 |
|---|---|---|
| RL 日志里的 policy loss/pg_loss、entropy、grad_norm、rewards/mean 各是什么；为什么 policy loss 一直在 0 附近 | ch03/09 GRPO | P16 指标解码器 + P03 预测题。已验证：ratio=1 时 loss = −Σ Aᵢ·Lᵢ / n，等长时为 0，梯度不为 0；改看准确率、截断比例、零方差组比例。再加一个对照：dlfs6 的 dropout 让首步 ratio 就偏离 1（33% 被截断），说明「先知道理论值，再读日志」 |
| 奖励饱和后「方差消失」，比较变得嘈杂 | ch03/09 `advantages = rewards − rewards.mean()` | 优势计算器：全对或全错的组 A ≡ 0，不贡献梯度；显示每批零方差组的比例 |
| gbs 是什么；不同设置下的比较怎样才公平 | ch09（mbs 32 × accum 4 × world_size） | P13 计算器 + 「8 卡会快 8 倍吗」；公平比较按 token 数或样本数对齐预算 |
| 每组几个 rollout（2 还是 8）的取舍 | ch03/09 `group_size = 8` | 拖动 group_size，看优势估计的方差和零方差组的概率 |
| 训练完怎么把 ckpt 取回来 | GPU 运行卡 | V14：Modal 用 Volume 加 `modal volume get`；RunPod/Lambda 用 scp/rsync |
| 希望成品不像随手让模型迭代出来的东西（对视频审美的反馈） | 全站视觉 | 借 Welch Labs / 3Blue1Brown 的统一视觉语法；配色克制；不做模板化卡片墙 |

---

## 9. 反模式与风险

- **为交互而交互**：每个交互必须绑定一句可验证的声明（distill 指出的头号成本就是制作成本）。
- **重型可视化**：3D/WebGL（Bycroft 式）开发维护贵；几百 MB 的模型（Transformer Explainer 式）拖垮首屏。用我们自己的 11M CodeBot。
- **数字没有出处**：正文里手抄的实验数字会过期，一律从产物读取（V14）。
- **可拖数字没人发现**：要有视觉提示、范围刻度和说明（Tangle 的批评）。
- **浏览器环境假设**：Pyodide 没有 torch、没有多进程；tiktoken 的词表下载可能遇到 CORS【待实测】。
- **环境与硬件**：Windows 本机需要 CUDA 版 torch；3060 Ti 不支持 FP8，Hopper 专属内核（FA3 等）跑不了；WebBot 全量配置要上云。
- **版权**：书的正文只能少量引用（每处最多 1–2 句）。代码是 MIT，可以改写和引用；日文注释在教程副本中翻译，原仓库不动。
- **维护**：锁版本（Pyodide、gpt-tokenizer、torch）；构建即测试（V12）。
- **可访问性**：色板对色弱友好；动画可暂停；交互支持键盘操作；折叠内容不藏主线步骤。
- **模板化外观**：用户明确在意。统一字体、图形语言和配色，避免通用的「AI 产物感」。

---

## 10. 建议的 MVP 顺序与开放问题

**MVP 顺序**

1. **构建流水线**（P21 + V01/V02/V12）：lesson 源文件 → HTML + 断言执行；Pyodide worker；注释金标的归一化比对。
2. **ch01 完整做一章样板**：P01/P03/P04/P05 + V01/V02/V04；验证「L0 覆盖率」能做到多高。
3. **ch02 注意力**：P02/P06/P07/P08/P09/P10 + V03（NumPy 移植对 PyTorch 快照）。
4. **GPU 产物管线**（E3 的骨架）：先用 CodeBot 在 3060 Ti 上跑出第一份 `report.json` 和曲线回放。
5. **E4 对拍 CS336**、**E5 后训练三部曲**、**E6 推理实验室**。

**开放问题**

- L0 的模型推理用 NumPy（Pyodide）还是 JS（onnxruntime-web 或手写 WebGPU）？前者与 Saito 的 NumPy 传统和读者的 Python 背景一致，后者更快。建议先 NumPy，性能不够再换。
- 托管方式：静态站（GitHub Pages）还是 claude.ai Artifact？需要能放几十 MB 的权重和 Pyodide 资源。
- 参考运行的费用由谁出、跑几个种子（决定 V10 区间的宽度）。
- ch08 没有代码，需要对照原书目录决定做成估算器还是纯文字。

---

## 附录 A：本地实测记录

脚本都在 scratch（`C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\patterns-survey\`），全部只读导入 dlfs6。运行方式：

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
R=/c/Users/chenf/Projects/_refs/dlfs6
uv run --project $R --frozen python golden_numbers.py $R     # 参数量 / 初始损失 / KV 字节
uv run --project $R --frozen python golden_comments.py $R/ch01/0{1,2,3,4,5}_*.py $R/ch06/04_mixed_precision.py $R/ch05/01_rope.py
uv run --project $R --frozen python kv_equiv.py $R           # KV cache 等价性
uv run --project $R --frozen python ctx_overflow.py $R       # 超过 max_context_len 的行为
uv run --project $R --frozen python grpo_loss_toy.py         # GRPO loss 值与梯度
uv run --project $R --frozen python grpo_dropout_ratio.py $R # train/eval 模式不一致时的 ratio 噪声
uv run --no-project --with numpy python bf16_numpy.py        # NumPy 复现 fp16/bf16 金标
```

输出摘录：

```
codebot  params=  11.12M  loss@init=6.989  ln(V)=6.908  KV/token(bf16)=9.0KiB  KV@ctx=2.2MiB  AdamW-train-mem=0.17GiB
storybot params=  22.70M  loss@init=9.299  ln(V)=9.210  KV/token(bf16)=8.0KiB  KV@ctx=2.0MiB  AdamW-train-mem=0.34GiB
webbot   params= 152.32M  loss@init=10.959 ln(V)=10.820 KV/token(bf16)=12.0KiB KV@ctx=12.0MiB AdamW-train-mem=2.27GiB

01_char_tokenizer.py: 8 match, 0 mismatch | 02_byte_tokenizer.py: 7/0 | 03_bpe_train.py: 2/1 (defaultdict 表示)
04_bpe_tokenizer.py: 2/0 | 05_special_token.py: 1/0 | 04_mixed_precision.py: 9/0 | 01_rope.py: 0/1 (torch.Size 表示)

storybot: max|cached - full| = 1.19e-07 -> OK   k_cache shape (1, 4, 12, 16)
webbot:   max|cached - full| = 1.49e-07 -> OK   k_cache shape (1, 2, 12, 16)

storybot: rope(offset=40)==rope(offset=31) -> True; after 48 tokens: k_cache len=48, cache_offset=48
webbot:   rope(offset=40)==rope(offset=31) -> True; after 48 tokens: k_cache len=32, cache_offset=32

lengths=[6]*8:              loss=+0.00000  predicted -sum(A_i*L_i)/G=-0.00000  |grad|=0.2898
lengths=[9,3,9,3,3,9,3,3]:  loss=-1.40625  predicted -sum(A_i*L_i)/G=-1.40625  |grad|=0.5279

dropout_rate in ckpt: 0.1 | model.training: True      (codebot/model_pretrain.pt，8×128 个 token)
model in train (as in ch03/09) : ratio mean=0.912 std=0.578 outside[0.8,1.2]=33.1%
model in eval                  : ratio mean=1.000 std=0.000 outside[0.8,1.2]=0.0%

fp16 1000+0.01 = 1e+03 | fp16 1e-8 = 0.0 | fp16 70000 = inf | bf16 1e-8 = 1.0011718e-08 | bf16 70000 = 70144.0
```

源码核对：

- `.venv/Lib/site-packages/torch/nn/parallel/distributed.py`（torch 2.9.1）中 `no_sync` 的文档写着：「The forward pass should be included inside the context manager, or else gradients will still be synchronized.」dlfs6 ch09 只把 `loss.backward()` 包进了 `no_sync()`。
- CS336 `cs336_basics/pretokenization_example.py::find_chunk_boundaries` 与 dlfs6 `storybot/tokenizer.py::find_chunk_boundaries` 是同一算法。
- dlfs6 ch03/09 `grpo_loss`：`ratio = probs / (old_probs + 1e-8)`，按 token 求和后除以 `n_samples`；`advantages = rewards − rewards.mean()`。`model` 来自 `GPT.load_from`（`nn.Module` 默认 train 模式），循环里只在评估时切到 eval，随即切回 train；`old_model` 一直是 eval。CodeBot 的检查点保存了 `dropout_rate=0.1`。

## 附录 B：主要链接

Karpathy：https://karpathy.ai/zero-to-hero.html · https://github.com/karpathy/minbpe · https://github.com/karpathy/build-nanogpt · https://github.com/karpathy/nanochat · https://github.com/karpathy/llm.c · https://github.com/karpathy/llama2.c · https://karpathy.github.io/2026/02/12/microgpt/ · https://karpathy.github.io/2019/04/25/recipe/ · https://github.com/karpathy/LLM101n
Raschka：https://github.com/rasbt/LLMs-from-scratch · https://github.com/rasbt/reasoning-from-scratch
CS336：https://cs336.stanford.edu/ · https://github.com/stanford-cs336/spring2025-lectures · https://github.com/stanford-cs336/assignment1-basics
可视化：https://poloclub.github.io/transformer-explainer/ · https://bbycroft.net/llm · https://tiktokenizer.vercel.app/ · https://jalammar.github.io/illustrated-gpt2/ · https://distill.pub/2020/communicating-with-interactive-articles/ · https://ciechanow.ski/exposing-floating-point/ · https://microgpt.jtauber.com/ · https://sequence.toys/
系统：https://huggingface.co/spaces/nanotron/ultrascale-playbook · https://github.com/huggingface/picotron_tutorial · https://jax-ml.github.io/scaling-book/ · https://blog.eleuther.ai/transformer-math/ · https://kipp.ly/transformer-inference-arithmetic/
中文：https://github.com/Lordog/dive-into-llms · https://github.com/datawhalechina/happy-llm · https://github.com/jingyaogong/minimind · https://github.com/datawhalechina/tiny-universe · https://zh.d2l.ai/
基础设施：https://pyodide.org/en/stable/usage/packages-in-pyodide.html · https://docs.marimo.io/guides/wasm/ · https://r-wasm.github.io/quarto-live/ · https://github.com/niieani/gpt-tokenizer · https://modal.com/docs/examples/hp_sweep_gpt
