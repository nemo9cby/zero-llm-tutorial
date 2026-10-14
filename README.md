# 从零构建大语言模型 · 可验证的互动教程

依据斋藤康毅《ゼロから作る Deep Learning ❻ — LLM編》（O'Reilly Japan，2026）的全部源代码反向编写的中文互动教程。
原书代码以 git submodule 固定在 `third_party/dlfs6`（MIT 许可）。

教程的约定只有一条：**每一个技术断言都要能被最小化地验证**。

- **A 层**：浏览器内验证。页面上带印章的代码块可以直接在浏览器里运行（Pyodide 314.0.7，NumPy 2.4.6），输出与构建时逐字一致才会盖章。
- **B 层**：本地 PyTorch 测试（`verify/`），或带溯源信息的 GPU 运行记录（`web/data/runs/`）。

## 快速开始

在 Git Bash（Windows）或 Linux/macOS 的终端里按顺序执行。需要 git、[uv](https://docs.astral.sh/uv/) 和 Node.js。

**1. 克隆**，连同原书代码（子模块 `third_party/dlfs6`，固定在 commit `c9b6e2e`）：

```bash
git clone --recurse-submodules <本仓库地址> zero-llm-tutorial
cd zero-llm-tutorial
```

**2. 只在 Windows 上：让子模块保持 LF。** Git for Windows 常把 `core.autocrlf` 设为 `true`，检出时会把换行改成 CRLF；
第 4 章起的 BPE 训练器以二进制读语料，在 CRLF 版的 TinyCodes 上，第 2 条合并就成了 `\r\n`，之后的规则全都对不上（站内第 1 章 1.7.3 节）。

```bash
git -C third_party/dlfs6 config core.autocrlf false    # 关掉换行转换，以后的检出也保持 LF
git -C third_party/dlfs6 rm -r -q --cached .           # 已经检出的文件要重新检出：先清空子模块的索引，
git -C third_party/dlfs6 reset -q --hard               # 再按 LF 写回全部文件
git -C third_party/dlfs6 ls-files --eol | grep w/crlf  # 检查：应当没有任何输出
```

只执行 `git -C third_party/dlfs6 checkout -- .` 不够可靠：它只重写 git 认为改动过的文件，而 git 先看时间戳和大小。
刚克隆完时多半会重写；克隆过一阵、索引被 git 刷新过之后，CRLF 的文件就会被原样跳过。
`verify/test_ch00.py::test_checkout_alone_leaves_crlf_files` 在一份临时克隆上重现了这两种情况。

**3. Python 环境与数据：**

```bash
uv sync                                # Python 3.12 + PyTorch 等，版本由 uv.lock 固定（Windows 上从 cu130 索引装 CUDA 版）
uv run python tools/fetch_data.py      # 官方 checkpoint 与验证集，约 0.6 GB，放进 third_party/dlfs6/<bot>/
```

`fetch_data.py --list` 只列出文件；加 `--full` 还会下载训练语料（1.1 GB + 5.3 GB），自己训练 StoryBot、WebBot 时才需要。

**4. B 层验证**（pytest 用真实的 DLFS6 代码检查正文结论，CPU 就能跑）。Windows 上先切到 UTF-8 模式：
中文 Windows 的默认编码是 GBK，而原书读 SFT 数据的两个脚本用的是不带 `encoding` 的 `open()`（站内第 0 章 0.7 节）。

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # PowerShell：$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
export CUDA_VISIBLE_DEVICES=-1               # 可选：隐藏 GPU（Windows 会丢掉空字符串，写 "" 不起作用）
uv run pytest                                # 只跑一章：uv run pytest verify/test_ch03.py
```

整套测试在一个进程里会用到约 8 GB 内存（各章的模型在整个会话里缓存着）。内存紧张时按文件分开跑，
例如 `for f in verify/test_*.py; do uv run pytest "$f"; done`。

**5. 构建站点**，同时执行全部 A 层验证单元：

```bash
uv run python tools/build.py                 # 写到 site/；任何一条验证失败，或站内链接、锚点失效，构建就失败
uv run python -m http.server 8000 -d site    # 浏览 http://localhost:8000
```

只预览几页时写 `uv run python tools/build.py --only ch01,ch02 --out build/preview`，这时失效的链接只给警告。
验证单元在 uv 另起的 Python 3.14 + NumPy 2.4.6 隔离环境里执行，与浏览器里的 Pyodide 一致；第一次构建时 uv 会先下载它们。

**6. 浏览器检查与 JS 测试：**

```bash
npm install                                  # Playwright，以及 JS 测试用来核对 WebAssembly 的 wabt
npx playwright install chromium              # 下载无头浏览器
node tools/check_pages.mjs site              # 每页无报错、无横向溢出，全部验证单元在真实的 Pyodide 中盖章
node --test "tests/js/*.test.mjs"            # 交互组件里的 JS 实现（BPE、前向计算、训练器）与 Python/PyTorch 对拍
```

`check_pages.mjs` 后面可以只写几个页面（`site ch01 ch02`）；加 `--shots <目录>` 会保存桌面与手机宽度的截图。

**7. GPU**：入口是本机的 `gpu/run_local.py` 和 Modal 的 `gpu/modal_app.py`，见下一节。

## GPU

运行器把原书代码复制到 `runs/<运行名>/work/`，在副本里原样运行脚本，不碰 `third_party/dlfs6`；
运行报告（两个提交号、GPU 与版本、每个脚本的用时、退出码和最终指标、费用）、逐步指标和图写进 `web/data/runs/<运行名>/`，
页面上的训练曲线只从这里读。作业（脚本顺序、所需数据、少数非改不可的补丁）定义在 `gpu/jobs.py`，本机和云端共用。

```bash
# 本机，或租来的 Linux GPU 机器（RunPod、Lambda 的步骤见 gpu/runpod_lambda.md 与 gpu/setup_gpu_box.sh）
uv run python gpu/run_local.py --run codebot-mine --job codebot      # 预训练 → 生成 → SFT → GRPO
uv run python gpu/fetch.py storybot                                   # StoryBot 的训练数据（约 1.1 GB，已有的文件跳过）
uv run python gpu/run_local.py --run storybot-mine --job storybot    # 预训练 → 生成 → DPO

# Modal（uv 的 gpu 依赖组）
uv run --group gpu modal setup                                            # 浏览器登录，一次就好
uv run --group gpu python gpu/modal_app.py plan --job webbot --gpu H100   # 只打印计划和成本估算，不连接 Modal
uv run --group gpu modal run gpu/modal_app.py::prepare --bots storybot    # 数据从 Hugging Face 进 Modal 的 Volume
uv run --group gpu modal run --detach gpu/modal_app.py::launch --job storybot --gpu H100 --run storybot-mine --budget 4
uv run --group gpu python gpu/modal_app.py pull --run storybot-mine       # 结果写回 web/data/runs/
```

**预算闸门**：`launch` 启动任何东西之前，先打印估算（每个脚本的小时数、合计美元）；估算的上限超过 `--budget`（美元，默认 5），
它就拒绝启动。离线算账用 `uv run python gpu/costs.py --scenario webbot-x8-trap`（按假设的 MFU；探针的实测见站内 GPU 实战手册）。
`--run` 请起一个新名字，不要沿用已入库的运行名，免得新的结果混进或覆盖已入库的记录：页面上的验证单元读的正是这些记录。
运行器自己也有两道检查：工作树里有没提交的改动时（`web/data/runs/` 除外），它拒绝开跑，加 `--allow-dirty` 才照跑，
并在报告里记下 `git diff HEAD` 的 SHA-256；已经有报告的运行名，换到另一种机器或在本机与 Modal 之间换着续写时，它也拒绝，
并要求换一个新的运行名。

已入库的运行记录：

| 运行记录 | 硬件 | 实测 |
|---|---|---|
| `codebot-3060ti` | 本机 RTX 3060 Ti | 预训练 41.9 分钟，SFT 2.1 分钟，GRPO 89.7 分钟（准确率 86.4%） |
| `codebot-grpo-evalfix-3060ti` | 本机 RTX 3060 Ti | 同一个 SFT 起点，GRPO 更新前加一行 `model.eval()`：76.5 分钟，准确率 100% |
| `storybot-h100` | Modal H100 | 预训练 16 分钟（验证 loss 1.304），连同生成与 DPO 共 1.17 美元 |
| `webbot-probe` | Modal H100 | 200 步计时探针：218 秒、0.27 美元；外推全程约 10.4 小时、41–43 美元 |

## 目录

| 路径 | 内容 |
|---|---|
| `content/` | 各章源文件（Markdown + 自定义指令，见 `STYLE.md`） |
| `web/` | 模板、样式、运行时、交互组件、数据（`web/data/runs/` 是 GPU 运行记录） |
| `tools/` | 构建、验证单元执行器、浏览器检查、数据下载、权重与分词器导出 |
| `verify/` | B 层 pytest |
| `tests/js/` | 交互组件里 JS 实现的测试（`node --test`） |
| `gpu/` | 本机与云端运行脚本、作业定义、成本估算 |
| `third_party/dlfs6` | 原书代码（git submodule） |
| `research/` | 编写本书时的研究档案（逐章运行原书代码的记录、风格分析、CS336 对照） |
| `DESIGN.md` / `STYLE.md` / `PROGRESS.md` | 总设计、写作规范、进度交接 |

`site/`（构建产物）、`build/`（预览构建）和 `runs/`（本机运行的工作目录）都不入库。
