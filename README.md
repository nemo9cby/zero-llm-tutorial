# 从零构建大语言模型 · 可验证的互动教程

依据斋藤康毅《ゼロから作る Deep Learning ❻ — LLM編》（O'Reilly Japan，2026）的全部源代码反向编写的中文互动教程。
原书代码以 git submodule 固定在 `third_party/dlfs6`（MIT 许可）。

教程的约定只有一条：**每一个技术断言都要能被最小化地验证**。

- **A 层**：浏览器内验证。页面上带印章的代码块可以直接在浏览器里运行（Pyodide 314.0.7，NumPy 2.4.6），输出与构建时逐字一致才会盖章。
- **B 层**：本地 PyTorch 测试（`verify/`），或带溯源信息的 GPU 运行记录（`web/data/runs/`）。

## 快速开始

```bash
git clone --recurse-submodules <this repo>
cd zero-llm-tutorial
git -C third_party/dlfs6 config core.autocrlf false   # Windows：保持原书数据为 LF
uv sync                                               # Python 3.12 + PyTorch（Windows 上自动用 CUDA 版）
npm install                                           # 只为无头浏览器检查（Playwright）

uv run python tools/fetch_data.py                     # 官方检查点与验证集（约 0.6 GB）
uv run python tools/build.py                          # 构建 site/（同时执行全部验证单元）
uv run python -m http.server 8000 -d site             # 打开 http://localhost:8000
```

Windows 上运行 Python 时请设置 `PYTHONUTF8=1`（控制台默认 GBK）。

## 验证

```bash
uv run pytest                                    # B 层：用真实的 DLFS6 代码检查正文结论
node --test tests/js/                            # 交互组件里的 JS 实现（BPE、前向计算、训练器）与 Python 对拍
node tools/check_pages.mjs site                  # 无头浏览器：每页无报错、全部验证单元在真实 Pyodide 中通过
```

## GPU

| 任务 | 命令 | 参考耗时 |
|---|---|---|
| CodeBot 全流程（本机） | `uv run python gpu/run_local.py --run codebot-3060ti --script ch03/01_pretrain.py --script ch03/02_generate.py --script ch03/03_sft.py --script ch03/09_grpo.py` | RTX 3060 Ti：预训练约 40 分钟 |
| Modal / RunPod / Lambda | 见 `gpu/` 与站内“GPU 实战手册” | 见手册中的成本表 |

运行记录（溯源、指标、样例）写入 `web/data/runs/<run>/`，页面上的训练曲线只从这里读取。

## 目录

| 路径 | 内容 |
|---|---|
| `content/` | 各章源文件（Markdown + 自定义指令，见 `STYLE.md`） |
| `web/` | 模板、样式、运行时、交互组件、数据 |
| `tools/` | 构建、验证单元执行器、浏览器检查、数据下载 |
| `verify/` | B 层 pytest |
| `gpu/` | 本机与云端运行脚本 |
| `research/` | 编写本书时的研究档案（逐章运行原书代码的记录、风格分析、CS336 对照） |
| `DESIGN.md` / `STYLE.md` / `PROGRESS.md` | 总设计、写作规范、进度交接 |
