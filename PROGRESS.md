# 进度与交接（Memento）

最后更新：2026-10-11（会话 2766cf54）

## 已完成

1. **研究**（15 个代理，结果在 `research/`）：DLFS6 全部代码逐章运行与验证（`repo-*.md`）、官方目录
   （`dlfs6-toc.md`，含每节编号）、斋藤四本书的写作模式量化分析（`style-*.md`）、CS336 2025/2026 对照
   （`cs336-crossref.md`）、同类教程模式（`tutorial-patterns.md`）、前端与 GPU 可行性及原型（`tech-frontend-gpu.md`,
   `tech-kit/`）、用户过往困惑 22 条（`private/confusions.md`，不入库）。
2. **基础设施**：`tools/build.py`（构建即验证：所有验证单元在 Pyodide 镜像环境执行，失败即构建失败）、
   Pyodide 314.0.7 模块 Worker 运行时、印章 UI、`tools/check_pages.mjs`（无头 Chromium 在真 Pyodide 中跑全部验证）、
   `_plot.js`（经验证的配色）、`runcurve.js`（训练运行回放）、`gpu/run_local.py`（原样运行原书脚本并记录溯源）、
   `tools/fetch_data.py`、Windows 上的 CUDA torch（cu130 索引）。
3. **样板章节**：`content/ch01.md` 的 1.1–1.2 节（语气与组件用法的标准）。

## 进行中

- 写作工作流 `wf_6d878c2a-0ac`：16 个代理并行撰写全部页面（见 `DESIGN.md` §2 页面清单）。
- 本机 RTX 3060 Ti 上运行 CodeBot 全流程（`runs/codebot-3060ti/`，原书脚本原样运行：预训练 → 生成 → SFT → GRPO）。

## 下一步

1. 汇总写作结果，处理各代理的 `infra_requests` 与 `open_issues`。
2. 对抗式审校工作流：技术正确性（重跑每条断言）、斋藤风格与中文行文、困惑覆盖、CS336 事实核查。
3. GPU：用自己训练的 CodeBot 权重替换 playground 中的官方权重（`tools/export_codebot.py`）；
   Modal 上跑 StoryBot 预训练 + DPO 与 WebBot 计时探针（用户批准的上限 30 美元，每次启动前打印成本估算）。
4. 整站构建 + 全量浏览器检查 + 提交；可选：发布为 Artifact 或 GitHub Pages。

## 约定与决定

- 浏览器检查用 Playwright（用户本次会话选择，不用 gstack browse）。
- `site/` 是构建产物，不入库；运行数据在 `web/data/runs/`。
- 原书子模块在 Windows 上必须 `core.autocrlf=false`（已对子模块设置），否则字节级 BPE 会学到 `\r\n`。
- 本机 GPU 训练期间，其他实验一律用 CPU（`CUDA_VISIBLE_DEVICES=""`）。
