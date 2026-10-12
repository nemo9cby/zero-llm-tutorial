# 进度与交接（Memento）

最后更新：2026-10-11 深夜（会话 2766cf54）

## 现状一句话

全书 26 个页面都已写完：541 个验证单元在构建时与真实浏览器（Pyodide）中全部通过，站内链接全部有效。
现在正在跑对抗式审校工作流（`wf_2f0bb04c-e03`）。

## 已完成

1. **研究**（`research/`）：逐章运行原书代码的档案、官方目录、斋藤四本书的风格量化、CS336 2025/2026 对照、
   同类教程模式、前端与 GPU 可行性原型、用户过往困惑 22 条（`private/`，不入库）。
2. **基础设施**：
   - `tools/build.py`：构建即验证。验证单元在 Pyodide 镜像环境、只含声明文件的沙箱目录中执行；畸形 fence 拒绝构建；全站链接校验。
   - `tools/check_pages.mjs`：无头浏览器跑全部验证单元，检查报错、溢出、组件越界，`--widget-shots` 可截每个组件的明暗主题图。
   - `web/`：设计令牌、印章 UI、`_plot.js`、`_bpe.js`（与 Python 分词器逐 id 一致）、`_gpt.js`（与 PyTorch 对拍）、`_tinygpt.js`。
3. **全部页面**：封面（书名实时分词）、第 0 章、三部导读、第 1–9 章、附录 A–E、CodeBot 工作台、浏览器内训练、
   GPU 手册、CS336 对照、术语、勘误。
4. **真实运行**（`web/data/runs/`，均带溯源）：
   - `codebot-3060ti`：原书脚本原样，本机 3060 Ti。预训练 41.9 min（loss 0.619），SFT（0.094），GRPO 1h29m（准确率 86.4%）。
   - `codebot-grpo-evalfix-3060ti`：同一 GRPO，只加 `model.eval()`（更新时关闭 dropout），同一 SFT 起点：第 51 步 98.8%，之后 100%。
   - `storybot-h100`：Modal H100，原书脚本原样。val loss 1.304（CS336 目标 ≤ 1.45），16 分钟，1.17 美元；DPO 34 秒。
   - `webbot-probe`：Modal H100 跑真实训练的前 200 步。稳态 0.317 s/步（约 41 万 token/s，MFU 约 38%），0.27 美元。
   - `web/data/ch05/kv_timing_{cpu,cuda}.json`：batch 1 时 KV cache 在 3060 Ti 上受开销限制（每步约 20 ms，与长度无关），CPU 上提速 4.5 倍。
   - Modal 累计花费约 1.5 美元（上限 30 美元）。

## 进行中

- 审校工作流 `wf_2f0bb04c-e03`：15 组页面，每组技术审查 + 编辑/读者审查 → 修复者，最后全站一致性检查。

## 下一步

1. 汇总审校结果，处理被推迟的问题和对共享文件的修改请求。
2. 全量构建 + 全量浏览器检查 + 提交。
3. 可选：GitHub Pages 部署（需要用户同意后再推送）；发布为 Artifact 需要自托管 Pyodide（CDN 在 Artifact 的 CSP 下不可用）。

## 约定与决定

- 浏览器检查用 Playwright（`tools/check_pages.mjs`），不用 gstack browse。
- `site/`、`build/`、`runs/` 不入库；运行数据在 `web/data/runs/`；playground 只附带本书自己训练的权重（34 MB）。
- 子模块必须 `core.autocrlf=false`；Windows 上 `PYTHONUTF8=1`、`CUDA_VISIBLE_DEVICES=-1`（空串无效）。
- 含反斜杠的编辑不要经 heredoc 里的 Python 字符串（会被吞掉）；用 Edit/Write 工具。
