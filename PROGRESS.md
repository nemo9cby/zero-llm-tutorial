# 进度与交接（Memento）

最后更新：2026-10-12（会话 2766cf54）

## 现状一句话

全书 26 个页面都已写完，经过一轮对抗式审校和一轮后续修整（提交 `282f259`）：24 个可见页面共 571 个验证单元
（另有隐藏的组件样板页 1 个），在构建时与真实浏览器（Pyodide）中全部通过，站内链接全部有效。
第三轮收尾工作流已写好，等用户同意后启动（见“进行中”）。

## 已完成

1. **研究**（`research/`）：逐章运行原书代码的档案、官方目录、斋藤四本书的风格量化、CS336 2025/2026 对照、
   同类教程模式、前端与 GPU 可行性原型、用户过往困惑 22 条（`private/`，不入库）。
2. **基础设施**：
   - `tools/build.py`：构建即验证。验证单元在 Pyodide 镜像环境、只含声明文件的沙箱目录中执行；畸形 fence、控制字符拒绝构建；全站链接校验。
   - `tools/check_pages.mjs`：无头浏览器跑全部验证单元，检查报错、溢出、组件越界、KaTeX 错误，`--widget-shots` 可截每个组件的明暗主题图。
   - `web/`：设计令牌、印章 UI、`_plot.js`、`_bpe.js`（与 Python 分词器逐 id 一致）、`_gpt.js`（与 PyTorch 对拍）、`_tinygpt.js`。
3. **全部页面**：封面（书名实时分词）、第 0 章、三部导读、第 1–9 章、附录 A–E、CodeBot 工作台、浏览器内训练、
   GPU 手册、CS336 对照、术语、勘误。
4. **真实运行**（`web/data/runs/`，均带溯源）：
   - `codebot-3060ti`：原书脚本原样，本机 3060 Ti。预训练 41.9 min（loss 0.619），SFT（0.094），GRPO 1h29m（准确率 86.4%）。
   - `codebot-grpo-evalfix-3060ti`：同一 GRPO，只加 `model.eval()`（更新时关闭 dropout），同一 SFT 起点：第 51 轮 98.8%，第 131 轮起 100%。
   - `storybot-h100`：Modal H100，原书脚本原样。val loss 1.304（CS336 目标 ≤ 1.45），16 分钟，1.17 美元（容器计时，下限）；DPO 34 秒。
   - `webbot-probe`：Modal H100 跑真实训练的前 200 步。稳态 0.317 s/步（约 41 万 token/s；MFU 按 n_mm + Attention 口径 31%，
     按 6N 口径 38%），0.27 美元。外推整次训练约 10.4 小时（步 8.8 h + 101 次评估 1.6 h），约 41–43 美元。
   - `web/data/ch05/kv_timing_{cpu,cuda}.json`：batch 1 时 KV cache 在 3060 Ti 上受开销限制，CPU 上提速数倍（页面给出负载与区间）。
   - 本地 CPU 全量复现：第 1 章朴素 BPE 68 分钟、规则与 `merge_rules.pkl` 逐字节一致；第 4 章 2.2 GB 训练集的合并规则与
     `tiny_stories_train.bin` 都与官方文件逐字节一致。
   - Modal 累计花费约 1.5 美元（上限 30 美元）。
5. **审校**（`wf_2f0bb04c-e03`，提交 `ac0b37f`）：15 组页面 × 技术审查 + 编辑/读者审查 → 修复者 → 全站一致性检查。
   修复 741 条；驳回 1 条（gpu-R8：tqdm 计数沿用“每个计数取第一行”，四个页面按此引用轮次）；3 条因涉及共享文件而推迟，
   交给后续工作流。审校后全量关卡：构建通过、浏览器 26/26 页 572/572 单元、JS 236/236、pytest 396 通过 1 跳过
   （跳过的需要 `CS336_A1_DIR`，本机 A1 仓库的现场抽查）。

6. **后续修整**（`wf_57d8dd52-3e9`，提交 `282f259`）：五个文件互不重叠的负责人 + 四个复核者。
   共享前端（锚点留白、行内代码保留空格、窄屏公式编号、导航分组、`_plot.js` 主题助手与 WCAG 标签色、折线图刻度、
   代码清单行号、callout 标题行内 Markdown、隐藏页 noindex、OS 暗色截图）；GPU 运行器溯源（工作树与 diff 哈希、
   checkpoint 哈希、拒绝脏树与跨主机追加、运行器自己写 timing.json 与 output.txt）；全站用词 D1–D9（记在 STYLE.md）；
   DESIGN/README/research 更新。复核者修了 6 个真问题。关卡：构建通过、浏览器 26/26 页 572/572 单元、JS 260/260；
   全量 pytest 由复核者在同一棵树上跑过（441 通过 1 跳过）。

## 进行中

- **pytest 已确认通过**：全量 pytest 在一个进程里会涨到约 8 GB（会话级 fixture 缓存着各章的模型），本机其他程序
  占了大部分内存时，后半段就会出现分配失败的 F/E。按文件逐个进程重跑：22 个文件全部通过，441 通过 1 跳过，
  与复核者的结果一致。**以后在本机跑 pytest，一个测试文件一个进程。**
- **第三轮收尾工作流运行中**：`wf_f817b1ac-27d`（脚本：会话目录 `workflows/scripts/dlfs6-round3.js`，基准提交 `282f259`）。
  按用户要求同时最多两个负责人（两个 worker 从队列里取：frontend、late、mid、early、ui），各负责人做完再由复核者复核
  （UI 文字不配复核者）。若被打断：`TaskStop`，`git status` 看半成品，再以 `resumeFromRunId` 重启。
  1. ui：组件里残留的旧词（检查点、greedy、嵌入、预热、迭代次数……）、重跑命令改成“记录这次运行的命令”、ch03-dropout 主题。
  2. frontend：表格内代码在 `/`、`_`、`.` 后插 `<wbr>`（四张表在桌面又能放下）、十个组件补主题钩子、ch04-dedup 改对数轴、
     暗色热力图红端标签统一、check_pages 等懒加载组件加载完再截图。
  3–5. early / mid / late 内容：新决定 E1（验证单元与 localcheck 测试源码里给读者看的注释、打印也统一用词）、
     E2（“轮”只数 GRPO 迭代，数据过一遍叫 epoch/遍）、E3（多头 Attention）；以及事实修正：ch00 的 Windows 重新检出命令
     （改用 README 验证过的 `rm -r --cached` + `reset --hard`）、勘误 E03 关于 `.gitattributes` 的错误说法、
     CS336 低资源预算是 4,000 万 token（4,096 万是助教的参考运行；已对照讲义 §7.2）、ch03 小结的 81.5%/91.5%、
     GPU 页的 26 秒停顿与运行器新规矩（措辞按复核者的要求留有余地）。
- 第三轮之后由编排者自己更新文档：STYLE.md（D3 措辞改为“运行记录里的 it=k 就是第 k 轮”、E1–E3、§3 新能力、
  §5 共享库与主题助手）、DESIGN.md §4（报告的新字段、运行器自己写 timing.json/output.txt）、README（脏树与 `--allow-dirty`）。

## 下一步

1. 第三轮结束后：汇总、处理复核者的报告、更新文档、全量关卡（pytest 按文件逐个进程）、提交。
2. 需要用户决定：`verify/cs336/adapters_dlfs6.py`（把原书实现接到 CS336 A1 测试上）公开推送前是否保留；
   GitHub Pages 部署（推送前须用户同意）；发布为 Artifact 需要自托管 Pyodide（CDN 在 Artifact 的 CSP 下不可用）。
3. 可选：在空闲机器上重测 `kv_timing_cpu.json` 与浏览器内训练的计时（现有记录标明了当时的 CPU 负载）。

## 约定与决定

- 浏览器检查用 Playwright（`tools/check_pages.mjs`），不用 gstack browse。
- `site/`、`build/`、`runs/` 不入库；运行数据在 `web/data/runs/`；playground 只附带本书自己训练的权重（34 MB）。
- 子模块必须 `core.autocrlf=false`；Windows 上 `PYTHONUTF8=1`、`CUDA_VISIBLE_DEVICES=-1`（空串无效）。
- 含反斜杠的编辑不要经 heredoc 里的 Python 字符串（会被吞掉）；用 Edit/Write 工具。Edit/Write 会把参数里的 `\uXXXX`
  变成字面字符，JS 里写 `\u{XXXX}`。
- 其他页面已链接的验证单元 id 保持不变；新单元用新字母。
- 工作树里的 CRLF：`.gitattributes` 规定 LF，提交时会规范化；全量关卡前用 `git ls-files --eol` 检查并转成 LF，
  让关卡测的就是全新克隆得到的字节。
