---
slug: index
order: 0
kind: extra
title: 从零构建大语言模型
bare: true
hidden: true
lede: 基于《ゼロから作るDeep Learning ❻ LLM編》的代码反向生成的可验证互动教程。
---

<header class="cover">
<h1 class="cover-title">从零构建<br>大语言模型</h1>
<p class="cover-lede">从一个字符一个编号的分词器出发，亲手搭出 GPT，训练会写 Python 的 CodeBot、会讲故事的 StoryBot，再用多块 GPU 训练 WebBot。书里的每一个技术断言，都配有一段可以在你浏览器里运行的最小验证。</p>
<p class="cover-byline">依据斋藤康毅《ゼロから作る Deep Learning ❻ LLM編》（O'Reilly Japan，2026）的全部源代码反向编写。原书代码以 MIT 许可发布。</p>
</header>

## 如何使用这本教程

正文里有三种可以“动手”的东西。

**验证**：带编号的代码块，右侧有一枚印章。点击它，代码会在你的浏览器里运行（借助 Pyodide，无需安装任何东西），如果输出与构建时完全一致，印章就会盖下。每段验证都用 `assert` 把结论写死，所以“通过”意味着正文里的说法被你亲手确认过一次。

**交互图**：编号为“图 X-Y”的图都可以拖动、输入、点击。它们用来建立直觉，验证则负责把直觉钉牢。

**本地验证与 GPU 实验**：涉及 PyTorch 的结论，用 `uv run pytest` 在本地验证；需要 GPU 的训练，给出在本机、Modal、RunPod 或 Lambda 上运行的完整命令与成本估算。

<!-- toc -->
