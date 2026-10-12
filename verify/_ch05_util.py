"""第 5 章本地验证用的辅助函数（文件名以下划线开头，pytest 不会把它当成测试收集）。

三件事：
  * load_defs：只执行 DLFS6 脚本里的 import / def / class，跳过模块级的演示代码
    （例如 ch05/05_kvcache.py 末尾那段约 10 秒的测速）；
  * page_cells：取出 content/ch05.md 里指定 id 的验证单元源码并依次执行，
    让 pytest 检查的正是页面上展示、浏览器里运行的那份 NumPy 代码；
  * load_storybot：加载 HF 上发布的 StoryBot 检查点（键名 mlp → ffn 改名后才能加载）。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
PAGE = ROOT / "content" / "ch05.md"
STORYBOT_CFG = ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim", "theta")

_VERIFY_RE = re.compile(r"^```verify[ \t]+(.*?)\n(.*?)^```[ \t]*$", re.S | re.M)


def load_defs(rel_path: str, names: set[str] | None = None, replace: dict[str, str] | None = None) -> dict:
    """Exec only the imports and top-level def/class nodes of a DLFS6 script.

    names:   keep only these def/class names (imports are always kept)
    replace: {old: new} substring edits applied to the kept source; each `old` must occur exactly once
    """
    src = (DLFS6 / rel_path).read_text(encoding="utf-8")
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names):
            keep.append(node)
    code = "\n\n".join(ast.get_source_segment(src, n) for n in keep)
    for old, new in (replace or {}).items():
        count = code.count(old)
        if count != 1:
            raise ValueError(f"{rel_path}: expected exactly one {old!r}, found {count}")
        code = code.replace(old, new)
    ns: dict = {"__name__": f"dlfs6_{Path(rel_path).stem}"}
    exec(compile(code, f"<dlfs6:{rel_path}>", "exec"), ns)
    return ns


def page_cell_sources() -> dict[str, str]:
    """{verify id: source} for every verify cell in content/ch05.md, in page order."""
    cells = {}
    for m in _VERIFY_RE.finditer(PAGE.read_text(encoding="utf-8")):
        found = re.search(r"\bid=(\S+)", m.group(1))
        if not found:
            raise ValueError(f"verify cell without id: {m.group(1)!r}")
        cells[found.group(1)] = m.group(2)
    return cells


def page_cells(*ids: str) -> dict:
    """Run the given verify cells of content/ch05.md, in order, in one shared namespace."""
    sources = page_cell_sources()
    missing = [i for i in ids if i not in sources]
    if missing:
        raise KeyError(f"content/ch05.md has no verify cell(s) {missing}")
    ns: dict = {"__name__": "__main__"}
    exec("import numpy as np", ns)          # 页面的第一个单元导入 numpy，后面的单元都默认它已在命名空间里
    for cid in ids:
        exec(compile(sources[cid], f"<ch05.md 验证 {cid}>", "exec"), ns)
    return ns


def load_storybot(path: Path, cls):
    """Build `cls` from a StoryBot checkpoint, renaming the published 'mlp' keys to the code's 'ffn'."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = cls(**{k: ck[k] for k in STORYBOT_CFG})
    model.load_state_dict({k.replace(".mlp.", ".ffn."): v for k, v in ck["model_state_dict"].items()})
    return model.eval()


def greedy_with_cache(model, prompt_ids, n: int):
    """generate_with_cache from ch05/05_kvcache.py, for any prompt length (clears the cache first)."""
    import torch

    model.eval()
    model.clear_cache()
    ids, nxt = prompt_ids, prompt_ids
    with torch.no_grad():
        for _ in range(n):
            nxt = model(nxt, use_cache=True)[:, -1:].argmax(-1)
            ids = torch.cat([ids, nxt], dim=1)
    return ids


def greedy_recompute(model, prompt_ids, n: int, window: int | None = None):
    """No cache: recompute the whole (optionally left-truncated) context every step."""
    import torch

    model.eval()
    ids = prompt_ids
    with torch.no_grad():
        for _ in range(n):
            ctx = ids if window is None else ids[:, -window:]
            nxt = model(ctx)[:, -1:].argmax(-1)
            ids = torch.cat([ids, nxt], dim=1)
    return ids
