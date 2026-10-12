"""第 9 章的本地验证辅助。

ch09/01_pretrain.py 在模块顶层就开始训练、写 checkpoint，不能直接 import。
load_defs 用 ast 只保留 import 语句和指定的 def，exec 到新命名空间里，得到原书的原始函数对象。
脚本顶层有 `import wandb`，而 wandb 不在默认依赖里，所以 exec 时临时放一个空壳模块顶替。
"""
from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
SCRIPT = "ch09/01_pretrain.py"


def script_tree(script: str = SCRIPT) -> ast.Module:
    path = DLFS6 / script
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def load_defs(names: list[str], script: str = SCRIPT) -> dict:
    tree = script_tree(script)
    body, found = [], set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
            found.add(node.name)
    missing = set(names) - found
    if missing:
        raise LookupError(f"{script}: no top-level def named {sorted(missing)}")
    saved = sys.modules.get("wandb")
    sys.modules["wandb"] = saved or types.ModuleType("wandb")
    try:
        ns: dict = {"__name__": "dlfs6_ch09_pretrain"}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(DLFS6 / script), "exec"), ns)
    finally:
        if saved is None:
            del sys.modules["wandb"]
    return {n: ns[n] for n in names}


def module_constants(script: str = SCRIPT) -> dict:
    """顶层的 `name = 常量` 赋值（超参数表），按脚本里的字面值取出。"""
    out = {}
    for node in script_tree(script).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def verify_cell(cell_id: str, page: str = "ch09") -> str:
    """content/<page>.md 里 ```verify id=<cell_id>``` 单元的源码（与 tools/build.py 用同一个正则）。"""
    text = (ROOT / "content" / f"{page}.md").read_text(encoding="utf-8")
    for m in re.finditer(r"^```verify[ \t]+(.*?)\n(.*?)^```[ \t]*$", text, re.S | re.M):
        if re.search(rf"(^|\s)id={re.escape(cell_id)}(\s|$)", m.group(1)):
            return m.group(2)
    raise LookupError(f"verify cell {cell_id} not found in content/{page}.md")
