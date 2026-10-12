"""第 3 章的本地验证辅助：从原书脚本里只取出类和函数，不运行脚本里的训练循环。

ch03 的脚本（01_pretrain.py、03_sft.py、09_grpo.py）在模块顶层直接开始训练并写 checkpoint，
所以不能 import。这里用 ast 只保留 import 语句和指定名字的 def/class，再 exec 到一个新命名空间里，
得到的就是原书的原始代码对象（不是抄写的副本）。
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
CODEBOT = DLFS6 / "codebot"

ALPACA = "### Instruction:\n{}\n\n### Response:\n"


def load_defs(script: str, names: list[str], *, keep_imports: bool = True, extra: dict | None = None) -> dict:
    """Return {name: object} for the top-level defs/classes `names` of third_party/dlfs6/<script>.

    keep_imports=False drops the script's own import lines (used to show what a bare `except:` hides).
    """
    path = DLFS6 / script
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and keep_imports:
            body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
            found.add(node.name)
    missing = set(names) - found
    if missing:
        raise LookupError(f"{script}: no top-level def/class named {sorted(missing)}")
    ns: dict = {"__name__": f"dlfs6_{Path(script).stem}", **(extra or {})}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return {n: ns[n] for n in names}


def load_tokenizer():
    from codebot.tokenizer import BPETokenizer
    return BPETokenizer.load_from(str(CODEBOT / "merge_rules.pkl"))


def load_model(name: str):
    """name in {'pretrain', 'sft', 'grpo'}: the official checkpoints, always on CPU."""
    from codebot.model import GPT
    path = CODEBOT / f"model_{name}.pt"
    if not path.is_file():
        import pytest
        pytest.skip(f"{path.name} missing; run tools/fetch_data.py")
    return GPT.load_from(str(path), device="cpu")


def arithmetic_prompts() -> list[tuple[str, int]]:
    """The 81 GRPO prompts, in GRPODataset order (i outer, j inner)."""
    return [(ALPACA.format(f"{i}+{j}="), i + j) for i in range(1, 10) for j in range(1, 10)]


def verify_cell(cell_id: str, page: str = "ch03") -> str:
    """Source of the ```verify id=<cell_id>``` cell in content/<page>.md (same regex as tools/build.py).
    Lets a test prove that the NumPy code shown on the page equals the book's PyTorch code."""
    import re
    text = (ROOT / "content" / f"{page}.md").read_text(encoding="utf-8")
    for m in re.finditer(r"^```verify[ \t]+(.*?)\n(.*?)^```[ \t]*$", text, re.S | re.M):
        if re.search(rf"(^|\s)id={re.escape(cell_id)}(\s|$)", m.group(1)):
            return m.group(2)
    raise LookupError(f"verify cell {cell_id} not found in content/{page}.md")
