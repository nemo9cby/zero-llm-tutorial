"""Run the NumPy verify cells of content/playground.md, so the local tests exercise the exact published code.

Cells run in page order in one shared namespace, from web/ so that their data paths (data/playground/...)
resolve as they do in the browser. The regex mirrors tools/build.py (VERIFY_RE).
"""
from __future__ import annotations

import contextlib
import io
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content" / "playground.md"
_CELL_RE = re.compile(r"^```verify[ \t]+id=(\S+)[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)


def cells() -> dict[str, str]:
    """{cell id: source}, in page order."""
    found = {m.group(1): m.group(2) for m in _CELL_RE.finditer(CONTENT.read_text(encoding="utf-8"))}
    if not found:
        raise RuntimeError(f"no verify cells found in {CONTENT}")
    return found


def run_cells_upto(cell_id: str) -> dict:
    """Execute every cell up to and including `cell_id` (prints swallowed); return the shared namespace."""
    ns: dict = {"__name__": "playground_cells"}
    here = os.getcwd()
    os.chdir(ROOT / "web")
    try:
        for cid, code in cells().items():
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, f"<playground 验证 {cid}>", "exec"), ns)
            if cid == cell_id:
                return ns
    finally:
        os.chdir(here)
    raise KeyError(f"verify cell {cell_id} not found in {CONTENT.name}")
