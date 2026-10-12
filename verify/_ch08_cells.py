"""Load the NumPy verify cells of content/ch08.md, so the local tests exercise the exact published code.

Cells run in page order in one shared namespace, the same way the browser runtime and tools/run_cells.py
run them. The regex mirrors tools/build.py (VERIFY_RE); the id is the first attribute of every cell.
Cells that declare files= read them as data/<...> relative to the site root, so they run with web/ as the
working directory (web/data is what the build copies to the site's data/).
"""
from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content" / "ch08.md"
_CELL_RE = re.compile(r"^```verify[ \t]+id=(\S+)[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)


def cells() -> dict[str, str]:
    """{cell id: source}, in page order."""
    found = {m.group(1): m.group(2) for m in _CELL_RE.finditer(CONTENT.read_text(encoding="utf-8"))}
    if not found:
        raise RuntimeError(f"no verify cells found in {CONTENT}")
    return found


def run_cells_upto(cell_id: str) -> dict:
    """Execute every cell up to and including `cell_id`; return the shared namespace.

    The cells' own prints are swallowed: the tests assert on the namespace, not on stdout.
    """
    ns: dict = {"__name__": "ch08_cells"}
    for cid, code in cells().items():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.chdir(ROOT / "web"):
            exec(compile(code, f"<ch08 验证 {cid}>", "exec"), ns)
        if cid == cell_id:
            return ns
    raise KeyError(f"verify cell {cell_id} not found in {CONTENT.name}")
