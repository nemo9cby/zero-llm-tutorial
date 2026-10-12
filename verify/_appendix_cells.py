"""Run an appendix page's verify cells inside pytest.

The B-tier tests in test_appendix.py compare the NumPy code that readers run in the browser with the
real DLFS6 (PyTorch) code. To make sure they test *exactly* the code on the page, the cells are read
from content/<slug>.md and executed in order in one shared namespace, the same way the browser and
tools/run_cells.py execute them.
"""
import contextlib
import io
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_VERIFY_RE = re.compile(r"^```verify[ \t]+(.*?)\n(.*?)^```[ \t]*$", re.S | re.M)


def _cells(slug):
    text = (ROOT / "content" / f"{slug}.md").read_text(encoding="utf-8")
    for m in _VERIFY_RE.finditer(text):
        attrs = dict(p.split("=", 1) for p in shlex.split(m.group(1)) if "=" in p)
        yield attrs["id"], m.group(2)


def cell_source(slug, cell_id):
    """The source code of one verify cell of content/<slug>.md."""
    for cid, code in _cells(slug):
        if cid == cell_id:
            return code
    raise KeyError(f"no verify cell {cell_id!r} in content/{slug}.md")


def page_namespace(slug, upto=None):
    """Execute the cells of content/<slug>.md in order and return the shared namespace.

    With `upto`, stop right after the cell with that id (KeyError if the page has no such cell).
    The cells' printed output is discarded; their assertions still run.
    """
    ns = {"__name__": f"page_{slug.replace('-', '_')}"}
    for cid, code in _cells(slug):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, f"<{slug} 验证 {cid}>", "exec"), ns)
        if cid == upto:
            return ns
    if upto is not None:
        raise KeyError(f"no verify cell {upto!r} in content/{slug}.md")
    return ns
