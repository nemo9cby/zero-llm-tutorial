"""Helpers for verify/test_ch02.py.

* load_defs(rel)   import only the import/def/class statements of a DLFS6 script, so its demo code
                   (random tensors, prints, os.chdir) never runs;
* run_script(rel)  run a side-effect-free DLFS6 script the way a reader would (GPU hidden);
* page_namespace() execute the NumPy verify cells of content/ch02.md in order, in one namespace,
                   exactly as the browser does, and return that namespace. The tests then check the
                   page's NumPy code against the book's PyTorch code.
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import re
import subprocess
import sys
import types
import warnings
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
PAGE = ROOT / "content" / "ch02.md"

# Scripts that write files (graph.py saves relu_gelu.png into the cwd) must never be run.
WRITES_FILES = {"ch02/graph.py"}
_VERIFY_RE = re.compile(r"^```verify[ \t]+(.*?)\n(.*?)^```[ \t]*$", re.S | re.M)


def load_defs(rel: str) -> types.ModuleType:
    """Return a module holding the imports, functions and classes of a DLFS6 script."""
    path = DLFS6 / rel
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    if not keep:
        raise ValueError(f"{rel}: no definitions found")
    mod = types.ModuleType("dlfs6_" + path.stem.replace("-", "_"))
    mod.__file__ = str(path)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), mod.__dict__)
    return mod


def source_lines(rel: str, start: int, end: int) -> str:
    """Lines start..end (1-based, inclusive) of a DLFS6 file."""
    lines = (DLFS6 / rel).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1:end])


def run_script(rel: str, timeout: int = 300) -> list[str]:
    """Run a DLFS6 script in a fresh interpreter from the repo root (GPU hidden); return stdout lines.
    A new interpreter re-imports torch (tens of seconds with CUDA wheels), so use it for NumPy-only scripts."""
    if rel in WRITES_FILES:
        raise ValueError(f"{rel} writes files into the working directory; do not run it")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "CUDA_VISIBLE_DEVICES": "-1"}
    proc = subprocess.run([sys.executable, str(DLFS6 / rel)], cwd=DLFS6, env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(f"{rel} exited with {proc.returncode}:\n{proc.stderr[-2000:]}")
    return proc.stdout.splitlines()


def run_in_process(rel: str, line_ranges: list[tuple[int, int]] | None = None) -> list[str]:
    """Execute a DLFS6 script (or only the given 1-based inclusive line ranges) inside this process,
    as `python <script>` would, and return its stdout lines. torch is already imported here, so this
    costs milliseconds instead of a fresh interpreter. cwd, sys.path and matplotlib state are restored."""
    if rel in WRITES_FILES:
        raise ValueError(f"{rel} writes files into the working directory; do not run it")
    path = DLFS6 / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    if line_ranges:
        # Blank out the skipped lines so tracebacks keep the script's own line numbers.
        keep = {i for a, b in line_ranges for i in range(a, b + 1)}
        lines = [ln if i + 1 in keep else "" for i, ln in enumerate(lines)]
    source = "\n".join(lines)
    if "matplotlib" in source:                   # never let a test open a GUI window
        import matplotlib
        matplotlib.use("Agg")
    code = compile(source, str(path), "exec")
    cwd, saved_path = os.getcwd(), list(sys.path)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")      # plt.show() under Agg warns that it cannot show
            exec(code, {"__name__": "__main__", "__file__": str(path)})
    finally:
        os.chdir(cwd)                            # 09/10 chdir to the repo root
        sys.path[:] = saved_path                 # ... and append '.' to sys.path
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")
    return buf.getvalue().splitlines()


def page_cells() -> list[tuple[str, str]]:
    """(id, code) of every verify cell on the chapter page, in page order."""
    out = []
    for m in _VERIFY_RE.finditer(PAGE.read_text(encoding="utf-8")):
        cid = re.search(r"id=(\S+)", m.group(1))
        out.append((cid.group(1) if cid else "?", m.group(2)))
    return out


@lru_cache(maxsize=1)
def page_namespace() -> dict:
    """Namespace after running all page cells in order (cached: the cells are deterministic).
    Their printed output is discarded; the build already checks it against the browser."""
    ns: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(ROOT / "web")          # cells read data/ch02/... relative to the site root
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for cid, code in page_cells():
                exec(compile(code, f"<verify {cid}>", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns
