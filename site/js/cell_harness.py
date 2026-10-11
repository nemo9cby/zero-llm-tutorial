"""Shared verify-cell harness.

The exact same file runs in two places, so a cell's output is identical in both:
  * in the browser, inside the Pyodide web worker (site/js/pyworker.js fetches it), and
  * at build time, in tools/run_cells.py under Python 3.14 + NumPy 2.4.6 (the Pyodide mirror).
"""
import contextlib
import io
import linecache
import sys
import time
import traceback


def run_cell(code, ns, cell_id="cell"):
    """Execute one cell in namespace `ns`; return {stdout, error, seconds}."""
    filename = f"<验证 {cell_id}>"
    # Register the source so tracebacks can show the failing line.
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    buf = io.StringIO()
    error = None
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, filename, "exec"), ns)
    except BaseException:
        etype, evalue, tb = sys.exc_info()
        frames = [f for f in traceback.extract_tb(tb) if f.filename == filename]
        where = ""
        if frames:
            f = frames[-1]
            where = f"第 {f.lineno} 行: {(f.line or '').strip()}\n"
        error = where + "".join(traceback.format_exception_only(etype, evalue)).strip()
    return {"stdout": buf.getvalue(), "error": error, "seconds": time.perf_counter() - t0}
