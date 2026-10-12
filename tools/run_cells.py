"""Execute a chapter's verify cells exactly as the browser would.

Invoked by tools/build.py inside the Pyodide-mirror env (Python 3.14 + NumPy 2.4.6):
    python -I tools/run_cells.py build/cells-ch01.in.json
Input JSON: {"cells": [{"id", "code", "files"}], "data_dir": ".../site/data"}
Prints a JSON list of {"id", "stdout", "error", "seconds"} to stdout.
All cells of a page share one namespace and run in order (like the in-browser runtime).
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web"))
from cell_harness import run_cell  # noqa: E402


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    site_dir = Path(spec["data_dir"]).parent
    # Like the browser: cells run in a directory that holds only the files declared so far with files=
    # (paths "data/<slug>/<name>" relative to the site root), so an undeclared read fails here too.
    work = Path(tempfile.mkdtemp(prefix="cells-"))
    os.chdir(work)
    ns = {"__name__": "__main__"}
    results = []
    try:
        for cell in spec["cells"]:
            for f in cell.get("files", []):
                src = site_dir / f
                if not src.is_file():
                    results.append({"id": cell["id"], "stdout": "", "seconds": 0.0,
                                    "error": f"缺少数据文件 {f}（应位于 web/{f}）"})
                    break
                (work / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, work / f)
            else:
                results.append(dict(id=cell["id"], **run_cell(cell["code"], ns, cell["id"])))
                continue
            break
    finally:
        os.chdir(site_dir)
        shutil.rmtree(work, ignore_errors=True)
    # -I ignores PYTHONIOENCODING, and the Windows console default is GBK: emit UTF-8 bytes explicitly.
    sys.stdout.buffer.write(json.dumps(results, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    main()
