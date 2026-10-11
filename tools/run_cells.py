"""Execute a chapter's verify cells exactly as the browser would.

Invoked by tools/build.py inside the Pyodide-mirror env (Python 3.14 + NumPy 2.4.6):
    python -I tools/run_cells.py build/cells-ch01.in.json
Input JSON: {"cells": [{"id", "code", "files"}], "data_dir": ".../site/data"}
Prints a JSON list of {"id", "stdout", "error", "seconds"} to stdout.
All cells of a page share one namespace and run in order (like the in-browser runtime).
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web"))
from cell_harness import run_cell  # noqa: E402


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    site_dir = Path(spec["data_dir"]).parent
    # Cells refer to data files as "data/<name>", relative to the site root (same layout as the browser FS).
    site_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(site_dir)
    for cell in spec["cells"]:
        for f in cell.get("files", []):
            if not Path(f).is_file():
                missing = [{"id": cell["id"], "stdout": "", "seconds": 0.0,
                            "error": f"缺少数据文件 {f}（应位于 site/{f}）"}]
                sys.stdout.buffer.write(json.dumps(missing, ensure_ascii=False).encode("utf-8"))
                return
    ns = {"__name__": "__main__"}
    results = [dict(id=c["id"], **run_cell(c["code"], ns, c["id"])) for c in spec["cells"]]
    # -I ignores PYTHONIOENCODING, and the Windows console default is GBK: emit UTF-8 bytes explicitly.
    sys.stdout.buffer.write(json.dumps(results, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    main()
