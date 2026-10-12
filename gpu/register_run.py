"""Register runs copied in from another machine (RunPod, Lambda) in web/data/runs/index.json.

    scp -r -P <port> root@<ip>:/workspace/zero-llm-tutorial/web/data/runs/storybot-4090 web/data/runs/
    uv run python gpu/register_run.py storybot-4090

The runcurve widgets and the run board on the GPU page only show runs listed in index.json.
(Modal runs need no copy: `python gpu/modal_app.py pull --run <run>` downloads and registers them.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_local  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Add copied run reports to web/data/runs/index.json.")
    ap.add_argument("runs", nargs="+", help="run ids: directories under web/data/runs/ with a report.json")
    a = ap.parse_args(argv)
    runs_dir = run_local.ROOT / "web" / "data" / "runs"
    for run in a.runs:
        if not (runs_dir / run / "report.json").is_file():
            sys.exit(f"{runs_dir / run / 'report.json'} not found: copy the whole run directory first")
        stems = run_local.register(runs_dir, run)
        print(f"{run}: registered ({', '.join(stems) or 'no metrics'})")


if __name__ == "__main__":
    main()
