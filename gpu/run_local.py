"""Run DLFS6 book scripts *unchanged* in an isolated workspace and record a run report.

    uv run python gpu/run_local.py --run codebot-3060ti \
        --script ch03/01_pretrain.py --script ch03/03_sft.py --script ch03/09_grpo.py

Each run gets runs/<run>/work/, a copy of the pinned submodule's tracked files, so the scripts can
write their checkpoints (codebot/model_pretrain.pt, ...) without touching third_party/dlfs6.
Scripts run in order in the same workspace, so a later script consumes the earlier checkpoint,
exactly like a reader running the book in sequence.

Outputs:
  runs/<run>/work/...                       checkpoints and everything the scripts write (gitignored)
  runs/<run>/<script>.log                   raw stdout+stderr (tqdm progress included)
  web/data/runs/<run>/report.json           provenance: commits, GPU, versions, wall times, exit codes
  web/data/runs/<run>/<stem>.metrics.jsonl {"it", <postfix values>} parsed from tqdm (loss, acc, ...)
  web/data/runs/<run>/<stem>.png            figures the script saved, copied for the page
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
TQDM_RE = re.compile(r"(\d+)/(\d+) \[([^\]]*)\]")
POSTFIX_RE = re.compile(r"(\w+)=([-+\d.eE]+)")


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def host_info() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["bf16"] = torch.cuda.is_bf16_supported()
    except ImportError:
        pass
    try:
        info["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def make_workspace(work: Path) -> None:
    if work.exists():
        return
    files = subprocess.check_output(["git", "-C", str(DLFS6), "ls-files"], text=True).splitlines()
    for rel in files:
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DLFS6 / rel, dst)


def link_data(work: Path, rels: list[str]) -> None:
    for rel in rels:
        src, dst = DLFS6 / rel, work / rel
        if dst.exists():
            continue
        if not src.is_file():
            sys.exit(f"--data {rel}: not found in third_party/dlfs6 (run tools/fetch_data.py)")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def parse_tqdm(log_text: str) -> list[dict]:
    points, last = [], None
    for chunk in re.split(r"[\r\n]", log_text):
        m = TQDM_RE.search(chunk)
        if not m:
            continue
        values = {k: float(v) for k, v in POSTFIX_RE.findall(m.group(3))}
        it = int(m.group(1))
        if values and it != last:
            points.append({"it": it, **values})
            last = it
    return points


def update_index(runs_dir: Path, run: str, stem: str | None) -> None:
    """web/data/runs/index.json: {run: {"steps": [stems with metrics]}} -- read by the runcurve widget."""
    path = runs_dir / "index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    steps = index.setdefault(run, {"steps": []})["steps"]
    if stem and stem not in steps:
        steps.append(stem)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id, e.g. codebot-3060ti")
    ap.add_argument("--script", action="append", required=True, help="book script, e.g. ch03/01_pretrain.py")
    ap.add_argument("--data", action="append", default=[], help="extra data file to link into the workspace")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    work = run_dir / "work"
    out = ROOT / "web" / "data" / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    make_workspace(work)
    link_data(work, args.data)

    report_path = out / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "run": args.run, "tutorial_commit": git_rev(ROOT), "dlfs6_commit": git_rev(DLFS6),
        "host": host_info(), "cost_usd": 0.0, "steps": []}
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "WANDB_MODE": os.environ.get("WANDB_MODE", "disabled")}

    for script in args.script:
        stem = Path(script).stem
        log_path = run_dir / f"{stem}.log"
        before = {p: p.stat().st_mtime for p in work.rglob("*") if p.is_file()}
        t0 = time.time()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{args.run}] {script} ... log: {log_path}", flush=True)
        with open(log_path, "wb") as log:
            proc = subprocess.run([sys.executable, str(work / script)], cwd=work, env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
        seconds = round(time.time() - t0, 1)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        points = parse_tqdm(text)
        if points:
            (out / f"{stem}.metrics.jsonl").write_text(
                "".join(json.dumps(p) + "\n" for p in points), encoding="utf-8")
        produced = [p for p in work.rglob("*") if p.is_file() and before.get(p) != p.stat().st_mtime]
        for p in produced:
            if p.suffix in (".png", ".json", ".jsonl") and p.stat().st_size < 5_000_000:
                shutil.copy2(p, out / p.name)
        report["steps"].append({
            "script": script, "started": started, "seconds": seconds, "exit_code": proc.returncode,
            "points": len(points), "final": points[-1] if points else None,
            "produced": sorted(str(p.relative_to(work)).replace("\\", "/") for p in produced),
            "log_tail": text.replace("\r", "\n").strip().splitlines()[-5:],
        })
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        update_index(out.parent, args.run, stem if points else None)
        print(f"[{args.run}] {script} exit={proc.returncode} in {seconds}s", flush=True)
        if proc.returncode != 0:
            sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
