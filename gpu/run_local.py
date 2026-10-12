"""Run DLFS6 book scripts *unchanged* in an isolated workspace and record a run report.

    uv run python gpu/run_local.py --run codebot-3060ti \
        --script ch03/01_pretrain.py --script ch03/03_sft.py --script ch03/09_grpo.py

or, with a named job from gpu/jobs.py (the same jobs gpu/modal_app.py runs on Modal):

    uv run python gpu/run_local.py --run storybot-3060ti --job storybot
    uv run python gpu/run_local.py --run webbot-runpod-8xh100 --job webbot --nproc 8     # Linux, 8 GPUs

Each run gets runs/<run>/work/, a copy of the pinned submodule's tracked files, so the scripts can
write their checkpoints (codebot/model_pretrain.pt, ...) without touching third_party/dlfs6.
Scripts run in order in the same workspace, so a later script consumes the earlier checkpoint,
exactly like a reader running the book in sequence. Before each run the script is restored from the
submodule; a job may then apply exact, single-occurrence patches (WebBot probe / multi-GPU budget),
which are recorded in the report.

Outputs:
  runs/<run>/work/...                       checkpoints and everything the scripts write (gitignored)
  runs/<run>/<script>.log                   raw stdout+stderr (tqdm progress included)
  web/data/runs/<run>/report.json           provenance: commits, GPU, versions, wall times, exit codes
  web/data/runs/<run>/<stem>.metrics.jsonl {"it", <postfix values>} parsed from tqdm (loss, acc, ...)
  web/data/runs/<run>/<stem>.png            figures the script saved, copied for the page
"""
from __future__ import annotations

import argparse
import hashlib
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
TQDM_RE = re.compile(r"(\d+)/(\d+) \[([^\]]*)\]")
POSTFIX_RE = re.compile(r"(\w+)=([-+\d.eE]+)")
ELAPSED_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+)<")
RATE_RE = re.compile(r"([\d.]+)(it/s|s/it)")


def _jobs():
    """gpu/jobs.py, the sibling module (shipped next to this file on Modal as well)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import jobs
    return jobs


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
            info["gpu_count"] = torch.cuda.device_count()
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


def parse_timing(log_text: str) -> dict:
    """Throughput of the main progress bar (the one with a postfix; eval bars have none).

    rate      : tqdm's own last reading, e.g. "8.96it/s"
    s_per_it  : steady state, from tqdm's elapsed clock over the second half of the run
                (skips start-up such as torch.compile and the evaluation at step 0)
    """
    marks, rate = [], None
    for chunk in re.split(r"[\r\n]", log_text):
        m = TQDM_RE.search(chunk)
        if not m or not POSTFIX_RE.search(m.group(3)):
            continue
        fields = m.group(3)
        e = ELAPSED_RE.match(fields)
        if e:
            h, mm, ss = (int(x) if x else 0 for x in e.groups())
            marks.append((int(m.group(1)), 3600 * h + 60 * mm + ss))
        r = RATE_RE.search(fields)
        if r:
            rate = r.group(0)
    out: dict = {"rate": rate} if rate else {}
    if len(marks) >= 20:
        (i0, t0), (i1, t1) = marks[len(marks) // 2], marks[-1]
        if i1 > i0 and t1 > t0:
            out["s_per_it"] = round((t1 - t0) / (i1 - i0), 4)
    return out


def update_index(runs_dir: Path, run: str, stem: str | None) -> None:
    """web/data/runs/index.json: {run: {"steps": [stems with metrics]}} -- read by the runcurve widget."""
    path = runs_dir / "index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    steps = index.setdefault(run, {"steps": []})["steps"]
    if stem and stem not in steps:
        steps.append(stem)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


def register(runs_dir: Path, run: str) -> list[str]:
    """Add a run copied from elsewhere (RunPod, Lambda, Modal) to index.json, in the order of its steps."""
    report = json.loads((runs_dir / run / "report.json").read_text(encoding="utf-8"))
    update_index(runs_dir, run, None)
    stems = []
    for step in report["steps"]:
        stem = Path(step["script"]).stem
        if (runs_dir / run / f"{stem}.metrics.jsonl").is_file():
            update_index(runs_dir, run, stem)
            stems.append(stem)
    return stems


def prepare_script(work: Path, script: str, patches: list, pristine: Path | None = None) -> list[dict]:
    """Restore the script from the pinned book code, then apply exact patches; return what changed."""
    src, dst = (pristine or DLFS6) / script, work / script
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not patches:
        shutil.copy2(src, dst)
        return []
    dst.write_text(_jobs().apply_patches(src.read_text(encoding="utf-8"), patches), encoding="utf-8")
    return [{"old": p.old, "new": p.new, "why": p.why} for p in patches]


def seed_checkpoints(work: Path, seeds: dict[str, str], source: Path | None = None) -> list[dict]:
    """Put the book's official checkpoints where a job expects them, unless the run already made its own."""
    fix_storybot_keys = _jobs().fix_storybot_keys
    source = source or DLFS6
    records = []
    for rel, how in seeds.items():
        dst = work / rel
        if dst.exists():
            continue
        from_run = how.startswith(_jobs().RUN_SEED)
        src = (ROOT / "runs" / how[len(_jobs().RUN_SEED):] / "work" / rel) if from_run else source / rel
        if not src.is_file():
            hint = "run that job first" if from_run else "run tools/fetch_data.py"
            sys.exit(f"{rel}: not in the run and not found at {src} ({hint})")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if from_run:
            shutil.copy2(src, dst)
            records.append({"path": rel, "from": f"tutorial run {how[len(_jobs().RUN_SEED):]}",
                            "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
        elif how == "official-keyfix":
            n = fix_storybot_keys(src, dst)
            records.append({"path": rel, "from": "huggingface koki0702/zero-llm-data", "fix": f"{n} keys .mlp. -> .ffn."})
        else:
            shutil.copy2(src, dst)
            records.append({"path": rel, "from": "huggingface koki0702/zero-llm-data"})
    return records


def launcher(nproc: int) -> list[str]:
    """`python script.py`, or torchrun (as `python -m torch.distributed.run`, same interpreter) for N GPUs."""
    if nproc <= 1:
        return [sys.executable]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}"]


def run_step(work: Path, out: Path, run_dir: Path, script: str, env: dict,
             cmd_prefix: list[str] | None = None, patched: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Run one prepared script in `work`; write its log, metrics and figures; return (step record, points)."""
    stem = Path(script).stem
    log_path = run_dir / f"{stem}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in work.rglob("*") if p.is_file()}
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cmd = (cmd_prefix or [sys.executable]) + [str(work / script)]
    with open(log_path, "wb") as log:
        proc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    seconds = round(time.time() - t0, 1)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    points = parse_tqdm(text)
    out.mkdir(parents=True, exist_ok=True)  # the page's data dir may have been reset while we trained
    if points:
        (out / f"{stem}.metrics.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in points), encoding="utf-8")
    produced = [p for p in work.rglob("*") if p.is_file() and before.get(p) != p.stat().st_mtime]
    for p in produced:
        if p.suffix in (".png", ".json", ".jsonl") and p.stat().st_size < 5_000_000:
            shutil.copy2(p, out / p.name)
    step = {
        "script": script, "started": started, "seconds": seconds, "exit_code": proc.returncode,
        "points": len(points), "final": points[-1] if points else None,
        "produced": sorted(str(p.relative_to(work)).replace("\\", "/") for p in produced),
        "log_tail": text.replace("\r", "\n").strip().splitlines()[-5:],
    }
    step.update(parse_timing(text))
    if patched:
        step["patches"] = patched
    if cmd_prefix and len(cmd_prefix) > 1:
        step["launcher"] = "torchrun " + " ".join(cmd_prefix[3:])
    return step, points


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id, e.g. codebot-3060ti")
    ap.add_argument("--script", action="append", default=[], help="book script, e.g. ch03/01_pretrain.py")
    ap.add_argument("--data", action="append", default=[], help="extra data file to link into the workspace")
    ap.add_argument("--job", help="named job from gpu/jobs.py instead of --script (e.g. storybot, webbot-probe)")
    ap.add_argument("--nproc", type=int, default=1, help="GPUs for torchrun (WebBot only; Linux)")
    ap.add_argument("--max-iters", type=int, help="WebBot only: total steps before dividing by --nproc")
    ap.add_argument("--scale", choices=["iters", "batch"], default="iters",
                    help="how WebBot keeps its token budget on --nproc GPUs (see gpu/jobs.py)")
    args = ap.parse_args(argv)
    if bool(args.script) == bool(args.job):
        ap.error("give either --script (one or more) or --job")

    plan = None
    if args.job:
        try:
            plan = _jobs().plan(args.job, nproc=args.nproc, max_iters=args.max_iters, scale=args.scale)
        except (KeyError, ValueError) as err:
            ap.error(str(err))
    elif args.nproc != 1 or args.max_iters:
        ap.error("--nproc / --max-iters need --job webbot")
    scripts = plan.scripts if plan else args.script

    run_dir = ROOT / "runs" / args.run
    work = run_dir / "work"
    out = ROOT / "web" / "data" / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    make_workspace(work)
    link_data(work, args.data + (plan.data if plan else []))

    report_path = out / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "run": args.run, "tutorial_commit": git_rev(ROOT), "dlfs6_commit": git_rev(DLFS6),
        "host": host_info(), "cost_usd": 0.0, "steps": []}
    if plan and plan.seeds:
        seeded = seed_checkpoints(work, plan.seeds)
        if seeded:
            report.setdefault("seeded", []).extend(seeded)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "WANDB_MODE": os.environ.get("WANDB_MODE", "disabled")}

    for script in scripts:
        patched = prepare_script(work, script, plan.patches_for(script) if plan else [])
        print(f"[{args.run}] {script} ... log: {run_dir / (Path(script).stem + '.log')}", flush=True)
        step, points = run_step(work, out, run_dir, script, env,
                                launcher(plan.nproc if plan else 1), patched)
        if plan:
            step["job"] = plan.job
        report["steps"].append(step)
        out.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        update_index(out.parent, args.run, Path(script).stem if points else None)
        print(f"[{args.run}] {script} exit={step['exit_code']} in {step['seconds']}s", flush=True)
        if step["exit_code"] != 0:
            sys.exit(step["exit_code"])


if __name__ == "__main__":
    main()
