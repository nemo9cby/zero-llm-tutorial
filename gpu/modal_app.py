"""Run the DLFS6 book scripts on Modal GPUs, unchanged, and record the same run report as gpu/run_local.py.

Setup, once (the Volume "zero-llm" holds data, checkpoints and reports):
  uv run --group gpu modal setup                                            # browser login -> ~/.modal.toml
  uv run --group gpu modal run gpu/modal_app.py::prepare --bots storybot     # Hugging Face -> Volume (CPU, cents)

Plan without touching Modal (the same estimate `launch` prints before it starts anything):
  uv run --group gpu python gpu/modal_app.py plan --job webbot --gpu H100:8

Launch. Prints the estimate first and refuses to start when its upper end exceeds --budget (USD):
  uv run --group gpu modal run --detach gpu/modal_app.py::launch --job storybot --gpu H100 --run storybot-h100 --budget 4

Bring the results into the book (report.json, metrics, figures -> web/data/runs/<run>/ + index.json):
  uv run --group gpu python gpu/modal_app.py pull --run storybot-h100

Jobs and the few patches they need (WebBot probe, multi-GPU token budget) live in gpu/jobs.py; the
cost model in gpu/costs.py reads the same price table as the handbook page (web/data/gpu/gpus.json).

Volume layout                                  what
  /vol/data/<bot>/<file>                         book data and official checkpoints from Hugging Face
  /vol/runs/<run>/work/<repo path>               files the scripts produced (checkpoints, figures)
  /vol/runs/<run>/logs/<stem>.log                raw stdout+stderr
  /vol/runs/<run>/out/                           report.json, <stem>.metrics.jsonl, figures (= web/data/runs/<run>/)
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal

HERE = Path(__file__).resolve().parent            # gpu/ locally; /root inside the container
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import costs  # noqa: E402  (pure Python, shipped next to this file)
import jobs  # noqa: E402
import run_local  # noqa: E402

ROOT = HERE.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
RUNS_OUT = ROOT / "web" / "data" / "runs"
BOOK = "/root/dlfs6"                              # the pinned book code inside the container (POSIX path)
VOL_MOUNT = "/vol"                                # where the Volume appears in containers
VOL = Path(VOL_MOUNT)
WS = Path("/root/ws")                             # per-run workspace on the container's local disk
VOLUME = "zero-llm"
# Same versions as the tutorial's uv.lock (Python 3.12, Linux); verify/test_gpu.py keeps them in sync.
PINS = ["torch==2.14.1", "numpy==2.5.4", "matplotlib==3.11.2", "tqdm==4.70.1", "regex==2026.9.29", "wandb==0.30.0"]
CPU_PER_GPU, GIB_PER_GPU = 2.0, 8                 # container request per GPU (billed by Modal on top of the GPU)
ALIASES = {"H100!": "H100", "B200+": "B200", "A100": "A100-40GB"}   # "A100" may land on 80 GB at the 40 GB price


@functools.lru_cache(maxsize=1)
def _tracked() -> frozenset[str]:
    out = subprocess.check_output(["git", "-C", str(DLFS6), "ls-files"], text=True)
    return frozenset(out.splitlines())


def _not_tracked(rel: Path) -> bool:
    """Ship only the book's git-tracked files; fetched corpora and checkpoints travel via the Volume."""
    return rel.as_posix() not in _tracked()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")               # ch09: torch.compile -> Triton compiles kernels with gcc
    .uv_pip_install(*PINS)                        # the Linux torch wheel bundles CUDA 13.0 (Modal: driver 580, CUDA 13.0)
    .env({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "MPLBACKEND": "Agg", "WANDB_MODE": "disabled"})
    .add_local_dir(DLFS6, BOOK, ignore=_not_tracked)
    .add_local_file(HERE / "run_local.py", "/root/run_local.py")
    .add_local_file(HERE / "jobs.py", "/root/jobs.py")
    .add_local_file(HERE / "costs.py", "/root/costs.py")
    .add_local_file(ROOT / "tools" / "fetch_data.py", "/root/fetch_data.py")
)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App("zero-llm", image=image)


# ---------------------------------------------------------------- container side

def _copy_tree(src: Path, dst: Path) -> None:
    for f in src.rglob("*") if src.is_dir() else []:
        if f.is_file():
            (dst / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst / f.relative_to(src))


def _from_volume(rel: str, bot: str) -> Path:
    """A file `prepare` put on the Volume (data or an official checkpoint)."""
    src = VOL / "data" / rel
    if not src.is_file():
        raise FileNotFoundError(f"{rel}: run `modal run gpu/modal_app.py::prepare --bots {bot}` first")
    return src


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _bill(report: dict, entry: dict, t_start: float) -> None:
    """Container time x (GPUs x GPU price + CPU/RAM). Attempts killed by preemption are billed but only
    recorded up to their last finished script, so cost_usd is a lower bound."""
    entry["seconds"] = round(time.time() - t_start, 1)
    entry["usd"] = round(entry["seconds"] / 3600 * entry["usd_per_hour"], 4)
    report["cost_usd"] = round(sum(b["usd"] for b in report.get("billing", [])), 2)


@app.function(volumes={VOL_MOUNT: vol}, timeout=3 * 3600, cpu=2.0)
def prepare(bots: str = "storybot", official: bool = True) -> list[str]:
    """Hugging Face -> /vol/data/<bot>/: resumable and size-checked with tools/fetch_data.py's manifest."""
    import fetch_data

    fetch_data.DLFS6 = VOL / "data"               # its fetch() writes under DLFS6/<path>
    sizes = {rel: size for rel, size, _ in fetch_data.MANIFEST}
    done = []
    for rel in jobs.files_for_bots(bots.split(","), official):
        (VOL / "data" / rel).parent.mkdir(parents=True, exist_ok=True)
        print(f"{rel}: {fetch_data.fetch(rel, sizes[rel])}", flush=True)
        done.append(rel)
    vol.commit()
    return done


@app.function(gpu="L4", volumes={VOL_MOUNT: vol}, timeout=24 * 3600, single_use_containers=True,
              retries=modal.Retries(initial_delay=0.0, max_retries=2))
def run_job(plan: dict, run: str, meta: dict, force: bool = False) -> dict:
    """Run one gpu/jobs.py plan in a fresh workspace on local disk, persisting outputs to the Volume.

    Idempotent per script: a retry after preemption skips scripts that already finished in this run
    (same script, same patches). A failing script is recorded and stops the job without raising, so a
    bug is not re-run (and re-billed) by the retry policy.
    """
    t_start = time.time()
    p = jobs.Plan.from_dict(plan)
    vol.reload()
    vroot = VOL / "runs" / run
    out, logs, saved = vroot / "out", vroot / "logs", vroot / "work"
    work = WS / run
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(Path(BOOK), work)
    _copy_tree(saved, work)                       # checkpoints an earlier job of this run produced
    for rel in p.data:                            # big memmaps: local NVMe beats the network file system
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_from_volume(rel, p.bot), work / rel)

    host = run_local.host_info()
    report = _read_json(out / "report.json") or {
        "run": run, "tutorial_commit": meta["tutorial_commit"], "dlfs6_commit": meta["dlfs6_commit"],
        "host": host, "provider": "modal", "cost_usd": 0.0, "steps": []}
    entry = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "job": p.job,
             "gpu": meta["gpu_request"], "usd_per_hour": meta["usd_per_hour"],
             "price_src": meta["price_src"], "price_checked": meta["price_checked"], "seconds": 0, "usd": 0}
    report.setdefault("billing", []).append(entry)

    for rel, how in p.seeds.items():
        if (work / rel).exists():
            continue
        src = _from_volume(rel, p.bot)
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        rec = {"path": rel, "from": "huggingface koki0702/zero-llm-data"}
        if how == "official-keyfix":
            rec["fix"] = f"{jobs.fix_storybot_keys(src, work / rel)} keys .mlp. -> .ffn."
        else:
            shutil.copyfile(src, work / rel)
        report.setdefault("seeded", []).append(rec)

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "WANDB_MODE": os.environ.get("WANDB_MODE", "disabled")}
    done = {(s["script"], json.dumps(s.get("patches", []), ensure_ascii=False))
            for s in report["steps"] if s["exit_code"] == 0}
    gpu_name = f"{host.get('gpu_count', 1)}× {host.get('gpu', '?')}"
    summary = []
    for script in p.scripts:
        patched = run_local.prepare_script(work, script, p.patches_for(script), pristine=Path(BOOK))
        if not force and (script, json.dumps(patched, ensure_ascii=False)) in done:
            print(f"[{run}] {script}: already finished in this run, skipped (launch with --force to rerun)", flush=True)
            continue
        print(f"[{run}] {script} on {gpu_name} ...", flush=True)
        step, _ = run_local.run_step(work, out, logs, script, env, run_local.launcher(p.nproc), patched)
        step.update(job=p.job, gpu=gpu_name)
        report["steps"].append(step)
        for rel in step["produced"]:              # persist checkpoints/figures for later jobs of this run
            (saved / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / rel, saved / rel)
        _bill(report, entry, t_start)
        _write_json(out / "report.json", report)
        vol.commit()
        summary.append({"script": script, "exit_code": step["exit_code"], "seconds": step["seconds"],
                        "final": step["final"]})
        print(f"[{run}] {script} exit={step['exit_code']} in {step['seconds']}s", flush=True)
        if step["exit_code"] != 0:
            break
    _bill(report, entry, t_start)
    _write_json(out / "report.json", report)
    vol.commit()
    return {"run": run, "steps": summary, "cost_usd": report["cost_usd"]}


# ---------------------------------------------------------------- local side

def modal_offer(gpu: str) -> str:
    """'H100' -> 'modal:h100' using the price table's modal_gpu names."""
    gpus, _ = costs.load()
    name = ALIASES.get(gpu, gpu)
    for o in gpus["offers"]:
        if o.get("modal_gpu") == name:
            return o["id"]
    raise SystemExit(f"unknown Modal GPU {gpu!r}; use one of "
                     + ", ".join(o["modal_gpu"] for o in gpus["offers"] if o.get("modal_gpu")))


def plan_and_estimate(job: str, gpu: str = "", max_iters: int = 0, scale: str = "iters",
                      probe_steps: int = 200) -> tuple[jobs.Plan, str, dict]:
    """Resolve the job and price it. gpu: '' (job default), 'H100', 'H100:8', ..."""
    name, _, count = gpu.partition(":")
    nproc = int(count) if count else 1
    p = jobs.plan(job, nproc=nproc, max_iters=max_iters or None, scale=scale, probe_steps=probe_steps)
    base = name or p.modal_gpu
    gpu_str = f"{base}:{nproc}" if nproc > 1 else base
    est = costs.estimate_plan(p.to_dict(), modal_offer(base), nproc,
                              cpu_cores=CPU_PER_GPU * nproc, mem_gib=GIB_PER_GPU * nproc)
    return p, gpu_str, est


def budget_error(est: dict, budget: float) -> str | None:
    """Why `launch` refuses to start, or None. The upper end of the estimate must fit the budget."""
    if est["usd"][1] > budget:
        return (f"the estimate goes up to ${est['usd'][1]:.2f}, above --budget {budget:.2f}. "
                "Rerun with a larger --budget if you mean it (or try the job on a cheaper GPU).")
    return None


def default_run_id(job: str, gpu_str: str) -> str:
    base, _, n = gpu_str.partition(":")
    return f"{job}-modal-{base.lower().rstrip('!+')}" + (f"x{n}" if n else "")


@app.local_entrypoint()
def launch(job: str, gpu: str = "", run: str = "", max_iters: int = 0, scale: str = "iters",
           budget: float = 5.0, dry_run: bool = False, force: bool = False, probe_steps: int = 200):
    p, gpu_str, est = plan_and_estimate(job, gpu, max_iters, scale, probe_steps)
    run = run or default_run_id(job, gpu_str)
    print(f"run {run}: job {job} on Modal gpu={gpu_str!r}")
    print(costs.format_plan_estimate(est))
    for note in p.notes:
        print(f"  note: {note}")
    if dry_run:
        print("dry run: nothing was launched")
        return
    refusal = budget_error(est, budget)
    if refusal:
        raise SystemExit(refusal)
    meta = {"tutorial_commit": run_local.git_rev(ROOT), "dlfs6_commit": run_local.git_rev(DLFS6),
            "gpu_request": gpu_str, "nproc": p.nproc,
            "usd_per_hour": round(p.nproc * est["price"] + est["extra_per_hour"], 4),
            "price_src": est["src"], "price_checked": est["checked"]}
    timeout = int(min(24 * 3600, max(1800, 2 * est["hours"][1] * 3600 + 1800)))
    fn = run_job.with_options(gpu=gpu_str, timeout=timeout, cpu=CPU_PER_GPU * p.nproc,
                              memory=GIB_PER_GPU * 1024 * p.nproc)
    call = fn.spawn(p.to_dict(), run, meta, force)
    print(f"launched {run} ({call.object_id}); timeout {timeout / 3600:.1f} h per attempt")
    print(f"when it finishes: uv run --group gpu python gpu/modal_app.py pull --run {run}")
    print(json.dumps(call.get(), ensure_ascii=False, indent=1))


def pull(run: str, dest_root: Path = RUNS_OUT, volume=None) -> list[Path]:
    """Copy /vol/runs/<run>/out/* into web/data/runs/<run>/ and register it in index.json."""
    v = volume if volume is not None else modal.Volume.from_name(VOLUME)
    prefix = f"runs/{run}/out"
    files = [e for e in v.listdir(prefix, recursive=True) if e.type.name == "FILE"]
    if not files:
        raise SystemExit(f"nothing at {VOLUME}:/{prefix} yet (has the job finished a script?)")
    dest = dest_root / run
    written = []
    for e in files:
        rel = PurePosixPath(e.path.lstrip("/")).relative_to(prefix)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            v.read_file_into_fileobj(e.path, f)
        written.append(target)
    run_local.register(dest_root, run)
    return written


def cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Plan Modal GPU jobs and pull their reports (no `modal run` needed).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="print the job plan and its cost estimate; contacts nothing")
    pl.add_argument("--job", required=True, choices=sorted(jobs.JOBS))
    pl.add_argument("--gpu", default="", help="Modal GPU string, e.g. L4, H100, H100:8 (default: the job's)")
    pl.add_argument("--max-iters", type=int, default=0)
    pl.add_argument("--scale", choices=["iters", "batch"], default="iters")
    pl.add_argument("--probe-steps", type=int, default=200)
    pu = sub.add_parser("pull", help="copy a finished run from the Volume into web/data/runs/")
    pu.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        p, gpu_str, est = plan_and_estimate(a.job, a.gpu, a.max_iters, a.scale, a.probe_steps)
        print(f"job {a.job} on Modal gpu={gpu_str!r}; patches: {len(p.patches)}")
        for patch in p.patches:
            print(f"  {patch.script}: {patch.old!r} -> {patch.new!r}  ({patch.why})")
        print(costs.format_plan_estimate(est))
        for note in p.notes:
            print(f"  note: {note}")
    else:
        for path in pull(a.run):
            print(f"pulled {path.relative_to(ROOT)}")


if __name__ == "__main__":
    cli()
