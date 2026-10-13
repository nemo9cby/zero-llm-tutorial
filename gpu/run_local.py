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

Two refusals keep the report true. A tutorial tree with uncommitted changes is refused unless
--allow-dirty: the report's commit would not be the code that ran (what runs write, web/data/runs/,
does not count; see repo_state). And new steps are never appended to a report recorded on another kind
of host or by another provider (Modal): its host block would mislabel them, so use a new run id.

Outputs:
  runs/<run>/work/...                       checkpoints and everything the scripts write (gitignored)
  runs/<run>/<script>.log                   raw stdout+stderr (tqdm progress included)
  web/data/runs/<run>/report.json           provenance: commits, GPU, versions, wall times, exit codes; per step
                                            the tree it ran from ("tutorial") and every checkpoint it wrote
                                            ("checkpoints": path, bytes, sha256)
  web/data/runs/<run>/<stem>.metrics.jsonl {"it", <postfix values>} parsed from tqdm (loss, acc, ...)
  web/data/runs/<run>/<name>.png            figures the script saved, under the names it gave them
  web/data/runs/<run>/timing.json           the pretraining step's speed, read from its log (timing_record)
  web/data/runs/<run>/<stem>.output.txt     what a script without a progress bar printed (output_text)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NamedTuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
TQDM_RE = re.compile(r"(\d+)/(\d+) \[([^\]]*)\]")
POSTFIX_RE = re.compile(r"(\w+)=([-+\d.eE]+)")
ELAPSED_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+)<")
RATE_RE = re.compile(r"([\d.]+)(it/s|s/it)")
LABEL_RE = re.compile(r"^(?:\x1b\[[\d;]*[A-Za-z]|\s)*(.*?):\s*\d+%\|")    # tqdm's desc: "Evaluating:  10%|" -> "Evaluating"
BAR_RE = re.compile(r"\d+%\|[^|]*\|\s*\d+/\d+ \[|\d+it \[\d+:\d+")       # a tqdm bar, with or without a total
WARMUP_STEPS = 20                       # steady readings start at this counter (the probe's 20-200 convention)
RUN_OUTPUTS = "web/data/runs"           # what runs write: never makes the tree dirty
RUN_INPUTS = ("gpu", "tools/fetch_data.py", "web/data/gpu")   # tutorial files a run reads, besides the book
DIFF = ("diff", "HEAD", "--binary", "--no-color", "--no-ext-diff")
HOST_KEYS = ("gpu", "gpu_count", "torch", "cuda", "python")    # what a report's host block claims for every step


def _sibling(name: str):
    """gpu/<name>.py, a module next to this file (shipped next to it on Modal as well)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return importlib.import_module(name)


def _jobs():
    return _sibling("jobs")


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


# ---------------------------------------------------------------- provenance: which tree, which host

def _git(cwd: Path, *args: str) -> bytes | None:
    """stdout of `git -C cwd args...`, or None when git is missing or the command fails."""
    try:
        proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True)
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def _paths(out: bytes | None) -> list[str]:
    """The paths of a `git ... -z` listing."""
    return [p for p in (out or b"").decode("utf-8", "replace").split("\0") if p]


def _is_checkout(path: Path) -> bool:
    """True when `path` is the top of a git work tree, not a directory inside some other repository."""
    top = _git(path, "rev-parse", "--show-toplevel")
    try:
        return bool(top) and os.path.samefile(top.decode("utf-8", "replace").strip(), path)
    except OSError:
        return False


def _label(path: Path) -> str:
    """A path as it reads from the tutorial root, e.g. third_party/dlfs6/ch03/01_pretrain.py."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def repo_state(scripts: Iterable[str] = ()) -> dict:
    """The tutorial tree a run starts from: its commit and how the files differ from it.

    dirty              tracked changes anywhere but in what runs write (web/data/runs/), the book included
                       (an edited book file also shows as the submodule third_party/dlfs6), or untracked
                       inputs: files under gpu/, tools/fetch_data.py, web/data/gpu/, and any of `scripts`
                       that the book's git does not track
    diff_sha256        sha256 of `git diff HEAD --binary --no-color --no-ext-diff -- . ':(exclude)web/data/runs'`
    dlfs6_diff_sha256  sha256 of `git diff HEAD --binary --no-color --no-ext-diff` inside third_party/dlfs6
    changed, untracked the paths behind the flag
    Not a git checkout (a zip download): {"commit": "unknown", "dirty": None}, since nothing can be checked.
    """
    if not _is_checkout(ROOT):
        return {"commit": "unknown", "dirty": None}
    commit = (_git(ROOT, "rev-parse", "HEAD") or b"unknown").decode().strip()
    outside_runs = ("--", ".", f":(exclude){RUN_OUTPUTS}")
    diff = _git(ROOT, *DIFF, *outside_runs) or b""
    changed = _paths(_git(ROOT, "diff", "HEAD", "--name-only", "-z", *outside_runs))
    untracked = _paths(_git(ROOT, "ls-files", "--others", "--exclude-standard", "-z", "--", *RUN_INPUTS))
    book_diff = b""
    if _is_checkout(DLFS6):
        book_diff = _git(DLFS6, *DIFF) or b""
        changed += [_label(DLFS6 / p) for p in _paths(_git(DLFS6, "diff", "HEAD", "--name-only", "-z"))]
        untracked += [_label(DLFS6 / s) for s in scripts
                      if (DLFS6 / s).is_file() and _git(DLFS6, "ls-files", "--error-unmatch", "--", s) is None]
    state = {"commit": commit, "dirty": bool(diff or book_diff or untracked)}
    for key, value in (("diff_sha256", diff), ("dlfs6_diff_sha256", book_diff)):
        if value:
            state[key] = hashlib.sha256(value).hexdigest()
    for key, value in (("changed", changed), ("untracked", untracked)):
        if value:
            state[key] = sorted(set(value))
    return state


def dirty_error(state: dict) -> str | None:
    """Why a run must not start from this tree without --allow-dirty, or None."""
    if not state.get("dirty"):
        return None
    paths = state.get("changed", []) + state.get("untracked", [])
    shown = ", ".join(paths[:8]) + (f" and {len(paths) - 8} more" if len(paths) > 8 else "")
    return (f"the tutorial tree differs from its commit {state['commit'][:7]} ({shown}). Commit first, so that "
            "the report names the code that ran, or pass --allow-dirty to record the run anyway (each step then "
            "keeps the sha256 of `git diff HEAD` and the changed paths).")


def host_signature(host: dict) -> dict:
    """What a report's host block claims for all its steps: GPU model and count, PyTorch, CUDA, Python, the OS.
    Kernel builds and driver versions are left out: they change between sessions on the same kind of
    machine (a Modal retry, a restarted pod). A block that names a GPU but no count was written before the
    runner recorded gpu_count (codebot-3060ti's): it claims no count, so none is compared."""
    sig = {k: host.get(k) for k in HOST_KEYS}
    if host.get("gpu") and "gpu_count" not in host:
        del sig["gpu_count"]
    sig["os"] = (host.get("platform") or "").split("-")[0]
    return sig


def append_error(report: dict, host: dict | None, provider: str | None) -> str | None:
    """Why new steps must not be appended to this existing report, or None.
    host None compares the provider only (the Modal launcher, before any container runs)."""
    def who(p):
        return p or "none (run_local.py)"

    hosts = []
    if host is not None:
        old, new = host_signature(report.get("host") or {}), host_signature(host)
        hosts = [f"{k} {old[k]!r} -> {new[k]!r}" for k in new if k in old and old[k] != new[k]]
    providers = [] if report.get("provider") == provider else [
        f"provider {who(report.get('provider'))} -> {who(provider)}"]
    if not hosts and not providers:
        return None
    where = " and ".join(w for w, hit in (("on another kind of host", hosts), ("by another provider", providers))
                         if hit)
    return (f"run {report.get('run')!r} was recorded {where} ({'; '.join(providers + hosts)}). Its report.json "
            "describes that setup, so new steps would be mislabelled: use a new run id (--run <new-id>).")


def checkpoint_records(work: Path, produced: Iterable[Path]) -> list[dict]:
    """Path, size and sha256 of every checkpoint (*.pt) a step wrote, so exported weights can be traced to it."""
    records = []
    for p in produced:
        if p.suffix == ".pt":
            with open(p, "rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            records.append({"path": p.relative_to(work).as_posix(), "bytes": p.stat().st_size, "sha256": digest})
    return sorted(records, key=lambda r: r["path"])


# ---------------------------------------------------------------- reading the logs

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


class Bar(NamedTuple):
    kind: str             # "main": the training bar (its lines carry a postfix); "eval": a labelled bar without one
    n: int                # tqdm's counter
    total: int
    t: int | None         # tqdm's elapsed clock, in whole seconds
    rate: float | None    # the rate the line shows, in it/s
    label: str | None


def _clock(fields: str) -> int | None:
    """tqdm's elapsed clock at the start of a bar's fields, in seconds ("02:21<7:48:41, ..." -> 141)."""
    e = ELAPSED_RE.match(fields)
    return 3600 * int(e.group(1) or 0) + 60 * int(e.group(2)) + int(e.group(3)) if e else None


def _rate(fields: str) -> float | None:
    """The rate a bar shows, in it/s ("35.13it/s" -> 35.13, "2.00s/it" -> 0.5)."""
    r = RATE_RE.search(fields)
    try:
        value = float(r.group(1)) if r else 0.0
    except ValueError:
        return None
    if not value:
        return None
    return value if r.group(2) == "it/s" else 1 / value


def _bars(log_text: str) -> list[Bar]:
    """The progress-bar lines of a log, in order. Lines of the training bar before its first postfix are left out."""
    rows = []
    for chunk in re.split(r"[\r\n]", log_text):
        m = TQDM_RE.search(chunk)
        if m:
            label = LABEL_RE.match(chunk)
            rows.append((bool(POSTFIX_RE.search(m.group(3))), label.group(1) if label else None,
                         int(m.group(1)), int(m.group(2)), _clock(m.group(3)), _rate(m.group(3))))
    main_labels = {label for post, label, *_ in rows if post}
    return [Bar("main" if post else "eval", n, total, t, rate, label) for post, label, n, total, t, rate in rows
            if post or (label is not None and label not in main_labels)]


def timing_summary(log_text: str, compiled: bool = False) -> dict:
    """How fast a log's progress bars went, from tqdm's elapsed clocks (1 s resolution). {} without a training bar.

    first_step_s              the training bar's clock when it first shows counter 1: step 0 with all its start-up
    steady_s_per_step_20_200  (clock at counter 200 - clock at counter 20) / 180, when no evaluation ran in between
    steady_s_per_step         seconds per step over every interval from counter 20 on that holds no evaluation
                              (steady_steps steps in steady_seconds s); parse_timing's s_per_it includes them
    eval_*                    the evaluation bar: label, batches per pass, passes, the first pass's seconds, the
                              median rate it showed (it/s), one pass at that rate (eval_s = batches / rate), and
                              the median seconds of the later passes
    note                      what the first step includes (compiled: the script calls torch.compile)
    An evaluation counts only if it shows a bar of its own (ch06, ch09 do; GRPO's accuracy check does not). The
    first line of counter k is printed when step k - 1 ends, and an evaluation runs between those of i and i + 1.
    """
    first: dict[int, int] = {}
    marks: list[tuple[int, int, bool]] = []      # (counter, clock, an evaluation ran since the previous mark)
    passes: list[dict] = []
    current, since_mark, eval_in_step0 = None, False, False
    for bar in _bars(log_text):
        if bar.kind == "eval":
            since_mark, eval_in_step0 = True, eval_in_step0 or 1 not in first
            if current is None or bar.n < current["n"] or bar.label != current["label"]:
                current = {"label": bar.label, "total": bar.total, "n": bar.n, "t": bar.t, "rates": []}
                passes.append(current)
            current["n"] = bar.n
            if bar.t is not None:
                current["t"] = bar.t
            if bar.rate is not None:
                current["rates"].append(bar.rate)
            continue
        current = None
        if bar.n >= 1 and bar.n not in first and bar.t is not None:
            first[bar.n] = bar.t
            marks.append((bar.n, bar.t, since_mark))
            since_mark = False
    if not marks:
        return {}
    out: dict = {"first_step_s": first[1]} if 1 in first else {}
    if 20 in first and 200 in first and not any(ev for n, _, ev in marks if 20 < n <= 200):
        out["steady_s_per_step_20_200"] = round((first[200] - first[20]) / 180, 4)
    starts = ["torch.compile"] * compiled + ["the step-0 evaluation"] * eval_in_step0
    out["note"] = ("from tqdm elapsed times in the raw log (1 s resolution)"
                   + (f"; step 0->1 includes {' and '.join(starts)}" if starts else ""))
    steady = [(n1 - n0, t1 - t0) for (n0, t0, _), (n1, t1, ev) in zip(marks, marks[1:])
              if n0 >= WARMUP_STEPS and not ev]
    steps, seconds = sum(s for s, _ in steady), sum(t for _, t in steady)
    if steps:
        out.update(steady_s_per_step=round(seconds / steps, 4), steady_steps=steps, steady_seconds=seconds)
    if passes:
        out.update(eval_bar=passes[0]["label"], eval_batches=passes[0]["total"], eval_passes=len(passes),
                   eval_first_pass_s=passes[0]["t"])
        rates = [r for p in passes for r in p["rates"]]
        if rates:
            rate = round(statistics.median(rates), 2)
            out.update(eval_it_per_s=rate, eval_s=round(passes[0]["total"] / rate, 1))
        later = [p["t"] for p in passes[1:] if p["t"] is not None]
        if later:
            out["eval_later_pass_s"] = statistics.median(later)
    return out


def timing_record(log_text: str, facts: dict | None = None, compiled: bool = False,
                  script: str | None = None) -> dict:
    """timing.json: timing_summary() plus what the log cannot tell (facts: tokens_per_step over all GPUs and
    nproc, from timing_facts). The first four keys are those of the probe's hand-cut timing.json, with the
    same meaning; the rest are additions. {} when the log has no training bar."""
    summary = timing_summary(log_text, compiled)
    if not summary:
        return {}
    facts = facts or {}
    record = {k: summary.pop(k) for k in ("first_step_s", "steady_s_per_step_20_200") if k in summary}
    if "tokens_per_step" in facts:
        record["tokens_per_step"] = facts["tokens_per_step"]
    record["note"] = summary.pop("note")
    if script:
        record["script"] = script
    if "nproc" in facts:
        record["nproc"] = facts["nproc"]
    return {**record, **summary}


def output_text(log_text: str) -> str | None:
    """What a script without a progress bar printed (the generate scripts): the log's non-blank lines, LF-joined.
    The hand extracts *_generate.output.txt follow the same rule. None for a log with a progress bar."""
    lines = re.split(r"[\r\n]", log_text)
    if any(BAR_RE.search(ln) for ln in lines):
        return None
    kept = [ln for ln in lines if ln.strip()]
    return "\n".join(kept) + "\n" if kept else None


def timing_facts(scripts: Iterable[str], plan=None) -> dict[str, dict]:
    """{script: {"tokens_per_step", "nproc"}} for the pretraining script among `scripts`, the one gpu/costs.py
    models (web/data/gpu/bots.json): tokens per optimizer step over all GPUs, after the plan's overrides."""
    costs = _sibling("costs")
    _, bots = costs.load()
    o, nproc = (plan.overrides, plan.nproc) if plan else ({}, 1)
    facts = {}
    for b in bots["bots"].values():
        if b["script"] in scripts:
            shape = costs.run_shape(b, nproc, o.get("max_iters"), o.get("micro_batch_size"),
                                    o.get("accumulation_steps"))
            facts[b["script"]] = {"tokens_per_step": shape["tokens_per_step"], "nproc": nproc}
    return facts


def update_index(runs_dir: Path, run: str, stem: str | None) -> None:
    """web/data/runs/index.json: {run: {"steps": [stems with metrics]}} -- read by the runcurve widget."""
    path = runs_dir / "index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    steps = index.setdefault(run, {"steps": []})["steps"]
    if stem and stem not in steps:
        steps.append(stem)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


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
    dst.write_text(_jobs().apply_patches(src.read_text(encoding="utf-8"), patches), encoding="utf-8", newline="\n")
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
             cmd_prefix: list[str] | None = None, patched: list[dict] | None = None,
             timing: dict | None = None) -> tuple[dict, list[dict]]:
    """Run one prepared script in `work`; write its log, metrics, figures and extracts; return (step record, points).

    timing: timing_facts() for this script when it is the pretraining script; its speed then goes to timing.json.
    A successful script without a progress bar gets <stem>.output.txt. Neither is written for a failed script.
    """
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
            "".join(json.dumps(p) + "\n" for p in points), encoding="utf-8", newline="\n")
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
    steady = timing_summary(text).get("steady_s_per_step")
    if steady is not None:
        step["steady_s_per_it"] = steady
    checkpoints = checkpoint_records(work, produced)
    if checkpoints:
        step["checkpoints"] = checkpoints
    extracts = {}
    if proc.returncode == 0:
        if timing is not None:
            compiled = "torch.compile(" in (work / script).read_text(encoding="utf-8", errors="replace")
            record = timing_record(text, timing, compiled, script)
            if record:
                extracts["timing.json"] = json.dumps(record, ensure_ascii=False, indent=1)
        output = output_text(text)
        if output:
            extracts[f"{stem}.output.txt"] = output
    for name, content in extracts.items():
        (out / name).write_text(content, encoding="utf-8", newline="\n")
    if extracts:
        step["extracts"] = list(extracts)
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
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run although the tutorial has uncommitted changes; each step records their sha256")
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

    # Both refusals come before anything is written.
    tree = repo_state(scripts)
    refusal = dirty_error(tree)
    if refusal and not args.allow_dirty:
        sys.exit(refusal)
    if tree["dirty"] is None:
        print("note: the tutorial is not a git checkout here, so the report cannot name the code that ran",
              file=sys.stderr)
    run_dir = ROOT / "runs" / args.run
    work = run_dir / "work"
    out = ROOT / "web" / "data" / "runs" / args.run
    report_path = out / "report.json"
    host = host_info()
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        clash = append_error(report, host, None)
        if clash:
            sys.exit(clash)
    else:
        report = {"run": args.run, "tutorial_commit": tree["commit"], "dlfs6_commit": git_rev(DLFS6),
                  "host": host, "cost_usd": 0.0, "steps": []}
    out.mkdir(parents=True, exist_ok=True)
    make_workspace(work)
    link_data(work, args.data + (plan.data if plan else []))
    if plan and plan.seeds:
        seeded = seed_checkpoints(work, plan.seeds)
        if seeded:
            report.setdefault("seeded", []).extend(seeded)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "WANDB_MODE": os.environ.get("WANDB_MODE", "disabled")}
    facts = timing_facts(scripts, plan)

    for script in scripts:
        patched = prepare_script(work, script, plan.patches_for(script) if plan else [])
        print(f"[{args.run}] {script} ... log: {run_dir / (Path(script).stem + '.log')}", flush=True)
        step, points = run_step(work, out, run_dir, script, env,
                                launcher(plan.nproc if plan else 1), patched, facts.get(script))
        if plan:
            step["job"] = plan.job
        step["tutorial"] = tree
        report["steps"].append(step)
        out.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        update_index(out.parent, args.run, Path(script).stem if points else None)
        print(f"[{args.run}] {script} exit={step['exit_code']} in {step['seconds']}s", flush=True)
        if step["exit_code"] != 0:
            sys.exit(step["exit_code"])


if __name__ == "__main__":
    main()
