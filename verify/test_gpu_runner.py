"""GPU runner provenance (gpu/run_local.py, gpu/modal_app.py): what a run report proves about the run.

Everything runs on the CPU in temporary directories and git repositories; nothing contacts Modal.
  * B1 the report names the tutorial tree that ran: HEAD, a dirty flag, the sha256 of `git diff HEAD`, the
    untracked inputs; run_local.py and the Modal launch refuse a dirty tree unless --allow-dirty;
  * B2 the runner writes timing.json (pretraining step) and <stem>.output.txt (bar-less scripts) itself, and
    regenerating them from the recorded raw logs (runs/<run>/*.log) reproduces the committed extracts;
  * B3 every checkpoint (*.pt) a step writes is recorded with its size and sha256;
  * B4 a step is never appended to a report recorded on another host or by another provider;
  * B5 the probe's log excerpt keeps the one-time 26 s stall at the end of the step-0 evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import DLFS6, ROOT

GPU = ROOT / "gpu"
RUNS = ROOT / "web" / "data" / "runs"
RAW_RUNS = ROOT / "runs"                      # raw logs of the recorded runs (gitignored, present where they ran)
sys.path.insert(0, str(GPU))
import costs  # noqa: E402
import jobs  # noqa: E402
import run_local  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

OLD_REPORT_KEYS = {"run", "tutorial_commit", "dlfs6_commit", "host", "cost_usd", "steps"}
NEW_TIMING_KEYS = {"script", "nproc", "steady_s_per_step", "steady_steps", "steady_seconds", "eval_bar",
                   "eval_batches", "eval_passes", "eval_first_pass_s", "eval_it_per_s", "eval_s"}
DIFF = ("diff", "HEAD", "--binary", "--no-color", "--no-ext-diff")   # the recipe gpu/runpod_lambda.md documents
EXCLUDE_OUTPUTS = ":(exclude)web/data/runs"


# ---------------------------------------------------------------- helpers

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(cwd: Path, *args: str) -> bytes:
    """git in a throwaway repository: no CRLF conversion, a fixed identity, local submodule URLs allowed."""
    proc = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "user.email=t@example.com", "-c", "user.name=t",
                           "-c", "protocol.file.allow=always", "-C", str(cwd), *args], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").decode().strip()


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def bar_script(steps: int, outputs: dict[str, bytes]) -> str:
    """A stand-in for a training script: chdir like the book, a tqdm-style bar with a loss postfix, files written."""
    return ("import os, sys\n"
            "os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))\n"
            f"STEPS = {steps}\n"
            "for i in range(STEPS):\n"
            "    sys.stdout.write(f'\\r{i * 100 // STEPS}%| | {i + 1}/{STEPS} [00:{i:02d}<00:10, 5.00it/s, loss={2 - i / 100:.4f}]')\n"
            "print()\n"
            f"for rel, data in {outputs!r}.items():\n"
            "    os.makedirs(os.path.dirname(rel) or '.', exist_ok=True)\n"
            "    open(rel, 'wb').write(data)\n")


def text_script(text: str, exit_code: int = 0) -> str:
    """A stand-in for a generate script: prints `text` verbatim (CRLF, blank lines and all), no progress bar."""
    return f"import sys\nsys.stdout.buffer.write({text.encode()!r})\nsys.exit({exit_code})\n"


def log_script(log: str, outputs: dict[str, bytes]) -> str:
    """Replays a recorded-looking tqdm log; mentions torch.compile like ch09/01_pretrain.py does."""
    return ("import os, sys\n"
            "os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))\n"
            "# the book compiles its model here: model = torch.compile(model)\n"
            f"sys.stdout.write({log!r})\n"
            f"for rel, data in {outputs!r}.items():\n"
            "    os.makedirs(os.path.dirname(rel) or '.', exist_ok=True)\n"
            "    open(rel, 'wb').write(data)\n")


def make_book(path: Path, files: dict[str, str]) -> Path:
    """A git repository standing in for third_party/dlfs6, with `files` committed."""
    for rel, text in files.items():
        write(path / rel, text)
    git(path.parent, "init", "-q", str(path))
    git(path, "config", "core.autocrlf", "false")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "book")
    return path


def make_tutorial(tmp_path: Path, book_files: dict[str, str]) -> tuple[Path, Path]:
    """A git repository standing in for the tutorial, with the book as its submodule at third_party/dlfs6."""
    book = make_book(tmp_path / "book-origin", book_files)
    root = tmp_path / "tutorial"
    write(root / "gpu" / "run_local.py", "# the runner\n")
    write(root / "content" / "notes.md", "notes\n")
    write(root / "web" / "data" / "runs" / "index.json", "{}\n")
    write(root / ".gitignore", "/runs/\n__pycache__/\n")
    git(tmp_path, "init", "-q", str(root))
    git(root, "config", "core.autocrlf", "false")
    git(root, "submodule", "add", "-q", book.as_posix(), "third_party/dlfs6")
    git(root / "third_party" / "dlfs6", "config", "core.autocrlf", "false")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "tutorial")
    return root, root / "third_party" / "dlfs6"


def clock(s: int) -> str:
    return f"{s // 60:02d}:{s % 60:02d}"


def synthetic_log(steps: int = 220, first_train_s: int = 10, step_s: int = 1, eval_every: int = 100,
                  batches: int = 40, rate: int = 20, compile_s: int = 1, stall_s: int = 3) -> str:
    """A tqdm log shaped like ch09/01_pretrain.py's. The evaluation of step i runs after its training step and
    before set_postfix, on its own labelled bar; every counter is printed twice (the refresh still carries the
    previous step's postfix). Clocks are whole seconds, like tqdm's. Like the probe's, the step-0 evaluation
    compiles on its first batch (`compile_s`) and stalls on its last one (`stall_s`); later ones do neither."""
    lines, t, post = [f"  0%|          | 0/{steps} [00:00<?, ?it/s]"], 0, ""
    for i in range(steps):
        if i:
            lines.append(f"{100 * i // steps:3d}%|#   | {i}/{steps} [{clock(t)}<00:10,  1.00it/s{post}]")
        t += first_train_s if i == 0 else step_s
        if i % eval_every == 0:
            c = compile_s if i == 0 else 0
            lines.append(f"Evaluating:   0%|          | 0/{batches} [00:00<?, ?it/s]\x1b[A")
            if c:
                lines.append(f"Evaluating:   2%|          | 1/{batches} [{clock(c)}<00:39, {c:5.2f}s/it]\x1b[A")
            for k in range(4, batches, 4):
                lines.append(f"Evaluating: {100 * k // batches:3d}%|##  | {k}/{batches} "
                             f"[{clock(c + k // rate)}<00:01, {rate:.2f}it/s]\x1b[A")
            e = c + batches // rate + (stall_s if i == 0 else 0)
            lines.append(f"Evaluating: 100%|####| {batches}/{batches} [{clock(e)}<00:00, {rate:.2f}it/s]\x1b[A")
            t += e
        post = f", loss={10 - i / 100:.4f}"
        lines.append(f"{100 * i // steps:3d}%|#   | {i}/{steps} [{clock(t)}<00:10,  1.00it/s{post}]")
    lines.append(f"100%|####| {steps}/{steps} [{clock(t)}<00:00,  1.00it/s{post}]")
    return "\r".join(lines) + "\n"


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """run_local pointed at a temporary tutorial root (not a git checkout) and a fake book (a git repo)."""
    book = make_book(tmp_path / "dlfs6", {
        "ch99/01_train.py": bar_script(40, {"fakebot/model.pt": b"m" * 32, "loss_train.png": b"png"}),
        "ch99/02_gen.py": text_script("--- sample 1 ---\r\n\r\ndef f():\r\n\r\n    return 1\r\n\r\n"),
        "ch99/03_fail.py": text_script("Traceback: boom\n", exit_code=1),
        "ch09/01_pretrain.py": log_script(synthetic_log(eval_every=1000), {"webbot/model_pretrain.pt": b"w" * 100}),
    })
    root = tmp_path / "root"
    monkeypatch.setattr(run_local, "ROOT", root)
    monkeypatch.setattr(run_local, "DLFS6", book)
    return root


# ---------------------------------------------------------------- B1: which tree ran

def test_repo_state_names_the_tree_and_its_changes(tmp_path, monkeypatch):
    root, book = make_tutorial(tmp_path, {"ch99/01_train.py": "print('train')\n"})
    monkeypatch.setattr(run_local, "ROOT", root)
    monkeypatch.setattr(run_local, "DLFS6", book)
    clean = {"commit": head(root), "dirty": False}
    assert run_local.repo_state(["ch99/01_train.py"]) == clean
    assert run_local.dirty_error(clean) is None

    # what runs write (web/data/runs/, runs/) and stray files outside the run's inputs do not count
    write(root / "web/data/runs/x/report.json", "{}\n")
    write(root / "web/data/runs/index.json", '{"x": {"steps": []}}\n')
    write(root / "runs/x/work/model.pt", "w")
    write(root / "scratch.png", "png")
    assert run_local.repo_state() == clean

    # an untracked file among the inputs (gpu/) does
    write(root / "gpu/helper.py", "x = 1\n")
    state = run_local.repo_state()
    assert state == {**clean, "dirty": True, "untracked": ["gpu/helper.py"]}
    msg = run_local.dirty_error(state)
    assert "gpu/helper.py" in msg and "--allow-dirty" in msg
    (root / "gpu/helper.py").unlink()

    # a tracked change anywhere: the sha256 is the one a reader recomputes with the documented command
    write(root / "content/notes.md", "notes, edited\n")
    state = run_local.repo_state()
    diff = git(root, *DIFF, "--", ".", EXCLUDE_OUTPUTS)
    assert state == {**clean, "dirty": True, "diff_sha256": sha256(diff), "changed": ["content/notes.md"]}
    assert state["diff_sha256"] != sha256(git(root, *DIFF))                     # index.json is left out
    git(root, "checkout", "--", "content/notes.md", "web/data/runs/index.json")

    # the book is an input too: an edited book file, or an untracked script the run was asked to run
    write(book / "ch99/01_train.py", "print('patched by hand')\n")
    state = run_local.repo_state(["ch99/01_train.py"])
    assert state["dirty"] and state["dlfs6_diff_sha256"] == sha256(git(book, *DIFF))
    assert {"third_party/dlfs6", "third_party/dlfs6/ch99/01_train.py"} <= set(state["changed"])
    git(book, "checkout", "--", "ch99/01_train.py")
    write(book / "ch99/extra.py", "print('not in the book')\n")
    assert run_local.repo_state(["ch99/01_train.py"]) == clean                    # not one of this run's scripts
    assert run_local.repo_state(["ch99/extra.py"])["untracked"] == ["third_party/dlfs6/ch99/extra.py"]

    # no git checkout at ROOT (a zip download, or a directory inside some other repository): unknown, not refused
    for elsewhere in (tmp_path / "plain", root / "content"):
        elsewhere.mkdir(exist_ok=True)
        monkeypatch.setattr(run_local, "ROOT", elsewhere)
        unknown = run_local.repo_state()
        assert unknown == {"commit": "unknown", "dirty": None} and run_local.dirty_error(unknown) is None


def test_run_local_refuses_a_dirty_tree_unless_allowed(tmp_path, monkeypatch):
    root, book = make_tutorial(tmp_path, {
        "ch99/01_train.py": bar_script(30, {"fakebot/model.pt": b"m" * 32})})
    monkeypatch.setattr(run_local, "ROOT", root)
    monkeypatch.setattr(run_local, "DLFS6", book)
    write(root / "content/notes.md", "uncommitted\n")
    with pytest.raises(SystemExit) as refused:
        run_local.main(["--run", "d1", "--script", "ch99/01_train.py"])
    assert "--allow-dirty" in str(refused.value) and "content/notes.md" in str(refused.value)
    assert not (root / "web/data/runs/d1").exists() and not (root / "runs/d1").exists()   # refused before any work

    run_local.main(["--run", "d1", "--script", "ch99/01_train.py", "--allow-dirty"])
    report = json.loads((root / "web/data/runs/d1/report.json").read_text(encoding="utf-8"))
    assert set(report) == OLD_REPORT_KEYS and report["tutorial_commit"] == head(root)
    diff = git(root, *DIFF, "--", ".", EXCLUDE_OUTPUTS)
    assert report["steps"][0]["tutorial"] == {"commit": head(root), "dirty": True, "diff_sha256": sha256(diff),
                                              "changed": ["content/notes.md"]}

    git(root, "commit", "-qam", "notes")
    run_local.main(["--run", "d2", "--script", "ch99/01_train.py"])
    run_local.main(["--run", "d2", "--script", "ch99/01_train.py"])       # appending: the first run's outputs
    steps = json.loads((root / "web/data/runs/d2/report.json").read_text(encoding="utf-8"))["steps"]
    assert [s["tutorial"] for s in steps] == [{"commit": head(root), "dirty": False}] * 2   # do not dirty the tree


def test_cli_help_lists_allow_dirty(capsys):
    with pytest.raises(SystemExit) as done:
        run_local.main(["--help"])
    assert done.value.code == 0 and "--allow-dirty" in capsys.readouterr().out


# ---------------------------------------------------------------- B4: one host per report

def test_append_refuses_another_host_or_provider(fake_root):
    here = run_local.host_info()
    other = {"platform": "Linux-4.19.0-gvisor-x86_64-with-glibc2.36", "python": "3.12.10", "torch": "2.14.1+cu130",
             "cuda": "13.0", "gpu": "NVIDIA H100 80GB HBM3", "gpu_count": 1, "gpu_capability": "9.0", "bf16": True,
             "nvidia_smi": "NVIDIA H100 80GB HBM3, 580.95.05, 81559 MiB"}
    # the same kind of machine: kernel build and driver may change between sessions, the claims do not
    later = {**other, "platform": "Linux-6.1.0-x86_64-with-glibc2.36",
             "nvidia_smi": "NVIDIA H100 80GB HBM3, 590.10.01, 81559 MiB"}
    assert run_local.append_error({"run": "x", "host": other, "provider": "modal"}, later, "modal") is None
    clash = run_local.append_error({"run": "x", "host": other, "provider": "modal"},
                                   {**later, "torch": "2.15.0+cu130", "gpu": "NVIDIA B200"}, "modal")
    assert "gpu" in clash and "torch" in clash and "new run id" in clash and "provider" not in clash
    assert "provider" in run_local.append_error({"run": "x", "host": here, "provider": "modal"}, here, None)
    assert run_local.append_error({"run": "x", "host": here}, here, None) is None

    # a host block written before the runner recorded gpu_count (codebot-3060ti's) makes no claim about the
    # count: the machine that wrote it may still append, another GPU may not, and a block without a GPU
    # still differs from a GPU host
    old_schema = {"platform": "Windows-11-10.0.26200-SP0", "python": "3.12.9", "torch": "2.14.1+cu130",
                  "cuda": "13.0", "gpu": "NVIDIA GeForce RTX 3060 Ti", "gpu_capability": "8.6", "bf16": True,
                  "nvidia_smi": "NVIDIA GeForce RTX 3060 Ti, 610.60, 8192 MiB"}
    today = {**old_schema, "gpu_count": 1}                         # what host_info() records there now
    assert run_local.append_error({"run": "x", "host": old_schema}, today, None) is None
    clash = run_local.append_error({"run": "x", "host": old_schema},
                                   {**today, "gpu": "NVIDIA GeForce RTX 4090"}, None)
    assert "gpu 'NVIDIA GeForce RTX 3060 Ti' -> 'NVIDIA GeForce RTX 4090'" in clash and "gpu_count" not in clash
    cpu_only = {k: v for k, v in old_schema.items() if k not in ("gpu", "gpu_capability", "bf16", "nvidia_smi")}
    assert "gpu_count None -> 1" in run_local.append_error({"run": "x", "host": cpu_only}, today, None)

    for run, host, provider in [("h100-modal", other, "modal"), ("this-pc-modal", here, "modal"),
                                ("another-box", other, None)]:
        path = write(fake_root / f"web/data/runs/{run}/report.json", json.dumps(
            {"run": run, "tutorial_commit": "t" * 40, "dlfs6_commit": "d" * 40, "host": host,
             **({"provider": provider} if provider else {}), "cost_usd": 0.27, "steps": []}))
        before = path.read_bytes()
        with pytest.raises(SystemExit) as refused:
            run_local.main(["--run", run, "--script", "ch99/01_train.py"])
        assert "new run id" in str(refused.value), run
        assert path.read_bytes() == before and not (fake_root / "runs" / run).exists(), run

    run_local.main(["--run", "same", "--script", "ch99/01_train.py"])
    run_local.main(["--run", "same", "--script", "ch99/01_train.py"])          # same machine: appends
    report = json.loads((fake_root / "web/data/runs/same/report.json").read_text(encoding="utf-8"))
    assert len(report["steps"]) == 2


# ---------------------------------------------------------------- B2 + B3: what a step leaves behind

def test_run_local_records_checkpoints_and_writes_extracts(fake_root):
    run_local.main(["--run", "p1", "--script", "ch09/01_pretrain.py", "--script", "ch99/01_train.py",
                    "--script", "ch99/02_gen.py"])
    out = fake_root / "web/data/runs/p1"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert set(report) == OLD_REPORT_KEYS
    pre, train, gen = report["steps"]

    # B3: every checkpoint the step wrote, with size and sha256; figures and other files are not checkpoints
    assert pre["checkpoints"] == [{"path": "webbot/model_pretrain.pt", "bytes": 100, "sha256": sha256(b"w" * 100)}]
    assert train["checkpoints"] == [{"path": "fakebot/model.pt", "bytes": 32, "sha256": sha256(b"m" * 32)}]
    assert "loss_train.png" in train["produced"] and "checkpoints" not in gen

    # B2: timing.json for the pretraining script gpu/costs.py models (bots.json), facts from the plan
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    assert pre["extracts"] == ["timing.json"] and "extracts" not in train
    assert timing == {
        "first_step_s": 16, "steady_s_per_step_20_200": 1.0, "tokens_per_step": 32 * 4 * 1024,
        "note": "from tqdm elapsed times in the raw log (1 s resolution); "
                "step 0->1 includes torch.compile and the step-0 evaluation",
        "script": "ch09/01_pretrain.py", "nproc": 1,
        "steady_s_per_step": 1.0, "steady_steps": 200, "steady_seconds": 200,
        "eval_bar": "Evaluating", "eval_batches": 40, "eval_passes": 1, "eval_first_pass_s": 6,
        "eval_it_per_s": 20.0, "eval_s": 2.0}
    assert pre["steady_s_per_it"] == 1.0 and {"rate", "s_per_it"} <= set(pre)       # old fields keep their meaning
    assert train["steady_s_per_it"] == 1.0

    # B2: a bar-less script's output, non-blank lines only (the rule of the committed hand extracts)
    assert gen["extracts"] == ["02_gen.output.txt"]
    assert (out / "02_gen.output.txt").read_bytes() == b"--- sample 1 ---\ndef f():\n    return 1\n"

    # a failing script leaves its log, but no extract that would look like a result
    work = fake_root / "runs/p1/work"
    run_local.prepare_script(work, "ch99/03_fail.py", [])
    step, _ = run_local.run_step(work, out, fake_root / "runs/p1", "ch99/03_fail.py", dict(os.environ))
    assert step["exit_code"] == 1 and "extracts" not in step and not (out / "03_fail.output.txt").exists()


def test_timing_summary_measures_between_evaluations():
    log = synthetic_log()                       # evaluations inside steps 0, 100 and 200
    t = run_local.timing_summary(log, compiled=True)
    assert t == {
        "first_step_s": 16,                     # 10 s of training + the step-0 evaluation: 1 + 2 + 3 s
        "note": "from tqdm elapsed times in the raw log (1 s resolution); "
                "step 0->1 includes torch.compile and the step-0 evaluation",
        "steady_s_per_step": 1.0, "steady_steps": 198, "steady_seconds": 198,     # [100,101], [200,201] left out
        "eval_bar": "Evaluating", "eval_batches": 40, "eval_passes": 3, "eval_first_pass_s": 6,
        "eval_it_per_s": 20.0, "eval_s": 2.0, "eval_later_pass_s": 2}
    # the legacy 20-200 reading exists only where that interval holds no evaluation
    assert "steady_s_per_step_20_200" not in t
    assert run_local.timing_summary(synthetic_log(eval_every=1000))["steady_s_per_step_20_200"] == 1.0
    # parse_timing keeps its meaning: the second half, evaluations included (so it reads slower)
    assert run_local.parse_timing(log)["s_per_it"] > 1.0
    # no compile, no step-0 evaluation: nothing to explain about the first step
    plain = "\r".join(f"{i}/40 [00:{i:02d}<00:10, 5.00it/s, loss=1.0]" for i in range(1, 41))
    assert run_local.timing_summary(plain)["note"] == "from tqdm elapsed times in the raw log (1 s resolution)"
    assert run_local.timing_summary("no progress bar here\n") == {}
    assert run_local.timing_record("no progress bar here\n", {"tokens_per_step": 8, "nproc": 1}) == {}


def test_output_text_keeps_what_the_script_printed():
    assert run_local.output_text("a\r\n\r\n  b\r\n\n") == "a\n  b\n"
    assert run_local.output_text("\n \n") is None
    assert run_local.output_text("Validation:  50%|#    | 334/668 [00:02<00:02, 160.0it/s]\nok\n") is None
    assert run_local.output_text("text\n 12it [00:01,  9.50it/s]\n") is None       # a bar without a total
    assert run_local.output_text("x = a/b [i]\n1/2 [ok]\n") == "x = a/b [i]\n1/2 [ok]\n"   # not a bar


def test_timing_facts_follow_bots_json_and_the_plan():
    _, bots = costs.load()
    web = bots["bots"]["webbot"]
    assert web["script"] == jobs.WEBBOT
    one = web["batch"] * web["accum"] * web["ctx"]
    assert run_local.timing_facts(["ch03/01_pretrain.py", "ch03/02_generate.py"]) == \
        {"ch03/01_pretrain.py": {"tokens_per_step": 32 * 256, "nproc": 1}}
    assert run_local.timing_facts(jobs.plan("webbot-probe").scripts, jobs.plan("webbot-probe")) == \
        {jobs.WEBBOT: {"tokens_per_step": one, "nproc": 1}}
    assert run_local.timing_facts([jobs.WEBBOT], jobs.plan("webbot", nproc=8)) == \
        {jobs.WEBBOT: {"tokens_per_step": 8 * one, "nproc": 8}}                 # max_iters / 8, 8x tokens per step
    assert run_local.timing_facts([jobs.WEBBOT], jobs.plan("webbot", nproc=8, scale="batch")) == \
        {jobs.WEBBOT: {"tokens_per_step": one, "nproc": 8}}                     # global batch kept
    assert run_local.timing_facts(["ch99/01_train.py"]) == {}


# ---------------------------------------------------------------- B2: the committed hand extracts, regenerated

def raw_log(run: str, stem: str) -> str:
    path = RAW_RUNS / run / f"{stem}.log"
    if not path.is_file():
        pytest.skip(f"raw log {path.relative_to(ROOT)} not here (runs/ is gitignored; it exists where the run ran)")
    return path.read_text(encoding="utf-8", errors="replace")


def test_regenerated_timing_json_reproduces_the_committed_one():
    """The probe's timing.json was cut by hand from the raw log. The runner's rules give the same values for
    every committed key; the differences are new keys only (eval_s etc.), listed here."""
    log = raw_log("webbot-probe", "01_pretrain")
    plan = jobs.plan("webbot-probe")
    facts = run_local.timing_facts(plan.scripts, plan)[jobs.WEBBOT]
    compiled = "torch.compile(" in (DLFS6 / jobs.WEBBOT).read_text(encoding="utf-8")
    regenerated = run_local.timing_record(log, facts, compiled=compiled, script=jobs.WEBBOT)
    committed = json.loads((RUNS / "webbot-probe" / "timing.json").read_text(encoding="utf-8"))
    assert {k: regenerated[k] for k in committed} == committed
    assert set(regenerated) - set(committed) == NEW_TIMING_KEYS
    new = {k: regenerated[k] for k in NEW_TIMING_KEYS}
    assert new == {"script": jobs.WEBBOT, "nproc": 1,
                   "steady_s_per_step": committed["steady_s_per_step_20_200"], "steady_steps": 180,
                   "steady_seconds": 57, "eval_bar": "Evaluating", "eval_batches": 1969, "eval_passes": 1,
                   "eval_first_pass_s": 101, "eval_it_per_s": 35.08, "eval_s": 56.1}
    # one evaluation at the steady rate (56 s) plus the wait for the first batch (19 s) and the stall on the
    # last one (26 s, see the next test) is the 101 s the step-0 evaluation took
    assert abs(new["eval_s"] + 19 + 26 - new["eval_first_pass_s"]) < 1


@pytest.mark.parametrize("run,stem", [("codebot-3060ti", "02_generate"), ("storybot-h100", "06_generate")])
def test_regenerated_output_txt_is_byte_identical(run, stem):
    regenerated = run_local.output_text(raw_log(run, stem))
    assert regenerated.encode("utf-8") == (RUNS / run / f"{stem}.output.txt").read_bytes()


def test_steady_rate_leaves_the_evaluations_out():
    """R10c on a real log: StoryBot's report says 0.0242 s/it, but that half-run average holds 41 evaluations.
    Between evaluations a step takes about 16 ms; each of the 81 evaluations about 4 s."""
    t = run_local.timing_summary(raw_log("storybot-h100", "05_pretrain"))
    report = json.loads((RUNS / "storybot-h100" / "report.json").read_text(encoding="utf-8"))
    step = next(s for s in report["steps"] if s["script"] == "ch06/05_pretrain.py")
    assert t["eval_bar"] == "Validation" and t["eval_batches"] == 668
    assert t["eval_passes"] == costs.n_evals(40_000, 500) == 81
    assert 0.015 < t["steady_s_per_step"] < 0.017 and step["s_per_it"] > 1.4 * t["steady_s_per_step"]
    assert abs(t["eval_s"] - t["eval_later_pass_s"]) < 0.5 and 3.5 < t["eval_s"] < 4.5
    assert run_local.output_text(raw_log("webbot-probe", "01_pretrain")) is None    # training logs get no output.txt


# ---------------------------------------------------------------- B5: the probe excerpt shows the stall

EVAL_LINE = re.compile(r"^Evaluating:.*?(\d+)/(\d+) \[(\d+):(\d+)<[^,]*, *([\d.]+)(it/s|s/it)\]")


def test_probe_excerpt_shows_the_one_time_eval_stall():
    """The excerpt now keeps the end of the step-0 evaluation: 1965/1969 at 01:15, still 1965 at 01:30, done at
    01:41. The last, shorter batch (8 sequences) stalled about 26 s, consistent with a recompile for the new
    shape. The stall lies inside the first step's 141 s, which the 10.4 h extrapolation counts as measured,
    and the later evaluations are counted at ~35 it/s (~56 s each), so the extrapolation is unchanged."""
    excerpt = (RUNS / "webbot-probe" / "01_pretrain.log.txt").read_text(encoding="utf-8").split("\n")
    bars = [(int(m[1]), int(m[2]), 60 * int(m[3]) + int(m[4]), float(m[5]), m[6])
            for m in map(EVAL_LINE.match, excerpt) if m]
    assert [(n, s) for n, _, s, _, _ in bars[-4:]] == [(1961, 74), (1965, 75), (1965, 90), (1969, 101)]
    stall = bars[-1][2] - bars[-3][2]
    steady = statistics.median(r for n, _, _, r, unit in bars if unit == "it/s" and n >= 100)
    first_batch = next(s for n, _, s, _, _ in bars if n == 1)
    assert stall == 26 and round(steady, 1) == 35.1 and round(1969 / steady) == 56 and first_batch == 19
    assert abs(first_batch + 1969 / steady + stall - bars[-1][2]) < 1           # 19 + 56 + 26 = 101 s
    timing = json.loads((RUNS / "webbot-probe" / "timing.json").read_text(encoding="utf-8"))
    assert timing["first_step_s"] - bars[-1][2] == 40      # the step-0 evaluation is inside the first step

    log = RAW_RUNS / "webbot-probe" / "01_pretrain.log"
    if log.is_file():                                      # every bar of the raw log from 1961 on, verbatim
        raw = [c for c in re.split(r"[\r\n]", log.read_text(encoding="utf-8", errors="replace"))
               if (m := EVAL_LINE.match(c)) and int(m[1]) >= 1961]
        assert raw and all(line in excerpt for line in raw)


# ---------------------------------------------------------------- Modal: the same rules on both sides

class FakeVolume:
    """Stands in for modal.Volume: a local directory."""

    def __init__(self, root: Path):
        self.root = root

    def reload(self):
        pass

    def commit(self):
        pass


@pytest.fixture
def modal_app():
    pytest.importorskip("modal")
    import modal_app
    return modal_app


def test_modal_launch_meta_refuses_a_dirty_tree_and_a_foreign_report(tmp_path, monkeypatch, modal_app):
    root, book = make_tutorial(tmp_path, {jobs.WEBBOT: "print('ch09')\n"})
    monkeypatch.setattr(run_local, "ROOT", root)
    monkeypatch.setattr(run_local, "DLFS6", book)
    monkeypatch.setattr(modal_app, "RUNS_OUT", root / "web/data/runs")
    p, gpu_str, est = modal_app.plan_and_estimate("webbot-probe", "H100")
    meta = modal_app.launch_meta(p, gpu_str, est, "probe-x")
    assert meta == {"tutorial_commit": head(root), "dlfs6_commit": head(book), "gpu_request": "H100", "nproc": 1,
                    "usd_per_hour": round(est["price"] + est["extra_per_hour"], 4), "price_src": est["src"],
                    "price_checked": est["checked"], "tutorial": {"commit": head(root), "dirty": False},
                    "timing": {jobs.WEBBOT: {"tokens_per_step": 32 * 4 * 1024, "nproc": 1}}}

    write(root / "content/notes.md", "uncommitted\n")
    with pytest.raises(SystemExit, match="--allow-dirty"):
        modal_app.launch_meta(p, gpu_str, est, "probe-x")
    assert modal_app.launch_meta(p, gpu_str, est, "probe-x", allow_dirty=True)["tutorial"]["dirty"] is True

    for provider, refused in [(None, True), ("modal", False)]:       # a run id another provider recorded
        write(root / "web/data/runs/probe-x/report.json", json.dumps(
            {"run": "probe-x", "host": {}, **({"provider": provider} if provider else {}), "steps": []}))
        if refused:
            with pytest.raises(SystemExit, match="new run id"):
                modal_app.launch_meta(p, gpu_str, est, "probe-x", allow_dirty=True)
        else:
            modal_app.launch_meta(p, gpu_str, est, "probe-x", allow_dirty=True)


@pytest.mark.filterwarnings("ignore:The run_job function is executing locally")
def test_modal_run_job_records_provenance_and_refuses_another_host(tmp_path, monkeypatch, modal_app):
    book, volroot = tmp_path / "book", tmp_path / "vol"
    write(book / "ch99/01_train.py", bar_script(30, {"fakebot/model.pt": b"w" * 64}))
    write(book / "ch99/02_gen.py", text_script("--- sample ---\n\nhello\n"))
    volroot.mkdir()
    monkeypatch.setattr(modal_app, "vol", FakeVolume(volroot))
    monkeypatch.setattr(modal_app, "VOL", volroot)
    monkeypatch.setattr(modal_app, "WS", tmp_path / "ws")
    monkeypatch.setattr(modal_app, "BOOK", str(book))
    plan = jobs.Plan(job="fake", bot="fakebot", scripts=["ch99/01_train.py", "ch99/02_gen.py"]).to_dict()
    tree = {"commit": "c" * 40, "dirty": True, "diff_sha256": "e" * 64, "changed": ["content/notes.md"]}
    meta = {"tutorial_commit": "c" * 40, "dlfs6_commit": "d" * 40, "gpu_request": "L4", "usd_per_hour": 0.96,
            "price_src": "https://modal.com/pricing", "price_checked": "2026-10-11", "tutorial": tree,
            "timing": {"ch99/01_train.py": {"tokens_per_step": 64, "nproc": 1}}}

    result = modal_app.run_job.local(plan, "prov-run", meta)
    out = volroot / "runs/prov-run/out"
    report_path = out / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "refused" not in result and [s["exit_code"] for s in result["steps"]] == [0, 0]
    train, gen = report["steps"]
    assert train["tutorial"] == gen["tutorial"] == tree
    assert train["checkpoints"] == [{"path": "fakebot/model.pt", "bytes": 64, "sha256": sha256(b"w" * 64)}]
    assert train["extracts"] == ["timing.json"] and gen["extracts"] == ["02_gen.output.txt"]
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    assert (timing["script"], timing["tokens_per_step"], timing["nproc"]) == ("ch99/01_train.py", 64, 1)
    assert (out / "02_gen.output.txt").read_text(encoding="utf-8") == "--- sample ---\nhello\n"

    # a retry that lands on another kind of machine must not append under this report's host block
    report["host"] = {**report["host"], "gpu": "NVIDIA B200", "gpu_count": 1}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    before = report_path.read_bytes()
    again = modal_app.run_job.local(plan, "prov-run", meta, True)
    assert "new run id" in again["refused"] and again["steps"] == [] and report_path.read_bytes() == before

    # nor under a report that run_local.py recorded (no provider)
    write(volroot / "runs/local-run/out/report.json", json.dumps(
        {"run": "local-run", "host": run_local.host_info(), "cost_usd": 0.0, "steps": []}))
    assert "provider" in modal_app.run_job.local(plan, "local-run", meta)["refused"]
