"""GPU 实战手册（content/gpu.md）：本地验证。

Everything here runs on the CPU in well under two minutes and never contacts Modal, RunPod or Lambda:
  * the numbers on the page come from the book's real models and scripts (meta-device models, ast);
  * the page's verify cells, gpu/costs.py and the widget's JS agree;
  * gpu/jobs.py patches keep WebBot's token budget on N GPUs, applied to the real ch09 script;
  * gpu/run_local.py and gpu/modal_app.py record runs in the same format (fake book, fake Volume);
  * gpu/setup_gpu_box.sh picks the right torch build for a host's driver (dry run, fake nvidia-smi);
  * the recorded runs the page quotes are what the raw logs and git history say they are.
The one exception is marked `slow`: it asks uv to resolve the setup script's torch flags (needs the network).
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import io
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from conftest import DLFS6, ROOT

GPU = ROOT / "gpu"
DATA = ROOT / "web" / "data" / "gpu"
RUNS = ROOT / "web" / "data" / "runs"
RAW_RUNS = ROOT / "runs"                      # raw logs of the recorded runs (gitignored, present where they ran)
sys.path.insert(0, str(GPU))
sys.path.insert(0, str(ROOT / "verify"))
import costs  # noqa: E402
import jobs  # noqa: E402
import run_local  # noqa: E402

BOTS = json.loads((DATA / "bots.json").read_text(encoding="utf-8"))
GPUS = json.loads((DATA / "gpus.json").read_text(encoding="utf-8"))
PLAN = json.loads((DATA / "plan.json").read_text(encoding="utf-8"))
PAGE = (ROOT / "content" / "gpu.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- helpers

def _value(node: ast.AST, env: dict):
    """Evaluate a constant expression from a book script (numbers, names already seen, + - * / //)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp):
        a, b = _value(node.left, env), _value(node.right, env)
        ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
               ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b}
        return ops[type(node.op)]()
    if isinstance(node, ast.List):
        return [_value(e, env) for e in node.elts]
    raise ValueError(ast.dump(node))


def script_constants(text: str) -> dict:
    """Top-level `name = <constant expression>` assignments of a script."""
    env: dict = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = _value(node.value, env)
            except (ValueError, KeyError):
                pass
    return env


def page_cells() -> list[str]:
    """The verify cells of content/gpu.md, in page order (same regex as tools/build.py)."""
    return [m.group(2) for m in re.finditer(r"^```verify[ \t]+(.*?)\n(.*?)^```[ \t]*$", PAGE, re.S | re.M)]


def page_ids(slug: str) -> set[str]:
    """Anchors a built page will have: numbered h2/h3 headings (as tools/build.py names them),
    verify cells (v-<id>) and explicit id="..." attributes."""
    text = (ROOT / "content" / f"{slug}.md").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([^"]+)"', text))
    ids |= {"s" + m.replace(".", "-") for m in re.findall(r"^#{2,3} ((?:\d+|[A-Z]+)(?:\.\d+)+)\s", text, re.M)}
    ids |= {"v-" + v for v in re.findall(r"^```verify[ \t]+id=(\S+)", text, re.M)}
    return ids


def git_show(commit: str, path: str) -> str | None:
    """The file at a commit, or None when it does not exist there."""
    proc = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.stdout if proc.returncode == 0 else None


def setup_dry_run(tmp_path: Path, driver: str) -> str:
    """Run gpu/setup_gpu_box.sh with DRY_RUN=1 against a fake nvidia-smi that reports `driver`."""
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    fake = tmp_path / f"bin{driver}"
    fake.mkdir()
    smi = fake / "nvidia-smi"
    smi.write_bytes((
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do case "$a" in --query-gpu=driver_version) echo "' + driver + '"; exit 0;; esac; done\n'
        'echo "NVIDIA H100 80GB HBM3, ' + driver + ', 81559 MiB"\n').encode())
    smi.chmod(0o755)
    env = {**os.environ, "DRY_RUN": "1", "WORK": str(tmp_path / f"ws{driver}"), "BOTS": "storybot",
           "TUTORIAL_SRC": "/x/zllm.bundle", "PATH": str(fake) + os.pathsep + os.environ["PATH"]}
    proc = subprocess.run([bash, str(GPU / "setup_gpu_box.sh")], env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=60, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture(scope="module")
def page_ns():
    """Run every verify cell of the page in one namespace, from web/ so data/gpu/*.json resolve."""
    ns: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(ROOT / "web")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for code in page_cells():
                exec(compile(code, "<gpu.md cell>", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


def write_fake_script(path: Path, steps: int, outputs: dict[str, bytes], bar: bool = True) -> None:
    """A stand-in for a book script: chdir to the repo root like the book does, print a tqdm-style
    progress bar with a loss postfix (bar=False: plain text, like the generate scripts), write some files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ("    sys.stdout.write(f'\\r{i * 100 // STEPS}%| | {i + 1}/{STEPS} [00:{i:02d}<00:10, 5.00it/s, loss={2 - i / 100:.4f}]')\n"
            if bar else "    print(f'sample {i}')\n")
    path.write_text(
        "import os, sys\n"
        "os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))\n"
        f"STEPS = {steps}\n"
        "for i in range(STEPS):\n"
        + body +
        "print()\n"
        f"for rel, data in {outputs!r}.items():\n"
        "    os.makedirs(os.path.dirname(rel) or '.', exist_ok=True)\n"
        "    open(rel, 'wb').write(data)\n"
        "print('done', STEPS)\n",
        encoding="utf-8")


# ---------------------------------------------------------------- G.2: the numbers on the page

def test_param_counts_match_the_real_models(dlfs6_path):
    import torch
    from torch import nn
    from codebot.model import GPT as CodeGPT
    from storybot.model import GPT as StoryGPT
    from webbot.model import GPT as WebGPT

    b = BOTS["bots"]
    with torch.device("meta"):                       # shapes only: no memory, no init cost
        models = {
            "codebot": CodeGPT(vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6,
                               ff_dim=4 * 384, dropout_rate=0.1),
            "storybot": StoryGPT(10000, 256, 512, 16, 4, 1344, 10000),
            "webbot": WebGPT(50000, 1024, 768, 12, 4, 12, 2048, 10000),
        }
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())          # tied weights counted once
        n_mm = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
        assert (params, n_mm) == (b[name]["params"], b[name]["n_mm"]) == costs.param_counts(b[name]), name
    assert models["codebot"].embed.weight is models["codebot"].unembed.weight      # tied, as bots.json says
    assert models["webbot"].embed.weight is not models["webbot"].unembed.weight


def test_bots_json_matches_the_book_scripts(dlfs6_path):
    b = BOTS["bots"]
    cb = script_constants((DLFS6 / "ch03/01_pretrain.py").read_text(encoding="utf-8"))
    sb_text = (DLFS6 / "ch06/05_pretrain.py").read_text(encoding="utf-8")
    sb = script_constants(sb_text)
    wb_text = (DLFS6 / "ch09/01_pretrain.py").read_text(encoding="utf-8")
    wb = script_constants(wb_text)
    assert (cb["context_len"], cb["vocab_size"], cb["batch_size"], cb["max_iters"], cb["embed_dim"],
            cb["n_head"], cb["n_layer"], cb["ff_dim"]) == (256, 1000, 32, 20000, 384, 6, 6, 1536)
    assert (b["codebot"]["ctx"], b["codebot"]["vocab"], b["codebot"]["batch"], b["codebot"]["max_iters"],
            b["codebot"]["d"], b["codebot"]["heads"], b["codebot"]["layers"], b["codebot"]["ff"]) == \
        (cb["context_len"], cb["vocab_size"], cb["batch_size"], cb["max_iters"], cb["embed_dim"],
         cb["n_head"], cb["n_layer"], cb["ff_dim"])
    for key, field in [("context_len", "ctx"), ("vocab_size", "vocab"), ("batch_size", "batch"),
                       ("max_iters", "max_iters"), ("embed_dim", "d"), ("n_head", "heads"),
                       ("n_layer", "layers"), ("ff_dim", "ff"), ("eval_iters", "eval_every")]:
        assert sb[key] == b["storybot"][field], key
    for key, field in [("context_len", "ctx"), ("vocab_size", "vocab"), ("micro_batch_size", "batch"),
                       ("accumulation_steps", "accum"), ("max_iters", "max_iters"), ("embed_dim", "d"),
                       ("n_head", "heads"), ("n_kv_head", "kv_heads"), ("n_layer", "layers"),
                       ("ff_dim", "ff"), ("eval_interval", "eval_every")]:
        assert wb[key] == b["webbot"][field], key
    # the loop and window conditions that costs.n_evals / costs.eval_windows reproduce
    assert "if (i % eval_iters) == 0 or i == max_iters - 1:" in sb_text
    assert "if is_main and ((step % eval_interval) == 0 or step == max_iters - 1):" in wb_text
    for text in (sb_text, wb_text):
        assert "ix = ix[ix + context_len + 1 < len(data)]" in text
    # precision: CodeBot has no autocast (fp32), StoryBot and WebBot use bf16 autocast
    assert "autocast" not in (DLFS6 / "ch03/01_pretrain.py").read_text(encoding="utf-8")
    assert "dtype=torch.bfloat16" in sb_text and "dtype=torch.bfloat16" in wb_text
    # validation set sizes (uint16 tokens) when the data has been fetched
    for bot, rel in [("storybot", "storybot/tiny_stories_valid.bin"), ("webbot", "webbot/owt_valid.bin")]:
        if (DLFS6 / rel).is_file():
            assert (DLFS6 / rel).stat().st_size // 2 == b[bot]["valid_tokens"], rel


def test_page_cells_agree_with_costs_module(page_ns):
    ns = page_ns
    gpu_by, offer_by = {g["id"]: g for g in GPUS["gpus"]}, {o["id"]: o for o in GPUS["offers"]}
    for name, b in BOTS["bots"].items():
        assert ns["param_counts"](b) == costs.param_counts(b)
    for it, every in [(40000, 500), (100000, 1000), (12500, 1000), (23000, 1000), (1001, 1000), (20000, 0)]:
        for steps in (None, 200, it):
            assert ns["n_evals"](it, every, steps) == costs.n_evals(it, every, steps)
    for s in BOTS["scenarios"]:
        b = BOTS["bots"][s["bot"]]
        kw = dict(world=s["world"], max_iters=s.get("max_iters"), batch=s.get("batch"), accum=s.get("accum"))
        mine, theirs = ns["run_shape"](b, **kw), costs.run_shape(b, **kw)
        assert {k: mine[k] for k in mine} == {k: theirs[k] for k in mine}
        assert ns["flops"](b, mine) == costs.flops(b, theirs)
        for oid in (s["offer"], "modal:h100", "runpod-community:rtx4090"):
            offer = offer_by[oid]
            gpu = gpu_by[offer["gpu"]]
            for floor in (0.0, 15.0):
                e1 = ns["estimate"](b, gpu, ns["price"](offer), mine, s["mfu"], floor)
                e2 = costs.estimate(b, gpu, costs.price_per_gpu_hour(offer), theirs, s["mfu"], floor)
                assert math.isclose(e1["wall_h"], e2["wall_hours"], rel_tol=1e-12)
                assert math.isclose(e1["usd"], e2["usd"], rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(e1["step_ms"], e2["step_ms"], rel_tol=1e-12)
    # G.7-a reads tqdm lines with the very patterns run_local.py records metrics with
    assert (ns["TQDM_RE"].pattern, ns["POSTFIX_RE"].pattern) == \
        (run_local.TQDM_RE.pattern, run_local.POSTFIX_RE.pattern)


def test_book_commit_on_the_page_is_the_submodule():
    """The runboard's '与子模块一致' badge compares reports with bots.json; the setup script checks it too."""
    head = run_local.git_rev(DLFS6)
    if head == "unknown":
        pytest.skip("third_party/dlfs6 is not a git checkout")
    assert BOTS["dlfs6_commit"] == head
    assert re.search(r"^DLFS6_COMMIT=(\w+)", (GPU / "setup_gpu_box.sh").read_text(encoding="utf-8"),
                     re.M).group(1) == head


def test_page_has_no_control_characters():
    """A backslash escape collapsed by a shell (\\t in \\text, \\b in \\beta) leaves an invisible control
    character that KaTeX renders without an error. tools/build.py tolerates TAB; this page never needs one."""
    for path in (ROOT / "content" / "gpu.md", GPU / "runpod_lambda.md"):
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            m = re.search(r"[\x00-\x08\x09\x0b-\x1f\x7f]", line)
            assert m is None, f"{path.relative_to(ROOT)}:{i}: U+{ord(m.group()):04X}"


def test_js_fixture_is_current():
    import _gpu_fixtures

    on_disk = json.loads(_gpu_fixtures.FIXTURE.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(_gpu_fixtures.build())), \
        "stale fixture: run `uv run python verify/_gpu_fixtures.py`, then `node --test tests/js/gpu-costmodel.test.mjs`"


def test_prices_cite_sources_checked_2026_10():
    assert GPUS["checked"].startswith("2026-10")
    gpu_ids = {g["id"] for g in GPUS["gpus"]}
    for g in GPUS["gpus"]:
        assert g["src"] and all(u.startswith("https://") for u in g["src"]), g["id"]
        assert g["bf16"] is None or g["bf16"] > 0
    for pid, p in GPUS["providers"].items():
        assert pid == "local" or p["src"].startswith("https://"), pid
    for o in GPUS["offers"]:
        assert o["gpu"] in gpu_ids and o["provider"] in GPUS["providers"], o["id"]
        price = costs.price_per_gpu_hour(o)
        assert (price == 0) == (o["provider"] == "local"), o["id"]
        assert o["counts"] and all(n in (1, 2, 4, 8) for n in o["counts"]), o["id"]


def test_plan_json_estimates_are_current():
    tree = ast.parse((GPU / "modal_app.py").read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:   # CPU_PER_GPU, GIB_PER_GPU = 2.0, 8
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            for t, v in zip(node.targets[0].elts, node.value.elts):
                consts[t.id] = ast.literal_eval(v)
    plan = json.loads((DATA / "plan.json").read_text(encoding="utf-8"))
    for p in plan["planned"]:
        modal = p["offer"].startswith("modal:")
        est = costs.estimate_plan(jobs.plan(p["job"], nproc=p["nproc"]).to_dict(), p["offer"], p["nproc"],
                                  cpu_cores=consts["CPU_PER_GPU"] * p["nproc"] if modal else 0,
                                  mem_gib=consts["GIB_PER_GPU"] * p["nproc"] if modal else 0)
        assert [round(x, 2) for x in est["usd"]] == p["estimate_usd"], p["run"]
        if modal:
            assert f"--job {p['job']}" in p["command"] and f"--budget" in p["command"]
            budget = float(p["command"].split("--budget")[1].split()[0])
            assert budget >= est["usd"][1], f"{p['run']}: the documented --budget would refuse its own job"


def test_page_links_into_other_pages_resolve():
    """`build.py --only gpu` checks links into other pages only for the page; check their #anchors here."""
    links = re.findall(r"\]\(([\w-]+)\.html#([^)\s]+)\)", PAGE)
    assert len(links) >= 10
    for slug, frag in links:
        assert frag in page_ids(slug), f"gpu.md links to {slug}.html#{frag}, which does not exist"


def test_runboard_cards_link_to_real_places():
    """Every recorded run gets a card with its re-run command; every 'used in' link points at a real anchor."""
    index = json.loads((RUNS / "index.json").read_text(encoding="utf-8"))
    planned = {p["run"]: p for p in PLAN["planned"]}
    assert set(index) <= set(planned), "a recorded run has no entry (command, links) in plan.json"
    for run, p in planned.items():
        assert p["job"] in jobs.JOBS and f"--job {p['job']}" in p["command"] and f"--run {run}" in p["command"]
        for link in p.get("used_in", []):
            page, _, frag = link["href"].partition("#")
            slug = page.removesuffix(".html") if page else "gpu"
            assert (ROOT / "content" / f"{slug}.md").is_file(), link
            assert not frag or frag in page_ids(slug), f"{run}: no #{frag} on {slug}"
    for run in index:
        report = json.loads((RUNS / run / "report.json").read_text(encoding="utf-8"))
        jobs_run = {s["job"] for s in report["steps"] if "job" in s}
        assert jobs_run <= {planned[run]["job"]}, run


# ---------------------------------------------------------------- G.3: multi-GPU budget

def test_ddp_patches_keep_the_token_budget(dlfs6_path):
    text = (DLFS6 / jobs.WEBBOT).read_text(encoding="utf-8")
    base = script_constants(text)

    def tokens(c, world):
        return c["micro_batch_size"] * c["accumulation_steps"] * c["context_len"] * c["max_iters"] * world

    one_gpu = tokens(base, 1)
    assert one_gpu == 13_107_200_000
    assert tokens(base, 8) == 8 * one_gpu                          # the trap: unchanged script, 8 ranks
    for n in (2, 4, 8):
        for scale in ("iters", "batch"):
            patched = jobs.apply_patches(text, jobs.webbot_patches(n, None, scale))
            compile(patched, jobs.WEBBOT, "exec")
            c = script_constants(patched)
            assert tokens(c, n) == one_gpu, (n, scale)
            assert c["learning_rate"] == base["learning_rate"] == 6e-4    # the peak LR is never rescaled
            if scale == "iters":                                    # same LR-vs-tokens warmup within 1%
                warm_tokens = c["warmup_iters"] * c["micro_batch_size"] * c["accumulation_steps"] * c["context_len"] * n
                assert abs(warm_tokens / (500 * 131072) - 1) < 0.01, n
            else:                                                   # same global batch, same schedule
                assert c["micro_batch_size"] * c["accumulation_steps"] * n == 128
                assert (c["max_iters"], c["warmup_iters"]) == (100000, 500)
    with pytest.raises(ValueError):
        jobs.webbot_patches(3, None, "iters")                       # 100000 % 3 != 0
    with pytest.raises(ValueError):
        jobs.apply_patches(text, [jobs.Patch(jobs.WEBBOT, "context_len", "x", "ambiguous")])


def test_probe_patch_runs_the_first_200_steps(dlfs6_path):
    text = (DLFS6 / jobs.WEBBOT).read_text(encoding="utf-8")
    plan = jobs.plan("webbot-probe")
    patched = jobs.apply_patches(text, plan.patches_for(jobs.WEBBOT))
    compile(patched, jobs.WEBBOT, "exec")
    assert "pbar = tqdm(range(200), disable=not is_main)" in patched
    c = script_constants(patched)
    assert (c["max_iters"], c["warmup_iters"]) == (100000, 500)    # LR schedule of the real run
    shape = costs.run_shape(BOTS["bots"]["webbot"], steps=plan.overrides["steps"])
    assert (shape["steps"], shape["evals"]) == (200, 1)


def test_grpo_constants_match_the_book(dlfs6_path):
    import torch
    from torch.utils.data import DataLoader
    from codebot.utils import generate

    text = (DLFS6 / "ch03/09_grpo.py").read_text(encoding="utf-8")
    c = script_constants(text)
    assert (c["batch_size"], c["group_size"], c["max_iters"], c["eval_interval"]) == (32, 8, 500, 10)
    assert "for i in range(1, 10):" in text and "for j in range(1, 10):" in text      # 9 × 9 = 81 problems
    assert "full_text = generate(model, tokenizer, prompt, temperature=1.0)" in text  # one answer per call
    # Batches: the DataLoader keeps the last, smaller batch and cycle() replays the first epoch (errata E07).
    assert "dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)" in text
    assert "data_iter = cycle(dataloader)" in text and "drop_last" not in text
    torch.manual_seed(0)
    it = itertools.cycle(DataLoader(list(range(81)), batch_size=32, shuffle=True))
    sizes = [len(next(it)) for _ in range(500)]
    assert sizes[:3] == [32, 32, 17] and sum(sizes) == 13_510         # × 8 answers = 108,080 sampled
    # generate(): one sequence per call, the whole context recomputed for every new token (no KV cache)
    src = inspect.getsource(generate)
    assert "ids = torch.tensor([ids]" in src and "logits = model(ids)[:, -1, :]" in src
    assert "cache" not in src


# ---------------------------------------------------------------- G.4: this PC

def test_windows_torch_comes_from_the_cu130_index():
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    index = {i["name"]: i for i in pp["tool"]["uv"]["index"]}["pytorch-cu130"]
    assert index["url"] == "https://download.pytorch.org/whl/cu130" and index["explicit"] is True
    assert pp["tool"]["uv"]["sources"]["torch"] == [{"index": "pytorch-cu130", "marker": "sys_platform == 'win32'"}]
    book_lock = tomllib.loads((DLFS6 / "uv.lock").read_text(encoding="utf-8"))
    torch_pkgs = [p for p in book_lock["package"] if p["name"] == "torch"]
    assert torch_pkgs and all("pypi.org" in p["source"]["registry"] for p in torch_pkgs)   # the book: PyPI only
    if sys.platform == "win32":
        import torch
        assert torch.__version__.endswith("+cu130") and torch.version.cuda.startswith("13.")


@pytest.mark.skipif(sys.platform != "win32", reason="uv's venv launcher is a Windows detail")
def test_venv_python_is_only_a_launcher():
    # The NVIDIA per-program setting must name the process that touches the GPU.
    assert Path(sys._base_executable).resolve() != Path(sys.executable).resolve()
    proc = subprocess.Popen([sys.executable, "-c", "import os, sys; print(os.getpid(), sys._base_executable)"],
                            stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(timeout=60)
    child_pid, base = out.split(maxsplit=1)
    assert int(child_pid) != proc.pid                      # the launcher started another process
    assert Path(base.strip()).resolve() == Path(sys._base_executable).resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="checks native Windows")
def test_windows_cannot_run_ch09_natively():
    import torch
    import torch.distributed as dist
    from torch._inductor.scheduler import Scheduler

    assert not dist.is_nccl_available()                    # ch09 initialises DDP with backend='nccl'
    # ch09 compiles a CUDA model: inductor's GPU backend needs Triton, which the Windows wheel does not ship
    assert importlib.util.find_spec("triton") is None
    backend = inspect.getsource(Scheduler.create_backend)
    assert "if not has_triton():" in backend and "raise TritonMissing" in backend
    if shutil.which("cl") is None:                         # no MSVC on PATH: inductor cannot compile CPU code
        with pytest.raises(Exception, match="cl is not found"):
            torch.compile(lambda x: torch.sin(x) * 2 + 1)(torch.randn(8))


def find_bash() -> str | None:
    """A POSIX bash. On Windows use Git's: System32's bash.exe would run the script inside WSL instead."""
    if sys.platform != "win32":
        return shutil.which("bash")
    git = shutil.which("git")
    candidates = [Path(git).resolve().parent.parent / "bin" / "bash.exe"] if git else []
    candidates.append(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe")
    return next((str(c) for c in candidates if c.is_file()), None)


CU126 = "https://download.pytorch.org/whl/cu126"


def test_setup_script_picks_torch_by_driver(tmp_path):
    assert b"\r" not in (GPU / "setup_gpu_box.sh").read_bytes()     # CRLF would break bash on Linux
    old = setup_dry_run(tmp_path, "575.57.08")
    install = [ln for ln in old.splitlines() if ln.startswith("+ uv pip install")]
    assert len(install) == 1 and "UV_NO_SYNC=1" in old
    # --index outranks PyPI; --index-url + --extra-index-url PyPI would let PyPI's CUDA 13.0 wheel win
    assert f"--index {CU126}" in install[0] and "torch==2.14.1+cu126" in install[0]
    assert "--index-url" not in install[0] and "--extra-index-url" not in install[0]
    new = setup_dry_run(tmp_path, "610.60")
    assert "cu126" not in new and "PyPI wheel (CUDA 13.0)" in new and "+ uv pip install" not in new
    for out in (old, new):
        assert "+ git clone /x/zllm.bundle" in out and "+ git submodule update --init third_party/dlfs6" in out
        assert "+ uv sync --frozen" in out and "+ uv run python gpu/fetch.py storybot" in out


@pytest.mark.slow
def test_setup_script_torch_flags_resolve_to_cu126(tmp_path):
    """Ask uv itself (dry: `uv pip compile`, Linux x86_64, Python 3.12) what the script's flags install."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    line = next(ln for ln in setup_dry_run(tmp_path, "575.57.08").splitlines() if ln.startswith("+ uv pip install"))
    args = shlex.split(line[2:])[3:]                          # after `uv pip install`
    i = args.index("--reinstall-package")
    del args[i:i + 2]                                         # an install-only option
    reqs = [a for a in args if "==" in a]
    flags = [a for a in args if a not in reqs]

    def resolve(req: str, index_flags: list[str]) -> str:
        (tmp_path / "req.txt").write_text(req + "\n", encoding="utf-8")
        proc = subprocess.run([uv, "pip", "compile", "req.txt", "--no-deps", "--no-config", "--no-header",
                               "--no-annotate", "--python-platform", "x86_64-manylinux_2_28",
                               "--python-version", "3.12", *index_flags],
                              capture_output=True, text=True, encoding="utf-8", timeout=300, cwd=tmp_path)
        if proc.returncode != 0:
            if "No solution found" in proc.stderr:
                pytest.fail(proc.stderr[-1500:])
            pytest.skip(f"uv could not reach the package indexes: {proc.stderr[-300:]}")
        return proc.stdout

    assert reqs == ["torch==2.14.1+cu126"] and flags == ["--index", CU126]
    assert "torch==2.14.1+cu126" in resolve(reqs[0], flags)
    # the pitfall the script avoids: every --extra-index-url outranks --index-url, so PyPI's wheel wins
    old = resolve("torch==2.14.1", ["--index-url", CU126, "--extra-index-url", "https://pypi.org/simple"])
    assert re.search(r"^torch==2\.14\.1$", old, re.M), old


def test_fetch_plans_only_what_a_bot_needs():
    import fetch

    story = dict(fetch.plan_files(["storybot"]))
    assert "storybot/tiny_stories_train.bin" in story and not any(r.startswith("webbot/") for r in story)
    assert story["storybot/model_pretrain.pt"] == 90_835_030           # official checkpoint for storybot-dpo
    web = dict(fetch.plan_files(["webbot"], official=False))
    assert set(web) == {"webbot/owt_train.bin", "webbot/owt_valid.bin", "webbot/merge_rules.pkl"}
    assert sum(web.values()) == 5_298_009_600 + 128_992_132 + 533_514
    assert set(dict(fetch.plan_files(["codebot"]))) == {"codebot/model_pretrain.pt", "codebot/model_sft.pt"}


# ---------------------------------------------------------------- G.5: Modal

def test_modal_image_matches_the_lockfile():
    tree = ast.parse((GPU / "modal_app.py").read_text(encoding="utf-8"))
    pins = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PINS")
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    linux312 = {}
    for p in lock["package"]:
        markers = " ".join(p.get("resolution-markers", []))
        if markers and "sys_platform != 'win32'" not in markers:
            continue                                         # e.g. the Windows-only torch+cu130
        if markers and "python_full_version >= '3.12'" not in markers:
            continue                                         # numpy 2.4.6 is for Python 3.11
        linux312[p["name"]] = p
    for pin in pins:
        name, version = pin.split("==")
        assert linux312[name]["version"] == version, pin
    torch_deps = {d["name"]: d.get("marker", "") for d in linux312["torch"]["dependencies"]}
    assert "pypi.org" in linux312["torch"]["source"]["registry"]
    for dep in ("nvidia-cudnn-cu13", "nvidia-nccl-cu13"):
        assert torch_deps[dep] == "sys_platform == 'linux'"
    # the image's Python is the one .python-version pins for local runs
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
    assert 'modal.Image.debian_slim(python_version="3.12")' in (GPU / "modal_app.py").read_text(encoding="utf-8")


def test_hf_storybot_checkpoint_needs_the_key_rename(dlfs6_path, tmp_path):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from storybot.model import GPT

    src = DLFS6 / "storybot" / "model_pretrain.pt"
    if not src.is_file():
        pytest.skip("run `uv run python tools/fetch_data.py` for the official StoryBot checkpoint")
    with pytest.raises(RuntimeError, match=r"blocks\.0\.ffn"):
        GPT.load_from(str(src))                              # as published: keys say `mlp`, the code says `ffn`
    fixed = tmp_path / "model_pretrain.pt"
    assert jobs.fix_storybot_keys(src, fixed) == 12          # 4 blocks × (W, V, O)
    model = GPT.load_from(str(fixed))
    valid = DLFS6 / "storybot" / "tiny_stories_valid.bin"
    if valid.is_file():
        ids = torch.from_numpy(np.memmap(valid, dtype=np.uint16, mode="r")[:2 * 257].astype(np.int64)).view(2, 257)
        with torch.no_grad():
            logits = model(ids[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)).item()
        assert loss < 3.0 < math.log(10000)                  # trained weights, not a random init (ln V = 9.21)


def test_modal_plan_cli_contacts_nothing(tmp_path):
    pytest.importorskip("modal")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore",
           "MODAL_TOKEN_ID": "ak-invalid", "MODAL_TOKEN_SECRET": "as-invalid",
           "MODAL_SERVER_URL": "http://127.0.0.1:9", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run([sys.executable, str(GPU / "modal_app.py"), "plan", "--job", "webbot", "--gpu", "H100:8"],
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=120, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "'max_iters = 100000' -> 'max_iters = 12500'" in proc.stdout
    assert "13,107,200,000 tokens, 14 evals" in proc.stdout


def test_modal_budget_guard():
    pytest.importorskip("modal")
    import modal_app

    _, gpu_str, est = modal_app.plan_and_estimate("webbot", "H100")
    assert gpu_str == "H100" and modal_app.budget_error(est, 5.0)          # default budget refuses WebBot
    _, gpu_str, est = modal_app.plan_and_estimate("webbot", "H100:8")
    assert gpu_str == "H100:8" and modal_app.budget_error(est, 70.0) is None
    _, _, est = modal_app.plan_and_estimate("storybot-dpo")
    assert modal_app.budget_error(est, 5.0) is None
    assert modal_app.default_run_id("webbot", "H100:8") == "webbot-modal-h100x8"
    with pytest.raises(ValueError):
        modal_app.plan_and_estimate("storybot", "T4")                     # T4 has no bf16


class FakeVolume:
    """Stands in for modal.Volume: a local directory, with commit/reload counted."""

    def __init__(self, root: Path):
        self.root, self.commits = root, 0

    def reload(self):
        pass

    def commit(self):
        self.commits += 1

    def listdir(self, path: str, recursive: bool = False):
        from types import SimpleNamespace
        base = self.root / path
        files = base.rglob("*") if recursive else base.iterdir()
        return [SimpleNamespace(path=f.relative_to(self.root).as_posix(), type=SimpleNamespace(name="FILE"))
                for f in files if f.is_file()]

    def read_file_into_fileobj(self, path: str, fileobj) -> int:
        data = (self.root / path).read_bytes()
        fileobj.write(data)
        return len(data)


@pytest.mark.filterwarnings("ignore:The run_job function is executing locally")
def test_modal_run_job_with_a_fake_volume(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    import modal_app

    book, volroot = tmp_path / "book", tmp_path / "vol"
    write_fake_script(book / "ch99/01_train.py", 50, {"fakebot/model.pt": b"w" * 64, "loss_train.png": b"png"})
    write_fake_script(book / "ch99/02_eval.py", 3, {"eval.txt": b"ok"}, bar=False)
    (volroot / "data/fakebot").mkdir(parents=True)
    (volroot / "data/fakebot/data.bin").write_bytes(b"\0" * 32)
    (volroot / "data/fakebot/start.pt").write_bytes(b"official")
    fake = FakeVolume(volroot)
    monkeypatch.setattr(modal_app, "vol", fake)
    monkeypatch.setattr(modal_app, "VOL", volroot)
    monkeypatch.setattr(modal_app, "WS", tmp_path / "ws")
    monkeypatch.setattr(modal_app, "BOOK", str(book))
    plan = jobs.Plan(job="fake", bot="fakebot", scripts=["ch99/01_train.py", "ch99/02_eval.py"],
                     data=["fakebot/data.bin"], seeds={"fakebot/start.pt": "official"},
                     patches=[jobs.Patch("ch99/01_train.py", "STEPS = 50", "STEPS = 30", "shorter, for the test")])
    meta = {"tutorial_commit": "t" * 40, "dlfs6_commit": "d" * 40, "gpu_request": "L4", "usd_per_hour": 0.96,
            "price_src": "https://modal.com/pricing", "price_checked": "2026-10-11"}

    first = modal_app.run_job.local(plan.to_dict(), "fake-run", meta)
    out = volroot / "runs/fake-run/out"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert [s["exit_code"] for s in first["steps"]] == [0, 0]
    assert report["provider"] == "modal" and report["dlfs6_commit"] == "d" * 40
    assert report["steps"][0]["patches"] == [{"old": "STEPS = 50", "new": "STEPS = 30", "why": "shorter, for the test"}]
    assert report["steps"][0]["final"]["it"] == 30 and report["steps"][0]["points"] == 30
    assert report["seeded"] == [{"path": "fakebot/start.pt", "from": "huggingface koki0702/zero-llm-data"}]
    assert (volroot / "runs/fake-run/work/fakebot/model.pt").read_bytes() == b"w" * 64   # persisted for later jobs
    assert (out / "01_train.metrics.jsonl").is_file() and (out / "loss_train.png").is_file()
    assert (book / "ch99/01_train.py").read_text(encoding="utf-8").count("STEPS = 50") == 1   # book untouched
    assert report["billing"][0]["seconds"] > 0 and report["cost_usd"] >= 0 and fake.commits >= 3

    second = modal_app.run_job.local(plan.to_dict(), "fake-run", meta)                    # a retry: nothing to redo
    report2 = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert second["steps"] == [] and len(report2["steps"]) == 2 and len(report2["billing"]) == 2

    dest = tmp_path / "web_runs"
    pulled = modal_app.pull("fake-run", dest_root=dest, volume=fake)
    assert {p.name for p in pulled} >= {"report.json", "01_train.metrics.jsonl", "loss_train.png"}
    assert json.loads((dest / "index.json").read_text(encoding="utf-8")) == {"fake-run": {"steps": ["01_train"]}}


# ---------------------------------------------------------------- G.7: run reports

def test_parse_timing_reads_the_main_bar_only():
    lines = [f"{i * 100 // 40}%| | {i}/40 [00:{i:02d}<00:10, 2.00it/s, loss={3 - i / 20:.4f}]" for i in range(1, 41)]
    lines.insert(5, "Validation: 100%|████| 668/668 [00:30<00:00, 22.0it/s]")      # an eval bar: no postfix
    t = run_local.parse_timing("\r".join(lines))
    assert t == {"rate": "2.00it/s", "s_per_it": 1.0}
    pts = run_local.parse_tqdm("\r".join(lines))
    assert len(pts) == 40 and pts[-1] == {"it": 40, "loss": 1.0}


def test_parse_tqdm_keeps_the_first_line_of_each_counter(dlfs6_path):
    """tqdm prints each counter twice: the refresh when the counter moves still carries the previous step's
    postfix, then set_postfix writes this step's. parse_tqdm keeps the first, so `it = k` holds what step k
    (i = k - 1) wrote: GRPO's evaluation at i = 10 is recorded at it = 11, while the book's own plot puts it
    at 10 (content/gpu.md G.7). The lines are verbatim from runs/codebot-3060ti/09_grpo.log."""
    lines = ["  2%|▏         | 10/500 [04:01<3:06:14, 22.81s/it, loss=0.1076, acc=53.1%]",
             "  2%|▏         | 10/500 [04:32<3:06:14, 22.81s/it, loss=0.0910, acc=61.7%]",
             "  2%|▏         | 11/500 [04:32<3:27:55, 25.51s/it, loss=0.0910, acc=61.7%]",
             "  2%|▏         | 11/500 [04:44<3:27:55, 25.51s/it, loss=0.2259, acc=61.7%]"]
    assert run_local.parse_tqdm("\r".join(lines)) == [{"it": 10, "loss": 0.1076, "acc": 53.1},
                                                      {"it": 11, "loss": 0.091, "acc": 61.7}]
    recorded = [json.loads(ln) for ln in (RUNS / "codebot-3060ti" / "09_grpo.metrics.jsonl")
                .read_text(encoding="utf-8").splitlines()]
    assert [r for r in recorded if r["it"] in (10, 11)] == run_local.parse_tqdm("\r".join(lines))
    # counter 0 has no refresh line, so in every recorded run it = 0 repeats it = 1 (the first step's postfix)
    for path in sorted(RUNS.glob("*/*.metrics.jsonl")):
        head = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()[:2]]
        assert head[0] == {**head[1], "it": 0}, path.relative_to(RUNS)
    log = RAW_RUNS / "codebot-3060ti" / "09_grpo.log"
    if log.is_file():                                       # consecutive in the raw log, in this order
        raw = [ln for ln in re.split(r"[\r\n]", log.read_text(encoding="utf-8", errors="replace")) if ln]
        k = raw.index(lines[0])
        assert raw[k:k + len(lines)] == lines
    # the book evaluates after the update of step i when i % 10 == 0, and plots accuracy against i itself
    text = (DLFS6 / "ch03/09_grpo.py").read_text(encoding="utf-8")
    loop = text[text.index("for i in pbar:"):text.index("model.save(grpo_model_save_path)")]
    assert loop.index("optimizer.step()") < loop.index("if i % eval_interval == 0:") < loop.index("pbar.set_postfix(")
    assert "steps = list(range(0, len(accuracies) * eval_interval, eval_interval))" in text


# Excerpts of runs/<run>/<stem>.log committed under web/data/runs/. Only the probe's log excerpt is purely
# hand-cut now: run_local.py writes timing.json and <stem>.output.txt itself, and the committed ones (cut by
# hand before it did) reproduce from the raw logs under its rules (test_gpu_runner.py).
HAND_EXTRACTS = [("webbot-probe", "01_pretrain.log.txt", "01_pretrain.log"),
                 ("codebot-3060ti", "02_generate.output.txt", "02_generate.log"),
                 ("storybot-h100", "06_generate.output.txt", "06_generate.log")]


def test_hand_extracts_match_the_raw_logs():
    """Each excerpt is verbatim log lines in log order; timing.json follows from the probe's raw log."""
    checked = 0
    for run, extract, raw in HAND_EXTRACTS:
        assert (RUNS / run / extract).is_file(), f"{run}/{extract}"
        log = RAW_RUNS / run / raw
        if not log.is_file():
            continue
        log_lines = iter(re.split(r"[\r\n]", log.read_text(encoding="utf-8", errors="replace")))
        lines = [ln for ln in (RUNS / run / extract).read_text(encoding="utf-8").split("\n") if ln.strip()]
        assert lines and all(any(ln == raw_ln for raw_ln in log_lines) for ln in lines), f"{run}/{extract}"
        checked += 1

    log = RAW_RUNS / "webbot-probe" / "01_pretrain.log"
    if log.is_file():
        first_seen: dict[int, int] = {}                      # counter -> tqdm's elapsed clock, in seconds
        for chunk in re.split(r"[\r\n]", log.read_text(encoding="utf-8", errors="replace")):
            m = run_local.TQDM_RE.search(chunk)
            if not m or not run_local.POSTFIX_RE.search(m.group(3)):
                continue                                     # evaluation bars carry no postfix
            h, mm, ss = (int(x) if x else 0 for x in run_local.ELAPSED_RE.match(m.group(3)).groups())
            first_seen.setdefault(int(m.group(1)), 3600 * h + 60 * mm + ss)
        timing = json.loads((RUNS / "webbot-probe" / "timing.json").read_text(encoding="utf-8"))
        wb = BOTS["bots"]["webbot"]
        assert timing["first_step_s"] == first_seen[1]
        assert timing["steady_s_per_step_20_200"] == round((first_seen[200] - first_seen[20]) / 180, 4)
        assert timing["tokens_per_step"] == wb["batch"] * wb["accum"] * wb["ctx"]
        checked += 1
    if not checked:
        pytest.skip("the raw logs (runs/<run>/*.log) are not in the repository; they exist where the runs ran")


def test_run_reports_point_at_commits_that_contain_their_code():
    """A report records the tutorial's HEAD when the run started. Where the code that ran was not committed
    yet, plan.json names the first commit that has it; git must agree on both counts."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    planned = {p["run"]: p for p in PLAN["planned"]}
    shas = {json.loads((RUNS / r / "report.json").read_text(encoding="utf-8"))["tutorial_commit"]
            for r in json.loads((RUNS / "index.json").read_text(encoding="utf-8"))}
    shas |= {p["code_commit"]["sha"] for p in PLAN["planned"] if "code_commit" in p}
    for sha in shas:
        if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{sha}^{{commit}}"]).returncode:
            pytest.skip(f"commit {sha[:7]} is not in this clone (shallow?)")

    for run in json.loads((RUNS / "index.json").read_text(encoding="utf-8")):
        report = json.loads((RUNS / run / "report.json").read_text(encoding="utf-8"))
        needs = [("gpu/run_local.py", "def parse_tqdm")]
        if report.get("provider") == "modal":
            needs.append(("gpu/modal_app.py", "def run_job"))
        needs += [("gpu/jobs.py", f'"{s["job"]}"') for s in report["steps"] if "job" in s]

        def has_code(commit: str) -> bool:
            return all((src := git_show(commit, path)) is not None and text in src for path, text in needs)

        fix = planned.get(run, {}).get("code_commit")
        if fix is None:
            assert has_code(report["tutorial_commit"]), f"{run}: its tutorial_commit lacks the code that ran"
            continue
        assert not has_code(report["tutorial_commit"]), f"{run}: the recorded commit is fine; drop code_commit"
        assert has_code(fix["sha"]), run
        assert fix["contains"] in (git_show(fix["sha"], fix["path"]) or ""), run
        assert fix["contains"] not in (git_show(fix["sha"] + "^", fix["path"]) or ""), \
            f"{run}: {fix['sha'][:7]} is not the first commit with {fix['path']}"
        assert subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", fix["sha"], "HEAD"]
                              ).returncode == 0, run


def test_run_local_records_a_run_end_to_end(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    dl = tmp_path / "dlfs6"
    write_fake_script(dl / "ch99/01_train.py", 40, {"fakebot/model.pt": b"m" * 32, "loss_train.png": b"png"})
    write_fake_script(dl / "ch99/02_gen.py", 2, {}, bar=False)
    (dl / "fakebot").mkdir(exist_ok=True)
    (dl / "fakebot/start.pt").write_bytes(b"official")        # untracked, like a fetched checkpoint
    git = ["git", "-C", str(dl), "-c", "user.email=t@example.com", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(dl)], check=True)
    subprocess.run(git + ["add", "ch99"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "fake book"], check=True)
    head = subprocess.check_output(["git", "-C", str(dl), "rev-parse", "HEAD"], text=True).strip()
    root = tmp_path / "root"
    monkeypatch.setattr(run_local, "ROOT", root)
    monkeypatch.setattr(run_local, "DLFS6", dl)

    # 1. the original CLI, unchanged: --run plus --script ...
    run_local.main(["--run", "t1", "--script", "ch99/01_train.py", "--script", "ch99/02_gen.py"])
    out = root / "web/data/runs/t1"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert set(report) == {"run", "tutorial_commit", "dlfs6_commit", "host", "cost_usd", "steps"}
    assert report["dlfs6_commit"] == head and report["cost_usd"] == 0.0
    s1, s2 = report["steps"]
    assert (s1["exit_code"], s1["points"], s1["final"]) == (0, 40, {"it": 40, "loss": 1.61})
    assert {"fakebot/model.pt", "loss_train.png"} <= set(s1["produced"]) and "patches" not in s1
    assert s2["points"] == 0 and s2["final"] is None
    assert (out / "01_train.metrics.jsonl").read_text(encoding="utf-8").count("\n") == 40
    assert (out / "loss_train.png").is_file()
    assert json.loads((root / "web/data/runs/index.json").read_text(encoding="utf-8")) == {"t1": {"steps": ["01_train"]}}
    assert (root / "runs/t1/work/fakebot/model.pt").is_file() and not (dl / "fakebot/model.pt").exists()

    # 2. a named job: official checkpoint seeded into the workspace, recorded in the report
    monkeypatch.setitem(jobs.JOBS, "fake", dict(bot="fakebot", scripts=["ch99/01_train.py"],
                                                seeds={"fakebot/start.pt": "official"}))
    run_local.main(["--run", "t2", "--job", "fake"])
    r2 = json.loads((root / "web/data/runs/t2/report.json").read_text(encoding="utf-8"))
    assert r2["seeded"] == [{"path": "fakebot/start.pt", "from": "huggingface koki0702/zero-llm-data"}]
    assert r2["steps"][0]["job"] == "fake"

    # 3. patches apply to the workspace copy only, and the next run restores the pristine script
    work = root / "runs/t1/work"
    patched = run_local.prepare_script(work, "ch99/01_train.py",
                                       [jobs.Patch("ch99/01_train.py", "STEPS = 40", "STEPS = 5", "quick")])
    assert patched == [{"old": "STEPS = 40", "new": "STEPS = 5", "why": "quick"}]
    step, points = run_local.run_step(work, out, root / "runs/t1", "ch99/01_train.py", dict(os.environ),
                                      run_local.launcher(1), patched)
    assert step["patches"] == patched and len(points) == 5
    assert "STEPS = 40" in (dl / "ch99/01_train.py").read_text(encoding="utf-8")
    assert run_local.prepare_script(work, "ch99/01_train.py", []) == []
    assert "STEPS = 40" in (work / "ch99/01_train.py").read_text(encoding="utf-8")

    # 4. torchrun for N GPUs, from the same interpreter; WebBot-only knobs are refused elsewhere
    assert run_local.launcher(8)[1:] == ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8"]
    with pytest.raises(SystemExit):
        run_local.main(["--run", "t3", "--script", "ch99/01_train.py", "--nproc", "8"])
    with pytest.raises(ValueError):
        jobs.plan("storybot", nproc=8)


def test_page_code_excerpts_match_their_sources():
    """Code shown on the page as ```<lang> <path> must be a verbatim excerpt of that file."""
    blocks = re.findall(r"^```(\w+) ([\w./-]+)\n(.*?)^```[ \t]*$", PAGE, re.S | re.M)
    shown = [(lang, path, body) for lang, path, body in blocks if lang not in ("verify", "localcheck", "widget", "listing")]
    assert len(shown) >= 5
    for lang, path, body in shown:
        src = (ROOT / path).read_text(encoding="utf-8")
        assert body in src, f"{path}: the excerpt on the page drifted from the file"
