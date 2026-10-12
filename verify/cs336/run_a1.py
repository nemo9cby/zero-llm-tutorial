"""Run CS336 Assignment 1's public tests against the DLFS6 book code and record the outcome.

    git clone https://github.com/stanford-cs336/assignment1-basics <A1>
    git -C <A1> checkout a158843b20107949f1a8d7df1b05cd33b9166712
    CUDA_VISIBLE_DEVICES=-1 uv run --with einops --with psutil python verify/cs336/run_a1.py --a1 <A1>

For each variant of verify/cs336/adapters_dlfs6.py ("as_shipped", "one_line_fixes") the A1 `tests/`
folder is copied into a temporary directory, its `adapters.py` is replaced by a two-line shim that
re-exports adapters_dlfs6, and pytest runs there (nothing is written into the clone, the repo or
third_party/dlfs6). A small pytest plugin records every test's outcome and the numbers behind it
(max |actual - expected| next to the test's own atol/rtol). Results go to verify/cs336/results.json
and, with --web, also to web/data/cs336/a1_results.json for the cs336 page.

On Windows the tokenizer tests `import resource` (Unix only) at module level; the runner provides an
empty stand-in module so the file can be collected. The two tests that actually use it are skipped by
the test file itself on non-Linux platforms.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINNED_A1 = "a158843b20107949f1a8d7df1b05cd33b9166712"
A1_URL = "https://github.com/stanford-cs336/assignment1-basics"
VARIANTS = ("as_shipped", "one_line_fixes")

TEST_ADAPTER = {
    "test_linear": "run_linear", "test_embedding": "run_embedding", "test_swiglu": "run_swiglu",
    "test_scaled_dot_product_attention": "run_scaled_dot_product_attention",
    "test_4d_scaled_dot_product_attention": "run_scaled_dot_product_attention",
    "test_multihead_self_attention": "run_multihead_self_attention",
    "test_multihead_self_attention_with_rope": "run_multihead_self_attention_with_rope",
    "test_transformer_lm": "run_transformer_lm", "test_transformer_lm_truncated_input": "run_transformer_lm",
    "test_transformer_block": "run_transformer_block", "test_rmsnorm": "run_rmsnorm", "test_rope": "run_rope",
    "test_silu_matches_pytorch": "run_silu", "test_get_batch": "run_get_batch",
    "test_softmax_matches_pytorch": "run_softmax", "test_cross_entropy": "run_cross_entropy",
    "test_gradient_clipping": "run_gradient_clipping", "test_adamw": "get_adamw_cls",
    "test_get_lr_cosine_schedule": "run_get_lr_cosine_schedule", "test_checkpointing": "run_save_checkpoint",
    "test_train_bpe_speed": "run_train_bpe", "test_train_bpe": "run_train_bpe",
    "test_train_bpe_special_tokens": "run_train_bpe",
}

SHIM = '''import sys
sys.path.insert(0, {here!r})
from adapters_dlfs6 import *  # noqa: F401,F403  (DLFS6 book code behind the A1 interface)
'''

PLUGIN = r'''
"""pytest plugin: record outcomes and the numeric comparisons behind them."""
import json, os, re, sys, time
import numpy as np
import numpy.testing
import torch

RECORD = {"tests": {}, "diagnostics": {}}
CURRENT = {"nodeid": None}
_np_allclose = numpy.testing.assert_allclose
_torch_allclose = torch.allclose


def _measure(kind, actual, desired, rtol, atol):
    a = np.asarray(actual, dtype=np.float64)
    d = np.asarray(desired, dtype=np.float64)
    diff = np.abs(a - d)
    tol = atol + rtol * np.abs(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tol > 0, diff / tol, np.where(diff > 0, np.inf, 0.0))
    m = {"kind": kind, "max_abs_diff": float(diff.max()) if diff.size else 0.0,
         "atol": float(atol), "rtol": float(rtol),
         "tol_ratio": float(ratio.max()) if diff.size else 0.0,
         "n": int(diff.size), "n_violations": int((diff > tol).sum())}
    if diff.size <= 32:
        m["actual"] = [round(float(x), 6) for x in a.ravel()]
        m["desired"] = [round(float(x), 6) for x in d.ravel()]
    return m


def _record(m):
    nid = CURRENT["nodeid"]
    if nid is None:
        return
    rec = RECORD["tests"].setdefault(nid, {})
    rec["n_comparisons"] = rec.get("n_comparisons", 0) + 1
    worst = rec.get("worst")
    if worst is None or m["tol_ratio"] > worst["tol_ratio"]:
        rec["worst"] = m


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg="", verbose=True, **kw):
    try:
        _record(_measure("numpy.testing.assert_allclose", actual, desired, rtol, atol))
    except Exception as e:  # the recorder must never change a test's outcome
        RECORD["tests"].setdefault(CURRENT["nodeid"] or "?", {})["record_error"] = repr(e)
    return _np_allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan,
                        err_msg=err_msg, verbose=verbose, **kw)


def allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    try:
        _record(_measure("torch.allclose", input.detach().cpu(), other.detach().cpu(), rtol, atol))
    except Exception as e:
        RECORD["tests"].setdefault(CURRENT["nodeid"] or "?", {})["record_error"] = repr(e)
    return _torch_allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


numpy.testing.assert_allclose = assert_allclose
torch.allclose = allclose


def _summary(text, limit=8):
    lines = [l[1:].strip() for l in text.splitlines() if l.startswith("E ")]
    if not lines:
        lines = [l.strip() for l in text.splitlines() if l.strip()][-3:]
    out = []
    for l in lines:
        l = re.sub(r"Counter\(\{.*?\}\)", "Counter({...})", l)
        out.append(l if len(l) <= 240 else l[:237] + "...")
    return "\n".join(out[:limit])


def pytest_runtest_setup(item):
    CURRENT["nodeid"] = item.nodeid


def pytest_runtest_teardown(item, nextitem):
    CURRENT["nodeid"] = None


def pytest_runtest_logreport(report):
    rec = RECORD["tests"].setdefault(report.nodeid, {})
    if report.when == "call" or (report.when == "setup" and not report.passed):
        rec["outcome"] = report.outcome
        rec["duration"] = round(report.duration, 4)
        if getattr(report, "wasxfail", None) is not None:
            rec["xfail"] = True
        if report.failed:
            rec["message"] = _summary(report.longreprtext)
        elif report.skipped and isinstance(report.longrepr, tuple):
            rec["message"] = str(report.longrepr[-1])


def _diag_bpe(A, common):
    dec = {v: k for k, v in common.gpt2_bytes_to_unicode().items()}
    fx = common.FIXTURES_PATH
    ref = []
    for line in (fx / "train-bpe-reference-merges.txt").read_text(encoding="utf-8").splitlines():
        a, b = line.rstrip().split(" ")
        ref.append((bytes(dec[c] for c in a), bytes(dec[c] for c in b)))
    times = []
    for _ in range(5):  # the A1 speed bar is 1.5 s for one call; report the spread, not one sample
        t0 = time.perf_counter()
        _, merges = A.run_train_bpe(fx / "corpus.en", 500, ["<|endoftext|>"])
        times.append(time.perf_counter() - t0)
    first = next((i for i, (x, y) in enumerate(zip(merges, ref)) if x != y), None)
    out = {"corpus": "corpus.en", "vocab_size": 500, "n_merges": len(merges), "n_reference": len(ref),
           "identical_positions": sum(x == y for x, y in zip(merges, ref)), "first_divergence": first,
           "exact": merges == ref, "same_set": set(merges) == set(ref),
           "train_seconds_runs": [round(t, 3) for t in times],
           "train_seconds_min": round(min(times), 3),
           "train_seconds_median": round(sorted(times)[len(times) // 2], 3)}
    if first is not None:
        out["dlfs6_at_first"] = repr(merges[first])
        out["reference_at_first"] = repr(ref[first])
    import pickle
    snap = pickle.loads((fx.parent / "_snapshots" / "test_train_bpe_special_tokens.pkl").read_bytes())
    vocab, merges = A.run_train_bpe(fx / "tinystories_sample_5M.txt", 1000, ["<|endoftext|>"])
    ref = snap["merges"]
    first = next((i for i, (x, y) in enumerate(zip(merges, ref)) if x != y), None)
    out5 = {"corpus": "tinystories_sample_5M.txt", "vocab_size": 1000, "n_merges": len(merges),
            "n_reference": len(ref), "identical_positions": sum(x == y for x, y in zip(merges, ref)),
            "first_divergence": first, "exact": merges == ref,
            "vocab_values_equal": set(vocab.values()) == snap["vocab_values"]}
    if first is not None:
        out5["dlfs6_at_first"] = repr(merges[first])
        out5["reference_at_first"] = repr(ref[first])
    return [out, out5]


def _diag_rmsnorm(A, common):
    sd = torch.load(common.FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    w = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}["layers.1.ln1.weight"]
    torch.manual_seed(4)
    x = torch.randn(4, 12, 64)                      # the A1 fixture `in_embeddings`
    want = np.load(common.FIXTURES_PATH.parent / "_snapshots" / "test_rmsnorm.npz")["array"]
    out = {}
    for name, eps in (("nn.RMSNorm(d) default eps", None), ("nn.RMSNorm(d, eps=1e-5)", 1e-5)):
        norm = torch.nn.RMSNorm(64, eps=eps)
        with torch.no_grad():
            norm.weight.copy_(w)
        out[name] = float(np.abs(norm(x).detach().numpy() - want).max())
    hand = A.load_defs("ch05/03_rmsnorm.py")["RMSNorm"](64)
    with torch.no_grad():
        hand.gamma.copy_(w)
    out["ch05/03 RMSNorm (eps=1e-5, 手写)"] = float(np.abs(hand(x).detach().numpy() - want).max())
    out["finfo(float32).eps"] = float(torch.finfo(torch.float32).eps)
    return out


def pytest_sessionfinish(session, exitstatus):
    A = sys.modules.get("adapters_dlfs6")
    try:
        from tests import common
        if A is None:
            import adapters_dlfs6 as A
        RECORD["diagnostics"]["train_bpe"] = _diag_bpe(A, common)
        RECORD["diagnostics"]["rmsnorm_vs_snapshot"] = _diag_rmsnorm(A, common)
    except Exception as e:
        import traceback
        RECORD["diagnostics"]["error"] = traceback.format_exc()
    RECORD["exitstatus"] = int(exitstatus)
    with open(os.environ["A1_RECORD_PATH"], "w", encoding="utf-8") as f:
        json.dump(RECORD, f, ensure_ascii=False, indent=1)
'''


def git(*args, cwd) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def run_variant(a1: Path, variant: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"a1-{variant}-") as tmp:
        tmp = Path(tmp)
        shutil.copytree(a1 / "tests", tmp / "tests")
        (tmp / "tests" / "adapters.py").write_text(SHIM.format(here=str(HERE)), encoding="utf-8")
        (tmp / "a1_recorder.py").write_text(PLUGIN, encoding="utf-8")
        if sys.platform == "win32":
            (tmp / "resource.py").write_text('"""Windows stand-in: the tests that use it are skipped off Linux."""\n',
                                             encoding="utf-8")
        record = tmp / "record.json"
        env = {**os.environ, "DLFS6_A1_VARIANT": variant, "A1_RECORD_PATH": str(record),
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "CUDA_VISIBLE_DEVICES": "-1", "MPLBACKEND": "Agg", "TQDM_DISABLE": "1",
               "PYTHONPATH": str(tmp)}
        cmd = [sys.executable, "-m", "pytest", "tests", "-p", "a1_recorder", "-p", "no:cacheprovider",
               "-q", "-o", "addopts=", "--rootdir", str(tmp), "--tb=short", "-W", "ignore::DeprecationWarning"]
        proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8")
        if not record.exists():
            raise RuntimeError(f"pytest did not produce a record for {variant}:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
        data = json.loads(record.read_text(encoding="utf-8"))
        data["pytest_tail"] = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a1", type=Path, default=os.environ.get("CS336_A1_DIR"),
                    help="clone of stanford-cs336/assignment1-basics (or set CS336_A1_DIR)")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    ap.add_argument("--web", action="store_true", help="also write web/data/cs336/a1_results.json")
    args = ap.parse_args()
    if not args.a1 or not (Path(args.a1) / "tests" / "adapters.py").is_file():
        sys.exit("need --a1 <clone of assignment1-basics> (see the module docstring)")
    a1 = Path(args.a1).resolve()
    commit = git("rev-parse", "HEAD", cwd=a1)
    if commit != PINNED_A1:
        print(f"warning: A1 clone is at {commit}, results were pinned at {PINNED_A1}", file=sys.stderr)

    sys.path.insert(0, str(HERE))
    import adapters_dlfs6 as A  # noqa: E402  (CATALOG and the exact one-line fixes)
    import numpy
    import torch

    raw = {v: run_variant(a1, v) for v in VARIANTS}
    tests = []
    nodeids = sorted({n for v in VARIANTS for n in raw[v]["tests"]},
                     key=lambda n: (n.split("::")[0], n))
    for nid in nodeids:
        file, _, name = nid.partition("::")
        adapter = TEST_ADAPTER.get(name, "get_tokenizer" if "test_tokenizer" in file else "?")
        kind, source = A.CATALOG.get(adapter, ("?", "?"))
        row = {"test": name, "file": Path(file).name, "adapter": adapter, "kind": kind, "dlfs6": source}
        for v in VARIANTS:
            r = raw[v]["tests"].get(nid, {})
            row[v] = {k: r[k] for k in ("outcome", "duration", "message", "worst", "n_comparisons", "xfail")
                      if k in r}
        tests.append(row)

    def summary(v):
        out = {"passed": 0, "failed": 0, "skipped": 0}
        for row in tests:
            out[row[v].get("outcome", "skipped")] += 1
        return out

    result = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cs336_a1": {"repo": A1_URL, "commit": commit, "commit_date": git("show", "-s", "--format=%cI", "HEAD", cwd=a1),
                     "handout": "cs336_assignment1_basics.pdf (v26.0.3)"},
        "dlfs6_commit": git("rev-parse", "HEAD", cwd=ROOT / "third_party" / "dlfs6"),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
                "platform": platform.platform(terse=True), "device": "cpu", "cpu_count": os.cpu_count(),
                "note": "timings were taken while another training job was running on the same machine"},
        "command": "CUDA_VISIBLE_DEVICES=-1 uv run --with einops --with psutil python verify/cs336/run_a1.py --a1 <A1 clone> --web",
        "fixes": {
            "rmsnorm_eps": {"file": "storybot/model.py", "before": "nn.RMSNorm(embed_dim)",
                            "after": "nn.RMSNorm(embed_dim, eps=1e-5)"},
            "bpe_tiebreak": {"file": "ch04/02_bpe_cache.py", "before": A.TIEBREAK_FIX[0], "after": A.TIEBREAK_FIX[1]},
            "get_batch": {"file": "ch06/05_pretrain.py", "before": A.GET_BATCH_FIX[0], "after": A.GET_BATCH_FIX[1]},
        },
        "summary": {v: summary(v) for v in VARIANTS},
        "pytest_tail": {v: raw[v]["pytest_tail"] for v in VARIANTS},
        "diagnostics": {v: raw[v]["diagnostics"] for v in VARIANTS},
        "catalog": {k: {"kind": kind, "dlfs6": src} for k, (kind, src) in A.CATALOG.items()},
        "tests": tests,
    }
    text = json.dumps(result, ensure_ascii=False, indent=1)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if args.web:
        web = ROOT / "web" / "data" / "cs336" / "a1_results.json"
        web.parent.mkdir(parents=True, exist_ok=True)
        web.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {web}")
    for v in VARIANTS:
        print(v, result["summary"][v], "|", result["pytest_tail"][v])


if __name__ == "__main__":
    main()
