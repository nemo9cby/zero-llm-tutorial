"""原书注释即断言：把 DLFS6 脚本里写在注释中的“期望输出”改写成断言来运行。

The DLFS6 scripts state many expected outputs as comments, either on the same line
(`print(ids)  # [259, 108, ...]`, 35 lines) or as a comment block right below the print
(ch03/03_alpaca.py, 3 prints). This module:

1. finds every single-argument `print(...)` that carries such a comment (AST + tokenize, no regex
   guessing about code), and checks that the inventory still matches the pinned submodule;
2. rewrites each of those calls into `__expect__(lineno, value)` and runs the script from a temporary
   copy of the repo (nothing is written into third_party/dlfs6; CUDA is hidden);
3. compares every recorded value with its comment and classifies the result:

   exact        the printed text equals the comment
   repr         the comment shows repr(value) where print shows str(value)      ('h' vs h)
   defaultdict  the comment shows the plain dict                                  ch01/03 count_pairs
   torch.Size   the comment shows the shape as a tuple                            (2, 10, 512)
   stale        a known out-of-date comment, asserted to differ exactly as documented in STALE

Only the scripts that contain such comments are run; all of them are CPU-only toys that write no
files. `uv run python verify/test_book_comments.py --json web/data/errata/book_comments.json`
writes the summary that the errata page reads (and test_summary_file_is_current keeps it honest).
"""
from __future__ import annotations

import ast
import contextlib
import functools
import io
import json
import os
import shutil
import sys
import tempfile
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import pytest

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
SUMMARY_FILE = ROOT / "web" / "data" / "errata" / "book_comments.json"

# The pinned submodule (c9b6e2e) has exactly these annotated prints.
EXPECTED_INVENTORY = {
    "ch01/01_char_tokenizer.py": 8, "ch01/02_byte_tokenizer.py": 7, "ch01/03_bpe_train.py": 3,
    "ch01/04_bpe_tokenizer.py": 2, "ch01/05_special_token.py": 1, "ch02/01_soft_dict.py": 1,
    "ch02/08_multi_head.py": 2, "ch03/03_alpaca.py": 3, "ch05/01_rope.py": 1,
    "ch05/04_improved_gpt.py": 1, "ch06/04_mixed_precision.py": 9,
}

# Known stale comments: (script, line) -> the exact documented difference.
STALE = {
    ("ch03/03_alpaca.py", 31): {"index": 30, "comment": 930, "actual": 932,
                                "comment_token": "alize", "actual_token": " you"},
}

_MISSING = object()


# ------------------------------------------------------------------ inventory

def annotated_prints(src: str) -> list[dict]:
    """Single-argument print(...) calls whose expected output is written in a comment."""
    comments = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            comments[tok.start[0]] = tok
    lines = src.splitlines()
    found = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "print"
                and len(node.value.args) == 1 and not node.value.keywords):
            continue
        end = node.end_lineno
        tok = comments.get(end)
        if tok is not None and lines[end - 1][:tok.start[1]].strip():          # same-line comment
            found.append({"line": node.lineno, "style": "trailing", "expected": tok.string[1:].strip()})
            continue
        block, j = [], end + 1                                                # comment block below
        while j <= len(lines) and lines[j - 1].lstrip().startswith("#"):
            text = lines[j - 1].lstrip()[1:]
            block.append(text[1:] if text.startswith(" ") else text)
            j += 1
        if block:
            found.append({"line": node.lineno, "style": "block", "expected": "\n".join(block)})
    return sorted(found, key=lambda f: f["line"])


def inventory() -> dict[str, list[dict]]:
    out = {}
    for path in sorted(DLFS6.rglob("*.py")):
        rel = path.relative_to(DLFS6).as_posix()
        if rel.startswith(".git"):
            continue
        found = annotated_prints(path.read_text(encoding="utf-8"))
        if found:
            out[rel] = found
    return out


# ------------------------------------------------------------------ rewrite + run

class _ToExpect(ast.NodeTransformer):
    def __init__(self, lines):
        self.lines = set(lines)

    def visit_Expr(self, node):
        if node.lineno in self.lines and isinstance(node.value, ast.Call):
            (arg,) = node.value.args
            args = [ast.Constant(node.lineno), arg]
            fields = [v.value for v in getattr(arg, "values", []) if isinstance(v, ast.FormattedValue)]
            if isinstance(arg, ast.JoinedStr) and len(fields) == 1:
                args.append(fields[0])          # an f-string's comment describes its one field
            node.value = ast.Call(ast.Name("__expect__", ast.Load()), args, [])
            return ast.fix_missing_locations(node)
        return node


def _copy_repo(dst: Path) -> None:
    big = shutil.ignore_patterns("*.pt", "*.bin", "*.txt", "__pycache__", ".git", "*.png")
    for sub in ("ch01", "ch02", "ch03", "ch05", "ch06", "codebot", "storybot"):
        shutil.copytree(DLFS6 / sub, dst / sub, ignore=big)


@functools.lru_cache(maxsize=None)
def run_script(rel: str) -> tuple[tuple, ...]:
    """Run one annotated script from a temp copy; return ((line, style, expected, verdict, shown), ...)."""
    import torch  # noqa: F401  (imported here so collection stays light)

    specs = {f["line"]: f for f in inventory()[rel]}
    src = (DLFS6 / rel).read_text(encoding="utf-8")
    tree = _ToExpect(specs).visit(ast.parse(src))
    seen = {}

    def __expect__(line, value, field=_MISSING):
        seen[line] = (value, field)

    saved_cwd, saved_path = os.getcwd(), list(sys.path)
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules)
                  if k.split(".")[0] in ("codebot", "storybot", "webbot")}
    with tempfile.TemporaryDirectory(prefix="dlfs6-comments-") as tmp:
        _copy_repo(Path(tmp))
        script = Path(tmp) / rel
        ns = {"__name__": "__main__", "__file__": str(script), "__expect__": __expect__}
        try:
            torch.manual_seed(0)
            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch("torch.cuda.is_available", return_value=False):
                exec(compile(tree, str(script), "exec"), ns)
        finally:
            os.chdir(saved_cwd)
            sys.path[:] = saved_path
            for k in [k for k in sys.modules if k.split(".")[0] in ("codebot", "storybot", "webbot")]:
                del sys.modules[k]
            sys.modules.update(saved_mods)
    rows = []
    for line, spec in sorted(specs.items()):
        assert line in seen, f"{rel}:{line} was never executed"
        value, field = seen[line]
        verdict, shown = classify(value, field, spec["expected"])
        rows.append((line, spec["style"], spec["expected"], verdict, shown))
    return tuple(rows)


def classify(value, field, expected: str) -> tuple[str, str]:
    import torch

    buf = io.StringIO()
    print(value, file=buf, end="")
    shown = buf.getvalue()
    if shown == expected:
        return "exact", shown
    if repr(value) == expected:
        return "repr", shown
    if isinstance(value, defaultdict) and repr(dict(value)) == expected:
        return "defaultdict", shown
    for v in (value, field):
        if isinstance(v, torch.Size) and str(tuple(v)) == expected:
            return "torch.Size", shown
    return "mismatch", shown


# ------------------------------------------------------------------ tests

def test_inventory_matches_pinned_submodule():
    inv = inventory()
    assert {k: len(v) for k, v in inv.items()} == EXPECTED_INVENTORY
    styles = Counter(f["style"] for v in inv.values() for f in v)
    assert styles == {"trailing": 35, "block": 3}


@pytest.mark.parametrize("rel", sorted(EXPECTED_INVENTORY))
def test_book_comments_hold(rel):
    rows = run_script(rel)
    for line, style, expected, verdict, shown in rows:
        stale = STALE.get((rel, line))
        if stale:
            want, got = ast.literal_eval(expected), ast.literal_eval(shown)
            diff = [(i, a, b) for i, (a, b) in enumerate(zip(want, got)) if a != b]
            assert len(want) == len(got) and diff == [(stale["index"], stale["comment"], stale["actual"])]
            continue
        assert verdict != "mismatch", f"{rel}:{line}: comment says {expected!r}, script prints {shown!r}"


def test_stale_alpaca_comment_is_a_merge_rules_artifact(dlfs6_path):
    from codebot.tokenizer import BPETokenizer

    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    for stale in STALE.values():
        assert tok.decode([stale["comment"]]) == stale["comment_token"]
        assert tok.decode([stale["actual"]]) == stale["actual_token"]


def summary() -> dict:
    per_script, totals = {}, Counter()
    for rel in sorted(EXPECTED_INVENTORY):
        verdicts = Counter()
        for line, style, expected, verdict, shown in run_script(rel):
            verdicts["stale" if (rel, line) in STALE else verdict] += 1
        per_script[rel] = dict(sorted(verdicts.items()))
        totals.update(verdicts)
    return {"scripts": len(per_script), "total": sum(totals.values()),
            "totals": dict(sorted(totals.items())), "per_script": per_script,
            "stale": [{"script": s, "line": ln, **v} for (s, ln), v in STALE.items()]}


def test_summary_counts():
    s = summary()
    assert s["total"] == 38
    assert s["totals"] == {"defaultdict": 1, "exact": 30, "repr": 3, "stale": 1, "torch.Size": 3}


def test_summary_file_is_current():
    assert json.loads(SUMMARY_FILE.read_text(encoding="utf-8")) == summary()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--json"]:
        out = Path(sys.argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {out}")
    else:
        for rel in sorted(EXPECTED_INVENTORY):
            for line, style, expected, verdict, shown in run_script(rel):
                print(f"{rel}:{line:<4} {style:8} {verdict:12} {shown[:60]!r}")
