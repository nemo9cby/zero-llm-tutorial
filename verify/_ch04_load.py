"""Helpers for chapter 4: load the book's ch04 scripts without running them.

Every ch04 script starts with ``os.chdir(...)`` and ends with top-level code that trains on the
2.2 GB corpus or writes ``storybot/merge_rules.pkl``. Importing such a script would run all of
that, so we parse it and execute only its ``import`` / ``def`` / ``class`` statements. The
function bodies are the book's code, line for line.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

DLFS6 = Path(__file__).resolve().parent.parent / "third_party" / "dlfs6"

KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)


def quiet_tqdm(iterable=None, *args, **kwargs):
    """Drop-in for tqdm that draws nothing (tests and benchmarks need clean output)."""
    return iterable


def load_defs(rel: str, quiet: bool = True) -> types.ModuleType:
    """Return a module holding only the definitions of third_party/dlfs6/<rel>.

    Imports inside the script still run (06/07 import from ``storybot.tokenizer``, so the
    caller must have third_party/dlfs6 on sys.path, e.g. via the ``dlfs6_path`` fixture).
    """
    path = DLFS6 / rel
    if not path.is_file():
        raise FileNotFoundError(f"{rel} not found under {DLFS6}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [node for node in tree.body if isinstance(node, KEEP)]
    if not any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in body):
        raise ValueError(f"{rel} defines no functions or classes")
    mod = types.ModuleType("dlfs6_" + Path(rel).stem.lstrip("0123456789_"))
    mod.__file__ = str(path)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), mod.__dict__)
    if quiet and "tqdm" in mod.__dict__:
        mod.tqdm = quiet_tqdm
    return mod


def function_ast(rel: str, name: str) -> ast.FunctionDef:
    """The AST of one top-level function (or method ``Class.method``) of a DLFS6 file."""
    tree = ast.parse((DLFS6 / rel).read_text(encoding="utf-8"))
    cls, _, meth = name.partition(".")
    for node in tree.body:
        if meth and isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == meth:
                    return sub
        if not meth and isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"{name} not found in {rel}")
