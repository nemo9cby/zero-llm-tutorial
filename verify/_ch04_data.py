"""Chapter 4 data builder: everything under web/data/ch04/ is produced (or checked) here.

    uv run python verify/_ch04_data.py            # regenerate the derived files
    uv run pytest verify/test_ch04.py             # checks they are still in sync with the sources

Files and where they come from:
  bpe_book.py            the book's BPE functions (codebot/tokenizer.py, ch04/01-03, storybot/tokenizer.py),
                         copied by AST transformation: tqdm calls removed, versions renamed (see RENAMES)
  storybot_merges.json   storybot/merge_rules.pkl as [[id1, id2], ...] in rank order (new id = 256 + index)
  codebot_merges.json    codebot/merge_rules.pkl, same format
  valid_head.txt         the first stories of storybot/tiny_stories_valid.txt, cut just before an <|endoftext|>
  valid_head.bin         the matching prefix of the official storybot/tiny_stories_valid.bin (uint16)
  valid_eot.json         byte offsets of every <|endoftext|> in tiny_stories_valid.txt (delta-encoded)
  codes_head10k.txt      the first 10,000 characters of codebot/tiny_codes.txt
Files that cannot be rebuilt from the repo (they come from the 2.2 GB train file, fetched once with an
HTTP Range request; provenance is in PROVENANCE below and in the page text):
  train_head10k.txt      the first 10,000 characters of storybot/tiny_stories_train.txt (the book's 4.5 sample)
  train_head.bin         the first 3,000 tokens of the official storybot/tiny_stories_train.bin
"""
from __future__ import annotations

import ast
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
OUT = ROOT / "web" / "data" / "ch04"
EOT = "<|endoftext|>"
VALID_HEAD_STORIES = 60          # stories in valid_head.txt (about 50 KB)

PROVENANCE = {
    "train_url": "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.txt",
    "train_bin_url": "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.bin",
}

# (source file, function or class) -> (name in bpe_book.py, renames applied to the body)
V1 = {"pretokenize": "pretokenize_v1", "count_pairs": "count_pairs_v1", "BPETokenizer": "BPETokenizerV1"}
RENAMES = [
    ("codebot/tokenizer.py", "pretokenize", "pretokenize_v1", {}),
    ("codebot/tokenizer.py", "count_pairs", "count_pairs_v1", {}),
    ("codebot/tokenizer.py", "train_bpe", "train_bpe_naive", V1),
    ("codebot/tokenizer.py", "BPETokenizer", "BPETokenizerV1", V1),
    ("storybot/tokenizer.py", "pretokenize", "pretokenize", {}),
    ("storybot/tokenizer.py", "count_pairs", "count_pairs", {}),
    ("storybot/tokenizer.py", "merge", "merge", {}),
    ("ch04/01_bpe_optimize.py", "train_bpe", "train_bpe_dedup", {}),
    ("ch04/02_bpe_cache.py", "train_bpe", "train_bpe_cache", {}),
    ("storybot/tokenizer.py", "find_chunk_boundaries", "find_chunk_boundaries", {}),
    ("ch04/03_bpe_chunk.py", "train_bpe", "train_bpe_chunk", {}),
    ("storybot/tokenizer.py", "pretoken_chunk", "pretoken_chunk", {}),
    ("storybot/tokenizer.py", "BPETokenizer", "BPETokenizer", {}),
]

HEADER = '''"""原书的 BPE 代码（浏览器里的验证单元直接 import 这个文件）。

逐行取自 third_party/dlfs6，只有两处机械改动：
  1. 去掉 tqdm 进度条（浏览器里没有终端）；
  2. 不同版本的同名函数加了后缀：
       pretokenize_v1 / count_pairs_v1 / train_bpe_naive / BPETokenizerV1  <- codebot/tokenizer.py（第 1 章）
       train_bpe_dedup <- ch04/01_bpe_optimize.py    train_bpe_cache <- ch04/02_bpe_cache.py
       train_bpe_chunk <- ch04/03_bpe_chunk.py       其余          <- storybot/tokenizer.py
由 verify/_ch04_data.py 生成；verify/test_ch04.py::test_bpe_book_is_verbatim 检查它与原书逐个语法节点相同。
"""
import os
import pickle
import shutil
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import regex as re
'''


class _Transform(ast.NodeTransformer):
    """Remove tqdm(...) wrappers and rename references to other versioned functions."""

    def __init__(self, renames: dict[str, str]):
        self.renames = renames

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "tqdm":
            return node.args[0]          # tqdm(x, desc=...) -> x
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self.renames:
            return ast.copy_location(ast.Name(id=self.renames[node.id], ctx=node.ctx), node)
        return node


def book_node(rel: str, name: str) -> ast.AST:
    tree = ast.parse((DLFS6 / rel).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise KeyError(f"{name} not in {rel}")


def transformed(rel: str, name: str, new_name: str, renames: dict) -> ast.AST:
    node = _Transform(renames).visit(book_node(rel, name))
    node.name = new_name
    return ast.fix_missing_locations(node)


def bpe_book_source() -> str:
    parts = [HEADER]
    for rel, name, new_name, renames in RENAMES:
        parts.append(f"\n\n# --- {rel} :: {name}" + (f"  (改名为 {new_name})" if new_name != name else "") + "\n")
        parts.append(ast.unparse(transformed(rel, name, new_name, renames)) + "\n")
    return "".join(parts)


def merges_list(pkl: Path) -> list[list[int]]:
    rules = pickle.loads(pkl.read_bytes())
    out = sorted(rules.items(), key=lambda kv: kv[1])
    assert [v for _, v in out] == list(range(256, 256 + len(out))), "ranks must be 256, 257, ..."
    return [[a, b] for (a, b), _ in out]


def valid_head() -> tuple[str, list[int]]:
    """First VALID_HEAD_STORIES stories of valid (text ends right before an EOT) + official ids."""
    import numpy as np

    text = (DLFS6 / "storybot" / "tiny_stories_valid.txt").read_bytes().decode("utf-8")
    pos = -1
    for _ in range(VALID_HEAD_STORIES):
        pos = text.index(EOT, pos + 1)
    head = text[:pos]
    sys.path.insert(0, str(DLFS6))
    try:
        from storybot.tokenizer import BPETokenizer
    finally:
        sys.path.remove(str(DLFS6))
    tok = BPETokenizer.load_from(DLFS6 / "storybot" / "merge_rules.pkl")
    ids = tok.encode(head)
    official = np.fromfile(DLFS6 / "storybot" / "tiny_stories_valid.bin", dtype=np.uint16)
    assert official[: len(ids)].tolist() == ids, "our encoding of the head differs from valid.bin"
    assert int(official[len(ids)]) == tok.end_token_id, "valid.bin should continue with <|endoftext|>"
    return head, ids


def eot_offsets() -> dict:
    data = (DLFS6 / "storybot" / "tiny_stories_valid.txt").read_bytes()
    marker = EOT.encode()
    offs, i = [], data.find(marker)
    while i != -1:
        offs.append(i)
        i = data.find(marker, i + 1)
    deltas = [offs[0]] + [b - a for a, b in zip(offs, offs[1:])]
    return {"file": "storybot/tiny_stories_valid.txt", "size": len(data), "count": len(offs),
            "marker_len": len(marker), "deltas": deltas}


def derived_files() -> dict[str, bytes]:
    """Every file that can be rebuilt from third_party/dlfs6, as bytes."""
    import numpy as np

    head, ids = valid_head()
    codes = (DLFS6 / "codebot" / "tiny_codes.txt").read_bytes().decode("utf-8")
    js = lambda obj: (json.dumps(obj, separators=(",", ":")) + "\n").encode()
    return {
        "bpe_book.py": bpe_book_source().encode("utf-8"),
        "storybot_merges.json": js(merges_list(DLFS6 / "storybot" / "merge_rules.pkl")),
        "codebot_merges.json": js(merges_list(DLFS6 / "codebot" / "merge_rules.pkl")),
        "valid_head.txt": head.encode("utf-8"),
        "valid_head.bin": np.array(ids, dtype=np.uint16).tobytes(),
        "valid_eot.json": js(eot_offsets()),
        "codes_head10k.txt": codes[:10000].encode("utf-8"),
    }


JS_FIXTURES = ROOT / "tests" / "js" / "fixtures" / "ch04"
TRICKY = ["Hello world! It's 2024.   Go", "don’t", "you'LL", "x = 1\r\ny = 2\r\n    z", "naïve 世界 2024",
          "“Hi!” she said.", "a b　c", "﻿BOM", "😁😁 ok", "tab\tsep\n\n\nend  "]


def js_fixtures() -> dict[str, bytes]:
    """Expected values for tests/js/ch04-bpe.test.mjs, computed with the book's own Python code."""
    sys.path.insert(0, str(DLFS6))
    try:
        from storybot.tokenizer import find_chunk_boundaries, pretokenize
    finally:
        sys.path.remove(str(DLFS6))
    import importlib.util

    spec = importlib.util.spec_from_file_location("bpe_book_fixture", OUT / "bpe_book.py")
    book = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # no __pycache__ inside web/data/ch04
    try:
        spec.loader.exec_module(book)
    finally:
        sys.dont_write_bytecode = saved
    sample = (OUT / "train_head10k.txt").read_text(encoding="utf-8")
    toy = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    valid = str(DLFS6 / "storybot" / "tiny_stories_valid.txt")
    head = str(OUT / "valid_head.txt")
    js = lambda obj: (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    return {
        "pretokenize.json": js({"tricky": [[t, list(pretokenize(t))] for t in TRICKY],
                                "sample": [p for piece in sample.split(EOT) for p in pretokenize(piece)]}),
        "merges.json": js({"toy": [list(p) for p in book.train_bpe_cache(toy, 256 + 10 + 1)],
                           "sample400": [list(p) for p in book.train_bpe_cache(sample, 400)]}),
        "boundaries.json": js({"valid": {str(k): find_chunk_boundaries(valid, k) for k in (1, 4, 64, 1000)},
                               "valid_count_100000": len(find_chunk_boundaries(valid, 100_000)),
                               "valid_count_1000000": len(find_chunk_boundaries(valid, 1_000_000)),
                               "head": {str(k): find_chunk_boundaries(head, k)
                                        for k in (2, 7, 30, 59, 60, 61, 100, 120, 5000)}}),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, blob in derived_files().items():
        (OUT / name).write_bytes(blob)
        print(f"wrote web/data/ch04/{name} ({len(blob):,} bytes)")
    JS_FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, blob in js_fixtures().items():
        (JS_FIXTURES / name).write_bytes(blob)
        print(f"wrote tests/js/fixtures/ch04/{name} ({len(blob):,} bytes)")


if __name__ == "__main__":
    main()
