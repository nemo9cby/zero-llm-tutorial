"""Shared helpers for chapter 7 (WebBot tokenizer): paths, the official WebBot artifacts, OWT samples.

The OWT *text* is not in the repo (webbot/owt_valid.txt is 290 MB), but the official token file
webbot/owt_valid.bin is. BPE decoding is lossless, so decoding a prefix of that file gives back exactly the
text the official tokenizer saw (test_ch07.py::test_official_valid_bin_facts checks the byte count). All OWT
samples used in this chapter are made that way.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
WEBBOT = DLFS6 / "webbot"
OUT = ROOT / "web" / "data" / "ch07"
FIXTURES = ROOT / "tests" / "js" / "fixtures" / "ch07"
EOT = "<|endoftext|>"
# The GPT-2 pre-tokenization regex exactly as written in storybot/tokenizer.py.
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Facts about the full OWT files on Hugging Face (koki0702/zero-llm-data, webbot/), not downloaded here.
OWT_TRAIN_TXT_BYTES = 11_920_511_059
OWT_TRAIN_BIN_BYTES = 5_298_009_600
OWT_VALID_TXT_BYTES = 289_998_753

# The in-browser sample: official tokens up to and including the first <|endoftext|> after this many bytes.
HEAD_MIN_BYTES = 100_000


def book_modules():
    """Import the book's storybot.tokenizer (progress bars silenced)."""
    if str(DLFS6) not in sys.path:
        sys.path.insert(0, str(DLFS6))
    import storybot.tokenizer as sb
    sb.tqdm = lambda it, **kw: it
    return sb


class SerialPool:
    """Stands in for multiprocessing.Pool: same imap() order, no processes (the browser cannot spawn any)."""

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, func, items):
        return map(func, items)


def official_rules() -> dict:
    with open(WEBBOT / "merge_rules.pkl", "rb") as f:
        return pickle.load(f)


def official_tokenizer():
    return book_modules().BPETokenizer(official_rules())


def valid_bin() -> np.ndarray:
    return np.memmap(WEBBOT / "owt_valid.bin", dtype=np.uint16, mode="r")


def head_token_count(tok, ids, min_bytes: int) -> int:
    """Number of leading tokens up to and including the first EOT once min_bytes of text precede it."""
    end_id = tok.end_token_id
    total = 0
    for i, t in enumerate(ids):
        if t == end_id and total >= min_bytes:
            return i + 1
        total += len(tok.id_to_bytes[int(t)])
    raise ValueError("ran out of tokens")


def owt_text(tok, min_bytes: int) -> str:
    """The first documents of OWT valid, decoded from the official .bin, cut just before an <|endoftext|>."""
    arr = valid_bin()
    # 5 bytes per token is a safe upper bound for this corpus (it averages 4.5)
    ids = arr[: max(10_000, min_bytes // 2)].tolist()
    k = head_token_count(tok, ids, min_bytes)
    return tok.decode(ids[: k - 1])


def webbot_tiktoken(tok):
    """WebBot's merge table loaded into tiktoken (Rust): ranks = token ids, same regex, same special token."""
    import tiktoken
    ranks = {tok.id_to_bytes[i]: i for i in range(tok.end_token_id)}
    assert len(ranks) == tok.end_token_id, "WebBot byte strings must be unique to be tiktoken ranks"
    return tiktoken.Encoding("webbot-50k", pat_str=PATTERN, mergeable_ranks=ranks,
                             special_tokens={EOT: tok.end_token_id})
