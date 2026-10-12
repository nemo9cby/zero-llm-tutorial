"""Shared helpers for chapter 7 (WebBot tokenizer): paths, the official WebBot artifacts, OWT samples.

The OWT *text* is not in the repo (webbot/owt_valid.txt is 290 MB), but the official token file
webbot/owt_valid.bin is. BPE decoding is lossless, so decoding that file gives back exactly the text the official
tokenizer saw: test_ch07.py::test_official_valid_bin_facts checks that the decoded bytes have the SHA-256 of
CS336's owt_valid.txt (research/cs336-crossref.md, V8). All OWT samples used in this chapter are made that way.

The author's artifacts are not in the submodule; webbot/download.sh lists them (commented out):
    webbot/merge_rules.pkl (533 KB), webbot/owt_valid.bin (129 MB)
test_ch07.py skips the checks that need them, with the download command (download_hint), when they are missing.
"""
from __future__ import annotations

import heapq
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
WEBBOT = DLFS6 / "webbot"
OUT = ROOT / "web" / "data" / "ch07"
FIXTURES = ROOT / "tests" / "js" / "fixtures" / "ch07"
CH04_DATA = ROOT / "web" / "data" / "ch04"
EOT = "<|endoftext|>"
DOWNLOADS = {"merge_rules.pkl": "533 KB", "owt_valid.bin": "129 MB"}
HF_WEBBOT = "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot"
# The GPT-2 pre-tokenization regex exactly as written in storybot/tokenizer.py.
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Facts about the full OWT files on Hugging Face (koki0702/zero-llm-data, webbot/), not downloaded here.
OWT_TRAIN_TXT_BYTES = 11_920_511_059
OWT_TRAIN_BIN_BYTES = 5_298_009_600
OWT_VALID_TXT_BYTES = 289_998_753
# SHA-256 of gunzip(stanford-cs336/owt-sample owt_valid.txt.gz) = DLFS6 webbot/owt_valid.txt (cs336-crossref.md V8)
OWT_VALID_TXT_SHA256 = "2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660"

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


def missing_downloads() -> list[str]:
    """The author's WebBot files this chapter needs but that are not on disk (download.sh has them commented out)."""
    return [name for name in DOWNLOADS if not (WEBBOT / name).is_file()]


def download_hint(names) -> str:
    """The commands that fetch the given webbot/ files (the same as the box at the end of section 7.1)."""
    return "cd third_party/dlfs6 && " + " && ".join(f"curl -L -o webbot/{n} {HF_WEBBOT}/{n}" for n in names)


def _webbot_file(name: str) -> Path:
    path = WEBBOT / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; download it first:  {download_hint([name])}")
    return path


def official_rules() -> dict:
    with open(_webbot_file("merge_rules.pkl"), "rb") as f:
        return pickle.load(f)


def official_tokenizer():
    return book_modules().BPETokenizer(official_rules())


def valid_bin() -> np.ndarray:
    return np.memmap(_webbot_file("owt_valid.bin"), dtype=np.uint16, mode="r")


def iter_valid_bytes(tok, step: int = 2_000_000):
    """The whole official owt_valid.bin decoded, as consecutive byte chunks (290 MB in total, ~20 s)."""
    arr = valid_bin()
    table = [tok.id_to_bytes[i] for i in range(tok.vocab_size)]
    for s in range(0, len(arr), step):
        yield b"".join([table[i] for i in arr[s:s + step].tolist()])


def iter_valid_docs(tok):
    """Every document of the official owt_valid.bin as (token ids, decoded bytes), in file order.

    The file is <doc> EOT <doc> EOT ... and ends with an EOT, so splitting at the EOT id gives all 59,059 documents.
    The ids are a uint16 view into the memmap (no copy)."""
    arr = valid_bin()
    table = [tok.id_to_bytes[i] for i in range(tok.vocab_size)]
    start = 0
    for end in np.flatnonzero(arr == tok.end_token_id).tolist():
        ids = arr[start:end]
        yield ids, b"".join([table[i] for i in ids.tolist()])
        start = end + 1


def ch04_ext():
    """Chapter 4's tutorial extension web/data/ch04/bpe_ext.py (not the book's code): the lazy-deletion heap.
    verify/test_ch04.py proves that it learns exactly the book's merges."""
    sys.dont_write_bytecode = True                       # no __pycache__ inside web/data/
    if str(CH04_DATA) not in sys.path:
        sys.path.insert(0, str(CH04_DATA))
    import bpe_ext
    return bpe_ext


_BITS = 17                                               # token ids < 2**17 (WebBot's are < 50,000)
_MASK = (1 << _BITS) - 1


def _heap_key(pair: tuple, count: int) -> int:
    """One int that orders like the book's max() key (count, id0, id1); negated for heapq's min-heap.
    (A tuple per entry would take ~150 bytes; on the 290 MB split the heap holds tens of millions of entries.)"""
    return -((count << 2 * _BITS) | (pair[0] << _BITS) | pair[1])


def heap_merge_sizes(ids_counts: dict, num_merges: int, compact: float = 3.0) -> tuple[list, list]:
    """bpe_ext.heap_merge_loop (chapter 4's lazy-deletion heap), recording len(pair_counts) before every merge.

    The heap keeps the same pair-count table as the book's train_bpe (bpe_ext.apply_merge is the book's update,
    line for line), so sizes[t] is the number of pairs the book's max() scans at step t: one key call each.
    When stale entries outnumber live ones `compact` to 1, the heap is rebuilt from the table (same order, less
    memory). test_ch07.py::test_heap_loop_scans_what_the_book_scans checks merges and sizes against the book.
    ids_counts ({pretoken bytes as a tuple of ints: count}) is consumed. Returns (merges, sizes).
    """
    bx = ch04_ext()
    pair_counts, pair_to_ids = bx.init_cache(ids_counts)
    heap = [_heap_key(p, c) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges, sizes = [], []
    for step in range(num_merges):
        while heap:
            k = -heap[0]
            best = ((k >> _BITS) & _MASK, k & _MASK)
            if pair_counts.get(best) == k >> 2 * _BITS:
                break
            heapq.heappop(heap)                          # stale entry: that pair's count has changed since
        else:
            break                                        # no pairs left (the book's `if not pair_counts: break`)
        sizes.append(len(pair_counts))
        merges.append(best)
        changed = set()
        bx.apply_merge(ids_counts, pair_counts, pair_to_ids, best, 256 + step, changed)
        for pair in changed:
            if pair in pair_counts:
                heapq.heappush(heap, _heap_key(pair, pair_counts[pair]))
        if len(heap) > compact * len(pair_counts):
            heap = [_heap_key(p, c) for p, c in pair_counts.items()]
            heapq.heapify(heap)
    return merges, sizes


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
