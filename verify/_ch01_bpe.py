"""Chapter 1 helpers: the book's BPE training loop on de-duplicated pre-tokens, with a pluggable tie-break.

The book's trainers (ch01/03-06, codebot/tokenizer.py) recount every pair of the whole corpus at every
step. That is exact but takes an hour on TinyCodes. The functions here do the same arithmetic on
{pre-token: count} (the idea of ch04) and update only the pre-tokens that contain the merged pair, so a
full TinyCodes run takes seconds. verify/test_ch01.py checks them against the book's own trainers before
using them for any claim:

  tie_break="ids"    codebot/storybot:     max(counts, key=lambda p: (counts[p], p[0], p[1]))
  tie_break="first"  ch01/03-06:           max(counts, key=counts.get)  -> the first pair in scan order
  tie_break="bytes"  CS336 A1 (not in the book): the lexicographically greater pair of *bytes*

`first` needs the scan order of the book's full recount. Distinct pre-tokens are kept in first-occurrence
order, so the pair the book's scan meets first is the one with the smallest (pre-token index, position).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

END = "<|endoftext|>"


def pretoken_table(text: str, pretokenize, end_token: str = END, sequence: list | None = None
                   ) -> tuple[list[str], list[int], int]:
    """Distinct pre-tokens in first-occurrence order, their counts, and the number of end tokens.

    If `sequence` is a list, the text is also recorded there in order: the index of each pre-token,
    and -1 for each end token (enough to rebuild encode(text) from the per-piece encodings)."""
    index: dict[str, int] = {}
    counts: list[int] = []
    docs = text.split(end_token)
    for d, doc in enumerate(docs):
        if d and sequence is not None:
            sequence.append(-1)
        for piece in pretokenize(doc):
            k = index.get(piece)
            if k is None:
                k = index[piece] = len(counts)
                counts.append(1)
            else:
                counts[k] += 1
            if sequence is not None:
                sequence.append(k)
    return list(index), counts, len(docs) - 1


def _merge(ids: tuple, pair: tuple, new_id: int) -> tuple:
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


@dataclass
class TrainResult:
    merges: list[tuple[int, int]]
    tokens: list[int]            # total tokens of the training text after 0, 1, ..., len(merges) merges
    counts: list[int]            # frequency of each chosen pair
    tied: list[int]              # how many pairs shared that maximum frequency
    vocab: dict[int, bytes] = field(default_factory=dict)


def train_dedup(pieces: list[str], counts: list[int], num_merges: int, tie_break: str = "ids",
                special_count: int = 0) -> TrainResult:
    """Train byte-level BPE on (pre-token, count) pairs. special_count end tokens add 1 token each."""
    if tie_break not in ("ids", "first", "bytes"):
        raise ValueError(f"unknown tie_break {tie_break!r}")
    words = [tuple(p.encode("utf-8")) for p in pieces]
    vocab = {i: bytes([i]) for i in range(256)}
    pair_count: dict[tuple, int] = defaultdict(int)
    where: dict[tuple, set] = defaultdict(set)
    for w, ids in enumerate(words):
        for pair in zip(ids, ids[1:]):
            pair_count[pair] += counts[w]
            where[pair].add(w)
    total = sum(len(ids) * c for ids, c in zip(words, counts)) + special_count
    res = TrainResult(merges=[], tokens=[total], counts=[], tied=[], vocab=vocab)

    def first_pos(pair):
        w = min(where[pair])
        ids = words[w]
        for i in range(len(ids) - 1):
            if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                return (w, i)
        raise AssertionError("pair index out of sync")

    for step in range(num_merges):
        if not pair_count:
            break
        top = max(pair_count.values())
        tied = [p for p, c in pair_count.items() if c == top]
        if tie_break == "ids":
            best = max(tied)
        elif tie_break == "bytes":
            best = max(tied, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        else:
            best = min(tied, key=first_pos)
        new_id = 256 + step
        for w in list(where[best]):
            old, c = words[w], counts[w]
            new = _merge(old, best, new_id)
            for pair in zip(old, old[1:]):
                pair_count[pair] -= c
                if pair_count[pair] == 0:
                    del pair_count[pair]
                where[pair].discard(w)
            for pair in zip(new, new[1:]):
                pair_count[pair] += c
                where[pair].add(w)
            total -= (len(old) - len(new)) * c
            words[w] = new
        where.pop(best, None)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        res.merges.append(best)
        res.tokens.append(total)
        res.counts.append(top)
        res.tied.append(len(tied))
    return res
