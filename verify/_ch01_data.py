"""Build the chapter-1 data files under web/data/ch01/ from the book's own artifacts.

    uv run python verify/_ch01_data.py          # (re)write the files
    uv run python verify/_ch01_data.py --check  # exit 1 if a file is stale

Every file is derived deterministically from third_party/dlfs6 (and, for the reference numbers, from
tiktoken's published encodings); verify/test_ch01.py::test_page_data_is_derived_from_the_book_files
regenerates them in memory and compares byte for byte.

  tiny_codes_head10k.txt     codebot/tiny_codes.txt[:10000], the sample of ch01/08_eval.py
  tiny_codes_head.bin        the first 4,709 uint16 ids of codebot/tiny_codes.bin (= the encoded head)
  tinycodes_pretokens.json   every distinct pre-token of TinyCodes with its count, first-occurrence order
  token_counts.json          np.bincount(codebot/tiny_codes.bin, minlength=1000)
  compression.json           bytes/token after k = 0..743 official merges (full corpus and 10k head),
                             plus other tokenizers on the same two samples
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
OUT = ROOT / "web" / "data" / "ch01"
END = "<|endoftext|>"
HEAD_CHARS = 10_000

sys.path.insert(0, str(DLFS6))
sys.path.insert(0, str(ROOT / "verify"))
from _ch01_bpe import pretoken_table  # noqa: E402


def _merge(ids: tuple, a: int, b: int, new_id: int) -> tuple:
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def replay_merges(pieces: list[str], counts: list[int], merges: list[tuple[int, int]],
                  special_count: int) -> list[int]:
    """Total tokens after applying the first k merges (k = 0..len(merges)), in the book's order.

    Applying merges one by one to every pre-token is exactly codebot's BPETokenizer._encode_text, so
    tokens[k] is the length of encode(text) for a tokenizer that keeps only the first k merges."""
    words = [tuple(p.encode("utf-8")) for p in pieces]
    where: dict[tuple, set] = {}
    for w, ids in enumerate(words):
        for pair in zip(ids, ids[1:]):
            where.setdefault(pair, set()).add(w)
    total = sum(len(ids) * c for ids, c in zip(words, counts)) + special_count
    tokens = [total]
    for k, (a, b) in enumerate(merges):
        new_id = 256 + k
        for w in where.pop((a, b), ()):
            old = words[w]
            new = _merge(old, a, b, new_id)
            for pair in zip(new, new[1:]):
                where.setdefault(pair, set()).add(w)
            total -= (len(old) - len(new)) * counts[w]
            words[w] = new
        tokens.append(total)
    return tokens


def load_book() -> dict:
    """TinyCodes as ch01/07-09 read it (text mode, LF), its merge rules, its .bin, and its pre-token table."""
    from codebot.tokenizer import pretokenize
    text = (DLFS6 / "codebot" / "tiny_codes.txt").read_text(encoding="utf-8")
    rules = pickle.loads((DLFS6 / "codebot" / "merge_rules.pkl").read_bytes())
    ids = np.fromfile(DLFS6 / "codebot" / "tiny_codes.bin", dtype=np.uint16)
    sequence: list[int] = []
    pieces, counts, n_end = pretoken_table(text, pretokenize, sequence=sequence)
    return {"text": text, "rules": rules, "merges": list(rules), "bin": ids, "pieces": pieces, "counts": counts,
            "n_end": n_end, "sequence": sequence, "pretokenize": pretokenize}


def reference_counts(text: str) -> list[dict]:
    """Token counts of other tokenizers on `text` (tiktoken encodings + the book's StoryBot/WebBot)."""
    import tiktoken
    from storybot.tokenizer import BPETokenizer, pretokenize
    refs = []
    for name, enc_name in [("GPT-2", "gpt2"), ("cl100k_base", "cl100k_base"), ("o200k_base", "o200k_base")]:
        enc = tiktoken.get_encoding(enc_name)
        refs.append({"name": name, "kind": "tiktoken", "vocab": enc.n_vocab,
                     "tokens": len(enc.encode(text, allowed_special={END}))})
    for bot in ("storybot", "webbot"):
        # storybot's encode, with a cache per distinct pre-token (the ids do not depend on the context)
        tok = BPETokenizer.load_from(DLFS6 / bot / "merge_rules.pkl")
        cache: dict[str, int] = {}
        n = 0
        for i, part in enumerate(text.split(END)):
            n += 1 if i else 0
            for p in pretokenize(part):
                if p not in cache:
                    cache[p] = len(tok._encode_text(p))
                n += cache[p]
        refs.append({"name": bot, "kind": "dlfs6", "vocab": tok.vocab_size, "tokens": n})
    return refs


def build_files(book: dict | None = None) -> dict[str, bytes]:
    book = book or load_book()
    text, merges = book["text"], book["merges"]
    head = text[:HEAD_CHARS]
    head_pieces, head_counts, head_end = pretoken_table(head, book["pretokenize"])
    head_tokens = replay_merges(head_pieces, head_counts, merges, head_end)
    full_tokens = replay_merges(book["pieces"], book["counts"], merges, book["n_end"])
    files: dict[str, bytes] = {}
    files["tiny_codes_head10k.txt"] = head.encode("utf-8")
    files["tiny_codes_head.bin"] = book["bin"][: head_tokens[-1]].astype("<u2").tobytes()
    dump = lambda obj: (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    files["tinycodes_pretokens.json"] = dump({
        "format": "dlfs6-pretokens/1",
        "source": "third_party/dlfs6/codebot/tiny_codes.txt (split on <|endoftext|>, then codebot.tokenizer.pretokenize)",
        "bytes": len(text.encode("utf-8")), "documents": book["n_end"] + 1, "end_tokens": book["n_end"],
        "pretokens": sum(book["counts"]), "pieces": [[p, c] for p, c in zip(book["pieces"], book["counts"])],
    })
    files["token_counts.json"] = dump({
        "source": "np.bincount(np.fromfile('third_party/dlfs6/codebot/tiny_codes.bin', dtype=np.uint16), minlength=1000)",
        "total": int(book["bin"].size), "counts": np.bincount(book["bin"], minlength=1000).tolist(),
    })
    files["compression.json"] = dump({
        "source": "tokens after the first k merges of codebot/merge_rules.pkl, k = 0..743 (end tokens count 1 each)",
        "full": {"label": "全语料", "bytes": len(text.encode("utf-8")), "tokens": full_tokens,
                 "refs": reference_counts(text)},
        "head": {"label": "前 10,000 个字符", "bytes": len(head.encode("utf-8")), "tokens": head_tokens,
                 "refs": reference_counts(head)},
    })
    return files


def main() -> None:
    check = "--check" in sys.argv
    files = build_files()
    stale = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = OUT / name
        if path.is_file() and path.read_bytes() == data:
            continue
        stale.append(name)
        if not check:
            path.write_bytes(data)
    for name in files:
        print(f"{'stale' if name in stale else 'ok   '} web/data/ch01/{name} ({len(files[name]):,} B)")
    if check and stale:
        sys.exit(1)


if __name__ == "__main__":
    main()
