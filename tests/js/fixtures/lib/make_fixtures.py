r"""Reference outputs for tests/js/lib-bpe.test.mjs, computed with the real DLFS6 Python tokenizers.

    uv run python tests/js/fixtures/lib/make_fixtures.py [--cs336 PATH/TO/assignment1-basics]

Writes, next to this file:
  unicode.json \p{L}, \p{N} and \s of Python's `regex`, as code-point ranges (the pre-tokenizer's classes)
  encode.json  mixed corpus (code, TinyStories, OWT, CJK, emoji, combining marks, special tokens, whitespace
               runs, ...) -> pretokenize() pieces, the regex branch behind each piece, and encode() ids for
               codebot / storybot / webbot
  decode.json  random id sequences -> decode() (errors="replace"), raw byte strings -> bytes.decode("utf-8",
               "replace"), incremental decoding, and a digest of every token's bytes (id_to_bytes)
  train.json   small corpora x {first, ids, bytes} tie-breaks -> merges + per-step [count, #tied, top-5]
  cs336.json   CS336 A1 corpus.en + reference merges (only rewritten when --cs336 is given)

Reference functions: ch01/03, ch01/05, ch01/06 train_bpe ('first' = dict-order tie-break), codebot.tokenizer
and storybot.tokenizer (train_bpe with the (count, id0, id1) key, BPETokenizer.encode/decode). The 'bytes'
rule (CS336: lexicographically greater pair of bytes) has no book implementation; it is the book's loop with
only the key changed, and the loop itself is first checked against every book trainer above.
"""
from __future__ import annotations

import argparse
import codecs
import contextlib
import hashlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DLFS6 = ROOT / "third_party" / "dlfs6"
sys.path.insert(0, str(DLFS6))

import numpy as np  # noqa: E402

from codebot import tokenizer as cb  # noqa: E402
from storybot import tokenizer as sb  # noqa: E402

END = "<|endoftext|>"


def load_script(rel: str):
    """Import a ch01 script as a module (its demo code runs once, silently; none of them write files)."""
    spec = importlib.util.spec_from_file_location("dlfs6_" + Path(rel).stem, DLFS6 / rel)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def dump(name: str, obj) -> None:
    path = HERE / name
    path.write_text(json.dumps(obj, ensure_ascii=True, separators=(",", ":")), encoding="utf-8", newline="\n")
    print(f"{path.relative_to(ROOT)}: {path.stat().st_size:,} bytes")


# --------------------------------------------------------------------------------------------- corpus
def read_text(rel: str, n_chars: int) -> str:
    with open(DLFS6 / rel, encoding="utf-8", newline="") as f:  # newline="": keep the file's own line endings
        return f.read(n_chars)


def owt_text(web_tok, min_bytes: int) -> str:
    """OpenWebText as the book encoded it: decode the head of webbot/owt_valid.bin, cut at a line break."""
    ids = np.fromfile(DLFS6 / "webbot" / "owt_valid.bin", dtype=np.uint16, count=6000).tolist()
    text = web_tok.decode(ids)
    cut = text.index("\n", min_bytes)
    assert "\ufffd" not in text[:cut]
    return text[: cut + 1]


def random_unicode(rng: random.Random, n: int) -> list[str]:
    """Strings mixing ASCII with characters assigned in Python's unicodedata (Unicode 15.0 on 3.12).

    Only characters known to that older table are used, so the JS engine's Unicode tables (16.0 in Node 24)
    know them too; see the "Known limit" note in _bpe.js."""
    pools = {
        "ascii": [chr(c) for c in range(0x20, 0x7F)] + ["\n", "\t", "  "],
        "latin": [chr(c) for c in range(0xA0, 0x250)],
        "space": list("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2007\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
                      "\x1c\x1d\x1e\x1f\u180e\u200b"),
        "marks": [chr(c) for c in range(0x300, 0x370)] + ["\u20dd", "\ufe0f", "\u200d"],
        "punct": [chr(c) for c in range(0x2010, 0x2060)] + list("「」『』（），。！？、"),
        "cjk": [chr(c) for c in range(0x4E00, 0x9FFF, 37)] + [chr(c) for c in range(0x3040, 0x30FF)]
               + [chr(c) for c in range(0xAC00, 0xD7A3, 97)],
        "digits": list("0123456789٠١٢٣٤٥٦٧٨٩०१२३४५६७८९⁰¹²³½¼ⅫⅣ①②０１２"),
        "emoji": [chr(c) for c in range(0x1F300, 0x1F650, 3)] + ["\U0001F3FB", "\U0001F1EF", "\U0001F1F5"],
        "scripts": [chr(c) for c in (*range(0x391, 0x3CA), *range(0x410, 0x450), *range(0x5D0, 0x5EB),
                                     *range(0x620, 0x64B), *range(0x905, 0x940), *range(0xE01, 0xE30))],
        "astral": [chr(c) for c in (0x10400, 0x1D400, 0x1D7CE, 0x20000, 0x2A6D6, 0x1F9D1, 0x10FFFD, 0xF0000)],
    }
    names, weights = zip(*[("ascii", 8), ("latin", 2), ("space", 3), ("marks", 1), ("punct", 2), ("cjk", 2),
                           ("digits", 2), ("emoji", 2), ("scripts", 2), ("astral", 1)])
    out = []
    for _ in range(n):
        s = "".join(rng.choice(pools[rng.choices(names, weights)[0]]) for _ in range(rng.randint(1, 40)))
        assert all(unicodedata.category(ch) not in ("Cn", "Cs") for ch in s)
        out.append(s)
    return out


CASES = [
    # ch01 examples
    ("book-hello", "hello世界😁"), ("book-train", "Hello world! This is BPE training."),
    ("book-say", "Say hello! Why hello? Just hello.<|endoftext|>Good morning!"),
    # research/repo-ch01-tokenizer.md J1 (29 tricky strings)
    ("j1-01", "hello世界😁"), ("j1-02", "e\u0301 café"), ("j1-03", "👨\u200d👩\u200d👧 family"), ("j1-04", "🇯🇵🇨🇳"),
    ("j1-05", "مرحبا بالعالم"), ("j1-06", "naïve\tTab\r\nCRLF"), ("j1-07", "\ufeffBOM"), ("j1-08", "\x00NUL"),
    ("j1-09", "日本語のテキスト。句読点、"), ("j1-10", "  \n\n  x"), ("j1-11", "I'm sure they'll say don't DON'T"),
    ("j1-12", "print(f'done')"), ("j1-13", "x  =   1 + 23456"), ("j1-14", "a\x1cb\x1dc\x1ed\x1fe"),
    ("j1-15", "a\x85b"), ("j1-16", "a\u180eb"), ("j1-17", "a\u200bb"), ("j1-18", "a\u2028b\u2029c"),
    ("j1-19", "a\u3000b\xa0c"), ("j1-20", "a\x0bb\x0cc"), ("j1-21", "😀🏻"), ("j1-22", "x²³ ½ Ⅻ ①"),
    ("j1-23", "Straße ǅ ﬁ"), ("j1-24", "İstanbul"), ("j1-25", "a \x85b"), ("j1-26", "a \ufeffb"),
    ("j1-27", "a \x1cb"), ("j1-28", "a\ufeff\ufeff b"), ("j1-29", "x \u180ey"),
    # special tokens
    ("special-only", END), ("special-twice", END + END), ("special-in-code", "print('<|endoftext|>')"),
    ("special-edges", END + "start and end" + END), ("special-spaces", "a " + END + " b\n" + END + "\n"),
    ("special-partial", "<|endoftext| <|endoftext |endoftext|> <|end" + "oftext|>|>"),
    # whitespace runs
    ("ws-mixed", "a  b   c\t\t d\n\n\ne \r\n f \xa0\xa0g\u3000\u3000h\u2028i \x85 j"),
    ("ws-trailing", "x   "), ("ws-leading", "   x"), ("ws-only", " \t\n \r\n"), ("ws-indent", "if x:\n        y\n\n\n    z"),
    ("ws-nel-bom", "\x85\x85 a\ufeff\ufeff\ufeffb \ufeff c"),
    # scripts, emoji, marks, numbers
    ("cjk", "日本語のテキスト。中文分词器，한국어 토크나이저「引用」"),
    ("emoji", "👍🏽 👨\u200d👩\u200d👧\u200d👦 🏳\ufe0f\u200d🌈 🇺🇸 ❤\ufe0f 😁😁 1\ufe0f\u20e3"),
    ("combining", "e\u0301 a\u0308 n\u0303 Z\u0351\u036b\u0343\u036a \u0915\u093f"),
    ("rtl", "שָׁלוֹם עולם مَرْحَبًا"), ("numbers", "3.14159 1,234,567 ٣٤٥ १२३ Ⅻ ½ ² ① ０１２"),
    ("contractions", "I'm you're we've they'd she'll it's DON'T O'Neil 'quoted' '' 's"),
    ("python", "def f(x):\n    return x**2  # square\n\tif x == 'done': pass\nprint(f'{x!r:>10}')\n"),
    ("mojibake", "Ã©tÃ© ÃÂÃÂÃÂ â€™ â€œ ï»¿"), ("controls", "\x00\x01\x07\x1b[0m\x7f\x80\x9f"),
    ("astral-letters", "𝐀𝐁𝐂 𐐀𐐁 𠀀𠀁 \U0010fffd\U000f0000"), ("long-word", "a" * 300), ("long-digits", "1" * 100),
    ("long-spaces", " " * 50 + "x" + "\n" * 30), ("empty", ""),
]


def build_corpus(web_tok) -> list[tuple[str, str]]:
    rng = random.Random(6)
    cases = list(CASES)
    cases.append(("tinycodes", read_text("codebot/tiny_codes.txt", 4000)))
    cases.append(("tinystories", read_text("storybot/tiny_stories_valid.txt", 4000)))
    cases.append(("owt-10k", owt_text(web_tok, 10_000)))
    cases += [(f"random-{i:02d}", s) for i, s in enumerate(random_unicode(rng, 60))]
    return cases


# --------------------------------------------------------------------------------------------- classes
def make_unicode() -> None:
    """The three character classes of the GPT-2 pattern, as Python's `regex` sees every code point."""
    import regex

    def ranges(pattern: str) -> list[list[int]]:
        rx, out, start = regex.compile(pattern), [], None
        for c in range(0x110001):
            ok = c <= 0x10FFFF and not 0xD800 <= c <= 0xDFFF and rx.match(chr(c)) is not None
            if ok and start is None:
                start = c
            elif not ok and start is not None:
                out.append([start, c - 1])
                start = None
        return out

    dump("unicode.json", {"regex": regex.__version__, "python": sys.version.split()[0],
                          "L": ranges(r"\p{L}"), "N": ranges(r"\p{N}"), "S": ranges(r"\s")})


# --------------------------------------------------------------------------------------------- encode
BRANCHES = [r"'(?:[sdmt]|ll|ve|re)", r" ?\p{L}+", r" ?\p{N}+", r" ?[^\s\p{L}\p{N}]+", r"\s+(?!\S)", r"\s+"]


def branch_of_each_piece(text: str) -> list[int]:
    """Which alternative of the GPT-2 pattern matched each piece (each branch wrapped in its own group)."""
    import regex

    grouped = regex.compile("|".join(f"({b})" for b in BRANCHES))
    matches = list(grouped.finditer(text))
    assert [m.group(0) for m in matches] == cb.pretokenize(text)  # wrapping in groups changes nothing
    return [m.lastindex - 1 for m in matches]


def make_encode(toks: dict) -> list[tuple[str, str]]:
    source = (DLFS6 / "codebot" / "tokenizer.py").read_text(encoding="utf-8")
    assert 'r"""' + "|".join(BRANCHES) + '"""' in source
    cases = build_corpus(toks["webbot"])
    out = {"cases": [{"name": n, "text": t} for n, t in cases],
           "pretokens": [cb.pretokenize(t) for _, t in cases],
           "branches": [branch_of_each_piece(t) for _, t in cases],
           "ids": {}}
    for bot, tok in toks.items():
        ids = [tok.encode(t) for _, t in cases]
        for (name, text), seq in zip(cases, ids):
            assert tok.decode(seq) == text, (bot, name)  # every case round-trips in Python
        out["ids"][bot] = ids
    # The two book encoders (apply every merge in training order vs. lowest rank first) agree on all cases.
    codebot_rank = sb.BPETokenizer(toks["codebot"].merge_rules)
    assert [codebot_rank.encode(t) for _, t in cases] == out["ids"]["codebot"]
    dump("encode.json", out)
    return cases


# --------------------------------------------------------------------------------------------- decode
EDGE_BYTES = [  # valid, BOM, truncated, stray continuation, overlong, surrogate, > U+10FFFF, never-valid bytes
    "", "41", "c3a9", "e38182", "f09f9881", "efbbbf41", "41efbbbf", "c3", "e381", "f09f98", "80", "bf41",
    "c080", "c1bf", "c0af", "e08080", "e09fbf", "eda080", "edbfbf", "edb0800a", "f08f8080", "f4908080",
    "f5", "fe", "ff", "c2", "f0", "e3e38182", "f09f41", "c3a9c3", "e28041", "41e2808a42", "f8888080",
    "fc8080808080", "ed9fbf", "ee8080", "f48fbfbf", "e0a080", "f0908080", "e182", "e1828283",
]


def make_decode(toks: dict) -> None:
    rng = random.Random(7)
    out = {"bytes": [], "ids": {}, "streams": {}, "digest": {}}
    lead = [0x41, 0x20, 0xC3, 0xE3, 0xF0, 0xED, 0xEF, 0xC0, 0xF4, 0xF5, 0xE0]
    raw = [bytes.fromhex(h) for h in EDGE_BYTES]
    for _ in range(400):
        raw.append(bytes(rng.choice(lead) if rng.random() < 0.3 else rng.choice(
            [rng.randrange(0x80, 0xC0), rng.randrange(0x20, 0x7F), rng.randrange(256)])
            for _ in range(rng.randint(1, 10))))
    out["bytes"] = [[b.hex(), b.decode("utf-8", errors="replace")] for b in raw]

    for bot, tok in toks.items():
        n = tok.vocab_size
        partial = [i for i, b in tok.id_to_bytes.items() if not _valid_utf8(b)]
        seqs = [[rng.randrange(n) for _ in range(rng.randint(1, 12))] for _ in range(150)]
        seqs += [[rng.choice(partial or [0]) for _ in range(rng.randint(1, 6))] for _ in range(50)]
        seqs += [[i] for i in range(min(n, 300))] + [[tok.end_token_id], [0xEF, 0xBB, 0xBF, 66]]
        out["ids"][bot] = [[s, tok.decode(s)] for s in seqs]
        # Incremental decoding, one token at a time (what a streaming chat UI must do).
        streams = []
        for s in seqs[:60] + seqs[150:170]:
            dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
            pieces = [dec.decode(tok.id_to_bytes[i]) for i in s] + [dec.decode(b"", final=True)]
            assert "".join(pieces) == tok.decode(s)
            streams.append([s, pieces])
        out["streams"][bot] = streams
        # Every token's bytes, as sha256 over (u32 little-endian length, bytes) for ids 0..vocab_size-1.
        h = hashlib.sha256()
        for i in range(n):
            b = tok.id_to_bytes[i]
            h.update(len(b).to_bytes(4, "little") + b)
        out["digest"][bot] = {"vocab_size": n, "sha256": h.hexdigest(), "partial_utf8": len(partial)}
    dump("decode.json", out)


def _valid_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# --------------------------------------------------------------------------------------------- train
def reference_train(text: str, vocab_size: int, tie_break: str, pretokenize: bool, special: str | None):
    """The book's training loop (ch01/05-06, codebot) with a pluggable tie-break; also records each step."""
    texts = text.split(special) if special else [text]
    if pretokenize:
        ids_list = [list(p.encode("utf-8")) for t in texts for p in cb.pretokenize(t)]
    else:
        ids_list = [list(t.encode("utf-8")) for t in texts]
    vocab = {i: bytes([i]) for i in range(256)}
    keys = {
        "first": None,  # max(counts, key=counts.get): first-inserted pair among the most frequent
        "ids": lambda c: lambda p: (c[p], p[0], p[1]),
        "bytes": lambda c: lambda p: (c[p], vocab[p[0]], vocab[p[1]]),
    }
    merges, steps = [], []
    for step in range(vocab_size - 256 - (1 if special else 0)):
        counts = defaultdict(int)
        for ids in ids_list:
            counts = cb.count_pairs(ids, counts)
        if not counts:
            break
        key = counts.get if tie_break == "first" else keys[tie_break](counts)
        best = max(counts, key=key)
        top_count = max(counts.values())
        order = sorted(counts, key=key, reverse=True) if tie_break != "first" else \
            sorted(counts, key=lambda p: -counts[p])  # stable: ties keep insertion order
        steps.append([counts[best], sum(c == top_count for c in counts.values()),
                      [[p[0], p[1], counts[p]] for p in order[:5]]])
        assert order[0] == best
        new_id = 256 + step
        merges.append([best[0], best[1]])
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        ids_list = [cb.merge(ids, best, new_id) for ids in ids_list]
    return merges, steps


def storybot_train(text: str, vocab_size: int) -> list[list[int]]:
    """ch04's trainer (dedupe + incremental cache + multiprocessing) reads a file in binary mode."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corpus.txt"
        path.write_bytes(text.encode("utf-8"))
        with contextlib.redirect_stderr(io.StringIO()):
            rules = sb.train_bpe(str(path), vocab_size, num_processes=1, num_chunks=4)
    return [list(p) for p in rules]


def train_corpora() -> list[tuple[str, str, int, list[tuple[bool, str | None]]]]:
    unicode_text = ("日本語のテキストを分割します。中文的分词器也一样。😁😁 emoji 👍🏽👍🏽 café café naïve "
                    "مرحبا مرحبا שלום שלום Ελληνικά Ελληνικά русский русский ") * 3
    codes = read_text("codebot/tiny_codes.txt", 4000)
    stories = read_text("storybot/tiny_stories_valid.txt", 3000)
    both = [(True, END), (False, END), (True, None), (False, None)]
    return [
        # name, text, vocab_size, [(pretokenize, special_token), ...]
        ("ch01-03", "Hello world! This is BPE training.", 260, [(False, None)]),
        ("ch01-03-exhaust", "Hello world! This is BPE training.", 300, [(False, None), (True, END)]),
        ("ch01-05", "Hello world!<|endoftext|>This is BPE training.", 260, [(False, END)]),
        ("ch01-06", "Say hello! Why hello? Just hello.<|endoftext|>Good morning!", 270, both),
        ("f44", END.join(["zzq"] * 3 + ["zz"] * 3 + ["aaq"] * 3 + ["aa"] * 2), 260, [(False, END), (True, END)]),
        ("runs", "aaaaaaa bbbbbbbb aaaa ab ba aaa\n\n\n\n    x    y  " * 3, 290, both),
        ("tinycodes-4k", codes, 400, [(True, END)]),
        ("tinystories-3k", stories, 360, [(True, END), (False, END)]),
        ("unicode", unicode_text, 340, [(True, END), (False, None)]),
        ("whitespace", "a\t\tb\r\n\r\nc \x85\x85d\u3000\u3000e\xa0\xa0f \ufeff\ufeffg\n\n\n  h  " * 4, 300, both),
    ]


def make_train() -> None:
    ch03, ch05, ch06 = (load_script(f"ch01/{n}.py") for n in ("03_bpe_train", "05_special_token", "06_pretokenize"))
    book = {  # (pretokenize, special) -> the book's own trainer for the 'first' tie-break
        (False, None): lambda t, v: ch03.train_bpe(t, v),
        (False, END): lambda t, v: ch05.train_bpe(t, v),
        (True, END): lambda t, v: ch06.train_bpe(t, v),
    }
    out, corpora, checked = [], {}, defaultdict(int)
    for name, text, vocab_size, configs in train_corpora():
        corpora[name] = text
        for pretok, special in configs:
            for tie in ("first", "ids", "bytes"):
                merges, steps = reference_train(text, vocab_size, tie, pretok, special)
                # The reference loop must reproduce every book trainer that implements this configuration.
                if tie == "first" and (pretok, special) in book:
                    with contextlib.redirect_stderr(io.StringIO()):
                        assert [list(p) for p in book[(pretok, special)](text, vocab_size)] == merges, name
                    checked["ch01 (first)"] += 1
                if tie == "ids" and (pretok, special) == (True, END):
                    with contextlib.redirect_stderr(io.StringIO()):
                        assert [list(p) for p in cb.train_bpe(text, vocab_size)] == merges, name
                    checked["codebot (ids)"] += 1
                    if "\r" not in text:  # storybot reads bytes; "\r\n" survives there but not in text mode
                        assert storybot_train(text, vocab_size) == merges, name
                        checked["storybot (ids)"] += 1
                out.append({"corpus": name, "vocab_size": vocab_size, "tie_break": tie, "pretokenize": pretok,
                            "special_token": special, "merges": merges, "steps": steps})
    print("reference loop checked against book trainers:", dict(checked))
    dump("train.json", {"corpora": corpora, "cases": out})


# --------------------------------------------------------------------------------------------- CS336
def make_cs336(a1: Path) -> None:
    """CS336 A1's own BPE fixture: corpus.en, vocab 500, special ["<|endoftext|>"] -> 243 reference merges."""
    fx = a1 / "tests" / "fixtures"
    sys.path.insert(0, str(a1))
    from tests.common import gpt2_bytes_to_unicode  # the test suite's own decoder

    text = (fx / "corpus.en").read_bytes().decode("utf-8").replace("\r\n", "\n")  # undo a CRLF checkout
    dec = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    ref = []
    for line in (fx / "train-bpe-reference-merges.txt").read_text(encoding="utf-8").splitlines():
        a, b = line.rstrip().split(" ")
        ref.append([bytes(dec[c] for c in a).hex(), bytes(dec[c] for c in b).hex()])
    merges, _ = reference_train(text, 500, "bytes", True, END)
    vocab = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(merges):
        vocab[256 + k] = vocab[a] + vocab[b]
    ours = [[vocab[a].hex(), vocab[b].hex()] for a, b in merges]
    assert ours == ref, "the bytes tie-break no longer reproduces CS336's reference merges"
    ids_merges, _ = reference_train(text, 500, "ids", True, END)
    first_diff = next(i for i, (x, y) in enumerate(zip(ids_merges, merges)) if x != y)
    print(f"CS336 corpus.en: bytes rule reproduces {len(ref)}/{len(ref)} merges; ids rule diverges at #{first_diff + 1}")
    dump("cs336.json", {
        "source": "https://github.com/stanford-cs336/assignment1-basics/tree/main/tests/fixtures",
        "license": "MIT, Copyright 2025 Stanford University",
        "vocab_size": 500, "special_tokens": [END], "text": text, "merges_hex": ref,
        "ids_rule_first_divergence": first_diff + 1,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs336", type=Path, help="checkout of stanford-cs336/assignment1-basics (rewrites cs336.json)")
    args = ap.parse_args()
    toks = {"codebot": cb.BPETokenizer.load_from(DLFS6 / "codebot" / "merge_rules.pkl"),
            "storybot": sb.BPETokenizer.load_from(DLFS6 / "storybot" / "merge_rules.pkl"),
            "webbot": sb.BPETokenizer.load_from(DLFS6 / "webbot" / "merge_rules.pkl")}
    make_unicode()
    make_encode(toks)
    make_decode(toks)
    make_train()
    if args.cs336:
        make_cs336(args.cs336)
    elif not (HERE / "cs336.json").is_file():
        print("cs336.json missing: pass --cs336 <assignment1-basics checkout> to create it")


if __name__ == "__main__":
    main()
