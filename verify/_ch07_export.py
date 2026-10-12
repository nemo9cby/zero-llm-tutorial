"""Chapter 7 data builder: everything under web/data/ch07/ (and tests/js/fixtures/ch07/) comes from here.

    uv run python verify/_ch07_export.py           # (re)write the files
    uv run python verify/_ch07_export.py --check   # exit 1 if any file is stale (test_ch07.py runs this)

Files:
  web/data/ch07/storybot_tokenizer.py  byte-for-byte copy of third_party/dlfs6/storybot/tokenizer.py (the browser
                                       cells import the book's own file; WebBot has no tokenizer module of its own)
  web/data/ch07/owt_head.bin           the first tokens of the official webbot/owt_valid.bin (uint16), up to and
                                       including the first <|endoftext|> after 100,000 bytes of text
  web/data/ch07/presets.json           sample texts for the three-tokenizer widget (figure 7-1)
  tests/js/fixtures/ch07/compare.json  the ids the book's BPETokenizer gives for every preset, per bot
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch07_common import (DLFS6, FIXTURES, HEAD_MIN_BYTES, OUT, book_modules,  # noqa: E402
                          head_token_count, official_tokenizer, valid_bin)

BOTS = {"codebot": 1_000, "storybot": 10_000, "webbot": 50_000}


def presets(owt_first_sentence: str) -> list[dict]:
    return [
        {"name": "网页（OWT）", "text": owt_first_sentence},
        {"name": "故事", "text": "Once upon a time, there was a little girl named Lily. She loved to play outside."},
        {"name": "代码", "text": "def greet(name):\n    print(f\"Hello, {name}!\")\n\nfor i in range(3):\n    greet('world')\n"},
        {"name": "中文", "text": "语言模型只认识整数，分词器负责把文字变成整数。"},
        {"name": "网址与数字", "text": "https://example.com/api/v2/items?id=1234567890&sort=desc"},
        {"name": "emoji", "text": "I love 🍕 and 🍣! 😁"},
        {"name": "特殊 token", "text": "print('<|endoftext|>')"},
    ]


def ordinary_ids(sb, tok, text: str) -> list[int]:
    """Encode with special tokens treated as plain text (what tiktoken's encode_ordinary does)."""
    return [i for piece in sb.pretokenize(text) for i in tok._encode_text(piece)]


def build() -> dict[Path, bytes]:
    sb = book_modules()
    files: dict[Path, bytes] = {}
    files[OUT / "storybot_tokenizer.py"] = (DLFS6 / "storybot" / "tokenizer.py").read_bytes()

    tok = official_tokenizer()
    arr = valid_bin()
    ids = arr[:HEAD_MIN_BYTES].tolist()          # far more than needed (~4.5 bytes per token)
    k = head_token_count(tok, ids, HEAD_MIN_BYTES)
    files[OUT / "owt_head.bin"] = arr[:k].tobytes()

    text = tok.decode(ids[:200])
    first_sentence = text[: text.index("night.") + len("night.")]
    pre = presets(first_sentence)
    files[OUT / "presets.json"] = (json.dumps({"presets": pre}, ensure_ascii=False, indent=1) + "\n").encode()

    toks = {}
    for bot in BOTS:
        with open(DLFS6 / bot / "merge_rules.pkl", "rb") as f:
            toks[bot] = sb.BPETokenizer(pickle.load(f))
        assert toks[bot].vocab_size == BOTS[bot], bot
    cases = []
    for p in pre:
        case = {"name": p["name"], "text": p["text"], "bytes": len(p["text"].encode("utf-8")), "ids": {}, "ordinary": {}}
        for bot, t in toks.items():
            case["ids"][bot] = t.encode(p["text"])
            case["ordinary"][bot] = ordinary_ids(sb, t, p["text"])
            assert t.decode(case["ids"][bot]) == p["text"]
        cases.append(case)
    files[FIXTURES / "compare.json"] = (json.dumps({"source": "verify/_ch07_export.py (storybot.tokenizer.BPETokenizer "
                                                   "+ the three merge_rules.pkl)", "cases": cases},
                                                  ensure_ascii=False) + "\n").encode()
    return files


def main() -> int:
    check = "--check" in sys.argv
    stale = []
    for path, data in build().items():
        if check:
            if not path.is_file() or path.read_bytes() != data:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"wrote {path.relative_to(OUT.parent.parent.parent)} ({len(data):,} bytes)")
    if stale:
        print("stale:", *[str(p) for p in stale], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
