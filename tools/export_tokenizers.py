"""Export the three official DLFS6 merge tables to compact JSON for web/widgets/_bpe.js.

    uv run python tools/export_tokenizers.py            # writes web/data/tokenizers/{codebot,storybot,webbot}.json
    uv run python tools/export_tokenizers.py --check    # exit 1 if a JSON file is missing or stale

Input:  third_party/dlfs6/<bot>/merge_rules.pkl, an ordered dict {(id1, id2): new_id}. Insertion order is the
        training order (= merge priority), and new_id == 256 + position.
Output: one JSON object per bot (no whitespace), read by `Tokenizer.load(url)` in _bpe.js:

    {"format": "dlfs6-bpe/1", "name": "codebot", "python": "codebot.tokenizer.BPETokenizer",
     "source": "third_party/dlfs6/codebot/merge_rules.pkl", "source_url": "...", "sha256": "<of the .pkl>",
     "pattern": "<GPT-2 pre-tokenizer regex, Python `regex` syntax>",
     "num_merges": 743, "vocab_size": 1000, "special_tokens": {"<|endoftext|>": 999},
     "merges": [a0, b0, a1, b1, ...]}      # merge i joins ids (a_i, b_i) into id 256 + i

The vocab_size and special-token id are taken from the book's own BPETokenizer class, so the JSON states
exactly what the Python code would do (end_token_id = 256 + len(merge_rules), vocab_size = len(id_to_bytes)).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
OUT = ROOT / "web" / "data" / "tokenizers"
COMMIT = "c9b6e2ed531b08dd9f451a091a34e9645148e2e2"  # pinned submodule (DESIGN.md §1)
HF = "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main"
# The GPT-2 pre-tokenizer pattern exactly as written in codebot/ and storybot/tokenizer.py (checked below).
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# bot -> (Python module whose BPETokenizer the book uses for it, where the .pkl comes from)
BOTS = {
    "codebot": ("codebot.tokenizer",  # ch01/07 trains it; ch03 loads it
                f"https://github.com/oreilly-japan/deep-learning-from-scratch-6/blob/{COMMIT}/codebot/merge_rules.pkl"),
    "storybot": ("storybot.tokenizer", f"{HF}/storybot/merge_rules.pkl"),  # ch04/04 trains it
    "webbot": ("storybot.tokenizer", f"{HF}/webbot/merge_rules.pkl"),      # ch07 reuses storybot's code
}


def build(bot: str) -> dict:
    module_name, url = BOTS[bot]
    pkl = DLFS6 / bot / "merge_rules.pkl"
    if not pkl.is_file():
        raise SystemExit(f"{pkl} is missing: run tools/fetch_data.py (or the bot's download.sh) first")
    raw = pkl.read_bytes()
    rules = pickle.loads(raw)  # trusted: the book's own artifact, pinned by sha256 below
    if not isinstance(rules, dict):
        raise SystemExit(f"{pkl}: expected a dict, got {type(rules).__name__}")

    flat: list[int] = []
    for rank, ((a, b), new_id) in enumerate(rules.items()):
        # The flat format relies on new_id == 256 + rank, and encode-by-rank relies on every merge using only
        # ids that already exist; check both instead of assuming them.
        if new_id != 256 + rank:
            raise SystemExit(f"{pkl}: merge #{rank} creates id {new_id}, expected {256 + rank}")
        if not (0 <= a < new_id and 0 <= b < new_id):
            raise SystemExit(f"{pkl}: merge #{rank} ({a}, {b}) -> {new_id} refers to a later id")
        flat += [a, b]

    source = (DLFS6 / Path(*module_name.split("."))).with_suffix(".py").read_text(encoding="utf-8")
    if f'pattern = r"""{PATTERN}"""' not in source:
        raise SystemExit(f"{module_name}: the pre-tokenizer pattern differs from the one this exporter documents")
    if str(DLFS6) not in sys.path:
        sys.path.insert(0, str(DLFS6))
    tok = importlib.import_module(module_name).BPETokenizer(rules)
    return {
        "format": "dlfs6-bpe/1",
        "name": bot,
        "python": f"{module_name}.BPETokenizer",
        "source": f"third_party/dlfs6/{bot}/merge_rules.pkl",
        "source_url": url,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "pattern": PATTERN,
        "num_merges": len(rules),
        "vocab_size": tok.vocab_size,
        "special_tokens": {tok.end_token: tok.end_token_id},
        "merges": flat,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the JSON files are up to date; write nothing")
    ap.add_argument("bots", nargs="*", default=list(BOTS), help="subset of: " + ", ".join(BOTS))
    args = ap.parse_args()
    stale = []
    for bot in args.bots:
        if bot not in BOTS:
            ap.error(f"unknown bot {bot!r}")
        data = build(bot)
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        path = OUT / f"{bot}.json"
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"{path.relative_to(ROOT)}: {data['num_merges']:,} merges, vocab {data['vocab_size']:,}, "
              f"special {data['special_tokens']}, {len(text.encode()):,} bytes")
    if stale:
        print("stale or missing:", ", ".join(stale), "-> run: uv run python tools/export_tokenizers.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
