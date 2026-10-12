"""Build the TinyCodes slice that the in-browser trainer (content/train-in-browser.md) learns from.

The slice is a deterministic function of DLFS6's own corpus, codebot/tiny_codes.txt:
every STRIDE-th document, skipping the few documents that contain characters outside printable
ASCII (tabs and one stray control character), so the character vocabulary stays at 96 + 1.
Every 10th selected document goes to the held-out (validation) file, the rest to the training file.

    uv run python verify/_trainer_data.py          # rewrite web/data/train-in-browser/tinycodes-*.txt

verify/test_trainer.py re-runs build_slice() and checks that the shipped files are byte-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "third_party" / "dlfs6" / "codebot" / "tiny_codes.txt"
OUT = ROOT / "web" / "data" / "train-in-browser"
EOT = "<|endoftext|>"
STRIDE = 25          # 21,836 documents / 25 -> about 870 documents, about 250 KB
VAL_EVERY = 10       # documents 9, 19, 29, ... of the selection are held out


def printable(doc: str) -> bool:
    return all(32 <= ord(c) < 127 or c == "\n" for c in doc)


def build_slice(corpus: Path = CORPUS) -> tuple[str, str]:
    """Return (train_text, val_text); documents are joined with <|endoftext|> exactly as in the corpus."""
    docs = corpus.read_text(encoding="utf-8").split(EOT)
    chosen = [d for d in docs[::STRIDE] if printable(d)]
    train = [d for i, d in enumerate(chosen) if i % VAL_EVERY != VAL_EVERY - 1]
    val = [d for i, d in enumerate(chosen) if i % VAL_EVERY == VAL_EVERY - 1]
    return EOT.join(train), EOT.join(val)


def main() -> None:
    if not CORPUS.is_file():
        sys.exit(f"{CORPUS} is missing: run `git submodule update --init` first")
    train, val = build_slice()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in (("tinycodes-train.txt", train), ("tinycodes-val.txt", val)):
        # newline="" keeps "\n" as is on Windows (no CRLF translation).
        (OUT / name).write_text(text, encoding="utf-8", newline="")
        print(f"{name}: {len(text):,} chars, {text.count(EOT) + 1} documents")


if __name__ == "__main__":
    main()
