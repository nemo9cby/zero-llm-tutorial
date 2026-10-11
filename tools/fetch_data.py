"""Download the book's data and official checkpoints into third_party/dlfs6/<bot>/.

The book scripts expect these files next to their bot package (see */download.sh). They are
excluded from the submodule's git status via .git/modules/third_party/dlfs6/info/exclude.

Usage:
  uv run python tools/fetch_data.py            # the "small" set (< 150 MB each, ~0.6 GB total)
  uv run python tools/fetch_data.py --full     # also the training corpora (1.1 GB + 5.3 GB)
  uv run python tools/fetch_data.py --list
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
HF = "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/"
LOCAL_CACHE = Path.home() / "Projects" / "_refs" / "dlfs6"  # earlier clone used during research

# (path, size in bytes, set)
MANIFEST = [
    ("codebot/model_pretrain.pt", 44_516_353, "small"),
    ("codebot/model_sft.pt", 44_515_938, "small"),
    ("codebot/model_grpo.pt", 44_516_021, "small"),
    ("storybot/merge_rules.pkl", 103_877, "small"),
    ("storybot/tiny_stories_valid.txt", 22_502_601, "small"),
    ("storybot/tiny_stories_valid.bin", 10_931_746, "small"),
    ("storybot/model_pretrain.pt", 90_835_030, "small"),
    ("storybot/model_iter_500.pt", 90_835_030, "small"),
    ("storybot/model_iter_5000.pt", 90_835_075, "small"),
    ("webbot/merge_rules.pkl", 533_514, "small"),
    ("webbot/owt_valid.bin", 128_992_132, "small"),
    ("storybot/tiny_stories_train.bin", 1_082_458_446, "full"),
    ("webbot/owt_train.bin", 5_298_009_600, "full"),
]


def fetch(rel: str, size: int) -> str:
    dst = DLFS6 / rel
    if dst.is_file() and dst.stat().st_size == size:
        return "present"
    cached = LOCAL_CACHE / rel
    if cached.is_file() and cached.stat().st_size == size:
        shutil.copy2(cached, dst)
        return "copied from local cache"
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(HF + rel, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab" if have else "wb") as f:
        if have and resp.status != 206:  # server ignored the range: start over
            f.seek(0)
            f.truncate()
        shutil.copyfileobj(resp, f, length=1 << 20)
    if part.stat().st_size != size:
        raise RuntimeError(f"{rel}: expected {size} bytes, got {part.stat().st_size}")
    part.replace(dst)
    return "downloaded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also fetch the large training corpora")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if not (DLFS6 / "codebot").is_dir():
        sys.exit("third_party/dlfs6 is empty — run: git submodule update --init")
    for rel, size, which in MANIFEST:
        if which == "full" and not args.full:
            continue
        if args.list:
            print(f"{rel:36} {size / 1e6:9.1f} MB  [{which}]")
            continue
        print(f"{rel:36} {size / 1e6:9.1f} MB  ... ", end="", flush=True)
        print(fetch(rel, size))


if __name__ == "__main__":
    main()
