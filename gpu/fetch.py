"""Download only what the given bots' GPU jobs read (data + official checkpoints) into third_party/dlfs6/.

    uv run python gpu/fetch.py storybot                 # TinyStories .bin files, merges, official checkpoint
    uv run python gpu/fetch.py webbot                   # OpenWebText .bin files (5.4 GB), merges
    uv run python gpu/fetch.py --list codebot storybot webbot

Resumable and size-checked: each file goes through tools/fetch_data.py's fetch(); gpu/jobs.py decides
which files a bot's jobs need, so a StoryBot box never downloads OpenWebText.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import fetch_data  # noqa: E402
import jobs  # noqa: E402


def plan_files(bots: list[str], official: bool = True) -> list[tuple[str, int]]:
    """(repo path, size in bytes) for every file these bots' jobs read."""
    sizes = {rel: size for rel, size, _ in fetch_data.MANIFEST}
    return [(rel, sizes[rel]) for rel in jobs.files_for_bots(bots, official)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bots", nargs="+", choices=["codebot", "storybot", "webbot"])
    ap.add_argument("--list", action="store_true", help="only list the files and sizes")
    ap.add_argument("--no-official", action="store_true", help="skip the book's pretrained checkpoints")
    a = ap.parse_args(argv)
    files = plan_files(a.bots, official=not a.no_official)
    for rel, size in files:
        print(f"{rel:36} {size / 1e6:9.1f} MB  ... ", end="", flush=True)
        print("(listed)" if a.list else fetch_data.fetch(rel, size))
    print(f"total {sum(s for _, s in files) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
