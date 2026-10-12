"""Chapter 4 measurements: where every timing and memory number in web/data/ch04/bench.json comes from.

Each experiment writes one raw record, web/data/ch04/bench_runs/<experiment>.json; `assemble` rebuilds
bench.json from those records alone, so nothing in bench.json is typed by hand:

    uv run --with psutil python verify/_ch04_bench.py run train_small encode_small   # CPU only, minutes to an hour
    uv run python verify/_ch04_bench.py assemble                                    # bench_runs/*.json -> bench.json

Experiments (every timed variant is checked against a reference, and the record says against which):
  naive_codes      ch01's train_bpe on the whole TinyCodes, vocab 1000 (about an hour; compared with codebot/merge_rules.pkl)
  train_small      ch01 / 01 / 02 / 03 / heap on the first 499,577 bytes of the TinyStories validation set, vocab 1000
  train_codes      01 / 02 / 03 / heap on the whole TinyCodes (compared with codebot/merge_rules.pkl)
  train_valid10k   02 on the whole validation set, vocab 10000, plus a stopwatch around max() and around the update
  loops_valid10k   the merge loop alone (no pretokenization): 02's loop vs the lazy heap
  scanned_valid10k pairs that max() looks at in each of the 9,743 steps
  shrink           token ids a naive trainer scans per step, relative to the first step (TinyCodes, validation set)
  pretok_scaling   pretokenization of the validation set: whole read, 03's chunked read, Pool(n) for n = 1..16
  pretok_repeat    whole read vs 03's chunked read, interleaved three times
  pool_startup     Pool(n) start + one trivial task per worker + shutdown, with 04's imports in the workers
  repeat_small     ch01 / 01 / 02 / heap on the train_small input again, interleaved three times
  encode_small     ch01's encoder vs 06 vs storybot vs memo on the first 29,642 bytes; ' accomplishment' alone
  encode_valid     06 and memo on the whole validation set (compared with the official tiny_stories_valid.bin)
  encode_file_scaling  storybot's encode_file with 16/8/4/2/1 processes (sha256 vs the official .bin)
  memory           peak working set of a fresh process per way of pretokenizing (needs --train-prefix)
  heaps            occurrences vs distinct pretokens on growing prefixes (needs --train-prefix)
  pep393           CPython string storage of the whole text vs the pieces between <|endoftext|> (needs --train-prefix)
  train_prefix_encode  encode_file on the train prefix vs the official train.bin prefix (needs both prefixes)
  train_full       ch04/04's training on the whole 2.2 GB train file (8 processes, 64 chunks), compared with the
                   official storybot/merge_rules.pkl (needs --train-full; about 10 minutes)
  encode_full      ch04/07's encode_file on the whole train file (8 processes, 64 chunks), compared by sha256 with the
                   official tiny_stories_train.bin (needs --train-full; under an hour)
  heaps_full       occurrences and distinct pretokens of the whole train file (needs --train-full; a few minutes)

Three inputs are not in the repository (see INPUTS for the URLs, sizes and sha256): --train-prefix (the first
100 MiB of tiny_stories_train.txt, cut before an <|endoftext|>) and --train-bin-prefix (the first 50,946,778
bytes of the official tiny_stories_train.bin), both fetched once with HTTP Range requests, and --train-full
(the whole tiny_stories_train.txt, the first file storybot/download.sh downloads).

verify/test_ch04.py::test_bench_json_consistent checks bench.json == assemble() and recomputes the
deterministic facts; test_bench_full_valid_facts (slow) re-runs the ones that need the whole validation set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
DATA = ROOT / "web" / "data" / "ch04"
RAW = DATA / "bench_runs"
DEST = DATA / "bench.json"
EOT = "<|endoftext|>"
VALID = DLFS6 / "storybot" / "tiny_stories_valid.txt"
VALID_BIN = DLFS6 / "storybot" / "tiny_stories_valid.bin"
CODES = DLFS6 / "codebot" / "tiny_codes.txt"
SMALL_LIMIT, SLICE30_LIMIT = 500_000, 30_000          # train_small / encode_small inputs: cut before an EOT
VALID_MARKS = [1 << 20, 2 << 20, 5 << 20, 10 << 20]   # Heaps checkpoints (bytes read)
PREFIX_MARKS = [1 << 20, 10 << 20, 50 << 20]

INPUTS = {
    "train_prefix": {"what": "the first 100 MiB of storybot/tiny_stories_train.txt, cut before an <|endoftext|>",
                     "url": "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.txt",
                     "bytes": 104_856_833,
                     "sha256": "3ba2ac7a6369e5f19c6cd62d5871d002241d8ede13fbe72056fc5ef52830a994"},
    "train_bin_prefix": {"what": "the first 50,946,778 bytes of the official storybot/tiny_stories_train.bin",
                         "url": "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.bin",
                         "bytes": 50_946_778,
                         "sha256": "0893d9cc166e91c361f096f867159332f236b97f230c1fdd2902b3f970b016a1"},
    "train_full": {"what": "the whole storybot/tiny_stories_train.txt (= TinyStoriesV2-GPT4-train.txt, see web/data/cs336/data_identity.json)",
                   "url": "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.txt",
                   "bytes": 2_227_753_162,
                   "sha256": "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443"},
}

# The official encoding of the whole train file, as published (encode_full compares with it, without downloading it).
OFFICIAL_TRAIN_BIN = {
    "file": "storybot/tiny_stories_train.bin", "bytes": 1_082_458_446,
    "sha256": "74c89ae1d5d46d474304841ea1b62bbe9139e11f7579ee5d5fff409b97c2777c",
    "source": "lfs.oid and size in GET https://huggingface.co/api/datasets/koki0702/zero-llm-data/tree/main/storybot "
              "(2026-10-12); the same listing gives merge_rules.pkl 8fb285d9... and tiny_stories_valid.bin 05a36c44..., "
              "which are the sha256 of the files in third_party/dlfs6/storybot"}

# The first batch of records (2026-10-11) was taken with a scratch copy of this script, before it was moved
# into the repository; those records carry no commit of their own, so the facts are stated here once.
FIRST_BATCH = {
    "records": ["naive_codes", "train_small", "train_codes", "train_valid10k", "loops_valid10k", "scanned_valid10k",
                "pretok_scaling", "encode_small", "encode_valid", "encode_file_scaling", "memory", "heaps", "pep393",
                "train_prefix_encode"],
    "measured": "2026-10-11 20:04-21:02 (UTC-4)",
    "dlfs6_commit": "c9b6e2ed531b08dd9f451a091a34e9645148e2e2",
    "tutorial_code": "working tree on top of c517071; the measured ch04 code (web/data/ch04/bpe_ext.py, "
                     "verify/_ch04_load.py) was first committed in 13841d1",
}


# ------------------------------------------------------------------ deterministic helpers (also used by the tests)
def slice_before_eot(data: bytes, limit: int) -> bytes:
    """The first stories of a corpus: everything before the last <|endoftext|> that starts before `limit`."""
    return data[: data.rindex(EOT.encode(), 0, limit)]


def heaps_rows(path: Path, marks: list[int], name: str | None = None) -> list[dict]:
    """Occurrences and distinct pretokens after reading about `marks` bytes, then for the whole file.

    One pass in the book's order (split on <|endoftext|>, pretokenize each piece); the last row also gives the
    share of all occurrences taken by the 10 and the 1,000 most frequent pretokens."""
    from storybot.tokenizer import pretokenize

    text = path.read_bytes().decode("utf-8")
    name = name or path.name
    rows, counts, consumed, k = [], defaultdict(int), 0, 0
    for piece in text.split(EOT):
        for p in pretokenize(piece):
            counts[p] += 1
        consumed += len(piece.encode("utf-8")) + len(EOT)
        while k < len(marks) and consumed >= marks[k]:
            rows.append({"file": name, "bytes": consumed, "occurrences": sum(counts.values()), "unique": len(counts)})
            k += 1
    rows.append({"file": name, "bytes": path.stat().st_size, "occurrences": sum(counts.values()), "unique": len(counts)})
    top = sorted(counts.values(), reverse=True)
    rows[-1]["top10_share"] = round(sum(top[:10]) / sum(top), 4)
    rows[-1]["top1000_share"] = round(sum(top[:1000]) / sum(top), 4)
    return rows


def pep393_stats(text: str) -> dict:
    """sys.getsizeof of the whole string vs the pieces between <|endoftext|> (PEP 393: width set by the widest char)."""
    pieces = text.split(EOT)
    return {"pieces": len(pieces), "wide_pieces": sum(1 for p in pieces if p and max(map(ord, p)) > 255),
            "whole_str_bytes": sys.getsizeof(text), "pieces_str_bytes": sum(sys.getsizeof(p) for p in pieces),
            "max_codepoint": max(map(ord, text))}


def merge_loop_profile(ids_counts: dict, num_merges: int, count_ids: bool = False, clock=None) -> tuple[dict, dict]:
    """ch04/02's merge loop (bpe_ext.init_cache / apply_merge are its lines) with counters per step:
    scanned  = pairs that max() looks at,  affected = distinct sequences the update touches,
    ids_occ  = token ids a naive trainer would scan (every occurrence), ids_uniq = ids the deduplicated one scans
    (count_ids=True; costs one extra merge per affected sequence). clock=time.perf_counter adds the time spent
    in max() and in the update. ids_counts is modified in place."""
    import bpe_ext as bx

    pair_counts, pair_to_ids = bx.init_cache(ids_counts)
    rules, stats = {}, {"scanned": [], "affected": [], "ids_occ": [], "ids_uniq": [], "t_max": 0.0, "t_update": 0.0}
    ids_occ = sum(c * len(k) for k, c in ids_counts.items())
    ids_uniq = sum(len(k) for k in ids_counts)
    for step in range(num_merges):
        if not pair_counts:
            break
        stats["scanned"].append(len(pair_counts))
        t0 = clock() if clock else 0.0
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        t1 = clock() if clock else 0.0
        if count_ids:
            stats["ids_occ"].append(ids_occ)
            stats["ids_uniq"].append(ids_uniq)
            for ids in pair_to_ids[best_pair]:
                saved = len(ids) - len(bx.merge(ids, best_pair, 256 + step))
                ids_occ -= ids_counts[ids] * saved
                ids_uniq -= saved
        rules[best_pair] = 256 + step
        n_affected, _ = bx.apply_merge(ids_counts, pair_counts, pair_to_ids, best_pair, 256 + step)
        if clock:
            t2 = clock()
            stats["t_max"] += t1 - t0
            stats["t_update"] += t2 - t1
        stats["affected"].append(n_affected)
    return rules, stats


def scanned_summary(scanned: list[int]) -> dict:
    return {"first": scanned[0], "max": max(scanned), "last": scanned[-1],
            "mean": round(sum(scanned) / len(scanned), 1), "steps": len(scanned)}


def shrink_summary(ids_occ: list[int]) -> dict:
    """How much the naive trainer's per-step work shrinks as merges proceed (ids scanned, relative to step 1)."""
    return {"first": ids_occ[0], "last": ids_occ[-1], "mean_over_first": round(sum(ids_occ) / len(ids_occ) / ids_occ[0], 3),
            "steps": len(ids_occ)}


def rules_digest(rules: dict) -> str:
    """sha256 of the merge rules as an ordered list: same pairs, same ids, same order <=> same digest."""
    return hashlib.sha256(repr(list(rules.items())).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """sha256 of a file, read in 16 MiB blocks (the inputs here are up to 2.2 GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ measurement helpers (psutil only needed here)
class Load:
    """Samples whole-machine CPU utilisation (all logical CPUs) every 0.5 s in a background thread."""

    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def __enter__(self):
        import psutil

        self._psutil = psutil
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._psutil.cpu_percent(interval=0.5))

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self):
        s = self.samples or [self._psutil.cpu_percent(0.2)]
        return {"mean": round(sum(s) / len(s), 1), "min": round(min(s), 1), "max": round(max(s), 1)}


class MemPeak:
    """Largest resident memory of this process plus all its children (pool workers), sampled every 0.5 s."""

    METHOD = "psutil rss of the process and its children, summed, sampled every 0.5 s"

    def __init__(self):
        self.peak, self._stop = 0, threading.Event()

    def __enter__(self):
        import psutil

        self._me = psutil.Process()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _sample(self):
        total = 0
        for p in [self._me, *self._me.children(recursive=True)]:
            try:
                total += p.memory_info().rss
            except Exception:              # a worker that exits between listing and reading
                pass
        self.peak = max(self.peak, total)

    def _run(self):
        while not self._stop.wait(0.5):
            self._sample()

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self._sample()

    @property
    def peak_mib(self) -> float:
        return round(self.peak / 2**20, 1)


def timed(fn, *args, **kwargs):
    with Load() as load:
        w0, c0 = time.perf_counter(), time.process_time()
        out = fn(*args, **kwargs)
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    return out, {"wall": round(wall, 3), "cpu": round(cpu, 3), "load": load.summary()}


class PhaseTimer:
    """Stand-in for tqdm that records the wall/CPU time of each loop it wraps, keyed by tqdm's desc."""

    def __init__(self):
        self.phases = {}

    def __call__(self, iterable=None, *args, desc=None, **kwargs):
        def gen():
            w0, c0 = time.perf_counter(), time.process_time()
            try:
                yield from iterable
            finally:
                self.phases[desc or "loop"] = {"wall": round(time.perf_counter() - w0, 3),
                                               "cpu": round(time.process_time() - c0, 3)}
        return gen()


def run_phased(mod, fn_name, *args, **kwargs):
    from _ch04_load import quiet_tqdm

    pt = PhaseTimer()
    mod.tqdm = pt
    try:
        out, total = timed(getattr(mod, fn_name), *args, **kwargs)
    finally:
        mod.tqdm = quiet_tqdm
    total["phases"] = pt.phases
    return out, total


def git_state(repo: Path) -> dict:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout
    return {"commit": head, "dirty": bool(dirty.strip())}


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    return platform.processor()


def env_info() -> dict:
    import numpy as np
    import psutil
    import regex

    return {"cpu": cpu_name(), "logical_cpus": psutil.cpu_count(), "physical_cores": psutil.cpu_count(False),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "os": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "regex": regex.__version__,
            "date": time.strftime("%Y-%m-%d %H:%M"), "utc_offset": time.strftime("%z")}


def command_line(argv: list[str]) -> str:
    """The command that made a record. Inputs from outside the repository are named, not given by their local path:
    `need` has checked their sha256 against INPUTS."""
    flags = {"--" + name.replace("_", "-"): name for name in INPUTS}
    out = []
    for prev, arg in zip([None, *argv], argv):
        flag, eq, _ = arg.partition("=")
        if prev in flags:                       # --train-full <path>
            arg = f"<{flags[prev]}>"
        elif eq and flag in flags:              # --train-full=<path>
            arg = f"{flag}=<{flags[flag]}>"
        out.append(arg)
    return "python verify/_ch04_bench.py " + " ".join(out)


def save(name: str, obj: dict) -> None:
    """Write a raw record, stamped with the environment, the commits and the command that produced it."""
    obj["env"] = env_info()
    obj["provenance"] = {"script": "verify/_ch04_bench.py", "command": command_line(sys.argv[1:]),
                         "tutorial": git_state(ROOT), "dlfs6": git_state(DLFS6)}
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(obj, ensure_ascii=False, indent=1)[:3000])


def same(a: dict, b: dict) -> bool:
    return list(a.items()) == list(b.items())


class Book:
    """The book's code, loaded once (definitions only, tqdm silenced) for the experiments."""

    def __init__(self):
        for p in (str(DLFS6), str(DATA), str(Path(__file__).resolve().parent)):
            if p not in sys.path:
                sys.path.insert(0, p)
        sys.dont_write_bytecode = True
        import bpe_ext
        import codebot.tokenizer as cb
        import storybot.tokenizer as sb
        from _ch04_load import load_defs, quiet_tqdm

        cb.tqdm = sb.tqdm = quiet_tqdm
        self.cb, self.sb, self.ext = cb, sb, bpe_ext
        self.v01, self.v02 = load_defs("ch04/01_bpe_optimize.py"), load_defs("ch04/02_bpe_cache.py")
        self.v03, self.v06 = load_defs("ch04/03_bpe_chunk.py"), load_defs("ch04/06_encode_optimize.py")
        self.story_rules = pickle.loads((DLFS6 / "storybot" / "merge_rules.pkl").read_bytes())
        self.code_rules = pickle.loads((DLFS6 / "codebot" / "merge_rules.pkl").read_bytes())
        self.tmp = Path(tempfile.mkdtemp(prefix="ch04-bench-"))

    def slice_file(self, limit: int) -> Path:
        path = self.tmp / f"valid_{limit}.txt"
        if not path.exists():
            path.write_bytes(slice_before_eot(VALID.read_bytes(), limit))
        return path

    def trainer(self, key: str, text: str, path: Path, vocab: int):
        """One training run of a variant, timed; returns (rules, timing)."""
        if key == "naive":
            return run_phased(self.cb, "train_bpe", text, vocab)
        if key == "dedup":
            return run_phased(self.v01, "train_bpe", text, vocab)
        if key == "cache":
            return run_phased(self.v02, "train_bpe", text, vocab)
        if key == "chunk":
            return run_phased(self.v03, "train_bpe", str(path), vocab)
        if key == "heap":
            return timed(self.ext.train_bpe_heap, text, vocab)
        raise KeyError(key)


def read_text(path: Path) -> str:
    return Path(path).read_bytes().decode("utf-8")


# ------------------------------------------------------------------ experiments
def exp_naive_codes(b: Book, args):
    rules, t = b.trainer("naive", read_text(CODES), CODES, 1000)
    save("naive_codes", {"input": "codebot/tiny_codes.txt (LF)", "bytes": CODES.stat().st_size, "vocab": 1000,
                         "naive": t, "equals_codebot_merge_rules_pkl": same(rules, b.code_rules)})


def exp_train_small(b: Book, args):
    path = b.slice_file(SMALL_LIMIT)
    text = read_text(path)
    runs, merges = {}, {}
    for key in ["dedup", "naive", "cache", "chunk", "heap", "dedup", "cache", "heap"]:
        out, t = b.trainer(key, text, path, 1000)
        runs.setdefault(key, []).append(t)
        merges.setdefault(key, out)
        print(key, t["cpu"], flush=True)
    save("train_small", {"input": "tiny_stories_valid.txt first 499,577 bytes", "bytes": path.stat().st_size,
                         "vocab": 1000, "runs": runs, "merges": len(merges["naive"]),
                         "identical_to_naive": {k: same(v, merges["naive"]) for k, v in merges.items()}})


def exp_repeat_small(b: Book, args):
    """The train_small input again, the four variants interleaved three times: how much do CPU times move?"""
    path = b.slice_file(SMALL_LIMIT)
    text = read_text(path)
    rounds, digests = [], {}
    for r in range(args.rounds):
        row = {}
        for key in ["naive", "dedup", "cache", "heap"]:
            out, t = b.trainer(key, text, path, 1000)
            row[key] = t
            digests.setdefault(key, rules_digest(out))
            print(r, key, t["cpu"], t["load"]["mean"], flush=True)
        rounds.append(row)
    save("repeat_small", {"input": "tiny_stories_valid.txt first 499,577 bytes", "bytes": path.stat().st_size,
                          "vocab": 1000, "rounds": rounds, "rules_sha256": digests,
                          "identical": len(set(digests.values())) == 1})


def exp_train_codes(b: Book, args):
    text = read_text(CODES)
    runs, merges = {}, {}
    for key in ["dedup", "cache", "chunk", "heap", "cache", "heap"]:
        out, t = b.trainer(key, text, CODES, 1000)
        runs.setdefault(key, []).append(t)
        merges.setdefault(key, out)
    save("train_codes", {"input": "codebot/tiny_codes.txt (LF)", "bytes": CODES.stat().st_size, "vocab": 1000,
                         "runs": runs, "equals_codebot_pkl": {k: same(v, b.code_rules) for k, v in merges.items()}})


def exp_train_valid10k(b: Book, args):
    text = read_text(VALID)
    book, t_book = run_phased(b.v02, "train_bpe", text, 10000)
    (rules, st), t_rep = timed(merge_loop_profile, b.ext.ids_counts_of(text, EOT), 9743, clock=time.perf_counter)
    heap, t_heap = timed(b.ext.train_bpe_heap, text, 10000)
    a = sorted(st["affected"])
    save("train_valid10k", {
        "input": "storybot/tiny_stories_valid.txt", "bytes": VALID.stat().st_size, "vocab": 10000,
        "book_02": t_book, "replica": t_rep, "replica_max_s": round(st["t_max"], 3),
        "replica_update_s": round(st["t_update"], 3), "max_share": round(st["t_max"] / (st["t_max"] + st["t_update"]), 4),
        "heap": t_heap, "unique_pretokens": len(b.ext.ids_counts_of(text, EOT)),
        "affected_mean": round(sum(a) / len(a), 2), "affected_median": a[len(a) // 2],
        "identical": {"replica": same(rules, book), "heap": same(heap, book)}, "merges": len(book)})


def exp_loops_valid10k(b: Book, args):
    text = read_text(VALID)
    book, t_book = run_phased(b.v02, "train_bpe", text, 10000)
    heap, t_heap = timed(b.ext.heap_merge_loop, dict(b.ext.ids_counts_of(text, EOT)), 9743)
    save("loops_valid10k", {"input": "storybot/tiny_stories_valid.txt", "vocab": 10000,
                            "book_loop": t_book["phases"]["Training BPE"], "book_load": t_book["load"],
                            "heap_loop": t_heap, "identical": same(heap, book)})


def exp_scanned_valid10k(b: Book, args):
    _, st = merge_loop_profile(b.ext.ids_counts_of(read_text(VALID), EOT), 9743)
    save("scanned_valid10k", scanned_summary(st["scanned"]))


def exp_shrink(b: Book, args):
    out = {}
    for name, path, merges in [("tiny_codes", CODES, 743), ("tiny_stories_valid", VALID, 9743)]:
        _, st = merge_loop_profile(b.ext.ids_counts_of(read_text(path), EOT), merges, count_ids=True)
        out[name] = shrink_summary(st["ids_occ"])
        print(name, out[name], flush=True)
    save("shrink", out)


def pretok_counts_serial(sb, text: str) -> dict:
    counts = defaultdict(int)
    for piece in text.split(EOT):
        for p in sb.pretokenize(piece):
            counts[p] += 1
    return counts


def pool_pretok(sb, n: int, num_chunks: int = 64):
    """ch04/04 steps 1-3 (storybot's pretoken_chunk = 04's), Pool start-up and shutdown included."""
    from multiprocessing import Pool

    w0 = time.perf_counter()
    bounds = sb.find_chunk_boundaries(str(VALID), num_chunks)
    info = [(str(VALID), s, e, EOT) for s, e in zip(bounds[:-1], bounds[1:])]
    p0 = time.perf_counter()
    with Pool(processes=n) as pool:
        p1 = time.perf_counter()
        results = list(pool.imap(sb.pretoken_chunk, info))
        p2 = time.perf_counter()
    p3 = time.perf_counter()
    total = defaultdict(int)
    for r in results:
        for k, v in r.items():
            total[k] += v
    return total, {"startup": round(p1 - p0, 3), "work": round(p2 - p1, 3), "shutdown": round(p3 - p2, 3),
                   "chunks": len(info), "wall": round(time.perf_counter() - w0, 3)}


def exp_pretok_scaling(b: Book, args):
    out = {"input": "storybot/tiny_stories_valid.txt", "bytes": VALID.stat().st_size}
    with Load() as load:
        w0 = time.perf_counter()
        ref = pretok_counts_serial(b.sb, read_text(VALID))
        out["serial_whole"] = {"wall": round(time.perf_counter() - w0, 3)}
    out["serial_whole"]["load"] = load.summary()
    _, out["serial_chunked_03"] = run_phased(b.v03, "train_bpe", str(VALID), 257)   # 0 merges: pretokenization only
    rows = []
    for n in [1, 2, 4, 8, 16, 8, 4]:
        with Load() as load:
            counts, info = pool_pretok(b.sb, n)
        info.update(n=n, load=load.summary(), identical=dict(counts) == dict(ref))
        rows.append(info)
    out["pool"] = rows
    out["occurrences"], out["unique"] = sum(ref.values()), len(ref)
    save("pretok_scaling", out)


def exp_pretok_repeat(b: Book, args):
    """Whole read + pretokenize (the 01/02 way) vs 03's chunked read, interleaved: is one of them faster?"""
    rounds, ref = [], None
    for _ in range(args.rounds):
        with Load() as load:
            w0 = time.perf_counter()
            counts = pretok_counts_serial(b.sb, read_text(VALID))
            whole = {"wall": round(time.perf_counter() - w0, 3)}
        whole["load"] = load.summary()
        ref = ref or counts
        _, chunked = run_phased(b.v03, "train_bpe", str(VALID), 257)
        rounds.append({"whole": whole, "chunked": {"wall": chunked["phases"]["Pretokenizing"]["wall"], "load": chunked["load"]}})
        print(rounds[-1], flush=True)
    save("pretok_repeat", {"input": "storybot/tiny_stories_valid.txt", "rounds": rounds,
                           "same_counts_every_round": dict(counts) == dict(ref)})


POOL_PROBE = r'''
import json, sys, time
from multiprocessing import Pool
from collections import defaultdict   # the imports of ch04/04_bpe_parallel.py: every spawned worker repeats them
import regex as re
from tqdm import tqdm
import pickle


def noop(x):
    return x


if __name__ == "__main__":
    out = {}
    for n in [int(a) for a in sys.argv[2:]]:
        runs = []
        for _ in range(int(sys.argv[1])):
            t0 = time.perf_counter()
            with Pool(n) as pool:
                pool.map(noop, range(n), chunksize=1)    # every worker has started and done one task
            runs.append(round(time.perf_counter() - t0, 3))
        out[n] = runs
    print(json.dumps(out))
'''


def exp_pool_startup(b: Book, args):
    probe = b.tmp / "pool_probe.py"
    probe.write_text(POOL_PROBE, encoding="utf-8")
    with Load() as load:
        res = subprocess.run([sys.executable, str(probe), str(args.rounds), "1", "8", "16"], capture_output=True,
                             text=True, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    runs = {int(k): v for k, v in json.loads(res.stdout).items()}
    save("pool_startup", {"what": "Pool(n): start, one trivial task per worker, shutdown; workers import what "
                                  "ch04/04_bpe_parallel.py imports (regex, tqdm, ...)", "start_method": "spawn",
                          "runs": {str(n): r for n, r in runs.items()}, "load": load.summary()})


def exp_encode_small(b: Book, args):
    path = b.slice_file(SLICE30_LIMIT)
    text = read_text(path)
    res = {}
    a, res["ch01"] = timed(b.cb.BPETokenizer(b.story_rules).encode, text)
    c06, res["06"] = timed(b.v06.BPETokenizer(b.story_rules).encode, text)
    c, res["storybot"] = timed(b.sb.BPETokenizer(b.story_rules).encode, text)
    d, res["memo"] = timed(b.ext.MemoTokenizer(b.story_rules).encode, text)
    word = " accomplishment"
    old, new = b.cb.BPETokenizer(b.story_rules), b.v06.BPETokenizer(b.story_rules)
    best_old = best_new = 1e9
    for _ in range(5):                       # wall time, best of 5; constructors outside the timed region
        t0 = time.perf_counter()
        x = old._encode_text(word)
        t1 = time.perf_counter()
        y = new._encode_text(word)
        t2 = time.perf_counter()
        best_old, best_new = min(best_old, t1 - t0), min(best_new, t2 - t1)
    save("encode_small", {"input": "tiny_stories_valid.txt first 29,642 bytes", "bytes": path.stat().st_size,
                          "merges": len(b.story_rules), "runs": res, "tokens": len(a), "identical": a == c06 == c == d,
                          "accomplishment": {"ids": x, "same": x == y, "ch01_ms": round(best_old * 1e3, 2),
                                             "06_ms": round(best_new * 1e3, 3)}})


def exp_encode_valid(b: Book, args):
    import numpy as np

    text = read_text(VALID)
    official = np.fromfile(VALID_BIN, dtype=np.uint16).tolist()
    a, t06 = timed(b.v06.BPETokenizer(b.story_rules).encode, text)
    m, tmemo = timed(b.ext.MemoTokenizer(b.story_rules).encode, text)
    save("encode_valid", {"input": "storybot/tiny_stories_valid.txt", "bytes": VALID.stat().st_size, "06": t06,
                          "memo": tmemo, "tokens": len(a),
                          "identical_to_official_bin": {"06": a == official, "memo": m == official}})


def exp_encode_file_scaling(b: Book, args):
    tok = b.sb.BPETokenizer(b.story_rules)
    ref = hashlib.sha256(VALID_BIN.read_bytes()).hexdigest()
    rows = []
    for n in [16, 8, 4, 2, 1]:
        outp, cache = b.tmp / f"valid_n{n}.bin", b.tmp / f"bpe_cache_n{n}"
        with Load() as load:
            w0 = time.perf_counter()
            total = tok.encode_file(str(VALID), str(outp), num_processes=n, cache_dir=str(cache))
            wall = time.perf_counter() - w0
        rows.append({"n": n, "wall": round(wall, 3), "tokens": total,
                     "identical_to_official": hashlib.sha256(outp.read_bytes()).hexdigest() == ref,
                     "cache_dir_removed": not cache.exists(), "load": load.summary()})
        outp.unlink()
    save("encode_file_scaling", {"input": "storybot/tiny_stories_valid.txt", "rows": rows, "sha256_official": ref})


MEM_CHILD = r'''
import sys, psutil
sys.path.insert(0, sys.argv[3]); sys.path.insert(0, sys.argv[4])
from _ch04_load import load_defs
mode, path = sys.argv[1], sys.argv[2]
if mode == "whole":            # ch04/02: open(path).read() then train_bpe(text, 257) = pretokenize only
    m = load_defs("ch04/02_bpe_cache.py")
    text = open(path, encoding="utf-8").read()
    m.train_bpe(text, 257)
    info = {"max_codepoint": max(map(ord, set(text))), "str_bytes": sys.getsizeof(text), "chars": len(text)}
elif mode == "chunked":        # ch04/03: 64 chunks, binary reads, generator pretokenize
    m = load_defs("ch04/03_bpe_chunk.py")
    m.train_bpe(path, 257)
    info = {}
elif mode == "findall_list":   # control (not the book's code): whole text, all pretokens kept in one list
    m = load_defs("ch04/01_bpe_optimize.py")
    text = open(path, encoding="utf-8").read()
    toks = [t for piece in text.split("<|endoftext|>") for t in m.pretokenize(piece)]
    info = {"pretokens": len(toks)}
elif mode == "naive_idslist":  # codebot/tokenizer.py train_bpe's first half: one list of ids per occurrence
    text = open(path, encoding="utf-8").read()
    from codebot.tokenizer import pretokenize
    ids_list = []
    for piece in text.split("<|endoftext|>"):
        for pretoken in pretokenize(piece):
            ids_list.append(list(pretoken.encode("utf-8")))
    info = {"occurrences": len(ids_list)}
print(psutil.Process().memory_info().peak_wset, info)
'''


def exp_memory(b: Book, args):
    rows = []
    for path, modes in [(VALID, ["naive_idslist", "findall_list", "whole", "chunked"]),
                        (need(args.train_prefix, "train_prefix"), ["whole", "chunked"])]:
        for mode in modes:
            w0 = time.perf_counter()
            out = subprocess.check_output([sys.executable, "-c", MEM_CHILD, mode, str(path), str(ROOT / "verify"),
                                           str(DLFS6)], text=True, env={**os.environ, "PYTHONUTF8": "1"})
            peak, info = out.strip().split(" ", 1)
            rows.append({"file": path.name, "bytes": path.stat().st_size, "mode": mode,
                         "peak_mib": round(int(peak) / 2**20, 1), "info": info, "wall": round(time.perf_counter() - w0, 1)})
    save("memory", {"rows": rows, "method": "psutil peak_wset of a fresh process (Windows peak working set)"})


def exp_heaps(b: Book, args):
    rows = heaps_rows(VALID, VALID_MARKS) + heaps_rows(need(args.train_prefix, "train_prefix"), PREFIX_MARKS, "train_prefix.txt")
    save("heaps", {"rows": rows})


def exp_heaps_full(b: Book, args):
    """Occurrences and distinct pretokens of the whole training file, which 4.1.1 extrapolates from the first 100 MiB.
    Counted the 04 way (steps 1-3: 64 chunks, 8 processes); chunked counts equal whole-file counts
    (test_parallel_pretokenize_equals_serial)."""
    from multiprocessing import Pool

    path = need(args.train_full, "train_full")
    bounds = b.sb.find_chunk_boundaries(str(path), 64)
    tasks = [(str(path), s, e, EOT) for s, e in zip(bounds[:-1], bounds[1:])]
    total = defaultdict(int)
    with Load() as load:
        w0 = time.perf_counter()
        with Pool(processes=8) as pool:
            for part in pool.imap(b.sb.pretoken_chunk, tasks):
                for k, v in part.items():
                    total[k] += v
        wall = time.perf_counter() - w0
    top = sorted(total.values(), reverse=True)
    save("heaps_full", {"input": "storybot/tiny_stories_train.txt (2,227,753,162 bytes)", "bytes": path.stat().st_size,
                        "occurrences": sum(top), "unique": len(top),
                        "top10_share": round(sum(top[:10]) / sum(top), 4),
                        "top1000_share": round(sum(top[:1000]) / sum(top), 4),
                        "wall": round(wall, 3), "load": load.summary()})


def exp_pep393(b: Book, args):
    save("pep393", {"valid": pep393_stats(read_text(VALID)),
                    "train_prefix": pep393_stats(read_text(need(args.train_prefix, "train_prefix")))})


def exp_train_prefix_encode(b: Book, args):
    prefix, official = need(args.train_prefix, "train_prefix"), need(args.train_bin_prefix, "train_bin_prefix")
    outp, cache = b.tmp / "train_prefix.bin", b.tmp / "bpe_cache_prefix"
    with Load() as load:
        w0 = time.perf_counter()
        total = b.sb.BPETokenizer(b.story_rules).encode_file(str(prefix), str(outp), num_processes=16, cache_dir=str(cache))
        wall = time.perf_counter() - w0
    save("train_prefix_encode", {"input": "tiny_stories_train.txt first 104,856,833 bytes (HTTP Range, ends before an EOT)",
                                 "tokens": total, "wall": round(wall, 3), "processes": 16, "load": load.summary(),
                                 "identical_to_official_train_bin_prefix": outp.read_bytes() == official.read_bytes(),
                                 "official_prefix_bytes": official.stat().st_size})


def exp_train_full(b: Book, args):
    """ch04/04 on the whole training file. storybot's train_bpe is 04's train_bpe statement for statement (only the
    default num_chunks differs: test_bpe_book_is_verbatim), and its pretoken_chunk can be imported by spawned
    workers, so it runs here with 04's arguments: 8 processes, 64 chunks."""
    path = need(args.train_full, "train_full")
    with MemPeak() as mem:
        rules, t = run_phased(b.sb, "train_bpe", str(path), 10000, num_processes=8, num_chunks=64)
    got, official = list(rules.items()), list(b.story_rules.items())
    first_diff = next((i for i, (x, y) in enumerate(zip(got, official)) if x != y), None)
    save("train_full", {"input": "storybot/tiny_stories_train.txt (2,227,753,162 bytes)", "bytes": path.stat().st_size,
                        "vocab": 10000, "num_processes": 8, "num_chunks": 64, "run": t, "peak_rss_mib": mem.peak_mib,
                        "peak_rss_method": MemPeak.METHOD, "merges": len(rules), "rules_sha256": rules_digest(rules),
                        "equals_storybot_merge_rules_pkl": got == official, "first_difference": first_diff,
                        "same_pairs_as_set": len(set(rules) & set(b.story_rules))})


def exp_encode_full(b: Book, args):
    """ch04/07's __main__ (its first call) on the whole training file: encode_file with 07's arguments, 8 processes and
    64 chunks. storybot's BPETokenizer is 07's method for method (only encode_file's default num_processes differs),
    and its bound _encode_chunk can be unpickled by spawned workers. The output is compared by size and sha256 with
    the official tiny_stories_train.bin as published (OFFICIAL_TRAIN_BIN), then deleted."""
    path = need(args.train_full, "train_full")
    outp, cache = b.tmp / "train_full.bin", b.tmp / "bpe_cache_full"
    tok = b.sb.BPETokenizer(b.story_rules)
    with MemPeak() as mem:
        total, t = timed(tok.encode_file, str(path), str(outp), num_processes=8, num_chunks=64, cache_dir=str(cache))
    size, digest = outp.stat().st_size, sha256_file(outp)
    outp.unlink()
    save("encode_full", {"input": "storybot/tiny_stories_train.txt (2,227,753,162 bytes)", "num_processes": 8,
                         "num_chunks": 64, "run": t, "peak_rss_mib": mem.peak_mib, "peak_rss_method": MemPeak.METHOD,
                         "tokens": total, "out_bytes": size, "out_sha256": digest, "official": OFFICIAL_TRAIN_BIN,
                         "identical_to_official_train_bin": (size, digest) == (OFFICIAL_TRAIN_BIN["bytes"], OFFICIAL_TRAIN_BIN["sha256"]),
                         "cache_dir_removed": not cache.exists()})


def need(path: Path | None, name: str) -> Path:
    """An input that is not in the repository: check it is exactly the file the records were made with."""
    if path is None:
        sys.exit(f"this experiment needs --{name.replace('_', '-')} ({INPUTS[name]['what']}; {INPUTS[name]['url']})")
    path = Path(path)
    digest = sha256_file(path)
    if digest != INPUTS[name]["sha256"]:
        sys.exit(f"{path} is not the recorded {name} (sha256 {digest})")
    return path


EXPERIMENTS = {name[4:]: fn for name, fn in globals().items() if name.startswith("exp_")}


# ------------------------------------------------------------------ bench.json
def load_raw(name: str) -> dict:
    path = RAW / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing raw record {path.relative_to(ROOT)}: run `verify/_ch04_bench.py run {name}`")
    return json.loads(path.read_text(encoding="utf-8"))


def _best(runs: list[dict], key: str = "cpu") -> float:
    return min(r[key] for r in runs)


def _mean_load(runs: list[dict]) -> float:
    return round(sum(r["load"]["mean"] for r in runs) / len(runs), 1)


def _row(key, name, runs, identical=None, reference=False, sub=None, ext=False, **extra):
    row = {"key": key, "name": name}
    if sub:
        row["sub"] = sub
    row["seconds"] = _best(runs)
    if ext:
        row["ext"] = True
    row["runs"] = len(runs)
    if reference:
        row["reference"] = True          # the output the other rows of this group were compared with
    else:
        row["identical"] = identical
    row.update(extra)
    return row


EXT = "本教程，非原书"


def assemble() -> dict:
    """bench.json, rebuilt from the raw records in bench_runs/ (pure function of those files)."""
    small, codes, naive = load_raw("train_small"), load_raw("train_codes"), load_raw("naive_codes")
    loops, enc_small, enc_valid = load_raw("loops_valid10k"), load_raw("encode_small"), load_raw("encode_valid")
    v10k, pretok, efile = load_raw("train_valid10k"), load_raw("pretok_scaling"), load_raw("encode_file_scaling")
    memory, heaps, prefix = load_raw("memory"), load_raw("heaps"), load_raw("train_prefix_encode")
    repeat, pretok_rep, pool, shrink = load_raw("repeat_small"), load_raw("pretok_repeat"), load_raw("pool_startup"), load_raw("shrink")
    full, enc_full, heaps_full = load_raw("train_full"), load_raw("encode_full"), load_raw("heaps_full")

    r, same_small = small["runs"], small["identical_to_naive"]
    c, same_codes = codes["runs"], codes["equals_codebot_pkl"]
    er = enc_small["runs"]
    races = [
        {"id": "train_small", "short": "训练 · 500 KB", "title": "训练：TinyStories 验证集前 499,577 字节，词表 1,000",
         "input": "TinyStories 验证集的前 499,577 字节，词表 1,000（学 743 条规则）",
         "metric": "train_bpe 一次调用的 CPU 秒（单进程；跑了几次就取最快的一次，次数见数据表）",
         "reference": "第 1 章的朴素版（其余版本学到的规则都与它逐项比较）",
         "load": _mean_load(sum(r.values(), [])),
         "rows": [_row("naive", "第 1 章：朴素", r["naive"], reference=True),
                  _row("dedup", "4.1 去重", r["dedup"], same_small["dedup"]),
                  _row("cache", "4.2 增量更新", r["cache"], same_small["cache"]),
                  _row("heap", "扩展：堆", r["heap"], same_small["heap"], sub=EXT, ext=True)],
         "extra": f"（4.3 分块版 {_best(r['chunk']):.2f} 秒，与增量版是同一个合并循环；学到的规则"
                  f"{'同样与朴素版逐项相同' if same_small['chunk'] else '与朴素版不同'}）"},
        {"id": "train_codes", "short": "训练 · TinyCodes", "title": "训练：整份 TinyCodes（6.5 MB），词表 1,000",
         "input": "整份 TinyCodes（6,487,033 字节，git 中的 LF 版本），词表 1,000",
         "metric": "train_bpe 一次调用的 CPU 秒（单进程；跑了几次就取最快的一次，次数见数据表）",
         "reference": "仓库里的 codebot/merge_rules.pkl（每个版本学到的 743 条规则都与它逐项比较）",
         "load": _mean_load(sum(c.values(), []) + [naive["naive"]]),
         "rows": [_row("naive", "第 1 章：朴素", [naive["naive"]], naive["equals_codebot_merge_rules_pkl"],
                       wall=naive["naive"]["wall"]),
                  _row("dedup", "4.1 去重", c["dedup"], same_codes["dedup"]),
                  _row("cache", "4.2 增量更新", c["cache"], same_codes["cache"]),
                  _row("heap", "扩展：堆", c["heap"], same_codes["heap"], sub=EXT, ext=True)],
         "extra": f"（4.3 分块版 {_best(c['chunk']):.2f} 秒，学到的规则"
                  f"{'同样与 merge_rules.pkl 逐项相同' if same_codes['chunk'] else '与 merge_rules.pkl 不同'}）"},
        {"id": "loops_valid", "short": "合并循环 · 词表 1 万", "title": "合并循环：整份 TinyStories 验证集，词表 10,000",
         "input": "整份 TinyStories 验证集（22,502,601 字节，13,111 个不同的预分词），词表 10,000（学 9,743 条规则）",
         "metric": "合并循环的 CPU 秒（不含预分词，各跑一次）",
         "reference": "原书 ch04/02 的合并循环（堆版本学到的 9,743 条规则与它逐项比较）",
         "load": round((loops["book_load"]["mean"] + loops["heap_loop"]["load"]["mean"]) / 2, 1),
         "rows": [_row("cache", "4.2 增量更新", [loops["book_loop"]], reference=True, sub="max() 全表扫描"),
                  _row("heap", "扩展：堆", [loops["heap_loop"]], loops["identical"], sub=EXT, ext=True)]},
        {"id": "encode_small", "short": "编码 · 30 KB", "title": "编码：TinyStories 验证集前 29,642 字节，9,743 条规则",
         "input": "TinyStories 验证集的前 29,642 字节（7,294 个 token），StoryBot 的 9,743 条规则",
         "metric": "encode 一次调用的 CPU 秒（单进程，各跑一次）",
         "reference": "几个编码器的输出互相逐个比较（第 1 章、06、storybot、缓存版）",
         "load": _mean_load(list(er.values())),
         "rows": [_row("ch01", "第 1 章：逐条试规则", [er["ch01"]], enc_small["identical"]),
                  _row("06", "4.6 按优先级合并", [er["06"]], enc_small["identical"]),
                  _row("memo", "扩展：缓存", [er["memo"]], enc_small["identical"], sub=EXT, ext=True)]},
        {"id": "encode_valid", "short": "编码 · 整份验证集", "title": "编码：整份 TinyStories 验证集（5,465,873 个 token）",
         "input": "整份 TinyStories 验证集（22,502,601 字节），StoryBot 的 9,743 条规则",
         "metric": "encode 一次调用的 CPU 秒（单进程，各跑一次）",
         "reference": "官方的 tiny_stories_valid.bin（两个版本的输出都与它逐个比较）",
         "load": _mean_load([enc_valid["06"], enc_valid["memo"]]),
         "rows": [_row("06", "4.6 按优先级合并", [enc_valid["06"]], enc_valid["identical_to_official_bin"]["06"]),
                  _row("memo", "扩展：缓存", [enc_valid["memo"]], enc_valid["identical_to_official_bin"]["memo"],
                       sub=EXT, ext=True)]},
    ]

    best_pool = {}
    for row in pretok["pool"]:
        best_pool[row["n"]] = min(best_pool.get(row["n"], 1e9), row["wall"])
    runs_pool = defaultdict(int)
    for row in pretok["pool"]:
        runs_pool[row["n"]] += 1
    rounds = repeat["rounds"]
    return {
        "env": {k: v for k, v in prefix["env"].items() if k != "note"},
        "provenance": {
            "script": "verify/_ch04_bench.py",
            "raw_records": "web/data/ch04/bench_runs/<experiment>.json",
            "run": "uv run --with psutil python verify/_ch04_bench.py run <experiment>",
            "assemble": "uv run python verify/_ch04_bench.py assemble",
            "first_batch": FIRST_BATCH,
            "later_records": "each carries its own commits, command and date in its 'provenance' and 'env' fields",
            "inputs_not_in_repo": INPUTS,
            "cpu_timer": "time.process_time() of the measuring process (on Windows it ticks in steps of 1/64 s)",
            "wall_timer": "time.perf_counter()",
            "load": "psutil.cpu_percent(interval=0.5) of the whole machine (16 logical CPUs) in a background thread; "
                    "mean over the run",
            "repeats": "a race row's seconds is the minimum over its runs; 'runs' says how many there were",
            "machine": "during every run the same machine was also training models and running tests for other chapters",
        },
        "races": races,
        "profile": {"input": "storybot/tiny_stories_valid.txt", "vocab": 10000, "merges": v10k["merges"],
                    "unique": v10k["unique_pretokens"], "affected_mean": v10k["affected_mean"],
                    "affected_median": v10k["affected_median"], "max_share": v10k["max_share"],
                    "max_share_timer": "time.perf_counter() around max() and around the update, summed over 9,743 steps",
                    "loop_cpu": v10k["book_02"]["phases"]["Training BPE"]["cpu"],
                    "loop_wall": v10k["book_02"]["phases"]["Training BPE"]["wall"],
                    "pretok_wall": v10k["book_02"]["phases"]["Pretokenizing"]["wall"], "load": v10k["book_02"]["load"]["mean"],
                    "identical": v10k["identical"]},
        "pretok": {"input": "storybot/tiny_stories_valid.txt", "bytes": pretok["bytes"], "chunks": 64,
                   "serial_wall": pretok["serial_whole"]["wall"],
                   "serial_chunked_wall": pretok["serial_chunked_03"]["phases"]["Pretokenizing"]["wall"],
                   "pool": [{"n": n, "wall": w, "runs": runs_pool[n]} for n, w in sorted(best_pool.items())],
                   "pool_startup_in_run": [{"n": row["n"], "startup": row["startup"]} for row in pretok["pool"]],
                   "identical": all(row["identical"] for row in pretok["pool"]),
                   "merge_loop_wall": loops["book_loop"]["wall"], "heap_loop_wall": loops["heap_loop"]["wall"],
                   "load": pretok["serial_whole"]["load"]["mean"]},
        "pretok_repeat": {"whole": [r["whole"]["wall"] for r in pretok_rep["rounds"]],
                          "chunked": [r["chunked"]["wall"] for r in pretok_rep["rounds"]],
                          "load": [r["whole"]["load"]["mean"] for r in pretok_rep["rounds"]],
                          "same_counts_every_round": pretok_rep["same_counts_every_round"], "date": pretok_rep["env"]["date"]},
        "pool_startup": {"runs": pool["runs"], "best": {n: min(v) for n, v in pool["runs"].items()},
                         "load": pool["load"]["mean"], "start_method": pool["start_method"], "date": pool["env"]["date"]},
        "repeat_small": {"cpu": {k: [rd[k]["cpu"] for rd in rounds] for k in rounds[0]},
                         "load": [round(sum(rd[k]["load"]["mean"] for k in rd) / len(rd), 1) for rd in rounds],
                         "rules_sha256": repeat["rules_sha256"], "identical": repeat["identical"],
                         "date": repeat["env"]["date"]},
        "shrink": {k: v for k, v in shrink.items() if k not in ("env", "provenance")},
        "encode_file": [{"n": row["n"], "wall": row["wall"], "identical": row["identical_to_official"],
                         "cache_dir_removed": row["cache_dir_removed"]} for row in efile["rows"]],
        "memory": memory["rows"],
        "heaps": heaps["rows"],
        "train_prefix_encode": {k: prefix[k] for k in ("tokens", "wall", "processes", "identical_to_official_train_bin_prefix",
                                                       "official_prefix_bytes")} | {"load": prefix["load"]["mean"]},
        "train_full": {"input": full["input"], "vocab": full["vocab"], "num_processes": full["num_processes"],
                       "num_chunks": full["num_chunks"], "wall": full["run"]["wall"],
                       "pretok_wall": full["run"]["phases"]["Pretokenizing"]["wall"],
                       "merge_loop_wall": full["run"]["phases"]["Training BPE"]["wall"],
                       "peak_rss_mib": full["peak_rss_mib"], "load": full["run"]["load"]["mean"], "merges": full["merges"],
                       "rules_sha256": full["rules_sha256"],
                       "identical_to_official_merge_rules": full["equals_storybot_merge_rules_pkl"],
                       "date": full["env"]["date"]},
        "encode_full": {"input": enc_full["input"], "num_processes": enc_full["num_processes"],
                        "num_chunks": enc_full["num_chunks"], "wall": enc_full["run"]["wall"],
                        "peak_rss_mib": enc_full["peak_rss_mib"], "load": enc_full["run"]["load"]["mean"],
                        "tokens": enc_full["tokens"], "out_bytes": enc_full["out_bytes"],
                        "identical_to_official_train_bin": enc_full["identical_to_official_train_bin"],
                        "cache_dir_removed": enc_full["cache_dir_removed"], "date": enc_full["env"]["date"]},
        "heaps_full": {k: heaps_full[k] for k in ("input", "bytes", "occurrences", "unique", "top10_share", "top1000_share")},
        "scanned": load_raw("scanned_valid10k"),
        "pep393": load_raw("pep393"),
        "accomplishment_ms": {"ch01": enc_small["accomplishment"]["ch01_ms"], "06": enc_small["accomplishment"]["06_ms"]},
        "accomplishment_timer": "wall time (time.perf_counter), best of 5",
    }


def write_bench() -> None:
    DEST.write_text(json.dumps(assemble(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {DEST.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run experiments and write their raw records")
    run.add_argument("experiments", nargs="+", choices=sorted(EXPERIMENTS))
    run.add_argument("--rounds", type=int, default=3, help="repeat_small / pretok_repeat / pool_startup")
    run.add_argument("--train-prefix", type=Path)
    run.add_argument("--train-bin-prefix", type=Path)
    run.add_argument("--train-full", type=Path)
    sub.add_parser("assemble", help="rebuild web/data/ch04/bench.json from the raw records")
    args = ap.parse_args()
    if args.cmd == "assemble":
        write_bench()
        return
    book = Book()
    try:
        for name in args.experiments:
            print(f"=== {name}", flush=True)
            EXPERIMENTS[name](book, args)
    finally:
        shutil.rmtree(book.tmp, ignore_errors=True)        # slices, probes and outputs written along the way


if __name__ == "__main__":
    main()
