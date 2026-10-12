"""Chapter 7 measurements -> web/data/ch07/bench.json (minutes of CPU; run by hand, not by pytest).

    uv run python verify/_ch07_record.py                       # every section except the optional HF trainer
    uv run --with tokenizers python verify/_ch07_record.py     # also time Hugging Face's Rust BPE trainer
    uv run python verify/_ch07_record.py valid290 heap8        # only these sections; the rest of the file is kept

Sections (bench.json keys):
  train     train / hf_tokenizers  storybot.train_bpe on 0.25-8 MB: max() share, pairs per step, cProfile, shrunk-run
                                   agreement with the official merges; Hugging Face's trainer on the 8 MB sample
  encode    encode_2mb   the book's encode vs whole-pretoken lookup vs tiktoken, interleaved rounds
  valid290  valid_290mb  the pairs the book's max() would scan, on prefixes of the whole validation split and on
                         all of it (49,743 merges), counted with chapter 4's heap (same table, same merges)
  heap8     heap_8mb     the book's train_bpe vs chapter 4's heap (pure Python, one process) on the 8 MB sample
  maxkey    max_key      max() over one step's pair table with the book's lambda key and with C-level keys

All inputs are OWT documents decoded from the official webbot/owt_valid.bin (see _ch07_common.py), so the
numbers can be reproduced without downloading the 12 GB corpus. Timings are wall-clock on whatever machine runs
this; every section records the date, the machine and its load so the page can state the conditions next to every
number. Every row whose output is compared (merges, token ids) also records whether it was identical.
"""
from __future__ import annotations

import builtins
import cProfile
import datetime
import io
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch07_common import (EOT, OUT, OWT_TRAIN_TXT_BYTES, SerialPool, book_modules, ch04_ext,  # noqa: E402
                          heap_merge_sizes, iter_valid_docs, official_rules, official_tokenizer, owt_text,
                          webbot_tiktoken)

SIZES_MB = [0.25, 0.5, 1, 2, 4, 8]
VALID_PREFIX_MB = [8, 16, 32, 64, 128]          # plus the whole split; 8 MB repeats the book's own 8 MB row
SECTIONS = ("train", "encode", "valid290", "heap8", "maxkey")


def cpu_load() -> int | None:
    """Current CPU load in percent (Windows only; provenance, not a measurement)."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).LoadPercentage"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def machine() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "logical_cpus": os.cpu_count()}
    try:
        info["cpu"] = subprocess.run(["powershell", "-NoProfile", "-Command",
                                      "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True,
                                     timeout=30).stdout.strip() if os.name == "nt" else platform.processor()
        if os.name == "nt":
            info["cpu_load_percent_at_start"] = cpu_load()
    except Exception as e:  # noqa: BLE001 - provenance only
        info["cpu"] = f"unknown ({e})"
    import numpy, regex, tiktoken
    info.update(numpy=numpy.__version__, regex=regex.__version__, tiktoken=tiktoken.__version__)
    # A fixed pure-Python loop, timed at the start and at the end: it shows how busy the machine was.
    t = time.perf_counter()
    s = 0
    for i in range(10_000_000):
        s += i
    info["python_10M_loop_s"] = round(time.perf_counter() - t, 3)
    return info


def write_sample(text: str, tmp: Path, name: str) -> Path:
    path = tmp / name
    path.write_bytes(text.encode("utf-8"))          # bytes: no newline translation on Windows
    return path


def items(rules: dict) -> list:
    return list(rules.items())


def first_difference(a: list, b: list) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def vocab_of(rules: dict) -> dict:
    v = {i: bytes([i]) for i in range(256)}
    for (a, b), n in rules.items():
        v[n] = v[a] + v[b]
    return v


def record_training(sb, samples: dict) -> tuple[dict, dict]:
    """Merge-loop profile of storybot.train_bpe (vocab 2,000) on OWT samples, plus shrunk-run agreement."""
    official = items(official_rules())
    vof = vocab_of(dict(official))
    sb.Pool = SerialPool          # identical results (test_ch07.py checks); no process start-up in the timings
    scaling, shrunk, rules_8mb = [], [], None
    for mb in SIZES_MB:
        path = samples[mb]
        sizes, spent = [], [0.0]

        def timed_max(it, *a, **kw):
            sizes.append(len(it))
            t0 = time.perf_counter()
            r = builtins.max(it, *a, **kw)
            spent[0] += time.perf_counter() - t0
            return r

        t0 = time.perf_counter()
        bounds = sb.find_chunk_boundaries(str(path), 8)
        counts = {}
        for s, e in zip(bounds[:-1], bounds[1:]):
            for p, c in sb.pretoken_chunk((str(path), s, e, EOT)).items():
                counts[p] = counts.get(p, 0) + c
        pretok_s = time.perf_counter() - t0
        sb.max = timed_max
        try:
            t0 = time.perf_counter()
            rules = sb.train_bpe(str(path), 2000, num_chunks=8)
            wall = time.perf_counter() - t0
        finally:
            del sb.max
        mine = items(rules)
        vm = vocab_of(rules)
        n = len(rules)
        first_n = {vof[256 + i] for i in range(n)}
        rec = {"mb": mb, "bytes": path.stat().st_size, "pretokens": sum(counts.values()), "distinct_pretokens": len(counts),
               "merges": n, "wall_s": round(wall, 2), "pretok_s": round(pretok_s, 2), "max_s": round(spent[0], 2),
               "max_share": round(spent[0] / wall, 3), "pairs_scanned": sum(sizes), "pairs_first_step": sizes[0],
               "pairs_last_step": sizes[-1], "ns_per_pair": round(spent[0] / sum(sizes) * 1e9),
               "pairs_every_100_steps": sizes[::100]}
        scaling.append(rec)
        shrunk.append({"mb": mb, "bytes": rec["bytes"], "identical_prefix": first_difference(mine, official),
                       "first_difference": [vm[256 + first_difference(mine, official)].decode("utf-8", "replace"),
                                            vof[256 + first_difference(mine, official)].decode("utf-8", "replace")],
                       "learned_in_official_first_n": sum(vm[i] in first_n for i in rules.values()), "n": n})
        print(f"train {mb} MB: wall {wall:.1f}s, max() {spent[0] / wall:.0%}, prefix {shrunk[-1]['identical_prefix']}",
              flush=True)
        if mb == 8:
            rules_8mb = rules

    # cProfile view of the same loop on the 1 MB sample (the profiler inflates per-call costs; shares are indicative)
    pr = cProfile.Profile()
    pr.enable()
    sb.train_bpe(str(samples[1]), 2000, num_chunks=8)
    pr.disable()
    st = pstats.Stats(pr, stream=io.StringIO())
    total = st.total_tt
    rows = []
    for (file, line, func), (cc, nc, tt, ct, _) in sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:6]:
        name = func if file == "~" else f"{Path(file).name}:{line}({func})"
        rows.append({"function": name, "ncalls": nc, "tottime_s": round(tt, 2), "cumtime_s": round(ct, 2)})
    return {"scaling": scaling, "shrunk": shrunk, "cprofile_1mb": {"total_s": round(total, 2), "top": rows}}, rules_8mb


ENCODE_ROUNDS = 5


def record_encode(sb, tok, text: str, rounds: int = ENCODE_ROUNDS) -> dict:
    """Same text, same 50K merge table: the book's pure-Python encoder vs tiktoken (Rust).

    Every method is timed once per round, in the same order, for `rounds` rounds; a row's time is its median.
    The book's row is the reference (identical_ids None); every other row except pretokenize-only (which yields
    no ids) is compared with it id by id."""
    import regex as re
    enc = webbot_tiktoken(tok)
    nbytes = len(text.encode("utf-8"))
    split = "(" + re.escape(EOT) + ")"
    ranks = {tok.id_to_bytes[i]: i for i in range(tok.end_token_id)}

    def lookup_then_bpe(t):          # tiktoken's fast path, written in Python
        out = []
        for part in re.split(split, t):
            if part == EOT:
                out.append(tok.end_token_id)
                continue
            for p in sb.pretokenize(part):
                i = ranks.get(p.encode("utf-8"))
                out.extend([i] if i is not None else tok._encode_text(p))
        return out

    def pretokenize_only(t):         # the book's generator pretokenize (regex.finditer, one yield per match)
        return sum(1 for part in re.split(split, t) for _ in sb.pretokenize(part))

    docs = text.split(EOT)

    def batch(threads):
        out = []
        for i, ids in enumerate(enc.encode_ordinary_batch(docs, num_threads=threads)):
            out.extend(ids)
            if i < len(docs) - 1:
                out.append(tok.end_token_id)
        return out

    methods = [
        ("book", "原书 encode（纯 Python）", lambda: tok.encode(text)),
        ("lookup", "原书 + 整词查表（纯 Python）", lambda: lookup_then_bpe(text)),
        ("pretokenize", "只做预分词（原书的 pretokenize）", lambda: pretokenize_only(text)),
        ("tiktoken1", "tiktoken，1 个线程", lambda: enc.encode(text, allowed_special="all")),
        ("tiktoken8", "tiktoken，8 个线程", lambda: batch(8)),
    ]
    times = {k: [] for k, _, _ in methods}
    outputs, loads = {}, []
    for r in range(rounds):
        loads.append(cpu_load())
        for key, _, fn in methods:
            t0 = time.perf_counter()
            out = fn()
            times[key].append(time.perf_counter() - t0)
            if r == 0:
                outputs[key] = out
        print(f"encode round {r + 1}/{rounds}: " + ", ".join(f"{k} {v[-1]:.3f}s" for k, v in times.items()),
              flush=True)
    ref = outputs["book"]
    med = {k: statistics.median(v) for k, v in times.items()}
    rows = []
    for key, label, _ in methods:
        same = None if key in ("book", "pretokenize") else outputs[key] == ref
        per_round = [b / t for b, t in zip(times["book"], times[key])]
        rows.append({"key": key, "label": label, "seconds": round(med[key], 3),
                     "seconds_all": [round(t, 3) for t in times[key]],
                     "mb_per_s": round(nbytes / med[key] / 1e6, 3), "speedup": round(med["book"] / med[key], 1),
                     "speedup_min": round(min(per_round), 1), "speedup_max": round(max(per_round), 1),
                     "identical_ids": same})
    return {"bytes": nbytes, "tokens": len(ref), "docs": len(docs), "rounds": rounds, "timing": "median of rounds",
            "cpu_load_percent_per_round": loads, "rows": rows,
            "owt_train_hours": {k: round(OWT_TRAIN_TXT_BYTES / (nbytes / t) / 3600, 2) for k, t in med.items()}}


def record_valid(sb, tok) -> dict:
    """How many pairs the book's max() would scan on the whole validation split (290 MB, 49,743 merges).

    The pretoken counts are those of the book's train_bpe (documents split at EOT, the book's pretokenize).
    The merge loop is chapter 4's heap (_ch07_common.heap_merge_sizes): it learns the book's merges in the book's
    order and keeps the book's pair table, so len(table) at step t is what the book's max() scans at step t
    (test_ch07.py::test_heap_loop_scans_what_the_book_scans). Prefixes are cut like owt_text(): at the first
    EOT once that many bytes precede it."""
    official = list(official_rules())
    eot_len = len(EOT.encode("utf-8"))
    counts: dict = defaultdict(int)
    targets = [int(mb * 1_000_000) for mb in VALID_PREFIX_MB]
    prefixes, pos, n_docs = [], 0, 0
    t_pretok, t_heap = 0.0, 0.0

    def run(num_merges):
        nonlocal t_heap
        ids_counts = {tuple(p.encode("utf-8")): c for p, c in counts.items()}
        t0 = time.perf_counter()
        merges, sizes = heap_merge_sizes(ids_counts, num_merges)
        t_heap += time.perf_counter() - t0
        k = next((i for i, (a, b) in enumerate(zip(merges, official)) if a != b), len(merges))
        return merges, sizes, k

    t0 = time.perf_counter()
    for _, data in iter_valid_docs(tok):
        for p in sb.pretokenize(data.decode("utf-8")):
            counts[p] += 1
        n_docs += 1
        pos += len(data)                       # the byte offset of the EOT that ends this document
        while targets and pos >= targets[0]:
            t_pretok += time.perf_counter() - t0
            _, sizes, k = run(1743)
            prefixes.append({"mb": VALID_PREFIX_MB[len(prefixes)], "bytes": pos, "docs": n_docs,
                             "distinct_pretokens": len(counts), "pairs_first_step": sizes[0],
                             "pairs_at_1743": sizes[1742], "pairs_scanned_1743": sum(sizes[:1743]),
                             "identical_prefix_vs_official": k})
            print(f"valid prefix {prefixes[-1]['mb']} MB: P_1743 {sizes[1742]:,}", flush=True)
            targets.pop(0)
            t0 = time.perf_counter()
        pos += eot_len
    t_pretok += time.perf_counter() - t0
    total_bytes = pos - eot_len                # the text before the final EOT
    merges, sizes, k = run(49_743)
    full = {"bytes": pos, "text_bytes_before_last_eot": total_bytes, "docs": n_docs,
            "distinct_pretokens": len(counts), "merges": len(merges), "pairs_first_step": sizes[0],
            "pairs_at": {str(t): sizes[t - 1] for t in (1, 1743, 9743, 49_743)},
            "pairs_scanned_1743": sum(sizes[:1743]), "pairs_scanned": sum(sizes),
            "pairs_every_100_steps": sizes[::100], "identical_prefix_vs_official": k}
    print(f"valid 290 MB: sum P_t {full['pairs_scanned']:,}, prefix vs official {k}", flush=True)
    return {"what": "pairs the book's max() scans per step, counted with chapter 4's heap",
            "pretokenize_s": round(t_pretok, 1), "heap_s": round(t_heap, 1), "prefixes": prefixes, "full": full}


def record_heap8(sb, path: Path, rounds: int = 2) -> dict:
    """The 8 MB sample: the book's train_bpe (one process) vs chapter 4's heap (pure Python, one process).
    Both include pretokenization; interleaved rounds, best time per method."""
    bx = ch04_ext()
    sb.Pool = SerialPool
    text = path.read_bytes().decode("utf-8")
    methods = [("book_1743", lambda: sb.train_bpe(str(path), 2000, num_chunks=8)),
               ("heap_1743", lambda: bx.train_bpe_heap(text, 2000)),
               ("heap_9743", lambda: bx.train_bpe_heap(text, 10_000))]
    times = {k: [] for k, _ in methods}
    out, loads = {}, []
    for _ in range(rounds):
        loads.append(cpu_load())
        for key, fn in methods:
            t0 = time.perf_counter()
            out[key] = fn()
            times[key].append(time.perf_counter() - t0)
        print("heap8 round: " + ", ".join(f"{k} {v[-1]:.1f}s" for k, v in times.items()), flush=True)
    book = list(out["book_1743"].items())
    return {"bytes": path.stat().st_size, "rounds": rounds, "timing": "best of rounds",
            "cpu_load_percent_per_round": loads,
            "rows": [{"key": k, "merges": len(out[k]), "seconds": round(min(v), 2),
                      "seconds_all": [round(t, 2) for t in v]} for k, v in times.items()],
            "heap_identical_to_book": list(out["heap_1743"].items()) == book,
            "heap_9743_starts_with_book": list(out["heap_9743"].items())[:1743] == book}


def record_max_key(sb, path: Path, reps: int = 30, rounds: int = 5) -> dict:
    """max() over the book's pair table at the last of 1,743 steps (1 MB sample), with different keys,
    and the book's train_bpe with its max() call replaced by a C-level equivalent."""
    sb.Pool = SerialPool
    seen = {}

    def capture(it, *a, **kw):
        seen["d"] = dict(it)
        return builtins.max(it, *a, **kw)

    sb.max = capture
    try:
        sb.train_bpe(str(path), 2000, num_chunks=8)
    finally:
        del sb.max
    d = seen["d"]
    variants = {
        "lambda": lambda: builtins.max(d, key=lambda pair: (d[pair], pair[0], pair[1])),   # the book's line 160
        "get": lambda: builtins.max(d, key=d.get),                                           # line 159 (commented out)
        "zip": lambda: builtins.max(zip(d.values(), d))[1],
        "values_only": lambda: builtins.max(d.values()),
    }
    ns = {}
    for key, fn in variants.items():
        best = float("inf")
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            best = min(best, (time.perf_counter() - t0) / reps)
        ns[key] = round(best / len(d) * 1e9, 1)
    picks = {k: fn() for k, fn in variants.items() if k != "values_only"}

    def zip_max(it, *a, **kw):      # same order as the book's key (count, id0, id1); pairs are unique
        return builtins.max(zip(it.values(), it))[1]

    timings = {"book": [], "zip": []}
    rules = {}
    loads = []
    for _ in range(2):
        loads.append(cpu_load())
        for key in timings:
            if key == "zip":
                sb.max = zip_max
            try:
                t0 = time.perf_counter()
                rules[key] = sb.train_bpe(str(path), 2000, num_chunks=8)
                timings[key].append(time.perf_counter() - t0)
            finally:
                if key == "zip":
                    del sb.max
    print(f"max key: {ns}; train 1 MB book {min(timings['book']):.1f}s, zip {min(timings['zip']):.1f}s", flush=True)
    return {"pairs": len(d), "reps": reps, "rounds": rounds, "ns_per_pair": ns,
            "same_pick": {k: list(v) for k, v in picks.items()},
            "train_1mb": {"bytes": path.stat().st_size, "merges": len(rules["book"]),
                          "book_s": round(min(timings["book"]), 2), "zip_s": round(min(timings["zip"]), 2),
                          "seconds_all": {k: [round(t, 2) for t in v] for k, v in timings.items()},
                          "identical_merges": list(rules["zip"].items()) == list(rules["book"].items())},
            "cpu_load_percent_per_round": loads}


def counts_before_merge(sb, path: Path, k: int) -> dict:
    """The book's pair counts at the moment it chooses merge number k (0-based)."""
    snapshot = {}

    def capture(it, *a, **kw):
        snapshot["counts"] = dict(it)
        return builtins.max(it, *a, **kw)

    sb.max = capture
    try:
        sb.train_bpe(str(path), 256 + 1 + k + 1, num_chunks=8)     # stops right after choosing merge k
    finally:
        del sb.max
    return snapshot["counts"]


def record_hf(sb, samples: dict, rules_8mb: dict) -> dict | None:
    try:
        import tokenizers
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    except ImportError:
        return None

    def byte_decoder():
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {chr(c): b for b, c in zip(bs, cs)}

    u2b = byte_decoder()
    text = samples[8].read_bytes().decode("utf-8")
    docs = text.split(EOT)
    mine = vocab_of(rules_8mb)
    mine_merges = [mine[i] for i in rules_8mb.values()]
    runs = []
    for vocab in (2000, 10000):
        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT], show_progress=False, min_frequency=0,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        t0 = time.perf_counter()
        tk.train_from_iterator(docs, trainer=trainer)
        dt = time.perf_counter() - t0
        merges = json.loads(tk.to_str())["model"]["merges"]
        pairs = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in merges]
        pairs = [(bytes(u2b[c] for c in a), bytes(u2b[c] for c in b)) for a, b in pairs]
        learned = [a + b for a, b in pairs]
        rec = {"vocab": vocab, "merges": len(merges), "seconds": round(dt, 2)}
        if vocab == 2000:
            k = first_difference(learned, mine_merges)
            rec["identical_prefix_vs_book"] = k
            rec["same_tokens_as_book"] = len(set(learned) & set(mine_merges))
            if k < len(learned):
                # Is the first difference a tie? Look up both choices in the book's counts at that step.
                counts = counts_before_merge(sb, samples[8], k)
                id_of = {mine[i]: i for i in list(rules_8mb.values())[:k]}
                id_of.update({bytes([i]): i for i in range(256)})
                book_pair = list(rules_8mb)[k]
                hf_pair = (id_of[pairs[k][0]], id_of[pairs[k][1]])
                rec["first_difference"] = {
                    "book": {"token": mine_merges[k].decode("utf-8", "replace"), "count": counts[book_pair]},
                    "hf": {"token": learned[k].decode("utf-8", "replace"), "count": counts.get(hf_pair, 0)}}
        runs.append(rec)
        print(f"HF tokenizers {vocab}: {dt:.1f}s", flush=True)
    return {"version": tokenizers.__version__, "sample_mb": 8, "runs": runs}


def stamped(fn, *args) -> dict:
    """Run one section; record when, where and how busy the machine was (before and after)."""
    info = machine()
    out = fn(*args)
    info["python_10M_loop_s_at_end"] = machine()["python_10M_loop_s"]
    return {"date": datetime.date.today().isoformat(), "machine": info, **out}


def main(argv: list[str]) -> int:
    wanted = set(argv) or set(SECTIONS)
    if wanted - set(SECTIONS):
        print(f"unknown sections {sorted(wanted - set(SECTIONS))}; choose from {SECTIONS}", file=sys.stderr)
        return 2
    path = OUT / "bench.json"
    bench = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    bench["what"] = "Chapter 7 measurements; produced by verify/_ch07_record.py"
    sb = book_modules()
    tok = official_tokenizer()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        big = owt_text(tok, int(SIZES_MB[-1] * 1_000_000))
        samples = {}
        for mb in SIZES_MB:
            cut = big.find(EOT, int(mb * 1_000_000)) if mb < SIZES_MB[-1] else len(big)
            samples[mb] = write_sample(big[:cut], tmp, f"owt_{mb}MB.txt")
        if "encode" in wanted:
            bench["encode_2mb"] = stamped(record_encode, sb, tok, samples[2].read_bytes().decode("utf-8"))
        if "train" in wanted:
            info = machine()
            print(info, flush=True)
            training, rules_8mb = record_training(sb, samples)
            hf = record_hf(sb, samples, rules_8mb)
            info["python_10M_loop_s_at_end"] = machine()["python_10M_loop_s"]
            bench.update(date=datetime.date.today().isoformat(), machine=info, train=training, hf_tokenizers=hf)
        if "heap8" in wanted:
            bench["heap_8mb"] = stamped(record_heap8, sb, samples[8])
        if "maxkey" in wanted:
            bench["max_key"] = stamped(record_max_key, sb, samples[1])
    if "valid290" in wanted:
        bench["valid_290mb"] = stamped(record_valid, sb, tok)
    path.write_text(json.dumps(bench, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
