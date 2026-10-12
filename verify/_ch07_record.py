"""Chapter 7 measurements -> web/data/ch07/bench.json (minutes of CPU; run by hand, not by pytest).

    uv run python verify/_ch07_record.py                       # everything except the optional HF trainer
    uv run --with tokenizers python verify/_ch07_record.py     # also time Hugging Face's Rust BPE trainer

All inputs are OWT documents decoded from the official webbot/owt_valid.bin (see _ch07_common.py), so the
numbers can be reproduced without downloading the 12 GB corpus. Timings are wall-clock on whatever machine runs
this; the file records the machine and the load so the page can state the conditions next to every number.
Every row whose output is compared (merges, token ids) also records whether it was identical.
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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch07_common import (EOT, OUT, OWT_TRAIN_TXT_BYTES, SerialPool, book_modules,  # noqa: E402
                          official_rules, official_tokenizer, owt_text, webbot_tiktoken)

SIZES_MB = [0.25, 0.5, 1, 2, 4, 8]


def machine() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "logical_cpus": os.cpu_count()}
    try:
        info["cpu"] = subprocess.run(["powershell", "-NoProfile", "-Command",
                                      "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True,
                                     timeout=30).stdout.strip() if os.name == "nt" else platform.processor()
        if os.name == "nt":
            info["cpu_load_percent_at_start"] = int(subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).LoadPercentage"],
                capture_output=True, text=True, timeout=30).stdout.strip() or -1)
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


def best_of(fn, n):
    best, out = float("inf"), None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


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


def record_encode(sb, tok, text: str) -> dict:
    """Same text, same 50K merge table: the book's pure-Python encoder vs tiktoken (Rust)."""
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

    def pretokenize_only(t):
        return sum(1 for part in re.split(split, t) for _ in sb.pretokenize(part))

    docs = text.split(EOT)

    def batch(threads):
        out = []
        for i, ids in enumerate(enc.encode_ordinary_batch(docs, num_threads=threads)):
            out.extend(ids)
            if i < len(docs) - 1:
                out.append(tok.end_token_id)
        return out

    t_book, ref = best_of(lambda: tok.encode(text), 1)
    rows = [("book", "原书 encode（纯 Python）", t_book, True)]
    for key, label, fn, n in [
        ("lookup", "原书 + 整词查表（纯 Python）", lambda: lookup_then_bpe(text), 3),
        ("pretokenize", "只做预分词（Python regex）", lambda: pretokenize_only(text), 3),
        ("tiktoken1", "tiktoken，1 个线程", lambda: enc.encode(text, allowed_special="all"), 5),
        ("tiktoken8", "tiktoken，8 个线程", lambda: batch(8), 5),
    ]:
        t, out = best_of(fn, n)
        rows.append((key, label, t, None if key == "pretokenize" else out == ref))   # pretokenize gives no ids
    return {"bytes": nbytes, "tokens": len(ref), "docs": len(docs),
            "rows": [{"key": k, "label": lab, "seconds": round(t, 3), "mb_per_s": round(nbytes / t / 1e6, 3),
                      "speedup": round(t_book / t, 1), "identical_ids": same} for k, lab, t, same in rows],
            "owt_train_hours": {k: round(OWT_TRAIN_TXT_BYTES / (nbytes / t) / 3600, 2) for k, _, t, _ in rows}}


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


def main() -> None:
    sb = book_modules()
    tok = official_tokenizer()
    info = machine()
    print(info, flush=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        big = owt_text(tok, int(SIZES_MB[-1] * 1_000_000))
        samples = {}
        for mb in SIZES_MB:
            cut = big.find(EOT, int(mb * 1_000_000)) if mb < SIZES_MB[-1] else len(big)
            samples[mb] = write_sample(big[:cut], tmp, f"owt_{mb}MB.txt")
        encode = record_encode(sb, tok, samples[2].read_bytes().decode("utf-8"))
        print("encode done", flush=True)
        training, rules_8mb = record_training(sb, samples)
        hf = record_hf(sb, samples, rules_8mb)
    info["python_10M_loop_s_at_end"] = machine()["python_10M_loop_s"]
    bench = {"what": "Chapter 7 measurements; produced by verify/_ch07_record.py", "date": datetime.date.today().isoformat(),
             "machine": info, "encode_2mb": encode, "train": training, "hf_tokenizers": hf}
    path = OUT / "bench.json"
    path.write_text(json.dumps(bench, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
