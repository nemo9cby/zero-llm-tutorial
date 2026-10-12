"""第 4 章 分词器【应用篇】：本地验证。

全部断言都针对 third_party/dlfs6 里的真实代码。ch04 的脚本顶层会 chdir 并在 2.2 GB 语料上训练，
所以用 _ch04_load.load_defs 只执行其中的 import / def / class（函数体就是原书代码）。
多进程测试用 storybot.tokenizer 里的工作函数（可以被 spawn 出来的子进程 import）。
"""
from __future__ import annotations

import ast
import inspect
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tracemalloc
from collections import Counter, defaultdict
from multiprocessing import Pool
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import DLFS6, ROOT
from _ch04_load import function_ast, load_defs, quiet_tqdm
import _ch04_bench as bench
import _ch04_data

DATA = ROOT / "web" / "data" / "ch04"
EOT = "<|endoftext|>"
VALID = DLFS6 / "storybot" / "tiny_stories_valid.txt"
VALID_BIN = DLFS6 / "storybot" / "tiny_stories_valid.bin"
CODES = DLFS6 / "codebot" / "tiny_codes.txt"
CODES_BIN = DLFS6 / "codebot" / "tiny_codes.bin"


def items(rules: dict) -> list:
    """Merge rules as an ordered list: same pairs, same ids, same order."""
    return list(rules.items())


def prefix_at_eot(data: bytes, approx: int) -> bytes:
    """The first stories of a corpus, cut just before the first <|endoftext|> at or after `approx`."""
    return data[: data.index(EOT.encode(), approx)]


def same_statements(name: str, *rels: str) -> bool:
    """True when function (or method `Class.name`) `name` has the same body, statement for statement, in every file
    of `rels` (default arguments may differ)."""
    return len({tuple(ast.dump(s) for s in function_ast(rel, name).body) for rel in rels}) == 1


@pytest.fixture(scope="module")
def m(dlfs6_path):
    """The book's modules (definitions only) plus this chapter's browser copies in web/data/ch04."""
    import codebot.tokenizer as cb
    import storybot.tokenizer as sb

    saved = cb.tqdm, sb.tqdm, sys.dont_write_bytecode
    cb.tqdm = sb.tqdm = quiet_tqdm
    sys.dont_write_bytecode = True        # web/data/ch04 is copied to the site: keep __pycache__ out of it
    sys.path.insert(0, str(DATA))
    import bpe_book
    import bpe_ext

    yield SimpleNamespace(
        cb=cb, sb=sb, book=bpe_book, ext=bpe_ext,
        v01=load_defs("ch04/01_bpe_optimize.py"), v02=load_defs("ch04/02_bpe_cache.py"),
        v03=load_defs("ch04/03_bpe_chunk.py"), v04=load_defs("ch04/04_bpe_parallel.py"),
        v06=load_defs("ch04/06_encode_optimize.py"), v07=load_defs("ch04/07_encode_parallel.py"),
        story_rules=pickle.loads((DLFS6 / "storybot" / "merge_rules.pkl").read_bytes()),
        code_rules=pickle.loads((DLFS6 / "codebot" / "merge_rules.pkl").read_bytes()),
    )
    sys.path.remove(str(DATA))
    cb.tqdm, sb.tqdm, sys.dont_write_bytecode = saved


@pytest.fixture(scope="module")
def valid_counts(m):
    """Pretoken counts of the whole TinyStories validation file (the 01/02 way: read, split, pretokenize)."""
    counts = Counter()
    for piece in VALID.read_bytes().decode("utf-8").split(EOT):
        counts.update(m.sb.pretokenize(piece))
    return counts


@pytest.fixture(scope="module")
def story_slice(tmp_path_factory):
    """About 30 KB of TinyStories validation stories as a file (for the trainers that take a path)."""
    path = tmp_path_factory.mktemp("ch04") / "valid_30k.txt"
    path.write_bytes(prefix_at_eot(VALID.read_bytes(), 30_000))
    return path


# ---------------------------------------------------------------- data shipped to the browser

def test_web_data_in_sync(dlfs6_path):
    """Every derived file in web/data/ch04 is exactly what verify/_ch04_data.py builds from the sources."""
    for name, blob in _ch04_data.derived_files().items():
        assert (DATA / name).read_bytes() == blob, f"web/data/ch04/{name} is stale: run verify/_ch04_data.py"
    for name, blob in _ch04_data.js_fixtures().items():      # expected values of tests/js/ch04-bpe.test.mjs
        assert (_ch04_data.JS_FIXTURES / name).read_bytes() == blob, f"tests/js/fixtures/ch04/{name} is stale"


def test_bpe_book_is_verbatim(m):
    """bpe_book.py = the book's functions, node for node (only tqdm removed and versions renamed)."""
    shipped = ast.parse((DATA / "bpe_book.py").read_text(encoding="utf-8"))
    shipped_defs = {n.name: n for n in shipped.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for rel, name, new_name, renames in _ch04_data.RENAMES:
        expected = _ch04_data.transformed(rel, name, new_name, renames)
        assert ast.dump(shipped_defs[new_name]) == ast.dump(expected), f"{new_name} differs from {rel}::{name}"
    # The storybot copies used in the browser are the same code as the ch04 scripts they came from.
    same = [("ch04/03_bpe_chunk.py", "find_chunk_boundaries"), ("ch04/04_bpe_parallel.py", "find_chunk_boundaries"),
            ("ch04/04_bpe_parallel.py", "pretoken_chunk"), ("ch04/03_bpe_chunk.py", "pretokenize")]
    for rel, name in same:
        assert ast.dump(_ch04_data.book_node(rel, name)) == ast.dump(_ch04_data.book_node("storybot/tokenizer.py", name))
    # The merge step of ch04/02 and storybot is literally the same code; only the defaults differ.
    loop = lambda rel: ast.dump(_ch04_data.book_node(rel, "train_bpe").body[-2])   # the `for step in ...` loop
    assert loop("ch04/02_bpe_cache.py") == loop("ch04/03_bpe_chunk.py") == loop("ch04/04_bpe_parallel.py") \
        == loop("storybot/tokenizer.py")
    # storybot's train_bpe is 04's, statement for statement; only the default num_chunks differs (8 vs 64), which is
    # why the full-train-set record (bench_runs/train_full.json) calls it with 04's arguments.
    assert same_statements("train_bpe", "ch04/04_bpe_parallel.py", "storybot/tokenizer.py")
    defaults = lambda rel: [ast.literal_eval(d) for d in function_ast(rel, "train_bpe").args.defaults]
    assert (defaults("ch04/04_bpe_parallel.py"), defaults("storybot/tokenizer.py")) == \
        (["<|endoftext|>", 8, 64], ["<|endoftext|>", 8, 8])


def test_merges_json_match_pickles(m):
    for name, rules in [("storybot_merges.json", m.story_rules), ("codebot_merges.json", m.code_rules)]:
        merges = json.loads((DATA / name).read_text())
        assert {tuple(p): 256 + i for i, p in enumerate(merges)} == rules
        assert list(rules.values()) == list(range(256, 256 + len(rules)))   # pickle order = training order


def test_train_head_sample_matches_official_train_bin(m):
    """train_head10k.txt (first 10,000 characters of the 2.2 GB train file, fetched with an HTTP Range request)
    is confirmed by the official train.bin: all but the last (cut-off) token agree."""
    sample = (DATA / "train_head10k.txt").read_text(encoding="utf-8")
    official = np.fromfile(DATA / "train_head.bin", dtype=np.uint16).tolist()
    ids = m.sb.BPETokenizer(m.story_rules).encode(sample)
    assert len(sample) == 10_000 and len(sample.encode("utf-8")) == 10_008
    assert len(ids) == 2394 and ids[:-1] == official[: len(ids) - 1] and ids[-1] != official[len(ids) - 1]


# ---------------------------------------------------------------- 4.1 dedup

def test_dedup_statistics_on_real_corpora(m, valid_counts):
    codes = Counter()
    for piece in CODES.read_text(encoding="utf-8").split(EOT):
        codes.update(m.cb.pretokenize(piece))
    assert (sum(codes.values()), len(codes)) == (1_826_876, 30_535)
    assert (sum(valid_counts.values()), len(valid_counts)) == (5_419_001, 13_111)
    assert sum(valid_counts.values()) / len(valid_counts) > 400
    # How skewed the frequencies are: the 10 most frequent pretokens take 32% of all occurrences,
    # the 41 most frequent just over half, the 1,000 most frequent 92.5%.
    ranked = valid_counts.most_common()
    share = lambda k: sum(c for _, c in ranked[:k]) / sum(valid_counts.values())
    assert [w for w, _ in ranked[:10]] == [".", ",", " the", " and", " a", "\n", " to", " was", " They", " it"]
    assert (round(share(10), 4), round(share(1000), 4)) == (0.3194, 0.9254)
    assert share(40) < 0.5 <= share(41)
    # The longest pretoken of the validation set: 23 bytes (a row of '='), so a word is merged at most 22 times.
    assert max(len(w.encode()) for w in valid_counts) == 23 and "=" * 23 in valid_counts


def test_small_slice_statistics(m):
    """4.1.4: the 499,577-byte slice (validation set cut before an <|endoftext|>) used by Fig. 4-1's first group.
    Bytes scanned per step: naive vs deduplicated, at the first step and as the 743 merges proceed."""
    small = bench.slice_before_eot(VALID.read_bytes(), bench.SMALL_LIMIT)
    assert len(small) == 499_577
    text = small.decode("utf-8")
    occ = [p for piece in text.split(EOT) for p in m.sb.pretokenize(piece)]
    counts = Counter(occ)
    b_occ, b_uniq = sum(len(p.encode()) for p in occ), sum(len(p.encode()) for p in counts)
    assert (len(occ), len(counts), b_occ, b_uniq) == (120_503, 3_523, 491_504, 22_792)
    assert round(b_occ / b_uniq, 1) == 21.6 and round(b_uniq / len(counts), 1) == 6.5    # short sequences
    # Frequent words become one token first, so the naive trainer's scan shrinks faster than the deduplicated one's:
    # ids scanned at the first step 21.6x apart, at the last step 15.1x, over all 743 steps 16.1x.
    rules, st = bench.merge_loop_profile(m.ext.ids_counts_of(text), 743, count_ids=True)
    assert (st["ids_occ"][0], st["ids_uniq"][0]) == (b_occ, b_uniq)
    assert (st["ids_occ"][-1], st["ids_uniq"][-1]) == (155_977, 10_307)
    assert round(st["ids_occ"][-1] / st["ids_uniq"][-1], 1) == 15.1
    assert round(sum(st["ids_occ"]) / sum(st["ids_uniq"]), 1) == 16.1
    assert items(rules) == items(m.v02.train_bpe(text, 1000))           # the counters do not change the loop


def test_count_pairs_signatures_differ(m):
    assert dict(m.cb.count_pairs([1, 2, 3], defaultdict(int))) == {(1, 2): 1, (2, 3): 1}
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for \+="):
        m.sb.count_pairs([1, 2, 3], defaultdict(int))          # counts lands in `weight`
    assert dict(m.sb.count_pairs([1, 2, 3], counts=defaultdict(int))) == {(1, 2): 1, (2, 3): 1}
    with pytest.raises(KeyError):
        m.sb.count_pairs([1, 2, 3], 1, {})                     # needs a defaultdict(int)


def test_all_trainers_agree(m, story_slice):
    """naive (ch01) = 01 dedup = 02 cache = 03 chunk = storybot (Pool) = heap and the browser copies."""
    text = story_slice.read_bytes().decode("utf-8")
    vocab = 320
    ref = items(m.cb.train_bpe(text, vocab))
    assert len(ref) == vocab - 257
    results = {
        "ch04/01": m.v01.train_bpe(text, vocab),
        "ch04/02": m.v02.train_bpe(text, vocab),
        "ch04/03": m.v03.train_bpe(str(story_slice), vocab),
        "storybot (Pool 2)": m.sb.train_bpe(str(story_slice), vocab, num_processes=2, num_chunks=8),
        "bpe_ext heap": m.ext.train_bpe_heap(text, vocab),
        "bpe_ext full": m.ext.train_full(text, vocab),
        "bpe_ext incremental": m.ext.train_incremental(text, vocab),
        "bpe_book cache": m.book.train_bpe_cache(text, vocab),
    }
    for name, rules in results.items():
        assert items(rules) == ref, name


# ---------------------------------------------------------------- 4.2 incremental update

def test_cache_trainer_reproduces_codebot_merges(m):
    """ch04/02 on the whole TinyCodes (LF, as stored in git) = codebot/merge_rules.pkl, order included."""
    assert items(m.v02.train_bpe(CODES.read_text(encoding="utf-8"), 1000)) == items(m.code_rules)


def test_incremental_invariant_on_valid_prefix(m):
    """After every one of 200 merges, ch04/02's incremental counts equal a full recount."""
    text = prefix_at_eot(VALID.read_bytes(), 200_000).decode("utf-8")
    ids_counts = m.ext.ids_counts_of(text)
    pair_counts, pair_to_ids = m.ext.init_cache(ids_counts)
    for step in range(200):
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        m.ext.apply_merge(ids_counts, pair_counts, pair_to_ids, best, 256 + step)
        full = defaultdict(int)
        for ids, c in ids_counts.items():
            m.sb.count_pairs(ids, c, full)
        assert dict(full) == dict(pair_counts), step


def test_heap_code_on_page_is_bpe_ext(m):
    """4.2.4 prints heap_merge_loop as a plain code block (listings only read the book's files): keep it verbatim."""
    page = (ROOT / "content" / "ch04.md").read_text(encoding="utf-8")
    shown = re.search(r"```python\n# web/data/ch04/bpe_ext\.py[^\n]*\n(.*?)```", page, re.S).group(1)
    assert shown == inspect.getsource(m.ext.heap_merge_loop)


# ---------------------------------------------------------------- 4.3 chunking

def test_gbk_default_encoding_breaks_on_tinystories():
    """What `open(path).read()` does on Chinese Windows without UTF-8 mode (locale encoding cp936 = GBK)."""
    raw = VALID.read_bytes()
    with pytest.raises(UnicodeDecodeError) as err:
        raw.decode("gbk")
    assert err.value.start == 30_846 and raw[30_846] == 0x9D        # offset counted from 0
    assert err.value.reason == "illegal multibyte sequence"
    assert raw[30_844:30_847].decode("utf-8") == "”"          # the right curly quote ” = e2 80 9d
    # GBK reads two bytes at a time: e2 80 becomes the character 鈥; then 9d (a lead byte) is followed by
    # 2e ('.'), which cannot be the second byte of a GBK character.
    assert raw[30_843:30_849].hex(" ") == "7a e2 80 9d 2e 20"
    assert raw[30_844:30_846].decode("gbk") == "鈥"
    assert all(c.encode("utf-8")[:2] == b"\xe2\x80" for c in "“”‘’–—…")    # all of them start with the same 鈥
    assert CODES.read_bytes().decode("gbk") == CODES.read_text(encoding="utf-8")   # pure ASCII: no problem


def test_crlf_checkout_changes_merges(m, tmp_path):
    lf = CODES.read_bytes()
    assert b"\r" not in lf
    crlf = tmp_path / "tiny_codes_crlf.txt"
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))
    # Text mode (01/02/06) hides the \r: the text is exactly the LF file, so the merges are the official ones
    # (test_cache_trainer_reproduces_codebot_merges trains ch04/02 on that very text).
    with open(crlf, encoding="utf-8") as f:
        assert f.read() == CODES.read_text(encoding="utf-8")
    # Binary mode (03/04/07) keeps it: the 2nd merge (id 257) is \r\n.
    rules = m.v03.train_bpe(str(crlf), 262)
    assert {v: k for k, v in rules.items()}[257] == (13, 10)


def test_chunk_boundaries_on_full_valid(m):
    data = VALID.read_bytes()
    b = m.sb.find_chunk_boundaries(str(VALID), 64)
    assert len(b) == 65 and b[:4] == [0, 351752, 703494, 1055567] and b[-1] == len(data)
    assert all(data[x:x + 13] == EOT.encode() for x in b[1:-1])
    assert len(m.sb.find_chunk_boundaries(str(VALID), 100_000)) == 27_629
    assert data.count(EOT.encode()) == 27_630         # so at most 27,630 + 2 boundaries
    data.decode("utf-8")      # valid UTF-8, so errors="ignore" drops nothing at '<' boundaries
    # More chunks do not mean more boundaries: the count wanders below 27,632 and falls again for huge k,
    # because chunk_size = S // k leaves the bytes after the last initial cut, (k-1)*(S//k), without any cut.
    counts = {k: len(m.sb.find_chunk_boundaries(str(VALID), k)) for k in (27_631, 40_000, 60_000, 200_000, 1_000_000)}
    assert counts == {27_631: 24_266, 40_000: 27_369, 60_000: 27_626, 200_000: 27_512, 1_000_000: 27_021}
    marker, offsets, i = EOT.encode(), [], data.find(EOT.encode())
    while i != -1:
        offsets.append(i)
        i = data.find(marker, i + 1)
    S = len(data)
    for k, last_cut, beyond in [(100_000, 22_499_775, 3), (1_000_000, 21_999_978, 612)]:
        assert (k - 1) * (S // k) == last_cut and sum(o >= last_cut for o in offsets) == beyond
    # The 60-story file of the figure: past 59 markers the count still moves (55, 54, 53, ... then 60, 61).
    head = str(DATA / "valid_head.txt")
    assert {k: len(m.sb.find_chunk_boundaries(head, k)) for k in (59, 60, 61, 70, 80, 120, 1000)} == \
        {59: 55, 60: 54, 61: 53, 70: 56, 80: 60, 120: 60, 1000: 61}
    # chunked counts == whole-file counts: test_parallel_pretokenize_equals_serial runs the same
    # pretoken_chunk on these same 64 chunks (in 4 processes) and compares with the whole-file counts.


def test_pretokenization_memory_column(m, tmp_path):
    """Column 4.3: Python-level peak memory of the pretokenization stage of each version, on 1 MB of stories.
    vocab_size=257 means zero merges, so each call does exactly the pretokenization part of the book's code."""
    path = tmp_path / "valid_2mb.txt"
    path.write_bytes(prefix_at_eot(VALID.read_bytes(), 1_000_000))

    def peak(fn):
        tracemalloc.start()
        fn()
        _, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return top

    p_naive = peak(lambda: m.cb.train_bpe(path.read_text(encoding="utf-8"), 257))   # one list per occurrence
    p_dedup = peak(lambda: m.v01.train_bpe(path.read_text(encoding="utf-8"), 257))  # counts, whole text in memory
    p_chunk = peak(lambda: m.v03.train_bpe(str(path), 257))                         # counts, 64 chunks
    print(f"peak MB: naive {p_naive / 1e6:.1f}, dedup {p_dedup / 1e6:.1f}, chunked {p_chunk / 1e6:.1f}")
    assert p_naive > 5 * p_dedup and p_dedup > 1.5 * p_chunk, (p_naive, p_dedup, p_chunk)
    assert (round(p_naive / 1e6, 1), round(p_dedup / 1e6, 1), round(p_chunk / 1e6, 1)) == (27.6, 4.0, 2.2)
    if sys.maxsize > 2**32:
        assert sys.getsizeof(list(b"abcd")) == 88              # ch01 keeps one such list per occurrence


# ---------------------------------------------------------------- 4.4 parallel

def run_killing_tree(cmd, timeout, stdin=None):
    """subprocess.run with a timeout that also kills grandchildren (pool workers) on Windows/POSIX."""
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin is not None else None, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8", **kw)
    try:
        out, err = p.communicate(input=stdin, timeout=timeout)
        return False, p.returncode, out, err
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, 9)
        out, err = p.communicate()
        return True, p.returncode, out, err


def test_spawn_needs_importable_worker(tmp_path):
    head = "import multiprocessing as mp\ndef square(x):\n    return x * x\nif __name__ == '__main__':\n" \
           "    mp.set_start_method('spawn')\n"
    pool_body = "    with mp.Pool(2) as pool:\n        print(list(pool.imap(square, range(4))))\n"
    # A Process whose target lives in `python -c` code: the child cannot find it and exits with code 1.
    hung, rc, out, err = run_killing_tree(
        [sys.executable, "-c", head + "    p = mp.Process(target=square, args=(3,)); p.start(); p.join()\n"
         "    print(p.exitcode)\n"], timeout=60)
    assert not hung and out.strip() == "1"
    assert "Can't get attribute 'square' on <module '__main__'" in err
    # The same function in a Pool: the workers die on every task and the parent waits forever.
    hung, rc, out, err = run_killing_tree([sys.executable, "-c", head + pool_body], timeout=10)
    assert hung and out == ""
    assert "Can't get attribute 'square' on <module '__main__'" in err
    # The guarded script fed through stdin (`python - < script.py`): the children cannot re-read the main module.
    hung, rc, out, err = run_killing_tree([sys.executable, "-"], timeout=10, stdin=head + pool_body)
    assert hung and out == "" and "<stdin>" in err
    if os.name == "nt":
        assert "OSError: [Errno 22] Invalid argument" in err
    # A script file without the guard: every spawned worker re-runs the top level, tries to start a Pool of its
    # own and dies with RuntimeError; the parent keeps replacing the workers and never finishes.
    noguard = tmp_path / "noguard.py"
    noguard.write_text("import multiprocessing as mp\ndef square(x):\n    return x * x\n"
                       "with mp.get_context('spawn').Pool(2) as pool:\n    print(list(pool.imap(square, range(4))))\n",
                       encoding="utf-8")
    hung, rc, out, err = run_killing_tree([sys.executable, str(noguard)], timeout=10)
    assert hung and out == ""
    assert "RuntimeError" in err and "An attempt has been made to start a new process before the" in err
    assert "if __name__ == '__main__':" in err             # the message shows the idiom that fixes it


def test_parallel_pretokenize_equals_serial(m, valid_counts):
    """ch04/04 steps 1-3 with Pool(4) (spawn on Windows) give exactly the serial counts."""
    b = m.sb.find_chunk_boundaries(str(VALID), 64)
    tasks = [(str(VALID), s, e, EOT) for s, e in zip(b[:-1], b[1:])]
    total, sizes = Counter(), []
    with Pool(processes=4) as pool:
        for part in pool.imap(m.sb.pretoken_chunk, tasks):
            total.update(part)
            sizes.append(len(part))
    assert total == valid_counts
    # what each worker sends back: a dict of about 3,000 entries (4.4.2)
    sizes.sort()
    assert (sizes[0], sizes[len(sizes) // 2], sizes[-1]) == (2_991, 3_117, 3_290)


def test_default_parameters(m):
    sig = lambda f: {k: v.default for k, v in inspect.signature(f).parameters.items()
                     if v.default is not inspect.Parameter.empty}
    assert sig(m.v04.train_bpe) == {"end_token": EOT, "num_processes": 8, "num_chunks": 64}
    assert sig(m.sb.train_bpe) == {"end_token": EOT, "num_processes": 8, "num_chunks": 8}
    assert sig(m.v07.BPETokenizer.encode_file) == {"num_processes": 8, "num_chunks": 64, "cache_dir": "bpe_cache"}
    assert sig(m.sb.BPETokenizer.encode_file) == {"num_processes": 4, "num_chunks": 64, "cache_dir": "bpe_cache"}
    assert "buffer_size = 4096" in inspect.getsource(m.sb.find_chunk_boundaries)
    assert "find_chunk_boundaries(file_path, num_chunks=64)" in inspect.getsource(m.v03.train_bpe)
    src = (DLFS6 / "ch04" / "04_bpe_parallel.py").read_text(encoding="utf-8")
    assert "train_bpe(file_path, vocab_size, num_processes=8)" in src and "vocab_size = 10000" in src
    # 278 MB per chunk when storybot's num_chunks=8 meets the 2.2 GB train file
    assert round(2_227_753_162 / 8 / 1e6) == 278


# ---------------------------------------------------------------- 4.5 evaluation

def test_eval_script_reconstruction(m, monkeypatch, capsys):
    """Run the 05_eval.py printed in content/ch04.md (the train file replaced by its first 10,000 characters)."""
    page = (ROOT / "content" / "ch04.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(# ch04/05_eval\.py.*?)```", page, re.S).group(1)
    code = code.replace("os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))\n", "")
    code = code.replace("sys.path.append('.')\n", "")
    code = code.replace('"storybot/tiny_stories_train.txt"', repr(str(DATA / "train_head10k.txt")))
    monkeypatch.chdir(DLFS6)
    exec(compile(code, "05_eval.py", "exec"), {"__name__": "__main__"})
    out = capsys.readouterr().out
    for line in ["  ID 256: ' t'", "  ID 262: ' the'", "  ID 9998: ' bold'", "  ID 9999: '<|endoftext|>'",
                 "字节数: 10,008", "token 数: 2,394", "压缩率: 4.18 倍（平均每个 token 4.18 字节）",
                 "CodeBot 分词器: token 数 4,879，压缩率 2.05 倍"]:
        assert line in out.splitlines(), line
    # f.read(10000) in text mode == read()[:10000], without reading the rest of the file
    with open(VALID, encoding="utf-8") as f:
        head = f.read(10000)
    assert head == VALID.read_text(encoding="utf-8")[:10000]


def test_official_valid_bin_facts():
    ids = np.fromfile(VALID_BIN, dtype=np.uint16)
    assert ids.shape == (5_465_873,) and VALID_BIN.stat().st_size == 2 * 5_465_873
    assert int(ids.max()) == 9999 and int((ids == 9999).sum()) == 27_630
    assert VALID.stat().st_size == 22_502_601
    assert round(VALID.stat().st_size / len(ids), 3) == 4.117


def test_storybot_vocab_facts(m):
    tok = m.sb.BPETokenizer(m.story_rules)
    assert (len(m.story_rules), tok.end_token_id, tok.vocab_size) == (9743, 9999, 10000)
    partial = [i for i, b in tok.id_to_bytes.items()
               if i >= 256 and b.decode("utf-8", "replace") != b.decode("utf-8", "ignore")]   # learned tokens only
    assert partial == [696, 4606] and tok.id_to_bytes[696] == b"\xe2\x80" and tok.id_to_bytes[4606] == b" \xe2\x80"

    def strict(b):
        try:
            b.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    # over the whole vocabulary, the 128 single bytes 0x80-0xFF fail too: the first failure is id 128
    assert [i for i in range(tok.vocab_size) if not strict(tok.id_to_bytes[i])] == list(range(128, 256)) + [696, 4606]
    # three learned tokens share the maximum length; max() just returns the first of them
    longest = max(len(tok.id_to_bytes[i]) for i in range(256, 9999))
    assert longest == 15 and [i for i in range(256, 9999) if len(tok.id_to_bytes[i]) == longest] == [7159, 9140, 9375]
    assert [tok.id_to_bytes[i] for i in (7159, 9140, 9375)] == [b" accomplishment", b" disappointment", b" responsibility"]
    assert max(range(256, 9999), key=lambda i: len(tok.id_to_bytes[i])) == 7159
    # no Chinese anywhere: not in the validation text, not in any learned token
    assert not any("一" <= c <= "鿿" for c in VALID.read_text(encoding="utf-8"))   # CJK unified ideographs
    assert not any(0xE4 <= b <= 0xE9 for i in range(256, 9999) for b in tok.id_to_bytes[i])
    assert sum(any(b >= 0x80 for b in tok.id_to_bytes[i]) for i in range(256, 9999)) == 20
    # ... all of them punctuation (curly quotes, dashes, the ellipsis); two of them are half a character
    non_ascii = [i for i in range(256, 9999) if any(b >= 0x80 for b in tok.id_to_bytes[i])]
    chars = {c for i in non_ascii for c in tok.id_to_bytes[i].decode("utf-8", "ignore") if ord(c) > 127}
    assert chars == set("–—‘’“”…") and [i for i in non_ascii if i in partial] == [696, 4606]


# ---------------------------------------------------------------- 4.6 rank-based encoding

def test_rank_encoder_matches_ch01_encoder(m):
    old, new = m.cb.BPETokenizer(m.story_rules), m.v06.BPETokenizer(m.story_rules)
    words = sorted(set(m.sb.pretokenize(VALID.read_text(encoding="utf-8")[:20_000])))[:100]
    assert all(old._encode_text(w) == new._encode_text(w) for w in words)
    # 06's own __main__: CodeBot rules on TinyCodes reproduce codebot/tiny_codes.bin (2,677,398 tokens).
    official = np.fromfile(CODES_BIN, dtype=np.uint16).tolist()
    assert len(official) == 2_677_398
    code_tok = m.v06.BPETokenizer(m.code_rules)
    head = prefix_at_eot(CODES.read_bytes(), 200_000).decode("utf-8")
    ids = code_tok.encode(head)
    assert ids == official[: len(ids)] and official[len(ids)] == 999

    class Memo(m.v06.BPETokenizer):            # same encoder, each distinct pretoken encoded once
        def __init__(self, rules):
            super().__init__(rules)
            self.memo = {}

        def _encode_text(self, text):
            if text not in self.memo:
                self.memo[text] = super()._encode_text(text)
            return self.memo[text]

    assert Memo(m.code_rules).encode(CODES.read_text(encoding="utf-8")) == official
    # The browser's extension (bpe_ext.MemoTokenizer) caches storybot's encoder, whose _encode_text is 06's;
    # encode (split on <|endoftext|>, then pretokenize) is the same code in 06, 07 and storybot (4.7.1).
    same_code = lambda name, *rels: len({ast.dump(function_ast(rel, name)) for rel in rels}) == 1
    assert same_code("BPETokenizer._encode_text", "storybot/tokenizer.py", "ch04/06_encode_optimize.py")
    assert same_code("BPETokenizer.encode", "storybot/tokenizer.py", "ch04/06_encode_optimize.py", "ch04/07_encode_parallel.py")
    # ... and the same logic as ch01's encode: only the tqdm progress bar is written differently
    no_tqdm = lambda rel: ast.dump(_ch04_data._Transform({}).visit(function_ast(rel, "BPETokenizer.encode")))
    assert no_tqdm("codebot/tokenizer.py") == no_tqdm("ch04/06_encode_optimize.py")
    assert ast.dump(function_ast("codebot/tokenizer.py", "BPETokenizer.encode")) != \
        ast.dump(function_ast("ch04/06_encode_optimize.py", "BPETokenizer.encode"))
    assert m.ext.MemoTokenizer(m.code_rules).encode(CODES.read_text(encoding="utf-8")) == official


def test_compression_on_whole_tinycodes(m):
    """4.5.3: on the first 10,000 characters of TinyCodes StoryBot's tokenizer looks slightly better than CodeBot's
    own (2.18 vs 2.12); on the whole corpus CodeBot's wins clearly (2.42 vs 2.06)."""
    codes = CODES.read_text(encoding="utf-8")
    ratio = lambda rules, text: len(text.encode("utf-8")) / len(m.ext.MemoTokenizer(rules).encode(text))
    head = (DATA / "codes_head10k.txt").read_text(encoding="utf-8")
    assert head == codes[:10_000]
    assert (f"{ratio(m.story_rules, head):.2f}", f"{ratio(m.code_rules, head):.2f}") == ("2.18", "2.12")
    whole = (len(m.ext.MemoTokenizer(m.code_rules).encode(codes)), len(m.ext.MemoTokenizer(m.story_rules).encode(codes)))
    assert whole == (2_677_398, 3_143_713)
    assert (f"{len(codes.encode()) / whole[0]:.2f}", f"{len(codes.encode()) / whole[1]:.2f}") == ("2.42", "2.06")


# ---------------------------------------------------------------- 4.7 encode_file

def test_encode_file_removes_cache_dir(m, tmp_path):
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("do not delete me", encoding="utf-8")
    src = tmp_path / "head.txt"
    src.write_bytes((DATA / "valid_head.txt").read_bytes())
    out = tmp_path / "head.bin"
    total = m.sb.BPETokenizer(m.story_rules).encode_file(str(src), str(out), num_processes=1, num_chunks=2,
                                                         cache_dir=str(precious))
    assert not precious.exists()                                  # the whole directory is gone, notes.txt included
    assert out.read_bytes() == (DATA / "valid_head.bin").read_bytes() and total == 12_636


def test_encode_chunk_task_pickles_whole_tokenizer(m):
    tok = m.sb.BPETokenizer(m.story_rules)
    per_task = len(ForkingPickler.dumps(tok._encode_chunk))       # what Pool.imap ships for every task
    assert 200_000 < per_task < 250_000
    assert per_task == len(pickle.dumps(tok._encode_chunk, protocol=4)) == 221_754   # Python 3.12's default protocol
    assert round(64 * per_task / 1e6) == 14                        # 64 chunks: about 14 MB in total


def test_encode_file_matches_official_bin(m, tmp_path):
    src = tmp_path / "valid_2mb.txt"
    src.write_bytes(prefix_at_eot(VALID.read_bytes(), 2_000_000))
    out = tmp_path / "valid_2mb.bin"
    total = m.sb.BPETokenizer(m.story_rules).encode_file(str(src), str(out), num_processes=2,
                                                         cache_dir=str(tmp_path / "cache"))
    official = np.fromfile(VALID_BIN, dtype=np.uint16)
    assert out.read_bytes() == official[:total].tobytes() and int(official[total]) == 9999
    # 4.7-e writes with tofile (07's _encode_chunk) instead of np.memmap, as the browser has no real disk:
    # the same ids written that way give the same bytes (raw uint16, no header)
    ids = m.ext.MemoTokenizer(m.story_rules).encode(src.read_bytes().decode("utf-8"))
    alt = tmp_path / "tofile.bin"
    np.array(ids, dtype=np.uint16).tofile(alt)
    assert len(ids) == total and alt.read_bytes() == out.read_bytes()


def test_uint16_limits():
    assert np.iinfo(np.uint16).max == 65535
    with pytest.raises(OverflowError):
        np.array([65536], dtype=np.uint16)
    assert np.array([65535, 65536, 70000]).astype(np.uint16).tolist() == [65535, 0, 4464]
    if sys.maxsize > 2**32:                                      # 64-bit CPython: 8-byte pointers per list slot
        assert (sys.getsizeof([0] * 1000) - sys.getsizeof([0] * 0)) / 1000 == 8
        assert sys.getsizeof(list(b"abcd")) == 88                # column 4.3: a 4-byte word as ch01's list of ids


def test_python_list_of_ids_memory(m):
    """4.7.3: kept as a Python list, token ids cost about 8.6 bytes each. The pointers take 8; the int objects are
    shared (ids < 256 are cached small ints, the others are the values of merge_rules), so they add little."""
    head = prefix_at_eot(VALID.read_bytes(), 2_000_000).decode("utf-8")
    tok = m.sb.BPETokenizer(m.story_rules)
    tracemalloc.start()
    ids = tok.encode(head)
    current, _ = tracemalloc.get_traced_memory()               # what is still allocated: the list of ids
    tracemalloc.stop()
    assert ids == np.fromfile(VALID_BIN, dtype=np.uint16)[: len(ids)].tolist() and len(ids) == 486_690
    per_token = sys.getsizeof(ids) / len(ids)                    # 8-byte pointers + the list's spare capacity
    assert round(per_token, 1) == 8.6 and round(current / len(ids), 1) == 8.6
    # no int object per token: one object per distinct id, shared by all its occurrences
    assert len({id(i) for i in ids}) == len(set(ids)) == 5_884
    assert per_token / np.dtype(np.uint16).itemsize > 4          # more than 4x the 2 bytes of uint16


NP1_PROBE = r'''
import multiprocessing as mp, sys
import numpy as np


def encode_chunk(ids):              # what 07's _encode_chunk does with the ids of a chunk
    return np.array(ids, dtype=np.uint16).tolist()


if __name__ == "__main__":
    print(np.__version__)
    if sys.argv[1] == "main":
        print(encode_chunk([65535, 65536, 70000]))
    else:
        mp.set_start_method("spawn")
        with mp.Pool(1) as pool:
            print(pool.map(encode_chunk, [[65535, 65536, 70000]])[0])
'''


@pytest.mark.slow
def test_numpy1_wraps_out_of_range_ids_silently(tmp_path):
    """4.7.3: the book's own environment pins numpy<2. NumPy 1.x wraps ids above 65,535 in np.array(ids, dtype=np.uint16)
    and only emits a DeprecationWarning, which Python's default filters show for code in __main__ alone:
    in a Pool worker (where _encode_chunk runs) the wrap is silent. Runs NumPy 1.26.4 from uv's cache."""
    assert '"numpy>=1.24.0,<2.0.0"' in (DLFS6 / "pyproject.toml").read_text(encoding="utf-8")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    probe = tmp_path / "np1_probe.py"
    probe.write_text(NP1_PROBE, encoding="utf-8")
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    results = {}
    for where in ("main", "worker"):
        proc = subprocess.run([uv, "run", "--offline", "--isolated", "--no-project", "--python", py, "--with",
                               "numpy==1.26.4", "python", str(probe), where], capture_output=True, text=True,
                              encoding="utf-8", timeout=300, cwd=tmp_path)
        if proc.returncode != 0 and "numpy" not in proc.stdout:
            pytest.skip(f"cannot build a NumPy 1.26.4 environment offline: {proc.stderr[-300:]}")
        results[where] = proc
    for where, proc in results.items():
        assert proc.stdout.splitlines() == ["1.26.4", "[65535, 0, 4464]"], where
    assert "DeprecationWarning" in results["main"].stderr
    assert "DeprecationWarning" not in results["worker"].stderr and results["worker"].stderr.strip() == ""


@pytest.mark.slow
def test_uv_run_uses_the_project_of_the_working_directory(tmp_path):
    """The box at the end of 4.7: `uv run` takes the project from the working directory. Run from the tutorial's root,
    a script inside third_party/dlfs6 gets the tutorial's environment (NumPy 2, as measured in this chapter); after
    `cd third_party/dlfs6`, uv builds the book's own environment (torch, numpy<2). Probed offline with two nested
    projects that have no dependencies, laid out like the tutorial and its submodule."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    book = (DLFS6 / "pyproject.toml").read_text(encoding="utf-8")
    assert '"torch>=2.5.0"' in book and '"numpy>=1.24.0,<2.0.0"' in book
    assert "workspace" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8")   # the submodule is its own project
    # 04 and 07 first change into the book's root, so run from anywhere they read and write the book's own files
    for rel in ("ch04/04_bpe_parallel.py", "ch04/07_encode_parallel.py"):
        head = (DLFS6 / rel).read_text(encoding="utf-8").splitlines()[:2]
        assert head == ["import os, sys", "os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))"]
    outer = tmp_path / "tutorial"
    inner = outer / "third_party" / "book"
    inner.mkdir(parents=True)
    for d, name in [(outer, "tutorial"), (inner, "book")]:
        (d / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
                                          'dependencies = []\n', encoding="utf-8")
    (inner / "probe.py").write_text("import sys\nprint(sys.prefix)\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")}

    def prefix(cwd, script):
        proc = subprocess.run([uv, "run", "--offline", "python", script], cwd=cwd, env=env, capture_output=True,
                              text=True, encoding="utf-8", timeout=300)
        if proc.returncode != 0:
            pytest.skip(f"uv could not create a probe environment offline: {proc.stderr[-300:]}")
        return os.path.normcase(os.path.realpath(proc.stdout.strip()))

    venv = lambda d: os.path.normcase(os.path.realpath(d / ".venv"))
    assert prefix(outer, "third_party/book/probe.py") == venv(outer)     # from the root: the root's environment
    assert prefix(inner, "probe.py") == venv(inner)                      # after cd: the inner project's own one


# ---------------------------------------------------------------- recorded measurements

def recorded_bench() -> dict:
    return json.loads((DATA / "bench.json").read_text(encoding="utf-8"))


def test_bench_json_consistent():
    """bench.json is exactly what verify/_ch04_bench.py assembles from the raw records in bench_runs/ (so no number
    in it is typed by hand), and every timed variant was compared with its group's reference when it was measured."""
    recorded = recorded_bench()
    assert recorded == json.loads(json.dumps(bench.assemble(), ensure_ascii=False))
    for group in recorded["races"]:
        assert group["reference"] and sum(bool(row.get("reference")) for row in group["rows"]) <= 1
        for row in group["rows"]:
            assert row.get("reference") or row["identical"] is True, (group["id"], row["name"])
            assert row["runs"] >= 1 and row["seconds"] > 0
    naive = next(r for r in recorded["races"] if r["id"] == "train_codes")["rows"][0]
    assert (naive["key"], naive["runs"], round(naive["seconds"] / 60), round(naive["wall"] / 60)) == ("naive", 1, 52, 77)
    assert recorded["profile"]["identical"] == {"replica": True, "heap": True}
    assert recorded["pretok"]["identical"] and recorded["pretok_repeat"]["same_counts_every_round"]
    assert all(r["identical"] and r["cache_dir_removed"] for r in recorded["encode_file"])
    assert recorded["train_prefix_encode"]["identical_to_official_train_bin_prefix"]
    assert recorded["repeat_small"]["identical"] and len(recorded["repeat_small"]["cpu"]["naive"]) == 3
    # records made after the script moved into the repository carry their own commits and command
    for name in ("repeat_small", "pretok_repeat", "pool_startup", "shrink", "train_full", "encode_full", "heaps_full"):
        prov = bench.load_raw(name)["provenance"]
        assert prov["script"] == "verify/_ch04_bench.py" and len(prov["dlfs6"]["commit"]) == 40, name


def test_full_train_set_encode_record():
    """The box at the end of 4.7: ch04/07's encode_file on the whole train file, recorded by verify/_ch04_bench.py
    (bench_runs/encode_full.json): the output is the official tiny_stories_train.bin, byte for byte (size and sha256
    as published on Hugging Face)."""
    rec = bench.load_raw("encode_full")
    assert rec["provenance"]["dlfs6"]["commit"].startswith("c9b6e2e") and not rec["provenance"]["dlfs6"]["dirty"]
    assert rec["provenance"]["command"] == "python verify/_ch04_bench.py run encode_full --train-full <train_full>"
    # 07's arguments: 8 processes (its __main__) and 64 chunks (its default). The run used storybot's BPETokenizer,
    # whose bound _encode_chunk spawned workers can unpickle; it is 07's class method for method.
    assert (rec["num_processes"], rec["num_chunks"]) == (8, 64)
    src = (DLFS6 / "ch04" / "07_encode_parallel.py").read_text(encoding="utf-8")
    assert '"storybot/tiny_stories_train.bin", num_processes=8)' in src
    for name in ("_encode_chunk", "encode_file", "encode", "_encode_text"):
        assert same_statements(f"BPETokenizer.{name}", "ch04/07_encode_parallel.py", "storybot/tokenizer.py"), name
    # the published file: its listing gives the sha256 of the files we do have, so it is the right listing
    official = rec["official"]
    assert official == bench.OFFICIAL_TRAIN_BIN and official["bytes"] == 1_082_458_446
    assert bench.sha256_file(DLFS6 / "storybot" / "merge_rules.pkl").startswith("8fb285d9")
    assert bench.sha256_file(VALID_BIN).startswith("05a36c44")
    assert rec["identical_to_official_train_bin"] is True and rec["cache_dir_removed"]
    assert (rec["out_bytes"], rec["out_sha256"]) == (official["bytes"], official["sha256"])
    assert 2 * rec["tokens"] == rec["out_bytes"] and rec["tokens"] == 541_229_223
    # wall clock with the machine fully loaded, and the memory of the process plus its 8 workers
    assert round(rec["run"]["wall"] / 60, 1) == 37.8 and rec["run"]["load"]["mean"] > 90
    assert round(rec["peak_rss_mib"] / 1024, 1) == 2.0


def test_full_train_set_heaps_record():
    """4.1.1: the whole train file counted (bench_runs/heaps_full.json), against the extrapolation from its first
    100 MiB in verify cell 4.1-b: 59,933 distinct pretokens where Heaps' law predicted about 2.7 x 21,808."""
    rec = bench.load_raw("heaps_full")
    assert rec["provenance"]["command"] == "python verify/_ch04_bench.py run heaps_full --train-full <train_full>"
    assert (rec["bytes"], rec["occurrences"], rec["unique"]) == (2_227_753_162, 536_592_168, 59_933)
    prefix = [r for r in recorded_bench()["heaps"] if r["file"] == "train_prefix.txt"]
    first, last = prefix[0], prefix[-1]
    beta = math.log(last["unique"] / first["unique"]) / math.log(last["occurrences"] / first["occurrences"])
    factor = round(rec["bytes"] / last["bytes"])
    predicted = last["unique"] * factor ** beta
    assert factor == 21 and abs(predicted / rec["unique"] - 1) < 0.03            # the extrapolation was within 3%
    assert round(rec["occurrences"] / last["occurrences"], 1) == 21.2
    # and the frequencies are as skewed as in the validation set
    assert (rec["top10_share"], rec["top1000_share"]) == (0.3192, 0.925)


def test_full_train_set_record(m):
    """4.5 and the box at the end of 4.7: ch04/04's training on the whole 2.2 GB train file, recorded by
    verify/_ch04_bench.py (bench_runs/train_full.json), reproduces the official storybot/merge_rules.pkl."""
    rec = bench.load_raw("train_full")
    assert rec["provenance"]["dlfs6"]["commit"].startswith("c9b6e2e") and not rec["provenance"]["dlfs6"]["dirty"]
    assert rec["provenance"]["command"] == "python verify/_ch04_bench.py run train_full --train-full <train_full>"
    # the input is the whole train file (need() checked its sha256), the same file CS336 A1 uses
    identity = json.loads((ROOT / "web" / "data" / "cs336" / "data_identity.json").read_text(encoding="utf-8"))
    assert bench.INPUTS["train_full"]["sha256"] in json.dumps(identity) and rec["bytes"] == 2_227_753_162
    # 04's arguments: 8 processes (its __main__) and 64 chunks (its default). The run called storybot's train_bpe,
    # whose pool workers can import it; it is 04's train_bpe statement for statement.
    assert (rec["vocab"], rec["num_processes"], rec["num_chunks"]) == (10000, 8, 64)
    assert same_statements("train_bpe", "ch04/04_bpe_parallel.py", "storybot/tokenizer.py")
    # the learned rules: the same list as the official pickle, which the recorded digest confirms independently
    assert rec["equals_storybot_merge_rules_pkl"] is True and rec["first_difference"] is None
    assert rec["merges"] == rec["same_pairs_as_set"] == len(m.story_rules) == 9743
    assert rec["rules_sha256"] == bench.rules_digest(m.story_rules)
    # where the time went (wall clock, the machine fully loaded): pretokenization, then the serial merge loop
    run, minutes = rec["run"], lambda s: round(s / 60, 1)
    pretok, loop = run["phases"]["Pretokenizing"]["wall"], run["phases"]["Training BPE"]["wall"]
    assert (minutes(run["wall"]), minutes(pretok), minutes(loop)) == (9.4, 3.2, 6.1)
    assert run["load"]["mean"] > 90 and round(rec["peak_rss_mib"] / 1024, 1) == 1.5   # GiB, far below CS336's 30 GB


def test_bench_deterministic_facts(m, valid_counts):
    """The parts of bench.json that do not depend on the clock, recomputed: the Heaps rows of the validation set,
    its PEP 393 string sizes, and the rules behind the repeated 500 KB runs (same as ch04/02's today)."""
    recorded = recorded_bench()
    rows = bench.heaps_rows(VALID, bench.VALID_MARKS)
    assert rows == [r for r in recorded["heaps"] if r["file"] == VALID.name]
    assert (rows[-1]["occurrences"], rows[-1]["unique"]) == (sum(valid_counts.values()), len(valid_counts))
    text = VALID.read_bytes().decode("utf-8")
    assert bench.pep393_stats(text) == recorded["pep393"]["valid"]
    whole = next(r for r in recorded["memory"] if r["file"] == VALID.name and r["mode"] == "whole")
    assert ast.literal_eval(whole["info"]) == {"max_codepoint": 0x2026, "str_bytes": sys.getsizeof(text), "chars": len(text)}
    small = bench.slice_before_eot(VALID.read_bytes(), bench.SMALL_LIMIT).decode("utf-8")
    digest = bench.rules_digest(m.v02.train_bpe(small, 1000))
    assert set(recorded["repeat_small"]["rules_sha256"].values()) == {digest}


@pytest.mark.slow
def test_bench_full_valid_facts(m):
    """The whole-validation-set facts of bench.json, re-run (several minutes): ch04/02 and the lazy heap learn the same
    9,743 rules; the per-step counters (pairs max() scans, sequences an update touches, ids a naive trainer would
    scan); the cached encoder reproduces the official tiny_stories_valid.bin. Also TinyCodes' shrink factor."""
    recorded = recorded_bench()
    text = VALID.read_bytes().decode("utf-8")
    book = m.v02.train_bpe(text, 10000)
    rules, st = bench.merge_loop_profile(m.ext.ids_counts_of(text), 9743, count_ids=True)
    assert items(rules) == items(book) == items(m.ext.train_bpe_heap(text, 10000))
    assert bench.scanned_summary(st["scanned"]) == recorded["scanned"]
    a = sorted(st["affected"])
    assert (round(sum(a) / len(a), 2), a[len(a) // 2]) == (recorded["profile"]["affected_mean"],
                                                           recorded["profile"]["affected_median"]) == (7.05, 1)
    assert bench.shrink_summary(st["ids_occ"]) == recorded["shrink"]["tiny_stories_valid"]
    _, codes_st = bench.merge_loop_profile(m.ext.ids_counts_of(CODES.read_text(encoding="utf-8")), 743, count_ids=True)
    assert bench.shrink_summary(codes_st["ids_occ"]) == recorded["shrink"]["tiny_codes"]
    # Trained on the 22.5 MB validation set, the rules part ways with the official ones (trained on the 2.2 GB
    # train set) almost at once: training data decides the rules, so 4.5 uses the published merge_rules.pkl.
    first_diff = next(i for i, (x, y) in enumerate(zip(items(book), items(m.story_rules))) if x != y)
    assert first_diff == 5                                  # 5 rules agree, then ' the' and 'nd' swap places
    learned = m.sb.BPETokenizer(book).id_to_bytes
    assert (learned[256 + 5], m.sb.BPETokenizer(m.story_rules).id_to_bytes[256 + 5]) == (b" the", b"nd")
    assert m.ext.MemoTokenizer(m.story_rules).encode(text) == np.fromfile(VALID_BIN, dtype=np.uint16).tolist()


def test_cs336_a1_bpe_record():
    """4.2.4's CS336 box, read from the recorded A1 runs (web/data/cs336/a1_results.json, written by
    verify/cs336/run_a1.py; its speed_history keeps the timings of every recording session)."""
    rec = json.loads((ROOT / "web" / "data" / "cs336" / "a1_results.json").read_text(encoding="utf-8"))
    assert rec["dlfs6_commit"].startswith("c9b6e2e") and rec["cs336_a1"]["handout"].endswith("(v26.0.3)")
    shipped = rec["diagnostics"]["as_shipped"]["train_bpe"][0]
    fixed = rec["diagnostics"]["one_line_fixes"]["train_bpe"][0]
    assert (shipped["corpus"], shipped["vocab_size"], shipped["n_merges"]) == ("corpus.en", 500, 243)
    assert shipped["identical_positions"] == 170 and shipped["first_divergence"] + 1 == 93   # 0-based index 92
    assert fixed["identical_positions"] == 243 and fixed["exact"]                              # the bytes key: all equal
    span = lambda xs: f"{min(xs):.2f}–{max(xs):.2f}"
    first, second = rec["speed_history"][:2]
    assert (first["generated"][:10], second["generated"][:10]) == ("2026-10-11", "2026-10-12")
    # first session: the book's key straddles the 1.5 s bar; a bytes key that calls a helper in the key is too slow
    runs = first["train_seconds_runs"]
    assert (span(runs["as_shipped"]), span(runs["one_line_fixes"])) == ("1.27–1.52", "1.80–2.48")
    assert len(runs["as_shipped"]) == 5 and sum(t > 1.5 for t in runs["as_shipped"]) == 1
    assert "id_bytes(pair[0], merge_rules)" in first["bpe_tiebreak_after"]
    assert first["speed_test"] == {"as_shipped": "passed", "one_line_fixes": "failed"}
    # second session: the bytes key reads a vocab_bytes table built once per merge, as fast as the book's key
    runs = second["train_seconds_runs"]
    assert (span(runs["as_shipped"]), span(runs["one_line_fixes"])) == ("0.95–1.11", "0.98–1.11")
    assert "vocab_bytes[pair[0]]" in "".join(second["bpe_tiebreak_after"])
    assert second["speed_test"] == {"as_shipped": "passed", "one_line_fixes": "passed"}
    # Windows: the tokenizer tests import the Unix-only `resource` module; the runner supplies an empty stand-in
    assert '(tmp / "resource.py").write_text(' in (ROOT / "verify" / "cs336" / "run_a1.py").read_text(encoding="utf-8")


def test_pep393_piecewise_storage():
    """Split into stories, most pieces are 1 byte/char: 2.6% need wider storage, the pieces take ~half the memory."""
    text = VALID.read_bytes().decode("utf-8")
    pieces = text.split(EOT)
    wide = sum(1 for p in pieces if p and max(map(ord, p)) > 255)
    recorded = json.loads((DATA / "bench.json").read_text(encoding="utf-8"))["pep393"]["valid"]
    assert (len(pieces), wide, max(map(ord, text))) == (recorded["pieces"], recorded["wide_pieces"], 0x2026)
    assert round(wide / len(pieces), 3) == 0.026
    whole, parts = sys.getsizeof(text), sum(sys.getsizeof(p) for p in pieces)
    assert 0 < whole - 2 * len(text) < 100                         # 2 bytes per char + a small object header
    assert 0.5 < parts / whole < 0.55
