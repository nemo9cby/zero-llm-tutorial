"""第 4 章 分词器【应用篇】：本地验证。

全部断言都针对 third_party/dlfs6 里的真实代码。ch04 的脚本顶层会 chdir 并在 2.2 GB 语料上训练，
所以用 _ch04_load.load_defs 只执行其中的 import / def / class（函数体就是原书代码）。
多进程测试用 storybot.tokenizer 里的工作函数（可以被 spawn 出来的子进程 import）。
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import pickle
import re
import subprocess
import sys
import tracemalloc
from collections import Counter, defaultdict
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import DLFS6, ROOT
from _ch04_load import load_defs, quiet_tqdm
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


# ---------------------------------------------------------------- 4.3 chunking

def test_gbk_default_encoding_breaks_on_tinystories():
    """What `open(path).read()` does on Chinese Windows without UTF-8 mode (locale encoding cp936 = GBK)."""
    raw = VALID.read_bytes()
    with pytest.raises(UnicodeDecodeError) as err:
        raw.decode("gbk")
    assert err.value.start == 30_846 and raw[30_846] == 0x9D
    assert raw[30_844:30_847].decode("utf-8") == "”"          # the right curly quote ” = e2 80 9d
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
    assert data.count(EOT.encode()) == 27_630
    data.decode("utf-8")      # valid UTF-8, so errors="ignore" drops nothing at '<' boundaries
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


# ---------------------------------------------------------------- 4.4 parallel

def run_killing_tree(cmd, timeout):
    """subprocess.run with a timeout that also kills grandchildren (pool workers) on Windows/POSIX."""
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", **kw)
    try:
        out, err = p.communicate(timeout=timeout)
        return False, p.returncode, out, err
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, 9)
        out, err = p.communicate()
        return True, p.returncode, out, err


def test_spawn_needs_importable_worker():
    head = "import multiprocessing as mp\ndef square(x):\n    return x * x\nif __name__ == '__main__':\n" \
           "    mp.set_start_method('spawn')\n"
    # A Process whose target lives in `python -c` code: the child cannot find it and exits with code 1.
    hung, rc, out, err = run_killing_tree(
        [sys.executable, "-c", head + "    p = mp.Process(target=square, args=(3,)); p.start(); p.join()\n"
         "    print(p.exitcode)\n"], timeout=60)
    assert not hung and out.strip() == "1"
    assert "Can't get attribute 'square' on <module '__main__'" in err
    # The same function in a Pool: the workers die on every task and the parent waits forever.
    hung, rc, out, err = run_killing_tree(
        [sys.executable, "-c", head + "    with mp.Pool(2) as pool:\n        print(list(pool.imap(square, range(4))))\n"],
        timeout=10)
    assert hung and out == ""
    assert "Can't get attribute 'square' on <module '__main__'" in err


def test_parallel_pretokenize_equals_serial(m, valid_counts):
    """ch04/04 steps 1-3 with Pool(4) (spawn on Windows) give exactly the serial counts."""
    b = m.sb.find_chunk_boundaries(str(VALID), 64)
    tasks = [(str(VALID), s, e, EOT) for s, e in zip(b[:-1], b[1:])]
    total = Counter()
    with Pool(processes=4) as pool:
        for part in pool.imap(m.sb.pretoken_chunk, tasks):
            total.update(part)
    assert total == valid_counts


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
    assert partial == [696, 4606] and tok.id_to_bytes[696] == b"\xe2\x80"
    assert max(range(256, 9999), key=lambda i: len(tok.id_to_bytes[i])) == 7159
    # no Chinese anywhere: not in the validation text, not in any learned token
    assert not any("一" <= c <= "鿿" for c in VALID.read_text(encoding="utf-8"))   # CJK unified ideographs
    assert not any(0xE4 <= b <= 0xE9 for i in range(256, 9999) for b in tok.id_to_bytes[i])
    assert sum(any(b >= 0x80 for b in tok.id_to_bytes[i]) for i in range(256, 9999)) == 20


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
    per_task = len(pickle.dumps(tok._encode_chunk))                # what Pool.imap ships for every task
    assert 200_000 < per_task < 250_000


def test_encode_file_matches_official_bin(m, tmp_path):
    src = tmp_path / "valid_2mb.txt"
    src.write_bytes(prefix_at_eot(VALID.read_bytes(), 2_000_000))
    out = tmp_path / "valid_2mb.bin"
    total = m.sb.BPETokenizer(m.story_rules).encode_file(str(src), str(out), num_processes=2,
                                                         cache_dir=str(tmp_path / "cache"))
    official = np.fromfile(VALID_BIN, dtype=np.uint16)
    assert out.read_bytes() == official[:total].tobytes() and int(official[total]) == 9999


def test_uint16_limits():
    assert np.iinfo(np.uint16).max == 65535
    with pytest.raises(OverflowError):
        np.array([65536], dtype=np.uint16)
    assert np.array([65535, 65536, 70000]).astype(np.uint16).tolist() == [65535, 0, 4464]
    if sys.maxsize > 2**32:                                      # 64-bit CPython: 8-byte pointers per list slot
        assert (sys.getsizeof([0] * 1000) - sys.getsizeof([0] * 0)) / 1000 == 8


# ---------------------------------------------------------------- recorded measurements

def test_bench_json_consistent():
    """The measurements quoted on the page: every timed variant produced identical output."""
    bench = json.loads((DATA / "bench.json").read_text(encoding="utf-8"))
    for group in bench["races"]:
        for row in group["rows"]:
            assert row["identical"] is True, (group["id"], row["name"])
    assert bench["profile"]["max_share"] > 0.9 and bench["profile"]["affected_median"] == 1
    assert bench["pretok"]["identical"] and all(r["identical"] for r in bench["encode_file"])
    assert bench["train_prefix_encode"]["identical_to_official_train_bin_prefix"]
    assert (bench["scanned"]["first"], bench["scanned"]["max"]) == (932, 12_025)


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
