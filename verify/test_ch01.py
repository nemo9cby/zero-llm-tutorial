"""第 1 章 分词器：本地验证（B 层）。

正文里需要真实语料、PyTorch、tiktoken 或多进程的结论，都在这里用 third_party/dlfs6 的原书代码核对。
整份 TinyCodes 只预分词一次（模块级 fixture），所有测试共用。
"""
import ast
import contextlib
import importlib.util
import io
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from conftest import DLFS6, ROOT, run_dlfs6_script

END = "<|endoftext|>"
PAGE = ROOT / "content" / "ch01.md"


# ------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def book(dlfs6_path):
    """TinyCodes, merge_rules.pkl, tiny_codes.bin and the pre-token table (about 10 s, shared)."""
    from _ch01_data import load_book
    return load_book()


@pytest.fixture(scope="module")
def page_files(book):
    """The web/data/ch01 files, rebuilt in memory from the book's files."""
    from _ch01_data import build_files
    return build_files(book)


@pytest.fixture(scope="module")
def tie_runs(book):
    """743 merges on TinyCodes under each tie-break rule (ids = codebot, first = ch01/03-06, bytes = CS336)."""
    from _ch01_bpe import train_dedup
    return {rule: train_dedup(book["pieces"], book["counts"], 743, rule, book["n_end"])
            for rule in ("ids", "first", "bytes")}


def load_script(rel: str):
    """Import a ch01 demo script as a module; its example code runs once, silently (none of 03-06 writes files)."""
    spec = importlib.util.spec_from_file_location("dlfs6_" + Path(rel).stem, DLFS6 / rel)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stderr(io.StringIO()):     # tqdm progress bars
        return fn(*args, **kwargs)


def vocab_of(merges):
    v = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(merges):
        v[256 + k] = v[a] + v[b]
    return v


def first_difference(xs, ys):
    return next((i for i, (x, y) in enumerate(zip(xs, ys)) if x != y), None)


# ------------------------------------------------------------------------------------- 1.1 / 1.2
def test_char_and_byte_scripts_match_book_comments():
    # 原书脚本里，每个 print 后面的注释写着预期输出；这里逐行核对。
    assert run_dlfs6_script("ch01/01_char_tokenizer.py") == [
        "['h', 'e', 'l', 'l', 'o', '世', '界', '😁']",
        "104",
        "128513",
        "h",
        "😁",
        "[104, 101, 108, 108, 111, 19990, 30028, 128513]",
        "[104, 101, 108, 108, 111, 19990, 30028, 128513]",
        "hello世界😁",
    ]
    assert run_dlfs6_script("ch01/02_byte_tokenizer.py") == [
        "b'A'",
        "[65]",
        "b'\\xe3\\x81\\x82'",
        "[227, 129, 130]",
        "A",
        "[104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]",
        "hello世界😁",
    ]


def test_codebot_size_and_context(dlfs6_path):
    # ch03/01_pretrain.py 的超参数（只解析，不运行），以及用它们建出的 GPT 的参数量
    tree = ast.parse((DLFS6 / "ch03" / "01_pretrain.py").read_text(encoding="utf-8"))
    hp = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    literal = {k: ast.literal_eval(hp[k]) for k in ("context_len", "vocab_size", "embed_dim", "n_head", "n_layer")}
    assert literal == {"context_len": 256, "vocab_size": 1000, "embed_dim": 384, "n_head": 6, "n_layer": 6}
    assert ast.unparse(hp["ff_dim"]) == "4 * embed_dim"

    import torch
    from codebot.model import GPT
    torch.manual_seed(0)
    model = GPT(vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6, ff_dim=4 * 384,
                dropout_rate=0.1)
    assert sum(p.numel() for p in model.parameters()) == 11_121_640
    assert tuple(model.embed.weight.shape) == (1000, 384) and model.unembed.out_features == 1000
    assert model.max_context_len == 256


# ------------------------------------------------------------------------------------- 1.3–1.6
def test_bpe_scripts_print_what_the_text_says():
    assert run_dlfs6_script("ch01/03_bpe_train.py") == [
        "defaultdict(<class 'int'>, {(1, 2): 2, (2, 3): 1, (3, 1): 1})",
        "[4, 3, 4]",
        "{(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}",
    ]
    assert run_dlfs6_script("ch01/04_bpe_tokenizer.py") == [
        "[259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]",
        "Hello世界😁",
    ]
    assert run_dlfs6_script("ch01/05_special_token.py") == [
        "{(105, 115): 256, (256, 32): 257, (105, 110): 258}",
        "[72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 259]",
        "Hello world!<|endoftext|>",
    ]
    assert run_dlfs6_script("ch01/06_pretokenize.py") == [
        "[262, 260, 33]",
        "Say hello!",
        "262 -> 'Say'",
        "260 -> ' hello'",
        "33 -> '!'",
    ]


def test_dedup_trainer_matches_the_book_trainers(book):
    """verify/_ch01_bpe.train_dedup == ch01/06 (tie-break 'first') and codebot (tie-break 'ids'), merge for merge."""
    from codebot import tokenizer as cb
    from _ch01_bpe import pretoken_table, train_dedup
    ch06 = load_script("ch01/06_pretokenize.py")
    text = book["text"]
    for sample, vocab_size in [("Say hello! Why hello? Just hello.<|endoftext|>Good morning!", 270),
                               (text[:4000], 400), (text[:20000], 330)]:
        pieces, counts, n_end = pretoken_table(sample, cb.pretokenize)
        assert train_dedup(pieces, counts, vocab_size - 257, "first", n_end).merges == \
            list(quiet(ch06.train_bpe, sample, vocab_size))
        assert train_dedup(pieces, counts, vocab_size - 257, "ids", n_end).merges == \
            list(quiet(cb.train_bpe, sample, vocab_size))


def test_tie_breaks_on_the_cs336_fixture(dlfs6_path):
    """CS336 A1's own train_bpe fixture (corpus.en, vocab 500, special <|endoftext|>): 243 reference merges."""
    from storybot import tokenizer as sb
    from codebot import tokenizer as cb
    from _ch01_bpe import pretoken_table, train_dedup
    fixture = ROOT / "tests" / "js" / "fixtures" / "lib" / "cs336.json"
    cs = json.loads(fixture.read_text(encoding="utf-8"))
    text = cs["text"]
    reference = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in cs["merges_hex"]]
    assert len(text) == 132_878 and len(reference) == 243

    def against_reference(merges):
        """(first differing merge, 1-based; positions equal; our pair; the reference pair), compared as bytes."""
        v = vocab_of(merges)
        as_bytes = [(v[a], v[b]) for a, b in merges]
        k = first_difference(as_bytes, reference)
        return k + 1, sum(x == y for x, y in zip(as_bytes, reference)), as_bytes[k], reference[k]

    # codebot's rule (count, id0, id1), with the book's own trainer of ch04, on a file, like CS336's run_train_bpe
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corpus.en"
        path.write_bytes(text.encode("utf-8"))
        book = list(quiet(sb.train_bpe, str(path), 500, num_processes=2, num_chunks=4))
    # ch01/03-06's rule (first wins, the line CS336's lecture 1 also uses), with the book's own trainer of ch01/06
    first_wins = list(quiet(load_script("ch01/06_pretokenize.py").train_bpe, text, 500))
    for name, merges in [("ids (codebot)", book), ("first wins (ch01/06)", first_wins)]:
        k, same, ours, theirs = against_reference(merges)
        print(f"{name}: first difference at merge #{k}: {ours} vs reference {theirs}; {same}/243 positions equal")
    assert against_reference(book)[:2] == (93, 170)
    assert against_reference(first_wins)[:2] == (65, 159)

    # the same loop with only the tie-break key changed to compare bytes: all 243 merges equal the reference
    pieces, counts, n_end = pretoken_table(text, cb.pretokenize)
    assert train_dedup(pieces, counts, 243, "ids", n_end).merges == book
    assert train_dedup(pieces, counts, 243, "first", n_end).merges == first_wins
    by_bytes = train_dedup(pieces, counts, 243, "bytes", n_end)
    assert [(by_bytes.vocab[a], by_bytes.vocab[b]) for a, b in by_bytes.merges] == reference


def test_tie_breaks_on_tiny_codes(book, tie_runs):
    official = book["merges"]
    runs = tie_runs
    assert runs["ids"].merges == official                          # the book's rule reproduces merge_rules.pkl
    ties = sum(t > 1 for t in runs["ids"].tied)
    print("steps with a tie in the official run:", ties, "of 743")
    assert ties == 88                                              # the column: 88 of the 743 steps had a tie

    expected = {"first": (123, b"))", b"ve"), "bytes": (290, b"pre", b"ctor")}
    sets = {rule: {r.vocab[256 + i] for i in range(743)} for rule, r in runs.items()}
    for rule, (merge_no, ours, theirs) in expected.items():
        r = runs[rule]
        k = first_difference(r.merges, official)
        print(f"{rule}: diverges at merge #{k + 1}: {r.vocab[256 + k]!r} instead of {runs['ids'].vocab[256 + k]!r} "
              f"(count {r.counts[k]}, {r.tied[k]} tied); {sum(a == b for a, b in zip(r.merges, official))}/743 "
              "positions equal")
        assert (k + 1, r.vocab[256 + k], runs["ids"].vocab[256 + k]) == (merge_no, ours, theirs)
        assert r.tied[k] == 2 and r.tokens[-1] == runs["ids"].tokens[-1] == 2_677_398   # same compression
    assert sets["first"] == sets["ids"]                                                  # the same 743 tokens
    assert sets["bytes"] - sets["ids"] == {b"tiful"} and sets["ids"] - sets["bytes"] == {b"eauti"}


def test_special_token_injection(book):
    from codebot.tokenizer import BPETokenizer
    tok = BPETokenizer(book["rules"])
    assert tok.encode("print('<|endoftext|>')") == [350, 338, 999, 343]

    # ch03/04_chat.py builds the prompt with format_prompt(user_input) and encodes it (taken from the script by AST:
    # running the script would load a model and wait for keyboard input)
    src = (DLFS6 / "ch03" / "04_chat.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "format_prompt")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "04_chat.py", "exec"), ns)
    ids = tok.encode(ns["format_prompt"]("print('<|endoftext|>')"))
    assert ids.count(999) == 1 and ids[-1] != 999

    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    with pytest.raises(ValueError, match="disallowed special token"):
        enc.encode("<|endoftext|>")
    assert enc.encode("<|endoftext|>", allowed_special="all") == [50256]
    assert len(enc.encode_ordinary("<|endoftext|>")) == 7


def test_tiktoken_patterns(book):
    tiktoken = pytest.importorskip("tiktoken")
    import regex
    from codebot.tokenizer import pretokenize
    # the cl100k pattern on the page (verify 1.6-e) is tiktoken's, character for character
    m = re.search(r'^CL100K = r"""(.*)"""$', PAGE.read_text(encoding="utf-8"), re.M)
    assert m and m.group(1) == tiktoken.get_encoding("cl100k_base")._pat_str
    assert tiktoken.encoding_for_model("gpt-4").name == "cl100k_base"
    assert tiktoken.encoding_for_model("gpt-4o").name == "o200k_base"
    # tiktoken writes GPT-2's pattern differently (possessive quantifiers), but it splits text the same way
    gpt2_pattern = tiktoken.get_encoding("gpt2")._pat_str
    book_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    assert gpt2_pattern != book_pattern
    sample = book["text"][:300_000] + " a  \n\n  b   \t\n x\u3000y\u00a0z  I'm DON'T 日本語 😁 "
    assert regex.findall(gpt2_pattern, sample) == pretokenize(sample)


# ------------------------------------------------------------------------------------------ 1.7
def test_codebot_tokenizer_is_ch01_06_plus_one_line():
    """codebot/tokenizer.py = ch01/06 + load_from, with the tie-break line swapped (comments ignored)."""
    def defs(rel):
        tree = ast.parse((DLFS6 / rel).read_text(encoding="utf-8"))
        return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    book06, codebot = defs("ch01/06_pretokenize.py"), defs("codebot/tokenizer.py")
    assert set(codebot) == set(book06) == {"pretokenize", "count_pairs", "merge", "train_bpe", "BPETokenizer"}
    for name in ("pretokenize", "count_pairs", "merge"):
        assert ast.dump(codebot[name]) == ast.dump(book06[name]), name

    def lines(node):                       # the code of a definition, without comments and blank lines
        return [re.sub(r"\s*#.*$", "", l).rstrip() for l in ast.unparse(node).splitlines() if l.strip()]
    removed = [l for l in lines(book06["train_bpe"]) if l not in lines(codebot["train_bpe"])]
    added = [l for l in lines(codebot["train_bpe"]) if l not in lines(book06["train_bpe"])]
    assert [l.strip() for l in removed] == ["best_pair = max(counts, key=counts.get)"]
    assert [l.strip() for l in added] == ["best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))"]

    methods = lambda cls: {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    m06, mcb = methods(book06["BPETokenizer"]), methods(codebot["BPETokenizer"])
    assert set(mcb) - set(m06) == {"load_from"} and all(mcb[k] == m06[k] for k in m06)

    # ch03: 5 of the 6 scripts import the tokenizer from here; 01_pretrain.py reads the encoded tiny_codes.bin instead
    sources = {p.name: p.read_text(encoding="utf-8") for p in (DLFS6 / "ch03").glob("*.py")}
    importers = {name for name, src in sources.items() if "from codebot.tokenizer import BPETokenizer" in src}
    assert len(sources) == 6 and set(sources) - importers == {"01_pretrain.py"}
    assert "codebot.tokenizer" not in sources["01_pretrain.py"] and "tiny_codes.bin" in sources["01_pretrain.py"]


def test_first_wins_breaks_in_the_ch04_trainer(tie_runs):
    """1.7.2: ch04's trainer deletes and re-inserts pairs, so with ch01/06's first-wins line it learns other merges."""
    from storybot import tokenizer as sb
    src = (DLFS6 / "storybot" / "tokenizer.py").read_text(encoding="utf-8")
    active = "best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))"
    assert src.count(active) == 1 and "del pair_counts[pair]" in src
    ns = {"__name__": "storybot_first_wins"}
    exec(compile(src.replace(active, "best_pair = max(pair_counts, key=pair_counts.get)"), "storybot_first_wins",
                 "exec"), ns)
    for name, obj in list(ns.items()):          # keep the real module's workers: multiprocessing must pickle them
        if callable(obj) and name != "train_bpe" and hasattr(sb, name):
            ns[name] = getattr(sb, name)
    corpus = str(DLFS6 / "codebot" / "tiny_codes.txt")
    fast_first = list(quiet(ns["train_bpe"], corpus, 1000, num_processes=8, num_chunks=64))
    naive_first = tie_runs["first"].merges        # == ch01/06 (test_dedup_trainer_matches_the_book_trainers)
    k = first_difference(fast_first, naive_first)
    print(f"first-wins: ch04 trainer vs ch01/06 differ from merge #{k + 1}; "
          f"{sum(a == b for a, b in zip(fast_first, naive_first))}/743 positions equal")
    assert k + 1 == 302 and set(fast_first) != set(naive_first)
    # with the (count, id0, id1) rule the same trainer reproduces merge_rules.pkl (see test_training_cost_naive_vs_fast)


def test_naive_run_record():
    """1.7.3: the recorded full run of ch01/07 (verify/_ch01_naive_run.py) matches the repository it ran on,
    and the page quotes it (minutes, seconds per merge, CPU)."""
    import hashlib
    import pickle
    if not shutil.which("git"):
        pytest.skip("git not found")
    record = json.loads((ROOT / "web" / "data" / "ch01" / "naive_run.json").read_text(encoding="utf-8"))
    sha = lambda data: hashlib.sha256(data).hexdigest()
    repo_pkl = (DLFS6 / "codebot" / "merge_rules.pkl").read_bytes()
    head = subprocess.run(["git", "-C", str(DLFS6), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert record["dlfs6_commit"] == head
    assert record["script_sha256"] == sha((DLFS6 / record["script"]).read_bytes())
    assert record["output_sha256"] == record["repo_sha256"] == sha(repo_pkl)
    assert record["byte_identical"] and record["same_rules_same_order"] and record["merges"] == 743
    assert "743/743" in record["tqdm_last_line"]
    python = tuple(int(x) for x in record["host"]["python"].split(".")[:2])
    assert record["pickle_default_protocol"] == 4 and python < (3, 14)
    minutes, per_merge = record["seconds"] / 60, record["seconds"] / 743
    print(f"{record['seconds']:,.0f} s = {minutes:.1f} min, {per_merge:.2f} s per merge on {record['host']['cpu']}")

    # 1.7.3 quotes the record: "743 步一共用了 M 分钟，平均每步 S 秒（<CPU> 单核", and M again in this test's title
    page = PAGE.read_text(encoding="utf-8")
    quoted = re.search(r"743 步一共用了 (\d+) 分钟，平均每步 ([\d.]+) 秒（(\S+) 单核", page)
    assert quoted, "1.7.3 no longer quotes the recorded run"
    assert (int(quoted.group(1)), quoted.group(2)) == (round(minutes), f"{per_merge:.1f}"), \
        "1.7.3 quotes another run: update the numbers in content/ch01.md to the record"
    assert quoted.group(3) in record["host"]["cpu"]
    assert f"naive_run.json）：{round(minutes)} 分钟" in page

    # "byte identical" needs pickle protocol 4: the default from Python 3.8 to 3.13 (DLFS6 asks for >=3.10);
    # 3.14's default, protocol 5, writes the same rules with a different second byte (the protocol number)
    assert 'requires-python = ">=3.10"' in (DLFS6 / "pyproject.toml").read_text(encoding="utf-8")
    rules = pickle.loads(repo_pkl)
    assert pickle.dumps(rules, protocol=4) == repo_pkl
    p5 = pickle.dumps(rules, protocol=5)
    assert len(p5) == len(repo_pkl) and [i for i, (a, b) in enumerate(zip(p5, repo_pkl)) if a != b] == [1]
    assert list(pickle.loads(p5).items()) == list(rules.items())


def test_training_cost_naive_vs_fast(book):
    """One step of the book's naive loop on the whole corpus vs the whole run of ch04's trainer."""
    from codebot import tokenizer as cb
    from storybot import tokenizer as sb
    official = book["merges"]
    # every occurrence of every pre-token, as codebot.train_bpe keeps them (order does not matter for counting)
    ids_list = [list(p.encode("utf-8")) for p, c in zip(book["pieces"], book["counts"]) for _ in range(c)]
    assert len(ids_list) == 1_826_876
    t0 = time.perf_counter()
    counts = defaultdict(int)
    for ids in ids_list:
        counts = cb.count_pairs(ids, counts)
    best = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))
    ids_list = [cb.merge(ids, best, 256) for ids in ids_list]
    naive_step = time.perf_counter() - t0
    assert best == official[0] == (32, 32) and counts[best] == 335_027

    t0 = time.perf_counter()
    fast = quiet(sb.train_bpe, str(DLFS6 / "codebot" / "tiny_codes.txt"), 1000, num_processes=8, num_chunks=64)
    fast_seconds = time.perf_counter() - t0
    assert list(fast.items()) == list(book["rules"].items())
    print(f"naive: {naive_step:.1f} s for one merge -> about {naive_step * 743 / 60:.0f} min for 743; "
          f"ch04's trainer: {fast_seconds:.1f} s for all 743 merges ({naive_step * 743 / fast_seconds:.0f}x faster)")
    # 1.7.3: one naive step takes seconds; ch04's trainer learns all 743 merges in under a minute, 100x+ faster
    # (wall clock: 6 s on an idle machine, 17-23 s while other jobs ran; the 60 s bound leaves room for load)
    assert naive_step > 1 and fast_seconds < 60 and naive_step * 743 > 100 * fast_seconds


def test_crlf_checkout_changes_the_second_merge(book):
    from storybot import tokenizer as sb
    if not shutil.which("git"):
        pytest.skip("git not found")
    # the repository stores tiny_codes.txt with LF; only core.autocrlf=true turns it into CRLF on checkout.
    # `git cat-file --filters` applies the same conversion as a checkout, so no clone is needed.
    git = lambda *args: subprocess.run(["git", "-C", str(DLFS6), *args], capture_output=True, check=True).stdout
    assert git("ls-files", "--eol", "codebot/tiny_codes.txt").decode().startswith("i/lf")
    checkout = {v: git("-c", f"core.autocrlf={v}", "cat-file", "--filters", "HEAD:codebot/tiny_codes.txt")
                for v in ("true", "input", "false")}
    lf = book["text"].encode("utf-8")
    assert checkout["input"] == checkout["false"] == lf and b"\r" not in lf
    assert checkout["true"] == book["text"].replace("\n", "\r\n").encode("utf-8")
    with tempfile.TemporaryDirectory() as d:
        crlf = Path(d) / "tiny_codes.txt"
        crlf.write_bytes(checkout["true"])                                      # what core.autocrlf=true checks out
        assert crlf.stat().st_size == 6_743_534
        rules = quiet(sb.train_bpe, str(crlf), 260, num_processes=8, num_chunks=64)   # ch04 reads bytes
        assert list(rules) == [(32, 32), (13, 10), (256, 256)]
        assert book["merges"][:2] == [(32, 32), (256, 256)]
        with open(crlf, encoding="utf-8") as f:                                  # text mode, like ch01/07
            assert f.read() == book["text"]


def test_tinystories_cannot_be_read_as_gbk():
    path = DLFS6 / "storybot" / "tiny_stories_valid.txt"
    if not path.is_file():
        pytest.skip("storybot/tiny_stories_valid.txt not downloaded (uv run python tools/fetch_data.py)")
    with open(path, "rb") as f:
        raw = f.read(200_000)
    with pytest.raises(UnicodeDecodeError) as err:
        raw.decode("gbk")
    start = err.value.start
    assert start == 30_846 and raw[start - 2:start + 1] == "”".encode("utf-8")


# ------------------------------------------------------------------------------------------ 1.8
def test_the_first_10k_chars_are_not_a_fair_sample(book):
    text = book["text"]
    lines = text.split("\n")
    deep = sum(line.startswith(" " * 8) for line in lines)
    print(f"{deep:,} of {len(lines):,} lines are indented by 8+ spaces ({deep / len(lines):.1%})")
    assert (len(lines), deep) == (256_502, 31_200)
    assert round(text.count("input(") / (len(text) / 1000), 2) == 0.16
    assert text[:10000].count("input(") == 53


def test_the_head_is_an_unlucky_sample(book):
    """1.8.2: 300 random 10k-character windows put the head near the bottom; dedenting closes only about half the gap."""
    from statistics import median
    from storybot.tokenizer import BPETokenizer as RankTokenizer   # == codebot's encoder, id for id (1.9's test)
    from _ch01_bpe import pretoken_table
    rank = RankTokenizer(book["rules"])
    cache = {}

    def ratio(sample):
        """bytes/token of CodeBot's encode(sample): <|endoftext|> is one token, every pre-token is encoded alone"""
        pieces, counts, n_end = pretoken_table(sample, book["pretokenize"])
        n = n_end
        for p, c in zip(pieces, counts):
            if p not in cache:
                cache[p] = len(rank._encode_text(p))
            n += cache[p] * c
        return len(sample.encode("utf-8")) / n

    text = book["text"]
    head, full = ratio(text[:10000]), ratio(text)
    assert (f"{head:.2f}", f"{full:.2f}") == ("2.12", "2.42")
    rng = random.Random(0)
    windows = [ratio(text[s:s + 10000]) for s in (rng.randrange(0, len(text) - 10000) for _ in range(300))]
    higher = sum(r > head for r in windows)
    print(f"300 random windows: min {min(windows):.3f}, median {median(windows):.3f}, max {max(windows):.3f}; "
          f"{higher} of them compress better than the head ({head:.3f})")
    assert f"{median(windows):.2f}" == "2.45" and higher == 290

    dedent = lambda t: "\n".join(line.lstrip(" \t") for line in t.split("\n"))
    flat_head, flat_full = ratio(dedent(text[:10000])), ratio(dedent(text))
    print(f"without indentation: head {flat_head:.4f}, full {flat_full:.4f}; gap {flat_full - flat_head:.4f} "
          f"(was {full - head:.4f})")
    assert (f"{flat_head:.2f}", f"{flat_full:.2f}", f"{flat_full - flat_head:.2f}") == ("2.08", "2.21", "0.13")
    assert 0.4 < (flat_full - flat_head) / (full - head) < 0.6        # indentation explains about half of the gap


def test_other_tokenizers_on_tiny_codes(book, page_files):
    """The table of 1.8.3, recomputed (tiktoken encodings + the book's StoryBot/WebBot) == compression.json."""
    pytest.importorskip("tiktoken")
    data = json.loads(page_files["compression.json"])
    table = {}
    for sample in ("head", "full"):
        s = data[sample]
        table[sample] = {"CodeBot": round(s["bytes"] / s["tokens"][-1], 2)}
        table[sample].update({r["name"]: round(s["bytes"] / r["tokens"], 2) for r in s["refs"]})
    assert table == {
        "head": {"CodeBot": 2.12, "storybot": 2.18, "webbot": 3.05, "GPT-2": 3.09, "cl100k_base": 3.97,
                 "o200k_base": 3.98},
        "full": {"CodeBot": 2.42, "storybot": 2.06, "webbot": 2.60, "GPT-2": 2.64, "cl100k_base": 3.86,
                 "o200k_base": 3.84},
    }
    vocabs = {r["name"]: r["vocab"] for r in data["full"]["refs"]}
    assert vocabs == {"GPT-2": 50_257, "cl100k_base": 100_277, "o200k_base": 200_019, "storybot": 10_000,
                      "webbot": 50_000}
    assert data["full"]["bytes"] == len(book["text"].encode("utf-8"))


def test_indentation_gpt2_vs_cl100k_vs_codebot(book):
    tiktoken = pytest.importorskip("tiktoken")
    from codebot.tokenizer import BPETokenizer
    code = "def f():\n        return 1\n"
    gpt2, cl100k = tiktoken.get_encoding("gpt2"), tiktoken.get_encoding("cl100k_base")
    codebot = BPETokenizer(book["rules"])
    pieces = {name: [enc.decode([i]) for i in enc.encode(code)] for name, enc in
              [("gpt2", gpt2), ("cl100k", cl100k), ("codebot", codebot)]}
    for name, p in pieces.items():
        print(f"{name:8} {len(p):2} {p}")
    assert {k: len(v) for k, v in pieces.items()} == {"gpt2": 14, "cl100k": 8, "codebot": 7}
    assert pieces["gpt2"].count(" ") == 7
    # GPT-2's vocabulary has no token made of two or more spaces only
    assert not [t for t in gpt2._mergeable_ranks if len(t) >= 2 and set(t) <= {32}]

    # 1.8.3, third point: without the whitespace-only pre-tokens (indentation, newlines) GPT-2 still leads by much more
    # on the head than on the whole corpus, so indentation is only part of why its lead shrinks from 0.97 to 0.21.
    # Both tokenizers split text with the same regex and merge inside pre-tokens only, so each pre-token is encoded alone.
    from storybot.tokenizer import BPETokenizer as RankTokenizer
    from _ch01_bpe import pretoken_table
    rank = RankTokenizer(book["rules"])

    def lead(pieces, counts):
        words = [(p, c) for p, c in zip(pieces, counts) if not p.isspace()]
        nbytes = sum(len(p.encode("utf-8")) * c for p, c in words)
        return (nbytes / sum(len(gpt2.encode_ordinary(p)) * c for p, c in words)
                - nbytes / sum(len(rank._encode_text(p)) * c for p, c in words))
    head_pieces, head_counts, _ = pretoken_table(book["text"][:10000], book["pretokenize"])
    leads = lead(head_pieces, head_counts), lead(book["pieces"], book["counts"])
    print(f"GPT-2's lead over CodeBot on pre-tokens that are not whitespace: head {leads[0]:.3f}, full {leads[1]:.3f}")
    assert [f"{x:.2f}" for x in leads] == ["1.23", "0.82"]


# ------------------------------------------------------------------------------------------ 1.9
def test_numpy1_silently_wraps_uint16():
    """In NumPy 1.26 (the version DLFS6's uv.lock pins) np.array(..., dtype=np.uint16) only warns and wraps."""
    lock = (DLFS6 / "uv.lock").read_text(encoding="utf-8")
    assert re.search(r'name = "numpy"\nversion = "1\.26\.4"', lock)
    if not shutil.which("uv"):
        pytest.skip("uv not found")
    code = ("import numpy as np, warnings\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    a = np.array([65535, 65536, 70000], dtype=np.uint16)\n"
            "print(np.__version__, a.tolist(), sorted({x.category.__name__ for x in w}))\n")
    proc = subprocess.run(["uv", "run", "--isolated", "--no-project", "--python", "3.12", "--with", "numpy==1.26.4",
                           "python", "-I", "-c", code], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 and "No solution" in proc.stderr + proc.stdout:
        pytest.skip("numpy 1.26.4 not installable here")
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert proc.stdout.strip() == "1.26.4 [65535, 0, 4464] ['DeprecationWarning']"
    # the NumPy in this environment (2.x) raises instead
    with pytest.raises(OverflowError):
        np.array([65535, 65536], dtype=np.uint16)


def test_tiny_codes_bin_is_the_encoded_corpus(book):
    from codebot.tokenizer import BPETokenizer as BookTokenizer          # ch01: every merge, in training order
    from storybot.tokenizer import BPETokenizer as RankTokenizer         # ch04: lowest-rank pair first
    text, rules, binary = book["text"], book["rules"], book["bin"]
    book_tok, rank_tok = BookTokenizer(rules), RankTokenizer(rules)

    # 1) the whole corpus, encoded piece by piece (same ids as encode(text): a piece's ids never depend on context)
    per_piece = [rank_tok._encode_text(p) for p in book["pieces"]]
    ids = []
    for k in book["sequence"]:
        ids.extend(per_piece[k] if k >= 0 else (999,))
    assert len(ids) == binary.size == 2_677_398 and ids == binary.tolist()
    assert binary.nbytes == (DLFS6 / "codebot" / "tiny_codes.bin").stat().st_size == 5_354_796

    # 2) ch04's rank-based encoder == the book's encoder: the 10k head in full, plus 2,000 random distinct pieces
    head_ids = book_tok.encode(text[:10000])
    assert head_ids == binary[:4709].tolist()
    for i in random.Random(0).sample(range(len(book["pieces"])), 2000):
        assert book_tok._encode_text(book["pieces"][i]) == per_piece[i]

    # 3) decoding the .bin gives the text back; 999 marks every <|endoftext|>
    assert book_tok.decode(binary.tolist()) == text
    assert int((binary == 999).sum()) == text.count(END) == 21_835
    assert round(len(text.encode("utf-8")) / binary.size, 4) == 2.4229
    assert round(10_000 / len(head_ids), 2) == 2.12
    # 1.8.1: every "bonacci" in the corpus belongs to a "fibonacci"
    assert text.count("bonacci") == text.lower().count("fibonacci") > 0


def test_initial_loss_is_about_ln_1000(book):
    """A freshly initialized CodeBot (ch03's hyperparameters) on real windows of tiny_codes.bin: loss ≈ ln 1000."""
    import math
    import torch
    import torch.nn.functional as F
    from codebot.model import GPT
    torch.manual_seed(0)
    model = GPT(vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6, ff_dim=4 * 384,
                dropout_rate=0.1).eval()
    data = book["bin"].astype(np.int64)
    starts = np.random.default_rng(0).integers(0, len(data) - 257, 8)
    x = torch.from_numpy(np.stack([data[s:s + 256] for s in starts]))
    y = torch.from_numpy(np.stack([data[s + 1:s + 257] for s in starts]))
    with torch.no_grad():
        loss = F.cross_entropy(model(x).reshape(-1, 1000), y.reshape(-1)).item()
    print(f"initial loss {loss:.4f}   ln 1000 = {math.log(1000):.4f}")
    assert 6.9 < loss < 7.1 and abs(loss - math.log(1000)) < 0.15


def test_page_data_is_derived_from_the_book_files(book, page_files):
    out = ROOT / "web" / "data" / "ch01"
    stale = [name for name, data in page_files.items() if not (out / name).is_file() or (out / name).read_bytes() != data]
    assert not stale, f"web/data/ch01 is stale: {stale} (run: uv run python verify/_ch01_data.py)"
    counts = json.loads(page_files["token_counts.json"])["counts"]
    assert counts == np.bincount(book["bin"], minlength=1000).tolist()
    assert page_files["tiny_codes_head.bin"] == book["bin"][:4709].astype("<u2").tobytes()
    table = json.loads(page_files["tinycodes_pretokens.json"])
    assert Counter(dict(table["pieces"])) == Counter(dict(zip(book["pieces"], book["counts"])))
