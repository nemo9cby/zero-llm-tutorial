"""第 7 章 分词器【挑战篇】：本地验证（B 层）。

全部断言都针对 third_party/dlfs6 里的真实代码与官方产物（webbot/merge_rules.pkl、webbot/owt_valid.bin）。
OWT 的文本由官方 owt_valid.bin 解码得到（解码是无损的，见 test_official_valid_bin_facts），不必下载 12 GB 语料。
ch07 的两个脚本会在 12 GB 语料上训练、编码，这里不运行脚本，只 import storybot.tokenizer 里的函数。

    uv run pytest verify/test_ch07.py
"""
from __future__ import annotations

import ast
import builtins
import cProfile
import inspect
import json
import pstats
import time
from collections import Counter

import numpy as np
import pytest

from conftest import DLFS6, ROOT
from _ch07_common import (EOT, OWT_VALID_TXT_BYTES, SerialPool, book_modules, official_rules, official_tokenizer,
                          owt_text, valid_bin, webbot_tiktoken)
import _ch07_export

BENCH = ROOT / "web" / "data" / "ch07" / "bench.json"


def first_difference(a: list, b: list) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


@pytest.fixture(scope="module")
def sb(dlfs6_path):
    return book_modules()


@pytest.fixture(scope="module")
def tok(sb):
    return official_tokenizer()


@pytest.fixture(scope="module")
def owt(tok, tmp_path_factory):
    """OWT samples (official tokens decoded), written as files the way train_bpe wants them."""
    d = tmp_path_factory.mktemp("owt")
    out = {}
    for name, size in [("250k", 250_000), ("500k", 500_000), ("1m", 1_000_000), ("4m", 4_000_000)]:
        path = d / f"owt_{name}.txt"
        path.write_bytes(owt_text(tok, size).encode("utf-8"))      # bytes: no CRLF translation on Windows
        out[name] = path
    return out


@pytest.fixture
def serial_pool(sb):
    saved = sb.Pool
    sb.Pool = SerialPool
    yield
    sb.Pool = saved


# ─── 7.1 ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_web_data_in_sync(dlfs6_path):
    """web/data/ch07 and the widget fixture are exactly what verify/_ch07_export.py builds from the repo."""
    stale = [str(p) for p, data in _ch07_export.build().items() if not p.is_file() or p.read_bytes() != data]
    assert not stale, f"rerun `uv run python verify/_ch07_export.py`: {stale}"
    assert (ROOT / "web/data/ch07/storybot_tokenizer.py").read_bytes() == (DLFS6 / "storybot/tokenizer.py").read_bytes()


def test_webbot_merge_table(tok):
    rules = official_rules()
    assert len(rules) == 49_743 and list(rules.values()) == list(range(256, 256 + 49_743))
    assert (tok.end_token_id, tok.vocab_size) == (49_999, 50_000)
    flat = json.loads((ROOT / "web/data/tokenizers/webbot.json").read_text(encoding="utf-8"))["merges"]
    assert list(zip(flat[0::2], flat[1::2])) == list(rules)                 # the JSON the browser loads == the pkl
    assert [tok.id_to_bytes[i].decode() for i in range(256, 266)] == \
        [" t", " a", "he", "in", "re", " the", "on", "er", " s", " w"]
    learned = [tok.id_to_bytes[i] for i in range(256, tok.end_token_id)]
    assert len(set(learned)) == len(learned) == 49_743                       # no two merges spell the same bytes
    half = [b for b in learned if b.decode("utf-8", "ignore").encode() != b]
    assert len(half) == 157 and half[0] == b"\xe2\x80"                       # tokens that are not valid UTF-8 alone


def test_official_valid_bin_facts(tok):
    arr = valid_bin()
    assert len(arr) == 64_496_066 and int(arr.max()) == 49_999
    counts = np.bincount(arr, minlength=tok.vocab_size)
    lengths = np.array([len(tok.id_to_bytes[i]) for i in range(tok.vocab_size)], dtype=np.int64)
    total = int((counts * lengths).sum())
    assert total == OWT_VALID_TXT_BYTES == 289_998_753        # decoding the .bin gives back the 290 MB text exactly
    assert int(counts[tok.end_token_id]) == 59_059            # documents separated by <|endoftext|>
    assert round(total / len(arr), 3) == 4.496                 # bytes per token on the whole validation split


def test_ch07_scripts_pass_these_arguments(sb):
    def calls(rel):
        tree = ast.parse((DLFS6 / rel).read_text(encoding="utf-8"))
        return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    train = next(c for c in calls("ch07/01_bpe_train.py") if getattr(c.func, "id", "") == "train_bpe")
    assert {k.arg: ast.literal_eval(k.value) for k in train.keywords} == {"num_processes": 8, "num_chunks": 64}
    src = (DLFS6 / "ch07/01_bpe_train.py").read_text(encoding="utf-8")
    assert "vocab_size = 50000" in src and '"webbot/owt_train.txt"' in src
    encodes = [c for c in calls("ch07/02_bpe_encode.py") if getattr(c.func, "attr", "") == "encode_file"]
    assert len(encodes) == 2 and all({k.arg: ast.literal_eval(k.value) for k in c.keywords} == {"num_processes": 8}
                                     for c in encodes)
    defaults = lambda f: {k: v.default for k, v in inspect.signature(f).parameters.items()
                          if v.default is not inspect.Parameter.empty}
    assert defaults(sb.train_bpe) == {"end_token": EOT, "num_processes": 8, "num_chunks": 8}
    assert defaults(sb.BPETokenizer.encode_file) == {"num_processes": 4, "num_chunks": 64, "cache_dir": "bpe_cache"}


def test_shrunk_run_reproduces_first_15_merges(sb, owt):
    """storybot.train_bpe with a real process pool on 1 MB of OWT: the first 15 merges are the official ones."""
    official = list(official_rules().items())
    rules = sb.train_bpe(str(owt["1m"]), 300, num_processes=4, num_chunks=16)
    mine = list(rules.items())
    assert first_difference(mine, official) == 15
    v = sb.BPETokenizer(rules).id_to_bytes
    assert (v[256 + 15], official_tokenizer().id_to_bytes[256 + 15]) == (b"an", b"is")
    saved = sb.Pool
    sb.Pool = SerialPool                         # what the browser cells do: same merges without processes
    try:
        assert list(sb.train_bpe(str(owt["1m"]), 300, num_chunks=16).items()) == mine
        # Why merge 16 differs: 'an' and 'is' are nearly tied in the samples (the full corpus picked 'is').
        ratio = {}
        for name in ["1m", "4m"]:
            seen = []
            sb.max = lambda it, *a, **kw: (seen.append(dict(it)), builtins.max(it, *a, **kw))[1]
            try:
                sb.train_bpe(str(owt[name]), 256 + 1 + 16, num_chunks=16)
            finally:
                del sb.max
            counts = seen[15]                                      # the counts when merge 16 is chosen
            ratio[name] = counts[(ord("a"), ord("n"))] / counts[(ord("i"), ord("s"))]
        assert round(ratio["1m"] - 1, 3) == 0.037 and round(ratio["4m"] - 1, 3) == 0.002
    finally:
        sb.Pool = saved


def test_vocab_size_tradeoff_on_sample(sb, tok, owt):
    """The first k official merges form a valid tokenizer; compression grows with k but ever more slowly."""
    text = owt["250k"].read_bytes().decode("utf-8")
    pieces = Counter(p for d in text.split(EOT) for p in sb.pretokenize(d))
    n_bytes = len(text.encode("utf-8"))
    rules = list(official_rules().items())
    ratio = {}
    for k in [0, 743, 9_743, 49_743]:
        t = sb.BPETokenizer(dict(rules[:k]))
        ratio[k] = n_bytes / (sum(len(t._encode_text(p)) * c for p, c in pieces.items()) + text.count(EOT))
    assert 1.0 <= ratio[0] < 1.01 and ratio[743] < ratio[9_743] < ratio[49_743]   # an EOT is 13 bytes in 1 token
    assert ratio[49_743] - ratio[9_743] < ratio[9_743] - ratio[743]                     # diminishing returns


# ─── 7.2 ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_merge_loop_profile(sb, owt, serial_pool):
    """cProfile on the book's train_bpe: max() at line 160 dominates; its key runs once per pair per step."""
    sizes = []

    def counting_max(it, *a, **kw):
        sizes.append(len(it))
        return builtins.max(it, *a, **kw)

    sb.max = counting_max
    prof = cProfile.Profile()
    try:
        prof.enable()
        sb.train_bpe(str(owt["500k"]), 1000, num_chunks=8)
        prof.disable()
    finally:
        del sb.max
    st = pstats.Stats(prof).stats
    key = next(v for (f, line, fn), v in st.items() if f.endswith("tokenizer.py") and fn == "<lambda>" and line == 160)
    max_entry = next(v for (f, line, fn), v in st.items() if fn == "<built-in method builtins.max>")
    total = sum(v[2] for v in st.values())                        # total own time of every function
    assert key[1] == sum(sizes)                                   # one key call per pair per step
    assert len(sizes) == 743 and sizes[-1] > 3 * sizes[0]         # the table of pairs grows during training
    assert max_entry[3] / total > 0.5                             # max() incl. its key: most of the time


def test_bench_json_consistent(sb, owt, serial_pool):
    b = json.loads(BENCH.read_text(encoding="utf-8"))
    rows = {r["key"]: r for r in b["encode_2mb"]["rows"]}
    assert all(r["identical_ids"] for k, r in rows.items() if k != "pretokenize")
    assert rows["pretokenize"]["identical_ids"] is None
    assert rows["book"]["mb_per_s"] < rows["lookup"]["mb_per_s"] < rows["pretokenize"]["mb_per_s"] \
        < rows["tiktoken1"]["mb_per_s"]
    scaling = b["train"]["scaling"]
    assert all(r["merges"] == 1743 and r["max_share"] > 0.5 for r in scaling)
    assert all(x["pairs_last_step"] < y["pairs_last_step"] for x, y in zip(scaling, scaling[1:]))
    assert [r["identical_prefix"] for r in b["train"]["shrunk"]] == [6, 10, 15, 15, 15, 20]
    # the smallest point is quick to redo here, and training is deterministic
    official = list(official_rules().items())
    assert first_difference(list(sb.train_bpe(str(owt["250k"]), 300, num_chunks=8).items()), official) == 6
    hf = b.get("hf_tokenizers")
    if hf:                                        # optional run: uv run --with tokenizers python verify/_ch07_record.py
        run = next(r for r in hf["runs"] if r["vocab"] == 2000)
        assert run["merges"] == run["same_tokens_as_book"] == 1743
        fd = run["first_difference"]                     # the first ordering difference is a tie
        assert fd["book"]["count"] == fd["hf"]["count"] and fd["book"]["token"] != fd["hf"]["token"]


# ─── 7.3 ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_every_webbot_token_encodes_to_itself(sb, tok):
    """Both ranking rules (DLFS6: by id pair, tiktoken: by merged bytes) rebuild every token from its bytes."""
    rules = tok.merge_rules
    by_bytes = {tok.id_to_bytes[i]: i for i in range(tok.end_token_id)}

    def pair_rank(bs):
        ids = list(bs)
        while len(ids) > 1:
            best = min(sb.count_pairs(ids), key=lambda p: rules.get(p, float("inf")))
            if best not in rules:
                break
            ids = sb.merge(ids, best, rules[best])
        return ids

    def bytes_rank(bs):
        parts = [bytes([x]) for x in bs]
        while True:
            cands = [(by_bytes[a + c], i) for i, (a, c) in enumerate(zip(parts, parts[1:])) if a + c in by_bytes]
            if not cands:
                return [by_bytes[p] for p in parts]
            _, i = min(cands)
            parts[i:i + 2] = [parts[i] + parts[i + 1]]

    for i in range(256, tok.end_token_id):
        bs = tok.id_to_bytes[i]
        assert pair_rank(bs) == [i] == bytes_rank(bs), i


def test_tiktoken_with_webbot_merges_matches_book(sb, tok, owt):
    enc = webbot_tiktoken(tok)
    presets = json.loads((ROOT / "web/data/ch07/presets.json").read_text(encoding="utf-8"))["presets"]
    tricky = [p["text"] for p in presets] + ["", " ", "\n\n\t  x", "don’t DON'T don't", "naïve café ｆｕｌｌ",
                                              "1234567890 3.14159", "<|endoftext|><|endoftext|>", "a\u0085b c　d",
                                              "ÃÂÃÂÃÂ", "————————", "😁👨‍👩‍👧", "日本語のテキスト"]
    for s in tricky:
        assert enc.encode(s, allowed_special="all") == tok.encode(s), repr(s)
    text = owt["250k"].read_bytes().decode("utf-8")
    t0 = time.perf_counter()
    book = tok.encode(text)
    t_book = time.perf_counter() - t0
    t_tik = min(_timed(lambda: enc.encode(text, allowed_special="all")) for _ in range(3))
    assert enc.encode(text, allowed_special="all") == book
    assert t_book / t_tik > 10, f"tiktoken only {t_book / t_tik:.1f}x faster"


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_gpt2_reference(tok, owt):
    """GPT-2 (tiktoken) has 50,257 ids; on OWT it compresses about as well as WebBot's 50,000."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        g = tiktoken.get_encoding("gpt2")
    except Exception as e:  # noqa: BLE001 - the BPE file is downloaded on first use
        pytest.skip(f"gpt2 encoding unavailable offline: {e}")
    assert g.n_vocab == 50_257
    text = owt["1m"].read_bytes().decode("utf-8")
    n = len(text.encode("utf-8"))
    web, gpt2 = n / len(tok.encode(text)), n / len(g.encode(text, allowed_special="all"))
    assert 4.4 < gpt2 < web < 4.7


# ─── 7.4 ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_special_token_injection(sb, tok):
    user_text = "print('<|endoftext|>')"
    ids = tok.encode(user_text)
    assert ids == [6521, 12167, 49_999, 16615] and tok.decode(ids) == user_text
    plain = [i for p in sb.pretokenize(user_text) for i in tok._encode_text(p)]
    assert 49_999 not in plain and tok.decode(plain) == user_text
    tiktoken = pytest.importorskip("tiktoken")
    try:
        g = tiktoken.get_encoding("gpt2")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"gpt2 encoding unavailable offline: {e}")
    with pytest.raises(ValueError, match="disallowed special token"):
        g.encode("<|endoftext|>")
    assert g.encode("<|endoftext|>", allowed_special="all") == [50256]
    assert g.encode_ordinary("<|endoftext|>") == [27, 91, 437, 1659, 5239, 91, 29]


def test_new_special_tokens_grow_the_book_model(dlfs6_path):
    """webbot.model.GPT: append rows for new tokens (initialised to the mean row); old logits stay unchanged."""
    import torch
    from webbot.model import GPT
    torch.manual_seed(0)
    m = GPT(vocab_size=300, max_context_len=16, embed_dim=32, n_head=4, n_kv_head=2, n_layer=2, ff_dim=64).eval()
    x = torch.randint(0, 300, (2, 8))
    with torch.no_grad():
        before = m(x)
        old_e, old_u = m.embed.weight.clone(), m.unembed.weight.clone()
        m.embed = torch.nn.Embedding(302, 32)
        m.unembed = torch.nn.Linear(32, 302, bias=False)
        m.embed.weight.copy_(torch.cat([old_e, old_e.mean(0, keepdim=True).repeat(2, 1)]))
        m.unembed.weight.copy_(torch.cat([old_u, old_u.mean(0, keepdim=True).repeat(2, 1)]))
        after = m(x)
    assert after.shape[-1] == 302
    torch.testing.assert_close(after[..., :300], before)
    torch.testing.assert_close(after[..., 300], before.mean(-1), rtol=0, atol=1e-6)
    assert np.iinfo(np.uint16).max >= 50_001                  # ids 50,000 and 50,001 still fit in the .bin files


# ─── 7.1 / 7.6 ────────────────────────────────────────────────────────────────────────────────────────────
def test_webbot_parameter_counts(dlfs6_path):
    import torch
    from webbot.model import GPT

    def build(vocab):
        with torch.device("meta"):
            return GPT(vocab, 1024, 768, 12, 4, 12, 2048)

    m = build(50_000)
    total = sum(p.numel() for p in m.parameters())
    assert total == 152_316_672
    assert m.embed.weight.numel() == m.unembed.weight.numel() == 38_400_000             # 50.4 % of WebBot
    linear = sum(mod.weight.numel() for blk in m.blocks for mod in blk.modules() if isinstance(mod, torch.nn.Linear))
    assert linear == 75_497_472                         # matrices every position goes through (7.6's FLOP count)
    assert sum(p.numel() for p in build(257).parameters()) == 75_911_424               # byte-level vocabulary
    kv = 2 * 12 * m.n_kv_head * (768 // 12) * 2                                        # K and V, bf16
    assert kv == 12 * 1024
