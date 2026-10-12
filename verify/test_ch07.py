"""第 7 章 分词器【挑战篇】：本地验证（B 层）。

全部断言都针对 third_party/dlfs6 里的真实代码与官方产物（webbot/merge_rules.pkl、webbot/owt_valid.bin）。
OWT 的文本由官方 owt_valid.bin 解码得到（解码是无损的，见 test_official_valid_bin_facts），不必下载 12 GB 语料。
ch07 的两个脚本会在 12 GB 语料上训练、编码，这里不运行脚本，只 import storybot.tokenizer 里的函数。
作者的两个文件不在子模块里（webbot/download.sh 里是注释掉的）；缺少它们时，用到它们的测试会跳过并给出下载命令。

    uv run pytest verify/test_ch07.py
"""
from __future__ import annotations

import ast
import builtins
import cProfile
import hashlib
import inspect
import json
import lzma
import pstats
import time
from collections import Counter

import numpy as np
import pytest
import regex as re

from conftest import DLFS6, ROOT
from _ch07_common import (EOT, OWT_VALID_TXT_BYTES, OWT_VALID_TXT_SHA256, SerialPool, book_modules, ch04_ext,
                          download_hint, heap_merge_sizes, iter_valid_bytes, iter_valid_docs, missing_downloads,
                          official_rules, official_tokenizer, owt_text, valid_bin, webbot_tiktoken)
import _ch07_export

BENCH = ROOT / "web" / "data" / "ch07" / "bench.json"
STORYBOT_RUN = ROOT / "runs" / "storybot-h100" / "work" / "storybot" / "model_pretrain.pt"


def first_difference(a: list, b: list) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


@pytest.fixture(scope="module")
def sb(dlfs6_path):
    return book_modules()


@pytest.fixture(scope="module")
def webbot_files(dlfs6_path):
    """webbot/merge_rules.pkl and webbot/owt_valid.bin, the author's files (not in the submodule)."""
    missing = missing_downloads()
    if missing:
        pytest.skip(f"needs the author's webbot/{', webbot/'.join(missing)}; download: {download_hint(missing)}")


@pytest.fixture(scope="module")
def tok(sb, webbot_files):
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
def test_web_data_in_sync(webbot_files):
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
    """The official owt_valid.bin: 64,496,066 tokens, 59,059 documents, and decoded it is CS336's owt_valid.txt
    byte for byte (SHA-256; decoding all of it takes about half a minute)."""
    arr = valid_bin()
    assert len(arr) == 64_496_066 and int(arr.max()) == 49_999
    counts = np.bincount(arr, minlength=tok.vocab_size)
    lengths = np.array([len(tok.id_to_bytes[i]) for i in range(tok.vocab_size)], dtype=np.int64)
    total = int((counts * lengths).sum())
    assert total == OWT_VALID_TXT_BYTES == 289_998_753        # decoding the .bin gives back the 290 MB text exactly
    ends = np.flatnonzero(arr == tok.end_token_id)
    assert len(ends) == int(counts[tok.end_token_id]) == 59_059              # documents separated by <|endoftext|>
    assert ends[-1] == len(arr) - 1 and ends[0] > 0 and np.diff(ends).min() > 1   # ...each one ends one, none empty
    assert round(total / len(arr), 3) == 4.496                 # bytes per token on the whole validation split
    h = hashlib.sha256()
    for chunk in iter_valid_bytes(tok):
        h.update(chunk)
    assert h.hexdigest() == OWT_VALID_TXT_SHA256 == \
        "2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660"   # gunzip(CS336 owt_valid.txt.gz)


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
    # Chapter 4's own script defaults to num_chunks=64 and its __main__ relies on that; ch07 passes 64 explicitly
    # because storybot/tokenizer.py, where the function was packaged, lowered the default to 8.
    ch04 = ast.parse((DLFS6 / "ch04/04_bpe_parallel.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(ch04) if isinstance(n, ast.FunctionDef) and n.name == "train_bpe")
    names = [a.arg for a in fn.args.args]
    ch04_defaults = dict(zip(names[len(names) - len(fn.args.defaults):], map(ast.literal_eval, fn.args.defaults)))
    assert ch04_defaults == {"end_token": EOT, "num_processes": 8, "num_chunks": 64}
    main = next(c for c in calls("ch04/04_bpe_parallel.py") if getattr(c.func, "id", "") == "train_bpe")
    assert [k.arg for k in main.keywords] == ["num_processes"]


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


def test_heap_loop_scans_what_the_book_scans(sb, owt, serial_pool):
    """Chapter 4's lazy-deletion heap (heap_merge_sizes) learns the book's merges in the book's order and keeps the
    book's pair table, so its table size before every step is exactly what the book's max() scans at that step.
    This is how bench.json counts the pairs for the whole 290 MB validation split (49,743 merges)."""
    sizes = []

    def counting_max(it, *a, **kw):
        sizes.append(len(it))
        return builtins.max(it, *a, **kw)

    sb.max = counting_max
    try:
        book = sb.train_bpe(str(owt["1m"]), 2000, num_chunks=8)
    finally:
        del sb.max
    ids_counts = ch04_ext().ids_counts_of(owt["1m"].read_bytes().decode("utf-8"))
    merges, heap_sizes = heap_merge_sizes(ids_counts, 1743, compact=1.2)    # 1.2: rebuilds the heap often
    assert merges == list(book) and heap_sizes == sizes and len(sizes) == 1743
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    full = bench["valid_290mb"]["full"]
    eight = next(p for p in bench["valid_290mb"]["prefixes"] if p["mb"] == 8)
    row8 = next(r for r in bench["train"]["scaling"] if r["mb"] == 8)       # the book's own 8 MB run
    assert (eight["bytes"], eight["pairs_first_step"], eight["pairs_at_1743"], eight["pairs_scanned_1743"]) == \
        (row8["bytes"], row8["pairs_first_step"], row8["pairs_last_step"], row8["pairs_scanned"])
    assert full["merges"] == 49_743 and full["bytes"] == OWT_VALID_TXT_BYTES


def test_bench_json_consistent(sb, owt, serial_pool):
    b = json.loads(BENCH.read_text(encoding="utf-8"))
    enc = b["encode_2mb"]
    rows = {r["key"]: r for r in enc["rows"]}
    # the book's row is the reference; pretokenize-only yields no ids; every other row was compared id by id
    assert {k: r["identical_ids"] for k, r in rows.items()} == \
        {"book": None, "lookup": True, "pretokenize": None, "tiktoken1": True, "tiktoken8": True}
    assert all(len(r["seconds_all"]) == enc["rounds"] and r["seconds"] == round(sorted(r["seconds_all"])[2], 3)
               for r in rows.values())                                      # median of the 5 interleaved rounds
    assert rows["book"]["mb_per_s"] < rows["lookup"]["mb_per_s"] < rows["pretokenize"]["mb_per_s"] \
        < rows["tiktoken1"]["mb_per_s"]
    heap8 = b["heap_8mb"]
    assert heap8["heap_identical_to_book"] and heap8["heap_9743_starts_with_book"]
    mk = b["max_key"]
    assert mk["train_1mb"]["identical_merges"] and mk["same_pick"]["lambda"] == mk["same_pick"]["zip"]
    valid = b["valid_290mb"]
    p1743 = [p["pairs_at_1743"] for p in valid["prefixes"]] + [valid["full"]["pairs_at"]["1743"]]
    assert p1743 == sorted(p1743) and len(set(p1743)) == len(p1743)     # more text, more distinct pairs
    every = valid["full"]["pairs_every_100_steps"]
    assert valid["full"]["pairs_scanned"] >= sum(every) * 100 * 0.99 and every[-1] > 3 * every[17]
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


def test_tiktoken_reproduces_official_valid_bin(tok):
    """tiktoken with WebBot's merges re-encodes all 59,059 documents of the validation split (about a minute):
    every one of the 64,496,066 official ids comes back, so owt_valid.bin is the product of these rules."""
    enc = webbot_tiktoken(tok)
    n_docs = n_ids = 0
    batch_ids, batch_texts = [], []

    def check():
        nonlocal n_docs, n_ids
        for ids, out in zip(batch_ids, enc.encode_ordinary_batch(batch_texts, num_threads=8)):
            assert np.array_equal(np.asarray(out, dtype=np.uint16), ids), f"document {n_docs}"
            n_docs += 1
            n_ids += len(ids) + 1                                           # + its <|endoftext|>
        batch_ids.clear()
        batch_texts.clear()

    for ids, data in iter_valid_docs(tok):
        batch_ids.append(ids)
        batch_texts.append(data.decode("utf-8"))                            # strict: every document is valid UTF-8
        if len(batch_texts) == 2_000:
            check()
    check()
    assert (n_docs, n_ids) == (59_059, 64_496_066)


def test_gpt2_reference(tok, owt):
    """GPT-2 (tiktoken) has 50,257 ids; on OWT it compresses about as well as WebBot's 50,000.
    Needs GPT-2's BPE file, which tiktoken downloads on first use (the test is skipped, saying so, when offline)."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        g = tiktoken.get_encoding("gpt2")
    except Exception as e:  # noqa: BLE001 - the BPE file is downloaded on first use
        pytest.skip(f"GPT-2's BPE file could not be downloaded (tiktoken.get_encoding('gpt2')): {e}")
    assert g.n_vocab == 50_257
    text = owt["1m"].read_bytes().decode("utf-8")
    n = len(text.encode("utf-8"))
    web, gpt2 = n / len(tok.encode(text)), n / len(g.encode(text, allowed_special="all"))
    assert (round(gpt2, 2), round(web, 2)) == (4.52, 4.61)                 # bytes per token, as on the page
    assert g.encode(EOT, allowed_special="all") == [50_256]                # GPT-2's own special token, last id


# ─── 7.4 ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_special_token_injection(sb, tok):
    """The book's encode turns <|endoftext|> in user text into 49,999; tiktoken (here with WebBot's own rules, no
    download needed) refuses by default, and encode_ordinary treats it as plain text."""
    user_text = "print('<|endoftext|>')"
    ids = tok.encode(user_text)
    assert ids == [6521, 12167, 49_999, 16615] and tok.decode(ids) == user_text
    plain = [i for p in sb.pretokenize(user_text) for i in tok._encode_text(p)]
    assert 49_999 not in plain and tok.decode(plain) == user_text
    enc = webbot_tiktoken(tok)
    with pytest.raises(ValueError, match="disallowed special token"):
        enc.encode(user_text)                                             # the default: refuse
    assert enc.encode(user_text, allowed_special="all") == ids           # explicitly allowed: like the book
    assert enc.encode_ordinary(user_text) == plain                       # plain text: no 49,999


def test_mean_row_on_trained_storybot(dlfs6_path):
    """A new token whose unembedding row is the mean of the old rows: on the StoryBot we trained with chapter 6's
    unchanged script (run storybot-h100), its probability at held-out positions is tiny, far below 1/(V+1)."""
    import torch
    if not STORYBOT_RUN.is_file():
        pytest.skip(f"needs the storybot-h100 run's checkpoint {STORYBOT_RUN.relative_to(ROOT)} "
                    "(web/data/runs/storybot-h100/report.json describes the run)")
    from storybot.model import GPT
    ckpt = torch.load(STORYBOT_RUN, map_location="cpu")
    m = GPT(ckpt["vocab_size"], ckpt["max_context_len"], ckpt["embed_dim"], ckpt["n_head"], ckpt["n_layer"],
            ckpt["ff_dim"], ckpt["theta"])
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    seen = {}
    m.unembed.register_forward_hook(lambda mod, inp, out: seen.update(h=inp[0]))   # what the unembedding sees
    data = np.memmap(DLFS6 / "storybot" / "tiny_stories_valid.bin", dtype=np.uint16, mode="r")
    x = torch.from_numpy(np.asarray(data[:256], dtype=np.int64))[None]      # 256 held-out positions
    with torch.no_grad():
        z = m(x)[0].double()                                                # (256, V) logits of the trained model
        h = seen["h"][0].double()
        u = m.unembed.weight.double()                                       # (V, d), no bias
        u_new = torch.cat([u, u.mean(0, keepdim=True)])                     # the new row: mean of the old rows
        z_new = h @ u_new.T
    v = u.shape[0]
    assert v == 10_000
    torch.testing.assert_close(z_new[:, :v], z, rtol=0, atol=1e-3)         # old logits: unchanged (fp32 vs fp64)
    torch.testing.assert_close(z_new[:, v], z_new[:, :v].mean(1), rtol=0, atol=1e-9)   # new logit = their mean
    p_new = torch.softmax(z_new, dim=1)[:, v]
    assert (p_new <= 1 / (v + 1)).all()                                     # Jensen: never above the uniform share
    assert p_new.max() < 1e-5 and p_new.median() < 1e-8                    # uniform would be ~1e-4
    assert 5.0e-10 < p_new.median() < 6.2e-10 and 1.0e-6 < p_new.max() < 1.4e-6   # page: ~5.6e-10, ~1.2e-6


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


# ─── 7.5 ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_word_vocab_oov_shrinks_but_stays_positive(tok):
    """Word-level vocabularies built from the first 50 KB, 1 MB and 8 MB of OWT valid, tested on the same 100 KB
    that follows the 8 MB: the share of unknown words falls (30.8% -> 10.3% -> 4.5%) but does not reach 0."""
    words = lambda s: re.findall(r"\p{L}+", s)                  # letters only, as in cell 7.5-a
    head = owt_text(tok, 8_000_000)
    after = owt_text(tok, 8_200_000)[len(head):][:100_000]       # the next 100 KB (characters) after the 8 MB
    test = words(after)
    rate, size = {}, {}
    for n in (50_000, 1_000_000, 8_000_000):
        vocab = set(words(owt_text(tok, n)))
        size[n] = len(vocab)
        rate[n] = round(sum(w not in vocab for w in test) / len(test), 3)
    assert rate == {50_000: 0.308, 1_000_000: 0.103, 8_000_000: 0.045}
    assert size[8_000_000] > tok.vocab_size == 50_000            # a bigger vocabulary than all of WebBot's


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


# ─── 7.6 ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_storybot_bytes_per_token_and_predictability(sb, tok):
    """StoryBot's official valid.bin decodes to exactly tiny_stories_valid.txt (4.117 bytes per token), and a
    general-purpose compressor finds OWT much harder to predict than TinyStories."""
    st = sb.BPETokenizer.load_from(str(DLFS6 / "storybot" / "merge_rules.pkl"))
    txt = DLFS6 / "storybot" / "tiny_stories_valid.txt"
    arr = np.memmap(DLFS6 / "storybot" / "tiny_stories_valid.bin", dtype=np.uint16, mode="r")
    counts = np.bincount(arr, minlength=st.vocab_size)
    lengths = np.array([len(st.id_to_bytes[i]) for i in range(st.vocab_size)], dtype=np.int64)
    assert int((counts * lengths).sum()) == txt.stat().st_size == 22_502_601   # the .bin decodes to the .txt
    assert round(txt.stat().st_size / len(arr), 3) == 4.117
    stories = txt.read_bytes()
    stories = stories[: stories.index(EOT.encode(), 1_000_000)]                # ~1 MB, cut at a document end
    web = owt_text(tok, 1_000_000).encode("utf-8")
    bpb = {name: len(lzma.compress(data, preset=9)) * 8 / len(data) for name, data in [("ts", stories), ("owt", web)]}
    assert 1.75 < bpb["ts"] < 1.85 and 2.5 < bpb["owt"] < 2.65, bpb           # page: "约 1.8" / "约 2.6"
