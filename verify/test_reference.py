"""参考页（errata / cs336 / glossary）的本地验证。

* test_eNN_*      one test per entry on content/errata.md (the minimal reproduction of each DLFS6 issue)
* test_cs336_*    the CS336 page: the recorded A1 run, StoryBot == A1 TinyStories config, DPO at init
* test_glossary_* naming facts that need the real DLFS6 code
* test_*_current  page data files under web/data/{errata,cs336,glossary} equal what the code produces

CPU only. DLFS6 scripts are loaded with `load_defs` (only their imports and def/class nodes run, never
the training loops) and nothing is written into third_party/dlfs6. The tests that need the CS336 A1
clone read CS336_A1_DIR and skip when it is not set.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import warnings
from itertools import cycle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs336.adapters_dlfs6 import DLFS6, load_defs  # noqa: E402

RESEARCH = ROOT / "research" / "data"
WEB = ROOT / "web" / "data"
CPU = torch.device("cpu")


def src(rel: str) -> str:
    return (DLFS6 / rel).read_text(encoding="utf-8")


def git(*args: str) -> str:
    """Run git in the pinned DLFS6 submodule (read-only commands only)."""
    import subprocess

    return subprocess.check_output(["git", "-C", str(DLFS6), *args], text=True, encoding="utf-8")


def consts(rel: str) -> dict:
    """Module-level `name = <literal>` assignments of a DLFS6 script, i.e. its hyperparameters."""
    return {t.id: ast.literal_eval(n.value) for n in ast.parse(src(rel)).body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}


def finetune_passes() -> dict[str, tuple[int, int, int, float]]:
    """(samples, batch_size, max_iters, passes) of the cycle(DataLoader) loops of SFT, DPO and GRPO.

    A pass is counted in the batches the real DataLoader yields: it keeps the last, partial batch.
    """
    from torch.utils.data import DataLoader

    grpo = load_defs("ch03/09_grpo.py", names={"GRPODataset"})["GRPODataset"](tokenizer=None)
    sizes = {"SFT": ("ch03/03_sft.py", len(json.loads(src("codebot/tiny_codes_sft.json")))),
             "DPO": ("ch06/08_dpo.py", len(json.loads(src("storybot/tiny_stories_dpo.json")))),
             "GRPO": ("ch03/09_grpo.py", len(grpo))}
    out = {}
    for name, (script, n) in sizes.items():
        c = consts(script)
        per_pass = len(DataLoader(range(n), batch_size=c["batch_size"], shuffle=True))
        out[name] = (n, c["batch_size"], c["max_iters"], c["max_iters"] / per_pass)
    return out


def page_cell(slug: str, cell_id: str) -> str:
    """Source of a verify cell on content/<slug>.md (to prove a page's copy of book code is verbatim)."""
    text = (ROOT / "content" / f"{slug}.md").read_text(encoding="utf-8")
    m = re.search(r"^```verify[ \t]+id=" + re.escape(cell_id) + r"\b[^\n]*\n(.*?)^```", text, re.S | re.M)
    assert m, f"verify cell {cell_id} not found on {slug}.md"
    return m.group(1)


def function_ast(code: str, name: str) -> str:
    fn = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(fn, include_attributes=False)


# ====================================================================== errata: high

def test_e01_grpo_policy_updates_in_train_mode(dlfs6_path):
    """09_grpo.py: model comes out of load_from in train mode (dropout 0.1), old_model is eval()."""
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer

    g = load_defs("ch03/09_grpo.py", names={"GRPODataset", "compute_probs"})
    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    model = GPT.load_from(str(DLFS6 / "codebot" / "model_sft.pt"))
    old = GPT.load_from(str(DLFS6 / "codebot" / "model_sft.pt"))
    old.eval()
    assert model.training and not old.training                      # exactly as in the script

    ds = g["GRPODataset"](tok)                                      # the 81 prompts i+j=
    ids, mask = ds.get_batch([p for p, _ in ds.data], [str(gt) for _, gt in ds.data], CPU)
    on = mask[:, 1:].bool()
    outside = lambda r: ((r < 0.8) | (r > 1.2)).float().mean().item()     # PPO clip range, eps=0.2
    with torch.no_grad():
        p_old = g["compute_probs"](old, ids)
        fractions = []
        for seed in range(20):                                      # 20 dropout masks, same weights
            torch.manual_seed(seed)
            r_train = (g["compute_probs"](model, ids) / (p_old + 1e-8))[on]    # the first update's ratio
            fractions.append(outside(r_train))
        model.eval()
        r_eval = (g["compute_probs"](model, ids) / (p_old + 1e-8))[on]
    assert r_train.numel() == 117
    assert round(sum(fractions) / 20, 2) == 0.48 and 0.41 < min(fractions) and max(fractions) < 0.59
    assert outside(r_eval) == 0 and (r_eval - 1).abs().max() < 1e-5


def test_e15_sft_greedy_baseline_is_44_of_81(dlfs6_path):
    """The SFT starting point is 44/81; the script's first logged accuracy is taken after 2 updates."""
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer
    from codebot.utils import generate

    g = load_defs("ch03/09_grpo.py", names={"GRPODataset", "calculate_reward"})
    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    sft = GPT.load_from(str(DLFS6 / "codebot" / "model_sft.pt"))
    correct = sum(g["calculate_reward"](gt, generate(sft, tok, p, temperature=0)) > 0
                  for p, gt in g["GRPODataset"](tok).data)
    assert correct == 44


def test_e02_hf_storybot_checkpoints_use_mlp_keys(dlfs6_path):
    from storybot.model import GPT

    with pytest.raises(RuntimeError, match=r"Missing key\(s\)") as err:
        GPT.load_from(str(DLFS6 / "storybot" / "model_pretrain.pt"))
    assert "blocks.0.ffn.W.weight" in str(err.value) and "blocks.0.mlp.W.weight" in str(err.value)
    for name in ("model_pretrain.pt", "model_iter_500.pt", "model_iter_5000.pt"):
        ck = torch.load(DLFS6 / "storybot" / name, map_location=CPU, mmap=True)
        keys = ck["model_state_dict"].keys()
        assert any(".mlp." in k for k in keys) and not any(".ffn." in k for k in keys), name
    # the fix: rename on load (strict loading then succeeds)
    ck = torch.load(DLFS6 / "storybot" / "model_pretrain.pt", map_location=CPU)
    model = GPT(*(ck[k] for k in ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim", "theta")))
    model.load_state_dict({k.replace(".mlp.", ".ffn."): v for k, v in ck["model_state_dict"].items()})
    assert "self.mlp" in src("webbot/model.py") and "self.ffn" in src("storybot/model.py")
    # the rename happened in 234a9c6 (the checkpoints were not re-uploaded)
    show = git("show", "234a9c6", "--format=%ad %s", "--date=short", "--", "storybot/model.py")
    assert show.startswith("2026-03-27 原稿との整合性修正")
    assert "-        self.mlp = SwiGLU(embed_dim, ff_dim)" in show and "+        self.ffn = SwiGLU(embed_dim, ff_dim)" in show


def test_e12_ch04_05_eval_is_missing():
    numbers = sorted(p.name[:2] for p in (DLFS6 / "ch04").glob("[0-9][0-9]_*.py"))
    assert numbers == ["01", "02", "03", "04", "06", "07"]


def test_e48_download_script_lists_unpublished_dpo_model():
    assert "storybot/model_dpo.pt" in src("storybot/download.sh")
    assert "model" not in src("webbot/download.sh").split("# === tokenizer files ===")[1].replace("merge_rules", "")
    assert "'dpo': 'storybot/model_dpo.pt'" in src("ch06/09_llm_judge.py")


def test_e03_crlf_checkout_changes_binary_mode_merges(dlfs6_path, tmp_path):
    from storybot.tokenizer import train_bpe

    text = (DLFS6 / "codebot" / "tiny_codes.txt").read_bytes()[:200_000].decode("ascii")
    assert "\r" not in text
    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes(text.encode())
    crlf.write_bytes(text.replace("\n", "\r\n").encode())
    rules_lf = list(train_bpe(str(lf), 300, num_processes=1, num_chunks=1))
    rules_crlf = list(train_bpe(str(crlf), 300, num_processes=1, num_chunks=1))
    assert (13, 10) not in rules_lf and rules_crlf.index((13, 10)) == 1        # '\r\n' becomes merge #2
    assert open(crlf, encoding="utf-8").read() == text                          # text mode hides the \r
    full = (DLFS6 / "codebot" / "tiny_codes.txt").read_bytes()                  # LF checkout of this repo
    assert (len(full), full.count(b"\n")) == (6_487_033, 256_501)               # CRLF: 6,743,534 bytes
    assert not (DLFS6 / ".gitattributes").exists()


# ====================================================================== errata: medium

def test_e04_storybot_cache_grows_and_rope_position_freezes(dlfs6_path):
    from storybot.model import GPT, RoPE

    rope = RoPE(10000, 8, max_context_len=16)
    x = torch.randn(1, 1, 1, 8)
    assert torch.equal(rope(x, offset=20), rope(x, offset=15))        # clamped to the last position
    assert not torch.equal(rope(x, offset=14), rope(x, offset=15))

    torch.manual_seed(0)
    m = GPT(50, 16, 32, 2, 2, 64).eval()
    seq, logits = [1], []
    with torch.no_grad():
        nxt = torch.tensor([[1]])
        for _ in range(30):                                            # what storybot.utils.generate does
            lg = m(nxt, use_cache=True)[:, -1]
            logits.append(lg)
            nxt = lg.argmax(-1, keepdim=True)
            seq.append(int(nxt))
        attn = m.blocks[0].attn
        assert attn.k_cache.shape[2] == 30 and attn.cache_offset == 30   # never truncated to 16
        diff = [float((m(torch.tensor([seq[:t + 1][-16:]]))[:, -1] - logits[t]).abs().max()) for t in range(30)]
    assert max(diff[:16]) < 1e-5 and min(diff[16:]) > 1e-3             # diverges exactly at position 16
    # how often a story is longer than StoryBot's 256 positions (TinyStories validation split)
    v = np.fromfile(DLFS6 / "storybot" / "tiny_stories_valid.bin", dtype=np.uint16)
    lens = np.diff(np.r_[-1, np.flatnonzero(v == 9999)]) - 1
    assert len(lens) == 27_630 and round(float((lens > 256).mean()), 4) == 0.1087 and np.median(lens) == 176
    assert "max_new_tokens = 300" in src("ch06/06_generate.py")


def test_e05_rope_length_max_plus_one_is_silently_wrong(dlfs6_path):
    from storybot.model import GPT, RoPE

    rope = RoPE(10000, 8, max_context_len=8)
    x = torch.randn(1, 1, 9, 8)
    y = rope(x)                                                         # offset becomes -1: one cos/sin row
    same_angle = torch.cat([rope(x[:, :, i:i + 1], offset=7) for i in range(9)], dim=2)
    assert torch.allclose(y, same_angle)                                # every position rotated as position 7
    with pytest.raises(RuntimeError):
        rope(torch.randn(1, 1, 10, 8))                                  # max+2 fails loudly
    m = GPT(50, 8, 16, 2, 1, 32).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 9, dtype=torch.long)).shape == (1, 9, 50)   # no error at max+1


def test_e06_multi_token_call_after_prefill_is_unmasked(dlfs6_path):
    from storybot.model import GPT

    torch.manual_seed(0)
    m = GPT(50, 32, 32, 2, 1, 64).eval()
    seq = torch.randint(0, 50, (1, 6))
    with torch.no_grad():
        full = m(seq)
        m.clear_cache(); m(seq[:, :3], use_cache=True)
        chunk = m(seq[:, 3:5], use_cache=True)                         # two new tokens in one call
        m.clear_cache(); m(seq[:, :3], use_cache=True)
        one = m(seq[:, 3:4], use_cache=True)                           # one token per call (generate())
    assert (chunk[:, 0] - full[:, 3]).abs().max() > 1e-3                # token 3 attended to token 4
    assert torch.allclose(chunk[:, 1], full[:, 4], atol=1e-5)
    assert torch.allclose(one[:, 0], full[:, 3], atol=1e-5)
    # the proposed fix: C new queries over K cached keys see a bottom-right aligned triangle
    C, K = 2, 5
    assert torch.equal(torch.ones(C, K).tril(diagonal=K - C), torch.ones(K, K).tril()[K - C:])


def test_e07_cycle_dataloader_replays_first_epoch_and_keeps_batches():
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    dl = DataLoader(list(range(10)), batch_size=3, shuffle=True)        # 4 batches per epoch
    it = cycle(dl)
    first = [next(it) for _ in range(4)]
    second = [next(it) for _ in range(4)]
    assert all(a is b for a, b in zip(first, second))                   # same objects: kept and replayed
    users = [p for p in ("ch03/01_pretrain.py", "ch03/03_sft.py", "ch03/09_grpo.py", "ch06/08_dpo.py")
             if "shuffle=True" in src(p) and "cycle(dataloader)" in src(p)]
    assert len(users) == 4
    # ch03/01 never finishes its first pass: 20000 x 32 windows < len(TokenDataset)
    n_windows = 2_677_398 - 256
    assert 20_000 * 32 < n_windows and 20_000 * 2 * 32 * 256 * 8 == 2_621_440_000   # bytes kept (x, y int64)
    # the fine-tuning loops do finish passes, and every pass after the first is a replay
    fp = finetune_passes()
    assert [round(fp[k][3], 1) for k in ("SFT", "DPO", "GRPO")] == [14.3, 55.6, 166.7]
    assert [len(b) for b in DataLoader(range(81), batch_size=fp["GRPO"][1])] == [32, 32, 17]


def test_e08_encode_file_deletes_the_whole_cache_dir(dlfs6_path, tmp_path):
    from storybot.tokenizer import BPETokenizer

    tok = BPETokenizer.load_from(str(DLFS6 / "storybot" / "merge_rules.pkl"))
    text = "Once upon a time, there was a cat.<|endoftext|>The end."
    (tmp_path / "in.txt").write_bytes(text.encode())
    cache = tmp_path / "precious"
    cache.mkdir()
    (cache / "notes.txt").write_text("keep me", encoding="utf-8")
    n = tok.encode_file(str(tmp_path / "in.txt"), str(tmp_path / "out.bin"), num_processes=1, num_chunks=1,
                        cache_dir=str(cache))
    assert not cache.exists()                                           # notes.txt is gone too
    ids = np.fromfile(tmp_path / "out.bin", dtype=np.uint16)
    assert len(ids) == n and tok.decode(ids.tolist()) == text
    raw = tmp_path / "chunk_00000.npy"                                  # how _encode_chunk writes its cache
    np.array(ids, dtype=np.uint16).tofile(raw)
    with pytest.raises(ValueError):
        np.load(raw)                                                    # no .npy header
    assert np.array_equal(np.fromfile(raw, dtype=np.uint16), ids)


def test_e15_grpo_logs_accuracy_after_the_updates():
    loop = next(n for n in ast.walk(ast.parse(src("ch03/09_grpo.py")))
                if isinstance(n, ast.For) and ast.unparse(n.target) == "i")
    kinds = [ast.unparse(s)[:40] for s in loop.body]
    upd = next(k for k, s in enumerate(kinds) if s.startswith("for _ in range(n_update_per_generation)"))
    ev = next(k for k, s in enumerate(kinds) if s.startswith("if i % eval_interval == 0"))
    assert upd < ev
    # the recorded research run (seed 0, as shipped) logged "iteration 0" at 55.56%, not the SFT start 44/81
    base = json.loads((WEB / "errata" / "grpo_dropout_runs.json").read_text(encoding="utf-8"))["runs"]["base"]
    assert (base["seed"], base["acc_it"][0], base["greedy_acc"][0]) == (0, 0, 55.56)
    assert round(100 * 44 / 81, 1) == 54.3


def test_e09_calculate_reward_swallows_every_error():
    g = load_defs("ch03/09_grpo.py", names={"calculate_reward"})
    f = g["calculate_reward"]
    assert [f(7, s) for s in ("3+4=7", "7 8", "8 7", "7.0", "-7")] == [1.0, 0.0, 1.0, 0.0, 0.0]
    code = ast.get_source_segment(src("ch03/09_grpo.py"), next(
        n for n in ast.parse(src("ch03/09_grpo.py")).body if isinstance(n, ast.FunctionDef) and n.name == "calculate_reward"))
    ns: dict = {}
    exec(code, ns)                                                      # same function, `re` never imported
    assert ns["calculate_reward"](7, "7") == 0.0                        # NameError became "wrong answer"
    assert function_ast(page_cell("errata", "Z.4-d"), "calculate_reward") == function_ast(code, "calculate_reward")


def test_e10_dpo_prompt_and_answer_are_tokenized_apart(dlfs6_path):
    from storybot.tokenizer import BPETokenizer

    tok = BPETokenizer.load_from(str(DLFS6 / "storybot" / "merge_rules.pkl"))
    data = json.loads((DLFS6 / "storybot" / "tiny_stories_dpo.json").read_text(encoding="utf-8"))
    assert len(data) == 144 and all(d["prompt"].endswith(" ") for d in data)
    extra = []
    for d in data:
        for answer in (d["chosen"], d["rejected"]):
            apart = tok.encode(d["prompt"]) + tok.encode(answer)       # ch06/08 _create_sample
            joint = tok.encode(d["prompt"] + answer)
            assert apart != joint                                       # all 288 sequences differ
            extra.append(len(apart) - len(joint))
    assert {k: extra.count(k) for k in set(extra)} == {0: 2, 1: 271, 2: 15}
    p = data[0]["prompt"]
    assert tok.decode(tok.encode(p)[-1:]) == " " and tok.decode(tok.encode(p + "Tim")[-1:]) == " Tim"


def test_e28_measure_generation_time_clears_the_wrong_cache():
    k = load_defs("ch05/05_kvcache.py")
    torch.manual_seed(0)
    m = k["GPT"](vocab_size=50, max_context_len=64, embed_dim=16, n_head=2, n_layer=1, ff_dim=32)
    start = torch.tensor([[1]])
    k["measure_generation_time"](m, start, use_cache=True, num_tokens=5)
    assert m.blocks[0].attn.cache_offset == 5
    k["measure_generation_time"](m, start, use_cache=True, num_tokens=5)
    assert m.blocks[0].attn.cache_offset == 10                          # continued the previous cache


def test_e21_grpo_has_no_length_cap_but_codebot_has_256_positions(dlfs6_path):
    import inspect

    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer
    from codebot.utils import generate

    call = next(n for n in ast.walk(ast.parse(src("ch03/09_grpo.py")))
                if isinstance(n, ast.Call) and ast.unparse(n.func) == "generate"
                and any(kw.arg == "temperature" and ast.unparse(kw.value) == "1.0" for kw in n.keywords))
    assert "max_new_tokens" not in {kw.arg for kw in call.keywords}
    assert inspect.signature(generate).parameters["max_new_tokens"].default == 1000
    g = load_defs("ch03/09_grpo.py", names={"GRPODataset", "compute_probs"})
    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    ds = g["GRPODataset"](tok)
    assert {len(tok.encode(p)) for p, _ in ds.data} == {22}              # every prompt is 22 tokens
    ids, _ = ds.get_batch([ds.data[0][0]], ["x " * 300], CPU)            # 22 prompt + ~300 response tokens
    assert ids.shape[1] > 256
    m = GPT(1000, 256, 32, 2, 1, 64, 0.1).eval()
    with pytest.raises(IndexError):
        g["compute_probs"](m, ids)
    # why the book never hit it: at T=1 the SFT model almost always answers one number and stops
    sft = GPT.load_from(str(DLFS6 / "codebot" / "model_sft.pt")).eval()
    eos = tok.encode("<|endoftext|>")
    assert {len(tok.encode(str(a))) for a in range(19)} == {1, 2}       # every sum 0..18 is 1-2 tokens
    short = []
    with torch.no_grad():
        for prompt, _ in ds.data:
            p = tok.encode(prompt)
            seqs = [p + tok.encode(str(a)) + eos for a in range(19)]
            width = max(map(len, seqs))
            logp = torch.log_softmax(sft(torch.tensor([s + [0] * (width - len(s)) for s in seqs])), -1)
            short.append(sum(row[torch.arange(len(p) - 1, len(s) - 1), torch.tensor(s[len(p):])].sum().exp().item()
                             for row, s in zip(logp, seqs)))                 # P(answer, then EOS)
    assert round(min(short), 3) == 0.976 and round(sum(short) / len(short), 3) == 0.997


def test_e11_special_token_text_is_injected(dlfs6_path):
    from codebot.tokenizer import BPETokenizer as CodeTok
    from storybot.tokenizer import BPETokenizer as StoryTok

    code = CodeTok.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    story = StoryTok.load_from(str(DLFS6 / "storybot" / "merge_rules.pkl"))
    assert code.encode("print('<|endoftext|>')") == [350, 338, 999, 343]
    assert story.end_token_id in story.encode("Say <|endoftext|> please")
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("gpt2")                              # needs a cached/downloadable BPE file
    except Exception as e:                                               # offline without cache
        pytest.skip(f"tiktoken gpt2 unavailable: {e}")
    with pytest.raises(ValueError, match="disallowed special token"):
        enc.encode("<|endoftext|>")                                     # tiktoken refuses by default


# ====================================================================== errata: low (numbers & interfaces)

def test_e13_get_batch_never_samples_the_last_window():
    for rel in ("ch06/05_pretrain.py", "ch09/01_pretrain.py"):                # same line in both scripts
        assert "torch.randint(len(data) - context_len - 1, (batch_size,))" in src(rel)
    get_batch = load_defs("ch06/05_pretrain.py", names={"get_batch"})["get_batch"]
    torch.manual_seed(0)
    data = np.arange(100)
    starts = set()
    for _ in range(1000):
        x, y = get_batch(data, 7, 32, CPU)
        starts.update(x[:, 0].tolist())
    assert min(starts) == 0 and max(starts) == 91 and len(starts) == 92  # start 92 is valid but never drawn
    assert list(data[92:99] + 1) == list(data[93:100])


def test_e14_evaluate_drops_the_last_full_window():
    get_batch = load_defs("ch06/05_pretrain.py", names={"get_batch"})["get_batch"]
    x, _ = get_batch(np.arange(513), 256, 2, CPU, random=False, offset=0)
    assert x.shape[0] == 1                                              # starts 0 and 256 both fit
    x, _ = get_batch(np.arange(514), 256, 2, CPU, random=False, offset=0)
    assert x.shape[0] == 2
    assert "ix = ix[ix + context_len + 1 < len(data)]" in src("ch09/01_pretrain.py")


def test_e16_ratio_is_taken_in_probability_space():
    assert "ratio = probs / (old_probs + 1e-8)" in src("ch03/09_grpo.py")
    p = torch.tensor([1e-4, 1e-6, 1e-7, 1e-8])
    assert torch.allclose(p / (p + 1e-8), torch.tensor([0.9999, 0.990099, 0.909091, 0.5]), atol=1e-6)


def test_e17_storybot_rmsnorm_uses_float32_machine_eps(dlfs6_path):
    from storybot.model import GPT

    m = GPT(50, 8, 16, 2, 1, 32)
    norms = [mod for mod in m.modules() if isinstance(mod, nn.RMSNorm)]
    assert len(norms) == 3 and all(n.eps is None for n in norms)
    assert "self.eps = 1e-5" in src("ch05/03_rmsnorm.py")
    x = torch.full((1, 8), 1e-3)                                        # a tiny activation: rms = 1e-3
    a, b = nn.RMSNorm(8)(x), nn.RMSNorm(8, eps=1e-5)(x)
    assert abs(a[0, 0].item() - 0.9452) < 1e-4 and abs(b[0, 0].item() - 0.3015) < 1e-4


def test_e18_codebot_runs_erf_gelu_and_torch_layernorm(dlfs6_path):
    from codebot.model import GPT

    m = GPT(50, 8, 16, 2, 1, 64, 0.1)
    block = m.blocks[0]
    assert type(block.norm1) is nn.LayerNorm and type(m.norm) is nn.LayerNorm
    gelus = [mod for mod in m.modules() if isinstance(mod, nn.GELU)]
    assert gelus and all(g.approximate == "none" for g in gelus)
    tree = ast.parse(src("codebot/model.py"))
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert {"LayerNorm", "GELU"}.isdisjoint(used)                       # defined, never instantiated
    from codebot.model import LayerNorm as HandLN

    torch.manual_seed(0)
    h = torch.randn(4, 6, 16)
    assert (HandLN(16)(h) - nn.LayerNorm(16)(h)).abs().max().item() < 1e-5   # numerically the same
    x = torch.linspace(-6, 6, 12001)
    diff = (nn.GELU()(x) - nn.GELU(approximate="tanh")(x)).abs().max().item()
    assert abs(diff - 4.74e-4) < 1e-5


def test_e19_head_dim_is_not_checked(dlfs6_path):
    from storybot.model import Block

    b = Block(100, 3, 64)                                               # 100 // 3 = 33, 3 * 33 = 99
    assert b.attn.W_q.weight.shape == (99, 100) and b.attn.W_o.weight.shape == (100, 99)
    assert b(torch.randn(1, 4, 100)).shape == (1, 4, 100)


def test_e20_eos_is_not_part_of_the_grpo_objective(dlfs6_path):
    from codebot.tokenizer import BPETokenizer

    g = load_defs("ch03/09_grpo.py", names={"GRPODataset"})
    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    ds = g["GRPODataset"](tok)
    p = ds.data[0][0]
    ids, mask = ds.get_batch([p, p], ["7", "12"], CPU)
    assert tok.end_token_id not in ids.tolist()[0] + ids.tolist()[1]
    assert mask.sum().item() == 3                                       # '7' and '1','2': digits only


def test_e22_generate_leaves_the_model_in_eval_mode(dlfs6_path):
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer
    from codebot.utils import generate

    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    m = GPT(1000, 16, 16, 2, 1, 32, 0.1)
    m.train()
    torch.manual_seed(0)
    generate(m, tok, "def", max_new_tokens=2)
    assert not m.training


def test_e23_tiny_temperature_gives_nan():
    logits = torch.tensor([[2.0, 1.0, 0.5]])
    probs = torch.softmax(logits / 1e-45, dim=-1)
    assert torch.isnan(probs).all()
    with pytest.raises(RuntimeError):
        torch.multinomial(probs, 1)


def test_e24_single_token_decode_and_webbot_partial_utf8(dlfs6_path):
    import pickle

    from codebot.tokenizer import BPETokenizer as CodeTok
    from storybot.tokenizer import BPETokenizer as StoryTok

    code = CodeTok.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    ids = code.encode("世")
    assert ids == [228, 184, 150] and [code.decode([i]) for i in ids] == ["�"] * 3
    web = StoryTok(pickle.loads((DLFS6 / "webbot" / "merge_rules.pkl").read_bytes()))
    bad = 0
    for b in web.id_to_bytes.values():
        try:
            b.decode("utf-8")
        except UnicodeDecodeError:
            bad += 1
    assert web.vocab_size == 50_000 and bad - 128 == 157               # 157 merged tokens (+ bytes 0x80-0xff)


def test_e25_uint16_cannot_hold_ids_above_65535():
    assert np.iinfo(np.uint16).max == 65535
    with pytest.raises(OverflowError):
        np.array([70000], dtype=np.uint16)
    assert "np.array(ids, dtype=np.uint16)" in src("storybot/tokenizer.py")


def test_e26_count_pairs_signatures_differ(dlfs6_path):
    import inspect

    from codebot.tokenizer import count_pairs as code_cp
    from storybot.tokenizer import count_pairs as story_cp

    assert list(inspect.signature(code_cp).parameters) == ["ids", "counts"]
    assert list(inspect.signature(story_cp).parameters) == ["ids", "weight", "counts"]
    with pytest.raises(TypeError):
        story_cp([1, 2, 3], {})                                         # dict lands in `weight`


def test_e27_merge_rules_order_is_part_of_the_data(dlfs6_path):
    import pickle

    from codebot.tokenizer import BPETokenizer

    rules = pickle.loads((DLFS6 / "codebot" / "merge_rules.pkl").read_bytes())
    shuffled = dict(sorted(rules.items(), key=lambda kv: -kv[1]))      # same pairs, reversed order
    with pytest.raises(KeyError):
        BPETokenizer(shuffled)


# ====================================================================== errata: comments, docs, portability

def test_z6_comment_doc_and_portability_facts():
    facts = [
        # (entry, file, text that must be present)
        ("E35", "ch05/05_kvcache.py", "# Causal Mask（KVキャッシュ使用時は不要）"),
        ("E35", "ch05/05_kvcache.py", "start_ids = torch.tensor([[42]])  # 固定シード"),
        ("E36", "README.md", "$ python ch02/10_gpt.py"),
        ("E39", "ch05/03_rmsnorm.py", "def __init__(self, x):"),
        ("E40", "ch02/08_multi_head.py", "H, D = self.n_head, self.head_dim  # ヘッドサイズ、ヘッドの次元数"),
        ("E41", "ch03/09_grpo.py", 'plt.savefig("loss_grpo.png")'),
        ("E42", "ch06/lr_graph.py", "plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS"),
        ("E44", "ch09/01_pretrain.py", "with autocast(device_type='cuda', dtype=torch.bfloat16):"),
        ("E45", "ch09/01_pretrain.py", "is_main = (not ddp) or (local_rank == 0)"),
        ("E45", "ch09/01_pretrain.py", "'effective_batch_size': micro_batch_size * accumulation_steps,"),
        ("E45", "ch09/01_pretrain.py", "train_loss = loss.item() * accumulation_steps"),
        ("E47", "ch01/08_eval.py", 'sample_text = open("codebot/tiny_codes.txt").read()[:10000]'),
        ("E38", "README.md", "* PyTorch 2.x"),
        ("E38", "requirements.txt", "torch>=2.5.0"),
        ("E38", "webbot/model.py", "enable_gqa=True"),
    ]
    for entry, rel, needle in facts:
        assert needle in src(rel), (entry, rel, needle)
    assert not (DLFS6 / "ch02" / "10_gpt.py").exists() and (DLFS6 / "ch02" / "10_gpt2.py").exists()
    assert "torch.manual_seed" not in src("ch05/05_kvcache.py")                                          # E35
    kv_tree = ast.parse(src("ch05/05_kvcache.py"))
    kv_names = {t.id for n in kv_tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    kv_strings = [n.value.value for n in kv_tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)]
    assert "max_new_tokens" not in kv_names and any("max_new_tokens=max_new_tokens" in s for s in kv_strings)  # E35
    unseeded = [r for r in ("ch03/01_pretrain.py", "ch03/03_sft.py", "ch03/09_grpo.py", "ch06/05_pretrain.py",
                            "ch06/08_dpo.py") if "manual_seed" not in src(r)]
    assert len(unseeded) == 5 and "torch.manual_seed(42 + seed_offset)" in src("ch09/01_pretrain.py")   # E43
    plots = ast.unparse(ast.parse(src("ch03/09_grpo.py")))
    assert "plt.plot(steps, accuracies)" in plots and "plt.ylabel('Accuracy (%)')" in plots              # E41
    strings = [n.value for n in ast.walk(ast.parse(src("ch01/08_eval.py")))
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert any("import tiktoken" in s for s in strings)                                                   # E47
    tok_tree = ast.parse(src("storybot/tokenizer.py"))
    fns = {n.name: n for n in tok_tree.body if isinstance(n, ast.FunctionDef)}
    body = lambda f: [ast.dump(s) for s in f.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert body(fns["process_single_chunk"]) == body(fns["pretoken_chunk"])[1:]   # E46: same body minus the args unpacking
    assert "process_single_chunk" not in {n.id for n in ast.walk(tok_tree) if isinstance(n, ast.Name)}
    assert git("show", "-s", "--format=%ad", "--date=short", "f85907f").strip() == "2026-04-10"           # E38
    assert set(git("show", "--name-only", "--format=", "f85907f").split()) == {"pyproject.toml", "requirements.txt"}


def test_e42_hiragino_falls_back_to_a_font_without_japanese():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    path = font_manager.findfont(font_manager.FontProperties(family="Hiragino Sans"))
    if "hiragino" in Path(path).name.lower():
        pytest.skip("this machine has Hiragino Sans")
    charmap = FT2Font(path).get_charmap()
    assert ord("A") in charmap and ord("学") not in charmap and ord("ウ") not in charmap


def test_e44_cuda_autocast_does_nothing_on_cpu():
    x = torch.randn(4, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = x @ x
    assert y.dtype == torch.float32                                     # CPU math stays float32
    if not torch.cuda.is_available():
        assert any("CUDA is not available" in str(w.message) for w in caught)


def test_e47_compression_on_the_first_10k_chars_vs_whole_corpus(dlfs6_path):
    from codebot.tokenizer import BPETokenizer

    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    text = (DLFS6 / "codebot" / "tiny_codes.txt").read_text(encoding="utf-8")
    sample = text[:10000]
    assert round(len(sample.encode()) / len(tok.encode(sample)), 4) == 2.1236
    n_tokens = (DLFS6 / "codebot" / "tiny_codes.bin").stat().st_size // 2
    assert round(len(text.encode()) / n_tokens, 4) == 2.4229


def test_e29_text_files_are_opened_with_the_locale_encoding():
    no_encoding = []
    for rel in ("ch01/07_tiny_codes.py", "ch03/03_alpaca.py", "ch03/03_sft.py", "ch04/01_bpe_optimize.py",
                "ch06/08_dpo.py"):
        for n in ast.walk(ast.parse(src(rel))):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"
                    and len(n.args) < 2 and "encoding" not in {k.arg for k in n.keywords}):
                no_encoding.append(rel)
    assert len(no_encoding) == 5                                        # text mode, locale encoding
    raw = (DLFS6 / "codebot" / "tiny_codes_sft.json").read_bytes()
    as_utf8, as_gbk = json.loads(raw.decode("utf-8")), json.loads(raw.decode("gbk"))   # GBK raises no error
    changed = [(a["response"], b["response"]) for a, b in zip(as_utf8, as_gbk) if a != b]
    assert len(changed) == 2 and all("O(n²)" in a and "O(n虏)" in b for a, b in changed)
    with pytest.raises(UnicodeDecodeError):
        (DLFS6 / "storybot" / "tiny_stories_valid.txt").read_bytes().decode("gbk")


def test_e30_soft_dictionary_softmax_overflows():
    soft = load_defs("ch02/01_soft_dict.py", names={"soft_dictionary"})["soft_dictionary"]
    movies = {(8, 2, 3): 85, (3, 9, 1): 70}
    rating, _ = soft((6, 4, 5), movies)                                 # returns (result, weights)
    assert 70 < rating < 85                                             # the book's scale: fine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rating, weights = soft((100, 100, 100), movies)                # scores 1300 and 1300: exp -> inf
    assert math.isnan(rating) and np.isnan(weights).all()


def test_e31_rope_tables_are_saved_into_every_layer(dlfs6_path):
    from storybot.model import GPT

    m = GPT(50, 16, 32, 2, 3, 64)
    assert sum("rope.cos_cache" in k or "rope.sin_cache" in k for k in m.state_dict()) == 6
    with pytest.raises(RuntimeError, match="size mismatch"):
        GPT(50, 32, 32, 2, 3, 64).load_state_dict(m.state_dict())     # same weights, longer context


def test_e32_grpo_evaluation_scores_prompt_plus_response():
    tree = ast.parse(src("ch03/09_grpo.py"))
    reward = load_defs("ch03/09_grpo.py", names={"calculate_reward"})["calculate_reward"]
    lines = src("ch03/09_grpo.py").splitlines()
    assert "response = full_text[len(prompt):]" in lines[84]            # training: response only
    assert lines[193].strip() == "response = generate(model, tokenizer, prompt, temperature=0)"   # eval: full text
    prompt = "### Instruction:\n{}+{}=\n\n### Response:\n"
    assert reward(5, prompt.format(0, 5)) == 1.0                         # no answer, yet the prompt's 5 counts
    assert all(reward(i + j, prompt.format(i, j)) == 0.0 for i in range(1, 10) for j in range(1, 10))
    assert tree is not None


def test_e33_shipped_merge_rules_pickles_hold_only_data():
    import pickletools

    data_ops = {"PROTO", "FRAME", "EMPTY_DICT", "MEMOIZE", "MARK", "BININT1", "BININT2", "BININT",
                "TUPLE2", "SETITEMS", "SETITEM", "STOP"}
    for bot in ("codebot", "storybot", "webbot"):
        ops = {op.name for op, _, _ in pickletools.genops((DLFS6 / bot / "merge_rules.pkl").read_bytes())}
        assert ops <= data_ops, (bot, ops - data_ops)                 # no GLOBAL / REDUCE: nothing executes
    assert "pickle.load(f)" in src("storybot/tokenizer.py")


def test_fixed_upstream_history_is_in_the_pinned_submodule():
    log = git("log", "--format=%h %s")
    for sha in ("8b3911f", "1760066", "c240183", "6855e70"):
        assert sha in log, sha
    assert "ß" not in src("ch06/07_llm_judge.py")                     # issue #4 (stray character) is fixed


# ====================================================================== errata: page data

def grpo_dropout_runs() -> dict:
    """Condense the recorded research runs (base vs. dropout-off) for the errata page."""
    runs = {}
    for name, label, seed in (("base", "原样（更新时 dropout 开）", 0), ("nodrop", "更新前 model.eval()", 0),
                              ("base_s1", "原样（更新时 dropout 开）", 1), ("nodrop_s1", "更新前 model.eval()", 1)):
        rows = [json.loads(l) for l in (RESEARCH / f"ch03_grpo_metrics_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l]
        evals = [r for r in rows if "greedy_acc" in r]
        runs[name] = {"label": label, "seed": seed,
                      "it": [r["it"] for r in rows],
                      "reward_mean": [round(r["reward_mean"], 4) for r in rows],
                      "grad_norm": [round(r["grad_norm0"], 3) for r in rows],
                      "acc_it": [r["it"] for r in evals],
                      "greedy_acc": [round(r["greedy_acc"], 2) for r in evals]}
    return {"source": "research/data/ch03_grpo_metrics_*.jsonl",
            "note": "研究阶段在本机 CPU 上的记录：批量采样版 GRPO（与 09_grpo.py 同一算法、同一超参），"
                    "每 10 次迭代（及最后一次）用 greedy 解码在 81 道题上测准确率；单 seed，各一次运行。",
            "runs": runs}


def test_errata_data_files_are_current():
    assert json.loads((WEB / "errata" / "grpo_dropout_runs.json").read_text(encoding="utf-8")) == grpo_dropout_runs()


# ====================================================================== cs336

def a1_record() -> dict:
    return json.loads((Path(__file__).resolve().parent / "cs336" / "results.json").read_text(encoding="utf-8"))


def test_cs336_a1_record_is_consistent():
    rec = a1_record()
    assert json.loads((WEB / "cs336" / "a1_results.json").read_text(encoding="utf-8")) == rec
    assert rec["cs336_a1"]["commit"] == "a158843b20107949f1a8d7df1b05cd33b9166712"
    tests = {t["test"]: t for t in rec["tests"]}
    assert len(tests) == 48
    for v in ("as_shipped", "one_line_fixes"):
        outcomes = [t[v]["outcome"] for t in rec["tests"]]
        assert rec["summary"][v] == {k: outcomes.count(k) for k in ("passed", "failed", "skipped")}
    flipped = {n for n, t in tests.items()
               if t["as_shipped"]["outcome"] == "failed" and t["one_line_fixes"]["outcome"] == "passed"}
    assert flipped == {"test_transformer_lm", "test_transformer_lm_truncated_input", "test_train_bpe",
                       "test_train_bpe_special_tokens", "test_get_batch"}
    still = {n for n, t in tests.items() if t["one_line_fixes"]["outcome"] == "failed"}
    assert still - {"test_train_bpe_speed"} == {"test_get_lr_cosine_schedule", "test_checkpointing",
                                                 "test_overlapping_special_tokens",
                                                 "test_encode_iterable_tinystories_sample_roundtrip",
                                                 "test_encode_iterable_tinystories_matches_tiktoken"}
    bpe = rec["diagnostics"]["as_shipped"]["train_bpe"][0]
    assert (bpe["first_divergence"], bpe["identical_positions"]) == (92, 170)
    assert rec["diagnostics"]["one_line_fixes"]["train_bpe"][0]["exact"]
    ts = rec["diagnostics"]["as_shipped"]["train_bpe"][1]                       # 5 MB TinyStories sample
    assert (ts["first_divergence"], ts["n_merges"], ts["vocab_values_equal"]) == (34, 743, True)
    assert rec["diagnostics"]["one_line_fixes"]["train_bpe"][1]["exact"]
    # the speed gate (1.5 s) on corpus.en: as shipped the five timings straddle it, after the fix all miss it
    shipped, fixed = (rec["diagnostics"][v]["train_bpe"][0]["train_seconds_runs"] for v in ("as_shipped", "one_line_fixes"))
    assert (round(min(shipped), 2), round(max(shipped), 2)) == (1.27, 1.52) and min(shipped) < 1.5 < max(shipped)
    assert (round(min(fixed), 2), round(max(fixed), 2)) == (1.80, 2.48)
    rms = rec["diagnostics"]["as_shipped"]["rmsnorm_vs_snapshot"]
    assert round(rms["nn.RMSNorm(d) default eps"], 6) == 1.7e-05                  # "only 1.7e-5" for test_rmsnorm
    worst = lambda name, v: tests[name][v]["worst"]
    assert round(worst("test_transformer_block", "as_shipped")["max_abs_diff"], 6) == 9.2e-05   # one block passes
    assert round(worst("test_transformer_lm", "as_shipped")["max_abs_diff"], 5) == 4.6e-04
    assert round(worst("test_transformer_lm", "one_line_fixes")["max_abs_diff"], 7) == 1.05e-05
    assert round(worst("test_transformer_lm_truncated_input", "as_shipped")["tol_ratio"], 2) == 1.53


@pytest.mark.skipif(not os.environ.get("CS336_A1_DIR"), reason="set CS336_A1_DIR to a clone of assignment1-basics")
def test_cs336_a1_live_spot_checks():
    """Re-derive the record's key numbers from the A1 fixtures (no CS336 code is run)."""
    from cs336 import adapters_dlfs6 as A

    a1 = Path(os.environ["CS336_A1_DIR"])
    sd = {k.replace("_orig_mod.", ""): v for k, v in torch.load(a1 / "tests/fixtures/ts_tests/model.pt",
                                                               map_location=CPU).items()}
    torch.manual_seed(6)
    idx = torch.randint(0, 10_000, (4, 12))                            # A1 fixture `in_indices`
    want = np.load(a1 / "tests/_snapshots/test_transformer_lm.npz")["array"]
    errs = {}
    for eps in (None, 1e-5):
        from storybot.model import GPT
        model = GPT(10_000, 16, 64, 4, 3, 128, 10000.0)
        for mod in model.modules():
            if isinstance(mod, nn.RMSNorm):
                mod.eps = eps
        A._load_exact(model, A.rename_lm_state(sd))
        with torch.no_grad():
            errs[eps] = float(np.abs(model.eval()(idx).numpy() - want).max())
    assert 4e-4 < errs[None] < 5e-4 and errs[1e-5] < 2e-5


def test_cs336_storybot_pretraining_equals_a1_tinystories_config():
    tree = ast.parse(src("ch06/05_pretrain.py"))
    consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    a1 = {"context_len": 256, "vocab_size": 10000, "embed_dim": 512, "n_head": 16, "n_layer": 4,
          "ff_dim": 1344, "theta": 10000, "batch_size": 32}                # A1 handout, TinyStories LM
    assert {k: consts[k] for k in a1} == a1
    assert consts["max_iters"] * consts["batch_size"] * consts["context_len"] == 327_680_000
    assert "save_iters = [500, 5000]" in src("ch06/05_pretrain.py")
    assert 5000 * 32 * 256 == 40_960_000                               # model_iter_5000 = A1 low-resource budget


def test_cs336_local_tinystories_valid_matches_the_recorded_hash():
    import hashlib

    ident = json.loads((WEB / "cs336" / "data_identity.json").read_text(encoding="utf-8"))
    pair = next(p for p in ident["pairs"] if p["dlfs6"]["url"].endswith("storybot/tiny_stories_valid.txt"))
    data = (DLFS6 / "storybot" / "tiny_stories_valid.txt").read_bytes()
    assert len(data) == pair["dlfs6"]["size"] == pair["cs336"]["size"]
    assert hashlib.sha256(data).hexdigest() == pair["dlfs6"]["sha256"] == pair["cs336"]["sha256"]


def test_cs336_storybot_parameter_count(dlfs6_path):
    from storybot.model import GPT

    m = GPT(10000, 256, 512, 16, 4, 1344, 10000)
    total = sum(p.numel() for p in m.parameters())
    emb = m.embed.weight.numel()
    assert total == 22_696_448 and total - emb == 17_576_448           # A1: "~17M non-embedding"
    assert sum(p.numel() for p in m.blocks.parameters()) == 12_455_936


def test_cs336_storybot_initial_logit_scale(dlfs6_path):
    """sigma(logits) ~ sqrt(d) * 0.02 because the final RMSNorm gives every vector RMS 1."""
    from storybot.model import GPT

    torch.manual_seed(0)
    m = GPT(10000, 256, 512, 16, 4, 1344).eval()
    ids = torch.randint(0, 10000, (4, 64))
    with torch.no_grad():
        logits = m(ids)
        ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 10000), ids[:, 1:].reshape(-1)).item()
    sigma = logits.std().item()
    assert abs(sigma - math.sqrt(512) * 0.02) < 0.02                   # 0.453
    assert abs(ce - (math.log(10000) + sigma ** 2 / 2)) < 0.03          # ~9.31, above ln V = 9.21


def test_cs336_dpo_loss_is_ln2_at_initialisation(dlfs6_path):
    from storybot.model import GPT

    d = load_defs("ch06/08_dpo.py", names={"get_sequence_logprobs", "compute_dpo_loss"})
    torch.manual_seed(0)
    m = GPT(50, 16, 16, 2, 1, 32).eval()
    ids = torch.randint(0, 50, (4, 10))
    mask = torch.ones(4, 10, dtype=torch.long)
    with torch.no_grad():
        loss = d["compute_dpo_loss"](m, m, ids, mask, ids.flip(1), mask, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6


def test_cs336_grpo_shape_matches_a5_rollout_batch():
    tree = ast.parse(src("ch03/09_grpo.py"))
    c = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
         for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert (c["group_size"], c["batch_size"], c["epsilon"], c["n_update_per_generation"]) == (8, 32, 0.2, 2)
    assert c["group_size"] * c["batch_size"] == 256                    # A5: rollout_batch_size 256, G = 8
    assert "advantages = rewards - rewards.mean()" in src("ch03/09_grpo.py")   # no std division


# ====================================================================== glossary

def test_glossary_sources_match_the_pinned_code():
    data = json.loads((WEB / "glossary" / "sources.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _glossary_sources import dlfs6_signatures

    assert data["dlfs6"]["signatures"] == dlfs6_signatures()
    a1 = os.environ.get("CS336_A1_DIR")
    if a1:
        from _glossary_sources import signature
        for name, args in data["cs336_a1"]["signatures"].items():
            assert signature(Path(a1) / "tests" / "adapters.py", None, name) == args


def test_glossary_epoch_two_ways():
    c = consts("ch03/01_pretrain.py")
    tokens = (DLFS6 / "codebot" / "tiny_codes.bin").stat().st_size // 2
    windows = tokens - c["context_len"]                                 # len(TokenDataset)
    seen = c["max_iters"] * c["batch_size"]
    assert round(seen * c["context_len"] / tokens, 1) == 61.2            # "61 epochs" of tokens
    assert round(seen / windows, 3) == 0.239                            # 0.24 epoch of the DataLoader
    fp = finetune_passes()                                              # fine-tuning: counted in real batches
    assert fp["SFT"][:3] == (1092, 32, 500) and round(fp["SFT"][3], 1) == 14.3      # 35 batches a pass
    assert round(500 * 32 / 1092, 1) == 14.7                                          # the naive count
    assert fp["DPO"][:3] == (144, 8, 1000) and round(fp["DPO"][3], 1) == 55.6


if __name__ == "__main__":
    out = WEB / "errata" / "grpo_dropout_runs.json"
    out.write_text(json.dumps(grpo_dropout_runs(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
