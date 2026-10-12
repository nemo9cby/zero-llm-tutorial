"""Recorded data for the opening pages (ch00, part1, part2, part3): scale table, dataset statistics,
and endpoint previews generated from the official checkpoints.

Everything runs on CPU with fixed seeds and only *imports* the book's modules, so nothing is written
into third_party/dlfs6. `--write` regenerates the files under web/data/{ch00,part1,part2,part3}/;
verify/test_ch00.py and verify/test_part{1,2,3}.py recompute the same quantities and compare.

    CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 uv run python verify/_parts_preview.py --write
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
import platform
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
WEB_DATA = ROOT / "web" / "data"
EOT = "<|endoftext|>"
TEMPLATE = "### Instruction:\n{}\n\n### Response:\n"          # ch03/04_chat.py format_prompt
CHAT_PROMPTS = ["Hello", "Who are you?", "write loop", "7+8=", "make variable x", "print hello"]
WEBBOT_PROMPTS = [  # ch09/02_generate.py
    "In 1991, Linus Torvalds created", "Monday, Tuesday, Wednesday,", "Python was created by",
    "Machine learning is defined as", "The capital of Japan is",
]
HF_LISTING = "https://huggingface.co/api/datasets/koki0702/zero-llm-data/tree/main?recursive=true"
# Sizes from the Hugging Face listing above (fetched 2026-10-11). The training corpora are not downloaded
# by default (tools/fetch_data.py --full), so their sizes are recorded facts, cross-checked against the
# manifest in tools/fetch_data.py by the tests.
HF_SIZES = {
    "storybot/tiny_stories_train.txt": 2_227_753_162,
    "storybot/tiny_stories_train.bin": 1_082_458_446,
    "webbot/owt_train.txt": 11_920_511_059,
    "webbot/owt_train.bin": 5_298_009_600,
    "webbot/owt_valid.txt": 289_998_753,
}
# sha256 of the official checkpoints, as listed by Hugging Face (LFS oid), fetched 2026-10-11.
HF_SHA256 = {
    "codebot/model_pretrain.pt": "3e32e4103c5d58a89bed032b2c497e86eb26c58dd0ca42eeef7565d5910a3ecb",
    "codebot/model_sft.pt": "8d3f09b5737e0fbd27c25e337e0f69373ff29e1c2aecf24c1323ca41a8354f14",
    "codebot/model_grpo.pt": "e89848d846fe1a8bac5651dfac0c454685f401addbaaf340cf6ac8e9d939832d",
}


# ---------------------------------------------------------------- small utilities

def book_path() -> None:
    """Make `import codebot...` work the way the book's scripts do."""
    if str(DLFS6) not in sys.path:
        sys.path.insert(0, str(DLFS6))


@contextmanager
def single_thread():
    """Token-by-token generation is tiny work per step: one thread is faster on a busy CPU (same outputs)."""
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(**extra) -> dict:
    import torch
    return {
        "generated_by": "CUDA_VISIBLE_DEVICES=-1 uv run python verify/_parts_preview.py --write",
        "date": date.today().isoformat(),
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dlfs6_commit": git_rev(DLFS6),
        **extra,
    }


def script_hparams(rel: str) -> dict:
    """Module-level `name = <constant expression>` assignments of a book script, evaluated without running it."""
    tree = ast.parse((DLFS6 / rel).read_text(encoding="utf-8"))
    env: dict = {}
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.List):
            return [ev(e) for e in node.elts]
        raise ValueError("not a constant expression")

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = ev(node.value)
            except (ValueError, KeyError):
                pass
    return env


def book_function(rel: str, name: str, **globs):
    """Extract one top-level function from a book script (AST) without running the script's training loop."""
    src = (DLFS6 / rel).read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = dict(globs)
            exec(compile(ast.Module(body=[node], type_ignores=[]), f"{rel}:{name}", "exec"), ns)
            return ns[name]
    raise LookupError(f"{name} not found in {rel}")


def non_ascii(raw: bytes) -> int:
    return int((np.frombuffer(raw, dtype=np.uint8) > 127).sum())


def doc_lengths(ids: np.ndarray, end_id: int) -> np.ndarray:
    """Tokens between consecutive end tokens (the end token itself excluded)."""
    cut = np.flatnonzero(ids == end_id)
    return np.diff(np.concatenate([[-1], cut, [len(ids)]])) - 1


def summarize(lengths: np.ndarray, ctx: int) -> dict:
    return {
        "n": int(len(lengths)), "median": float(np.median(lengths)), "mean": round(float(lengths.mean()), 2),
        "p90": float(np.percentile(lengths, 90)), "max": int(lengths.max()),
        "over_ctx": int((lengths > ctx).sum()), "over_ctx_frac": round(float((lengths > ctx).mean()), 4),
    }


def take_docs(text: str, nbytes: int, skip_first: bool = False) -> str:
    """Whole documents from the start of `text` until at least `nbytes` UTF-8 bytes are collected."""
    docs = text.split(EOT)[1 if skip_first else 0:]
    out, n = [], 0
    for d in docs:
        out.append(d)
        n += len(d.encode("utf-8"))
        if n >= nbytes:
            break
    return EOT.join(out)


# ---------------------------------------------------------------- ch00: the three bots

def param_split(model) -> dict:
    """Group a DLFS6 GPT's parameters (shared tensors counted once) the way the scale table shows them."""
    split = {"embedding": 0, "position": 0, "blocks": 0, "final_norm": 0, "output_extra": 0}
    for name, p in model.named_parameters():          # remove_duplicate=True: a tied matrix appears once
        key = ("embedding" if name.startswith("embed.") else "position" if name.startswith("pos_embed.")
               else "blocks" if name.startswith("blocks.") else "final_norm" if name.startswith("norm.")
               else "output_extra")
        split[key] += p.numel()
    return split


PRETRAIN = {"codebot": "ch03/01_pretrain.py", "storybot": "ch06/05_pretrain.py", "webbot": "ch09/01_pretrain.py"}


def build_models(seed: int = 0) -> dict:
    """The three bots exactly as their pretraining scripts construct them (CPU, eval mode)."""
    import torch
    book_path()
    from codebot.model import GPT as CodeGPT
    from storybot.model import GPT as StoryGPT
    from webbot.model import GPT as WebGPT
    c, s, w = (script_hparams(PRETRAIN[k]) for k in ("codebot", "storybot", "webbot"))
    torch.manual_seed(seed)
    models = {
        "codebot": CodeGPT(c["vocab_size"], c["context_len"], c["embed_dim"], c["n_head"], c["n_layer"],
                           c["ff_dim"], c["dropout_rate"]),
        "storybot": StoryGPT(s["vocab_size"], s["context_len"], s["embed_dim"], s["n_head"], s["n_layer"],
                             s["ff_dim"], s["theta"]),
        "webbot": WebGPT(w["vocab_size"], w["context_len"], w["embed_dim"], w["n_head"], w["n_kv_head"],
                         w["n_layer"], w["ff_dim"], w["theta"]),
    }
    return {k: m.eval() for k, m in models.items()}


def initial_loss(model, tokens: np.ndarray, rows: int = 4, width: int = 256) -> float:
    """Cross-entropy of an untrained model on the first rows×width tokens of a real corpus."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.asarray(tokens[: rows * width + 1]).astype(np.int64))
    inp, tgt = x[:-1].view(rows, width), x[1:].view(rows, width)
    with torch.no_grad():
        logits = model(inp)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()


CORPORA = {  # (training corpus as the scripts read it, text file it was encoded from, end-token id)
    "codebot": ("codebot/tiny_codes.bin", "codebot/tiny_codes.txt", 999),
    "storybot": ("storybot/tiny_stories_train.bin", "storybot/tiny_stories_train.txt", 9999),
    "webbot": ("webbot/owt_train.bin", "webbot/owt_train.txt", 49999),
}
LOCAL_TOKENS = {  # a real corpus that is on disk after tools/fetch_data.py, used for the initial-loss check
    "codebot": "codebot/tiny_codes.bin", "storybot": "storybot/tiny_stories_valid.bin", "webbot": "webbot/owt_valid.bin",
}


def file_size(rel: str) -> int:
    p = DLFS6 / rel
    return p.stat().st_size if p.is_file() else HF_SIZES[rel]


def bots_table(models: dict | None = None) -> dict:
    models = models or build_models()
    hp = {k: script_hparams(s) for k, s in PRETRAIN.items()}
    sft = json.loads((DLFS6 / "codebot/tiny_codes_sft.json").read_text(encoding="utf-8"))
    dpo = json.loads((DLFS6 / "storybot/tiny_stories_dpo.json").read_text(encoding="utf-8"))
    sft_hp, grpo_hp, dpo_hp = (script_hparams(s) for s in ("ch03/03_sft.py", "ch03/09_grpo.py", "ch06/08_dpo.py"))
    meta = {
        "codebot": dict(name="CodeBot", part="I", page="part1.html", data="TinyCodes",
                        chapters={"tokenizer": "ch01", "model": "ch02", "training": "ch03"},
                        precision="fp32", arch="GPT-2"),
        "storybot": dict(name="StoryBot", part="II", page="part2.html", data="TinyStories V2-GPT4",
                         chapters={"tokenizer": "ch04", "model": "ch05", "training": "ch06"},
                         precision="bf16 autocast", arch="RoPE + SwiGLU + RMSNorm"),
        "webbot": dict(name="WebBot", part="III", page="part3.html", data="OpenWebText (CS336 sample)",
                       chapters={"tokenizer": "ch07", "model": "ch08", "training": "ch09"},
                       precision="bf16 autocast + torch.compile", arch="StoryBot + GQA + SDPA"),
    }
    bots = []
    for key, m in models.items():
        h = hp[key]
        token_file, text_file, end_id = CORPORA[key]
        accum = h.get("accumulation_steps", 1)
        batch = h.get("batch_size", h.get("micro_batch_size"))
        tokens_per_iter = batch * accum * h["context_len"]
        post = []
        if key == "codebot":
            post = [
                {"stage": "SFT", "script": "ch03/03_sft.py", "data": "codebot/tiny_codes_sft.json",
                 "examples": len(sft), "max_iters": sft_hp["max_iters"], "batch_size": sft_hp["batch_size"],
                 "learning_rate": sft_hp["learning_rate"]},
                {"stage": "GRPO", "script": "ch03/09_grpo.py", "data": "81 道 i+j=（i, j = 1…9）",
                 "examples": 81, "group_size": grpo_hp["group_size"], "batch_size": grpo_hp["batch_size"],
                 "max_iters": grpo_hp["max_iters"], "learning_rate": grpo_hp["learning_rate"]},
            ]
        elif key == "storybot":
            post = [{"stage": "DPO", "script": "ch06/08_dpo.py", "data": "storybot/tiny_stories_dpo.json",
                     "examples": len(dpo), "max_iters": dpo_hp["max_iters"], "batch_size": dpo_hp["batch_size"],
                     "learning_rate": dpo_hp["learning_rate"], "beta": dpo_hp["beta"]}]
        bots.append({
            "id": key, **meta[key],
            "corpus": {"text_file": text_file, "text_bytes": file_size(text_file),
                       "token_file": token_file, "tokens": file_size(token_file) // 2, "end_token_id": end_id},
            "model": {"vocab_size": h["vocab_size"], "context_len": h["context_len"], "embed_dim": h["embed_dim"],
                      "n_layer": h["n_layer"], "n_head": h["n_head"], "n_kv_head": h.get("n_kv_head", h["n_head"]),
                      "ff_dim": h["ff_dim"]},
            "params": int(sum(p.numel() for p in m.parameters())),
            "params_split": param_split(m),
            "pretrain": {"script": PRETRAIN[key],
                         "max_iters": h["max_iters"], "batch_size": batch, "accumulation_steps": accum,
                         "context_len": h["context_len"], "tokens_per_iter": tokens_per_iter,
                         "tokens": h["max_iters"] * tokens_per_iter, "learning_rate": h["learning_rate"]},
            "post": post,
            "init_loss": round(initial_loss(m, np.memmap(DLFS6 / LOCAL_TOKENS[key], dtype=np.uint16, mode="r")), 4),
            "init_loss_data": LOCAL_TOKENS[key],
        })
    return {"_provenance": provenance(seed=0, hf_listing=HF_LISTING, hf_listing_date="2026-10-11",
                                      note="初始 loss：torch.manual_seed(0) 依次构造三个模型（eval 模式），"
                                           "在 init_loss_data 的前 4×256 个 token 上计算交叉熵"),
            "bots": bots}


# ---------------------------------------------------------------- part I: CodeBot

def codebot_preview() -> dict:
    book_path()
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer
    from codebot.utils import generate
    calculate_reward = book_function("ch03/09_grpo.py", "calculate_reward", re=re)
    tok = BPETokenizer.load_from(str(DLFS6 / "codebot/merge_rules.pkl"))

    def reply(full: str) -> str:            # ch03/04_chat.py: keep what follows "### Response:"
        return full.split("### Response:")[-1].strip()

    models = {}
    for name in ("pretrain", "sft", "grpo"):
        rel = f"codebot/model_{name}.pt"
        model = GPT.load_from(str(DLFS6 / rel), device="cpu").eval()
        entry = {"file": rel, "sha256": sha256(DLFS6 / rel)}
        with single_thread():
            entry["greedy"] = [reply(generate(model, tok, TEMPLATE.format(p), max_new_tokens=200, temperature=0))
                               for p in CHAT_PROMPTS]
            if name == "pretrain":       # a base model only continues text: show what it does with "def"
                entry["def_continuation"] = generate(model, tok, "def", max_new_tokens=40, temperature=0)
            else:
                answers, correct = {}, 0
                for i in range(1, 10):
                    for j in range(1, 10):
                        full = generate(model, tok, TEMPLATE.format(f"{i}+{j}="), temperature=0)  # as in 09_grpo.py
                        answers[f"{i}+{j}"] = reply(full)
                        correct += calculate_reward(i + j, full) > 0
                entry.update(addition_greedy=answers, addition_correct=int(correct))
        models[name] = entry
    return {"_provenance": provenance(
                chat="ch03/04_chat.py 的模板与 max_new_tokens=200，但用贪心解码（temperature 0）以便复现；"
                     "原脚本用 temperature 1.0 采样，每次的回答会不同",
                addition="ch03/09_grpo.py 的评测方式：generate(temperature=0) + calculate_reward（最后一个整数等于答案）",
                def_continuation="预训练检查点从 prompt \"def\" 贪心续写 40 个 token"),
            "prompts": CHAT_PROMPTS, "template": TEMPLATE, "models": models}


def tinycodes_stats() -> tuple[dict, str]:
    raw = (DLFS6 / "codebot/tiny_codes.txt").read_bytes()
    text = raw.decode("utf-8")
    programs = text.split(EOT)
    ids = np.fromfile(DLFS6 / "codebot/tiny_codes.bin", dtype=np.uint16)
    lengths = doc_lengths(ids, 999)
    head = EOT.join(programs[:500])
    stats = {
        "_provenance": provenance(source="codebot/tiny_codes.txt + codebot/tiny_codes.bin（完整语料）"),
        "name": "TinyCodes", "unit": "token（CodeBot 分词器）", "end_token_id": 999, "context_len": 256,
        "text_bytes": len(raw), "line_feeds": raw.count(b"\n"), "carriage_returns": raw.count(b"\r"),
        "non_ascii_bytes": non_ascii(raw), "programs": len(programs), "separators": text.count(EOT),
        "tokens": int(len(ids)), "summary": summarize(lengths, 256), "lengths": lengths.tolist(),
    }
    return stats, head


def ast_failures(programs: list[str]) -> int:
    import warnings
    bad = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # the corpus has invalid escape sequences (SyntaxWarning)
        for p in programs:
            try:
                ast.parse(p)
            except SyntaxError:
                bad += 1
    return bad


# ---------------------------------------------------------------- part II: StoryBot

def load_storybot(rel: str):
    """Official HF StoryBot checkpoints name the SwiGLU block `mlp`; storybot/model.py calls it `ffn`."""
    import torch
    book_path()
    from storybot.model import GPT
    ck = torch.load(DLFS6 / rel, map_location="cpu")
    state = {k.replace(".mlp.", ".ffn."): v for k, v in ck["model_state_dict"].items()}
    model = GPT(ck["vocab_size"], ck["max_context_len"], ck["embed_dim"], ck["n_head"], ck["n_layer"],
                ck["ff_dim"], ck["theta"])
    model.load_state_dict(state)
    return model.eval()


STORY_CKPTS = [("storybot/model_iter_500.pt", 500), ("storybot/model_iter_5000.pt", 5000),
               ("storybot/model_pretrain.pt", 40000)]


def storybot_val_loss(model, windows: int = 32, seed: int = 0) -> float:
    """Mean cross-entropy on `windows` random 256-token windows of the validation set (fixed rng)."""
    import torch
    import torch.nn.functional as F
    val = np.fromfile(DLFS6 / "storybot/tiny_stories_valid.bin", dtype=np.uint16)
    starts = np.random.default_rng(seed).integers(0, len(val) - 257, size=windows)
    x = torch.from_numpy(np.stack([val[s:s + 256] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([val[s + 1:s + 257] for s in starts]).astype(np.int64))
    total = 0.0
    with torch.no_grad():
        for k in range(0, windows, 16):
            logits = model(x[k:k + 16])
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[k:k + 16].reshape(-1),
                                     reduction="sum").item()
    return total / y.numel()


def storybot_preview(seeds=(0, 1)) -> dict:
    import torch
    book_path()
    from storybot.tokenizer import BPETokenizer
    from storybot.utils import generate
    tok = BPETokenizer.load_from(str(DLFS6 / "storybot/merge_rules.pkl"))
    out = []
    for rel, iters in STORY_CKPTS:
        model = load_storybot(rel)
        samples = []
        with single_thread():
            for seed in seeds:
                torch.manual_seed(seed)
                story = generate(model, tok, EOT, max_new_tokens=255, temperature=1.0)
                samples.append({"seed": seed, "text": story, "tokens": len(tok.encode(story))})
        out.append({"file": rel, "iters": iters, "tokens_seen": iters * 32 * 256,
                    "val_loss_subset": round(storybot_val_loss(model), 4), "samples": samples})
    return {"_provenance": provenance(
                generate="storybot/utils.generate，prompt = <|endoftext|>（同 ch06/06_generate.py），temperature 1.0，"
                         "max_new_tokens 255（书中为 300；255 + 1 个 prompt token 正好不超过上下文 256）",
                val_loss_subset="验证集中 32 个随机 256-token 窗口（np.random.default_rng(0)），fp32",
                key_rename="HF 检查点的 blocks.N.mlp.* 改名为 blocks.N.ffn.*"),
            "checkpoints": out}


def tinystories_stats() -> tuple[dict, str]:
    raw = (DLFS6 / "storybot/tiny_stories_valid.txt").read_bytes()
    text = raw.decode("utf-8")
    segments = text.split(EOT)
    ids = np.fromfile(DLFS6 / "storybot/tiny_stories_valid.bin", dtype=np.uint16)
    lengths = doc_lengths(ids, 9999)
    head = EOT.join(segments[1:201])          # segment 0 starts mid-sentence: skip it
    stats = {
        "_provenance": provenance(source="storybot/tiny_stories_valid.txt + .bin（验证集全量）"),
        "name": "TinyStories（验证集）", "unit": "token（StoryBot 分词器）", "end_token_id": 9999, "context_len": 256,
        "text_bytes": len(raw), "non_ascii_bytes": non_ascii(raw), "segments": len(segments),
        "separators": text.count(EOT), "first_segment_head": segments[0][:60],
        "tokens": int(len(ids)), "summary": summarize(lengths, 256), "lengths": lengths.tolist(),
    }
    return stats, head


# ---------------------------------------------------------------- part III: WebBot

OWT_EXCERPT_DOC = 9   # a short science-news document near the start of owt_valid.bin


def owt_stats() -> dict:
    book_path()
    from storybot.tokenizer import BPETokenizer
    ids = np.memmap(DLFS6 / "webbot/owt_valid.bin", dtype=np.uint16, mode="r")
    lengths = doc_lengths(np.asarray(ids), 49999)
    tok = BPETokenizer.load_from(str(DLFS6 / "webbot/merge_rules.pkl"))
    sample = np.asarray(ids[:2_000_000]).tolist()
    sample_bytes = len(tok.decode(sample).encode("utf-8"))
    return {
        "_provenance": provenance(source="webbot/owt_valid.bin（验证集全量）+ webbot/merge_rules.pkl"),
        "name": "OpenWebText（验证集）", "unit": "token（WebBot 分词器）", "end_token_id": 49999, "context_len": 1024,
        "tokens": int(len(ids)), "summary": summarize(lengths, 1024),
        "bytes_per_token_first_2M": round(sample_bytes / len(sample), 4), "lengths": lengths.tolist(),
    }


def owt_document(k: int) -> str:
    book_path()
    from storybot.tokenizer import BPETokenizer
    ids = np.memmap(DLFS6 / "webbot/owt_valid.bin", dtype=np.uint16, mode="r")
    cut = np.flatnonzero(np.asarray(ids[:3_000_000]) == 49999)
    start = 0 if k == 0 else int(cut[k - 1]) + 1
    tok = BPETokenizer.load_from(str(DLFS6 / "webbot/merge_rules.pkl"))
    return tok.decode(np.asarray(ids[start:int(cut[k])]).tolist())


def tokenizer_matrix() -> dict:
    """bytes/token of each bot's tokenizer on ~50 KB of each bot's data, plus one web sentence as chips."""
    import pickle
    book_path()
    from storybot.tokenizer import BPETokenizer
    toks = {b: BPETokenizer(pickle.load(open(DLFS6 / f"{b}/merge_rules.pkl", "rb")))
            for b in ("codebot", "storybot", "webbot")}
    web_ids = np.asarray(np.memmap(DLFS6 / "webbot/owt_valid.bin", dtype=np.uint16, mode="r")[:40_000]).tolist()
    corpora = {
        "TinyCodes": take_docs((DLFS6 / "codebot/tiny_codes.txt").read_text(encoding="utf-8"), 50_000),
        "TinyStories": take_docs((DLFS6 / "storybot/tiny_stories_valid.txt").read_text(encoding="utf-8"), 50_000,
                                 skip_first=True),
        "OpenWebText": take_docs(toks["webbot"].decode(web_ids), 50_000),
    }
    matrix = []
    for cname, text in corpora.items():
        nbytes = len(text.encode("utf-8"))
        row = {"corpus": cname, "bytes": nbytes, "tokens": {}}
        for tname, tk in toks.items():
            ids = tk.encode(text)
            assert tk.decode(ids) == text
            row["tokens"][tname] = len(ids)
        matrix.append(row)
    sentence = owt_document(OWT_EXCERPT_DOC).split("\n")[0]
    sentence = sentence[: sentence.index("”") + 1]          # first sentence, ends with a closing quote
    chips = {}
    for tname, tk in toks.items():
        ids = tk.encode(sentence)
        chips[tname] = [{"id": i, "bytes": tk.id_to_bytes[i].hex()} for i in ids]
    return {"_provenance": provenance(
                note="所有分词器都用 storybot.tokenizer.BPETokenizer 的按 rank 合并编码；对 CodeBot 的合并规则，"
                     "它与 codebot.tokenizer 的逐条套用结果相同（test_parts.py 有对拍）",
                samples="TinyCodes 开头、TinyStories 验证集开头（跳过半截的第一段）、OWT 验证集开头，各取整篇文档直到 ≥ 50,000 字节"),
            "matrix": matrix, "sentence": sentence, "sentence_source": f"owt_valid.bin 第 {OWT_EXCERPT_DOC + 1} 篇文档的第一句",
            "chips": chips}


def webbot_preview(new_tokens: int = 30) -> dict:
    import torch
    book_path()
    from storybot.tokenizer import BPETokenizer
    from webbot.utils import generate
    model = build_models(seed=0)["webbot"]
    tok = BPETokenizer.load_from(str(DLFS6 / "webbot/merge_rules.pkl"))
    outs = []
    with single_thread():
        for p in WEBBOT_PROMPTS:
            torch.manual_seed(0)
            outs.append(generate(model, tok, p, max_new_tokens=new_tokens, temperature=0.5))
    loss = initial_loss(model, np.memmap(DLFS6 / "webbot/owt_valid.bin", dtype=np.uint16, mode="r"))
    cont_ids = [tok.encode(o[len(p):]) for p, o in zip(WEBBOT_PROMPTS, outs)]
    return {"_provenance": provenance(
                weights="未训练：torch.manual_seed(0) 后依次构造 CodeBot、StoryBot、WebBot（与 ch00 的 bots.json 同一顺序），取 WebBot",
                generate=f"webbot/utils.generate，temperature 0.5（同 ch09/02_generate.py），max_new_tokens {new_tokens}"
                         "（书中为 100），每个 prompt 前 torch.manual_seed(0)",
                continuation_ids="续写部分（去掉 prompt）用 WebBot 分词器重新编码的 id"),
            "prompts": WEBBOT_PROMPTS, "outputs": outs, "continuation_ids": cont_ids, "init_loss": round(loss, 4),
            "ln_vocab": round(math.log(50000), 4)}


# ---------------------------------------------------------------- write everything

def dump(obj, rel: str) -> None:
    path = WEB_DATA / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1 if "lengths" not in obj else None) + "\n",
                    encoding="utf-8")
    print(f"wrote web/data/{rel} ({path.stat().st_size:,} B)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="regenerate web/data/{ch00,part1,part2,part3}/")
    ap.add_argument("--only", help="comma-separated subset: bots,codebot,tinycodes,storybot,tinystories,owt,tokenizers,webbot")
    args = ap.parse_args()
    if not args.write:
        ap.print_help()
        return
    only = set(args.only.split(",")) if args.only else None
    want = lambda k: only is None or k in only  # noqa: E731
    if want("bots"):
        dump(bots_table(), "ch00/bots.json")
    if want("codebot"):
        dump(codebot_preview(), "part1/preview.json")
    if want("tinycodes"):
        stats, head = tinycodes_stats()
        dump(stats, "part1/doclen.json")
        (WEB_DATA / "part1/tiny_codes_head.txt").write_text(head, encoding="utf-8", newline="\n")
        shutil.copyfile(DLFS6 / "codebot/tiny_codes_sft.json", WEB_DATA / "part1/tiny_codes_sft.json")
        print("wrote web/data/part1/tiny_codes_head.txt, part1/tiny_codes_sft.json")
    if want("storybot"):
        dump(storybot_preview(), "part2/preview.json")
    if want("tinystories"):
        stats, head = tinystories_stats()
        dump(stats, "part2/doclen.json")
        (WEB_DATA / "part2/stories_head.txt").write_text(head, encoding="utf-8", newline="\n")
        print("wrote web/data/part2/stories_head.txt")
    if want("owt"):
        dump(owt_stats(), "part3/doclen.json")
    if want("tokenizers"):
        dump(tokenizer_matrix(), "part3/tokenizers.json")
    if want("webbot"):
        dump(webbot_preview(), "part3/preview.json")


if __name__ == "__main__":
    main()
