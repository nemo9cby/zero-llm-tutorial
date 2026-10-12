"""重新生成第 3 章页面用到的小数据文件（web/data/ch03/），全部来自原书代码与官方 checkpoint。

    CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch03_export.py [--only name,...]

每个文件都能被 verify/test_ch03.py 重新推导出来（测试会逐项核对），所以页面上的数字有据可查。
只在 CPU 上运行，固定随机种子。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch03_dlfs6 import ALPACA, CODEBOT, DLFS6, ROOT, arithmetic_prompts, load_defs  # noqa: E402

OUT = ROOT / "web" / "data" / "ch03"
sys.path.insert(0, str(DLFS6))

# 采样台（图 3-x）用到的上下文：(模型, 标签, 原始 prompt)
NEXT_TOKEN_CONTEXTS = [
    ("pretrain", "def", "def"),
    ("pretrain", "for i in", "for i in"),
    ("pretrain", "def add(a, b):↵    return", "def add(a, b):\n    return"),
    ("pretrain", "print(", "print("),
    ("sft", "SFT：3+4=", ALPACA.format("3+4=")),
    ("grpo", "GRPO：3+4=", ALPACA.format("3+4=")),
]


def dump(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote web/data/ch03/{name} ({(OUT / name).stat().st_size:,} B)")


def models():
    from codebot.model import GPT
    out = {}
    for k in ("pretrain", "sft", "grpo"):
        m = GPT.load_from(str(CODEBOT / f"model_{k}.pt"), device="cpu")
        m.eval()
        out[k] = m
    return out


def token_strings(tok) -> list[str]:
    return [tok.id_to_bytes[i].decode("utf-8", errors="replace") for i in range(tok.vocab_size)]


@torch.no_grad()
def next_token_logits(model, tok, prompt: str) -> list[float]:
    ids = torch.tensor([tok.encode(prompt)])
    return model(ids)[0, -1].tolist()


@torch.no_grad()
def answer_dists(model, tok) -> list[list[float]]:
    """P(response == str(a), then <|endoftext|>) for a = 0..18 under T=1; last column = 1 - sum.
    Only the canonical tokenization of each answer is counted (the tokenizer's own encode)."""
    eos = tok.end_token_id
    rows = []
    for prompt, _ in arithmetic_prompts():
        p = tok.encode(prompt)
        row = []
        for a in range(19):
            ans = tok.encode(str(a)) + [eos]
            seq = torch.tensor([p + ans])
            lp = F.log_softmax(model(seq)[0, len(p) - 1:-1], dim=-1)
            row.append(lp.gather(-1, torch.tensor(ans)[:, None]).sum().exp().item())
        rows.append(row + [max(0.0, 1.0 - sum(row))])
    return rows


def export_vocab(tok) -> None:
    dump("vocab.json", {"source": "codebot/merge_rules.pkl (BPETokenizer.id_to_bytes, decoded with errors='replace')",
                        "end_token_id": tok.end_token_id, "tokens": token_strings(tok)})


def export_next_token(tok, ms) -> None:
    ctx = []
    for model, label, prompt in NEXT_TOKEN_CONTEXTS:
        lg = next_token_logits(ms[model], tok, prompt)
        ctx.append({"model": model, "label": label, "prompt": prompt, "logits": [round(x, 4) for x in lg]})
    dump("next_token.json", {"source": "official codebot checkpoints (model_pretrain/sft/grpo.pt), eval mode, CPU",
                             "contexts": ctx})


def export_sft_tokens(tok) -> None:
    data = json.loads((CODEBOT / "tiny_codes_sft.json").read_text(encoding="utf-8"))
    prompt_ids, response_ids = [], []
    for item in data:
        prompt_ids.append(tok.encode(ALPACA.format(item["instruction"])))
        response_ids.append(tok.encode(f"{item['response']}<|endoftext|>"))
    dump("sft_tokens.json", {
        "source": "codebot/tiny_codes_sft.json tokenized exactly like SFTDataset._create_sample (ch03/03_sft.py)",
        "context_len": 256, "instructions": [d["instruction"] for d in data],
        "prompt_ids": prompt_ids, "response_ids": response_ids})


def export_answer_dists(tok, ms) -> None:
    res = {"source": "official codebot checkpoints; P(response == answer, then <|endoftext|>) at T=1, canonical "
                     "tokenization; last column 'other' = 1 - sum",
           "prompts": [f"{i}+{j}=" for i in range(1, 10) for j in range(1, 10)],
           "answers": [str(a) for a in range(19)] + ["other"]}
    for k in ("sft", "grpo"):
        res[k] = [[round(x, 5) for x in r] for r in answer_dists(ms[k], tok)]
    dump("answer_dists.json", res)


def first_update_ratios(tok, seed: int = 0, n_prompts: int = 32, group_size: int = 8) -> dict:
    """Reproduce the very first GRPO update of ch03/09_grpo.py: old_model == model (same weights),
    responses sampled by the book's own generate_group. Returns per-response-token ratios with the
    policy in train() mode (as the script does) and in eval() mode."""
    from codebot.model import GPT
    g = load_defs("ch03/09_grpo.py", ["GRPODataset", "calculate_reward", "generate_group", "compute_probs"])
    torch.manual_seed(seed)
    model = GPT.load_from(str(CODEBOT / "model_sft.pt"), device="cpu")
    old = GPT.load_from(str(CODEBOT / "model_sft.pt"), device="cpu")
    old.eval()
    ds = g["GRPODataset"](tok)
    prompts = [p for p, _ in ds.data[:n_prompts]]
    gts = [t for _, t in ds.data[:n_prompts]]
    P, R, A = g["generate_group"](old, tok, prompts, gts, group_size)
    ids, mask = ds.get_batch(P, R, "cpu")
    m = mask[:, 1:].bool()
    with torch.no_grad():
        po = g["compute_probs"](old, ids)
        torch.manual_seed(seed + 1)
        model.train()
        rt = (g["compute_probs"](model, ids) / (po + 1e-8))[m]
        model.eval()
        re_ = (g["compute_probs"](model, ids) / (po + 1e-8))[m]
    return {"responses": R, "advantages": [round(a, 6) for a in A.tolist()],
            "train": [round(x, 5) for x in rt.tolist()], "eval": [round(x, 7) for x in re_.tolist()]}


def export_ratios(tok) -> None:
    r = first_update_ratios(tok)
    rt = np.array(r["train"])
    out = {"source": "ch03/09_grpo.py functions on model_sft.pt; first 32 prompts x 8 rollouts; torch.manual_seed(0) "
                     "before sampling, manual_seed(1) before the train-mode forward; CPU",
           **r,
           "summary": {"n_tokens": int(rt.size), "train_mean": float(rt.mean()), "train_std": float(rt.std()),
                       "train_outside": float(np.mean((rt < 0.8) | (rt > 1.2)))}}
    dump("ratio_first_update.json", out)


def export_corpus_head() -> None:
    ids = np.fromfile(CODEBOT / "tiny_codes.bin", dtype=np.uint16)
    OUT.mkdir(parents=True, exist_ok=True)
    ids[:4096].tofile(OUT / "tiny_codes_head.bin")
    print("wrote web/data/ch03/tiny_codes_head.bin (first 4096 tokens)")


def free_running_indices(n: int = 48, seed: int = 0) -> list[int]:
    return sorted(np.random.default_rng(seed).choice(1092, size=n, replace=False).tolist())


@torch.no_grad()
def sft_free_running_lengths(tok, model, indices: list[int], max_new_tokens: int = 200) -> dict:
    """Teacher forcing vs free running: sample one response per prompt from the SFT model (T=1,
    the book's generate()) and compare its token length with the dataset's target response.
    Each prompt gets its own seed (torch.manual_seed(i)), so any subset can be re-checked alone."""
    from codebot.utils import generate
    data = json.loads((CODEBOT / "tiny_codes_sft.json").read_text(encoding="utf-8"))
    gen, tgt = [], []
    for i in indices:
        prompt = ALPACA.format(data[i]["instruction"])
        torch.manual_seed(i)
        full = generate(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=1.0)
        gen.append(len(tok.encode(full[len(prompt):])))
        tgt.append(len(tok.encode(data[i]["response"])))
    return {"indices": list(indices), "generated": gen, "target": tgt}


def export_free_running(tok, ms) -> None:
    r = sft_free_running_lengths(tok, ms["sft"], free_running_indices())
    dump("sft_free_running.json", {"source": "model_sft.pt + codebot.utils.generate (T=1, max_new_tokens=200), "
                                             "torch.manual_seed(index) per prompt; 48 prompts chosen by "
                                             "np.random.default_rng(0); lengths in tokens, EOS excluded", **r})


CHAT_PROMPTS = ["Hello", "Who are you?", "print hello", "write loop", "3+4=", "7+8=", "5+7="]


@torch.no_grad()
def chat_greedy(tok, model, instruction: str, max_new_tokens: int = 100) -> str:
    """04_chat.py 的做法（套模板、取 ### Response: 之后），但用 T=0，结果可复现。"""
    from codebot.utils import generate
    prompt = ALPACA.format(instruction)
    return generate(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=0)[len(prompt):].strip()


@torch.no_grad()
def pretrain_sample(tok, model, seed: int, max_new_tokens: int = 60) -> str:
    from codebot.utils import generate
    torch.manual_seed(seed)
    return generate(model, tok, "def", max_new_tokens=max_new_tokens, temperature=1.0)


def export_samples(tok, ms) -> None:
    dump("samples.json", {
        "source": "official checkpoints + codebot.utils.generate on CPU; chat = greedy (T=0); "
                  "pretrain = T=1 with torch.manual_seed(seed)",
        "chat": [{"instruction": q, "sft": chat_greedy(tok, ms["sft"], q), "grpo": chat_greedy(tok, ms["grpo"], q)}
                 for q in CHAT_PROMPTS],
        "pretrain_def": [{"seed": s, "text": pretrain_sample(tok, ms["pretrain"], s)} for s in (0, 1, 2)]})


EXPORTS = {
    "vocab": lambda tok, ms: export_vocab(tok),
    "next_token": export_next_token,
    "sft_tokens": lambda tok, ms: export_sft_tokens(tok),
    "answer_dists": export_answer_dists,
    "ratios": lambda tok, ms: export_ratios(tok),
    "corpus_head": lambda tok, ms: export_corpus_head(),
    "free_running": export_free_running,
    "samples": export_samples,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated subset of: " + ",".join(EXPORTS))
    args = ap.parse_args()
    from codebot.tokenizer import BPETokenizer
    torch.set_num_threads(4)
    tok = BPETokenizer.load_from(str(CODEBOT / "merge_rules.pkl"))
    ms = models()
    for name in (args.only.split(",") if args.only else EXPORTS):
        EXPORTS[name](tok, ms)


if __name__ == "__main__":
    main()
