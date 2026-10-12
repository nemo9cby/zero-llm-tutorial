"""第 6 章的 DPO 记录运行：在 CPU 上跑 ch06/08_dpo.py 的训练，并做训练后分析，结果写成 JSON。

与原书脚本相比只有四处不同（每一处都有 verify/test_ch06.py 里的测试支撑）：
  1. 加载 HF 上的权重时把 `.mlp.` 改名为 `.ffn.`，否则 load_state_dict 报错
     （test_hf_checkpoint_needs_key_rename）。
  2. context_len 256 -> 96：最长的样本只有 81 个 token，右侧 padding 全被 mask 掉，
     每条序列的 log-prob 与 256 时相同（test_dpo_masks_on_real_model），只是省了 2/3 的计算。
  3. 开头加 torch.manual_seed(0) 与 torch.set_num_threads()：原脚本没有设种子，结果不可复现。
  4. 每步的 loss 写进 JSON；训练前后在 144 条训练 pair 与 8 条留出 pair 上计算隐式奖励，
     在留出的 TinyStories 文本上计算语言模型 loss，并用相同种子各生成几段样本。
其余（数据、模型、β=0.1、lr=5e-6、batch 8、1000 步、AdamW 默认参数、cycle(DataLoader)）与原书相同，
DPODataset / get_sequence_logprobs / compute_dpo_loss 直接取自原书脚本。

用法（约 1 小时，取决于 CPU）：
  CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_dpo_run.py --out web/data/ch06/dpo_cpu.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch06_common import DLFS6, ROOT, STORYBOT, heldout_tokens, lm_loss, load_defs, load_storybot  # noqa: E402

BETA = 0.1


def pair_metrics(model, ref, ns, tokenizer, pairs, context_len, beta=BETA):
    """Implicit rewards beta*(log pi - log pi_ref) on (prompt, chosen, rejected) pairs, via the book's code."""
    ds = ns["DPODataset"].__new__(ns["DPODataset"])  # reuse the book's _create_sample on in-memory pairs
    ds.tokenizer, ds.context_len = tokenizer, context_len
    ds.samples = [ds._create_sample(p["prompt"], p["chosen"], p["rejected"]) for p in pairs]
    c, cm, r, rm = [torch.stack(t) for t in zip(*[ds[i] for i in range(len(ds))])]
    lp = ns["get_sequence_logprobs"]
    out = {k: [] for k in ("pc", "pr", "qc", "qr")}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(c), 16):
            sl = slice(i, i + 16)
            out["pc"].append(lp(model, c[sl], cm[sl]))
            out["pr"].append(lp(model, r[sl], rm[sl]))
            out["qc"].append(lp(ref, c[sl], cm[sl]))
            out["qr"].append(lp(ref, r[sl], rm[sl]))
    pc, pr, qc, qr = (torch.cat(out[k]) for k in ("pc", "pr", "qc", "qr"))
    margin = beta * ((pc - qc) - (pr - qr))
    return {
        "n": len(pairs),
        "reward_acc": (margin > 0).float().mean().item(),
        "margin_mean": margin.mean().item(),
        "dpo_loss": (-F.logsigmoid(margin)).mean().item(),
        "chosen_shift_nats": (pc - qc).mean().item(),
        "rejected_shift_nats": (pr - qr).mean().item(),
        "ref_prefers_chosen_by_sum": ((qc - qr) > 0).float().mean().item(),
        "chosen_tokens_mean": cm[:, 1:].sum(1).float().mean().item(),
        "rejected_tokens_mean": rm[:, 1:].sum(1).float().mean().item(),
    }


def setup(context_len: int = 96, threads: int = 4):
    """Prepare DPO exactly in the book's order: data -> DataLoader -> two models -> optimizer.

    GPT's constructor draws random numbers before the weights are overwritten, so the order matters
    for the shuffle that DataLoader draws on its first iteration.
    """
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    ns = load_defs("ch06/08_dpo.py", ["DPODataset", "get_sequence_logprobs", "compute_dpo_loss"])
    from storybot.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.load_from(str(STORYBOT / "merge_rules.pkl"))
    dataset = ns["DPODataset"](str(STORYBOT / "tiny_stories_dpo.json"), tokenizer, context_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = load_storybot(STORYBOT / "model_pretrain.pt")
    ref_model = load_storybot(STORYBOT / "model_pretrain.pt")
    ref_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    return ns, tokenizer, model, ref_model, optimizer, cycle(dataloader)


def train_step(ns, model, ref_model, optimizer, data_iter) -> float:
    """One iteration of the book's training loop (08_dpo.py lines 133-148)."""
    chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)
    loss = ns["compute_dpo_loss"](model, ref_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, BETA)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--context-len", type=int, default=96)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-at", default="0,100,300,1000", help="steps at which to measure the 144 training pairs")
    ap.add_argument("--heldout-tokens", type=int, default=65536)
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data" / "ch06" / "dpo_cpu.json")
    ap.add_argument("--save-model", type=Path, default=None, help="optional path for the DPO weights (90 MB)")
    args = ap.parse_args()

    t_start = time.time()
    ns, tokenizer, model, ref_model, optimizer, data_iter = setup(args.context_len, args.threads)
    from storybot.utils import generate

    train_pairs = json.loads((STORYBOT / "tiny_stories_dpo.json").read_text(encoding="utf-8"))
    heldout_pairs = json.loads((ROOT / "web" / "data" / "ch06" / "dpo_heldout_pairs.json").read_text(encoding="utf-8"))
    eval_at = {int(s) for s in args.eval_at.split(",") if s}

    losses, evals = [], {}
    t_train = time.time()
    for i in range(args.iters + 1):
        if i in eval_at:
            evals[str(i)] = pair_metrics(model, ref_model, ns, tokenizer, train_pairs, args.context_len)
            model.train()  # the book's model stays in train mode (StoryBot has no dropout, so it does not matter)
            print(f"[eval @ {i}] {evals[str(i)]}", flush=True)
        if i == args.iters:
            break
        losses.append(train_step(ns, model, ref_model, optimizer, data_iter))
        if i % 10 == 0 or i == args.iters - 1:
            el = time.time() - t_train
            print(f"step {i:4d} loss {losses[-1]:.4f}  ({el / (i + 1):.2f} s/step)", flush=True)
    train_seconds = time.time() - t_train

    if args.save_model:
        model.save(str(args.save_model))

    post = {
        "train_pairs": evals.get(str(args.iters)),
        "heldout_pairs_before": pair_metrics(ref_model, ref_model, ns, tokenizer, heldout_pairs, args.context_len),
        "heldout_pairs": pair_metrics(model, ref_model, ns, tokenizer, heldout_pairs, args.context_len),
    }
    tokens = heldout_tokens(args.heldout_tokens)
    post["lm_loss_pretrain"] = lm_loss(ref_model, tokens)
    post["lm_loss_dpo"] = lm_loss(model, tokens)
    post["lm_loss_tokens"] = args.heldout_tokens

    samples = []
    for seed in (1, 2, 3):
        row = {"seed": seed}
        for name, m in (("pretrain", ref_model), ("dpo", model)):
            torch.manual_seed(seed)
            row[name] = generate(m, tokenizer, "Once upon a time", 120, 1.0).strip()
        samples.append(row)

    report = {
        "what": "ch06/08_dpo.py on CPU (context_len 96, seed 0) + post-training analysis",
        "command": "CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_dpo_run.py",
        "provenance": {
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dlfs6_commit": git_rev(DLFS6), "platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "threads": args.threads, "train_seconds": round(train_seconds, 1),
            "total_seconds": round(time.time() - t_start, 1),
            "note": "CPU shared with other jobs while measuring; wall time is an upper bound.",
        },
        "config": {"beta": BETA, "lr": 5e-6, "batch_size": 8, "iters": args.iters, "context_len": args.context_len,
                   "optimizer": "torch.optim.AdamW defaults (betas 0.9/0.999, eps 1e-8, wd 0.01)",
                   "reference": "storybot/model_pretrain.pt (frozen)", "n_train_pairs": len(train_pairs),
                   "n_heldout_pairs": len(heldout_pairs)},
        "losses": [round(x, 6) for x in losses],
        "evals": evals,
        "post": post,
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({train_seconds / 60:.1f} min of training)")


if __name__ == "__main__":
    main()
