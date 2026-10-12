"""第 6 章的记录运行：在 StoryBot 同款结构的缩小版上扫描学习率（6.3.1 节“第一个学习率是从哪里来的？”）。

模型：storybot.model.GPT，vocab 10000、ctx 64、d 64、4 头、2 层、ff 172（1,379,136 个参数）。
数据：storybot/tiny_stories_valid.bin 的前 900,000 个 token 用来训练，紧接着的 4,097 个 token 做验证
     （训练语料 tiny_stories_train.bin 有 1.08 GB，这里只演示方法，用不着它）。
训练：每个学习率 150 步，batch 16，原书 05_pretrain.py 的 get_lr（预热 10 步、D2Z）、get_batch、
     梯度裁剪 1.0、torch.optim.AdamW 默认参数；每个学习率都从 torch.manual_seed(0) 开始。

用法（CPU，几分钟）：
  CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_lr_sweep.py --out web/data/ch06/lr_sweep.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch06_common import ROOT, heldout_tokens, load_defs  # noqa: E402

CONFIG = dict(vocab_size=10000, max_context_len=64, embed_dim=64, n_head=4, n_layer=2, ff_dim=172, theta=10000)
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
STEPS, BATCH, WARMUP, N_TRAIN = 150, 16, 10, 900_000


def sweep_data():
    data = heldout_tokens(N_TRAIN + 64 * 64).astype(np.uint16)  # N_TRAIN + 4097 tokens
    return data[:N_TRAIN], data[N_TRAIN:]


def train_one(lr: float, steps: int = STEPS):
    """Train the tiny StoryBot for `steps` steps at peak learning rate `lr`; return (per-step losses, model)."""
    ns = load_defs("ch06/05_pretrain.py", ["get_lr", "get_batch"])
    from storybot.model import GPT

    train, _ = sweep_data()
    torch.manual_seed(0)
    model = GPT(**CONFIG)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for i in range(steps):
        for group in opt.param_groups:
            group["lr"] = ns["get_lr"](i, lr, WARMUP, STEPS)
        x, y = ns["get_batch"](train, CONFIG["max_context_len"], BATCH, torch.device("cpu"))
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses, model


def val_loss(model) -> float:
    _, val = sweep_data()
    t = torch.from_numpy(val.astype(np.int64))
    x, y = t[:-1].view(64, 64), t[1:].view(64, 64)
    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data" / "ch06" / "lr_sweep.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    rows = []
    for lr in LRS:
        losses, model = train_one(lr)
        rows.append({"lr": lr, "losses": [round(x, 6) for x in losses], "val_loss": val_loss(model)})
        print(f"lr {lr:g}: last train loss {losses[-1]:.4f}, val loss {rows[-1]['val_loss']:.4f}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "what": "learning-rate sweep on a tiny StoryBot (storybot.model.GPT)",
        "command": "CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_lr_sweep.py",
        "config": {**CONFIG, "params": n_params, "steps": STEPS, "batch_size": BATCH, "warmup": WARMUP,
                   "schedule": "get_lr from ch06/05_pretrain.py (warmup + linear decay to 0)", "grad_clip": 1.0,
                   "optimizer": "torch.optim.AdamW defaults", "seed": 0,
                   "train_tokens": "tiny_stories_valid.bin[:900000]", "val_tokens": 4096},
        "provenance": {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "platform": platform.platform(), "torch": torch.__version__, "threads": args.threads,
                       "seconds": round(time.time() - t0, 1)},
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
