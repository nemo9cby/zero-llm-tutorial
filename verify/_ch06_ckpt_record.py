"""第 6 章的记录运行：官方 StoryBot checkpoint 在留出数据上的 loss，以及各 checkpoint 写出的故事。

  * loss：storybot/tiny_stories_valid.bin 开头的若干 token（训练只用了 tiny_stories_train.bin，所以这是留出数据），
    不重叠的 256-token 窗口，fp32（不开 autocast），nats/token；同时换算成困惑度和 bits/byte。
  * 样本：与 ch06/07_llm_judge.py 相同的设置（prompt "<|endoftext|>"、200 个新 token、温度 1.0），
    用 storybot.utils.generate 生成，种子固定。
  * 第 0 次迭代用 torch.manual_seed(0) 新建的随机初始化模型代替（官方没有发布）。

用法（CPU，几分钟）：
  CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_ckpt_record.py --out web/data/ch06/checkpoints.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch06_common import ROOT, STORYBOT, STORYBOT_CONFIG, ensure_dlfs6_on_path, heldout_tokens, lm_loss, load_storybot  # noqa: E402

CHECKPOINTS = [(500, "model_iter_500.pt"), (5000, "model_iter_5000.pt"), (40000, "model_pretrain.pt")]
N_LARGE, N_SMALL = 131072, 8192
SAMPLE_SEEDS = (0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data" / "ch06" / "checkpoints.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ensure_dlfs6_on_path()
    from storybot.model import GPT
    from storybot.tokenizer import BPETokenizer
    from storybot.utils import generate

    tok = BPETokenizer.load_from(str(STORYBOT / "merge_rules.pkl"))
    big = heldout_tokens(N_LARGE)
    # bytes of text behind the predicted (target) tokens; the <|endoftext|> separators are not text
    n_bytes = sum(len(tok.id_to_bytes[i]) for i in big[1:].tolist() if i != tok.end_token_id)
    t0 = time.time()

    rows = []
    torch.manual_seed(0)
    fresh = GPT(**STORYBOT_CONFIG)
    with torch.no_grad():
        logits = fresh(torch.from_numpy(big[:256][None]))
    models = [(0, fresh)] + [(it, load_storybot(STORYBOT / f)) for it, f in CHECKPOINTS]
    for it, model in models:
        loss_big = lm_loss(model, big)
        loss_small = lm_loss(model, big[: N_SMALL + 1])
        steps = {0: 0, 500: 501, 5000: 5001, 40000: 40000}[it]  # save_iters are saved after that step's update
        row = {"iter": it, "steps": steps, "tokens_seen": steps * 32 * 256,
               "val_loss": loss_big, "val_loss_8192": loss_small,
               "ppl": math.exp(loss_big), "bits_per_token": loss_big / math.log(2),
               "bits_per_byte": loss_big * N_LARGE / (math.log(2) * n_bytes)}
        if it:
            row["samples"] = []
            for seed in SAMPLE_SEEDS:
                torch.manual_seed(seed)
                row["samples"].append(generate(model, tok, "<|endoftext|>", 200, 1.0).strip())
        rows.append(row)
        print(f"iter {it:5d}: val {loss_big:.4f} (8192 tok: {loss_small:.4f}) ppl {row['ppl']:.2f}", flush=True)

    report = {
        "what": "official StoryBot checkpoints on held-out TinyStories tokens + samples",
        "command": "CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_ckpt_record.py",
        "heldout": {"file": "storybot/tiny_stories_valid.bin", "tokens": N_LARGE, "bytes": n_bytes,
                    "bytes_per_token": n_bytes / N_LARGE, "context_len": 256, "dtype": "fp32"},
        "fresh_init": {"seed": 0, "logit_std": logits.std().item(),
                       "ln_v": math.log(10000), "ln_v_plus_var_over_2": math.log(10000) + logits.var().item() / 2},
        "sampling": {"prompt": "<|endoftext|>", "max_new_tokens": 200, "temperature": 1.0, "seeds": list(SAMPLE_SEEDS),
                     "threads": args.threads},
        "provenance": {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                       "seconds": round(time.time() - t0, 1)},
        "checkpoints": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
