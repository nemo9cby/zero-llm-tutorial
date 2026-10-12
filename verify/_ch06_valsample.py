"""第 6 章的记录运行：在整个验证集上随机抽窗口，估计官方 checkpoint 的验证 loss 及其置信区间。

checkpoints.json 只用了 tiny_stories_valid.bin 开头的 131,072 个 token（约 2.4%）。CS336 的两条及格线
（4096 万 token 时 ≤ 2.00、3.28 亿 token 时 ≤ 1.45）是按整个验证集算的，所以这里把整个验证集切成
不重叠的 256-token 窗口（与 ch06/05_pretrain.py 的 evaluate 取的窗口相同），用固定种子不放回地随机抽
一部分，在 fp32 下（不开 autocast）逐窗口算平均 loss。页面上的验证单元用这些逐窗口的 loss 自己算均值和
置信区间。

用法（CPU，每个 checkpoint 约 5 分钟）：
  CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_valsample.py --out web/data/ch06/val_sample.json
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
from _ch06_common import ROOT, STORYBOT, load_storybot  # noqa: E402

CHECKPOINTS = [(5000, "model_iter_5000.pt"), (40000, "model_pretrain.pt")]
CONTEXT, BATCH, N_SAMPLE, SEED = 256, 32, 2048, 0


def validation_data() -> np.memmap:
    return np.memmap(STORYBOT / "tiny_stories_valid.bin", dtype=np.uint16, mode="r")


def sampled_windows(n_windows: int, n_sample: int = N_SAMPLE, seed: int = SEED) -> np.ndarray:
    """Window indices drawn without replacement (sorted, so reads go through the file in order)."""
    return np.sort(np.random.default_rng(seed).choice(n_windows, size=n_sample, replace=False))


def window_losses(model, data, windows) -> np.ndarray:
    """Mean next-token cross-entropy (nats) of each 256-token window, fp32, eval mode."""
    out = np.empty(len(windows))
    model.eval()
    with torch.no_grad():
        for s in range(0, len(windows), BATCH):
            idx = windows[s:s + BATCH]
            x = torch.from_numpy(np.stack([data[i * CONTEXT:(i + 1) * CONTEXT] for i in idx]).astype(np.int64))
            y = torch.from_numpy(np.stack([data[i * CONTEXT + 1:(i + 1) * CONTEXT + 1] for i in idx]).astype(np.int64))
            logits = model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
            out[s:s + len(idx)] = nll.view(len(idx), CONTEXT).mean(dim=1).double().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data" / "ch06" / "val_sample.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    data = validation_data()
    n_windows = (len(data) - 1) // CONTEXT          # every window whose shifted target also fits
    windows = sampled_windows(n_windows)
    t0 = time.time()
    rows = []
    for it, name in CHECKPOINTS:
        losses = window_losses(load_storybot(STORYBOT / name), data, windows)
        rows.append({"iter": it, "checkpoint": f"storybot/{name}", "window_losses": [round(float(v), 5) for v in losses]})
        print(f"iter {it:5d}: mean {losses.mean():.4f} over {len(losses)} windows ({time.time() - t0:.0f} s)", flush=True)

    report = {
        "what": "official StoryBot checkpoints: mean fp32 loss of randomly sampled 256-token windows of the whole validation set",
        "command": "CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_valsample.py",
        "validation": {"file": "storybot/tiny_stories_valid.bin", "tokens": int(len(data)), "context_len": CONTEXT,
                       "n_windows": int(n_windows)},
        "sample": {"n": N_SAMPLE, "seed": SEED, "without_replacement": True,
                   "indices": "np.sort(np.random.default_rng(seed).choice(n_windows, size=n, replace=False))"},
        "provenance": {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                       "threads": args.threads, "seconds": round(time.time() - t0, 1)},
        "checkpoints": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
