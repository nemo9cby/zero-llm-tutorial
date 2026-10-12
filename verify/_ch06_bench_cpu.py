"""第 6 章的记录运行：CPU 上 StoryBot 一步前向加反向的耗时，fp32 与 bf16 autocast 对比（6.4.3 节）。

没有原生 bf16 矩阵指令（AVX512-BF16 或 AMX）的 CPU 上，bf16 autocast 反而更慢；有这些指令的 CPU 上结果可能相反。
结果与机器有关，所以记录里写明了处理器型号；测量时如有其他程序占用 CPU，数字只能作为上界参考。

用法：
  CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_bench_cpu.py --out web/data/ch06/cpu_bf16_bench.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch06_common import ROOT, STORYBOT_CONFIG, ensure_dlfs6_on_path  # noqa: E402


def seconds_per_step(autocast: bool, batch: int = 8, steps: int = 3) -> float:
    """Mean wall time of one StoryBot forward + backward (batch x 256 tokens) on the CPU, after one warm-up step."""
    ensure_dlfs6_on_path()
    from storybot.model import GPT

    torch.manual_seed(0)
    model = GPT(**STORYBOT_CONFIG)
    x = torch.randint(0, STORYBOT_CONFIG["vocab_size"], (batch, 256))
    y = torch.randint(0, STORYBOT_CONFIG["vocab_size"], (batch, 256))

    def step():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    step()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    return (time.perf_counter() - t0) / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data" / "ch06" / "cpu_bf16_bench.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    fp32 = seconds_per_step(False, steps=args.steps)
    bf16 = seconds_per_step(True, steps=args.steps)
    report = {
        "what": "StoryBot (22.7M) forward+backward, batch 8 x 256 tokens, CPU, fp32 vs bf16 autocast",
        "command": "CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_bench_cpu.py",
        "fp32_seconds_per_step": round(fp32, 3),
        "bf16_seconds_per_step": round(bf16, 3),
        "bf16_over_fp32": round(bf16 / fp32, 2),
        "provenance": {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "processor": platform.processor(), "platform": platform.platform(),
                       "torch": torch.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                       "threads": args.threads, "steps": args.steps,
                       "note": "other jobs may have shared the CPU; compare the ratio, not the absolute times"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
