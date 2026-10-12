"""第 8 章 8.5 节：比较 eager 与 torch.compile 的训练步耗时（WebBot 的结构，原书 webbot/model.py）。

在 Linux 的 GPU 上（或 Windows 的 WSL2 里）运行：
    uv run python verify/_ch08_compile_bench.py --device cuda --micro-batch 4 --steps 30
在本机 CPU 上只做冒烟测试（Windows 上默认的 inductor 需要 cl.exe，所以换成不生成代码的 aot_eager）：
    uv run python verify/_ch08_compile_bench.py --device cpu --tiny --backend aot_eager --warmup 1 --steps 2

两种模式用同一个随机种子初始化、喂同样的 batch，训练步与 ch09/01_pretrain.py 相同
（AdamW、bf16 autocast（只在 CUDA 上）、梯度裁剪 1.0）。最后打印一行 JSON：
环境、配置、eager 与 compile 每步耗时的中位数（毫秒）、各自第一步的耗时（compile 的第一步包含编译），
以及两者第一步 loss 的差。这个脚本不写任何文件。
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "dlfs6"))
from webbot.model import GPT  # noqa: E402

WEBBOT = dict(vocab_size=50000, max_context_len=1024, embed_dim=768, n_head=12, n_kv_head=4, n_layer=12, ff_dim=2048)
TINY = dict(WEBBOT, max_context_len=128, embed_dim=128, n_head=4, n_kv_head=2, n_layer=2, ff_dim=344)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--micro-batch", type=int, default=4, help="ch09 用 32；8 GB 显存的卡用 4")
    ap.add_argument("--steps", type=int, default=30, help="计时的步数")
    ap.add_argument("--warmup", type=int, default=5, help="不计时的预热步数（compile 的编译发生在第一步）")
    ap.add_argument("--backend", default="inductor", help="torch.compile 的后端：inductor（默认）或 aot_eager 等")
    ap.add_argument("--tiny", action="store_true", help="2 层、宽 128 的小模型，用于在 CPU 上冒烟")
    return ap.parse_args(argv)


def train_steps(model, cfg, args, device):
    """跑 warmup + steps 个训练步，返回 (每步秒数列表, 第一步的 loss)。"""
    use_amp = device.type == "cuda"
    T = cfg["max_context_len"]
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    seconds, first_loss = [], None
    for step in range(args.warmup + args.steps):
        g = torch.Generator().manual_seed(step)                         # 两种模式喂同样的数据
        ids = torch.randint(0, cfg["vocab_size"], (args.micro_batch, T + 1), generator=g)
        x, y = ids[:, :-1].contiguous().to(device), ids[:, 1:].contiguous().to(device)
        sync()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(model(x).view(-1, cfg["vocab_size"]), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        seconds.append(time.perf_counter() - t0)
        if first_loss is None:
            first_loss = loss.item()
    return seconds, first_loss


def main(argv=None):
    args = parse_args(argv)
    if args.warmup < 1 or args.steps < 1:
        sys.exit("--warmup 和 --steps 都至少为 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("没有可用的 CUDA 设备：去掉 --device cuda，或者换一台有 GPU 的机器")
    cfg = TINY if args.tiny else WEBBOT

    results = {}
    for mode in ("eager", "compile"):
        torch.manual_seed(0)                                            # 两种模式的初始权重相同
        model = GPT(**cfg).to(device)
        if mode == "compile":
            torch._dynamo.reset()
            model = torch.compile(model, backend=args.backend)
        try:
            seconds, first_loss = train_steps(model, cfg, args, device)
        except Exception as e:                                          # 例如 Windows 上没有 cl.exe
            sys.exit(f"{mode} 模式失败：{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        results[mode] = {"first_step_s": round(seconds[0], 3),
                         "median_ms": round(1000 * statistics.median(seconds[args.warmup:]), 2),
                         "first_loss": first_loss}

    report = {
        "host": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                 "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()},
        "config": {**cfg, "micro_batch": args.micro_batch, "backend": args.backend,
                   "warmup": args.warmup, "steps": args.steps, "bf16_autocast": device.type == "cuda"},
        "eager": results["eager"],
        "compile": results["compile"],
        "speedup": round(results["eager"]["median_ms"] / results["compile"]["median_ms"], 3),
        "first_loss_diff": abs(results["eager"]["first_loss"] - results["compile"]["first_loss"]),
    }
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
