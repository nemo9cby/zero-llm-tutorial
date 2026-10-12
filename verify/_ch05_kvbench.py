"""第 5 章：KV cache 的逐步延迟，CPU 或 GPU 都能跑。

    uv run python verify/_ch05_kvbench.py --device cpu --threads 1   # → web/data/ch05/kv_timing_cpu.json
    uv run python verify/_ch05_kvbench.py --device cuda              # → web/data/ch05/kv_timing_cuda.json

模型是 ch05/05_kvcache.py 测速用的配置（vocab 1000、E=384、6 头、6 层、ff 1024，权重随机），
只把 max_context_len 放大到 1024，好测到 t = 800。对每个长度 t 测两件事，各取多次运行的中位数：
  full   —— 不用 cache 时第 t 步要做的事：model(ids[:, :t])，把 t 个 token 全部重算；
  decode —— 用 cache 时第 t 步要做的事：先 prefill t−1 个 token，再只送最后 1 个 token。
另外照原书的方式完整生成 n 个 token（起始 token 42），记录两种生成的总耗时。
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import date

sys.dont_write_bytecode = True

from _ch05_util import ROOT, load_defs  # noqa: E402

MODEL = dict(vocab_size=1000, max_context_len=1024, embed_dim=384, n_head=6, n_layer=6, ff_dim=1024)


def device_name(device: str) -> str:
    import torch

    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def cpu_load_percent() -> int | None:
    """Other processes' load while we measure (the numbers are only as clean as the machine)."""
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                           "(Get-CimInstance Win32_Processor).LoadPercentage"], text=True, timeout=30)
            return int(out.split()[0])
        import os

        return round(100 * os.getloadavg()[0] / os.cpu_count())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def measure(device: str = "cpu", lengths=(1, 25, 50, 100, 200, 400, 800), repeats: int = 7,
            gen_lengths=(50, 100, 200), model_cfg: dict | None = None) -> dict:
    import torch

    ns = load_defs("ch05/05_kvcache.py")
    cfg = {**MODEL, **(model_cfg or {})}
    torch.manual_seed(0)
    model = ns["GPT"](**cfg).to(device).eval()
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    def timed(fn) -> float:
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        return time.perf_counter() - t0

    ids = torch.randint(0, cfg["vocab_size"], (1, max(lengths)), device=device)
    per_step = []
    with torch.no_grad():
        for _ in range(3):                                   # warm-up (allocator, kernels, thread pool)
            model(ids[:, :min(50, ids.shape[1])])
        for t in lengths:
            full = statistics.median(timed(lambda: model(ids[:, :t])) for _ in range(repeats))
            model.clear_cache()
            if t > 1:
                model(ids[:, :t - 1], use_cache=True)            # prefill the first t-1 tokens once
            saved = [(b.attn.k_cache, b.attn.v_cache, b.attn.cache_offset) for b in model.blocks]
            dts = []
            for _ in range(repeats):
                dts.append(timed(lambda: model(ids[:, t - 1:t], use_cache=True)))
                for blk, (k, v, off) in zip(model.blocks, saved):   # roll the cache back to t-1 tokens
                    blk.attn.k_cache, blk.attn.v_cache, blk.attn.cache_offset = k, v, off
            per_step.append({"t": t, "full_ms": round(1000 * full, 3),
                             "decode_ms": round(1000 * statistics.median(dts), 3)})

    start = torch.tensor([[42]], device=device)
    generate = []
    for n in gen_lengths:
        no_cache = timed(lambda: ns["generate_without_cache"](model, start, n))
        model.clear_cache()
        cache = timed(lambda: ns["generate_with_cache"](model, start, n))
        generate.append({"n": n, "no_cache_s": round(no_cache, 3), "cache_s": round(cache, 3)})
    model.clear_cache()
    return {"per_step": per_step, "generate": generate, "model": cfg}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU only)")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available")
    if args.threads:
        torch.set_num_threads(args.threads)
    load_before = cpu_load_percent() if args.device == "cpu" else None
    res = measure(args.device, repeats=args.repeats)
    report = {
        "device": args.device,
        "device_name": device_name(args.device),
        "threads": torch.get_num_threads() if args.device == "cpu" else None,
        "cpu_load_before_percent": load_before,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "date": date.today().isoformat(),
        "command": "uv run python verify/_ch05_kvbench.py " + " ".join(sys.argv[1:]),
        **res,
    }
    out = args.out or ROOT / "web" / "data" / "ch05" / f"kv_timing_{args.device}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
