"""Cost model for the GPU handbook (content/gpu.md).

One formula, shared by the page's verify cells, its calculator widget (web/widgets/gpu-costmodel.js,
checked against tests/js/fixtures/gpu/cases.json) and the pre-launch estimate in gpu/modal_app.py:

    training FLOPs per token = 6·n_mm + 6·L·T·d      (all matmul weights + causal attention scores)
    eval FLOPs per token     = 2·n_mm + 2·L·T·d      (forward only)
    wall seconds = train FLOPs / (world · peak · MFU) + eval FLOPs / (peak · MFU)   (evals run on rank 0 only)
    GPU hours    = world · wall hours,     USD = GPU hours · price per GPU-hour

Model shapes and hyperparameters come from web/data/gpu/bots.json, GPU peaks and prices from
web/data/gpu/gpus.json (the same files the page reads).

    uv run python gpu/costs.py --scenario webbot-x8-trap
    uv run python gpu/costs.py --bot webbot --offer modal:h100 --world 8 --max-iters 12500
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "web" / "data" / "gpu"


def load(data_dir: Path = DATA) -> tuple[dict, dict]:
    """(gpus.json, bots.json) as dicts."""
    gpus = json.loads((data_dir / "gpus.json").read_text(encoding="utf-8"))
    bots = json.loads((data_dir / "bots.json").read_text(encoding="utf-8"))
    return gpus, bots


def price_per_gpu_hour(offer: dict) -> float:
    """Modal publishes per-second prices; everyone else per hour."""
    if "usd_per_gpu_second" in offer:
        return offer["usd_per_gpu_second"] * 3600
    return offer["usd_per_gpu_hour"]


def param_counts(b: dict) -> tuple[int, int]:
    """(all parameters, matmul weights n_mm) from the architecture in bots.json.

    gelu  = GPT-2 style (CodeBot): LayerNorm with bias, FFN with bias, tied embedding, unembed bias.
    swiglu = modern style (StoryBot, WebBot): RMSNorm (weight only), no biases, untied unembed.
    """
    E, L, ff, V, T = b["d"], b["layers"], b["ff"], b["vocab"], b["ctx"]
    kv = b["kv_heads"] * (E // b["heads"])            # GQA: K and V project to kv_heads × head_dim
    attn = 2 * E * E + 2 * E * kv                      # W_q, W_o + W_k, W_v (no bias)
    gpt2 = b["ffn"] == "gelu"
    ffn = (2 if gpt2 else 3) * E * ff                  # GELU MLP: 2 matrices; SwiGLU: W, V, O
    block = attn + ffn + (ff + E + 4 * E if gpt2 else 2 * E)   # + FFN biases + 2 LayerNorms | 2 RMSNorms
    params = V * E + (T * E if b["learned_pos"] else 0) + L * block + (2 * E if gpt2 else E)
    params += 0 if b["tied"] else V * E                # unembed weight (shared with embed when tied)
    params += V if gpt2 else 0                         # unembed bias
    n_mm = L * (attn + ffn) + V * E                    # every token is multiplied by these, incl. the LM head
    return params, n_mm


def n_evals(max_iters: int, every: int, steps: int | None = None) -> int:
    """Evaluations in `for i in range(steps): if i % every == 0 or i == max_iters - 1`.

    steps < max_iters is the WebBot probe: the loop stops early, so the last-step evaluation never comes.
    """
    if not every:
        return 0
    steps = steps or max_iters
    last = 1 if max_iters - 1 < steps and (max_iters - 1) % every else 0
    return math.ceil(steps / every) + last


def eval_windows(valid_tokens: int, ctx: int) -> int:
    """Windows in one full validation pass: starts 0, ctx, 2·ctx, ... with start + ctx + 1 < len(data)."""
    return max(0, math.ceil((valid_tokens - ctx - 1) / ctx))


def run_shape(b: dict, world: int = 1, max_iters: int | None = None,
              batch: int | None = None, accum: int | None = None, steps: int | None = None) -> dict:
    """How many tokens a run touches. Every rank draws its own micro-batches (ch09 seeds 42 + rank).

    steps: stop after this many steps of a max_iters-step schedule (the WebBot probe).
    """
    it = max_iters or b["max_iters"]
    run = min(steps, it) if steps else it
    mb = batch or b["batch"]
    acc = accum or b["accum"]
    per_rank = mb * acc * b["ctx"]
    evals = n_evals(it, b["eval_every"], run)
    return {
        "world": world, "max_iters": it, "steps": run, "batch": mb, "accum": acc,
        "tokens_per_step_per_rank": per_rank,
        "tokens_per_step": per_rank * world,
        "train_tokens": per_rank * world * run,
        "evals": evals,
        "eval_tokens": eval_windows(b["valid_tokens"], b["ctx"]) * b["ctx"] * evals,
    }


def flops(b: dict, shape: dict) -> tuple[float, float]:
    """(training FLOPs, evaluation FLOPs) for a run of this shape."""
    attn = b["layers"] * b["ctx"] * b["d"]
    per_tok_train = 6 * b["n_mm"] + 6 * attn
    per_tok_eval = 2 * b["n_mm"] + 2 * attn
    return float(per_tok_train * shape["train_tokens"]), float(per_tok_eval * shape["eval_tokens"])


def peak_tflops(b: dict, gpu: dict) -> float | None:
    """Dense peak for the precision the bot trains in (CodeBot fp32, StoryBot/WebBot bf16)."""
    return gpu["bf16"] if b["precision"] == "bf16" else gpu["fp32"]


def estimate(b: dict, gpu: dict, usd_per_gpu_hour: float, shape: dict, mfu: float,
             floor_ms: float = 0.0) -> dict:
    """FLOPs-based plan. Evaluation runs on rank 0 while the other ranks wait (and are billed).

    floor_ms: the shortest a training step can take whatever the FLOPs (kernel launches, Python,
    get_batch). It only matters when a small model meets a fast GPU; 0 disables it.
    """
    peak = peak_tflops(b, gpu)
    if peak is None:
        raise ValueError(f"{gpu['name']} has no published {b['precision']} peak")
    eff = peak * 1e12 * mfu
    train_f, eval_f = flops(b, shape)
    per_tok = 6 * b["n_mm"] + 6 * b["layers"] * b["ctx"] * b["d"]
    step_s = shape["tokens_per_step_per_rank"] * per_tok / eff
    wall_s = shape["steps"] * max(step_s, floor_ms / 1e3) + eval_f / eff
    gpu_hours = wall_s * shape["world"] / 3600
    return {"train_flops": train_f, "eval_flops": eval_f, "step_ms": step_s * 1e3,
            "floor_bound": step_s * 1e3 < floor_ms, "wall_hours": wall_s / 3600,
            "gpu_hours": gpu_hours, "usd": gpu_hours * usd_per_gpu_hour}


def estimate_measured(steps: int, s_per_it: float, evals: int, eval_s: float, world: int,
                      usd_per_gpu_hour: float) -> dict:
    """Probe-based ETA: what tqdm measured beats any MFU guess."""
    wall_s = steps * s_per_it + evals * eval_s
    gpu_hours = wall_s * world / 3600
    return {"wall_hours": wall_s / 3600, "gpu_hours": gpu_hours, "usd": gpu_hours * usd_per_gpu_hour}


def find(items: list[dict], key: str) -> dict:
    for it in items:
        if it["id"] == key:
            return it
    raise KeyError(key)


# ---------------------------------------------------------------- whole jobs (gpu/jobs.py plans)
# Scripts other than pretraining are not FLOP-bound; these are planning ranges in minutes on one GPU.
# Sources: CodeBot SFT/GRPO from the local RTX 3060 Ti runs (GRPO is generation-bound: 256 sampled
# answers per step on the CPU side, so a bigger GPU barely helps); DPO = 1000 small steps; generation
# scripts produce a few hundred tokens. Replace them with your own report.json numbers once you have them.
SCRIPT_MINUTES = {
    "ch03/02_generate.py": (0.5, 3), "ch03/03_sft.py": (1, 6), "ch03/09_grpo.py": (90, 180),
    "ch06/06_generate.py": (0.5, 3), "ch06/08_dpo.py": (2, 15), "ch09/02_generate.py": (0.5, 3),
}
MFU_RANGE = {"codebot": (0.10, 0.30), "storybot": (0.10, 0.25), "webbot": (0.25, 0.45)}
FLOOR_MS = (8, 25)               # per-step floor of eager PyTorch on a fast GPU (assumption; probes replace it)
STARTUP_MINUTES = (3, 10)        # container start, image pull, copying data to local disk
COMPILE_MINUTES = {"webbot": (1, 4)}   # torch.compile warm-up in ch09


def estimate_plan(plan: dict, offer_id: str, nproc: int = 1, cpu_cores: float = 0.0, mem_gib: float = 0.0,
                  data_dir: Path = DATA) -> dict:
    """(low, high) hours and USD for a gpu/jobs.py plan on `nproc` GPUs of one offer.

    cpu_cores / mem_gib: what the container requests; billed on top of the GPU where the provider
    publishes per-second CPU and memory prices (Modal).
    """
    gpus, bots = load(data_dir)
    offer = find(gpus["offers"], offer_id)
    gpu = find(gpus["gpus"], offer["gpu"])
    prov = gpus["providers"][offer["provider"]]
    extra = 3600 * (cpu_cores * prov.get("usd_per_core_second", 0) + mem_gib * prov.get("usd_per_gib_second", 0))
    price = price_per_gpu_hour(offer)
    b = bots["bots"][plan["bot"]]
    o = plan.get("overrides", {})
    lines, lo, hi = [], STARTUP_MINUTES[0] / 60, STARTUP_MINUTES[1] / 60
    for script in plan["scripts"]:
        if script == b["script"]:
            shape = run_shape(b, nproc, o.get("max_iters"), o.get("micro_batch_size"),
                              o.get("accumulation_steps"), o.get("steps"))
            m_lo, m_hi = MFU_RANGE[plan["bot"]]
            c_lo, c_hi = COMPILE_MINUTES.get(plan["bot"], (0, 0))
            h_lo = estimate(b, gpu, price, shape, m_hi, FLOOR_MS[0])["wall_hours"] + c_lo / 60
            h_hi = estimate(b, gpu, price, shape, m_lo, FLOOR_MS[1])["wall_hours"] + c_hi / 60
            what = f"{shape['steps']:,} steps, {shape['train_tokens']:,} tokens, {shape['evals']} evals"
        else:
            m = SCRIPT_MINUTES[script]
            h_lo, h_hi, what = m[0] / 60, m[1] / 60, "planning range"
        lines.append({"script": script, "hours": (h_lo, h_hi), "what": what})
        lo, hi = lo + h_lo, hi + h_hi
    per_hour = nproc * price + extra
    return {"offer": offer, "gpu": gpu, "price": price, "extra_per_hour": extra, "nproc": nproc,
            "lines": lines, "hours": (lo, hi), "usd": (lo * per_hour, hi * per_hour),
            "checked": gpus["checked"], "src": prov["src"]}


def format_plan_estimate(e: dict) -> str:
    out = [f"{e['nproc']} × {e['offer']['label']} at ${e['price']:.4g}/GPU-h"
           + (f" + ${e['extra_per_hour']:.2f}/h CPU and RAM" if e["extra_per_hour"] else "")
           + f"  (prices checked {e['checked']}: {e['src']})"]
    for ln in e["lines"]:
        out.append(f"  {ln['script']:22} {ln['hours'][0]:6.2f}–{ln['hours'][1]:<6.2f} h  ({ln['what']})")
    out.append(f"  + start-up {STARTUP_MINUTES[0]}–{STARTUP_MINUTES[1]} min")
    out.append(f"  total {e['hours'][0]:.2f}–{e['hours'][1]:.2f} h wall  ->  ${e['usd'][0]:.2f}–${e['usd'][1]:.2f}")
    return "\n".join(out)


def describe(bot_id: str, offer_id: str, world: int = 1, max_iters: int | None = None,
             batch: int | None = None, accum: int | None = None, mfu: float = 0.35,
             floor_ms: float = 15.0, data_dir: Path = DATA) -> dict:
    """Everything the CLI and the Modal launcher print, as one dict."""
    gpus, bots = load(data_dir)
    b = bots["bots"][bot_id]
    offer = find(gpus["offers"], offer_id)
    gpu = find(gpus["gpus"], offer["gpu"])
    shape = run_shape(b, world, max_iters, batch, accum)
    price = price_per_gpu_hour(offer)
    est = estimate(b, gpu, price, shape, mfu, floor_ms)
    return {"bot": bot_id, "offer": offer, "gpu": gpu, "price": price, "mfu": mfu, "floor_ms": floor_ms,
            **shape, **est}


def format_estimate(d: dict) -> str:
    src = d["offer"].get("provider")
    return "\n".join([
        f"{d['bot']} on {d['world']} × {d['offer']['label']} (${d['price']:.2f}/GPU-h, MFU {d['mfu']:.0%})",
        f"  tokens  : {d['train_tokens']:,} train ({d['steps']:,} steps × {d['tokens_per_step']:,}/step)"
        f" + {d['eval_tokens']:,} eval ({d['evals']} passes, rank 0)",
        f"  FLOPs   : {d['train_flops']:.3e} train + {d['eval_flops']:.3e} eval",
        f"  time    : {d['wall_hours']:.2f} h wall, {d['gpu_hours']:.2f} GPU-h"
        f"  ({d['step_ms']:.1f} ms of compute per step" + (f", below the {d['floor_ms']:g} ms floor)" if d["floor_bound"] else ")"),
        f"  cost    : ${d['usd']:.2f}  (prices checked 2026-10-11, provider: {src})",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate DLFS6 training time and cost.")
    ap.add_argument("--scenario", help="preset id from web/data/gpu/bots.json, e.g. webbot-x8-trap")
    ap.add_argument("--bot", choices=["codebot", "storybot", "webbot"])
    ap.add_argument("--offer", help="offer id from web/data/gpu/gpus.json, e.g. modal:h100")
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--max-iters", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--accum", type=int)
    ap.add_argument("--mfu", type=float)
    ap.add_argument("--floor-ms", type=float, default=15.0, help="shortest possible step (launch overhead)")
    args = ap.parse_args()
    if args.scenario:
        _, bots = load()
        s = find(bots["scenarios"], args.scenario)
        kw = dict(bot_id=s["bot"], offer_id=args.offer or s["offer"], world=s["world"],
                  max_iters=s.get("max_iters"), batch=s.get("batch"), accum=s.get("accum"),
                  mfu=args.mfu or s["mfu"], floor_ms=args.floor_ms)
    else:
        if not (args.bot and args.offer):
            ap.error("give --scenario, or --bot and --offer")
        kw = dict(bot_id=args.bot, offer_id=args.offer, world=args.world, max_iters=args.max_iters,
                  batch=args.batch, accum=args.accum, mfu=args.mfu or 0.35, floor_ms=args.floor_ms)
    print(format_estimate(describe(**kw)))


if __name__ == "__main__":
    main()
