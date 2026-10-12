"""第 6 章的记录：我们在 H100 上原样运行 05_pretrain.py 与 08_dpo.py（运行 storybot-h100）得到的两个 checkpoint，
在 CPU 上比较 DPO 前后。DPO 到底改变了什么，这里不靠训练 loss，而是直接看模型。

  * 两个 checkpoint：runs/storybot-h100/work/storybot/model_pretrain.pt 与 model_dpo.pt（本地文件，不入库；
    记录里写下它们的 sha256。这次运行的出处是 web/data/runs/storybot-h100/report.json）。
  * 故事：两个模型用同样的开头、同样的种子各写一遍（原书的 storybot.utils.generate，温度 1.0）。开头有两组：
      - 8 条留出 pair 的 prompt（web/data/ch06/dpo_heldout_pairs.json），与 DPO 训练的格式相同：给故事开头，写结局；
      - 09_llm_judge.py 用的 "Once upon a time"。
    max_new_tokens = 256 − prompt 的 token 数：整个故事都在训练时的上下文长度之内（超过 256 以后 RoPE 的位置被钳住，
    见 6.6.1 节）。每个故事都记下新写了多少个 token、是写到 <|endoftext|> 自然结束，还是写满了上限。
  * 偏好对：8 条留出 pair 与 144 条训练 pair 上，两个模型对 chosen、rejected 的对数概率之和（原书 08_dpo.py 的
    DPODataset 与 get_sequence_logprobs，context_len 与原书相同，为 256），由此得到隐式奖励与“按对数概率之和偏好哪一个”。
  * 语言建模：验证文件前 65,536 个 token 上的 loss（与 dpo_cpu.json 同一口径）。
  * 参数移动了多少：|θ_dpo − θ_pretrain| 的最大值与平均值。

用法（CPU，几分钟）：
  CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_h100_dpo.py --out web/data/ch06/h100_dpo.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch06_common import (ROOT, STORYBOT, ensure_dlfs6_on_path, heldout_tokens, lm_loss,  # noqa: E402
                          load_defs, load_storybot)

RUN_ID = "storybot-h100"
RUN_DIR = ROOT / "runs" / RUN_ID / "work" / "storybot"
CHECKPOINTS = {"pretrain": RUN_DIR / "model_pretrain.pt", "dpo": RUN_DIR / "model_dpo.pt"}
HELDOUT_PAIRS = ROOT / "web" / "data" / "ch06" / "dpo_heldout_pairs.json"
CONTEXT = 256              # StoryBot 的 max_context_len，也是 08_dpo.py 的 context_len
BETA = 0.1                 # 08_dpo.py 的 beta
HELDOUT_SEEDS = (1, 2, 3)  # 每条留出 prompt 用 3 个种子
OPEN_PROMPT = "Once upon a time"
OPEN_SEEDS = tuple(range(1, 9))
LM_TOKENS = 65536


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RecordingTokenizer:
    """Delegates to the book's tokenizer and remembers the ids that generate() decodes at the end.

    generate() appends a sampled token only if it is not <|endoftext|>, so the number of new tokens tells
    whether the story stopped by itself (fewer than max_new_tokens) or ran into the limit.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.end_token_id = tokenizer.end_token_id
        self.decoded_ids: list[int] = []

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        self.decoded_ids = list(ids)
        return self.tokenizer.decode(ids)


def write_story(model, tokenizer, prompt: str, seed: int) -> dict:
    """One story with the book's generate(); the same seed gives both models the same random numbers."""
    from storybot.utils import generate

    n_prompt = len(tokenizer.encode(prompt))
    max_new = CONTEXT - n_prompt
    rec = RecordingTokenizer(tokenizer)
    torch.manual_seed(seed)
    text = generate(model, rec, prompt, max_new, 1.0)
    if not text.startswith(prompt):
        raise ValueError(f"decoded text does not start with the prompt {prompt!r}")
    n_new = len(rec.decoded_ids) - n_prompt
    return {"text": text[len(prompt):], "new_tokens": n_new, "ended": n_new < max_new, "max_new_tokens": max_new}


def pair_logprobs(models: dict, tokenizer, pairs: list[dict], context_len: int = CONTEXT) -> dict:
    """Sum of log-probs of the chosen and rejected answers under each model (the book's 08_dpo.py functions)."""
    ns = load_defs("ch06/08_dpo.py", ["DPODataset", "get_sequence_logprobs"])
    ds = ns["DPODataset"].__new__(ns["DPODataset"])   # the book's _create_sample on in-memory pairs
    ds.tokenizer, ds.context_len = tokenizer, context_len
    ds.samples = [ds._create_sample(p["prompt"], p["chosen"], p["rejected"]) for p in pairs]
    c, cm, r, rm = [torch.stack(t) for t in zip(*[ds[i] for i in range(len(ds))])]
    lp = ns["get_sequence_logprobs"]
    out = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            chosen, rejected = [], []
            for i in range(0, len(c), 16):
                chosen.append(lp(model, c[i:i + 16], cm[i:i + 16]))
                rejected.append(lp(model, r[i:i + 16], rm[i:i + 16]))
            out[name] = {"chosen": torch.cat(chosen).tolist(), "rejected": torch.cat(rejected).tolist()}
    out["chosen_tokens"] = cm[:, 1:].sum(1).tolist()
    out["rejected_tokens"] = rm[:, 1:].sum(1).tolist()
    return out


def pair_summary(lp: dict, beta: float = BETA) -> dict:
    pc, pr = np.array(lp["dpo"]["chosen"]), np.array(lp["dpo"]["rejected"])
    qc, qr = np.array(lp["pretrain"]["chosen"]), np.array(lp["pretrain"]["rejected"])
    margin = beta * ((pc - qc) - (pr - qr))
    return {
        "n": len(pc),
        "reward_acc": float((margin > 0).mean()),
        "margin_mean": float(margin.mean()),
        "dpo_loss": float(np.logaddexp(0.0, -margin).mean()),
        "chosen_shift_nats": float((pc - qc).mean()),
        "rejected_shift_nats": float((pr - qr).mean()),
        "pretrain_prefers_chosen_by_sum": float((qc > qr).mean()),
        "dpo_prefers_chosen_by_sum": float((pc > pr).mean()),
    }


def parameter_shift(a: torch.nn.Module, b: torch.nn.Module) -> dict:
    sa, sb = a.state_dict(), b.state_dict()
    diffs = torch.cat([(sb[k].float() - sa[k].float()).abs().flatten()
                       for k in sa if not k.endswith(("cos_cache", "sin_cache"))])
    return {"n_params": diffs.numel(), "max_abs": diffs.max().item(), "mean_abs": diffs.mean().item(),
            "frac_above_0.005": (diffs > 0.005).float().mean().item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data" / "ch06" / "h100_dpo.json")
    args = ap.parse_args()
    missing = [str(p) for p in CHECKPOINTS.values() if not p.is_file()]
    if missing:
        sys.exit(f"missing {missing}: copy the run back first (gpu/modal_app.py pull --run {RUN_ID})")
    torch.set_num_threads(args.threads)
    t0 = time.time()
    ensure_dlfs6_on_path()
    from storybot.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.load_from(str(STORYBOT / "merge_rules.pkl"))
    models = {name: load_storybot(path) for name, path in CHECKPOINTS.items()}
    heldout = json.loads(HELDOUT_PAIRS.read_text(encoding="utf-8"))
    train = json.loads((STORYBOT / "tiny_stories_dpo.json").read_text(encoding="utf-8"))

    stories = []
    for prompt, seeds, kind in [(p["prompt"], HELDOUT_SEEDS, "heldout") for p in heldout] + [
            (OPEN_PROMPT, OPEN_SEEDS, "open")]:
        for seed in seeds:
            row = {"kind": kind, "prompt": prompt, "seed": seed}
            for name, model in models.items():
                row[name] = write_story(model, tokenizer, prompt, seed)
            stories.append(row)
            print(f"[{kind} seed {seed}] {prompt[:40]!r}: pretrain {row['pretrain']['new_tokens']} tok "
                  f"(ended {row['pretrain']['ended']}), dpo {row['dpo']['new_tokens']} tok (ended {row['dpo']['ended']})",
                  flush=True)

    held_lp = pair_logprobs(models, tokenizer, heldout)
    train_lp = pair_logprobs(models, tokenizer, train)
    tokens = heldout_tokens(LM_TOKENS)
    report = {
        "what": "the storybot-h100 run's own checkpoints before and after 08_dpo.py, compared on CPU",
        "command": "CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch06_h100_dpo.py",
        "source_run": {"run": RUN_ID, "report": f"web/data/runs/{RUN_ID}/report.json",
                       "checkpoints": {name: {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(p)}
                                       for name, p in CHECKPOINTS.items()}},
        "sampling": {"generate": "storybot.utils.generate (unchanged)", "temperature": 1.0,
                     "max_new_tokens": f"{CONTEXT} - len(prompt tokens)", "heldout_seeds": list(HELDOUT_SEEDS),
                     "open_prompt": OPEN_PROMPT, "open_seeds": list(OPEN_SEEDS), "threads": args.threads},
        "pairs": {"beta": BETA, "context_len": CONTEXT,
                  "heldout": {**pair_summary(held_lp), "logprobs": held_lp},
                  "train": pair_summary(train_lp)},
        "lm_loss": {"tokens": LM_TOKENS, "pretrain": lm_loss(models["pretrain"], tokens),
                    "dpo": lm_loss(models["dpo"], tokens)},
        "param_shift": parameter_shift(models["pretrain"], models["dpo"]),
        "stories": stories,
        "provenance": {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "platform": platform.platform(), "processor": platform.processor(),
                       "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                       "threads": args.threads, "seconds": round(time.time() - t0, 1)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {args.out} in {report['provenance']['seconds']} s")


if __name__ == "__main__":
    main()
