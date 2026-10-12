"""Greedy single-digit addition for the workbench (content/playground.md, P.8).

The tutorial's own SFT and GRPO checkpoints (runs/codebot-3060ti, trained by running the book's ch03 scripts
unchanged) are asked every problem with the Alpaca template of ch03, once in float32 and once with exactly the
int8 weights the browser runs (rebuilt here from the shards in web/data/playground/<name>/). The book's official
checkpoints are asked too, for comparison with ch03 3.4.2.

    uv run python verify/_playground_addition.py        # rewrites web/data/playground/addition.json

Decoding follows codebot.utils.generate at temperature 0 (crop to the last max_context_len tokens, argmax, stop
at <|endoftext|> without appending it) with at most MAX_NEW new tokens. Every recorded answer stops at
<|endoftext|> well before that cap (the tests check it), so the result equals the book's default of 1000.
Correctness is 09_grpo.py's own calculate_reward, extracted from the script, applied to the response (what the
training rewards see).
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DLFS6 = ROOT / "third_party" / "dlfs6"
EXPORTS = ROOT / "web" / "data" / "playground"
OUT = EXPORTS / "addition.json"
RUN = ROOT / "runs" / "codebot-3060ti" / "work" / "codebot"
MAX_NEW = 64
FORMAT = "dlfs6-codebot-addition/1"

SETS = {
    "train": [(a, b) for a in range(1, 10) for b in range(1, 10)],      # GRPODataset: the 81 prompts of 09_grpo.py
    "zero": [(0, b) for b in range(10)] + [(a, 0) for a in range(1, 10)],
    "teens": [(a, b) for a in range(10, 19) for b in range(1, 10)],
}
# name -> (checkpoint path, int8 export folder or None, label)
MODELS = {
    "ours-sft-f32": (RUN / "model_sft.pt", None, "SFT（本书训练，float32）"),
    "ours-sft-q8": (RUN / "model_sft.pt", "sft", "SFT（本书训练，浏览器里的 int8）"),
    "ours-grpo-f32": (RUN / "model_grpo.pt", None, "GRPO（本书训练，float32）"),
    "ours-grpo-q8": (RUN / "model_grpo.pt", "grpo", "GRPO（本书训练，浏览器里的 int8）"),
    "official-sft-f32": (DLFS6 / "codebot" / "model_sft.pt", None, "SFT（官方，float32）"),
    "official-grpo-f32": (DLFS6 / "codebot" / "model_grpo.pt", None, "GRPO（官方，float32）"),
}


def alpaca(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def book_reward():
    """calculate_reward exactly as written in ch03/09_grpo.py (the script itself trains on import)."""
    src = (DLFS6 / "ch03" / "09_grpo.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "calculate_reward")
    ns: dict = {}
    exec(compile("import re\n" + ast.get_source_segment(src, fn), "09_grpo.py", "exec"), ns)
    return ns["calculate_reward"]


def modules():
    if str(DLFS6) not in sys.path:
        sys.path.insert(0, str(DLFS6))
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer
    return GPT, BPETokenizer


def load_export(name: str) -> dict[str, torch.Tensor]:
    """The float32 weights the browser reconstructs from web/data/playground/<name>/ (q * row scale)."""
    folder = EXPORTS / name
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    blob = b"".join((folder / s["file"]).read_bytes() for s in manifest["shards"])
    if hashlib.sha256(blob).hexdigest() != manifest["sha256"]:
        raise ValueError(f"{name}: shards do not match the manifest's sha256")
    sd = {}
    for t in manifest["tensors"]:
        shape = tuple(t["shape"])
        if t["dtype"] == "q8":
            q = np.frombuffer(blob, np.int8, int(np.prod(shape)), t["offset"]).reshape(shape)
            scale = np.frombuffer(blob, "<f4", shape[0], t["scales_offset"])
            w = q.astype(np.float32) * scale[:, None]
        else:
            w = np.frombuffer(blob, "<f4", int(np.prod(shape)), t["offset"]).reshape(shape)
        sd[t["name"]] = torch.from_numpy(w.copy())
    sd["unembed.weight"] = sd["embed.weight"]
    return sd


def load_model(path: Path, export: str | None):
    GPT, _ = modules()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = {k: int(ckpt[k]) for k in ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim")}
    model = GPT(**cfg, dropout_rate=0.0)
    model.load_state_dict(load_export(export) if export else ckpt["model_state_dict"])
    return model.eval()


@torch.no_grad()
def greedy(model, ids: list[int], end_id: int, max_new: int = MAX_NEW) -> tuple[list[int], bool]:
    """codebot.utils.generate at temperature 0, returning the new ids and whether <|endoftext|> stopped it."""
    seq, out = list(ids), []
    for _ in range(max_new):
        nxt = int(model(torch.tensor([seq[-model.max_context_len:]]))[0, -1].argmax())
        if nxt == end_id:
            return out, True
        out.append(nxt)
        seq.append(nxt)
    return out, False


def evaluate(model, tok, reward, keep_ids: bool) -> dict:
    results = {}
    for set_name, problems in SETS.items():
        rec = {"responses": [], "correct": [], "eos": []}
        if keep_ids:
            rec["ids"] = []
        for a, b in problems:
            new, eos = greedy(model, tok.encode(alpaca(f"{a}+{b}=")), tok.end_token_id)
            text = tok.decode(new)
            rec["responses"].append(text)
            rec["correct"].append(int(reward(a + b, text) > 0))
            rec["eos"].append(eos)
            if keep_ids:
                rec["ids"].append(new)
        rec["n_correct"] = sum(rec["correct"])
        results[set_name] = rec
    return results


def build(threads: int = 4) -> dict:
    torch.set_num_threads(threads)
    _, BPETokenizer = modules()
    tok = BPETokenizer.load_from(DLFS6 / "codebot" / "merge_rules.pkl")
    reward = book_reward()
    out = {
        "format": FORMAT, "max_new_tokens": MAX_NEW, "template": alpaca("{a}+{b}="),
        "sets": {k: [f"{a}+{b}=" for a, b in v] for k, v in SETS.items()},
        "answers": {k: [a + b for a, b in v] for k, v in SETS.items()},
        "models": {},
    }
    for name, (path, export, label) in MODELS.items():
        model = load_model(path, export)
        out["models"][name] = {
            "label": label, "checkpoint": path.relative_to(ROOT).as_posix(), "export": export,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "results": evaluate(model, tok, reward, keep_ids=export is not None),
        }
    return out


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    for name, m in data["models"].items():
        counts = "  ".join(f"{k} {r['n_correct']}/{len(r['correct'])}" for k, r in m["results"].items())
        print(f"{name:18} {counts}")


if __name__ == "__main__":
    main()
