"""Export a DLFS6 CodeBot checkpoint for the in-browser workbench (content/playground.md).

The workbench's default weights are the tutorial's own run of the book's ch03 scripts (gpu/run_local.py; the
manifest copies the producing step from web/data/runs/<run>/report.json into provenance.run_step):

    uv run python tools/export_codebot.py --ckpt runs/codebot-3060ti/work/codebot/model_sft.pt --name sft \
        --label SFT --template alpaca --order 1 --source "gpu/run_local.py --run codebot-3060ti" --description "..."

The book's official checkpoints (Hugging Face, fetched by tools/fetch_data.py) go through the same command;
a new --name adds an entry to the workbench's checkpoint switch:

    uv run python tools/export_codebot.py --ckpt third_party/dlfs6/codebot/model_sft.pt --name official-sft \
        --label "SFT（官方）" --template alpaca --order 11 \
        --source https://huggingface.co/datasets/koki0702/zero-llm-data/blob/main/codebot/model_sft.pt

Writes web/data/playground/<name>/
  manifest.json   format "dlfs6-codebot/1": config, provenance, packing, tensor table, shard list
  weights-NN.bin  the packed weights cut into shards of at most --shard-bytes (concatenate them in order)
  ref.json        PyTorch reference for the parity tests: last-position logits, greedy tokens, logit lens,
                  attention weights; once with the checkpoint's float32 weights ("f32") and once with exactly
                  the weights the browser reconstructs from the shards ("q8")
and adds (or replaces) the entry <name> in web/data/playground/index.json.

Packing: each 2-D weight is stored as int8, one scale per output row (symmetric: scale = max|w| / 127,
q = round(w / scale)), followed by its float32 row scales; each 1-D parameter (biases, LayerNorm) stays float32.
unembed.weight is not stored, because CodeBot ties it to embed.weight; a checkpoint where the two differ is refused.
The reference is computed with the book's own model class (third_party/dlfs6/codebot/model.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
OUT = ROOT / "web" / "data" / "playground"
FORMAT, REF_FORMAT = "dlfs6-codebot/1", "dlfs6-codebot-ref/1"
CONFIG_KEYS = ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim")
END_TOKEN = "<|endoftext|>"


def alpaca(instruction: str) -> str:
    """The prompt template of ch03/03_sft.py, 04_chat.py and 09_grpo.py."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# The prompts every export is checked on (the same for every checkpoint, so they can be compared).
PROMPTS = [
    {"name": "code", "text": "def add(a, b):\n", "new_tokens": 32, "attention": True},
    {"name": "chat", "text": alpaca("print hello"), "new_tokens": 32},
    {"name": "math", "text": alpaca("3+4="), "new_tokens": 32},
    {"name": "long", "corpus_tokens": 250, "new_tokens": 16},   # crosses the 256-token window while generating
]


# ----------------------------------------------------------------------------- checkpoint
def expected_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    V, C, E, L, F = (cfg[k] for k in ("vocab_size", "max_context_len", "embed_dim", "n_layer", "ff_dim"))
    shapes = {"embed.weight": (V, E), "pos_embed.weight": (C, E)}
    for l in range(L):
        p = f"blocks.{l}."
        shapes |= {p + "norm1.weight": (E,), p + "norm1.bias": (E,)}
        shapes |= {p + f"attn.W_{m}.weight": (E, E) for m in "qkvo"}
        shapes |= {p + "norm2.weight": (E,), p + "norm2.bias": (E,),
                   p + "ffn.layers.0.weight": (F, E), p + "ffn.layers.0.bias": (F,),
                   p + "ffn.layers.2.weight": (E, F), p + "ffn.layers.2.bias": (E,)}
    shapes |= {"norm.weight": (E,), "norm.bias": (E,), "unembed.weight": (V, E), "unembed.bias": (V,)}
    return shapes


def load_checkpoint(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Read a checkpoint written by codebot.model.GPT.save() and check it is the architecture the JS runs."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in (*CONFIG_KEYS, "model_state_dict") if k not in ckpt]
    if missing:
        raise SystemExit(f"{path}: not a codebot.model.GPT checkpoint (missing {missing})")
    cfg = {k: int(ckpt[k]) for k in CONFIG_KEYS}
    if cfg["embed_dim"] % cfg["n_head"]:
        raise SystemExit(f"{path}: embed_dim {cfg['embed_dim']} is not divisible by n_head {cfg['n_head']}")
    if cfg["embed_dim"] % 16 or cfg["ff_dim"] % 16:
        raise SystemExit(f"{path}: the browser's SIMD kernel needs embed_dim and ff_dim divisible by 16")
    if cfg["vocab_size"] > 65536:
        raise SystemExit(f"{path}: vocab_size {cfg['vocab_size']} does not fit the workbench's token ids")
    sd = {k: v.detach().float().contiguous() for k, v in ckpt["model_state_dict"].items()}
    want = expected_shapes(cfg)
    if set(sd) != set(want):
        extra, lack = sorted(set(sd) - set(want)), sorted(set(want) - set(sd))
        raise SystemExit(f"{path}: unexpected parameters (extra {extra[:5]}, missing {lack[:5]})")
    for k, shape in want.items():
        if tuple(sd[k].shape) != shape:
            raise SystemExit(f"{path}: {k} has shape {tuple(sd[k].shape)}, expected {shape}")
    if not torch.equal(sd["embed.weight"], sd["unembed.weight"]):
        raise SystemExit(f"{path}: unembed.weight differs from embed.weight; the workbench assumes tied weights")
    if not all(torch.isfinite(v).all() for v in sd.values()):
        raise SystemExit(f"{path}: the checkpoint contains inf or NaN")
    return cfg, sd


# ----------------------------------------------------------------------------- packing
def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per output row: w ≈ q * scale[:, None], |q| ≤ 127, error ≤ scale / 2."""
    scale = (np.abs(w).max(axis=1) / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0                       # an all-zero row: any scale reconstructs it exactly
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantize_rows(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """What the browser computes: float32(q) * float32(scale), rounded once to float32."""
    return q.astype(np.float32) * scale[:, None]


def pack(sd: dict[str, torch.Tensor]) -> tuple[bytes, list[dict], dict[str, torch.Tensor]]:
    """Pack every tensor but unembed.weight; return the blob, the tensor table and the reconstructed weights."""
    chunks, table, recon, off = [], [], {}, 0
    for name, t in sd.items():
        if name == "unembed.weight":
            continue
        a = t.numpy().astype(np.float32)
        entry = {"name": name, "shape": list(a.shape), "offset": off}
        if a.ndim == 2:
            q, scale = quantize_rows(a)
            entry |= {"dtype": "q8", "scales_offset": off + q.nbytes}
            data = q.tobytes() + scale.astype("<f4").tobytes()
            recon[name] = torch.from_numpy(dequantize_rows(q, scale))
        else:
            entry["dtype"] = "f32"
            data = a.astype("<f4").tobytes()
            recon[name] = torch.from_numpy(a.copy())
        entry["nbytes"] = len(data)
        pad = (-len(data)) % 4                    # keep every tensor 4-byte aligned for typed-array views
        chunks.append(data + b"\0" * pad)
        table.append(entry)
        off += len(data) + pad
    recon["unembed.weight"] = recon["embed.weight"]
    return b"".join(chunks), table, recon


def write_shards(blob: bytes, out: Path, shard_bytes: int) -> list[dict]:
    for old in out.glob("weights-*.bin"):
        old.unlink()
    shards = []
    for i, start in enumerate(range(0, len(blob), shard_bytes)):
        part = blob[start:start + shard_bytes]
        name = f"weights-{i:02d}.bin"
        (out / name).write_bytes(part)
        shards.append({"file": name, "bytes": len(part), "sha256": hashlib.sha256(part).hexdigest()})
    return shards


# ----------------------------------------------------------------------------- reference (PyTorch, the book's model)
def dlfs6_modules():
    if str(DLFS6) not in sys.path:
        sys.path.insert(0, str(DLFS6))
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer
    return GPT, BPETokenizer


def build_model(cfg: dict, sd: dict[str, torch.Tensor]):
    GPT, _ = dlfs6_modules()
    model = GPT(**cfg, dropout_rate=0.0)
    model.load_state_dict(sd)                     # embed.weight and unembed.weight are one tensor: both keys load it
    return model.eval()


def corpus_prompt(tok, n_tokens: int) -> tuple[str, list[int]]:
    """The shortest prefix of TinyCodes, cut between pre-tokens, that encodes to at least n_tokens tokens."""
    from codebot.tokenizer import pretokenize
    text = (DLFS6 / "codebot" / "tiny_codes.txt").read_text(encoding="utf-8")[:20_000]
    pieces, ids = [], []
    for part in text.split(END_TOKEN):
        for piece in ([END_TOKEN] if pieces else []) + pretokenize(part):
            pieces.append(piece)
            ids.extend([tok.end_token_id] if piece == END_TOKEN else tok._encode_text(piece))
            if len(ids) >= n_tokens:
                prompt = "".join(pieces)
                if tok.encode(prompt) != ids:
                    raise RuntimeError("the long prompt does not re-encode to the same ids")
                return prompt, ids
    raise RuntimeError("tiny_codes.txt is too short for the long prompt")


@torch.no_grad()
def run_prompt(model, ids: list[int], new_tokens: int, max_len: int, attention: bool) -> dict:
    """Last-position logits, logit lens and (optionally) attention for the prompt; then greedy decoding with the
    book's semantics (codebot.utils.generate at temperature 0: crop to the last max_len tokens before every forward),
    except that it does not stop at <|endoftext|>, so that every export has the same number of tokens to compare."""
    captured, hooks = {}, []
    if attention:
        for l, block in enumerate(model.blocks):   # the input of attention_dropout is the softmax weights (B, H, C, C)
            hooks.append(block.attn.attention_dropout.register_forward_hook(
                lambda m, inp, out, l=l: captured.__setitem__(("attn", l), inp[0][0].clone())))
    hooks.append(model.dropout.register_forward_hook(          # input = token embedding + position embedding
        lambda m, inp, out: captured.__setitem__(("x", 0), inp[0][0, -1].clone())))
    for l, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(
            lambda m, inp, out, l=l: captured.__setitem__(("x", l + 1), out[0, -1].clone())))
    try:
        logits = model(torch.tensor([ids]))[0, -1]
    finally:
        for hk in hooks:
            hk.remove()
    lens = []
    for l in range(len(model.blocks) + 1):
        p = torch.softmax(model.unembed(model.norm(captured[("x", l)])), dim=-1)
        top = torch.topk(p, 5)
        lens.append({"layer": l, "ids": top.indices.tolist(), "probs": [round(float(v), 6) for v in top.values]})
    seq, greedy = list(ids), []
    for _ in range(new_tokens):
        window = seq[-max_len:]
        nxt = int(model(torch.tensor([window]))[0, -1].argmax())
        greedy.append(nxt)
        seq.append(nxt)
    out = {"logits": [round(float(v), 6) for v in logits], "greedy": greedy, "lens": lens}
    if attention:
        out["attention"] = [[[[round(float(v), 6) for v in row] for row in head] for head in captured[("attn", l)]]
                            for l in range(len(model.blocks))]
    return out


def reference(cfg: dict, sd: dict, recon: dict, threads: int) -> dict:
    _, BPETokenizer = dlfs6_modules()
    torch.set_num_threads(threads)
    tok = BPETokenizer.load_from(DLFS6 / "codebot" / "merge_rules.pkl")
    if tok.vocab_size != cfg["vocab_size"]:
        raise SystemExit(f"tokenizer vocab {tok.vocab_size} != model vocab {cfg['vocab_size']}")
    models = {"f32": build_model(cfg, sd), "q8": build_model(cfg, recon)}
    prompts = []
    for spec in PROMPTS:
        if "corpus_tokens" in spec:
            text, ids = corpus_prompt(tok, spec["corpus_tokens"])
        else:
            text, ids = spec["text"], tok.encode(spec["text"])
        rec = {"name": spec["name"], "text": text, "ids": ids, "new_tokens": spec["new_tokens"]}
        for variant, model in models.items():
            rec[variant] = run_prompt(model, ids, spec["new_tokens"], cfg["max_context_len"], spec.get("attention", False))
        prompts.append(rec)
    return {"format": REF_FORMAT, "end_token_id": tok.end_token_id, "max_context_len": cfg["max_context_len"],
            "prompts": prompts}


# ----------------------------------------------------------------------------- provenance & index
def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name                           # outside the repo: do not leak a local absolute path


def run_step(run: str, ckpt_rel: str) -> dict | None:
    """The step of web/data/runs/<run>/report.json (gpu/run_local.py) whose script wrote this checkpoint."""
    path = ROOT / "web/data/runs" / run / "report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    inner = ckpt_rel.split("/work/", 1)[-1]          # runs/<run>/work/codebot/model_sft.pt -> codebot/model_sft.pt
    found = None
    for step in report.get("steps", []):             # a re-run writes the file again: the last step wins
        if inner in step.get("produced", []):
            found = step
    if found is None:
        return None
    host = report.get("host", {})
    return {"report": f"data/runs/{run}/report.json", "script": found["script"], "started": found.get("started"),
            "seconds": found.get("seconds"), "exit_code": found.get("exit_code"), "final": found.get("final"),
            "gpu": host.get("gpu"), "torch": host.get("torch"), "dlfs6_commit": report.get("dlfs6_commit")}


def provenance(ckpt: Path, source: str | None) -> dict:
    raw = ckpt.read_bytes()
    where = rel(ckpt)
    parts = Path(where).parts
    run = parts[1] if len(parts) > 2 and parts[0] == "runs" else None   # runs/<run>/work/... (gpu/run_local.py)
    step = run_step(run, where) if run else None
    return {
        "checkpoint": where, "checkpoint_bytes": len(raw), "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "source": source, "run": run, "run_report": step["report"] if step else None, "run_step": step,
        "dlfs6_commit": git_rev(DLFS6), "tutorial_commit": git_rev(ROOT),
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "exporter": "tools/export_codebot.py", "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
    }


def update_index(out_root: Path, entry: dict) -> None:
    path = out_root / "index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"checkpoints": []}
    rest = [c for c in index["checkpoints"] if c["name"] != entry["name"]]
    index["checkpoints"] = sorted(rest + [entry], key=lambda c: (c.get("order", 99), c["name"]))
    path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True, help="a checkpoint saved by codebot.model.GPT.save()")
    ap.add_argument("--name", required=True, help="folder name under web/data/playground (letters, digits, -)")
    ap.add_argument("--label", help="short label for the workbench's checkpoint switch (default: --name)")
    ap.add_argument("--description", default="", help="one sentence shown under the switch")
    ap.add_argument("--template", choices=["none", "alpaca"], default="none",
                    help="whether the workbench wraps prompts in the ch03 Alpaca template by default")
    ap.add_argument("--source", help="where the checkpoint came from (URL or note), recorded in the manifest")
    ap.add_argument("--order", type=int, default=50, help="position in the checkpoint switch")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--shard-bytes", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads for the reference")
    args = ap.parse_args()
    if not args.name.replace("-", "").isalnum():
        raise SystemExit("--name may only contain letters, digits and '-'")
    if not args.ckpt.is_file():
        raise SystemExit(f"{args.ckpt} not found (official checkpoints: uv run python tools/fetch_data.py)")

    cfg, sd = load_checkpoint(args.ckpt)
    blob, table, recon = pack(sd)
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    shards = write_shards(blob, out, args.shard_bytes)
    manifest = {
        "format": FORMAT, "name": args.name, "label": args.label or args.name, "config": cfg,
        "tied_unembed": True, "end_token_id": cfg["vocab_size"] - 1,
        "packing": "2-D weights: int8 per output row, symmetric, scale = max|w|/127, then float32 row scales; "
                   "1-D parameters: float32; unembed.weight = embed.weight (tied, not stored)",
        "provenance": provenance(args.ckpt, args.source),
        "total_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
        "shards": shards, "tensors": table,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                       newline="\n")
    ref = reference(cfg, sd, recon, args.threads)
    ref["name"] = args.name
    (out / "ref.json").write_text(json.dumps(ref, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8",
                                  newline="\n")
    update_index(args.out, {
        "name": args.name, "label": args.label or args.name, "description": args.description,
        "template": None if args.template == "none" else args.template, "order": args.order,
        "manifest": f"{args.name}/manifest.json", "ref": f"{args.name}/ref.json", "bytes": len(blob),
    })
    params = sum(int(np.prod(t["shape"])) for t in table)
    q8 = {p["name"]: max(abs(a - b) for a, b in zip(p["f32"]["logits"], p["q8"]["logits"])) for p in ref["prompts"]}
    print(f"{args.name}: {params:,} parameters -> {len(blob):,} bytes in {len(shards)} shards "
          f"(float32 would be {params * 4:,})")
    print("  max |logit(q8) - logit(f32)| per prompt:", {k: round(v, 4) for k, v in q8.items()})
    for p in ref["prompts"]:
        same = sum(a == b for a, b in zip(p["f32"]["greedy"], p["q8"]["greedy"]))
        print(f"  {p['name']:5} {len(p['ids']):3} prompt tokens; greedy q8 == f32 on {same}/{len(p['f32']['greedy'])}")


if __name__ == "__main__":
    main()
