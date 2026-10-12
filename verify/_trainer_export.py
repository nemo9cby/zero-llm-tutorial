"""Export PyTorch reference values for the in-browser trainer (web/widgets/_tinygpt.js).

The reference model is DLFS6's own `codebot.model.GPT`, built at a tiny size, with dropout 0 and in
float64, so that the JavaScript implementation can be compared with it to ~1e-12:

* parity-tiny.json: weights, a batch, the loss, every parameter's gradient, and three
  torch.optim.AdamW steps (losses and the parameters afterwards).
* parity-default.json: the trainer's real size (V=96, ctx 64, embed 64, 4 heads, 2 layers, ff 192)
  with weights from a formula that JavaScript can reproduce; stores the loss and per-parameter
  gradient checksums (the full gradients would be 750 KB).
* erf.json: reference values for the JavaScript port of erf.

    uv run python verify/_trainer_export.py      # rewrite tests/js/fixtures/trainer/*.json

Arrays are stored as base64 of little-endian float64. verify/test_trainer.py rebuilds every fixture
and checks that the committed files still match.
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "js" / "fixtures" / "trainer"
TINY = dict(vocab_size=23, max_context_len=16, embed_dim=16, n_head=4, n_layer=2, ff_dim=64)
DEFAULT = dict(vocab_size=96, max_context_len=64, embed_dim=64, n_head=4, n_layer=2, ff_dim=192)
ADAMW = dict(lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)  # torch defaults except lr


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f8").tobytes()).decode("ascii")


def build_gpt(cfg: dict):
    """DLFS6's GPT in float64 with dropout off (the same module the book trains in ch03)."""
    dlfs6 = str(ROOT / "third_party" / "dlfs6")
    if dlfs6 not in sys.path:
        sys.path.insert(0, dlfs6)
    from codebot.model import GPT

    return GPT(**cfg, dropout_rate=0.0).double()


def tiny_weights(model, rng: np.random.Generator) -> None:
    """Larger-than-init random weights, so every path of the backward pass carries signal."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "norm.weight":
                p.copy_(torch.from_numpy(1 + 0.1 * rng.standard_normal(p.shape)))
            else:
                p.copy_(torch.from_numpy(0.3 * rng.standard_normal(p.shape)))


def formula_weights(model) -> None:
    """w[i] = 0.05 * sin(1.7 i + 0.3 k + 0.1) for the k-th parameter; LayerNorm weights get 1 + w[i].
    The same formula lives in tests/js/trainer-parity.test.mjs."""
    with torch.no_grad():
        for k, (name, p) in enumerate(model.named_parameters()):
            i = np.arange(p.numel(), dtype=np.float64)
            w = 0.05 * np.sin(1.7 * i + 0.3 * k + 0.1)
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "norm.weight":
                w = 1 + w
            p.copy_(torch.from_numpy(w.reshape(p.shape)))


def split_xy(ids: np.ndarray):
    """(B, T+1) token rows -> inputs ids[:, :-1] and next-token targets ids[:, 1:], as in TokenDataset."""
    return (torch.from_numpy(np.ascontiguousarray(ids[:, :-1])),
            torch.from_numpy(np.ascontiguousarray(ids[:, 1:])))


def loss_and_grads(model, x, y):
    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  # as in ch03/01_pretrain.py
    loss.backward()
    return loss.item(), {n: p.grad.detach().numpy().copy() for n, p in model.named_parameters()}


def build_tiny() -> dict:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build_gpt(TINY)
    tiny_weights(model, rng)
    B, T, V = 3, TINY["max_context_len"], TINY["vocab_size"]
    ids = rng.integers(0, V, size=(4, B, T + 1))  # batch 0 for the gradients, 1..3 for the AdamW steps
    x, y = split_xy(ids[0])
    weights = {n: p.detach().numpy().copy() for n, p in model.named_parameters()}
    loss, grads = loss_and_grads(model, x, y)

    opt = torch.optim.AdamW(model.parameters(), **ADAMW)
    losses = []
    for s in range(1, 4):
        xs, ys = split_xy(ids[s])
        opt.zero_grad()
        logits = model(xs)
        step_loss = F.cross_entropy(logits.view(-1, V), ys.view(-1))
        step_loss.backward()
        opt.step()
        losses.append(step_loss.item())
    return {
        "source": "DLFS6 codebot.model.GPT, float64, dropout_rate=0.0",
        "config": TINY, "B": B, "T": T, "ids": ids.tolist(),
        "names": list(weights), "shapes": {n: list(w.shape) for n, w in weights.items()},
        "weights": {n: b64(w) for n, w in weights.items()},
        "loss": loss, "grads": {n: b64(g) for n, g in grads.items()},
        "adamw": {**ADAMW, "betas": list(ADAMW["betas"]), "losses": losses,
                  "params_after": {n: b64(p.detach().numpy()) for n, p in model.named_parameters()}},
    }


def build_default() -> dict:
    model = build_gpt(DEFAULT)
    formula_weights(model)
    B, T, V = 2, DEFAULT["max_context_len"], DEFAULT["vocab_size"]
    ids = np.random.default_rng(1).integers(0, V, size=(B, T + 1))
    loss, grads = loss_and_grads(model, *split_xy(ids))
    n_params = sum(p.numel() for p in model.parameters())  # the shared embedding counts once
    checks = {n: {"sum": float(g.sum()), "sumsq": float((g * g).sum()), "head": g.reshape(-1)[:8].tolist()}
              for n, g in grads.items()}
    return {"source": "DLFS6 codebot.model.GPT, float64, dropout_rate=0.0", "config": DEFAULT,
            "B": B, "T": T, "ids": ids.tolist(), "names": list(grads), "param_count": n_params,
            "ln_v": math.log(V), "loss": loss, "grad_checks": checks}


def build_erf() -> dict:
    """Python's math.erf on a grid over [-7, 7] plus the branch points of fdlibm's erf (JavaScript has no Math.erf)."""
    xs = [i / 64 for i in range(-64 * 7, 64 * 7 + 1)]
    xs += [1e-30, -1e-30, 1e-10, 0.84375, 1.25, 1 / 0.35, 5.999, 6.0, 7.5, -0.3, 2.857142857142857]
    return {"x": xs, "erf": [math.erf(x) for x in xs]}


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, data in (("parity-tiny.json", build_tiny()), ("parity-default.json", build_default())):
        (FIXTURES / name).write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
        print(f"{name}: loss {data['loss']:.12f}")
    (FIXTURES / "erf.json").write_text(json.dumps(build_erf()), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
