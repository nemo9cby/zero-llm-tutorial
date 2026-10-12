"""Export the data files of chapter 2 (web/data/ch02/) from the real DLFS6 CodeBot code and checkpoint.

Regenerate (CPU only; needs codebot/model_pretrain.pt from codebot/download.sh):
    CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch02_export.py

Writes:
  attn_maps.json     attention weights of the trained CodeBot (model_pretrain.pt) on two code snippets
  pos_embed_cos.json mean cosine similarity of position embeddings vs. offset (trained vs. fresh init)
  tiny_ref.npz       weights of a tiny random codebot GPT + ids + PyTorch reference logits
verify/test_ch02.py re-derives all three independently from the same code/checkpoint and checks them.
This script is a tool, not a test: pytest never runs it.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLFS6 = os.path.join(ROOT, "third_party", "dlfs6")
OUT = os.path.join(ROOT, "web", "data", "ch02")
CKPT = os.path.join(DLFS6, "codebot", "model_pretrain.pt")
SNIPPETS = ["def add(a, b):\n    return a + b\n",
            "def square(x):\n    return x * x\n\nprint(square(3))\n"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def attention_maps(model, ids):
    """(L, H, C, C) causal attention weights, recomputed from each block's attention input (eval mode)."""
    caps = []
    hooks = [b.attn.register_forward_pre_hook(lambda m, inp: caps.append(inp[0])) for b in model.blocks]
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0]
    for hk in hooks:
        hk.remove()
    C, H = len(ids), model.n_head
    D = model.embed_dim // H
    maps = []
    with torch.no_grad():
        for blk, xin in zip(model.blocks, caps):
            a = blk.attn
            q = a.W_q(xin).view(1, C, H, D).transpose(1, 2)
            k = a.W_k(xin).view(1, C, H, D).transpose(1, 2)
            s = (q @ k.transpose(-2, -1)) / D ** 0.5
            s = s.masked_fill(torch.tril(torch.ones(C, C)) == 0, float("-inf"))
            maps.append(F.softmax(s, -1)[0])
    return torch.stack(maps), logits


def main():
    sys.path.insert(0, DLFS6)
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer

    if not os.path.isfile(CKPT):
        sys.exit(f"{CKPT} not found: run codebot/download.sh first")
    os.makedirs(OUT, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", DLFS6, "rev-parse", "HEAD"], text=True).strip()
    tok = BPETokenizer.load_from(os.path.join(DLFS6, "codebot", "merge_rules.pkl"))
    model = GPT.load_from(CKPT).eval()

    snippets = []
    for text in SNIPPETS:
        ids = tok.encode(text)
        W, logits = attention_maps(model, ids)
        probs = F.softmax(logits, -1)
        top = probs.argmax(-1)
        snippets.append({
            "text": text,
            "ids": ids,
            "tokens": [tok.decode([i]) for i in ids],
            "pred": [tok.decode([int(i)]) for i in top],
            "pred_prob": [round(float(probs[t, top[t]]), 4) for t in range(len(ids))],
            "attn": [[[[round(float(v), 4) for v in row] for row in W[l, h]]
                      for h in range(W.shape[1])] for l in range(W.shape[0])],
        })
    meta = {"checkpoint": "codebot/model_pretrain.pt", "bytes": os.path.getsize(CKPT),
            "sha256": sha256(CKPT)[:16], "dlfs6_commit": commit[:7], "torch": torch.__version__,
            "layout": "attn[layer][head][query][key], eval mode, rounded to 4 decimals"}
    with open(os.path.join(OUT, "attn_maps.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "snippets": snippets}, f, ensure_ascii=False, separators=(",", ":"))

    def mean_cos(P):
        """Mean cosine similarity between position p and p + k, for k = 1 .. n-1."""
        P = F.normalize(P, dim=-1)
        return [round(float((P[:-k] * P[k:]).sum(-1).mean()), 4) for k in range(1, P.shape[0])]
    torch.manual_seed(0)
    fresh = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
    with torch.no_grad():
        pos = {"meta": {**meta, "layout": "cos[k-1] = mean_p cos(pos_embed[p], pos_embed[p+k]), k = 1..255; "
                                          "fresh = GPT(...) after torch.manual_seed(0)"},
               "trained": mean_cos(model.pos_embed.weight), "fresh": mean_cos(fresh.pos_embed.weight)}
    with open(os.path.join(OUT, "pos_embed_cos.json"), "w", encoding="utf-8") as f:
        json.dump(pos, f, separators=(",", ":"))

    # Tiny random codebot GPT for the NumPy end-to-end forward (2.10.4).
    torch.manual_seed(0)
    tiny = GPT(vocab_size=50, max_context_len=16, embed_dim=32, n_head=4, n_layer=2, ff_dim=128,
               dropout_rate=0.1).eval()
    ids = np.array([3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9, 32], dtype=np.int64)
    with torch.no_grad():
        ref = tiny(torch.tensor(ids[None]))[0].numpy()
    arrs = {k: v.numpy() for k, v in tiny.state_dict().items()}
    arrs.update(ids=ids, ref_logits=ref)
    np.savez(os.path.join(OUT, "tiny_ref.npz"), **arrs)

    for name in ("attn_maps.json", "pos_embed_cos.json", "tiny_ref.npz"):
        print(f"{name}: {os.path.getsize(os.path.join(OUT, name)):,} bytes")


if __name__ == "__main__":
    main()
