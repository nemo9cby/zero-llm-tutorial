# Inspect the book's trained codebot checkpoint (downloaded to scratch): config, loss, generation, attention maps.
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

REPO = r"C:\Users\chenf\Projects\_refs\dlfs6"
sys.path.insert(0, REPO)
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate

HERE = os.path.dirname(os.path.abspath(__file__))
from bookload import checkpoint
CK = checkpoint()
torch.manual_seed(0)

print("== T1: checkpoint config and size ==")
raw = torch.load(CK, map_location="cpu")            # weights_only=True by default in torch>=2.6
print({k: v for k, v in raw.items() if k != "model_state_dict"})
model = GPT.load_from(CK).eval()
print(f"params = {sum(p.numel() for p in model.parameters()):,} | tied after load: {model.embed.weight is model.unembed.weight}")
print("state_dict has both embed.weight and unembed.weight:", "embed.weight" in raw["model_state_dict"], "unembed.weight" in raw["model_state_dict"])

print("\n== T2: tokenizer ==")
tok = BPETokenizer.load_from(os.path.join(REPO, "codebot", "merge_rules.pkl"))
print("merges =", len(tok.merge_rules), "| vocab_size =", tok.vocab_size, "| end_token_id =", tok.end_token_id)
snippet = "def add(a, b):\n    return a + b\n"
ids = tok.encode(snippet)
print("ids =", ids)
print("tokens =", [tok.decode([i]) for i in ids])

print("\n== T3: loss of the trained model on tiny_codes.bin windows vs init ==")
data = np.fromfile(os.path.join(REPO, "codebot", "tiny_codes.bin"), dtype=np.uint16)
print("tokens in tiny_codes.bin =", len(data), "| max id =", int(data.max()))
rng = np.random.default_rng(0)
starts = rng.integers(0, len(data) - 257, size=32)
xb = torch.tensor(np.stack([data[s:s + 256] for s in starts]).astype(np.int64))
yb = torch.tensor(np.stack([data[s + 1:s + 257] for s in starts]).astype(np.int64))
with torch.no_grad():
    t0 = time.perf_counter()
    loss = F.cross_entropy(model(xb).reshape(-1, 1000), yb.reshape(-1)).item()
    dt = time.perf_counter() - t0
    fresh = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval()
    loss0 = F.cross_entropy(fresh(xb).reshape(-1, 1000), yb.reshape(-1)).item()
print(f"trained CE = {loss:.4f} (ppl {math.exp(loss):.2f}) | fresh-init CE = {loss0:.4f} | ln(1000) = {math.log(1000):.4f} | fwd 32x256 took {dt:.2f}s")

print("\n== T4: greedy generation from 'def' (temperature=0, 60 tokens) ==")
print(repr(generate(model, tok, "def", max_new_tokens=60, temperature=0)))

print("\n== T5: attention maps on the snippet (eval mode) ==")
caps = []
hooks = [b.attn.register_forward_pre_hook(lambda m, inp: caps.append(inp[0])) for b in model.blocks]
with torch.no_grad():
    model(torch.tensor([ids]))
for h in hooks: h.remove()
C = len(ids); H, D = 6, 64
maps = []
with torch.no_grad():
    for li, (blk, xin) in enumerate(zip(model.blocks, caps)):
        a = blk.attn
        q = a.W_q(xin).view(1, C, H, D).transpose(1, 2)
        k = a.W_k(xin).view(1, C, H, D).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / D ** 0.5
        w = F.softmax(s.masked_fill(torch.tril(torch.ones(C, C)) == 0, float("-inf")), -1)[0]   # (H,C,C)
        maps.append(w)
W = torch.stack(maps)                                   # (L,H,C,C)
prev = torch.stack([W[:, :, t, t - 1] for t in range(1, C)], -1).mean(-1)      # weight on t-1
first = W[:, :, 1:, 0].mean(-1)                                                # weight on token 0
selfw = torch.stack([W[:, :, t, t] for t in range(1, C)], -1).mean(-1)
ent = -(W.clamp_min(1e-12).log() * W).sum(-1)[:, :, 1:].mean(-1)
print("mean weight on previous token (rows=layer, cols=head):\n", prev.numpy().round(2))
print("mean weight on first token:\n", first.numpy().round(2))
print("mean weight on itself:\n", selfw.numpy().round(2))
li, hi = divmod(int(prev.argmax()), H)
print(f"strongest previous-token head: layer {li} head {hi} (mean {prev[li, hi]:.2f})")

out = {"model": "codebot/model_pretrain.pt (DLFS6 ch03)", "snippet": snippet, "ids": ids,
       "tokens": [tok.decode([i]) for i in ids], "shape": "[layer][head][query][key]",
       "attn": [[[[round(float(v), 4) for v in row] for row in W[l, h]] for h in range(H)] for l in range(6)]}
path = os.path.join(os.path.dirname(HERE), "ch02_attn_maps.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("saved", path, os.path.getsize(path), "bytes")
