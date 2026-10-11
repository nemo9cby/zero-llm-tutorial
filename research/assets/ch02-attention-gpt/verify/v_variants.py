# codebot vs storybot vs webbot model variants: param counts at their training configs + attention-core equivalences
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\chenf\Projects\_refs\dlfs6")
import codebot.model as cb
import storybot.model as sb
import webbot.model as wb

count = lambda m: sum(p.numel() for p in m.parameters())
with torch.device("meta"):
    c = cb.GPT(1000, 256, 384, 6, 6, 1536, 0.1)                 # ch03/01_pretrain.py
    s = sb.GPT(10000, 256, 512, 16, 4, 1344, 10000)             # ch06/05_pretrain.py
    w = wb.GPT(50000, 1024, 768, 12, 4, 12, 2048, 10000)        # ch09/01_pretrain.py
for name, m, V, E in (("codebot (ch03)", c, 1000, 384), ("storybot (ch06)", s, 10000, 512), ("webbot (ch09)", w, 50000, 768)):
    n = count(m)
    emb = V * E * (1 if name.startswith("codebot") else 2)    # codebot tied; others untied embed + unembed
    print(f"{name:16s}: {n:>12,} params | embedding+unembedding {emb:>11,} ({emb/n:5.1%}) | non-embedding {n-emb:>11,}")

torch.manual_seed(0)
print("\n== storybot MHA (rope=None, no cache) == codebot MHA (eval) with the same weights ==")
mc = cb.MultiHeadAttention(64, 4, 16).eval()
ms = sb.MultiHeadAttention(64, 4, 16, rope=None)
ms.load_state_dict(mc.state_dict())
x = torch.randn(2, 9, 64)
with torch.no_grad():
    print("max|diff| =", (mc(x) - ms(x)).abs().max().item())

print("\n== webbot GQA via SDPA(enable_gqa=True) == explicit repeat_interleave of KV heads ==")
mw = wb.MultiHeadAttention(64, 8, 2, 8, rope=None)
with torch.no_grad():
    B, C = 2, 9
    Q = mw.W_q(x).view(B, C, 8, 8).transpose(1, 2)
    K = mw.W_k(x).view(B, C, 2, 8).transpose(1, 2).repeat_interleave(4, dim=1)   # each KV head serves 4 query heads
    V = mw.W_v(x).view(B, C, 2, 8).transpose(1, 2).repeat_interleave(4, dim=1)
    s_ = (Q @ K.transpose(-2, -1)) / 8 ** 0.5
    s_ = s_.masked_fill(torch.tril(torch.ones(C, C)) == 0, float("-inf"))
    ref = mw.W_o((F.softmax(s_, -1) @ V).transpose(1, 2).reshape(B, C, 64))
    print("max|diff| =", (mw(x) - ref).abs().max().item(),
          "| K/V params per layer:", mw.W_k.weight.numel() + mw.W_v.weight.numel(), "vs MHA", 2 * 64 * 64)
