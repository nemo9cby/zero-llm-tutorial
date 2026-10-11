# Checks for the TOC sections that have no / indirect code: 2.5 positional encoding, 2.7 value low-rank, 2.8.3 two formulas
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from bookload import load

torch.manual_seed(0)
m06 = load("ch02/06_attn_mask.py")
m07 = load("ch02/07_attn_value.py")
from codebot.model import GPT, MultiHeadAttention

print("== P1 (2.7): 07 == 06 with an E x E value matrix W_eff = W_o W_v of rank <= D ==")
a7 = m07.Attention(embed_dim=256, key_dim=64).eval()
a6 = m06.Attention(embed_dim=256, key_dim=64).eval()
with torch.no_grad():
    a6.W_q.weight.copy_(a7.W_q.weight); a6.W_k.weight.copy_(a7.W_k.weight)
    W_eff = a7.W_o.weight @ a7.W_v.weight                       # (256,64)@(64,256) -> (256,256)
    a6.W_v.weight.copy_(W_eff)
    x = torch.randn(2, 5, 256)
    print("max|07(x) - 06_with_W_eff(x)| =", (a7(x) - a6(x)).abs().max().item())
    print("rank(W_eff) =", torch.linalg.matrix_rank(W_eff).item(), "of", W_eff.shape,
          "| params: full E*E =", 256 * 256, " factorized 2*E*D =", 2 * 256 * 64)
    sv = torch.linalg.svdvals(W_eff)
    print("singular values #63,#64,#65 =", [f"{v:.2e}" for v in sv[62:65].tolist()])

print("\n== P2 (2.8.3): concat(heads) @ W_o^T == sum_h head_h @ W_o[:, hD:(h+1)D]^T ==")
mha = MultiHeadAttention(512, 8, 64).eval()
x = torch.randn(2, 10, 512)
with torch.no_grad():
    B, C, _ = x.shape
    Q, K, V = (lin(x).view(B, C, 8, 64).transpose(1, 2) for lin in (mha.W_q, mha.W_k, mha.W_v))
    heads = F.scaled_dot_product_attention(Q, K, V, is_causal=True)          # (B,H,C,D)
    concat = heads.transpose(1, 2).reshape(B, C, 512) @ mha.W_o.weight.T
    summed = sum(heads[:, h] @ mha.W_o.weight[:, h * 64:(h + 1) * 64].T for h in range(8))
    print("max|concat form - sum form| =", (concat - summed).abs().max().item(),
          "| vs module:", (mha(x) - concat).abs().max().item())

print("\n== P3 (2.5.3): sinusoidal PE -- PE(p).PE(q) depends only on p - q ==")
def sinusoidal(n_pos, d):
    pos = np.arange(n_pos)[:, None]; i = np.arange(d // 2)[None, :]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((n_pos, d)); pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang)
    return pe
pe = sinusoidal(256, 64)
for k in (1, 5, 50):
    dots = [pe[p] @ pe[p + k] for p in (0, 37, 150)]
    print(f"offset {k:2d}: PE(p).PE(p+k) for p=0,37,150 ->", np.round(dots, 6))
print("|PE(p)|^2 =", np.round((pe ** 2).sum(1)[:3], 6), "(= d/2)")

print("\n== P4 (2.5.4): learned position embeddings of the TRAINED codebot model are smooth ==")

from bookload import checkpoint
ck = checkpoint()
trained = GPT.load_from(ck)
P = F.normalize(trained.pos_embed.weight.detach(), dim=-1)
fresh = F.normalize(GPT(1000, 256, 384, 6, 6, 1536, 0.1).pos_embed.weight.detach(), dim=-1)
for name, M in (("trained", P), ("random init", fresh)):
    cs = lambda k: (M[:-k] * M[k:]).sum(-1).mean().item()
    print(f"{name:11s}: mean cos(pos_p, pos_p+k) for k=1: {cs(1):+.3f}  k=8: {cs(8):+.3f}  k=64: {cs(64):+.3f}")
print("pos_embed row norms trained: first/mid/last =",
      [round(v, 3) for v in trained.pos_embed.weight.norm(dim=-1)[[0, 128, 255]].tolist()],
      "| token embed mean norm =", round(trained.embed.weight.norm(dim=-1).mean().item(), 3))
