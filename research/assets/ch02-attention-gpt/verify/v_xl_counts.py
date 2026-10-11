# GPT-2 XL shape (V=50257, ctx=1024, E=1600, L=48, H=25, ff=6400) in three architectures. Needs: --with transformers
import sys
import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, r"C:\Users\chenf\Projects\_refs\dlfs6")
import codebot.model as cb
import webbot.model as wb

n = lambda m: sum(p.numel() for p in m.parameters())
V, CTX, E, L, H, FF = 50257, 1024, 1600, 48, 25, 6400
with torch.device("meta"):
    hf = GPT2LMHeadModel(GPT2Config(n_embd=E, n_layer=L, n_head=H))
    c = cb.GPT(V, CTX, E, H, L, FF, 0.1)
    w = wb.GPT(V, CTX, E, H, H, L, FF, 10000)   # RMSNorm + SwiGLU + RoPE + untied, no biases (CS336-A1-like)
print(f"HF GPT-2 XL           : {n(hf):,}")
print(f"codebot @ GPT-2 XL    : {n(c):,}")
print(f"LLaMA-style @ XL shape: {n(w):,}  formula 2VE + L(4E^2+3E*FF+2E) + E = {2*V*E + L*(4*E*E + 3*E*FF + 2*E) + E:,}")
