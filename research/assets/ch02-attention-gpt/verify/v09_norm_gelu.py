# torch checks for ch02/09_norm_gelu.py vs torch built-ins and vs codebot.model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bookload import load

torch.manual_seed(0)
m09 = load("ch02/09_norm_gelu.py")
import codebot.model as cb

E = 384
x = torch.randn(4, 16, E) * 3 + 1.5

print("== N1: book LayerNorm == nn.LayerNorm (eps=1e-5, biased variance) ==")
ln_book, ln_torch = m09.LayerNorm(E), nn.LayerNorm(E)
with torch.no_grad():
    g, b = torch.randn(E), torch.randn(E)
    ln_book.gamma.copy_(g); ln_book.beta.copy_(b)
    ln_torch.weight.copy_(g); ln_torch.bias.copy_(b)
    print("max|diff| =", (ln_book(x) - ln_torch(x)).abs().max().item())

print("\n== N2: unbiased=True would be a (small) bug ==")
with torch.no_grad():
    mean = x.mean(-1, keepdim=True)
    y_b = (x - mean) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)
    y_u = (x - mean) / torch.sqrt(x.var(-1, keepdim=True, unbiased=True) + 1e-5)
    print("ratio y_unbiased / y_biased ~", (y_u / y_b).mean().item(), " sqrt((E-1)/E) =", math.sqrt((E - 1) / E))

print("\n== N3: per-token stats after LN (gamma=1, beta=0) ==")
y = m09.LayerNorm(E)(x)
print("row mean max|.| =", y.mean(-1).abs().max().item(),
      "| row var (biased) min/max =", y.var(-1, unbiased=False).min().item(), y.var(-1, unbiased=False).max().item())

print("\n== N4: LN is per token -> a sample's output does not depend on the rest of the batch ==")
with torch.no_grad():
    bn = nn.BatchNorm1d(E).train()
    full = ln_torch(x)[0]; alone = ln_torch(x[:1])[0]
    print("LayerNorm  max|full-alone| =", (full - alone).abs().max().item())
    xb = x[:, 0, :]                                   # (4, E) as a batch of 4 vectors
    print("BatchNorm  max|full-alone| =", (bn(xb)[0] - bn(torch.cat([xb[:1], torch.randn(3, E)]))[0]).abs().max().item())

print("\n== N5: GELU variants ==")
t = torch.linspace(-6, 6, 120001)
g_book = m09.GELU()(t)
g_erf = nn.GELU()(t)                                   # default approximate='none' (exact, erf)
g_tanh = nn.GELU(approximate='tanh')(t)
g_def = t * 0.5 * (1 + torch.erf(t / math.sqrt(2)))     # x * Phi(x)
print("book(tanh) vs nn.GELU(approximate='tanh'): max|diff| =", (g_book - g_tanh).abs().max().item())
print("book(tanh) vs nn.GELU() [erf]          : max|diff| =", (g_book - g_erf).abs().max().item(),
      "at x =", t[(g_book - g_erf).abs().argmax()].item())
print("nn.GELU() vs x*Phi(x)                   : max|diff| =", (g_erf - g_def).abs().max().item())
i = g_erf.argmin()
print("GELU min =", g_erf[i].item(), "at x =", t[i].item(), "| GELU(0) =", nn.GELU()(torch.tensor(0.)).item(),
      "| GELU(3)/3 =", (nn.GELU()(torch.tensor(3.)) / 3).item())

print("\n== N6: book FFN/Block vs codebot FFN/Block (codebot swapped to nn.GELU() and nn.LayerNorm) ==")
blk09 = m09.Block(E, 6).eval()          # ff_dim=None -> 4E; book LayerNorm + book GELU(tanh)
blkcb = cb.Block(E, 6, 4 * E).eval()     # nn.LayerNorm + nn.GELU() (erf)
k09, kcb = list(blk09.state_dict().keys()), list(blkcb.state_dict().keys())
print("09 keys :", k09)
print("cb keys :", kcb)
try:
    blk09.load_state_dict(blkcb.state_dict())
except RuntimeError as e:
    print("strict load fails:", str(e).splitlines()[0][:100], "...")
mapped = {k09[i]: v for i, v in enumerate(blkcb.state_dict().values())}  # same order, rename only
blk09.load_state_dict(mapped)
with torch.no_grad():
    d = (blk09(x) - blkcb(x)).abs().max().item()
    blkcb.ffn.layers[1] = nn.GELU(approximate='tanh')
    d2 = (blk09(x) - blkcb(x)).abs().max().item()
print(f"max|Block09 - BlockCodebot| = {d:.3e}  (only GELU erf vs tanh differs)")
print(f"after setting codebot GELU to approximate='tanh': {d2:.3e}")

print("\n== N7: Block parameter count (E=384, H=6, ff=4E) ==")
n = sum(p.numel() for p in blkcb.parameters())
ln_p, attn_p, ffn_p = 2 * E, 4 * E * E, (E * 4 * E + 4 * E) + (4 * E * E + E)
print("numel =", n, "| formula 2*LN + 4E^2 + FFN =", 2 * ln_p + attn_p + ffn_p,
      f"(LN {ln_p} x2, attn {attn_p}, ffn {ffn_p})")
print("~ 12E^2 =", 12 * E * E)

print("\n== N8: residual stream: zero the two output projections -> Block is the identity ==")
with torch.no_grad():
    blk = cb.Block(E, 6, 4 * E).eval()
    blk.attn.W_o.weight.zero_()
    blk.ffn.layers[2].weight.zero_(); blk.ffn.layers[2].bias.zero_()
    print("max|Block(x) - x| =", (blk(x) - x).abs().max().item())
