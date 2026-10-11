# torch checks for ch02/08_multi_head.py and codebot.model.MultiHeadAttention
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from bookload import load

torch.manual_seed(0)
m08 = load("ch02/08_multi_head.py")
from codebot.model import MultiHeadAttention as CodebotMHA

print("== H1: shape trace of 08's functional part (B=2,C=4,E=16,H=3,D=8) ==")
B, C, E, H, D = 2, 4, 16, 3, 8
x = torch.randn(B, C, E)
W_q, W_k, W_v = (nn.Linear(E, H * D, bias=False) for _ in range(3))
Q = W_q(x); print("W_q(x)                  ", tuple(Q.shape))
Q = Q.view(B, C, H, D); print(".view(B,C,H,D)          ", tuple(Q.shape))
Q = Q.transpose(1, 2); print(".transpose(1,2)         ", tuple(Q.shape), " contiguous?", Q.is_contiguous())
print("H*D =", H * D, "!= E =", E, "-> W_o maps", H * D, "->", E)

print("\n== H2: multi-head == H independent single heads on weight row-slices, concatenated ==")
mha = CodebotMHA(512, 8, 64).eval()
x = torch.randn(2, 10, 512)
with torch.no_grad():
    heads = []
    for h in range(8):
        sl = slice(h * 64, (h + 1) * 64)
        q, k, v = x @ mha.W_q.weight[sl].T, x @ mha.W_k.weight[sl].T, x @ mha.W_v.weight[sl].T
        s = (q @ k.transpose(-2, -1)) / 64 ** 0.5
        s = s.masked_fill(torch.tril(torch.ones(10, 10)) == 0, float('-inf'))
        heads.append(F.softmax(s, -1) @ v)                      # (2,10,64)
    loop_out = mha.W_o(torch.cat(heads, dim=-1))
    print("max|batched - per-head loop| =", (mha(x) - loop_out).abs().max().item())

print("\n== H3: .view after .transpose needs .contiguous() ==")
hid = torch.randn(2, 8, 10, 64).transpose(1, 2)                 # (2,10,8,64), non-contiguous
try:
    hid.view(2, 10, 512)
except RuntimeError as e:
    print("RuntimeError:", str(e)[:120])
print("reshape works and equals contiguous().view:",
      torch.equal(hid.reshape(2, 10, 512), hid.contiguous().view(2, 10, 512)))

print("\n== H4: splitting heads with view(B,H,C,D) (no transpose) silently breaks causality ==")
def mha_wrong(m, x):
    B, C, E = x.shape; H, D = m.n_head, m.head_dim
    Q, K, V = (lin(x).view(B, H, C, D) for lin in (m.W_q, m.W_k, m.W_v))   # WRONG
    s = (Q @ K.transpose(-2, -1)) / D ** 0.5
    s = s.masked_fill(torch.tril(torch.ones(C, C)) == 0, float('-inf'))
    hdn = (F.softmax(s, -1) @ V).transpose(1, 2).contiguous().view(B, C, H * D)
    return m.W_o(hdn)
with torch.no_grad():
    x2 = x.clone(); x2[:, 5:] = torch.randn(2, 5, 512)          # change only tokens 5..9
    for name, f in [("correct", mha), ("wrong view", lambda z: mha_wrong(mha, z))]:
        d = (f(x) - f(x2))[:, :5].abs().max().item()
        print(f"{name:10s}: max|dy| at positions 0..4 = {d:.3e}")

print("\n== H5: dropout is ON by default (modules start in train mode) ==")
m = CodebotMHA(512, 8, 64)                                       # dropout_rate=0.1, training=True
print("training =", m.training, "| two calls equal?", torch.equal(m(x), m(x)))
m.eval()
print("after eval(): two calls equal?", torch.equal(m(x), m(x)))
drop = nn.Dropout(0.1).train()
w = F.softmax(torch.randn(10, 10), -1)
print("row sums of dropped-out attention weights:", drop(w).sum(-1)[:5].round(decimals=3))

print("\n== H6: codebot MHA == F.scaled_dot_product_attention(is_causal=True) (eval mode) ==")
with torch.no_grad():
    Bq, Cq = x.shape[:2]
    Qh, Kh, Vh = (lin(x).view(Bq, Cq, 8, 64).transpose(1, 2) for lin in (mha.W_q, mha.W_k, mha.W_v))
    sdpa = F.scaled_dot_product_attention(Qh, Kh, Vh, is_causal=True)
    ref = mha.W_o(sdpa.transpose(1, 2).reshape(Bq, Cq, 512))
    print("max|codebot MHA - SDPA path| =", (mha(x) - ref).abs().max().item())

print("\n== H7: codebot MHA == torch.nn.MultiheadAttention with copied weights ==")
tm = nn.MultiheadAttention(512, 8, bias=False, batch_first=True).eval()
with torch.no_grad():
    tm.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight], 0))
    tm.out_proj.weight.copy_(mha.W_o.weight)
    causal = torch.triu(torch.ones(10, 10, dtype=torch.bool), 1)   # True = NOT allowed
    out, _ = tm(x, x, x, attn_mask=causal, need_weights=False)
    print("max|codebot - nn.MultiheadAttention| =", (mha(x) - out).abs().max().item())

print("\n== H8: parameter count = 4*E*(H*D) ==")
print("codebot MHA(512,8,64):", sum(p.numel() for p in mha.parameters()), "=", 4 * 512 * 512)
print("08 functional (E=16,H*D=24): ", 3 * 16 * 24 + 24 * 16)
print("nn.MultiheadAttention(bias=False):", sum(p.numel() for p in tm.parameters()))

print("\n== H9: the 08 class and codebot class are the same code ==")
m8 = m08.MultiHeadAttention(512, 8, 64).eval()
m8.load_state_dict(mha.state_dict())
print("state_dict keys:", list(mha.state_dict().keys()))
print("max|08 class - codebot class| =", (m8(x) - mha(x)).abs().max().item())

print("\n== H10: cost of materializing (B,H,C,C) scores: hand-written vs SDPA on CPU ==")
torch.set_num_threads(8)
for Cc in (256, 1024):
    Qb, Kb, Vb = (torch.randn(8, 8, Cc, 64) for _ in range(3))
    def hand():
        s = (Qb @ Kb.transpose(-2, -1)) / 8.0
        s = s.masked_fill(torch.tril(torch.ones(Cc, Cc)) == 0, float('-inf'))
        return F.softmax(s, -1) @ Vb
    def fused():
        return F.scaled_dot_product_attention(Qb, Kb, Vb, is_causal=True)
    with torch.no_grad():
        print(f"C={Cc}: max|hand-fused|={(hand()-fused()).abs().max().item():.2e}", end="  ")
        for name, f in (("hand", hand), ("sdpa", fused)):
            f(); t = time.perf_counter()
            for _ in range(5): f()
            print(f"{name}={1000*(time.perf_counter()-t)/5:.1f}ms", end="  ")
        print(f"scores tensor = {8*8*Cc*Cc*4/2**20:.0f} MiB fp32")
