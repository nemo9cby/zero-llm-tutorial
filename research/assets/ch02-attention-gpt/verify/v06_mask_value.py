# torch checks for ch02/06_attn_mask.py and 07_attn_value.py, using the book's own classes
import torch
import torch.nn.functional as F
from bookload import load

torch.manual_seed(0)
m06 = load("ch02/06_attn_mask.py")
m07 = load("ch02/07_attn_value.py")

print("== M1: the causal mask and the resulting weights (C=5) ==")
C = 5
mask = torch.tril(torch.ones(C, C))
print(mask)
scores = torch.randn(C, C)
w = F.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
print(w.round(decimals=3))
print("upper triangle all exactly 0:", bool((w.triu(1) == 0).all()), "| row sums:", w.sum(-1))

print("\n== M2: causality test on the book's 06 Attention ==")
att = m06.Attention(embed_dim=256, key_dim=64)
x = torch.randn(2, 5, 256)
y = att(x)
x2 = x.clone(); x2[:, 3:] = torch.randn(2, 2, 256)       # change tokens 3,4 only
y2 = att(x2)
print("max|dy| at positions 0..2 =", (y - y2)[:, :3].abs().max().item())
print("max|dy| at positions 3..4 =", (y - y2)[:, 3:].abs().max().item())

print("\n== M3: hand-written == F.scaled_dot_product_attention(is_causal=True) ==")
with torch.no_grad():
    Q, K, V = att.W_q(x), att.W_k(x), att.W_v(x)          # V has dim 256 != 64 (Ev != D is allowed)
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print("06: shapes Q,K,V =", tuple(Q.shape), tuple(K.shape), tuple(V.shape),
          " max|diff| =", (att(x) - ref).abs().max().item())
    a7 = m07.Attention(embed_dim=256, key_dim=64)
    ref7 = a7.W_o(F.scaled_dot_product_attention(a7.W_q(x), a7.W_k(x), a7.W_v(x), is_causal=True))
    print("07: max|diff| =", (a7(x) - ref7).abs().max().item())

print("\n== M4: position 0 can only see itself -> output[0] == V[0] ==")
with torch.no_grad():
    print("06: max|y[:,0]-V[:,0]| =", (att(x)[:, 0] - att.W_v(x)[:, 0]).abs().max().item())
    print("07: max|y[:,0]-W_o(V[:,0])| =", (a7(x)[:, 0] - a7.W_o(a7.W_v(x)[:, 0])).abs().max().item())

print("\n== M5: a fully masked row gives NaN (not possible with tril; possible with padding masks) ==")
print(F.softmax(torch.tensor([float('-inf')] * 3), dim=-1))

print("\n== M6: softmax over the WRONG dim leaks the future even with the mask ==")
def attn_dim(Q, K, V, dim):
    s = (Q @ K.transpose(-2, -1)) / Q.shape[-1] ** 0.5
    s = s.masked_fill(torch.tril(torch.ones(s.shape[-1], s.shape[-1])) == 0, float('-inf'))
    return F.softmax(s, dim=dim) @ V
with torch.no_grad():
    Q, K, V = att.W_q(x), att.W_k(x), att.W_v(x)
    Q2, K2, V2 = att.W_q(x2), att.W_k(x2), att.W_v(x2)
    for dim in (-1, -2):
        d = (attn_dim(Q, K, V, dim) - attn_dim(Q2, K2, V2, dim))[:, :3].abs().max().item()
        print(f"softmax dim={dim}: max|dy| at positions 0..2 after changing tokens 3,4 = {d:.3e}")

print("\n== M7: parameter counts 06 vs 07 (E=256, D=64) ==")
n06 = sum(p.numel() for p in att.parameters())
n07 = sum(p.numel() for p in a7.parameters())
print("06 (W_q,W_k: E*D, W_v: E*E) =", n06, "=", 2 * 256 * 64 + 256 * 256)
print("07 (W_q,W_k,W_v: E*D, W_o: D*E) =", n07, "=", 4 * 256 * 64)

print("\n== M8: without mask & position info, attention is permutation-equivariant ==")
with torch.no_grad():
    perm = torch.tensor([4, 2, 0, 1, 3])
    f = lambda z: F.scaled_dot_product_attention(att.W_q(z), att.W_k(z), att.W_v(z), is_causal=False)
    print("max|f(x[perm]) - f(x)[perm]| =", (f(x[:, perm]) - f(x)[:, perm]).abs().max().item())
    g = lambda z: F.scaled_dot_product_attention(att.W_q(z), att.W_k(z), att.W_v(z), is_causal=True)
    print("with causal mask: max|g(x[perm]) - g(x)[perm]| =", (g(x[:, perm]) - g(x)[:, perm]).abs().max().item())
