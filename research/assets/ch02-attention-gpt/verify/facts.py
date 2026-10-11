# Minimal verifiable facts for DLFS6 ch02 (Attention -> GPT-2). Each `code` is a standalone snippet.
# browser=True  -> NumPy/math/json only (runs in Pyodide); needs: "" | "torch" | "ckpt" | "transformers"
# Run all with run_facts.py (CPython) and the browser subset with ../pyodide_test/run_facts.mjs.
# Snippets assume cwd = kit root (research/assets/ch02-attention-gpt) and sys.path has the repo + verify/.

FACTS = [
# ---------------- 2.1 soft dictionary ----------------
dict(id="F01", sec="2.1", browser=True, needs="", claim="Soft dictionary for new movie (6,4,5) predicts 78.66; weights 72.62% on (7,6,2), 26.71% on (5,5,5)",
code=r'''
import numpy as np
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float)
V = np.array([85,70,60,75,80,65,90], float)
s = K @ np.array([6., 4., 5.])
w = np.exp(s - s.max()); w /= w.sum()
print((w * 100).round(2), round(float(w @ V), 2))
'''),
dict(id="F02", sec="2.1", browser=True, needs="", claim="Dot-product similarity is magnitude-sensitive: argmax dot = (7,6,2) but nearest/cosine = (5,5,5)",
code=r'''
import numpy as np
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float); q = np.array([6., 4., 5.])
print("dot:", K[np.argmax(K @ q)], "cos:", K[np.argmax(K @ q / np.linalg.norm(K, axis=1))],
      "nearest:", K[np.argmin(np.linalg.norm(K - q, axis=1))])
'''),
dict(id="F03", sec="2.1", browser=True, needs="", claim="Hard lookup is the T->0 limit only without ties: querying with each key returns 87.5 (not 85) and 73.33 (not 75)",
code=r'''
import numpy as np
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float); V = np.array([85,70,60,75,80,65,90.])
def soft(q, T):
    s = K @ q / T; w = np.exp(s - s.max()); return float((w / w.sum()) @ V)
print([round(soft(k, 0.001), 2) for k in K], "true:", V.tolist())
'''),
dict(id="F04", sec="2.1", browser=True, needs="", claim="01's naive softmax (no max-subtraction) overflows to NaN once scores reach ~710; the stable form does not",
code=r'''
import numpy as np
s = np.array([71., 59, 59, 75, 76, 70, 63]) * 10        # 01's scores with features on a 0..100 scale
with np.errstate(over="ignore", invalid="ignore"):
    print("naive :", np.exp(s) / np.exp(s).sum())
e = np.exp(s - s.max()); print("stable:", (e / e.sum()).round(4))
'''),
dict(id="F05", sec="2.1", browser=True, needs="", claim="softmax is shift-invariant but not scale-invariant",
code=r'''
import numpy as np
sm = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
x = np.array([1., 2., 3.])
print(sm(x).round(4), sm(x + 100).round(4), sm(100 * x).round(4))
'''),
# ---------------- 2.2 matrix form ----------------
dict(id="F06", sec="2.2", browser=True, needs="", claim="Matrix form softmax(QK^T)V reproduces 78.66 / 69.76 / 61.20 for the three queries of 02; rows sum to 1",
code=r'''
import numpy as np
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float); V = np.array([[85],[70],[60],[75],[80],[65],[90.]])
Q = np.array([[6,4,5],[2,8,3],[4,3,7]], float)
S = Q @ K.T; W = np.exp(S - S.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
print(S.shape, W.sum(1), (W @ V).ravel().round(2))
'''),
# ---------------- 2.3 scaling ----------------
dict(id="F07", sec="2.3", browser=True, needs="", claim="softmax([100,200,300]) saturates (float32: [0, 3.8e-44, 1]); dividing by 100 first gives [0.090, 0.245, 0.665]",
code=r'''
import numpy as np
for x in (np.array([100., 200., 300.], dtype=np.float32), np.array([1., 2., 3.])):
    e = np.exp(x - x.max()); print(e / e.sum())
'''),
dict(id="F08", sec="2.3", browser=True, needs="", claim="For q,k ~ N(0, I_d): Var(q.k) ~ d and Var(q.k/sqrt(d)) ~ 1 (ratio exactly d)",
code=r'''
import numpy as np
rng = np.random.default_rng(0); d = 64
q, k = rng.standard_normal((10000, d)), rng.standard_normal((10000, d))
dots = (q * k).sum(1)
print(round(dots.var(), 2), round((dots / np.sqrt(d)).var(), 4), round(dots.var() / (dots / np.sqrt(d)).var(), 6))
'''),
dict(id="F09", sec="2.3", browser=True, needs="", claim="Without scaling, attention over 256 keys saturates more as d grows; with 1/sqrt(d) its sharpness is d-invariant",
code=r'''
import numpy as np
rng = np.random.default_rng(0)
for d in (16, 64, 256, 1024):
    raw = np.array([rng.standard_normal((256, d)) @ rng.standard_normal(d) for _ in range(200)])
    out = []
    for z in (raw, raw / np.sqrt(d)):
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        out.append(f"maxw={p.max(1).mean():.3f} sat={(p.max(1) > .99).mean():.2f}")
    print(f"d={d:4d}  unscaled: {out[0]}  |  scaled: {out[1]}")
'''),
dict(id="F10", sec="2.3", browser=False, needs="torch", claim="At init (std 0.02) the real model's layer-0 scores are tiny (std ~0.15) -> attention ~uniform regardless of scaling",
code=r'''
import torch, torch.nn.functional as F
from codebot.model import GPT
torch.manual_seed(0); m = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval(); a = m.blocks[0].attn
ids = torch.randint(0, 1000, (1, 256))
with torch.no_grad():
    h = m.blocks[0].norm1(m.embed(ids) + m.pos_embed(torch.arange(256)))
    q, k = (l(h).view(1, 256, 6, 64).transpose(1, 2) for l in (a.W_q, a.W_k))
    s = q @ k.transpose(-2, -1) / 8
print("score std:", round(s.std().item(), 4), "| last-row max weight:", round(F.softmax(s[0, 0, -1], -1).max().item(), 5), "vs 1/256 =", round(1 / 256, 5))
'''),
# ---------------- 2.4/2.5 position ----------------
dict(id="F11", sec="2.5", browser=True, needs="", claim="Without mask and position info, attention is permutation-equivariant; the causal mask breaks this",
code=r'''
import numpy as np
rng = np.random.default_rng(0); C, E = 5, 8
Wq, Wk, Wv = (rng.standard_normal((E, E)) for _ in range(3)); x = rng.standard_normal((C, E))
def attn(x, causal):
    s = (x @ Wq) @ (x @ Wk).T / np.sqrt(E)
    if causal: s = np.where(np.tril(np.ones((C, C), bool)), s, -np.inf)
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); return p @ (x @ Wv)
perm = [4, 2, 0, 1, 3]
for causal in (False, True):
    print("causal" if causal else "no mask", np.abs(attn(x[perm], causal) - attn(x, causal)[perm]).max())
'''),
dict(id="F12", sec="2.5", browser=True, needs="", claim="Sinusoidal PE: PE(p).PE(p+k) depends only on k; |PE(p)|^2 = d/2",
code=r'''
import numpy as np
pos, i = np.arange(256)[:, None], np.arange(32)[None, :]
pe = np.zeros((256, 64)); ang = pos / 10000 ** (2 * i / 64); pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang)
print([round(float(pe[p] @ pe[p + 5]), 6) for p in (0, 37, 150)], (pe ** 2).sum(1)[:3])
'''),
dict(id="F13", sec="2.5", browser=False, needs="ckpt", claim="The trained model's learned pos_embed is smooth: mean cos(p, p+k) = 0.949 / 0.562 / 0.037 for k = 1 / 8 / 64 (random init ~0)",
code=r'''
import torch, torch.nn.functional as F
from codebot.model import GPT
from bookload import checkpoint
P = F.normalize(GPT.load_from(checkpoint()).pos_embed.weight.detach(), dim=-1)
print({k: round((P[:-k] * P[k:]).sum(-1).mean().item(), 3) for k in (1, 8, 64)})
'''),
dict(id="F14", sec="2.5", browser=False, needs="torch", claim="Learned absolute positions cap the context: 257 tokens into a max_context_len=256 model raises IndexError",
code=r'''
import torch
from codebot.model import GPT
m = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
try: m(torch.randint(0, 1000, (1, 257)))
except IndexError as e: print("IndexError:", e)
'''),
# ---------------- 2.6 mask ----------------
dict(id="F15", sec="2.6", browser=True, needs="", claim="Causal softmax: upper triangle exactly 0, every row sums to 1, row 0 = [1,0,...]",
code=r'''
import numpy as np
C = 5; s = np.random.default_rng(0).standard_normal((C, C))
s = np.where(np.tril(np.ones((C, C), bool)), s, -np.inf)
w = np.exp(s - s.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
print(w.round(3)); print("upper==0:", bool((np.triu(w, 1) == 0).all()), "row sums:", w.sum(1))
'''),
dict(id="F16", sec="2.6", browser=False, needs="torch", claim="Causality test on the book's 06 Attention: changing tokens 3,4 leaves outputs 0..2 bit-identical",
code=r'''
import torch
from bookload import load
torch.manual_seed(0); att = load("ch02/06_attn_mask.py").Attention(256, 64)
x = torch.randn(2, 5, 256); x2 = x.clone(); x2[:, 3:] = torch.randn(2, 2, 256)
d = (att(x) - att(x2)).abs()
print("positions 0-2:", d[:, :3].max().item(), "| positions 3-4:", round(d[:, 3:].max().item(), 3))
'''),
dict(id="F17", sec="2.6", browser=False, needs="torch", claim="06 Attention == F.scaled_dot_product_attention(is_causal=True), even with value dim 256 != key dim 64",
code=r'''
import torch, torch.nn.functional as F
from bookload import load
torch.manual_seed(0); att = load("ch02/06_attn_mask.py").Attention(256, 64); x = torch.randn(2, 5, 256)
with torch.no_grad():
    ref = F.scaled_dot_product_attention(att.W_q(x), att.W_k(x), att.W_v(x), is_causal=True)
    print(tuple(ref.shape), (att(x) - ref).abs().max().item())
'''),
dict(id="F18", sec="2.6", browser=True, needs="", claim="A fully masked row gives NaN (impossible with tril, possible with padding masks)",
code=r'''
import numpy as np
s = np.full(3, -np.inf)
with np.errstate(invalid="ignore"): print(np.exp(s - s.max()) / np.exp(s - s.max()).sum())
'''),
dict(id="F19", sec="2.6", browser=True, needs="", claim="Softmax over the wrong axis leaks the future even though masked entries are 0",
code=r'''
import numpy as np
rng = np.random.default_rng(0); C, E = 5, 8
Wq, Wk, Wv = (rng.standard_normal((E, E)) for _ in range(3))
x = rng.standard_normal((C, E)); x2 = x.copy(); x2[3:] = rng.standard_normal((2, E))
def attn(x, axis):
    s = np.where(np.tril(np.ones((C, C), bool)), (x @ Wq) @ (x @ Wk).T / np.sqrt(E), -np.inf)
    p = np.exp(s - s.max(axis, keepdims=True)); p /= p.sum(axis, keepdims=True); return p @ (x @ Wv)
for axis in (-1, -2): print("softmax axis", axis, "-> max change at positions 0-2:", np.abs(attn(x, axis) - attn(x2, axis))[:3].max())
'''),
dict(id="F20", sec="2.6", browser=False, needs="torch", claim="Why the mask (2.6.1-2.6.2): on i.i.d. random tokens a causal model stays at ln V, a mask-less one 'cheats' down to ln(V)/C",
code=r'''
import math
from v_cheat import run
print("causal:", round(run(True).mean().item(), 4), "| no mask:", round(run(False).mean().item(), 4),
      "| ln 8 =", round(math.log(8), 4), "| ln(8)/8 =", round(math.log(8) / 8, 4))
'''),
dict(id="F21", sec="2.6", browser=False, needs="torch", claim="SDPA's is_causal=True is top-left aligned: with 1 query and 5 cached keys it only sees key 0 (so KV-cache decode must use is_causal=False)",
code=r'''
import torch, torch.nn.functional as F
torch.manual_seed(0); q, K, V = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8)
print(torch.allclose(F.scaled_dot_product_attention(q, K, V, is_causal=True), V[:, :, :1]))
'''),
# ---------------- 2.7 value low-rank ----------------
dict(id="F22", sec="2.7", browser=False, needs="torch", claim="07 == 06 whose E x E value matrix is W_o W_v (rank 64); params 98,304 -> 65,536",
code=r'''
import torch
from bookload import load
torch.manual_seed(0)
a7, a6 = load("ch02/07_attn_value.py").Attention(256, 64), load("ch02/06_attn_mask.py").Attention(256, 64)
with torch.no_grad():
    a6.W_q.weight.copy_(a7.W_q.weight); a6.W_k.weight.copy_(a7.W_k.weight)
    a6.W_v.weight.copy_(a7.W_o.weight @ a7.W_v.weight)
    x = torch.randn(2, 5, 256)
    print((a7(x) - a6(x)).abs().max().item(), torch.linalg.matrix_rank(a6.W_v.weight).item(),
          sum(p.numel() for p in a6.parameters()), sum(p.numel() for p in a7.parameters()))
'''),
dict(id="F23", sec="2.7", browser=False, needs="torch", claim="Position 0 can only attend to itself, so its output is exactly its own value vector",
code=r'''
import torch
from bookload import load
torch.manual_seed(0); att = load("ch02/06_attn_mask.py").Attention(256, 64); x = torch.randn(2, 5, 256)
with torch.no_grad(): print((att(x)[:, 0] - att.W_v(x)[:, 0]).abs().max().item())
'''),
# ---------------- 2.8 multi-head ----------------
dict(id="F24", sec="2.8", browser=False, needs="torch", claim="Shape trace of 08: (2,4,24) -view-> (2,4,3,8) -transpose-> (2,3,4,8), which is non-contiguous",
code=r'''
import torch, torch.nn as nn
B, C, E, H, D = 2, 4, 16, 3, 8
Q = nn.Linear(E, H * D, bias=False)(torch.randn(B, C, E)); Qh = Q.view(B, C, H, D).transpose(1, 2)
print(tuple(Q.shape), tuple(Q.view(B, C, H, D).shape), tuple(Qh.shape), Qh.is_contiguous())
'''),
dict(id="F25", sec="2.8", browser=False, needs="torch", claim=".view after .transpose raises RuntimeError; .contiguous().view == .reshape",
code=r'''
import torch
h = torch.randn(2, 8, 10, 64).transpose(1, 2)
try: h.view(2, 10, 512)
except RuntimeError as e: print("RuntimeError:", str(e)[:58])
print(torch.equal(h.reshape(2, 10, 512), h.contiguous().view(2, 10, 512)))
'''),
dict(id="F26", sec="2.8", browser=True, needs="", claim="Batched heads (reshape+transpose) == looping over weight row-slices per head",
code=r'''
import numpy as np
rng = np.random.default_rng(0); C, E, H = 6, 16, 4; D = E // H
Wq, Wk, Wv = (rng.standard_normal((H * D, E)) for _ in range(3)); x = rng.standard_normal((C, E))
mask = np.tril(np.ones((C, C), bool))
def attn(q, k, v):
    s = np.where(mask, q @ np.swapaxes(k, -1, -2) / np.sqrt(D), -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); return p @ v
split = lambda t: t.reshape(C, H, D).transpose(1, 0, 2)                      # (C, H*D) -> (H, C, D)
batched = attn(split(x @ Wq.T), split(x @ Wk.T), split(x @ Wv.T))
loop = np.stack([attn(*(x @ W[h*D:(h+1)*D].T for W in (Wq, Wk, Wv))) for h in range(H)])
print(batched.shape, np.abs(batched - loop).max())
'''),
dict(id="F27", sec="2.8", browser=True, needs="", claim="Splitting heads with reshape(H, C, D) without the transpose silently breaks causality",
code=r'''
import numpy as np
rng = np.random.default_rng(0); C, E, H = 6, 16, 4; D = E // H
Wq, Wk, Wv = (rng.standard_normal((H * D, E)) for _ in range(3))
x = rng.standard_normal((C, E)); x2 = x.copy(); x2[4:] = rng.standard_normal((2, E))
def mha(x, split):
    q, k, v = (split(x @ W.T) for W in (Wq, Wk, Wv))
    s = np.where(np.tril(np.ones((C, C), bool)), q @ k.transpose(0, 2, 1) / np.sqrt(D), -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); return (p @ v).transpose(1, 0, 2).reshape(C, E)
right = lambda t: t.reshape(C, H, D).transpose(1, 0, 2)
wrong = lambda t: t.reshape(H, C, D)
for name, sp in (("right", right), ("wrong", wrong)): print(name, "max change at positions 0-3:", np.abs(mha(x, sp) - mha(x2, sp))[:4].max())
'''),
dict(id="F28", sec="2.8", browser=False, needs="torch", claim="codebot MultiHeadAttention == SDPA(is_causal=True) path == torch.nn.MultiheadAttention with copied weights",
code=r'''
import torch, torch.nn as nn, torch.nn.functional as F
from codebot.model import MultiHeadAttention
torch.manual_seed(0); m = MultiHeadAttention(512, 8, 64).eval(); x = torch.randn(2, 10, 512)
with torch.no_grad():
    Q, K, V = (l(x).view(2, 10, 8, 64).transpose(1, 2) for l in (m.W_q, m.W_k, m.W_v))
    sdpa = m.W_o(F.scaled_dot_product_attention(Q, K, V, is_causal=True).transpose(1, 2).reshape(2, 10, 512))
    t = nn.MultiheadAttention(512, 8, bias=False, batch_first=True).eval()
    t.in_proj_weight.copy_(torch.cat([m.W_q.weight, m.W_k.weight, m.W_v.weight])); t.out_proj.weight.copy_(m.W_o.weight)
    ref, _ = t(x, x, x, attn_mask=torch.triu(torch.ones(10, 10, dtype=torch.bool), 1), need_weights=False)
    print((m(x) - sdpa).abs().max().item(), (m(x) - ref).abs().max().item())
'''),
dict(id="F29", sec="2.8", browser=True, needs="", claim="2.8.3 'two equivalent formulas': concat(heads) W_o^T == sum_h head_h W_o[:, hD:(h+1)D]^T",
code=r'''
import numpy as np
rng = np.random.default_rng(0); C, H, D, E = 6, 4, 8, 32
heads = rng.standard_normal((H, C, D)); Wo = rng.standard_normal((E, H * D))
concat = heads.transpose(1, 0, 2).reshape(C, H * D) @ Wo.T
summed = sum(heads[h] @ Wo[:, h*D:(h+1)*D].T for h in range(H))
print(np.abs(concat - summed).max())
'''),
dict(id="F30", sec="2.8", browser=False, needs="torch", claim="Modules start in train mode: dropout makes two forwards differ and dropped attention rows stop summing to 1",
code=r'''
import torch, torch.nn.functional as F
from codebot.model import MultiHeadAttention
torch.manual_seed(0); m = MultiHeadAttention(512, 8, 64); x = torch.randn(2, 10, 512)
print("training:", m.training, "| equal:", torch.equal(m(x), m(x)), end=" ")
m.eval(); print("| after eval():", torch.equal(m(x), m(x)))
print(torch.nn.Dropout(0.1)(F.softmax(torch.randn(4, 10), -1)).sum(-1))
'''),
dict(id="F31", sec="2.8", browser=False, needs="torch", claim="MHA parameter count = 4 * E * (H*D) = 1,048,576 for E=512, H=8, D=64 (no biases)",
code=r'''
from codebot.model import MultiHeadAttention
print(sum(p.numel() for p in MultiHeadAttention(512, 8, 64).parameters()), 4 * 512 * 8 * 64)
'''),
# ---------------- 2.9 block ----------------
dict(id="F32", sec="2.9", browser=False, needs="torch", claim="Book LayerNorm == nn.LayerNorm (eps 1e-5, biased variance)",
code=r'''
import torch, torch.nn as nn
from bookload import load
torch.manual_seed(0); ln = load("ch02/09_norm_gelu.py").LayerNorm(384); ref = nn.LayerNorm(384)
with torch.no_grad():
    g, b = torch.randn(384), torch.randn(384); ln.gamma.copy_(g); ln.beta.copy_(b); ref.weight.copy_(g); ref.bias.copy_(b)
    x = torch.randn(4, 16, 384) * 3 + 1.5
    print((ln(x) - ref(x)).abs().max().item())
'''),
dict(id="F33", sec="2.9", browser=True, needs="", claim="Using the unbiased variance would rescale LN output by sqrt((E-1)/E) = 0.998697 for E=384",
code=r'''
import numpy as np
x = np.random.default_rng(0).standard_normal((4, 384)) * 3 + 1.5
z = lambda ddof: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, ddof=ddof, keepdims=True) + 1e-5)
print(round(float((z(1) / z(0)).mean()), 6), round(float(np.sqrt(383 / 384)), 6))
'''),
dict(id="F34", sec="2.9", browser=True, needs="", claim="After LN (gamma=1, beta=0) every token vector has mean 0 and variance 1 (minus eps effect)",
code=r'''
import numpy as np
x = np.random.default_rng(0).standard_normal((5, 384)) * 3 + 2
y = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
print(np.abs(y.mean(-1)).max(), y.var(-1).round(6))
'''),
dict(id="F35", sec="2.9", browser=False, needs="torch", claim="LayerNorm is per token: a sample's output ignores the rest of the batch (BatchNorm does not)",
code=r'''
import torch, torch.nn as nn
torch.manual_seed(0); x = torch.randn(4, 384) * 3 + 1.5
ln, bn = nn.LayerNorm(384), nn.BatchNorm1d(384).train()
with torch.no_grad():
    other = torch.cat([x[:1], torch.randn(3, 384)])
    print("LN:", (ln(x)[0] - ln(other)[0]).abs().max().item(), "| BN:", round((bn(x)[0] - bn(other)[0]).abs().max().item(), 3))
'''),
dict(id="F36", sec="2.9", browser=True, needs="", claim="tanh-GELU vs exact (erf) GELU differ by at most 4.73e-4; GELU min = -0.16997 at x = -0.752",
code=r'''
import math, numpy as np
t = np.linspace(-6, 6, 120001)
exact = 0.5 * t * (1 + np.vectorize(math.erf)(t / math.sqrt(2)))
tanh = 0.5 * t * (1 + np.tanh(math.sqrt(2 / math.pi) * (t + 0.044715 * t ** 3)))
i = exact.argmin()
print(f"max|tanh-erf|={np.abs(tanh - exact).max():.3e}  min={exact[i]:.5f} at x={t[i]:.3f}  GELU(0)={exact[60000]:.1f}")
'''),
dict(id="F37", sec="2.9", browser=False, needs="torch", claim="Book GELU == nn.GELU(approximate='tanh'); codebot's FFN uses nn.GELU() which is the erf version",
code=r'''
import torch, torch.nn as nn
from bookload import load
import codebot.model as cb
t = torch.linspace(-6, 6, 120001); g = load("ch02/09_norm_gelu.py").GELU()(t)
print((g - nn.GELU(approximate="tanh")(t)).abs().max().item(), (g - nn.GELU()(t)).abs().max().item(), cb.FFN(8, 32, 0.1).layers[1])
'''),
dict(id="F38", sec="2.9", browser=False, needs="torch", claim="09 Block (gamma/beta) and codebot Block (weight/bias) state_dicts are not interchangeable without renaming",
code=r'''
from bookload import load
import codebot.model as cb
b09, bcb = load("ch02/09_norm_gelu.py").Block(384, 6), cb.Block(384, 6, 1536)
print([k for k in b09.state_dict() if "norm1" in k], [k for k in bcb.state_dict() if "norm1" in k])
try: b09.load_state_dict(bcb.state_dict())
except RuntimeError as e: print("load_state_dict fails:", " ".join(str(e).split())[:90])
'''),
dict(id="F39", sec="2.9", browser=False, needs="torch", claim="Residual stream: zeroing W_o and the FFN output layer turns a Block into the exact identity",
code=r'''
import torch
import codebot.model as cb
torch.manual_seed(0); blk = cb.Block(384, 6, 1536).eval(); x = torch.randn(2, 16, 384)
with torch.no_grad():
    blk.attn.W_o.weight.zero_(); blk.ffn.layers[2].weight.zero_(); blk.ffn.layers[2].bias.zero_()
    print((blk(x) - x).abs().max().item())
'''),
dict(id="F40", sec="2.9", browser=False, needs="torch", claim="One Block (E=384, H=6, ff=4E) has 1,772,928 params = 2 LN + 4E^2 + FFN (about 12E^2 = 1,769,472)",
code=r'''
import codebot.model as cb
E = 384; n = sum(p.numel() for p in cb.Block(E, 6, 4 * E).parameters())
print(n, 2 * 2 * E + 4 * E * E + (E * 4 * E + 4 * E) + (4 * E * E + E), 12 * E * E)
'''),
# ---------------- 2.10 GPT-2 ----------------
dict(id="F41", sec="2.10", browser=True, needs="", claim="Closed-form param count of the ch02/ch03 GPT: 11,121,640 (untied would be 11,505,640)",
code=r'''
V, CTX, E, L, FF = 1000, 256, 384, 6, 1536
block = 2 * 2 * E + 4 * E * E + (E * FF + FF) + (FF * E + E)
print(f"{V*E + V + CTX*E + L*block + 2*E:,}  untied: {2*V*E + V + CTX*E + L*block + 2*E:,}")
'''),
dict(id="F42", sec="2.10", browser=False, needs="torch", claim="numel over parameters() = 11,121,640 (tied matrix counted once); summing state_dict double-counts it; embed.weight IS unembed.weight",
code=r'''
from codebot.model import GPT
m = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
print(f"{sum(p.numel() for p in m.parameters()):,}", f"{sum(t.numel() for t in m.state_dict().values()):,}", m.embed.weight is m.unembed.weight)
'''),
dict(id="F43", sec="2.10", browser=False, needs="torch", claim="A saved checkpoint is ~44.5 MB ~ 4 bytes x 11.12M params, so the tied matrix is stored once (the official model_pretrain.pt is 44,516,353 bytes by HTTP HEAD)",
code=r'''
import os, tempfile
from codebot.model import GPT
p = os.path.join(tempfile.mkdtemp(), "m.pt"); GPT(1000, 256, 384, 6, 6, 1536, 0.1).save(p)
print(os.path.getsize(p), 11_121_640 * 4); os.remove(p)
'''),
dict(id="F44", sec="2.10", browser=False, needs="torch", claim="Init: Linear/Embedding ~ N(0, 0.02), biases 0, LayerNorm 1/0; on the tied tensor the Linear init runs last and wins",
code=r'''
import torch.nn as nn
from codebot.model import GPT
class Probe(GPT):
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.constant_(m.weight, 2.0)
        elif isinstance(m, nn.Embedding): nn.init.constant_(m.weight, 1.0)
print("tied value:", Probe(50, 8, 16, 2, 1, 64, 0.0).embed.weight[0, 0].item())
m = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
print(round(m.blocks[0].attn.W_q.weight.std().item(), 4), m.blocks[0].norm1.weight.mean().item(), m.unembed.bias.abs().max().item())
'''),
dict(id="F45", sec="2.10", browser=False, needs="torch", claim="Loss at init is ~6.98 = ln V + sigma^2/2 (sigma = logit std ~0.39), i.e. slightly above ln(1000) = 6.9078",
code=r'''
import math, torch, torch.nn.functional as F
from codebot.model import GPT
torch.manual_seed(0); m = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval(); ids = torch.randint(0, 1000, (8, 257))
with torch.no_grad(): lg = m(ids[:, :-1])
ce = F.cross_entropy(lg.reshape(-1, 1000), ids[:, 1:].reshape(-1)).item(); s = lg.std().item()
print(f"CE={ce:.4f}  lnV={math.log(1000):.4f}  lnV+s^2/2={math.log(1000) + s * s / 2:.4f}  s={s:.3f}")
'''),
dict(id="F46", sec="2.10", browser=False, needs="torch", claim="Whole-model causality: changing token 100 leaves logits 0..99 bit-identical",
code=r'''
import torch
from codebot.model import GPT
torch.manual_seed(0); m = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval()
a = torch.randint(0, 1000, (1, 256)); b = a.clone(); b[0, 100] = (b[0, 100] + 1) % 1000
with torch.no_grad(): d = (m(a) - m(b)).abs()
print(d[:, :100].max().item(), round(d[:, 100:].max().item(), 3))
'''),
dict(id="F47", sec="2.10", browser=False, needs="torch", claim="eval() makes the model deterministic and batch-independent; train() (dropout) does not",
code=r'''
import torch
from codebot.model import GPT
torch.manual_seed(0); m = GPT(1000, 256, 384, 6, 6, 1536, 0.1); ids = torch.randint(0, 1000, (4, 64))
print("train equal:", torch.equal(m(ids), m(ids)), end=" | ")
m.eval()
with torch.no_grad(): print("eval equal:", torch.equal(m(ids), m(ids)), "| batch vs alone:", (m(ids)[0] - m(ids[:1])[0]).abs().max().item())
'''),
dict(id="F48", sec="2.10", browser=False, needs="transformers", claim="HF GPT-2 small has 124,439,808 params; codebot GPT at the same hyperparameters has 124,453,201 (+50,257 lm_head bias, -36,864 qkv/proj biases)",
code=r'''
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from codebot.model import GPT
with torch.device("meta"):
    hf, cb = GPT2LMHeadModel(GPT2Config()), GPT(50257, 1024, 768, 12, 12, 3072, 0.1)
n = lambda m: sum(p.numel() for p in m.parameters())
print(f"{n(hf):,} {n(cb):,} {n(cb) - n(hf):+,} = {50257 - 12 * 4 * 768:+,}")
'''),
dict(id="F49", sec="2.10", browser=False, needs="transformers", claim="With HF GPT-2 weights copied in (qkv/proj biases zeroed), codebot GPT logits match HF to 3.5e-4 (erf GELU) and 1.9e-6 after switching to tanh-GELU",
code=r'''
import runpy, contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf): runpy.run_path("verify/v_hf_gpt2.py")
print("\n".join(l for l in buf.getvalue().splitlines() if "max|logits diff|" in l))
'''),
dict(id="F50", sec="2.10", browser=False, needs="ckpt", claim="Trained codebot: exact token ids of a snippet, CE 0.40 on 32 training windows (fresh init ~6.94-6.98), greedy sample from 'def'",
code=r'''
import os, numpy as np, torch, torch.nn.functional as F
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate
from bookload import checkpoint, REPO
tok = BPETokenizer.load_from(os.path.join(REPO, "codebot", "merge_rules.pkl")); m = GPT.load_from(checkpoint()).eval()
print(tok.encode("def add(a, b):\n    return a + b\n"))
data = np.fromfile(os.path.join(REPO, "codebot", "tiny_codes.bin"), dtype=np.uint16)
st = np.random.default_rng(0).integers(0, len(data) - 257, size=32)
x = torch.tensor(np.stack([data[s:s + 257] for s in st]).astype(np.int64))
with torch.no_grad(): print("CE on 32 train windows:", round(F.cross_entropy(m(x[:, :-1]).reshape(-1, 1000), x[:, 1:].reshape(-1)).item(), 4))
print(repr(generate(m, tok, "def", max_new_tokens=24, temperature=0)))
'''),
dict(id="F51", sec="2.10", browser=True, needs="", claim="Trained heads (from ch02_attn_maps.json): L4H3 is a previous-token head (' b' -> ' +' = 0.99); layer 1 puts ~half its weight on the first token (attention sink)",
code=r'''
import json, numpy as np
A = np.array(json.load(open("ch02_attn_maps.json", encoding="utf-8"))["attn"])     # (layer, head, query, key)
prev = np.mean([A[:, :, t, t - 1] for t in range(1, A.shape[-1])], axis=0)
print("best prev-token head (layer, head):", tuple(int(i) for i in np.unravel_index(prev.argmax(), prev.shape)), round(float(prev.max()), 2),
      "| L4H3 ' b'->' +':", A[4, 3, 11, 10], "| layer-1 mean weight on token 0:", round(float(A[1, :, 1:, 0].mean()), 2))
'''),
dict(id="F52", sec="2.10", browser=True, needs="", claim="A NumPy-only GPT forward (np_gpt.py) reproduces torch logits of a tiny codebot GPT to ~1e-7",
code=r'''
import numpy as np
from np_gpt import gpt_forward
Z = np.load("pyodide_test/tiny_ref.npz"); P = {k: Z[k].astype(np.float64) for k in Z.files if k not in ("ids", "ref_logits")}
out = gpt_forward([int(i) for i in Z["ids"]], P, n_layer=2, n_head=4)
print(out.shape, np.abs(out - Z["ref_logits"]).max())
'''),
dict(id="F53", sec="2.10", browser=False, needs="torch", claim="End-to-end sanity: the 11.1M GPT overfits one fixed batch of real TinyCodes tokens (7.0 -> ~0.02 in 50 steps, CPU)",
code=r'''
import numpy as np, torch, torch.nn.functional as F, os
from codebot.model import GPT
from bookload import REPO
torch.manual_seed(0)
data = np.fromfile(os.path.join(REPO, "codebot", "tiny_codes.bin"), dtype=np.uint16)
st = np.random.default_rng(0).integers(0, len(data) - 65, size=4)
x = torch.tensor(np.stack([data[s:s + 65] for s in st]).astype(np.int64))
m = GPT(1000, 256, 384, 6, 6, 1536, 0.0); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
for step in range(51):
    loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 1000), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0: print(step, round(loss.item(), 4))
'''),
]
