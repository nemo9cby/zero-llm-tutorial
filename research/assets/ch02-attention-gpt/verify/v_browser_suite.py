# NumPy + math only. Runs under CPython (numpy 1.x/2.x) and Pyodide. Each check prints PASS/FAIL with the observed value.
import math, os, time
import numpy as np
from np_gpt import softmax, layer_norm, gelu, causal_attention, gpt_forward

results = []
def check(name, ok, observed):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {observed}")

rng = np.random.default_rng(0)

# B1 soft dictionary
K = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], float)
Vv = np.array([85,70,60,75,80,65,90], float)
w = softmax(K @ np.array([6.,4.,5.]))
check("B1 soft-dict prediction", round(float(w @ Vv), 2) == 78.66, f"{w @ Vv:.4f}, weights sum {w.sum():.6f}")

# B2 Var(q.k) = d ; scaled -> 1
d = 64
q, k = rng.standard_normal((20000, d)), rng.standard_normal((20000, d))
dots = (q * k).sum(1)
check("B2 var(q.k)~d, var(q.k/sqrt d)~1", abs(dots.var() / d - 1) < 0.05, f"{dots.var():.2f} / {(dots / math.sqrt(d)).var():.4f}")

# B3 causal attention: weights + causality
C, E, H = 6, 16, 4
Wq, Wk, Wv, Wo = (rng.standard_normal((E, E)) * 0.3 for _ in range(4))
x = rng.standard_normal((C, E))
x2 = x.copy(); x2[4:] = rng.standard_normal((2, E))
dy = np.abs(causal_attention(x, Wq, Wk, Wv, Wo, H) - causal_attention(x2, Wq, Wk, Wv, Wo, H))
check("B3 causality (change tokens 4,5)", dy[:4].max() == 0 and dy[4:].max() > 0, f"max|dy| pos0-3={dy[:4].max():.1e}, pos4-5={dy[4:].max():.3f}")

# B4 wrong softmax axis leaks the future
def attn_axis(x, axis):
    s = (x @ Wq.T) @ (x @ Wk.T).T / math.sqrt(E)
    s = np.where(np.tril(np.ones((C, C), bool)), s, -np.inf)
    return softmax(s, axis=axis) @ (x @ Wv.T)
leak = np.abs(attn_axis(x, -2) - attn_axis(x2, -2))[:4].max()
check("B4 softmax over axis=-2 leaks future", leak > 1e-3, f"max|dy| pos0-3 = {leak:.3f}")

# B5 multi-head: wrong split (reshape to (H,C,D) without transpose) leaks future
def mha_wrong(x):
    D = E // H
    qh, kh, vh = ((x @ W.T).reshape(H, C, D) for W in (Wq, Wk, Wv))     # WRONG split
    s = qh @ kh.transpose(0, 2, 1) / math.sqrt(D)
    s = np.where(np.tril(np.ones((C, C), bool)), s, -np.inf)
    return (softmax(s) @ vh).transpose(1, 0, 2).reshape(C, E) @ Wo.T
leak2 = np.abs(mha_wrong(x) - mha_wrong(x2))[:4].max()
check("B5 wrong head split leaks future", leak2 > 1e-3, f"max|dy| pos0-3 = {leak2:.3f}")

# B6 LayerNorm and GELU
y = layer_norm(rng.standard_normal((5, 384)) * 3 + 2, np.ones(384), np.zeros(384))
check("B6a LN row mean~0 var~1", np.abs(y.mean(-1)).max() < 1e-9 and np.abs(y.var(-1) - 1).max() < 1e-5,
      f"max|mean|={np.abs(y.mean(-1)).max():.1e}, var={y.var(-1).min():.6f}")
t = np.linspace(-6, 6, 120001)
gd = np.abs(gelu(t, exact=False) - gelu(t, exact=True))
gmin = gelu(t).min()
check("B6b GELU tanh vs erf max diff ~4.7e-4", 4.0e-4 < gd.max() < 5.5e-4, f"{gd.max():.3e} at x={t[gd.argmax()]:.3f}")
check("B6c GELU min ~ -0.170 at x ~ -0.752", abs(gmin + 0.16997) < 1e-4, f"{gmin:.5f} at x={t[gelu(t).argmin()]:.3f}")

# B7 param count of the ch03 codebot GPT, by formula only
Vc, CTX, Ec, L, FF = 1000, 256, 384, 6, 1536
blk = 4 * Ec + 4 * Ec * Ec + 2 * Ec * FF + FF + Ec
check("B7 codebot GPT param count", Vc * Ec + Vc + CTX * Ec + L * blk + 2 * Ec == 11_121_640, f"{Vc*Ec + Vc + CTX*Ec + L*blk + 2*Ec:,}")

# B8 NumPy GPT == torch reference logits (tiny random model), causality, loss at init
path = "tiny_ref.npz"
if os.path.exists(path):
    Z = np.load(path)
    P = {k: Z[k].astype(np.float64) for k in Z.files if k not in ("ids", "ref_logits")}
    ids = [int(i) for i in Z["ids"]]
    t0 = time.perf_counter()
    out = gpt_forward(ids, P, n_layer=2, n_head=4)
    check("B8a numpy GPT == torch logits", np.abs(out - Z["ref_logits"]).max() < 1e-5,
          f"max|diff|={np.abs(out - Z['ref_logits']).max():.2e} ({(time.perf_counter()-t0)*1000:.0f} ms)")
    ids2 = ids.copy(); ids2[8] = (ids2[8] + 1) % 50
    dd = np.abs(gpt_forward(ids2, P, 2, 4) - out)
    check("B8b whole-model causality (change token 8)", dd[:8].max() == 0 and dd[8:].max() > 0, f"pos0-7 {dd[:8].max():.1e}, pos8+ {dd[8:].max():.3f}")
    logp = out - out.max(-1, keepdims=True); logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    ce = -logp[np.arange(len(ids) - 1), ids[1:]].mean()
    check("B8c CE at init ~ ln(50)=3.912", abs(ce - math.log(50)) < 0.1, f"{ce:.4f}")
else:
    print("(B8 skipped: tiny_ref.npz not found)")

print(f"\n{sum(results)}/{len(results)} checks passed | numpy {np.__version__}")
