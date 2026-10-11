# NumPy-only checks for ch02/03_attn_scaling.py (Pyodide-feasible)
import numpy as np
np.set_printoptions(precision=4, suppress=False)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

print("== S1: softmax([100,200,300]) saturates (float64 vs float32) ==")
x = np.array([100.0, 200.0, 300.0])
print("float64:", softmax(x))
print("float32:", softmax(x.astype(np.float32)))
print("divide by 100 first:", softmax(x / 100).round(4))

print("\n== S2: Var(q.k) = d for q,k ~ N(0, I_d); scaling by 1/sqrt(d) -> 1 ==")
rng = np.random.default_rng(0)
for d in [10, 64, 512]:
    q = rng.standard_normal((10000, d))
    k = rng.standard_normal((10000, d))
    dots = (q * k).sum(axis=1)
    print(f"d={d:4d}: var(dot)={dots.var():8.3f}  var(dot/sqrt(d))={(dots/np.sqrt(d)).var():.4f}  ratio={dots.var()/(dots/np.sqrt(d)).var():.4f}")

print("\n== S3: what it does to an attention row (d=64, 256 keys, 2000 trials) ==")
d, n, trials = 64, 256, 2000
q = rng.standard_normal((trials, 1, d))
K = rng.standard_normal((trials, n, d))
raw = (q @ K.transpose(0, 2, 1))[:, 0, :]          # (trials, n)
for name, s in [("unscaled", raw), ("scaled", raw / np.sqrt(d))]:
    p = softmax(s)
    ent = -(p * np.log(p + 1e-30)).sum(axis=1)
    print(f"{name:8s}: mean max-weight={p.max(axis=1).mean():.3f}  mean entropy={ent.mean():.3f} nats (uniform = ln {n} = {np.log(n):.3f})")

print("\n== S4: ...and to its gradient: softmax Jacobian J = diag(p) - p p^T ==")
def jac_norm(s):
    p = softmax(s)
    J = np.diag(p) - np.outer(p, p)
    return np.linalg.norm(J)
jr = np.mean([jac_norm(raw[t]) for t in range(200)])
js = np.mean([jac_norm(raw[t] / np.sqrt(d)) for t in range(200)])
print(f"mean ||J||_F unscaled={jr:.4f}  scaled={js:.4f}  (one-hot p gives exactly 0)")
p = softmax(np.array([0.0, 100.0, 0.0]))
print("one-hot-ish p:", p, " ||J||_F =", np.linalg.norm(np.diag(p) - np.outer(p, p)))
