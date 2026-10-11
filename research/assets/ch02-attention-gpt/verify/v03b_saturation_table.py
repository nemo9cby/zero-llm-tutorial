# 2.3 supplement: softmax saturation and Jacobian norm vs d, unscaled vs scaled (NumPy only; ~10 s on CPython).
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def jacobian_fro(P):
    """Row-wise ||diag(p) - p p^T||_F, vectorised; clipped at 0 against round-off."""
    s2, s4 = (P ** 2).sum(1), (P ** 4).sum(1)
    return np.sqrt(np.maximum((P ** 2 * (1 - P) ** 2).sum(1) + s2 ** 2 - s4, 0))


n, trials = 256, 1000
print(" d    | unscaled: maxw  sat>0.99  ||J||_F | scaled: maxw  sat>0.99  ||J||_F")
for d in [4, 16, 64, 256, 1024, 4096]:
    q = rng.standard_normal((trials, d))
    K = rng.standard_normal((trials, n, d))
    raw = np.einsum("td,tnd->tn", q, K)
    cols = []
    for s in (raw, raw / np.sqrt(d)):
        P = softmax(s)
        cols.append((P.max(1).mean(), (P.max(1) > 0.99).mean(), jacobian_fro(P).mean()))
    (a, b, c), (e, f, g) = cols
    print(f"{d:5d} |        {a:.3f}  {b:6.3f}   {c:.4f}  |        {e:.3f}  {f:6.3f}   {g:.4f}")
