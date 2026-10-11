# NumPy-only checks for ch02/01_soft_dict.py and 02_attn_math.py (Pyodide-feasible)
import numpy as np
np.set_printoptions(precision=4, suppress=True)

keys = np.array([[8,2,3],[3,9,1],[1,2,9],[5,5,5],[7,6,2],[2,7,6],[9,1,1]], dtype=float)
vals = np.array([85,70,60,75,80,65,90], dtype=float)
q = np.array([6,4,5], dtype=float)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # stable: subtract max
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

print("== F1: soft dictionary reproduces 78.66 ==")
s = keys @ q
w = softmax(s)
print("scores =", s)
print("weights(%) =", (w*100).round(2), " sum =", w.sum())
print("prediction =", round(float(w @ vals), 2))

print("\n== F2: hard dict is the T->0 limit of the soft dict (query = exact key) ==")
for T in [10.0, 1.0, 0.1, 0.01]:
    wt = softmax((keys @ keys[4]) / T)
    print(f"T={T:<5} query=key[4]=(7,6,2): weight on key[4]={wt[4]:.4f} -> output {wt @ vals:.2f}")

print("\n== F3: dot product != 'closest' (magnitude matters) ==")
cos = (keys @ q) / (np.linalg.norm(keys, axis=1) * np.linalg.norm(q))
dist = np.linalg.norm(keys - q, axis=1)
print("argmax dot      ->", keys[np.argmax(s)])
print("argmax cosine   ->", keys[np.argmax(cos)], " cos =", cos.round(4))
print("argmin distance ->", keys[np.argmin(dist)], " dist =", dist.round(3))

print("\n== F4: naive exp overflows (the book's 01 has no max-subtraction) ==")
with np.errstate(over='ignore', invalid='ignore'):
    big = keys @ (q * 10)          # features on a 0..100 scale -> scores up to ~800
    naive = np.exp(big) / np.exp(big).sum()
print("max score =", big.max(), "  naive weights:", naive)
print("stable    weights:", softmax(big).round(4))

print("\n== F5: matrix form (02) == loop form (01) ==")
Q = np.array([[6,4,5],[2,8,3],[4,3,7]], dtype=float)
W = softmax(Q @ keys.T, axis=1)
out = W @ vals[:, None]
print("Q K^T shape", (Q @ keys.T).shape, " W shape", W.shape, " out shape", out.shape)
print("row sums =", W.sum(axis=1))
print("predictions =", out.ravel().round(2))

print("\n== F6: softmax is shift-invariant, NOT scale-invariant ==")
x = np.array([1.0, 2.0, 3.0])
print("softmax(x)      =", softmax(x).round(4))
print("softmax(x+100)  =", softmax(x+100).round(4))
print("softmax(100*x)  =", softmax(100*x).round(4))
