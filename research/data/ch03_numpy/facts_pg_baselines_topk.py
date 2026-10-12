# NumPy-only (Pyodide-feasible)
import itertools, math
import numpy as np
np.set_printoptions(precision=5, suppress=True)
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

print('--- (A) "policy loss ~ 0" does not mean "gradient ~ 0"')
th = np.array([0.5, 0.0, -0.5]); pi = softmax(th)
a, r = np.array([0, 1]), np.array([1.0, 0.0])          # one correct, one wrong sample
A = r - r.mean()                                        # [+0.5, -0.5]
ratio = pi[a] / pi[a]                                   # first update: pi == pi_old
loss = -(ratio * A).sum() / len(a)
grad_logits = -sum(A[k] * ratio[k] * (np.eye(3)[a[k]] - pi) for k in range(2)) / len(a)   # d loss / d theta
print(f'  A={A}, loss={loss:+.4f}, dloss/dtheta={grad_logits}  (non-zero!)')
th2 = th - 1.0 * grad_logits; pi2 = softmax(th2)
print(f'  after one SGD step (lr=1): pi {pi} -> {pi2}; E[reward] {pi[0]:.4f} -> {pi2[0]:.4f}')
print(f'  the loss value re-evaluated on the same samples is now {-(pi2[a]/pi[a]*A).sum()/2:+.4f} (tiny), yet reward went up')

print('--- (B) baseline unbiasedness (exact): sum_a pi(a) * b * dlogpi(a)/dtheta == 0')
th = np.array([0.3, -0.2, 1.0]); pi = softmax(th); rew = np.array([1.0, 0.0, 0.5])
score = np.eye(3) - pi                                 # row a: d log pi(a) / d theta
true_grad = (pi[:, None] * rew[:, None] * score).sum(0)
print('  E[b*score] for b=0.7:', (pi[:, None] * 0.7 * score).sum(0), '| true grad of E[r]:', true_grad)

print('--- (C) exact expectation of the GRPO (group-mean) estimator vs RLOO, by enumerating all G-tuples')
for G in [1, 2, 3, 4]:
    g_grpo = np.zeros(3); g_rloo = np.zeros(3)
    for tup in itertools.product(range(3), repeat=G):
        p = np.prod(pi[list(tup)]); rr = rew[list(tup)]
        A = rr - rr.mean()
        g_grpo += p * sum(A[i] * score[tup[i]] for i in range(G)) / G
        if G > 1:
            A_loo = np.array([rr[i] - (rr.sum() - rr[i]) / (G - 1) for i in range(G)])
            g_rloo += p * sum(A_loo[i] * score[tup[i]] for i in range(G)) / G
    print(f'  G={G}: E[grpo grad]/true = {np.round(g_grpo / true_grad, 6)}  ((G-1)/G = {(G-1)/G:.4f}) | E[rloo grad]/true = {np.round(g_rloo / true_grad, 6) if G > 1 else "n/a"}')

print('--- (D) top-k / top-p (nucleus) filtering, then renormalise')
z = np.array([2.0, 1.0, 0.5, 0.0, -1.0]); p = softmax(z)
def top_k(p, k):
    q = np.where(p >= np.sort(p)[-k], p, 0.0); return q / q.sum()
def top_p(p, thr):
    idx = np.argsort(-p); c = np.cumsum(p[idx]); keep = idx[: int(np.searchsorted(c, thr) + 1)]
    q = np.zeros_like(p); q[keep] = p[keep]; return q / q.sum()
print('  p        ', p)
print('  top-k=1  ', top_k(p, 1), '(== greedy)')
print('  top-k=2  ', top_k(p, 2))
print('  top-p=0.7', top_p(p, 0.7), '| top-p=0.9', top_p(p, 0.9), '| top-p=1.0', top_p(p, 1.0))
