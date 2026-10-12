# NumPy-only facts (Pyodide-feasible). No torch import.
import re, math
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def entropy(p): return float(-(p * np.log(np.clip(p, 1e-300, None))).sum())

print('--- (1) temperature')
z = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
for T in [0.01, 0.5, 1.0, 2.0, 100.0]:
    p = softmax(z / T); print(f'T={T:<5} p={p} H={entropy(p):.4f} (ln5={math.log(5):.4f})')
print('argmax == T->0 limit:', np.argmax(z) == np.argmax(softmax(z / 1e-3)))

print('--- (2) sampling frequency matches probs (law of large numbers)')
p = softmax(z); s = rng.choice(len(p), size=200000, p=p)
print('empirical', np.bincount(s, minlength=5) / len(s), 'vs p', p)

print('--- (3) cross entropy of uniform logits == ln V; random small logits slightly above')
V = 1000
logits = np.zeros((8, V)); y = rng.integers(0, V, 8)
ce = lambda L, y: float(np.mean(np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1) - L[np.arange(len(y)), y]))
print(f'uniform: {ce(logits, y):.6f} vs ln(V) = {math.log(V):.6f}')
L = rng.normal(0, 0.392, (4096, V)); y = rng.integers(0, V, 4096)
print(f'N(0, 0.392^2) logits: {ce(L, y):.4f} vs ln V + sigma^2/2 = {math.log(V) + 0.392**2/2:.4f}')

print('--- (4) reward function (verbatim regex logic)')
def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            return 1.0 if int(matches[-1]) == ground_truth else 0.0
        return 0.0
    except:
        return 0.0
for resp in ['7', ' 7', 'The answer is 7', '3+4=7', '7 8', '8 7', '07', '7.0', '-7', 'seven', '', '7<|endoftext|>', '1 2']:
    print(f'  gt=7  {resp!r:22} -> {calculate_reward(7, resp)}')
print('  gt=12 "1 2" ->', calculate_reward(12, '1 2'), '| gt=12 "12" ->', calculate_reward(12, '12'))

print('--- (5) group advantages  A = r - mean(r)   (09_grpo.py: no std division)')
for r in [[1,0,0,0,0,0,0,0], [1,1,1,1,0,0,0,0], [1]*8, [0]*8, [1]]:
    r = np.array(r, float); A = r - r.mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        A_std0 = (r - r.mean()) / r.std()                 # population std, no eps -> NaN when all equal
    A_std1 = (r - r.mean()) / (r.std(ddof=1) + 1e-4) if len(r) > 1 else np.full_like(r, np.nan)  # torch.std default (unbiased) + eps
    print(f'  r={r.astype(int).tolist()}: A={A} sum={A.sum():.1e} | /std(ddof0, no eps)={A_std0} | /(std(ddof1)+1e-4)={A_std1}')
print('  binary rewards, k of G correct: A_correct = 1-k/G (no std) vs sqrt((1-p)/p) (std-normalized, ddof0)')
G = 8
for k in range(1, 8):
    p = k / G; print(f'   k={k}: no-std A+={1-p:.3f} A-={-p:.3f} | std-norm A+={math.sqrt((1-p)/p):.3f} A-={-math.sqrt(p/(1-p)):.3f}')

print('--- (6) P(zero-variance group) = p^G + (1-p)^G  (no learning signal)')
for p in [0.05, 0.2, 0.44, 0.8, 0.95]:
    print('  p=%.2f ' % p + ' '.join(f'G={G}:{p**G + (1-p)**G:.3f}' for G in [1, 2, 4, 8, 16]))

print('--- (7) PPO-clip surrogate min(r*A, clip(r,1-e,1+e)*A): d/dr  (e=0.2)')
def surr(r, A, e=0.2): return min(r * A, min(max(r, 1 - e), 1 + e) * A)
for A in [+1.0, -1.0]:
    for r in [0.5, 0.9, 1.0, 1.1, 1.5]:
        g = (surr(r + 1e-6, A) - surr(r - 1e-6, A)) / 2e-6
        print(f'  A={A:+.0f} ratio={r:<4} objective={surr(r, A):+.3f} d/dratio={g:+.3f}')

print('--- (8) d ratio / d theta at theta_old == d log pi / d theta (softmax toy, finite differences)')
th = rng.normal(size=5); a = 2; old = softmax(th)[a]
num_ratio = np.array([(softmax(th + 1e-6*np.eye(5)[k])[a] / old - softmax(th - 1e-6*np.eye(5)[k])[a] / old) / 2e-6 for k in range(5)])
num_logp = np.array([(np.log(softmax(th + 1e-6*np.eye(5)[k])[a]) - np.log(softmax(th - 1e-6*np.eye(5)[k])[a])) / 2e-6 for k in range(5)])
print('  grad ratio', num_ratio, '\n  grad logpi', num_logp, '\n  analytic onehot-p', np.eye(5)[a] - softmax(th))

print('--- (9) prob-space ratio bias: p/(p+1e-8) when p_new == p_old')
for p in [1e-2, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9]:
    print(f'  p={p:.0e} ratio={p/(p+1e-8):.6f}')
