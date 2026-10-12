# NumPy-only (Pyodide-feasible): pass@k from real answer distributions + tabular GRPO toy initialised from the SFT model
import json, sys, math
import numpy as np
d = json.load(open(sys.argv[1]))
corr = np.array([int(p[0]) + int(p[2]) for p in d['prompts']])
P_sft, P_grpo = np.array(d['sft']), np.array(d['grpo'])
print('--- pass@k = mean_prompt[1-(1-p_correct)^k]')
for name, Pm in [('SFT ', P_sft), ('GRPO', P_grpo)]:
    pc = Pm[np.arange(81), corr]
    print(f'  {name}: ' + '  '.join(f'k={k}:{np.mean(1-(1-pc)**k):.3f}' for k in [1, 2, 4, 8, 16, 64, 256]))

def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def run(G=8, eps=0.2, n_upd=2, lr=30.0, iters=150, std_norm=False, seed=0, log_every=None):
    rng = np.random.default_rng(seed)
    th = np.log(np.clip(P_sft, 1e-8, None))
    order = rng.permutation(81); batches = [order[:32], order[32:64], order[64:]]   # cycle(DataLoader): same 3 batches forever
    hist = []
    for it in range(iters):
        b = batches[it % 3]; nb = len(b)
        pi_old = softmax(th[b])
        u = rng.random((nb, G))[..., None]
        a = (pi_old.cumsum(-1)[:, None, :] < u).sum(-1).clip(max=19)            # (nb, G) inverse-CDF sampling
        r = (a == corr[b][:, None]).astype(float)
        A = r - r.mean(1, keepdims=True)
        if std_norm: A = A / (r.std(1, ddof=1, keepdims=True) + 1e-4) if G > 1 else A * 0
        rows = np.arange(nb)[:, None]
        for _ in range(n_upd):
            pi = softmax(th[b])
            ratio = pi[rows, a] / pi_old[rows, a]
            dobj = np.where(((A > 0) & (ratio <= 1 + eps)) | ((A < 0) & (ratio >= 1 - eps)), A, 0.0)
            coef = dobj * ratio / (nb * G)                                         # d obj / d logit = coef * (onehot(a) - pi)
            grad = np.zeros_like(pi)
            np.add.at(grad, (np.repeat(np.arange(nb), G), a.ravel()), coef.ravel())
            grad -= coef.sum(1, keepdims=True) * pi
            th[b] += lr * grad
        P = softmax(th); pc = P[np.arange(81), corr]
        hist.append(dict(it=it, reward=r.mean(), zero_var=np.mean(r.std(1) == 0), greedy=np.mean(P.argmax(-1) == corr),
                         p_corr=pc.mean(), ent=np.mean(-(P * np.log(P + 1e-12)).sum(-1)), min_pc=pc.min()))
        if log_every and (it % log_every == 0 or it == iters - 1):
            h = hist[-1]; print(f"    it {it:3d} reward {h['reward']:.3f} zero_var {h['zero_var']:.2f} greedy {h['greedy']:.3f} meanP {h['p_corr']:.3f} minP {h['min_pc']:.3f} H {h['ent']:.3f}")
    return hist
print('--- tabular GRPO from the SFT distribution, G=8 (verbose)')
run(log_every=25)
print('--- ablations (final values after 150 iterations, mean of 5 seeds)')
for kw in [dict(G=1), dict(G=2), dict(G=4), dict(G=8), dict(G=16), dict(G=8, std_norm=True), dict(G=8, n_upd=1), dict(G=8, eps=0.05), dict(G=8, lr=0.0)]:
    fin = [run(seed=s, **kw)[-1] for s in range(5)]
    first = [next((h['it'] for h in run(seed=s, **kw) if h['greedy'] >= 0.95), None) for s in range(2)]
    print(f"  {str(kw):32s} greedy {np.mean([f['greedy'] for f in fin]):.3f} meanP {np.mean([f['p_corr'] for f in fin]):.3f} zero_var(last batch) {np.mean([f['zero_var'] for f in fin]):.2f} entropy {np.mean([f['ent'] for f in fin]):.3f} | iters to greedy>=95%: {first}")
