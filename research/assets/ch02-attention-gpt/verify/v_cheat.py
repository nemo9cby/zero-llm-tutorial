# 2.6.1/2.6.2: training on all C positions at once needs the mask; without it the model copies the answer.
# Data = i.i.d. uniform random tokens (fresh every step) -> nothing is learnable for an honest (causal) model.
# Prediction: causal CE -> ln V ; no-mask CE -> ln(V)/C (only the last position cannot peek at its target).
import math, time
import torch
import torch.nn.functional as F
import codebot.model as cb


class MHA(cb.MultiHeadAttention):
    causal = True

    def forward(self, x):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim
        Q, K, V = (l(x).view(B, C, H, D).transpose(1, 2) for l in (self.W_q, self.W_k, self.W_v))
        s = (Q @ K.transpose(-2, -1)) / D ** 0.5
        if self.causal:
            s = s.masked_fill(torch.tril(torch.ones(C, C)) == 0, float('-inf'))
        h = F.softmax(s, -1) @ V
        return self.W_o(h.transpose(1, 2).contiguous().view(B, C, H * D))


def run(causal, Vsz=8, C=8, E=32, H=2, L=1, steps=150, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    m = cb.GPT(Vsz, C, E, H, L, 4 * E, 0.0)
    for b in m.blocks:
        new = MHA(E, H, E // H, 0.0)
        new.load_state_dict(b.attn.state_dict())
        new.causal = causal
        b.attn = new
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        seq = torch.randint(0, Vsz, (64, C + 1))
        x, y = seq[:, :-1], seq[:, 1:]                       # "efficient format": C (input,target) pairs per row
        loss = F.cross_entropy(m(x).reshape(-1, Vsz), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        seq = torch.randint(0, Vsz, (512, C + 1))
        return F.cross_entropy(m(seq[:, :-1]).transpose(1, 2), seq[:, 1:], reduction='none').mean(0)


if __name__ == "__main__":
    t = time.perf_counter()
    Vsz, C = 8, 8
    for causal in (True, False):
        pp = run(causal, Vsz, C)
        print(f"causal={causal!s:5s}: mean CE = {pp.mean():.4f} | per position = {[round(v, 3) for v in pp.tolist()]}")
    print(f"ln V = {math.log(Vsz):.4f} | ln(V)/C = {math.log(Vsz)/C:.4f} | {time.perf_counter()-t:.1f}s")
