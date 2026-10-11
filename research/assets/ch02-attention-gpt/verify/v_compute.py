# Compute / memory measurements for training codebot GPT (the model defined in ch02, trained in ch03)
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\chenf\Projects\_refs\dlfs6")
from codebot.model import GPT

torch.manual_seed(0)
torch.set_num_threads(8)
V, CTX, E, H, L, FF = 1000, 256, 384, 6, 6, 1536


def saved_bytes(model, B, C):
    """Bytes of tensors autograd saves for backward in one forward pass (deduplicated by storage)."""
    seen = {}
    def pack(t):
        try:
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            seen[key] = t.untyped_storage().nbytes()
        except Exception:
            pass
        return t
    ids = torch.randint(0, V, (B, C))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, V), ids.view(-1))
    return sum(seen.values()), loss


model = GPT(V, CTX, E, H, L, FF, 0.1).train()
N = sum(p.numel() for p in model.parameters())
N_matmul = N - V * E - CTX * E - V   # drop embedding lookups (tied matrix still used once in unembed: add back)
N_matmul += V * E
print(f"N total = {N:,} | N used in matmuls per token ~ {N_matmul:,}")

print("\n== C1: activation memory saved for backward (train mode, fp32) ==")
for B, C in [(1, 256), (8, 256), (32, 256)]:
    nbytes, _ = saved_bytes(model, B, C)
    print(f"B={B:2d} C={C}: saved activations = {nbytes/2**20:8.1f} MiB  (per sample {nbytes/B/2**20:.1f} MiB)")
static = N * 4 * 4
print(f"static (params+grads+AdamW m,v) = {static/2**20:.0f} MiB")

print("\n== C2: one training step on this CPU (i7-11700F, 8 threads) ==")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
for B, C in [(32, 256), (8, 128)]:
    ids = torch.randint(0, V, (B, C + 1))
    x, y = ids[:, :-1], ids[:, 1:]
    times = []
    for i in range(4):
        t = time.perf_counter()
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        times.append(time.perf_counter() - t)
    step = sum(times[1:]) / 3
    print(f"B={B:2d} C={C}: {step:.3f} s/step (first step {times[0]:.3f}s) -> 20,000 steps ~ {20000*step/3600:.1f} h")

print("\n== C3: FLOP model for ch03's pretraining run ==")
tokens = 20000 * 32 * 256
per_tok_dense = 6 * N_matmul
per_tok_attn = 3 * L * 4 * CTX * E          # fwd QK^T + AV = 4*C*E per layer (full CxC as written), x3 for fwd+bwd
total = tokens * (per_tok_dense + per_tok_attn)
print(f"tokens D = {tokens:,} ({tokens/2677398:.1f} epochs of tiny_codes.bin)")
print(f"per token: 6N = {per_tok_dense/1e6:.1f} MFLOP + attention {per_tok_attn/1e6:.1f} MFLOP")
print(f"total = {total:.3e} FLOP (6ND alone = {tokens*per_tok_dense:.3e})")
for name, peak, mfu in [("RTX 3060 Ti fp32 (16.2 TF)", 16.2e12, 0.35), ("A100 bf16 (312 TF)", 312e12, 0.25),
                        ("H100 SXM bf16 (989 TF)", 989e12, 0.2), ("A100 fp32 no-TF32 (19.5 TF)", 19.5e12, 0.5)]:
    print(f"  {name:28s} @ {int(mfu*100)}% MFU -> {total/(peak*mfu)/60:6.1f} min")
