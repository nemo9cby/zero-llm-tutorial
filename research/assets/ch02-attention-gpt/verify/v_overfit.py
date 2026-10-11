# End-to-end sanity check: the ch02 GPT (ch03 size, 11.1M params) can overfit ONE fixed batch of real TinyCodes tokens.
import math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

REPO = r"C:\Users\chenf\Projects\_refs\dlfs6"
sys.path.insert(0, REPO)
from codebot.model import GPT

torch.manual_seed(0)
data = np.fromfile(os.path.join(REPO, "codebot", "tiny_codes.bin"), dtype=np.uint16)
B, C = 4, 64
starts = np.random.default_rng(0).integers(0, len(data) - C - 1, size=B)
x = torch.tensor(np.stack([data[s:s + C] for s in starts]).astype(np.int64))
y = torch.tensor(np.stack([data[s + 1:s + C + 1] for s in starts]).astype(np.int64))

model = GPT(1000, 256, 384, 6, 6, 1536, dropout_rate=0.0)      # dropout off so the batch can be memorised exactly
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
t = time.perf_counter()
for step in range(101):
    loss = F.cross_entropy(model(x).reshape(-1, 1000), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (0, 10, 25, 50, 100):
        print(f"step {step:3d}: loss {loss.item():.4f}")
print(f"ln(1000) = {math.log(1000):.4f} | {time.perf_counter() - t:.1f}s for 101 steps (B={B}, C={C}, CPU)")
