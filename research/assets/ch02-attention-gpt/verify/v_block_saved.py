# What autograd keeps for backward in ONE codebot Block (train mode, B=1, C=256, fp32, CPU).
import collections, sys
import torch

sys.path.insert(0, r"C:\Users\chenf\Projects\_refs\dlfs6")
import codebot.model as cb

torch.manual_seed(0)
blk = cb.Block(384, 6, 1536, 0.1).train()
x = torch.randn(1, 256, 384, requires_grad=True)
seen = collections.OrderedDict()


def pack(t):
    s = t.untyped_storage()
    seen.setdefault((s.data_ptr(), s.nbytes()), (tuple(t.shape), str(t.dtype).replace("torch.", ""), s.nbytes()))
    return t


with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    blk(x)
params = {p.untyped_storage().data_ptr() for p in blk.parameters()}
total = 0
for (ptr, _), (shape, dt, nb) in seen.items():
    is_param = ptr in params
    total += 0 if is_param else nb
    print(f"{str(shape):22s} {dt:8s} {nb / 2**20:7.3f} MiB {'PARAM' if is_param else ''}")
C, E, F, H = 256, 384, 1536, 6
print(f"activations (excl. params) per block per sample: {total / 2**20:.2f} MiB")
print(f"analytic lower bound 8CE+2CF+2HC^2 floats = {(8*C*E + 2*C*F + 2*H*C*C) * 4 / 2**20:.2f} MiB (+ masks)")
