# Check np_gpt.gpt_forward == codebot.model.GPT (eval) on a random tiny model and on the trained checkpoint.
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\chenf\Projects\_refs\dlfs6")
from codebot.model import GPT
from np_gpt import gpt_forward

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)


def compare(model, ids, label):
    model.eval()
    P = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    with torch.no_grad():
        ref = model(torch.tensor([ids]))[0].double().numpy()
    t = time.perf_counter()
    out = gpt_forward(ids, P, model.n_layer, model.n_head, exact_gelu=True)
    dt = time.perf_counter() - t
    print(f"{label}: max|numpy - torch| = {np.abs(out - ref).max():.3e}  (numpy fwd {dt*1000:.0f} ms, C={len(ids)})")
    return P


tiny = GPT(vocab_size=50, max_context_len=16, embed_dim=32, n_head=4, n_layer=2, ff_dim=128, dropout_rate=0.1)
P = compare(tiny, list(np.random.default_rng(0).integers(0, 50, 12)), "tiny random GPT (V=50,E=32,H=4,L=2)")
print("tiny params:", sum(p.numel() for p in tiny.parameters()), "| npz keys:", len(P))

from bookload import checkpoint
trained = GPT.load_from(checkpoint())
ids = [304, 882, 40, 97, 44, 345, 285, 265, 307, 273, 347, 345, 10]        # "def add(a, b):\n    return a + b\n"
compare(trained, ids, "trained codebot (11.1M)")
compare(trained, list(np.random.default_rng(1).integers(0, 1000, 256)), "trained codebot, full context")
