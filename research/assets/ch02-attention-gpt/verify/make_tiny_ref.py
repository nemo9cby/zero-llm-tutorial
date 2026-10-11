# Build tiny_ref.npz: weights of a tiny random codebot GPT + ids + torch reference logits (for the Pyodide check).
import os, sys
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\chenf\Projects\_refs\dlfs6")
from codebot.model import GPT

SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0)
m = GPT(vocab_size=50, max_context_len=16, embed_dim=32, n_head=4, n_layer=2, ff_dim=128, dropout_rate=0.1).eval()
ids = np.array([3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9, 32], dtype=np.int64)
with torch.no_grad():
    ref = m(torch.tensor(ids[None]))[0].numpy()
arrs = {k: v.numpy() for k, v in m.state_dict().items()}
arrs.update(ids=ids, ref_logits=ref)
np.savez(os.path.join(SCR, "pyodide_test", "tiny_ref.npz"), **arrs)
print("saved", len(arrs), "arrays; ref_logits shape", ref.shape)
