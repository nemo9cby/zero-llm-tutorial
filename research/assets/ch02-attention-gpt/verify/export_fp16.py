# Export the trained codebot checkpoint as fp16 .npz (22.3 MB, tied matrix stored once) + torch reference logits
# for ../pyodide_test/run_trained.mjs (in-browser NumPy timing of the real model).
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\chenf\Projects\_refs\dlfs6")
sys.path.insert(0, HERE)
from codebot.model import GPT
from bookload import checkpoint

m = GPT.load_from(checkpoint()).eval()
sd = {k: v.numpy() for k, v in m.state_dict().items() if k != "embed.weight"}   # tied: keep unembed.weight only
ids = np.array([304, 882, 40, 97, 44, 345, 285, 265, 307, 273, 347, 345, 10])   # "def add(a, b):\n    return a + b\n"
with torch.no_grad():
    ref = m(torch.tensor(ids[None]))[0].numpy()
out = os.path.join(os.path.dirname(HERE), "pyodide_test", "codebot_fp16.npz")
np.savez(out, ids=ids, ref_logits=ref, **{k: v.astype(np.float16) for k, v in sd.items()})
print("saved", out, os.path.getsize(out), "bytes")
