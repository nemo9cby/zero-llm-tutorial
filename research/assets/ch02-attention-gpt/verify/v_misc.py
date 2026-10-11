# Small traps: non-divisible head_dim runs silently; corpus facts behind the trained model.
import os, sys
import torch

REPO = r"C:\Users\chenf\Projects\_refs\dlfs6"
sys.path.insert(0, REPO)
import codebot.model as cb

torch.manual_seed(0)
b = cb.Block(100, 3, 400).eval()           # 100 // 3 = 33 -> H*D = 99 != 100, no assertion anywhere
print("E=100,H=3: head_dim =", b.attn.head_dim, "| W_q", tuple(b.attn.W_q.weight.shape), "| out", tuple(b(torch.randn(1, 4, 100)).shape))

with open(os.path.join(REPO, "codebot", "tiny_codes.txt"), encoding="utf-8") as fh:
    t = fh.read()
greedy = "def sort_list(lst):\n    for i in range(len(lst) - 1):\n        for j in range(i + 1, len(lst)):\n            if lst[i] < lst[j]:\n"
print("corpus chars:", len(t), "| documents:", t.count("<|endoftext|>"), "| greedy 4-line prefix occurs verbatim:", t.count(greedy))
print("tokens in tiny_codes.bin:", os.path.getsize(os.path.join(REPO, "codebot", "tiny_codes.bin")) // 2,
      "| ch03 pretraining epochs = 20000*32*256 / tokens =", round(20000 * 32 * 256 / (os.path.getsize(os.path.join(REPO, "codebot", "tiny_codes.bin")) // 2), 1))
