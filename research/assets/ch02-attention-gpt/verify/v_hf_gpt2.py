# Is codebot.model.GPT really "GPT-2"? Copy a randomly initialised HF GPT-2 into it and compare logits.
# No download: GPT2LMHeadModel(GPT2Config(...)) is built from a config with random weights.
import sys
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, r"C:\Users\chenf\Projects\_refs\dlfs6")
from codebot.model import GPT

torch.manual_seed(0)
print("== X1: parameter count of HF GPT-2 small (default config) ==")
with torch.device("meta"):
    small = GPT2LMHeadModel(GPT2Config())
print(f"HF GPT-2 small: {sum(p.numel() for p in small.parameters()):,}")
with torch.device("meta"):
    cb_small = GPT(50257, 1024, 768, 12, 12, 3072, 0.1)
print(f"codebot GPT, same hyperparameters: {sum(p.numel() for p in cb_small.parameters()):,}")

print("\n== X2: weight-mapped equivalence at the ch02/ch03 size (V=1000, ctx=256, E=384, L=6, H=6) ==")
V, CTX, E, L, H = 1000, 256, 384, 6, 6
cfg = GPT2Config(vocab_size=V, n_positions=CTX, n_embd=E, n_layer=L, n_head=H,
                 resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
hf = GPT2LMHeadModel(cfg).eval()
cb = GPT(V, CTX, E, H, L, 4 * E, 0.0).eval()
print("HF activation_function =", cfg.activation_function, "| layer_norm_epsilon =", cfg.layer_norm_epsilon)

with torch.no_grad():
    t = hf.transformer
    cb.embed.weight.copy_(t.wte.weight)            # tied with cb.unembed.weight
    cb.unembed.bias.zero_()                         # GPT-2's lm_head has no bias
    cb.pos_embed.weight.copy_(t.wpe.weight)
    for b, h in zip(cb.blocks, t.h):
        h.attn.c_attn.bias.zero_(); h.attn.c_proj.bias.zero_()   # codebot has no qkv/proj biases
        b.norm1.weight.copy_(h.ln_1.weight); b.norm1.bias.copy_(h.ln_1.bias)
        Wq, Wk, Wv = h.attn.c_attn.weight.split(E, dim=1)        # Conv1D stores (in, out)
        b.attn.W_q.weight.copy_(Wq.T); b.attn.W_k.weight.copy_(Wk.T); b.attn.W_v.weight.copy_(Wv.T)
        b.attn.W_o.weight.copy_(h.attn.c_proj.weight.T)
        b.norm2.weight.copy_(h.ln_2.weight); b.norm2.bias.copy_(h.ln_2.bias)
        b.ffn.layers[0].weight.copy_(h.mlp.c_fc.weight.T); b.ffn.layers[0].bias.copy_(h.mlp.c_fc.bias)
        b.ffn.layers[2].weight.copy_(h.mlp.c_proj.weight.T); b.ffn.layers[2].bias.copy_(h.mlp.c_proj.bias)
    cb.norm.weight.copy_(t.ln_f.weight); cb.norm.bias.copy_(t.ln_f.bias)

    ids = torch.randint(0, V, (2, 128))
    ref = hf(ids).logits
    d_erf = (cb(ids) - ref).abs().max().item()
    for b in cb.blocks:
        b.ffn.layers[1] = nn.GELU(approximate="tanh")              # = HF "gelu_new" = book's 09 GELU
    d_tanh = (cb(ids) - ref).abs().max().item()
print(f"max|logits diff| with codebot's nn.GELU() (erf): {d_erf:.3e}")
print(f"max|logits diff| after switching to tanh-GELU  : {d_tanh:.3e}")
print("lm_head tied in HF:", hf.lm_head.weight is hf.transformer.wte.weight)
c_proj_std = t.h[0].attn.c_proj.weight.std().item()
print(f"HF init: c_attn std={t.h[0].attn.c_attn.weight.std().item():.4f}, attn.c_proj std={c_proj_std:.4f} "
      f"(0.02/sqrt(2L)={0.02/(2*L)**0.5:.4f} -> GPT-2 scales residual projections; codebot does not)")
