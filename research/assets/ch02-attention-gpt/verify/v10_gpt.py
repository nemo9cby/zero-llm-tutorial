# torch checks for ch02/10_gpt2.py and codebot.model.GPT
import math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from bookload import load

SCR = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
m10 = load("ch02/10_gpt2.py")
from codebot.model import GPT

V, CTX, E, H, L = 1000, 256, 384, 6, 6
FF = 4 * E
model = GPT(V, CTX, E, H, L, FF, 0.1)

print("== G1: parameter count ==")
total = sum(p.numel() for p in model.parameters())
blk = 2 * 2 * E + 4 * E * E + (E * FF + FF) + (FF * E + E)
formula = V * E + V + CTX * E + L * blk + 2 * E
print(f"sum(p.numel()) = {total:,}")
print(f"formula: tok_emb(tied) {V*E:,} + unembed.bias {V:,} + pos_emb {CTX*E:,} + {L}x block {blk:,} + final LN {2*E:,} = {formula:,}")
named = sum(p.numel() for _, p in model.named_parameters())
sd = sum(t.numel() for t in model.state_dict().values())
print(f"named_parameters (dedups shared) = {named:,} | summing state_dict tensors (double-counts tied) = {sd:,}")
print(f"untied would be {total + V*E:,} | non-embedding (blocks + final LN) = {L*blk + 2*E:,}")

print("\n== G2: weight tying is one Parameter object ==")
print("embed.weight is unembed.weight:", model.embed.weight is model.unembed.weight,
      "| shape", tuple(model.embed.weight.shape),
      "| same storage:", model.embed.weight.data_ptr() == model.unembed.weight.data_ptr())
ck = os.path.join(SCR, "tmp_gpt.pt")
model.save(ck)
size = os.path.getsize(ck)
print(f"checkpoint size = {size/1e6:.2f} MB vs total*4 bytes = {total*4/1e6:.2f} MB (shared storage saved once)")
re = GPT.load_from(ck)
print("after load_from, still tied:", re.embed.weight is re.unembed.weight)
os.remove(ck)

print("\n== G3: init statistics ==")
for name, p in [("embed/unembed (tied)", model.embed.weight), ("pos_embed", model.pos_embed.weight),
                ("blocks.0.attn.W_q", model.blocks[0].attn.W_q.weight),
                ("blocks.0.ffn.layers.0.weight", model.blocks[0].ffn.layers[0].weight),
                ("blocks.0.ffn.layers.0.bias", model.blocks[0].ffn.layers[0].bias),
                ("blocks.0.norm1.weight", model.blocks[0].norm1.weight),
                ("unembed.bias", model.unembed.bias)]:
    print(f"{name:30s} mean={p.mean().item():+.4f} std={p.std().item():.4f}")

print("\n== G4: apply() order -> the Linear init is the one that sticks on the tied tensor ==")
class Probe(GPT):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): nn.init.constant_(module.weight, 2.0)
        elif isinstance(module, nn.Embedding): nn.init.constant_(module.weight, 1.0)
p = Probe(50, 8, 16, 2, 1, 64, 0.0)
print("tied weight value after init =", p.embed.weight[0, 0].item(), "(1.0 = Embedding init, 2.0 = Linear init)")
print("children order:", [n for n, _ in p.named_children()])

print("\n== G5: loss at init vs ln(V) ==")
model.eval()
ids = torch.randint(0, V, (8, CTX + 1))
with torch.no_grad():
    logits = model(ids[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1)).item()
    s = logits.std().item()
print(f"CE at init = {loss:.4f} | ln(V) = {math.log(V):.4f} | ln(V)+sigma^2/2 = {math.log(V)+s*s/2:.4f} (logit std sigma={s:.3f})")
with torch.no_grad():
    cur = logits.gather(-1, ids[:, :-1, None]).squeeze(-1)          # logit of the CURRENT input token
    rank = (logits > cur[..., None]).sum(-1).float()
print(f"tied embeddings at init: mean logit of the current token = {cur.mean().item():.3f} vs all logits mean {logits.mean().item():.3f}; mean rank of current token = {rank.mean().item():.1f} / {V}")

print("\n== G6: causality of the whole model (eval mode) ==")
with torch.no_grad():
    a = ids[:1, :CTX].clone(); b = a.clone(); b[0, 100] = (b[0, 100] + 1) % V
    la, lb = model(a), model(b)
    print("max|dlogits| positions 0..99 =", (la - lb)[:, :100].abs().max().item(),
          "| positions 100.. =", (la - lb)[:, 100:].abs().max().item())

print("\n== G7: batch independence (eval) ==")
with torch.no_grad():
    print("max|model(batch)[0] - model(batch[:1])[0]| =", (model(ids[:4, :64])[0] - model(ids[:1, :64])[0]).abs().max().item())

print("\n== G8: longer than max_context_len -> IndexError from pos_embed ==")
try:
    model(torch.randint(0, V, (1, CTX + 1)))
except IndexError as e:
    print("IndexError:", e)

print("\n== G9: train() vs eval() ==")
x = ids[:1, :32]
model.train()
print("train: two forwards equal?", torch.equal(model(x), model(x)))
model.eval()
print("eval : two forwards equal?", torch.equal(model(x), model(x)))

print("\n== G10: the book's 10_gpt2.GPT and codebot.model.GPT are the same network ==")
g10 = m10.GPT(V, CTX, E, H, L, FF, 0.1).eval()
g10.load_state_dict(model.state_dict())
with torch.no_grad():
    print("max|10_gpt2 - codebot| =", (g10(ids[:2, :64]) - model(ids[:2, :64])).abs().max().item(),
          "(10_gpt2 imports codebot Block, so both use nn.GELU()/nn.LayerNorm)")

print("\n== G11: attention at init is nearly uniform (layer 0, head 0) ==")
with torch.no_grad():
    xin = model.blocks[0].norm1(model.embed(ids[:1, :CTX]) + model.pos_embed(torch.arange(CTX)))
    at = model.blocks[0].attn
    q = at.W_q(xin).view(1, CTX, H, E // H).transpose(1, 2)
    k = at.W_k(xin).view(1, CTX, H, E // H).transpose(1, 2)
    sc = (q @ k.transpose(-2, -1)) / (E // H) ** 0.5
    print(f"score std scaled={sc.std().item():.4f}  unscaled={(sc*(E//H)**0.5).std().item():.4f}")
    w = F.softmax(sc.masked_fill(torch.tril(torch.ones(CTX, CTX)) == 0, float('-inf')), -1)
    t = CTX - 1
    print(f"row {t}: max weight={w[0,0,t].max().item():.5f} vs uniform 1/{t+1}={1/(t+1):.5f}")

print("\n== G12: GPT-2-small hyperparameters in this codebase (meta device, no memory) ==")
with torch.device("meta"):
    g2 = GPT(50257, 1024, 768, 12, 12, 3072, 0.1)
n2 = sum(p.numel() for p in g2.parameters())
print(f"codebot GPT @ GPT-2 small config = {n2:,}; HF GPT-2 = 124,439,808; diff = {n2-124439808:+,} "
      f"(= +{50257} unembed bias - {12*4*768} missing qkv/proj biases = {50257-12*4*768:+,})")
