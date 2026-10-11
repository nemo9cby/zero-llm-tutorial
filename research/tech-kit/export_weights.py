"""Export a DLFS6 CodeBot checkpoint to browser-friendly files:
  <out>/codebot.{f32,f16,q8}.bin + <out>/codebot.{f32,f16,q8}.json  (manifest: config + tensor table)
  <out>/tokenizer.json (ordered BPE merges), <out>/ref.json (PyTorch reference logits + greedy ids)
Usage: uv run --with regex python export_weights.py <repo_dir> <ckpt.pt> <out_dir>"""
import sys, json, pickle, importlib.util, numpy as np, torch
repo, ckpt_path, out = sys.argv[1], sys.argv[2], sys.argv[3]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
model_mod = load("codebot_model", f"{repo}/codebot/model.py")
tok_mod = load("codebot_tokenizer", f"{repo}/codebot/tokenizer.py")

ckpt = torch.load(ckpt_path, map_location="cpu")
cfg = {k: ckpt[k] for k in ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim")}
sd = {k: v.float().contiguous() for k, v in ckpt["model_state_dict"].items() if k != "unembed.weight"}  # tied with embed.weight

def write(fmt):
    table, chunks, off = [], [], 0
    for name, t in sd.items():
        a = t.numpy()
        entry = {"name": name, "shape": list(a.shape), "offset": off}
        if fmt == "f32":
            b = a.astype("<f4").tobytes()
        elif fmt == "f16":
            b = a.astype("<f2").tobytes()
        else:  # q8: symmetric per-output-row int8 for 2-D weights, fp16 for 1-D params
            if a.ndim == 2:
                scale = np.abs(a).max(axis=1, keepdims=True) / 127.0
                scale[scale == 0] = 1.0
                q = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
                entry["scales_offset"] = off + q.nbytes
                b = q.tobytes() + scale.astype("<f4").ravel().tobytes()
            else:
                b = a.astype("<f2").tobytes(); entry["dtype"] = "f16"
        entry.setdefault("dtype", {"f32": "f32", "f16": "f16", "q8": "q8"}[fmt])
        entry["nbytes"] = len(b)
        pad = (-len(b)) % 4  # keep every tensor 4-byte aligned for typed-array views
        chunks.append(b + b"\0" * pad); table.append(entry); off += len(b) + pad
    open(f"{out}/codebot.{fmt}.bin", "wb").write(b"".join(chunks))
    json.dump({"config": cfg, "tied_unembed": True, "tensors": table}, open(f"{out}/codebot.{fmt}.json", "w"), indent=0)
    print(fmt, f"{off/1e6:.2f} MB")
for fmt in ("f32", "f16", "q8"): write(fmt)

merges = pickle.load(open(f"{repo}/codebot/merge_rules.pkl", "rb"))
json.dump({"merges": [[a, b, n] for (a, b), n in sorted(merges.items(), key=lambda kv: kv[1])], "end_token": "<|endoftext|>",
           "end_token_id": 256 + len(merges)}, open(f"{out}/tokenizer.json", "w"))

# PyTorch reference: last-position logits for a prompt + 64 greedy tokens (no early stop) for JS parity checks.
tok = tok_mod.BPETokenizer(merges)
model = model_mod.GPT.load_from(ckpt_path, device="cpu").eval()
prompt = "def add(a, b):\n"
ids = tok.encode(prompt)
with torch.no_grad():
    logits = model(torch.tensor([ids]))[0, -1]
    seq = list(ids)
    for _ in range(64):
        nxt = int(model(torch.tensor([seq[-cfg['max_context_len']:]]))[0, -1].argmax())
        seq.append(nxt)
json.dump({"prompt": prompt, "prompt_ids": ids, "last_logits": logits.tolist(), "greedy_ids": seq[len(ids):],
           "greedy_text": tok.decode(seq[len(ids):])}, open(f"{out}/ref.json", "w"))
print("prompt ids", ids, "| greedy:", repr(tok.decode(seq[len(ids):])[:120]))
