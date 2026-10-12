"""DLFS6 → CS336 Assignment 1 adapter mapping (an evaluation harness, not a CS336 solution).

CS336 A1 (github.com/stanford-cs336/assignment1-basics) ships fixed tests that call the functions
in `tests/adapters.py`. This module fills that interface with *the code of the DLFS6 book*
(third_party/dlfs6), so the public A1 tests can measure how close DLFS6's implementations are to
the course's reference. Nothing here is new model/tokenizer/optimizer logic: every adapter either
calls a DLFS6 class/function, calls the PyTorch built-in that DLFS6 itself uses, or raises
NotImplementedError when DLFS6 has no counterpart. The glue only renames weights and converts
data formats. (DLFS6 freely uses torch.nn / torch.nn.functional, which A1 forbids, so this file is
also not a valid A1 submission.)

Variants (environment variable DLFS6_A1_VARIANT):
  as_shipped      DLFS6 exactly as published (commit pinned by the git submodule).
  one_line_fixes  three one-line deltas applied, each explained on the cs336 page:
                  RMSNorm eps=1e-5, BPE tie-break on bytes, get_batch includes the last window.

Used by verify/cs336/run_a1.py (which writes verify/cs336/results.json) and verify/test_reference.py.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ into third_party/dlfs6

ROOT = Path(__file__).resolve().parents[2]
DLFS6 = ROOT / "third_party" / "dlfs6"
if str(DLFS6) not in sys.path:
    sys.path.insert(0, str(DLFS6))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from storybot import model as sb_model  # noqa: E402
from storybot import tokenizer as sb_tok  # noqa: E402

VARIANT = os.environ.get("DLFS6_A1_VARIANT", "as_shipped")
if VARIANT not in ("as_shipped", "one_line_fixes"):
    raise ValueError(f"unknown DLFS6_A1_VARIANT={VARIANT!r}")
FIXES = VARIANT == "one_line_fixes"


# ------------------------------------------------------------------ loading book scripts safely

def load_defs(rel_path: str, names=None, replace: dict | None = None, extra: dict | None = None) -> dict:
    """Exec only the imports and def/class nodes of a DLFS6 script (its training loop never runs).

    replace: {old: new} source substitutions; each `old` must occur, so a changed upstream file
             fails loudly instead of silently testing something else.
    """
    src = (DLFS6 / rel_path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (names is None or n.name in names))]
    code = "\n\n".join(ast.get_source_segment(src, n) for n in keep)
    for old, new in (replace or {}).items():
        if code.count(old) != 1:
            raise ValueError(f"{rel_path}: expected exactly one occurrence of {old!r}")
        code = code.replace(old, new)
    ns: dict = {"__name__": f"dlfs6_{Path(rel_path).stem}"}
    exec(compile(code, f"<dlfs6:{rel_path}>", "exec"), ns)
    ns.update(extra or {})
    return ns


def _quiet(iterable, **_kwargs):
    """Stand-in for tqdm: the book's progress bars are not part of what we measure."""
    return iterable


# The three one-line deltas (shown verbatim on the cs336 page).
GET_BATCH_FIX = ("torch.randint(len(data) - context_len - 1, (batch_size,))",
                 "torch.randint(len(data) - context_len, (batch_size,))")
TIEBREAK_FIX = ("key=lambda pair: (pair_counts[pair], pair[0], pair[1])",
                "key=lambda pair: (pair_counts[pair], id_bytes(pair[0], merge_rules), id_bytes(pair[1], merge_rules))")
RMSNORM_EPS_FIX = 1e-5  # nn.RMSNorm(embed_dim) -> nn.RMSNorm(embed_dim, eps=1e-5)


_BYTES = {"owner": None, "map": {}, "seen": 0}


def id_bytes(tok_id: int, merge_rules: dict) -> bytes:
    """Bytes of a DLFS6 token id, rebuilt from its {(id1, id2): new_id} merge table (cached per run)."""
    if _BYTES["owner"] is not merge_rules:
        _BYTES.update(owner=merge_rules, map={i: bytes([i]) for i in range(256)}, seen=0)
    if len(merge_rules) != _BYTES["seen"]:
        m = _BYTES["map"]
        for (a, b), new in list(merge_rules.items())[_BYTES["seen"]:]:
            m[new] = m[a] + m[b]
        _BYTES["seen"] = len(merge_rules)
    return _BYTES["map"][tok_id]


_BPE = load_defs("ch04/02_bpe_cache.py", names={"pretokenize", "count_pairs", "merge", "train_bpe"},
                 replace=dict([TIEBREAK_FIX]) if FIXES else None,
                 extra={"tqdm": _quiet, "id_bytes": id_bytes})
_PRETRAIN = load_defs("ch06/05_pretrain.py", names={"get_batch"},
                      replace=dict([GET_BATCH_FIX]) if FIXES else None)
_ADAMW = load_defs("ch06/02_adamw.py", names={"AdamW"})
_LR = load_defs("ch06/03_lr_scheduler.py")


def _apply_variant(module: nn.Module) -> nn.Module:
    """StoryBot builds every norm as nn.RMSNorm(embed_dim): eps=None means finfo(dtype).eps."""
    if FIXES:
        for m in module.modules():
            if isinstance(m, nn.RMSNorm):
                m.eps = RMSNORM_EPS_FIX
    return module


def _load_exact(module: nn.Module, state: dict) -> nn.Module:
    missing, unexpected = module.load_state_dict(state, strict=False)
    rope_buffers = [k for k in missing if k.endswith(("cos_cache", "sin_cache"))]
    if unexpected or set(missing) - set(rope_buffers):
        raise KeyError(f"weight mapping incomplete: missing={missing}, unexpected={unexpected}")
    return module


# Weight names: CS336 reference (keys of tests/fixtures/ts_tests/model.pt) -> DLFS6 StoryBot.
BLOCK_KEYS = {
    "attn.q_proj.weight": "attn.W_q.weight",
    "attn.k_proj.weight": "attn.W_k.weight",
    "attn.v_proj.weight": "attn.W_v.weight",
    "attn.output_proj.weight": "attn.W_o.weight",
    "ln1.weight": "norm1.weight",
    "ln2.weight": "norm2.weight",
    "ffn.w1.weight": "ffn.W.weight",   # SiLU branch
    "ffn.w3.weight": "ffn.V.weight",   # linear (gate) branch
    "ffn.w2.weight": "ffn.O.weight",   # down projection
}
LM_KEYS = {"token_embeddings.weight": "embed.weight", "ln_final.weight": "norm.weight",
           "lm_head.weight": "unembed.weight"}


def rename_lm_state(weights: dict) -> dict:
    out = {}
    for k, v in weights.items():
        if k in LM_KEYS:
            out[LM_KEYS[k]] = v
            continue
        prefix, i, rest = k.split(".", 2)            # layers.{i}.<block key>
        if prefix != "layers" or rest not in BLOCK_KEYS:
            raise KeyError(f"no DLFS6 counterpart for weight {k!r}")
        out[f"blocks.{i}.{BLOCK_KEYS[rest]}"] = v
    return out


def _offset(token_positions, seq_len: int) -> int:
    """DLFS6's RoPE takes one integer offset; CS336 passes explicit token_positions."""
    if token_positions is None:
        return 0
    pos = torch.as_tensor(token_positions).reshape(-1, seq_len)
    start = int(pos[0, 0])
    if not torch.equal(pos, torch.arange(start, start + seq_len).expand_as(pos)):
        raise NotImplementedError("DLFS6 RoPE only supports contiguous positions offset..offset+C-1")
    return start


# ------------------------------------------------------------------ A1 adapter functions

def run_linear(d_in, d_out, weights, in_features):
    layer = nn.Linear(d_in, d_out, bias=False)          # how storybot/model.py builds projections
    return _load_exact(layer, {"weight": weights})(in_features)


def run_embedding(vocab_size, d_model, weights, token_ids):
    emb = nn.Embedding(vocab_size, d_model)               # storybot GPT.embed
    return _load_exact(emb, {"weight": weights})(token_ids)


def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    ffn = sb_model.SwiGLU(d_model, d_ff)
    return _load_exact(ffn, {"W.weight": w1_weight, "V.weight": w3_weight, "O.weight": w2_weight})(in_features)


def run_scaled_dot_product_attention(Q, K, V, mask=None):
    # DLFS6 has no stand-alone SDPA with an arbitrary mask; WebBot delegates to PyTorch's kernel.
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)


def run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight,
                                 o_proj_weight, in_features):
    mha = sb_model.MultiHeadAttention(d_model, num_heads, d_model // num_heads, rope=None)
    _load_exact(mha, {"W_q.weight": q_proj_weight, "W_k.weight": k_proj_weight,
                      "W_v.weight": v_proj_weight, "W_o.weight": o_proj_weight})
    return mha(in_features)


def run_multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, q_proj_weight,
                                           k_proj_weight, v_proj_weight, o_proj_weight, in_features,
                                           token_positions=None):
    if _offset(token_positions, in_features.shape[-2]) != 0:
        raise NotImplementedError("DLFS6 MultiHeadAttention applies RoPE from position 0 when use_cache=False")
    head_dim = d_model // num_heads
    mha = sb_model.MultiHeadAttention(d_model, num_heads, head_dim, rope=sb_model.RoPE(theta, head_dim, max_seq_len))
    _load_exact(mha, {"W_q.weight": q_proj_weight, "W_k.weight": k_proj_weight,
                      "W_v.weight": v_proj_weight, "W_o.weight": o_proj_weight})
    return mha(in_features)


def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    x = in_query_or_key
    seq = x.shape[-2]
    rope = sb_model.RoPE(theta, d_k, max_seq_len)
    out = rope(x.reshape(-1, 1, seq, d_k), _offset(token_positions, seq))   # DLFS6 expects (B, H, C, D)
    return out.reshape(x.shape)


def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    rope = sb_model.RoPE(theta, d_model // num_heads, max_seq_len)
    block = _apply_variant(sb_model.Block(d_model, num_heads, d_ff, rope))
    return _load_exact(block, {BLOCK_KEYS[k]: v for k, v in weights.items()})(in_features)


def run_transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                       weights, in_indices):
    model = sb_model.GPT(vocab_size, context_length, d_model, num_heads, num_layers, d_ff, rope_theta)
    _load_exact(_apply_variant(model), rename_lm_state(weights))
    return model.eval()(in_indices)


def run_rmsnorm(d_model, eps, weights, in_features):
    norm = nn.RMSNorm(d_model)                # storybot/model.py ignores eps: nn.RMSNorm(embed_dim)
    _apply_variant(norm)
    return _load_exact(norm, {"weight": weights})(in_features)


def run_silu(in_features):
    return sb_model.silu(in_features)         # storybot/model.py: x * torch.sigmoid(x)


def run_get_batch(dataset, batch_size, context_length, device):
    return _PRETRAIN["get_batch"](dataset, context_length, batch_size, device)


def run_softmax(in_features, dim):
    return F.softmax(in_features, dim=dim)    # every DLFS6 model calls F.softmax


def run_cross_entropy(inputs, targets):
    return F.cross_entropy(inputs, targets)   # ch03/ch06 training loops call F.cross_entropy


def run_gradient_clipping(parameters, max_l2_norm):
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)   # ch06/05, ch09/01, ch03/09


def get_adamw_cls():
    return _ADAMW["AdamW"]                    # ch06/02_adamw.py


def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    # DLFS6 decays linearly to 0 ("D2Z", ch06/03) instead of cosine-to-min_lr: a different schedule.
    return _LR["get_lr"](it, max_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(model, optimizer, iteration, out):
    raise NotImplementedError("DLFS6 GPT.save() stores weights + config only (no optimizer state, no iteration)")


def run_load_checkpoint(src, model, optimizer):
    raise NotImplementedError("DLFS6 GPT.load_from() rebuilds a model from weights + config only")


class DLFS6Tokenizer:
    """storybot.tokenizer.BPETokenizer driven by CS336-style (vocab, merges).

    DLFS6 numbers tokens internally (bytes 0-255, merge r -> 256+r, the end token last); this wrapper
    only translates between those ids and the ids of the given vocab. Encoding is DLFS6's own
    rank-based `_encode_text`; there is no encode_iterable (DLFS6 encodes big files with encode_file).
    """

    def __init__(self, vocab, merges, special_tokens=None):
        specials = list(special_tokens or [])
        if len(specials) > 1:
            raise NotImplementedError("DLFS6 BPETokenizer supports exactly one special token (end_token)")
        to_internal = {bytes([i]): i for i in range(256)}
        merge_rules = {}
        for rank, (a, b) in enumerate(merges):
            merge_rules[(to_internal[a], to_internal[b])] = 256 + rank
            to_internal.setdefault(a + b, 256 + rank)
        self.tok = sb_tok.BPETokenizer(merge_rules, end_token=specials[0] if specials else "<|endoftext|>")
        vocab_id = {b: i for i, b in vocab.items()}
        self.to_vocab = {i: vocab_id[b] for i, b in self.tok.id_to_bytes.items() if b in vocab_id}
        self.from_vocab = {v: i for i, v in self.to_vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.to_vocab[i] for i in self.tok.encode(text)]

    def decode(self, ids) -> str:
        return self.tok.decode([self.from_vocab[i] for i in ids])


def get_tokenizer(vocab, merges, special_tokens=None):
    return DLFS6Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    if len(special_tokens) > 1:
        raise NotImplementedError("DLFS6 train_bpe splits on exactly one end_token")
    end_token = special_tokens[0] if special_tokens else "<|endoftext|>"
    text = Path(input_path).read_text(encoding="utf-8")
    merge_rules = _BPE["train_bpe"](text, vocab_size, end_token)          # ch04/02_bpe_cache.py
    vocab = dict(sb_tok.BPETokenizer(merge_rules, end_token).id_to_bytes)  # DLFS6's own vocab layout
    return vocab, [(vocab[a], vocab[b]) for (a, b) in merge_rules]


# What each adapter maps to; run_a1.py copies this into results.json.
#   dlfs6     DLFS6's own hand-written code       torch   the PyTorch built-in DLFS6 itself calls
#   design    DLFS6 deliberately does something else          none    no counterpart in DLFS6
CATALOG = {
    "run_linear": ("torch", "nn.Linear(bias=False)，storybot/model.py 的全部投影"),
    "run_embedding": ("torch", "nn.Embedding，storybot GPT.embed"),
    "run_swiglu": ("dlfs6", "storybot/model.py SwiGLU（W/V/O ↔ w1/w3/w2）"),
    "run_scaled_dot_product_attention": ("torch", "F.scaled_dot_product_attention（webbot/model.py）"),
    "run_multihead_self_attention": ("dlfs6", "storybot/model.py MultiHeadAttention（rope=None）"),
    "run_multihead_self_attention_with_rope": ("dlfs6", "storybot/model.py MultiHeadAttention + RoPE"),
    "run_rope": ("dlfs6", "storybot/model.py RoPE（相邻两维配对）"),
    "run_transformer_block": ("dlfs6", "storybot/model.py Block"),
    "run_transformer_lm": ("dlfs6", "storybot/model.py GPT"),
    "run_rmsnorm": ("torch", "nn.RMSNorm(embed_dim)，eps 用默认值"),
    "run_silu": ("dlfs6", "storybot/model.py silu"),
    "run_get_batch": ("dlfs6", "ch06/05_pretrain.py get_batch"),
    "run_softmax": ("torch", "F.softmax"),
    "run_cross_entropy": ("torch", "F.cross_entropy"),
    "run_gradient_clipping": ("torch", "torch.nn.utils.clip_grad_norm_"),
    "get_adamw_cls": ("dlfs6", "ch06/02_adamw.py AdamW"),
    "run_get_lr_cosine_schedule": ("design", "ch06/03_lr_scheduler.py get_lr（warmup + 线性衰减到 0）"),
    "run_save_checkpoint": ("none", "GPT.save 只存权重与配置"),
    "run_load_checkpoint": ("none", "GPT.load_from 只读权重与配置"),
    "get_tokenizer": ("dlfs6", "storybot/tokenizer.py BPETokenizer（按 rank 合并）"),
    "run_train_bpe": ("dlfs6", "ch04/02_bpe_cache.py train_bpe"),
}
