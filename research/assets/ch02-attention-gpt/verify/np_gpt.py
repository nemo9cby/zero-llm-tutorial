# NumPy-only GPT-2-style forward pass mirroring codebot.model.GPT (eval mode). Pyodide-friendly: numpy + math only.
import math
import numpy as np

_erf = np.vectorize(math.erf, otypes=[np.float64])


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)            # NumPy default ddof=0 == torch unbiased=False
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x, exact=True):
    if exact:                                   # nn.GELU() default (erf) -- what codebot uses
        return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))   # book 09 / GPT-2


def causal_attention(x, Wq, Wk, Wv, Wo, n_head):
    C, E = x.shape
    D = Wq.shape[0] // n_head
    split = lambda t: t.reshape(C, n_head, D).transpose(1, 0, 2)          # (H, C, D)
    q, k, v = split(x @ Wq.T), split(x @ Wk.T), split(x @ Wv.T)
    s = q @ k.transpose(0, 2, 1) / math.sqrt(D)                           # (H, C, C)
    s = np.where(np.tril(np.ones((C, C), dtype=bool)), s, -np.inf)
    h = softmax(s) @ v                                                    # (H, C, D)
    return h.transpose(1, 0, 2).reshape(C, n_head * D) @ Wo.T


def gpt_forward(ids, P, n_layer, n_head, exact_gelu=True):
    """ids: list[int] (one sequence). P: dict of numpy arrays named like codebot's state_dict."""
    C = len(ids)
    x = P["embed.weight"][ids] + P["pos_embed.weight"][:C]
    for i in range(n_layer):
        p = lambda n: P[f"blocks.{i}.{n}"]
        h = layer_norm(x, p("norm1.weight"), p("norm1.bias"))
        x = x + causal_attention(h, p("attn.W_q.weight"), p("attn.W_k.weight"),
                                 p("attn.W_v.weight"), p("attn.W_o.weight"), n_head)
        h = layer_norm(x, p("norm2.weight"), p("norm2.bias"))
        h = gelu(h @ p("ffn.layers.0.weight").T + p("ffn.layers.0.bias"), exact_gelu)
        x = x + h @ p("ffn.layers.2.weight").T + p("ffn.layers.2.bias")
    x = layer_norm(x, P["norm.weight"], P["norm.bias"])
    return x @ P["unembed.weight"].T + P["unembed.bias"]                 # (C, V); unembed.weight is the tied matrix
