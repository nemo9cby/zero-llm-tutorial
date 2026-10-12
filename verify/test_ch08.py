"""第 8 章 模型【挑战篇】：本地验证（需要 PyTorch，只用 CPU）。

本章的 NumPy 验证单元由 verify/_ch08_cells.py 直接从 content/ch08.md 读出并执行，
所以这里对拍的就是页面上发布的那份代码。真实权重的测试需要先下载 StoryBot 的 checkpoint：
    uv run python tools/fetch_data.py
运行：
    uv run pytest verify/test_ch08.py
"""
import math
import shutil
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from _ch08_cells import run_cells_upto


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def wb(dlfs6_path):
    import webbot.model as wb
    return wb


@pytest.fixture(scope="module")
def sb(dlfs6_path):
    import storybot.model as sb
    return sb


@pytest.fixture(scope="module")
def storybot_ckpt(dlfs6_path):
    path = dlfs6_path / "storybot" / "model_pretrain.pt"
    if not path.is_file():
        pytest.skip("缺少 storybot/model_pretrain.pt：先运行 uv run python tools/fetch_data.py")
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def valid_tokens(dlfs6_path):
    path = dlfs6_path / "storybot" / "tiny_stories_valid.bin"
    if not path.is_file():
        pytest.skip("缺少 storybot/tiny_stories_valid.bin：先运行 uv run python tools/fetch_data.py")
    return np.memmap(path, dtype=np.uint16, mode="r")


def storybot_config(ck):
    return [ck[k] for k in ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim", "theta")]


def windows(tokens, n, T, stride, start=0):
    """n 个长 T 的窗口及其右移一位的目标（与原书 get_batch 的 x、y 相同）。"""
    ix = start + np.arange(n) * stride
    x = torch.from_numpy(np.stack([tokens[i:i + T].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([tokens[i + 1:i + T + 1].astype(np.int64) for i in ix]))
    return x, y


def count(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- 8.1 模型规模

def test_param_counts_match_formula(wb, sb):
    ns = run_cells_upto("8.1-a")                         # 页面上的 webbot_params / total_params
    with torch.device("meta"):                           # 只建形状，不分配内存
        for kv, expected in [(12, 161_753_856), (4, 152_316_672), (1, 148_777_728)]:
            model = wb.GPT(50000, 1024, 768, 12, kv, 12, 2048)
            assert count(model) == ns["total_params"](ns["webbot_params"](KV=kv)) == expected
        story = sb.GPT(10000, 256, 512, 16, 4, 1344)     # ch06/05_pretrain.py 的配置
    assert count(story) == 22_696_448
    assert model.blocks[0].attn.W_k.weight.shape == (64, 768)   # 最后一个是 KV=1：W_k 只有一个头


# ---------------------------------------------------------------- 8.2 Flash Attention

def test_numpy_flash_matches_torch_sdpa(wb, sb):
    flash = run_cells_upto("8.2-c")["attention_flash"]  # 页面上的 NumPy 分块实现
    rng = np.random.default_rng(42)
    for T, Br, Bc, causal in [(64, 16, 16, True), (50, 16, 8, True), (50, 16, 8, False), (37, 8, 4, True)]:
        Q, K, V = (rng.normal(size=(T, 16)) for _ in range(3))
        O, lse, _ = flash(Q, K, V, causal=causal, Br=Br, Bc=Bc)
        q, k, v = (torch.from_numpy(a)[None, None] for a in (Q, K, V))      # (1, 1, T, d)，float64
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)[0, 0].numpy()
        assert np.abs(O - ref).max() < 1e-12
        S = q @ k.transpose(-2, -1) / math.sqrt(16)
        if causal:
            S = S.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        assert np.abs(lse - torch.logsumexp(S, dim=-1)[0, 0].numpy()).max() < 1e-12

    # 原书的两种写法：WebBot 的 SDPA（KV = H 时）与 StoryBot 的手写 attention，权重相同则输出相同
    torch.manual_seed(0)
    story_attn = sb.MultiHeadAttention(64, 4, 16, rope=sb.RoPE(10000, 16, 32)).double()
    web_attn = wb.MultiHeadAttention(64, 4, 4, 16, rope=wb.RoPE(10000, 16, 32)).double()
    web_attn.load_state_dict(story_attn.state_dict())
    x = torch.randn(2, 20, 64, dtype=torch.float64)
    with torch.no_grad():
        assert (story_attn(x) - web_attn(x)).abs().max() < 1e-12


def test_sdpa_is_causal_is_top_left_aligned():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 5, 8)   # decode：1 个新 query，缓存里 5 个 key
    eye = torch.eye(5).view(1, 1, 5, 5)                       # value 取 one-hot，输出就是注意力权重本身
    causal = F.scaled_dot_product_attention(q, k, eye, is_causal=True)[0, 0, 0]
    full = F.scaled_dot_product_attention(q, k, eye, is_causal=False)[0, 0, 0]
    assert causal.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]       # 左上角对齐：只看得到第 0 个 key
    assert (full > 0).all() and abs(full.sum().item() - 1) < 1e-6   # WebBot decode 的做法：看得到全部缓存
    assert "upper left" in F.scaled_dot_product_attention.__doc__   # 文档里写明的对齐方式


def test_sdpa_on_cpu_is_flash_kernel_with_linear_memory():
    from torch.profiler import ProfilerActivity, profile

    def naive(q, k, v):                       # StoryBot 的写法：摆出完整的 (T, T) 分数矩阵
        T = q.shape[-2]
        s = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        s = s.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
        return torch.softmax(s, dim=-1) @ v

    def sdpa(q, k, v):                        # WebBot 的写法
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def measure(fn, T):
        q, k, v = (torch.randn(1, 1, T, 64) for _ in range(3))
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            fn(q, k, v)
        mib = sum(max(e.self_cpu_memory_usage, 0) for e in prof.key_averages()) / 2**20
        return mib, {e.name for e in prof.events()}

    sizes = [1024, 2048, 4096]
    naive_mib = [measure(naive, T)[0] for T in sizes]
    sdpa_runs = [measure(sdpa, T) for T in sizes]
    sdpa_mib = [mib for mib, _ in sdpa_runs]
    print("naive MiB:", [round(m, 1) for m in naive_mib], "| sdpa MiB:", [round(m, 1) for m in sdpa_mib])
    assert "aten::_scaled_dot_product_flash_attention_for_cpu" in sdpa_runs[0][1]
    for small, big in zip(naive_mib, naive_mib[1:]):
        assert 3.5 < big / small < 4.5        # 长度翻倍，朴素写法分配的内存变成约 4 倍
    assert sdpa_mib[-1] < 2 * sdpa_mib[0]     # SDPA 几乎不随长度增长
    assert naive_mib[-1] > 50 * sdpa_mib[-1]


# ---------------------------------------------------------------- 8.3 GQA

def test_enable_gqa_is_repeat_interleave():
    torch.manual_seed(0)
    B, H, KV, T, D = 2, 12, 4, 7, 64          # WebBot：12 个 query 头，4 个 KV 头
    Q, K, V = torch.randn(B, H, T, D), torch.randn(B, KV, T, D), torch.randn(B, KV, T, D)
    gqa = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
    inter = F.scaled_dot_product_attention(Q, K.repeat_interleave(3, 1), V.repeat_interleave(3, 1), is_causal=True)
    tiled = F.scaled_dot_product_attention(Q, K.repeat(1, 3, 1, 1), V.repeat(1, 3, 1, 1), is_causal=True)
    assert (gqa - inter).abs().max() < 1e-6
    assert (gqa - tiled).abs().max() > 0.1
    assert "repeat_interleave" in F.scaled_dot_product_attention.__doc__   # 文档里的参考实现

    ns = run_cells_upto("8.3-a")              # 页面上 NumPy 版的分组与 torch 一致
    q, k, v = (torch.from_numpy(ns[n])[None] for n in ("Q", "K", "V"))
    torch_gqa = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)[0].numpy()
    assert np.abs(torch_gqa - ns["inter"]).max() < 1e-12


def test_webbot_cache_holds_only_kv_heads(wb):
    torch.manual_seed(0)
    model = wb.GPT(1000, 64, 768, 12, 4, 1, 2048).eval()     # WebBot 的宽度与头数，只取 1 层
    with torch.no_grad():
        model.clear_cache()
        model(torch.randint(0, 1000, (1, 10)), use_cache=True)
    k_cache = model.blocks[0].attn.k_cache
    assert tuple(k_cache.shape) == (1, 4, 10, 64)            # (B, KV, T, head_dim)
    per_token_one_layer = 2 * 4 * 64 * 2                     # K 与 V × 4 个头 × 64 维 × bf16 的 2 字节
    assert run_cells_upto("8.3-b")["kv_bytes_per_token"](12, 4, 64) == 12 * per_token_one_layer


def test_gqa_with_all_kv_heads_is_storybot_mha(wb, sb, storybot_ckpt, valid_tokens):
    V, C, E, H, L, FF, theta = storybot_config(storybot_ckpt)
    sd = storybot_ckpt["model_state_dict"]                   # HF 上的权重：FFN 的键名是 mlp
    story = sb.GPT(V, C, E, H, L, FF, theta).eval()
    story.load_state_dict({k.replace(".mlp.", ".ffn."): w for k, w in sd.items()})
    web = wb.GPT(V, C, E, H, H, L, FF, theta).eval()         # n_kv_head = n_head = 16
    web.load_state_dict(sd)                                  # 键名原封不动就能装进去
    x, _ = windows(valid_tokens, 2, 64, 1_000_000)
    with torch.no_grad():
        assert (story(x) - web(x)).abs().max() < 1e-4

    def greedy(model, prompt, n=20):                         # 都走 KV cache
        model.clear_cache()
        out, nxt = [], prompt
        with torch.no_grad():
            for _ in range(n):
                nxt = model(nxt, use_cache=True)[:, -1:].argmax(-1)
                out.append(nxt.item())
        return out

    prompt = x[:1, :8]
    assert greedy(story, prompt) == greedy(web, prompt)


def test_mean_pooled_gqa_needs_uptraining(wb, storybot_ckpt, valid_tokens):
    V, C, E, H, L, FF, theta = storybot_config(storybot_ckpt)
    D = E // H
    sd = storybot_ckpt["model_state_dict"]
    x, y = windows(valid_tokens, 8, C, 680_000)              # 验证集上 8 个长 256 的窗口

    def mean_pooled(kv):                                     # GQA 论文的做法：每组各头的 K、V 投影取平均
        g, out = H // kv, dict(sd)
        for layer in range(L):
            for name in ("W_k", "W_v"):
                w = sd[f"blocks.{layer}.attn.{name}.weight"]          # (H*D, E)：第 h 个头占 h*D 到 (h+1)*D 行
                out[f"blocks.{layer}.attn.{name}.weight"] = w.view(kv, g, D, E).mean(1).reshape(kv * D, E)
        return out

    losses = {}
    for kv in (16, 8, 4, 1):
        model = wb.GPT(V, C, E, H, kv, L, FF, theta).eval()
        model.load_state_dict(mean_pooled(kv))
        with torch.no_grad():
            losses[kv] = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
    print({kv: round(loss, 3) for kv, loss in losses.items()})
    expected = {16: 1.266, 8: 4.380, 4: 7.321, 1: 7.465}     # 正文 8.3 节引用的数
    for kv, value in expected.items():
        assert abs(losses[kv] - value) < 0.002, (kv, losses[kv])


# ---------------------------------------------------------------- 8.5 torch.compile

def test_default_inductor_needs_a_cxx_compiler():
    import torch._dynamo as dynamo

    def f(x):
        return F.silu(x) * x + 1

    x = torch.randn(8)
    dynamo.reset()
    if sys.platform == "win32" and shutil.which("cl") is None:
        with pytest.raises(Exception, match="cl is not found"):     # InductorError: InvalidCxxCompiler
            torch.compile(f)(x)
    else:
        assert torch.allclose(torch.compile(f)(x), f(x))
    dynamo.reset()


def test_compile_captures_webbot_in_one_graph(wb):
    import torch._dynamo as dynamo

    torch.manual_seed(0)
    model = wb.GPT(1000, 64, 128, 4, 2, 2, 344)              # 结构与 WebBot 相同的小模型
    x, y = torch.randint(0, 1000, (2, 16)), torch.randint(0, 1000, (2, 16))
    graphs = []

    def counting_backend(gm, example_inputs):                # 只记录捕获到的图，不编译
        graphs.append(gm)
        return gm.forward

    dynamo.reset()
    with torch.no_grad():                                    # fullgraph=True：只要有一处中断就报错
        torch.compile(model, backend=counting_backend, fullgraph=True)(x)
    assert len(graphs) == 1                                  # 整个前向是 1 张图、0 次中断

    dynamo.reset()
    compiled = torch.compile(model, backend="aot_eager")     # 捕获图、追踪反向，但不生成新代码
    logits_c = compiled(x)
    F.cross_entropy(logits_c.view(-1, 1000), y.view(-1)).backward()
    grads_c = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    logits_e = model(x)
    F.cross_entropy(logits_e.view(-1, 1000), y.view(-1)).backward()
    assert torch.equal(logits_c, logits_e)                   # logits 逐位相同
    assert all(torch.equal(g, p.grad) for g, p in zip(grads_c, model.parameters()))   # 梯度逐位相同
    dynamo.reset()


def test_compile_recompiles_for_eval_and_last_batch(wb):
    import torch._dynamo as dynamo

    torch.manual_seed(0)
    model = wb.GPT(1000, 64, 128, 4, 2, 2, 344)
    compiles = []

    def counting_backend(gm, example_inputs):                # 只计数，不编译
        compiles.append(len(gm.graph.nodes))
        return gm.forward

    dynamo.reset()
    compiled = torch.compile(model, backend=counting_backend)

    def train_step(batch):
        x, y = torch.randint(0, 1000, (batch, 16)), torch.randint(0, 1000, (batch, 16))
        F.cross_entropy(compiled(x).view(-1, 1000), y.view(-1)).backward()

    def evaluate(batch):
        model.eval()
        with torch.no_grad():
            compiled(torch.randint(0, 1000, (batch, 16)))
        model.train()

    for _ in range(3):
        train_step(4)                                        # 训练：每个 batch 的形状都相同
    assert len(compiles) == 1
    evaluate(4)                                              # 第一次评估：切换到推理状态
    assert len(compiles) == 2
    evaluate(2)                                              # 评估的最后一个 batch 比较小（ch09 里是 8 条）
    assert len(compiles) == 3                                # 这一次编译出的是 batch 维可变的版本
    for _ in range(2):                                       # 之后来回切换，都命中缓存
        train_step(4)
        evaluate(4)
        evaluate(3)
    assert len(compiles) == 3
    # 注意：大小为 1 的维度总是被特化，所以这里不用 batch=1 来代表“最后一个小 batch”
    dynamo.reset()


def test_compiled_save_has_clean_keys(wb, tmp_path):
    model = wb.GPT(1000, 64, 128, 4, 2, 2, 344)
    compiled = torch.compile(model, backend="eager")
    assert type(compiled).__name__ == "OptimizedModule" and compiled._orig_mod is model
    assert all(k.startswith("_orig_mod.") for k in compiled.state_dict())
    path = tmp_path / "webbot_tiny.pt"
    compiled.save(path)                                      # ch09/01_pretrain.py 第 223 行的调用方式
    saved = torch.load(path, weights_only=True)["model_state_dict"]
    assert not any(k.startswith("_orig_mod.") for k in saved)
    reloaded = wb.GPT.load_from(path)                        # 原书的加载函数可以直接读回来
    assert all(torch.equal(a, b) for a, b in zip(reloaded.state_dict().values(), model.state_dict().values()))


# ---------------------------------------------------------------- 8.6 上下文扩展

def test_rope_clamp_in_book_models(wb, sb):
    ns = run_cells_upto("8.6-a")                             # 页面上照搬的 NumPy 版 RoPE
    cos, sin = ns["rope_cache"](10000.0, 64, 1024)
    book_rope = wb.RoPE(10000, 64, 1024)
    x = np.random.default_rng(0).normal(size=(1, 64))
    xt = torch.from_numpy(x)[None, None]                     # (1, 1, 1, 64)
    for offset in (0, 500, 1023, 1100, 1500):
        ours = ns["rope"](x, cos, sin, offset)
        book = book_rope(xt, offset)[0, 0].numpy()           # 原书在 float32 下算角度
        assert np.abs(ours - book).max() < 1e-3
    assert torch.equal(book_rope(xt, 1100), book_rope(xt, 1023))      # 超出表长：钳在 1023
    assert torch.equal(book_rope(xt, 1500), book_rope(xt, 1023))

    # 上下文只有 16 的小模型：先 prefill 4 个 token，再逐个生成到第 33 个位置
    lengths = {}
    for name, model in [("storybot", sb.GPT(1000, 16, 64, 4, 2, 172)), ("webbot", wb.GPT(1000, 16, 64, 4, 2, 2, 172))]:
        model.eval()
        model.clear_cache()
        nxt = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            for _ in range(30):
                nxt = model(nxt, use_cache=True)[:, -1:].argmax(-1)
        attn = model.blocks[0].attn
        lengths[name] = attn.k_cache.shape[2]
        probe = torch.randn(1, 1, 1, 16)
        assert torch.equal(attn.rope(probe, attn.cache_offset), attn.rope(probe, 15))   # 位置钳在 15
    assert lengths == {"storybot": 33, "webbot": 16}         # StoryBot 一直变长，WebBot 像滑动窗口


def test_context_extension_on_storybot(sb, storybot_ckpt, valid_tokens):
    V, C, E, H, L, FF, theta = storybot_config(storybot_ckpt)
    D = E // H
    sd = {k.replace(".mlp.", ".ffn."): w for k, w in storybot_ckpt["model_state_dict"].items()}
    n, T = 8, 2 * C                                           # 8 个长 512 的窗口：训练长度的 2 倍
    x, y = windows(valid_tokens, n, T, len(valid_tokens) // n - 600, start=777)
    s = T / C

    def extended(pos_scale=1.0, base=theta):
        model = sb.GPT(V, T, E, H, L, FF, base).eval()
        model.load_state_dict({k: w for k, w in sd.items() if "_cache" not in k}, strict=False)
        inv_freq = 1.0 / base ** (2.0 * torch.arange(D // 2) / D)          # 与 RoPE.__init__ 相同
        angles = (torch.arange(T) / pos_scale)[:, None] * inv_freq[None, :]  # PI：位置 m 换成 m/s
        rope = model.blocks[0].attn.rope                                     # 各层共用同一个 RoPE
        rope.cos_cache.copy_(torch.cos(angles))
        rope.sin_cache.copy_(torch.sin(angles))
        return model

    def halves(model):                                        # 前 256 个位置与后 256 个位置的平均 loss
        with torch.no_grad():
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction="none").view(n, T)
        return loss[:, :C].mean().item(), loss[:, C:].mean().item()

    results = {
        "extrapolate": halves(extended()),
        "pi": halves(extended(pos_scale=s)),
        "ntk": halves(extended(base=theta * s ** (D / (D - 2)))),
    }
    original = sb.GPT(V, C, E, H, L, FF, theta).eval()
    original.load_state_dict(sd)
    with torch.no_grad():
        results["reference"] = tuple(
            F.cross_entropy(original(x[:, a:a + C]).reshape(-1, V), y[:, a:a + C].reshape(-1)).item() for a in (0, C))
    print({k: tuple(round(v, 3) for v in pair) for k, pair in results.items()})
    expected = {                                              # 正文 8.6 节表格里的数
        "extrapolate": (1.384, 2.496),
        "pi": (1.553, 1.418),
        "ntk": (1.388, 1.330),
        "reference": (1.384, 1.336),
    }
    for key, pair in expected.items():
        assert all(abs(a - b) < 0.002 for a, b in zip(results[key], pair)), (key, results[key])
