"""第 5 章 模型【应用篇】：本地验证。

用 third_party/dlfs6 里的真实代码检查正文的结论，并与页面上的 NumPy 验证单元对拍。全部在 CPU 上运行：
    uv run pytest verify/test_ch05.py
"""
import contextlib
import inspect
import io
import json
import math
import sys

sys.dont_write_bytecode = True          # 导入 storybot / webbot / codebot 时不往 third_party/dlfs6 里写 __pycache__

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from _ch05_kvbench import measure  # noqa: E402
from _ch05_record import BUG as STORYBOT_BUG  # noqa: E402
from _ch05_record import CKPT as STORYBOT_CKPT  # noqa: E402
from _ch05_record import OUT as STORYBOT_RECORD  # noqa: E402
from _ch05_record import storybot_samples  # noqa: E402
from _ch05_util import DLFS6, ROOT, greedy_recompute, greedy_with_cache, load_defs, load_storybot, page_cells  # noqa: E402

KV_BUG = {"if not use_cache or is_first_call:": "if not use_cache:"}   # 05_kvcache.py 在 8b3911f 之前的条件


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def improved_gpt():
    return load_defs("ch05/04_improved_gpt.py")["GPT"]


# ---------------------------------------------------------------- 5.1 RoPE

def run_script_in_process(rel: str) -> list[str]:
    """Execute a DLFS6 script's full source (it only prints; nothing is written) and return its stdout lines."""
    src = (DLFS6 / rel).read_text(encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(src, str(DLFS6 / rel), "exec"), {"__name__": "__main__"})
    return out.getvalue().splitlines()


def test_book_scripts_print_the_shapes_in_their_comments():
    # 01 与 04 末尾的 print 后面，注释里写着期望的输出（在同一个进程里执行整份脚本，省掉重复 import torch 的时间）
    assert run_script_in_process("ch05/01_rope.py") == ["torch.Size([2, 10, 512])"]
    assert run_script_in_process("ch05/04_improved_gpt.py") == ["torch.Size([8, 256, 10000])"]


def test_rope_equals_the_numpy_version_on_the_page_and_has_no_parameters():
    RoPE = load_defs("ch05/01_rope.py", {"RoPE"})["RoPE"]
    rope = RoPE(10000, 64, 1024)
    assert n_params(rope) == 0
    assert {k: tuple(v.shape) for k, v in rope.named_buffers()} == {"cos_cache": (1024, 32), "sin_cache": (1024, 32)}
    rope_np = page_cells("5.1-c")["rope"]
    torch.manual_seed(0)
    x = torch.randn(1, 1, 10, 64)
    ref = rope_np(x[0, 0].double().numpy(), np.arange(10))
    assert np.abs(rope(x)[0, 0].double().numpy() - ref).max() < 1e-6


def test_rope_table_is_built_in_float32_and_its_error_grows_with_position():
    RoPE = load_defs("ch05/01_rope.py", {"RoPE"})["RoPE"]
    inv_freq = 1.0 / 10000 ** (np.arange(32) * 2.0 / 64)
    err = {}
    for L in (256, 1024, 32768):
        rope = RoPE(10000, 64, L)
        assert rope.cos_cache.dtype == torch.float32
        err[L] = np.abs(rope.cos_cache.double().numpy() - np.cos(np.arange(L)[:, None] * inv_freq)).max()
    assert err[256] < 2e-5 and err[1024] < 1e-4 and err[32768] > 5e-4
    assert err[32768] > 30 * err[256]


def test_rope_rotates_bf16_inputs_in_float32():
    RoPE = load_defs("ch05/01_rope.py", {"RoPE"})["RoPE"]
    rope = RoPE(10000, 64, 1024)
    torch.manual_seed(0)
    xb = torch.randn(1, 1, 10, 64).bfloat16()
    out = rope(xb)
    assert out.dtype == torch.bfloat16                          # 进出都是 bf16……
    truth = rope(xb.float())                                    # ……中间按 fp32 旋转同一组 bf16 数
    all_bf16 = xb[..., 0::2] * rope.cos_cache[:10].bfloat16() - xb[..., 1::2] * rope.sin_cache[:10].bfloat16()
    err_book = (out.float()[..., 0::2] - truth[..., 0::2]).abs().max().item()
    err_bf16 = (all_bf16.float() - truth[..., 0::2]).abs().max().item()
    assert err_book < err_bf16


def test_rope_offset_moves_q_and_k_together_without_changing_scores():
    RoPE = load_defs("ch05/05_kvcache.py", {"RoPE"})["RoPE"]
    rope = RoPE(10000, 64, 256)
    torch.manual_seed(0)
    Q, K = torch.randn(1, 2, 8, 64), torch.randn(1, 2, 8, 64)
    s0 = rope(Q, 0) @ rope(K, 0).transpose(-2, -1)
    s100 = rope(Q, 100) @ rope(K, 100).transpose(-2, -1)
    assert s0.abs().mean() > 1                                  # 分数本身是 O(1)–O(10) 的量
    assert (s0 - s100).abs().max() < 1e-4                       # 差别只是 float32 的舍入


def test_rope_buffers_are_saved_once_per_layer_and_pin_max_context_len():
    GPT = improved_gpt()
    sd = GPT(1000, 256, 384, 6, 6, 1024).state_dict()
    keys = [k for k in sd if "rope" in k]
    assert len(keys) == 12                                      # 6 层 × (cos_cache, sin_cache)
    assert len({sd[k].data_ptr() for k in keys if k.endswith("cos_cache")}) == 1   # 其实是同一块内存
    with pytest.raises(RuntimeError, match="cos_cache"):        # max_context_len 改成 512 就加载不进去
        GPT(1000, 512, 384, 6, 6, 1024).load_state_dict(sd)


# ---------------------------------------------------------------- 5.2 SwiGLU

def test_swiglu_matches_a_4d_ffn_in_parameters_and_the_page_numpy_version():
    ns = load_defs("ch05/02_swiglu.py")
    m = ns["SwiGLU"](384)
    assert m.W.out_features == 1024
    assert n_params(m) == 2 * 384 * (4 * 384) == 1_179_648
    assert all(lin.bias is None for lin in (m.W, m.V, m.O))
    assert ns["SwiGLU"](512).W.out_features == 1365             # int(512 * 8 / 3)，不取整
    x = torch.randn(1000)
    assert torch.allclose(ns["silu"](x), F.silu(x), atol=1e-6)
    assert torch.equal(F.silu(x), nn.SiLU()(x))
    swiglu_np = page_cells("5.2-a", "5.2-c")["swiglu"]
    torch.manual_seed(0)
    xx = torch.randn(2, 5, 384)
    w = lambda lin: lin.weight.detach().T.double().numpy()     # nn.Linear 存的是 (out, in)
    ref = swiglu_np(xx.double().numpy(), w(m.W), w(m.V), w(m.O))
    assert np.abs(m(xx).detach().double().numpy() - ref).max() < 1e-6


# ---------------------------------------------------------------- 5.3 RMSNorm 与其他改良

def test_hand_written_rmsnorm_vs_nn_rmsnorm_default_eps():
    RMSNorm = load_defs("ch05/03_rmsnorm.py")["RMSNorm"]
    d = 384
    mine = RMSNorm(d)
    assert mine.eps == 1e-5 and nn.RMSNorm(d).eps is None
    assert torch.finfo(torch.float32).eps == 2 ** -23
    torch.manual_seed(0)
    x = torch.randn(4, 7, d)
    with torch.no_grad():
        assert (mine(x) - nn.RMSNorm(d, eps=1e-5)(x)).abs().max() < 1e-6
        rms = lambda t: t.pow(2).mean(-1).sqrt().mean().item()
        assert rms(mine(x * 1e-3)) < 0.35 and rms(nn.RMSNorm(d)(x * 1e-3)) > 0.9   # 输入很小时差别巨大
        ln = nn.LayerNorm(d, elementwise_affine=False)
        rn = nn.RMSNorm(d, eps=1e-5, elementwise_affine=False)
        assert (ln(x) - rn(x - x.mean(-1, keepdim=True))).abs().max() < 1e-5
        rms_norm_np = page_cells("5.3-a")["rms_norm"]
        assert np.abs(mine(x).double().numpy() - rms_norm_np(x.double().numpy())).max() < 1e-5
    assert n_params(nn.RMSNorm(d)) == d and n_params(nn.LayerNorm(d)) == 2 * d


def test_dropout_makes_the_forward_pass_random_until_eval(dlfs6_path):
    from codebot.model import GPT as CodeGPT
    torch.manual_seed(0)
    m = CodeGPT(1000, 64, 64, 4, 2, 256, 0.1)
    ids = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        assert not torch.equal(m.train()(ids), m(ids))            # train 模式：两次前向不同
        assert torch.equal(m.eval()(ids), m(ids))                 # eval 模式：确定性
        Improved = improved_gpt()
        im = Improved(1000, 64, 64, 4, 2, 170).train()
        assert torch.equal(im(ids), im(ids))                      # 改良版没有 dropout，train 模式也确定


def test_codebot_vs_improved_gpt_parameter_accounting(dlfs6_path):
    from codebot.model import GPT as CodeGPT
    import storybot.model as sb
    GPT = improved_gpt()
    m = GPT(10000, 256, 384, 6, 6, int(384 * 8 / 3), 10000)
    assert n_params(m) == 18_301_824
    assert [n for n, _ in m.named_parameters() if n.endswith("bias")] == []
    cb, im = CodeGPT(1000, 256, 384, 6, 6, 4 * 384, 0.1), GPT(1000, 256, 384, 6, 6, 1024)
    assert n_params(cb) == 11_121_640 and n_params(im) == 11_389_824
    assert cb.embed.weight is cb.unembed.weight                 # CodeBot：权重共享
    assert im.embed.weight is not im.unembed.weight             # 改良版：各有一份
    assert n_params(sb.GPT(10000, 256, 512, 16, 4, 1344)) == 22_696_448
    params = page_cells("5.4-a")["improved_gpt_params"]
    assert params(10000, 384, 6, 1024) == n_params(m) and params(10000, 512, 4, 1344) == 22_696_448
    # storybot/model.py 的 GPT 与 04 的 GPT 是同一个模型：键名相同，不用 cache 时输出相同
    story = sb.GPT(10000, 256, 384, 6, 6, 1024, 10000)
    story.load_state_dict(m.state_dict())
    ids = torch.randint(0, 10000, (1, 32))
    with torch.no_grad():
        assert torch.allclose(story(ids), m(ids), atol=1e-6)


# ---------------------------------------------------------------- 5.4 改良版 GPT

def test_initial_loss_is_ln_v_plus_half_logit_variance():
    GPT = improved_gpt()
    V = 10_000
    shown = {384: (9.290, 9.287), 512: (9.303, 9.313)}         # 正文里写的数（随机目标、对目标取平均）
    for E, H, L, F_ in [(384, 6, 6, 1024), (512, 16, 4, 1344)]:
        torch.manual_seed(0)
        m = GPT(V, 256, E, H, L, F_, 10000)
        ids, tgt = torch.randint(0, V, (8, 256)), torch.randint(0, V, (8, 256))
        with torch.no_grad():
            logits = m(ids)
        loss = F.cross_entropy(logits.view(-1, V), tgt.view(-1)).item()
        expected = (torch.logsumexp(logits, -1) - logits.mean(-1)).mean().item()   # 对目标 token 取平均
        pred = math.log(V) + (0.02 * math.sqrt(E)) ** 2 / 2
        print(f"E={E}: 随机目标的 loss {loss:.4f}，期望值 {expected:.4f}，ln V + σ²/2 = {pred:.4f}")
        assert abs(expected - pred) < 0.01 and abs(loss - pred) < 0.03
        assert abs(loss - shown[E][0]) < 0.0015 and abs(expected - shown[E][1]) < 0.0015
        assert abs(logits.std().item() - 0.02 * math.sqrt(E)) < 0.01


def test_improved_gpt_is_causal():
    GPT = improved_gpt()
    torch.manual_seed(0)
    m = GPT(1000, 256, 384, 6, 2, 1024)
    ids = torch.randint(0, 1000, (2, 128))
    ids2 = ids.clone()
    ids2[:, 100] = (ids2[:, 100] + 1) % 1000
    with torch.no_grad():
        a, b = m(ids), m(ids2)
    assert torch.allclose(a[:, :100], b[:, :100], atol=1e-6)    # 位置 100 之前看不到改动
    assert (a[:, 100:] - b[:, 100:]).abs().max() > 1e-3


def test_published_storybot_checkpoint_needs_the_mlp_to_ffn_rename(dlfs6_path):
    if not STORYBOT_CKPT.is_file():
        pytest.skip("先运行 uv run python tools/fetch_data.py")
    from storybot.model import GPT
    ck = torch.load(STORYBOT_CKPT, map_location="cpu", weights_only=True)
    assert {k: ck[k] for k in ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim", "theta")} \
        == {"vocab_size": 10000, "max_context_len": 256, "embed_dim": 512, "n_head": 16, "n_layer": 4,
            "ff_dim": 1344, "theta": 10000}
    assert sum(".mlp." in k for k in ck["model_state_dict"]) == 12
    assert sum(".ffn." in k for k in ck["model_state_dict"]) == 0
    with pytest.raises(RuntimeError, match=r"blocks\.0\.ffn\.W\.weight"):
        GPT.load_from(STORYBOT_CKPT)
    assert n_params(load_storybot(STORYBOT_CKPT, GPT)) == 22_696_448


# ---------------------------------------------------------------- 5.5 KV cache

def test_page_numpy_gpt_matches_storybot_model(dlfs6_path):
    from storybot.model import GPT
    ns = page_cells("5.5-c")
    torch.manual_seed(0)
    tm = GPT(64, 32, 32, 4, 2, 85).eval()
    for name, p in tm.named_parameters():                     # 放大权重，让贪心解码没有“几乎平票”的情况
        if "norm" not in name:
            nn.init.normal_(p, 0.0, 0.2)
    P = ns["init_params"](V=64, C=32, E=32, H=4, L=2, F=85)
    sd = {k: v.double().numpy() for k, v in tm.state_dict().items()}
    P.update(emb=sd["embed.weight"], unemb=sd["unembed.weight"].T, g=sd["norm.weight"])
    for i, b in enumerate(P["blocks"]):
        g = lambda n: sd[f"blocks.{i}.{n}"]
        b.update(g1=g("norm1.weight"), g2=g("norm2.weight"),
                 Wq=g("attn.W_q.weight").T, Wk=g("attn.W_k.weight").T, Wv=g("attn.W_v.weight").T,
                 Wo=g("attn.W_o.weight").T, W=g("ffn.W.weight").T, V=g("ffn.V.weight").T, O=g("ffn.O.weight").T)
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        ref = tm(ids)[0].double().numpy()
    assert np.abs(ns["forward"](ids[0].numpy(), P) - ref).max() < 1e-4
    prompt = ids[0, :6].tolist()
    assert ns["generate"](P, prompt, 15, use_cache=True) == greedy_recompute(tm, torch.tensor([prompt]), 15)[0].tolist()


def test_kv_cache_generation_equals_full_recompute():
    ns = load_defs("ch05/05_kvcache.py")
    torch.manual_seed(0)
    model = ns["GPT"](vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6, ff_dim=1024)
    start = torch.tensor([[42]])
    out_nc = ns["generate_without_cache"](model, start, 30)
    model.clear_cache()
    out_c = ns["generate_with_cache"](model, start, 30)
    assert out_c.shape == (1, 31) and torch.equal(out_nc, out_c)
    model.clear_cache()
    worst = 0.0
    with torch.no_grad():                                       # 逐步比较 logits
        for t in range(1, 31):
            full = model(out_nc[:, :t])[:, -1]
            inc = model(out_nc[:, :t] if t == 1 else out_nc[:, t - 1:t], use_cache=True)[:, -1]
            worst = max(worst, (full - inc).abs().max().item())
    assert worst < 1e-4
    prompt8 = torch.tensor([[42, 7, 99, 3, 500, 17, 256, 8]])  # 多 token 的 prompt 也一样
    assert torch.equal(greedy_with_cache(model, prompt8, 20), greedy_recompute(model, prompt8, 20))


def _stepwise_logits(model, ids, n_prompt):
    """prefill ids[:n_prompt]，再沿 ids 逐个 decode；返回每一步最后位置的 logits。"""
    model.eval()
    model.clear_cache()
    with torch.no_grad():
        out = [model(ids[:, :n_prompt], use_cache=True)[:, -1]]
        out += [model(ids[:, t:t + 1], use_cache=True)[:, -1] for t in range(n_prompt, ids.shape[1] - 1)]
    return torch.stack(out)


def test_prefill_mask_bug_8b3911f_needs_two_prompt_tokens_and_two_layers():
    fixed, buggy = load_defs("ch05/05_kvcache.py"), load_defs("ch05/05_kvcache.py", replace=KV_BUG)
    one, eight = torch.tensor([[42]]), torch.tensor([[42, 7, 99, 3, 500, 17, 256, 8]])
    for n_layer in (1, 2):
        torch.manual_seed(0)
        good = fixed["GPT"](1000, 256, 384, 6, n_layer, 1024)
        bad = buggy["GPT"](1000, 256, 384, 6, n_layer, 1024)
        bad.load_state_dict(good.state_dict())
        for prompt in (one, eight):
            ids = greedy_with_cache(good, prompt, 12)
            gap = (_stepwise_logits(good, ids, prompt.shape[1]) - _stepwise_logits(bad, ids, prompt.shape[1])).abs().max()
            if prompt.shape[1] >= 2 and n_layer >= 2:
                assert gap > 1e-3, (n_layer, prompt.shape[1], gap)
            else:
                assert gap < 1e-5, (n_layer, prompt.shape[1], gap)
    # 原脚本的测速配置（6 层）：1 个 token 的 prompt 生成结果完全一样，8 个 token 的就不一样了
    torch.manual_seed(0)
    good = fixed["GPT"](1000, 256, 384, 6, 6, 1024)
    bad = buggy["GPT"](1000, 256, 384, 6, 6, 1024)
    bad.load_state_dict(good.state_dict())
    assert torch.equal(greedy_with_cache(good, one, 15), greedy_with_cache(bad, one, 15))
    assert not torch.equal(greedy_with_cache(good, eight, 15), greedy_with_cache(bad, eight, 15))


def test_cache_holds_rotated_keys_and_chunked_prefill_is_not_supported(dlfs6_path):
    import storybot.model as sb
    torch.manual_seed(0)
    m = sb.GPT(1000, 64, 128, 4, 2, 344).eval()
    ids = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        m.clear_cache()
        m(ids, use_cache=True)
        a = m.blocks[0].attn
        h = m.blocks[0].norm1(m.embed(ids))
        K = a.W_k(h).view(1, 8, 4, 32).transpose(1, 2)
        V = a.W_v(h).view(1, 8, 4, 32).transpose(1, 2)
        assert torch.allclose(a.k_cache, a.rope(K), atol=1e-6) and not torch.allclose(a.k_cache, K, atol=1e-6)
        assert torch.allclose(a.v_cache, V, atol=1e-6)
        assert m.blocks[0].attn.k_cache is not m.blocks[1].attn.k_cache
        # 第二次调用一次送进多个 token 时没有 mask：结果与一次性 prefill 不同
        full = m(ids)[:, -1]
        m.clear_cache()
        m(ids[:, :4], use_cache=True)
        two_chunks = m(ids[:, 4:], use_cache=True)[:, -1]
        assert (two_chunks - full).abs().max() > 1e-3


def test_measure_generation_time_clears_the_wrong_cache():
    ns = load_defs("ch05/05_kvcache.py")
    torch.manual_seed(0)
    model = ns["GPT"](1000, 256, 384, 6, 6, 1024)
    start = torch.tensor([[42]])
    offset = lambda: model.blocks[0].attn.cache_offset
    ns["measure_generation_time"](model, start, use_cache=True, num_tokens=5)
    assert offset() == 5
    ns["measure_generation_time"](model, start, use_cache=True, num_tokens=5)
    assert offset() == 10                                       # 第二次接着上一次的 cache 继续
    ns["measure_generation_time"](model, start, use_cache=False, num_tokens=1)
    assert offset() == 0                                        # 反倒是“不用 cache”的那次把它清空了
    first = ns["generate_with_cache"](model, start, 20)
    second = ns["generate_with_cache"](model, start, 20)        # 忘了 clear_cache()
    assert not torch.equal(first, second)
    # “# 固定シード”固定的只是起始 token：不设随机种子时，两次初始化的权重不同
    assert not torch.equal(ns["GPT"](1000, 256, 384, 6, 1, 1024).embed.weight,
                           ns["GPT"](1000, 256, 384, 6, 1, 1024).embed.weight)


def test_commented_out_consistency_check_needs_max_new_tokens():
    src = (DLFS6 / "ch05" / "05_kvcache.py").read_text(encoding="utf-8")
    block = src.split('"""')[-2]                                # 文件末尾用三引号注释掉的“出力の一致確認”
    assert "出力の一致確認" in block
    ns = load_defs("ch05/05_kvcache.py")
    torch.manual_seed(0)
    ns["model"] = ns["GPT"](vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=2, ff_dim=1024)
    ns["start_ids"] = torch.tensor([[42]])
    with pytest.raises(NameError, match="max_new_tokens"):     # 这个变量在 23046f6 中被删掉了
        exec(block, ns)
    ns["max_new_tokens"] = 20
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(block, ns)
    assert "✓ 出力が一致しました!" in out.getvalue()


def test_one_decode_step_makes_hundreds_of_small_torch_calls():
    from torch.overrides import TorchFunctionMode
    ns = load_defs("ch05/05_kvcache.py")
    torch.manual_seed(0)
    model = ns["GPT"](1000, 256, 384, 6, 6, 1024).eval()

    class Count(TorchFunctionMode):                              # 数一数 Python 层调用了多少次 torch 运算
        n = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            Count.n += 1
            return func(*args, **(kwargs or {}))

    with torch.no_grad():
        model(torch.randint(0, 1000, (1, 50)), use_cache=True)     # prefill 50 个 token
        with Count():
            model(torch.tensor([[7]]), use_cache=True)              # decode 1 个 token
    print("decode 一步调用的 torch 运算次数：", Count.n)
    assert 300 < Count.n < 500


def test_positions_freeze_past_max_context_len(dlfs6_path):
    import storybot.model as sb
    import webbot.model as wb
    torch.manual_seed(0)
    m = sb.GPT(100, 16, 32, 4, 1, 85).eval()
    with torch.no_grad():
        for T in (16, 17):                                      # max_context_len = 16
            m.clear_cache()
            assert m(torch.randint(0, 100, (1, T)), use_cache=True).shape == (1, T, 100)   # 17 也“正常”运行
        m.clear_cache()
        with pytest.raises(RuntimeError):
            m(torch.randint(0, 100, (1, 18)), use_cache=True)
        with pytest.raises(RuntimeError):                       # 04 的 RoPE 没有钳位：17 就直接报错
            improved_gpt()(100, 16, 32, 4, 1, 85)(torch.randint(0, 100, (1, 17)))
    RoPE = load_defs("ch05/05_kvcache.py", {"RoPE"})["RoPE"]
    rope = RoPE(10000, 8, 16)
    x = torch.randn(1, 1, 1, 8)
    assert torch.equal(rope(x, 20), rope(x, 15)) and not torch.equal(rope(x, 14), rope(x, 15))
    for mod, args, grows in [(sb, (1000, 16, 64, 4, 2, 172), True), (wb, (1000, 16, 64, 4, 2, 2, 172), False)]:
        m = mod.GPT(*args).eval()
        m.clear_cache()
        nxt = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            for _ in range(30):
                nxt = m(nxt, use_cache=True)[:, -1:].argmax(-1)
        assert m.blocks[0].attn.k_cache.shape[2] == (4 + 29 if grows else 16)   # StoryBot 一直变长，WebBot 滑动
    # WebBot 的 Attention：GQA（K、V 只有 n_kv_head 个头）+ F.scaled_dot_product_attention
    src = inspect.getsource(wb.MultiHeadAttention.forward)
    assert "scaled_dot_product_attention" in src and "enable_gqa=True" in src
    assert wb.GPT(1000, 16, 64, 4, 2, 1, 172).blocks[0].attn.W_k.out_features == 2 * 16
    with torch.device("meta"):                                  # 只数参数，不分配内存
        assert n_params(wb.GPT(50000, 1024, 768, 12, 4, 12, 2048)) == 152_316_672    # ch09 的 WebBot
        assert n_params(wb.GPT(50000, 1024, 768, 12, 12, 12, 2048)) == 161_753_856   # 若不用 GQA


def test_trained_storybot_reproduces_the_recorded_samples():
    if not STORYBOT_CKPT.is_file():
        pytest.skip("先运行 uv run python tools/fetch_data.py")
    recorded = json.loads(STORYBOT_RECORD.read_text(encoding="utf-8"))
    fresh = storybot_samples()
    assert fresh == recorded
    assert fresh["same_with_and_without_cache"] and fresh["utils_generate_same_text"]
    assert fresh["buggy"] != fresh["correct"]
    assert fresh["max_context_len"] < fresh["beyond_first_diff_position"] < fresh["beyond_total_tokens"]


def test_kv_benchmark_runs_and_the_recorded_cpu_numbers_have_the_expected_shape():
    res = measure("cpu", lengths=(1, 8), repeats=1, gen_lengths=(3,),
                  model_cfg=dict(max_context_len=16, embed_dim=32, n_head=4, n_layer=1, ff_dim=85))
    assert [r["t"] for r in res["per_step"]] == [1, 8] and res["generate"][0]["n"] == 3
    rec = json.loads((ROOT / "web" / "data" / "ch05" / "kv_timing_cpu.json").read_text(encoding="utf-8"))
    assert rec["model"]["n_layer"] == 6 and rec["device"] == "cpu"
    assert [r["t"] for r in rec["per_step"]] == [1, 25, 50, 100, 200, 400, 800]
    assert [g["n"] for g in rec["generate"]] == [50, 100, 200]
