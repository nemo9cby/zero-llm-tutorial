"""第 2 章 模型【基础篇】：本地验证（需要 PyTorch；只用 CPU）。

两类断言：
1. 原书脚本与 codebot/model.py 的真实行为（输出、形状、陷阱），直接导入 third_party/dlfs6 的代码；
2. 页面上 NumPy 验证单元里的实现，与原书的 PyTorch 实现在同样的权重、同样的输入上给出同样的结果。
运行：uv run pytest verify/test_ch02.py
"""
import hashlib
import json
import math
from functools import lru_cache

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from _ch02_book import DLFS6, ROOT, load_defs, page_namespace, run_in_process, run_script, source_lines

pytestmark = pytest.mark.usefixtures("dlfs6_path")
DATA = ROOT / "web" / "data" / "ch02"
CKPT = DLFS6 / "codebot" / "model_pretrain.pt"
# The author's checkpoint on Hugging Face (koki0702/zero-llm-data, codebot/model_pretrain.pt; the file page shows this sha256).
OFFICIAL_SHA256 = "3e32e4103c5d58a89bed032b2c497e86eb26c58dd0ca42eeef7565d5910a3ecb"
# This book's own run of ch03/01_pretrain.py (gpu/run_local.py --run codebot-3060ti); the playground serves it.
OURS_CKPT = ROOT / "runs" / "codebot-3060ti" / "work" / "codebot" / "model_pretrain.pt"
SQUARE = "def square(x):\n    return x * x\n\nprint(square(3))\n"     # the second snippet of attn_maps.json


@pytest.fixture(autouse=True, scope="module")
def _few_threads():
    """These models are tiny: extra intra-op threads only add contention (30 s with 8 threads vs 4 s with 1 on a busy
    8-core machine for the cheating experiment). The previous setting is restored for other test modules."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def f64(t):
    """torch tensor -> float64 NumPy array."""
    return t.detach().double().numpy()


@lru_cache(maxsize=None)
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def needs_checkpoint():
    """Skip without the official checkpoint; fail loudly if a different file sits in its place."""
    if not CKPT.is_file():
        pytest.skip("codebot/model_pretrain.pt 不存在：先运行 codebot/download.sh")
    assert sha256(CKPT) == OFFICIAL_SHA256, "codebot/model_pretrain.pt 不是原书作者发布的那个文件"


# ---------------------------------------------------------------- 2.1–2.3 原书脚本

def test_soft_dict_and_matrix_scripts_print_the_book_values():
    # 01 只用 NumPy：在新的解释器里原样运行；02 用 torch：在本进程里执行（省掉重新 import torch 的几十秒）
    assert run_script("ch02/01_soft_dict.py") == [
        "200",
        "新しい映画 (6, 4, 5) の予測評価: 78.66 点",
        "",
        "各映画の重み:",
        "映画 (8, 2, 3): 0.49%",
        "映画 (3, 9, 1): 0.00%",
        "映画 (1, 2, 9): 0.00%",
        "映画 (5, 5, 5): 26.71%",
        "映画 (7, 6, 2): 72.62%",
        "映画 (2, 7, 6): 0.18%",
        "映画 (9, 1, 1): 0.00%",
    ]
    assert run_in_process("ch02/02_attn_math.py") == [
        "映画 [6. 4. 5.] の予測評価: 78.66",
        "映画 [2. 8. 3.] の予測評価: 69.76",
        "映画 [4. 3. 7.] の予測評価: 61.20",
    ]


def test_scaling_script_saturates_and_variance_ratio_is_exactly_d():
    # 第 10 行导入 pyplot、第 41–46 行画图：跳过这几行，其余原样执行
    out = run_in_process("ch02/03_attn_scaling.py", [(1, 9), (11, 40), (47, 49)])
    assert out[0] == "tensor([0.0000e+00, 3.7835e-44, 1.0000e+00])"
    raw = float(out[-2].split(":")[1])
    scaled = float(out[-1].split(":")[1])
    assert abs(raw / scaled - 10) < 1e-9      # Var(x/√d) = Var(x)/d：代数恒等式，与采样无关
    assert 9 < raw < 11                       # 脚本没有设种子；10,000 个样本的方差约为 d = 10（标准误约 0.16）


def test_page_softmax_and_attention_match_torch_and_the_book():
    ns = page_namespace()
    rng = np.random.default_rng(0)
    s = rng.normal(size=(4, 7)) * 3
    for shift in (0.0, 100.0, 1000.0):
        ref = F.softmax(torch.tensor(s + shift), dim=-1).numpy()
        assert np.abs(ns["softmax"](s + shift) - ref).max() < 1e-12
    book = load_defs("ch02/02_attn_math.py")
    Q, K, V = rng.normal(size=(3, 3)), rng.normal(size=(7, 3)), rng.normal(size=(7, 1))
    out, _ = book.attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V))
    assert np.abs(ns["attention"](Q, K, V) - out.numpy()).max() < 1e-12


def _layer0_scores(m, ids):
    """Scaled layer-0 attention scores (1, H, C, C) of a codebot GPT in eval mode."""
    a, C = m.blocks[0].attn, ids.shape[1]
    with torch.no_grad():
        h = m.blocks[0].norm1(m.embed(ids) + m.pos_embed(torch.arange(C)))
        q, k = (lin(h).view(1, C, a.n_head, a.head_dim).transpose(1, 2) for lin in (a.W_q, a.W_k))
        return q @ k.transpose(-2, -1) / a.head_dim ** 0.5


def test_attention_is_nearly_uniform_at_initialization_but_not_after_training():
    from codebot.model import GPT
    torch.manual_seed(0)
    s = _layer0_scores(GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval(), torch.randint(0, 1000, (1, 256)))
    # 手算：q 的每个分量方差 = 384 × 0.02² = 0.154；分数的方差 = 64 × 0.154² / 64 = 0.154²，标准差也约为 0.154
    assert abs(s.std().item() - 384 * 0.02 ** 2) < 0.02
    w = F.softmax(s[0, :, -1], -1)                       # 每个头的最后一行：256 个可见的 key
    assert w.max().item() < 0.008                        # 6 个头都接近均匀的 1/256 = 0.0039
    assert ((w.max(-1).values > 0.005) & (w.max(-1).values < 0.007)).all()   # 正文：“都只有 0.006 左右”
    raw = s * 8                                          # 去掉 1/√64：初始化时缩放也在起作用
    assert abs(raw.std().item() - 1.21) < 0.01
    w_raw = F.softmax(raw[0, :, -1], -1).max(-1).values
    assert abs(w_raw.min().item() - 0.04) < 0.005 and abs(w_raw.max().item() - 0.13) < 0.005   # 0.04–0.13

    needs_checkpoint()
    trained = GPT.load_from(CKPT).eval()
    tokens = np.fromfile(DLFS6 / "codebot" / "tiny_codes.bin", dtype=np.uint16)
    window = lambda a: torch.tensor(tokens[a:a + 256].astype(np.int64))[None]
    visible = torch.tril(torch.ones(256, 256, dtype=torch.bool))
    s_init = s[0][:, visible].std().item()
    st = _layer0_scores(trained, window(1000))            # 正文的输入：tokens[1000:1256]
    sv = st[0][:, visible].std().item()
    assert sv > 10 * s_init                              # 训练后分数大了一个数量级……
    assert abs(sv - 2.45) < 0.01 and 15.5 < sv / s_init < 16.5   # ……具体是 2.45，约 16 倍
    mx = F.softmax(st[0, :, -1], -1).max(-1).values
    assert mx.min().item() > 0.15                        # 每个头都明显集中了
    assert abs(mx.min().item() - 0.19) < 0.005 and abs(mx.max().item() - 0.74) < 0.005   # 0.19–0.74
    unscaled = F.softmax(st[0, :, -1] * 8, -1).max(-1).values      # 原始点积 q·k = 分数 × 8
    assert abs(8 * sv - 19.6) < 0.1 and int((unscaled > 0.99).sum()) == 5   # 标准差约 19.6；6 个头里 5 个饱和
    mx5000 = F.softmax(_layer0_scores(trained, window(5000))[0, :, -1], -1).max(-1).values
    assert abs(mx5000.min().item() - 0.09) < 0.005 and abs(mx5000.max().item() - 0.71) < 0.005   # 换一段：0.09–0.71
    starts = np.random.default_rng(0).integers(0, len(tokens) - 256, 200)   # 随机 200 段：标准差都差不多
    stds = [_layer0_scores(trained, window(a))[0][:, visible].std().item() for a in starts]
    assert 2.4 < min(stds) and max(stds) < 2.65


# ---------------------------------------------------------------- 2.5 位置编码

def test_return_a_minus_b_and_b_minus_a_are_the_same_tokens():
    from codebot.tokenizer import BPETokenizer
    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    ab, ba = tok.encode("return a - b"), tok.encode("return b - a")
    assert [tok.decode([i]) for i in ab] == ["re", "turn", " a", " -", " b"]
    assert sorted(ab) == sorted(ba) and ab != ba          # 同样的 5 个 token，只是顺序不同


# ---------------------------------------------------------------- 2.6 掩码

def test_book_masked_attention_is_causal_matches_sdpa_and_the_page():
    torch.manual_seed(0)
    att = load_defs("ch02/06_attn_mask.py").Attention(256, 64)
    x = torch.randn(2, 5, 256)
    with torch.no_grad():
        y = att(x)
        ref = F.scaled_dot_product_attention(att.W_q(x), att.W_k(x), att.W_v(x), is_causal=True)
        assert ref.shape == (2, 5, 256)                      # value 宽 256 ≠ key 宽 64 也可以
        assert (y - ref).abs().max().item() < 1e-5
        assert torch.equal(y[:, 0], att.W_v(x)[:, 0])        # 位置 0 只能看到自己
        x2 = x.clone()
        x2[:, 3:] = torch.randn(2, 2, 256)                   # 改动“未来”的位置 3、4
        y2 = att(x2)
        assert torch.equal(y[:, :3], y2[:, :3])              # 过去的输出逐位不变
        assert (y[:, 3:] - y2[:, 3:]).abs().max().item() > 0.1
    port = page_namespace()["attn06"](f64(x), *(f64(m.weight) for m in (att.W_q, att.W_k, att.W_v)))
    assert np.abs(port - f64(y)).max() < 1e-5


def test_sdpa_is_causal_aligns_the_mask_to_the_top_left():
    torch.manual_seed(0)
    q, K, V = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8)
    out = F.scaled_dot_product_attention(q, K, V, is_causal=True)
    assert torch.allclose(out, V[:, :, :1])   # 1 个 query 对 5 个 key：只看得到 key 0


def test_without_the_mask_the_model_cheats_on_random_tokens():
    import codebot.model as cb

    class SwitchableMHA(cb.MultiHeadAttention):
        causal = True

        def forward(self, x):
            B, C, _ = x.shape
            H, D = self.n_head, self.head_dim
            Q, K, V = (lin(x).view(B, C, H, D).transpose(1, 2) for lin in (self.W_q, self.W_k, self.W_v))
            s = (Q @ K.transpose(-2, -1)) / D ** 0.5
            if self.causal:
                s = s.masked_fill(torch.tril(torch.ones(C, C)) == 0, float("-inf"))
            h = F.softmax(s, -1) @ V
            return self.W_o(h.transpose(1, 2).contiguous().view(B, C, H * D))

    def train(causal, V=8, C=8, E=32, H=2, steps=100):
        torch.manual_seed(0)
        m = cb.GPT(V, C, E, H, 1, 4 * E, 0.0)
        for blk in m.blocks:
            new = SwitchableMHA(E, H, E // H, 0.0)
            new.load_state_dict(blk.attn.state_dict())
            new.causal = causal
            blk.attn = new
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for _ in range(steps):
            seq = torch.randint(0, V, (64, C + 1))           # 独立均匀的随机 token：没有任何规律可学
            loss = F.cross_entropy(m(seq[:, :-1]).reshape(-1, V), seq[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            seq = torch.randint(0, V, (512, C + 1))
            return F.cross_entropy(m(seq[:, :-1]).transpose(1, 2), seq[:, 1:], reduction="none").mean(0)

    honest, cheat = train(True), train(False)
    assert abs(honest.mean().item() - math.log(8)) < 0.05       # 诚实的模型只能停在 ln V
    assert abs(cheat.mean().item() - math.log(8) / 8) < 0.1     # 作弊的模型接近 ln(V)/C
    assert cheat[:-1].max().item() < 0.02                       # 前 C-1 个位置偷看了答案：loss 接近 0（实测 < 0.004）
    assert cheat[-1].item() > 2.0                               # 最后一个位置的答案不在输入里：仍在 2 以上（实测 2.10）


# ---------------------------------------------------------------- 2.7 Value 低秩化

def test_value_lowrank_07_equals_06_with_the_product_matrix():
    torch.manual_seed(0)
    a7 = load_defs("ch02/07_attn_value.py").Attention(256, 64)
    a6 = load_defs("ch02/06_attn_mask.py").Attention(256, 64)
    with torch.no_grad():
        a6.W_q.weight.copy_(a7.W_q.weight)
        a6.W_k.weight.copy_(a7.W_k.weight)
        W_eff = a7.W_o.weight @ a7.W_v.weight                # (256, 64) @ (64, 256) -> (256, 256)
        a6.W_v.weight.copy_(W_eff)
        x = torch.randn(2, 5, 256)
        y7 = a7(x)
        assert (y7 - a6(x)).abs().max().item() < 1e-5
        assert torch.linalg.matrix_rank(W_eff).item() == 64
    assert sum(p.numel() for p in a6.parameters()) == 98_304  # 2·E·D + E²
    assert sum(p.numel() for p in a7.parameters()) == 65_536  # 4·E·D
    port = page_namespace()["attn07"](f64(x), *(f64(m.weight) for m in (a7.W_q, a7.W_k, a7.W_v, a7.W_o)))
    assert np.abs(port - f64(y7)).max() < 1e-5


# ---------------------------------------------------------------- 2.8 多头

def test_multihead_reshape_traps():
    import codebot.model as cb
    B, C, E, H, D = 2, 4, 16, 3, 8
    torch.manual_seed(0)
    Q = nn.Linear(E, H * D, bias=False)(torch.randn(B, C, E)).detach()
    Qh = Q.view(B, C, H, D).transpose(1, 2)
    assert tuple(Q.shape) == (2, 4, 24) and tuple(Qh.shape) == (2, 3, 4, 8)
    assert not Qh.is_contiguous()
    h = torch.randn(2, 8, 10, 64).transpose(1, 2)            # (B, C, H, D)，不连续
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        h.view(2, 10, 512)
    assert torch.equal(h.contiguous().view(2, 10, 512), h.reshape(2, 10, 512))

    # 拆头时不先 transpose：不报错，但把位置和头搅在一起，未来的信息漏进了过去。
    m = cb.MultiHeadAttention(16, 4, 4, 0.0).eval()
    x = torch.randn(1, 6, 16)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 16)

    def mha(x, right):
        B, C, _ = x.shape
        split = (lambda t: t.view(B, C, 4, 4).transpose(1, 2)) if right else (lambda t: t.view(B, 4, C, 4))
        q, k, v = (split(lin(x)) for lin in (m.W_q, m.W_k, m.W_v))
        s = (q @ k.transpose(-2, -1)) / 2.0
        s = s.masked_fill(torch.tril(torch.ones(C, C)) == 0, float("-inf"))
        return m.W_o((F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, C, 16))

    with torch.no_grad():
        assert torch.allclose(mha(x, True), m(x), atol=1e-6)  # 正确的拆法就是 codebot 的 MHA
        assert torch.equal(mha(x, True)[:, :4], mha(x2, True)[:, :4])
        assert (mha(x, False)[:, :4] - mha(x2, False)[:, :4]).abs().max().item() > 1e-3
    # （E 不能被 H 整除的坑在 2.9.4 节讲，见 test_block_matches_page_residual_identity_and_key_names）


def test_codebot_mha_matches_sdpa_torch_mha_the_book_and_the_page():
    import codebot.model as cb
    torch.manual_seed(0)
    m = cb.MultiHeadAttention(512, 8, 64).eval()
    x = torch.randn(2, 10, 512)
    with torch.no_grad():
        y = m(x)
        Q, K, V = (lin(x).view(2, 10, 8, 64).transpose(1, 2) for lin in (m.W_q, m.W_k, m.W_v))
        sdpa = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        sdpa = m.W_o(sdpa.transpose(1, 2).reshape(2, 10, 512))
        t = nn.MultiheadAttention(512, 8, bias=False, batch_first=True).eval()
        t.in_proj_weight.copy_(torch.cat([m.W_q.weight, m.W_k.weight, m.W_v.weight]))
        t.out_proj.weight.copy_(m.W_o.weight)
        ref, _ = t(x, x, x, attn_mask=torch.triu(torch.ones(10, 10, dtype=torch.bool), 1), need_weights=False)
        book = load_defs("ch02/08_multi_head.py").MultiHeadAttention(512, 8, 64).eval()
        book.load_state_dict(m.state_dict())
        assert torch.equal(book(x), y)                       # 08 的类与 codebot 的类是同一份代码
    assert (y - sdpa).abs().max().item() < 1e-5
    assert (y - ref).abs().max().item() < 1e-5
    assert sum(p.numel() for p in m.parameters()) == 4 * 512 * 8 * 64 == 1_048_576
    # 2.8.2：512、8、64 正是原书 08 末尾使用示例的配置；CodeBot 的 384、6、64 是 589,824
    assert source_lines("ch02/08_multi_head.py", 99, 101) == "embed_dim = 512\nn_head = 8\nhead_dim = 64"
    assert sum(p.numel() for p in cb.MultiHeadAttention(384, 6, 64).parameters()) == 589_824
    port = page_namespace()["mha"](f64(x), *(f64(lin.weight) for lin in (m.W_q, m.W_k, m.W_v, m.W_o)), 8)
    assert np.abs(port - f64(y)).max() < 1e-5


def test_dropout_makes_train_mode_random():
    import codebot.model as cb
    torch.manual_seed(0)
    m = cb.MultiHeadAttention(512, 8, 64)
    assert m.training                                        # 新建的模块处于 train 模式
    x = torch.randn(2, 10, 512)
    with torch.no_grad():
        assert not torch.equal(m(x), m(x))                   # dropout：两次前向结果不同
        m.eval()
        assert torch.equal(m(x), m(x))
        w = nn.Dropout(0.1)(F.softmax(torch.randn(64, 10), -1))
    assert (w.sum(-1) - 1).abs().max().item() > 0.01         # 被 dropout 的注意力行之和不再是 1


# ---------------------------------------------------------------- 2.9 LayerNorm / GELU / Block

def test_layernorm_book_torch_and_page_agree():
    torch.manual_seed(0)
    ln = load_defs("ch02/09_norm_gelu.py").LayerNorm(384)
    ref = nn.LayerNorm(384)
    with torch.no_grad():
        g, b = torch.randn(384), torch.randn(384)
        ln.gamma.copy_(g)
        ln.beta.copy_(b)
        ref.weight.copy_(g)
        ref.bias.copy_(b)
        x = torch.randn(4, 16, 384) * 3 + 1.5
        y = ref(x)
        assert (ln(x) - y).abs().max().item() < 1e-5
        assert (ln(x) - y).abs().max().item() < 2e-6                    # 正文：float32 下最大差不到 2e-6（不是逐位相同）

        def z(unbiased):
            var = x.var(-1, keepdim=True, unbiased=unbiased)
            return (x - x.mean(-1, keepdim=True)) / torch.sqrt(var + 1e-5)
        assert abs((z(True) / z(False)).mean().item() - math.sqrt(383 / 384)) < 1e-5   # 忘写 unbiased=False 的后果
    assert np.abs(page_namespace()["layer_norm"](f64(x), f64(g), f64(b)) - f64(y)).max() < 1e-5

    x0 = torch.randn(4, 384)
    other = torch.cat([x0[:1], torch.randn(3, 384)])
    bn = nn.BatchNorm1d(384).train()
    with torch.no_grad():
        assert (ref(x0)[0] - ref(other)[0]).abs().max().item() < 1e-6     # LN：只看自己这个 token
        assert torch.equal(ref.train()(x0), ref.eval()(x0))               # LN：训练和推理是同一套计算
        assert (bn(x0)[0] - bn(other)[0]).abs().max().item() > 0.1        # BN 训练时：受同一 batch 里别人的影响
        bn.eval()                                                          # BN 推理时：改用累计的均值和方差
        assert torch.equal(bn(x0)[0], bn(other)[0])


def test_gelu_versions():
    import codebot.model as cb
    t = torch.linspace(-6, 6, 120001)
    g_book = load_defs("ch02/09_norm_gelu.py").GELU()(t)
    assert (g_book - nn.GELU(approximate="tanh")(t)).abs().max().item() < 1e-6
    assert 4.6e-4 < (g_book - nn.GELU()(t)).abs().max().item() < 4.8e-4    # tanh 近似 vs erf
    act = cb.FFN(8, 32, 0.1).layers[1]
    assert isinstance(act, nn.GELU) and act.approximate == "none"          # codebot 用的是 erf 版
    graph = {"np": np}
    exec(source_lines("ch02/graph.py", 5, 11), graph)                      # 只执行 graph.py 的公式部分
    assert np.abs(graph["gelu"] - nn.GELU(approximate="tanh")(torch.tensor(graph["x"])).numpy()).max() < 1e-12
    ns, td = page_namespace(), t.double()
    assert np.abs(ns["gelu_tanh"](td.numpy()) - nn.GELU(approximate="tanh")(td).numpy()).max() < 1e-12
    assert np.abs(ns["gelu_erf"](td.numpy()) - nn.GELU()(td).numpy()).max() < 1e-12


def test_block_matches_page_residual_identity_and_key_names():
    import codebot.model as cb
    torch.manual_seed(0)
    blk = cb.Block(384, 6, 1536).eval()
    x = torch.randn(2, 16, 384)
    P = {f"blocks.0.{k}": f64(v) for k, v in blk.state_dict().items()}
    with torch.no_grad():
        y = blk(x)
    assert np.abs(page_namespace()["block_forward"](f64(x), P, 0, 6) - f64(y)).max() < 1e-4
    with torch.no_grad():
        blk.attn.W_o.weight.zero_()
        blk.ffn.layers[2].weight.zero_()
        blk.ffn.layers[2].bias.zero_()
        assert torch.equal(blk(x), x)                        # 两个子层都写入 0：残差流原样通过
    b09 = load_defs("ch02/09_norm_gelu.py").Block(384, 6)
    assert [k for k in b09.state_dict() if "norm1" in k] == ["norm1.gamma", "norm1.beta"]
    assert [k for k in blk.state_dict() if "norm1" in k] == ["norm1.weight", "norm1.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        b09.load_state_dict(blk.state_dict())

    # head_dim = embed_dim // n_head 不检查整除：E=100、H=3 时悄悄变成 3 × 33 = 99 维，照样运行
    odd = cb.Block(100, 3, 400).eval()
    assert odd.attn.head_dim == 33 and tuple(odd.attn.W_q.weight.shape) == (99, 100)
    assert tuple(odd(torch.randn(1, 4, 100)).shape) == (1, 4, 100)


# ---------------------------------------------------------------- 2.10 GPT-2

def test_param_counts_tying_and_the_page_formula():
    from codebot.model import GPT
    m = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
    assert sum(p.numel() for p in m.parameters()) == 11_121_640
    assert sum(t.numel() for t in m.state_dict().values()) == 11_505_640    # 共享矩阵被算了两次
    assert m.embed.weight is m.unembed.weight
    assert m.unembed.bias is not None and m.unembed.bias.numel() == 1000
    assert sum(p.numel() for p in m.blocks[0].parameters()) == 1_772_928
    with torch.device("meta"):
        g2 = GPT(50257, 1024, 768, 12, 12, 3072, 0.1)
    assert sum(p.numel() for p in g2.parameters()) == 124_453_201           # GPT-2 small 的超参
    n_params = page_namespace()["n_params"]
    assert n_params(1000, 256, 384, 6, 1536, tied=False) == 11_505_640
    for V, ctx, E, H, L, ff in [(1000, 256, 384, 6, 6, 1536), (50257, 1024, 768, 12, 12, 3072),
                                (37, 9, 24, 2, 3, 40), (10000, 256, 512, 16, 4, 2048)]:
        with torch.device("meta"):
            g = GPT(V, ctx, E, H, L, ff, 0.0)
        assert n_params(V, ctx, E, L, ff) == sum(p.numel() for p in g.parameters())


def test_tied_matrix_keeps_the_linear_init():
    from codebot.model import GPT

    class Probe(GPT):                                        # 两种初始化各设一个可区分的常数
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 2.0)
            elif isinstance(module, nn.Embedding):
                nn.init.constant_(module.weight, 1.0)
    p = Probe(50, 8, 16, 2, 1, 64, 0.0)
    children = [n for n, _ in p.named_children()]
    assert children.index("embed") < children.index("unembed")   # apply() 先访问 embed，后访问 unembed
    assert p.embed.weight.eq(2.0).all() and p.pos_embed.weight.eq(1.0).all()   # 共享矩阵最终是 Linear 的初始化


def test_init_loss_is_ln_v_plus_half_the_logit_variance():
    from codebot.model import GPT
    torch.manual_seed(0)
    m = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval()
    assert abs(m.blocks[0].attn.W_q.weight.std().item() - 0.02) < 1e-3
    # 残差分支的输出投影（W_o、FFN 的第二个 Linear）也是 0.02：CodeBot 没有 GPT-2 论文那种 1/√N 的缩放
    for w in (m.blocks[0].attn.W_o.weight, m.blocks[5].ffn.layers[2].weight):
        assert abs(w.std().item() - 0.02) < 1e-3
    assert m.unembed.bias.abs().max().item() == 0 and bool(m.blocks[0].norm1.weight.eq(1).all())
    ids = torch.randint(0, 1000, (8, 257))
    with torch.no_grad():
        logits = m(ids[:, :-1])
    assert logits.min().item() < 0 and not torch.allclose(logits.sum(-1), torch.ones(8, 256), atol=1e-3)   # 输出的是 logits，没做 softmax
    ce = F.cross_entropy(logits.reshape(-1, 1000), ids[:, 1:].reshape(-1)).item()
    nll = -F.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).mean().item()
    assert abs(ce - nll) < 1e-5                              # softmax 由 F.cross_entropy 在内部完成
    s = logits.std().item()
    assert abs(s - 0.02 * math.sqrt(384)) < 0.02             # 手算：σ ≈ 0.02 × √384 = 0.392
    assert abs(ce - (math.log(1000) + s * s / 2)) < 0.01     # ln V + σ²/2
    assert 6.95 < ce < 7.02


def test_whole_model_context_limit_causality_and_eval_mode():
    from codebot.model import GPT
    torch.manual_seed(0)
    m = GPT(1000, 256, 384, 6, 6, 1536, 0.1)
    with pytest.raises(IndexError):
        m(torch.randint(0, 1000, (1, 257)))                  # 位置表只有 256 行
    ids = torch.randint(0, 1000, (4, 64))
    with torch.no_grad():
        assert not torch.equal(m(ids), m(ids))               # train 模式：dropout 开着
        m.eval()
        assert torch.equal(m(ids), m(ids))
        a = torch.randint(0, 1000, (1, 256))
        b = a.clone()
        b[0, 100] = (b[0, 100] + 1) % 1000
        d = (m(a) - m(b)).abs()
    assert d[:, :100].max().item() == 0.0 and d[:, 100:].max().item() > 0.1


def test_save_load_roundtrip_and_the_official_checkpoint(tmp_path):
    from codebot.model import GPT
    torch.manual_seed(0)
    m = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval()
    path = tmp_path / "m.pt"
    m.save(path)
    assert 11_121_640 * 4 < path.stat().st_size < 11_121_640 * 4 * 1.002    # 共享矩阵只存了一份
    r = GPT.load_from(path).eval()
    assert r.embed.weight is r.unembed.weight
    ids = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        assert torch.equal(m(ids), r(ids))
    needs_checkpoint()
    assert CKPT.stat().st_size == 44_516_353
    assert sha256(CKPT) == OFFICIAL_SHA256 and OFFICIAL_SHA256.startswith("3e32e410")   # 与 Hugging Face 上的文件相同
    ck = torch.load(CKPT, map_location="cpu")
    assert {k: v for k, v in ck.items() if k != "model_state_dict"} == {
        "vocab_size": 1000, "max_context_len": 256, "embed_dim": 384, "n_head": 6,
        "n_layer": 6, "ff_dim": 1536, "dropout_rate": 0.1}


def test_book_10_gpt2_is_the_codebot_gpt():
    from codebot.model import GPT
    assert run_in_process("ch02/10_gpt2.py") == ["出力形状: torch.Size([1, 256, 1000])"]
    torch.manual_seed(0)
    cb = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval()
    g10 = load_defs("ch02/10_gpt2.py").GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval()
    g10.load_state_dict(cb.state_dict())
    ids = torch.randint(0, 1000, (2, 64))
    with torch.no_grad():
        assert torch.equal(g10(ids), cb(ids))


def test_page_numpy_gpt_matches_the_codebot_gpt():
    from codebot.model import GPT
    gpt_forward = page_namespace()["gpt_forward"]
    Z = np.load(DATA / "tiny_ref.npz")
    weights = [k for k in Z.files if k not in ("ids", "ref_logits")]
    tiny = GPT(50, 16, 32, 4, 2, 128, 0.1).eval()
    tiny.load_state_dict({k: torch.tensor(Z[k]) for k in weights})
    with torch.no_grad():
        ref = tiny(torch.tensor(Z["ids"][None]))[0].numpy()
    assert np.abs(ref - Z["ref_logits"]).max() < 1e-5        # 页面附带的参考 logits 就是 codebot 的输出
    P = {k: Z[k].astype(np.float64) for k in weights}
    assert np.abs(gpt_forward(Z["ids"].tolist(), P, 2, 4) - ref).max() < 1e-5
    torch.manual_seed(1)
    other = GPT(70, 20, 48, 3, 3, 96, 0.1).eval()            # 换一个形状再比一次
    ids = torch.randint(0, 70, (1, 20))
    with torch.no_grad():
        ref2 = other(ids)[0].numpy()
    P2 = {k: f64(v) for k, v in other.state_dict().items()}
    assert np.abs(gpt_forward(ids[0].tolist(), P2, 3, 3) - ref2).max() < 1e-5


# ---------------------------------------------------------------- 页面数据 == 真实 checkpoint

def _attention_maps(model, ids):
    """Independent re-derivation (no hooks): walk the blocks by hand in eval mode."""
    x = model.embed(torch.tensor([ids])) + model.pos_embed(torch.arange(len(ids)))
    maps = []
    for blk in model.blocks:
        a, h = blk.attn, blk.norm1(x)
        C, H, D = len(ids), a.n_head, a.head_dim
        q = a.W_q(h).view(1, C, H, D).transpose(1, 2)
        k = a.W_k(h).view(1, C, H, D).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / D ** 0.5
        future = ~torch.tril(torch.ones(C, C, dtype=torch.bool))
        maps.append(F.softmax(s.masked_fill(future, float("-inf")), -1)[0])
        x = blk(x)
    return torch.stack(maps), model.unembed(model.norm(x))[0]


def test_attention_map_data_is_what_the_checkpoint_computes():
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer
    needs_checkpoint()
    data = json.loads((DATA / "attn_maps.json").read_text(encoding="utf-8"))
    assert data["meta"]["bytes"] == CKPT.stat().st_size
    assert data["meta"]["sha256"] == sha256(CKPT)[:16]                       # 页面数据来自官方权重
    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    model = GPT.load_from(CKPT).eval()
    for snip in data["snippets"]:
        ids = tok.encode(snip["text"])
        assert ids == snip["ids"] and [tok.decode([i]) for i in ids] == snip["tokens"]
        with torch.no_grad():
            W, logits = _attention_maps(model, ids)
        assert np.abs(W.numpy() - np.array(snip["attn"])).max() < 6e-5      # 存的是保留 4 位小数的值
        assert [tok.decode([int(i)]) for i in logits.argmax(-1)] == snip["pred"]


def _next_token_probs(model, tok, text):
    with torch.no_grad():
        return F.softmax(model(torch.tensor([tok.encode(text)]))[0], -1)


def test_context_not_one_head_completes_the_name():
    """2.8.1: the attention map of L4H0 is a clue, not the mechanism. Ablating that one head barely changes the
    completion of `print(s` -> 'qu'; taking `square` out of the context does."""
    from codebot.model import GPT
    from codebot.tokenizer import BPETokenizer
    needs_checkpoint()
    tok = BPETokenizer.load_from(str(DLFS6 / "codebot" / "merge_rules.pkl"))
    qu, are, lpar = (tok.encode(t)[0] for t in ("qu", "are", "("))
    model = GPT.load_from(CKPT).eval()
    assert json.loads((DATA / "attn_maps.json").read_text(encoding="utf-8"))["snippets"][1]["text"] == SQUARE
    ids = tok.encode(SQUARE)
    assert [tok.decode([i]) for i in ids[16:19]] == ["s", "qu", "are"]          # 第二次出现的 square
    p = _next_token_probs(model, tok, SQUARE)
    full = [p[16, qu].item(), p[17, are].item(), p[18, lpar].item()]
    assert full[0] > 0.99 and abs(full[0] - 0.997) < 0.001                      # 正文：0.997
    with torch.no_grad():
        model.blocks[4].attn.W_o.weight[:, 0:64] = 0      # L4H0 的输出进入 W_o 的那 64 列：置 0 就是去掉这个头
    p_abl = _next_token_probs(model, tok, SQUARE)
    ablated = [p_abl[16, qu].item(), p_abl[17, are].item(), p_abl[18, lpar].item()]
    assert abs(ablated[0] - 0.994) < 0.001 and min(ablated) > 0.9                # 0.997 -> 0.994：这个头几乎不是必需的
    assert [int(p_abl[t].argmax()) for t in (16, 17, 18)] == [qu, are, lpar]     # 三处的预测都没变

    model = GPT.load_from(CKPT).eval()
    cube = "def cube(x):\n    return x * x * x\n\nprint(square(3))\n"
    toks = [tok.decode([t]) for t in tok.encode(cube)]
    i = next(k for k in range(2, len(toks)) if toks[k - 2:k + 1] == ["print", "(", "s"])   # print( 之后的 's'
    p_cube = _next_token_probs(model, tok, cube)[i, qu].item()
    p_alone = _next_token_probs(model, tok, "print(s")[-1, qu].item()          # 没有任何前文
    assert abs(p_cube - 0.14) < 0.005 and abs(p_alone - 0.07) < 0.005            # 前文里没有 square，概率就掉下来了


def test_head_roles_belong_to_one_checkpoint():
    """2.8.1: the head labels hold for the official checkpoint only. The book's own run of the same script
    (the checkpoint the playground serves) puts the same behaviours in other heads."""
    from codebot.model import GPT
    if not OURS_CKPT.is_file():
        pytest.skip("runs/codebot-3060ti 的 checkpoint 不存在：先运行 gpu/run_local.py --run codebot-3060ti")
    manifest = json.loads((ROOT / "web" / "data" / "playground" / "pretrain" / "manifest.json").read_text(encoding="utf-8"))
    assert sha256(OURS_CKPT) == manifest["provenance"]["checkpoint_sha256"] != OFFICIAL_SHA256   # 工作台用的就是它
    data = json.loads((DATA / "attn_maps.json").read_text(encoding="utf-8"))
    ids = data["snippets"][0]["ids"]                                                 # def add(a, b): ...
    with torch.no_grad():
        A, _ = _attention_maps(GPT.load_from(OURS_CKPT).eval(), ids)
    A = A.numpy()
    prev = np.array([[np.mean([A[l, h, t, t - 1] for t in range(1, 13)]) for h in range(6)] for l in range(6)])
    top2 = [divmod(int(i), 6) for i in np.argsort(prev.ravel())[::-1][:2]]
    assert top2 == [(3, 0), (0, 0)]                      # 最强的 previous-token head 是 L3H0，其次是 L0H0
    assert round(float(prev[3, 0]), 2) == 0.60 and round(float(prev[0, 0]), 2) == 0.48
    assert abs(prev[4, 3] - 0.35) < 0.01                                             # 官方权重里的 L4H3 是 0.57
    assert sum(int(A[5, 2, t].argmax()) == 1 for t in range(1, 13)) == 0             # L5H2 一次也不看 ' add'
    official = np.array(data["snippets"][0]["attn"])
    off_prev = np.array([[np.mean([official[l, h, t, t - 1] for t in range(1, 13)]) for h in range(6)] for l in range(6)])
    assert tuple(int(i) for i in np.unravel_index(off_prev.argmax(), off_prev.shape)) == (4, 3)


def test_position_embedding_data_is_what_the_checkpoint_computes():
    from codebot.model import GPT
    needs_checkpoint()
    data = json.loads((DATA / "pos_embed_cos.json").read_text(encoding="utf-8"))

    def mean_cos(W):
        P = F.normalize(W.detach(), dim=-1)
        return np.array([(P[:-k] * P[k:]).sum(-1).mean().item() for k in range(1, P.shape[0])])
    assert np.abs(mean_cos(GPT.load_from(CKPT).pos_embed.weight) - data["trained"]).max() < 1e-4
    assert np.abs(np.array(data["fresh"][:128])).max() < 0.03    # 随机初始化：方向互不相关
    if data["meta"]["torch"] == torch.__version__:
        torch.manual_seed(0)
        fresh = GPT(1000, 256, 384, 6, 6, 1536, 0.1).pos_embed.weight
        assert np.abs(mean_cos(fresh) - data["fresh"]).max() < 1e-4
