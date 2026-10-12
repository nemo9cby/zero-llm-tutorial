"""附录 A–E：本地验证（需要 PyTorch 和 third_party/dlfs6）。

页面上的验证单元只用 NumPy。这里拿真实的 DLFS6 代码检查同一结论；页面上的代码不在这里重抄，
而是由 _appendix_cells.page_namespace 从 content/appendix-*.md 里读出来原样执行。
DLFS6 的训练脚本在模块顶层就开始训练，所以只用 ast 抽出需要的函数和类，不运行脚本本身。
"""
import ast
import random
from collections import Counter

import pytest
from _appendix_cells import cell_source, page_namespace

# 不往 third_party/dlfs6 里写 __pycache__ 由 conftest.py 统一处理；本文件的张量都建在 CPU 上。


def _load_defs(path, names, ns):
    """只执行脚本里指定的顶层函数和类的定义（脚本本身在模块顶层就开始训练，不能直接 import）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), f"{path.name} 里找不到 {names}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _script_constant(path, name):
    """读出脚本顶层的常量赋值 name = <字面量>（不执行脚本）。"""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{path.name} 里没有顶层常量 {name}")


def _called_names(path):
    """脚本里所有以名字直接调用的函数，例如 generate(...)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _is_utf8(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# ---------------------------------------------------------------- 附录 A：UTF-8

def test_a_decode_matches_dlfs6(dlfs6_path):
    from storybot.tokenizer import BPETokenizer  # WebBot 复用 StoryBot 的分词器（ch07/01_bpe_train.py）

    utf8_decode = page_namespace("appendix-a", upto="A-j")["utf8_decode"]
    tok = BPETokenizer.load_from(str(dlfs6_path / "webbot" / "merge_rules.pkl"))
    assert tok.vocab_size == 50_000

    # 合并出来的 token 里，单独解码不合法的有 157 个：150 个以不完整的序列结尾，7 个以后续字节开头
    reasons = Counter()
    for i in range(256, tok.vocab_size):
        try:
            tok.id_to_bytes[i].decode("utf-8")
        except UnicodeDecodeError as e:
            reasons[e.reason] += 1
    assert reasons == {"unexpected end of data": 150, "invalid start byte": 7}
    assert tok.id_to_bytes[290] == "’".encode("utf-8")[:2] == b"\xe2\x80"
    assert all(c.encode("utf-8")[:2] == b"\xe2\x80" for c in "’“”—")   # 弯引号与破折号共用这两个字节
    assert tok.id_to_bytes[6091] == "€".encode("utf-8")[1:] == b"\x82\xac"

    # 本页的解码器与 DLFS6 的 decode（errors="replace"）逐字相同：先逐个 token，再随机拼成序列
    for i in range(tok.vocab_size):
        b = tok.id_to_bytes[i]
        assert utf8_decode(b, errors="replace") == tok.decode([i]), i
        try:
            ref = b.decode("utf-8")
        except UnicodeDecodeError as e:
            ref = (e.start, e.end, e.reason)
        try:
            mine = utf8_decode(b)
        except UnicodeDecodeError as e:
            mine = (e.start, e.end, e.reason)
        assert mine == ref, i
    partial = [i for i in range(tok.vocab_size) if not _is_utf8(tok.id_to_bytes[i])]
    rng = random.Random(0)
    for _ in range(3000):
        ids = [rng.choice(partial) if rng.random() < 0.5 else rng.randrange(tok.vocab_size)
               for _ in range(rng.randint(1, 8))]
        joined = b"".join(tok.id_to_bytes[i] for i in ids)
        assert utf8_decode(joined, errors="replace") == tok.decode(ids), ids


def test_a_corpora_are_ascii(dlfs6_path):
    """A.1：DLFS6 的三份语料几乎全是 ASCII。OWT 只有分词之后的 owt_valid.bin，用 WebBot 的词表把 token 换回字节再数。"""
    import numpy as np
    from storybot.tokenizer import BPETokenizer

    paths = [dlfs6_path / p for p in ("codebot/tiny_codes.txt", "storybot/tiny_stories_valid.txt",
                                      "webbot/owt_valid.bin", "webbot/merge_rules.pkl")]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"数据文件不在本地：{missing}（运行 tools/fetch_data.py）")
    codes = np.fromfile(paths[0], dtype=np.uint8)
    stories = np.fromfile(paths[1], dtype=np.uint8)
    assert codes.size == 6_487_033 and int((codes >= 0x80).sum()) == 0          # TinyCodes：纯 ASCII
    assert round(float((stories >= 0x80).mean()) * 100, 2) == 0.06             # TinyStories 验证集：0.06%

    tok = BPETokenizer.load_from(str(paths[3]))
    lengths = np.array([len(tok.id_to_bytes[i]) for i in range(tok.vocab_size)])
    high = np.array([sum(x >= 0x80 for x in tok.id_to_bytes[i]) for i in range(tok.vocab_size)])
    counts = np.bincount(np.fromfile(paths[2], dtype=np.uint16), minlength=tok.vocab_size)
    assert counts.size == tok.vocab_size == 50_000
    assert round(float(counts @ high) / float(counts @ lengths) * 100, 1) == 1.4  # OWT 验证集：1.4%


def test_a_chunk_boundaries(dlfs6_path, tmp_path):
    from storybot.tokenizer import find_chunk_boundaries

    end = b"<|endoftext|>"
    docs = ["从前有一只猫，它喜欢晒太阳。", "Hello, 世界! 😁 naïve café", "def f(x):\n    return x  # ✓",
            "日本語のテキストも混ぜておく。"] * 300
    blob = end.join(d.encode("utf-8") for d in docs)
    path = tmp_path / "corpus.txt"
    path.write_bytes(blob)

    naive_bad = 0
    for num_chunks in (3, 7, 16, 64):
        bounds = find_chunk_boundaries(str(path), num_chunks)
        assert bounds[0] == 0 and bounds[-1] == len(blob)
        for a, b in zip(bounds, bounds[1:]):
            assert a == 0 or blob[a:a + len(end)] == end   # 每个内部切点都落在 <|endoftext|> 上
            blob[a:b].decode("utf-8")                       # 严格解码：切开字符就会抛异常
        size = len(blob) // num_chunks
        naive = [i * size for i in range(num_chunks)] + [len(blob)]
        naive_bad += sum(not _is_utf8(blob[a:b]) for a, b in zip(naive, naive[1:]))
    assert naive_bad > 0  # 对照：不挪切点的话，确实会把字符切开


# ---------------------------------------------------------------- 附录 B：Attention 缩放

def _causal_attention(q, k, v, scale):
    import torch
    C = q.size(-2)
    causal = torch.tril(torch.ones(C, C, dtype=torch.bool))
    scores = (q @ k.transpose(-2, -1) * scale).masked_fill(~causal, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def test_b_codebot_scores(dlfs6_path):
    import math
    import torch
    from codebot.model import GPT

    pred = 384 * 0.02**2 * 0.0008 / (0.0008 + 1e-5)       # 附录 B.3 的预测：0.1517
    measured = {0: 0.1511, 1: 0.1514, 2: 0.1515}             # 正文引用的三个种子的实测值
    for seed, expected in measured.items():
        torch.manual_seed(seed)
        model = GPT(1000, 256, 384, 6, 6, 1536, 0.1).eval()   # ch03/01_pretrain.py 的配置
        attn = model.blocks[0].attn
        assert (attn.n_head, attn.head_dim) == (6, 64)
        ids = torch.randint(0, 1000, (4, 256))
        with torch.no_grad():
            h = model.blocks[0].norm1(model.embed(ids) + model.pos_embed(torch.arange(256)))
            B, C, _ = h.shape
            q, k, v = (w(h).view(B, C, 6, 64).transpose(1, 2) for w in (attn.W_q, attn.W_k, attn.W_v))

            def full(scale):
                return attn.W_o(_causal_attention(q, k, v, scale).transpose(1, 2).reshape(B, C, 384))

            assert torch.allclose(attn(h), full(1 / math.sqrt(64)), atol=1e-6)      # 除以 √64 = 8
            assert not torch.allclose(attn(h), full(1 / math.sqrt(384)), atol=1e-4)  # 不是 √384

            std = (q @ k.transpose(-2, -1) / math.sqrt(64)).std().item()
            assert abs(std - expected) < 1e-4, (seed, std)
            assert abs(std / pred - 1) < 0.01, (seed, std, pred)   # 与预测相差不到 1%


def test_b_sdpa_default_scale(dlfs6_path):
    import math
    import torch
    from webbot.model import MultiHeadAttention

    torch.manual_seed(0)
    E, H, KV, D, B, C = 128, 4, 2, 32, 2, 10            # 缩小版的 WebBot 注意力（GQA，无 RoPE）
    attn = MultiHeadAttention(E, H, KV, D).eval()
    x = torch.randn(B, C, E)
    with torch.no_grad():
        q = attn.W_q(x).view(B, C, H, D).transpose(1, 2)
        k = attn.W_k(x).view(B, C, KV, D).transpose(1, 2).repeat_interleave(H // KV, dim=1)
        v = attn.W_v(x).view(B, C, KV, D).transpose(1, 2).repeat_interleave(H // KV, dim=1)
        manual = attn.W_o(_causal_attention(q, k, v, 1 / math.sqrt(D)).transpose(1, 2).reshape(B, C, H * D))
        assert torch.allclose(attn(x), manual, atol=1e-5)   # SDPA 默认除以 √(每个头的维度)


# ---------------------------------------------------------------- 附录 C：基线

def test_c_grpo_grad(dlfs6_path):
    import copy

    import torch
    import torch.nn.functional as F
    from codebot.model import GPT

    ns = _load_defs(dlfs6_path / "ch03" / "09_grpo.py", ["compute_probs", "grpo_loss"], {"torch": torch, "F": F})
    torch.manual_seed(0)
    model = GPT(50, 16, 32, 4, 2, 64, 0.1)          # 缩小版 CodeBot（有 dropout 0.1）
    old_model = copy.deepcopy(model).eval()
    model.eval()                                     # 先关掉 dropout：新旧策略完全相同

    # 一个问题、G = 4 个回答；与 GRPODataset.get_batch 一样右填充 0，mask 只覆盖回答
    prompt, responses = [5, 6, 7], [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]]
    width = len(prompt) + max(map(len, responses))
    ids = torch.tensor([prompt + r + [0] * (width - len(prompt) - len(r)) for r in responses])
    mask = torch.tensor([[0.0] * len(prompt) + [1.0] * len(r) + [0.0] * (width - len(prompt) - len(r))
                         for r in responses])
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    adv = rewards - rewards.mean()                   # 09_grpo.py 第 89 行

    params = list(model.parameters())

    def flat(loss):
        return torch.cat([g.flatten() for g in torch.autograd.grad(loss, params)])

    # 附录 C 的估计量：−(1/N) Σ_i A_i ∇log π(y_i)，log π(y_i) 是回答部分 token 的对数概率之和
    logp = torch.log_softmax(model(ids)[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    estimator = flat(-(adv[:, None] * logp * mask[:, 1:]).sum() / ids.size(0))

    with torch.no_grad():
        ratio = ns["compute_probs"](model, ids) / (ns["compute_probs"](old_model, ids) + 1e-8)
    assert (ratio - 1)[mask[:, 1:] > 0].abs().max() < 1e-5          # 比率为 1（只差 +1e-8）
    g_eval = flat(ns["grpo_loss"](model, old_model, ids, mask, adv))
    assert torch.allclose(g_eval, estimator, rtol=1e-4, atol=1e-8)
    assert F.cosine_similarity(g_eval, estimator, dim=0) > 0.999999

    # 原书在更新时没有切到 eval 模式：dropout 让比率偏离 1，梯度也跟着偏了
    model.train()
    torch.manual_seed(1)
    with torch.no_grad():
        ratio_train = ns["compute_probs"](model, ids) / (ns["compute_probs"](old_model, ids) + 1e-8)
    assert (ratio_train - 1)[mask[:, 1:] > 0].abs().max() > 0.01
    g_train = flat(ns["grpo_loss"](model, old_model, ids, mask, adv))
    assert F.cosine_similarity(g_train, estimator, dim=0) < 0.999


# ---------------------------------------------------------------- 附录 D：Adam 的偏差校正

def test_d_adamw(dlfs6_path):
    import copy

    import numpy as np
    import torch
    from conftest import run_dlfs6_script

    adamw_step = page_namespace("appendix-d", upto="D-h")["adamw_step"]

    # 1. 验证 D-h 写死的初始权重，正是 02_adamw.py 里 torch.manual_seed(0) 之后的 Linear(2, 1)
    literals = {}
    for node in ast.parse(cell_source("appendix-d", "D-h")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and node.value.args:
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
            if name in ("w", "b"):
                literals[name] = ast.literal_eval(node.value.args[0])
    torch.manual_seed(0)
    linear = torch.nn.Linear(2, 1)
    assert linear.weight.detach().flatten().tolist() == literals["w"]
    assert linear.bias.detach().tolist() == literals["b"]

    # 2. 原书脚本的输出就是验证 D-h 断言的那五个 loss
    out = run_dlfs6_script("ch06/02_adamw.py")
    assert out == [f"Step {i}: loss = {v}" for i, v in enumerate(["8.0011", "5.8991", "4.1294", "2.6933", "1.5855"])]

    # 3. 本页的 adamw_step、原书的 AdamW 类、torch.optim.AdamW：同一个问题上跑 200 步（float64）
    source = (dlfs6_path / "ch06" / "02_adamw.py").read_text(encoding="utf-8")
    defs = {}
    exec(compile(source.split("torch.manual_seed(0)")[0], "02_adamw.py", "exec"), defs)  # 只取类定义
    torch.manual_seed(1)
    X, Y = torch.randn(64, 3, dtype=torch.float64), torch.randn(64, 2, dtype=torch.float64)
    base = torch.nn.Linear(3, 2).double()
    book_model, torch_model, grad_model = (copy.deepcopy(base) for _ in range(3))
    optimizers = [(book_model, defs["AdamW"](book_model.parameters(), lr=1e-2, weight_decay=0.1)),
                  (torch_model, torch.optim.AdamW(torch_model.parameters(), lr=1e-2, weight_decay=0.1))]
    mine = [p.detach().numpy().copy() for p in base.parameters()]
    states = [{"t": 0, "m": np.zeros_like(p), "v": np.zeros_like(p)} for p in mine]
    for _ in range(200):
        for model, opt in optimizers:
            opt.zero_grad()
            ((model(X) - Y) ** 2).mean().backward()
            opt.step()
        with torch.no_grad():                         # 用 torch 求 NumPy 参数处的梯度
            for p, value in zip(grad_model.parameters(), mine):
                p.copy_(torch.from_numpy(value))
        grad_model.zero_grad()
        ((grad_model(X) - Y) ** 2).mean().backward()
        mine = [adamw_step(value, p.grad.numpy(), s, lr=1e-2, weight_decay=0.1)
                for value, p, s in zip(mine, grad_model.parameters(), states)]
    for book_p, torch_p, mine_p in zip(book_model.parameters(), torch_model.parameters(), mine):
        assert np.abs(book_p.detach().numpy() - torch_p.detach().numpy()).max() < 1e-10
        assert np.abs(mine_p - torch_p.detach().numpy()).max() < 1e-10

    # 4. 正文引用的超参数：CodeBot 预训练 lr 3e-4；StoryBot 预训练与 GRPO 都把梯度裁剪到 1.0
    assert _script_constant(dlfs6_path / "ch03" / "01_pretrain.py", "learning_rate") == 3e-4
    assert _script_constant(dlfs6_path / "ch06" / "05_pretrain.py", "grad_clip") == 1.0
    grpo_src = (dlfs6_path / "ch03" / "09_grpo.py").read_text(encoding="utf-8")
    assert "clip_grad_norm_(model.parameters(), max_norm=1.0)" in grpo_src


# ---------------------------------------------------------------- 附录 E：DPO

def _load_storybot(path):
    import torch
    from storybot.model import GPT

    ckpt = torch.load(path, map_location="cpu")
    config = {k: ckpt[k] for k in ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim", "theta")}
    model = GPT(**config)
    # Hugging Face 上发布的检查点把 FFN 叫作 mlp，仓库里的 storybot/model.py 叫 ffn
    model.load_state_dict({k.replace(".mlp.", ".ffn."): v for k, v in ckpt["model_state_dict"].items()})
    return model.eval()


def test_e_dpo_loss(dlfs6_path):
    import json
    import math

    import torch
    import torch.nn.functional as F
    from storybot.tokenizer import BPETokenizer
    from torch.utils.data import Dataset

    script = dlfs6_path / "ch06" / "08_dpo.py"
    ns = _load_defs(script, ["DPODataset", "get_sequence_logprobs", "compute_dpo_loss"],
                    {"torch": torch, "F": F, "json": json, "Dataset": Dataset})
    dpo_loss = page_namespace("appendix-e", upto="E-g")["dpo_loss"]
    beta = _script_constant(script, "beta")
    assert beta == 0.1

    # 正文引用的设置：AdamW、学习率 5e-6；model 与 ref_model 都从预训练完成的同一个检查点加载
    source = script.read_text(encoding="utf-8")
    assert _script_constant(script, "learning_rate") == 5e-6
    assert "torch.optim.AdamW(model.parameters(), lr=learning_rate)" in source
    assert "model = GPT.load_from(pretrain_model_path" in source
    assert "ref_model = GPT.load_from(pretrain_model_path" in source
    assert _script_constant(script, "pretrain_model_path") == "storybot/model_pretrain.pt"

    # 偏好数据：144 对，每对共用一个故事开头；正文举的第一对，chosen 是快乐的结局，rejected 是悲伤的结局
    pairs = json.loads((dlfs6_path / "storybot" / "tiny_stories_dpo.json").read_text(encoding="utf-8"))
    assert len(pairs) == 144 and all(sorted(p) == ["chosen", "prompt", "rejected"] for p in pairs)
    assert "Tim was so happy" in pairs[0]["chosen"] and "Tim was very sad" in pairs[0]["rejected"]

    tok = BPETokenizer.load_from(str(dlfs6_path / "storybot" / "merge_rules.pkl"))
    data = ns["DPODataset"](str(dlfs6_path / "storybot" / "tiny_stories_dpo.json"), tok, 256)
    batch_size, max_iters = _script_constant(script, "batch_size"), _script_constant(script, "max_iters")
    assert (len(data), batch_size, max_iters) == (144, 8, 1000)
    assert round(max_iters * batch_size / len(data), 1) == 55.6        # 附录 E.6：同一批数据看 55 遍以上
    chosen_ids, chosen_mask, rejected_ids, rejected_mask = (torch.stack(col) for col in zip(*(data[i] for i in range(4))))

    policy = _load_storybot(dlfs6_path / "storybot" / "model_iter_5000.pt")
    ref = _load_storybot(dlfs6_path / "storybot" / "model_pretrain.pt")
    lp_fn, loss_fn = ns["get_sequence_logprobs"], ns["compute_dpo_loss"]
    with torch.no_grad():
        book = loss_fn(policy, ref, chosen_ids, chosen_mask, rejected_ids, rejected_mask, beta).item()
        lp = [lp_fn(m, ids, mask).double().numpy()
              for m in (policy, ref) for ids, mask in ((chosen_ids, chosen_mask), (rejected_ids, rejected_mask))]
        assert abs(book - dpo_loss(lp[0], lp[1], lp[2], lp[3], beta)) < 1e-5   # 原书 = 本页的 NumPy 版

        same = loss_fn(ref, ref, chosen_ids, chosen_mask, rejected_ids, rejected_mask, beta).item()
        assert abs(same - math.log(2)) < 1e-6                                   # 策略 = 参考时恰为 ln 2

        def with_prompt(mask):                                                  # 把问题部分也算进去
            full = mask.clone()
            for row in full:
                row[: int(row.argmax())] = 1
            return full

        full = loss_fn(policy, ref, chosen_ids, with_prompt(chosen_mask), rejected_ids, with_prompt(rejected_mask), beta)
        assert abs(full.item() - book) < 1e-4                                   # 问题部分在差值里抵消

    # DPO 训练时不生成回答；GRPO 要生成
    assert "generate" not in _called_names(script)
    assert "generate" in _called_names(dlfs6_path / "ch03" / "09_grpo.py")
