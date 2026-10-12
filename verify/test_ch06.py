"""第 6 章 训练【应用篇】：本地验证（需要 PyTorch，CPU 即可）。

测试直接使用 third_party/dlfs6 里原书的代码：脚本里的函数和类通过 load_defs() 取出（只执行定义，
不执行训练循环），StoryBot 的官方权重从 HF 下载（tools/fetch_data.py）。页面上的 NumPy 版代码通过
chapter_cells() 原样执行，再与 PyTorch 的实现对拍。

  CUDA_VISIBLE_DEVICES=-1 uv run pytest verify/test_ch06.py
"""
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import run_dlfs6_script
from _ch06_common import (ROOT, STORYBOT, STORYBOT_CONFIG, chapter_cells, function_source, heldout_tokens,
                          lm_loss, load_defs, load_storybot)

DATA = ROOT / "web" / "data" / "ch06"


def read_record(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 6.2 AdamW 的实现

def test_book_adamw_matches_torch(dlfs6_path):
    # 原样运行 ch06/02_adamw.py：输出与 6.2.3 节一致
    assert run_dlfs6_script("ch06/02_adamw.py") == [
        "Step 0: loss = 8.0011", "Step 1: loss = 5.8991", "Step 2: loss = 4.1294",
        "Step 3: loss = 2.6933", "Step 4: loss = 1.5855"]
    book = load_defs("ch06/02_adamw.py", ["SGD", "AdamW"])

    # 原书的 SGD 与 torch.optim.SGD：同样的 20 步
    def train(make_opt, steps=20):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3)
        opt = make_opt(model.parameters())
        for _ in range(steps):
            loss = (model(torch.randn(8, 4)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    sgd_diff = (train(lambda p: book["SGD"](p, lr=0.1)) - train(lambda p: torch.optim.SGD(p, lr=0.1))).abs().max()
    assert sgd_diff < 1e-6

    # CS336 作业 1 的 test_adamw 流程：Linear(3, 2)、1000 步，与 torch.optim.AdamW 的差要在 1e-4 以内
    def cs336_optimize(opt_class):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2, bias=False)
        opt = opt_class(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        for _ in range(1000):
            opt.zero_grad()
            x = torch.rand(model.in_features)
            y = torch.tensor([x[0] + x[1], -x[2]])
            loss = ((y - model(x)) ** 2).sum()
            loss.backward()
            opt.step()
        return model.weight.detach()

    diff = (cs336_optimize(book["AdamW"]) - cs336_optimize(torch.optim.AdamW)).abs().max().item()
    print(f"原书 AdamW 与 torch.optim.AdamW，1000 步后参数的最大差: {diff:.2e}")
    assert diff < 1e-4   # CS336 的容差
    assert diff < 2e-5   # 实际只差在 float32 的舍入顺序上


def test_numpy_adamw_matches_torch():
    # 页面上 6.2-a 的 NumPy 版 AdamW（原样执行）与 torch.optim.AdamW 在 float64 下对拍
    AdamW = chapter_cells("6.2-a")["AdamW"]
    rng = np.random.default_rng(0)
    w0, b0 = rng.normal(size=(5, 3)), rng.normal(size=3)
    xs, ys = rng.normal(size=(100, 32, 5)), rng.normal(size=(100, 32, 3))

    w, b = w0.copy(), b0.copy()
    opt = AdamW([w, b], lr=1e-2, weight_decay=0.1)
    for x, y in zip(xs, ys):
        dout = 2 * (x @ w + b - y) / y.size          # MSE 对输出的梯度
        opt.step([x.T @ dout, dout.sum(axis=0)])

    tw = torch.tensor(w0, requires_grad=True)
    tb = torch.tensor(b0, requires_grad=True)
    topt = torch.optim.AdamW([tw, tb], lr=1e-2, weight_decay=0.1)
    for x, y in zip(xs, ys):
        loss = ((torch.from_numpy(x) @ tw + tb - torch.from_numpy(y)) ** 2).mean()
        topt.zero_grad()
        loss.backward()
        topt.step()
    assert np.abs(w - tw.detach().numpy()).max() < 1e-12
    assert np.abs(b - tb.detach().numpy()).max() < 1e-12

    # 6.2-b 抄下来的初始值，确实是 torch.manual_seed(0) 之后 nn.Linear(2, 1) 的初始化
    torch.manual_seed(0)
    lin = torch.nn.Linear(2, 1)
    assert lin.weight.tolist() == [[-0.0052939653396606445, 0.37932294607162476]]
    assert lin.bias.tolist() == [-0.5819807648658752]


# ---------------------------------------------------------------- 6.3 学习率调度

def test_lr_sweep_record(dlfs6_path):
    # 6.3.1 节的扫描记录（verify/_ch06_lr_sweep.py 生成）：重放其中一个学习率，前 20 步的 loss 一致
    import _ch06_lr_sweep as sweep

    rec = read_record("lr_sweep.json")
    runs = {r["lr"]: r for r in rec["runs"]}
    assert rec["config"]["params"] == 1_379_136
    torch.set_num_threads(rec["provenance"]["threads"])
    losses, _ = sweep.train_one(3e-3, steps=20)
    assert np.allclose(losses, runs[3e-3]["losses"][:20], atol=1e-4)

    # 记录呈 U 形：太小学得慢，太大学得乱
    lrs = sorted(runs)
    vals = [runs[lr]["val_loss"] for lr in lrs]
    best = int(np.argmin(vals))
    assert lrs[best] in (3e-3, 1e-2)
    assert vals[0] > vals[best] + 2.0 and vals[-1] > vals[best] + 0.5


def test_get_lr_identical_and_lr_zero_step(dlfs6_path):
    # 03_lr_scheduler.py、05_pretrain.py、ch09/01_pretrain.py 里的 get_lr 去掉注释后逐字相同
    sources = {rel: function_source(rel, "get_lr")
               for rel in ("ch06/03_lr_scheduler.py", "ch06/05_pretrain.py", "ch09/01_pretrain.py")}
    assert len(set(sources.values())) == 1

    # 页面上 6.3-d 的 get_lr 与原书函数在整个训练区间上取值相同
    book = load_defs("ch06/03_lr_scheduler.py", ["get_lr"])["get_lr"]
    page = chapter_cells("6.3-d")["get_lr"]
    assert all(page(i, 1e-3, 200, 40000) == book(i, 1e-3, 200, 40000) for i in range(0, 40001))
    assert sum(book(i, 1e-3, 200, 40000) for i in range(40000)) == pytest.approx(20.0, abs=1e-9)

    # torch.optim.AdamW 在 lr = 0 时：参数不动，但步数和一阶矩已经更新
    p = torch.nn.Parameter(torch.tensor([0.5, -0.3]))
    before = p.detach().clone()
    opt = torch.optim.AdamW([p], lr=book(0, 1e-3, 200, 40000))
    p.grad = torch.tensor([0.2, 0.4])
    opt.step()
    state = opt.state[p]
    assert torch.equal(p.detach(), before)
    assert int(state["step"]) == 1 and bool(torch.all(state["exp_avg"] != 0))


# ---------------------------------------------------------------- 6.4 混合精度

def test_bf16_emulation_and_script_output(dlfs6_path):
    # 原样运行 ch06/04_mixed_precision.py：每一行输出都与原书注释一致
    assert run_dlfs6_script("ch06/04_mixed_precision.py") == [
        "----- FP16 -----", "tensor(1000., dtype=torch.float16)", "tensor(0., dtype=torch.float16)",
        "tensor(inf, dtype=torch.float16)", "----- BF16 -----", "tensor(0., dtype=torch.float16)",
        "tensor(1.0012e-08, dtype=torch.bfloat16)", "tensor(inf, dtype=torch.float16)",
        "tensor(70144., dtype=torch.bfloat16)", "----- 自動混合精度 -----", "torch.bfloat16", "torch.float32"]

    # 页面上 6.4-d 的 to_bf16 与 PyTorch 的 fp32 → bf16 转换逐位相同
    to_bf16 = chapter_cells("6.4-d")["to_bf16"]
    rng = np.random.default_rng(0)
    with np.errstate(over="ignore"):
        x = (rng.standard_normal(200_000) * 10.0 ** rng.uniform(-45, 39, 200_000)).astype(np.float32)
    special = np.array([0.0, -0.0, 1e-8, 3e-8, 70000.0, np.pi, 1 / 3, 65520.0, 3.4e38, -3.4e38, 1e-45,
                        np.inf, -np.inf], dtype=np.float32)
    x = np.concatenate([x, special])
    ours = to_bf16(x).view(np.uint32)
    theirs = torch.from_numpy(x).to(torch.bfloat16).to(torch.float32).numpy().view(np.uint32)
    assert np.array_equal(ours, theirs)
    assert np.isnan(to_bf16(np.float32("nan")))


def test_autocast_dtypes_in_storybot(dlfs6_path):
    from storybot.model import GPT

    torch.manual_seed(0)
    model = GPT(vocab_size=1000, max_context_len=32, embed_dim=64, n_head=4, n_layer=2, ff_dim=128)
    seen = {}

    def record(name):
        return lambda module, inputs, output: seen.__setitem__(name, output.dtype)

    block = model.blocks[0]
    for name, module in [("embedding", model.embed), ("RMSNorm", block.norm1), ("W_q", block.attn.W_q),
                         ("attention", block.attn), ("SwiGLU", block.ffn), ("block 输出（残差流）", block),
                         ("logits", model.unembed)]:
        module.register_forward_hook(record(name))
    x = torch.randint(0, 1000, (2, 32))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
    loss.backward()
    print({k: str(v) for k, v in seen.items()}, "loss:", loss.dtype)

    assert seen["embedding"] == seen["RMSNorm"] == seen["block 输出（残差流）"] == torch.float32
    assert seen["W_q"] == seen["attention"] == seen["SwiGLU"] == seen["logits"] == torch.bfloat16
    assert loss.dtype == torch.float32
    assert all(p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in model.parameters())

    # CPU 的 autocast 不把 sum 提升为 fp32（CUDA 会）；a.sum() 是 fp32 只因为 a 本来就是 fp32
    a = torch.randn(64, 64)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert (a @ a).dtype == torch.bfloat16
        assert (a @ a).sum().dtype == torch.bfloat16
        assert a.sum().dtype == torch.float32


def test_cpu_bf16_benchmark_record(dlfs6_path):
    # 记录（verify/_ch06_bench_cpu.py 生成）：在没有原生 bf16 指令的 CPU 上，bf16 autocast 比 fp32 慢
    import platform
    import _ch06_bench_cpu as bench

    rec = read_record("cpu_bf16_bench.json")
    assert rec["bf16_seconds_per_step"] > rec["fp32_seconds_per_step"]
    # 在你的机器上重测一次（batch 小一些）；只有与记录是同一型号的 CPU 时才断言快慢
    fp32 = bench.seconds_per_step(autocast=False, batch=2, steps=2)
    bf16 = bench.seconds_per_step(autocast=True, batch=2, steps=2)
    print(f"本机: fp32 {fp32:.3f} s/step，bf16 autocast {bf16:.3f} s/step（{platform.processor()}）")
    if platform.processor() == rec["provenance"]["processor"]:
        assert bf16 > fp32


def test_gradscaler_rescues_fp16_underflow():
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 16)
    x = torch.randn(4, 16)
    ref = torch.nn.Linear(16, 16)
    ref.load_state_dict(lin.state_dict())
    (ref(x).sum() * 1e-9).backward()                          # fp32 下的参考梯度

    with torch.autocast("cpu", dtype=torch.float16):          # 不缩放：fp16 里的梯度全部下溢
        loss = lin(x).float().sum() * 1e-9
    loss.backward()
    assert lin.weight.grad.abs().max().item() == 0.0

    lin.zero_grad()
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu")
    assert scaler.get_scale() == 65536.0                      # 默认从 2**16 起步
    assert (scaler.get_growth_interval(), scaler.get_growth_factor(), scaler.get_backoff_factor()) == (2000, 2.0, 0.5)
    with torch.autocast("cpu", dtype=torch.float16):
        loss = lin(x).float().sum() * 1e-9
    scaler.scale(loss).backward()
    scaler.unscale_(opt)                                      # 除回原来的大小（裁剪应当在这之后）
    assert torch.allclose(lin.weight.grad, ref.weight.grad, rtol=1e-2, atol=0)

    # 梯度溢出成 inf 时：这一步被跳过，缩放因子减半
    scaler = torch.amp.GradScaler("cpu")
    opt.zero_grad()
    with torch.autocast("cpu", dtype=torch.float16):
        loss = lin(x).float().sum() * 1e30
    scaler.scale(loss).backward()
    before = lin.weight.detach().clone()
    scaler.step(opt)
    scaler.update()
    assert torch.equal(lin.weight.detach(), before) and scaler.get_scale() == 32768.0


# ---------------------------------------------------------------- 6.5 预训练（改进版）

def test_clip_grad_norm_semantics():
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))
    a.grad, b.grad = torch.tensor([3.0, 4.0]), torch.tensor([12.0])
    norm = torch.nn.utils.clip_grad_norm_([a, b], 1.0)
    assert norm.item() == 13.0                                 # 返回裁剪前的范数
    coef = 1.0 / (13.0 + 1e-6)
    assert torch.allclose(a.grad, torch.tensor([3.0, 4.0]) * coef)
    assert torch.allclose(b.grad, torch.tensor([12.0]) * coef)

    # 页面上 6.5-b 的 NumPy 版与 torch 一致（float64）
    clip = chapter_cells("6.5-b")["clip_grad_norm"]
    rng = np.random.default_rng(0)
    grads = [rng.normal(size=shape) * 5 for shape in [(3, 4), (7,), (2, 2, 2)]]
    ours, total = clip(grads, 1.0)
    params = [torch.nn.Parameter(torch.zeros(g.shape, dtype=torch.float64)) for g in grads]
    for p, g in zip(params, grads):
        p.grad = torch.from_numpy(g.copy())
    assert abs(torch.nn.utils.clip_grad_norm_(params, 1.0).item() - total) < 1e-9
    assert all(np.allclose(o, p.grad.numpy(), rtol=1e-12) for o, p in zip(ours, params))


def test_get_batch_like_cs336(dlfs6_path):
    # CS336 作业 1 的 get_batch 测试流程：arange(100)、上下文长度 7、抽 1000 × 32 次
    get_batch = load_defs("ch06/05_pretrain.py", ["get_batch"])["get_batch"]
    data = np.arange(100)
    torch.manual_seed(0)
    starts = []
    for _ in range(1000):
        x, y = get_batch(data, 7, 32, torch.device("cpu"))
        assert torch.equal(y[:, :-1], x[:, 1:])                 # y 是 x 错开一位
        starts += x[:, 0].tolist()
    print(f"起点范围 {min(starts)}…{max(starts)}，共 {len(set(starts))} 种")
    assert min(starts) == 0 and max(starts) == 91 and len(set(starts)) == 92   # 永远抽不到 92


def test_checkpoint_losses_match_record(dlfs6_path):
    # 记录（verify/_ch06_ckpt_record.py 生成）：在前 8192 个留出 token 上重算，结果一致
    from storybot.model import GPT

    rec = read_record("checkpoints.json")
    tokens = heldout_tokens(8192)
    files = {500: "model_iter_500.pt", 5000: "model_iter_5000.pt", 40000: "model_pretrain.pt"}
    for c in rec["checkpoints"]:
        if c["iter"] == 0:
            torch.manual_seed(0)
            model = GPT(**STORYBOT_CONFIG)
        else:
            model = load_storybot(STORYBOT / files[c["iter"]])
        measured = lm_loss(model, tokens)
        print(f"iter {c['iter']:>5}: {measured:.4f}（记录 {c['val_loss_8192']:.4f}）")
        assert abs(measured - c["val_loss_8192"]) < 1e-4
    loss = {c["iter"]: c["val_loss"] for c in rec["checkpoints"]}
    assert loss[5000] <= 2.00 and loss[40000] <= 1.45           # CS336 的两条及格线


def test_storybot_size_and_optimizer(dlfs6_path):
    from storybot.model import GPT

    model = GPT(**STORYBOT_CONFIG)
    total = sum(p.numel() for p in model.parameters())
    assert total == 22_696_448 and total - model.embed.weight.numel() == 17_576_448
    assert model.embed.weight is not model.unembed.weight       # StoryBot 不共享输入、输出的权重

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)         # 05_pretrain.py 第 107 行
    assert len(opt.param_groups) == 1
    group = opt.param_groups[0]
    assert group["weight_decay"] == 0.01 and group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
    assert any(p is model.blocks[0].norm1.weight for p in group["params"])   # RMSNorm 的权重也在衰减

    # 验证集：5,465,873 个 token（10,931,746 字节），故事平均约 198 个 token
    data = np.memmap(STORYBOT / "tiny_stories_valid.bin", dtype=np.uint16, mode="r")
    stories = int((data == 9999).sum())
    print(f"{len(data):,} 个 token，{stories:,} 个故事，平均 {len(data) / stories:.1f} 个 token")
    assert len(data) == 5_465_873 and 195 < len(data) / stories < 200


def test_activation_memory_dominates(dlfs6_path):
    # 反向传播要保存的张量（激活值），用 saved_tensors_hooks 按存储去重后计数；batch 2 减 batch 1 得到每条序列的量
    from storybot.model import GPT

    def saved_bytes(batch):
        torch.manual_seed(0)
        model = GPT(**STORYBOT_CONFIG)
        params = {p.untyped_storage().data_ptr() for p in model.parameters()}
        x = torch.randint(0, 10000, (batch, 256))
        storages = {}

        def pack(t):
            s = t.untyped_storage()
            if s.data_ptr() not in params:
                storages[s.data_ptr()] = s.nbytes()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                F.cross_entropy(model(x).view(-1, 10000), x.view(-1))
        return sum(storages.values())

    per_seq = saved_bytes(2) - saved_bytes(1)
    static = 22_696_448 * 16
    print(f"每条序列 {per_seq / 2**20:.1f} MiB；batch 32 共 {32 * per_seq / 2**30:.2f} GiB；静态 {static / 2**30:.2f} GiB")
    assert 32 * per_seq > 4 * static


def test_storybot_overfits_one_batch(dlfs6_path):
    # 6.5.4 节的第一道检查：同一个 batch 反复训练，loss 应该降到接近 0
    from storybot.model import GPT

    torch.manual_seed(0)
    model = GPT(vocab_size=10000, max_context_len=64, embed_dim=64, n_head=4, n_layer=2, ff_dim=172)
    tokens = torch.from_numpy(heldout_tokens(4 * 64))
    x, y = tokens[:-1].view(4, 64), tokens[1:].view(4, 64)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(120):
        loss = F.cross_entropy(model(x).view(-1, 10000), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"第 0 步 {losses[0]:.3f} → 第 119 步 {losses[-1]:.4f}")
    assert losses[0] > 9.0 and losses[-1] < 0.1


# ---------------------------------------------------------------- 6.6 文本生成与模型评估

def test_hf_checkpoint_needs_key_rename(dlfs6_path):
    from storybot.model import GPT

    with pytest.raises(RuntimeError, match=r"blocks\.0\.ffn\.W\.weight"):
        GPT.load_from(str(STORYBOT / "model_pretrain.pt"))              # 原书的加载方式
    ckpt = torch.load(STORYBOT / "model_pretrain.pt", map_location="cpu")
    assert any(".mlp." in k for k in ckpt["model_state_dict"])
    ckpt["model_state_dict"] = {k.replace(".mlp.", ".ffn."): v for k, v in ckpt["model_state_dict"].items()}
    model = GPT(ckpt["vocab_size"], ckpt["max_context_len"], ckpt["embed_dim"], ckpt["n_head"],
                ckpt["n_layer"], ckpt["ff_dim"], ckpt["theta"])
    model.load_state_dict(ckpt["model_state_dict"])                    # 改名之后加载成功
    assert lm_loss(model, heldout_tokens(1024)) < 2.0                    # 预测也正常


def test_checkpoint_sample_reproduces(dlfs6_path):
    from storybot.tokenizer import BPETokenizer
    from storybot.utils import generate

    rec = read_record("checkpoints.json")
    torch.set_num_threads(rec["sampling"]["threads"])
    tok = BPETokenizer.load_from(str(STORYBOT / "merge_rules.pkl"))
    model = load_storybot(STORYBOT / "model_pretrain.pt")
    final = next(c for c in rec["checkpoints"] if c["iter"] == 40000)
    torch.manual_seed(1)
    story = generate(model, tok, "<|endoftext|>", 200, 1.0).strip()
    assert story == final["samples"][1]
    # 本章开头引用的故事就是这一段
    opener = (ROOT / "content" / "ch06.md").read_text(encoding="utf-8")
    assert " ".join(story.split()) in " ".join(opener.split())


# ---------------------------------------------------------------- 6.7、6.8 DPO

def _dpo_batch(ns, tok, pairs, context_len):
    """用原书 DPODataset 的 _create_sample 把若干条 pair 编成一个 batch。"""
    ds = ns["DPODataset"].__new__(ns["DPODataset"])
    ds.tokenizer, ds.context_len = tok, context_len
    ds.samples = [ds._create_sample(p["prompt"], p["chosen"], p["rejected"]) for p in pairs]
    return [torch.stack(t) for t in zip(*[ds[i] for i in range(len(ds))])]


def test_dpo_dataset_facts(dlfs6_path):
    from storybot.tokenizer import BPETokenizer

    data = json.loads((STORYBOT / "tiny_stories_dpo.json").read_text(encoding="utf-8"))
    assert len(data) == 144 and len({d["prompt"] for d in data}) == 144
    assert all(d["prompt"].endswith(". ") for d in data)                       # prompt 以空格结尾
    assert not any("<|endoftext|>" in d["chosen"] + d["rejected"] for d in data)  # 回答末尾没有 EOS

    tok = BPETokenizer.load_from(str(STORYBOT / "merge_rules.pkl"))
    longest = max(len(tok.encode(d["prompt"])) + max(len(tok.encode(d["chosen"])), len(tok.encode(d["rejected"])))
                  for d in data)
    assert longest == 81                                                          # context_len 96 放得下

    # 分开编码与一起编码：prompt 末尾的空格单独成了一个 token，后面才是 'Tim'
    p0 = data[0]
    n = len(tok.encode(p0["prompt"]))
    apart = tok.encode(p0["prompt"]) + tok.encode(p0["chosen"])
    together = tok.encode(p0["prompt"] + p0["chosen"])
    assert (len(apart), len(together)) == (70, 69)
    assert tok.decode([apart[n - 1]]) == " " and tok.decode([apart[n]]) == "Tim"
    assert tok.decode([together[n - 1]]) == " Tim"


def test_dpo_masks_on_real_model(dlfs6_path):
    import copy
    from storybot.tokenizer import BPETokenizer

    ns = load_defs("ch06/08_dpo.py", ["DPODataset", "get_sequence_logprobs", "compute_dpo_loss"])
    lp, dpo_loss = ns["get_sequence_logprobs"], ns["compute_dpo_loss"]
    tok = BPETokenizer.load_from(str(STORYBOT / "merge_rules.pkl"))
    pairs = json.loads((STORYBOT / "tiny_stories_dpo.json").read_text(encoding="utf-8"))[:8]
    c, cm, r, rm = _dpo_batch(ns, tok, pairs, 96)
    policy = load_storybot(STORYBOT / "model_iter_5000.pt")      # 一个与参考模型不同的“策略”
    ref = load_storybot(STORYBOT / "model_pretrain.pt").eval()
    same = load_storybot(STORYBOT / "model_pretrain.pt")          # 与参考模型相同，处于 train 模式

    with torch.no_grad():
        # 第 0 步：策略 = 参考模型（StoryBot 没有 dropout，train/eval 模式的输出相同）→ loss = ln 2
        assert torch.equal(same(c), ref(c))
        assert abs(dpo_loss(same, ref, c, cm, r, rm, 0.1).item() - math.log(2)) < 1e-6

        # prompt 也计入对数概率：loss 不变（共用的 prompt 在差值里抵消）
        def with_prompt(mask):
            full = torch.zeros_like(mask)
            for i in range(len(mask)):
                full[i, : int(mask[i].nonzero().max()) + 1] = 1
            return full
        masked = dpo_loss(policy, ref, c, cm, r, rm, 0.1).item()
        unmasked = dpo_loss(policy, ref, c, with_prompt(cm), r, with_prompt(rm), 0.1).item()
        print(f"只算回答 {masked:.6f}，prompt 也计入 {unmasked:.6f}")
        assert abs(masked - unmasked) < 1e-5

        # 右边多填充的 padding 不影响每条序列的对数概率：context_len 256 与 96 相同
        c256, cm256, _, _ = _dpo_batch(ns, tok, pairs, 256)
        assert torch.allclose(lp(ref, c256, cm256), lp(ref, c, cm), atol=1e-4)

        # 分词边界：回答的第一个 token，分开编码（' ' 之后的 'Tim'）远比一起编码（' Tim'）罕见
        everything = json.loads((STORYBOT / "tiny_stories_dpo.json").read_text(encoding="utf-8"))
        first = []
        for d in everything:
            prompt_ids = tok.encode(d["prompt"])
            n = len(prompt_ids)
            apart = prompt_ids + tok.encode(d["chosen"])
            together = tok.encode(d["prompt"] + d["chosen"])
            la = torch.log_softmax(ref(torch.tensor([apart[:n]]))[0, -1], -1)[apart[n]].item()
            lt = torch.log_softmax(ref(torch.tensor([together[:n - 1]]))[0, -1], -1)[together[n - 1]].item()
            first.append((la, lt))
        print(f"第一条：{first[0][0]:.2f} 对 {first[0][1]:.2f}；144 条平均："
              f"{np.mean([a for a, _ in first]):.2f} 对 {np.mean([b for _, b in first]):.2f}")
        assert (round(first[0][0], 1), round(first[0][1], 1)) == (-12.2, -1.8)
        assert (round(np.mean([a for a, _ in first]), 1), round(np.mean([b for _, b in first]), 1)) == (-14.4, -3.3)

        # 长度：按对数概率之和，参考模型只在 9/144 条上“偏好” chosen；按每个 token 的平均，是 91/144
        C, CM, R, RM = _dpo_batch(ns, tok, everything, 96)
        qc, qr = lp(ref, C, CM), lp(ref, R, RM)
        nc, nr = CM[:, 1:].sum(1), RM[:, 1:].sum(1)
        assert int((qc > qr).sum()) == 9 and int((qc / nc > qr / nr).sum()) == 91
        assert (round(nc.float().mean().item(), 1), round(nr.float().mean().item(), 1)) == (47.0, 32.7)

    # 对照：带 dropout 的模型处于 train 模式时，“策略 = 参考模型”的第 0 步 loss 不再是 ln 2
    torch.manual_seed(0)
    toy = torch.nn.Sequential(torch.nn.Embedding(50, 16), torch.nn.Dropout(0.1), torch.nn.Linear(16, 50))
    toy_ref = copy.deepcopy(toy).eval()
    ids = torch.randint(1, 50, (4, 12))
    ones = torch.ones_like(ids)
    assert abs(dpo_loss(toy.train(), toy_ref, ids, ones, ids.flip(1), ones, 0.1).item() - math.log(2)) > 1e-4


def test_dpo_record_reproduces(dlfs6_path):
    # 记录（verify/_ch06_dpo_run.py 生成）：用同样的设置重放前 3 步，loss 与记录一致
    import _ch06_dpo_run as run

    rec = read_record("dpo_cpu.json")
    ns, _, model, ref, opt, data_iter = run.setup(rec["config"]["context_len"], rec["provenance"]["threads"])
    losses = [run.train_step(ns, model, ref, opt, data_iter) for _ in range(3)]
    print("重放:", [f"{x:.6f}" for x in losses], "记录:", rec["losses"][:3])
    assert np.allclose(losses, rec["losses"][:3], atol=1e-5)
    assert rec["config"]["iters"] == 1000 and len(rec["losses"]) == 1000
    assert rec["evals"]["0"]["ref_prefers_chosen_by_sum"] * 144 == pytest.approx(9)   # 只按对数概率之和，9/144
