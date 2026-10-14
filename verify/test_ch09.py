"""第 9 章 训练【挑战篇】：本地验证（只用 CPU，不需要 GPU）。

需要 PyTorch 的断言都在这里，页面上用 ```localcheck 引用。模型一律是原书 webbot/model.py 的 GPT；
原书脚本 ch09/01_pretrain.py 不能 import（顶层就开始训练），所以用 _ch09_dlfs6.load_defs 只取出函数。
多进程的部分（DDP、ZeRO-1）用 gloo 在 CPU 上起 2 个真正的进程，worker 在 _ch09_dist.py 里。
"""
from __future__ import annotations

import ast
import json
import math
import sys

sys.dont_write_bytecode = True   # 导入 webbot.model 时不要在 third_party/dlfs6 里留下 __pycache__

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import _ch09_dist as D  # noqa: E402
from _ch09_dlfs6 import ROOT, load_defs, module_constants, script_tree, verify_cell  # noqa: E402

WEBBOT = dict(vocab_size=50000, max_context_len=1024, embed_dim=768, n_head=12, n_kv_head=4,
              n_layer=12, ff_dim=2048, theta=10000)          # ch09/01_pretrain.py 的配置
RUNS = ROOT / "web" / "data" / "ch09"


def tiny_gpt(seed=0):
    from webbot.model import GPT
    torch.manual_seed(seed)
    return GPT(**D.TINY)


def lm_loss(logits, y):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))


def forward_blocks(model, ids, use_checkpoint: bool, calls: list | None = None):
    """GPT.forward 的同一套计算；use_checkpoint=True 时每个 Block 都包进 torch.utils.checkpoint。
    calls 记录 Block 被执行了几次（重算时 forward hook 不会触发，所以直接数函数调用）。"""
    def run(block, h):
        if calls is not None:
            calls.append(1)
        return block(h)

    x = model.embed(ids)
    for block in model.blocks:
        x = checkpoint(run, block, x, use_reentrant=False) if use_checkpoint else run(block, x)
    return model.unembed(model.norm(x))


# ---------------------------------------------------------------- 9.1 梯度累积

def test_accumulation_equals_big_batch(dlfs6_path):
    """WebBot 的 GPT 上：8 条序列一次算，和拆成 4 个 micro-batch、每个 loss 除以 4 再累积，梯度相同。"""
    torch.manual_seed(1)
    x = torch.randint(0, D.TINY["vocab_size"], (8, D.CTX + 1))
    x, y = x[:, :-1], x[:, 1:]
    K = 4

    big = tiny_gpt()
    lm_loss(big(x), y).backward()

    acc, forgot = tiny_gpt(), tiny_gpt()
    for k in range(K):
        s = slice(2 * k, 2 * k + 2)
        (lm_loss(acc(x[s]), y[s]) / K).backward()        # 原书：loss = loss / accumulation_steps
        lm_loss(forgot(x[s]), y[s]).backward()            # 忘了除以 K

    for p_big, p_acc, p_forgot in zip(big.parameters(), acc.parameters(), forgot.parameters()):
        assert torch.allclose(p_acc.grad, p_big.grad, rtol=1e-4, atol=1e-7)
        assert torch.allclose(p_forgot.grad, K * p_big.grad, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------- 9.2 梯度检查点

def test_checkpoint_same_grads_and_recompute(dlfs6_path):
    """每个 Block 包进 checkpoint：梯度不变；代价是反向时每个 Block 的 forward 又跑了一遍。"""
    torch.manual_seed(2)
    x = torch.randint(0, D.TINY["vocab_size"], (4, D.CTX + 1))
    x, y = x[:, :-1], x[:, 1:]

    plain, ckpt = tiny_gpt(), tiny_gpt()
    calls_plain, calls_ckpt = [], []
    lm_loss(forward_blocks(plain, x, False, calls_plain), y).backward()
    lm_loss(forward_blocks(ckpt, x, True, calls_ckpt), y).backward()

    for p1, p2 in zip(plain.parameters(), ckpt.parameters()):
        assert torch.equal(p1.grad, p2.grad)          # 重算出来的激活逐位相同，梯度也逐位相同
    n_blocks = len(plain.blocks)
    assert (len(calls_plain), len(calls_ckpt)) == (n_blocks, 2 * n_blocks)


def saved_bytes(model, ids, use_checkpoint: bool, bf16: bool = False) -> int:
    """前向时为反向保存下来的激活一共多少字节（按底层 storage 去重，不算参数和 buffer 本身）。
    用了检查点时，Block 内部的张量不再保存，但每个 Block 的输入要留着，这部分单独加上。
    bf16=True 时像原书一样把前向和 loss 包进 autocast（这里是 CPU 的 autocast，规则与 CUDA 的不完全相同）。"""
    seen: dict[int, int] = {}
    fixed = {t.untyped_storage().data_ptr() for t in [*model.parameters(), *model.buffers()]}

    def pack(t):
        st = t.untyped_storage()
        if st.data_ptr() not in fixed:
            seen[st.data_ptr()] = st.nbytes()
        return t

    hooks = []
    if use_checkpoint:
        def keep_input(module, args):
            st = args[0].untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
        hooks = [b.register_forward_pre_hook(keep_input) for b in model.blocks]
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), \
            torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        logits = forward_blocks(model, ids, use_checkpoint)
        lm_loss(logits, ids)
    for h in hooks:
        h.remove()
    return sum(seen.values())


def test_webbot_activation_memory_per_sequence(dlfs6_path):
    """全尺寸 WebBot（CPU）：一条 1024-token 的序列要为反向保存多少激活；加检查点后剩多少；
    打开 bf16 autocast 时，每多一条序列多多少，与 batch 无关的部分（矩阵乘权重的 bf16 副本）又有多少。
    页面上验证 9.2-c 写死的几个数，必须与这里的测量一致。"""
    from webbot.model import GPT
    torch.manual_seed(0)
    model = GPT(**WEBBOT)
    ids = torch.randint(0, WEBBOT["vocab_size"], (2, WEBBOT["max_context_len"]))
    mib = {ck: saved_bytes(model, ids[:1], ck) / 2**20 for ck in (False, True)}
    bf16 = {b: saved_bytes(model, ids[:b], False, bf16=True) / 2**20 for b in (1, 2)}
    bf16_per_seq, bf16_fixed = bf16[2] - bf16[1], 2 * bf16[1] - bf16[2]
    logits_mib = WEBBOT["max_context_len"] * WEBBOT["vocab_size"] * 4 / 2**20
    n_matmul = sum(p.numel() for n, p in model.named_parameters() if p.ndim == 2 and not n.startswith("embed."))
    print(f"\nper sequence: plain {mib[False]:.1f} MiB, checkpointed {mib[True]:.1f} MiB, "
          f"fp32 log-softmax alone {logits_mib:.1f} MiB")
    print(f"bf16 autocast: batch 1 {bf16[1]:.1f} MiB, batch 2 {bf16[2]:.1f} MiB -> "
          f"{bf16_per_seq:.1f} MiB per sequence + {bf16_fixed:.1f} MiB fixed")

    page: dict = {"np": np}
    exec(verify_cell("9.2-c"), page)                         # 页面上的数字（单元里的断言也一起跑）
    assert abs(mib[False] - page["per_seq_plain"]) < 0.05 and abs(mib[True] - page["per_seq_ckpt"]) < 0.05
    assert abs(bf16_per_seq - page["per_seq_bf16"]) < 0.05 and abs(bf16_fixed - page["bf16_fixed"]) < 0.05
    assert abs(bf16_fixed - 2 * n_matmul / 2**20) < 0.5      # 固定部分 = 全部矩阵乘权重（含输出层）的 bf16 副本
    assert mib[False] / mib[True] > 3.5                      # 检查点后减少到约 1/4
    assert bf16_per_seq < 0.8 * mib[False]                   # bf16 只让每条序列少了约四分之一


# ---------------------------------------------------------------- 9.3 分布式训练 & 9.4 ZeRO（2 个 gloo 进程）

@pytest.fixture(scope="module")
def gloo_runs(dlfs6_path, tmp_path_factory):
    """一次 spawn 2 个进程，先跑 DDP 的两种 no_sync 写法，再跑 ZeRO-1；结果存盘后在主进程核对。"""
    out = tmp_path_factory.mktemp("gloo")
    world = 2
    mp.spawn(D.worker, args=(world, (out / "store").as_posix(), str(dlfs6_path), str(out), 3, 1e-2),
             nprocs=world, join=True)
    return {"world": world,
            "ddp": [torch.load(out / f"ddp_rank{r}.pt") for r in range(world)],
            "zero": [torch.load(out / f"zero_rank{r}.pt") for r in range(world)]}


def test_ddp_two_processes_gloo(dlfs6_path, gloo_runs):
    """2 个 gloo 进程跑真正的 DDP：初值被广播；梯度等于单进程的大 batch；
    原书把 forward 放在 no_sync 外面，于是每个 micro-step 都发生了 all-reduce。"""
    world, ranks = gloo_runs["world"], gloo_runs["ddp"]

    # 种子是 42+rank，包装前两个进程的初值不同；DDP 构造时把 rank 0 的参数广播给所有进程
    assert not torch.equal(ranks[0]["before"], ranks[1]["before"])
    assert torch.equal(ranks[1]["after"], ranks[0]["before"])

    # 单进程参照：rank 0 的初值，把两个进程的全部 2×4 个 micro-batch 当作一个大 batch
    ref = D.make_model(seed=42)
    for r in range(world):
        for m in range(D.ACCUM):
            (D.lm_loss(ref, *D.micro_batch(r, 0, m)) / (D.ACCUM * world)).backward()
    g_ref = D.flat_grads(ref)
    for rk in ranks:
        for pattern in ("book", "fixed"):
            assert torch.allclose(rk[pattern]["grads"], g_ref, rtol=1e-5, atol=1e-7), pattern

    for rk in ranks:                                              # 页面上写的计数：[1, 1, 1, 1] 与 [0, 0, 0, 1]
        book = rk["book"]["allreduce_calls_per_micro"]
        fixed = rk["fixed"]["allreduce_calls_per_micro"]
        assert book == [1] * D.ACCUM                              # 原书写法：4 个 micro-step 各通信一次
        assert fixed == [0] * (D.ACCUM - 1) + [1]                 # forward 也放进 no_sync：只在最后通信一次


def test_zero1_equals_adamw(dlfs6_path, gloo_runs):
    """ZeroRedundancyOptimizer(AdamW)：每个进程只存一部分 Adam 状态，参数却和普通 AdamW 完全一样地更新。"""
    world, ranks = gloo_runs["world"], gloo_runs["zero"]
    ref = D.make_model(seed=42)
    opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        for r in range(world):                                    # DDP 求平均 = 两份 loss 的平均
            (D.lm_loss(ref, *D.micro_batch(r, step, 0)) / world).backward()
        opt.step()
    p_ref = D.flat_params(ref)
    n = ranks[0]["n_params"]
    for rk in ranks:
        assert torch.allclose(rk["params"], p_ref, rtol=1e-5, atol=1e-6)
        assert 0 < rk["local_state_numel"] < 2 * n                # 只有一部分参数的 m 和 v
    assert sum(rk["local_state_numel"] for rk in ranks) == 2 * n  # 合起来恰好是一整份


def test_adamw_state_is_16_bytes_per_param(dlfs6_path):
    """原书的写法（fp32 参数 + autocast，torch.optim.AdamW）：参数、梯度、m、v 各 4 字节，合计 16 字节/参数。"""
    model = tiny_gpt()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, D.TINY["vocab_size"], (2, D.CTX + 1))
    lm_loss(model(x[:, :-1]), x[:, 1:]).backward()
    opt.step()
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts)  # noqa: E731
    states = [t for st in opt.state.values() for k, t in st.items() if k in ("exp_avg", "exp_avg_sq")]
    assert nbytes(params) == nbytes(p.grad for p in params) == 4 * n
    assert nbytes(states) == 8 * n
    assert nbytes(params) + nbytes(p.grad for p in params) + nbytes(states) == 16 * n


# ---------------------------------------------------------------- 9.5 Muon

def test_webbot_param_count_and_muon_split(dlfs6_path):
    """WebBot 一共 152,316,672 个参数；交给 Muon 的二维隐藏层矩阵和交给 AdamW 的其余参数各有多少。"""
    from webbot.model import GPT
    model = GPT(**WEBBOT)
    total = sum(p.numel() for p in model.parameters())
    muon = sum(p.numel() for n, p in model.named_parameters()
               if p.ndim == 2 and not n.startswith(("embed.", "unembed.")))
    assert total == 152_316_672
    assert muon == 75_497_472 and total - muon == 76_819_200


def test_numpy_newton_schulz_matches_torch_muon():
    """页面上验证 9.5-a 的 NumPy 版 ns5，与 PyTorch 官方 Muon 的 Newton–Schulz（内部用 bf16）一致。"""
    from torch.optim._muon import DEFAULT_A, DEFAULT_B, DEFAULT_C, EPS, _zeropower_via_newtonschulz
    ns: dict = {}
    exec(verify_cell("9.5-a"), ns)                     # 取出页面上的 NumPy 代码原样执行
    assert (DEFAULT_A, DEFAULT_B, DEFAULT_C, EPS) == (3.4445, -4.7750, 2.0315, 1e-7)
    rng = np.random.default_rng(0)
    for shape in ((64, 32), (32, 64), (48, 48)):
        G = rng.normal(size=shape)
        ours = ns["ns5"](G)
        theirs = _zeropower_via_newtonschulz(torch.tensor(G, dtype=torch.float32),
                                             (DEFAULT_A, DEFAULT_B, DEFAULT_C), 5, EPS).float().numpy()
        rel = np.linalg.norm(ours - theirs) / np.linalg.norm(ours)
        assert rel < 3e-2, (shape, rel)                # 实测约 1.6%–1.8%：bf16 只有 8 位有效位


def test_torch_muon_step_is_the_formula():
    """torch.optim.Muon 的一步 = 解耦权重衰减 + 按形状缩放的学习率 × NS(Nesterov 动量)。"""
    from torch.optim._muon import DEFAULT_A, DEFAULT_B, DEFAULT_C, EPS, _zeropower_via_newtonschulz
    torch.manual_seed(0)
    W = torch.randn(96, 32)                             # d_out = 96、d_in = 32：学习率乘 s = sqrt(96/32)
    G1, G2 = torch.randn(96, 32), torch.randn(96, 32)
    p = torch.nn.Parameter(W.clone())
    opt = torch.optim.Muon([p], lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True)
    expect, buf = W.clone(), torch.zeros_like(W)
    for G in (G1, G2):
        p.grad = G.clone()
        opt.step()
        buf = buf.lerp(G, 1 - 0.95)                     # M ← 0.95·M + 0.05·G（动量的指数平均写法；文档记作 B）
        upd = G.lerp(buf, 0.95)                         # Nesterov：0.05·G + 0.95·M
        O = _zeropower_via_newtonschulz(upd, (DEFAULT_A, DEFAULT_B, DEFAULT_C), 5, EPS)
        expect = expect * (1 - 0.02 * 0.1)              # 先做解耦权重衰减（用没调整过的 lr）
        expect = expect.add(O, alpha=-0.02 * math.sqrt(96 / 32))   # 再减去 lr·s·O，s = sqrt(max(1, d_out/d_in))
    assert torch.equal(p.detach(), expect)


# ---------------------------------------------------------------- 9.3 / 9.6 原书脚本本身

def test_script_constants_and_get_lr(dlfs6_path):
    """ch09/01_pretrain.py 的超参数，以及原书 get_lr 在关键步数上的值（warmup 500 步，然后线性降到 0）。"""
    c = module_constants()
    assert (c["micro_batch_size"], c["accumulation_steps"], c["context_len"]) == (32, 4, 1024)
    assert (c["max_iters"], c["warmup_iters"], c["learning_rate"], c["eval_interval"]) == (100000, 500, 6e-4, 1000)
    assert c["micro_batch_size"] * c["accumulation_steps"] * c["context_len"] * c["max_iters"] == 13_107_200_000
    get_lr = load_defs(["get_lr"])["get_lr"]
    lr = lambda it: get_lr(it, 6e-4, 500, 100000)  # noqa: E731
    assert lr(0) == 0.0 and lr(500) == 6e-4 and math.isclose(lr(250), 3e-4)
    assert math.isclose(lr(50250), 3e-4) and lr(100000) == 0.0 and 0 < lr(99999) < 1e-8

    # 优化器只传了 lr，其余都是 torch.optim.AdamW 的默认值
    opt_call = next(_find_call(script_tree(), "torch.optim.AdamW"))
    assert [k.arg for k in opt_call.keywords] == ["lr"]
    defaults = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))]).defaults
    assert defaults["betas"] == (0.9, 0.999) and defaults["weight_decay"] == 0.01 and defaults["eps"] == 1e-8


def test_batch_constants_quoted_on_the_page(dlfs6_path):
    """验证 9.3-e 写死的几个数：WebBot 的 micro-batch 32 × 累积 4；第 3 章 GRPO 每次 32 道题、每题 8 个回答、
    同一批数据更新 2 次（都是脚本顶层的字面量）。"""
    c9 = module_constants()
    c3 = module_constants("ch03/09_grpo.py")
    assert (c9["micro_batch_size"], c9["accumulation_steps"], c9["context_len"]) == (32, 4, 1024)
    assert (c3["batch_size"], c3["group_size"], c3["n_update_per_generation"]) == (32, 8, 2)
    page = verify_cell("9.3-e")
    assert "micro, accum, world, ctx = 32, 4, 8, 1024" in page and "prompts, G, n_update = 32, 8, 2" in page


def _find_call(tree, dotted: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted:
            yield node


def test_script_ddp_and_logging_details(dlfs6_path):
    """直接读原书脚本的语法树：no_sync 里只有 backward；is_main 用的是 local_rank；
    wandb 记的 effective_batch_size 没有乘卡数；train/loss 是最后一个 micro-batch 的值。"""
    tree = script_tree()
    src = ast.unparse(tree)
    with_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.With)
                  and ast.unparse(n.items[0].context_expr) == "model.no_sync()"]
    assert len(with_nodes) == 1
    assert [ast.unparse(s) for s in with_nodes[0].body] == ["loss.backward()"]   # forward 在外面
    assert "is_main = not ddp or local_rank == 0" in src
    update = next(_find_call(tree, "wandb.config.update"))
    cfg = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(update.args[0].keys, update.args[0].values)}
    assert cfg["effective_batch_size"] == "micro_batch_size * accumulation_steps"
    assert "train_loss = loss.item() * accumulation_steps" in src
    logged = [ast.unparse(c.args[0]) for c in _find_call(tree, "wandb.log")]
    assert logged == ["{'train/loss': train_loss, 'train/lr': lr}", "{'val/loss': val_loss}"]


# ---------------------------------------------------------------- 9.7 记录下来的 CPU 运行

def _calls(run):
    return [json.loads(line) for line in (RUNS / run / "wandb.jsonl").read_text(encoding="utf-8").splitlines()]


def test_recorded_cpu_run_is_consistent(dlfs6_path):
    """web/data/ch09/webbot-cpu-shrunk：记录下来的学习率逐步等于原书 get_lr，loss 从 ln 50000 附近开始下降。
    页面上关于这次运行的几句话也在这里核对：原书的 autocast 在 CPU 上只留下一条警告就关掉了（所以是 fp32）；
    训练用 owt_valid.bin 的前 90%，紧接着的 65,536 个 token 做验证。"""
    import re
    report = json.loads((RUNS / "webbot-cpu-shrunk" / "report.json").read_text(encoding="utf-8"))
    calls = _calls("webbot-cpu-shrunk")
    cfg = next(c["data"] for c in calls if c["call"] == "config.update")
    assert [c["call"] for c in calls][:2] == ["init", "config.update"] and calls[-1]["call"] == "finish"
    assert report["dlfs6_commit"].startswith("c9b6e2e") and report["exit_codes"] == [0]
    assert cfg["effective_batch_size"] == cfg["micro_batch_size"] * cfg["accumulation_steps"]
    assert report["warnings"] == ["UserWarning: CUDA is not available or torch_xla is imported. Disabling autocast."]

    probe_log = (ROOT / "web" / "data" / "runs" / "webbot-probe" / "01_pretrain.log.txt").read_text(encoding="utf-8")
    n_valid = int(re.search(r"検証データ: ([\d,]+) tokens", probe_log).group(1).replace(",", ""))  # 整个 owt_valid.bin
    (t0, t1), (v0, v1) = [tuple(int(x or 0) for x in re.search(r"\[(\d*):(\d+)\]", report["data"][k]).groups())
                          for k in ("train", "val")]
    assert report["data"]["train"].startswith("webbot/owt_valid.bin[") and t0 == 0 and t1 == int(0.9 * n_valid)
    assert v0 == t1 and v1 - v0 == 65_536

    get_lr = load_defs(["get_lr"])["get_lr"]
    train = [c for c in calls if c["call"] == "log" and "train/loss" in c["data"]]
    val = [c for c in calls if c["call"] == "log" and "val/loss" in c["data"]]
    assert [c["step"] for c in train] == list(range(cfg["max_iters"]))
    for c in train:
        assert c["data"]["train/lr"] == pytest.approx(
            get_lr(c["step"], cfg["learning_rate"], cfg["warmup_iters"], cfg["max_iters"]), rel=1e-12, abs=0)
    last = cfg["max_iters"] - 1
    eval_interval = int(next(p["new"] for p in report["patches"] if p["old"] == "eval_interval = 1000").split("=")[1])
    assert [c["step"] for c in val] == sorted({*range(0, cfg["max_iters"], eval_interval), last})
    assert abs(val[0]["data"]["val/loss"] - math.log(cfg["vocab_size"])) < 0.1
    assert val[-1]["data"]["val/loss"] < val[0]["data"]["val/loss"] - 2.0
    samples = json.loads((RUNS / "webbot-cpu-shrunk" / "samples.json").read_text(encoding="utf-8"))
    assert [s["prompt"] for s in samples["samples"]] == [
        "In 1991, Linus Torvalds created", "Monday, Tuesday, Wednesday,", "Python was created by",
        "Machine learning is defined as", "The capital of Japan is"]


def test_unigram_baseline_on_the_shrunk_split(dlfs6_path):
    """只按词频猜（unigram）：在缩小版的训练数据上数词频，在它的验证数据上、用原书 evaluate 取的同一批目标 token
    算交叉熵。页面上写的是 7.61；缩小版训练 300 步后的 val/loss（7.02）比它低 0.6 nats 左右。"""
    import re
    report = json.loads((RUNS / "webbot-cpu-shrunk" / "report.json").read_text(encoding="utf-8"))
    path = dlfs6_path / "webbot" / "owt_valid.bin"
    if not path.is_file():
        pytest.skip("webbot/owt_valid.bin 没有下载（见 9.3 节的 GPU 框）")
    (t0, t1), (v0, v1) = [tuple(int(x or 0) for x in re.search(r"\[(\d*):(\d+)\]", report["data"][k]).groups())
                          for k in ("train", "val")]
    data = np.memmap(path, dtype=np.uint16, mode="r")
    train, val = np.asarray(data[t0:t1]), data[v0:v1]
    cfg = next(c["data"] for c in _calls("webbot-cpu-shrunk") if c["call"] == "config.update")
    ctx, bs = cfg["context_len"], cfg["micro_batch_size"]

    get_batch = load_defs(["get_batch"])["get_batch"]             # 原书 evaluate 取验证数据的同一个函数
    n_batches = ((len(val) - ctx - 1) // ctx) // bs + 1
    targets = torch.cat([get_batch(val, ctx, bs, "cpu", random=False, offset=b * bs * ctx)[1].reshape(-1)
                         for b in range(n_batches)]).numpy()
    p = np.bincount(train, minlength=cfg["vocab_size"]) / len(train)
    assert (p[targets] > 0).all()                                 # 验证里的 token 在训练数据里都出现过
    ce = float(-np.log(p[targets]).mean())
    final = [c for c in _calls("webbot-cpu-shrunk") if c["call"] == "log" and "val/loss" in c["data"]][-1]
    print(f"\nunigram cross-entropy {ce:.4f} on {len(targets):,} target tokens; "
          f"shrunk model val/loss {final['data']['val/loss']:.4f}")
    assert round(ce, 2) == 7.61
    assert 0.55 < ce - final["data"]["val/loss"] < 0.65


def test_recorded_ddp_run_logs_only_on_rank0(dlfs6_path):
    """web/data/ch09/webbot-cpu-ddp2：原书脚本走 DDP 分支、2 个进程；只有 rank 0 调用 wandb，
    而它记下的 effective_batch_size 只是单卡的 micro×accum。改动的行与单进程的缩小版相比，只多出 4 行
    （nccl → gloo、cuda:{local_rank} → cpu、去掉 torch.cuda.set_device、DDP 不传 device_ids）。"""
    report = json.loads((RUNS / "webbot-cpu-ddp2" / "report.json").read_text(encoding="utf-8"))
    assert report["world_size"] == 2 and report["exit_codes"] == [0, 0]
    shrunk = json.loads((RUNS / "webbot-cpu-shrunk" / "report.json").read_text(encoding="utf-8"))
    old = lambda r: [p["old"] for p in r["patches"] if p.get("file", "ch09/01_pretrain.py") == "ch09/01_pretrain.py"]  # noqa: E731
    extra = [line for line in old(report) if line not in old(shrunk)]
    assert set(old(shrunk)) <= set(old(report)) and len(old(report)) == len(old(shrunk)) + 4
    assert extra == ["dist.init_process_group(backend='nccl')", "device = torch.device(f'cuda:{local_rank}')",
                     "torch.cuda.set_device(device)", "model = DDP(model, device_ids=[local_rank])"]
    calls = _calls("webbot-cpu-ddp2")
    assert report["wandb_calls_per_rank"]["1"] == 0
    assert report["wandb_calls_per_rank"]["0"] == len(calls)     # 记录下来的每一次调用都来自 rank 0
    cfg = next(c["data"] for c in calls if c["call"] == "config.update")
    assert cfg["effective_batch_size"] == cfg["micro_batch_size"] * cfg["accumulation_steps"]   # 没有乘卡数
    train = [c for c in calls if c["call"] == "log" and "train/loss" in c["data"]]
    assert [c["step"] for c in train] == list(range(cfg["max_iters"]))  # 每步一条，不会因为有 2 个进程而重复


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="记录的是这台 Windows 机器上的现象")
def test_windows_torch_has_gloo_but_no_nccl_and_no_torchrun(tmp_path):
    """这台 Windows 上的 PyTorch：gloo 可用、NCCL 不可用；torchrun 起不来（这个构建没有带 libuv），
    设 USE_LIBUV=0 也一样。所以 9.3 节的 2 进程运行绕过 torchrun，手动给每个进程设环境变量。"""
    import os
    import subprocess
    import torch.distributed as dist
    assert dist.is_gloo_available() and not dist.is_nccl_available()
    script = tmp_path / "hello.py"
    script.write_text("print('hello from a worker')\n", encoding="utf-8")
    base = {k: v for k, v in os.environ.items() if k != "USE_LIBUV"}
    for extra in ({}, {"USE_LIBUV": "0"}):
        proc = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                               "--nproc_per_node=1", str(script)], env={**base, "PYTHONUTF8": "1", **extra},
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        assert proc.returncode != 0, extra
        assert "use_libuv was requested but PyTorch was built without libuv support" in proc.stderr, extra
        assert "hello from a worker" not in proc.stdout


WANDB_PROBE = """
import wandb
wandb.init(project="webbot-pretrain", config={"micro_batch_size": 32})
wandb.log({"train/loss": 10.8, "train/lr": 0.0}, step=0)
wandb.finish()
"""


def test_wandb_offline_and_disabled(tmp_path):
    """和原书一样的三行调用：WANDB_MODE=disabled 时一个文件都不写（当前目录、主目录、临时目录和 wandb 的
    缓存、配置、数据目录都指到 tmp_path 下，最后整棵树里一个文件都没有）；offline 时写到当前目录的
    wandb/offline-run-*，wandb 命令行也确实有 sync 子命令可以以后上传。每种模式用一个新进程，
    因为 wandb 在进程里只读一次这个环境变量。（wandb 在默认的 dev 依赖组里；没装时本测试跳过。）"""
    import os
    import subprocess
    pytest.importorskip("wandb")
    from wandb.cli.cli import cli
    assert "sync" in cli.commands                         # wandb sync：把 offline 的运行上传到服务器
    homes = ("HOME", "USERPROFILE", "TMP", "TEMP", "APPDATA", "LOCALAPPDATA", "XDG_CACHE_HOME",
             "XDG_CONFIG_HOME", "WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR")
    procs = {}
    for mode in ("disabled", "offline"):                 # 两个进程同时跑，省一半时间
        root = tmp_path / mode
        (root / "cwd").mkdir(parents=True)
        dirs = {name: str(root / name.lower()) for name in homes}
        for d in dirs.values():
            os.makedirs(d)
        env = {**os.environ, **dirs, "WANDB_MODE": mode, "WANDB_SILENT": "true", "PYTHONUTF8": "1"}
        for k in ("WANDB_API_KEY", "WANDB_DIR"):          # WANDB_DIR 不设：运行目录就是当前目录下的 wandb/
            env.pop(k, None)
        procs[mode] = subprocess.Popen([sys.executable, "-c", WANDB_PROBE], cwd=root / "cwd", env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for mode, proc in procs.items():
        _, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, err[-2000:]
        root = tmp_path / mode
        offline = list((root / "cwd" / "wandb").glob("offline-run-*"))
        assert (len(offline) == 1) == (mode == "offline"), (mode, offline)
        if mode == "disabled":
            assert [p for p in root.rglob("*") if p.is_file()] == []
