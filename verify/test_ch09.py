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


def saved_bytes(model, ids, use_checkpoint: bool) -> int:
    """前向时为反向保存下来的激活一共多少字节（按底层 storage 去重，不算参数和 buffer 本身）。
    用了检查点时，Block 内部的张量不再保存，但每个 Block 的输入要留着，这部分单独加上。"""
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
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        logits = forward_blocks(model, ids, use_checkpoint)
        lm_loss(logits, ids)
    for h in hooks:
        h.remove()
    return sum(seen.values())


def test_webbot_activation_memory_per_sequence(dlfs6_path):
    """全尺寸 WebBot（fp32，CPU）：一条 1024-token 的序列要为反向保存多少激活；加检查点后剩多少。"""
    from webbot.model import GPT
    torch.manual_seed(0)
    model = GPT(**WEBBOT)
    ids = torch.randint(0, WEBBOT["vocab_size"], (1, WEBBOT["max_context_len"]))
    mib = {ck: saved_bytes(model, ids, ck) / 2**20 for ck in (False, True)}
    logits_mib = WEBBOT["max_context_len"] * WEBBOT["vocab_size"] * 4 / 2**20
    print(f"\nper sequence: plain {mib[False]:.1f} MiB, checkpointed {mib[True]:.1f} MiB, "
          f"fp32 log-softmax alone {logits_mib:.1f} MiB")
    assert 900 < mib[False] < 980             # 页面上写的是“约 940 MiB”
    assert 230 < mib[True] < 255              # 检查点后约 240 MiB，其中 195 MiB 是 logits 的 log-softmax
    assert mib[False] / mib[True] > 3.5       # 减少到约 1/4


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

    book = ranks[0]["book"]["allreduce_calls_per_micro"]
    fixed = ranks[0]["fixed"]["allreduce_calls_per_micro"]
    assert all(c > 0 for c in book)                               # 原书写法：4 个 micro-step 都在通信
    assert fixed[:-1] == [0] * (D.ACCUM - 1) and fixed[-1] > 0     # forward 也放进 no_sync：只在最后通信一次


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
    W = torch.randn(96, 32)                             # 输出 96 维、输入 32 维：学习率乘 sqrt(96/32)
    G1, G2 = torch.randn(96, 32), torch.randn(96, 32)
    p = torch.nn.Parameter(W.clone())
    opt = torch.optim.Muon([p], lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True)
    expect, buf = W.clone(), torch.zeros_like(W)
    for G in (G1, G2):
        p.grad = G.clone()
        opt.step()
        buf = buf.lerp(G, 1 - 0.95)                     # B ← 0.95·B + 0.05·G（动量的指数平均写法）
        upd = G.lerp(buf, 0.95)                         # Nesterov：0.05·G + 0.95·B
        O = _zeropower_via_newtonschulz(upd, (DEFAULT_A, DEFAULT_B, DEFAULT_C), 5, EPS)
        expect = expect * (1 - 0.02 * 0.1)              # 先做解耦权重衰减（用没调整过的 lr）
        expect = expect.add(O, alpha=-0.02 * math.sqrt(96 / 32))   # 再减去 lr·sqrt(max(1, A/B))·O
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
    """web/data/ch09/webbot-cpu-shrunk：记录下来的学习率逐步等于原书 get_lr，loss 从 ln 50000 附近开始下降。"""
    report = json.loads((RUNS / "webbot-cpu-shrunk" / "report.json").read_text(encoding="utf-8"))
    calls = _calls("webbot-cpu-shrunk")
    cfg = next(c["data"] for c in calls if c["call"] == "config.update")
    assert [c["call"] for c in calls][:2] == ["init", "config.update"] and calls[-1]["call"] == "finish"
    assert report["dlfs6_commit"].startswith("c9b6e2e") and report["exit_codes"] == [0]
    assert cfg["effective_batch_size"] == cfg["micro_batch_size"] * cfg["accumulation_steps"]

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


def test_recorded_ddp_run_logs_only_on_rank0(dlfs6_path):
    """web/data/ch09/webbot-cpu-ddp2：原书脚本走 DDP 分支、2 个进程；只有 rank 0 调用 wandb，
    而它记下的 effective_batch_size 只是单卡的 micro×accum。"""
    report = json.loads((RUNS / "webbot-cpu-ddp2" / "report.json").read_text(encoding="utf-8"))
    assert report["world_size"] == 2 and report["exit_codes"] == [0, 0]
    assert report["wandb_calls_per_rank"]["1"] == 0 and report["wandb_calls_per_rank"]["0"] > 0
    cfg = next(c["data"] for c in _calls("webbot-cpu-ddp2") if c["call"] == "config.update")
    assert cfg["effective_batch_size"] == cfg["micro_batch_size"] * cfg["accumulation_steps"]
    assert cfg["effective_batch_size"] * report["world_size"] == 2 * cfg["effective_batch_size"]


WANDB_PROBE = """
import wandb
wandb.init(project="webbot-pretrain", config={"micro_batch_size": 32})
wandb.log({"train/loss": 10.8, "train/lr": 0.0}, step=0)
wandb.finish()
"""


def test_wandb_offline_and_disabled(tmp_path):
    """和原书一样的三行调用：WANDB_MODE=disabled 时一个文件都不写；offline 时写到本地 wandb/offline-run-*，
    以后可以 wandb sync。每种模式用一个新进程，因为 wandb 在进程里只读一次这个环境变量。
    （wandb 在 gpu 依赖组里；没装时本测试跳过：uv run --group gpu pytest verify/test_ch09.py -k wandb）"""
    import os
    import subprocess
    pytest.importorskip("wandb")
    procs = {}
    for mode in ("disabled", "offline"):                 # 两个进程同时跑，省一半时间
        cwd = tmp_path / mode
        cwd.mkdir()
        env = {**os.environ, "WANDB_MODE": mode, "WANDB_SILENT": "true", "PYTHONUTF8": "1"}
        env.pop("WANDB_API_KEY", None)
        procs[mode] = subprocess.Popen([sys.executable, "-c", WANDB_PROBE], cwd=cwd, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for mode, proc in procs.items():
        _, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, err[-2000:]
        cwd = tmp_path / mode
        offline = list((cwd / "wandb").glob("offline-run-*")) if (cwd / "wandb").exists() else []
        assert (len(offline) == 1) == (mode == "offline"), (mode, offline)
        if mode == "disabled":
            assert not (cwd / "wandb").exists()
