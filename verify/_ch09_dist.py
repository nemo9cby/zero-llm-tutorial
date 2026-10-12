"""第 9 章的多进程验证：在 CPU 上用 gloo 起真正的 2 个进程，跑 DDP 与 ZeRO-1。

test_ch09.py 用 torch.multiprocessing.spawn 调用这里的 worker（spawn 要求 worker 定义在可导入的模块里）。
模型是原书 webbot/model.py 的 GPT，只是尺寸缩小；数据由 (rank, step, micro_step) 决定，主进程可以复现。
每个 worker 把结果 torch.save 到 out_dir，由测试在主进程里核对。
"""
from __future__ import annotations

import contextlib
import sys

sys.dont_write_bytecode = True   # 子进程导入 webbot.model 时不要在 third_party/dlfs6 里留下 __pycache__

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TINY = dict(vocab_size=256, max_context_len=32, embed_dim=32, n_head=4, n_kv_head=2,
            n_layer=2, ff_dim=64, theta=10000)
CTX, MICRO, ACCUM = 16, 2, 4


def micro_batch(rank: int, step: int, micro: int):
    """每个 (rank, step, micro_step) 一份固定的数据；主进程用同样的种子就能拼出“大 batch”。"""
    g = torch.Generator().manual_seed(10_000 * rank + 100 * step + micro)
    seq = torch.randint(0, TINY["vocab_size"], (MICRO, CTX + 1), generator=g)
    return seq[:, :-1], seq[:, 1:]


def make_model(seed: int):
    from webbot.model import GPT
    torch.manual_seed(seed)
    return GPT(**TINY)


def lm_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))


def flat_grads(model) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1).clone() for p in model.parameters()])


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])


def _init(rank: int, world: int, init_file: str, dlfs6: str) -> None:
    if dlfs6 not in sys.path:
        sys.path.insert(0, dlfs6)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=world)


def worker(rank: int, world: int, init_file: str, dlfs6: str, out_dir: str, steps: int, lr: float) -> None:
    """一次起进程，先后跑 DDP 和 ZeRO-1 两个实验（Windows 上起一次进程要十几秒，合在一起省时间）。"""
    _init(rank, world, init_file, dlfs6)
    ddp_part(rank, out_dir)
    zero_part(rank, out_dir, steps, lr)
    dist.destroy_process_group()


def ddp_part(rank: int, out_dir: str) -> None:
    """同一进程里依次跑两种写法：原书写法（只有 backward 在 no_sync 里）和修正写法（forward 也在里面）。"""
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = make_model(seed=42 + rank)            # 原书：torch.manual_seed(42 + seed_offset)，各 rank 初值不同
    before = flat_params(model)
    ddp = DDP(model)                              # CPU 上不传 device_ids
    after = flat_params(model)

    calls: list[int] = []

    def counting_hook(state, bucket):             # 每次真正发起 all-reduce 都会经过这里
        calls.append(bucket.buffer().numel())
        return allreduce_hook(None, bucket)

    ddp.register_comm_hook(None, counting_hook)

    result = {"before": before, "after": after}
    for pattern in ("book", "fixed"):
        ddp.zero_grad(set_to_none=True)
        per_micro = []
        for micro in range(ACCUM):
            n0 = len(calls)
            x, y = micro_batch(rank, 0, micro)
            last = micro == ACCUM - 1
            if pattern == "book":                 # ch09/01_pretrain.py 的写法
                loss = lm_loss(ddp, x, y) / ACCUM
                if not last:
                    with ddp.no_sync():
                        loss.backward()
                else:
                    loss.backward()
            else:                                 # 页面上的修正写法：forward 和 backward 都放进 no_sync
                sync = last
                with contextlib.nullcontext() if sync else ddp.no_sync():
                    loss = lm_loss(ddp, x, y) / ACCUM
                    loss.backward()
            per_micro.append(len(calls) - n0)
        result[pattern] = {"grads": flat_grads(model), "allreduce_calls_per_micro": per_micro}
    torch.save(result, f"{out_dir}/ddp_rank{rank}.pt")


def zero_part(rank: int, out_dir: str, steps: int, lr: float) -> None:
    """ZeroRedundancyOptimizer(AdamW)：每个 rank 只保存一部分参数的 Adam 状态。"""
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = make_model(seed=42)
    ddp = DDP(model)
    opt = ZeroRedundancyOptimizer(ddp.parameters(), optimizer_class=torch.optim.AdamW, lr=lr)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = micro_batch(rank, step, 0)
        lm_loss(ddp, x, y).backward()
        opt.step()
    local_state = sum(t.numel() for st in opt.optim.state.values() for k, t in st.items()
                      if k in ("exp_avg", "exp_avg_sq"))
    torch.save({"params": flat_params(model), "local_state_numel": local_state,
                "n_params": sum(p.numel() for p in model.parameters())}, f"{out_dir}/zero_rank{rank}.pt")
