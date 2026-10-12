"""第 9 章 9.7 节：在 CPU 上跑“缩小版”的原书脚本 ch09/01_pretrain.py（和 02_generate.py），并记录运行报告。

    uv run python verify/_ch09_shrunk_run.py --work <临时目录>                  # 单进程
    uv run python verify/_ch09_shrunk_run.py --work <临时目录> --ddp 2 --iters 20 --no-generate
                                                                                # 2 个 gloo 进程，走原书的 DDP 分支

做法：把原书脚本复制进工作目录，逐条做“必须恰好匹配一次”的文本替换（缩小模型、减少步数、关掉 torch.compile；
DDP 时再把 nccl/cuda 换成 gloo/cpu），训练循环、学习率调度、梯度累积和日志代码一字不改。
wandb 换成一个“记录调用”的替身模块：它把 init / config.update / log / finish 的参数写进 JSONL，
不联网、不需要账号。这份记录就是页面上曲线的数据来源。

输出（页面读取）：web/data/ch09/<run>/report.json、wandb.jsonl、samples.json（只有单进程运行才生成样例）
工作目录里留下的 checkpoint 和日志不进仓库。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
OUT = ROOT / "web" / "data" / "ch09"

SPLIT = 58_046_459      # owt_valid.bin 的前 90% 当训练数据（owt_train.bin 有 5.3 GB，这里不下载）
VAL_TOKENS = 65_536     # 紧接着的 65,536 个 token 当验证数据

COPY = ["ch09/01_pretrain.py", "ch09/02_generate.py", "webbot/model.py", "webbot/utils.py",
        "storybot/tokenizer.py", "webbot/merge_rules.pkl"]
LINK = ["webbot/owt_valid.bin"]

WANDB_STUB = '''"""记录调用的 wandb 替身：把参数写进 $WANDB_STUB_LOG（JSONL）。不联网，也不需要账号。"""
import json, os

_path = os.environ["WANDB_STUB_LOG"]


def _write(call, **payload):
    with open(_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"call": call, **payload}, ensure_ascii=False) + "\\n")


class _Config(dict):
    def update(self, d=None, **kw):
        d = dict(d or {}, **kw)
        super().update(d)
        _write("config.update", data=d)


config = _Config()


def init(project=None, config=None, **kw):
    _write("init", project=project, config=dict(config or {}))


def log(data, step=None, **kw):
    _write("log", step=step, data={k: float(v) for k, v in data.items()})


def finish(**kw):
    _write("finish")
'''


def pretrain_patches(a) -> list[tuple[str, str]]:
    p = [
        ("context_len = 1024", "context_len = 128"),
        ("embed_dim = 768", "embed_dim = 128"),
        ("n_head = 12", "n_head = 4"),
        ("n_kv_head = 4", "n_kv_head = 2"),
        ("n_layer = 12", "n_layer = 2"),
        ("ff_dim = 2048", "ff_dim = 384"),
        ("micro_batch_size = 32", f"micro_batch_size = {a.micro}"),
        ("warmup_iters = 500", f"warmup_iters = {a.warmup}"),
        ("max_iters = 100000", f"max_iters = {a.iters}"),
        ("eval_interval = 1000", f"eval_interval = {a.eval_interval}"),
        ("train_data_path = 'webbot/owt_train.bin'", "train_data_path = 'webbot/owt_valid.bin'"),
        ("train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')",
         f"train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')[:{SPLIT}]"),
        ("val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')",
         f"val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')[{SPLIT}:{SPLIT + VAL_TOKENS}]"),
        ("model = torch.compile(model)", "pass  # model = torch.compile(model)：这台 Windows 机器上没有 cl.exe"),
    ]
    if a.ddp:
        p += [
            ("dist.init_process_group(backend='nccl')", "dist.init_process_group(backend='gloo')"),
            ("device = torch.device(f'cuda:{local_rank}')", "device = torch.device('cpu')"),
            ("torch.cuda.set_device(device)", "pass  # torch.cuda.set_device(device)"),
            ("model = DDP(model, device_ids=[local_rank])", "model = DDP(model)"),
        ]
    return p


GENERATE_PATCHES = [
    ("device = get_device()", "device = get_device()\nimport torch; torch.manual_seed(0)  # 采样可复现（原脚本没有固定种子）"),
]


def apply(path: Path, patches: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in patches:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path.name}: expected exactly one {old!r}, found {n}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8", newline="\n")


def make_workspace(work: Path) -> None:
    for rel in COPY:
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DLFS6 / rel, dst)
    for rel in LINK:
        dst = work / rel
        if not dst.exists():
            try:
                os.link(DLFS6 / rel, dst)
            except OSError:
                shutil.copy2(DLFS6 / rel, dst)
    stub = work / "_stub"
    stub.mkdir(exist_ok=True)
    (stub / "wandb.py").write_text(WANDB_STUB, encoding="utf-8")


def base_env(work: Path) -> dict:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
            "CUDA_VISIBLE_DEVICES": "-1",            # 只用 CPU（Windows 上空字符串的环境变量会被丢掉，所以用 -1）
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(work / "_stub"), "USE_LIBUV": "0"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_pretrain(work: Path, a) -> tuple[float, list[int], list[Path]]:
    env = base_env(work)
    logs, procs = [], []
    world = a.ddp or 1
    port = free_port()
    t0 = time.time()
    for rank in range(world):
        e = dict(env, WANDB_STUB_LOG=str(work / f"wandb_rank{rank}.jsonl"))
        if a.ddp:
            e.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                     MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), OMP_NUM_THREADS=str(a.threads))
        (work / f"wandb_rank{rank}.jsonl").unlink(missing_ok=True)
        log = work / f"pretrain_rank{rank}.log"
        logs.append(log)
        procs.append(subprocess.Popen([sys.executable, "ch09/01_pretrain.py"], cwd=work, env=e,
                                      stdout=open(log, "wb"), stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    return round(time.time() - t0, 1), codes, logs


def parse_generate(text: str, prompts: list[str]) -> list[dict]:
    """02_generate.py 的输出：====、プロンプト i: …、====、续写正文、空行。"""
    out = []
    blocks = re.split(r"={70}\nプロンプト \d+: .*\n={70}\n", text)[1:]
    for prompt, block in zip(prompts, blocks):
        out.append({"prompt": prompt, "text": block.rstrip("\n")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True, help="工作目录（复制原书脚本、写 checkpoint 的地方）")
    ap.add_argument("--run", default=None, help="运行名，默认 webbot-cpu-shrunk 或 webbot-cpu-ddp<N>")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--ddp", type=int, default=0, help="进程数；0 表示不走 DDP 分支")
    ap.add_argument("--threads", type=int, default=4, help="DDP 时每个进程的 OMP 线程数")
    ap.add_argument("--no-generate", action="store_true")
    a = ap.parse_args()
    run = a.run or (f"webbot-cpu-ddp{a.ddp}" if a.ddp else "webbot-cpu-shrunk")
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    make_workspace(work)
    patches = {"ch09/01_pretrain.py": pretrain_patches(a)}
    apply(work / "ch09/01_pretrain.py", patches["ch09/01_pretrain.py"])

    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")   # 本地时间，带时区
    seconds, codes, logs = run_pretrain(work, a)
    print(f"pretrain exit={codes} in {seconds}s", flush=True)
    if any(codes):
        print(logs[0].read_text(encoding="utf-8", errors="replace")[-3000:])
        raise SystemExit(1)

    import torch
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)
    calls = {r: [json.loads(line) for line in (work / f"wandb_rank{r}.jsonl").read_text(encoding="utf-8").splitlines()]
             if (work / f"wandb_rank{r}.jsonl").exists() else [] for r in range(a.ddp or 1)}
    (out / "wandb.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in calls[0]),
                                     encoding="utf-8")
    log0 = logs[0].read_text(encoding="utf-8", errors="replace")
    report = {
        "run": run,
        "what": "缩小版 ch09/01_pretrain.py（CPU）。训练循环、调度、梯度累积、日志代码与原书一致，只改了下面列出的行。",
        "dlfs6_commit": subprocess.check_output(["git", "-C", str(DLFS6), "rev-parse", "HEAD"], text=True).strip(),
        "started": started,
        "host": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0],
                 "torch": torch.__version__, "cpu_count": os.cpu_count(),
                 "note": "运行时机器上还有其他任务在占用 CPU，墙钟时间只是上限"},
        "world_size": a.ddp or 1,
        "command": " ".join(["uv run python verify/_ch09_shrunk_run.py --work <dir>"]
                            + ([f"--ddp {a.ddp}"] if a.ddp else [])
                            + [f"--iters {a.iters} --micro {a.micro} --warmup {a.warmup} --eval-interval {a.eval_interval}"]
                            + (["--no-generate"] if a.no_generate else [])),
        "patches": [{"file": f, "old": o, "new": n} for f, ps in patches.items() for o, n in ps],
        "data": {"train": f"webbot/owt_valid.bin[:{SPLIT}]", "val": f"webbot/owt_valid.bin[{SPLIT}:{SPLIT + VAL_TOKENS}]"},
        "seconds": {"pretrain": seconds},
        "exit_codes": codes,
        "wandb_calls_per_rank": {str(r): len(c) for r, c in calls.items()},
        "stdout_head": [l for l in log0.replace("\r", "\n").splitlines()
                        if l.startswith(("学習データ", "検証データ", "パラメータ数", "step "))][:40],
        "warnings": sorted(set(re.findall(r"\w*Warning: [^\r\n]*", log0)))[:5],
        "cost_usd": 0.0,
    }

    if not a.no_generate and not a.ddp:
        apply(work / "ch09/02_generate.py", GENERATE_PATCHES)
        report["patches"] += [{"file": "ch09/02_generate.py", "old": o, "new": n} for o, n in GENERATE_PATCHES]
        t0 = time.time()
        proc = subprocess.run([sys.executable, "ch09/02_generate.py"], cwd=work, env=base_env(work),
                              capture_output=True, text=True, encoding="utf-8")
        report["seconds"]["generate"] = round(time.time() - t0, 1)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-3000:])
            raise SystemExit(1)
        prompts = ["In 1991, Linus Torvalds created", "Monday, Tuesday, Wednesday,", "Python was created by",
                   "Machine learning is defined as", "The capital of Japan is"]
        samples = parse_generate(proc.stdout, prompts)
        (out / "samples.json").write_text(json.dumps({"temperature": 0.5, "max_new_tokens": 100, "seed": 0,
                                                      "samples": samples}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
