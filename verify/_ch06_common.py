"""第 6 章本地验证的共用工具：从原书脚本里只取定义、加载 HF 上的 StoryBot 权重、在留出数据上算 loss。

原书的 ch06 脚本在模块顶层就开始训练（或调用付费 API），不能直接 import。
load_defs() 只执行脚本里的 import / def / class 语句，所以测试用到的一定是原书写的那份函数。
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
STORYBOT = DLFS6 / "storybot"

# StoryBot 的配置：与 ch06/05_pretrain.py 第 83–93 行一致（也与 CS336 A1 的 TinyStories 配置一致）。
STORYBOT_CONFIG = dict(vocab_size=10000, max_context_len=256, embed_dim=512, n_head=16, n_layer=4,
                       ff_dim=1344, theta=10000)


def ensure_dlfs6_on_path() -> None:
    if str(DLFS6) not in sys.path:
        sys.path.insert(0, str(DLFS6))


def load_defs(rel: str, names: list[str] | None = None) -> dict:
    """Execute only the import/def/class statements of a DLFS6 script; return the namespace.

    Module-level statements (training loops, os.chdir, API calls) are skipped on purpose.
    """
    ensure_dlfs6_on_path()
    os.environ.setdefault("MPLBACKEND", "Agg")
    src = (DLFS6 / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
    body = [n for n in tree.body if isinstance(n, keep)]
    ns: dict = {"__name__": f"dlfs6_{Path(rel).stem}"}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(DLFS6 / rel), "exec"), ns)
    if names:
        missing = [n for n in names if n not in ns]
        if missing:
            raise KeyError(f"{rel}: {missing} not defined")
    return ns


def function_source(rel: str, name: str) -> str:
    """Source of one top-level function with comments and blank lines removed (for 'identical' checks)."""
    src = (DLFS6 / rel).read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise KeyError(f"{name} not found in {rel}")


def rename_hf_keys(state_dict: dict) -> dict:
    """HF 上发布的 StoryBot 权重把 SwiGLU 子模块叫 `mlp`，仓库里的 storybot/model.py 叫 `ffn`。"""
    return {k.replace(".mlp.", ".ffn."): v for k, v in state_dict.items()}


def load_storybot(path: Path | str, device: str = "cpu"):
    """Like GPT.load_from, but accepts the published checkpoints (mlp -> ffn)."""
    import torch

    ensure_dlfs6_on_path()
    from storybot.model import GPT

    ckpt = torch.load(path, map_location=device)
    model = GPT(ckpt["vocab_size"], ckpt["max_context_len"], ckpt["embed_dim"], ckpt["n_head"],
                ckpt["n_layer"], ckpt["ff_dim"], ckpt["theta"])
    model.load_state_dict(rename_hf_keys(ckpt["model_state_dict"]))
    return model.to(device)


def heldout_tokens(n_tokens: int):
    """The first n_tokens (+1 for the shifted targets) of storybot/tiny_stories_valid.bin.

    The published checkpoints were trained on tiny_stories_train.bin only, so this split is held out.
    """
    import numpy as np

    data = np.memmap(STORYBOT / "tiny_stories_valid.bin", dtype=np.uint16, mode="r")
    return np.asarray(data[: n_tokens + 1], dtype=np.int64)


def lm_loss(model, tokens, context_len: int = 256, batch_size: int = 32) -> float:
    """Mean next-token cross-entropy (nats) over non-overlapping windows, fp32 (no autocast).

    Same windowing as evaluate() in ch06/05_pretrain.py, except that every window that fits is used.
    """
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(tokens)
    n_win = (len(t) - 1) // context_len
    x = t[: n_win * context_len].view(n_win, context_len)
    y = t[1: n_win * context_len + 1].view(n_win, context_len)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, n_win, batch_size):
            logits = model(x[i:i + batch_size])
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[i:i + batch_size].reshape(-1),
                                     reduction="sum").item()
    return total / y.numel()


CHAPTER_MD = ROOT / "content" / "ch06.md"


def chapter_cells(upto: str | None = None) -> dict:
    """Run the chapter's NumPy verify cells in order (shared namespace, like the browser) and return it.

    Lets a pytest compare the exact code shown on the page against the real PyTorch implementation.
    """
    import contextlib
    import io
    import re

    text = CHAPTER_MD.read_text(encoding="utf-8")
    ns: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(ROOT / "web")  # cells read data files as data/<slug>/..., relative to the site root
    try:
        for m in re.finditer(r"^```verify[ \t]+id=(\S+)[^\n]*\n(.*?)^```[ \t]*$", text, re.S | re.M):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(m.group(2), f"<verify {m.group(1)}>", "exec"), ns)
            if m.group(1) == upto:
                return ns
    finally:
        os.chdir(cwd)
    if upto is not None:
        raise KeyError(f"verify cell {upto} not found in {CHAPTER_MD}")
    return ns
