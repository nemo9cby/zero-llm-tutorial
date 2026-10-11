"""Run the DLFS6 (ゼロから作るDeep Learning ❻) training scripts on Modal GPUs.

Layout on the Volume "zero-llm":  /vol/ws/<repo-relative path>  (datasets, tokenizers, checkpoints)
The book scripts read/write repo-relative paths, so each run links /vol/ws into the repo checkout
and copies new *.pt / *.png back afterwards. That also chains stages (pretrain -> SFT -> GRPO/DPO).

  modal run modal_app.py::prepare --bot storybot                      # HF -> Volume, CPU only
  modal run --detach modal_app.py --script ch06/05_pretrain.py --gpu L4
  modal run --detach modal_app.py::storybot_resumable                 # checkpoint/resume demo
  modal run --detach modal_app.py::ddp --script ch09/01_pretrain.py   # 8 x H100, torchrun
  modal volume get zero-llm ws/storybot/model_pretrain.pt .
"""
import os
import shutil
import subprocess
from pathlib import Path

import modal

LOCAL_REPO = Path(os.environ.get("DLFS6_DIR", Path(__file__).parent / "dlfs6"))  # git clone of the book repo
REMOTE_REPO = "/root/dlfs6"
REPO = os.environ.get("ZLLM_REPO", REMOTE_REPO)  # env overrides only for local `.local()` smoke tests
WS = Path(os.environ.get("ZLLM_WS", "/vol/ws"))
HF = "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main"
FILES = {  # what each bot needs that is NOT committed to the repo (codebot data ships with the repo)
    "codebot": ["model_pretrain.pt", "model_sft.pt"],
    "storybot": ["tiny_stories_train.bin", "tiny_stories_valid.bin", "merge_rules.pkl", "model_pretrain.pt"],
    "webbot": ["owt_train.bin", "owt_valid.bin", "merge_rules.pkl"],
}

vol = modal.Volume.from_name("zero-llm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")  # Triton (torch.compile in ch09) shells out to gcc
    .uv_pip_install("torch==2.14.1", "numpy<2", "matplotlib", "tqdm", "regex", "wandb")  # Linux wheel = CUDA 13.0
    .env({"PYTHONUNBUFFERED": "1", "WANDB_MODE": os.environ.get("WANDB_MODE", "disabled")})
    .add_local_dir(LOCAL_REPO, REMOTE_REPO, ignore=["**/.git/**", "**/.venv/**", "storybot/*.bin", "storybot/*.txt",
                                             "webbot/*.bin", "webbot/*.txt", "**/*.pt"])
)
app = modal.App("zero-llm", image=image)
RUN = dict(volumes={"/vol": vol}, timeout=24 * 3600, single_use_containers=True,
           retries=modal.Retries(initial_delay=0.0, max_retries=3))  # GPU containers are preemptible


def _commit() -> None:
    if not modal.is_local():  # inside a container: persist now (Volumes also auto-commit every few seconds)
        vol.commit()


def _link_workspace(copy_over_bytes: int = 1 << 30) -> set[Path]:
    """Expose /vol/ws/** inside the repo. Big memmapped corpora are copied to local disk for fast random reads."""
    ws, linked = WS, set()
    for src in ws.rglob("*"):
        if not src.is_file():
            continue
        dst = Path(REPO) / src.relative_to(ws)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        if src.stat().st_size > copy_over_bytes:
            shutil.copyfile(src, dst)
        else:
            dst.symlink_to(src)
        linked.add(dst)
    return linked


def _save_outputs(linked: set[Path]) -> None:
    for f in Path(REPO).rglob("*"):
        if f.suffix in {".pt", ".png"} and f.is_file() and not f.is_symlink() and f not in linked:
            dst = WS / f.relative_to(REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            print(f"saved {dst}")
    _commit()


@app.function(volumes={"/vol": vol}, timeout=3 * 3600)
def prepare(bot: str = "storybot", include_models: bool = False):
    """Download a bot's data from the book's Hugging Face dataset straight into the Volume."""
    import urllib.request
    out = WS / bot
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES[bot]:
        if name.endswith(".pt") and not include_models:
            continue
        dst = out / name
        if not dst.exists():
            print(f"downloading {bot}/{name}")
            urllib.request.urlretrieve(f"{HF}/{bot}/{name}", dst)
        print(f"{dst}: {dst.stat().st_size / 1e6:.1f} MB")
    _commit()


@app.function(gpu="L4", **RUN)
def book(script: str):
    """Run one book script unchanged, e.g. ch03/01_pretrain.py or ch06/05_pretrain.py."""
    linked = _link_workspace()
    subprocess.run(["nvidia-smi", "-L"], check=False)
    subprocess.run(["python", script], cwd=REPO, check=True)
    _save_outputs(linked)


@app.function(gpu="H100:8", **RUN)
def ddp(script: str = "ch09/01_pretrain.py", nproc: int = 8):
    """Single-node multi-GPU run with torchrun (the ch09 script initialises DDP from RANK/LOCAL_RANK)."""
    linked = _link_workspace()
    subprocess.run(["torchrun", "--standalone", f"--nproc-per-node={nproc}", script], cwd=REPO, check=True)
    _save_outputs(linked)


@app.function(gpu="L4", **RUN)
def storybot_resumable(max_iters: int = 40000, ckpt_every: int = 1000, batch_size: int = 32, max_lr: float = 1e-3):
    """ch06 StoryBot pretraining loop with atomic checkpoints, so a preempted container resumes, not restarts."""
    import sys
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, REPO)
    from storybot.model import GPT

    ws = WS / "storybot"
    ckpt = ws / "resume.pt"
    data = np.memmap(ws / "tiny_stories_train.bin", dtype=np.uint16, mode="r")
    model = GPT(10000, 256, 512, 16, 4, 1344, 10000).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr)
    start = 0
    if ckpt.exists():
        s = torch.load(ckpt, map_location="cpu", weights_only=False)  # RNG state must stay a CPU ByteTensor
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); start = s["step"] + 1
        torch.set_rng_state(s["rng"]); print(f"resuming at step {start}")

    def lr_at(it, warmup=200):
        return max_lr * it / warmup if it < warmup else max_lr * (1 - (it - warmup) / (max_iters - warmup))

    for it in range(start, max_iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        ix = torch.randint(len(data) - 257, (batch_size,))
        xy = torch.stack([torch.from_numpy(data[i:i + 257].astype(np.int64)) for i in ix]).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xy[:, :-1])
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), xy[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 500 == 0:
            print(f"step {it} loss {loss.item():.4f}")
        if (it + 1) % ckpt_every == 0 or it == max_iters - 1:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": it, "rng": torch.get_rng_state()}, tmp)
            tmp.replace(ckpt)  # atomic: a preemption mid-save never leaves a torn checkpoint
            _commit()
    model.save(str(ws / "model_pretrain.pt"))
    _commit()


@app.local_entrypoint()
def main(script: str = "ch03/01_pretrain.py", gpu: str = "L4"):
    # .spawn().get() rather than .remote(): Modal's long-training example notes .remote() calls expire after 24 h.
    book.with_options(gpu=gpu).spawn(script).get()
