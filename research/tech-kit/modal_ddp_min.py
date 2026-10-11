"""Minimal 8-GPU DDP job on Modal: torchrun inside one container (single node, 8 x H100).
  modal run modal_app.py::prepare --bot webbot      # once: data -> Volume "zero-llm" (see modal_app.py)
  modal run --detach modal_ddp_min.py
"""
import os
import shutil
import subprocess
from pathlib import Path

import modal

LOCAL_REPO = Path(os.environ.get("DLFS6_DIR", Path(__file__).parent / "dlfs6"))
vol = modal.Volume.from_name("zero-llm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")                        # gcc for Triton / torch.compile
    .uv_pip_install("torch==2.14.1", "numpy<2", "matplotlib", "tqdm", "regex", "wandb")
    .env({"WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})  # set "online" + add the wandb secret to log
    .add_local_dir(LOCAL_REPO, "/root/dlfs6", ignore=["**/.git/**", "**/.venv/**", "**/*.bin", "**/*.pt",
                                                      "storybot/*.txt", "webbot/*.txt"])
)
app = modal.App("zero-llm-ddp", image=image)


@app.function(gpu="H100:8", volumes={"/vol": vol}, timeout=24 * 3600,
              retries=modal.Retries(initial_delay=0.0, max_retries=2), single_use_containers=True)
              # secrets=[modal.Secret.from_name("wandb-secret")]  <- after `modal secret create wandb-secret WANDB_API_KEY=...`
def train(max_iters: int = 12_500):
    repo = Path("/root/dlfs6")
    for f in ("owt_train.bin", "owt_valid.bin"):        # 5.3 GB memmap: local NVMe beats network FS for random reads
        shutil.copyfile(f"/vol/ws/webbot/{f}", repo / "webbot" / f)
    script = repo / "ch09" / "01_pretrain.py"           # 8 ranks x 131k tokens/step: 12.5k steps = the 1-GPU 100k-step token budget
    src = script.read_text().replace("max_iters = 100000", f"max_iters = {max_iters}")
    (repo / "ch09" / "01_pretrain_modal.py").write_text(src)
    subprocess.run(["torchrun", "--standalone", "--nproc-per-node=8", "ch09/01_pretrain_modal.py"], cwd=repo, check=True)
    out = Path("/vol/ws/webbot"); out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / "webbot" / "model_pretrain.pt", out / "model_pretrain.pt")
    vol.commit()


@app.local_entrypoint()
def main(max_iters: int = 12_500):
    train.spawn(max_iters).get()
