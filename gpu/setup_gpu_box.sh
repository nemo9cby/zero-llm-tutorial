#!/usr/bin/env bash
# Set up the tutorial on a rented Linux GPU box (RunPod pod or Lambda instance), reproducibly.
#
#   RunPod : WORK=/workspace                   (network volume or volume disk; survives a pod stop)
#   Lambda : WORK=/lambda/nfs/<filesystem>     (persistent filesystem attached at launch)
#
# The tutorial reaches the box either from a git remote, or as a bundle you copy over (no remote needed):
#   on your PC :  git bundle create zllm.bundle HEAD  &&  scp -P <port> zllm.bundle root@<ip>:/workspace/
#   on the box :  WORK=/workspace TUTORIAL_SRC=/workspace/zllm.bundle BOTS="storybot" bash setup_gpu_box.sh
#
# BOTS     : any of codebot storybot webbot (what to download; see gpu/jobs.py)
# DRY_RUN=1: print the commands instead of running them (verify/test_gpu.py uses this)
set -euo pipefail

WORK="${WORK:-/workspace}"
BOTS="${BOTS:-storybot}"
TUTORIAL_SRC="${TUTORIAL_SRC:-}"
DRY_RUN="${DRY_RUN:-0}"
DLFS6_COMMIT=c9b6e2ed531b08dd9f451a091a34e9645148e2e2   # the book commit the tutorial's submodule pins
REPO="$WORK/zero-llm-tutorial"

run() { if [ "$DRY_RUN" = 1 ]; then echo "+ $*"; else "$@"; fi; }

export UV_CACHE_DIR="$WORK/.cache/uv"            # wheels and Python stay on the persistent disk
export UV_PYTHON_INSTALL_DIR="$WORK/.cache/python"
export UV_LINK_MODE=copy                          # cache and venv may sit on different filesystems

echo "== GPU and driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)

# uv.lock pins torch 2.14.1; on Linux that PyPI wheel is built for CUDA 13.0 and needs driver >= 580.
# With an older driver (>= 525) install the same torch version built for CUDA 12.6 instead. It covers
# sm_80-sm_90 (A100, A10, L4, L40S, RTX 4090, H100) but not Blackwell; B200 hosts ship new drivers anyway.
if [ "$DRIVER_MAJOR" -ge 580 ]; then
  TORCH_INDEX=""
elif [ "$DRIVER_MAJOR" -ge 525 ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
  echo "driver $DRIVER_MAJOR is older than 525: pick another host" >&2
  exit 1
fi
echo "torch 2.14.1 build: ${TORCH_INDEX:-PyPI wheel (CUDA 13.0)}"

if ! command -v uv >/dev/null 2>&1; then
  run sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "== tutorial and pinned book code"
run mkdir -p "$WORK"
if [ ! -d "$REPO/.git" ]; then
  if [ -z "$TUTORIAL_SRC" ]; then
    echo "set TUTORIAL_SRC to a git URL or a bundle file (see the header of this script)" >&2
    exit 1
  fi
  run git clone "$TUTORIAL_SRC" "$REPO"
fi
run cd "$REPO"
run git submodule update --init third_party/dlfs6       # the book, from GitHub, at the pinned commit
if [ "$DRY_RUN" != 1 ]; then
  have=$(git -C third_party/dlfs6 rev-parse HEAD)
  [ "$have" = "$DLFS6_COMMIT" ] || { echo "book commit $have != pinned $DLFS6_COMMIT" >&2; exit 1; }
fi

echo "== Python environment (uv.lock)"
run uv sync --frozen
ENV_FILE="$WORK/zllm.env"
if [ -n "$TORCH_INDEX" ]; then
  # `--index` ranks above PyPI (uv's default index), so torch comes from the CUDA 12.6 index. Do not write
  # `--index-url <cu126> --extra-index-url <PyPI>`: uv ranks every --extra-index-url *above* --index-url and
  # would reinstall PyPI's CUDA 13.0 wheel. The +cu126 pin turns a wrong pick into an error.
  run uv pip install --reinstall-package torch "torch==2.14.1+cu126" --index "$TORCH_INDEX"
  # From now on `uv run` must not "repair" torch back to the CUDA 13 wheel.
  run sh -c "echo 'export UV_NO_SYNC=1' > '$ENV_FILE'"
  export UV_NO_SYNC=1
else
  run sh -c ": > '$ENV_FILE'"
fi
run uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: compare nvidia-smi with torch.version.cuda'; p = torch.cuda.get_device_properties(0); print(f'torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.device_count()} x {p.name} | bf16 {torch.cuda.is_bf16_supported()}')"

if ! command -v gcc >/dev/null 2>&1; then              # ch09 calls torch.compile; Triton needs a C compiler
  if [ "$(id -u)" = 0 ]; then run apt-get install -y build-essential; else run sudo apt-get install -y build-essential; fi
fi

echo "== data for: $BOTS"
run uv run python gpu/fetch.py $BOTS

cat <<EOF

Ready. Keep long runs alive across SSH disconnects with tmux, and record them like on your PC:
  tmux new -s train
  cd $REPO && source $ENV_FILE
  uv run python gpu/run_local.py --run storybot-<gpu> --job storybot
  uv run python gpu/run_local.py --run webbot-<gpu>x8 --job webbot --nproc 8     # 8 GPUs, same token budget
Copy the report back to your PC, then rebuild the site:
  scp -r -P <port> root@<ip>:$REPO/web/data/runs/<run> web/data/runs/
  uv run python gpu/register_run.py <run>                                   (adds it to index.json)
Stop or terminate the instance when you are done: an idle GPU still bills by the minute.
EOF
