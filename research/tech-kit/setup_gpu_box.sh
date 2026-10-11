#!/usr/bin/env bash
# Reproducible DLFS6 environment on a fresh Linux GPU box (RunPod pod or Lambda instance).
#   RunPod : WORK=/workspace   (volume disk or network volume; survives pod stop)
#   Lambda : WORK=/lambda/nfs/<filesystem-name>   (persistent filesystem attached at launch)
# Usage    : WORK=/workspace BOTS="storybot" bash setup_gpu_box.sh
set -euo pipefail

WORK="${WORK:-/workspace}"
BOTS="${BOTS:-storybot}"                      # any of: codebot storybot webbot
REPO_URL="https://github.com/oreilly-japan/deep-learning-from-scratch-6.git"
HF="https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main"

mkdir -p "$WORK"
export UV_CACHE_DIR="$WORK/.cache/uv"          # keep wheel downloads on the persistent disk
export UV_PYTHON_INSTALL_DIR="$WORK/.cache/python"
export UV_LINK_MODE=copy                       # cache and venv may sit on different filesystems

echo "== GPU / driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

cd "$WORK"
[ -d dlfs6/.git ] || git clone --depth 1 "$REPO_URL" dlfs6
cd dlfs6

# The book's uv.lock resolves torch 2.9.1 from PyPI; on Linux that wheel is the CUDA 12.8 build
# (CUDA 12.x minor-version compatibility needs driver >= 525; >= 570 recommended).
uv sync --frozen --python 3.12
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: compare `nvidia-smi` driver with torch.version.cuda"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {p.name} sm_{p.major}{p.minor} | "
      f"{p.total_memory / 2**30:.0f} GiB | bf16 {torch.cuda.is_bf16_supported()}")
PY

command -v gcc >/dev/null 2>&1 || echo "WARNING: no gcc; torch.compile (ch09) needs it: apt-get install -y build-essential"

fetch() {  # fetch <bot> <file>: skip files that already exist (re-runs are cheap)
  local dst="$1/$2"
  [ -s "$dst" ] && { echo "have $dst"; return; }
  curl -fL --retry 5 --retry-delay 3 -o "$dst.part" "$HF/$1/$2" && mv "$dst.part" "$dst"
}
for bot in $BOTS; do
  case "$bot" in
    codebot)  fetch codebot model_pretrain.pt ;;   # tiny_codes.* and merge_rules.pkl ship with the repo
    storybot) for f in tiny_stories_train.bin tiny_stories_valid.bin merge_rules.pkl; do fetch storybot "$f"; done ;;
    webbot)   for f in owt_train.bin owt_valid.bin merge_rules.pkl; do fetch webbot "$f"; done ;;
    *) echo "unknown bot: $bot" >&2; exit 1 ;;
  esac
done

cat <<EOF
Ready. Long runs should survive SSH disconnects:
  cd $WORK/dlfs6 && tmux new -s train
  uv run python ch06/05_pretrain.py                                    # single GPU
  WANDB_MODE=disabled uv run torchrun --standalone --nproc-per-node=8 ch09/01_pretrain.py
EOF
