#!/usr/bin/env bash
# One-shot setup on a RunPod GPU pod (or any Ubuntu 22.04/24.04 box with an NVIDIA GPU), as root,
# from the repo root:   bash tools/pod_setup.sh
#
# Idempotent. Everything slow to rebuild (venv, model cache) lives on the persistent volume
# ($WORKSPACE, default /workspace); apt packages live on the container disk, so after a pod
# restart just run this again (~3 min, the venv and models are kept).
#   FLASH_ATTN=1 bash tools/pod_setup.sh   also builds flash-attn (optional; sdpa works fine)
set -euo pipefail
WS=${WORKSPACE:-/workspace}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export DEBIAN_FRONTEND=noninteractive

echo "== apt: LaTeX, ffmpeg, cairo/pango (Manim), fonts"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    ffmpeg fontconfig fonts-liberation2 libcairo2-dev libpango1.0-dev pkg-config build-essential \
    git curl ca-certificates sox \
    texlive-xetex texlive-latex-extra texlive-lang-cyrillic texlive-fonts-recommended >/dev/null
apt-get install -y -qq --no-install-recommends dvisvgm >/dev/null 2>&1 || true  # own package on 24.04
command -v dvisvgm >/dev/null || { echo "dvisvgm missing"; exit 1; }

echo "== python 3.12 venv in $WS/venv (uv)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH=$HOME/.local/bin:$PATH
[ -x "$WS/venv/bin/python" ] || uv venv -q -p 3.12 "$WS/venv"
# shellcheck disable=SC1091
. "$WS/venv/bin/activate"
uv pip install -q fonttools
# torch + torchaudio from ONE PyTorch index (qwen-tts would otherwise pull a torchaudio built for
# another CUDA). cu128 runs on drivers >= 570; override with TORCH_CUDA=cu130 etc.
TORCH_CUDA=${TORCH_CUDA:-cu128}
python -c "import torch, torchaudio" 2>/dev/null || \
    uv pip install -q torch torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
uv pip install -q -e "$REPO[align,dev,tts]"
python -c "import torchaudio" || { echo "torchaudio broken: torch/torchaudio CUDA builds differ"; exit 1; }
if [ "${FLASH_ATTN:-0}" = 1 ]; then
    uv pip install -q packaging ninja wheel
    MAX_JOBS=${MAX_JOBS:-8} uv pip install -q flash-attn --no-build-isolation || echo "flash-attn build failed; sdpa will be used"
fi

echo "== fonts"
bash "$REPO/tools/install_fonts.sh" >/dev/null

echo "== env file: $WS/factory.env"
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
cat > "$WS/factory.env" <<EOF
. $WS/venv/bin/activate
export FACTORY_CACHE=$WS/cache HF_HOME=$WS/cache/hf TORCH_HOME=$WS/cache/torch
# faster-whisper (CTranslate2) loads cuBLAS/cuDNN from the pip wheels that torch installed
export LD_LIBRARY_PATH=$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}
cd $REPO
EOF
. "$WS/factory.env"

echo "== check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'EOF'
import torch, ctranslate2
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu ok:", torch.cuda.is_available())
print("ctranslate2 cuda devices:", ctranslate2.get_cuda_device_count())
assert torch.cuda.is_available(), "torch can't see the GPU (pick a pod with CUDA >= 12.8 drivers)"
EOF
echo
echo "Done. In every new shell:  source $WS/factory.env"
echo "Next: bash tools/pod_smoke.sh"
