#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/project/rrg-vislearn/kini1409"
ENV_DIR="${PROJECT_DIR}/envs/deepeyes"

echo "[setup] project: ${PROJECT_DIR}"
echo "[setup] env: ${ENV_DIR}"

cd "${PROJECT_DIR}"
mkdir -p envs logs models/deepeyes

if [ ! -d "${ENV_DIR}" ]; then
  echo "[setup] creating venv"
  python -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

echo "[setup] python: $(which python)"
python -m pip install -U pip

echo "[setup] installing PyTorch CUDA wheel"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[setup] installing DeepEyes/Qwen-VL inference deps"
python -m pip install "transformers==4.51.3" accelerate qwen-vl-utils pillow safetensors requests regex psutil

echo "[setup] smoke import"
python -u -c "import torch; print('torch', torch.__version__, 'cuda_visible', torch.cuda.is_available(), flush=True); import transformers; print('transformers', transformers.__version__, flush=True)"

echo "[setup] done"
