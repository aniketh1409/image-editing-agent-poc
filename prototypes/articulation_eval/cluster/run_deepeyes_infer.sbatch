#!/usr/bin/env bash
#SBATCH --account=rrg-vislearn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=deepeyes_infer
#SBATCH --output=/project/rrg-vislearn/kini1409/logs/deepeyes_infer_%j.out
#SBATCH --error=/project/rrg-vislearn/kini1409/logs/deepeyes_infer_%j.err

set -euo pipefail

PROJECT_DIR="/project/rrg-vislearn/kini1409"
SCRIPT_DIR="${PROJECT_DIR}/repos/image-editing-agent-poc/prototypes/articulation_eval/cluster"
ENV_DIR="${PROJECT_DIR}/envs/deepeyes"

cd "${PROJECT_DIR}"
mkdir -p logs

echo "[job] host: $(hostname)"
echo "[job] date: $(date)"
echo "[job] project: ${PROJECT_DIR}"
echo "[job] script dir: ${SCRIPT_DIR}"

nvidia-smi

source "${ENV_DIR}/bin/activate"
echo "[job] python: $(which python)"

python -u "${SCRIPT_DIR}/test_deepeyes_imports.py"
python -u "${SCRIPT_DIR}/test_deepeyes_infer.py"
