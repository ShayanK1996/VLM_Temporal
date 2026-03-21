#!/bin/bash
#SBATCH --job-name=vlm_e2e_train
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_e2e_%j.out
#SBATCH --error=logs/train_e2e_%j.err

# ============================================================
# Stage 2: End-to-End LoRA + Temporal Module Training
# Requires A100 80GB (full VLM in forward pass)
# ============================================================

set -euo pipefail

if command -v module &>/dev/null; then
  module load conda/latest 2>/dev/null || true
fi
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
elif [ -x "/home/$USER/miniconda3/bin/conda" ]; then
  eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)"
elif [ -x "/home/$USER/anaconda3/bin/conda" ]; then
  eval "$(/home/$USER/anaconda3/bin/conda shell.bash hook)"
else
  echo "ERROR: conda not found."
  exit 1
fi
conda activate VLM_EatingBehavior

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$HOME/VLM_Temporal}}"
if [ ! -d "$REPO_DIR/src" ]; then
  if [ -d "$HOME/VLM_Temporal/src" ]; then
    REPO_DIR="$HOME/VLM_Temporal"
  elif [ -d "$HOME/VLM_TemporalBranch/src" ]; then
    REPO_DIR="$HOME/VLM_TemporalBranch"
  else
    echo "ERROR: REPO_DIR does not look like the repo root (missing src/)."
    exit 2
  fi
fi

# SLURM copies the batch script to /var/spool/... — must not use BASH_SOURCE for lib path
# shellcheck source=scripts/lib_vlm_work_root.sh
source "${REPO_DIR}/scripts/lib_vlm_work_root.sh"

VIDEO_DIR="$HOME/VLM_EatingBehavior/data/segmented_videos"
METADATA_CSV="$HOME/VLM_EatingBehavior/data/segments_metadata.csv"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-${VLM_WORK_ROOT}/checkpoints/temporal_v1/fold_0/best_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${VLM_WORK_ROOT}/checkpoints/e2e_v1}"
FOLD=0

mkdir -p logs "$OUTPUT_DIR"
cd "$REPO_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

python -m src.training.train_e2e \
    --temporal-checkpoint "$TEMPORAL_CKPT" \
    --video-dir "$VIDEO_DIR" \
    --metadata-csv "$METADATA_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --fold "$FOLD" \
    --num-epochs 5 \
    --batch-size 2 \
    --lr 2e-5 \
    --grad-accum 8

echo "Finished: $(date)"
