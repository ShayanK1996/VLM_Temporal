#!/bin/bash
#SBATCH --job-name=vlm_extract_features
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/extract_features_%j.out
#SBATCH --error=logs/extract_features_%j.err

# ============================================================
# Stage 0: Extract & cache VLM visual features (run ONCE)
# Requires A100 80GB to load full Qwen2.5-VL
# Output: cached .pt files with per-frame patch tokens
# ============================================================

set -euo pipefail

# --- Conda setup (Unity HPC uses module; fallback for local miniconda) ---
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
  echo "ERROR: conda not found. Run 'module load conda/latest' or install conda."
  exit 1
fi
conda activate VLM_EatingBehavior

# --- Paths ---
# IMPORTANT: Under SLURM, the script runs from a temp spool directory
# (e.g. /var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT your repo.
# Use the submit directory (repo root) unless REPO_DIR is explicitly set.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$HOME/VLM_Temporal}}"
if [ ! -d "$REPO_DIR/src" ]; then
  if [ -d "$HOME/VLM_Temporal/src" ]; then
    REPO_DIR="$HOME/VLM_Temporal"
  elif [ -d "$HOME/VLM_TemporalBranch/src" ]; then
    REPO_DIR="$HOME/VLM_TemporalBranch"
  else
    echo "ERROR: REPO_DIR does not look like the repo root (missing src/)."
    echo "Submit from the repo root, e.g.:"
    echo "  cd ~/VLM_Temporal && sbatch scripts/extract_features.sh"
    echo "Or set REPO_DIR explicitly in the sbatch command:"
    echo "  REPO_DIR=/path/to/repo sbatch scripts/extract_features.sh"
    exit 2
  fi
fi
# Reads qwen_dataset.jsonl directly — no intermediate CSV needed.
# Video paths inside the JSONL are absolute Unity paths:
#   /home/skhodabakhsh_uri_edu/VLM_EatingBehavior/data/processed/training_clips/*.mp4
DATASET_JSONL="$HOME/VLM_EatingBehavior/qwen_dataset.jsonl"

# Cached .pt + manifest live ONLY under /work/.../VLM_Temporal (see lib_vlm_work_root.sh)
# Local dev: export VLM_WORK_ROOT=/your/writable/path
# SLURM copies this script to /var/spool/... — use REPO_DIR, not BASH_SOURCE
# shellcheck source=scripts/lib_vlm_work_root.sh
source "${REPO_DIR}/scripts/lib_vlm_work_root.sh"
OUTPUT_DIR="${OUTPUT_DIR:-${VLM_WORK_ROOT}/cached_features}"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
RUN_ID="20260311_171539_A53569"
LORA_DIR="${LORA_DIR:-$HOME/VLM_EatingBehavior/checkpoints/fold_0}"

if [ ! -d "$LORA_DIR" ]; then
  # Common layout: checkpoints stored under a run id folder (you mentioned this exists on Unity)
  if [ -d "$HOME/VLM_EatingBehavior/checkpoints/${RUN_ID}/fold_0" ]; then
    LORA_DIR="$HOME/VLM_EatingBehavior/checkpoints/${RUN_ID}/fold_0"
  elif [ -d "$HOME/VLM_EatingBehavior/checkpoints/${RUN_ID}_main/fold_0" ]; then
    LORA_DIR="$HOME/VLM_EatingBehavior/checkpoints/${RUN_ID}_main/fold_0"
  # Fallback: repo-local uploaded checkpoints (useful for local testing / staging)
  elif [ -d "$REPO_DIR/${RUN_ID}_main/fold_0" ]; then
    LORA_DIR="$REPO_DIR/${RUN_ID}_main/fold_0"
  elif [ -d "$REPO_DIR/${RUN_ID}/fold_0" ]; then
    LORA_DIR="$REPO_DIR/${RUN_ID}/fold_0"
  fi
fi

if [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: Could not find LoRA directory. Set LORA_DIR env var to the fold folder containing adapter_model.safetensors."
  echo "Tried:"
  echo "  - \$HOME/VLM_EatingBehavior/checkpoints/fold_0"
  echo "  - \$HOME/VLM_EatingBehavior/checkpoints/${RUN_ID}/fold_0"
  echo "  - \$HOME/VLM_EatingBehavior/checkpoints/${RUN_ID}_main/fold_0"
  echo "  - \$REPO_DIR (${REPO_DIR})/${RUN_ID}_main/fold_0"
  echo "  - \$REPO_DIR/${RUN_ID}/fold_0"
  exit 2
fi

cd "$REPO_DIR"
mkdir -p logs "$OUTPUT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo "Repo dir: $REPO_DIR"
echo "Artifact root (VLM_WORK_ROOT): $VLM_WORK_ROOT"
echo "Feature output: $OUTPUT_DIR"
echo "LoRA dir: $LORA_DIR"
echo ""

python -m src.training.extract_features \
    --dataset-jsonl "$DATASET_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --lora-dir "$LORA_DIR" \
    --num-frames 16 \
    --segment-types bite

echo ""
echo "Finished: $(date)"
echo "Features cached to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR" | tail -5
echo "Total files: $(ls "$OUTPUT_DIR"/*.pt 2>/dev/null | wc -l)"
