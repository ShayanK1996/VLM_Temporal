#!/bin/bash
#SBATCH --job-name=vlm_temporal_train
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_temporal_%j.out
#SBATCH --error=logs/train_temporal_%j.err

# ============================================================
# Stage 1: Train Spatial Decomposition + Temporal Attention
# Operates on CACHED features — no VLM forward pass needed
# Can run on L40S 48GB (much cheaper/more available than A100)
# ============================================================

set -euo pipefail

# --- Conda setup (Unity HPC: module; fallback miniconda) ---
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

# --- Paths (SLURM-safe repo root; same as extract_features.sh) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$HOME/VLM_Temporal}}"
if [ ! -d "$REPO_DIR/src" ]; then
  if [ -d "$HOME/VLM_Temporal/src" ]; then
    REPO_DIR="$HOME/VLM_Temporal"
  elif [ -d "$HOME/VLM_TemporalBranch/src" ]; then
    REPO_DIR="$HOME/VLM_TemporalBranch"
  else
    echo "ERROR: REPO_DIR does not look like the repo root (missing src/). Submit from repo root or set REPO_DIR."
    exit 2
  fi
fi

# Features + temporal checkpoints default next to repo on local disk, or under /work on Unity
if [ -n "${VLM_WORK_ROOT:-}" ]; then
  :
else
  _w="/work/pi_walls_uri_edu/$USER/VLM_Temporal"
  if mkdir -p "$_w" 2>/dev/null; then
    VLM_WORK_ROOT="$_w"
  else
    VLM_WORK_ROOT="$REPO_DIR"
  fi
fi
FEATURE_DIR="${FEATURE_DIR:-${VLM_WORK_ROOT}/cached_features}"
MANIFEST="${MANIFEST:-$FEATURE_DIR/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${VLM_WORK_ROOT}/checkpoints/temporal_v1}"

# --- Hyperparameters ---
NUM_EPOCHS=30
BATCH_SIZE=32
LR=1e-3
D_BRANCH=128
N_BRANCHES=4
TEMPORAL_HIDDEN=64
TEMPORAL_OUT=64
N_HEADS=4
N_ATTN_LAYERS=2
DIVERSITY_WEIGHT=0.1

# --- Run single fold for validation/debug ---
# Default: fold 0 only
FOLD_ARG="--fold 0"

mkdir -p logs "$OUTPUT_DIR"

cd "$REPO_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo "Repo: $REPO_DIR"
echo "Artifact root (VLM_WORK_ROOT): $VLM_WORK_ROOT"
echo "Features: $FEATURE_DIR"
echo ""
echo "Config: epochs=$NUM_EPOCHS, bs=$BATCH_SIZE, lr=$LR"
echo "Architecture: d_branch=$D_BRANCH, n_branches=$N_BRANCHES, heads=$N_HEADS, layers=$N_ATTN_LAYERS"
echo ""

python -m src.training.train_temporal \
    --manifest "$MANIFEST" \
    --feature-dir "$FEATURE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --d-branch "$D_BRANCH" \
    --n-branches "$N_BRANCHES" \
    --temporal-hidden "$TEMPORAL_HIDDEN" \
    --temporal-out "$TEMPORAL_OUT" \
    --n-heads "$N_HEADS" \
    --n-attn-layers "$N_ATTN_LAYERS" \
    --diversity-weight "$DIVERSITY_WEIGHT" \
    $FOLD_ARG

echo ""
echo "Finished: $(date)"
echo "Results saved to: $OUTPUT_DIR"

# Print summary if exists
if [ -f "$OUTPUT_DIR/cv_summary.json" ]; then
    echo ""
    echo "Cross-validation summary:"
    cat "$OUTPUT_DIR/cv_summary.json"
fi
