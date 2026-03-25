#!/bin/bash
#SBATCH --job-name=vlm_e2e_train
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs_stage_2/train_e2e_%j.out
#SBATCH --error=logs_stage_2/train_e2e_%j.err

# ============================================================
# Stage 2: End-to-End LoRA (visual encoder) + Temporal Module
# Requires A100 80GB — VLM forward pass with LoRA gradients.
#
# The temporal head weights are loaded from Stage 1 checkpoints.
# LoRA adapters are randomly initialized on the VLM visual encoder.
# Gradient checkpointing is enabled to fit in 80GB VRAM.
#
# IMPORTANT: Create the log dir first:  mkdir -p logs_stage_2
# ============================================================

set -euo pipefail

# --- Conda setup ---
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

# shellcheck source=scripts/lib_vlm_work_root.sh
source "${REPO_DIR}/scripts/lib_vlm_work_root.sh"

# --- Paths ---
MANIFEST="${VLM_WORK_ROOT}/cached_features/manifest.json"
# Use {fold} placeholder — train_e2e.py substitutes the fold number
TEMPORAL_CKPT="${VLM_WORK_ROOT}/checkpoints/temporal_v1/fold_{fold}/best_model.pt"
OUTPUT_DIR="${OUTPUT_DIR:-${VLM_WORK_ROOT}/checkpoints/e2e_v1}"

# --- Hyperparameters ---
NUM_EPOCHS=5
LORA_LR=2e-5
TEMPORAL_LR=2e-4
GRAD_ACCUM_STEPS=16          # effective bs = 1 * 16 = 16
EARLY_STOP_PATIENCE=3
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.1
IMBALANCE_RATIO_THRESHOLD=1.25
NUM_WORKERS=0

# Architecture (MUST match Stage 1)
D_VISION=1280
D_BRANCH=32
N_BRANCHES=4
TEMPORAL_HIDDEN=16
TEMPORAL_OUT=16
N_HEADS=1
N_ATTN_LAYERS=1
TEMPORAL_KERNEL=7
NUM_FRAMES=16

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Resolution cap — limits ViT attention N² (default Qwen2.5-VL is way too high).
# 256*28*28 = 200704 → ~320 merged patches/frame (matches Stage 1 features).
MAX_PIXELS=200704

# --- Fold selection ---
# "--fold 0" for single-fold debug; remove for all 5 folds
FOLD_ARG="--fold 0"

mkdir -p logs_stage_2 "$OUTPUT_DIR"
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Slurm diagnostics ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}  Partition: ${SLURM_JOB_PARTITION:-n/a}"
echo "manifest: ${MANIFEST} -> $([ -f "${MANIFEST}" ] && echo OK || echo MISSING)"
echo "temporal ckpt pattern: ${TEMPORAL_CKPT}"
echo "========================="

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
_gpu_info="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
echo "GPU: ${_gpu_info}"
echo "Start: $(date)"
echo ""
echo "Config: epochs=$NUM_EPOCHS, accum=$GRAD_ACCUM_STEPS (eff=$GRAD_ACCUM_STEPS), workers=$NUM_WORKERS"
echo "LR: lora=$LORA_LR, temporal=$TEMPORAL_LR | LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "Regularization: focal_gamma=$FOCAL_GAMMA, label_smooth=$LABEL_SMOOTHING, early_stop=$EARLY_STOP_PATIENCE"
echo "Architecture: d_vision=$D_VISION, d_branch=$D_BRANCH, n_branches=$N_BRANCHES, heads=$N_HEADS, layers=$N_ATTN_LAYERS, kernel=$TEMPORAL_KERNEL"
echo "Resolution: max_pixels=$MAX_PIXELS | CUDA alloc: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

python -u -m src.training.train_e2e \
    --manifest "$MANIFEST" \
    --temporal-checkpoint "$TEMPORAL_CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --lora-lr "$LORA_LR" \
    --temporal-lr "$TEMPORAL_LR" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --num-workers "$NUM_WORKERS" \
    --focal-gamma "$FOCAL_GAMMA" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --imbalance-ratio-threshold "$IMBALANCE_RATIO_THRESHOLD" \
    --d-vision "$D_VISION" \
    --d-branch "$D_BRANCH" \
    --n-branches "$N_BRANCHES" \
    --temporal-hidden "$TEMPORAL_HIDDEN" \
    --temporal-out "$TEMPORAL_OUT" \
    --n-heads "$N_HEADS" \
    --n-attn-layers "$N_ATTN_LAYERS" \
    --temporal-kernel "$TEMPORAL_KERNEL" \
    --num-frames "$NUM_FRAMES" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --max-pixels "$MAX_PIXELS" \
    $FOLD_ARG

echo ""
echo "Finished: $(date)"
echo "Results saved to: $OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/cv_summary.json" ]; then
    echo ""
    echo "Cross-validation summary:"
    cat "$OUTPUT_DIR/cv_summary.json"
fi
