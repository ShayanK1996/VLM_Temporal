#!/bin/bash
#SBATCH --job-name=vlm_temporal_train
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
# If jobs are rejected or killed for time: try --time=12:00:00 (uri-gpu QOS varies by account)
#SBATCH --time=1-00:00:00
#SBATCH --output=logs_stage_1/train_temporal_%j.out
#SBATCH --error=logs_stage_1/train_temporal_%j.err

# ============================================================
# Stage 1: Train Spatial Decomposition + Temporal Attention
# Operates on CACHED features — no VLM forward pass needed
# Can run on L40S 48GB (much cheaper/more available than A100)
#
# IMPORTANT: Slurm opens logs_stage_1/*.out BEFORE this script runs.
# Create the dir first:  mkdir -p logs_stage_1
# Or use:                ./scripts/submit_train_temporal.sh
# If the job "vanishes": sacct -j JOBID -o JobID,State,ExitCode,Reason,Elapsed,Timelimit
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
# conda activate can return non-zero under strict set -e in some batch setups
if ! conda activate VLM_EatingBehavior; then
  echo "ERROR: conda activate VLM_EatingBehavior failed."
  exit 1
fi

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

# SLURM copies the batch script to /var/spool/... — must not use BASH_SOURCE for lib path
# shellcheck source=scripts/lib_vlm_work_root.sh
source "${REPO_DIR}/scripts/lib_vlm_work_root.sh"
FEATURE_DIR="${FEATURE_DIR:-${VLM_WORK_ROOT}/cached_features}"
MANIFEST="${MANIFEST:-$FEATURE_DIR/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${VLM_WORK_ROOT}/checkpoints/temporal_v1}"

# --- Hyperparameters ---
NUM_EPOCHS=30
BATCH_SIZE="${BATCH_SIZE:16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
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

mkdir -p logs_stage_1 "$OUTPUT_DIR"

cd "$REPO_DIR"
# Ensure `python -m src....` resolves (some Slurm/env setups omit cwd from sys.path)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Slurm diagnostics ==="
echo "PYTHONPATH (repo root): ${REPO_DIR}"
echo "src/data/feature_dataset.py: $([ -f "${REPO_DIR}/src/data/feature_dataset.py" ] && echo OK || echo MISSING)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}  SUBMIT_DIR=${SLURM_SUBMIT_DIR:-n/a}"
echo "Timelimit env: ${SLURM_TIMELIMIT:-n/a}  Partition: ${SLURM_JOB_PARTITION:-n/a}"
echo "lib_vlm_work_root.sh: $([ -f "${REPO_DIR}/scripts/lib_vlm_work_root.sh" ] && echo OK || echo MISSING)"
echo "manifest: ${MANIFEST} -> $([ -f "${MANIFEST}" ] && echo OK || echo MISSING)"
_pt_n=0
[ -d "${FEATURE_DIR}" ] && _pt_n=$(find "${FEATURE_DIR}" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
echo "feature dir: ${FEATURE_DIR} (${_pt_n} .pt files)"
echo "========================="

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
# Do not fail the whole job if nvidia-smi hiccups (set -e treats subshell failure as fatal)
_gpu_info="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
echo "GPU: ${_gpu_info}"
echo "Start: $(date)"
echo "Repo: $REPO_DIR"
echo "Artifact root (VLM_WORK_ROOT): $VLM_WORK_ROOT"
echo "Features: $FEATURE_DIR"
echo ""
echo "Config: epochs=$NUM_EPOCHS, bs=$BATCH_SIZE, workers=$NUM_WORKERS, lr=$LR"
echo "Architecture: d_branch=$D_BRANCH, n_branches=$N_BRANCHES, heads=$N_HEADS, layers=$N_ATTN_LAYERS"
echo ""

python -u -m src.training.train_temporal \
    --manifest "$MANIFEST" \
    --feature-dir "$FEATURE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
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
