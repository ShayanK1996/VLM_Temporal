#!/bin/bash
#SBATCH --job-name=vlm_extract_features
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --mem=64G
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

# --- Conda setup ---
eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook 2>/dev/null || /home/$USER/anaconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate VLM_EatingBehavior

# --- Paths ---
# Reads qwen_dataset.jsonl directly — no intermediate CSV needed.
# Video paths inside the JSONL are absolute Unity paths:
#   /home/skhodabakhsh_uri_edu/VLM_EatingBehavior/data/processed/training_clips/*.mp4
REPO_DIR="$HOME/VLM_TemporalBranch"
DATASET_JSONL="$HOME/VLM_EatingBehavior/qwen_dataset.jsonl"
OUTPUT_DIR="$HOME/VLM_TemporalBranch/cached_features"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

mkdir -p logs "$OUTPUT_DIR"

cd "$REPO_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo ""

python -m src.training.extract_features \
    --dataset-jsonl "$DATASET_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --num-frames 16

echo ""
echo "Finished: $(date)"
echo "Features cached to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR" | tail -5
echo "Total files: $(ls "$OUTPUT_DIR"/*.pt 2>/dev/null | wc -l)"
