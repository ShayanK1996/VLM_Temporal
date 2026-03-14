# Experiment Log — VLM_TemporalBranch

## Project Goal
Augment fine-tuned Qwen2.5-VL with spatial-branch temporal attention for
eating behavior assessment. Target: CVPR MetaFood workshop (near-term),
CVPR 2027 main track (with distillation + Study 1b data).

## Baseline (from IMWUT paper / VLM_EatingBehavior repo)
- LoRA-only fine-tuning: **63.2% ± 6.7%** binary accuracy (5-fold CV)
- Zero-shot: 24.1%
- Majority class: 54.3%
- Per-food: chips 72.6%, carrots 63.8%, rice+beans 53.6%, churros 52.0%

## Experiment Index

| Run ID | Date | Description | Accuracy | Notes |
|--------|------|-------------|----------|-------|
| — | — | (no experiments yet) | — | — |

## Run Log

### EXP-001: [PENDING] Feature extraction
- **Status**: Not started
- **Goal**: Cache per-frame patch tokens from Qwen2.5-VL for all 4,223 segments
- **Script**: `scripts/extract_features.sh`
- **Expected output**: `cached_features/` with ~4,223 .pt files + manifest.json
- **Notes**: Run on A100 node. Should take ~2-4 hours.

### EXP-002: [PENDING] Temporal module baseline (all defaults)
- **Status**: Not started  
- **Goal**: First training run with default hyperparameters
- **Script**: `scripts/train_temporal.sh`
- **Config**: d_branch=128, n_branches=4, temporal_hidden=64, kernel=3, heads=4, layers=2
- **Expected**: If > 63.2%, temporal module adds value. If < 63.2%, need tuning.
- **Notes**: Run on L40S. Should take ~1-2 hours for all 5 folds.
