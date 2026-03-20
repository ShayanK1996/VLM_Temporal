# VLM_TemporalBranch

**Spatial-Branch Temporal Attention for Vision-Language Eating Behavior Assessment**

Augments a fine-tuned Qwen2.5-VL with a sensor-inspired temporal reasoning module
that decomposes video frames into learned spatial streams (jaw, hand, food, context)
and processes each stream through dilated-CNN + RoPE self-attention — the same
architecture proven on IMU bite detection (F1=0.731), now operating on visual features.

## Architecture Overview

```
Video Frames (16 frames @ 1fps)
        │
        ▼
┌─────────────────────────┐
│  Qwen2.5-VL Vision      │  (frozen or LoRA)
│  Encoder                 │
│  → per-frame patch tokens│
│    (16, H*W, d_vision)   │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Learned Spatial Decomposition       │
│  4 soft-attention heads → 4 streams  │
│  (jaw, hand, food, context)          │
│  Each: (16, d_branch)                │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Parallel Dilated-CNN Branches       │
│  (adapted from RF_CNN_Attention_v3)  │
│  Coprime dilations [1,2,3], k=7      │
│  → temporal features per stream      │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  RoPE Multi-Head Self-Attention      │
│  Cross-branch temporal fusion        │
│  → behavior temporal representation  │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Integration with LM backbone        │
│  → classification + feedback         │
└─────────────────────────────────────┘
```

## Quick Start

### 1. Extract & cache visual features (run once)

```bash
sbatch scripts/extract_features.sh
```

### 2. Train temporal module on cached features

```bash
sbatch scripts/train_temporal.sh
```

### 3. End-to-end fine-tuning (temporal + LoRA)

```bash
sbatch scripts/train_e2e.sh
```

### Unity HPC: where features & checkpoints go

**Cached features and temporal/e2e checkpoints use one tree only:**  
`/work/pi_walls_uri_edu/$USER/VLM_Temporal/` (not under the repo clone — keeps `$HOME` free).

- `cached_features/` — `.pt` caches + `manifest.json` (same path training reads)
- `checkpoints/temporal_v1/` — stage 1
- `checkpoints/e2e_v1/` — stage 2

If that directory cannot be created, the job exits with an error. For **local machines** without `/work`, set a writable root:

```bash
export VLM_WORK_ROOT=/your/preferred/root
sbatch scripts/extract_features.sh
```

On Unity the repo is often `~/VLM_Temporal`; submit from that directory (or set `REPO_DIR`).

## Repo Structure

```
VLM_TemporalBranch/
├── configs/
│   ├── extract_features.yaml    # Feature extraction config
│   ├── train_temporal.yaml      # Temporal module training
│   └── train_e2e.yaml           # End-to-end training
├── src/
│   ├── models/
│   │   ├── spatial_decomposition.py   # Learned spatial attention heads
│   │   ├── temporal_branches.py       # Dilated-CNN + RoPE (adapted from sensor model)
│   │   ├── vlm_temporal_model.py      # Combined VLM + temporal module
│   │   └── __init__.py
│   ├── data/
│   │   ├── feature_dataset.py         # Dataset for cached visual features
│   │   └── __init__.py
│   ├── training/
│   │   ├── extract_features.py        # Cache per-frame VLM features
│   │   ├── train_temporal.py          # Train temporal module (stage 1)
│   │   ├── train_e2e.py               # End-to-end fine-tuning (stage 2)
│   │   └── __init__.py
│   └── evaluation/
│       ├── evaluate.py
│       └── __init__.py
├── scripts/
│   ├── extract_features.sh            # SLURM: feature extraction
│   ├── train_temporal.sh              # SLURM: temporal module training
│   └── train_e2e.sh                   # SLURM: end-to-end
├── EXPERIMENT_LOG.md
├── results_registry.json
└── README.md
```

## Hardware Requirements

- **Feature extraction**: 1x A100 80GB (loads full Qwen2.5-VL)
- **Temporal module training**: 1x L40S 48GB (operates on cached features, no VLM needed)
- **End-to-end**: 1x A100 80GB

## Relationship to Other Repos

- `VLM_EatingBehavior`: IMWUT paper (LoRA fine-tuning, binary classification)
- `RF_CNN_Attention_v3.py`: Sensor-based bite detection (source architecture for temporal module)
- **This repo**: CVPR paper (temporal attention on visual features, building toward cross-modal distillation)
