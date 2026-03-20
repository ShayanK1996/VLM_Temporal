# VLM_TemporalBranch — Full Project Context for AI Assistant

## What This Project Is

This is the second paper in a research arc on eating behavior assessment using Vision-Language Models. The first paper (targeting ACM IMWUT) fine-tuned Qwen2.5-VL-3B with LoRA to classify eating behavior from profile-view meal video. This second paper (targeting CVPR MetaFood workshop, then CVPR 2027 main track) adds a **temporal reasoning module** on top of the fine-tuned VLM that captures the sequential rhythm of eating (bite → chew → pause → bite) — something standard VLMs miss because they treat video as a bag of frames.

The temporal module architecture is adapted from a proven IMU-based bite detection model (`RF_CNN_Attention_v3.py`) that achieved F1=0.731 on sensor data. We swap sensor channels for visual feature streams extracted from the VLM's vision encoder.

## The IMWUT Paper Baselines (what we need to beat)

- Zero-shot Qwen2.5-VL-3B: **24.1%** accuracy (4-class)
- Majority class: **54.3%** (binary)
- LoRA fine-tuning only: **63.2% ± 6.7%** (binary, 5-fold participant-level CV)
- Per-food-type: chips 72.6%, carrots 63.8%, rice+beans 53.6%, churros 52.0%

## Repository Relationships

```
~/VLM_EatingBehavior/          ← IMWUT paper repo (ALREADY EXISTS)
├── qwen_dataset.jsonl         ← 4,251 training records (bite/g3/g5/phase segments)
├── data/processed/
│   └── training_clips/*.mp4   ← pre-cut video clips (already segmented)
└── checkpoints/
    └── fold_0/                ← LoRA checkpoint from IMWUT fine-tuning
        ├── adapter_config.json
        └── adapter_model.safetensors

~/VLM_Temporal/ or ~/VLM_TemporalBranch/   ← THIS REPO (CVPR paper; Unity often ~/VLM_Temporal)
├── src/models/                ← new temporal architecture
├── src/training/              ← training pipeline
├── src/evaluation/            ← evaluation with comparison to IMWUT baselines
└── scripts/                   ← SLURM job scripts for Unity HPC

/work/pi_walls_uri_edu/$USER/VLM_Temporal/   ← default large-artifact root on Unity (env: VLM_WORK_ROOT)
├── cached_features/           ← Stage 0: .pt + manifest.json (falls back to REPO_DIR if /work unavailable)
└── checkpoints/               ← temporal_v1, e2e_v1
```

## Data Format

### qwen_dataset.jsonl (source of truth)

Each line is a JSON object with this structure:
```json
{
  "id": "111001_128.31",
  "messages": [
    {"role": "system", "content": "You are a clinical eating behavior analyst..."},
    {"role": "user", "content": [{"type": "video", "video": "/absolute/path/to/clip.mp4", "fps": 1.0}, {"type": "text", "text": "..."}]},
    {"role": "assistant", "content": "CATEGORY: needs_improvement\n\nFeedback text..."}
  ],
  "metadata": {
    "participant_id": "111001",
    "category": "needs_improvement",          // binary label (needs_improvement or good)
    "category_original": "needs_improvement",  // 4-class (eating_too_fast, needs_improvement, adequate, good)
    "metrics": {
      "mean_cbr": 11.0,
      "bite_rate_per_min": 5.2,
      "mean_pause_sec": 0.21,
      "food_types": [1]                       // 1=chips_and_salsa, 2=carrots, 3=rice_and_beans, 4=churros
    },
    "clip_path": "/home/skhodabakhsh_uri_edu/VLM_EatingBehavior/data/processed/training_clips/111001_bite_000.mp4"
  }
}
```

### Dataset Statistics

- **Total records**: 4,251
- **Participants**: 26
- **Segment types**: bite (1,460), g3 (1,382), g5 (1,332), phase (77)
- **Binary labels**: needs_improvement 2,292 (53.9%), good 1,959 (46.1%)
- **Food types**: chips_and_salsa 1,466, rice_and_beans 1,287, carrots 1,086, churros 412
- **Segment type is inferred from clip filename**: `_bite_`, `_g3_`, `_g5_`, `_phase_`

### Video Clips

- Pre-segmented .mp4 files at absolute paths on Unity HPC
- Per-bite clips: ~15-20 seconds, 8 frames sampled at 1fps
- g3 clips: 3 consecutive bites (longer, more temporal structure)
- g5 clips: 5 consecutive bites (ideal for temporal module)
- Profile (side) view of participant eating

## Architecture Overview

### Variant B: Spatial-Branch Temporal Attention

```
Video Clip (.mp4)
    │
    ▼
Qwen2.5-VL Vision Encoder (with merged LoRA weights)
    │  outputs per-frame patch tokens
    │  shape: (num_frames, num_patches, d_vision=1536)
    ▼
SpatialDecomposition (src/models/spatial_decomposition.py)
    │  4 learnable query vectors soft-attend over patches
    │  each learns a different spatial region (jaw, hand, food, context)
    │  includes diversity loss to push branches apart
    │  shape: (num_frames, 4, d_branch=128)
    ▼
VisualTemporalAttention (src/models/temporal_branches.py)
    │  4 parallel dilated-CNN branches (one per spatial stream)
    │  coprime dilations [1, 2, 3], kernel_size=3
    │  RoPE multi-head self-attention for cross-branch temporal fusion
    │  adapted from RF_CNN_Attention_v3.py (sensor bite detection model)
    │  shape: (d_model=256) after pooling
    ▼
Classifier → binary label (needs_improvement vs good)
```

### Key Design Decisions

1. **Kernel size = 3** (not 7 like sensor model) because we have 16 frames, not 350 IMU timesteps
2. **Single pool layer** (not 3x) to preserve temporal resolution on short sequences
3. **Learned spatial decomposition** instead of hard-coded ROIs (jaw detector etc.) — avoids fragile dependencies
4. **Diversity regularization** on spatial branches to encourage specialization
5. **RoPE attention** encodes relative temporal position — same proven component from the sensor model

## Three-Stage Training Pipeline

### Stage 0: Extract & Cache Features (run ONCE)

```bash
sbatch scripts/extract_features.sh
```

- Loads base Qwen2.5-VL + merges LoRA checkpoint from IMWUT fine-tuning
- Runs each video clip through the fine-tuned vision encoder
- Saves per-frame patch tokens as `.pt` files in `cached_features/` (default location on Unity: `/work/pi_walls_uri_edu/$USER/VLM_Temporal/cached_features/`; override with env `VLM_WORK_ROOT`, or falls back to repo root if `/work` is unavailable)
- Creates `manifest.json` in that same directory with all metadata + fold assignments
- **Requires**: A100 80GB, ~3-4 hours for 4,251 segments
- **Important**: Uses fine-tuned (not base) vision encoder so features already understand eating behavior

### Stage 1: Train Temporal Module (fast iteration)

```bash
sbatch scripts/train_temporal.sh
```

- Loads cached .pt files (NO VLM needed, NO GPU-hungry model loading)
- Trains SpatialDecomposition + VisualTemporalAttention
- 5-fold participant-level cross-validation
- **Requires**: L40S 48GB (much cheaper/more available), ~1-2 hours for all folds
- **This is where most experimentation happens** — hyperparameter sweeps, ablations

### Stage 2: End-to-End Fine-Tuning (optional, after Stage 1 succeeds)

```bash
sbatch scripts/train_e2e.sh
```

- Joint optimization: LoRA on VLM + temporal module
- Uses Stage 1 temporal weights as initialization
- Generates both classification AND natural-language feedback
- **Requires**: A100 80GB, ~6-12 hours

## Fold Assignments (deterministic, seed=42)

| Fold | Participants | 
|------|-------------|
| 0 | 111010, 211003, 111001, 211013, 111013, 111011 |
| 1 | 111015, 111003, 211011, 111007, 111004 |
| 2 | 111014, 211002, 111005, 111006, 211009 |
| 3 | 211004, 211010, 211005, 211015, 211008 |
| 4 | 111009, 111012, 211001, 211006, 111008 |

## Key Files

### Models
- `src/models/spatial_decomposition.py` — SpatialBranchAttention, SpatialDecomposition (with diversity loss)
- `src/models/temporal_branches.py` — TemporalCNNBranch, VisualTemporalAttention, RoPE implementation
- `src/models/vlm_temporal_model.py` — TemporalBehaviorModel (full pipeline), TemporalModelConfig, TemporalTokenInjector

### Training
- `src/training/extract_features.py` — reads qwen_dataset.jsonl, loads VLM + LoRA, caches vision features
- `src/training/train_temporal.py` — Stage 1 training loop, 5-fold CV, per-food-type eval
- `src/training/train_e2e.py` — Stage 2 end-to-end (LoRA + temporal joint training)

### Data
- `src/data/feature_dataset.py` — CachedFeatureDataset, get_fold_split (reads manifest.json, uses pre-assigned folds)

### Evaluation
- `src/evaluation/evaluate.py` — cross-validation summary, comparison table vs IMWUT baselines, per-food-type analysis

### Scripts
- `scripts/extract_features.sh` — SLURM job for Stage 0 (A100)
- `scripts/train_temporal.sh` — SLURM job for Stage 1 (L40S)
- `scripts/train_e2e.sh` — SLURM job for Stage 2 (A100)
- `scripts/generate_metadata.py` — optional utility to export qwen_dataset.jsonl as CSV

### Tests
- `tests/sanity_check.py` — verifies all module shapes, loss computation, backprop with random data (CPU, no GPU needed)

## HPC Environment (Unity)

- **Cluster**: Unity HPC at URI
- **Partition**: uri-gpu
- **GPUs**: L40S 48GB, A100 80GB
- **Conda env**: VLM_EatingBehavior (has peft, trl, bitsandbytes, transformers, qwen-vl-utils)
- **Job scheduler**: SLURM
- **Workflow**: edit locally → commit → push → `git pull --rebase` on Unity → `sbatch scripts/...`

## What Needs to Happen Next (in order)

1. **Upload LoRA checkpoints to Unity**: `scp -r checkpoints/fold_0/ unity:~/VLM_EatingBehavior/checkpoints/fold_0/`
2. **Verify clip paths resolve**: `head -1 ~/VLM_EatingBehavior/qwen_dataset.jsonl | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['metadata']['clip_path'])" | xargs ls -lh`
3. **Run sanity check**: `cd ~/VLM_TemporalBranch && python tests/sanity_check.py`
4. **Run Stage 0** (extract features): `sbatch scripts/extract_features.sh`
5. **Run Stage 1** (train temporal): `sbatch scripts/train_temporal.sh` — start with `--fold 0` for debugging
6. **Check if accuracy > 63.2%** — if yes, temporal module adds value and CVPR story holds

## Important Considerations

### g3/g5 segments are better than per-bite for temporal module
Per-bite clips cover one bite cycle — no multi-bite rhythm for temporal attention to learn. g3/g5 clips have 3-5 consecutive bites, which is the sequential structure the dilated-CNN + RoPE was designed to capture. Use `--segment-types g3 g5` flag in extract_features.sh to filter.

### LoRA checkpoint for feature extraction
The extraction script merges LoRA weights into the base model before extracting features. This means the vision encoder already "sees" eating behavior (adapted by IMWUT fine-tuning). Do NOT skip this — base model features would be much weaker.

### The temporal module's pooling is different from the sensor model
Sensor model pools 3x (T → T//8) because input is ~350 timesteps at 100Hz. Visual temporal module pools only 1x (T → T//2) because input is only 16 frames. This is handled in TemporalCNNBranch.

### Diversity loss weight
`diversity_weight=0.1` in SpatialDecomposition encourages branches to attend to different patches. If all branches collapse to the same region, increase this. If accuracy is low and branches are too different, decrease it.

## Future Extensions (for CVPR 2027 main track)

- **Cross-modal distillation**: Use trained sensor model (RF_CNN_Attention_v3, F1=0.731) as teacher, distill temporal knowledge into the visual temporal module. At inference, throw away sensors.
- **Study 1b data**: 60 participants, more food variety, richer annotations
- **Multiple VLM backbones**: Compare against InternVL, LLaVA-Video
- **Frame sampling ablation**: 4, 8, 16, 32 frames with different strategies
