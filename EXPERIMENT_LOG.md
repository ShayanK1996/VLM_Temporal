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

---

## Bug & Fix Log

### BUG-001: Stale cluster copy (jobs ran old config)
- **Symptom**: Job 53826355 printed `epochs=30, bs=8, d_branch=128, heads=4, layers=2` — values that didn't match the local repo.
- **Cause**: The HPC cluster at `~/VLM_Temporal` had not been synced with the local `VLM_TemporalBranch` repo. Jobs submitted from the cluster always ran the old script.
- **Fix**: Push local changes to GitHub and `git pull` on the cluster before every submission.

### BUG-002: DataLoader workers OOM-killed (RAM)
- **Symptom**: Job 53826378 — `RuntimeError: DataLoader worker (pid 221909) exited unexpectedly` + Slurm `oom_kill` event. Crashed before training started.
- **Cause**: `--num-workers 0` was passed to `main()` → `run_fold(num_workers=0)` → but `run_fold` never forwarded `num_workers` to `get_fold_split()`. The function fell back to its default of `num_workers=4`, spawning 4 worker processes. Each process loaded massive `(16, N_patches, 1280)` feature tensors from NFS in parallel, exhausting the 160 GB RAM allocation.
- **Fix** (`src/training/train_temporal.py`): Added `num_workers=num_workers` to the `get_fold_split()` call inside `run_fold`.
- **Fix** (`src/data/feature_dataset.py`): `pin_memory` now only `True` when `num_workers > 0` and CUDA is available (no benefit otherwise).

### BUG-003: CUDA OOM during backward (GPU VRAM)
- **Symptom**: Job 53826776 — `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.28 GiB`. Crashed during `loss.backward()` on first epoch. GPU had 69.47 GiB already allocated.
- **Cause**: `_compute_diversity_loss` in `SpatialDecomposition` re-ran `k_proj` and full attention for all 4 branches *again* on the flattened `(BT=512, N_patches, 1280)` tensor — i.e., the computation graph was built twice, doubling stored activations. `bs=32` × `T=16` = 512 frame-batches × `N_patches×1280` float32 filled the GPU before the model even started.
- **Fix** (`src/models/spatial_decomposition.py`): `SpatialBranchAttention.forward()` now returns the pre-dropout head-averaged attention distribution alongside the branch feature. `_compute_diversity_loss` uses these cached values — no redundant second pass.
- **Fix** (`src/training/train_temporal.py`): Added AMP (`torch.amp.autocast("cuda")` + `GradScaler`) — activations computed in float16, halving VRAM usage.
- **Fix** (`scripts/train_temporal.sh`): Reduced per-step batch from 32 → 8 (with `GRAD_ACCUM_STEPS=4` to keep effective batch = 32). Also fixed deprecated `torch.cuda.amp.*` API → `torch.amp.*`.

### BUG-004: Model overtrained by epoch 8, wasted compute (overfitting)
- **Symptom**: Job 53827575 — model peaked at `val_acc=0.741` at epoch 8, then degraded to 0.659 by epoch 19. `train_acc` went from 53% to 96% while val stayed flat.
- **Cause**: LR=1e-3 was too aggressive (epoch 1 val_acc dropped from 0.637 to 0.541 due to large initial update). No mechanism to stop at the best epoch. No input regularization.
- **Fix** (`src/training/train_temporal.py`): Added early stopping with configurable patience (`--early-stop-patience`). Added `feat_dropout` (random patch zeroing during training). Added LR warmup (10% of epochs, linear ramp) + cosine decay. Added `--grad-accum-steps` CLI arg.
- **Fix** (`src/models/vlm_temporal_model.py`): Added `label_smoothing` to CE loss (`F.cross_entropy(..., label_smoothing=0.1)`).
- **Fix** (`scripts/train_temporal.sh`): LR 1e-3 → 3e-4, `EARLY_STOP_PATIENCE=7`, `FEAT_DROPOUT=0.1`, `LABEL_SMOOTHING=0.1`.
- **Result**: Job 53849283 stopped at epoch 14 (early stopping triggered). Best epoch 7, `val_acc=0.710`, `macro_f1=0.667`.

### BUG-005: Class imbalance — model predicts majority class (NI) almost exclusively
- **Symptom**: Job 53849283 epoch 0 — `val_acc=0.631, NI_f1=0.774, G_f1=0.000`. The model started by predicting *everything* as "needs improvement". Majority class baseline = 63.1%. Best result was only 71.0% accuracy = **+8% above guessing**. G_f1 oscillated wildly (0 → 0.557 → 0 → 0.557) between epochs because small batches (bs=8) sometimes contained zero "good" samples.
- **Cause (updated after job 53851545 instrumentation)**: Train split for fold 0 is actually near-balanced (`561 vs 516`). The stronger skew appears in the **validation split** (participant-level split with only 6 val participants), so fold-level metrics can swing heavily. Enabling weighted sampling on a near-balanced train split can also add noise due to replacement sampling.
- **Fix** (`src/data/feature_dataset.py`): Weighted sampling is now **conditional** (auto only when train imbalance ratio exceeds threshold; default 1.25). Train/val class counts and the sampler decision are logged each run.
- **Fix** (`src/models/vlm_temporal_model.py`): `forward()` now accepts `class_weight` tensor. Loss computed as `F.cross_entropy(..., weight=class_weight)`.
- **Fix** (`src/training/train_temporal.py`): `run_fold` computes inverse-frequency class weights from training labels **only when imbalance is meaningful** (same threshold). Model selection switched from `val_acc` to **macro-F1** (`(NI_f1 + G_f1) / 2`) — accuracy is misleading when folds are skewed.
- **Fix** (`scripts/train_temporal.sh`): Model simplified further (`d_branch=32`, `temporal_hidden/out=16`, `n_heads=1`), LR lowered (`2e-4`), stronger regularization (`feat_dropout=0.15`), and faster stopping (`patience=5`).

### BUG-006: Prediction collapse — G_f1 ≈ 0 in early epochs (majority-class shortcut)
- **Symptom**: Job 53857029 — Focal loss not yet applied. G_f1 started at 0.030 (epoch 0), briefly rose to 0.286 (epoch 2), then collapsed back to 0.015 (epoch 3). The model repeatedly fell into predicting NI for all samples because the classifier's bias term could minimize CE loss by simply matching the class prior. Job was externally cancelled (SIGTERM) at epoch 3.
- **Cause**: Standard CE loss rewards confident majority-class predictions. With random-init features, the final linear layer's **bias** nudges toward the larger class within the first epoch. Once the NI logit is slightly higher, the positive feedback loop drives G_f1 → 0. Label smoothing and class weighting alone are not sufficient to break this collapse.
- **Fix** (`src/models/vlm_temporal_model.py`): Replaced CE with **focal loss** — `(1 - p_t)^gamma * CE` with `gamma=2.0`. Confident correct predictions (easy NI samples) get near-zero loss weight; mistakes on "good" samples get full weight. Configurable via `focal_gamma` in `TemporalModelConfig`.
- **Fix** (`src/models/temporal_branches.py`): Removed bias from the final classifier layer (`nn.Linear(mlp_hidden, num_classes, bias=False)`) so the model cannot learn the class prior through bias alone.
- **Fix** (`scripts/train_temporal.sh`): Added `FOCAL_GAMMA=2.0` and `--focal-gamma` CLI arg.
- **Result**: Job 53887524 (fold 0 only) — G_f1 rose to **0.656** at epoch 4 (vs 0.286 before), macro_f1 = **0.731** (vs 0.544 before), val_acc = **75.2%**. Model no longer collapses to majority-class prediction.

---

## Experiment Index

| Run ID | Job ID | Date | Config | Val Acc | Macro-F1 | Notes |
|--------|--------|------|--------|---------|----------|-------|
| EXP-001 | 53826355 | 2026-03-23 | Old cluster copy: epochs=30, bs=8, d_branch=128, heads=4 | — | — | Cancelled — old config |
| EXP-002 | 53826378 | 2026-03-23 | epochs=20, bs=32, workers=0 (BUG-002 fix) | — | — | Crashed — DataLoader OOM |
| EXP-003 | 53826776 | 2026-03-23 | epochs=20, bs=32, workers=0 (BUG-002 fix applied) | — | — | Crashed — CUDA OOM during backward |
| EXP-004 | 53827575 | 2026-03-23 | epochs=20, bs=8, accum=4, lr=1e-3, AMP (BUG-003 fixes) | 0.741 (ep8) | ~0.699 | Overfit; ran all 20 epochs |
| EXP-005 | 53849283 | 2026-03-24 | epochs=30, bs=8, accum=4, lr=3e-4, warmup, early_stop=7, feat_drop=0.1, label_smooth=0.1 | 0.710 (ep7) | 0.667 | Early stopped ep14; G_f1 unstable |
| EXP-006 | 53851545 | 2026-03-24 | +WeightedSampler/class-weighting/macro-F1, bs=8 accum=2 | — | — | Instrumented split counts: train near-balanced (561/516) |
| EXP-007 | 53857029 | 2026-03-24 | Smaller model + conditional balancing, bs=8 accum=2, eff=16 | — | — | Externally cancelled (SIGTERM) at epoch 3; no code error |
| EXP-008 | 53887524 | 2026-03-24 | +Focal loss (γ=2), no classifier bias, bs=8 accum=2 | 0.752 (ep4) | 0.731 | Fold 0 only. Early stopped ep9. G_f1=0.656 |
| **EXP-009** | **53940459** | **2026-03-24** | **Same as EXP-008, full 5-fold CV** | **0.760 ± 0.038** | **0.754 ± 0.039** | **Stage 1 complete. Results below.** |

---

## EXP-009 — Full 5-Fold CV Results (Stage 1 Final)

**Job**: 53940459 | **Date**: 2026-03-24/25 | **Runtime**: ~16 hours | **GPU**: A100-SXM4-80GB

### Cross-Validation Summary

| Metric | Mean ± Std |
|--------|-----------|
| **Macro-F1** | **0.754 ± 0.039** |
| **Accuracy** | **0.760 ± 0.038** |

### Per-Fold Breakdown

| Fold | Samples (train/val) | Balanced? | Best Epoch | Val Acc | Macro-F1 | NI_f1 | G_f1 | Stopped |
|------|--------------------:|-----------|:----------:|--------:|---------:|------:|-----:|--------:|
| 0 | 1077 / 355 | No  (1.09) | 4  | 0.749 | 0.731 | 0.806 | 0.656 | ep 9 |
| 1 | 1195 / 237 | Yes (1.28) | 10 | 0.785 | 0.785 | 0.787 | 0.783 | ep 15 |
| 2 | 1085 / 347 | No  (1.12) | 19 | 0.810 | 0.799 | 0.846 | 0.752 | full 20 |
| 3 | 1215 / 217 | Yes (1.25) | 10 | 0.765 | 0.764 | 0.749 | 0.779 | ep 15 |
| 4 | 1156 / 276 | Yes (1.32) | 5  | 0.696 | 0.691 | 0.641 | 0.741 | ep 10 |

### Per-Food-Type Accuracy

| Food | Accuracy (mean ± std) |
|------|-----------------------|
| Churros | **0.824 ± 0.069** |
| Chips & Salsa | 0.764 ± 0.055 |
| Carrots | 0.736 ± 0.062 |
| Rice & Beans | 0.732 ± 0.077 |

### Comparison to Baseline

| Method | Accuracy (5-fold CV) | Improvement |
|--------|---------------------|-------------|
| Majority class | 54.3% | — |
| LoRA-only (IMWUT baseline) | 63.2% ± 6.7% | — |
| **Stage 1: Spatial+Temporal on cached features** | **76.0% ± 3.8%** | **+12.8 pp** |

### Key Observations
- **G_f1 is stable across all folds** — focal loss eliminated the prediction collapse.
- **Best fold** (2): macro_f1=0.799, val_acc=81.0%. Ran all 20 epochs without early stopping.
- **Weakest fold** (4): macro_f1=0.691. This fold had the highest train imbalance (1.32) and smallest effective training per epoch (early stopped at ep 5).
- **No overfitting**: train_acc peaks at ~84% vs val_acc ~76% — healthy generalization gap.
- **Conditional balancing worked correctly**: folds 1, 3, 4 (ratio ≥ 1.25) used weighted sampling + class weights; folds 0, 2 did not.

### Local Results
- Checkpoints: `results/temporal_v1_5fold/temporal_v1/`
- Training log: `results/temporal_v1_5fold/train_log_53940459.out`
- CV summary: `results/temporal_v1_5fold/temporal_v1/cv_summary.json`

---

## Current Config (scripts/train_temporal.sh)
```
NUM_EPOCHS=20            BATCH_SIZE=8
GRAD_ACCUM_STEPS=2       # effective bs = 16
LR=2e-4                  D_BRANCH=32
N_BRANCHES=4             TEMPORAL_HIDDEN=16
TEMPORAL_OUT=16          N_HEADS=1
N_ATTN_LAYERS=1          TEMPORAL_KERNEL=7
AMP=1                    LABEL_SMOOTHING=0.1
FOCAL_GAMMA=2.0          EARLY_STOP_PATIENCE=5
FEAT_DROPOUT=0.15        BALANCED_SAMPLING=1
IMBALANCE_RATIO_THRESHOLD=1.25
```
