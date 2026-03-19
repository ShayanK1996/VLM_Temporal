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

### EXP-001: [COMPLETE] Feature extraction
- **Status**: Complete
- **Goal**: Cache per-frame patch tokens from Qwen2.5-VL for all 4,251 segments
- **Script**: `python -m src.training.extract_features --dataset-jsonl <path>/qwen_dataset.jsonl --output-dir cached_features/`
- **Output**: `cached_features/` with .pt files + `manifest.json`
- **Output .pt format**: `{"patches": tensor(T, H*W, 1536), "label": int, "num_frames": int, "patches_per_frame": int, "d_vision": 1536, "metadata": dict}`

#### Bugs encountered and fixed (sequential blockers)

Each bug below masked the next — they could only be discovered and fixed one at a time.

**Bug 1: Visual encoder attribute path mismatch**
- **Error**: `'Qwen2_5_VLForConditionalGeneration' object has no attribute 'visual'`
- **Root cause**: Qwen2.5-VL nests the vision encoder at `model.model.visual` (the outer `model` is the conditional-generation wrapper, the inner `model` is the core Qwen2.5 model). When PEFT wraps the model, it adds yet another layer: `model.model.model.visual`. Our initial code assumed `model.visual`.
- **Fix** (`extract_features.py:119-136`): Hierarchical attribute lookup that checks three paths in order:
  1. `model.model.visual` (standard transformers)
  2. `model.visual` (unlikely but defensive)
  3. `model.model.model.visual` (PeftModel wrapping)
- **Impact**: 100% failure — no videos could be processed.

**Bug 2: `fps` / `nframes` parameter conflict in qwen_vl_utils**
- **Error**: `Only accept either 'fps' or 'nframes'` — all videos silently rejected by `qwen_vl_utils.process_vision_info()`
- **Root cause**: We passed `{"type": "video", "video": path, "fps": 1.0}` in the message content. The `qwen_vl_utils` library internally sets a default `nframes` and raises when both are present.
- **Fix** (`extract_features.py:163`): Replaced `fps` with `nframes: num_frames` in the message dict. This tells the processor exactly how many frames we want.
- **Impact**: 100% failure — 0 .pt files produced.

**Bug 3: `.mov` files crash decord and torchvision video backends**
- **Error**: `video_reader_backend torchvision error, use torchvision as default, msg: 'video_fps'`
- **Root cause**: The dataset contains a mix of `.mov` (HEVC/H.265) and `.mp4` files. Neither `decord` nor `torchvision`'s video reader can reliably extract metadata from `.mov` containers — they crash with a KeyError on `video_fps`. The `qwen_vl_utils.process_vision_info()` function uses these backends internally and has no fallback.
- **Fix** (`extract_features.py:39-74`): Bypassed `qwen_vl_utils` video reading entirely. Wrote a custom `_read_video_pyav()` function using PyAV (libav/ffmpeg bindings) that:
  - Opens any container format PyAV/ffmpeg supports (.mov, .mp4, .avi, etc.)
  - Handles containers that don't report frame count (decodes once to count)
  - Uniformly samples `num_frames` via `np.linspace` over frame indices
  - Returns `List[PIL.Image]`, which is passed directly to the processor as `videos=[pil_frames]`
  - Pads by repeating the last frame if the video is shorter than requested
- **Impact**: Crash at ~64% of dataset — all progress lost on each attempt.

**Bug 4: Visual encoder returns wrapper object, not raw tensor**
- **Error**: `'BaseModelOutputWithPooling' object has no attribute 'shape'`
- **Root cause**: `visual_encoder(pixel_values, grid_thw=...)` returns a `BaseModelOutputWithPooling` named tuple (from `transformers`), not a bare tensor. The raw hidden states are inside `.last_hidden_state`.
- **Fix** (`extract_features.py:200-201`):
  ```python
  if hasattr(vision_output, 'last_hidden_state'):
      vision_output = vision_output.last_hidden_state
  ```
- **Impact**: 100% failure — 0 .pt files, every segment errored.

#### Key implementation decisions

1. **PyAV over decord/torchvision**: PyAV wraps ffmpeg and handles virtually any container/codec combination. It's the only backend that works reliably across our mixed .mov/.mp4 dataset.
2. **Pre-read frames, then pass to processor**: Instead of letting `qwen_vl_utils` handle video I/O (which uses broken backends), we read frames ourselves and pass `List[PIL.Image]` directly to `Qwen2_5_VLProcessor.__call__(videos=[pil_frames])`.
3. **`pixel_values` vs `pixel_values_videos`**: The processor may return video tensors under either key depending on transformers version. We check both (`extract_features.py:186-191`).
4. **Reshape using `image_grid_thw`**: The vision encoder returns a flat sequence of patch tokens. We use the `(t, h, w)` grid info to reshape into `(T, H*W, D)` where `T` = temporal frames, `H*W` = spatial patches per frame, `D` = 1536 (vision hidden dim).
5. **float16 storage**: Cached tensors are downcast from bfloat16 to float16 to halve disk usage without meaningful precision loss for downstream training.

### EXP-002: [PENDING] Temporal module baseline (all defaults)
- **Status**: Not started  
- **Goal**: First training run with default hyperparameters
- **Script**: `scripts/train_temporal.sh`
- **Config**: d_branch=128, n_branches=4, temporal_hidden=64, kernel=3, heads=4, layers=2
- **Expected**: If > 63.2%, temporal module adds value. If < 63.2%, need tuning.
- **Notes**: Run on L40S. Should take ~1-2 hours for all 5 folds.
