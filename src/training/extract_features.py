#!/usr/bin/env python3
"""
Extract and cache per-frame patch tokens from Qwen2.5-VL for all video segments.

Run this ONCE on an A100 node. Cached features are stored as .pt files and used
by the temporal module training (which runs on cheaper L40S nodes).

Usage:
    python -m src.training.extract_features \
        --video-dir /path/to/segmented_videos \
        --metadata-csv /path/to/segments_metadata.csv \
        --output-dir /path/to/cached_features \
        --model-name Qwen/Qwen2.5-VL-3B-Instruct \
        --num-frames 16 \
        --batch-size 1

Output structure:
    cached_features/
        segment_0001.pt   # {'patches': tensor(T, N, D), 'label': int, 'metadata': dict}
        segment_0002.pt
        ...
        manifest.json     # index of all cached features with metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def _read_video_pyav(path: str, num_frames: int) -> List[Image.Image]:
    """Read uniformly-sampled frames from a video using PyAV.

    Handles both .mov and .mp4 reliably, unlike decord/torchvision backends
    which fail on many .mov files with 'video_fps' KeyError.
    """
    import av

    container = av.open(path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:
        # Some containers don't report frame count; decode to count
        total_frames = sum(1 for _ in container.decode(video=0))
        container.close()
        container = av.open(path)

    if total_frames == 0:
        container.close()
        raise ValueError(f"No video frames found in {path}")

    indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_image())  # -> PIL.Image
        if len(frames) == num_frames:
            break
    container.close()

    if len(frames) < num_frames:
        # Pad by repeating last frame if video is shorter than requested
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())

    return frames


def extract_features_batch(
    video_paths: List[str],
    labels: List[int],
    metadata_list: List[Dict],
    model_name: str,
    lora_dir: Optional[str],
    num_frames: int,
    output_dir: Path,
    device: str = "cuda",
):
    """Extract and cache features for a batch of video segments."""
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )
    print(f"Loading model: {model_name}")
    processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)

    if lora_dir is not None:
        lora_path = Path(lora_dir)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA dir not found: {lora_dir}")
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "peft is required to load LoRA adapters. Install it in your environment."
            ) from e

        print(f"Merging LoRA adapter from: {lora_dir}")
        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Find the visual encoder — attribute path varies across transformers versions
    # and PeftModel wrapping depth.
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        visual_encoder = model.model.visual
    elif hasattr(model, "visual"):
        visual_encoder = model.visual
    elif (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "visual")
    ):
        # PeftModel wrapping: peft -> qwen_conditional -> qwen_model -> visual
        visual_encoder = model.model.model.visual
    else:
        raise AttributeError(
            f"Cannot find visual encoder on {type(model).__name__}. "
            f"Top-level attrs: {[a for a in dir(model) if not a.startswith('_')]}"
        )
    print(f"Visual encoder: {type(visual_encoder).__name__}")

    manifest = []
    output_dir.mkdir(parents=True, exist_ok=True)
    n_errors = 0

    for idx, (vpath, label, meta) in enumerate(
        tqdm(zip(video_paths, labels, metadata_list), total=len(video_paths), desc="Extracting")
    ):
        segment_id = f"segment_{idx:05d}"
        output_path = output_dir / f"{segment_id}.pt"

        if output_path.exists():
            manifest.append({
                "segment_id": segment_id,
                "video_path": str(vpath),
                "label": label,
                "cached_path": str(output_path),
                **meta,
            })
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(vpath), "nframes": num_frames},
                    {"type": "text", "text": "Analyze eating behavior."},
                ],
            }
        ]

        try:
            # Read frames with PyAV — handles .mov and .mp4 reliably,
            # bypassing qwen_vl_utils's buggy decord/torchvision backends.
            pil_frames = _read_video_pyav(str(vpath), num_frames)

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=None,
                videos=[pil_frames],
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                pixel_values = inputs.get("pixel_values")
                image_grid_thw = inputs.get("image_grid_thw")

                if pixel_values is None:
                    pixel_values = inputs.get("pixel_values_videos")
                    image_grid_thw = inputs.get("video_grid_thw")

                if pixel_values is None:
                    print(f"  WARNING: No pixel values for {vpath}, skipping")
                    n_errors += 1
                    continue

                vision_output = visual_encoder(pixel_values, grid_thw=image_grid_thw)
                # Unwrap BaseModelOutputWithPooling → raw tensor
                if hasattr(vision_output, 'last_hidden_state'):
                    vision_output = vision_output.last_hidden_state

                t, h, w = image_grid_thw[0].tolist()
                t, h, w = int(t), int(h), int(w)
                patches_per_frame = h * w
                d_vision = vision_output.shape[-1]

                frame_patches = vision_output.reshape(
                    t, patches_per_frame, d_vision
                ).cpu().to(torch.float16)

                torch.save({
                    "patches": frame_patches,
                    "label": label,
                    "num_frames": t,
                    "patches_per_frame": patches_per_frame,
                    "d_vision": d_vision,
                    "metadata": meta,
                }, output_path)

                manifest.append({
                    "segment_id": segment_id,
                    "video_path": str(vpath),
                    "label": label,
                    "num_frames": t,
                    "patches_per_frame": patches_per_frame,
                    "d_vision": d_vision,
                    "cached_path": str(output_path),
                    **meta,
                })

        except Exception as e:
            print(f"  ERROR on {vpath}: {e}")
            n_errors += 1
            continue

        if (idx + 1) % 100 == 0:
            _save_manifest(manifest, output_dir)
            print(f"  Saved manifest ({len(manifest)} segments, {n_errors} errors)")

    _save_manifest(manifest, output_dir)
    print(f"\nDone! Extracted {len(manifest)} segments to {output_dir} ({n_errors} errors)")
    return manifest


def _save_manifest(manifest, output_dir):
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract VLM visual features")
    parser.add_argument("--dataset-jsonl", type=str, required=True,
                        help="Path to qwen_dataset.jsonl (your training data)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save cached features")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=None,
        help="Optional LoRA adapter directory to merge into the base model before feature extraction.",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--segment-types", type=str, nargs="*", default=None,
                        help="Filter to specific segment types (e.g., bite g3 g5)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Load JSONL directly
    print(f"Loading: {args.dataset_jsonl}")
    with open(args.dataset_jsonl) as f:
        records = [json.loads(line) for line in f]
    print(f"  Total records: {len(records)}")
    
    # Food type code → name
    FOOD_MAP = {1: "chips_and_salsa", 2: "carrots", 3: "rice_and_beans", 4: "churros"}
    
    # Binary label mapping
    LABEL_MAP = {"needs_improvement": 0, "good": 1}
    
    # Parse records
    video_paths = []
    labels = []
    metadata_list = []
    
    for r in records:
        meta = r["metadata"]
        clip_path = meta["clip_path"]
        
        # Detect segment type from filename
        seg_type = "unknown"
        for st in ["bite", "g3", "g5", "phase"]:
            if f"_{st}_" in Path(clip_path).stem:
                seg_type = st
                break
        
        # Filter by segment type if requested
        if args.segment_types and seg_type not in args.segment_types:
            continue
        
        # Food type
        food_codes = meta["metrics"].get("food_types", [])
        food_type = FOOD_MAP.get(food_codes[0], "unknown") if food_codes else "unknown"
        
        video_paths.append(clip_path)
        labels.append(LABEL_MAP.get(meta["category"], 0))
        metadata_list.append({
            "participant_id": meta["participant_id"],
            "food_type": food_type,
            "segment_type": seg_type,
            "category_binary": meta["category"],
            "category_original": meta.get("category_original", meta["category"]),
            "mean_cbr": meta["metrics"].get("mean_cbr", ""),
            "bite_rate_per_min": meta["metrics"].get("bite_rate_per_min", ""),
            "mean_pause_sec": meta["metrics"].get("mean_pause_sec", ""),
            "segment_id": r["id"],
        })
    
    if args.segment_types:
        print(f"  After filtering to {args.segment_types}: {len(video_paths)} segments")
    
    # Assign participant-level folds to match fixed CV splits from project context.
    # If participant IDs don't match the expected set, fall back to a deterministic shuffle.
    n_folds = 5
    fixed_folds = {
        0: ["111010", "211003", "111001", "211013", "111013", "111011"],
        1: ["111015", "111003", "211011", "111007", "111004"],
        2: ["111014", "211002", "111005", "111006", "211009"],
        3: ["211004", "211010", "211005", "211015", "211008"],
        4: ["111009", "111012", "211001", "211006", "111008"],
    }
    fixed_fold_map = {pid: fid for fid, pids in fixed_folds.items() for pid in pids}

    unique_pids = sorted(set(m["participant_id"] for m in metadata_list))
    if set(unique_pids).issubset(set(fixed_fold_map.keys())):
        fold_map = fixed_fold_map
        print("  Using fixed participant-level fold mapping (from CURSOR_CONTEXT.md).")
    else:
        print(
            "  WARNING: participant IDs don't match fixed fold mapping; "
            "falling back to deterministic shuffle (seed=42)."
        )
        rng = np.random.RandomState(42)
        shuffled = list(unique_pids)
        rng.shuffle(shuffled)
        fold_map = {}
        fold_size = len(shuffled) // n_folds
        remainder = len(shuffled) % n_folds
        idx = 0
        for fid in range(n_folds):
            size = fold_size + (1 if fid < remainder else 0)
            for _ in range(size):
                if idx < len(shuffled):
                    fold_map[shuffled[idx]] = fid
                    idx += 1

    for m in metadata_list:
        m["fold"] = fold_map.get(m["participant_id"], -1)
    
    print(f"  Fold assignments: { {fid: sum(1 for m in metadata_list if m['fold'] == fid) for fid in range(n_folds)} }")
    
    extract_features_batch(
        video_paths=video_paths,
        labels=labels,
        metadata_list=metadata_list,
        model_name=args.model_name,
        lora_dir=args.lora_dir,
        num_frames=args.num_frames,
        output_dir=Path(args.output_dir),
        device=device,
    )


if __name__ == "__main__":
    main()
