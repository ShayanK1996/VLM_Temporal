#!/usr/bin/env python3
"""
Generate segments_metadata.csv from qwen_dataset.jsonl.

Bridges the existing VLM_EatingBehavior data format to the
VLM_TemporalBranch feature extraction pipeline.

Input:  qwen_dataset.jsonl (from VLM_EatingBehavior preprocessing)
Output: segments_metadata.csv with columns:
    video_path, label, participant_id, food_type, segment_type,
    category_original, fold, mean_cbr, bite_rate_per_min, mean_pause_sec

Usage:
    python scripts/generate_metadata.py \
        --input ~/VLM_EatingBehavior/qwen_dataset.jsonl \
        --output ~/VLM_TemporalBranch/data/segments_metadata.csv \
        --n-folds 5 \
        --seed 42

    # Filter to specific segment types:
    python scripts/generate_metadata.py \
        --input ~/VLM_EatingBehavior/qwen_dataset.jsonl \
        --output ~/VLM_TemporalBranch/data/segments_metadata_bite_only.csv \
        --segment-types bite \
        --n-folds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


# Food type code → name (from DIBS Study 1a meal order)
FOOD_MAP = {
    1: "chips_and_salsa",
    2: "carrots",
    3: "rice_and_beans",
    4: "churros",
}

# Binary label mapping (matches IMWUT paper)
# eating_too_fast, needs_improvement, adequate → 0 (needs_improvement)
# good → 1 (good)
BINARY_LABEL_MAP = {
    "needs_improvement": 0,
    "good": 1,
}


def detect_segment_type(clip_path: str) -> str:
    """Infer segment type from clip filename."""
    name = Path(clip_path).stem
    if "_bite_" in name:
        return "bite"
    elif "_g3_" in name:
        return "g3"
    elif "_g5_" in name:
        return "g5"
    elif "_phase_" in name:
        return "phase"
    return "unknown"


def assign_folds(
    participant_ids: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, int]:
    """Assign participants to folds deterministically.
    
    Same logic as the IMWUT paper's 5-fold participant-level CV.
    """
    unique_pids = sorted(set(participant_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_pids)
    
    fold_map = {}
    fold_size = len(unique_pids) // n_folds
    remainder = len(unique_pids) % n_folds
    
    idx = 0
    for fold_id in range(n_folds):
        # Distribute remainder across first folds
        size = fold_size + (1 if fold_id < remainder else 0)
        for _ in range(size):
            if idx < len(unique_pids):
                fold_map[unique_pids[idx]] = fold_id
                idx += 1
    
    return fold_map


def main():
    parser = argparse.ArgumentParser(description="Generate metadata CSV from qwen_dataset.jsonl")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to qwen_dataset.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment-types", type=str, nargs="*", default=None,
                        help="Filter to specific segment types (e.g., bite g3 g5)")
    args = parser.parse_args()
    
    # Load qwen dataset
    print(f"Loading: {args.input}")
    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    print(f"  Total records: {len(records)}")
    
    # Extract metadata
    all_pids = [r["metadata"]["participant_id"] for r in records]
    fold_map = assign_folds(all_pids, n_folds=args.n_folds, seed=args.seed)
    
    # Print fold assignments
    from collections import Counter
    fold_counts = Counter(fold_map.values())
    print(f"\nFold assignments ({len(set(all_pids))} participants):")
    for fold_id in sorted(fold_counts):
        pids_in_fold = [p for p, f in fold_map.items() if f == fold_id]
        print(f"  Fold {fold_id}: {len(pids_in_fold)} participants — {pids_in_fold}")
    
    # Build rows
    rows = []
    skipped = 0
    
    for r in records:
        meta = r["metadata"]
        metrics = meta["metrics"]
        clip_path = meta["clip_path"]
        pid = meta["participant_id"]
        binary_cat = meta["category"]
        original_cat = meta["category_original"]
        segment_type = detect_segment_type(clip_path)
        
        # Filter by segment type
        if args.segment_types and segment_type not in args.segment_types:
            skipped += 1
            continue
        
        # Food type (take first; most segments have one food type)
        food_codes = metrics.get("food_types", [])
        food_type = FOOD_MAP.get(food_codes[0], "unknown") if food_codes else "unknown"
        
        # Binary label
        label = BINARY_LABEL_MAP.get(binary_cat, 0)
        
        rows.append({
            "video_path": clip_path,
            "label": label,
            "participant_id": pid,
            "food_type": food_type,
            "segment_type": segment_type,
            "category_binary": binary_cat,
            "category_original": original_cat,
            "fold": fold_map[pid],
            "mean_cbr": metrics.get("mean_cbr", ""),
            "bite_rate_per_min": metrics.get("bite_rate_per_min", ""),
            "mean_pause_sec": metrics.get("mean_pause_sec", ""),
            "n_bites": metrics.get("n_bites", ""),
            "segment_id": r["id"],
        })
    
    if skipped > 0:
        print(f"\n  Skipped {skipped} segments (filtered by type: {args.segment_types})")
    
    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nWritten: {output_path}")
    print(f"  Rows: {len(rows)}")
    
    # Summary
    seg_types = Counter(r["segment_type"] for r in rows)
    food_types = Counter(r["food_type"] for r in rows)
    label_dist = Counter(r["label"] for r in rows)
    fold_dist = Counter(r["fold"] for r in rows)
    
    print(f"\n  Segment types: {dict(seg_types)}")
    print(f"  Food types: {dict(food_types)}")
    print(f"  Labels: needs_improvement={label_dist[0]}, good={label_dist[1]}")
    print(f"  Per-fold sample counts: {dict(sorted(fold_dist.items()))}")
    
    # Per-fold label balance
    print(f"\n  Per-fold label balance:")
    for fold_id in sorted(fold_dist):
        fold_rows = [r for r in rows if r["fold"] == fold_id]
        fold_labels = Counter(r["label"] for r in fold_rows)
        pct_good = fold_labels[1] / len(fold_rows) * 100
        print(f"    Fold {fold_id}: {len(fold_rows)} samples, "
              f"{fold_labels[0]} NI / {fold_labels[1]} good ({pct_good:.1f}% good)")


if __name__ == "__main__":
    main()
