#!/usr/bin/env python3
"""
Evaluation for VLM_TemporalBranch models.

Computes all metrics from the IMWUT paper (for direct comparison) plus
temporal-module-specific analyses:
    - Overall accuracy (binary: needs_improvement vs good)
    - Per-class precision / recall / F1
    - Per-food-type accuracy breakdown
    - Per-fold accuracy with confidence intervals
    - Comparison table: zero-shot → LoRA-only → LoRA+temporal

Usage:
    python -m src.evaluation.evaluate \
        --checkpoint-dir /path/to/temporal_v1 \
        --manifest /path/to/cached_features/manifest.json \
        --feature-dir /path/to/cached_features \
        --output-dir /path/to/evaluation_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_fold_results(checkpoint_dir: Path, n_folds: int = 5) -> List[Dict]:
    """Load all fold results from a training run."""
    results = []
    for fold_id in range(n_folds):
        fold_dir = checkpoint_dir / f"fold_{fold_id}"
        result_path = fold_dir / "fold_result.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            print(f"  WARNING: Missing fold {fold_id} results at {result_path}")
    return results


def compute_summary(fold_results: List[Dict]) -> Dict:
    """Compute cross-validation summary matching IMWUT paper format."""
    
    accs = [r["final_metrics"]["accuracy"] for r in fold_results]
    
    summary = {
        "overall_accuracy": {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "per_fold": [float(a) for a in accs],
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
            "range": float(np.max(accs) - np.min(accs)),
        },
    }
    
    # Per-class metrics
    for cls in ["needs_improvement", "good"]:
        for metric in ["precision", "recall", "f1"]:
            key = f"{cls}_{metric}"
            vals = [r["final_metrics"].get(key, 0.0) for r in fold_results]
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    
    # Per-food-type accuracy
    food_accs = {}
    for r in fold_results:
        for key, val in r["final_metrics"].items():
            if key.startswith("acc_"):
                food = key[4:]
                food_accs.setdefault(food, []).append(val)
    
    summary["per_food_type"] = {}
    for food, vals in sorted(food_accs.items()):
        summary["per_food_type"][food] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n_folds": len(vals),
        }
    
    # Fold sizes
    summary["fold_sizes"] = [
        r["fold_info"]["val_size"] for r in fold_results
    ]
    summary["total_samples"] = sum(summary["fold_sizes"])
    
    return summary


def print_comparison_table(
    temporal_summary: Dict,
    imwut_baseline: Dict = None,
):
    """Print comparison table: zero-shot → LoRA-only → LoRA+temporal."""
    
    # IMWUT baselines (hardcoded from paper)
    if imwut_baseline is None:
        imwut_baseline = {
            "zero_shot": {"accuracy": 0.241, "std": None},
            "majority_class": {"accuracy": 0.543, "std": None},
            "lora_only": {"accuracy": 0.632, "std": 0.067},
            "per_food": {
                "chips_and_salsa": 0.726,
                "carrots": 0.638,
                "rice_and_beans": 0.536,
                "churros": 0.520,
            },
        }
    
    temporal_acc = temporal_summary["overall_accuracy"]["mean"]
    temporal_std = temporal_summary["overall_accuracy"]["std"]
    lora_acc = imwut_baseline["lora_only"]["accuracy"]
    lora_std = imwut_baseline["lora_only"]["std"]
    
    delta = temporal_acc - lora_acc
    direction = "+" if delta > 0 else ""
    
    print("\n" + "=" * 70)
    print("COMPARISON TABLE: VLM Eating Behavior Assessment")
    print("=" * 70)
    print(f"{'Method':<35} {'Accuracy':>12} {'Δ vs LoRA':>12}")
    print("-" * 70)
    print(f"{'Zero-shot Qwen2.5-VL-3B':<35} {'24.1%':>12} {'—':>12}")
    print(f"{'Majority class':<35} {'54.3%':>12} {'—':>12}")
    print(f"{'LoRA fine-tuning (IMWUT)':<35} {f'{lora_acc*100:.1f}% ± {lora_std*100:.1f}%':>12} {'baseline':>12}")
    print(f"{'LoRA + Temporal (this work)':<35} {f'{temporal_acc*100:.1f}% ± {temporal_std*100:.1f}%':>12} {f'{direction}{delta*100:.1f} pp':>12}")
    print("-" * 70)
    
    # Per-food comparison
    print(f"\n{'Per-Food-Type Accuracy:':<35}")
    print(f"{'Food Type':<25} {'LoRA-only':>12} {'LoRA+Temp':>12} {'Δ':>8}")
    print("-" * 60)
    
    food_map = {
        "chips_and_salsa": "Chips & salsa",
        "carrots": "Carrots",
        "rice_and_beans": "Rice & beans",
        "churros": "Churros",
    }
    
    for food_key, food_name in food_map.items():
        lora_food = imwut_baseline["per_food"].get(food_key, 0)
        temporal_food_data = temporal_summary["per_food_type"].get(food_key, {})
        temporal_food = temporal_food_data.get("mean", 0)
        delta_food = temporal_food - lora_food
        dir_food = "+" if delta_food > 0 else ""
        
        print(
            f"  {food_name:<23} "
            f"{lora_food*100:>10.1f}% "
            f"{temporal_food*100:>10.1f}% "
            f"{dir_food}{delta_food*100:>6.1f} pp"
        )
    
    print()
    
    # Key question: does temporal module help on soft foods?
    soft_foods = ["rice_and_beans", "churros"]
    soft_lora = np.mean([imwut_baseline["per_food"][f] for f in soft_foods])
    soft_temporal = np.mean([
        temporal_summary["per_food_type"].get(f, {}).get("mean", 0) 
        for f in soft_foods
    ])
    soft_delta = soft_temporal - soft_lora
    
    print(f"  Soft food avg improvement: {'+' if soft_delta > 0 else ''}{soft_delta*100:.1f} pp")
    if soft_delta > 0.02:
        print("  → Temporal module helps on visually ambiguous foods ✓")
    elif soft_delta > 0:
        print("  → Marginal improvement on soft foods")
    else:
        print("  → No improvement on soft foods — temporal module may need tuning")


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal model")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    fold_results = load_fold_results(checkpoint_dir, args.n_folds)
    if not fold_results:
        print("No fold results found!")
        return
    
    print(f"Loaded {len(fold_results)} fold results from {checkpoint_dir}")
    
    # Compute summary
    summary = compute_summary(fold_results)
    
    # Save
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print
    print(f"\nOverall: {summary['overall_accuracy']['mean']*100:.1f}% ± {summary['overall_accuracy']['std']*100:.1f}%")
    print(f"Range: {summary['overall_accuracy']['min']*100:.1f}% – {summary['overall_accuracy']['max']*100:.1f}%")
    
    print(f"\nPer-class:")
    for cls in ["needs_improvement", "good"]:
        p = summary[f"{cls}_precision"]["mean"]
        r = summary[f"{cls}_recall"]["mean"]
        f1 = summary[f"{cls}_f1"]["mean"]
        print(f"  {cls}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    
    print(f"\nPer-food-type:")
    for food, data in summary["per_food_type"].items():
        print(f"  {food}: {data['mean']*100:.1f}% ± {data['std']*100:.1f}%")
    
    # Comparison table
    print_comparison_table(summary)
    
    print(f"\nFull results saved to {output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
