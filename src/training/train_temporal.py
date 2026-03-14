#!/usr/bin/env python3
"""
Stage 1: Train Spatial Decomposition + Temporal Attention on cached VLM features.

This is the fast-iteration training loop. No VLM forward pass — operates
entirely on cached per-frame patch tokens. Can run on L40S 48GB.

5-fold participant-level cross-validation (same protocol as IMWUT paper).

Usage:
    python -m src.training.train_temporal \
        --manifest /path/to/cached_features/manifest.json \
        --feature-dir /path/to/cached_features \
        --output-dir /path/to/temporal_checkpoints \
        --num-epochs 30 \
        --batch-size 32 \
        --lr 1e-3 \
        --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.vlm_temporal_model import TemporalBehaviorModel, TemporalModelConfig
from src.data.feature_dataset import get_fold_split


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_div = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        patches = batch["patches"].to(device)   # (B, T, N, D)
        labels = batch["label"].to(device)       # (B,)
        
        output = model(patches, labels=labels)
        loss = output["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        total_ce += output["ce_loss"].item() * labels.size(0)
        total_div += output["diversity_loss"].item() * labels.size(0)
        preds = output["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {
        "loss": total_loss / total,
        "ce_loss": total_ce / total,
        "div_loss": total_div / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: str,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_food_types = []
    
    for batch in loader:
        patches = batch["patches"].to(device)
        labels = batch["label"].to(device)
        
        output = model(patches, labels=labels)
        
        total_loss += output["loss"].item() * labels.size(0)
        preds = output["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_food_types.extend(batch["food_type"])
    
    # Overall metrics
    accuracy = correct / total
    
    # Per-class metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    results = {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "n_samples": total,
    }
    
    # Per-class precision/recall/f1
    for cls_id, cls_name in enumerate(["needs_improvement", "good"]):
        mask_pred = all_preds == cls_id
        mask_true = all_labels == cls_id
        tp = (mask_pred & mask_true).sum()
        fp = (mask_pred & ~mask_true).sum()
        fn = (~mask_pred & mask_true).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[f"{cls_name}_precision"] = float(precision)
        results[f"{cls_name}_recall"] = float(recall)
        results[f"{cls_name}_f1"] = float(f1)
    
    # Per-food-type accuracy
    food_types = np.array(all_food_types)
    for ft in sorted(set(all_food_types)):
        mask = food_types == ft
        if mask.sum() > 0:
            ft_acc = (all_preds[mask] == all_labels[mask]).mean()
            results[f"acc_{ft}"] = float(ft_acc)
    
    return results


def run_fold(
    fold_id: int,
    config: TemporalModelConfig,
    manifest_path: str,
    feature_dir: str,
    output_dir: Path,
    device: str,
) -> Dict:
    """Train and evaluate one fold."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_id}")
    print(f"{'='*60}")
    
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    train_loader, val_loader, fold_info = get_fold_split(
        manifest_path=manifest_path,
        feature_dir=feature_dir,
        fold_id=fold_id,
        n_folds=5,
        max_frames=config.num_frames,
        batch_size=config.batch_size,
    )
    
    print(f"  Train: {fold_info['train_size']} samples ({len(fold_info['train_participants'])} participants)")
    print(f"  Val:   {fold_info['val_size']} samples ({len(fold_info['val_participants'])} participants)")
    
    # Peek at one sample to get d_vision
    sample = next(iter(train_loader))
    actual_d_vision = sample["patches"].shape[-1]
    if actual_d_vision != config.d_vision:
        print(f"  NOTE: Adjusting d_vision from {config.d_vision} to {actual_d_vision}")
        config.d_vision = actual_d_vision
    
    # Model
    model = TemporalBehaviorModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6,
    )
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(config.num_epochs):
        t0 = time.time()
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=config.grad_clip,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        history.append(epoch_record)
        
        # Logging
        print(
            f"  Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} | "
            f"NI_f1={val_metrics['needs_improvement_f1']:.3f} G_f1={val_metrics['good_f1']:.3f} | "
            f"{elapsed:.1f}s"
        )
        
        # Save best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config.__dict__,
            }, fold_dir / "best_model.pt")
    
    # Save training history
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    # Load best model for final evaluation
    ckpt = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_metrics = evaluate(model, val_loader, device)
    
    fold_result = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "final_metrics": final_metrics,
        "fold_info": fold_info,
    }
    
    with open(fold_dir / "fold_result.json", "w") as f:
        json.dump(fold_result, f, indent=2, default=str)
    
    print(f"  Best: epoch {best_epoch}, val_acc={best_val_acc:.3f}")
    return fold_result


def main():
    parser = argparse.ArgumentParser(description="Train temporal module (Stage 1)")
    
    # Data
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    # Architecture
    parser.add_argument("--d-vision", type=int, default=1536)
    parser.add_argument("--d-branch", type=int, default=128)
    parser.add_argument("--n-branches", type=int, default=4)
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--temporal-out", type=int, default=64)
    parser.add_argument("--temporal-kernel", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-attn-layers", type=int, default=2)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    
    # Training
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run specific fold only (default: all folds)")
    
    # General
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Config
    config = TemporalModelConfig(
        d_vision=args.d_vision,
        d_branch=args.d_branch,
        n_branches=args.n_branches,
        temporal_hidden=args.temporal_hidden,
        temporal_out=args.temporal_out,
        temporal_kernel=args.temporal_kernel,
        n_heads_temporal=args.n_heads,
        n_attn_layers=args.n_attn_layers,
        diversity_weight=args.diversity_weight,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_frames=16,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    # Run folds
    folds_to_run = [args.fold] if args.fold is not None else list(range(args.n_folds))
    all_results = []
    
    for fold_id in folds_to_run:
        result = run_fold(
            fold_id=fold_id,
            config=config,
            manifest_path=args.manifest,
            feature_dir=args.feature_dir,
            output_dir=output_dir,
            device=device,
        )
        all_results.append(result)
    
    # Summary across folds
    if len(all_results) > 1:
        accs = [r["best_val_accuracy"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SUMMARY ({len(all_results)} folds)")
        print(f"{'='*60}")
        print(f"  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Per-fold: {[f'{a:.3f}' for a in accs]}")
        
        # Per-food-type summary
        food_accs = {}
        for r in all_results:
            for key, val in r["final_metrics"].items():
                if key.startswith("acc_"):
                    food = key[4:]
                    food_accs.setdefault(food, []).append(val)
        
        if food_accs:
            print(f"\n  Per-food-type accuracy:")
            for food, vals in sorted(food_accs.items()):
                print(f"    {food}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        
        # Save summary
        summary = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "per_fold_accuracy": accs,
            "food_type_accuracy": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                    for k, v in food_accs.items()},
            "n_folds": len(all_results),
        }
        with open(output_dir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
