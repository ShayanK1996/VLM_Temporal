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
import traceback
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
    use_amp: bool = False,
    scaler: "torch.amp.GradScaler | None" = None,
    grad_accum_steps: int = 1,
    feat_dropout: float = 0.0,
    class_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Train for one epoch with optional AMP and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_div = 0.0
    correct = 0
    total = 0
    skipped_non_finite = 0

    optimizer.zero_grad()
    num_batches = len(loader)

    for step, batch in enumerate(loader):
        patches = batch["patches"].to(device)   # (B, T, N, D)
        patches = torch.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
        if feat_dropout > 0.0:
            mask = torch.bernoulli(
                torch.full(patches.shape[:3], 1.0 - feat_dropout, device=device)
            ).unsqueeze(-1)
            patches = patches * mask
        labels = batch["label"].to(device)       # (B,)

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(patches, labels=labels, class_weight=class_weight)
            loss = output["loss"]

        if not torch.isfinite(loss):
            skipped_non_finite += 1
            continue

        scaled_loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            if scaler is not None:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        total_ce += output["ce_loss"].item() * labels.size(0)
        total_div += output["diversity_loss"].item() * labels.size(0)
        preds = output["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise RuntimeError(
            f"All training batches were non-finite (skipped={skipped_non_finite}). "
            "Check cached feature tensors for NaN/Inf or extreme values."
        )

    return {
        "loss": total_loss / total,
        "ce_loss": total_ce / total,
        "div_loss": total_div / total,
        "accuracy": correct / total,
        "skipped_non_finite": skipped_non_finite,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: str,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_food_types = []
    skipped_non_finite = 0

    for batch in loader:
        patches = batch["patches"].to(device)
        patches = torch.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(patches, labels=labels)
        if not torch.isfinite(output["loss"]):
            skipped_non_finite += 1
            continue
        
        total_loss += output["loss"].item() * labels.size(0)
        preds = output["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_food_types.extend(batch["food_type"])

    if total == 0:
        raise RuntimeError(
            f"All validation batches were non-finite (skipped={skipped_non_finite}). "
            "Check cached feature tensors for NaN/Inf or extreme values."
        )
    
    # Overall metrics
    accuracy = correct / total
    
    # Per-class metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    results = {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "n_samples": total,
        "skipped_non_finite": skipped_non_finite,
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
    num_workers: int = 0,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    early_stop_patience: int = 0,
    feat_dropout: float = 0.0,
    label_smoothing: float = 0.0,
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
        num_workers=num_workers,
    )

    class_counts = fold_info["class_counts"]
    n_train = fold_info["train_size"]
    class_weight = torch.tensor(
        [n_train / (len(class_counts) * c) for c in class_counts],
        dtype=torch.float, device=device,
    )
    print(f"  Train: {n_train} samples ({len(fold_info['train_participants'])} participants)")
    print(f"  Val:   {fold_info['val_size']} samples ({len(fold_info['val_participants'])} participants)")
    print(f"  Class counts: {dict(enumerate(class_counts))} | weights: {class_weight.tolist()}")

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

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Warmup for first 10% of epochs, then cosine decay
    warmup_epochs = max(1, int(0.1 * config.num_epochs))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.num_epochs - warmup_epochs, eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Training loop — track macro-F1 as primary metric (robust to imbalance)
    best_macro_f1 = 0.0
    best_epoch = 0
    history = []
    epochs_no_improve = 0

    for epoch in range(config.num_epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=config.grad_clip,
            use_amp=use_amp,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
            feat_dropout=feat_dropout,
            class_weight=class_weight,
        )
        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)
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
        
        lr_now = optimizer.param_groups[0]["lr"]
        ni_f1 = val_metrics["needs_improvement_f1"]
        g_f1 = val_metrics["good_f1"]
        macro_f1 = (ni_f1 + g_f1) / 2.0

        # Logging
        print(
            f"  Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} | "
            f"NI_f1={ni_f1:.3f} G_f1={g_f1:.3f} macro_f1={macro_f1:.3f} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        # Save best / early stopping — use macro-F1, not accuracy
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config.__dict__,
            }, fold_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch} (macro_f1 no improvement for {early_stop_patience} epochs)")
                break

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
        "best_macro_f1": best_macro_f1,
        "best_val_accuracy": final_metrics["accuracy"],
        "final_metrics": final_metrics,
        "fold_info": fold_info,
    }

    with open(fold_dir / "fold_result.json", "w") as f:
        json.dump(fold_result, f, indent=2, default=str)

    print(f"  Best: epoch {best_epoch}, macro_f1={best_macro_f1:.3f}, val_acc={final_metrics['accuracy']:.3f}")
    return fold_result


def main():
    parser = argparse.ArgumentParser(description="Train temporal module (Stage 1)")
    
    # Data
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    # Architecture
    parser.add_argument("--d-vision", type=int, default=1536)
    parser.add_argument("--d-branch", type=int, default=64)
    parser.add_argument("--n-branches", type=int, default=4)
    parser.add_argument("--temporal-hidden", type=int, default=32)
    parser.add_argument("--temporal-out", type=int, default=32)
    parser.add_argument("--temporal-kernel", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-attn-layers", type=int, default=2)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    
    # Training
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for CE loss (0=off, 0.1=recommended)")
    parser.add_argument("--early-stop-patience", type=int, default=7,
                        help="Stop if val_acc does not improve for this many epochs (0=disabled)")
    parser.add_argument("--feat-dropout", type=float, default=0.1,
                        help="Randomly zero out this fraction of patches per frame during training")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run specific fold only (default: all folds)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default 0 — safest on NFS / Slurm; try 2–4 if local SSD)",
    )

    # Memory / performance
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (float16)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective bs = batch_size * accum)")

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
    if args.amp:
        eff_bs = args.batch_size * args.grad_accum_steps
        print(f"AMP: enabled | grad_accum: {args.grad_accum_steps} | effective bs: {eff_bs}")
    
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
        label_smoothing=args.label_smoothing,
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
            num_workers=args.num_workers,
            use_amp=args.amp,
            grad_accum_steps=args.grad_accum_steps,
            early_stop_patience=args.early_stop_patience,
            feat_dropout=args.feat_dropout,
            label_smoothing=args.label_smoothing,
        )
        all_results.append(result)
    
    # Summary across folds
    if len(all_results) > 1:
        accs = [r["best_val_accuracy"] for r in all_results]
        macro_f1s = [r["best_macro_f1"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SUMMARY ({len(all_results)} folds)")
        print(f"{'='*60}")
        print(f"  Macro-F1:  {np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}")
        print(f"  Accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Per-fold macro-F1: {[f'{v:.3f}' for v in macro_f1s]}")

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
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s)),
            "per_fold_macro_f1": macro_f1s,
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
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
