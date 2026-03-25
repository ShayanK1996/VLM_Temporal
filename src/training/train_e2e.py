#!/usr/bin/env python3
"""
Stage 2: End-to-End Fine-Tuning — LoRA on VLM visual encoder + temporal head.

Joint optimisation of:
    - Qwen2.5-VL visual encoder via LoRA adapters
    - Spatial decomposition module  (loaded from Stage 1)
    - Temporal attention module      (loaded from Stage 1)

The language model head is NOT used here — this stage focuses purely on
improving the classification path by letting gradients flow through the
vision encoder so it can learn eating-behaviour-specific features.

Requires A100 80GB (VLM + temporal model + activations for backprop).

Usage:
    python -m src.training.train_e2e \\
        --manifest /path/to/cached_features/manifest.json \\
        --temporal-checkpoint /path/to/temporal_v1/fold_0/best_model.pt \\
        --output-dir /path/to/e2e_checkpoints \\
        --fold 0 \\
        --num-epochs 5 \\
        --lr 2e-5 \\
        --grad-accum-steps 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.vlm_temporal_model import TemporalBehaviorModel, TemporalModelConfig
from src.data.video_dataset import get_e2e_fold_split


# ---------------------------------------------------------------------------
# VLM setup
# ---------------------------------------------------------------------------

def setup_vlm_with_lora(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_pixels: int = 200704,
    device: str = "cuda",
):
    """Load Qwen2.5-VL and apply LoRA to the **visual encoder** only.

    LoRA targets ``qkv`` (fused Q/K/V projection in each ViT block).
    The LM decoder stays on CPU (never called) to save ~5 GB of VRAM.

    Returns (vlm_model, processor, visual_encoder).
    """
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )
    from peft import LoraConfig, get_peft_model

    print(f"Loading {model_name} ...")

    # Cap resolution so the visual encoder sees ~300 patches/frame instead
    # of thousands.  Default max_pixels is huge and produces an N² attention
    # matrix that blows up 80 GB VRAM.
    # 256 * 28 * 28 = 200704 → roughly 16×20 grid after Qwen's 14px patches
    # and 2×2 spatial merge → ~320 patches/frame (matches Stage 1 features).
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_name,
        min_pixels=4 * 28 * 28,
        max_pixels=max_pixels,
    )
    print(f"  Processor max_pixels={max_pixels} "
          f"(~{max_pixels // (28*28)} spatial blocks → "
          f"~{max_pixels // (28*28*4)} merged patches/frame)")

    # Load full model on CPU first — we'll selectively move parts to GPU
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # LoRA on visual encoder attention only.
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["qkv"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing on the visual encoder to save VRAM
    visual = _get_visual_encoder(model)
    if hasattr(visual, "gradient_checkpointing_enable"):
        visual.gradient_checkpointing_enable()
        print("  Visual encoder gradient checkpointing: enabled")

    # Move ONLY the visual encoder to GPU.  The LM decoder (~2.4B params)
    # stays on CPU — it's never called, so keeping it on GPU wastes ~5 GB.
    visual.to(device)
    # Patch embed / merger might sit outside the blocks list
    for name, param in model.named_parameters():
        if "visual" in name and param.device.type != device:
            param.data = param.data.to(device)

    n_gpu = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
    n_cpu = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
    print(f"  GPU params: {n_gpu/1e6:.0f}M  |  CPU params (LM, unused): {n_cpu/1e6:.0f}M")

    return model, processor, visual


def _get_visual_encoder(model: nn.Module) -> nn.Module:
    """Navigate PeftModel / HF wrapping to find the visual encoder."""
    base = model.base_model if hasattr(model, "base_model") else model
    if hasattr(base, "model") and hasattr(base.model, "visual"):
        return base.model.visual
    if hasattr(base, "visual"):
        return base.visual
    if (
        hasattr(base, "model")
        and hasattr(base.model, "model")
        and hasattr(base.model.model, "visual")
    ):
        return base.model.model.visual
    raise AttributeError(
        f"Cannot find visual encoder on {type(model).__name__}. "
        f"Attrs: {[a for a in dir(base) if not a.startswith('_')]}"
    )


# ---------------------------------------------------------------------------
# Forward pass helper
# ---------------------------------------------------------------------------

def forward_e2e(
    visual_encoder: nn.Module,
    temporal_model: TemporalBehaviorModel,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    class_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """VLM visual encoder → reshape → temporal model → output dict."""
    vision_output = visual_encoder(pixel_values, grid_thw=image_grid_thw)
    if hasattr(vision_output, "last_hidden_state"):
        vision_output = vision_output.last_hidden_state

    t, h, w = image_grid_thw[0].tolist()
    patches_per_frame = int(h) * int(w)
    frame_patches = vision_output.reshape(1, int(t), patches_per_frame, -1)

    return temporal_model(
        frame_patches.float(), labels=labels, class_weight=class_weight,
    )


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    visual_encoder: nn.Module,
    temporal_model: TemporalBehaviorModel,
    vlm_model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    class_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    vlm_model.train()
    temporal_model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    correct = 0
    total = 0
    steps_done = 0

    for step, batch in enumerate(train_loader):
        pv = batch["pixel_values"].to(device)
        grid = batch["image_grid_thw"].to(device)
        labels = batch["label"].unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = forward_e2e(
                visual_encoder, temporal_model,
                pv, grid, labels=labels, class_weight=class_weight,
            )
            loss = output["loss"] / grad_accum_steps

        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            nn.utils.clip_grad_norm_(
                list(vlm_model.parameters()) + list(temporal_model.parameters()),
                max_norm=grad_clip,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_done += 1

        with torch.no_grad():
            total_loss += output["loss"].item() * grad_accum_steps
            pred = output["logits"].argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.shape[0]

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "optimizer_steps": steps_done,
    }


@torch.no_grad()
def evaluate(
    visual_encoder: nn.Module,
    temporal_model: TemporalBehaviorModel,
    vlm_model: nn.Module,
    val_loader,
    device: str,
) -> Dict[str, float]:
    vlm_model.eval()
    temporal_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_food_types = []

    for batch in val_loader:
        pv = batch["pixel_values"].to(device)
        grid = batch["image_grid_thw"].to(device)
        labels = batch["label"].unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = forward_e2e(visual_encoder, temporal_model, pv, grid, labels=labels)

        total_loss += output["loss"].item()
        pred = output["logits"].argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.shape[0]
        all_preds.append(pred.cpu().item())
        all_labels.append(labels.cpu().item())
        all_food_types.append(batch.get("food_type", "unknown"))

    accuracy = correct / max(total, 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results: Dict[str, float] = {
        "loss": total_loss / max(total, 1),
        "accuracy": accuracy,
        "n_samples": total,
    }

    for cls_id, cls_name in enumerate(["needs_improvement", "good"]):
        mask_pred = all_preds == cls_id
        mask_true = all_labels == cls_id
        tp = int((mask_pred & mask_true).sum())
        fp = int((mask_pred & ~mask_true).sum())
        fn = int((~mask_pred & mask_true).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[f"{cls_name}_f1"] = float(f1)

    food_arr = np.array(all_food_types)
    for ft in sorted(set(all_food_types)):
        mask = food_arr == ft
        if mask.sum() > 0:
            results[f"acc_{ft}"] = float((all_preds[mask] == all_labels[mask]).mean())

    return results


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def run_fold(
    fold_id: int,
    temporal_config: TemporalModelConfig,
    vlm_model: nn.Module,
    visual_encoder: nn.Module,
    processor,
    manifest_path: str,
    temporal_checkpoint: Optional[str],
    output_dir: Path,
    device: str,
    num_frames: int = 16,
    video_root: Optional[str] = None,
    num_workers: int = 0,
    num_epochs: int = 5,
    lora_lr: float = 2e-5,
    temporal_lr: float = 2e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 16,
    early_stop_patience: int = 3,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    imbalance_ratio_threshold: float = 1.25,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"FOLD {fold_id}")
    print(f"{'='*60}")

    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    train_loader, val_loader, fold_info = get_e2e_fold_split(
        manifest_path=manifest_path,
        fold_id=fold_id,
        processor=processor,
        num_frames=num_frames,
        video_root=video_root,
        num_workers=num_workers,
    )

    n_train = fold_info["train_size"]
    train_class_counts = fold_info["train_class_counts"]
    train_imbalance_ratio = fold_info["train_imbalance_ratio"]
    use_class_weight = train_imbalance_ratio >= imbalance_ratio_threshold

    class_weight = None
    if use_class_weight:
        class_weight = torch.tensor(
            [n_train / (len(train_class_counts) * c) for c in train_class_counts],
            dtype=torch.float, device=device,
        )

    print(f"  Train: {n_train} samples ({len(fold_info['train_participants'])} participants)")
    print(f"  Val:   {fold_info['val_size']} samples ({len(fold_info['val_participants'])} participants)")
    print(f"  Train class counts: {dict(enumerate(train_class_counts))} | "
          f"val class counts: {dict(enumerate(fold_info['val_class_counts']))}")
    print(f"  Train imbalance: {train_imbalance_ratio:.2f} | class_weighting={use_class_weight}")

    # ---- temporal model (from Stage 1) ----
    temporal_config.focal_gamma = focal_gamma
    temporal_config.label_smoothing = label_smoothing

    temporal_model = TemporalBehaviorModel(temporal_config).to(device)

    if temporal_checkpoint:
        ckpt_path = temporal_checkpoint.replace("{fold}", str(fold_id))
        if Path(ckpt_path).exists():
            print(f"  Loading Stage 1 checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            temporal_model.load_state_dict(ckpt["model_state_dict"])
            ckpt_epoch = ckpt.get("epoch", "?")
            ckpt_f1 = ckpt.get("val_metrics", {}).get("good_f1", "?")
            print(f"    epoch={ckpt_epoch}, G_f1={ckpt_f1}")
        else:
            print(f"  WARNING: checkpoint not found: {ckpt_path} — training from scratch")

    # ---- optimizer: different LR for LoRA vs temporal ----
    lora_params = [p for n, p in vlm_model.named_parameters() if p.requires_grad]
    temporal_params = list(temporal_model.parameters())

    n_lora = sum(p.numel() for p in lora_params)
    n_temporal = sum(p.numel() for p in temporal_params)
    print(f"  LoRA params: {n_lora:,} (lr={lora_lr:.1e})")
    print(f"  Temporal params: {n_temporal:,} (lr={temporal_lr:.1e})")

    optimizer = AdamW([
        {"params": lora_params, "lr": lora_lr},
        {"params": temporal_params, "lr": temporal_lr},
    ], weight_decay=weight_decay)

    # Cosine decay (no warmup — both components are pre-trained)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    # ---- training loop ----
    best_macro_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(num_epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            visual_encoder, temporal_model, vlm_model, train_loader,
            optimizer, device, grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps, class_weight=class_weight,
        )
        val_metrics = evaluate(
            visual_encoder, temporal_model, vlm_model, val_loader, device,
        )
        scheduler.step()

        elapsed = time.time() - t0
        ni_f1 = val_metrics["needs_improvement_f1"]
        g_f1 = val_metrics["good_f1"]
        macro_f1 = (ni_f1 + g_f1) / 2.0

        lr_lora = optimizer.param_groups[0]["lr"]
        lr_temp = optimizer.param_groups[1]["lr"]
        print(
            f"  Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} | "
            f"NI_f1={ni_f1:.3f} G_f1={g_f1:.3f} macro_f1={macro_f1:.3f} | "
            f"lr_lora={lr_lora:.2e} lr_temp={lr_temp:.2e} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch, "train": train_metrics, "val": val_metrics,
            "lr_lora": lr_lora, "lr_temp": lr_temp, "elapsed_s": elapsed,
        })

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            # Save temporal model + LoRA adapter separately
            torch.save({
                "epoch": epoch,
                "temporal_state_dict": temporal_model.state_dict(),
                "val_metrics": val_metrics,
                "config": temporal_config.__dict__,
            }, fold_dir / "best_temporal.pt")
            vlm_model.save_pretrained(str(fold_dir / "best_lora_adapter"))
        else:
            epochs_no_improve += 1
            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(macro_f1 no improvement for {early_stop_patience} epochs)")
                break

    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # ---- final evaluation with best checkpoint ----
    best_temporal_ckpt = torch.load(
        fold_dir / "best_temporal.pt", map_location="cpu", weights_only=False,
    )
    temporal_model.load_state_dict(best_temporal_ckpt["temporal_state_dict"])
    final_metrics = evaluate(
        visual_encoder, temporal_model, vlm_model, val_loader, device,
    )

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

    print(f"  Best: epoch {best_epoch}, macro_f1={best_macro_f1:.3f}, "
          f"val_acc={final_metrics['accuracy']:.3f}")
    return fold_result


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 — End-to-end LoRA + temporal training",
    )

    # Data
    parser.add_argument("--manifest", type=str, required=True,
                        help="manifest.json from Stage 1 feature extraction")
    parser.add_argument("--video-root", type=str, default=None,
                        help="Optional root dir to resolve video paths")
    parser.add_argument("--temporal-checkpoint", type=str, default=None,
                        help="Stage 1 checkpoint (use {fold} placeholder for per-fold)")
    parser.add_argument("--output-dir", type=str, required=True)

    # VLM / LoRA
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-pixels", type=int, default=200704,
                        help="Max total pixels per frame sent to VLM processor "
                             "(256*28*28=200704). Reduces ViT attention N².")

    # Architecture (must match Stage 1)
    parser.add_argument("--d-vision", type=int, default=1280)
    parser.add_argument("--d-branch", type=int, default=32)
    parser.add_argument("--n-branches", type=int, default=4)
    parser.add_argument("--temporal-hidden", type=int, default=16)
    parser.add_argument("--temporal-out", type=int, default=16)
    parser.add_argument("--temporal-kernel", type=int, default=7)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--n-attn-layers", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=16)

    # Training
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--temporal-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=16,
                        help="Gradient accumulation (effective bs = 1 * accum)")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--imbalance-ratio-threshold", type=float, default=1.25)

    # Fold selection
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a single fold (default: all folds)")

    # General
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- VLM + LoRA ----
    vlm_model, processor, visual_encoder = setup_vlm_with_lora(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_pixels=args.max_pixels,
        device=device,
    )

    # ---- temporal config (must match Stage 1) ----
    temporal_config = TemporalModelConfig(
        d_vision=args.d_vision,
        d_branch=args.d_branch,
        n_branches=args.n_branches,
        temporal_hidden=args.temporal_hidden,
        temporal_out=args.temporal_out,
        temporal_kernel=args.temporal_kernel,
        n_heads_temporal=args.n_heads,
        n_attn_layers=args.n_attn_layers,
        num_frames=args.num_frames,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    # Save config
    with open(output_dir / "e2e_config.json", "w") as f:
        json.dump({**vars(args), "temporal_config": temporal_config.__dict__}, f, indent=2, default=str)

    print(f"\nDevice: {device}")
    print(f"Grad accum: {args.grad_accum_steps} (effective bs = {args.grad_accum_steps})")

    # ---- run folds ----
    folds_to_run = [args.fold] if args.fold is not None else list(range(args.n_folds))
    all_results = []

    for fold_id in folds_to_run:
        result = run_fold(
            fold_id=fold_id,
            temporal_config=temporal_config,
            vlm_model=vlm_model,
            visual_encoder=visual_encoder,
            processor=processor,
            manifest_path=args.manifest,
            temporal_checkpoint=args.temporal_checkpoint,
            output_dir=output_dir,
            device=device,
            num_frames=args.num_frames,
            video_root=args.video_root,
            num_workers=args.num_workers,
            num_epochs=args.num_epochs,
            lora_lr=args.lora_lr,
            temporal_lr=args.temporal_lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            early_stop_patience=args.early_stop_patience,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            imbalance_ratio_threshold=args.imbalance_ratio_threshold,
        )
        all_results.append(result)

    # ---- cross-validation summary ----
    if len(all_results) > 1:
        macro_f1s = [r["best_macro_f1"] for r in all_results]
        accs = [r["best_val_accuracy"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SUMMARY ({len(all_results)} folds)")
        print(f"{'='*60}")
        print(f"  Macro-F1:  {np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}")
        print(f"  Accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Per-fold macro-F1: {[f'{v:.3f}' for v in macro_f1s]}")

        food_accs: Dict[str, list] = {}
        for r in all_results:
            for key, val in r["final_metrics"].items():
                if key.startswith("acc_"):
                    food_accs.setdefault(key[4:], []).append(val)
        if food_accs:
            print(f"\n  Per-food-type accuracy:")
            for food, vals in sorted(food_accs.items()):
                print(f"    {food}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

        summary = {
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s)),
            "per_fold_macro_f1": macro_f1s,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "per_fold_accuracy": accs,
            "food_type_accuracy": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in food_accs.items()
            },
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
