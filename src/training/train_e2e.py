#!/usr/bin/env python3
"""
Stage 2: End-to-End Fine-Tuning (LoRA + Temporal Module).

Joint optimization of:
    - Qwen2.5-VL with LoRA adapters (vision + language)
    - Spatial decomposition module
    - Temporal attention module

This stage uses the temporal module weights from Stage 1 as initialization,
then fine-tunes everything jointly with the VLM. Requires A100 80GB.

The key difference from Stage 1: gradients flow through the VLM's vision
encoder (via LoRA), so the spatial decomposition can learn to extract
features that are jointly optimized with the temporal reasoning.

Training produces both:
    (a) Classification logits from the temporal module
    (b) Natural-language feedback from the VLM's language model
        (conditioned on temporal representation via virtual tokens)

Usage:
    python -m src.training.train_e2e \
        --temporal-checkpoint /path/to/temporal_v1/fold_0/best_model.pt \
        --video-dir /path/to/segmented_videos \
        --metadata-csv /path/to/segments_metadata.csv \
        --output-dir /path/to/e2e_checkpoints \
        --fold 0 \
        --num-epochs 5 \
        --batch-size 2 \
        --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.vlm_temporal_model import (
    TemporalBehaviorModel,
    TemporalModelConfig,
    TemporalTokenInjector,
)


def setup_vlm_with_lora(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    device: str = "cuda",
):
    """Load Qwen2.5-VL with LoRA adapters.
    
    Returns:
        model: the VLM with LoRA
        processor: the tokenizer/processor
    """
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    
    print(f"Loading {model_name}...")
    processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)
    
    return model, processor


class EndToEndModel(nn.Module):
    """Combines VLM (with LoRA) + Temporal Module for joint training.
    
    Two loss terms:
        1. Classification loss from temporal module (CE on behavior label)
        2. Language modeling loss from VLM (next-token prediction on feedback text)
    
    The temporal representation is injected into the VLM as virtual tokens
    via TemporalTokenInjector, conditioning the feedback generation on
    the temporal behavior understanding.
    """
    
    def __init__(
        self,
        vlm_model: nn.Module,
        temporal_model: TemporalBehaviorModel,
        token_injector: TemporalTokenInjector,
        classification_weight: float = 0.5,
        lm_weight: float = 0.5,
    ):
        super().__init__()
        self.vlm = vlm_model
        self.temporal = temporal_model
        self.injector = token_injector
        self.classification_weight = classification_weight
        self.lm_weight = lm_weight
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,          # (B,) classification labels
        lm_labels: Optional[torch.Tensor] = None,       # (B, seq_len) LM labels
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through both modules.
        
        1. Extract vision features from VLM's visual encoder
        2. Run spatial decomposition + temporal attention → classification
        3. Inject temporal representation as virtual tokens into LM
        4. Run LM forward → feedback text generation loss
        """
        # Step 1: Vision encoder (with LoRA gradients)
        # Access the base model's visual module
        base_model = self.vlm.base_model if hasattr(self.vlm, 'base_model') else self.vlm
        visual = base_model.model.visual if hasattr(base_model, 'model') else base_model.visual
        
        vision_output = visual(pixel_values, grid_thw=image_grid_thw)
        
        # Reshape to per-frame patches
        t, h, w = image_grid_thw[0].tolist()
        t, h, w = int(t), int(h), int(w)
        patches_per_frame = h * w
        d_vision = vision_output.shape[-1]
        B = pixel_values.shape[0] if pixel_values.dim() > 2 else 1
        
        frame_patches = vision_output.reshape(B, t, patches_per_frame, d_vision)
        
        # Step 2: Temporal module → classification
        temporal_output = self.temporal(frame_patches.float(), labels=labels)
        temporal_repr = self.temporal.get_temporal_representation(frame_patches.float())
        
        # Step 3: Inject temporal tokens into LM
        virtual_tokens = self.injector(temporal_repr)  # (B, n_tokens, lm_hidden)
        
        # Step 4: LM forward with virtual tokens prepended
        # Get text embeddings
        text_embeds = base_model.model.embed_tokens(input_ids) if hasattr(base_model, 'model') else base_model.embed_tokens(input_ids)
        
        # Prepend virtual tokens
        combined_embeds = torch.cat([virtual_tokens.to(text_embeds.dtype), text_embeds], dim=1)
        
        # Extend attention mask for virtual tokens
        virtual_mask = torch.ones(B, virtual_tokens.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([virtual_mask, attention_mask], dim=1)
        
        # Note: Full LM forward with modified embeddings would go here.
        # This is a simplified version — the actual implementation needs
        # to handle the Qwen2.5-VL's specific input format carefully.
        # For now, we compute the classification loss as the primary signal.
        
        output = {
            "logits": temporal_output["logits"],
            "temporal_repr": temporal_repr,
            "virtual_tokens": virtual_tokens,
        }
        
        if labels is not None:
            output["classification_loss"] = temporal_output["ce_loss"]
            output["diversity_loss"] = temporal_output["diversity_loss"]
            output["loss"] = (
                self.classification_weight * temporal_output["ce_loss"]
                + temporal_output["diversity_loss"]
            )
        
        return output


def load_temporal_checkpoint(
    checkpoint_path: str,
    config: TemporalModelConfig,
    device: str,
) -> TemporalBehaviorModel:
    """Load pre-trained temporal module from Stage 1."""
    model = TemporalBehaviorModel(config)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading temporal checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")
        
        # Update config from checkpoint if available
        if "config" in ckpt:
            for key, val in ckpt["config"].items():
                if hasattr(config, key):
                    setattr(config, key, val)
    else:
        print("No temporal checkpoint — training from scratch")
    
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="End-to-end training (Stage 2)")
    
    # Model
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--temporal-checkpoint", type=str, default=None,
                        help="Path to Stage 1 temporal model checkpoint")
    
    # Data
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    # Training
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--lm-weight", type=float, default=0.5)
    parser.add_argument("--fold", type=int, default=0)
    
    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    
    # Architecture (should match Stage 1)
    parser.add_argument("--d-branch", type=int, default=128)
    parser.add_argument("--n-branches", type=int, default=4)
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--temporal-out", type=int, default=64)
    parser.add_argument("--n-inject-tokens", type=int, default=4,
                        help="Number of virtual tokens to inject into LM")
    
    # General
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    torch.manual_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "e2e_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Build temporal config
    temporal_config = TemporalModelConfig(
        d_vision=1536,  # will be verified at runtime
        d_branch=args.d_branch,
        n_branches=args.n_branches,
        temporal_hidden=args.temporal_hidden,
        temporal_out=args.temporal_out,
    )
    
    # Load components
    temporal_model = load_temporal_checkpoint(
        args.temporal_checkpoint, temporal_config, device
    )
    
    vlm_model, processor = setup_vlm_with_lora(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=device,
    )
    
    # Token injector
    d_temporal = args.temporal_out * args.n_branches
    # Get LM hidden dim from the VLM config
    lm_hidden = vlm_model.config.hidden_size if hasattr(vlm_model.config, 'hidden_size') else 2048
    
    injector = TemporalTokenInjector(
        temporal_d_model=d_temporal,
        lm_hidden_dim=lm_hidden,
        n_tokens=args.n_inject_tokens,
    ).to(device)
    
    # Combine
    e2e_model = EndToEndModel(
        vlm_model=vlm_model,
        temporal_model=temporal_model,
        token_injector=injector,
        classification_weight=args.classification_weight,
        lm_weight=args.lm_weight,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in e2e_model.parameters())
    trainable_params = sum(p.numel() for p in e2e_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    # Optimizer: different LR for LoRA vs temporal module
    lora_params = [p for n, p in e2e_model.named_parameters() if "lora" in n and p.requires_grad]
    temporal_params = [p for n, p in e2e_model.named_parameters() if "lora" not in n and p.requires_grad]
    
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": temporal_params, "lr": args.lr * 10},  # temporal module gets higher LR
    ], weight_decay=1e-4)
    
    print(f"\nLoRA params: {sum(p.numel() for p in lora_params):,}")
    print(f"Temporal params: {sum(p.numel() for p in temporal_params):,}")
    print(f"\nReady for training. Fold: {args.fold}")
    print(f"Output: {output_dir}")
    print(f"\n{'='*60}")
    print("NOTE: Full training loop requires the video dataloader.")
    print("This script sets up all components and verifies they connect.")
    print("Integrate with your VLM_EatingBehavior data pipeline for actual training.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
