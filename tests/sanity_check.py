#!/usr/bin/env python3
"""
Sanity check: verify all modules work end-to-end with random data.

Run this LOCALLY (no GPU needed) before submitting SLURM jobs.

    python tests/sanity_check.py

Expected output: all shape checks pass, loss computes, backprop works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.models.spatial_decomposition import SpatialDecomposition
from src.models.temporal_branches import VisualTemporalAttention
from src.models.vlm_temporal_model import TemporalBehaviorModel, TemporalModelConfig


def test_spatial_decomposition():
    print("Testing SpatialDecomposition...")
    
    B, T, N, D = 4, 16, 196, 1536  # batch, frames, patches, d_vision
    x = torch.randn(B, T, N, D)
    
    model = SpatialDecomposition(
        d_vision=D, d_branch=128, n_branches=4,
        n_heads_per_branch=4, diversity_weight=0.1,
    )
    
    streams, div_loss = model(x)
    assert streams.shape == (B, T, 4, 128), f"Expected (4,16,4,128), got {streams.shape}"
    assert div_loss.ndim == 0, "Diversity loss should be scalar"
    
    # With attention maps
    streams, div_loss, attn = model(x, return_attention=True)
    assert attn.shape[0] == B
    assert attn.shape[1] == T
    assert attn.shape[2] == 4  # n_branches
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK — streams: {streams.shape}, div_loss: {div_loss.item():.4f}")
    print(f"  Parameters: {n_params:,}")


def test_temporal_attention():
    print("\nTesting VisualTemporalAttention...")
    
    B, T, N_br, D_br = 4, 16, 4, 128
    x = torch.randn(B, T, N_br, D_br)
    
    model = VisualTemporalAttention(
        d_branch=D_br, n_branches=N_br,
        temporal_hidden=64, temporal_out=64,
        kernel_size=3, n_heads=4, n_attn_layers=2,
        mlp_hidden=128, num_classes=2,
    )
    
    logits = model(x)
    assert logits.shape == (B, 2), f"Expected (4,2), got {logits.shape}"
    
    # With features
    logits, extras = model(x, return_features=True, return_attention=True)
    assert extras['features'].shape == (B, 64 * 4), f"Got {extras['features'].shape}"
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK — logits: {logits.shape}, features: {extras['features'].shape}")
    print(f"  Parameters: {n_params:,}")


def test_full_model():
    print("\nTesting TemporalBehaviorModel (full pipeline)...")
    
    B, T, N, D = 4, 16, 196, 1536
    x = torch.randn(B, T, N, D)
    labels = torch.randint(0, 2, (B,))
    
    config = TemporalModelConfig(
        d_vision=D, d_branch=128, n_branches=4,
        temporal_hidden=64, temporal_out=64,
        temporal_kernel=3, n_heads_temporal=4,
        n_attn_layers=2, mlp_hidden=128, num_classes=2,
        diversity_weight=0.1,
    )
    
    model = TemporalBehaviorModel(config)
    
    # Forward with loss
    output = model(x, labels=labels)
    assert 'logits' in output
    assert 'loss' in output
    assert 'ce_loss' in output
    assert 'diversity_loss' in output
    assert output['logits'].shape == (B, 2)
    
    # Backward
    output['loss'].backward()
    grad_norms = {name: p.grad.norm().item() 
                  for name, p in model.named_parameters() 
                  if p.grad is not None}
    assert len(grad_norms) > 0, "No gradients computed!"
    
    # Temporal representation extraction
    model.zero_grad()
    repr = model.get_temporal_representation(x)
    assert repr.shape == (B, 64 * 4)
    
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  OK — loss: {output['loss'].item():.4f}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Temporal repr: {repr.shape}")
    print(f"  Total params: {n_params:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Gradient-receiving modules: {len(grad_norms)}")


def test_memory_estimate():
    """Rough estimate of GPU memory for training."""
    print("\nMemory estimate...")
    
    # Each cached feature sample: 16 frames * 196 patches * 1536 dim * 2 bytes (fp16)
    sample_bytes = 16 * 196 * 1536 * 2
    sample_mb = sample_bytes / 1024 / 1024
    
    # Batch of 32
    batch_mb = sample_mb * 32
    
    # Model params (rough)
    config = TemporalModelConfig(d_vision=1536)
    model = TemporalBehaviorModel(config)
    param_bytes = sum(p.numel() * 4 for p in model.parameters())  # fp32 training
    param_mb = param_bytes / 1024 / 1024
    
    print(f"  Single sample (fp16): {sample_mb:.1f} MB")
    print(f"  Batch of 32 (fp16→fp32): {batch_mb * 2:.1f} MB")
    print(f"  Model params (fp32): {param_mb:.1f} MB")
    print(f"  Estimated total (with activations ~3x): {(batch_mb * 2 + param_mb) * 3:.0f} MB")
    print(f"  → Should fit on L40S 48GB easily ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("VLM_TemporalBranch — Architecture Sanity Check")
    print("=" * 60)
    
    test_spatial_decomposition()
    test_temporal_attention()
    test_full_model()
    test_memory_estimate()
    
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED ✓")
    print("=" * 60)
