"""
Temporal Branch Processing for Visual Feature Streams.

Adapted from RF_CNN_Attention_v3.py (ParallelCNNAttention), which achieved
F1=0.731 on bite detection from IMU data. The architecture is nearly identical:
    - Parallel 1D dilated-CNN branches (one per spatial stream)
    - Coprime dilations [1, 2, 3] for artifact-free receptive fields
    - RoPE multi-head self-attention for cross-branch temporal fusion

Key adaptation: instead of 4 sensor modality channels (condyle_acc, condyle_gyr,
wrist_acc, wrist_gyr) with 3 channels each, we process 4 spatial streams from
the learned spatial decomposition, each with d_branch dimensions.

The temporal module learns the RHYTHM of eating behavior across frames:
bite → chew → pause → bite. Standard VLMs treat video as a bag of frames
and miss this sequential structure.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn


# =============================================================================
# RoPE (identical to sensor model — proven to work)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding for 1-D sequences.
    
    Encodes relative position by rotating query/key pairs.
    Identical to RF_CNN_Attention_v3.py implementation.
    """
    def __init__(self, dim: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._max_cached = 0
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        if seq_len <= self._max_cached:
            return
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
        self._max_cached = seq_len

    def forward(self, seq_len: int):
        self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), \
           (k * cos) + (_rotate_half(k) * sin)


# =============================================================================
# Temporal CNN Branch (adapted for visual features)
# =============================================================================

class TemporalCNNBranch(nn.Module):
    """1D dilated-CNN for temporal processing of a single spatial stream.
    
    Adapted from CNNBranch in RF_CNN_Attention_v3.py.
    Key difference: input is d_branch features per frame (not 3-channel IMU).
    
    Uses coprime dilations [1, 2, 3] with kernel_size=7 (same as sensor model).
    Why kernel=7 makes sense even with only 16 frames:
        - Dilated conv effective span: d*(k-1)+1
          → d=1,k=7: 7 frames (half the sequence)
          → d=2,k=7: 13 frames (most of the sequence)
          → d=3,k=7: 19 frames (full sequence + context via padding)
        - This lets each branch see full eating bouts (bite→chew→pause→bite)
          which typically span 5-10 frames at 1fps
    
    No pooling: 16 frames is too short to halve (would give RoPE attention only 8
    positions, losing temporal resolution for the rhythm we care about).
    """
    
    def __init__(
        self,
        in_channels: int,       # d_branch from spatial decomposition
        hidden_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 7,   # same as sensor model — dilations give full temporal RF
        dropout: float = 0.2,
    ):
        super().__init__()
        dilations = [1, 2, 3]
        
        self.conv1 = nn.Conv1d(
            in_channels, hidden_channels, kernel_size,
            padding=dilations[0] * (kernel_size // 2),
            dilation=dilations[0],
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size,
            padding=dilations[1] * (kernel_size // 2),
            dilation=dilations[1],
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = nn.Conv1d(
            hidden_channels, out_channels, kernel_size,
            padding=dilations[2] * (kernel_size // 2),
            dilation=dilations[2],
        )
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # No pooling: preserve all 16 temporal positions for RoPE attention.
        # Sensor model pooled because it had ~350 timesteps; we have 16.
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_frames, d_branch) — one spatial stream across time
        Returns:
            (B, num_frames, out_channels) — temporally processed features, full resolution
        """
        x = x.transpose(1, 2)  # (B, d_branch, T)
        x = torch.relu(self.bn1(self.conv1(x)))   # (B, hidden, T) — no pool
        x = torch.relu(self.bn2(self.conv2(x)))   # (B, hidden, T)
        x = torch.relu(self.bn3(self.conv3(x)))   # (B, out_ch, T)
        x = self.dropout(x)
        return x.transpose(1, 2)  # (B, T, out_channels)


# =============================================================================
# Multi-Head Self-Attention with RoPE (identical structure to sensor model)
# =============================================================================

class MultiHeadAttentionRoPE(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len=max_len)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(T)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn_drop = self.attn_dropout(attn)
        
        out = torch.matmul(attn_drop, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn
        return out


class TransformerBlockRoPE(nn.Module):
    """Pre-norm transformer block with RoPE attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.attn = MultiHeadAttentionRoPE(d_model, n_heads, dropout, max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# Full Temporal Module (the Variant B architecture)
# =============================================================================

class VisualTemporalAttention(nn.Module):
    """Parallel-branch temporal attention over visual spatial streams.
    
    This is the visual analog of ParallelCNNAttention from the sensor model.
    Instead of 4 IMU modalities, it processes 4 learned spatial streams.
    
    Architecture:
        4 spatial streams (from SpatialDecomposition)
            ├── TemporalCNNBranch 1 (stream: jaw-like)
            ├── TemporalCNNBranch 2 (stream: hand-like)
            ├── TemporalCNNBranch 3 (stream: food-like)
            └── TemporalCNNBranch 4 (stream: context-like)
                    ↓ concatenate
            RoPE Multi-Head Self-Attention (cross-stream temporal fusion)
                    ↓
            Temporal behavior representation (d_model)
    
    The output can be:
        (a) Used directly for classification (standalone temporal module)
        (b) Injected into the VLM's language model as conditioning tokens
    """
    
    def __init__(
        self,
        d_branch: int = 128,        # feature dim per spatial stream
        n_branches: int = 4,         # number of spatial streams
        temporal_hidden: int = 64,   # CNN hidden channels per branch
        temporal_out: int = 64,      # CNN output channels per branch
        kernel_size: int = 3,        # temporal conv kernel (smaller for 16 frames)
        cnn_dropout: float = 0.2,
        n_heads: int = 4,            # attention heads (same as sensor model)
        n_attn_layers: int = 2,      # attention layers (same as sensor model)
        attn_dropout: float = 0.2,
        mlp_hidden: int = 128,       # classifier hidden dim
        mlp_dropout: float = 0.3,
        num_classes: int = 2,        # needs_improvement vs good
    ):
        super().__init__()
        self.n_branches = n_branches
        
        # Parallel temporal CNN branches (one per spatial stream)
        self.temporal_branches = nn.ModuleList([
            TemporalCNNBranch(
                in_channels=d_branch,
                hidden_channels=temporal_hidden,
                out_channels=temporal_out,
                kernel_size=kernel_size,
                dropout=cnn_dropout,
            )
            for _ in range(n_branches)
        ])
        
        # Cross-branch temporal attention
        d_model = temporal_out * n_branches  # concat all branch outputs
        self.d_model = d_model
        
        self.attention_layers = nn.ModuleList([
            TransformerBlockRoPE(d_model, n_heads, attn_dropout, max_len=64)
            for _ in range(n_attn_layers)
        ])
        
        # Classification head — final layer has no bias so the model can't
        # learn the class prior through the bias term alone (prevents the
        # "predict majority class for everything" collapse in early epochs)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, num_classes, bias=False),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        spatial_streams: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Args:
            spatial_streams: (B, num_frames, n_branches, d_branch)
                Output from SpatialDecomposition
        
        Returns:
            logits: (B, num_classes) — classification logits
            extras: dict with 'features', 'attention' (optional)
        """
        B, T, N_br, D = spatial_streams.shape
        assert N_br == self.n_branches
        
        # Process each spatial stream through its temporal CNN branch
        branch_outputs = []
        for i, branch in enumerate(self.temporal_branches):
            stream_i = spatial_streams[:, :, i, :]  # (B, T, d_branch)
            temporal_feat = branch(stream_i)          # (B, T//2, temporal_out)
            branch_outputs.append(temporal_feat)
        
        # Concatenate across branches: (B, T//2, d_model)
        fused = torch.cat(branch_outputs, dim=-1)
        
        # Cross-branch temporal attention
        last_attn = None
        for layer in self.attention_layers:
            if return_attention:
                fused, attn = layer(fused, return_attention=True)
                last_attn = attn
            else:
                fused = layer(fused)
        
        # Global temporal pooling
        temporal_repr = fused.mean(dim=1)  # (B, d_model)
        
        # Classify
        logits = self.classifier(temporal_repr)
        
        if return_features or return_attention:
            extras = {}
            if return_features:
                extras['features'] = temporal_repr
            if return_attention:
                extras['attention'] = last_attn
            return logits, extras
        
        return logits
    
    def get_temporal_representation(self, spatial_streams: torch.Tensor) -> torch.Tensor:
        """Extract temporal behavior representation without classification.
        
        Used when injecting into the VLM's language model as conditioning.
        
        Returns:
            (B, d_model) — temporal behavior representation
        """
        B, T, N_br, D = spatial_streams.shape
        
        branch_outputs = []
        for i, branch in enumerate(self.temporal_branches):
            stream_i = spatial_streams[:, :, i, :]
            branch_outputs.append(branch(stream_i))
        
        fused = torch.cat(branch_outputs, dim=-1)
        for layer in self.attention_layers:
            fused = layer(fused)
        
        return fused.mean(dim=1)  # (B, d_model)
