"""
Learned Spatial Decomposition for Video Frame Features.

Instead of hard-coded ROIs (jaw, hand, food, context), we use learnable
query vectors that soft-attend over the spatial patch tokens of each frame.
Each query learns to specialize on a different spatial region — the model
discovers which regions matter for eating behavior assessment.

This is the visual analog of having 4 sensor modalities (condyle_acc,
condyle_gyr, wrist_acc, wrist_gyr) in the IMU pipeline.

Input:  (B, num_frames, num_patches, d_vision)  — per-frame patch tokens from VLM
Output: (B, num_frames, num_branches, d_branch)  — per-frame spatial stream features
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SpatialBranchAttention(nn.Module):
    """Single spatial branch: a learnable query that attends over patches.
    
    Each branch learns a query vector that soft-attends over the spatial
    patch tokens of a frame, producing a single feature vector per frame.
    Different branches learn to focus on different spatial regions.
    """
    
    def __init__(self, d_vision: int, d_branch: int, n_heads: int = 4):
        super().__init__()
        self.d_vision = d_vision
        self.d_branch = d_branch
        self.n_heads = n_heads
        assert d_branch % n_heads == 0
        self.head_dim = d_branch // n_heads
        
        # Learnable query for this branch (what spatial region to attend to)
        self.query = nn.Parameter(torch.randn(1, 1, n_heads, self.head_dim) * 0.02)
        
        # Project vision features to key/value space
        self.k_proj = nn.Linear(d_vision, d_branch)
        self.v_proj = nn.Linear(d_vision, d_branch)
        self.out_proj = nn.Linear(d_branch, d_branch)
        
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        patch_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, num_patches, d_vision) — patches from ONE frame
            return_attention: if True, also return per-head attention weights

        Returns:
            branch_feature: (B, d_branch)
            attn_avg: (B, num_patches) — head-averaged pre-dropout attention (for diversity loss)
            attn_per_head: (B, n_heads, num_patches) — only when return_attention=True
        """
        B, N, _ = patch_tokens.shape

        k = self.k_proj(patch_tokens).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(patch_tokens).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.query.expand(B, -1, -1, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn, dim=-1)

        # Pre-dropout, head-averaged attention for diversity regularization
        attn_avg = attn_weights.mean(dim=1).squeeze(1)  # (B, N)

        out = torch.matmul(self.attn_dropout(attn_weights), v)
        out = out.squeeze(2).reshape(B, -1)  # (B, d_branch)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_avg, attn_weights.squeeze(2)
        return out, attn_avg


class SpatialDecomposition(nn.Module):
    """Decompose frame patch tokens into N spatial streams.
    
    Uses N learnable branch queries to soft-attend over spatial patches,
    producing N feature vectors per frame. Each branch discovers a different
    spatial region of interest (ideally: jaw, hand, food, context).
    
    Includes diversity regularization to encourage branches to attend to
    different spatial locations.
    """
    
    def __init__(
        self,
        d_vision: int,
        d_branch: int,
        n_branches: int = 4,
        n_heads_per_branch: int = 4,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.n_branches = n_branches
        self.d_branch = d_branch
        self.diversity_weight = diversity_weight
        
        self.branches = nn.ModuleList([
            SpatialBranchAttention(d_vision, d_branch, n_heads_per_branch)
            for _ in range(n_branches)
        ])
        
        # Optional: shared layer norm on input patches
        self.input_norm = nn.LayerNorm(d_vision)
    
    def forward(
        self,
        frame_patches: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_patches: (B, num_frames, num_patches, d_vision)
            
        Returns:
            streams: (B, num_frames, n_branches, d_branch)
            diversity_loss: scalar tensor (for training regularization)
            attn_maps: (B, num_frames, n_branches, n_heads, num_patches) — optional
        """
        B, T, N, D = frame_patches.shape

        patches_flat = frame_patches.reshape(B * T, N, D)
        patches_flat = self.input_norm(patches_flat)

        branch_outputs = []
        branch_attn_dists = []
        attn_maps = [] if return_attention else None

        for branch in self.branches:
            if return_attention:
                feat, attn_avg, attn_per_head = branch(patches_flat, return_attention=True)
                attn_maps.append(attn_per_head)
            else:
                feat, attn_avg = branch(patches_flat)
            branch_outputs.append(feat)
            branch_attn_dists.append(attn_avg)

        streams = torch.stack(branch_outputs, dim=1)
        streams = streams.reshape(B, T, self.n_branches, self.d_branch)

        diversity_loss = self._compute_diversity_loss(branch_attn_dists)

        if return_attention:
            attn_stack = torch.stack(attn_maps, dim=1)
            attn_stack = attn_stack.reshape(B, T, self.n_branches, -1, N)
            return streams, diversity_loss, attn_stack

        return streams, diversity_loss
    
    def _compute_diversity_loss(
        self, branch_attn_dists: list[torch.Tensor],
    ) -> torch.Tensor:
        """Encourage branches to attend to different spatial regions.

        Uses the pre-computed (pre-dropout, head-averaged) attention
        distributions from each branch's forward pass — no redundant
        recomputation of k_proj / attention.
        """
        if self.diversity_weight == 0.0:
            return torch.tensor(0.0, device=branch_attn_dists[0].device)

        attn_stack = torch.stack(branch_attn_dists, dim=0)  # (n_branches, BT, N)

        loss = torch.tensor(0.0, device=attn_stack.device)
        count = 0
        for i in range(self.n_branches):
            for j in range(i + 1, self.n_branches):
                cos_sim = F.cosine_similarity(attn_stack[i], attn_stack[j], dim=-1)
                loss = loss + cos_sim.mean()
                count += 1

        if count > 0:
            loss = loss / count

        return self.diversity_weight * loss
