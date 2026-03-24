"""
VLM + Temporal Module: The Full Variant B Architecture.

Wires together:
    1. Qwen2.5-VL vision encoder → per-frame patch tokens
    2. SpatialDecomposition → 4 spatial streams per frame
    3. VisualTemporalAttention → temporal behavior representation
    4. Two output modes:
       (a) Classification-only: temporal repr → classifier → label
       (b) VLM-integrated: temporal repr → special tokens → LM → label + feedback

Training stages:
    Stage 1: Train spatial decomposition + temporal module on cached VLM features
             (fast iteration, no VLM forward pass needed)
    Stage 2: End-to-end LoRA fine-tuning with temporal module
             (joint optimization, LoRA on VLM + temporal module)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TemporalModelConfig:
    """Configuration for the full architecture."""
    
    # Vision encoder (Qwen2.5-VL)
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    d_vision: int = 1536          # Qwen2.5-VL-3B hidden dim (verify from model config)
    num_frames: int = 16
    
    # Spatial decomposition
    d_branch: int = 32            # feature dim per spatial stream
    n_branches: int = 4           # number of spatial streams
    n_heads_spatial: int = 4      # attention heads per branch query
    diversity_weight: float = 0.05 # branch specialization loss weight
    
    # Temporal processing
    temporal_hidden: int = 16
    temporal_out: int = 16
    temporal_kernel: int = 7   # kernel=7 with dilations [1,2,3] covers full 16-frame RF
    cnn_dropout: float = 0.2
    n_heads_temporal: int = 1
    n_attn_layers: int = 1
    attn_dropout: float = 0.2
    
    # Classifier
    mlp_hidden: int = 64
    mlp_dropout: float = 0.3
    num_classes: int = 2
    
    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    num_epochs: int = 20
    batch_size: int = 8
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0     # focal loss gamma (0 = standard CE)
    
    # LoRA (for stage 2 end-to-end)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class VLMFeatureExtractor(nn.Module):
    """Wrapper to extract per-frame patch tokens from Qwen2.5-VL.
    
    In Stage 1, this is used ONCE to cache features to disk.
    In Stage 2, this runs as part of the forward pass with LoRA.
    """
    
    def __init__(self, config: TemporalModelConfig):
        super().__init__()
        self.config = config
        self.model = None  # Lazy-loaded to save memory
    
    def load_model(self):
        """Load the VLM. Call this explicitly — not in __init__ to allow
        the temporal module to be instantiated without the full VLM."""
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.vlm_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # flash attention can cause issues
        )
        self.model.eval()
        # Freeze everything — we only want features
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def extract_frame_patches(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-frame patch tokens from the vision encoder.
        
        Args:
            pixel_values: preprocessed video frames from Qwen2.5-VL processor
            image_grid_thw: temporal/height/width grid info
            
        Returns:
            (B, num_frames, num_patches_per_frame, d_vision)
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first")
        
        # Get vision encoder output
        # Qwen2.5-VL's visual module outputs patch embeddings
        vision_output = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        # vision_output shape: (total_patches, d_vision)
        
        # Reshape to per-frame: need to know patches per frame from grid_thw
        # grid_thw: (num_videos, 3) where columns are [temporal, height, width]
        # For a single video: temporal=num_frames, h_patches, w_patches
        # patches_per_frame = h_patches * w_patches
        t, h, w = image_grid_thw[0].tolist()
        patches_per_frame = int(h * w)
        total_frames = int(t)
        
        # Reshape: (total_patches,  d_vision) -> (1, total_frames, patches_per_frame, d_vision)
        frame_patches = vision_output.reshape(1, total_frames, patches_per_frame, -1)
        
        return frame_patches


class TemporalBehaviorModel(nn.Module):
    """The full Variant B model: spatial decomposition + temporal attention.
    
    Operates on pre-extracted visual features (Stage 1) or can be connected
    to the VLM for end-to-end training (Stage 2).
    """
    
    def __init__(self, config: TemporalModelConfig):
        super().__init__()
        self.config = config
        
        # Import here to avoid circular deps
        from .spatial_decomposition import SpatialDecomposition
        from .temporal_branches import VisualTemporalAttention
        
        self.spatial = SpatialDecomposition(
            d_vision=config.d_vision,
            d_branch=config.d_branch,
            n_branches=config.n_branches,
            n_heads_per_branch=config.n_heads_spatial,
            diversity_weight=config.diversity_weight,
        )
        
        self.temporal = VisualTemporalAttention(
            d_branch=config.d_branch,
            n_branches=config.n_branches,
            temporal_hidden=config.temporal_hidden,
            temporal_out=config.temporal_out,
            kernel_size=config.temporal_kernel,
            cnn_dropout=config.cnn_dropout,
            n_heads=config.n_heads_temporal,
            n_attn_layers=config.n_attn_layers,
            attn_dropout=config.attn_dropout,
            mlp_hidden=config.mlp_hidden,
            mlp_dropout=config.mlp_dropout,
            num_classes=config.num_classes,
        )
    
    def forward(
        self,
        frame_patches: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        class_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frame_patches: (B, num_frames, num_patches, d_vision)
                Either cached features or live VLM output
            labels: (B,) — optional, for computing loss
            
        Returns:
            dict with 'logits', 'loss' (if labels given), 'diversity_loss',
            and optionally 'spatial_attn', 'temporal_attn'
        """
        # Step 1: Spatial decomposition → 4 streams per frame
        if return_attention:
            streams, div_loss, spatial_attn = self.spatial(
                frame_patches, return_attention=True
            )
        else:
            streams, div_loss = self.spatial(frame_patches)
        # streams: (B, num_frames, n_branches, d_branch)
        
        # Step 2: Temporal processing → classification
        if return_attention:
            logits, extras = self.temporal(
                streams, return_attention=True
            )
        else:
            logits = self.temporal(streams)
        
        # Build output
        output = {
            'logits': logits,
            'diversity_loss': div_loss,
        }
        
        if labels is not None:
            ce_loss = self._focal_loss(logits, labels, class_weight)
            output['ce_loss'] = ce_loss
            output['loss'] = ce_loss + div_loss
        
        if return_attention:
            output['spatial_attn'] = spatial_attn
            output['temporal_attn'] = extras.get('attention')
        
        return output
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Focal loss: (1 - p_t)^gamma * CE.

        When gamma=0 this is standard CE. When gamma>0, confident-and-correct
        predictions (the "predict majority class" shortcut) are downweighted,
        forcing the model to pay attention to the minority class from epoch 0.
        """
        gamma = self.config.focal_gamma
        smoothing = self.config.label_smoothing

        ce = F.cross_entropy(
            logits, labels,
            weight=class_weight,
            label_smoothing=smoothing,
            reduction="none",
        )
        if gamma == 0.0:
            return ce.mean()

        with torch.no_grad():
            p_t = torch.exp(-ce)  # probability of the true class
        modulator = (1.0 - p_t) ** gamma
        return (modulator * ce).mean()

    def predict(self, frame_patches: torch.Tensor) -> torch.Tensor:
        """Simple prediction interface.
        
        Returns:
            (B,) — predicted class indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(frame_patches)
            return output['logits'].argmax(dim=-1)
    
    def get_temporal_representation(self, frame_patches: torch.Tensor) -> torch.Tensor:
        """Extract temporal behavior representation for VLM integration.
        
        Returns:
            (B, d_model) — can be projected and injected into LM
        """
        streams, _ = self.spatial(frame_patches)
        return self.temporal.get_temporal_representation(streams)


class TemporalTokenInjector(nn.Module):
    """Projects temporal representation into VLM token space for LM conditioning.
    
    Used in Stage 2 (end-to-end) to inject temporal behavior understanding
    into the language model's generation of classification + feedback.
    
    The temporal representation is projected to match the LM's hidden dim
    and prepended as special tokens before the text instruction.
    """
    
    def __init__(
        self,
        temporal_d_model: int,   # d_model from VisualTemporalAttention
        lm_hidden_dim: int,      # Qwen2.5-VL's LM hidden dimension
        n_tokens: int = 4,       # number of virtual tokens to inject
    ):
        super().__init__()
        self.n_tokens = n_tokens
        
        # Project temporal repr to n_tokens * lm_hidden_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(temporal_d_model),
            nn.Linear(temporal_d_model, lm_hidden_dim * n_tokens),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.lm_hidden_dim = lm_hidden_dim
    
    def forward(self, temporal_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_repr: (B, temporal_d_model)
        Returns:
            virtual_tokens: (B, n_tokens, lm_hidden_dim)
                Ready to be prepended to the LM's input embeddings
        """
        projected = self.projector(temporal_repr)  # (B, n_tokens * lm_dim)
        return projected.reshape(-1, self.n_tokens, self.lm_hidden_dim)
