"""
Dataset for cached visual features with participant-level fold splits.

Loads pre-extracted .pt files (from extract_features.py) and provides
train/val splits based on participant-level 5-fold CV — same protocol
as the IMWUT paper to ensure comparable results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _norm_pid(x) -> str:
    """JSON manifests may use int or str for participant_id; folds must match."""
    if x is None:
        return "unknown"
    return str(x).strip()


def _load_feature_pt(path: Path) -> Dict:
    """Load a cache .pt file saved by extract_features.py.

    Files contain tensors plus Python ints and a metadata dict; they are not
    ``weights_only``-safe. PyTorch 2.4+ rejects those under ``weights_only=True``,
    which previously caused Stage 1 to exit immediately with code 1.
    """
    p = str(path)
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only kwarg
        return torch.load(p, map_location="cpu")


class CachedFeatureDataset(Dataset):
    """Dataset that loads cached VLM visual features from disk.
    
    Each sample is a .pt file containing:
        'patches': (num_frames, num_patches, d_vision)
        'label': int
        'metadata': dict with participant_id, food_type, fold, etc.
    """
    
    def __init__(
        self,
        manifest_path: str,
        feature_dir: str,
        fold_indices: Optional[List[int]] = None,
        participant_ids: Optional[List[str]] = None,
        max_frames: int = 16,
    ):
        """
        Args:
            manifest_path: path to manifest.json from feature extraction
            feature_dir: directory containing .pt files
            fold_indices: if provided, only include segments from these folds
            participant_ids: if provided, only include these participants
            max_frames: pad/truncate to this many frames
        """
        self.feature_dir = Path(feature_dir)
        self.max_frames = max_frames
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Filter by fold or participant
        self.entries = []
        want_pids = {_norm_pid(p) for p in participant_ids} if participant_ids is not None else None
        for entry in manifest:
            if fold_indices is not None:
                if entry.get("fold") not in fold_indices:
                    continue
            if want_pids is not None:
                if _norm_pid(entry.get("participant_id")) not in want_pids:
                    continue
            self.entries.append(entry)
        
        # Cache labels for quick access
        self.labels = [e["label"] for e in self.entries]
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        cached_path = entry["cached_path"]
        
        data = _load_feature_pt(Path(cached_path))
        patches = data["patches"].float()  # (T, N_patches, d_vision)
        label = data["label"]
        
        T, N, D = patches.shape
        
        # Pad or truncate to max_frames
        if T < self.max_frames:
            padding = torch.zeros(self.max_frames - T, N, D)
            patches = torch.cat([patches, padding], dim=0)
        elif T > self.max_frames:
            # Uniform sample
            indices = torch.linspace(0, T - 1, self.max_frames).long()
            patches = patches[indices]
        
        return {
            "patches": patches,                           # (max_frames, N_patches, d_vision)
            "label": torch.tensor(label, dtype=torch.long),
            "participant_id": entry.get("participant_id", "unknown"),
            "food_type": entry.get("food_type", "unknown"),
            "segment_id": entry.get("segment_id", f"idx_{idx}"),
        }


def get_fold_split(
    manifest_path: str,
    feature_dir: str,
    fold_id: int,
    n_folds: int = 5,
    max_frames: int = 16,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train/val dataloaders for a specific fold.
    
    Uses pre-assigned fold IDs from the manifest (set by generate_metadata.py)
    for consistency with the IMWUT paper's participant-level CV splits.
    Falls back to random assignment if fold field is missing.
    
    Returns:
        train_loader, val_loader, fold_info dict
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Use pre-assigned folds from manifest if available
    has_fold_field = all("fold" in e for e in manifest)
    
    if has_fold_field:
        # Use existing fold assignments (from generate_metadata.py)
        fold_assignments = {}
        for e in manifest:
            if "participant_id" not in e:
                continue
            fold_assignments[_norm_pid(e["participant_id"])] = e["fold"]
    else:
        # Fallback: assign folds by shuffling participants
        participants = sorted(set(_norm_pid(e.get("participant_id")) for e in manifest))
        np.random.seed(42)
        np.random.shuffle(participants)
        fold_assignments = {}
        fold_size = len(participants) // n_folds
        for i, pid in enumerate(participants):
            fold_assignments[pid] = i // fold_size if i // fold_size < n_folds else n_folds - 1
    
    # Split
    val_pids = sorted(set(pid for pid, f in fold_assignments.items() if f == fold_id))
    train_pids = sorted(set(pid for pid, f in fold_assignments.items() if f != fold_id))
    
    train_ds = CachedFeatureDataset(
        manifest_path, feature_dir,
        participant_ids=train_pids,
        max_frames=max_frames,
    )
    val_ds = CachedFeatureDataset(
        manifest_path, feature_dir,
        participant_ids=val_pids,
        max_frames=max_frames,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_fn,
    )
    
    if len(train_ds) == 0:
        raise ValueError(
            f"No training samples for fold {fold_id}. "
            "Check manifest has 'participant_id' and 'fold' fields, and that this fold has val participants "
            "while others are assigned to train."
        )
    if len(val_ds) == 0:
        raise ValueError(
            f"No validation samples for fold {fold_id}. Check manifest fold assignments."
        )
    
    fold_info = {
        "fold_id": fold_id,
        "train_participants": train_pids,
        "val_participants": val_pids,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    
    return train_loader, val_loader, fold_info


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate to handle mixed tensor/string fields."""
    # Each sample's patches are shaped (T, N_patches, D_vision).
    # N_patches varies across segments because input frame resolutions differ.
    # We pad to the maximum N_patches inside the batch so we can stack.
    patches_list = [b["patches"] for b in batch]
    labels_list = [b["label"] for b in batch]

    B = len(batch)
    T = patches_list[0].shape[0]
    D = patches_list[0].shape[2]
    max_n = max(p.shape[1] for p in patches_list)

    padded = patches_list[0].new_zeros((B, T, max_n, D))
    for i, p in enumerate(patches_list):
        t_i, n_i, d_i = p.shape
        if d_i != D:
            raise ValueError(f"Inconsistent d_vision in batch: got {d_i} vs {D}")
        padded[i, :t_i, :n_i, :] = p

    return {
        "patches": padded,  # (B, T, max_n, D)
        "label": torch.stack(labels_list),
        "participant_id": [b["participant_id"] for b in batch],
        "food_type": [b["food_type"] for b in batch],
        "segment_id": [b["segment_id"] for b in batch],
    }
