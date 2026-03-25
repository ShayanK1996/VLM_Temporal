"""Video dataset for Stage 2 end-to-end training.

Loads raw video segments, processes them through the Qwen2.5-VL processor,
and returns pixel_values + image_grid_thw for the VLM visual encoder.

Reuses the same manifest.json and participant-level fold splits from Stage 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _read_video_pyav(path: str, num_frames: int) -> list:
    """Read uniformly-sampled frames from a video using PyAV."""
    import av

    container = av.open(path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(video=0))
        container.close()
        container = av.open(path)

    if total_frames == 0:
        container.close()
        raise ValueError(f"No video frames found in {path}")

    indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_image())
        if len(frames) == num_frames:
            break
    container.close()

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames


class VideoSegmentDataset(Dataset):
    """Dataset that loads raw video segments for end-to-end VLM training.

    Each sample is processed through the Qwen2.5-VL processor and returns
    pixel_values + image_grid_thw ready for the visual encoder.
    Batch size must be 1 (variable-size pixel_values across videos).
    """

    def __init__(
        self,
        entries: List[Dict],
        processor,
        num_frames: int = 16,
        video_root: Optional[str] = None,
    ):
        self.entries = entries
        self.processor = processor
        self.num_frames = num_frames
        self.video_root = Path(video_root) if video_root else None
        self.labels = [e["label"] for e in entries]

    def __len__(self):
        return len(self.entries)

    def _resolve_video_path(self, entry: Dict) -> str:
        vpath = entry["video_path"]
        if Path(vpath).exists():
            return vpath
        if self.video_root is not None:
            candidate = self.video_root / Path(vpath).name
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"Video not found: {vpath}"
            + (f" (also tried {self.video_root})" if self.video_root else "")
        )

    def __getitem__(self, idx):
        entry = self.entries[idx]
        video_path = self._resolve_video_path(entry)
        label = entry["label"]

        frames = _read_video_pyav(video_path, self.num_frames)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": self.num_frames},
                    {"type": "text", "text": "Analyze eating behavior."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=None, videos=[frames],
            return_tensors="pt", padding=True,
        )

        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None:
            pixel_values = inputs.get("pixel_values_videos")
            image_grid_thw = inputs.get("video_grid_thw")

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "label": torch.tensor(label, dtype=torch.long),
            "food_type": entry.get("food_type", "unknown"),
        }


def _identity_collate(batch):
    """For bs=1: return the single sample without additional stacking."""
    return batch[0]


def get_e2e_fold_split(
    manifest_path: str,
    fold_id: int,
    processor,
    num_frames: int = 16,
    video_root: Optional[str] = None,
    num_workers: int = 0,
):
    """Create train/val dataloaders for a fold using raw video segments.

    Returns the same fold splits as Stage 1 (participant-level, from manifest).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    train_entries = [e for e in manifest if e.get("fold") != fold_id]
    val_entries = [e for e in manifest if e.get("fold") == fold_id]

    train_ds = VideoSegmentDataset(train_entries, processor, num_frames, video_root)
    val_ds = VideoSegmentDataset(val_entries, processor, num_frames, video_root)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=num_workers, pin_memory=False,
        collate_fn=_identity_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=False,
        collate_fn=_identity_collate,
    )

    train_labels = [e["label"] for e in train_entries]
    val_labels = [e["label"] for e in val_entries]
    train_pids = set(e.get("participant_id", "") for e in train_entries)
    val_pids = set(e.get("participant_id", "") for e in val_entries)
    n_classes = max(max(train_labels), max(val_labels)) + 1
    train_class_counts = [train_labels.count(c) for c in range(n_classes)]
    val_class_counts = [val_labels.count(c) for c in range(n_classes)]
    train_imbalance_ratio = max(train_class_counts) / max(1, min(train_class_counts))

    fold_info = {
        "train_size": len(train_entries),
        "val_size": len(val_entries),
        "train_participants": sorted(train_pids),
        "val_participants": sorted(val_pids),
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
        "train_imbalance_ratio": train_imbalance_ratio,
    }
    return train_loader, val_loader, fold_info
