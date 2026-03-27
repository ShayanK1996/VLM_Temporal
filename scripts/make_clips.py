#!/usr/bin/env python3
"""
Generate g3 and g5 video clips from raw participant videos + bite annotation CSVs.

g3 clips: 3 consecutive bites (sliding window, step=1)
g5 clips: 5 consecutive bites (sliding window, step=1)

Clip boundary:
  start = bite_window[0] start - PRE_ROLL_SEC
  end   = bite_window[-1+1] start (next bite after window), so the clip includes
          the chewing / pause period following the last bite in the window.
          If no next bite exists (end of food type), end = last_bite_end + POST_ROLL_SEC.

Clips are kept within a single food type. Bites beyond the video duration are skipped.

Usage:
    python scripts/make_clips.py \
        --csv-dir  data/participant_csvs/ \
        --video-dir data/participant_videos/ \
        --out-dir  data/processed/training_clips/ \
        [--participant 111001]   # single participant for testing
        [--segment-types g3 g5] # default: both
        [--pre-roll 2.0]         # seconds before first bite start
        [--post-roll 15.0]       # seconds after last bite end when no next bite
        [--dry-run]              # print commands, don't run ffmpeg
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

FOOD_TYPE_MAP = {
    1: "chips_and_salsa",
    2: "carrots",
    3: "rice_and_beans",
    4: "churros",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def ts_to_sec(ts: str) -> float:
    """Convert 'HH:MM:SS.mmm' CSV timestamp to seconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def extract_bites(csv_path: Path) -> pd.DataFrame:
    """
    Return a DataFrame with one row per bite event:
        bite_idx, food_type, start_sec, end_sec
    Rows are sorted by start_sec.
    """
    df = pd.read_csv(csv_path)
    bite_col = df["Bite"].values
    transitions = np.diff(bite_col, prepend=0, append=0)
    starts_idx = np.where(transitions == 1)[0]
    ends_idx   = np.where(transitions == -1)[0]

    rows = []
    for i, (si, ei) in enumerate(zip(starts_idx, ends_idx)):
        rows.append({
            "bite_idx":  i,
            "food_type": int(df["Food Type"].iloc[si]),
            "start_sec": ts_to_sec(df["Formatted Timestamp"].iloc[si]),
            "end_sec":   ts_to_sec(df["Formatted Timestamp"].iloc[ei - 1]),
        })
    return pd.DataFrame(rows)


def ffmpeg_cut(
    video_path: Path,
    out_path: Path,
    start: float,
    end: float,
    dry_run: bool = False,
) -> bool:
    """Cut [start, end] seconds from video_path into out_path. Returns True on success."""
    duration = max(end - start, 0.1)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t",  f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "18",
        "-preset", "fast",
        "-an",         # drop audio (profile-view video only)
        str(out_path),
    ]
    if dry_run:
        print("  [dry-run]", " ".join(cmd))
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ffmpeg ERROR] {out_path.name}")
        print(result.stderr[-500:])
        return False
    return True


# ── core logic ────────────────────────────────────────────────────────────────

def make_group_clips(
    pid: str,
    csv_path: Path,
    video_path: Path,
    out_dir: Path,
    window: int,
    pre_roll: float,
    post_roll: float,
    dry_run: bool,
    video_duration: float,
) -> list[dict]:
    """
    Slide a window of `window` consecutive same-food-type bites across each
    participant's bite list. Yield one clip per window position.

    Returns list of metadata dicts for the manifest CSV.
    """
    bites = extract_bites(csv_path)
    # Drop bites that start beyond the video
    bites = bites[bites["start_sec"] < video_duration].reset_index(drop=True)

    seg_label = f"g{window}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    clip_global_idx = 0

    for food_type, group_df in bites.groupby("food_type", sort=True):
        if food_type == 0:
            continue  # non-eating rows
        group_df = group_df.reset_index(drop=True)
        n = len(group_df)
        if n < window:
            continue  # not enough bites for this window size

        food_name = FOOD_TYPE_MAP.get(food_type, f"food{food_type}")

        for w_start in range(n - window + 1):
            w_bites = group_df.iloc[w_start : w_start + window]

            clip_start = max(0.0, w_bites.iloc[0]["start_sec"] - pre_roll)

            # Clip end: start of next bite after window, or last_bite_end + post_roll
            next_idx = w_start + window
            if next_idx < n:
                clip_end = group_df.iloc[next_idx]["start_sec"]
            else:
                clip_end = min(w_bites.iloc[-1]["end_sec"] + post_roll, video_duration)

            clip_end = min(clip_end, video_duration)
            if clip_end <= clip_start:
                continue

            out_name = f"{pid}_{seg_label}_{clip_global_idx:03d}.mp4"
            out_path = out_dir / out_name

            success = ffmpeg_cut(video_path, out_path, clip_start, clip_end, dry_run)
            if success:
                records.append({
                    "clip_path":     str(out_path),
                    "participant_id": pid,
                    "segment_type":  seg_label,
                    "food_type":     food_name,
                    "food_type_code": food_type,
                    "clip_start_sec": f"{clip_start:.3f}",
                    "clip_end_sec":   f"{clip_end:.3f}",
                    "duration_sec":   f"{clip_end - clip_start:.3f}",
                    "window_bite_start_idx": w_start,
                    "n_bites_in_window": window,
                    "bite_starts": ";".join(f"{b['start_sec']:.3f}" for _, b in w_bites.iterrows()),
                })
                clip_global_idx += 1

    return records


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate g3/g5 video clips")
    parser.add_argument("--csv-dir",   type=Path,
                        default=Path("data/participant_csvs"))
    parser.add_argument("--video-dir", type=Path,
                        default=Path("data/participant_videos"))
    parser.add_argument("--out-dir",   type=Path,
                        default=Path("data/processed/training_clips"))
    parser.add_argument("--participant", type=str, default=None,
                        help="Process only this participant ID (e.g. 111001)")
    parser.add_argument("--segment-types", nargs="+",
                        choices=["g3", "g5"], default=["g3", "g5"])
    parser.add_argument("--pre-roll",  type=float, default=2.0,
                        help="Seconds of video before first bite in window")
    parser.add_argument("--post-roll", type=float, default=15.0,
                        help="Seconds after last bite end when no next bite exists")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print ffmpeg commands but don't execute")
    parser.add_argument("--manifest",  type=Path,
                        default=Path("data/processed/clips_manifest.csv"),
                        help="Output manifest CSV path")
    args = parser.parse_args()

    # Collect participant IDs
    csv_files = sorted(args.csv_dir.glob("*_final.csv"))
    if args.participant:
        csv_files = [f for f in csv_files if f.stem.startswith(args.participant)]
        if not csv_files:
            sys.exit(f"No CSV found for participant: {args.participant}")

    # Find video extensions (.mov or .MP4 or .mp4)
    video_ext_map: dict[str, Path] = {}
    for ext in ("*.mov", "*.MP4", "*.mp4", "*.MOV"):
        for vf in args.video_dir.glob(ext):
            pid = vf.stem
            video_ext_map[pid] = vf

    all_records: list[dict] = []
    window_sizes = {"g3": 3, "g5": 5}

    for csv_path in csv_files:
        pid = csv_path.stem.replace("_final", "")
        video_path = video_ext_map.get(pid)
        if video_path is None:
            print(f"[SKIP] {pid}: no video found")
            continue

        video_duration = get_video_duration(video_path)
        print(f"\n{'='*60}")
        print(f"Participant {pid}  |  video={video_duration:.1f}s  |  {video_path.name}")

        for seg_type in args.segment_types:
            w = window_sizes[seg_type]
            print(f"  → {seg_type} (window={w})")
            records = make_group_clips(
                pid=pid,
                csv_path=csv_path,
                video_path=video_path,
                out_dir=args.out_dir,
                window=w,
                pre_roll=args.pre_roll,
                post_roll=args.post_roll,
                dry_run=args.dry_run,
                video_duration=video_duration,
            )
            print(f"     {len(records)} clips generated")
            all_records.extend(records)

    # Write manifest
    if all_records and not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_records[0].keys())
        with open(args.manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\nManifest written: {args.manifest}  ({len(all_records)} rows)")

    # Summary
    from collections import Counter
    seg_counts = Counter(r["segment_type"] for r in all_records)
    food_counts = Counter(r["food_type"] for r in all_records)
    print("\n── Summary ──────────────────────────────────────")
    print(f"  Total clips: {len(all_records)}")
    for k, v in sorted(seg_counts.items()):
        print(f"  {k}: {v}")
    print(f"  Food types: {dict(food_counts)}")


if __name__ == "__main__":
    main()
