#!/bin/bash
# Sync local VLM_TemporalBranch to Unity HPC.
# Run this from your local machine whenever you change scripts or code.
#
# Usage:
#   ./scripts/sync_to_unity.sh
#
# Requires: rsync, SSH access to Unity (e.g. unityhpc or your configured host)

set -euo pipefail

# --- Config (customize if needed) ---
UNITY_HOST="${UNITY_HOST:-unityhpc}"
REMOTE_DIR="${REMOTE_DIR:-VLM_Temporal}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing $LOCAL_DIR -> ${UNITY_HOST}:~/${REMOTE_DIR}/"
echo ""

rsync -avz --progress \
  --exclude '.git' \
  --exclude 'VLM_Temporal' \
  --exclude 'cached_features' \
  --exclude '20260311_171539_*' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.ipynb_checkpoints' \
  --exclude '*.egg-info' \
  --exclude 'logs' \
  "$LOCAL_DIR/" "${UNITY_HOST}:~/${REMOTE_DIR}/"

echo ""
echo "Done. On Unity run: cd ~/${REMOTE_DIR} && sbatch scripts/extract_features.sh"
