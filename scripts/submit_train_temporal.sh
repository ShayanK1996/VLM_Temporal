#!/bin/bash
# Run from anywhere: ensures logs_stage_1 exists under the repo BEFORE sbatch opens log paths.
# Usage:  ./scripts/submit_train_temporal.sh
#     or:  bash scripts/submit_train_temporal.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${REPO}/logs_stage_1"
cd "${REPO}"
exec sbatch "${REPO}/scripts/train_temporal.sh"
