#!/usr/bin/env bash
# Source from batch scripts after REPO_DIR is set, e.g.:
#   source "${REPO_DIR}/scripts/lib_vlm_work_root.sh"
# (Do not use dirname "${BASH_SOURCE[0]}" — under SLURM the script runs from /var/spool/slurm/...)
#
# Unity HPC: cached_features (.pt + manifest) and temporal/e2e checkpoints live ONLY
# under this tree — not under the repo clone. Override for local dev:
#   export VLM_WORK_ROOT=/path/to/writable/root

if [ -n "${VLM_WORK_ROOT:-}" ]; then
  :
else
  VLM_WORK_ROOT="/work/pi_walls_uri_edu/${USER}/VLM_Temporal"
  if ! mkdir -p "${VLM_WORK_ROOT}" 2>/dev/null; then
    echo "ERROR: Cannot use artifact root ${VLM_WORK_ROOT} (mkdir failed or no permission)."
    echo "On Unity, cached features are stored here; create the directory or set VLM_WORK_ROOT."
    exit 2
  fi
fi
