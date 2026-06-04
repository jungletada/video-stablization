#!/usr/bin/env bash
set -euo pipefail

# Full-shape ReCamMaster test on one GPU.
#
# This uses height=480, width=832, num_frames=81, but keeps
# num_inference_steps=1 and cfg_scale=1.0 by default.
#
# Example:
#   bash models/ReCamMaster/run_recammaster_full_shape_test.sh 0
#
# Extra arguments are forwarded to run_recammaster_full_shape_test.py.

GPU_ID="${1:-0}"
if [[ $# -ge 1 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/recammaster_full_shape_test_gpu${GPU_ID}.log"

echo "Launching ReCamMaster full-shape test"
echo "  physical GPU: ${GPU_ID}"
echo "  height: 480"
echo "  width: 832"
echo "  num_frames: 81"
echo "  log: ${LOG_FILE}"

(
  cd "${REPO_ROOT}"
  python models/ReCamMaster/run_recammaster_full_shape_test.py \
    --cuda_devices "${GPU_ID}" \
    --device cuda \
    "$@"
) 2>&1 | tee "${LOG_FILE}"

