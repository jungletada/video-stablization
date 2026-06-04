#!/usr/bin/env bash
set -euo pipefail

# Full-shape ReCamMaster test with selected visible GPU ids.
#
# This uses height=480, width=832, num_frames=81, but keeps
# num_inference_steps=1 and cfg_scale=1.0 by default.
#
# Example:
#   bash models/ReCamMaster/run_recammaster_full_shape_test.sh 0
#
# To expose all 8 GPUs to this process:
#   bash models/ReCamMaster/run_recammaster_full_shape_test.sh "0,1,2,3,4,5,6,7"
#
# Extra arguments are forwarded to run_recammaster_full_shape_test.py.

GPU_IDS="${1:-0}"
if [[ $# -ge 1 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

SAFE_GPU_IDS="${GPU_IDS//,/}"
LOG_FILE="${LOG_DIR}/recammaster_full_shape_test_gpu${SAFE_GPU_IDS}.log"

echo "Launching ReCamMaster full-shape test"
echo "  visible physical GPUs: ${GPU_IDS}"
echo "  note: the current pipeline still runs a single inference on logical cuda:0"
echo "  height: 480"
echo "  width: 832"
echo "  num_frames: 81"
echo "  log: ${LOG_FILE}"

(
  cd "${REPO_ROOT}"
  python models/ReCamMaster/run_recammaster_full_shape_test.py \
    --cuda_devices "${GPU_IDS}" \
    --device cuda \
    "$@"
) 2>&1 | tee "${LOG_FILE}"
