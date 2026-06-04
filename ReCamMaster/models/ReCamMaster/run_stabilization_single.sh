#!/usr/bin/env bash
set -euo pipefail

# Run one ReCamMaster stabilization variant on one selected GPU.
#
# Example:
#   bash models/ReCamMaster/run_stabilization_single.sh 0 smooth \
#     --enable_vram_management \
#     --height 384 --width 672 \
#     --dataset_path ./example_test_data \
#     --pose_file ./example_test_data/cameras/camera_extrinsics.json
#
# Arguments:
#   1. Physical GPU id, for example 0.
#   2. Variant: raw, smooth, or slow_static. Defaults to smooth.
#   3+. Extra arguments forwarded to run_stabilization_experiment.py.

GPU_ID="${1:-0}"
VARIANT="${2:-smooth}"

if [[ $# -ge 1 ]]; then
  shift
fi
if [[ $# -ge 1 ]]; then
  shift
fi

case "${VARIANT}" in
  raw|smooth|slow_static)
    ;;
  *)
    echo "Unknown variant: ${VARIANT}. Use raw, smooth, or slow_static."
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/stabilization_${VARIANT}_gpu${GPU_ID}.log"

echo "Launching one stabilization run"
echo "  physical GPU: ${GPU_ID}"
echo "  variant: ${VARIANT}"
echo "  log: ${LOG_FILE}"

(
  cd "${REPO_ROOT}"
  python models/ReCamMaster/run_stabilization_experiment.py \
    --cuda_devices "${GPU_ID}" \
    --device cuda \
    --variant "${VARIANT}" \
    "$@"
) 2>&1 | tee "${LOG_FILE}"

