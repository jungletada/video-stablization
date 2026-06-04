#!/usr/bin/env bash
set -euo pipefail

# Minimal ReCamMaster smoke test on one GPU.
#
# Example:
#   bash models/ReCamMaster/run_recammaster_smoke_test.sh 0
#
# Extra arguments are forwarded to run_recammaster_smoke_test.py, so you can
# override defaults, for example:
#   bash models/ReCamMaster/run_recammaster_smoke_test.sh 0 --height 256 --width 448

GPU_ID="${1:-0}"
if [[ $# -ge 1 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/recammaster_smoke_test_gpu${GPU_ID}.log"

echo "Launching ReCamMaster smoke test"
echo "  physical GPU: ${GPU_ID}"
echo "  log: ${LOG_FILE}"

(
  cd "${REPO_ROOT}"
  python models/ReCamMaster/run_recammaster_smoke_test.py \
    --cuda_devices "${GPU_ID}" \
    --device cuda \
    "$@"
) 2>&1 | tee "${LOG_FILE}"

