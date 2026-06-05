#!/usr/bin/env bash
set -euo pipefail

# Run ReCamMaster stabilization with DVS-exported camera embeddings.
#
# Example:
#   bash models/ReCamMaster/run_dvs_embedding_stabilization.sh 0
#   bash models/ReCamMaster/run_dvs_embedding_stabilization.sh "0,1,2,3,4,5,6,7" --enable_vram_management

GPU_IDS="${1:-0}"
if [[ $# -ge 1 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECAMMASTER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${RECAMMASTER_ROOT}/logs"
mkdir -p "${LOG_DIR}"

SAFE_GPU_IDS="${GPU_IDS//,/}"
LOG_FILE="${LOG_DIR}/dvs_embedding_stabilization_gpu${SAFE_GPU_IDS}.log"

echo "Launching DVS embedding ReCamMaster stabilization"
echo "  visible physical GPUs: ${GPU_IDS}"
echo "  height: 480"
echo "  width: 832"
echo "  num_frames: 81"
echo "  num_inference_steps: 50"
echo "  cfg_scale: 5.0"
echo "  torch_dtype: bfloat16"
echo "  log: ${LOG_FILE}"

(
  cd "${RECAMMASTER_ROOT}"
  python models/ReCamMaster/run_dvs_embedding_stabilization.py \
    --cuda_devices "${GPU_IDS}" \
    --device cuda \
    "$@"
) 2>&1 | tee "${LOG_FILE}"
