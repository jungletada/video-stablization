#!/usr/bin/env bash
set -euo pipefail

# Run trajectory variants in separate processes and assign each process to one GPU.
#
# Example:
#   bash models/ReCamMaster/run_stabilization_multi_gpu.sh "0,1,2" \
#     --enable_vram_management \
#     --height 384 --width 672 \
#     --dataset_path ./example_test_data \
#     --pose_file ./example_test_data/cameras/camera_extrinsics.json
#
# The first argument is a comma-separated list of physical GPU ids. All remaining
# arguments are forwarded to run_stabilization_experiment.py.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 GPU_IDS [run_stabilization_experiment.py args...]"
  echo "Example: $0 0,1,2 --enable_vram_management --height 384 --width 672 --dataset_path ./example_test_data --pose_file ./example_test_data/cameras/camera_extrinsics.json"
  exit 2
fi

GPU_IDS="$1"
shift

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
VARIANTS=("raw" "smooth" "slow_static")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${REPO_ROOT}/logs"

for idx in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  log_file="${REPO_ROOT}/logs/stabilization_${variant}_gpu${gpu}.log"
  echo "Launching variant=${variant} on physical GPU ${gpu}; log=${log_file}"
  (
    cd "${REPO_ROOT}"
    python models/ReCamMaster/run_stabilization_experiment.py \
      --cuda_devices "${gpu}" \
      --device cuda \
      --variants "${variant}" \
      "$@"
  ) >"${log_file}" 2>&1 &
done

wait
echo "All variants finished."

