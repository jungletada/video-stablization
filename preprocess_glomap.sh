#!/bin/bash
# Video to COLMAP global SfM (GLOMAP) reconstruction pipeline.
#
# Pipeline: video -> ffmpeg frames -> COLMAP feature extraction + sequential
# matching -> global SfM (GLOMAP via COLMAP 4.0+ global_mapper) -> undistort
# to PINHOLE camera model.
#
# COLMAP 4.0+ integrates GLOMAP as the `global_mapper` subcommand, so a
# standalone glomap binary is not required.
#
# Usage:
#   bash scripts/preprocess_video_generic.sh
# Override any variable from the environment, e.g.:
#   VIDEO=/path/to/clip.mp4 FRAME_DISTANCE=120 bash scripts/preprocess_video_generic.sh
#
# Requirements:
#   - colmap 4.x with the `global_mapper` subcommand (CUDA build)
#   - ffmpeg

set -euo pipefail

# =============================================================================
# 0. Configuration
# =============================================================================
VIDEO=${VIDEO:-input.mp4}
WORK=${WORK:-colmap_workspace}
CAMERA_MODEL=${CAMERA_MODEL:-OPENCV}

# Optional frame decimation for COLMAP (1 = keep every frame). Larger value =>
# fewer frames, more parallax per pair, faster matching.
FRAME_STEP=${FRAME_STEP:-1}

IMAGES_RAW=${WORK}/images_raw
DATABASE=${WORK}/database.db
SPARSE=${WORK}/sparse
UNDISTORTED=${WORK}/undistorted

mkdir -p "${WORK}"

echo "=========================================================="
echo " Video to COLMAP global SfM pipeline"
echo "   VIDEO           = ${VIDEO}"
echo "   WORK            = ${WORK}"
echo "   CAMERA_MODEL    = ${CAMERA_MODEL}"
echo "   FRAME_STEP      = ${FRAME_STEP}"
echo "=========================================================="

# =============================================================================
# 1. Extract frames (ffmpeg)
# =============================================================================
if [ -d "${IMAGES_RAW}" ] && [ -n "$(ls -A "${IMAGES_RAW}" 2>/dev/null)" ]; then
    echo "[1/6] Frames already present in ${IMAGES_RAW}, skipping extraction."
else
    echo "[1/6] Extracting frames from video..."
    mkdir -p "${IMAGES_RAW}"
    if [ "${FRAME_STEP}" -gt 1 ]; then
        ffmpeg -hide_banner -loglevel warning -i "${VIDEO}" \
            -vf "select=not(mod(n\,${FRAME_STEP}))" -vsync 0 \
            -qscale:v 2 -start_number 0 "${IMAGES_RAW}/frame_%05d.png"
    else
        ffmpeg -hide_banner -loglevel warning -i "${VIDEO}" \
            -qscale:v 2 -start_number 0 "${IMAGES_RAW}/frame_%05d.png"
    fi
    echo "      Extracted $(ls "${IMAGES_RAW}" | wc -l) frames."
fi

# =============================================================================
# 2. Feature extraction (single camera, distortion-capable model)
# =============================================================================
echo "[2/6] COLMAP feature_extractor..."
colmap feature_extractor \
    --database_path "${DATABASE}" \
    --image_path "${IMAGES_RAW}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "${CAMERA_MODEL}"

# =============================================================================
# 3. Sequential matching (video frames are temporally ordered)
# =============================================================================
echo "[3/6] COLMAP sequential_matcher..."
colmap sequential_matcher \
    --database_path "${DATABASE}"

# =============================================================================
# 4. View-graph calibration (recommended before global SfM)
# =============================================================================
echo "[4/6] COLMAP view_graph_calibrator..."
colmap view_graph_calibrator \
    --database_path "${DATABASE}"

# =============================================================================
# 5. Global SfM (GLOMAP via COLMAP 4.0+ global_mapper)
# =============================================================================
echo "[5/6] COLMAP global_mapper (GLOMAP global SfM)..."
mkdir -p "${SPARSE}"
colmap global_mapper \
    --database_path "${DATABASE}" \
    --image_path "${IMAGES_RAW}" \
    --output_path "${SPARSE}"

# Resolve the sparse model dir (global_mapper writes sparse/0/, but be defensive).
if [ -f "${SPARSE}/0/cameras.bin" ] || [ -f "${SPARSE}/0/cameras.txt" ]; then
    SPARSE_MODEL="${SPARSE}/0"
elif [ -f "${SPARSE}/cameras.bin" ] || [ -f "${SPARSE}/cameras.txt" ]; then
    SPARSE_MODEL="${SPARSE}"
else
    echo "ERROR: no COLMAP model found under ${SPARSE}. global_mapper may have failed." >&2
    exit 1
fi
echo "      Sparse model: ${SPARSE_MODEL}"
colmap model_analyzer --path "${SPARSE_MODEL}" || true

# =============================================================================
# 6. Undistort to PINHOLE (removes lens distortion, projects to standard camera)
# =============================================================================
echo "[6/6] COLMAP image_undistorter -> PINHOLE..."
colmap image_undistorter \
    --image_path "${IMAGES_RAW}" \
    --input_path "${SPARSE_MODEL}" \
    --output_path "${UNDISTORTED}" \
    --output_type COLMAP

# image_undistorter writes <out>/images and <out>/sparse (cameras/images/points3D
# directly, no 0/ nesting).
if [ -f "${UNDISTORTED}/sparse/cameras.bin" ] || [ -f "${UNDISTORTED}/sparse/cameras.txt" ]; then
    true
elif [ -f "${UNDISTORTED}/sparse/0/cameras.bin" ] || [ -f "${UNDISTORTED}/sparse/0/cameras.txt" ]; then
    true
else
    echo "ERROR: undistorted sparse model not found under ${UNDISTORTED}/sparse." >&2
    exit 1
fi
echo "      Undistorted scene: ${UNDISTORTED}"

echo "=========================================================="
echo " Done. Scene ready at: ${UNDISTORTED}"
echo "   - Images:     ${UNDISTORTED}/images"
echo "   - Sparse SfM: ${UNDISTORTED}/sparse"
echo "=========================================================="

