"""Build a ReCamMaster embedding from a straightened C2W camera-center path.

This probe keeps the sampled pose rotations, converts W2C -> C2W, then replaces
camera centers with [0, 0, original_z].  The resulting C2W sequence is converted
into ReCamMaster's relative [N, 12] camera embedding without applying the extra
ReCamMaster axis conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajectory_utils import c2w_to_recammaster_embedding, project_rotation, trajectory_summary


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSE_NPY = (
    "da3/outputs/giant_l1opt/PRO_VID_20260602_153852_00_023_ud_smooth_trajectory/"
    "sampled_smooth_w2c_start90.npy"
)
DEFAULT_OUTPUT_DIR = "da3/outputs/recammaster_condition_giant_1_1_smooth_l1_straight_c2w_center_no_axes"
DEFAULT_SAMPLE_NAME = "PRO_VID_20260602_153852_00_023"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_pose_sequence(path: Path, convention: str) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(np.load(path), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected [N, 4, 4] poses at {path}, got {poses.shape}")

    if convention == "w2c":
        c2ws = np.linalg.inv(poses)
    elif convention == "c2w":
        c2ws = poses.copy()
    else:
        raise ValueError(f"Unsupported convention: {convention}")

    for i in range(len(c2ws)):
        c2ws[i, :3, :3] = project_rotation(c2ws[i, :3, :3])
        c2ws[i, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return poses, c2ws


def straighten_camera_centers(c2ws: np.ndarray, keep_rotation: bool = True) -> np.ndarray:
    straight = c2ws.copy()
    centers = c2ws[:, :3, 3].copy()
    straight_centers = np.zeros_like(centers)
    straight_centers[:, 2] = centers[:, 2]
    straight[:, :3, 3] = straight_centers

    if not keep_rotation:
        straight[:, :3, :3] = c2ws[0, :3, :3]
    return straight


def embedding_translation_stats(embedding: np.ndarray) -> dict[str, float | list[float]]:
    poses = embedding.reshape(-1, 3, 4)
    translation = poses[:, :3, 3]
    norms = np.linalg.norm(translation, axis=1)
    return {
        "translation_max_abs": float(np.max(np.abs(translation))),
        "translation_max_norm": float(np.max(norms)),
        "translation_mean_norm": float(np.mean(norms)),
        "translation_first": translation[0].astype(float).tolist(),
        "translation_last": translation[-1].astype(float).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Straighten C2W camera centers to x=0,y=0 while preserving z, then export ReCamMaster embedding."
    )
    parser.add_argument("--pose_npy", default=DEFAULT_POSE_NPY)
    parser.add_argument("--matrix_convention", choices=["w2c", "c2w"], default="w2c")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_name", default=DEFAULT_SAMPLE_NAME)
    parser.add_argument(
        "--freeze_rotation",
        action="store_true",
        help="Use the first pose rotation for all frames. By default sampled rotations are preserved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_path = project_path(args.pose_npy)
    output_dir = project_path(args.output_dir) / args.sample_name
    output_dir.mkdir(parents=True, exist_ok=True)

    source_poses, source_c2ws = load_pose_sequence(pose_path, args.matrix_convention)
    straight_c2ws = straighten_camera_centers(source_c2ws, keep_rotation=not args.freeze_rotation)
    straight_w2cs = np.linalg.inv(straight_c2ws)
    embedding = c2w_to_recammaster_embedding(straight_c2ws)

    source_centers = source_c2ws[:, :3, 3]
    straight_centers = straight_c2ws[:, :3, 3]

    np.save(output_dir / "source_poses.npy", source_poses.astype(np.float32))
    np.save(output_dir / "source_c2w.npy", source_c2ws.astype(np.float32))
    np.save(output_dir / "straight_c2w.npy", straight_c2ws.astype(np.float32))
    np.save(output_dir / "straight_w2c.npy", straight_w2cs.astype(np.float32))
    np.save(output_dir / "source_camera_centers_c2w.npy", source_centers.astype(np.float32))
    np.save(output_dir / "straight_camera_centers_c2w.npy", straight_centers.astype(np.float32))
    np.save(output_dir / "recammaster_camera_embedding.npy", embedding.astype(np.float32))
    np.savetxt(output_dir / "recammaster_camera_embedding.txt", embedding, fmt="%.9g", delimiter=" ")

    summary = {
        "input_pose_npy": str(pose_path),
        "matrix_convention": args.matrix_convention,
        "output_dir": str(output_dir),
        "shape": list(straight_c2ws.shape),
        "camera_center_edit": "C2W center x=0, y=0, z=source_z",
        "rotation": "first pose frozen" if args.freeze_rotation else "source sampled rotations preserved",
        "apply_recammaster_axes": False,
        "recammaster_flatten_order": "row-major [r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz]",
        "source_center_first": source_centers[0].astype(float).tolist(),
        "source_center_last": source_centers[-1].astype(float).tolist(),
        "straight_center_first": straight_centers[0].astype(float).tolist(),
        "straight_center_last": straight_centers[-1].astype(float).tolist(),
        "straight_center_min": straight_centers.min(axis=0).astype(float).tolist(),
        "straight_center_max": straight_centers.max(axis=0).astype(float).tolist(),
        "embedding_translation_stats": embedding_translation_stats(embedding),
        "straight_trajectory_summary": trajectory_summary(straight_c2ws),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Loaded {pose_path}: {source_poses.shape} ({args.matrix_convention})")
    print(f"Saved straight C2W center embedding: {output_dir / 'recammaster_camera_embedding.npy'} {embedding.shape}")
    print(f"Source center first -> last: {source_centers[0]} -> {source_centers[-1]}")
    print(f"Straight center first -> last: {straight_centers[0]} -> {straight_centers[-1]}")
    print(f"Embedding translation max_norm: {summary['embedding_translation_stats']['translation_max_norm']:.6g}")


if __name__ == "__main__":
    main()
