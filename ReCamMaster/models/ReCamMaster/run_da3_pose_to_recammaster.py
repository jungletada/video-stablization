"""Convert Depth-Anything-3 pose npz outputs to ReCamMaster camera embeddings.

The DA3 output produced by da3/depth_pose_estimate.py stores one npz per frame:

- depth: [H, W]
- pose: [4, 4], saved as global world-to-camera by that script
- intrinsic: [3, 3]

This converter loads the pose sequence, converts it to c2w, samples 21 poses,
and writes a ReCamMaster-ready [21, 12] relative camera embedding.  It can
also read a precomputed pose npy file, such as a smoothed DA3 W2C trajectory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from trajectory_utils import (
    c2w_to_recammaster_embedding,
    convert_to_recammaster_camera_axes,
    project_rotation,
    trajectory_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = "da3/outputs/small_l1opt/PRO_VID_20260602_153852_00_023_ud"
DEFAULT_OUTPUT_DIR = "da3/outputs/recammaster_condition"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def frame_index(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.npz$)", path.name)
    if match:
        return int(match.group(1))
    return 10**12


def find_npz_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.npz"), key=lambda p: (frame_index(p), p.name))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")
    return files


def load_da3_poses(
    input_dir: Path,
    pose_key: str,
    matrix_convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    files = find_npz_files(input_dir)
    matrices = []
    for path in files:
        with np.load(path) as data:
            if pose_key not in data:
                raise KeyError(f"{pose_key!r} not found in {path}; keys={list(data.keys())}")
            matrix = np.asarray(data[pose_key], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected {pose_key} shape [4, 4] in {path}, got {matrix.shape}")
        matrices.append(matrix)

    poses = np.stack(matrices, axis=0)
    if matrix_convention == "w2c":
        c2ws = np.linalg.inv(poses)
    elif matrix_convention == "c2w":
        c2ws = poses.copy()
    else:
        raise ValueError(f"Unsupported matrix_convention: {matrix_convention}")

    for i in range(len(c2ws)):
        c2ws[i, :3, :3] = project_rotation(c2ws[i, :3, :3])
        c2ws[i, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return poses, c2ws


def load_pose_npy(
    pose_npy: Path,
    matrix_convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(np.load(pose_npy), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose npy shape [N, 4, 4] at {pose_npy}, got {poses.shape}")

    if matrix_convention == "w2c":
        c2ws = np.linalg.inv(poses)
    elif matrix_convention == "c2w":
        c2ws = poses.copy()
    else:
        raise ValueError(f"Unsupported matrix_convention: {matrix_convention}")

    for i in range(len(c2ws)):
        c2ws[i, :3, :3] = project_rotation(c2ws[i, :3, :3])
        c2ws[i, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return poses, c2ws


def sample_pose_indices(
    length: int,
    start_frame: int,
    mode: str,
    num_frames: int,
    stride: int,
    count: int,
) -> np.ndarray:
    if start_frame < 0:
        raise ValueError("--start_frame must be >= 0")
    if start_frame >= length:
        raise ValueError(f"--start_frame {start_frame} is outside pose sequence length {length}")

    if mode == "stride":
        indices = start_frame + np.arange(0, num_frames, stride, dtype=np.int64)
        if len(indices) != count:
            raise ValueError(
                f"num_frames={num_frames} and stride={stride} produce {len(indices)} poses, "
                f"but pose_count={count} is required."
            )
        if indices[-1] >= length:
            raise ValueError(
                f"Need at least frame index {indices[-1]} for stride sampling, got {length} poses."
            )
        return indices

    if mode == "uniform":
        available = length - start_frame
        if available < count:
            raise ValueError(f"Need at least {count} poses from start_frame={start_frame}, got {available}")
        return np.rint(np.linspace(start_frame, length - 1, count)).astype(np.int64)

    raise ValueError(f"Unsupported sample_mode: {mode}")


def embedding_translation_stats(embedding: np.ndarray) -> dict[str, float | list[float]]:
    poses = embedding.reshape(-1, 3, 4)
    translation = poses[:, :3, 3]
    norms = np.linalg.norm(translation, axis=1)
    return {
        "translation_max_abs": float(np.max(np.abs(translation))),
        "translation_max_norm": float(np.max(norms)),
        "translation_mean_norm": float(np.mean(norms)),
        "translation_last": translation[-1].astype(float).tolist(),
    }


def maybe_rescale_to_target(
    c2ws: np.ndarray,
    sample_indices: np.ndarray,
    target_max_translation_norm: float | None,
) -> tuple[np.ndarray, float, dict[str, float | list[float]]]:
    embedding = c2w_to_recammaster_embedding(c2ws[sample_indices])
    stats = embedding_translation_stats(embedding)
    if target_max_translation_norm is None:
        return c2ws, 1.0, stats
    if target_max_translation_norm <= 0:
        raise ValueError("--target_max_translation_norm must be > 0")

    current = float(stats["translation_max_norm"])
    if current < 1e-8:
        return c2ws, 1.0, stats

    auto_scale = current / target_max_translation_norm
    scaled = c2ws.copy()
    scaled[:, :3, 3] /= auto_scale
    scaled_embedding = c2w_to_recammaster_embedding(scaled[sample_indices])
    return scaled, auto_scale, embedding_translation_stats(scaled_embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DA3 pose npz files to ReCamMaster camera embeddings.")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--pose_npy",
        default=None,
        help="Optional [N, 4, 4] pose npy file. Use this for smoothed DA3 trajectories.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_name", default=None, help="Defaults to the input directory name.")
    parser.add_argument("--pose_key", default="pose")
    parser.add_argument(
        "--matrix_convention",
        choices=["w2c", "c2w"],
        default="w2c",
        help="DA3 depth_pose_estimate.py saves pose as global w2c.",
    )
    parser.add_argument(
        "--no_recammaster_axes",
        action="store_true",
        help="Skip the camera-axis conversion used by ReCamMaster's original data loader.",
    )
    parser.add_argument("--translation_scale", type=float, default=1.0)
    parser.add_argument(
        "--target_max_translation_norm",
        type=float,
        default=None,
        help=(
            "Optional extra auto scale for arbitrary-scale DA3 translation. "
            "For example, 2.0 makes the sampled embedding's max relative translation norm about 2."
        ),
    )
    parser.add_argument("--sample_mode", choices=["stride", "uniform"], default="stride")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--camera_stride", type=int, default=4)
    parser.add_argument("--pose_count", type=int, default=21)
    parser.add_argument(
        "--pose_already_sampled",
        action="store_true",
        help="Treat the loaded pose sequence as the exact camera samples to embed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = project_path(args.input_dir)
    pose_npy = project_path(args.pose_npy) if args.pose_npy else None
    output_root = project_path(args.output_dir)
    sample_name = args.sample_name or (pose_npy.stem if pose_npy else input_dir.name)
    output_dir = output_root / sample_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if pose_npy is not None:
        raw_poses, c2ws = load_pose_npy(
            pose_npy=pose_npy,
            matrix_convention=args.matrix_convention,
        )
        input_description = str(pose_npy)
    else:
        raw_poses, c2ws = load_da3_poses(
            input_dir=input_dir,
            pose_key=args.pose_key,
            matrix_convention=args.matrix_convention,
        )
        input_description = str(input_dir)

    if not args.no_recammaster_axes:
        c2ws = convert_to_recammaster_camera_axes(c2ws)

    if args.translation_scale == 0:
        raise ValueError("--translation_scale must be non-zero")
    c2ws = c2ws.copy()
    c2ws[:, :3, 3] /= args.translation_scale

    if args.pose_already_sampled:
        if len(c2ws) != args.pose_count:
            raise ValueError(
                f"--pose_already_sampled expects pose_count={args.pose_count} poses, got {len(c2ws)}."
            )
        sample_indices = np.arange(len(c2ws), dtype=np.int64)
    else:
        sample_indices = sample_pose_indices(
            len(c2ws),
            start_frame=args.start_frame,
            mode=args.sample_mode,
            num_frames=args.num_frames,
            stride=args.camera_stride,
            count=args.pose_count,
        )
    c2ws, auto_translation_scale, embedding_stats = maybe_rescale_to_target(
        c2ws,
        sample_indices=sample_indices,
        target_max_translation_norm=args.target_max_translation_norm,
    )

    sampled_c2ws = c2ws[sample_indices]
    camera_embedding = c2w_to_recammaster_embedding(sampled_c2ws)

    np.save(output_dir / "da3_pose_raw.npy", raw_poses.astype(np.float32))
    np.save(output_dir / "da3_c2w_recammaster_axes.npy", c2ws.astype(np.float32))
    np.save(output_dir / "sampled_da3_c2w_recammaster_axes.npy", sampled_c2ws.astype(np.float32))
    np.save(output_dir / "recammaster_camera_embedding.npy", camera_embedding)
    np.savetxt(output_dir / "recammaster_camera_embedding.txt", camera_embedding, fmt="%.9g", delimiter=" ")
    np.savetxt(output_dir / "sample_indices.txt", sample_indices, fmt="%d")

    summary = {
        "sample": sample_name,
        "input_dir": str(input_dir),
        "pose_npy": str(pose_npy) if pose_npy is not None else None,
        "input_source": input_description,
        "output_dir": str(output_dir),
        "pose_key": args.pose_key,
        "matrix_convention": args.matrix_convention,
        "source_pose_shape": list(raw_poses.shape),
        "c2w_shape": list(c2ws.shape),
        "apply_recammaster_axes": not args.no_recammaster_axes,
        "translation_scale": args.translation_scale,
        "auto_translation_scale": auto_translation_scale,
        "effective_translation_scale": args.translation_scale * auto_translation_scale,
        "target_max_translation_norm": args.target_max_translation_norm,
        "sample_mode": args.sample_mode,
        "pose_already_sampled": args.pose_already_sampled,
        "start_frame": args.start_frame,
        "num_frames": args.num_frames,
        "camera_stride": args.camera_stride,
        "pose_count": args.pose_count,
        "sample_indices": sample_indices.astype(int).tolist(),
        "sampled_c2w_shape": list(sampled_c2ws.shape),
        "recammaster_camera_embedding_shape": list(camera_embedding.shape),
        "recammaster_flatten_order": "row-major [r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz]",
        "embedding_translation_stats": embedding_stats,
        "sampled_trajectory_summary": trajectory_summary(sampled_c2ws),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Loaded DA3 poses: {raw_poses.shape} from {input_description}")
    print(f"Sample indices: {sample_indices[0]}..{sample_indices[-1]} ({len(sample_indices)} poses)")
    print(f"Saved ReCamMaster embedding: {output_dir / 'recammaster_camera_embedding.npy'} {camera_embedding.shape}")
    print(f"Embedding translation max_norm: {embedding_stats['translation_max_norm']:.6g}")
    print(f"Effective translation scale: {summary['effective_translation_scale']:.6g}")


if __name__ == "__main__":
    main()
