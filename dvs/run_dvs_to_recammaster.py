"""Run DVS virtual-camera inference and export ReCamMaster camera embeddings.

This script keeps the DVS output and the ReCamMaster-ready conversion side by
side:

- virtual_queue.txt: [N, 5], columns are timestamp_ns qx qy qz qw.
- recammaster_camera_embedding.npy/txt: [21, 12].

DVS quaternions follow the gyro/virtual-projection convention used by DVS
warping. ReCamMaster expects camera-to-world-style relative camera embeddings,
so the default conversion transposes the DVS rotation matrix before embedding.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import get_inference_data_loader
from gyro.gyro_function import ConvertQuaternionToRotationMatrix
from inference import run as run_dvs
from model import Model


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cf: dict, use_cuda: bool) -> Model:
    model = Model(cf)
    load_model_path = cf["model"]["load_model"]
    if load_model_path is None:
        checkpoint_dir = Path(cf["data"]["checkpoints_dir"]) / cf["data"]["exp"]
        load_model_path = checkpoint_dir / f"{cf['data']['exp']}_last.checkpoint"
    checkpoint = torch.load(load_model_path, map_location="cpu")
    model.net.load_state_dict(checkpoint["state_dict"])
    model.unet.load_state_dict(checkpoint["unet"])
    if use_cuda:
        model.net.cuda()
        model.unet.cuda()
    print(f"Loaded DVS checkpoint: {load_model_path}")
    return model


def quaternion_sequence_to_c2w(quaternions_xyzw: np.ndarray, rotation_mode: str) -> np.ndarray:
    c2ws = []
    for quat in quaternions_xyzw:
        rotation = ConvertQuaternionToRotationMatrix(quat).astype(np.float32)
        if rotation_mode == "inverse":
            rotation = rotation.T
        elif rotation_mode != "as_is":
            raise ValueError(f"Unsupported rotation_mode: {rotation_mode}")
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        c2ws.append(matrix)
    return np.asarray(c2ws, dtype=np.float32)


def sample_pose_indices(
    length: int,
    mode: str,
    num_frames: int,
    stride: int,
    count: int,
    start_frame: int = 0,
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
                f"but count={count} is required."
            )
        if indices[-1] >= length:
            raise ValueError(
                f"Need at least {indices[-1] + 1} DVS poses for stride sampling, got {length}."
            )
        return indices
    if mode == "uniform":
        available = length - start_frame
        if available < count:
            raise ValueError(f"Need at least {count} poses from start_frame={start_frame}, got {available}.")
        return np.rint(np.linspace(start_frame, length - 1, count)).astype(np.int64)
    raise ValueError(f"Unsupported sample mode: {mode}")


def c2w_to_recammaster_embedding(c2ws: np.ndarray) -> np.ndarray:
    anchor_inv = np.linalg.inv(c2ws[0])
    relative = []
    for c2w in c2ws:
        rel = anchor_inv @ c2w
        relative.append(rel[:3, :].reshape(-1))
    return np.asarray(relative, dtype=np.float32)


def prepend_initial_pose(virtual_queue: np.ndarray, first_timestamp: float) -> np.ndarray:
    first = np.zeros((1, 5), dtype=np.float32)
    first[:, 0] = first_timestamp
    first[:, 1:] = virtual_queue[0, 1:]
    return np.concatenate((first, virtual_queue.astype(np.float32)), axis=0)


def export_sample(
    model: Model,
    cf: dict,
    data_path: Path,
    output_dir: Path,
    use_cuda: bool,
    sample_mode: str,
    num_frames: int,
    camera_stride: int,
    pose_count: int,
    start_frame: int,
    rotation_mode: str,
) -> None:
    print(f"\n=== {data_path.name} ===")
    loader = get_inference_data_loader(cf, str(data_path), no_flo=False)
    data = loader.dataset.data[0]
    virtual_queue = run_dvs(model, loader, cf, USE_CUDA=use_cuda, compute_loss=False)
    virtual_queue = prepend_initial_pose(virtual_queue, data.frame[0, 0])

    quaternions = virtual_queue[:, 1:5]
    c2ws = quaternion_sequence_to_c2w(quaternions, rotation_mode=rotation_mode)
    sample_indices = sample_pose_indices(
        len(c2ws),
        mode=sample_mode,
        num_frames=num_frames,
        stride=camera_stride,
        count=pose_count,
        start_frame=start_frame,
    )
    sampled_c2ws = c2ws[sample_indices]
    camera_embedding = c2w_to_recammaster_embedding(sampled_c2ws)

    sample_out = output_dir / data_path.name
    sample_out.mkdir(parents=True, exist_ok=True)
    np.savetxt(sample_out / "virtual_queue.txt", virtual_queue, fmt="%.9g", delimiter=" ")
    np.save(sample_out / "virtual_queue.npy", virtual_queue)
    np.save(sample_out / "rotation_matrices.npy", c2ws[:, :3, :3])
    np.save(sample_out / "c2w_zero_translation.npy", c2ws)
    np.save(sample_out / "sampled_c2w_zero_translation.npy", sampled_c2ws)
    np.save(sample_out / "recammaster_camera_embedding.npy", camera_embedding)
    np.savetxt(sample_out / "recammaster_camera_embedding.txt", camera_embedding, fmt="%.9g", delimiter=" ")

    summary = {
        "sample": data_path.name,
        "virtual_queue_shape": list(virtual_queue.shape),
        "quaternion_order": "qx qy qz qw",
        "rotation_mode": rotation_mode,
        "c2w_zero_translation_shape": list(c2ws.shape),
        "sample_mode": sample_mode,
        "start_frame": start_frame,
        "sample_indices": sample_indices.astype(int).tolist(),
        "recammaster_camera_embedding_shape": list(camera_embedding.shape),
        "recammaster_flatten_order": "row-major [r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz]",
        "translation": "zero",
    }
    with (sample_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved virtual queue: {sample_out / 'virtual_queue.txt'} {virtual_queue.shape}")
    print(f"Saved ReCamMaster embedding: {sample_out / 'recammaster_camera_embedding.npy'} {camera_embedding.shape}")


def reexport_saved_virtual_queue(
    input_dir: Path,
    output_dir: Path,
    sample_mode: str,
    num_frames: int,
    camera_stride: int,
    pose_count: int,
    start_frame: int,
    rotation_mode: str,
) -> None:
    sample_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No saved DVS sample directories found under {input_dir}")

    for data_path in sample_dirs:
        print(f"\n=== {data_path.name} ===")
        virtual_queue_path = data_path / "virtual_queue.npy"
        if not virtual_queue_path.exists():
            raise FileNotFoundError(f"Missing saved virtual queue: {virtual_queue_path}")
        virtual_queue = np.load(virtual_queue_path).astype(np.float32)
        quaternions = virtual_queue[:, 1:5]
        c2ws = quaternion_sequence_to_c2w(quaternions, rotation_mode=rotation_mode)
        sample_indices = sample_pose_indices(
            len(c2ws),
            mode=sample_mode,
            num_frames=num_frames,
            stride=camera_stride,
            count=pose_count,
            start_frame=start_frame,
        )
        sampled_c2ws = c2ws[sample_indices]
        camera_embedding = c2w_to_recammaster_embedding(sampled_c2ws)

        sample_out = output_dir / data_path.name
        sample_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(sample_out / "virtual_queue.txt", virtual_queue, fmt="%.9g", delimiter=" ")
        np.save(sample_out / "virtual_queue.npy", virtual_queue)
        np.save(sample_out / "rotation_matrices.npy", c2ws[:, :3, :3])
        np.save(sample_out / "c2w_zero_translation.npy", c2ws)
        np.save(sample_out / "sampled_c2w_zero_translation.npy", sampled_c2ws)
        np.save(sample_out / "recammaster_camera_embedding.npy", camera_embedding)
        np.savetxt(sample_out / "recammaster_camera_embedding.txt", camera_embedding, fmt="%.9g", delimiter=" ")

        summary = {
            "sample": data_path.name,
            "source_virtual_queue_dir": str(data_path),
            "virtual_queue_shape": list(virtual_queue.shape),
            "quaternion_order": "qx qy qz qw",
            "rotation_mode": rotation_mode,
            "c2w_zero_translation_shape": list(c2ws.shape),
            "sample_mode": sample_mode,
            "start_frame": start_frame,
            "sample_indices": sample_indices.astype(int).tolist(),
            "recammaster_camera_embedding_shape": list(camera_embedding.shape),
            "recammaster_flatten_order": "row-major [r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz]",
            "translation": "zero",
        }
        with (sample_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Re-exported ReCamMaster embedding: {sample_out / 'recammaster_camera_embedding.npy'} {camera_embedding.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DVS virtual poses as ReCamMaster camera embeddings.")
    parser.add_argument("--config", default="./conf/stabilzation.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="./test/dvs_recammaster_condition")
    parser.add_argument(
        "--from_virtual_queue_dir",
        default=None,
        help="Re-export embeddings from an existing DVS output directory without running DVS inference.",
    )
    parser.add_argument(
        "--rotation_mode",
        choices=["inverse", "as_is"],
        default="inverse",
        help=(
            "How to convert DVS quaternion rotations before ReCamMaster embedding. "
            "'inverse' transposes the DVS rotation matrix and is the corrected default; "
            "'as_is' reproduces the earlier legacy output."
        ),
    )
    parser.add_argument("--sample_mode", choices=["stride", "uniform"], default="stride")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--camera_stride", type=int, default=4)
    parser.add_argument("--pose_count", type=int, default=21)
    parser.add_argument("--cuda_devices", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cf = load_config(args.config)
    use_cuda = bool(cf["data"]["use_cuda"]) and not args.cpu and torch.cuda.is_available()
    if cf["data"]["use_cuda"] and not use_cuda:
        print("CUDA requested by config but unavailable or disabled; using CPU.")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_virtual_queue_dir is not None:
        reexport_saved_virtual_queue(
            input_dir=Path(args.from_virtual_queue_dir),
            output_dir=output_dir,
            sample_mode=args.sample_mode,
            num_frames=args.num_frames,
            camera_stride=args.camera_stride,
            pose_count=args.pose_count,
            start_frame=args.start_frame,
            rotation_mode=args.rotation_mode,
        )
        return

    model = load_model(cf, use_cuda=use_cuda)
    sample_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found under {data_dir}")

    for data_path in sample_dirs:
        export_sample(
            model=model,
            cf=cf,
            data_path=data_path,
            output_dir=output_dir,
            use_cuda=use_cuda,
            sample_mode=args.sample_mode,
            num_frames=args.num_frames,
            camera_stride=args.camera_stride,
            pose_count=args.pose_count,
            start_frame=args.start_frame,
            rotation_mode=args.rotation_mode,
        )


if __name__ == "__main__":
    main()
