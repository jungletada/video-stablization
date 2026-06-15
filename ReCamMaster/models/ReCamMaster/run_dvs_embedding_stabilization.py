"""Run ReCamMaster using DVS-exported camera embeddings.

The script pairs each video under ../data with the matching camera embedding
exported by dvs/run_dvs_to_recammaster.py and generates a stabilized video.
For Wan/ReCamMaster, num_frames must be 4n + 1 and the camera embedding length
must be (num_frames - 1) // 4 + 1.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

THIS_DIR = Path(__file__).resolve().parent
RECAMMASTER_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = RECAMMASTER_ROOT.parent
sys.path.insert(0, str(RECAMMASTER_ROOT))
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from run_stabilization_experiment import (
    NEGATIVE_PROMPT,
    load_pipeline,
    print_startup_diagnostics,
    resolve_torch_dtype,
)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def recammaster_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return RECAMMASTER_ROOT / path


def sample_dirs(data_dir: Path, selected_samples: Iterable[str] | None = None) -> list[Path]:
    selected = set(selected_samples or [])
    dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if selected:
        dirs = [p for p in dirs if p.name in selected]
    if not dirs:
        raise FileNotFoundError(f"No sample directories found in {data_dir}")
    return dirs


def find_video(sample_dir: Path) -> Path:
    preferred = sample_dir / f"{sample_dir.name}.mp4"
    if preferred.exists():
        return preferred
    candidates = sorted(sample_dir.glob("*.mp4"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one mp4 in {sample_dir}, found {len(candidates)}")
    return candidates[0]


def load_source_video(video_path: Path, args):
    import imageio
    import torch
    import torchvision
    from einops import rearrange
    from PIL import Image
    from torchvision.transforms import v2

    frame_process = v2.Compose(
        [
            v2.CenterCrop(size=(args.height, args.width)),
            v2.Resize(size=(args.height, args.width), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def crop_and_resize(image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = max(args.width / width, args.height / height)
        return torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )

    reader = imageio.get_reader(str(video_path))
    try:
        frame_count = reader.count_frames()
        last_needed = args.start_frame + (args.num_frames - 1) * args.frame_interval
        if frame_count <= last_needed:
            raise ValueError(
                f"{video_path} has {frame_count} frames, but frame {last_needed} is required."
            )

        frames = []
        for frame_id in range(args.num_frames):
            source_id = args.start_frame + frame_id * args.frame_interval
            frame = Image.fromarray(reader.get_data(source_id))
            frame = frame_process(crop_and_resize(frame))
            frames.append(frame)
    finally:
        reader.close()

    video = torch.stack(frames, dim=0)
    video = rearrange(video, "T C H W -> 1 C T H W")
    return video


def expected_pose_count(num_frames: int) -> int:
    if num_frames % 4 != 1:
        raise ValueError(
            f"--num_frames must satisfy num_frames % 4 == 1 for ReCamMaster, got {num_frames}."
        )
    return (num_frames - 1) // 4 + 1


def load_camera_embedding(embedding_path: Path, torch_dtype, pose_count: int, num_frames: int):
    import torch

    embedding = np.load(embedding_path).astype(np.float32)
    expected_shape = (pose_count, 12)
    if embedding.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape[0]} x 12 camera embedding at {embedding_path}, "
            f"got {embedding.shape}. Re-export it with --num_frames {num_frames} "
            f"--pose_count {pose_count}."
        )
    return torch.as_tensor(embedding, dtype=torch_dtype).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ReCamMaster with DVS camera embeddings")
    parser.add_argument("--data_dir", default="data", help="Project-relative data directory")
    parser.add_argument(
        "--embedding_dir",
        default="dvs/test/dvs_recammaster_condition",
        help="Project-relative directory containing per-sample recammaster_camera_embedding.npy",
    )
    parser.add_argument("--output_dir", default="./results/dvs_recammaster_stabilization")
    parser.add_argument("--samples", default=None, help="Comma-separated sample names. Defaults to all.")
    parser.add_argument(
        "--prompt",
        default="A handheld action camera video, stabilized camera motion, realistic details.",
    )
    parser.add_argument("--ckpt_path", default="./models/ReCamMaster/checkpoints/step20000.ckpt")
    parser.add_argument("--dit_path", default="./models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--text_encoder_path", default="./models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", default="./models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_devices", default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--enable_vram_management", action="store_true")
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true", help="Only validate input pairs and camera shapes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_devices is None and args.num_gpus is not None:
        if args.num_gpus < 1:
            raise ValueError("--num_gpus must be >= 1")
        args.cuda_devices = ",".join(str(i) for i in range(args.num_gpus))
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        print(f"CUDA_VISIBLE_DEVICES={args.cuda_devices}")
        if args.device == "cuda":
            args.device = "cuda:0"

    print_startup_diagnostics(args)
    os.chdir(RECAMMASTER_ROOT)

    import torch
    from diffsynth import save_video

    data_dir = project_path(args.data_dir)
    embedding_dir = project_path(args.embedding_dir)
    output_dir = recammaster_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [s.strip() for s in args.samples.split(",")] if args.samples else None
    samples = sample_dirs(data_dir, selected)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    pose_count = expected_pose_count(args.num_frames)

    print("Input pairs")
    pairs = []
    for sample_dir in samples:
        video_path = find_video(sample_dir)
        embedding_path = embedding_dir / sample_dir.name / "recammaster_camera_embedding.npy"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing camera embedding: {embedding_path}")
        embedding = np.load(embedding_path)
        expected_shape = (pose_count, 12)
        if embedding.shape != expected_shape:
            raise ValueError(
                f"Expected {expected_shape[0]} x 12 camera embedding at {embedding_path}, "
                f"got {embedding.shape}. Re-export DVS with --num_frames {args.num_frames} "
                f"--pose_count {pose_count}."
            )
        output_path = output_dir / f"{sample_dir.name}_dvs_recammaster_stabilized.mp4"
        pairs.append((sample_dir.name, video_path, embedding_path, output_path))
        print(f"  {sample_dir.name}")
        print(f"    video: {video_path}")
        print(f"    camera: {embedding_path} shape={embedding.shape} dtype={embedding.dtype}")
        print(f"    num_frames: {args.num_frames}, camera poses: {pose_count}")
        print(f"    output: {output_path}")

    if args.dry_run:
        print("Dry run complete. Skipping model loading and video generation.")
        return

    pipe = load_pipeline(args)

    for index, (name, video_path, embedding_path, output_path) in enumerate(pairs):
        print(f"\nRunning {index + 1}/{len(pairs)}: {name}")
        source_video = load_source_video(video_path, args)
        target_camera = load_camera_embedding(embedding_path, torch_dtype, pose_count, args.num_frames)
        video = pipe(
            prompt=[args.prompt],
            negative_prompt=NEGATIVE_PROMPT,
            source_video=source_video,
            target_camera=target_camera,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            tiled=True,
        )
        save_video(video, str(output_path), fps=args.fps, quality=args.quality)
        print(f"Saved {output_path}")

        del video, source_video, target_camera
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished. Stabilized videos are under {output_dir}")


if __name__ == "__main__":
    main()
