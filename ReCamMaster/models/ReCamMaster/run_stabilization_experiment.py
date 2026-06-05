"""Run a low-risk ReCamMaster stabilization probe.

This script generates three camera-trajectory conditions from an existing
per-frame pose sequence:

1. raw: the original trajectory, useful as a control.
2. smooth: a low-pass/Savitzky-Golay smoothed trajectory.
3. slow_static: a heavily smoothed trajectory with motion shrunk around frame 0.

It then feeds each condition into the existing ReCamMaster inference pipeline.
Use --dry_run to only export camera embeddings and trajectory summaries.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from trajectory_utils import (
    build_stabilization_variants,
    c2w_to_recammaster_embedding,
    convert_to_recammaster_camera_axes,
    load_matrix_sequence,
    sample_cameras,
    trajectory_summary,
    write_json,
)


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_torch_dtype(dtype_name: str):
    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype_map[dtype_name]


def print_startup_diagnostics(args) -> None:
    import torch

    print("Startup diagnostics")
    print(f"  torch: {torch.__version__}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  requested device: {args.device}")
    print(f"  requested torch_dtype: {args.torch_dtype}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

    if not torch.cuda.is_available():
        print("  CUDA available: False")
    else:
        print("  CUDA available: True")
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        try:
            print(f"  torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}")
        except Exception as exc:
            print(f"  torch.cuda.is_bf16_supported(): unavailable ({exc})")
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            name = torch.cuda.get_device_name(index)
            total_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
            print(
                f"  logical cuda:{index}: {name}, "
                f"capability={capability}, total_vram={total_gb:.1f} GiB"
            )

    try:
        from diffsynth.models import wan_video_dit

        if wan_video_dit.FLASH_ATTN_3_AVAILABLE:
            selected_backend = "FlashAttention 3"
        elif wan_video_dit.FLASH_ATTN_2_AVAILABLE:
            selected_backend = "FlashAttention 2"
        elif wan_video_dit.SAGE_ATTN_AVAILABLE:
            selected_backend = "SageAttention"
        else:
            selected_backend = "torch.scaled_dot_product_attention"
        print("  attention backend availability:")
        print(f"    FlashAttention 3: {wan_video_dit.FLASH_ATTN_3_AVAILABLE}")
        print(f"    FlashAttention 2: {wan_video_dit.FLASH_ATTN_2_AVAILABLE}")
        print(f"    SageAttention: {wan_video_dit.SAGE_ATTN_AVAILABLE}")
        print(f"    selected by ReCamMaster: {selected_backend}")
    except Exception as exc:
        print(f"  attention backend diagnostics unavailable: {exc}")


def make_fixed_camera_dataset_class():
    import imageio
    import pandas as pd
    import torch
    import torchvision
    from einops import rearrange
    from PIL import Image
    from torchvision.transforms import v2

    class FixedCameraVideoDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            base_path: str | Path,
            metadata_path: str | Path,
            camera_embedding: np.ndarray,
            camera_dtype,
            max_num_frames: int = 81,
            frame_interval: int = 1,
            num_frames: int = 81,
            height: int = 480,
            width: int = 832,
        ) -> None:
            metadata = pd.read_csv(metadata_path)
            self.path = [str(Path(base_path) / "videos" / file_name) for file_name in metadata["file_name"]]
            self.text = metadata["text"].to_list()
            self.camera_embedding = torch.as_tensor(camera_embedding, dtype=camera_dtype)
            self.max_num_frames = max_num_frames
            self.frame_interval = frame_interval
            self.num_frames = num_frames
            self.height = height
            self.width = width
            self.frame_process = v2.Compose(
                [
                    v2.CenterCrop(size=(height, width)),
                    v2.Resize(size=(height, width), antialias=True),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        def crop_and_resize(self, image: Image.Image) -> Image.Image:
            width, height = image.size
            scale = max(self.width / width, self.height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height * scale), round(width * scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            return image

        def load_frames_using_imageio(self, file_path: str, start_frame_id: int):
            reader = imageio.get_reader(file_path)
            try:
                frame_count = reader.count_frames()
                last_needed = start_frame_id + (self.num_frames - 1) * self.frame_interval
                if frame_count < self.max_num_frames or frame_count - 1 < last_needed:
                    return None

                frames = []
                for frame_id in range(self.num_frames):
                    frame = reader.get_data(start_frame_id + frame_id * self.frame_interval)
                    frame = Image.fromarray(frame)
                    frame = self.crop_and_resize(frame)
                    frame = self.frame_process(frame)
                    frames.append(frame)
                frames = torch.stack(frames, dim=0)
                return rearrange(frames, "T C H W -> C T H W")
            finally:
                reader.close()

        def load_video(self, file_path: str):
            max_start = self.max_num_frames - (self.num_frames - 1) * self.frame_interval
            start_frame_id = int(torch.randint(0, max_start, (1,))[0])
            return self.load_frames_using_imageio(file_path, start_frame_id)

        def __getitem__(self, data_id: int) -> Dict:
            text = self.text[data_id]
            path = self.path[data_id]
            video = self.load_video(path)
            if video is None:
                raise ValueError(f"{path} is not a valid video with {self.num_frames} frames.")
            return {
                "text": text,
                "video": video,
                "path": path,
                "camera": self.camera_embedding,
            }

        def __len__(self) -> int:
            return len(self.path)

    return FixedCameraVideoDataset


def build_and_export_embeddings(args) -> Dict[str, np.ndarray]:
    c2ws = load_matrix_sequence(
        repo_path(args.pose_file),
        pose_format=args.pose_format,
        cam_key=args.source_cam,
        matrix_convention=args.matrix_convention,
        translation_scale=args.translation_scale,
    )

    variants = build_stabilization_variants(
        c2ws,
        smooth_method=args.smooth_method,
        smooth_window=args.smooth_window,
        slow_window=args.slow_window,
        ema_alpha=args.ema_alpha,
        savgol_polyorder=args.savgol_polyorder,
        slow_rotation_scale=args.slow_rotation_scale,
        slow_translation_scale=args.slow_translation_scale,
    )

    out_dir = repo_path(args.output_dir)
    camera_dir = out_dir / "camera_variants"
    camera_dir.mkdir(parents=True, exist_ok=True)

    selected = [v.strip() for v in args.variants.split(",") if v.strip()]
    embeddings = {}
    summary = {
        "pose_file": str(repo_path(args.pose_file)),
        "pose_format": args.pose_format,
        "source_cam": args.source_cam,
        "num_frames": args.num_frames,
        "camera_stride": args.camera_stride,
        "apply_recammaster_axis_transform": args.apply_recammaster_axis_transform,
        "variants": {},
    }

    for name in selected:
        if name not in variants:
            raise ValueError(f"Unknown variant {name}. Available: {sorted(variants)}")
        sampled_c2ws, frame_indices = sample_cameras(
            variants[name], num_frames=args.num_frames, stride=args.camera_stride
        )
        sampled_embedding_c2ws = (
            convert_to_recammaster_camera_axes(sampled_c2ws)
            if args.apply_recammaster_axis_transform
            else sampled_c2ws
        )
        embedding = c2w_to_recammaster_embedding(sampled_embedding_c2ws)
        embeddings[name] = embedding
        np.save(camera_dir / f"{name}_sampled_c2w.npy", sampled_c2ws.astype(np.float32))
        np.save(
            camera_dir / f"{name}_sampled_embedding_c2w.npy",
            sampled_embedding_c2ws.astype(np.float32),
        )
        np.save(camera_dir / f"{name}_camera_embedding.npy", embedding)
        summary["variants"][name] = {
            "frame_indices": frame_indices,
            "embedding_shape": list(embedding.shape),
            "trajectory_summary_full": trajectory_summary(variants[name]),
            "trajectory_summary_sampled": trajectory_summary(sampled_c2ws),
        }

    write_json(camera_dir / "trajectory_summary.json", summary)
    return embeddings


def load_pipeline(args):
    import torch
    import torch.nn as nn
    from diffsynth import ModelManager, WanVideoReCamMasterPipeline

    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    print(f"Using torch dtype: {torch_dtype}")
    model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
    model_manager.load_models(
        [
            str(repo_path(args.dit_path)),
            str(repo_path(args.text_encoder_path)),
            str(repo_path(args.vae_path)),
        ]
    )
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=args.device)

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(12, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    state_dict = torch.load(repo_path(args.ckpt_path), map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)
    if args.enable_vram_management:
        pipe.to(dtype=torch_dtype)
        pipe.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit
        )
    else:
        pipe.to(args.device)
        pipe.to(dtype=torch_dtype)
    return pipe


def run_inference(args, embeddings: Dict[str, np.ndarray]) -> None:
    import torch
    from diffsynth import save_video

    if torch.cuda.is_available():
        print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
        current_device = torch.device(args.device)
        if current_device.type == "cuda":
            index = torch.cuda.current_device() if current_device.index is None else current_device.index
            print(f"Using logical CUDA device {index}: {torch.cuda.get_device_name(index)}")

    pipe = load_pipeline(args)
    output_dir = repo_path(args.output_dir)
    dataset_path = repo_path(args.dataset_path)
    metadata_path = dataset_path / "metadata.csv"
    FixedCameraVideoDataset = make_fixed_camera_dataset_class()
    camera_dtype = resolve_torch_dtype(args.torch_dtype)

    for variant_name, embedding in embeddings.items():
        variant_dir = output_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        dataset = FixedCameraVideoDataset(
            dataset_path,
            metadata_path,
            embedding,
            camera_dtype,
            max_num_frames=args.max_num_frames,
            frame_interval=args.frame_interval,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.dataloader_num_workers,
        )

        for batch_idx, batch in enumerate(dataloader):
            target_text = batch["text"]
            source_video = batch["video"]
            target_camera = batch["camera"]
            video = pipe(
                prompt=target_text,
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
            save_video(video, str(variant_dir / f"video{batch_idx}.mp4"), fps=args.fps, quality=5)

            del video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="ReCamMaster low-risk stabilization probe")
    parser.add_argument("--dataset_path", default="./example_test_data", help="Dataset with videos/ and metadata.csv")
    parser.add_argument("--pose_file", default="./example_test_data/cameras/camera_extrinsics.json")
    parser.add_argument(
        "--pose_format",
        choices=["recammaster_json", "npy", "json_list"],
        default="recammaster_json",
    )
    parser.add_argument("--source_cam", default="cam01", help="Camera key for recammaster_json, e.g. cam01 or 1")
    parser.add_argument("--matrix_convention", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument(
        "--translation_scale",
        type=float,
        default=100.0,
        help="Divide loaded translations by this value. ReCamMaster JSON uses 100.",
    )
    parser.add_argument(
        "--apply_recammaster_axis_transform",
        "--apply-recammaster-axis-transform",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="apply_recammaster_axis_transform",
        help="Apply the axis conversion used by inference_recammaster.py before embedding.",
    )

    parser.add_argument("--output_dir", default="./results/stabilization_probe")
    parser.add_argument(
        "--variant",
        choices=["raw", "smooth", "slow_static"],
        default=None,
        help="Run a single trajectory variant. Defaults to smooth.",
    )
    parser.add_argument(
        "--variants",
        default="smooth",
        help="Comma-separated variants to run. Kept for compatibility; default is smooth only.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only export camera embeddings and summaries")

    parser.add_argument("--smooth_method", choices=["moving_average", "ema", "savgol"], default="moving_average")
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--slow_window", type=int, default=21)
    parser.add_argument("--ema_alpha", type=float, default=0.18)
    parser.add_argument("--savgol_polyorder", type=int, default=2)
    parser.add_argument("--slow_rotation_scale", type=float, default=0.15)
    parser.add_argument("--slow_translation_scale", type=float, default=0.15)

    parser.add_argument("--ckpt_path", default="./models/ReCamMaster/checkpoints/step20000.ckpt")
    parser.add_argument("--dit_path", default="./models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--text_encoder_path", default="./models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", default="./models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch_dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Computation dtype. Use float16 on V100; bfloat16 is better for Ampere/A100-class GPUs.",
    )
    parser.add_argument(
        "--cuda_devices",
        default=None,
        help="Comma-separated physical GPU ids to expose, e.g. 0 or 0,1. Must be set before torch loads CUDA.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Convenience option: expose GPUs 0..N-1. Ignored if --cuda_devices is set.",
    )
    parser.add_argument(
        "--enable_vram_management",
        action="store_true",
        help="Enable CPU/GPU offload in ReCamMaster to reduce peak CUDA memory.",
    )
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        help="With VRAM management, keep this many DiT params persistent on GPU; lower uses less VRAM.",
    )

    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_num_frames", type=int, default=81)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--camera_stride", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.variant is not None:
        args.variants = args.variant
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
    os.chdir(REPO_ROOT)
    if args.apply_recammaster_axis_transform is None:
        args.apply_recammaster_axis_transform = args.pose_format == "recammaster_json"
    embeddings = build_and_export_embeddings(args)
    print(f"Exported camera variants to {repo_path(args.output_dir) / 'camera_variants'}")
    for name, embedding in embeddings.items():
        print(f"  {name}: camera embedding shape {tuple(embedding.shape)}")

    if args.dry_run:
        print("Dry run complete. Skipping model loading and video generation.")
        return

    run_inference(args, embeddings)
    print(f"Finished. Videos are under {repo_path(args.output_dir)}")


if __name__ == "__main__":
    main()
