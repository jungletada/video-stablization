"""Minimal ReCamMaster inference smoke test.

This is intentionally much smaller than the official demo command.  It runs a
single-frame, low-resolution, one-step generation to verify that the downloaded
Wan2.1 and ReCamMaster checkpoints can load and that the pipeline can produce an
mp4 without exhausting GPU memory.
"""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_stabilization_experiment import main as run_stabilization_main


DEFAULT_ARGS = [
    "--variant",
    "smooth",
    "--output_dir",
    "./results/recammaster_smoke_test",
    "--height",
    "128",
    "--width",
    "224",
    "--num_frames",
    "1",
    "--max_num_frames",
    "81",
    "--camera_stride",
    "4",
    "--num_inference_steps",
    "1",
    "--cfg_scale",
    "1.0",
    "--dataloader_num_workers",
    "0",
    "--enable_vram_management",
]


if __name__ == "__main__":
    # User-provided CLI args come after defaults, so simple argparse options such
    # as --height or --num_inference_steps can override this smoke-test preset.
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    run_stabilization_main()

