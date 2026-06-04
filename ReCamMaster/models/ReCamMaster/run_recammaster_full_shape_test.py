"""Full-shape ReCamMaster inference test.

This keeps the official spatial/temporal shape used by ReCamMaster
(`height=480`, `width=832`, `num_frames=81`) while keeping denoising cheap
(`num_inference_steps=1`, `cfg_scale=1.0`).  Use it after the minimal smoke test
passes to check whether the server can handle the official token scale.
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
    "./results/recammaster_full_shape_test",
    "--height",
    "480",
    "--width",
    "832",
    "--num_frames",
    "81",
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
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    run_stabilization_main()

