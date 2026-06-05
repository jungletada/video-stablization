"""ReCamMaster inference test with official demo-scale parameters.

This mirrors the original ReCamMaster demo scale:
`height=480`, `width=832`, `num_frames=81`, `num_inference_steps=50`,
`cfg_scale=5.0`, `seed=0`, `fps=30`, and BF16 inference.
"""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_stabilization_experiment import main as run_stabilization_main


DEFAULT_ARGS = [
    "--variant",
    "raw",
    "--output_dir",
    "./results/recammaster_test",
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
    "50",
    "--cfg_scale",
    "5.0",
    "--seed",
    "0",
    "--fps",
    "30",
    "--torch_dtype",
    "bfloat16",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    run_stabilization_main()
