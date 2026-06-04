# ReCamMaster Stabilization Probe

This folder adds a low-risk experiment for using a smoothed camera trajectory as
the target condition for ReCamMaster.

## What It Tests

For one source video and one existing per-frame camera trajectory, the script
generates three target trajectories:

- `raw`: original camera trajectory. This is the control group.
- `smooth`: low-pass/Savitzky-Golay smoothed trajectory.
- `slow_static`: heavily smoothed trajectory with motion shrunk around frame 0.

Each trajectory is converted to the current ReCamMaster camera condition shape:
`21 x 12`, representing 21 sampled `3 x 4` relative camera poses.

## Dry Run

Run this first on any machine. It does not load the video generation model.

```bash
cd /path/to/ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json \
  --source_cam cam01 \
  --dry_run
```

Outputs:

```text
results/stabilization_probe/camera_variants/
  raw_camera_embedding.npy
  smooth_camera_embedding.npy
  slow_static_camera_embedding.npy
  raw_sampled_c2w.npy
  smooth_sampled_c2w.npy
  slow_static_sampled_c2w.npy
  trajectory_summary.json
```

## Full Inference

After checkpoints are ready on the server:

```bash
cd /path/to/ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json \
  --source_cam cam01 \
  --output_dir ./results/stabilization_probe \
  --smooth_method moving_average \
  --smooth_window 9 \
  --slow_window 21
```

If CUDA runs out of memory, enable the built-in VRAM management mode:

```bash
python models/ReCamMaster/run_stabilization_experiment.py \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json \
  --source_cam cam01 \
  --output_dir ./results/stabilization_probe \
  --smooth_method moving_average \
  --smooth_window 9 \
  --slow_window 21 \
  --enable_vram_management
```

To expose only a subset of GPUs:

```bash
python models/ReCamMaster/run_stabilization_experiment.py \
  --num_gpus 1 \
  --device cuda \
  --enable_vram_management \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

or explicitly:

```bash
python models/ReCamMaster/run_stabilization_experiment.py \
  --cuda_devices 0,1 \
  --device cuda \
  --enable_vram_management \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

Note: `--cuda_devices 0,1` controls which physical GPUs are visible to the
process. The current ReCamMaster pipeline still runs one inference on one
logical device (`cuda:0`) unless the underlying model is modified for model
parallelism. For a single-video OOM, `--enable_vram_management`, smaller
`--height/--width`, or fewer `--num_inference_steps` are the practical fixes.

If the log says a single attention call tried to allocate tens of GiB, for
example `Tried to allocate 95.96 GiB. GPU 0 ...`, the failure is in DiT
self-attention activations. CPU/GPU offload reduces model weight memory, but it
does not reduce that attention activation peak. Start with a lower resolution:

```bash
python models/ReCamMaster/run_stabilization_experiment.py \
  --variants smooth \
  --enable_vram_management \
  --height 384 \
  --width 672 \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

If needed, try `--height 256 --width 448`.

To run variants on different GPUs in separate processes:

```bash
bash models/ReCamMaster/run_stabilization_multi_gpu.sh 0,1,2 \
  --enable_vram_management \
  --height 384 \
  --width 672 \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

This runs `raw`, `smooth`, and `slow_static` on different GPU processes. It is
not model parallelism; each single inference still uses one GPU.

For quick debugging, run only one trajectory variant:

```bash
python models/ReCamMaster/run_stabilization_experiment.py \
  --variants smooth \
  --enable_vram_management \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

Generated videos are saved as:

```text
results/stabilization_probe/raw/video0.mp4
results/stabilization_probe/smooth/video0.mp4
results/stabilization_probe/slow_static/video0.mp4
```

## Using Your Own R,t

If your poses are already an array of `F x 4 x 4` matrices:

```bash
python models/ReCamMaster/run_stabilization_experiment.py \
  --dataset_path /path/to/your/test_data \
  --pose_file /path/to/your/c2w_poses.npy \
  --pose_format npy \
  --matrix_convention c2w \
  --translation_scale 1.0 \
  --no-apply-recammaster-axis-transform
```

If the matrices are world-to-camera, use:

```bash
--matrix_convention w2c
```

The script expects at least 81 poses by default and samples frames
`0, 4, 8, ..., 80`, matching the current ReCamMaster inference code.

For ReCamMaster's native `camera_extrinsics.json`, the script smooths poses in
the raw continuous c2w coordinate system first, then applies the same axis
conversion used by `inference_recammaster.py` before creating the camera
embedding. For your own `.npy` or `json_list` matrices, only enable
`--apply-recammaster-axis-transform` if they follow the same raw UE-style camera
format as the released dataset.

## What To Inspect

Compare `raw`, `smooth`, and `slow_static` on:

- Stability: whether global shake is reduced.
- Identity/content preservation: whether the subject and scene remain consistent.
- Motion synchronization: whether body/object motion still matches the source.
- Edge completion: whether newly visible borders look plausible.
- Generation artifacts: hands, faces, duplicated limbs, texture drift, flicker.
