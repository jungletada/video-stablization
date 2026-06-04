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
