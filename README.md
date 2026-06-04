# Video Stabilization with ReCamMaster

本项目用于探索一种生成式视频稳定方案：先根据已有相机位姿 `R,t` 生成更平滑的目标相机轨迹，再把这条轨迹作为条件输入到 ReCamMaster，让模型按新相机轨迹重新渲染源视频。

## 核心想法

传统 video stabilization 通常通过估计相机运动、平滑相机路径、再对原视频做裁剪和 warping 来减少抖动。优点是保真，缺点是容易裁剪画面，遇到大运动、视差、边缘空洞时效果受限。

ReCamMaster 的思路不同。它不是直接对原视频做几何变换，而是把源视频、文本描述和目标相机轨迹输入到视频生成模型中，生成一段“同一动态场景在新相机轨迹下拍摄”的视频。因此，用 ReCamMaster 做稳定时，本质是：

```text
抖动源视频 + 平滑目标相机轨迹 -> 生成式重拍摄后的稳定视频
```

这带来两个重要特点：

- 它可以避免传统稳定中的黑边和大幅裁剪问题，因为模型可以生成新可见区域。
- 它不是逐像素保真的稳定器，可能引入生成式伪影，例如手部错误、纹理漂移、身份不一致或时序闪烁。

因此，ReCamMaster 更适合探索“创作型视频稳定”或“镜头重拍摄”，而不是严格保真的司法/测量类视频修复。

## ReCamMaster 在稳定任务中的输入和输出

如果目标是 video stabilization，可以把 ReCamMaster 看成一个条件视频生成器。

### 输入

| 输入 | 形状/格式 | 作用 |
| --- | --- | --- |
| 源视频 `source_video` | 默认 81 帧，推理时处理为 `C x T x H x W` | 需要被稳定或重拍摄的原始视频 |
| 文本描述 `prompt` | 字符串 | 描述源视频内容，帮助生成模型保持语义和外观 |
| 目标相机轨迹 `target_camera` | 当前代码中为 `21 x 12` | 控制输出视频采用什么相机运动 |
| 模型权重 | Wan2.1 T2V 权重 + ReCamMaster checkpoint | 生成视频所需的基础模型和相机控制模块 |

其中 `target_camera` 是最关键的稳定条件。当前 ReCamMaster 代码会从 81 帧相机轨迹中每 4 帧采样一次：

```text
frame 0, 4, 8, ..., 80 -> 共 21 个相机位姿
```

每个相机位姿是一个相对 `3 x 4` 外参矩阵，展平成 12 维，所以最终输入为：

```text
21 x 12
```

如果我们已经有每帧旋转和平移矩阵 `R,t`，则稳定实验的关键步骤是：

```text
原始每帧 R,t
-> 低通/EMA/Savitzky-Golay 平滑
-> 得到平滑目标相机轨迹
-> 转成 ReCamMaster 需要的 21 x 12 camera embedding
-> 输入 ReCamMaster
```

### 输出

| 输出 | 格式 | 含义 |
| --- | --- | --- |
| 生成视频 | `.mp4` | 按目标平滑相机轨迹重新生成的视频 |
| 相机条件文件 | `.npy` | 当前实验轨迹对应的 camera embedding |
| 轨迹统计 | `.json` | 平滑前后旋转/平移步长与加速度统计 |

当前默认只跑一组稳定实验，也就是 `smooth`：

```text
smooth/video0.mp4
```

可选轨迹含义：

- `raw`：原始相机轨迹，作为对照组。
- `smooth`：简单平滑后的相机轨迹，目标是减少抖动但保留原始运镜趋势。默认运行这一组。
- `slow_static`：强平滑并压缩运动幅度，近似静止或缓慢运动镜头。

## 低风险实验脚本

实验代码位于：

```text
ReCamMaster/models/ReCamMaster/
```

关键文件：

- `trajectory_utils.py`：读取相机位姿、平滑轨迹、生成 camera embedding。
- `run_stabilization_experiment.py`：生成目标轨迹并调用 ReCamMaster 推理。默认只运行 `smooth`。
- `run_stabilization_single.sh`：服务器单次运行脚本，指定一张 GPU 和一组轨迹。
- `run_recammaster_smoke_test.py`：最小推理测试，只验证 ReCamMaster 是否能成功生成一个 mp4。
- `run_recammaster_smoke_test.sh`：服务器 smoke test 脚本，指定一张 GPU 并保存日志。
- `STABILIZATION_PROBE.md`：更详细的运行说明。

## 最小推理 Smoke Test

官方仓库的示例测试是：

```bash
python inference_recammaster.py --cam_type 1
```

这个命令会按官方 demo 配置跑完整视频推理，默认 81 帧、480x832、50 个 denoise steps。它适合最终测试，但不适合先排查环境，因为在没有高效 attention 后端时很容易 OOM。

本项目新增了更小的 smoke test，只生成 1 帧、128x224、1 个 denoise step、`cfg_scale=1.0`。它的目的不是评估画质，而是验证：

- Wan2.1 和 ReCamMaster checkpoint 能正确加载。
- VAE、text encoder、DiT 和 camera condition 能跑通。
- pipeline 能成功输出 mp4。

服务器上建议先跑：

```bash
cd ReCamMaster
bash models/ReCamMaster/run_recammaster_smoke_test.sh 0
```

V100 不支持原生 BF16 Tensor Core，因此 smoke/full-shape 测试脚本默认使用：

```text
--torch_dtype float16
```

如果在 3090/A100 等 BF16 支持更好的卡上测试，也可以覆盖成：

```bash
bash models/ReCamMaster/run_recammaster_smoke_test.sh 0 --torch_dtype bfloat16
```

输出位置：

```text
results/recammaster_smoke_test/smooth/video0.mp4
logs/recammaster_smoke_test_gpu0.log
```

如果 smoke test 成功，再逐步增加配置，例如：

```bash
bash models/ReCamMaster/run_recammaster_smoke_test.sh 0 \
  --height 256 \
  --width 448 \
  --num_inference_steps 4
```

如果要测试 ReCamMaster 官方尺寸：

```text
height = 480
width = 832
num_frames = 81
```

先跑 full-shape 但低步数的测试：

```bash
cd ReCamMaster
bash models/ReCamMaster/run_recammaster_full_shape_test.sh 0
```

它默认仍然只跑：

```text
num_inference_steps = 1
cfg_scale = 1.0
torch_dtype = float16
```

输出位置：

```text
results/recammaster_full_shape_test/smooth/video0.mp4
logs/recammaster_full_shape_test_gpu0.log
```

先只验证相机轨迹和 embedding，不加载大模型：

```bash
cd ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json \
  --source_cam cam01 \
  --dry_run
```

服务器上模型权重准备好后，运行完整推理：

```bash
cd ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json \
  --source_cam cam01 \
  --output_dir ./results/stabilization_probe \
  --variant smooth \
  --smooth_method moving_average \
  --smooth_window 9 \
  --slow_window 21
```

如果服务器上遇到 CUDA out of memory，优先启用低显存模式：

```bash
cd ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json \
  --source_cam cam01 \
  --output_dir ./results/stabilization_probe \
  --variant smooth \
  --smooth_method moving_average \
  --smooth_window 9 \
  --slow_window 21 \
  --enable_vram_management
```

也可以指定当前进程可见的 GPU 数量：

```bash
cd ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --num_gpus 8 \
  --device cuda \
  --enable_vram_management \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

或显式指定 GPU id：

```bash
cd ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --cuda_devices 0,1,2,3,4,5,5,6,7 \
  --device cuda \
  --enable_vram_management \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

注意：`--cuda_devices 0,1` 只是控制进程能看到哪些物理 GPU。当前 ReCamMaster pipeline 没有做模型并行，单次推理仍会跑在一个逻辑设备上，默认是 `cuda:0`。如果是单次推理 OOM，真正有效的选项通常是 `--enable_vram_management`、降低 `--height/--width`，或者减少 `--num_inference_steps`。

如果日志里出现类似：

```text
torch.cuda.OutOfMemoryError: Tried to allocate 95.96 GiB. GPU 0 ...
```

这通常发生在 DiT self-attention，而不是模型权重加载阶段。`--enable_vram_management` 可以减少权重占用，但不能减少单次 attention 的峰值激活显存。建议先用较低分辨率测试：

```bash
cd ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --variant smooth \
  --enable_vram_management \
  --height 384 \
  --width 672 \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

如果仍然 OOM，可以继续降到：

```bash
--height 256 --width 448
```

服务器上推荐用单次脚本指定一张 GPU 和一组轨迹：

```bash
cd ReCamMaster
bash models/ReCamMaster/run_stabilization_single.sh 0 smooth \
  --enable_vram_management \
  --height 384 \
  --width 672 \
  --dataset_path ./example_test_data \
  --pose_file ./example_test_data/cameras/camera_extrinsics.json
```

这只会运行 `smooth` 一组实验，并把日志写到 `logs/stabilization_smooth_gpu0.log`。

如果使用自己的每帧 `R,t`，并且已经保存为 `F x 4 x 4` 的 c2w 矩阵：

```bash
cd ReCamMaster
python models/ReCamMaster/run_stabilization_experiment.py \
  --dataset_path /path/to/your/test_data \
  --pose_file /path/to/your/c2w_poses.npy \
  --pose_format npy \
  --matrix_convention c2w \
  --translation_scale 1.0 \
  --no-apply-recammaster-axis-transform
```

如果你的矩阵是 w2c：

```bash
--matrix_convention w2c
```

## 如何评价稳定结果

建议先观察 `smooth` 输出，重点看：

- 稳定性：全局抖动是否减少。
- 身份保持：人物、物体、场景是否和源视频一致。
- 动作同步：人物动作和物体运动是否仍与源视频同步。
- 边缘补全：新露出的画面边缘是否自然。
- 生成伪影：是否出现手部错误、脸部变形、纹理漂移、闪烁或重复肢体。

如果 `smooth` 的全局抖动明显减少，同时身份和动作同步保持较好，说明“平滑目标相机轨迹 + ReCamMaster 重拍摄”的方向可继续推进。

如果需要对照，再分别运行 `--variant raw` 或 `--variant slow_static`。如果 `slow_static` 稳定但内容明显漂移，说明目标轨迹与源视频差异过大，后续应限制虚拟相机轨迹不要偏离原始轨迹太远。

## 项目结构

```text
.
├── Paper/                  # 本地论文资料，不同步到 GitHub
├── ReCamMaster/            # ReCamMaster 代码和稳定实验脚本
├── deep-stabilization/     # Deep-FVS 相关代码
└── README.md               # 当前项目说明
```

`Paper/`、模型权重、checkpoint 和生成结果已通过 `.gitignore` 排除。
