"""Camera trajectory utilities for ReCamMaster stabilization probes.

The helpers in this file intentionally avoid heavyweight dependencies.  They
read the camera format used by ReCamMaster, generate conservative smoothed
target trajectories, and convert them to the 21 x 12 camera embedding expected
by the current ReCamMaster inference code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_matrix(matrix_str: str) -> np.ndarray:
    rows = matrix_str.strip().split("] [")
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append(list(map(float, row.split())))
    return np.array(matrix, dtype=np.float64)


def _frame_index(frame_key: str) -> int:
    return int(frame_key.replace("frame", ""))


def project_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vt
    return projected


def load_recammaster_json_c2w(
    pose_file: str | Path,
    cam_key: str = "cam01",
    translation_scale: float = 100.0,
) -> np.ndarray:
    """Load ReCamMaster camera_extrinsics.json as raw c2w matrices.

    ReCamMaster's original inference code applies an additional axis conversion
    right before pose embedding.  We intentionally do not apply that conversion
    here, because smoothing should happen in the physically continuous pose
    space.  Use convert_to_recammaster_camera_axes before creating embeddings.
    """

    pose_file = Path(pose_file)
    with pose_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if cam_key.isdigit():
        cam_key = f"cam{int(cam_key):02d}"

    frame_keys = sorted(data.keys(), key=_frame_index)
    matrices = []
    for frame_key in frame_keys:
        if cam_key not in data[frame_key]:
            raise KeyError(f"{cam_key} not found in {pose_file} at {frame_key}")
        matrices.append(parse_matrix(data[frame_key][cam_key]))

    c2ws = np.stack(matrices, axis=0).transpose(0, 2, 1).astype(np.float64)
    loaded = []
    for c2w in c2ws:
        c2w = c2w.copy()
        if translation_scale != 0:
            c2w[:3, 3] /= translation_scale
        c2w[:3, :3] = project_rotation(c2w[:3, :3])
        loaded.append(c2w)
    return np.asarray(loaded, dtype=np.float64)


def convert_to_recammaster_camera_axes(c2ws: np.ndarray) -> np.ndarray:
    """Apply the axis conversion used by the original ReCamMaster inference code."""

    converted = []
    for c2w in c2ws:
        c2w = c2w.copy()
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        converted.append(c2w)
    return np.asarray(converted, dtype=np.float64)


def load_matrix_sequence(
    pose_file: str | Path,
    pose_format: str = "recammaster_json",
    cam_key: str = "cam01",
    matrix_convention: str = "c2w",
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Load a sequence of 4x4 matrices.

    Supported formats:
    - recammaster_json: frameN -> camXX -> matrix string, as in this repo.
    - npy: numpy array with shape [F, 4, 4].
    - json_list: JSON list with shape [F, 4, 4].
    """

    pose_file = Path(pose_file)
    if pose_format == "recammaster_json":
        return load_recammaster_json_c2w(
            pose_file, cam_key=cam_key, translation_scale=translation_scale
        )

    if pose_format == "npy":
        c2ws = np.load(pose_file).astype(np.float64)
    elif pose_format == "json_list":
        with pose_file.open("r", encoding="utf-8") as f:
            c2ws = np.asarray(json.load(f), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported pose_format: {pose_format}")

    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        raise ValueError(f"Expected [F, 4, 4] matrices, got {c2ws.shape}")

    if matrix_convention == "w2c":
        c2ws = np.linalg.inv(c2ws)
    elif matrix_convention != "c2w":
        raise ValueError(f"Unsupported matrix_convention: {matrix_convention}")

    if translation_scale != 0:
        c2ws = c2ws.copy()
        c2ws[:, :3, 3] /= translation_scale

    for i in range(len(c2ws)):
        c2ws[i, :3, :3] = project_rotation(c2ws[i, :3, :3])
    return c2ws


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return quaternion as [w, x, y, z]."""

    r = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ],
            dtype=np.float64,
        )

    diag = np.diag(r)
    idx = int(np.argmax(diag))
    if idx == 0:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = np.array(
            [
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif idx == 1:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = np.array(
            [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = np.array(
            [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return normalize_quaternion(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return normalize_quaternion(q0 + amount * (q1 - q0))
    theta_0 = math.acos(dot)
    theta = theta_0 * amount
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return normalize_quaternion((s0 * q0) + (s1 * q1))


def _continuous_quaternions(c2ws: np.ndarray) -> np.ndarray:
    quats = np.stack([rotation_matrix_to_quaternion(m[:3, :3]) for m in c2ws])
    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] *= -1.0
    return quats


def _odd_window(window: int, n: int) -> int:
    window = max(1, min(int(window), int(n)))
    if window % 2 == 0:
        window -= 1
    return max(1, window)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    window = _odd_window(window, n)
    if window <= 1:
        return values.copy()
    half = window // 2
    padded = np.pad(values, [(half, half)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    smoothed = []
    for i in range(n):
        smoothed.append(padded[i : i + window].mean(axis=0))
    return np.asarray(smoothed, dtype=np.float64)


def smooth_c2w_moving_average(c2ws: np.ndarray, window: int = 9) -> np.ndarray:
    quats = _continuous_quaternions(c2ws)
    translations = c2ws[:, :3, 3]
    q_smooth = moving_average(quats, window)
    t_smooth = moving_average(translations, window)

    out = c2ws.copy()
    for i in range(len(out)):
        out[i, :3, :3] = quaternion_to_rotation_matrix(q_smooth[i])
        out[i, :3, 3] = t_smooth[i]
    return out


def smooth_c2w_ema(c2ws: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    quats = _continuous_quaternions(c2ws)
    translations = c2ws[:, :3, 3]
    q_out = [quats[0]]
    t_out = [translations[0]]
    for i in range(1, len(c2ws)):
        q_out.append(slerp(q_out[-1], quats[i], alpha))
        t_out.append((1.0 - alpha) * t_out[-1] + alpha * translations[i])

    out = c2ws.copy()
    for i, (q, t) in enumerate(zip(q_out, t_out)):
        out[i, :3, :3] = quaternion_to_rotation_matrix(q)
        out[i, :3, 3] = t
    return out


def smooth_c2w_savgol(c2ws: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return smooth_c2w_moving_average(c2ws, window=window)

    n = len(c2ws)
    window = _odd_window(window, n)
    if window <= polyorder:
        return smooth_c2w_moving_average(c2ws, window=window)

    quats = _continuous_quaternions(c2ws)
    translations = c2ws[:, :3, 3]
    q_smooth = savgol_filter(quats, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    t_smooth = savgol_filter(
        translations, window_length=window, polyorder=polyorder, axis=0, mode="interp"
    )

    out = c2ws.copy()
    for i in range(len(out)):
        out[i, :3, :3] = quaternion_to_rotation_matrix(q_smooth[i])
        out[i, :3, 3] = t_smooth[i]
    return out


def smooth_c2w(
    c2ws: np.ndarray,
    method: str = "moving_average",
    window: int = 9,
    ema_alpha: float = 0.18,
    polyorder: int = 2,
) -> np.ndarray:
    if method == "moving_average":
        return smooth_c2w_moving_average(c2ws, window=window)
    if method == "ema":
        return smooth_c2w_ema(c2ws, alpha=ema_alpha)
    if method == "savgol":
        return smooth_c2w_savgol(c2ws, window=window, polyorder=polyorder)
    raise ValueError(f"Unsupported smoothing method: {method}")


def shrink_motion_from_first_pose(
    c2ws: np.ndarray,
    rotation_scale: float = 0.15,
    translation_scale: float = 0.15,
) -> np.ndarray:
    """Create a near-static trajectory by shrinking motion around frame 0."""

    first = c2ws[0]
    first_inv = np.linalg.inv(first)
    out = []
    identity_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for c2w in c2ws:
        rel = first_inv @ c2w
        rel_q = rotation_matrix_to_quaternion(rel[:3, :3])
        rel_q = slerp(identity_q, rel_q, rotation_scale)
        rel_t = rel[:3, 3] * translation_scale

        rel_shrunk = np.eye(4, dtype=np.float64)
        rel_shrunk[:3, :3] = quaternion_to_rotation_matrix(rel_q)
        rel_shrunk[:3, 3] = rel_t
        out.append(first @ rel_shrunk)
    return np.asarray(out, dtype=np.float64)


def sample_cameras(c2ws: np.ndarray, num_frames: int = 81, stride: int = 4) -> Tuple[np.ndarray, List[int]]:
    if len(c2ws) < num_frames:
        raise ValueError(f"Need at least {num_frames} camera poses, got {len(c2ws)}")
    frame_indices = list(range(num_frames))[::stride]
    return c2ws[frame_indices], frame_indices


def c2w_to_recammaster_embedding(c2ws: np.ndarray) -> np.ndarray:
    """Convert sampled c2w matrices to ReCamMaster's [N, 12] relative pose embedding."""

    anchor_inv = np.linalg.inv(c2ws[0])
    relative = []
    for c2w in c2ws:
        rel = anchor_inv @ c2w
        relative.append(rel[:3, :].reshape(-1))
    return np.asarray(relative, dtype=np.float32)


def build_stabilization_variants(
    c2ws: np.ndarray,
    smooth_method: str = "moving_average",
    smooth_window: int = 9,
    slow_window: int = 21,
    ema_alpha: float = 0.18,
    savgol_polyorder: int = 2,
    slow_rotation_scale: float = 0.15,
    slow_translation_scale: float = 0.15,
) -> Dict[str, np.ndarray]:
    raw = c2ws.copy()
    smooth = smooth_c2w(
        c2ws,
        method=smooth_method,
        window=smooth_window,
        ema_alpha=ema_alpha,
        polyorder=savgol_polyorder,
    )
    heavily_smooth = smooth_c2w(
        c2ws,
        method=smooth_method,
        window=slow_window,
        ema_alpha=ema_alpha,
        polyorder=savgol_polyorder,
    )
    slow_static = shrink_motion_from_first_pose(
        heavily_smooth,
        rotation_scale=slow_rotation_scale,
        translation_scale=slow_translation_scale,
    )
    return {"raw": raw, "smooth": smooth, "slow_static": slow_static}


def relative_rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    cos_angle = (np.trace(rel) - 1.0) * 0.5
    cos_angle = float(min(max(cos_angle, -1.0), 1.0))
    return math.degrees(math.acos(cos_angle))


def trajectory_summary(c2ws: np.ndarray) -> Dict[str, float]:
    rot_steps = np.array(
        [relative_rotation_angle_deg(c2ws[i - 1], c2ws[i]) for i in range(1, len(c2ws))],
        dtype=np.float64,
    )
    translations = c2ws[:, :3, 3]
    trans_steps = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    rot_accel = np.diff(rot_steps) if len(rot_steps) > 1 else np.array([0.0])
    trans_accel = np.diff(trans_steps) if len(trans_steps) > 1 else np.array([0.0])
    return {
        "frames": int(len(c2ws)),
        "rotation_step_mean_deg": float(rot_steps.mean()) if len(rot_steps) else 0.0,
        "rotation_step_std_deg": float(rot_steps.std()) if len(rot_steps) else 0.0,
        "rotation_accel_std_deg": float(rot_accel.std()) if len(rot_accel) else 0.0,
        "translation_step_mean": float(trans_steps.mean()) if len(trans_steps) else 0.0,
        "translation_step_std": float(trans_steps.std()) if len(trans_steps) else 0.0,
        "translation_accel_std": float(trans_accel.std()) if len(trans_accel) else 0.0,
    }


def write_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
