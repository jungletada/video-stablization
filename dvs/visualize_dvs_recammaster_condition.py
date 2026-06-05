"""Visualize DVS virtual camera queues and ReCamMaster camera embeddings.

The preferred renderer is Matplotlib. If Matplotlib is not installed, the script
falls back to plain SVG and optionally Pillow PNG output.
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np


BLUE = "#2F6BBA"
ORANGE = "#A45A00"
OLIVE = "#5B7F25"
PINK = "#8E3A59"
GRID = "#E6E8EB"
INK = "#222831"
MUTED = "#68707A"


def quaternion_angle_deg(quat_xyzw):
    quat = quat_xyzw.astype(np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    w = np.clip(np.abs(quat[:, 3]), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def embedding_rotation_angle_deg(embedding):
    mats = embedding.reshape(-1, 3, 4)[:, :3, :3]
    trace = np.trace(mats, axis1=1, axis2=2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def load_sample(sample_dir):
    virtual_queue = np.load(sample_dir / "virtual_queue.npy")
    embedding = np.load(sample_dir / "recammaster_camera_embedding.npy")
    summary_path = sample_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return virtual_queue, embedding, summary


def prepare_sample(sample_dir):
    virtual_queue, embedding, summary = load_sample(sample_dir)
    t = virtual_queue[:, 0].astype(np.float64)
    t_sec = (t - t[0]) / 1e9
    quat = virtual_queue[:, 1:5]
    sample_indices = np.asarray(summary.get("sample_indices", list(range(embedding.shape[0]))), dtype=np.int64)
    sample_indices = np.clip(sample_indices, 0, len(t_sec) - 1)

    return {
        "name": sample_dir.name,
        "virtual_queue": virtual_queue,
        "embedding": embedding,
        "t_sec": t_sec,
        "quat": quat,
        "sample_indices": sample_indices,
        "sampled_t_sec": t_sec[sample_indices],
        "quat_angle": quaternion_angle_deg(quat),
        "embedding_angle": embedding_rotation_angle_deg(embedding),
        "embedding_delta": embedding - embedding[0:1],
    }


def render_sample_matplotlib(sample_dir, output_dir, save_pdf=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = prepare_sample(sample_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13.6, 11.2),
        gridspec_kw={"height_ratios": [2.3, 1.4, 1.4, 2.1]},
        constrained_layout=True,
    )
    fig.suptitle(data["name"], fontsize=15, fontweight="bold")

    colors = [BLUE, ORANGE, OLIVE, PINK]
    for i, (label, color) in enumerate(zip(["qx", "qy", "qz", "qw"], colors)):
        axes[0].plot(data["t_sec"], data["quat"][:, i], label=label, color=color, linewidth=1.35)
    for idx in data["sample_indices"]:
        axes[0].axvline(data["t_sec"][idx], color="#A0A7B0", linewidth=0.5, alpha=0.32)
    axes[0].set_title("DVS virtual_queue quaternion over time", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Quaternion")
    axes[0].legend(ncol=4, frameon=False, loc="upper right")

    axes[1].plot(data["t_sec"], data["quat_angle"], color=BLUE, linewidth=1.55)
    axes[1].scatter(
        data["sampled_t_sec"],
        data["quat_angle"][data["sample_indices"]],
        s=24,
        color=ORANGE,
        zorder=3,
        label="21 sampled poses",
    )
    axes[1].set_title("Virtual camera rotation angle from identity", loc="left", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Degrees")

    pose_id = np.arange(data["embedding"].shape[0])
    axes[2].plot(pose_id, data["embedding_angle"], marker="o", color=OLIVE, linewidth=1.55, markersize=4)
    axes[2].set_title("ReCamMaster sampled 21-pose relative rotation angle", loc="left", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Pose index")
    axes[2].set_ylabel("Degrees")
    axes[2].set_xticks(pose_id)

    vmax = max(float(np.max(np.abs(data["embedding_delta"][:, :9]))), 1e-6)
    im = axes[3].imshow(
        data["embedding_delta"],
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[3].set_title("21 x 12 ReCamMaster embedding delta from first pose", loc="left", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Flattened 3 x 4 camera embedding dimension")
    axes[3].set_ylabel("Pose index")
    axes[3].set_xticks(np.arange(12))
    axes[3].set_yticks(pose_id)
    fig.colorbar(im, ax=axes[3], orientation="vertical", fraction=0.025, pad=0.01, label="Delta")

    for ax in axes[:3]:
        ax.grid(True, color=GRID, linewidth=0.8)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_paths = []
    png_path = output_dir / f"{data['name']}_dvs_recammaster_condition.png"
    fig.savefig(png_path, dpi=180)
    output_paths.append(png_path)
    if save_pdf:
        pdf_path = output_dir / f"{data['name']}_dvs_recammaster_condition.pdf"
        fig.savefig(pdf_path)
        output_paths.append(pdf_path)
    plt.close(fig)
    return output_paths


def normalize_range(values, low=None, high=None):
    values = np.asarray(values, dtype=np.float64)
    if low is None:
        low = float(np.nanmin(values))
    if high is None:
        high = float(np.nanmax(values))
    if abs(high - low) < 1e-12:
        pad = max(abs(high) * 0.05, 1e-3)
        low -= pad
        high += pad
    return low, high


def sx(x, x_min, x_max, left, width):
    return left + (np.asarray(x) - x_min) / (x_max - x_min) * width


def sy(y, y_min, y_max, top, height):
    return top + height - (np.asarray(y) - y_min) / (y_max - y_min) * height


def polyline(x, y, color, stroke=1.8):
    points = " ".join(f"{float(a):.2f},{float(b):.2f}" for a, b in zip(x, y))
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke}" />'


def text(x, y, value, size=13, color=INK, anchor="start", weight="400"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}">{html.escape(str(value))}</text>'
    )


def axes(title, left, top, width, height, x_label, y_label, y_min, y_max, x_min, x_max):
    parts = [
        text(left, top - 12, title, size=15, weight="700"),
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="{MUTED}" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="{MUTED}" />',
        text(left + width / 2, top + height + 42, x_label, size=12, color=MUTED, anchor="middle"),
    ]
    for i in range(5):
        yy = top + i * height / 4
        value = y_max - i * (y_max - y_min) / 4
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + width}" y2="{yy:.1f}" stroke="{GRID}" />')
        parts.append(text(left - 10, yy + 4, f"{value:.3g}", size=11, color=MUTED, anchor="end"))
    for i in range(6):
        xx = left + i * width / 5
        value = x_min + i * (x_max - x_min) / 5
        parts.append(f'<line x1="{xx:.1f}" y1="{top + height}" x2="{xx:.1f}" y2="{top + height + 5}" stroke="{MUTED}" />')
        parts.append(text(xx, top + height + 22, f"{value:.2g}", size=11, color=MUTED, anchor="middle"))
    return parts


def diverging_color(value, vmax):
    ratio = float(np.clip(value / max(vmax, 1e-12), -1, 1))
    if ratio >= 0:
        start = np.array([247, 248, 250])
        end = np.array([180, 70, 55])
        color = start * (1 - ratio) + end * ratio
    else:
        ratio = abs(ratio)
        start = np.array([247, 248, 250])
        end = np.array([47, 107, 186])
        color = start * (1 - ratio) + end * ratio
    return "#%02x%02x%02x" % tuple(np.round(color).astype(int))


def render_sample_svg(sample_dir, output_dir):
    virtual_queue, embedding, summary = load_sample(sample_dir)
    name = sample_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    t = virtual_queue[:, 0].astype(np.float64)
    t_sec = (t - t[0]) / 1e9
    quat = virtual_queue[:, 1:5]
    sample_indices = np.asarray(summary.get("sample_indices", list(range(embedding.shape[0]))), dtype=np.int64)
    sample_indices = np.clip(sample_indices, 0, len(t_sec) - 1)
    sampled_t_sec = t_sec[sample_indices]

    quat_angle = quaternion_angle_deg(quat)
    embedding_angle = embedding_rotation_angle_deg(embedding)
    embedding_delta = embedding - embedding[0:1]

    width, height = 1240, 1040
    left, plot_width = 96, 1050
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF" />',
        text(40, 42, name, size=20, weight="700"),
        text(40, 66, "DVS virtual_queue and ReCamMaster 21 x 12 camera embedding", size=13, color=MUTED),
    ]

    x_min, x_max = normalize_range(t_sec, 0, float(t_sec[-1]))
    q_min, q_max = normalize_range(quat)
    parts.extend(axes("DVS virtual_queue quaternion over time", left, 112, plot_width, 210, "Time since first frame (s)", "Quaternion", q_min, q_max, x_min, x_max))
    for idx in sample_indices:
        xx = sx(t_sec[idx], x_min, x_max, left, plot_width)
        parts.append(f'<line x1="{xx:.1f}" y1="112" x2="{xx:.1f}" y2="322" stroke="#A0A7B0" stroke-width="0.5" opacity="0.35" />')
    for i, (label, color) in enumerate(zip(["qx", "qy", "qz", "qw"], [BLUE, ORANGE, OLIVE, PINK])):
        x = sx(t_sec, x_min, x_max, left, plot_width)
        y = sy(quat[:, i], q_min, q_max, 112, 210)
        parts.append(polyline(x, y, color))
        parts.append(f'<rect x="{left + 820 + i * 58}" y="91" width="12" height="3" fill="{color}" />')
        parts.append(text(left + 837 + i * 58, 96, label, size=12, color=MUTED))

    a_min, a_max = normalize_range(quat_angle, 0, None)
    parts.extend(axes("Virtual camera rotation angle from identity", left, 402, plot_width, 150, "Time since first frame (s)", "Degrees", a_min, a_max, x_min, x_max))
    parts.append(polyline(sx(t_sec, x_min, x_max, left, plot_width), sy(quat_angle, a_min, a_max, 402, 150), BLUE, stroke=2.0))
    for xx, yy in zip(sx(sampled_t_sec, x_min, x_max, left, plot_width), sy(quat_angle[sample_indices], a_min, a_max, 402, 150)):
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.4" fill="{ORANGE}" />')

    pose = np.arange(embedding.shape[0])
    e_min, e_max = normalize_range(embedding_angle, 0, None)
    parts.extend(axes("Sampled 21-pose relative rotation angle", left, 632, plot_width, 130, "Pose index", "Degrees", e_min, e_max, 0, 20))
    parts.append(polyline(sx(pose, 0, 20, left, plot_width), sy(embedding_angle, e_min, e_max, 632, 130), OLIVE, stroke=2.0))
    for xx, yy in zip(sx(pose, 0, 20, left, plot_width), sy(embedding_angle, e_min, e_max, 632, 130)):
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.4" fill="{OLIVE}" />')

    heat_left, heat_top = left, 842
    cell_w, cell_h = plot_width / 12, 150 / embedding.shape[0]
    vmax = max(float(np.max(np.abs(embedding_delta[:, :9]))), 1e-6)
    parts.append(text(heat_left, heat_top - 16, "21 x 12 embedding delta from first pose", size=15, weight="700"))
    for row in range(embedding_delta.shape[0]):
        for col in range(embedding_delta.shape[1]):
            color = diverging_color(embedding_delta[row, col], vmax)
            parts.append(
                f'<rect x="{heat_left + col * cell_w:.1f}" y="{heat_top + row * cell_h:.1f}" '
                f'width="{cell_w + 0.4:.1f}" height="{cell_h + 0.4:.1f}" fill="{color}" />'
            )
    parts.append(f'<rect x="{heat_left}" y="{heat_top}" width="{plot_width}" height="150" fill="none" stroke="{MUTED}" />')
    for col in range(12):
        parts.append(text(heat_left + (col + 0.5) * cell_w, heat_top + 174, str(col), size=11, color=MUTED, anchor="middle"))
    for row in range(0, 21, 5):
        parts.append(text(heat_left - 10, heat_top + (row + 0.7) * cell_h, str(row), size=11, color=MUTED, anchor="end"))
    parts.append(text(heat_left + plot_width / 2, heat_top + 196, "Flattened 3 x 4 camera embedding dimension", size=12, color=MUTED, anchor="middle"))
    parts.append(text(heat_left - 55, heat_top + 80, "Pose index", size=12, color=MUTED, anchor="middle"))
    parts.append(text(heat_left + plot_width + 14, heat_top + 6, f"+{vmax:.3g}", size=11, color=MUTED))
    parts.append(text(heat_left + plot_width + 14, heat_top + 78, "0", size=11, color=MUTED))
    parts.append(text(heat_left + plot_width + 14, heat_top + 150, f"-{vmax:.3g}", size=11, color=MUTED))

    parts.append("</svg>")
    output_path = output_dir / f"{name}_dvs_recammaster_condition.svg"
    output_path.write_text("\n".join(parts))
    return output_path


def render_sample_png(sample_dir, output_dir):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    virtual_queue, embedding, summary = load_sample(sample_dir)
    name = sample_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = 2
    width, height = 1240, 1040
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image)

    def font(size, bold=False):
        names = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"]
        for font_name in names:
            try:
                return ImageFont.truetype(font_name, size * scale)
            except OSError:
                pass
        return ImageFont.load_default()

    def xy(point):
        return tuple(int(round(v * scale)) for v in point)

    def draw_text(x, y, value, size=13, fill=INK, anchor="la", bold=False):
        draw.text(xy((x, y)), str(value), fill=fill, font=font(size, bold), anchor=anchor)

    def draw_line(points, fill, width_px=2):
        pts = [xy(point) for point in points]
        draw.line(pts, fill=fill, width=width_px * scale, joint="curve")

    def draw_axes(title, left, top, plot_width, plot_height, x_label, y_label, y_min, y_max, x_min, x_max):
        draw_text(left, top - 12, title, size=15, bold=True)
        draw.line([xy((left, top + plot_height)), xy((left + plot_width, top + plot_height))], fill=MUTED, width=scale)
        draw.line([xy((left, top)), xy((left, top + plot_height))], fill=MUTED, width=scale)
        draw_text(left + plot_width / 2, top + plot_height + 42, x_label, size=12, fill=MUTED, anchor="ma")
        for i in range(5):
            yy = top + i * plot_height / 4
            value = y_max - i * (y_max - y_min) / 4
            draw.line([xy((left, yy)), xy((left + plot_width, yy))], fill=GRID, width=scale)
            draw_text(left - 10, yy, f"{value:.3g}", size=11, fill=MUTED, anchor="rm")
        for i in range(6):
            xx = left + i * plot_width / 5
            value = x_min + i * (x_max - x_min) / 5
            draw.line([xy((xx, top + plot_height)), xy((xx, top + plot_height + 5))], fill=MUTED, width=scale)
            draw_text(xx, top + plot_height + 22, f"{value:.2g}", size=11, fill=MUTED, anchor="ma")

    t = virtual_queue[:, 0].astype(np.float64)
    t_sec = (t - t[0]) / 1e9
    quat = virtual_queue[:, 1:5]
    sample_indices = np.asarray(summary.get("sample_indices", list(range(embedding.shape[0]))), dtype=np.int64)
    sample_indices = np.clip(sample_indices, 0, len(t_sec) - 1)
    sampled_t_sec = t_sec[sample_indices]

    quat_angle = quaternion_angle_deg(quat)
    embedding_angle = embedding_rotation_angle_deg(embedding)
    embedding_delta = embedding - embedding[0:1]

    left, plot_width = 96, 1050
    draw_text(40, 42, name, size=20, bold=True)
    draw_text(40, 66, "DVS virtual_queue and ReCamMaster 21 x 12 camera embedding", size=13, fill=MUTED)

    x_min, x_max = normalize_range(t_sec, 0, float(t_sec[-1]))
    q_min, q_max = normalize_range(quat)
    draw_axes("DVS virtual_queue quaternion over time", left, 112, plot_width, 210, "Time since first frame (s)", "Quaternion", q_min, q_max, x_min, x_max)
    for idx in sample_indices:
        xx = float(sx(t_sec[idx], x_min, x_max, left, plot_width))
        draw.line([xy((xx, 112)), xy((xx, 322))], fill="#A0A7B0", width=scale)
    for i, (label, color) in enumerate(zip(["qx", "qy", "qz", "qw"], [BLUE, ORANGE, OLIVE, PINK])):
        xs = sx(t_sec, x_min, x_max, left, plot_width)
        ys = sy(quat[:, i], q_min, q_max, 112, 210)
        draw_line(list(zip(xs, ys)), color, width_px=2)
        draw.rectangle([xy((left + 820 + i * 58, 90)), xy((left + 832 + i * 58, 94))], fill=color)
        draw_text(left + 837 + i * 58, 96, label, size=12, fill=MUTED)

    a_min, a_max = normalize_range(quat_angle, 0, None)
    draw_axes("Virtual camera rotation angle from identity", left, 402, plot_width, 150, "Time since first frame (s)", "Degrees", a_min, a_max, x_min, x_max)
    draw_line(list(zip(sx(t_sec, x_min, x_max, left, plot_width), sy(quat_angle, a_min, a_max, 402, 150))), BLUE, width_px=2)
    for xx, yy in zip(sx(sampled_t_sec, x_min, x_max, left, plot_width), sy(quat_angle[sample_indices], a_min, a_max, 402, 150)):
        draw.ellipse([xy((xx - 3.4, yy - 3.4)), xy((xx + 3.4, yy + 3.4))], fill=ORANGE)

    pose = np.arange(embedding.shape[0])
    e_min, e_max = normalize_range(embedding_angle, 0, None)
    draw_axes("Sampled 21-pose relative rotation angle", left, 632, plot_width, 130, "Pose index", "Degrees", e_min, e_max, 0, 20)
    xs = sx(pose, 0, 20, left, plot_width)
    ys = sy(embedding_angle, e_min, e_max, 632, 130)
    draw_line(list(zip(xs, ys)), OLIVE, width_px=2)
    for xx, yy in zip(xs, ys):
        draw.ellipse([xy((xx - 3.4, yy - 3.4)), xy((xx + 3.4, yy + 3.4))], fill=OLIVE)

    heat_left, heat_top = left, 842
    cell_w, cell_h = plot_width / 12, 150 / embedding.shape[0]
    vmax = max(float(np.max(np.abs(embedding_delta[:, :9]))), 1e-6)
    draw_text(heat_left, heat_top - 16, "21 x 12 embedding delta from first pose", size=15, bold=True)
    for row in range(embedding_delta.shape[0]):
        for col in range(embedding_delta.shape[1]):
            color = diverging_color(embedding_delta[row, col], vmax)
            draw.rectangle(
                [xy((heat_left + col * cell_w, heat_top + row * cell_h)),
                 xy((heat_left + (col + 1) * cell_w, heat_top + (row + 1) * cell_h))],
                fill=color,
            )
    draw.rectangle([xy((heat_left, heat_top)), xy((heat_left + plot_width, heat_top + 150))], outline=MUTED, width=scale)
    for col in range(12):
        draw_text(heat_left + (col + 0.5) * cell_w, heat_top + 174, str(col), size=11, fill=MUTED, anchor="ma")
    for row in range(0, 21, 5):
        draw_text(heat_left - 10, heat_top + (row + 0.7) * cell_h, str(row), size=11, fill=MUTED, anchor="rm")
    draw_text(heat_left + plot_width / 2, heat_top + 196, "Flattened 3 x 4 camera embedding dimension", size=12, fill=MUTED, anchor="ma")

    output_path = output_dir / f"{name}_dvs_recammaster_condition.png"
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./test/dvs_recammaster_condition")
    parser.add_argument("--output_dir", default="./test/dvs_recammaster_condition_visualizations")
    parser.add_argument(
        "--renderer",
        choices=["auto", "matplotlib", "fallback"],
        default="auto",
        help="auto prefers Matplotlib and falls back to SVG/Pillow when Matplotlib is unavailable.",
    )
    parser.add_argument("--save_pdf", action="store_true", help="Also save a PDF when using Matplotlib.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sample_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found under {input_dir}")

    for sample_dir in sample_dirs:
        rendered = False
        if args.renderer in ("auto", "matplotlib"):
            try:
                for path in render_sample_matplotlib(sample_dir, output_dir, save_pdf=args.save_pdf):
                    print(path)
                rendered = True
            except ImportError:
                if args.renderer == "matplotlib":
                    raise

        if not rendered:
            print(render_sample_svg(sample_dir, output_dir))
            png_path = render_sample_png(sample_dir, output_dir)
            if png_path is not None:
                print(png_path)


if __name__ == "__main__":
    main()
