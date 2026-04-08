from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a .ply point cloud to an interactive HTML viewer.")
    parser.add_argument("--input", required=True, type=str, help="Path to the input .ply point cloud.")
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Path to the output HTML file. Defaults to <input_stem>.html next to the .ply.",
    )
    parser.add_argument(
        "--max_points",
        default=50000,
        type=int,
        help="Maximum number of points to keep for the interactive viewer.",
    )
    parser.add_argument(
        "--near_clip_percentiles",
        default=[35.0, 60.0],
        nargs=2,
        type=float,
        metavar=("MID_FAR", "FAR_ONLY"),
        help="Distance percentiles used for the optional near-point filtering traces.",
    )
    parser.add_argument("--title", default=None, type=str, help="Optional title shown in the HTML viewer.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for point subsampling.")
    parser.add_argument("--rotate_deg_x", default=0.0, type=float, help="Rotate the cloud around X before export.")
    parser.add_argument("--rotate_deg_y", default=0.0, type=float, help="Rotate the cloud around Y before export.")
    parser.add_argument("--rotate_deg_z", default=0.0, type=float, help="Rotate the cloud around Z before export.")
    parser.add_argument(
        "--default_view",
        default="isometric",
        choices=["isometric", "side", "top", "front"],
        help="Initial camera view when the HTML opens.",
    )
    parser.add_argument(
        "--default_trace",
        default="mid_far",
        choices=["mid_far", "all", "far_only"],
        help="Initial point subset shown when the HTML opens.",
    )
    parser.set_defaults(open_html=True)
    parser.add_argument("--open", dest="open_html", action="store_true", help="Open the HTML after export.")
    parser.add_argument("--no-open", dest="open_html", action="store_false", help="Only write the HTML file.")
    return parser.parse_args()


def sample_points(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices], colors[indices]


def rgb_strings(colors: np.ndarray) -> list[str]:
    rgb = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def axis_limits(points_xyz: np.ndarray) -> dict[str, list[float]]:
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins)) / 2.0
    return {
        "x": [float(center[0] - span), float(center[0] + span)],
        "y": [float(center[1] - span), float(center[1] + span)],
        "z": [float(center[2] - span), float(center[2] + span)],
    }


def rotation_matrix_x(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_y(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_z(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def apply_rotation(points_xyz: np.ndarray, rotate_deg_x: float, rotate_deg_y: float, rotate_deg_z: float) -> np.ndarray:
    rotation = rotation_matrix_z(rotate_deg_z) @ rotation_matrix_y(rotate_deg_y) @ rotation_matrix_x(rotate_deg_x)
    return points_xyz @ rotation.T


def camera_eye_for_view(view: str) -> dict[str, float]:
    eyes = {
        "isometric": {"x": 1.6, "y": 1.2, "z": 0.7},
        "side": {"x": 2.3, "y": 0.05, "z": 0.2},
        "top": {"x": 0.01, "y": 0.01, "z": 2.3},
        "front": {"x": 0.05, "y": 2.3, "z": 0.25},
    }
    return eyes[view]


def trace_visibility_for(selection: str) -> list[bool | str]:
    return {
        "mid_far": [True, "legendonly", "legendonly"],
        "all": ["legendonly", True, "legendonly"],
        "far_only": ["legendonly", "legendonly", True],
    }[selection]


def make_trace(
    points: np.ndarray,
    colors: np.ndarray,
    name: str,
    visible: bool | str,
    point_size: float,
) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        visible=visible,
        marker={
            "size": point_size,
            "opacity": 0.9,
            "color": rgb_strings(colors),
        },
        hoverinfo="skip",
    )


def export_html(
    input_path: Path,
    output_path: Path,
    max_points: int,
    near_clip_percentiles: list[float],
    seed: int,
    rotate_deg_x: float,
    rotate_deg_y: float,
    rotate_deg_z: float,
    default_view: str,
    default_trace: str,
    title: str | None,
) -> Path:
    cloud = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if len(points) == 0:
        raise RuntimeError(f"Point cloud is empty: {input_path}")

    if colors.shape != points.shape:
        colors = np.full_like(points, 0.8)

    points, colors = sample_points(points, colors, max_points=max_points, seed=seed)

    # The maritime camera is mounted upside down, so flip the vertical axis for viewing.
    points_xyz = np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])
    points_xyz = apply_rotation(points_xyz, rotate_deg_x, rotate_deg_y, rotate_deg_z)
    distances = np.linalg.norm(points, axis=1)
    mid_far_threshold = float(np.percentile(distances, near_clip_percentiles[0]))
    far_only_threshold = float(np.percentile(distances, near_clip_percentiles[1]))

    mid_far_mask = distances >= mid_far_threshold
    far_only_mask = distances >= far_only_threshold

    limits = axis_limits(points_xyz)
    trace_visibility = trace_visibility_for(default_trace)
    default_trace_label = {
        "mid_far": "Mid/Far",
        "all": "All Points",
        "far_only": "Far Only",
    }[default_trace]
    default_view_label = {
        "isometric": "Isometric",
        "side": "Side",
        "top": "Top",
        "front": "Front",
    }[default_view]

    fig = go.Figure()
    fig.add_trace(make_trace(points_xyz[mid_far_mask], colors[mid_far_mask], "Mid/Far", trace_visibility[0], 1.1))
    fig.add_trace(make_trace(points_xyz, colors, "All Points", trace_visibility[1], 1.0))
    fig.add_trace(make_trace(points_xyz[far_only_mask], colors[far_only_mask], "Far Only", trace_visibility[2], 1.2))

    fig.update_layout(
        title=title or f"Point cloud viewer: {input_path.name}",
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        title_font={"size": 30, "color": "#F3F3F3"},
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 17, "color": "#FFFFFF"},
            "bgcolor": "rgba(0,0,0,0.90)",
            "bordercolor": "rgba(255,255,255,0.35)",
            "borderwidth": 2,
        },
        margin={"l": 0, "r": 210, "t": 170, "b": 0},
        scene={
            "xaxis": {"visible": False, "range": limits["x"]},
            "yaxis": {"visible": False, "range": limits["y"]},
            "zaxis": {"visible": False, "range": limits["z"]},
            "aspectmode": "cube",
            "camera": {"eye": camera_eye_for_view(default_view)},
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.18,
                "showactive": True,
                "bgcolor": "rgba(20,20,20,0.95)",
                "bordercolor": "rgba(255,255,255,0.30)",
                "borderwidth": 2,
                "font": {"size": 17, "color": "#F5F5F5"},
                "buttons": [
                    {
                        "label": "Isometric",
                        "method": "relayout",
                        "args": [{"scene.camera.eye": camera_eye_for_view("isometric")}],
                    },
                    {
                        "label": "Top",
                        "method": "relayout",
                        "args": [{"scene.camera.eye": camera_eye_for_view("top")}],
                    },
                    {
                        "label": "Side",
                        "method": "relayout",
                        "args": [{"scene.camera.eye": camera_eye_for_view("side")}],
                    },
                    {
                        "label": "Front",
                        "method": "relayout",
                        "args": [{"scene.camera.eye": camera_eye_for_view("front")}],
                    },
                ],
            },
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.43,
                "y": 1.18,
                "showactive": True,
                "bgcolor": "rgba(20,20,20,0.95)",
                "bordercolor": "rgba(255,255,255,0.30)",
                "borderwidth": 2,
                "font": {"size": 17, "color": "#F5F5F5"},
                "buttons": [
                    {
                        "label": "Mid/Far",
                        "method": "update",
                        "args": [{"visible": trace_visibility_for("mid_far")}],
                    },
                    {
                        "label": "All Points",
                        "method": "update",
                        "args": [{"visible": trace_visibility_for("all")}],
                    },
                    {
                        "label": "Far Only",
                        "method": "update",
                        "args": [{"visible": trace_visibility_for("far_only")}],
                    },
                ],
            }
        ],
        annotations=[
            {
                "text": f"Opened as: {default_view_label} + {default_trace_label}",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.08,
                "showarrow": False,
                "align": "left",
                "font": {"size": 18, "color": "#141414"},
                "bgcolor": "rgba(255,244,179,0.98)",
                "bordercolor": "rgba(255,255,255,0.55)",
                "borderwidth": 2,
                "borderpad": 8,
            },
            {
                "text": "Top buttons change view and point subset.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.025,
                "showarrow": False,
                "align": "left",
                "font": {"size": 16, "color": "#111111"},
                "bgcolor": "rgba(190,230,255,0.96)",
                "bordercolor": "rgba(255,255,255,0.55)",
                "borderwidth": 2,
                "borderpad": 8,
            },
            {
                "text": "Legend moved to the right. Use Far Only if the mast dominates.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.58,
                "y": 1.08,
                "showarrow": False,
                "align": "left",
                "font": {"size": 16, "color": "#111111"},
                "bgcolor": "rgba(255,209,209,0.96)",
                "bordercolor": "rgba(255,255,255,0.55)",
                "borderwidth": 2,
                "borderpad": 8,
            },
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input point cloud does not exist: {input_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.with_suffix(".html")
    )

    html_path = export_html(
        input_path=input_path,
        output_path=output_path,
        max_points=args.max_points,
        near_clip_percentiles=args.near_clip_percentiles,
        seed=args.seed,
        rotate_deg_x=args.rotate_deg_x,
        rotate_deg_y=args.rotate_deg_y,
        rotate_deg_z=args.rotate_deg_z,
        default_view=args.default_view,
        default_trace=args.default_trace,
        title=args.title,
    )
    print(html_path)

    if args.open_html:
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
