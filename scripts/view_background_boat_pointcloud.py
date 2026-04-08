from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from scripts.view_pointcloud import (
    apply_rotation,
    axis_limits,
    camera_eye_for_view,
    rgb_strings,
    sample_points,
)


def _load_cloud(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if colors.shape != points.shape:
        colors = np.full_like(points, 0.8)
    return sample_points(points, colors, max_points=max_points, seed=seed)


def _make_trace(points: np.ndarray, colors: np.ndarray, name: str, visible: bool | str, size: float) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        visible=visible,
        marker={
            "size": size,
            "opacity": 0.9,
            "color": rgb_strings(colors),
        },
        hoverinfo="skip",
    )


def export_background_boat_html(
    background_path: Path,
    boat_path: Path,
    output_path: Path,
    *,
    background_max_points: int = 70000,
    boat_max_points: int = 35000,
    near_clip_percentiles: tuple[float, float] = (35.0, 60.0),
    seed: int = 0,
    default_view: str = "side",
    default_mode: str = "both_far",
    title: str | None = None,
    rotate_deg_x: float = 0.0,
    rotate_deg_y: float = 0.0,
    rotate_deg_z: float = 0.0,
) -> Path:
    bg_points, bg_colors = _load_cloud(background_path, background_max_points, seed)
    boat_points, boat_colors = _load_cloud(boat_path, boat_max_points, seed + 1)
    if len(bg_points) == 0 and len(boat_points) == 0:
        raise RuntimeError("No points found for semantic viewer.")

    def to_view(points: np.ndarray) -> np.ndarray:
        pts = np.column_stack([points[:, 0], points[:, 2], -points[:, 1]]) if len(points) else np.zeros((0, 3))
        return apply_rotation(pts, rotate_deg_x, rotate_deg_y, rotate_deg_z)

    bg_view = to_view(bg_points)
    boat_view = to_view(boat_points)

    distances = np.linalg.norm(bg_points, axis=1) if len(bg_points) else np.zeros((0,), dtype=np.float32)
    mid_far_threshold = float(np.percentile(distances, near_clip_percentiles[0])) if len(distances) else 0.0
    far_only_threshold = float(np.percentile(distances, near_clip_percentiles[1])) if len(distances) else 0.0
    bg_mid_far_mask = distances >= mid_far_threshold if len(distances) else np.zeros((0,), dtype=bool)
    bg_far_only_mask = distances >= far_only_threshold if len(distances) else np.zeros((0,), dtype=bool)

    # Tint boat points warm so they remain visually distinct from the world map.
    if len(boat_colors):
        tint = np.array([1.0, 0.55, 0.15], dtype=np.float32)
        boat_colors = np.clip(0.35 * boat_colors + 0.65 * tint, 0.0, 1.0)

    all_for_limits = []
    if len(bg_view):
        all_for_limits.append(bg_view)
    if len(boat_view):
        all_for_limits.append(boat_view)
    limits = axis_limits(np.concatenate(all_for_limits, axis=0))

    visibility = {
        "background_mid_far": [True, "legendonly", "legendonly", "legendonly"],
        "background_all": ["legendonly", True, "legendonly", "legendonly"],
        "background_far": ["legendonly", "legendonly", True, "legendonly"],
        "boat_only": ["legendonly", "legendonly", "legendonly", True],
        "both_mid_far": [True, "legendonly", "legendonly", True],
        "both_far": ["legendonly", "legendonly", True, True],
    }[default_mode]

    fig = go.Figure()
    fig.add_trace(_make_trace(bg_view[bg_mid_far_mask], bg_colors[bg_mid_far_mask], "Background Mid/Far", visibility[0], 1.0))
    fig.add_trace(_make_trace(bg_view, bg_colors, "Background All", visibility[1], 0.95))
    fig.add_trace(_make_trace(bg_view[bg_far_only_mask], bg_colors[bg_far_only_mask], "Background Far", visibility[2], 1.05))
    fig.add_trace(_make_trace(boat_view, boat_colors, "Boat Local", visibility[3], 1.15))

    mode_visibility = {
        "Background Mid/Far": [True, False, False, False],
        "Background All": [False, True, False, False],
        "Background Far": [False, False, True, False],
        "Boat Only": [False, False, False, True],
        "Both Mid/Far": [True, False, False, True],
        "Both Far": [False, False, True, True],
    }

    fig.update_layout(
        title=title or f"Background + boat viewer: {background_path.name}",
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
        margin={"l": 0, "r": 220, "t": 170, "b": 0},
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
                    {"label": "Isometric", "method": "relayout", "args": [{"scene.camera.eye": camera_eye_for_view("isometric")}]},
                    {"label": "Top", "method": "relayout", "args": [{"scene.camera.eye": camera_eye_for_view("top")}]},
                    {"label": "Side", "method": "relayout", "args": [{"scene.camera.eye": camera_eye_for_view("side")}]},
                    {"label": "Front", "method": "relayout", "args": [{"scene.camera.eye": camera_eye_for_view("front")}]},
                ],
            },
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.38,
                "y": 1.18,
                "showactive": True,
                "bgcolor": "rgba(20,20,20,0.95)",
                "bordercolor": "rgba(255,255,255,0.30)",
                "borderwidth": 2,
                "font": {"size": 17, "color": "#F5F5F5"},
                "buttons": [
                    {"label": label, "method": "update", "args": [{"visible": visible}]}
                    for label, visible in mode_visibility.items()
                ],
            },
        ],
        annotations=[
            {
                "text": "Background is world-aligned. Boat is local to the camera/rig.",
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
                "text": "Use the top buttons to show background, boat, or both without conflating them.",
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
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path
