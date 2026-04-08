from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.maritime_da360 import (  # noqa: E402
    build_erp_remap,
    build_fisheye_to_erp_remap,
    detect_fisheye_geometry,
    erp_to_pointcloud_arrays,
    save_circle_preview,
)
from scripts.view_background_boat_pointcloud import export_background_boat_html  # noqa: E402
from scripts.view_pointcloud import export_html  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate a denser maritime point cloud across a short time window.")
    parser.add_argument(
        "--backend",
        default="da360",
        choices=["da360", "dap"],
        help="Per-frame depth backend used before aggregation.",
    )
    parser.add_argument("--input", required=True, type=str, help="Path to the maritime fisheye video.")
    parser.add_argument("--model_path", required=True, type=str, help="Model checkpoint path.")
    parser.add_argument("--model_name", default="DA360_large", type=str, help="Label used in outputs.")
    parser.add_argument("--net", default=None, type=str, help="Override network architecture if needed.")
    parser.add_argument("--device", default=None, type=str, help="cuda, cpu, or leave unset for auto.")
    parser.add_argument("--erp_height", default=None, type=int, help="ERP height override.")
    parser.add_argument("--erp_width", default=None, type=int, help="ERP width override.")
    parser.add_argument(
        "--dap_root",
        default=str(ROOT.parents[0] / "DAP"),
        type=str,
        help="Path to the official DAP repo checkout when --backend dap is used.",
    )
    parser.add_argument("--midas_model_type", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--fine_tune_type", default="hypersim", type=str)
    parser.add_argument("--min_depth", default=0.01, type=float)
    parser.add_argument("--max_depth", default=1.0, type=float)
    parser.add_argument("--output_dir", default=None, type=str, help="Output directory.")
    parser.add_argument("--timestamp_sec", default=20.0, type=float, help="Center timestamp for aggregation.")
    parser.add_argument("--window_sec", default=1.0, type=float, help="Aggregate +/- this many seconds around timestamp.")
    parser.add_argument("--frame_step", default=2, type=int, help="Use every Nth frame within the chosen window.")
    parser.add_argument(
        "--erp_point_stride",
        default=4,
        type=int,
        help="Sample every Nth ERP pixel per axis when building the cloud.",
    )
    parser.add_argument(
        "--drop_near_percentile",
        default=0.0,
        type=float,
        help="Optionally drop the nearest depth percentile from each frame before aggregation.",
    )
    parser.add_argument("--voxel_size", default=0.03, type=float, help="Voxel downsampling size after merging.")
    parser.add_argument("--turntable_seconds", default=10.0, type=float, help="Length of the rotating preview video.")
    parser.add_argument("--turntable_fps", default=20, type=int, help="FPS of the rotating preview video.")
    parser.add_argument(
        "--turntable_drop_near_percentile",
        default=35.0,
        type=float,
        help="Drop nearest points by this percentile for the rotating preview video.",
    )
    parser.add_argument(
        "--viewer_max_points",
        default=80000,
        type=int,
        help="Maximum number of points kept in the interactive HTML viewer.",
    )
    parser.add_argument("--fisheye_fov_deg", default=200.0, type=float, help="Assumed fisheye FOV in degrees.")
    parser.add_argument(
        "--fisheye_model",
        default="equidistant",
        choices=["equidistant", "equisolid", "orthographic", "stereographic"],
        help="Fisheye projection model.",
    )
    parser.add_argument("--black_threshold", default=40, type=int, help="Threshold for the visible fisheye footprint.")
    parser.add_argument("--edge_trim_px", default=4.0, type=float, help="Trim for the detected ellipse boundary.")
    parser.add_argument(
        "--registration_mode",
        default="pairwise_icp",
        choices=["none", "pairwise_icp"],
        help="How to register frames before fusing them into one cloud.",
    )
    parser.add_argument(
        "--registration_pair",
        default="previous",
        choices=["previous", "reference"],
        help="Whether to register each frame against the previous accepted frame or the first/reference frame.",
    )
    parser.add_argument(
        "--registration_far_percentile",
        default=65.0,
        type=float,
        help="Use only points beyond this depth percentile inside the registration band.",
    )
    parser.add_argument(
        "--registration_row_min_frac",
        default=0.20,
        type=float,
        help="Top of the ERP row band used for registration, expressed as a fraction of height.",
    )
    parser.add_argument(
        "--registration_row_max_frac",
        default=0.82,
        type=float,
        help="Bottom of the ERP row band used for registration, expressed as a fraction of height.",
    )
    parser.add_argument(
        "--registration_point_stride",
        default=6,
        type=int,
        help="Sample every Nth ERP pixel per axis when building the registration cloud.",
    )
    parser.add_argument(
        "--registration_voxel_size",
        default=0.06,
        type=float,
        help="Voxel size used for the downsampled ICP clouds.",
    )
    parser.add_argument(
        "--registration_max_corr",
        default=0.0,
        type=float,
        help="ICP maximum correspondence distance. Set 0 to choose automatically from cloud scale.",
    )
    parser.add_argument(
        "--registration_min_fitness",
        default=0.18,
        type=float,
        help="Reject pairwise registrations below this Open3D ICP fitness score.",
    )
    parser.add_argument(
        "--registration_scale_mode",
        default="none",
        choices=["none", "median"],
        help="Whether to estimate an extra per-pair uniform scale from depth statistics before ICP.",
    )
    parser.add_argument(
        "--anchor_point_json",
        default=None,
        type=str,
        help="Optional JSON file with a clicked fisheye pixel used as a known-distance anchor.",
    )
    parser.add_argument(
        "--anchor_distance",
        default=None,
        type=float,
        help="True camera-to-point distance for the anchor point, in the desired world units.",
    )
    parser.add_argument(
        "--anchor_patch_radius",
        default=4,
        type=int,
        help="Half-size of the square patch used to read the predicted depth around the anchor point.",
    )
    parser.add_argument("--boat_mask_video", default=None, type=str, help="Optional video whose white pixels mark boat regions to exclude.")
    parser.add_argument("--water_mask_video", default=None, type=str, help="Optional video whose white pixels mark water regions to exclude.")
    parser.add_argument(
        "--mask_white_threshold",
        default=245,
        type=int,
        help="Pixels brighter than this in the mask videos are treated as excluded.",
    )
    parser.add_argument(
        "--mask_dilate_px",
        default=3,
        type=int,
        help="Dilate extracted exclusion masks by this many pixels after upscaling.",
    )
    parser.add_argument(
        "--max_distance",
        default=0.0,
        type=float,
        help="If >0, exclude background points farther than this metric distance after anchor scaling.",
    )
    parser.add_argument(
        "--boat_max_distance",
        default=0.0,
        type=float,
        help="If >0, exclude boat points farther than this metric distance after anchor scaling.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_backend_api(backend_name: str):
    if backend_name == "dap":
        for module_name in list(sys.modules):
            if module_name == "networks" or module_name.startswith("networks."):
                sys.modules.pop(module_name, None)
    module_name = {
        "da360": "scripts.maritime_da360",
        "dap": "scripts.maritime_dap",
    }[backend_name]
    return importlib.import_module(module_name)


def read_video_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_frame_indices(fps: float, frame_count: int, timestamp_sec: float, window_sec: float, frame_step: int) -> List[int]:
    center_frame = int(round(timestamp_sec * fps))
    start_frame = max(0, int(round((timestamp_sec - window_sec) * fps)))
    end_frame = min(frame_count - 1, int(round((timestamp_sec + window_sec) * fps)))

    indices = list(range(start_frame, end_frame + 1, max(frame_step, 1)))
    if center_frame < frame_count and center_frame not in indices:
        indices.append(center_frame)
    return sorted(set(i for i in indices if 0 <= i < frame_count))


def filter_depth_mask(valid_mask: np.ndarray, depth: np.ndarray, drop_near_percentile: float) -> np.ndarray:
    filtered_mask = valid_mask.copy()
    if drop_near_percentile > 0.0 and np.any(filtered_mask):
        threshold = float(np.percentile(depth[filtered_mask], drop_near_percentile))
        filtered_mask &= depth >= threshold
    return filtered_mask


def cloud_from_arrays(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def make_uniform_scale_matrix(scale: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= float(scale)
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    homog = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    transformed = homog @ matrix.T
    return transformed[:, :3].astype(np.float32)


def load_anchor_point(path: Path | None) -> Dict[str, int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "x": int(payload["x"]),
        "y": int(payload["y"]),
    }


def estimate_anchor_scale(
    fisheye_depth: np.ndarray,
    anchor_point: Dict[str, int] | None,
    anchor_distance: float | None,
    patch_radius: int,
) -> Tuple[float, float | None]:
    if anchor_point is None or anchor_distance is None:
        return 1.0, None

    x = int(np.clip(anchor_point["x"], 0, fisheye_depth.shape[1] - 1))
    y = int(np.clip(anchor_point["y"], 0, fisheye_depth.shape[0] - 1))
    r = max(int(patch_radius), 0)
    patch = fisheye_depth[max(0, y - r): min(fisheye_depth.shape[0], y + r + 1), max(0, x - r): min(fisheye_depth.shape[1], x + r + 1)]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return 1.0, None
    predicted_distance = float(np.median(valid))
    scale = float(anchor_distance) / max(predicted_distance, 1e-6)
    return scale, predicted_distance


def open_optional_video(path_str: str | None) -> cv2.VideoCapture | None:
    if not path_str:
        return None
    capture = cv2.VideoCapture(str(Path(path_str).resolve()))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path_str}")
    return capture


def read_exclusion_mask_frame(
    capture: cv2.VideoCapture | None,
    frame_time_sec: float,
    target_width: int,
    target_height: int,
    white_threshold: int,
    dilate_px: int,
) -> np.ndarray:
    if capture is None:
        return np.zeros((target_height, target_width), dtype=bool)

    mask_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    mask_frame_index = int(round(frame_time_sec * mask_fps)) if mask_fps > 0.0 else 0
    if frame_count > 0:
        mask_frame_index = int(np.clip(mask_frame_index, 0, frame_count - 1))

    capture.set(cv2.CAP_PROP_POS_FRAMES, mask_frame_index)
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        return np.zeros((target_height, target_width), dtype=bool)

    if frame_bgr.shape[1] != target_width or frame_bgr.shape[0] != target_height:
        frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray >= int(white_threshold)
    if dilate_px > 0 and np.any(mask):
        kernel_size = int(max(1, dilate_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return mask


def remap_fisheye_mask_to_erp(mask: np.ndarray, erp_map_x: np.ndarray, erp_map_y: np.ndarray) -> np.ndarray:
    erp_mask = cv2.remap(
        mask.astype(np.uint8) * 255,
        erp_map_x,
        erp_map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return erp_mask > 0


def apply_max_distance_mask(valid_mask: np.ndarray, depth: np.ndarray, max_distance: float) -> np.ndarray:
    if max_distance <= 0.0:
        return valid_mask
    return valid_mask & np.isfinite(depth) & (depth > 0) & (depth <= float(max_distance))


def save_mask_debug_outputs(
    debug_dir: Path,
    center_rgb: np.ndarray,
    boat_mask: np.ndarray,
    water_mask: np.ndarray,
) -> None:
    ensure_dir(debug_dir)
    combined_mask = boat_mask | water_mask
    cv2.imwrite(str(debug_dir / "boat_mask_aligned.png"), (boat_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(debug_dir / "water_mask_aligned.png"), (water_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(debug_dir / "exclude_mask_aligned.png"), (combined_mask.astype(np.uint8) * 255))

    overlay = center_rgb.copy()
    overlay[boat_mask] = (0.55 * overlay[boat_mask] + 0.45 * np.array([255, 80, 80], dtype=np.uint8)).astype(np.uint8)
    overlay[water_mask] = (0.55 * overlay[water_mask] + 0.45 * np.array([80, 160, 255], dtype=np.uint8)).astype(np.uint8)
    cv2.imwrite(str(debug_dir / "exclude_mask_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def build_registration_mask(
    valid_mask: np.ndarray,
    depth: np.ndarray,
    row_min_frac: float,
    row_max_frac: float,
    far_percentile: float,
) -> np.ndarray:
    mask = valid_mask.astype(bool).copy()
    height = mask.shape[0]
    row_min = int(np.clip(round(height * row_min_frac), 0, height - 1))
    row_max = int(np.clip(round(height * row_max_frac), row_min + 1, height))
    band_mask = np.zeros_like(mask, dtype=bool)
    band_mask[row_min:row_max] = True
    mask &= band_mask
    if not np.any(mask):
        return mask

    threshold = float(np.percentile(depth[mask], far_percentile))
    mask &= depth >= threshold
    return mask


def prepare_registration_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    point_stride: int,
    voxel_size: float,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    points, colors = erp_to_pointcloud_arrays(
        rgb,
        depth,
        valid_mask,
        point_stride=max(point_stride, 1),
    )
    cloud = cloud_from_arrays(points, colors)
    if voxel_size > 0.0 and len(points) > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    return cloud, points


def choose_icp_distance(points: np.ndarray, requested_distance: float) -> float:
    if requested_distance > 0.0:
        return float(requested_distance)
    if len(points) == 0:
        return 0.25
    median_range = float(np.median(np.linalg.norm(points, axis=1)))
    return float(np.clip(median_range * 0.18, 0.12, 1.5))


def register_cloud_pair(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    source_depth = source_depth[np.isfinite(source_depth)]
    target_depth = target_depth[np.isfinite(target_depth)]
    scale = 1.0
    if args.registration_scale_mode == "median":
        source_median = float(np.median(source_depth)) if source_depth.size else 1.0
        target_median = float(np.median(target_depth)) if target_depth.size else 1.0
        scale = target_median / max(source_median, 1e-6)

    scale_matrix = make_uniform_scale_matrix(scale)
    source_scaled = o3d.geometry.PointCloud(source_cloud)
    source_scaled.transform(scale_matrix)

    source_points_scaled = np.asarray(source_scaled.points)
    target_points = np.asarray(target_cloud.points)
    max_corr = choose_icp_distance(target_points, args.registration_max_corr)
    if len(source_points_scaled) < 80 or len(target_points) < 80:
        return {
            "accepted": False,
            "fitness": 0.0,
            "rmse": float("inf"),
            "scale": scale,
            "transformation": scale_matrix,
            "max_corr": max_corr,
            "reason": "too_few_points",
        }

    coarse = o3d.pipelines.registration.registration_icp(
        source_scaled,
        target_cloud,
        max_corr * 2.0,
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    fine = o3d.pipelines.registration.registration_icp(
        source_scaled,
        target_cloud,
        max_corr,
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    accepted = float(fine.fitness) >= float(args.registration_min_fitness)
    transform = fine.transformation @ scale_matrix if accepted else scale_matrix
    return {
        "accepted": accepted,
        "fitness": float(fine.fitness),
        "rmse": float(fine.inlier_rmse),
        "scale": scale,
        "transformation": transform,
        "max_corr": max_corr,
        "reason": "ok" if accepted else "low_fitness",
    }


def export_turntable_video(
    cloud_path: Path,
    output_path: Path,
    seconds: float,
    fps: int,
    drop_near_percentile: float,
    max_points: int = 60000,
) -> None:
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if len(points) == 0:
        raise RuntimeError(f"Point cloud is empty: {cloud_path}")

    rng = np.random.default_rng(0)
    if len(points) > max_points:
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    distances = np.linalg.norm(points, axis=1)
    if drop_near_percentile > 0.0:
        threshold = float(np.percentile(distances, drop_near_percentile))
        keep = distances >= threshold
        points = points[keep]
        colors = colors[keep]

    # Match the interactive viewer: flip vertical so the upside-down camera reads upright.
    points_xyz = np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins)) / 2.0

    fig = plt.figure(figsize=(8, 8), dpi=180, facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()

    ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], c=colors, s=0.15, linewidths=0)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.view_init(elev=25, azim=-60)

    frame_total = max(int(round(seconds * fps)), 1)

    def update(frame_idx: int):
        azim = -60.0 + 360.0 * frame_idx / frame_total
        ax.view_init(elev=25, azim=azim)
        return ()

    animation = FuncAnimation(fig, update, frames=frame_total, interval=1000 / max(fps, 1), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    animation.save(str(output_path), writer=writer)
    plt.close(fig)


def save_center_outputs(output_dir: Path, center_outputs: Dict[str, np.ndarray], center_rgb: np.ndarray) -> None:
    debug_dir = output_dir / "center_frame"
    ensure_dir(debug_dir)
    cv2.imwrite(str(debug_dir / "original.jpg"), cv2.cvtColor(center_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / "overlay.jpg"), cv2.cvtColor(center_outputs["overlay"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / "erp_valid.jpg"), cv2.cvtColor(center_outputs["erp_rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / "erp_depth.jpg"), cv2.cvtColor(center_outputs["erp_depth_rgb"], cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    backend_api = load_backend_api(args.backend)
    model, erp_height, erp_width = backend_api.load_model(args, device)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (ROOT / "maritime_output" / f"aggregate_t{args.timestamp_sec:05.1f}s").resolve()
    )
    ensure_dir(output_dir)
    anchor_point = load_anchor_point(Path(args.anchor_point_json).resolve() if args.anchor_point_json else None)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    boat_mask_capture = open_optional_video(args.boat_mask_video)
    water_mask_capture = open_optional_video(args.water_mask_video)

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        center_frame = int(round(args.timestamp_sec * fps))
        center_frame = min(max(center_frame, 0), max(frame_count - 1, 0))
        center_rgb = read_video_frame(capture, center_frame)
        geometry = detect_fisheye_geometry(center_rgb, args.black_threshold, args.edge_trim_px)
        geometry["reference_frame_index"] = center_frame
        geometry["reference_time_sec"] = center_frame / max(fps, 1.0)

        save_circle_preview(output_dir / "circle_preview.jpg", center_rgb, geometry)

        erp_map_x, erp_map_y, erp_valid_mask = build_erp_remap(
            erp_height, erp_width, geometry, args.fisheye_fov_deg, args.fisheye_model
        )
        fisheye_map_x, fisheye_map_y, fisheye_valid_mask = build_fisheye_to_erp_remap(
            center_rgb.shape[0],
            center_rgb.shape[1],
            geometry,
            erp_height,
            erp_width,
            args.fisheye_fov_deg,
            args.fisheye_model,
        )

        frame_indices = build_frame_indices(
            fps=fps,
            frame_count=frame_count,
            timestamp_sec=args.timestamp_sec,
            window_sec=args.window_sec,
            frame_step=args.frame_step,
        )
        center_time_sec = float(center_frame / max(fps, 1.0))
        center_boat_mask = read_exclusion_mask_frame(
            boat_mask_capture,
            frame_time_sec=center_time_sec,
            target_width=center_rgb.shape[1],
            target_height=center_rgb.shape[0],
            white_threshold=args.mask_white_threshold,
            dilate_px=args.mask_dilate_px,
        )
        center_water_mask = read_exclusion_mask_frame(
            water_mask_capture,
            frame_time_sec=center_time_sec,
            target_width=center_rgb.shape[1],
            target_height=center_rgb.shape[0],
            white_threshold=args.mask_white_threshold,
            dilate_px=args.mask_dilate_px,
        )
        save_mask_debug_outputs(output_dir / "center_frame", center_rgb, center_boat_mask, center_water_mask)

        background_points_all = []
        background_colors_all = []
        boat_points_all = []
        boat_colors_all = []
        center_outputs = None
        reference_frame_index = frame_indices[0] if frame_indices else center_frame
        reference_registration_cloud = None
        reference_registration_depth = None
        reference_frame_transform = np.eye(4, dtype=np.float64)
        last_registration_cloud = None
        last_registration_depth = None
        last_frame_transform = np.eye(4, dtype=np.float64)
        last_anchor_frame_index = reference_frame_index
        registration_stats = []

        progress = tqdm.tqdm(frame_indices, desc="Aggregating cloud")
        for frame_index in progress:
            frame_rgb = read_video_frame(capture, frame_index)
            frame_time_sec = float(frame_index / max(fps, 1.0))
            boat_mask_fisheye = read_exclusion_mask_frame(
                boat_mask_capture,
                frame_time_sec=frame_time_sec,
                target_width=frame_rgb.shape[1],
                target_height=frame_rgb.shape[0],
                white_threshold=args.mask_white_threshold,
                dilate_px=args.mask_dilate_px,
            )
            water_mask_fisheye = read_exclusion_mask_frame(
                water_mask_capture,
                frame_time_sec=frame_time_sec,
                target_width=frame_rgb.shape[1],
                target_height=frame_rgb.shape[0],
                white_threshold=args.mask_white_threshold,
                dilate_px=args.mask_dilate_px,
            )
            exclusion_mask_fisheye = boat_mask_fisheye | water_mask_fisheye
            boat_mask_erp = remap_fisheye_mask_to_erp(boat_mask_fisheye, erp_map_x, erp_map_y)
            water_mask_erp = remap_fisheye_mask_to_erp(water_mask_fisheye, erp_map_x, erp_map_y)
            exclusion_mask_erp = boat_mask_erp | water_mask_erp
            outputs = backend_api.process_frame(
                frame_rgb,
                model,
                device,
                args.black_threshold,
                erp_map_x,
                erp_map_y,
                erp_valid_mask,
                fisheye_map_x,
                fisheye_map_y,
                fisheye_valid_mask,
            )
            current_valid_mask = filter_depth_mask(
                outputs["erp_valid_mask"] > 0,
                outputs["erp_depth"],
                args.drop_near_percentile,
            )
            anchor_scale, anchor_predicted_distance = estimate_anchor_scale(
                outputs["fisheye_depth"],
                anchor_point,
                args.anchor_distance,
                args.anchor_patch_radius,
            )
            scaled_erp_depth = outputs["erp_depth"] * float(anchor_scale)
            background_valid_mask = current_valid_mask & ~exclusion_mask_erp
            background_valid_mask = apply_max_distance_mask(background_valid_mask, scaled_erp_depth, args.max_distance)
            boat_valid_mask = (outputs["erp_valid_mask"] > 0) & boat_mask_erp & ~water_mask_erp
            boat_valid_mask = apply_max_distance_mask(boat_valid_mask, scaled_erp_depth, args.boat_max_distance)

            background_points, background_colors = erp_to_pointcloud_arrays(
                outputs["erp_rgb"],
                scaled_erp_depth,
                background_valid_mask,
                point_stride=max(args.erp_point_stride, 1),
            )
            boat_points_local, boat_colors_local = erp_to_pointcloud_arrays(
                outputs["erp_rgb"],
                scaled_erp_depth,
                boat_valid_mask,
                point_stride=max(args.erp_point_stride, 1),
            )
            frame_transform = np.eye(4, dtype=np.float64)
            registration_info = {
                "accepted": frame_index == reference_frame_index,
                "fitness": 1.0 if frame_index == reference_frame_index else 0.0,
                "rmse": 0.0,
                "scale": 1.0,
                "max_corr": 0.0,
                "reason": "reference",
                "target_frame_index": frame_index,
                "anchor_scale": float(anchor_scale),
                "anchor_predicted_distance": (
                    float(anchor_predicted_distance) if anchor_predicted_distance is not None else None
                ),
                "excluded_fraction_fisheye": float(np.mean(exclusion_mask_fisheye)),
                "excluded_fraction_erp": float(np.mean(exclusion_mask_erp)),
                "background_fraction_erp": float(np.mean(background_valid_mask)),
                "boat_fraction_erp": float(np.mean(boat_valid_mask)),
            }

            if args.registration_mode == "pairwise_icp" and len(background_points) > 0:
                registration_mask = build_registration_mask(
                    outputs["erp_valid_mask"] > 0,
                    scaled_erp_depth,
                    args.registration_row_min_frac,
                    args.registration_row_max_frac,
                    args.registration_far_percentile,
                )
                registration_mask &= ~exclusion_mask_erp
                registration_mask = apply_max_distance_mask(registration_mask, scaled_erp_depth, args.max_distance)
                registration_cloud, registration_points = prepare_registration_cloud(
                    outputs["erp_rgb"],
                    scaled_erp_depth,
                    registration_mask,
                    point_stride=args.registration_point_stride,
                    voxel_size=args.registration_voxel_size,
                )
                registration_depth = np.linalg.norm(registration_points, axis=1)
                if reference_registration_cloud is None:
                    reference_registration_cloud = registration_cloud
                    reference_registration_depth = registration_depth
                    last_registration_cloud = registration_cloud
                    last_registration_depth = registration_depth
                    last_frame_transform = np.eye(4, dtype=np.float64)
                else:
                    use_previous = args.registration_pair == "previous" and last_registration_cloud is not None
                    target_cloud = last_registration_cloud if use_previous else reference_registration_cloud
                    target_depth = last_registration_depth if use_previous else reference_registration_depth
                    target_transform = last_frame_transform if use_previous else reference_frame_transform
                    target_frame_index = last_anchor_frame_index if use_previous else reference_frame_index
                    pair_result = register_cloud_pair(
                        source_cloud=registration_cloud,
                        target_cloud=target_cloud,
                        source_depth=registration_depth,
                        target_depth=target_depth,
                        args=args,
                    )
                    frame_transform = target_transform @ pair_result["transformation"]
                    registration_info = {
                        "accepted": bool(pair_result["accepted"]),
                        "fitness": float(pair_result["fitness"]),
                        "rmse": float(pair_result["rmse"]),
                        "scale": float(pair_result["scale"]),
                        "max_corr": float(pair_result["max_corr"]),
                        "reason": str(pair_result["reason"]),
                        "target_frame_index": int(target_frame_index),
                        "anchor_scale": float(anchor_scale),
                        "anchor_predicted_distance": (
                            float(anchor_predicted_distance) if anchor_predicted_distance is not None else None
                        ),
                        "excluded_fraction_fisheye": float(np.mean(exclusion_mask_fisheye)),
                        "excluded_fraction_erp": float(np.mean(exclusion_mask_erp)),
                        "background_fraction_erp": float(np.mean(background_valid_mask)),
                        "boat_fraction_erp": float(np.mean(boat_valid_mask)),
                    }
                    if pair_result["accepted"] or args.registration_pair == "reference":
                        last_registration_cloud = registration_cloud
                        last_registration_depth = registration_depth
                        last_frame_transform = frame_transform
                        last_anchor_frame_index = frame_index

            transformed_background_points = transform_points(background_points, frame_transform)
            background_points_all.append(transformed_background_points)
            background_colors_all.append(background_colors)
            boat_points_all.append(boat_points_local)
            boat_colors_all.append(boat_colors_local)
            registration_stats.append(
                {
                    "frame_index": int(frame_index),
                    "time_sec": float(frame_index / max(fps, 1.0)),
                    **registration_info,
                }
            )
            progress.set_postfix(
                fit=f"{registration_info['fitness']:.2f}",
                scale=f"{registration_info['scale']:.2f}",
                mode=registration_info["reason"],
            )

            if frame_index == center_frame:
                center_outputs = outputs

        merged_background_points = np.concatenate(background_points_all, axis=0)
        merged_background_colors = np.concatenate(background_colors_all, axis=0)
        background_cloud = cloud_from_arrays(merged_background_points, merged_background_colors)
        if args.voxel_size > 0.0:
            background_cloud = background_cloud.voxel_down_sample(args.voxel_size)

        cloud_path = output_dir / "aggregated_pointcloud.ply"
        background_cloud_path = output_dir / "background_pointcloud.ply"
        o3d.io.write_point_cloud(str(cloud_path), background_cloud)
        o3d.io.write_point_cloud(str(background_cloud_path), background_cloud)

        boat_cloud = None
        boat_cloud_path = None
        if any(len(points) > 0 for points in boat_points_all):
            merged_boat_points = np.concatenate(boat_points_all, axis=0)
            merged_boat_colors = np.concatenate(boat_colors_all, axis=0)
            boat_cloud = cloud_from_arrays(merged_boat_points, merged_boat_colors)
            if args.voxel_size > 0.0:
                boat_cloud = boat_cloud.voxel_down_sample(args.voxel_size)
            boat_cloud_path = output_dir / "boat_pointcloud_local.ply"
            o3d.io.write_point_cloud(str(boat_cloud_path), boat_cloud)

        html_path = output_dir / "aggregated_pointcloud.html"
        export_html(
            input_path=cloud_path,
            output_path=html_path,
            max_points=args.viewer_max_points,
            near_clip_percentiles=[35.0, 60.0],
            seed=0,
            rotate_deg_x=0.0,
            rotate_deg_y=0.0,
            rotate_deg_z=0.0,
            default_view="isometric",
            default_trace="mid_far",
            title=f"Aggregated maritime cloud around {args.timestamp_sec:.2f}s",
        )

        if boat_cloud_path is not None:
            export_background_boat_html(
                background_path=background_cloud_path,
                boat_path=boat_cloud_path,
                output_path=output_dir / "background_boat_view.html",
                background_max_points=args.viewer_max_points,
                boat_max_points=max(args.viewer_max_points // 2, 15000),
                default_view="side",
                default_mode="both_far",
                title=f"Background + boat cloud around {args.timestamp_sec:.2f}s",
            )

        video_path = output_dir / "aggregated_pointcloud_turntable.mp4"
        export_turntable_video(
            cloud_path=cloud_path,
            output_path=video_path,
            seconds=args.turntable_seconds,
            fps=args.turntable_fps,
            drop_near_percentile=args.turntable_drop_near_percentile,
        )

        if center_outputs is not None:
            save_center_outputs(output_dir, center_outputs, center_rgb)

        metadata = {
            "input": str(input_path),
            "backend": args.backend,
            "model_path": str(Path(args.model_path).resolve()),
            "model_name": args.model_name,
            "dap_root": str(Path(args.dap_root).resolve()) if args.backend == "dap" else None,
            "midas_model_type": args.midas_model_type if args.backend == "dap" else None,
            "fine_tune_type": args.fine_tune_type if args.backend == "dap" else None,
            "min_depth": args.min_depth if args.backend == "dap" else None,
            "max_depth": args.max_depth if args.backend == "dap" else None,
            "timestamp_sec": args.timestamp_sec,
            "window_sec": args.window_sec,
            "fps": fps,
            "frame_step": args.frame_step,
            "frame_indices": frame_indices,
            "erp_point_stride": args.erp_point_stride,
            "drop_near_percentile": args.drop_near_percentile,
            "voxel_size": args.voxel_size,
            "turntable_drop_near_percentile": args.turntable_drop_near_percentile,
            "fisheye_model": args.fisheye_model,
            "fisheye_fov_deg": args.fisheye_fov_deg,
            "geometry": geometry,
            "reference_frame_index": int(reference_frame_index),
            "registration_mode": args.registration_mode,
            "registration_pair": args.registration_pair,
            "registration_far_percentile": args.registration_far_percentile,
            "registration_row_min_frac": args.registration_row_min_frac,
            "registration_row_max_frac": args.registration_row_max_frac,
            "registration_point_stride": args.registration_point_stride,
            "registration_voxel_size": args.registration_voxel_size,
            "registration_max_corr": args.registration_max_corr,
            "registration_min_fitness": args.registration_min_fitness,
            "registration_scale_mode": args.registration_scale_mode,
            "anchor_point_json": str(Path(args.anchor_point_json).resolve()) if args.anchor_point_json else None,
            "anchor_distance": args.anchor_distance,
            "anchor_patch_radius": args.anchor_patch_radius,
            "boat_mask_video": str(Path(args.boat_mask_video).resolve()) if args.boat_mask_video else None,
            "water_mask_video": str(Path(args.water_mask_video).resolve()) if args.water_mask_video else None,
            "mask_white_threshold": args.mask_white_threshold,
            "mask_dilate_px": args.mask_dilate_px,
            "max_distance": args.max_distance,
            "boat_max_distance": args.boat_max_distance,
            "registration_stats": registration_stats,
            "point_count_after_downsample": int(np.asarray(background_cloud.points).shape[0]),
            "background_point_count_after_downsample": int(np.asarray(background_cloud.points).shape[0]),
            "boat_point_count_after_downsample": (
                int(np.asarray(boat_cloud.points).shape[0]) if boat_cloud is not None else 0
            ),
            "background_pointcloud_path": str(background_cloud_path),
            "boat_pointcloud_path": str(boat_cloud_path) if boat_cloud_path is not None else None,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        capture.release()
        if boat_mask_capture is not None:
            boat_mask_capture.release()
        if water_mask_capture is not None:
            water_mask_capture.release()


if __name__ == "__main__":
    main()
