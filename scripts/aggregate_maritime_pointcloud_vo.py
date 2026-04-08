from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_maritime_pointcloud import (  # noqa: E402
    apply_max_distance_mask,
    build_frame_indices,
    build_registration_mask,
    choose_icp_distance,
    cloud_from_arrays,
    ensure_dir,
    estimate_anchor_scale,
    export_turntable_video,
    load_anchor_point,
    load_backend_api,
    open_optional_video,
    prepare_registration_cloud,
    read_exclusion_mask_frame,
    read_video_frame,
    remap_fisheye_mask_to_erp,
    save_center_outputs,
    save_mask_debug_outputs,
    transform_points,
)
from scripts.maritime_da360 import (  # noqa: E402
    build_erp_remap,
    build_fisheye_to_erp_remap,
    detect_fisheye_geometry,
    erp_to_pointcloud_arrays,
    save_circle_preview,
)
from scripts.view_background_boat_pointcloud import export_background_boat_html  # noqa: E402
from scripts.view_pointcloud import export_html  # noqa: E402


@dataclass
class ViewConfig:
    name: str
    yaw_deg: float
    pitch_deg: float
    size: int
    fov_deg: float
    map_x: np.ndarray
    map_y: np.ndarray
    camera_matrix: np.ndarray
    rotation_world_from_view: np.ndarray


@dataclass
class FrameState:
    frame_index: int
    time_sec: float
    rgb: np.ndarray
    outputs: Dict[str, np.ndarray]
    scaled_erp_depth: np.ndarray
    background_mask_erp: np.ndarray
    boat_mask_erp: np.ndarray
    world_from_frame: np.ndarray
    registration_cloud: o3d.geometry.PointCloud | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype maritime visual odometry using background-only virtual horizon views."
    )
    parser.add_argument("--backend", default="da360", choices=["da360", "dap"])
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--model_name", default="DA360_large", type=str)
    parser.add_argument("--net", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--erp_height", default=None, type=int)
    parser.add_argument("--erp_width", default=None, type=int)
    parser.add_argument("--dap_root", default=str(ROOT.parents[0] / "DAP"), type=str)
    parser.add_argument("--midas_model_type", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--fine_tune_type", default="hypersim", type=str)
    parser.add_argument("--min_depth", default=0.01, type=float)
    parser.add_argument("--max_depth", default=1.0, type=float)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--timestamp_sec", default=4.0, type=float)
    parser.add_argument("--window_sec", default=1.0, type=float)
    parser.add_argument("--frame_step", default=1, type=int)
    parser.add_argument("--erp_point_stride", default=4, type=int)
    parser.add_argument("--voxel_size", default=0.03, type=float)
    parser.add_argument("--viewer_max_points", default=80000, type=int)
    parser.add_argument("--turntable_seconds", default=10.0, type=float)
    parser.add_argument("--turntable_fps", default=20, type=int)
    parser.add_argument("--turntable_drop_near_percentile", default=0.0, type=float)
    parser.add_argument("--fisheye_fov_deg", default=200.0, type=float)
    parser.add_argument(
        "--fisheye_model",
        default="equidistant",
        choices=["equidistant", "equisolid", "orthographic", "stereographic"],
    )
    parser.add_argument("--black_threshold", default=40, type=int)
    parser.add_argument("--edge_trim_px", default=4.0, type=float)
    parser.add_argument("--anchor_point_json", default=None, type=str)
    parser.add_argument("--anchor_distance", default=None, type=float)
    parser.add_argument("--anchor_patch_radius", default=4, type=int)
    parser.add_argument("--boat_mask_video", default=None, type=str)
    parser.add_argument("--water_mask_video", default=None, type=str)
    parser.add_argument("--mask_white_threshold", default=245, type=int)
    parser.add_argument("--mask_dilate_px", default=4, type=int)
    parser.add_argument("--max_distance", default=30.0, type=float)
    parser.add_argument("--boat_max_distance", default=8.0, type=float)
    parser.add_argument("--view_size", default=320, type=int)
    parser.add_argument("--view_fov_deg", default=90.0, type=float)
    parser.add_argument("--view_yaws_deg", default="0,45,90,135,180,225,270,315", type=str)
    parser.add_argument("--view_pitch_deg", default=0.0, type=float)
    parser.add_argument("--orb_features", default=1800, type=int)
    parser.add_argument("--match_ratio", default=0.78, type=float)
    parser.add_argument("--depth_patch_radius", default=2, type=int)
    parser.add_argument("--pnp_min_matches", default=40, type=int)
    parser.add_argument("--pnp_min_inliers", default=24, type=int)
    parser.add_argument("--pnp_reproj_error", default=4.0, type=float)
    parser.add_argument("--pair_debug_count", default=6, type=int)
    parser.add_argument("--pose_refine_icp", action="store_true")
    parser.add_argument("--registration_row_min_frac", default=0.2, type=float)
    parser.add_argument("--registration_row_max_frac", default=0.62, type=float)
    parser.add_argument("--registration_far_percentile", default=0.0, type=float)
    parser.add_argument("--registration_point_stride", default=6, type=int)
    parser.add_argument("--registration_voxel_size", default=0.05, type=float)
    parser.add_argument("--registration_max_corr", default=0.0, type=float)
    parser.add_argument("--registration_min_fitness", default=0.05, type=float)
    return parser.parse_args()


def parse_yaw_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def spherical_direction(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    x = math.cos(lat) * math.sin(lon)
    y = math.cos(lat) * math.cos(lon)
    z = -math.sin(lat)
    vec = np.array([x, y, z], dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def build_view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    forward = spherical_direction(yaw_deg, pitch_deg)
    world_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(forward, right)
    up = up / max(float(np.linalg.norm(up)), 1e-8)
    down = -up
    return np.stack([right, down, forward], axis=1).astype(np.float32)


def build_erp_to_view_remap(
    erp_height: int,
    erp_width: int,
    size: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fx = fy = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    xs = (np.arange(size, dtype=np.float32) - cx) / fx
    ys = (np.arange(size, dtype=np.float32) - cy) / fy
    grid_x, grid_y = np.meshgrid(xs, ys)
    rays_view = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    rays_view /= np.linalg.norm(rays_view, axis=-1, keepdims=True)

    rotation_world_from_view = build_view_rotation(yaw_deg, pitch_deg)
    rays_world = rays_view @ rotation_world_from_view.T
    lon = np.arctan2(rays_world[..., 0], rays_world[..., 1])
    lat = -np.arcsin(np.clip(rays_world[..., 2], -1.0, 1.0))
    map_x = ((lon + np.pi) / (2.0 * np.pi) * erp_width - 0.5).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * erp_height - 0.5).astype(np.float32)

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return map_x, map_y, camera_matrix, rotation_world_from_view


def build_view_configs(erp_height: int, erp_width: int, args: argparse.Namespace) -> List[ViewConfig]:
    configs: List[ViewConfig] = []
    for yaw_deg in parse_yaw_list(args.view_yaws_deg):
        map_x, map_y, camera_matrix, rotation_world_from_view = build_erp_to_view_remap(
            erp_height=erp_height,
            erp_width=erp_width,
            size=args.view_size,
            fov_deg=args.view_fov_deg,
            yaw_deg=yaw_deg,
            pitch_deg=args.view_pitch_deg,
        )
        configs.append(
            ViewConfig(
                name=f"yaw_{int(round(yaw_deg)):03d}",
                yaw_deg=float(yaw_deg),
                pitch_deg=float(args.view_pitch_deg),
                size=int(args.view_size),
                fov_deg=float(args.view_fov_deg),
                map_x=map_x,
                map_y=map_y,
                camera_matrix=camera_matrix,
                rotation_world_from_view=rotation_world_from_view,
            )
        )
    return configs


def remap_to_view(array: np.ndarray, view: ViewConfig, interpolation: int, border_value) -> np.ndarray:
    return cv2.remap(
        array,
        view.map_x,
        view.map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def sample_depth_patch(depth: np.ndarray, x: float, y: float, radius: int) -> float | None:
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= depth.shape[1] or yi >= depth.shape[0]:
        return None
    r = max(int(radius), 0)
    patch = depth[max(0, yi - r): min(depth.shape[0], yi + r + 1), max(0, xi - r): min(depth.shape[1], xi + r + 1)]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_unit_ray(point: tuple[float, float], camera_matrix: np.ndarray) -> np.ndarray:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    x = (float(point[0]) - cx) / fx
    y = (float(point[1]) - cy) / fy
    ray = np.array([x, y, 1.0], dtype=np.float32)
    return ray / max(float(np.linalg.norm(ray)), 1e-8)


def build_view_images(state: FrameState, view: ViewConfig) -> Dict[str, np.ndarray]:
    view_rgb = remap_to_view(state.outputs["erp_rgb"], view, cv2.INTER_LINEAR, (0, 0, 0))
    view_depth = remap_to_view(state.scaled_erp_depth, view, cv2.INTER_LINEAR, 0.0)
    view_background = remap_to_view(
        (state.background_mask_erp.astype(np.uint8) * 255),
        view,
        cv2.INTER_NEAREST,
        0,
    ) > 0
    return {
        "rgb": view_rgb,
        "gray": cv2.cvtColor(view_rgb, cv2.COLOR_RGB2GRAY),
        "depth": view_depth,
        "background_mask": view_background,
    }


def estimate_pose_single_view(
    prev_view: Dict[str, np.ndarray],
    curr_view: Dict[str, np.ndarray],
    view: ViewConfig,
    orb: cv2.ORB,
    args: argparse.Namespace,
) -> Dict[str, object] | None:
    prev_mask_u8 = prev_view["background_mask"].astype(np.uint8) * 255
    curr_mask_u8 = curr_view["background_mask"].astype(np.uint8) * 255
    keypoints_prev, descriptors_prev = orb.detectAndCompute(prev_view["gray"], prev_mask_u8)
    keypoints_curr, descriptors_curr = orb.detectAndCompute(curr_view["gray"], curr_mask_u8)
    if descriptors_prev is None or descriptors_curr is None:
        return None
    if len(keypoints_prev) < args.pnp_min_matches or len(keypoints_curr) < args.pnp_min_matches:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(descriptors_prev, descriptors_curr, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < args.match_ratio * second.distance:
            good_matches.append(first)
    if len(good_matches) < args.pnp_min_matches:
        return None

    object_points = []
    image_points = []
    kept_matches = []
    for match in sorted(good_matches, key=lambda item: item.distance):
        kp_prev = keypoints_prev[match.queryIdx]
        kp_curr = keypoints_curr[match.trainIdx]
        depth_value = sample_depth_patch(prev_view["depth"], kp_prev.pt[0], kp_prev.pt[1], args.depth_patch_radius)
        if depth_value is None or depth_value > args.max_distance:
            continue
        ray_prev = pixel_to_unit_ray(kp_prev.pt, view.camera_matrix)
        object_points.append(ray_prev * depth_value)
        image_points.append(kp_curr.pt)
        kept_matches.append(match)

    if len(object_points) < args.pnp_min_matches:
        return None

    object_points_np = np.asarray(object_points, dtype=np.float32)
    image_points_np = np.asarray(image_points, dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points_np,
        imagePoints=image_points_np,
        cameraMatrix=view.camera_matrix,
        distCoeffs=dist_coeffs,
        iterationsCount=200,
        reprojectionError=float(args.pnp_reproj_error),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < args.pnp_min_inliers:
        return None

    inlier_indices = inliers.reshape(-1)
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points_np[inlier_indices],
        imagePoints=image_points_np[inlier_indices],
        cameraMatrix=view.camera_matrix,
        distCoeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation_view, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        object_points_np[inlier_indices],
        rvec,
        tvec,
        view.camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    reproj_error = float(
        np.sqrt(np.mean(np.sum((projected - image_points_np[inlier_indices]) ** 2, axis=1)))
    )

    transform_view = np.eye(4, dtype=np.float64)
    transform_view[:3, :3] = rotation_view.astype(np.float64)
    transform_view[:3, 3] = tvec.reshape(3).astype(np.float64)

    similarity = np.eye(4, dtype=np.float64)
    similarity[:3, :3] = view.rotation_world_from_view.astype(np.float64)
    similarity_inv = np.eye(4, dtype=np.float64)
    similarity_inv[:3, :3] = view.rotation_world_from_view.T.astype(np.float64)
    transform_frame = similarity @ transform_view @ similarity_inv

    return {
        "view_name": view.name,
        "yaw_deg": view.yaw_deg,
        "pitch_deg": view.pitch_deg,
        "match_count": len(good_matches),
        "object_match_count": len(object_points_np),
        "inlier_count": int(len(inlier_indices)),
        "reproj_error": reproj_error,
        "transform_frame": transform_frame,
        "transform_view": transform_view,
        "keypoints_prev": keypoints_prev,
        "keypoints_curr": keypoints_curr,
        "kept_matches": kept_matches,
        "inlier_indices": inlier_indices,
        "prev_rgb": prev_view["rgb"],
        "curr_rgb": curr_view["rgb"],
    }


def estimate_pose_multi_view(
    prev_state: FrameState,
    curr_state: FrameState,
    view_configs: Sequence[ViewConfig],
    orb: cv2.ORB,
    args: argparse.Namespace,
) -> Dict[str, object] | None:
    best_result = None
    for view in view_configs:
        prev_view = build_view_images(prev_state, view)
        curr_view = build_view_images(curr_state, view)
        result = estimate_pose_single_view(prev_view, curr_view, view, orb, args)
        if result is None:
            continue
        if best_result is None:
            best_result = result
            continue
        current_score = (int(result["inlier_count"]), -float(result["reproj_error"]))
        best_score = (int(best_result["inlier_count"]), -float(best_result["reproj_error"]))
        if current_score > best_score:
            best_result = result
    return best_result


def refine_pose_with_icp(
    prev_cloud: o3d.geometry.PointCloud | None,
    curr_cloud: o3d.geometry.PointCloud | None,
    init_prev_from_curr: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object] | None:
    if prev_cloud is None or curr_cloud is None:
        return None

    prev_points = np.asarray(prev_cloud.points)
    curr_points = np.asarray(curr_cloud.points)
    if len(prev_points) < 80 or len(curr_points) < 80:
        return None

    max_corr = choose_icp_distance(prev_points, args.registration_max_corr)
    coarse = o3d.pipelines.registration.registration_icp(
        curr_cloud,
        prev_cloud,
        max_corr * 2.0,
        init_prev_from_curr,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    fine = o3d.pipelines.registration.registration_icp(
        curr_cloud,
        prev_cloud,
        max_corr,
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    if fine.fitness < args.registration_min_fitness:
        return None
    return {
        "transformation": fine.transformation,
        "fitness": float(fine.fitness),
        "rmse": float(fine.inlier_rmse),
        "max_corr": float(max_corr),
    }


def save_match_debug(path: Path, result: Dict[str, object], max_matches: int = 80) -> None:
    matches = result["kept_matches"]
    inlier_indices = set(int(index) for index in result["inlier_indices"])
    selected_matches = [matches[idx] for idx in inlier_indices if idx < len(matches)]
    selected_matches = selected_matches[:max_matches]
    if not selected_matches:
        return
    debug = cv2.drawMatches(
        cv2.cvtColor(result["prev_rgb"], cv2.COLOR_RGB2BGR),
        result["keypoints_prev"],
        cv2.cvtColor(result["curr_rgb"], cv2.COLOR_RGB2BGR),
        result["keypoints_curr"],
        selected_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 180, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(path), debug)


def plot_trajectory(path: Path, transforms: List[np.ndarray]) -> None:
    positions = np.array([transform[:3, 3] for transform in transforms], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    axes[0].plot(positions[:, 0], positions[:, 1], marker="o", linewidth=1.5)
    axes[0].set_title("Trajectory XY")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(positions[:, 0], label="X")
    axes[1].plot(positions[:, 1], label="Y")
    axes[1].plot(positions[:, 2], label="Z")
    axes[1].set_title("Translation Components")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Meters")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)


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
        else (ROOT / "maritime_output" / f"vo_t{args.timestamp_sec:05.1f}s_{args.backend}").resolve()
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
        view_configs = build_view_configs(erp_height, erp_width, args)
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
        registration_stats = []
        frame_transforms: List[np.ndarray] = []
        prev_state: FrameState | None = None
        center_outputs = None
        orb = cv2.ORB_create(nfeatures=args.orb_features, fastThreshold=12)
        pair_debug_dir = output_dir / "pair_debug"
        ensure_dir(pair_debug_dir)
        debug_pairs_written = 0

        progress = tqdm.tqdm(frame_indices, desc="VO aggregation")
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
            anchor_scale, anchor_predicted_distance = estimate_anchor_scale(
                outputs["fisheye_depth"],
                anchor_point,
                args.anchor_distance,
                args.anchor_patch_radius,
            )
            scaled_erp_depth = outputs["erp_depth"] * float(anchor_scale)
            background_mask_erp = (outputs["erp_valid_mask"] > 0) & ~exclusion_mask_erp
            background_mask_erp = apply_max_distance_mask(background_mask_erp, scaled_erp_depth, args.max_distance)
            boat_mask_local = (outputs["erp_valid_mask"] > 0) & boat_mask_erp & ~water_mask_erp
            boat_mask_local = apply_max_distance_mask(boat_mask_local, scaled_erp_depth, args.boat_max_distance)
            registration_mask = build_registration_mask(
                background_mask_erp,
                scaled_erp_depth,
                args.registration_row_min_frac,
                args.registration_row_max_frac,
                args.registration_far_percentile,
            )
            registration_cloud = None
            if np.any(registration_mask):
                registration_cloud, _ = prepare_registration_cloud(
                    outputs["erp_rgb"],
                    scaled_erp_depth,
                    registration_mask,
                    point_stride=max(args.registration_point_stride, 1),
                    voxel_size=args.registration_voxel_size,
                )

            if prev_state is None:
                world_from_frame = np.eye(4, dtype=np.float64)
                pose_info = {
                    "accepted": True,
                    "reason": "reference",
                    "view_name": None,
                    "yaw_deg": None,
                    "pitch_deg": None,
                    "match_count": 0,
                    "object_match_count": 0,
                    "inlier_count": 0,
                    "reproj_error": 0.0,
                    "icp_refined": False,
                    "icp_fitness": 0.0,
                    "icp_rmse": 0.0,
                    "icp_max_corr": 0.0,
                }
            else:
                curr_state_probe = FrameState(
                    frame_index=frame_index,
                    time_sec=frame_time_sec,
                    rgb=frame_rgb,
                    outputs=outputs,
                    scaled_erp_depth=scaled_erp_depth,
                    background_mask_erp=background_mask_erp,
                    boat_mask_erp=boat_mask_local,
                    world_from_frame=np.eye(4, dtype=np.float64),
                    registration_cloud=registration_cloud,
                )
                pose_result = estimate_pose_multi_view(prev_state, curr_state_probe, view_configs, orb, args)
                if pose_result is None:
                    world_from_frame = prev_state.world_from_frame.copy()
                    pose_info = {
                        "accepted": False,
                        "reason": "no_pose",
                        "view_name": None,
                        "yaw_deg": None,
                        "pitch_deg": None,
                        "match_count": 0,
                        "object_match_count": 0,
                        "inlier_count": 0,
                        "reproj_error": float("inf"),
                        "icp_refined": False,
                        "icp_fitness": 0.0,
                        "icp_rmse": float("inf"),
                        "icp_max_corr": 0.0,
                    }
                else:
                    transform_curr_from_prev = pose_result["transform_frame"]
                    transform_prev_from_curr = np.linalg.inv(transform_curr_from_prev)
                    icp_result = None
                    if args.pose_refine_icp:
                        icp_result = refine_pose_with_icp(
                            prev_state.registration_cloud,
                            registration_cloud,
                            transform_prev_from_curr,
                            args,
                        )
                    if icp_result is not None:
                        transform_prev_from_curr = icp_result["transformation"]
                    world_from_frame = prev_state.world_from_frame @ transform_prev_from_curr
                    pose_info = {
                        "accepted": True,
                        "reason": "pnp_icp" if icp_result is not None else "pnp",
                        "view_name": pose_result["view_name"],
                        "yaw_deg": float(pose_result["yaw_deg"]),
                        "pitch_deg": float(pose_result["pitch_deg"]),
                        "match_count": int(pose_result["match_count"]),
                        "object_match_count": int(pose_result["object_match_count"]),
                        "inlier_count": int(pose_result["inlier_count"]),
                        "reproj_error": float(pose_result["reproj_error"]),
                        "icp_refined": bool(icp_result is not None),
                        "icp_fitness": float(icp_result["fitness"]) if icp_result is not None else 0.0,
                        "icp_rmse": float(icp_result["rmse"]) if icp_result is not None else 0.0,
                        "icp_max_corr": float(icp_result["max_corr"]) if icp_result is not None else 0.0,
                    }
                    if debug_pairs_written < args.pair_debug_count:
                        save_match_debug(
                            pair_debug_dir / f"pair_{prev_state.frame_index:06d}_{frame_index:06d}_{pose_result['view_name']}.jpg",
                            pose_result,
                        )
                        debug_pairs_written += 1

            current_state = FrameState(
                frame_index=frame_index,
                time_sec=frame_time_sec,
                rgb=frame_rgb,
                outputs=outputs,
                scaled_erp_depth=scaled_erp_depth,
                background_mask_erp=background_mask_erp,
                boat_mask_erp=boat_mask_local,
                world_from_frame=world_from_frame,
                registration_cloud=registration_cloud,
            )
            prev_state = current_state
            frame_transforms.append(world_from_frame.copy())

            background_points, background_colors = erp_to_pointcloud_arrays(
                outputs["erp_rgb"],
                scaled_erp_depth,
                background_mask_erp,
                point_stride=max(args.erp_point_stride, 1),
            )
            boat_points_local, boat_colors_local = erp_to_pointcloud_arrays(
                outputs["erp_rgb"],
                scaled_erp_depth,
                boat_mask_local,
                point_stride=max(args.erp_point_stride, 1),
            )
            background_points_all.append(transform_points(background_points, world_from_frame))
            background_colors_all.append(background_colors)
            boat_points_all.append(boat_points_local)
            boat_colors_all.append(boat_colors_local)

            registration_stats.append(
                {
                    "frame_index": int(frame_index),
                    "time_sec": frame_time_sec,
                    "anchor_scale": float(anchor_scale),
                    "anchor_predicted_distance": (
                        float(anchor_predicted_distance) if anchor_predicted_distance is not None else None
                    ),
                    "excluded_fraction_erp": float(np.mean(exclusion_mask_erp)),
                    "background_fraction_erp": float(np.mean(background_mask_erp)),
                    "boat_fraction_erp": float(np.mean(boat_mask_local)),
                    "translation_world": world_from_frame[:3, 3].tolist(),
                    **pose_info,
                }
            )
            progress.set_postfix(
                mode=pose_info["reason"],
                view=pose_info["view_name"] or "-",
                inliers=pose_info["inlier_count"],
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
            default_view="side",
            default_trace="mid_far",
            title=f"VO aggregated maritime cloud around {args.timestamp_sec:.2f}s",
        )

        semantic_view_path = None
        if boat_cloud_path is not None:
            semantic_view_path = output_dir / "background_boat_view.html"
            export_background_boat_html(
                background_path=background_cloud_path,
                boat_path=boat_cloud_path,
                output_path=semantic_view_path,
                background_max_points=args.viewer_max_points,
                boat_max_points=max(args.viewer_max_points // 2, 15000),
                default_view="side",
                default_mode="both_far",
                title=f"VO background + boat cloud around {args.timestamp_sec:.2f}s",
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

        trajectory_path = output_dir / "trajectory.png"
        plot_trajectory(trajectory_path, frame_transforms)

        accepted_frames = sum(1 for item in registration_stats if bool(item["accepted"]))
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
            "voxel_size": args.voxel_size,
            "viewer_max_points": args.viewer_max_points,
            "turntable_drop_near_percentile": args.turntable_drop_near_percentile,
            "fisheye_model": args.fisheye_model,
            "fisheye_fov_deg": args.fisheye_fov_deg,
            "geometry": geometry,
            "view_size": args.view_size,
            "view_fov_deg": args.view_fov_deg,
            "view_yaws_deg": parse_yaw_list(args.view_yaws_deg),
            "view_pitch_deg": args.view_pitch_deg,
            "orb_features": args.orb_features,
            "match_ratio": args.match_ratio,
            "depth_patch_radius": args.depth_patch_radius,
            "pnp_min_matches": args.pnp_min_matches,
            "pnp_min_inliers": args.pnp_min_inliers,
            "pnp_reproj_error": args.pnp_reproj_error,
            "pose_refine_icp": bool(args.pose_refine_icp),
            "registration_row_min_frac": args.registration_row_min_frac,
            "registration_row_max_frac": args.registration_row_max_frac,
            "registration_far_percentile": args.registration_far_percentile,
            "registration_point_stride": args.registration_point_stride,
            "registration_voxel_size": args.registration_voxel_size,
            "registration_max_corr": args.registration_max_corr,
            "registration_min_fitness": args.registration_min_fitness,
            "anchor_point_json": str(Path(args.anchor_point_json).resolve()) if args.anchor_point_json else None,
            "anchor_distance": args.anchor_distance,
            "anchor_patch_radius": args.anchor_patch_radius,
            "boat_mask_video": str(Path(args.boat_mask_video).resolve()) if args.boat_mask_video else None,
            "water_mask_video": str(Path(args.water_mask_video).resolve()) if args.water_mask_video else None,
            "mask_white_threshold": args.mask_white_threshold,
            "mask_dilate_px": args.mask_dilate_px,
            "max_distance": args.max_distance,
            "boat_max_distance": args.boat_max_distance,
            "accepted_frame_count": int(accepted_frames),
            "frame_count_processed": int(len(frame_indices)),
            "registration_stats": registration_stats,
            "trajectory_path": str(trajectory_path),
            "point_count_after_downsample": int(np.asarray(background_cloud.points).shape[0]),
            "background_point_count_after_downsample": int(np.asarray(background_cloud.points).shape[0]),
            "boat_point_count_after_downsample": (
                int(np.asarray(boat_cloud.points).shape[0]) if boat_cloud is not None else 0
            ),
            "aggregated_pointcloud_path": str(cloud_path),
            "background_pointcloud_path": str(background_cloud_path),
            "boat_pointcloud_path": str(boat_cloud_path) if boat_cloud_path is not None else None,
            "html_path": str(html_path),
            "semantic_view_path": str(semantic_view_path) if semantic_view_path is not None else None,
            "turntable_path": str(video_path),
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
