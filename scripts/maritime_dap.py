from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm

ROOT = Path(__file__).resolve().parents[1]
DAP_DEFAULT_ROOT = ROOT.parents[0] / "DAP"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver import kitti_colormap  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt DAP to a downward-looking maritime fisheye image or video."
    )
    parser.add_argument("--input", required=True, type=str, help="Path to the fisheye image or video.")
    parser.add_argument(
        "--input_type",
        default="auto",
        choices=["auto", "image", "video"],
        help="Force image or video processing when auto-detection is ambiguous.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory. Defaults to ./maritime_output/<input_name>_<model_name>.",
    )
    parser.add_argument(
        "--dap_root",
        default=str(DAP_DEFAULT_ROOT),
        type=str,
        help="Path to the official DAP repo checkout.",
    )
    parser.add_argument("--model_path", required=True, type=str, help="DAP checkpoint path.")
    parser.add_argument("--model_name", default="DAP_vitl", type=str, help="Label used in outputs.")
    parser.add_argument("--device", default=None, type=str, help="cuda, cpu, or leave unset for auto.")
    parser.add_argument("--erp_height", default=512, type=int, help="ERP height fed to DAP.")
    parser.add_argument("--erp_width", default=1024, type=int, help="ERP width fed to DAP.")
    parser.add_argument("--midas_model_type", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--fine_tune_type", default="hypersim", type=str)
    parser.add_argument("--min_depth", default=0.01, type=float)
    parser.add_argument("--max_depth", default=1.0, type=float)
    parser.add_argument(
        "--fisheye_fov_deg",
        default=200.0,
        type=float,
        help="Assumed full fisheye field of view in degrees.",
    )
    parser.add_argument(
        "--fisheye_model",
        default="equidistant",
        choices=["equidistant", "equisolid", "orthographic", "stereographic"],
        help="Projection model used for fisheye-to-ERP conversion.",
    )
    parser.add_argument(
        "--black_threshold",
        default=40,
        type=int,
        help="Pixels darker than this are treated as outside the fisheye footprint.",
    )
    parser.add_argument(
        "--edge_trim_px",
        default=4.0,
        type=float,
        help="Shrink the detected fisheye radius slightly to avoid sampling the black border.",
    )
    parser.add_argument(
        "--sample_stride",
        default=20,
        type=int,
        help="Process every Nth frame for video input. 20 means 1 fps on a 20 fps source.",
    )
    parser.add_argument(
        "--max_frames",
        default=None,
        type=int,
        help="Optional hard limit on processed frames after stride is applied.",
    )
    parser.add_argument(
        "--save_debug_frames",
        default=5,
        type=int,
        help="How many processed frames should also save ERP intermediates.",
    )
    parser.add_argument(
        "--write_video",
        action="store_true",
        help="Write fisheye depth and overlay preview videos for video input.",
    )
    parser.add_argument(
        "--save_pointcloud",
        action="store_true",
        help="Save a point cloud for the first debug frame using only valid ERP pixels.",
    )
    return parser.parse_args()


def detect_input_type(path: Path, forced: str) -> str:
    if forced != "auto":
        return forced

    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    capture.release()
    if opened and frame_count != 1:
        return "video"

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return "image"

    if opened:
        return "video"

    raise FileNotFoundError(f"Could not identify input type for {path}")


def projection_radius(theta: np.ndarray, theta_max: float, radius: float, model: str) -> np.ndarray:
    if model == "equidistant":
        return radius * theta / theta_max
    if model == "equisolid":
        return radius * np.sin(theta / 2.0) / np.sin(theta_max / 2.0)
    if model == "orthographic":
        return radius * np.sin(theta) / np.sin(theta_max)
    if model == "stereographic":
        return radius * np.tan(theta / 2.0) / np.tan(theta_max / 2.0)
    raise ValueError(f"Unsupported fisheye model: {model}")


def inverse_projection_theta(norm_r: np.ndarray, theta_max: float, model: str) -> np.ndarray:
    eps = 1e-6
    norm_r = np.clip(norm_r, 0.0, 1.0)
    if model == "equidistant":
        return norm_r * theta_max
    if model == "equisolid":
        return 2.0 * np.arcsin(np.clip(norm_r * np.sin(theta_max / 2.0), 0.0, 1.0))
    if model == "orthographic":
        return np.arcsin(np.clip(norm_r * np.sin(theta_max), 0.0, 1.0))
    if model == "stereographic":
        return 2.0 * np.arctan(norm_r * np.tan(theta_max / 2.0) + eps)
    raise ValueError(f"Unsupported fisheye model: {model}")


def normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg <= -90.0:
        angle_deg += 180.0
    while angle_deg > 90.0:
        angle_deg -= 180.0
    return angle_deg


def detect_fisheye_geometry(frame_rgb: np.ndarray, black_threshold: int, edge_trim_px: float) -> Dict[str, float]:
    mask = (frame_rgb.max(axis=2) > black_threshold).astype(np.uint8) * 255
    if not np.any(mask):
        raise RuntimeError("No non-black pixels found while detecting the fisheye footprint.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Could not extract a valid fisheye contour.")

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2.0
        cy = y + h / 2.0
        radius_x = max(w / 2.0 - edge_trim_px, 1.0)
        radius_y = max(h / 2.0 - edge_trim_px, 1.0)
        angle_deg = 0.0
    else:
        (cx, cy), (axis_a, axis_b), angle_deg = cv2.fitEllipse(contour)
        if axis_a >= axis_b:
            radius_x = max(axis_a / 2.0 - edge_trim_px, 1.0)
            radius_y = max(axis_b / 2.0 - edge_trim_px, 1.0)
            angle_deg = normalize_angle_deg(angle_deg)
        else:
            radius_x = max(axis_b / 2.0 - edge_trim_px, 1.0)
            radius_y = max(axis_a / 2.0 - edge_trim_px, 1.0)
            angle_deg = normalize_angle_deg(angle_deg - 90.0)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "radius_x": float(radius_x),
        "radius_y": float(radius_y),
        "angle_deg": float(angle_deg),
        "footprint_area": float(math.pi * radius_x * radius_y),
    }


def estimate_video_geometry(input_path: Path, args: argparse.Namespace, candidate_count: int = 5) -> Dict[str, float]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video {input_path} while estimating geometry.")

    best_geometry = None
    try:
        for candidate_idx in range(candidate_count):
            frame_index = candidate_idx * max(args.sample_stride, 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            geometry = detect_fisheye_geometry(frame_rgb, args.black_threshold, args.edge_trim_px)
            geometry["reference_frame_index"] = frame_index
            if best_geometry is None or geometry["footprint_area"] < best_geometry["footprint_area"]:
                best_geometry = geometry
    finally:
        capture.release()

    if best_geometry is None:
        raise RuntimeError("Could not estimate fisheye geometry from the video.")
    return best_geometry


def build_erp_remap(
    erp_height: int,
    erp_width: int,
    geometry: Dict[str, float],
    fisheye_fov_deg: float,
    fisheye_model: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_max = math.radians(fisheye_fov_deg / 2.0)
    rows = (np.arange(erp_height, dtype=np.float32) + 0.5) / erp_height
    cols = (np.arange(erp_width, dtype=np.float32) + 0.5) / erp_width
    lat = (0.5 - rows) * np.pi
    lon = cols * 2.0 * np.pi - np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x = np.cos(lat_grid) * np.sin(lon_grid)
    y = np.cos(lat_grid) * np.cos(lon_grid)
    z = -np.sin(lat_grid)

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    psi = np.arctan2(x, y)
    valid = theta <= theta_max
    radial = projection_radius(theta, theta_max, 1.0, fisheye_model)

    local_x = geometry["radius_x"] * radial * np.sin(psi)
    local_y = geometry["radius_y"] * radial * np.cos(psi)
    angle_rad = math.radians(geometry["angle_deg"])
    dx = local_x * math.cos(angle_rad) - local_y * math.sin(angle_rad)
    dy = local_x * math.sin(angle_rad) + local_y * math.cos(angle_rad)

    map_x = (geometry["cx"] + dx).astype(np.float32)
    map_y = (geometry["cy"] - dy).astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid


def build_fisheye_to_erp_remap(
    frame_height: int,
    frame_width: int,
    geometry: Dict[str, float],
    erp_height: int,
    erp_width: int,
    fisheye_fov_deg: float,
    fisheye_model: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_max = math.radians(fisheye_fov_deg / 2.0)
    xs = np.arange(frame_width, dtype=np.float32)
    ys = np.arange(frame_height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dx = grid_x - geometry["cx"]
    dy = geometry["cy"] - grid_y
    angle_rad = math.radians(geometry["angle_deg"])
    local_x = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
    local_y = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    norm_x = local_x / geometry["radius_x"]
    norm_y = local_y / geometry["radius_y"]
    norm_r = np.sqrt(norm_x * norm_x + norm_y * norm_y)
    valid = norm_r <= 1.0

    theta = inverse_projection_theta(norm_r, theta_max, fisheye_model)
    psi = np.arctan2(norm_x, norm_y)

    x = np.sin(theta) * np.sin(psi)
    y = np.sin(theta) * np.cos(psi)
    z = np.cos(theta)
    lat = -np.arcsin(np.clip(z, -1.0, 1.0))
    lon = np.arctan2(x, y)

    erp_x = ((lon + np.pi) / (2.0 * np.pi) * erp_width - 0.5).astype(np.float32)
    erp_y = ((0.5 - lat / np.pi) * erp_height - 0.5).astype(np.float32)
    erp_x[~valid] = -1.0
    erp_y[~valid] = -1.0
    return erp_x, erp_y, valid


def fill_unseen_zenith(erp_rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    filled = erp_rgb.copy()
    height, width = valid_mask.shape
    for col in range(width):
        valid_rows = np.flatnonzero(valid_mask[:, col])
        if valid_rows.size == 0:
            continue
        top = int(valid_rows[0])
        if top > 0:
            filled[:top, col] = filled[top, col]

    cap_rows = np.flatnonzero(~valid_mask.any(axis=1))
    if cap_rows.size:
        cap_height = int(cap_rows[-1]) + 1
        blurred = cv2.GaussianBlur(
            filled,
            (0, 0),
            sigmaX=max(width / 80.0, 1.0),
            sigmaY=max(cap_height / 6.0, 1.0),
        )
        filled[:cap_height] = blurred[:cap_height]
    return filled


def prepare_dap_imports(dap_root: Path) -> None:
    if str(dap_root) not in sys.path:
        sys.path.insert(0, str(dap_root))


def load_model(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, int, int]:
    dap_root = Path(args.dap_root).resolve()
    if not dap_root.exists():
        raise FileNotFoundError(f"DAP repo path does not exist: {dap_root}")

    prepare_dap_imports(dap_root)
    from networks.models import make  # type: ignore

    spec = {
        "name": "dap",
        "args": {
            "midas_model_type": args.midas_model_type,
            "fine_tune_type": args.fine_tune_type,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "train_decoder": True,
        },
    }
    old_cwd = Path.cwd()
    os.chdir(dap_root)
    try:
        model = make(spec)
    finally:
        os.chdir(old_cwd)

    state = torch.load(str(Path(args.model_path).resolve()), map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        model = nn.DataParallel(model)

    model = model.to(device)
    model_state = model.state_dict()
    if isinstance(state, dict):
        state = {k: v for k, v in state.items() if k in model_state}
        model.load_state_dict(state, strict=False)
    else:
        raise RuntimeError("Unexpected checkpoint format for DAP model.")

    model.eval()
    return model, int(args.erp_height), int(args.erp_width)


def infer_depth(model: torch.nn.Module, erp_rgb_filled: np.ndarray, device: torch.device) -> np.ndarray:
    image = erp_rgb_filled.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
        if isinstance(outputs, dict) and "pred_depth" in outputs:
            pred = outputs["pred_depth"][0].detach().cpu().squeeze().numpy()
        else:
            pred = outputs[0].detach().cpu().squeeze().numpy()
    pred = np.asarray(pred, dtype=np.float32)
    pred[~np.isfinite(pred)] = 0.0
    pred[pred < 0.0] = 0.0
    return pred


def colorize_depth(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    depth_vis = depth.copy()
    depth_vis[~valid_mask] = 0.0
    positive = depth_vis > 0
    if not np.any(positive):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    normalized = depth_vis / depth_vis[positive].max()
    disparity = np.zeros_like(normalized, dtype=np.float32)
    disparity[positive] = 1.0 / np.maximum(normalized[positive], 1e-8)
    depth_rgb = kitti_colormap(disparity)
    depth_rgb[~valid_mask] = 0
    return depth_rgb


def save_circle_preview(path: Path, frame_rgb: np.ndarray, geometry: Dict[str, float]) -> None:
    preview = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    center = (int(round(geometry["cx"])), int(round(geometry["cy"])))
    axes = (int(round(geometry["radius_x"])), int(round(geometry["radius_y"])))
    cv2.ellipse(preview, center, axes, geometry["angle_deg"], 0, 360, (0, 255, 255), 3)
    cv2.circle(preview, center, 4, (0, 0, 255), -1)
    cv2.imwrite(str(path), preview)


def erp_to_pointcloud_arrays(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    point_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    sample_mask = valid_mask.copy()
    if point_stride > 1:
        stride_mask = np.zeros_like(valid_mask, dtype=bool)
        stride_mask[::point_stride, ::point_stride] = True
        sample_mask &= stride_mask

    theta = np.pi - (np.arange(h, dtype=np.float32).reshape(h, 1) + 0.5) * np.pi / h
    theta = np.repeat(theta, w, axis=1)
    phi = (np.arange(w, dtype=np.float32).reshape(1, w) + 0.5) * 2.0 * np.pi / w - np.pi
    phi = np.repeat(phi, h, axis=0)

    masked_depth = depth[sample_mask]
    x = masked_depth * np.sin(theta[sample_mask]) * np.sin(phi[sample_mask])
    y = masked_depth * np.cos(theta[sample_mask])
    z = masked_depth * np.sin(theta[sample_mask]) * np.cos(phi[sample_mask])
    xyz = np.stack([x, y, z], axis=1)
    colors = rgb[sample_mask].astype(np.float32) / 255.0
    return xyz, colors


def maybe_write_pointcloud(path: Path, rgb: np.ndarray, depth: np.ndarray, valid_mask: np.ndarray) -> None:
    try:
        import open3d as o3d
    except ImportError:
        return

    xyz, colors = erp_to_pointcloud_arrays(rgb, depth, valid_mask, point_stride=1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), cloud)


def process_frame(
    frame_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    black_threshold: int,
    erp_map_x: np.ndarray,
    erp_map_y: np.ndarray,
    erp_valid_mask: np.ndarray,
    fisheye_map_x: np.ndarray,
    fisheye_map_y: np.ndarray,
    fisheye_valid_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    erp_rgb = cv2.remap(
        frame_rgb,
        erp_map_x,
        erp_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    erp_filled = fill_unseen_zenith(erp_rgb, erp_valid_mask)
    erp_depth = infer_depth(model, erp_filled, device)
    erp_depth_masked = erp_depth.copy()
    erp_depth_masked[~erp_valid_mask] = 0.0
    erp_depth_rgb = colorize_depth(erp_depth_masked, erp_valid_mask)

    fisheye_depth = cv2.remap(
        erp_depth_masked,
        fisheye_map_x,
        fisheye_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    observed_mask = fisheye_valid_mask & (frame_rgb.max(axis=2) > black_threshold)
    fisheye_depth_rgb = colorize_depth(fisheye_depth, observed_mask)
    overlay = frame_rgb.copy()
    overlay[observed_mask] = (
        0.55 * overlay[observed_mask] + 0.45 * fisheye_depth_rgb[observed_mask]
    ).astype(np.uint8)

    return {
        "erp_rgb": erp_rgb,
        "erp_filled": erp_filled,
        "erp_valid_mask": (erp_valid_mask.astype(np.uint8) * 255),
        "erp_depth": erp_depth_masked,
        "erp_depth_rgb": erp_depth_rgb,
        "fisheye_depth": fisheye_depth,
        "fisheye_depth_rgb": fisheye_depth_rgb,
        "fisheye_valid_mask": (observed_mask.astype(np.uint8) * 255),
        "overlay": overlay,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_debug_outputs(
    frame_id: str,
    outputs: Dict[str, np.ndarray],
    original_rgb: np.ndarray,
    debug_dir: Path,
    save_pointcloud: bool,
) -> None:
    ensure_dir(debug_dir)
    cv2.imwrite(str(debug_dir / f"{frame_id}_original.jpg"), cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / f"{frame_id}_erp_valid.jpg"), cv2.cvtColor(outputs["erp_rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / f"{frame_id}_erp_filled.jpg"), cv2.cvtColor(outputs["erp_filled"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / f"{frame_id}_erp_mask.png"), outputs["erp_valid_mask"])
    cv2.imwrite(str(debug_dir / f"{frame_id}_fisheye_mask.png"), outputs["fisheye_valid_mask"])
    cv2.imwrite(str(debug_dir / f"{frame_id}_erp_depth.jpg"), cv2.cvtColor(outputs["erp_depth_rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(debug_dir / f"{frame_id}_fisheye_depth.jpg"),
        cv2.cvtColor(outputs["fisheye_depth_rgb"], cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(debug_dir / f"{frame_id}_overlay.jpg"), cv2.cvtColor(outputs["overlay"], cv2.COLOR_RGB2BGR))
    np.save(str(debug_dir / f"{frame_id}_erp_depth.npy"), outputs["erp_depth"])
    np.save(str(debug_dir / f"{frame_id}_fisheye_depth.npy"), outputs["fisheye_depth"])
    if save_pointcloud:
        maybe_write_pointcloud(
            debug_dir / f"{frame_id}_erp_pointcloud.ply",
            outputs["erp_rgb"],
            outputs["erp_depth"],
            outputs["erp_valid_mask"] > 0,
        )


def init_video_writer(path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, frame_size)


def run_image(
    input_path: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    erp_height: int,
    erp_width: int,
) -> None:
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image {input_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    geometry = detect_fisheye_geometry(image_rgb, args.black_threshold, args.edge_trim_px)
    erp_map_x, erp_map_y, erp_valid_mask = build_erp_remap(
        erp_height, erp_width, geometry, args.fisheye_fov_deg, args.fisheye_model
    )
    fisheye_map_x, fisheye_map_y, fisheye_valid_mask = build_fisheye_to_erp_remap(
        image_rgb.shape[0],
        image_rgb.shape[1],
        geometry,
        erp_height,
        erp_width,
        args.fisheye_fov_deg,
        args.fisheye_model,
    )
    save_circle_preview(output_dir / "circle_preview.jpg", image_rgb, geometry)

    outputs = process_frame(
        image_rgb,
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
    save_debug_outputs("image", outputs, image_rgb, output_dir / "debug", args.save_pointcloud)

    metadata = {
        "input": str(input_path),
        "input_type": "image",
        "dap_root": str(Path(args.dap_root).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "model_name": args.model_name,
        "midas_model_type": args.midas_model_type,
        "fine_tune_type": args.fine_tune_type,
        "fisheye_model": args.fisheye_model,
        "fisheye_fov_deg": args.fisheye_fov_deg,
        "erp_height": erp_height,
        "erp_width": erp_width,
        "geometry": geometry,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_video(
    input_path: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    erp_height: int,
    erp_width: int,
) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video {input_path}")

    input_fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    processed_fps = input_fps / max(args.sample_stride, 1)
    debug_dir = output_dir / "debug"
    ensure_dir(debug_dir)

    writer_depth = None
    writer_overlay = None
    if args.write_video:
        video_dir = output_dir / "video"
        ensure_dir(video_dir)
        writer_depth = init_video_writer(video_dir / "fisheye_depth.mp4", processed_fps, (source_width, source_height))
        writer_overlay = init_video_writer(video_dir / "overlay.mp4", processed_fps, (source_width, source_height))

    geometry = estimate_video_geometry(input_path, args)
    erp_map_x = erp_map_y = erp_valid_mask = None
    fisheye_map_x = fisheye_map_y = fisheye_valid_mask = None

    frame_index = 0
    processed_index = 0
    max_processed = args.max_frames if args.max_frames is not None else math.inf
    estimated_total = int(math.ceil(frame_count / max(args.sample_stride, 1))) if frame_count > 0 else None
    progress = tqdm.tqdm(total=estimated_total, desc="Processing video")

    try:
        while processed_index < max_processed:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            if frame_index % args.sample_stride != 0:
                frame_index += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if erp_map_x is None:
                erp_map_x, erp_map_y, erp_valid_mask = build_erp_remap(
                    erp_height, erp_width, geometry, args.fisheye_fov_deg, args.fisheye_model
                )
                fisheye_map_x, fisheye_map_y, fisheye_valid_mask = build_fisheye_to_erp_remap(
                    frame_rgb.shape[0],
                    frame_rgb.shape[1],
                    geometry,
                    erp_height,
                    erp_width,
                    args.fisheye_fov_deg,
                    args.fisheye_model,
                )
                save_circle_preview(output_dir / "circle_preview.jpg", frame_rgb, geometry)

            outputs = process_frame(
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

            frame_id = f"frame_{frame_index:06d}"
            if processed_index < args.save_debug_frames:
                save_debug_outputs(
                    frame_id,
                    outputs,
                    frame_rgb,
                    debug_dir,
                    save_pointcloud=args.save_pointcloud and processed_index == 0,
                )

            if writer_depth is not None:
                writer_depth.write(cv2.cvtColor(outputs["fisheye_depth_rgb"], cv2.COLOR_RGB2BGR))
            if writer_overlay is not None:
                writer_overlay.write(cv2.cvtColor(outputs["overlay"], cv2.COLOR_RGB2BGR))

            processed_index += 1
            frame_index += 1
            progress.update(1)

        metadata = {
            "input": str(input_path),
            "input_type": "video",
            "dap_root": str(Path(args.dap_root).resolve()),
            "model_path": str(Path(args.model_path).resolve()),
            "model_name": args.model_name,
            "midas_model_type": args.midas_model_type,
            "fine_tune_type": args.fine_tune_type,
            "fisheye_model": args.fisheye_model,
            "fisheye_fov_deg": args.fisheye_fov_deg,
            "erp_height": erp_height,
            "erp_width": erp_width,
            "input_fps": input_fps,
            "processed_fps": processed_fps,
            "sample_stride": args.sample_stride,
            "processed_frames": processed_index,
            "source_frames": frame_count,
            "geometry": geometry,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        progress.close()
        capture.release()
        if writer_depth is not None:
            writer_depth.release()
        if writer_overlay is not None:
            writer_overlay.release()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    model, erp_height, erp_width = load_model(args, device)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (ROOT / "maritime_output" / f"{input_path.stem}_{args.model_name}").resolve()
    )
    ensure_dir(output_dir)

    input_type = detect_input_type(input_path, args.input_type)
    if input_type == "image":
        run_image(input_path, args, model, device, output_dir, erp_height, erp_width)
    else:
        run_video(input_path, args, model, device, output_dir, erp_height, erp_width)


if __name__ == "__main__":
    main()
