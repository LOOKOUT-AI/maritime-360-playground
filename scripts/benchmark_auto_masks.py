from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class SemanticBundle:
    model_id: str
    processor: object
    model: object
    task_kind: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg <= -90.0:
        angle_deg += 180.0
    while angle_deg > 90.0:
        angle_deg -= 180.0
    return angle_deg


def detect_fisheye_geometry(frame_rgb: np.ndarray, black_threshold: int, edge_trim_px: float) -> dict[str, float]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark automatic boat/water masking methods on sampled video frames.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--sample_times_sec", default="0,15,30,45,59", type=str)
    parser.add_argument("--sam2_root", required=True, type=str)
    parser.add_argument("--sam2_checkpoint", required=True, type=str)
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_s.yaml", type=str)
    parser.add_argument("--segformer_model", default="nvidia/segformer-b5-finetuned-ade-640-640", type=str)
    parser.add_argument("--mask2former_model", default="facebook/mask2former-swin-base-ade-semantic", type=str)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def parse_times(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def read_video_frame_at_time(video_path: Path, time_sec: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = int(round(time_sec * fps))
    if frame_count > 0:
        frame_index = int(np.clip(frame_index, 0, frame_count - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def ellipse_valid_mask(shape: tuple[int, int], geometry: dict[str, float]) -> np.ndarray:
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    xs -= float(geometry["cx"])
    ys -= float(geometry["cy"])
    theta = math.radians(float(geometry["angle_deg"]))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xr = cos_t * xs + sin_t * ys
    yr = -sin_t * xs + cos_t * ys
    rx = max(float(geometry["radius_x"]), 1.0)
    ry = max(float(geometry["radius_y"]), 1.0)
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def relative_point(geometry: dict[str, float], x_frac: float, y_frac: float) -> tuple[float, float]:
    rx = float(geometry["radius_x"])
    ry = float(geometry["radius_y"])
    theta = math.radians(float(geometry["angle_deg"]))
    dx = x_frac * rx
    dy = y_frac * ry
    xr = math.cos(theta) * dx - math.sin(theta) * dy
    yr = math.sin(theta) * dx + math.cos(theta) * dy
    return float(geometry["cx"] + xr), float(geometry["cy"] + yr)


def geometry_prompt_config(geometry: dict[str, float]) -> dict[str, object]:
    boat_points = np.array(
        [
            relative_point(geometry, 0.0, -0.45),
            relative_point(geometry, 0.0, -0.05),
            relative_point(geometry, 0.0, 0.30),
            relative_point(geometry, -0.22, 0.08),
            relative_point(geometry, 0.22, 0.08),
            relative_point(geometry, 0.0, 0.62),
            relative_point(geometry, -0.65, 0.45),
            relative_point(geometry, 0.65, 0.45),
            relative_point(geometry, -0.82, 0.18),
            relative_point(geometry, 0.82, 0.18),
        ],
        dtype=np.float32,
    )
    boat_labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    water_points = np.array(
        [
            relative_point(geometry, -0.65, 0.52),
            relative_point(geometry, 0.65, 0.52),
            relative_point(geometry, -0.85, 0.20),
            relative_point(geometry, 0.85, 0.20),
            relative_point(geometry, 0.0, -0.45),
            relative_point(geometry, 0.0, 0.0),
            relative_point(geometry, 0.0, 0.35),
            relative_point(geometry, -0.22, 0.08),
            relative_point(geometry, 0.22, 0.08),
        ],
        dtype=np.float32,
    )
    water_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
    cx = float(geometry["cx"])
    cy = float(geometry["cy"])
    rx = float(geometry["radius_x"])
    ry = float(geometry["radius_y"])
    boat_box = np.array([cx - 0.58 * rx, cy - 0.90 * ry, cx + 0.58 * rx, cy + 0.95 * ry], dtype=np.float32)
    water_box = np.array([cx - 0.95 * rx, cy - 0.05 * ry, cx + 0.95 * rx, cy + 1.05 * ry], dtype=np.float32)
    boat_seed_points = [tuple(map(int, map(round, point))) for point in boat_points[:6]]
    water_seed_points = [tuple(map(int, map(round, point))) for point in water_points[:4]]
    return {
        "boat_points": boat_points,
        "boat_labels": boat_labels,
        "boat_box": boat_box,
        "water_points": water_points,
        "water_labels": water_labels,
        "water_box": water_box,
        "boat_seed_points": boat_seed_points,
        "water_seed_points": water_seed_points,
    }


def mask_contains(mask: dict, point: tuple[int, int]) -> bool:
    x, y = point
    seg = mask["segmentation"]
    return 0 <= y < seg.shape[0] and 0 <= x < seg.shape[1] and bool(seg[y, x])


def select_seed_masks(
    masks: list[dict],
    seed_points: Iterable[tuple[int, int]],
    reject_points: Iterable[tuple[int, int]] = (),
    max_masks: int = 12,
) -> list[tuple[int, dict]]:
    selected: list[tuple[int, dict]] = []
    used: set[int] = set()
    reject_points = list(reject_points)
    for seed in seed_points:
        candidates = []
        for idx, mask in enumerate(masks):
            if idx in used or not mask_contains(mask, seed):
                continue
            if any(mask_contains(mask, pt) for pt in reject_points):
                continue
            seg = mask["segmentation"]
            border_touch = np.mean(np.concatenate([seg[0], seg[-1], seg[:, 0], seg[:, -1]]).astype(np.float32))
            score = (mask["predicted_iou"], mask["stability_score"], mask["area"], -border_touch)
            candidates.append((score, idx, mask))
        candidates.sort(reverse=True)
        for _, idx, mask in candidates[:3]:
            if idx not in used:
                selected.append((idx, mask))
                used.add(idx)
        if len(selected) >= max_masks:
            break
    return selected


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def overlay_image(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    if np.any(mask):
        canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.array(color, dtype=np.uint8)).astype(np.uint8)
    return canvas


def save_overlay(path: Path, image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    Image.fromarray(overlay_image(image, mask, color)).save(path)


def label_ids_for_keywords(id2label: dict[int, str], keywords: Iterable[str]) -> list[int]:
    lowered = {idx: label.lower() for idx, label in id2label.items()}
    result = []
    for idx, label in lowered.items():
        if any(keyword in label for keyword in keywords):
            result.append(int(idx))
    return sorted(set(result))


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask.astype(np.uint8))
    if count <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
    largest_label = int(np.argmax(sizes) + 1)
    return labeled == largest_label


def distance_peak_points(mask: np.ndarray, count: int, min_distance_px: int = 24) -> list[tuple[float, float]]:
    if not np.any(mask):
        return []
    distance = ndimage.distance_transform_edt(mask)
    points: list[tuple[float, float]] = []
    work = distance.copy()
    for _ in range(count):
        index = np.argmax(work)
        if work.flat[index] <= 0.0:
            break
        y, x = np.unravel_index(index, work.shape)
        points.append((float(x), float(y)))
        y0 = max(0, y - min_distance_px)
        y1 = min(work.shape[0], y + min_distance_px + 1)
        x0 = max(0, x - min_distance_px)
        x1 = min(work.shape[1], x + min_distance_px + 1)
        work[y0:y1, x0:x1] = 0.0
    return points


def bbox_from_mask(mask: np.ndarray, pad_px: int = 16) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) - pad_px)
    y0 = max(0, int(ys.min()) - pad_px)
    x1 = min(mask.shape[1] - 1, int(xs.max()) + pad_px)
    y1 = min(mask.shape[0] - 1, int(ys.max()) + pad_px)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def compute_metrics(mask: np.ndarray, valid_mask: np.ndarray, geometry: dict[str, float], kind: str) -> dict[str, float]:
    valid_area = max(float(valid_mask.sum()), 1.0)
    area = float(mask.sum())
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).astype(np.float32)

    center_mask = np.zeros_like(mask, dtype=bool)
    lower_mask = np.zeros_like(mask, dtype=bool)
    cx = float(geometry["cx"])
    cy = float(geometry["cy"])
    rx = float(geometry["radius_x"])
    ry = float(geometry["radius_y"])
    ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype(np.float32)
    center_mask = (((xs - cx) / max(rx * 0.35, 1.0)) ** 2 + ((ys - cy) / max(ry * 0.55, 1.0)) ** 2) <= 1.0
    lower_mask = valid_mask & (ys >= cy + 0.15 * ry)
    outer_ring = valid_mask & ((((xs - cx) / max(rx * 0.75, 1.0)) ** 2 + ((ys - cy) / max(ry * 0.75, 1.0)) ** 2) >= 1.0)

    labeled, count = ndimage.label(mask.astype(np.uint8))
    component_sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1)) if count else np.array([], dtype=np.float32)
    largest_frac = float(component_sizes.max() / max(area, 1.0)) if component_sizes.size else 0.0

    metrics = {
        "area_frac": area / valid_area,
        "border_touch": float(border.mean()),
        "center_frac": float((mask & center_mask).sum() / valid_area),
        "lower_frac": float((mask & lower_mask).sum() / valid_area),
        "outer_ring_frac": float((mask & outer_ring).sum() / valid_area),
        "largest_component_frac": largest_frac,
        "component_count": int(count),
    }
    if kind == "boat":
        score = (
            4.0 * metrics["center_frac"]
            + 1.5 * metrics["largest_component_frac"]
            + 0.5 * metrics["area_frac"]
            - 1.5 * metrics["border_touch"]
            - 1.0 * metrics["outer_ring_frac"]
        )
    else:
        score = (
            2.5 * metrics["lower_frac"]
            + 2.0 * metrics["outer_ring_frac"]
            + 1.0 * metrics["largest_component_frac"]
            + 0.5 * metrics["area_frac"]
            - 1.2 * metrics["center_frac"]
        )
    metrics["score"] = float(score)
    return metrics


def create_contact_sheet(frame_dir: Path, frame_image: np.ndarray, method_results: dict[str, dict[str, np.ndarray]]) -> None:
    methods = list(method_results.keys())
    thumb_width = 420
    thumb_height = int(frame_image.shape[0] * (thumb_width / frame_image.shape[1]))
    rows = []
    rows.append(("original", Image.fromarray(frame_image).resize((thumb_width, thumb_height))))
    for method_name in methods:
        boat_overlay = Image.fromarray(
            overlay_image(frame_image, method_results[method_name]["boat"], (255, 80, 80))
        ).resize((thumb_width, thumb_height))
        water_overlay = Image.fromarray(
            overlay_image(frame_image, method_results[method_name]["water"], (80, 160, 255))
        ).resize((thumb_width, thumb_height))
        rows.append((f"{method_name} boat", boat_overlay))
        rows.append((f"{method_name} water", water_overlay))

    label_width = 260
    sheet = Image.new("RGB", (label_width + thumb_width, len(rows) * thumb_height), color=(16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for row_idx, (label, image) in enumerate(rows):
        y = row_idx * thumb_height
        sheet.paste(image, (label_width, y))
        draw.text((16, y + 16), label, fill=(240, 240, 240))
    sheet.save(frame_dir / "contact_sheet.jpg")


def load_semantic_bundle(model_id: str, device: torch.device, task_kind: str) -> SemanticBundle:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(model_id)
    if task_kind == "semantic":
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id, use_safetensors=True)
    elif task_kind == "universal":
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, use_safetensors=True)
    else:
        raise ValueError(f"Unsupported task kind: {task_kind}")
    model.to(device)
    model.eval()
    return SemanticBundle(model_id=model_id, processor=processor, model=model, task_kind=task_kind)


def run_semantic_bundle(
    image: np.ndarray,
    valid_mask: np.ndarray,
    bundle: SemanticBundle,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    id2label = {int(k): str(v) for k, v in bundle.model.config.id2label.items()}
    boat_ids = label_ids_for_keywords(id2label, ("boat", "ship", "vessel"))
    water_ids = label_ids_for_keywords(id2label, ("water", "sea", "river", "lake", "ocean"))
    inputs = bundle.processor(images=Image.fromarray(image), return_tensors="pt")
    inputs = {key: value.to(bundle.model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = bundle.model(**inputs)
    segmentation = bundle.processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.shape[:2]],
    )[0].detach().cpu().numpy()
    boat_mask = np.isin(segmentation, boat_ids) & valid_mask
    water_mask = np.isin(segmentation, water_ids) & valid_mask & (~boat_mask)
    return boat_mask, water_mask, {
        "label_ids": {"boat": boat_ids, "water": water_ids},
        "labels": {str(idx): id2label[idx] for idx in sorted(set(boat_ids + water_ids))},
    }


def run_sam2_methods(image: np.ndarray, valid_mask: np.ndarray, prompt_cfg: dict[str, object], sam2_model: object) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=800,
    )
    predictor = SAM2ImagePredictor(sam2_model)

    with torch.inference_mode():
        masks = generator.generate(image)

    boat_selected = select_seed_masks(
        masks,
        prompt_cfg["boat_seed_points"],
        reject_points=prompt_cfg["water_seed_points"],
        max_masks=12,
    )
    water_selected = select_seed_masks(
        masks,
        prompt_cfg["water_seed_points"],
        reject_points=prompt_cfg["boat_seed_points"],
        max_masks=12,
    )
    boat_auto = np.zeros(valid_mask.shape, dtype=bool)
    for _, mask in boat_selected:
        boat_auto |= mask["segmentation"]
    boat_auto &= valid_mask
    water_auto = np.zeros(valid_mask.shape, dtype=bool)
    for _, mask in water_selected:
        water_auto |= mask["segmentation"]
    water_auto &= valid_mask & (~boat_auto)

    with torch.inference_mode():
        predictor.set_image(image)
        boat_prompted, boat_ious, _ = predictor.predict(
            point_coords=prompt_cfg["boat_points"],
            point_labels=prompt_cfg["boat_labels"],
            box=prompt_cfg["boat_box"],
            multimask_output=False,
            normalize_coords=False,
        )
        predictor.set_image(image)
        water_prompted, water_ious, _ = predictor.predict(
            point_coords=prompt_cfg["water_points"],
            point_labels=prompt_cfg["water_labels"],
            box=prompt_cfg["water_box"],
            multimask_output=False,
            normalize_coords=False,
        )

    results = {
        "sam2_auto_seeded_boat": boat_auto,
        "sam2_auto_seeded_water": water_auto,
        "sam2_prompted_boat": boat_prompted[0].astype(bool) & valid_mask,
        "sam2_prompted_water": water_prompted[0].astype(bool) & valid_mask,
    }
    results["sam2_prompted_water"] &= ~results["sam2_prompted_boat"]
    return results, {
        "auto_mask_count": len(masks),
        "boat_auto_indices": [idx for idx, _ in boat_selected],
        "water_auto_indices": [idx for idx, _ in water_selected],
        "boat_prompt_iou": float(boat_ious[0]),
        "water_prompt_iou": float(water_ious[0]),
    }


def refine_with_sam2(
    image: np.ndarray,
    valid_mask: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    fallback_box: np.ndarray,
    predictor: object,
) -> np.ndarray:
    positive_mask = largest_component(positive_mask & valid_mask)
    if not np.any(positive_mask):
        return np.zeros(valid_mask.shape, dtype=bool)

    box = bbox_from_mask(positive_mask, pad_px=24)
    if box is None:
        box = fallback_box

    pos_points = distance_peak_points(positive_mask, count=6, min_distance_px=40)
    neg_points = distance_peak_points((negative_mask & valid_mask) | (valid_mask & (~positive_mask)), count=6, min_distance_px=40)
    if not pos_points:
        return np.zeros(valid_mask.shape, dtype=bool)

    point_coords = np.array(pos_points + neg_points, dtype=np.float32)
    point_labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)
    with torch.inference_mode():
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
            normalize_coords=False,
        )
    return masks[0].astype(bool) & valid_mask


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sample_times = parse_times(args.sample_times_sec)

    sam2_root = Path(args.sam2_root).resolve()
    os.chdir(sam2_root)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(args.sam2_config, str(Path(args.sam2_checkpoint).resolve()), device=device.type)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    semantic_bundles: dict[str, SemanticBundle] = {}
    model_load_errors: dict[str, str] = {}
    for method_name, model_spec in (
        ("segformer_semantic", (args.segformer_model, "semantic")),
        ("mask2former_semantic", (args.mask2former_model, "universal")),
    ):
        try:
            model_id, task_kind = model_spec
            semantic_bundles[method_name] = load_semantic_bundle(model_id, device, task_kind)
        except Exception as exc:  # pragma: no cover - environment/model download failures should not abort all methods
            model_load_errors[method_name] = repr(exc)

    frame_summaries: list[dict[str, object]] = []
    aggregate_scores: dict[str, dict[str, float]] = {}

    for sample_idx, time_sec in enumerate(sample_times):
        frame_rgb = read_video_frame_at_time(video_path, time_sec)
        frame_dir = output_dir / f"frame_{sample_idx:02d}_{int(round(time_sec)):03d}s"
        ensure_dir(frame_dir)
        Image.fromarray(frame_rgb).save(frame_dir / "frame_rgb.jpg")

        geometry = detect_fisheye_geometry(frame_rgb, black_threshold=40, edge_trim_px=4.0)
        valid_mask = ellipse_valid_mask(frame_rgb.shape[:2], geometry)
        save_mask(frame_dir / "valid_mask.png", valid_mask)

        prompt_cfg = geometry_prompt_config(geometry)
        sam2_results, sam2_summary = run_sam2_methods(frame_rgb, valid_mask, prompt_cfg, sam2_model)

        method_results: dict[str, dict[str, np.ndarray]] = {
            "sam2_auto_seeded": {
                "boat": sam2_results["sam2_auto_seeded_boat"],
                "water": sam2_results["sam2_auto_seeded_water"],
            },
            "sam2_prompted": {
                "boat": sam2_results["sam2_prompted_boat"],
                "water": sam2_results["sam2_prompted_water"],
            },
        }
        method_summaries: dict[str, object] = {
            "sam2_auto_seeded": {"source": "sam2_auto_seeded"},
            "sam2_prompted": {"source": "sam2_prompted", **sam2_summary},
        }

        for method_name, bundle in semantic_bundles.items():
            boat_mask, water_mask, semantic_summary = run_semantic_bundle(frame_rgb, valid_mask, bundle)
            method_results[method_name] = {"boat": boat_mask, "water": water_mask}
            method_summaries[method_name] = semantic_summary

            refined_method_name = method_name.replace("_semantic", "_sam2_refined")
            refined_boat = refine_with_sam2(
                frame_rgb,
                valid_mask,
                positive_mask=boat_mask,
                negative_mask=water_mask,
                fallback_box=prompt_cfg["boat_box"],
                predictor=sam2_predictor,
            )
            refined_water = refine_with_sam2(
                frame_rgb,
                valid_mask,
                positive_mask=water_mask,
                negative_mask=refined_boat | boat_mask,
                fallback_box=prompt_cfg["water_box"],
                predictor=sam2_predictor,
            )
            refined_water &= ~refined_boat
            method_results[refined_method_name] = {"boat": refined_boat, "water": refined_water}
            method_summaries[refined_method_name] = {
                "source_method": method_name,
                "refinement": "sam2_image_predictor",
            }

        per_method_metrics: dict[str, object] = {}
        for method_name, masks in method_results.items():
            method_dir = frame_dir / method_name
            ensure_dir(method_dir)
            save_mask(method_dir / "boat_mask.png", masks["boat"])
            save_mask(method_dir / "water_mask.png", masks["water"])
            save_overlay(method_dir / "boat_overlay.jpg", frame_rgb, masks["boat"], (255, 80, 80))
            save_overlay(method_dir / "water_overlay.jpg", frame_rgb, masks["water"], (80, 160, 255))
            boat_metrics = compute_metrics(masks["boat"], valid_mask, geometry, kind="boat")
            water_metrics = compute_metrics(masks["water"], valid_mask, geometry, kind="water")
            overlap = float((masks["boat"] & masks["water"]).sum() / max(float(valid_mask.sum()), 1.0))
            combined_score = float(boat_metrics["score"] + water_metrics["score"] - 4.0 * overlap)
            per_method_metrics[method_name] = {
                "boat_metrics": boat_metrics,
                "water_metrics": water_metrics,
                "overlap_frac": overlap,
                "combined_score": combined_score,
                "summary": method_summaries.get(method_name, {}),
            }
            stats = aggregate_scores.setdefault(method_name, {"score_sum": 0.0, "frame_count": 0.0})
            stats["score_sum"] += combined_score
            stats["frame_count"] += 1.0

        create_contact_sheet(frame_dir, frame_rgb, method_results)
        frame_summary = {
            "sample_time_sec": time_sec,
            "frame_dir": str(frame_dir),
            "geometry": geometry,
            "methods": per_method_metrics,
        }
        (frame_dir / "summary.json").write_text(json.dumps(frame_summary, indent=2), encoding="utf-8")
        frame_summaries.append(frame_summary)

    ranking = []
    for method_name, stats in aggregate_scores.items():
        average_score = stats["score_sum"] / max(stats["frame_count"], 1.0)
        ranking.append({"method": method_name, "average_score": average_score})
    ranking.sort(key=lambda item: item["average_score"], reverse=True)

    best_method = ranking[0]["method"] if ranking else None
    summary = {
        "video_path": str(video_path),
        "sample_times_sec": sample_times,
        "device": str(device),
        "sam2": {
            "root": str(sam2_root),
            "checkpoint": str(Path(args.sam2_checkpoint).resolve()),
            "config": args.sam2_config,
        },
        "semantic_models_loaded": {name: bundle.model_id for name, bundle in semantic_bundles.items()},
        "model_load_errors": model_load_errors,
        "ranking": ranking,
        "best_method": best_method,
        "frames": frame_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
