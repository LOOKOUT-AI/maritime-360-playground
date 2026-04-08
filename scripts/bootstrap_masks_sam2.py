from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap boat/water masks on a maritime frame using SAM 2.")
    parser.add_argument("--image", required=True, type=str, help="Path to the RGB frame.")
    parser.add_argument("--sam2_root", required=True, type=str, help="Path to the cloned facebookresearch/sam2 repo.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the SAM 2 checkpoint.")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml", type=str, help="SAM 2 config path.")
    parser.add_argument("--output_dir", required=True, type=str, help="Where to save masks and overlays.")
    return parser.parse_args()


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
            border_touch = np.mean(
                np.concatenate([seg[0], seg[-1], seg[:, 0], seg[:, -1]]).astype(np.float32)
            )
            score = (
                mask["area"],
                mask["predicted_iou"],
                mask["stability_score"],
                -border_touch,
            )
            candidates.append((score, idx, mask))
        candidates.sort(reverse=True)
        for _, idx, mask in candidates[:3]:
            if idx not in used:
                selected.append((idx, mask))
                used.add(idx)
        if len(selected) >= max_masks:
            break
    return selected


def save_overlay(path: Path, image: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
    canvas = image.copy()
    canvas[mask] = (0.55 * canvas[mask] + 0.45 * color).astype(np.uint8)
    Image.fromarray(canvas).save(path)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    sam2_root = Path(args.sam2_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(sam2_root)

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = np.array(Image.open(image_path).convert("RGB"))
    valid_region = image.max(axis=2) > 20

    # These prompts match the static boat-centric framing of the camera and are intended
    # only as a first-pass bootstrap, not final masks.
    boat_seed_points = [(960, 290), (960, 520), (960, 770), (760, 520), (1160, 520)]
    water_seed_points = [(420, 760), (1510, 760), (300, 610), (1650, 610)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.config, str(checkpoint), device=device)

    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=800,
    )
    with torch.inference_mode():
        masks = mask_generator.generate(image)

    boat_selected = select_seed_masks(masks, boat_seed_points, reject_points=water_seed_points, max_masks=12)
    water_selected = select_seed_masks(masks, water_seed_points, reject_points=boat_seed_points, max_masks=12)

    boat_auto = np.zeros(valid_region.shape, dtype=bool)
    for _, mask in boat_selected:
        boat_auto |= mask["segmentation"]
    boat_auto &= valid_region

    water_auto = np.zeros(valid_region.shape, dtype=bool)
    for _, mask in water_selected:
        water_auto |= mask["segmentation"]
    water_auto &= valid_region & (~boat_auto)

    predictor = SAM2ImagePredictor(model)
    boat_points = np.array(
        [
            [960, 290],
            [960, 520],
            [960, 770],
            [760, 520],
            [1160, 520],
            [870, 845],
            [1050, 845],
            [420, 760],
            [1510, 760],
            [300, 610],
            [1650, 610],
        ],
        dtype=np.float32,
    )
    boat_labels = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    boat_box = np.array([250, 40, 1690, 975], dtype=np.float32)

    water_points = np.array(
        [
            [420, 760],
            [1510, 760],
            [300, 610],
            [1650, 610],
            [960, 290],
            [960, 520],
            [960, 770],
            [760, 520],
            [1160, 520],
            [330, 180],
            [1610, 180],
        ],
        dtype=np.float32,
    )
    water_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    water_box = np.array([120, 300, 1800, 1050], dtype=np.float32)

    with torch.inference_mode():
        predictor.set_image(image)
        boat_prompted, boat_ious, _ = predictor.predict(
            point_coords=boat_points,
            point_labels=boat_labels,
            box=boat_box,
            multimask_output=False,
            normalize_coords=False,
        )
        predictor.set_image(image)
        water_prompted, water_ious, _ = predictor.predict(
            point_coords=water_points,
            point_labels=water_labels,
            box=water_box,
            multimask_output=False,
            normalize_coords=False,
        )

    boat_prompted_mask = boat_prompted[0].astype(bool) & valid_region
    water_prompted_mask = water_prompted[0].astype(bool) & valid_region & (~boat_prompted_mask)

    Image.fromarray((boat_auto.astype(np.uint8) * 255)).save(output_dir / "boat_mask_sam2_auto.png")
    Image.fromarray((water_auto.astype(np.uint8) * 255)).save(output_dir / "water_mask_sam2_auto.png")
    Image.fromarray((boat_prompted_mask.astype(np.uint8) * 255)).save(output_dir / "boat_mask_sam2_prompted.png")
    Image.fromarray((water_prompted_mask.astype(np.uint8) * 255)).save(output_dir / "water_mask_sam2_prompted.png")

    overview = image.copy()
    rng = np.random.default_rng(0)
    for mask in sorted(masks, key=lambda m: m["area"], reverse=True)[:80]:
        seg = mask["segmentation"]
        color = rng.integers(64, 255, size=3, dtype=np.uint8)
        overview[seg] = (0.65 * overview[seg] + 0.35 * color).astype(np.uint8)
    Image.fromarray(overview).save(output_dir / "sam2_masks_overview.jpg")

    save_overlay(output_dir / "boat_mask_auto_overlay.jpg", image, boat_auto, np.array([255, 80, 80], dtype=np.uint8))
    save_overlay(output_dir / "water_mask_auto_overlay.jpg", image, water_auto, np.array([80, 160, 255], dtype=np.uint8))
    save_overlay(
        output_dir / "boat_mask_prompted_overlay.jpg",
        image,
        boat_prompted_mask,
        np.array([255, 80, 80], dtype=np.uint8),
    )
    save_overlay(
        output_dir / "water_mask_prompted_overlay.jpg",
        image,
        water_prompted_mask,
        np.array([80, 160, 255], dtype=np.uint8),
    )

    summary = {
        "image_path": str(image_path),
        "checkpoint": str(checkpoint),
        "config": args.config,
        "mask_count": len(masks),
        "boat_seed_points": boat_seed_points,
        "water_seed_points": water_seed_points,
        "boat_auto_mask_indices": [idx for idx, _ in boat_selected],
        "water_auto_mask_indices": [idx for idx, _ in water_selected],
        "boat_prompt_box": boat_box.tolist(),
        "boat_prompt_points": boat_points.tolist(),
        "boat_prompt_labels": boat_labels.tolist(),
        "boat_prompt_iou": float(boat_ious[0]),
        "water_prompt_box": water_box.tolist(),
        "water_prompt_points": water_points.tolist(),
        "water_prompt_labels": water_labels.tolist(),
        "water_prompt_iou": float(water_ious[0]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
