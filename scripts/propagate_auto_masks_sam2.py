from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propagate automatic boat/water seed masks through a video with SAM2.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--benchmark_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--sam2_root", required=True, type=str)
    parser.add_argument("--sam2_checkpoint", required=True, type=str)
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_s.yaml", type=str)
    parser.add_argument("--boat_method", default="sam2_auto_seeded", type=str)
    parser.add_argument("--water_method", default="segformer_sam2_refined", type=str)
    parser.add_argument("--frame_step", default=4, type=int)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, frames_dir: Path, frame_step: int) -> tuple[float, list[int], tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved_indices: list[int] = []

    try:
        frame_idx = 0
        saved_idx = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            if frame_idx % max(frame_step, 1) == 0:
                cv2.imwrite(str(frames_dir / f"{saved_idx:05d}.jpg"), frame_bgr)
                saved_indices.append(frame_idx)
                saved_idx += 1
            frame_idx += 1
            if frame_count > 0 and frame_idx >= frame_count:
                break
    finally:
        capture.release()

    return fps, saved_indices, (width, height)


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def overlay_frame(frame_rgb: np.ndarray, boat_mask: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    canvas = frame_rgb.copy()
    if np.any(water_mask):
        canvas[water_mask] = (0.55 * canvas[water_mask] + 0.45 * np.array([80, 160, 255], dtype=np.uint8)).astype(np.uint8)
    if np.any(boat_mask):
        canvas[boat_mask] = (0.55 * canvas[boat_mask] + 0.45 * np.array([255, 80, 80], dtype=np.uint8)).astype(np.uint8)
    return canvas


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    frames_dir = output_dir / "frames"
    mask_dir = output_dir / "propagated_masks"
    ensure_dir(output_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    ensure_dir(frames_dir)
    ensure_dir(mask_dir)

    summary = json.loads((benchmark_dir / "summary.json").read_text(encoding="utf-8"))
    sample_times = [float(item) for item in summary["sample_times_sec"]]

    fps, saved_indices, frame_size = extract_frames(video_path, frames_dir, args.frame_step)
    width, height = frame_size

    os.chdir(Path(args.sam2_root).resolve())
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    predictor = build_sam2_video_predictor(args.sam2_config, str(Path(args.sam2_checkpoint).resolve()), device=device)
    inference_state = predictor.init_state(str(frames_dir))

    seed_entries = []
    for frame_summary in summary["frames"]:
        time_sec = float(frame_summary["sample_time_sec"])
        extracted_index = int(round(time_sec * fps / max(args.frame_step, 1)))
        extracted_index = int(np.clip(extracted_index, 0, max(len(saved_indices) - 1, 0)))
        frame_dir = Path(frame_summary["frame_dir"])
        boat_mask_path = frame_dir / args.boat_method / "boat_mask.png"
        water_mask_path = frame_dir / args.water_method / "water_mask.png"
        if not boat_mask_path.exists() or not water_mask_path.exists():
            continue
        boat_mask = load_mask(boat_mask_path)
        water_mask = load_mask(water_mask_path) & (~boat_mask)

        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=extracted_index,
            obj_id=1,
            mask=torch.from_numpy(boat_mask.astype(np.uint8)),
        )
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=extracted_index,
            obj_id=2,
            mask=torch.from_numpy(water_mask.astype(np.uint8)),
        )
        seed_entries.append(
            {
                "sample_time_sec": time_sec,
                "extracted_frame_idx": extracted_index,
                "boat_mask_path": str(boat_mask_path),
                "water_mask_path": str(water_mask_path),
            }
        )

    overlay_video_path = output_dir / "propagated_overlay.mp4"
    boat_video_path = output_dir / "boat_mask_propagated.mp4"
    water_video_path = output_dir / "water_mask_propagated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(overlay_video_path), fourcc, fps / max(args.frame_step, 1), (width, height))
    boat_writer = cv2.VideoWriter(str(boat_video_path), fourcc, fps / max(args.frame_step, 1), (width, height), False)
    water_writer = cv2.VideoWriter(str(water_video_path), fourcc, fps / max(args.frame_step, 1), (width, height), False)

    frame_results = []
    try:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask_map = {int(obj_id): (out_mask_logits[idx, 0] > 0.0).detach().cpu().numpy() for idx, obj_id in enumerate(out_obj_ids)}
            boat_mask = mask_map.get(1, np.zeros((height, width), dtype=bool))
            water_mask = mask_map.get(2, np.zeros((height, width), dtype=bool)) & (~boat_mask)
            frame_rgb = np.array(Image.open(frames_dir / f"{out_frame_idx:05d}.jpg").convert("RGB"))
            overlay = overlay_frame(frame_rgb, boat_mask, water_mask)
            overlay_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            boat_writer.write((boat_mask.astype(np.uint8) * 255))
            water_writer.write((water_mask.astype(np.uint8) * 255))

            frame_subdir = mask_dir / f"{out_frame_idx:05d}"
            ensure_dir(frame_subdir)
            Image.fromarray((boat_mask.astype(np.uint8) * 255)).save(frame_subdir / "boat_mask.png")
            Image.fromarray((water_mask.astype(np.uint8) * 255)).save(frame_subdir / "water_mask.png")
            Image.fromarray(overlay).save(frame_subdir / "overlay.jpg")
            frame_results.append(
                {
                    "frame_idx": int(out_frame_idx),
                    "source_frame_idx": int(saved_indices[out_frame_idx]),
                    "boat_area_frac": float(boat_mask.mean()),
                    "water_area_frac": float(water_mask.mean()),
                }
            )
    finally:
        overlay_writer.release()
        boat_writer.release()
        water_writer.release()

    propagation_summary = {
        "video_path": str(video_path),
        "benchmark_dir": str(benchmark_dir),
        "boat_method": args.boat_method,
        "water_method": args.water_method,
        "frame_step": args.frame_step,
        "fps_original": fps,
        "fps_output": fps / max(args.frame_step, 1),
        "frame_count_output": len(saved_indices),
        "seed_entries": seed_entries,
        "overlay_video_path": str(overlay_video_path),
        "boat_video_path": str(boat_video_path),
        "water_video_path": str(water_video_path),
        "frames_dir": str(frames_dir),
        "propagated_masks_dir": str(mask_dir),
        "frame_results": frame_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(propagation_summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
