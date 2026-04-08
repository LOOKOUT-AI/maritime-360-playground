from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click a known-distance point and save its image coordinates.")
    parser.add_argument("--image", required=True, type=str, help="Path to the reference image.")
    parser.add_argument("--output", required=True, type=str, help="Where to save the picked point JSON.")
    parser.add_argument("--label", default="known_distance_point", type=str, help="Label stored in the JSON output.")
    parser.add_argument(
        "--max_display_width",
        default=1600,
        type=int,
        help="Resize the image for display if it is wider than this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image_bgr.shape[:2]
    display_scale = min(1.0, float(args.max_display_width) / float(width))
    if display_scale < 1.0:
        display_bgr = cv2.resize(
            image_bgr,
            (int(round(width * display_scale)), int(round(height * display_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display_bgr = image_bgr.copy()

    selected_point: dict[str, int] = {}
    window_name = "Pick Known-Distance Point"

    def redraw() -> None:
        canvas = display_bgr.copy()
        instructions = [
            "Left click: select point",
            "Enter or S: save",
            "U: clear selection",
            "Esc or Q: quit",
        ]
        for idx, text in enumerate(instructions):
            y = 32 + idx * 28
            cv2.putText(canvas, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if selected_point:
            x_disp = int(round(selected_point["x"] * display_scale))
            y_disp = int(round(selected_point["y"] * display_scale))
            cv2.drawMarker(
                canvas,
                (x_disp, y_disp),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=2,
            )
            label = f"{args.label}: ({selected_point['x']}, {selected_point['y']})"
            cv2.putText(canvas, label, (18, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(
                canvas,
                label,
                (18, canvas.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        selected_point["x"] = int(round(x / display_scale))
        selected_point["y"] = int(round(y / display_scale))
        redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("u"):
            selected_point.clear()
            redraw()
            continue
        if key in (13, ord("s")) and selected_point:
            payload = {
                "image_path": str(image_path),
                "label": args.label,
                "x": int(selected_point["x"]),
                "y": int(selected_point["y"]),
                "image_width": int(width),
                "image_height": int(height),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(output_path)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
