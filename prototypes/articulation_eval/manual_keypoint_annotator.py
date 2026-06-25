"""Click-based keypoint annotator for ArticulationEval examples.

The annotator saves manual 2D keypoints for selected video frames. These are
the observed points from the generated video; later evaluators can compare them
against hinge/slider motion constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_KEYPOINTS = [
    "left_pivot",
    "right_pivot",
    "lid_back_center",
    "lid_midpoint",
    "lid_front_tip",
]


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def draw_annotation(frame, points, keypoints, current_idx, frame_idx):
    canvas = frame.copy()
    for name, point in points.items():
        if point is None:
            continue
        x, y = point
        cv2.circle(canvas, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if current_idx < len(keypoints):
        current = keypoints[current_idx]
        status = f"frame {frame_idx} | click: {current} | u undo | k skip | Enter next | q quit"
    else:
        status = f"frame {frame_idx} | all points marked | Enter next | u undo | q quit"

    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (8, 8), (min(w - 8, 980), 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        status,
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def annotate_frame(video_path: Path, frame_idx: int, keypoints: list[str]):
    frame = read_frame(video_path, frame_idx)
    points = {name: None for name in keypoints}
    current_idx = 0
    window_name = f"annotate frame {frame_idx}"

    def redraw():
        cv2.imshow(window_name, draw_annotation(frame, points, keypoints, current_idx, frame_idx))

    def mouse_cb(event, x, y, flags, param):
        nonlocal current_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if current_idx >= len(keypoints):
            return
        points[keypoints[current_idx]] = [int(x), int(y)]
        current_idx += 1
        redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_cb)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key in [ord("q"), 27]:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Annotation cancelled")
        if key in [13, ord("n")]:
            break
        if key == ord("u"):
            if current_idx > 0:
                current_idx -= 1
                points[keypoints[current_idx]] = None
            redraw()
        if key == ord("k"):
            if current_idx < len(keypoints):
                points[keypoints[current_idx]] = None
                current_idx += 1
            redraw()

    cv2.destroyWindow(window_name)
    return points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video to annotate")
    parser.add_argument("--out_json", required=True, help="Output keypoint JSON")
    parser.add_argument("--case_name", default="case_toilet1_0")
    parser.add_argument("--motion_type", default="hinge_rotation")
    parser.add_argument("--frames", default="0,46,60,80", help="Comma-separated frame indices")
    parser.add_argument(
        "--keypoints",
        default=",".join(DEFAULT_KEYPOINTS),
        help="Comma-separated keypoint names, clicked in this order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    out_path = Path(args.out_json)
    frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    keypoints = [x.strip() for x in args.keypoints.split(",") if x.strip()]

    annotations = []
    for frame_idx in frames:
        print(f"Annotating frame {frame_idx}")
        points = annotate_frame(video_path, frame_idx, keypoints)
        annotations.append({"frame_idx": frame_idx, "points": points})

    out = {
        "case_name": args.case_name,
        "motion_type": args.motion_type,
        "video": str(video_path),
        "keypoint_order": keypoints,
        "frames": annotations,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"Saved keypoints: {out_path}")


if __name__ == "__main__":
    main()
