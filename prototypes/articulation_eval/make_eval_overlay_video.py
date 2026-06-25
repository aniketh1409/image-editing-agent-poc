"""Create a full-video ArticulationEval overlay demo.

This turns sparse manual keypoints into a presentation-friendly video by
linearly interpolating them across frames, then drawing the hinge/keypoint
diagnostics on top of the generated video. It is a visual demo, not a new
tracking algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

from keypoint_articulation_eval import (
    FRONT_TIP_KEY,
    LID_KEYS,
    PIVOT_KEY,
    color_for_keypoint,
    distance,
    evaluate_frames,
    first_visible_baseline,
    normalize_frames,
)


def read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")
    return frames, fps


def make_writer(path: Path, fps: float, size: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_name in ["mp4v", "avc1", "H264"]:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not create video writer: {path}")


def load_keypoint_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def interpolate_point(frame_idx: int, annotated: list[dict], key: str):
    visible = [
        (item["frame_idx"], item["points"].get(key))
        for item in annotated
        if item["points"].get(key) is not None
    ]
    if not visible:
        return None
    if frame_idx <= visible[0][0]:
        return visible[0][1]
    if frame_idx >= visible[-1][0]:
        return visible[-1][1]

    for (f0, p0), (f1, p1) in zip(visible, visible[1:]):
        if f0 <= frame_idx <= f1:
            if f1 == f0:
                return p0
            alpha = (frame_idx - f0) / float(f1 - f0)
            return (
                p0[0] * (1.0 - alpha) + p1[0] * alpha,
                p0[1] * (1.0 - alpha) + p1[1] * alpha,
            )
    return None


def nearest_metric_row(frame_idx: int, rows: list[dict]):
    return min(rows, key=lambda row: abs(int(row["frame_idx"]) - frame_idx))


def draw_panel(canvas, lines: list[str], status: str):
    color = (25, 135, 84) if status == "PASS-ish" else (0, 140, 255)
    cv2.rectangle(canvas, (12, 12), (850, 120), (0, 0, 0), -1)
    cv2.rectangle(canvas, (12, 12), (850, 120), color, 2)
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (26, 40 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        status,
        (700, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_keypoint(canvas, name: str, point):
    if point is None:
        return
    xy = (int(round(point[0])), int(round(point[1])))
    color = color_for_keypoint(name)
    cv2.circle(canvas, xy, 6, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, xy, 10, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        name,
        (xy[0] + 8, xy[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--keypoints_json", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--slowdown", type=int, default=1, help="Repeat each frame N times")
    args = parser.parse_args()

    data = load_keypoint_data(Path(args.keypoints_json))
    annotated = normalize_frames(data)
    rows, summary = evaluate_frames(data)
    baseline = first_visible_baseline(annotated)
    pivot0 = baseline.get(PIVOT_KEY)
    tip0 = baseline.get(FRONT_TIP_KEY)
    tip_radius0 = distance(pivot0, tip0) if pivot0 and tip0 else None

    frames, fps = read_video(Path(args.video))
    height, width = frames[0].shape[:2]
    writer = make_writer(Path(args.out_video), fps / max(args.slowdown, 1), (width, height))

    tip_trail = []
    key_order = data.get("keypoint_order", [])
    if PIVOT_KEY not in key_order:
        key_order = [PIVOT_KEY] + key_order

    annotated_frame_set = {item["frame_idx"] for item in annotated}

    for frame_idx, frame in enumerate(frames):
        canvas = frame.copy()
        points = {
            key: interpolate_point(frame_idx, annotated, key)
            for key in key_order
        }
        pivot = points.get(PIVOT_KEY)
        tip = points.get(FRONT_TIP_KEY)
        row = nearest_metric_row(frame_idx, rows)

        if tip is not None:
            tip_trail.append((int(round(tip[0])), int(round(tip[1]))))
        if len(tip_trail) > 80:
            tip_trail = tip_trail[-80:]

        if pivot is not None and tip_radius0 is not None:
            cv2.circle(
                canvas,
                (int(round(pivot[0])), int(round(pivot[1]))),
                int(round(tip_radius0)),
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )

        if pivot is not None:
            pivot_xy = (int(round(pivot[0])), int(round(pivot[1])))
            for key in LID_KEYS:
                point = points.get(key)
                if point is None:
                    continue
                xy = (int(round(point[0])), int(round(point[1])))
                cv2.line(canvas, pivot_xy, xy, (130, 130, 130), 1, cv2.LINE_AA)

        for i in range(1, len(tip_trail)):
            cv2.line(canvas, tip_trail[i - 1], tip_trail[i], (60, 220, 60), 2, cv2.LINE_AA)

        for key in key_order:
            draw_keypoint(canvas, key, points.get(key))

        if frame_idx in annotated_frame_set:
            cv2.putText(
                canvas,
                "manual keyframe",
                (26, height - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        status = "REVIEW"
        if (
            math.isfinite(summary["mean_rigidity_error_norm"])
            and summary["mean_rigidity_error_norm"] < 0.25
            and math.isfinite(summary["mean_radius_error_norm"])
            and summary["mean_radius_error_norm"] < 0.25
        ):
            status = "PASS-ish"

        lines = [
            f"ArticulationEval overlay | frame {frame_idx}",
            f"tip motion: {float(row['tip_motion_px']):.1f}px   pivot drift: {float(row['pivot_drift_px']) if str(row['pivot_drift_px']) != 'nan' else float('nan'):.1f}px",
            f"radius error: {float(row['radius_error_mean_norm']) if str(row['radius_error_mean_norm']) != 'nan' else float('nan'):.2f}   rigidity error: {float(row['rigidity_error_mean_norm']):.2f}",
            "Goal: verify whether the generated lid behaves like a rigid hinged part",
        ]
        draw_panel(canvas, lines, status)

        for _ in range(max(args.slowdown, 1)):
            writer.write(canvas)

    writer.release()
    print(f"Saved overlay demo video: {args.out_video}")


if __name__ == "__main__":
    main()
