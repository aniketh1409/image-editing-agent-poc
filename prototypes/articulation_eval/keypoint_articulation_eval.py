"""Evaluate side-view articulated-object motion from semantic 2D keypoints.

This is ArticulationEval v1. It intentionally starts with manual keypoints
instead of automatic detection so the metric definitions are easy to inspect.

Input JSON is produced by manual_keypoint_annotator.py. For the toilet-lid
side-view case, expected keypoints are:

- back_pivot
- lid_front_tip
- lid_midpoint
- lid_back_edge
- lid_front_edge_mid

The evaluator compares observed keypoints against simple hinge-motion
constraints:

- visibility: how many semantic points are actually labelable
- pivot stability: the hinge-ish origin should stay mostly fixed
- opening motion: the lid front tip should move relative to frame 0
- radius consistency: lid points should keep similar distance to the pivot
- rigidity: distances between lid keypoints should stay similar over time
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable


PIVOT_KEY = "back_pivot"
LEFT_PIVOT_KEY = "left_pivot"
RIGHT_PIVOT_KEY = "right_pivot"
FRONT_TIP_KEY = "lid_front_tip"
LID_KEYS = [
    "lid_front_tip",
    "lid_midpoint",
    "lid_back_center",
]


def point_or_none(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        return None
    x, y = value
    if x is None or y is None:
        return None
    return float(x), float(y)


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_deg(origin, point) -> float:
    return math.degrees(math.atan2(point[1] - origin[1], point[0] - origin[0]))


def wrap_angle_delta_deg(a: float, b: float) -> float:
    """Return signed smallest difference b - a in degrees."""
    return (b - a + 180.0) % 360.0 - 180.0


def safe_ratio(error: float, scale: float) -> float:
    if scale <= 1e-6:
        return math.nan
    return error / scale


def finite_values(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def mean_or_nan(values: Iterable[float]) -> float:
    vals = finite_values(values)
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def max_or_nan(values: Iterable[float]) -> float:
    vals = finite_values(values)
    if not vals:
        return math.nan
    return max(vals)


def load_keypoints(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_frames(data: dict) -> list[dict]:
    frames = data.get("frames", [])
    normalized = []
    for frame in frames:
        points = {
            name: point_or_none(value)
            for name, value in frame.get("points", {}).items()
        }
        if points.get(PIVOT_KEY) is None:
            left = points.get(LEFT_PIVOT_KEY)
            right = points.get(RIGHT_PIVOT_KEY)
            if left is not None and right is not None:
                points[PIVOT_KEY] = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
        normalized.append({
            "frame_idx": int(frame["frame_idx"]),
            "points": points,
        })
    normalized.sort(key=lambda item: item["frame_idx"])
    return normalized


def first_visible_baseline(frames: list[dict]) -> dict:
    """Find baseline point positions from the earliest frame where each appears."""
    baseline = {}
    for frame in frames:
        for name, point in frame["points"].items():
            if point is not None and name not in baseline:
                baseline[name] = point
    return baseline


def baseline_pair_distances(baseline: dict, keys: list[str]) -> dict[tuple[str, str], float]:
    out = {}
    for a, b in combinations(keys, 2):
        if a in baseline and b in baseline:
            out[(a, b)] = distance(baseline[a], baseline[b])
    return out


def evaluate_frames(data: dict) -> tuple[list[dict], dict]:
    frames = normalize_frames(data)
    if not frames:
        raise RuntimeError("No frames found in keypoint JSON")

    keypoint_order = data.get("keypoint_order") or sorted(frames[0]["points"].keys())
    baseline = first_visible_baseline(frames)
    if PIVOT_KEY not in baseline:
        raise RuntimeError(f"Need at least one visible {PIVOT_KEY} point")

    pivot0 = baseline[PIVOT_KEY]
    tip0 = baseline.get(FRONT_TIP_KEY)
    pivot_radius0 = {
        name: distance(pivot0, point)
        for name, point in baseline.items()
        if name in LID_KEYS
    }
    pair_dist0 = baseline_pair_distances(baseline, LID_KEYS)

    rows = []
    prev_tip_angle = None
    for frame in frames:
        frame_idx = frame["frame_idx"]
        points = frame["points"]
        visible_count = sum(1 for name in keypoint_order if points.get(name) is not None)
        visibility_ratio = safe_ratio(visible_count, len(keypoint_order))

        pivot = points.get(PIVOT_KEY)
        tip = points.get(FRONT_TIP_KEY)

        pivot_drift_px = distance(pivot, pivot0) if pivot is not None else math.nan
        tip_motion_px = distance(tip, tip0) if tip is not None and tip0 is not None else math.nan

        radius_errors = []
        radius_error_norms = []
        if pivot is not None:
            for name in LID_KEYS:
                point = points.get(name)
                base_radius = pivot_radius0.get(name)
                if point is None or base_radius is None:
                    continue
                err = abs(distance(pivot, point) - base_radius)
                radius_errors.append(err)
                radius_error_norms.append(safe_ratio(err, base_radius))

        rigidity_errors = []
        rigidity_error_norms = []
        for (a, b), base_dist in pair_dist0.items():
            pa = points.get(a)
            pb = points.get(b)
            if pa is None or pb is None:
                continue
            err = abs(distance(pa, pb) - base_dist)
            rigidity_errors.append(err)
            rigidity_error_norms.append(safe_ratio(err, base_dist))

        tip_angle_deg = math.nan
        tip_angle_delta_from_start_deg = math.nan
        tip_angle_delta_prev_deg = math.nan
        if pivot is not None and tip is not None:
            tip_angle_deg = angle_deg(pivot, tip)
            if PIVOT_KEY in baseline and FRONT_TIP_KEY in baseline:
                start_angle = angle_deg(pivot0, baseline[FRONT_TIP_KEY])
                tip_angle_delta_from_start_deg = wrap_angle_delta_deg(start_angle, tip_angle_deg)
            if prev_tip_angle is not None:
                tip_angle_delta_prev_deg = wrap_angle_delta_deg(prev_tip_angle, tip_angle_deg)
            prev_tip_angle = tip_angle_deg

        rows.append({
            "frame_idx": frame_idx,
            "visible_keypoints": visible_count,
            "total_keypoints": len(keypoint_order),
            "visibility_ratio": visibility_ratio,
            "pivot_visible": pivot is not None,
            "tip_visible": tip is not None,
            "pivot_drift_px": pivot_drift_px,
            "tip_motion_px": tip_motion_px,
            "radius_error_mean_px": mean_or_nan(radius_errors),
            "radius_error_max_px": max_or_nan(radius_errors),
            "radius_error_mean_norm": mean_or_nan(radius_error_norms),
            "radius_error_max_norm": max_or_nan(radius_error_norms),
            "rigidity_error_mean_px": mean_or_nan(rigidity_errors),
            "rigidity_error_max_px": max_or_nan(rigidity_errors),
            "rigidity_error_mean_norm": mean_or_nan(rigidity_error_norms),
            "rigidity_error_max_norm": max_or_nan(rigidity_error_norms),
            "tip_angle_deg": tip_angle_deg,
            "tip_angle_delta_from_start_deg": tip_angle_delta_from_start_deg,
            "tip_angle_delta_prev_deg": tip_angle_delta_prev_deg,
        })

    summary = {
        "case_name": data.get("case_name", ""),
        "motion_type": data.get("motion_type", ""),
        "num_frames": len(rows),
        "mean_visibility_ratio": mean_or_nan(r["visibility_ratio"] for r in rows),
        "max_pivot_drift_px": max_or_nan(r["pivot_drift_px"] for r in rows),
        "max_tip_motion_px": max_or_nan(r["tip_motion_px"] for r in rows),
        "mean_radius_error_norm": mean_or_nan(r["radius_error_mean_norm"] for r in rows),
        "max_radius_error_norm": max_or_nan(r["radius_error_max_norm"] for r in rows),
        "mean_rigidity_error_norm": mean_or_nan(r["rigidity_error_mean_norm"] for r in rows),
        "max_rigidity_error_norm": max_or_nan(r["rigidity_error_max_norm"] for r in rows),
        "max_abs_tip_angle_delta_from_start_deg": max_or_nan(
            abs(r["tip_angle_delta_from_start_deg"]) for r in rows
        ),
        "mean_abs_tip_angle_delta_prev_deg": mean_or_nan(
            abs(r["tip_angle_delta_prev_deg"]) for r in rows
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def color_for_keypoint(name: str) -> tuple[int, int, int]:
    colors = {
        "back_pivot": (80, 80, 255),
        "lid_front_tip": (60, 220, 60),
        "lid_midpoint": (255, 180, 40),
        "lid_back_edge": (220, 80, 220),
        "lid_front_edge_mid": (40, 220, 255),
    }
    return colors.get(name, (255, 255, 255))


def read_video_frames(video_path: Path) -> tuple[list, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
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
        raise RuntimeError(f"No frames read from: {video_path}")
    return frames, fps


def draw_text_box(canvas, lines: list[str], x: int = 10, y: int = 10) -> None:
    import cv2

    line_h = 22
    width = min(canvas.shape[1] - 20, 760)
    height = 14 + line_h * len(lines)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x + 10, y + 24 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def make_video_writer(path: Path, fps: float, size: tuple[int, int]):
    import cv2

    # Prefer mp4v here because Windows OpenCV often has a broken OpenH264 DLL
    # in the path. H.264 can be produced later with ffmpeg if needed.
    for fourcc_name in ["mp4v", "avc1", "H264"]:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not create video writer: {path}")


def write_visual_report(
    data: dict,
    rows: list[dict],
    video_path: Path,
    viz_dir: Path | None = None,
    out_video: Path | None = None,
) -> None:
    """Draw keypoints and hinge-style diagnostics on selected video frames."""
    import cv2

    frames_bgr, fps = read_video_frames(video_path)
    key_frames = normalize_frames(data)
    baseline = first_visible_baseline(key_frames)
    pivot0 = baseline.get(PIVOT_KEY)
    tip0 = baseline.get(FRONT_TIP_KEY)
    tip_radius0 = distance(pivot0, tip0) if pivot0 is not None and tip0 is not None else None
    rows_by_frame = {int(row["frame_idx"]): row for row in rows}

    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames_bgr[0].shape[:2]
        writer = make_video_writer(out_video, fps, (w, h))

    tip_trail = []
    for item in key_frames:
        frame_idx = item["frame_idx"]
        if frame_idx < 0 or frame_idx >= len(frames_bgr):
            continue
        canvas = frames_bgr[frame_idx].copy()
        points = item["points"]
        row = rows_by_frame.get(frame_idx, {})
        pivot = points.get(PIVOT_KEY)
        tip = points.get(FRONT_TIP_KEY)

        if tip is not None:
            tip_trail.append((int(round(tip[0])), int(round(tip[1]))))

        if pivot is not None and tip_radius0 is not None and math.isfinite(tip_radius0):
            cv2.circle(
                canvas,
                (int(round(pivot[0])), int(round(pivot[1]))),
                int(round(tip_radius0)),
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        if pivot is not None:
            pivot_xy = (int(round(pivot[0])), int(round(pivot[1])))
            for name in LID_KEYS:
                point = points.get(name)
                if point is None:
                    continue
                xy = (int(round(point[0])), int(round(point[1])))
                cv2.line(canvas, pivot_xy, xy, (120, 120, 120), 1, cv2.LINE_AA)

        for i in range(1, len(tip_trail)):
            cv2.line(canvas, tip_trail[i - 1], tip_trail[i], (60, 220, 60), 2, cv2.LINE_AA)

        for name, point in points.items():
            if point is None:
                continue
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

        lines = [
            f"ArticulationEval | frame {frame_idx}",
            f"visibility={row.get('visibility_ratio', math.nan):.2f}  tip_motion={row.get('tip_motion_px', math.nan):.1f}px  pivot_drift={row.get('pivot_drift_px', math.nan):.1f}px",
            f"radius_err={row.get('radius_error_mean_norm', math.nan):.3f}  rigidity_err={row.get('rigidity_error_mean_norm', math.nan):.3f}  angle_delta={row.get('tip_angle_delta_from_start_deg', math.nan):.1f}deg",
        ]
        draw_text_box(canvas, lines)

        if viz_dir is not None:
            cv2.imwrite(str(viz_dir / f"frame_{frame_idx:04d}_articulation_eval.png"), canvas)
        if writer is not None:
            writer.write(canvas)

    if writer is not None:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_json", required=True, help="Manual keypoint JSON")
    parser.add_argument("--out_csv", required=True, help="Frame-level metrics CSV")
    parser.add_argument("--summary_json", required=True, help="Summary metrics JSON")
    parser.add_argument("--video", default=None, help="Optional video path for annotated visual report")
    parser.add_argument("--viz_dir", default=None, help="Optional directory for annotated PNG frames")
    parser.add_argument("--out_video", default=None, help="Optional annotated MP4 path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_keypoints(Path(args.keypoints_json))
    rows, summary = evaluate_frames(data)
    write_csv(Path(args.out_csv), rows)
    write_json(Path(args.summary_json), summary)

    video_arg = args.video or data.get("video")
    visual_report_error = None
    if args.viz_dir or args.out_video:
        if not video_arg:
            raise RuntimeError("--video is required when writing visual outputs")
        try:
            write_visual_report(
                data,
                rows,
                Path(video_arg),
                viz_dir=Path(args.viz_dir) if args.viz_dir else None,
                out_video=Path(args.out_video) if args.out_video else None,
            )
        except ModuleNotFoundError as exc:
            visual_report_error = (
                f"Could not write visual report because {exc.name} is not installed. "
                "Install opencv-python and rerun with --viz_dir/--out_video."
            )

    print("=== ArticulationEval summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"Saved frame metrics: {args.out_csv}")
    print(f"Saved summary: {args.summary_json}")
    if visual_report_error:
        print(f"Visual report skipped: {visual_report_error}")
    elif args.viz_dir:
        print(f"Saved annotated frames: {args.viz_dir}")
    if not visual_report_error and args.out_video:
        print(f"Saved annotated video: {args.out_video}")


if __name__ == "__main__":
    main()
