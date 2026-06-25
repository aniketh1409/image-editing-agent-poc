#!/usr/bin/env python3
"""
Synthetic articulation sandbox.

This script generates controlled 2D semantic-keypoint trajectories for simple
articulated motions, then scores them with a shared "model-fit residual" idea.

Why this exists:
- Real generated videos mix together many failure sources: occlusion, blur,
  annotation noise, bad tracking, and actual articulation failure.
- Synthetic keypoints let us test the scoring math when the correct behavior is
  known. Perfect cases should score high; injected failure cases should score
  lower.

The score is intentionally simple:
    total_error = model_fit_error + rigidity_error + anchor_drift
    articulation_score = 100 * exp(-4 * total_error)

The motion-specific part is only the model-fit error:
- hinge/rotation: moving points should preserve radius around a pivot/center.
- slider: moving points should translate together along one dominant line.

The final residual/score format is shared across motion types.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Point = Tuple[float, float]
Frame = Dict[str, Point]


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def distance(a: Point, b: Point) -> float:
    return norm(sub(a, b))


def rotate_around(p: Point, center: Point, angle_rad: float) -> Point:
    x, y = sub(p, center)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (center[0] + c * x - s * y, center[1] + s * x + c * y)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def object_scale(frames: List[Frame], moving_keys: List[str]) -> float:
    first = frames[0]
    dists = []
    for i, key_a in enumerate(moving_keys):
        for key_b in moving_keys[i + 1 :]:
            dists.append(distance(first[key_a], first[key_b]))
    return max(max(dists, default=1.0), 1e-6)


def pairwise_rigidity_error(frames: List[Frame], moving_keys: List[str], scale: float) -> float:
    first = frames[0]
    errors = []
    for i, key_a in enumerate(moving_keys):
        for key_b in moving_keys[i + 1 :]:
            d0 = distance(first[key_a], first[key_b])
            for frame in frames[1:]:
                dt = distance(frame[key_a], frame[key_b])
                errors.append(abs(dt - d0) / scale)
    return mean(errors)


def anchor_drift_error(frames: List[Frame], anchor_key: str | None, scale: float) -> float:
    if anchor_key is None:
        return 0.0
    anchor0 = frames[0][anchor_key]
    return mean(distance(frame[anchor_key], anchor0) / scale for frame in frames[1:])


def radius_model_error(
    frames: List[Frame],
    center_key: str,
    moving_keys: List[str],
    scale: float,
) -> float:
    """For hinge/rotation: moving points should keep radius around center."""
    first = frames[0]
    radii0 = {key: distance(first[key], first[center_key]) for key in moving_keys}
    errors = []
    for frame in frames[1:]:
        center = frame[center_key]
        for key in moving_keys:
            rt = distance(frame[key], center)
            errors.append(abs(rt - radii0[key]) / scale)
    return mean(errors)


def slider_model_error(frames: List[Frame], moving_keys: List[str], scale: float) -> float:
    """For sliders: all points should share one translation per frame."""
    first = frames[0]
    last = frames[-1]
    final_displacements = [sub(last[key], first[key]) for key in moving_keys]
    final_avg = (
        mean(d[0] for d in final_displacements),
        mean(d[1] for d in final_displacements),
    )
    final_len = norm(final_avg)
    if final_len < 1e-6:
        return 0.0
    global_axis = (final_avg[0] / final_len, final_avg[1] / final_len)

    errors = []
    for frame in frames[1:]:
        displacements = [sub(frame[key], first[key]) for key in moving_keys]
        avg_disp = (
            mean(d[0] for d in displacements),
            mean(d[1] for d in displacements),
        )
        avg_parallel = mul(global_axis, dot(avg_disp, global_axis))
        path_off_axis_error = norm(sub(avg_disp, avg_parallel)) / scale

        for disp in displacements:
            shared_translation_error = norm(sub(disp, avg_disp)) / scale
            parallel = mul(global_axis, dot(disp, global_axis))
            off_axis_error = norm(sub(disp, parallel)) / scale
            errors.append(shared_translation_error + off_axis_error + path_off_axis_error)
    return mean(errors)


def score_from_error(total_error: float) -> float:
    return 100.0 * math.exp(-4.0 * total_error)


def summarize_case(
    case_name: str,
    motion_type: str,
    frames: List[Frame],
    moving_keys: List[str],
    model_error: float,
    anchor_key: str | None = None,
) -> Dict[str, float | str | int]:
    scale = object_scale(frames, moving_keys)
    rigidity_error = pairwise_rigidity_error(frames, moving_keys, scale)
    drift_error = anchor_drift_error(frames, anchor_key, scale)
    total_error = model_error + rigidity_error + drift_error
    return {
        "case_name": case_name,
        "motion_type": motion_type,
        "num_frames": len(frames),
        "object_scale": scale,
        "model_fit_error": model_error,
        "rigidity_error": rigidity_error,
        "anchor_drift_error": drift_error,
        "total_error": total_error,
        "articulation_score": score_from_error(total_error),
    }


def make_hinge_case(case_name: str, failure: str | None = None, num_frames: int = 13) -> List[Frame]:
    pivot0 = (160.0, 220.0)
    local_points = {
        "pivot": (0.0, 0.0),
        "part_mid": (70.0, 0.0),
        "part_tip": (140.0, 0.0),
        "part_upper_edge": (110.0, -28.0),
    }
    frames = []
    for t in range(num_frames):
        phase = t / (num_frames - 1)
        angle = math.radians(phase * 95.0)
        pivot = pivot0
        stretch = 1.0
        bend = 0.0

        if failure == "pivot_drift":
            pivot = add(pivot0, (phase * 24.0, math.sin(phase * math.pi) * 10.0))
        elif failure == "deformation":
            stretch = 1.0 + 0.24 * phase
            bend = 24.0 * math.sin(phase * math.pi)

        frame: Frame = {"pivot": pivot}
        for key, local in local_points.items():
            if key == "pivot":
                continue
            local_deformed = (local[0] * stretch, local[1] + bend * (local[0] / 140.0))
            point0 = add(pivot, local_deformed)
            frame[key] = rotate_around(point0, pivot, angle)
        frames.append(frame)
    return frames


def make_slider_case(case_name: str, failure: str | None = None, num_frames: int = 13) -> List[Frame]:
    base = {
        "front_left": (140.0, 190.0),
        "front_right": (260.0, 190.0),
        "back_left": (140.0, 250.0),
        "back_right": (260.0, 250.0),
    }
    frames = []
    for t in range(num_frames):
        phase = t / (num_frames - 1)
        translation = (phase * 130.0, 0.0)
        wobble = 0.0
        shear = 0.0
        if failure == "off_axis_wobble":
            wobble = 24.0 * math.sin(phase * math.pi * 2.0)
        elif failure == "deformation":
            shear = 34.0 * phase

        frame: Frame = {}
        for key, point in base.items():
            extra_y = wobble
            extra_x = shear if "right" in key else 0.0
            frame[key] = add(point, (translation[0] + extra_x, translation[1] + extra_y))
        frames.append(frame)
    return frames


def make_rotation_case(case_name: str, failure: str | None = None, num_frames: int = 13) -> List[Frame]:
    center0 = (210.0, 210.0)
    local_points = {
        "center": (0.0, 0.0),
        "mark_a": (60.0, 0.0),
        "mark_b": (0.0, 60.0),
        "mark_c": (-45.0, -35.0),
    }
    frames = []
    for t in range(num_frames):
        phase = t / (num_frames - 1)
        angle = math.radians(phase * 150.0)
        center = center0
        stretch = 1.0
        if failure == "center_drift":
            center = add(center0, (phase * 22.0, -phase * 16.0))
        elif failure == "deformation":
            stretch = 1.0 + 0.30 * phase

        frame: Frame = {"center": center}
        for key, local in local_points.items():
            if key == "center":
                continue
            point0 = add(center, (local[0] * stretch, local[1]))
            frame[key] = rotate_around(point0, center, angle)
        frames.append(frame)
    return frames


def build_cases() -> Dict[str, Dict[str, object]]:
    return {
        "perfect_hinge": {
            "motion_type": "hinge",
            "frames": make_hinge_case("perfect_hinge"),
            "moving_keys": ["part_mid", "part_tip", "part_upper_edge"],
            "anchor_key": "pivot",
        },
        "hinge_pivot_drift": {
            "motion_type": "hinge",
            "frames": make_hinge_case("hinge_pivot_drift", failure="pivot_drift"),
            "moving_keys": ["part_mid", "part_tip", "part_upper_edge"],
            "anchor_key": "pivot",
        },
        "hinge_deformation": {
            "motion_type": "hinge",
            "frames": make_hinge_case("hinge_deformation", failure="deformation"),
            "moving_keys": ["part_mid", "part_tip", "part_upper_edge"],
            "anchor_key": "pivot",
        },
        "perfect_slider": {
            "motion_type": "slider",
            "frames": make_slider_case("perfect_slider"),
            "moving_keys": ["front_left", "front_right", "back_left", "back_right"],
            "anchor_key": None,
        },
        "slider_off_axis_wobble": {
            "motion_type": "slider",
            "frames": make_slider_case("slider_off_axis_wobble", failure="off_axis_wobble"),
            "moving_keys": ["front_left", "front_right", "back_left", "back_right"],
            "anchor_key": None,
        },
        "slider_deformation": {
            "motion_type": "slider",
            "frames": make_slider_case("slider_deformation", failure="deformation"),
            "moving_keys": ["front_left", "front_right", "back_left", "back_right"],
            "anchor_key": None,
        },
        "perfect_rotation": {
            "motion_type": "rotation",
            "frames": make_rotation_case("perfect_rotation"),
            "moving_keys": ["mark_a", "mark_b", "mark_c"],
            "anchor_key": "center",
        },
        "rotation_center_drift": {
            "motion_type": "rotation",
            "frames": make_rotation_case("rotation_center_drift", failure="center_drift"),
            "moving_keys": ["mark_a", "mark_b", "mark_c"],
            "anchor_key": "center",
        },
        "rotation_deformation": {
            "motion_type": "rotation",
            "frames": make_rotation_case("rotation_deformation", failure="deformation"),
            "moving_keys": ["mark_a", "mark_b", "mark_c"],
            "anchor_key": "center",
        },
    }


def score_cases(cases: Dict[str, Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    serializable_cases: Dict[str, object] = {}

    for case_name, case in cases.items():
        motion_type = str(case["motion_type"])
        frames = case["frames"]  # type: ignore[assignment]
        moving_keys = case["moving_keys"]  # type: ignore[assignment]
        anchor_key = case["anchor_key"]  # type: ignore[assignment]
        scale = object_scale(frames, moving_keys)

        if motion_type in {"hinge", "rotation"}:
            center_key = "pivot" if motion_type == "hinge" else "center"
            model_error = radius_model_error(frames, center_key, moving_keys, scale)
        elif motion_type == "slider":
            model_error = slider_model_error(frames, moving_keys, scale)
        else:
            raise ValueError(f"Unsupported motion type: {motion_type}")

        rows.append(
            summarize_case(
                case_name=case_name,
                motion_type=motion_type,
                frames=frames,
                moving_keys=moving_keys,
                model_error=model_error,
                anchor_key=anchor_key,
            )
        )

        serializable_cases[case_name] = {
            "motion_type": motion_type,
            "moving_keys": moving_keys,
            "anchor_key": anchor_key,
            "frames": [
                {
                    "frame": idx,
                    "keypoints": {key: [round(x, 4), round(y, 4)] for key, (x, y) in frame.items()},
                }
                for idx, frame in enumerate(frames)
            ],
        }

    return rows, serializable_cases


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = [
        "case_name",
        "motion_type",
        "num_frames",
        "object_scale",
        "model_fit_error",
        "rigidity_error",
        "anchor_drift_error",
        "total_error",
        "articulation_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def svg_polyline(points: List[Point], color: str) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" />'


def write_svg(path: Path, cases: Dict[str, object]) -> None:
    colors = ["#d1495b", "#00798c", "#edae49", "#30638e", "#4c956c", "#8d5a97"]
    width = 1100
    height = 760
    cell_w = 360
    cell_h = 250
    margin_x = 35
    margin_y = 45
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8" />',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:17px;font-weight:700}.label{font-size:12px;fill:#555}</style>',
    ]
    for idx, (case_name, case_obj) in enumerate(cases.items()):
        case = case_obj  # type: ignore[assignment]
        col = idx % 3
        row = idx // 3
        ox = margin_x + col * cell_w
        oy = margin_y + row * cell_h
        chunks.append(f'<text class="title" x="{ox}" y="{oy - 15}">{case_name}</text>')
        chunks.append(f'<rect x="{ox}" y="{oy}" width="300" height="190" fill="#fff" stroke="#ddd" />')

        frames = case["frames"]  # type: ignore[index]
        keypoints = frames[0]["keypoints"].keys()  # type: ignore[index]
        for key_idx, key in enumerate(keypoints):
            points = []
            for frame in frames:
                x, y = frame["keypoints"][key]  # type: ignore[index]
                points.append((ox + (x - 80) * 0.9, oy + (y - 120) * 0.9))
            chunks.append(svg_polyline(points, colors[key_idx % len(colors)]))
            sx, sy = points[-1]
            chunks.append(
                f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{colors[key_idx % len(colors)]}" />'
            )
        chunks.append(f'<text class="label" x="{ox}" y="{oy + 213}">trajectories over synthetic frames</text>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks), encoding="utf-8")


def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n=== Synthetic Articulation Sandbox ===")
    print("Lower errors are better. Higher articulation_score is better.\n")
    header = f"{'case':26s} {'motion':9s} {'model':>8s} {'rigid':>8s} {'anchor':>8s} {'total':>8s} {'score':>8s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['case_name']):26s} "
            f"{str(row['motion_type']):9s} "
            f"{float(row['model_fit_error']):8.3f} "
            f"{float(row['rigidity_error']):8.3f} "
            f"{float(row['anchor_drift_error']):8.3f} "
            f"{float(row['total_error']):8.3f} "
            f"{float(row['articulation_score']):8.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="prototypes/articulation_eval/outputs/synthetic_articulation_sandbox",
        help="Directory for generated synthetic keypoints, scores, and SVG plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases()
    rows, serializable_cases = score_cases(cases)

    write_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "synthetic_keypoints.json").write_text(
        json.dumps(serializable_cases, indent=2), encoding="utf-8"
    )
    write_svg(out_dir / "trajectories.svg", serializable_cases)

    print_summary(rows)
    print(f"\nSaved summary CSV: {out_dir / 'summary.csv'}")
    print(f"Saved synthetic keypoints: {out_dir / 'synthetic_keypoints.json'}")
    print(f"Saved trajectory plot: {out_dir / 'trajectories.svg'}")


if __name__ == "__main__":
    main()
