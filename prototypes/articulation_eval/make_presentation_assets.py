"""Create presentation-ready ArticulationEval plots from CSV/JSON outputs.

This intentionally uses only the Python standard library so it works even in
minimal environments. It writes SVG plots and a Markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PLOT_SPECS = [
    ("motion", "Tip Motion vs Pivot Drift", [
        ("tip_motion_px", "tip motion px", "#2f9e44"),
        ("pivot_drift_px", "pivot drift px", "#d9480f"),
    ]),
    ("errors", "Hinge/Rigidity Error", [
        ("radius_error_mean_norm", "radius error norm", "#1971c2"),
        ("rigidity_error_mean_norm", "rigidity error norm", "#9c36b5"),
    ]),
    ("angle", "Opening Angle Change", [
        ("tip_angle_delta_from_start_deg", "angle delta deg", "#e67700"),
    ]),
    ("visibility", "Keypoint Visibility", [
        ("visibility_ratio", "visibility ratio", "#0ca678"),
    ]),
]


def as_float(value: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def nice_num(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def polyline_points(rows: list[dict], metric: str, x_min: float, x_span: float, y_min: float, y_span: float):
    points = []
    for row in rows:
        x = as_float(row["frame_idx"])
        y = as_float(row.get(metric, "nan"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        px = 70 + ((x - x_min) / max(x_span, 1e-6)) * 700
        py = 310 - ((y - y_min) / max(y_span, 1e-6)) * 240
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def write_svg(path: Path, rows: list[dict], title: str, series: list[tuple[str, str, str]]) -> None:
    xs = [as_float(row["frame_idx"]) for row in rows]
    ys = []
    for metric, _label, _color in series:
        ys.extend(as_float(row.get(metric, "nan")) for row in rows)
    xs = [x for x in xs if math.isfinite(x)]
    ys = [y for y in ys if math.isfinite(y)]
    if not xs or not ys:
        return

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(ys)
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1.0
    x_span = x_max - x_min
    y_span = y_max - y_min

    legend = []
    y_legend = 34
    for metric, label, color in series:
        legend.append(f'<circle cx="620" cy="{y_legend}" r="5" fill="{color}" />')
        legend.append(f'<text x="632" y="{y_legend + 5}" class="small">{label}</text>')
        y_legend += 20

    lines = []
    dots = []
    for metric, _label, color in series:
        pts = polyline_points(rows, metric, x_min, x_span, y_min, y_span)
        if pts:
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" />')
            for pt in pts.split(" "):
                x, y = pt.split(",")
                dots.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}" />')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="840" height="380" viewBox="0 0 840 380">
  <style>
    .title {{ font: 700 22px Arial, sans-serif; fill: #111; }}
    .label {{ font: 14px Arial, sans-serif; fill: #333; }}
    .small {{ font: 13px Arial, sans-serif; fill: #333; }}
    .grid {{ stroke: #ddd; stroke-width: 1; }}
    .axis {{ stroke: #333; stroke-width: 2; }}
  </style>
  <rect width="840" height="380" fill="#fff" />
  <text x="70" y="38" class="title">{title}</text>
  <line x1="70" y1="310" x2="770" y2="310" class="axis" />
  <line x1="70" y1="70" x2="70" y2="310" class="axis" />
  <line x1="70" y1="190" x2="770" y2="190" class="grid" />
  <line x1="420" y1="70" x2="420" y2="310" class="grid" />
  <text x="70" y="340" class="label">frame {nice_num(x_min)}</text>
  <text x="715" y="340" class="label">frame {nice_num(x_max)}</text>
  <text x="18" y="314" class="label">{nice_num(y_min)}</text>
  <text x="18" y="76" class="label">{nice_num(y_max)}</text>
  {''.join(legend)}
  {''.join(lines)}
  {''.join(dots)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_report(path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# ArticulationEval Result Summary",
        "",
        "## Main Takeaway",
        "",
        "The generated toilet-lid video shows large front-tip motion with relatively low pivot drift, but radius and rigidity errors remain high. This suggests the lid opens visually, yet does not behave like a clean rigid hinge under the semantic keypoint model.",
        "",
        "## Summary Metrics",
        "",
        f"- Visibility ratio: `{summary['mean_visibility_ratio']:.2f}`",
        f"- Max pivot drift: `{summary['max_pivot_drift_px']:.2f}px`",
        f"- Max lid-tip motion: `{summary['max_tip_motion_px']:.2f}px`",
        f"- Mean radius error: `{summary['mean_radius_error_norm']:.2f}`",
        f"- Mean rigidity error: `{summary['mean_rigidity_error_norm']:.2f}`",
        f"- Max angle change: `{summary['max_abs_tip_angle_delta_from_start_deg']:.2f} deg`",
        "",
        "## Slide Interpretation",
        "",
        "- Low pivot drift means the approximate hinge origin is fairly stable.",
        "- Large tip motion confirms the lid opens substantially.",
        "- High radius error means lid points do not preserve distance to the pivot.",
        "- High rigidity error means distances between lid points change too much.",
        "- This supports the ArticulationEval motivation: visual plausibility alone is not enough for articulated-object editing.",
        "",
        "## Frame-Level Notes",
        "",
    ]
    for row in rows:
        lines.append(
            f"- Frame `{row['frame_idx']}`: visibility `{as_float(row['visibility_ratio']):.2f}`, "
            f"tip motion `{as_float(row['tip_motion_px']):.1f}px`, "
            f"pivot drift `{nice_num(as_float(row['pivot_drift_px']))}px`, "
            f"rigidity error `{nice_num(as_float(row['rigidity_error_mean_norm']))}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.metrics_csv))
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))

    for slug, title, series in PLOT_SPECS:
        write_svg(out_dir / f"{slug}.svg", rows, title, series)
    write_report(out_dir / "result_summary.md", rows, summary)
    print(f"Saved presentation assets to: {out_dir}")


if __name__ == "__main__":
    main()
