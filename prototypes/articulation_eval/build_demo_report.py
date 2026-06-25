"""Build a visual HTML report for the ArticulationEval prototype.

The report is intentionally static: it can be opened directly in a browser and
shown during a meeting without needing a server.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def metric_card(label: str, value: str, note: str) -> str:
    return f"""
      <section class="metric-card">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(value)}</div>
        <div class="metric-note">{html.escape(note)}</div>
      </section>
    """


def build_table(rows: list[dict]) -> str:
    headers = [
        "frame_idx",
        "visibility_ratio",
        "pivot_drift_px",
        "tip_motion_px",
        "radius_error_mean_norm",
        "rigidity_error_mean_norm",
        "tip_angle_delta_from_start_deg",
    ]
    head = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = []
        for h in headers:
            value = row.get(h, "")
            try:
                value = f"{float(value):.3f}"
            except (TypeError, ValueError):
                pass
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_report(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root)
    out_html = Path(args.out_html)
    summary = read_json(Path(args.summary_json))
    rows = read_csv(Path(args.metrics_csv))
    viz_dir = Path(args.viz_dir)
    assets_dir = Path(args.assets_dir)
    video_path = Path(args.video) if args.video else None

    frames = sorted(viz_dir.glob("*.png"))
    plots = [
        assets_dir / "motion.svg",
        assets_dir / "errors.svg",
        assets_dir / "angle.svg",
        assets_dir / "visibility.svg",
    ]

    metric_cards = [
        metric_card("Visibility", f"{summary['mean_visibility_ratio']:.2f}", "How many semantic points were labelable."),
        metric_card("Pivot Drift", f"{summary['max_pivot_drift_px']:.2f}px", "Low drift means the hinge origin is stable."),
        metric_card("Tip Motion", f"{summary['max_tip_motion_px']:.2f}px", "Large motion confirms the lid opened."),
        metric_card("Radius Error", f"{summary['mean_radius_error_norm']:.2f}", "Distance to pivot should stay stable for hinge motion."),
        metric_card("Rigidity Error", f"{summary['mean_rigidity_error_norm']:.2f}", "Distances between lid points should stay stable."),
        metric_card("Angle Change", f"{summary['max_abs_tip_angle_delta_from_start_deg']:.2f} deg", "Opening amount around the pivot."),
    ]

    frame_imgs = "\n".join(
        f'<figure><img src="{rel(frame, out_html.parent)}" /><figcaption>{html.escape(frame.stem)}</figcaption></figure>'
        for frame in frames
    )
    plot_imgs = "\n".join(
        f'<figure><img src="{rel(plot, out_html.parent)}" /><figcaption>{html.escape(plot.stem)}</figcaption></figure>'
        for plot in plots
        if plot.exists()
    )
    video_block = ""
    if video_path and video_path.exists():
        video_block = f"""
        <section class="panel">
          <h2>Annotated Video</h2>
          <video controls src="{rel(video_path, out_html.parent)}"></video>
        </section>
        """

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ArticulationEval Demo Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #17202a;
      background: #f5f7fb;
    }}
    header {{
      background: #111827;
      color: white;
      padding: 36px 48px;
    }}
    header h1 {{ margin: 0 0 10px; font-size: 36px; }}
    header p {{ margin: 0; max-width: 980px; font-size: 18px; line-height: 1.45; color: #d1d5db; }}
    main {{ padding: 28px 48px 48px; }}
    .panel {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 22px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }}
    h2 {{ margin: 0 0 16px; font-size: 24px; }}
    .takeaway {{
      font-size: 22px;
      line-height: 1.4;
      border-left: 5px solid #e67700;
      padding-left: 18px;
      margin: 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 14px;
    }}
    .metric-card {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 16px;
      background: #fbfdff;
    }}
    .metric-label {{ color: #4b5563; font-size: 14px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .metric-note {{ color: #6b7280; font-size: 13px; line-height: 1.35; margin-top: 8px; }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 14px;
    }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 14px;
    }}
    figure {{
      margin: 0;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
      background: white;
    }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 9px 12px; color: #4b5563; font-size: 13px; }}
    video {{ width: 100%; max-height: 560px; background: #000; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f9fafb; color: #374151; }}
    .loop {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      align-items: stretch;
    }}
    .loop div {{
      background: #eef6ff;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      padding: 14px;
      font-weight: 700;
      text-align: center;
    }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      header {{ padding: 28px 22px; }}
      .metrics, .gallery, .plots, .loop {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ArticulationEval Demo Report</h1>
    <p>Case: {html.escape(summary.get('case_name', 'unknown'))}. A semantic-keypoint evaluator for hinge-like articulated-object motion in generated video.</p>
  </header>
  <main>
    <section class="panel">
      <h2>Main Takeaway</h2>
      <p class="takeaway">The generated toilet lid opens visually, but the keypoint geometry suggests the lid does not behave like a clean rigid hinge. This turns a qualitative failure into measurable feedback.</p>
    </section>

    <section class="panel">
      <h2>Pipeline</h2>
      <div class="loop">
        <div>Generated video</div>
        <div>Semantic keypoints</div>
        <div>ArticulationEval</div>
        <div>Failure signal</div>
        <div>Filter / refine</div>
      </div>
    </section>

    <section class="panel">
      <h2>Summary Metrics</h2>
      <div class="metrics">{''.join(metric_cards)}</div>
    </section>

    {video_block}

    <section class="panel">
      <h2>Annotated Keypoint Frames</h2>
      <div class="gallery">{frame_imgs}</div>
    </section>

    <section class="panel">
      <h2>Metric Plots</h2>
      <div class="plots">{plot_imgs}</div>
    </section>

    <section class="panel">
      <h2>Frame-Level Metrics</h2>
      {build_table(rows)}
    </section>
  </main>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")
    print(f"Saved demo report: {out_html}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--viz_dir", required=True)
    parser.add_argument("--assets_dir", required=True)
    parser.add_argument("--video", default="")
    parser.add_argument("--out_html", required=True)
    return parser.parse_args()


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()
