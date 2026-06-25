"""Generate slide-ready concept diagrams for the ArticulationEval story."""

from __future__ import annotations

from pathlib import Path


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def metric_map_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <style>
    .title { font: 700 34px Arial, sans-serif; fill: #111; }
    .subtitle { font: 18px Arial, sans-serif; fill: #444; }
    .box { fill: #f8f9fa; stroke: #343a40; stroke-width: 2; rx: 12; }
    .ours { fill: #fff4e6; stroke: #e67700; stroke-width: 3; rx: 12; }
    .kps { fill: #e7f5ff; stroke: #1971c2; stroke-width: 3; rx: 12; }
    .text { font: 20px Arial, sans-serif; fill: #111; }
    .small { font: 16px Arial, sans-serif; fill: #333; }
    .arrow { stroke: #555; stroke-width: 3; fill: none; marker-end: url(#arrow); }
  </style>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/>
    </marker>
  </defs>
  <rect width="1280" height="720" fill="#ffffff"/>
  <text x="60" y="70" class="title">Metric-First Evaluation: Where ArticulationEval Fits</text>
  <text x="60" y="105" class="subtitle">PhysX-Omni defines multiple evidence-based metrics; KPS is closest to articulated motion, but human-object generated video needs interaction-aware checks.</text>

  <rect x="60" y="155" width="260" height="95" class="box"/>
  <text x="85" y="190" class="text">RQS / MCS</text>
  <text x="85" y="220" class="small">render quality + multi-view consistency</text>

  <rect x="360" y="155" width="260" height="95" class="box"/>
  <text x="385" y="190" class="text">DCS / DQS</text>
  <text x="385" y="220" class="small">description + dimension consistency</text>

  <rect x="660" y="155" width="260" height="95" class="box"/>
  <text x="685" y="190" class="text">APS / MPS</text>
  <text x="685" y="220" class="small">affordance + material plausibility</text>

  <rect x="960" y="155" width="260" height="95" class="kps"/>
  <text x="985" y="190" class="text">KPS</text>
  <text x="985" y="220" class="small">kinematic plausibility from articulation video</text>

  <path d="M 1090 250 C 1090 325, 920 335, 820 390" class="arrow"/>

  <rect x="200" y="390" width="880" height="190" class="ours"/>
  <text x="235" y="430" class="text">ArticulationEval for Generated Interaction Videos</text>
  <text x="235" y="470" class="small">Evidence: semantic keypoints + generated video frames</text>
  <text x="235" y="505" class="small">Checks: visibility, pivot stability, front-tip motion, radius consistency, part rigidity, angle change</text>
  <text x="235" y="540" class="small">Future: contact / attachment consistency, motion leakage, pseudo-3D depth, agent-assisted verification</text>

  <text x="250" y="640" class="subtitle">Takeaway: my metric prototype is a video-interaction counterpart to metric-first benchmarking, focused on articulated motion failures that visual quality metrics miss.</text>
</svg>
"""


def agent_loop_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <style>
    .title { font: 700 34px Arial, sans-serif; fill: #111; }
    .subtitle { font: 18px Arial, sans-serif; fill: #444; }
    .box { fill: #f8f9fa; stroke: #343a40; stroke-width: 2; rx: 14; }
    .eval { fill: #fff4e6; stroke: #e67700; stroke-width: 3; rx: 14; }
    .agent { fill: #e7f5ff; stroke: #1971c2; stroke-width: 3; rx: 14; }
    .text { font: 20px Arial, sans-serif; fill: #111; }
    .small { font: 15px Arial, sans-serif; fill: #333; }
    .arrow { stroke: #555; stroke-width: 3; fill: none; marker-end: url(#arrow); }
  </style>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/>
    </marker>
  </defs>
  <rect width="1280" height="720" fill="#ffffff"/>
  <text x="60" y="70" class="title">DeepEyes-Style Agent + ArticulationEval Verification Loop</text>
  <text x="60" y="105" class="subtitle">Use the agent for localization/planning; use ArticulationEval for motion-specific verification and filtering.</text>

  <rect x="60" y="190" width="185" height="120" class="box"/>
  <text x="90" y="230" class="text">Input</text>
  <text x="90" y="262" class="small">image/video</text>
  <text x="90" y="287" class="small">+ edit intent</text>

  <path d="M245 250 L300 250" class="arrow"/>
  <rect x="300" y="190" width="210" height="120" class="agent"/>
  <text x="330" y="225" class="text">Visual Agent</text>
  <text x="330" y="258" class="small">locate object parts</text>
  <text x="330" y="283" class="small">keypoints / masks</text>

  <path d="M510 250 L565 250" class="arrow"/>
  <rect x="565" y="190" width="210" height="120" class="box"/>
  <text x="595" y="225" class="text">Editor</text>
  <text x="595" y="258" class="small">VACE / KinEdit</text>
  <text x="595" y="283" class="small">Wan-style video</text>

  <path d="M775 250 L830 250" class="arrow"/>
  <rect x="830" y="190" width="260" height="120" class="eval"/>
  <text x="860" y="225" class="text">ArticulationEval</text>
  <text x="860" y="258" class="small">rigidity / hinge / contact</text>
  <text x="860" y="283" class="small">visibility-aware score</text>

  <path d="M1090 250 L1145 250" class="arrow"/>
  <rect x="1145" y="190" width="95" height="120" class="box"/>
  <text x="1165" y="238" class="text">Keep</text>
  <text x="1165" y="268" class="small">or reject</text>

  <path d="M960 310 C 960 470, 405 470, 405 320" class="arrow"/>
  <rect x="430" y="420" width="420" height="110" class="box"/>
  <text x="465" y="458" class="text">Self-Verifying Refinement</text>
  <text x="465" y="492" class="small">metric failures guide prompt/mask/control changes</text>

  <text x="150" y="620" class="subtitle">Dataset construction use: generate candidates -> verify interactions -> keep high-quality samples -> flag failure modes for refinement.</text>
</svg>
"""


def main() -> None:
    out_dir = Path("prototypes/articulation_eval/outputs/presentation_assets")
    write(out_dir / "metric_taxonomy_map.svg", metric_map_svg())
    write(out_dir / "agent_verification_loop.svg", agent_loop_svg())
    print(f"Saved concept diagrams to: {out_dir}")


if __name__ == "__main__":
    main()
