# ArticulationEval Presentation Outline

Use this as a 15-20 minute meeting story.

Main thesis:

Generated video editing is getting strong visually, but current metrics do not
directly test whether articulated objects move correctly. My near-term direction
is an evaluator for articulation quality, starting from RTSS/local rigidity and
moving toward semantic keypoint and motion-type-specific constraints.

Mentor-aligned framing:

The immediate goal is not to directly improve a diffusion/video model. The more
useful near-term problem is verification and filtering: given generated
human/articulated-object interaction videos, can we automatically identify which
ones preserve meaningful interaction and articulation quality well enough to
use for analysis, dataset construction, or later refinement?

## Timed Agenda

1. Motivation and research gap: 3 min
2. Background: Wan/VACE/KinEdit and why editing/evaluation matters: 3-4 min
3. RTSS baseline experiment: 4 min
4. RTSS failure analysis on toilet lid: 3 min
5. ArticulationEval v1 keypoint evaluator: 4-5 min
6. Limitations and next steps: 2-3 min

## 1. Problem

Generic video metrics such as FID/FVD or background consistency do not directly
measure whether an articulated object moves correctly.

For human/articulated-object editing, we care about:

- the moving part staying rigid
- the part staying attached to the base
- the motion matching the articulation type
- occlusion and surface visibility changes over time

Talking point:

Image/video editing models may produce plausible frames, but for articulated
objects we need to ask whether the edit respects the object's mechanism.

Examples:

- toilet lid opening should rotate around a hinge
- drawer opening should translate along a slide axis
- door opening should rotate while staying attached to the frame
- human hand/object contact should remain plausible

## 2. Background Context

Wan/Wan2.2:

- strong video generation family
- useful base for text/image-to-video
- not specifically an articulation evaluator

VACE:

- closer to editing because it uses prompts, context frames, and masks
- useful framing: editing is controlled by spatiotemporal conditions
- still does not directly guarantee contact/articulation correctness

KinEdit/InfiniKin:

- more directly related to articulated-object motion
- suggests articulation-specific data/modules help
- existing metrics still do not fully capture rigidity, attachment, or correct
  motion type

DeepEyes-style agents:

- possible future layer for visual reasoning and keypoint/mask localization
- too heavy to put in a full loop immediately
- better as a later automation layer after the evaluator is defined

## 3. RTSS Baseline

I first tested a windowed RTSS-style rigidity evaluator using CoTracker points
inside manually annotated masks.

Workflow:

1. Local annotation: define visibility-consistent windows and draw moving-part
   masks with `local_window_mask_annotator_v2.py`.
2. Transfer annotated case to the 4090 server.
3. Server evaluation: run `server_manual_window_rtss.py` with 120 sampled points
   per mask/window.
4. Inspect `annotated_*.mp4` before trusting CSV metrics.
5. Read `manual_window_scores.csv` and `manual_window_summary.csv`.

Observation from `case_toilet1_0`:

- A visibility-consistent window produced finite 2D RTSS.
- A later window failed because the toilet lid surface became occluded or
  changed visibility.

Interpretation:

RTSS is useful for local visible surface rigidity, but it is not enough for a
whole articulated motion because the same surface may not remain visible.

Concrete result from `case_toilet1_0`:

- `window_00`, frames `0-38`: valid but mostly still
- `window_01`, frames `39-46`: valid early-motion score
- `window_02`, frames `47-80`: invalid endpoint visibility, because the visible
  lid surface changes/occludes

Numbers:

- `window_00`: `endpoint_visible_points=113`,
  `rtss2d_action_p90=0.022529`
- `window_01`: `endpoint_visible_points=64`,
  `rtss2d_action_p90=0.070457`
- `window_02`: `endpoint_visible_points=0`, endpoint RTSS becomes NaN

Slide visual:

- show annotated RTSS tracking video or stills if available
- show table with the three windows
- emphasize that NaN is a meaningful visibility failure, not just a code crash

## 4. Keypoint Direction

The next evaluator uses semantic keypoints instead of many generic surface
points.

For the side-view toilet lid example, the first keypoints are:

- `back_pivot`
- `left_pivot`
- `right_pivot`
- `lid_back_center`
- `lid_front_tip`
- `lid_midpoint`

These let us ask motion-type-specific questions:

- Does the front tip move away from the initial position?
- Does the pivot stay mostly stable?
- Do distances between lid points stay stable?
- Do lid points roughly preserve distance to the pivot?

Why this is stronger than generic points:

- generic points know texture, not object semantics
- semantic points encode the mechanism: pivot, front tip, moving part
- the same framework can be adapted by motion type

Motion-type examples:

- hinge rotation: preserve distance to pivot/hinge axis
- drawer sliding: preserve part shape while translating along one main axis
- knob rotation: preserve radius around central axis
- door opening: hinge rotation plus attachment consistency

## 5. Current Artifact

Implemented:

- `manual_keypoint_annotator.py`
- `keypoint_articulation_eval.py`
- toy keypoint JSON example
- CSV and JSON metric outputs
- optional visual report with annotated keypoints, pivot rays, radius circle,
  trajectory, and metric text

The current evaluator reports:

- visibility ratio
- pivot drift
- tip motion
- pivot-radius consistency
- pairwise lid rigidity
- approximate angle change around the pivot

Real `case_toilet1_0` result:

- `mean_visibility_ratio=1.0`
- `max_pivot_drift_px=22.47`
- `max_tip_motion_px=316.62`
- `mean_radius_error_norm=1.51`
- `max_radius_error_norm=5.38`
- `mean_rigidity_error_norm=0.52`
- `max_rigidity_error_norm=1.08`
- `max_abs_tip_angle_delta_from_start_deg=91.09`

Interpretation:

The lid visibly opens, but the semantic metrics suggest the generated motion is
not cleanly rigid or hinge-consistent under the manual keypoint model. This is
exactly the kind of failure generic video metrics may miss.

## 6. Limitations

- Current keypoints are manual.
- Current version is 2D.
- Generated-video depth may be inconsistent, so 3D should be treated as
  pseudo-3D unless ground-truth synthetic data is available.
- The pivot may be visually hidden, so the evaluator must handle missing
  keypoints honestly.

Important honesty point:

The current metric is not a final benchmark. It is a prototype for defining
what should be measured before automating the measurement.

## 7. Stronger Metric Roadmap

Short term:

- make keypoint evaluator robust to missing points
- compare one good-looking case and one bad-looking case
- add visual reports for each evaluated case
- use ArticulationEval as a filtering/ranking signal over generated candidates

Medium term:

- add pseudo-3D using Video Depth Anything or MiDaS, while treating generated
  depth as approximate
- add separate metric modes:
  - hinge rotation
  - sliding translation
  - contact/attachment preservation
  - background/motion leakage

Long term:

- use DeepEyes or another vision agent to localize semantic keypoints/masks
- use ArticulationEval as reward/feedback for an editing or RL loop
- evaluate multiple generators/editors such as KinEdit, VACE, and Wan-style
  baselines
- support agent-assisted dataset construction: generate or collect interaction
  videos, verify articulation/contact quality, keep high-quality samples, and
  flag failure modes for refinement

## Agent/Data Construction Connection

Hang's DeepEyes/RL direction can be framed as a verification/refinement loop:

1. Generate or collect candidate interaction video.
2. Use a visual agent to understand the scene and localize object parts,
   keypoints, masks, or contact regions.
3. Use custom metrics such as ArticulationEval to verify motion quality.
4. Filter good samples for a dataset or send bad samples back for refinement.

This connects image compositing and video editing:

- image compositing: localize object/part, edit it, verify geometry/occlusion
- video editing: localize object/part over time, edit it, verify temporal
  articulation/contact consistency

The evaluator supplies the "self-verifying" signal that a DeepEyes-style agent
or RL loop could use.

## 8. Next Step

Annotate real keypoints on `case_toilet1_0`, run the evaluator, and compare:

- good/valid hinge-like motion
- blurry or deformed motion
- visibility failures

Longer term, DeepEyes or another visual reasoning tool could localize keypoints
automatically, while ArticulationEval provides feedback or reward.

## Backup Slide Ideas

If more time is needed:

- Explain temporal vs spatiotemporal masks.
- Explain why surface occlusion broke RTSS.
- Show how a drawer metric would differ from a hinge metric.
- Show the agentic loop:
  prompt/edit intent -> locate part/keypoints -> generate/edit -> evaluate ->
  refine.
- Discuss why generated-video depth is useful but risky.

## Final-Day Work Plan

Priority 1: strengthen the semantic keypoint story.

- Create a visual report from the real `case_toilet1_0` annotation.
- Show annotated frames/video with keypoints, pivot rays, radius circle, and
  metric text.
- Add one slide explaining each metric in plain English:
  visibility, pivot drift, tip motion, radius consistency, rigidity, angle
  change.

Priority 2: depth estimation as a feasibility probe, not a dependency.

- Present MiDaS or Video Depth Anything as possible pseudo-3D lifting tools.
- Be explicit that generated-video depth may be inconsistent and should not be
  treated as ground truth.
- The useful research question is whether depth helps distinguish perspective
  effects from true non-rigid deformation.

Priority 3: DeepEyes as the future automation/agent layer.

- Do not claim DeepEyes solves video editing.
- Frame it as a visual reasoning/localization agent that could identify
  keypoints, inspect failures, or choose where to look/edit.
- Connect it to image compositing: image-level object/part localization and
  compositing are simpler versions of the same control problem that video
  editing extends over time.
