# ArticulationEval Prototype

This prototype is the next step after the windowed RTSS rigidity baseline.

The current RTSS workflow samples generic surface points inside masks and checks
whether those tracked points move like a rigid patch. That is useful, but it
breaks when articulated motion hides one surface and reveals another surface,
as in a toilet lid opening.

ArticulationEval moves toward semantic, motion-type-aware evaluation:

- hinge rotation for lids, doors, and flaps
- sliding translation for drawers
- attachment consistency between moving part and base
- local rigidity/deformation of the moving part
- visibility-aware handling of occluded keypoints

## Planned v1

Start with manually annotated keypoints in a small JSON file. For a side-view
toilet-lid hinge case, the keypoints can include:

- `back_pivot`
- `left_pivot`
- `right_pivot`
- `lid_back_center`
- `lid_front_tip`
- `lid_midpoint`

The evaluator should compute simple scores first:

- hinge stability: hinge keypoints should move less than front keypoints
- opening amount: front keypoint should move over time
- rigidity proxy: distances between same-part keypoints should stay stable
- arc consistency: front keypoint should move approximately around the hinge axis
- visibility ratio: missing/occluded points should be reported separately

DeepEyes-style visual reasoning can later help localize keypoints or masks, but
the first version should stay lightweight and manually inspectable.

## Manual Keypoint Annotation

For the local toilet-lid case, run:

```bash
python prototypes/articulation_eval/manual_keypoint_annotator.py \
  --video case_toilet1_0/input.mp4 \
  --out_json prototypes/articulation_eval/examples/toilet_lid_keypoints.json \
  --frames 0,52,60,70,80
```

Click points in this order on each frame:

1. `left_pivot`: left visible hinge/pivot-side anchor; skip if hidden
2. `right_pivot`: right visible hinge/pivot-side anchor; skip if hidden
3. `lid_back_center`: center of the lid's rear/back edge near the hinge side
4. `lid_midpoint`: middle of the visible lid slab or edge
5. `lid_front_tip`: most visible front tip/end of the lid

The evaluator derives `back_pivot` as the midpoint between `left_pivot` and
`right_pivot` when both are visible.

Controls:

- left click: place the current keypoint
- `u`: undo the previous point
- `k`: skip a point if it is not visible
- Enter or `n`: go to the next frame
- `q` or Esc: cancel

## Run The Evaluator

After annotation, run:

```bash
python prototypes/articulation_eval/keypoint_articulation_eval.py \
  --keypoints_json prototypes/articulation_eval/examples/toilet_lid_keypoints.json \
  --out_csv prototypes/articulation_eval/outputs/toilet_lid_metrics.csv \
  --summary_json prototypes/articulation_eval/outputs/toilet_lid_summary.json \
  --video case_toilet1_0/input.mp4 \
  --viz_dir prototypes/articulation_eval/outputs/toilet_lid_viz \
  --out_video prototypes/articulation_eval/outputs/toilet_lid_articulation_eval.mp4
```

If the real annotation is not ready yet, run the toy example:

```bash
python prototypes/articulation_eval/keypoint_articulation_eval.py \
  --keypoints_json prototypes/articulation_eval/examples/toilet_lid_keypoints.example.json \
  --out_csv prototypes/articulation_eval/outputs/toy_toilet_lid_metrics.csv \
  --summary_json prototypes/articulation_eval/outputs/toy_toilet_lid_summary.json \
  --video case_toilet1_0/input.mp4 \
  --viz_dir prototypes/articulation_eval/outputs/toy_toilet_lid_viz \
  --out_video prototypes/articulation_eval/outputs/toy_toilet_lid_articulation_eval.mp4
```

The visual report draws:

- semantic keypoints
- pivot-to-lid rays
- the expected front-tip radius circle
- the observed front-tip trajectory
- per-frame metric text

## Build Demo Report

To create a browser-friendly report for presentations:

```bash
python prototypes/articulation_eval/build_demo_report.py \
  --metrics_csv prototypes/articulation_eval/outputs/toilet_lid_metrics.csv \
  --summary_json prototypes/articulation_eval/outputs/toilet_lid_summary.json \
  --viz_dir prototypes/articulation_eval/outputs/toilet_lid_viz \
  --assets_dir prototypes/articulation_eval/outputs/presentation_assets \
  --video prototypes/articulation_eval/outputs/toilet_lid_articulation_eval.mp4 \
  --out_html prototypes/articulation_eval/outputs/articulation_eval_demo.html
```

Open `prototypes/articulation_eval/outputs/articulation_eval_demo.html` in a
browser to show the semi-constructed pipeline end-to-end.

Presentation framing:

- RTSS is the local rigidity baseline.
- RTSS worked on visibility-consistent windows but failed when the lid surface
  became occluded or changed.
- Semantic keypoints are the next layer: they let us ask whether the lid moves
  like a hinged part, whether it deforms, and whether keypoints remain visible.
- Depth/3D can be added later, but generated-video depth should be treated as
  pseudo-3D rather than ground truth.
