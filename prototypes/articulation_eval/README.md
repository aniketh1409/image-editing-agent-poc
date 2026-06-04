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

Start with manually annotated keypoints in a small JSON file. For a hinge case,
the keypoints can include:

- `hinge_left`
- `hinge_right`
- `front_center`
- optional lid corners or edge points

The evaluator should compute simple scores first:

- hinge stability: hinge keypoints should move less than front keypoints
- opening amount: front keypoint should move over time
- rigidity proxy: distances between same-part keypoints should stay stable
- arc consistency: front keypoint should move approximately around the hinge axis
- visibility ratio: missing/occluded points should be reported separately

DeepEyes-style visual reasoning can later help localize keypoints or masks, but
the first version should stay lightweight and manually inspectable.
