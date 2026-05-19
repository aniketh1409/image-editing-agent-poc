# Research Map

Public-safe map for the internship direction. Keep private lab paper details, private data, and unpublished ideas outside this repo.

## Current Direction

Interaction-aware image/video editing for human-articulated object scenes.

The goal is to understand and eventually improve editing pipelines where edits preserve:

- human-object contact
- occlusion relationships
- articulated object state
- temporal consistency across frames
- plausible human-object interaction geometry

## Why This Is Hard

- Articulated objects have parts and states, such as open/closed drawers, doors, cabinets, laptops, and appliances.
- Human hands often occlude the object part being edited.
- A visually good frame can still be physically implausible if contact or object state breaks.
- Video adds temporal consistency: the edit must stay stable across frames.

## Public Models And Papers To Read

- Wan2.2 / Wan paper: base video generation model family.
- VACE: video creation/editing framework built around Wan-style video generation.
- FLUX.1 Kontext: image editing and in-context generation reference point.
- DeepEyes / DeepEyesV2: visual reasoning agent references for later.

## Candidate Research Questions

- Can an editing pipeline detect when human-object contact is broken after generation?
- Can masks, pose, depth, segmentation, or object part state guide video edits better?
- Where do Wan/VACE-style baselines fail on drawer, door, cabinet, and laptop interactions?
- Can an agent iteratively critique and refine video edits based on contact, occlusion, and state consistency?

## Next Experiments

- Create toy video tensor exercises with moving masks.
- Run a public Wan2.2 baseline on simple human-articulated-object prompts.
- Compare outputs against failure categories: contact, occlusion, object state, temporal flicker.
- Read VACE after the first Wan pass and map what VACE adds.

## Open Questions For Supervisor

- Which articulated object categories matter most for the lab direction?
- Should the first baseline be image-to-video, video-to-video, or text-to-video?
- Are there approved public datasets or demo videos to use first?
- When should this work move into a private repo?
