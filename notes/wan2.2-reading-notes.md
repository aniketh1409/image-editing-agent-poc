# Wan2.2 Reading Notes

Public-safe notes for reading the Wan2.2 repo and paper.

Repo: https://github.com/Wan-Video/Wan2.2

## What Wan2.2 Is

Wan2.2 is a public video generation model family. For this project, the main value is understanding how modern large video generation models represent motion, conditioning, and temporal consistency.

## Model Variants

- T2V-A14B: text-to-video model.
- I2V-A14B: image-to-video model.
- TI2V-5B: text/image-to-video model, smaller than the A14B variants.
- Animate-14B: character animation and replacement.
- S2V-14B: speech-to-video, lower priority for this project.

## Input And Output

Input examples:

- text prompt
- reference image
- optional conditioning depending on task

Output:

- generated video frames

Tensor mental model:

```text
single video: T x C x H x W
batched video: B x T x C x H x W
```

## Important For This Project

Human-object interaction:

- Does the generated hand remain attached to the handle/object part?
- Does the hand slide, float, or disappear?

Articulated object state:

- Does a door/drawer/laptop preserve a coherent open/closed state?
- Does the state transition look physically plausible?

Occlusion:

- Are hand/object boundaries stable?
- Does the model hallucinate impossible overlaps?

Temporal consistency:

- Does the object flicker?
- Does the edited part drift across frames?

## First Baseline Prompt Ideas

```text
A person opens a wooden drawer while keeping one hand on the handle, realistic indoor lighting.
```

```text
A person closes a laptop with one hand, the hand remains in contact with the screen, realistic desk scene.
```

```text
A person opens a refrigerator door while standing in a kitchen, hand on the handle, realistic video.
```

## Questions While Reading

- What conditioning does each Wan2.2 task use?
- What does the VAE compress: individual frames, video chunks, or spatiotemporal patches?
- Where does the DiT operate?
- How is temporal consistency encouraged?
- Which model variant is practical on available Rorqual GPUs?
- Which output settings give the cheapest useful baseline?

## Notes From README

- Fill this section while reading the README.

## Notes From Paper

- Fill this section while reading the Wan paper.

## Things To Try On Rorqual

- Clone Wan2.2 into project storage.
- Download TI2V-5B first.
- Run one text-to-video prompt related to articulated objects.
- Save prompt, command, output path, GPU type, runtime, and visible failures.

## My Takeaway

Wan2.2 is a strong base video generation model family, not specifically an articulated-object editing system. The most practical starting point is TI2V-5B because it supports text/image-to-video and has lower GPU requirements. For my project, Wan2.2 is useful as a baseline to test failures in human-object contact, occlusion, articulated state, and temporal consistency.

## Direction

The goal is to study and improve video editing for human-articulated-object interactions, where edits must preserve spatial relationships such as contact, occlusion, object part motion, and temporal consistency.