# VACE Reading Notes

Paper: VACE: All-in-One Video Creation and Editing  
arXiv: https://arxiv.org/abs/2503.07598

## One-Sentence Takeaway

VACE is a unified video creation and editing framework that turns text, images, videos, and masks into a common conditioning format so one video model can handle many tasks such as reference-to-video, video-to-video editing, and masked video editing.

## Why This Matters For My Project

Wan2.2 is mostly the base video generation model family. VACE is closer to the actual editing problem because it asks how to condition a video model on existing videos, masks, references, and controls.

For human-articulated-object scenes, the important question is whether these general editing controls are enough to preserve:

- hand-object contact
- occlusion boundaries
- articulated object state
- plausible part motion
- temporal consistency

## Main Tasks

VACE groups video tasks around four main modalities:

- text
- image/reference
- video
- mask

The main task types are:

- T2V: text-to-video generation
- R2V: reference-to-video generation
- V2V: video-to-video editing
- MV2V: masked video-to-video editing
- task composition: combinations of the above

## Key Idea: Video Condition Unit

The Video Condition Unit, or VCU, is VACE's unified input interface.

Instead of making a separate model for every task, VACE represents task inputs as:

```text
text prompt
+ context frame sequence
+ mask sequence
```

Important shape intuition:

```text
context frames: T x C x H x W
context masks:  T x 1 x H x W
```

This connects directly to the video tensor exercise.

Plain-English version:

```text
VCU = a standardized package of "what the model should pay attention to"
```

For normal generation, the package might mostly be a text prompt. For editing, the package can also include existing video frames and masks that say which pixels/frames are known, protected, or editable.

This means VACE can express many tasks using the same basic format:

```text
generate from text:
  prompt + empty/no context + empty/no mask

edit a video:
  prompt + original video frames + mask over the edit region

extend or compose video:
  prompt + partial context frames + masks showing what is available
```

The VCU is not the final generated video. It is the conditioning information given to the model so the model knows what task it is doing.

## Key Idea: Masks Are Spatiotemporal

For masked video-to-video editing, the mask is not just a 2D image mask. It is a 3D region of interest across time:

```text
frame 0 mask
frame 1 mask
frame 2 mask
...
```

This is why moving masks matter. The edited region can move as the object, hand, or camera moves.

## Key Idea: Context Adapter

VACE injects task-specific context into a video Diffusion Transformer using a Context Adapter.

High-level interpretation:

```text
base video model generates video
context adapter tells it how to use video/mask/reference controls
```

This lets the system support many creation/editing tasks without training a totally separate model for each one.

## Important For Articulated Objects

VACE supports masks, videos, references, pose, depth, optical flow, layout, and other RGB-representable controls. These are useful for articulated-object editing, but the paper does not appear to center the problem around explicit physical interaction constraints.

Possible gap:

```text
VACE can condition on where to edit,
but may not explicitly know whether a hand stays in contact with a drawer handle
or whether a door hinge motion is physically coherent.
```

## Failure Modes To Watch For

- hand floats away from handle
- hand clips through object
- drawer/door state changes inconsistently across frames
- object part deforms instead of rotating/sliding
- occlusion boundary flickers
- mask follows the wrong region over time
- generated object identity changes across frames

## Questions While Reading

- What inputs does VACE use for each editing task?
- How are masks represented across time?
- How does VACE separate unchanged regions from regions to regenerate?
- Which controls are closest to articulated-object state: depth, pose, optical flow, layout, mask?
- Does VACE evaluate human-object interaction quality directly?
- Could a contact/articulation metric be added on top of VACE outputs?
- Could Blender-generated labels provide stronger supervision or evaluation?

## My Current Research Interpretation

VACE is a strong general-purpose video editing framework. My possible research angle is not to rebuild VACE, but to study or improve how VACE/Wan-style systems handle human-articulated-object interaction edits, where preserving contact, occlusion, part motion, and temporal consistency is harder than simply making the video look good.
