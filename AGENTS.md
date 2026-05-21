# Agent Notes

Project goal: build AI agent prototypes for image/video editing using deep learning and generative models, with the research direction shifting toward interaction-aware video editing for human-articulated-object scenes.

Working-memory rule:

- Treat this file as the repo's compact project memory.
- At the start of future work in this repo, read `AGENTS.md` before deciding what to do.
- After meaningful discussion or implementation, update `AGENTS.md` with relevant current context so future chats can resume quickly.

Repo organization:

- `foundation/`: public/basic learning and simple editor POC.
- `foundation/pytorch-basics/`: small runnable PyTorch exercises.
- `foundation/scripts/simple_editor.py`: classical mask-based image editor.
- `foundation/images/`: sample inputs for beginner scripts.
- `foundation/outputs/`: generated beginner-script outputs; generally not important research artifacts.
- `notes/`: public-safe research notes only.
- Keep private lab paper notes, unpublished ideas, private data, and non-shareable experiments out of this public repo.
- Create a separate private repo only when work touches the lab's unpublished articulated-object paper, private datasets, or lab-specific confidential experiments.

Current learning track:

1. PyTorch fundamentals: tensors, shapes, indexing, broadcasting, autograd.
2. Neural network basics in PyTorch: `torch.nn.Module`, losses, optimizers, training loops.
3. Vision basics: image tensors, transforms, masks, convolutions, simple segmentation.
4. Video tensor basics: `T x C x H x W` videos, `B x T x C x H x W` batches, moving masks, frame differences, toy contact checks.
5. Generative video models: Wan/Wan2.2 as base video generation model family.
6. Video editing frameworks: VACE as a unified video creation/editing framework with text, context frames, and mask sequences.
7. Agent pipeline: interpret edit intent, choose/edit masks or regions, call model, evaluate/refine output.

Current implemented files:

- `foundation/pytorch-basics/01_tensors.py`: tensor basics, image tensors, masks, broadcasting, matrix multiplication, autograd.
- `foundation/pytorch-basics/02_training_loop.py`: first PyTorch training loop for `y = 3x + 2`.
- `foundation/pytorch-basics/03_real_image_tensors.py`: load real images, create masks, black/average/blur/soft-mask fills, crop/resize/batch/normalize.
- `foundation/pytorch-basics/04_augmentations.py`: basic image augmentations.
- `foundation/pytorch-basics/05_video_tensors.py`: fake video tensors, moving spatiotemporal masks, black-fill video edit, frame differences, toy hand-object contact failure metric.
- `foundation/scripts/simple_editor.py`: CLI image + mask + edit mode tool with `black`, `average`, `blur`, and `soft-blur`.

Current public notes:

- `notes/research-map.md`: public-safe research map for interaction-aware image/video editing.
- `notes/wan2.2-reading-notes.md`: Wan2.2 reading notes.
- `notes/vace-reading-notes.md`: VACE reading notes.

Research direction:

- Main direction: interaction-aware image/video editing for human-articulated-object scenes.
- Core challenge: edit video while preserving human-object contact, occlusion relationships, articulated object state, plausible part motion, and temporal consistency.
- Example tasks: open/close drawer while preserving hand contact; replace articulated object while keeping contact plausible; edit object state across frames; remove/modify an occluding articulated object without breaking human pose/object geometry.
- Blender/simulation can provide precise synthetic data and labels, but Wan/VACE-style models are needed for flexible pixel-space generation/editing when full 3D scenes are unavailable.

Key model/paper understanding:

- Wan/Wan2.2: base video generation model family. Practical first target is `TI2V-5B` because it supports text/image-to-video and has lower GPU requirements than A14B models.
- VACE: closer to actual video editing. It uses a unified Video Condition Unit: text prompt + context frame sequence + mask sequence.
- Plain-English VCU understanding: a standardized conditioning package for VACE. It bundles the prompt, existing/partial video frames, and spatiotemporal masks so generation, video editing, masked editing, and task composition can be represented in one format.
- "Temporal" means related to time. In video tensors, the temporal dimension is `T`, the sequence of frames. Temporal consistency means objects, masks, edits, and interactions stay coherent from frame to frame.
- "Spatiotemporal" means space plus time. In video, it refers to where something is in each frame and how that location/shape changes across frames. A spatiotemporal mask has shape like `T x 1 x H x W`.
- Moving video masks matter because video editing needs "where to edit at each time step," not just a static 2D image mask.
- The likely research gap is that general video editing controls may not explicitly preserve contact, occlusion, or articulated object state.

Near-term next steps:

1. Read VACE selectively first, not cover-to-cover: focus on abstract/introduction, method overview, Video Condition Unit, masked video-to-video editing, task composition, experiments/examples, and limitations/failure modes if present.
2. Keep updating `notes/vace-reading-notes.md` with public-safe takeaways.
3. Debug Rorqual Slurm account/partition access.
4. Run a public Wan2.2 `TI2V-5B` baseline if GPU access works.
5. Evaluate generated videos for contact, occlusion, object state, and temporal flicker.

Rorqual/Alliance status:

- User has Alliance access and successfully logged into Rorqual.
- Project path found: `/project/rrg-vislearn/kini1409`.
- Existing virtualenv: `/project/rrg-vislearn/kini1409/envs/image-editing`.
- Installed there: `torch==2.4.1+computecanada`, `torchvision==0.19.1+computecanada`, `torchaudio==2.4.1+computecanada`.
- Previous Slurm issue: `--account=rrg-vislearn` with GPU partitions failed as invalid account/partition combination.
- Suggested diagnostic commands:
  - `sacctmgr show assoc user=$USER format=Cluster,Account,User,Partition,QOS%40,DefaultQOS`
  - `sacctmgr show assoc user=$USER -P`

Tutoring style:

- Keep examples small, runnable, and tied to image-editing tasks.
- Emphasize tensor shapes constantly.
- Prefer hands-on scripts under `foundation/pytorch-basics/` before larger prototypes.
- Use CPU-friendly examples unless the environment changes.
- Keep public notes concise and safe for GitHub.
