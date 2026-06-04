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
- `prototypes/contact-eval/`: reusable prototype for evaluating hand-object contact preservation across short frame sequences.
- `prototypes/articulation_eval/`: public-safe scaffold for the next evaluator: semantic keypoint and motion-type-specific articulated-object metrics.
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
- `prototypes/contact-eval/contact_eval.py`: early reusable contact evaluator. It loads hand/object boxes from JSON, optionally overlays them on frame folders, computes center and edge distance, uses edge distance for contact pass/fail, and writes per-frame CSV metrics. Output folders are ignored by git.
- `prototypes/contact-eval/generate_example_frames.py`: helper that generates simple good/broken example frame folders from the JSON annotations for testing frame-folder support. Generated frame folders are ignored by git.
- `prototypes/articulation_eval/README.md`: scaffold notes for moving from windowed RTSS/local rigidity to semantic keypoint articulation metrics.
- `prototypes/articulation_eval/keypoint_articulation_eval.py`: placeholder entry point for the upcoming keypoint-based ArticulationEval prototype.
- `requirements-lab-eval.txt`: public dependency list for locally copied private lab evaluation scripts; `lab_eval_tools/` itself is ignored and should not be committed.

Current public notes:

- `notes/research-map.md`: public-safe research map for interaction-aware image/video editing.
- `notes/wan2.2-reading-notes.md`: Wan2.2 reading notes.
- `notes/vace-reading-notes.md`: VACE reading notes.
- `notes/june-10-demo-plan.md`: local-only ignored plan for June 10 demo.

Research direction:

- Main direction: interaction-aware image/video editing for human-articulated-object scenes.
- Core challenge: edit video while preserving human-object contact, occlusion relationships, articulated object state, plausible part motion, and temporal consistency.
- Example tasks: open/close drawer while preserving hand contact; replace articulated object while keeping contact plausible; edit object state across frames; remove/modify an occluding articulated object without breaking human pose/object geometry.
- Blender/simulation can provide precise synthetic data and labels, but Wan/VACE-style models are needed for flexible pixel-space generation/editing when full 3D scenes are unavailable.
- Supervisor/mentor update: the professor likes seeing polished/fancy work and will ask concept-understanding questions. For the next meeting, the expectation is not a huge finished system, but there should be a working artifact or experimental findings, research understanding, limitations, and a clear next-step story.
- Current June 10 strategy: maintain a good mix of theoretical understanding and practical progress. A live demo is not required; a slideshow with findings, limitations, diagrams, and optionally a short video/GIF of the prototype is sufficient. Show a working contact-evaluation artifact plus research findings/limitations around VACE/Wan/KinEdit-style articulated-object editing and agentic multimodal reasoning.

Current RTSS/lab evaluation status:

- Local 4090 account path: `/mnt/SSD4T_1/aniketh`.
- Lab data path: `/mnt/SSD4T_1/aniketh/original_data`.
- Lab evaluation path: `/mnt/SSD4T_1/aniketh/infiniKin_Evaluation`.
- Local dependency target path on 4090: `/mnt/SSD4T_1/aniketh/envs/rtss_packages`; use `export PYTHONPATH=/mnt/SSD4T_1/aniketh/envs/rtss_packages:$PYTHONPATH`.
- Torch CUDA was fixed by installing `torch==2.5.1+cu121`; CUDA is available on NVIDIA RTX 4090.
- CoTracker downloads/loads through `torch.hub` and `server_manual_window_rtss.py` runs on CUDA.
- Current test case: `case_toilet1_0`, based on `/mnt/SSD4T_1/aniketh/original_data/02_toilet/WAN/toilet1_0.mp4`.
- First RTSS attempt used one full window `0-79`; it ran but failed endpoint visibility: `endpoint_visible_points = 0`, so window-level RTSS was NaN.
- Second RTSS attempt used two windows:
  - `window_00`: frames `0-39`, valid result with `endpoint_visible_points = 112`, `rtss2d_action_p90 = 0.04090580344200134`, `endpoint_motion_p90_norm = 0.10264723747968674`, `cumulative_motion_p90_norm = 0.1926405429840088`, `return_penalty = 1.0`.
  - `window_01`: frames `40-80`, failed endpoint visibility with `endpoint_visible_points = 0`, so RTSS values were NaN.
- Interpretation: RTSS pipeline is working, but annotation/window choice matters. Shorter visibility-consistent windows can produce usable 2D RTSS, while windows where the visible surface disappears produce NaNs.
- Need to inspect/convert annotated videos to verify whether tracked points stay on the correct moving part before trusting metrics.

Key model/paper understanding:

- Wan/Wan2.2: base video generation model family. Practical first target is `TI2V-5B` because it supports text/image-to-video and has lower GPU requirements than A14B models.
- VACE: closer to actual video editing. It uses a unified Video Condition Unit: text prompt + context frame sequence + mask sequence.
- Plain-English VCU understanding: a standardized conditioning package for VACE. It bundles the prompt, existing/partial video frames, and spatiotemporal masks so generation, video editing, masked editing, and task composition can be represented in one format.
- "Temporal" means related to time. In video tensors, the temporal dimension is `T`, the sequence of frames. Temporal consistency means objects, masks, edits, and interactions stay coherent from frame to frame.
- "Spatiotemporal" means space plus time. In video, it refers to where something is in each frame and how that location/shape changes across frames. A spatiotemporal mask has shape like `T x 1 x H x W`.
- Moving video masks matter because video editing needs "where to edit at each time step," not just a static 2D image mask.
- The likely research gap is that general video editing controls may not explicitly preserve contact, occlusion, or articulated object state.
- Contact evaluator understanding: center distance is useful as a diagnostic, but edge distance is a better simple pass/fail metric because contact is a boundary relationship. Future upgrades can use masks, keypoints, depth, or temporal smoothness.
- DeepEyes/DeepEyesV2 are now relevant because they may provide the agentic multimodal bridge: an agent can inspect images, reason with visual evidence, decide where to look/edit, and potentially help connect high-level human intent to masks/prompts/evaluation.
- DeepEyes current understanding: useful as an image/frame-level visual reasoning and localization agent. It may help before editing by locating target hand/object/part boxes and after editing by verifying or critiquing visual failures. It is not assumed to solve full video-temporal reasoning by itself; for video it may need frame-by-frame use plus tracking/temporal logic.
- DeepEyes likely outputs/uses boxes rather than native masks. For mask-conditioned editors, a possible bridge is DeepEyes-style box localization -> SAM/segmentation/tracking -> mask sequence -> VACE/KinEdit-style editor.
- Nano Banana/Gemini-style image editing is a supervisor-mentioned reference point for strong modern image editing. Treat it as benchmark/inspiration, not necessarily a system to reproduce. The research angle is that image editing is becoming strong, but video plus human-articulated-object interaction adds temporal/contact/articulation constraints.
- InfiniKin has known inconsistencies according to mentor discussion; treat it as relevant but not a fully solved path.
- InfiniKin/KinEdit paper understanding:
  - InfiniKin is the synthetic/procedural data engine for indoor articulated-object videos.
  - KinEdit is closer to VACE + articulation fine-tuning + a Kinematic Part Binding (KPB) module.
  - KinEdit input still depends on user-provided text prompt and target part mask; it does not provide the automatic reasoning/object detection/mask-generation loop.
  - Existing metrics in the paper such as FID, FVD, subject consistency, and background consistency do not directly measure articulation quality: attachment, rigidity, correct articulation, localized motion, or leakage.
- DeepEyes lesson for this project: do not make one model do everything. Use an agentic loop: reason, call tools, inspect results, evaluate, and repeat. DeepEyes brings vision + reasoning + tools/agents, but it is mainly a bridge layer rather than the articulated video editor itself.
- FLUX is strong image generation/image editing, not articulated reasoning, kinematics, or human interaction. Treat it as a possible image editing backend/reference, not the core articulated-video solution.
- Wan/VACE is general video generation/editing. KinEdit specializes this direction for articulated-object motion by adding data/fine-tuning/KPB.
- Current most defensible 10-day deliverable: ArticulationEval, an articulation-quality evaluator for generated articulated-object videos. It should prioritize metrics missing from generic video benchmarks: motion leakage, background stability, part rigidity, and attachment consistency.

Near-term next steps:

1. For immediate June 10 progress, prioritize concrete RTSS results and visualizations over expanding scope.
2. Inspect/convert annotated RTSS videos for `case_toilet1_0` and record whether tracking points stay on the correct moving part.
3. Improve or replace the failed second window for `case_toilet1_0`, or choose an easier video/category where the moving surface stays visible.
4. Ask Hang/supervisor/Jiahui for 3-5 generated articulated-object videos plus target masks if available: KinEdit outputs first, then VACE/Wan outputs or paper/example videos as fallback.
5. Build ArticulationEval metrics around available videos, starting with one model/output source first rather than all models:
   - motion leakage score
   - background stability score
   - attachment consistency score
   - part rigidity proxy
6. Use the existing `prototypes/contact-eval/` as the seed, but reframe it from hand-object contact to moving-part/base attachment.
7. Only after one evaluation path works on one source, expand to comparisons such as KinEdit vs VACE vs Wan or good vs bad examples.
8. Create a concise contact-eval/ArticulationEval README and slides that explain inputs, outputs, metric definitions, current limitations, and how it fits into an agentic editing pipeline.
9. DeepEyes should be treated as a parallel/next-layer exploration for automatic reasoning/localization/mask planning. Try running it only if setup is practical and time-boxed; do not let it block ArticulationEval.
10. For June 10, prepare a story: generic metrics miss articulation quality, RTSS gives a rigid-surface tracking baseline, ArticulationEval can add targeted failure-mode metrics, and DeepEyes-style agents can later automate prompt/mask/evaluation loops.

Current RTSS decision protocol:

- Do not spend time adding new metrics until the annotated CoTracker videos are visually checked.
- First artifact to produce/check is `annotated_window_*.mp4` plus `manual_window_scores.csv` from `server_manual_window_rtss.py`.
- If points stay on the intended moving rigid part for a window and endpoint-visible points are enough, keep that window as a valid result.
- If points drift to background, slide onto other object parts, or disappear by the endpoint, re-annotate a shorter visibility-consistent window or skip that segment.
- For `case_toilet1_0`, `window_00` is currently the best candidate to present; `window_01` is evidence that endpoint visibility/occlusion breaks the metric, not evidence that RTSS failed as a pipeline.
- New interpretation for `case_toilet1_0`: frames `0-39` mostly show the toilet lid still, while frames `40-80` contain the actual lid motion. The outer lid surface naturally disappears/reveals the inner surface during opening, so a mask drawn on the outer surface at frame 40 may fail endpoint visibility for physical reasons. For the motion segment, annotate a visible physical patch on the moving lid that persists through the chosen subwindow, likely inner surface/edge/hinge-region after it appears, or split into shorter windows across the visibility change.
- Current local annotation was redone as three windows: `window_00` frames `0-38`, `window_01` frames `39-46`, and `window_02` frames `47-80`. The JSON mask paths should stay relative to the annotation folder as `masks/...` so `server_manual_window_rtss.py` resolves them correctly.
- Three-window RTSS run on the 4090 completed with `--depth_backend none` under `/mnt/SSD4T_1/aniketh/infiniKin_Evaluation/cases/case_toilet1_0/rtss_three_windows`. The console printed `nan` for `rtss_action_p90`, `rtss_rot_deg_p90`, and `rtss_trans_p90` because the script's console summary reports 3D RTSS fields, and no depth backend was used. Check `manual_window_scores.csv` for `rtss2d_action_p90` and endpoint-visible point counts, and inspect the `annotated_window_*.mp4` files before deciding whether the three windows are valid.
- Three-window 2D results: `window_00` frames `0-38` has `endpoint_visible_points=113`, `rtss2d_action_p90=0.02252914011478424`, mostly sanity/still segment; `window_01` frames `39-46` has `endpoint_visible_points=64`, `rtss2d_action_p90=0.07045704126358032`, likely the best current motion result; `window_02` frames `47-80` has `endpoint_visible_points=0`, so endpoint RTSS is NaN and this late-motion window is invalid unless re-split/re-masked around a persistent visible patch.
- Annotated videos written with OpenCV `mp4v` may not play in some viewers; prefer H.264 (`avc1`/`H264`) or transcode existing outputs with ffmpeg/libx264 before copying/viewing. Local `lab_eval_tools/server_manual_window_rtss.py` was patched to try H.264 first and fall back to `mp4v`.
- Hang guidance/interpretation: image and video editing should be treated as interrelated, with image/frame-level reasoning tools such as DeepEyes potentially helping fine-tune or guide an RL/agent model. This does not replace the ArticulationEval direction; it suggests the evaluator can become reward/feedback for an agent that reasons over frames, predicts/localizes keypoints or masks, edits, and then scores results.
- Current articulation-eval framing: RTSS/windowed CoTracker results are useful as local rigidity/trackability evidence, but not sufficient as a whole-video articulation metric because articulated objects self-occlude and reveal new surfaces. A stronger direction is keypoint/part-based evaluation: choose semantically meaningful points on the lid/part, predict where they should move under the intended articulation, compare against tracked/generated positions, and use motion-type-specific metrics such as hinge rotation consistency, sliding/prismatic consistency, attachment consistency, rigidity/deformation, and background leakage.
- Lab evaluation script interpretation: `/mnt/SSD4T_1/aniketh/infiniKin_Evaluation` contains several exploratory scripts. `depth_aware_rigid_motion_consistency01.py` is an older/no-mask automatic LK+MiDaS D-RMC/HRC prototype; it auto-selects moving points and can mix object/background/shadows. `server_drmc_cotracker_from_mask.py` and `server_drmc_cotracker_rtss_vda.py` appear to be more advanced mask/CoTracker/depth variants. The current public repo intentionally keeps only the two most relevant local pieces: `local_window_mask_annotator_v2.py` and `server_manual_window_rtss.py`. For the next project step, build a clean ArticulationEval/keypoint evaluator in the repo rather than copying every lab exploratory script.

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
