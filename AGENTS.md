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
- `prototypes/articulation_eval/keypoint_articulation_eval.py`: runnable semantic keypoint evaluator. It reads manual keypoint JSON and writes frame-level CSV plus summary JSON for visibility, pivot drift, front-tip motion, radius consistency, pairwise lid rigidity, and approximate angle change.
- `prototypes/articulation_eval/manual_keypoint_annotator.py`: OpenCV click annotator for marking semantic 2D keypoints on selected video frames before later 2D/3D articulation scoring.
- `prototypes/articulation_eval/examples/toilet_lid_keypoints.example.json`: toy public-safe example for testing the keypoint evaluator without private video annotations.
- `prototypes/articulation_eval/presentation_outline.md`: concise meeting story for RTSS baseline -> semantic keypoint ArticulationEval.
- `prototypes/articulation_eval/synthetic_articulation_sandbox.py`: dependency-light synthetic keypoint sandbox for validating the unified articulation score idea on controlled hinge, slider, and rotation trajectories before using noisy real videos.
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
- Hang's "3 axes/origin" framing can guide the next evaluator: represent a moving articulated part with a local coordinate frame anchored at an origin, such as the hinge. Track semantic keypoints relative to that origin/frame over time. For hinge motion, the key check is whether points preserve distances to the hinge axis and move along rotation-like arcs; for drawer/sliding motion, points should move mostly along one translation axis while preserving same-part distances.
- Immediate ArticulationEval v1 target: use the same `case_toilet1_0` toilet-lid video and manually establish semantic keypoints on a few frames first, rather than solving automatic keypoint detection. Start with hinge points and lid front/edge points, then compute 2D hinge metrics before adding tracking or DeepEyes-assisted localization.
- Keypoint evaluator assumption: begin with manually clicked 2D keypoints as the observed generated-video points. Depth-based lifting can be added later as pseudo-3D, but generated videos may have inconsistent monocular depth, so depth should be treated as an optional approximation, not ground truth. The evaluator should distinguish observed keypoints from expected keypoints predicted by a simple articulation model, e.g. hinge rotation preserving distances to a hinge axis.
- Manual marking workflow: use `prototypes/articulation_eval/manual_keypoint_annotator.py` on `case_toilet1_0/input.mp4`. Current default keypoints are `left_pivot`, `right_pivot`, `lid_back_center`, `lid_midpoint`, and `lid_front_tip`, with frames such as `0,52,60,70,80`. The evaluator derives `back_pivot` from the midpoint of the two pivots when visible. Save to `prototypes/articulation_eval/examples/toilet_lid_keypoints.json`; the local video/case remains ignored by git.
- ArticulationEval v1 now runs on a toy example and outputs metrics under ignored `prototypes/articulation_eval/outputs/`. Verified command: `python prototypes/articulation_eval/keypoint_articulation_eval.py --keypoints_json prototypes/articulation_eval/examples/toilet_lid_keypoints.example.json --out_csv prototypes/articulation_eval/outputs/toy_toilet_lid_metrics.csv --summary_json prototypes/articulation_eval/outputs/toy_toilet_lid_summary.json`.
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

June 9 presentation state:

- Current talk framing: ArticulationEval is a verification/filtering prototype for generated human-articulated-object interaction videos, not a generator. It turns qualitative failures like "the lid looks rubbery" into measurable signals that can be used for dataset filtering, candidate ranking, self-verification, or future RL/agent reward.
- Professor Cheng reportedly expects vivid/visual/product-like results. Best presentation posture: show a mini verifier product rather than just a report:
  1. RTSS baseline videos/table.
  2. RTSS limitation: 120 generic surface points work only on visibility-consistent patches; points disappear when the toilet lid surface changes from outer to inner surface or visible area changes.
  3. Semantic keypoint annotation frames.
  4. ArticulationEval metrics/plots.
  5. Cross-model comparison on toilet `1_0`.
  6. Pipeline framing: generated candidates -> visual/keypoint localization -> ArticulationEval -> PASS/REVIEW/FAIL -> dataset filtering/refinement.
- RTSS workflow from server README: local `local_window_mask_annotator_v2.py` defines visibility-consistent windows and masks, case is transferred to server, `server_manual_window_rtss.py` samples/tracks 120 points per window, then `manual_window_scores.csv`, `manual_window_summary.csv`, and `annotated_*.mp4` are inspected. Important rule: inspect annotated tracking videos before trusting CSV metrics.
- RTSS result table for `case_toilet1_0`:
  - `window_00`, frames `0-38`: `endpoint_visible_points=113`, `rtss2d_action_p90=0.022529`, mostly still/sanity window.
  - `window_01`, frames `39-46`: `endpoint_visible_points=64`, `rtss2d_action_p90=0.070457`, valid early motion.
  - `window_02`, frames `47-80`: `endpoint_visible_points=0`, endpoint RTSS NaN because surface visibility changes during opening.
- Semantic keypoint setup evolved to current keypoints: `left_pivot`, `right_pivot`, `lid_back_center`, `lid_midpoint`, `lid_front_tip`. The evaluator derives `back_pivot` as midpoint of visible left/right pivots. Frames should be motion-phase normalized when videos differ in length.
- Current evaluator files:
  - `prototypes/articulation_eval/manual_keypoint_annotator.py`: click annotator.
  - `prototypes/articulation_eval/keypoint_articulation_eval.py`: computes metrics and visual outputs.
  - `prototypes/articulation_eval/make_presentation_assets.py`: SVG plots and result summary.
  - `prototypes/articulation_eval/make_concept_diagrams.py`: PhysX-Omni metric map and DeepEyes/ArticulationEval loop diagrams.
  - `prototypes/articulation_eval/build_demo_report.py`: local static HTML report.
  - `prototypes/articulation_eval/make_eval_overlay_video.py`: full-video overlay demo using interpolated keypoints; it is a visual demo, not a tracking algorithm.
- Current metrics and interpretation:
  - Base WAN / original toilet: visibility `0.92`, max pivot drift `6.96 px`, max lid-tip motion `312.87 px`, mean radius error `1.13`, mean rigidity error `0.61`, angle change `98.55 deg`. Interpretation: lid opens and pivot is fairly stable, but high radius/rigidity errors suggest non-rigid or non-hinge-consistent generated motion.
  - WAN_Finetune: visibility `0.92`, max pivot drift `7.21 px`, max lid-tip motion `275.94 px`, mean radius error `0.50`, mean rigidity error `0.23`, angle change `109.73 deg`. Interpretation: finetuning improves shape preservation and hinge-like rigidity compared with base WAN.
  - VACE: visibility `0.80`, max pivot drift `5.50 px`, max lid-tip motion `294.26 px`, mean radius error `1.03`, mean rigidity error `0.15`, angle change `112.50 deg`. Interpretation: visually clean opening and best rigidity score, but lower visibility / high radius error partly caused by ambiguity around the moving lid back edge and visible moving-part extent.
  - VACE_Finetune: visibility `0.88`, max pivot drift `13.34 px`, max lid-tip motion `294.33 px`, mean radius error `1.11`, mean rigidity error `0.22`, angle change `110.75 deg`. Interpretation: visually very similar to VACE; small differences likely reflect manual annotation ambiguity/noise rather than reliable model separation.
- Cross-model takeaway: WAN_Finetune is clearly better than base WAN by radius/rigidity metrics; VACE and VACE_Finetune are both visually strong and metric differences are likely within manual annotation noise. A careful slide should say the evaluator agrees with the broad visual trend but manual annotation variance limits fine-grained ranking.
- Important limitation to state: semantic keypoint metrics depend on consistent part definitions. When the generated model changes apparent moving-part extent, especially the lid back edge, visibility/radius metrics may reflect annotation ambiguity rather than true physical failure.
- Prompt-to-metric framing for toilet opening:
  - Ideal prompt: "Open the toilet lid smoothly around its rear hinge while preserving lid rigidity, hinge attachment, and a static toilet base."
  - Rear hinge rotation -> angle change + radius consistency.
  - Rigid lid -> rigidity error.
  - Stable base/hinge -> pivot drift.
  - Visible/localized part -> visibility and visual/keypoint checks.
- End goal framing: automatic verification and refinement pipeline for generated interaction videos:
  generated candidates -> agent/vision model localizes object parts/keypoints/masks/contact -> ArticulationEval scores motion quality -> filter/rank/reject/refine -> keep high-quality samples for dataset construction and use failures for generation refinement or RL reward.
- DeepEyes guidance from Hang on June 8: he said it would be great to show intermediate visual features like keypoints. Next direction is to run/train DeepEyes (ICLR'26 paper) and understand research progress about agentic training. He is training DeepEyesV2 with SFT under standard configuration; 6-hour LoRA SFT has not improved original Qwen2.5-VL-7B yet, and he is continuing.
- DeepEyes interpretation for presentation/reply: Do not claim DeepEyes is integrated. Say current work is the verification layer; DeepEyes-style agent can become the localization/planning layer that outputs keypoints/masks, while ArticulationEval provides self-verification/reward/filtering. Full DeepEyes training is heavy (multi-GPU/Ray/Qwen judge etc.), so immediate next step is to run/evaluate the provided setup or study the repo and identify the adapter point from visual agent outputs to ArticulationEval inputs.
- PhysX-Omni benchmark note: useful related framing for metric-first evaluation. It includes metrics such as RQS, MCS, DCS, DQS, APS, KPS, and MPS. KPS (Kinematic Plausibility Score) is closest conceptually. ArticulationEval can be framed as a lightweight video-interaction counterpart focused on generated human-articulated-object videos with keypoint evidence.
- Candidate filtering/PASS-REVIEW-FAIL idea: It is simple weighted scoring, not ML, but still useful as a verification component. If used, present as a prototype Articulation Quality Score, not a final benchmark.

June 9 presentation feedback:

- Presentation went alright. Professor Cheng emphasized that future work needs specific numbers and a clear end goal, not vague descriptors.
- He wants a formula/metric direction that can work across different articulated motion types. The key challenge is whether different motions (hinge rotation, drawer sliding/prismatic motion, knob rotation, etc.) can share a standard unified score rather than requiring unrelated metrics for each category.
- Next research target should be a unified Articulation Quality Score with motion-type-specific expected transforms underneath. The score can be standard even if the motion model differs:
  - infer/define moving part, base/anchor, and expected degree(s) of freedom;
  - estimate observed keypoint/track motion;
  - fit the best valid articulation transform for that motion type;
  - score residuals normalized by object scale;
  - combine with visibility, attachment/base stability, rigidity, and motion localization into one numeric score.
- A possible end goal: a general verifier for generated articulated-object videos that outputs one interpretable score and failure breakdown:
  `AQS = 100 - penalties(visibility, rigidity residual, articulation-model residual, attachment drift, motion leakage, temporal smoothness)`.
- Continue DeepEyesV2 setup in parallel. DeepEyes can eventually automate part/keypoint/mask localization, while the unified score is the verification target/reward.
- June 23 next-step decision: first validate the scoring math on synthetic keypoints rather than real generated video. Synthetic keypoints are generated by script, not manually annotated, and are used to separate two unknowns: whether the formula behaves correctly vs whether real-video keypoints/tracking are noisy. The current sandbox writes ignored outputs under `prototypes/articulation_eval/outputs/synthetic_articulation_sandbox/`.
- Current unified-score prototype: `synthetic_articulation_sandbox.py` generates perfect and broken cases for hinge, slider, and simple rotation, then scores each case with shared components:
  - `model_fit_error`: motion-type-specific residual. Hinge/rotation preserve radius around pivot/center; slider translates along one global axis.
  - `rigidity_error`: moving-part pairwise distances should stay stable.
  - `anchor_drift_error`: hinge/center anchors should stay stable when applicable.
  - `total_error = model_fit_error + rigidity_error + anchor_drift_error`.
  - `articulation_score = 100 * exp(-4 * total_error)`.
- Verified sandbox output pattern: perfect hinge/slider/rotation score `100.0`; injected failures score lower, e.g. hinge pivot drift around `42`, hinge deformation around `33`, slider off-axis wobble around `41`, slider deformation around `54`, rotation drift/deformation around `59-60`. This is only a controlled validation step, not the final metric.
- How to use the synthetic scores: treat them as a sanity check, not a final result. The current claim is only that the normalized residual idea distinguishes clean low-DOF articulation from injected failures under controlled keypoints. The breakdown is more important than the final score: hinge pivot drift gives high `anchor_drift_error`, hinge deformation gives high model/rigidity errors, slider wobble gives high slider model error, etc. Next step is to compare this breakdown against real toilet/video metrics and then refine the formula with literature and real annotations.
- Current motion-type scope: v1 should focus on rigid articulated parts: hinge/pivot objects (toilet lid, cabinet/oven/microwave door), drawer/slider/prismatic motion, and simple rotation/knob-like motion. Flexible or deformable objects such as bendable lamp necks are future work because shape change may be valid rather than a failure.
- Papers/knowledge direction for the unified score: do not begin by copying a complex formula. Start from geometry and model-fit residuals, then read papers to refine terminology/model fitting. Useful areas: revolute/prismatic robot kinematics, PartNet-Mobility style joint definitions, articulated-object model estimation from trajectories/point clouds, and later screw theory / product-of-exponentials if a more formal unified language is needed.

Hang June 23/24 articulated reconstruction thread:

- Hang said he is investigating reconstructing articulated objects from a single-view scene image. His first step is training an autoregressive model that sequentially reconstructs latent codes for each object part, then reconstructs each part's 3D mesh using a TRELLIS2 prior. Later he plans to explore RL for performance improvement and SFT with paired data from InfiniKin; he suggested pulling the git repo if time permits.
- Interpretation: this is adjacent to, but not identical with, the current articulation-scoring work. Hang's thread is about single-image 3D part/mesh reconstruction; current ArticulationEval work is about evaluating articulated motion quality from keypoints/video. The conceptual overlap is part-level articulated object structure and possible future joint/plausibility evaluation, but do not force the metric work into his pipeline prematurely.
- Safer framing if discussing with Hang: first understand the AR latent-code + TRELLIS2 reconstruction pipeline. Possible future bridge: after reconstructed parts/joints exist, an articulation plausibility score could validate whether the parts support plausible hinge/slider/rotation behavior or serve as a later reward/diagnostic. Treat that as a potential bridge, not the immediate main task.

DeepEyesV2 setup/debug status:

- Downloaded checkpoint target: `honglyhly/DeepEyesV2_7B_1031`.
- Rorqual checkpoint path used: `/project/rrg-vislearn/kini1409/models/deepeyes/DeepEyesV2_7B_1031`.
- Existing `image-editing` env became unreliable for DeepEyes inference. `transformers==5.10.2` was incompatible with `torch==2.4.1` because Transformers tried to access `torch.float8_e8m0fnu`. Downgrading to `transformers==4.51.3` fixed that specific mismatch, but later `import torch` appeared to hang on the compute node in this env.
- Rorqual interactive debugging lessons:
  - Run DeepEyes only on a GPU allocation where `nvidia-smi` works.
  - Login nodes such as `rorqual1/3/4` do not expose the GPU.
  - Use `python -u ... 2>&1 | tee deepeyes_infer.log` for live logs.
  - Use `sbatch` for long runs instead of babysitting `salloc`.
  - If `nvidia-smi` shows `0MiB` for many minutes and log is empty, the script has not reached GPU/model placement.
- Added cluster helper scripts under `prototypes/articulation_eval/cluster/`:
  - `setup_deepeyes_env.sh`: creates clean `/project/rrg-vislearn/kini1409/envs/deepeyes` and installs a DeepEyes/Qwen-VL inference stack.
  - `test_deepeyes_imports.py`: prints import/CUDA progress.
  - `test_deepeyes_infer.py`: runs one toilet-frame inference with progress logs.
  - `run_deepeyes_infer.sbatch`: 2-hour GPU batch job with logs under `/project/rrg-vislearn/kini1409/logs/`.
- Next DeepEyes action: copy/sync the repo scripts to Rorqual, run `setup_deepeyes_env.sh` inside `tmux`, then submit `run_deepeyes_infer.sbatch`. Keep the manual keypoint annotations as the future answer key for DeepEyes localization validation.
- Vulcan worked much better than Rorqual for DeepEyes inference. Vulcan project path used: `~/projects/aip-vislearn/kini1409`; env: `envs/deepeyes-vulcan`; model path: `models/deepeyes/DeepEyesV2_7B_1031`. Inside a Vulcan GPU allocation, `torch 2.12.0` reported CUDA true on an NVIDIA L40S and the model loaded successfully.
- DeepEyesV2 inference result: on a toilet keypoint-overlay frame, the model loaded all 7 checkpoint shards in about 39 seconds, recognized labels such as `lid_back_center`, `hinge_midpoint`, and `lid_front_tip`, and explained that the hinge midpoint is the pivot region. Important caveat: this only shows the model can interpret an already-labeled overlay frame; it does not prove automatic keypoint discovery from raw video.
- DeepEyes validation mini-set: user built about 27 images / 31 questions across small text, counting, object-text, spatial, chart/table, and confusion-matrix tasks. Results comparing DeepEyesV2 final vs SFT:
  - Overall: final `22/31` (`71.0%`), SFT `21/31` (`67.7%`).
  - Small text: both `9/10`.
  - Counting: final `2/5`, SFT `1/5`.
  - Object-text: final `3/3`, SFT `2/3`.
  - Spatial: both `5/5`.
  - Chart/table: final `3/7`, SFT `4/7`.
  - Confusion matrix: both `0/1`.
- DeepEyes validation interpretation: final model narrowly wins overall, SFT is better on chart/table reading, both struggle with counting and confusion-matrix questions. Dataset is small, so use this only as a preliminary model sanity check and error-analysis seed, not a strong benchmark claim.
- DeepEyes relationship to current work: keep it as the future automation/perception layer. Manual keypoints and synthetic keypoints are the answer key / scoring validation; DeepEyes or another agent could later localize keypoints/masks automatically, and ArticulationEval/unified score could act as verifier or reward.
