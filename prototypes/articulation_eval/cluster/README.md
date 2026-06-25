# DeepEyesV2 Rorqual Inference Helpers

These scripts are for running the released DeepEyesV2 checkpoint on Rorqual
without hand-typing fragile commands.

Expected cluster paths:

- Project root: `/project/rrg-vislearn/kini1409`
- Clean env: `/project/rrg-vislearn/kini1409/envs/deepeyes`
- Checkpoint: `/project/rrg-vislearn/kini1409/models/deepeyes/DeepEyesV2_7B_1031`
- Test image: `/project/rrg-vislearn/kini1409/toilet_frame.png`

## Copy To Rorqual

From local PowerShell:

```powershell
scp -r prototypes/articulation_eval/cluster kini1409@rorqual.alliancecan.ca:/project/rrg-vislearn/kini1409/repos/image-editing-agent-poc/prototypes/articulation_eval/
```

## Setup Env In tmux

On Rorqual:

```bash
tmux new -s deepeyes_setup
cd /project/rrg-vislearn/kini1409/repos/image-editing-agent-poc/prototypes/articulation_eval/cluster
bash setup_deepeyes_env.sh
```

Detach with `Ctrl+b`, then `d`.

## Submit Inference Job

```bash
cd /project/rrg-vislearn/kini1409/repos/image-editing-agent-poc/prototypes/articulation_eval/cluster
sbatch run_deepeyes_infer.sbatch
```

Watch logs:

```bash
squeue -u $USER
tail -f /project/rrg-vislearn/kini1409/logs/deepeyes_infer_<JOBID>.out
```

If the job fails, inspect both `.out` and `.err` logs.
