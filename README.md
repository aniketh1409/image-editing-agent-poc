# Image Editing Agent POC

Early research/learning repo for building AI agents for image editing.

Current public focus:

- PyTorch fundamentals
- image tensors and masks
- simple mask-based editing tools
- preparing for future inpainting/diffusion and agent workflows

Existing beginner/editor work now lives under `foundation/` so the repo root can stay clean for research notes and later prototypes.

## Simple Editor

`foundation/scripts/simple_editor.py` is a small command-line image editing POC.

It takes:

```text
image + mask + edit mode -> edited image
```

Supported modes:

- `black`: fills the masked region with black
- `average`: fills the masked region with the average image color
- `blur`: fills the masked region with a blurred version of the image
- `soft-blur`: uses a softened mask edge for smoother blur blending

Example:

```powershell
.\.venv\Scripts\python.exe foundation\scripts\simple_editor.py --image foundation\images\sample.jpg --mask foundation\outputs\mask.jpg --mode soft-blur --output foundation\outputs\cli_soft_blur.jpg
```

## Learning Files

- `foundation/pytorch-basics/01_tensors.py`: tensor basics, image tensors, masks, broadcasting, matrix multiplication, autograd
- `foundation/pytorch-basics/02_training_loop.py`: first PyTorch training loop
- `foundation/pytorch-basics/03_real_image_tensors.py`: loading real images, masks, fills, blending
- `foundation/pytorch-basics/04_augmentations.py`: basic image augmentations

## Direction

This POC currently uses classical image processing. The next step is to add public research notes on Wan/VACE and start video-tensor exercises before running larger video editing baselines on the cluster.
