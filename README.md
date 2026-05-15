# Image Editing Agent POC

Early research/learning repo for building AI agents for image editing.

Current focus:

- PyTorch fundamentals
- image tensors and masks
- simple mask-based editing tools
- preparing for future inpainting/diffusion and agent workflows

## Simple Editor

`scripts/simple_editor.py` is a small command-line image editing POC.

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
.\.venv\Scripts\python.exe scripts\simple_editor.py --image images\sample.jpg --mask outputs\mask.jpg --mode soft-blur --output outputs\cli_soft_blur.jpg
```

## Learning Files

- `pytorch-basics/01_tensors.py`: tensor basics, image tensors, masks, broadcasting, matrix multiplication, autograd
- `pytorch-basics/02_training_loop.py`: first PyTorch training loop
- `pytorch-basics/03_real_image_tensors.py`: loading real images, masks, fills, blending
- `pytorch-basics/04_augmentations.py`: basic image augmentations

## Direction

This POC currently uses classical image processing. The next step is to replace the fill operation with pretrained inpainting/diffusion models, then wrap the editing tools in a simple agent pipeline.
