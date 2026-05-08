# Agent Notes

Project goal: build AI agent prototypes for image editing using deep learning and generative models, especially object removal and spatially aware edits.

Current learning track:

1. PyTorch fundamentals: tensors, shapes, indexing, broadcasting, autograd.
2. Neural network basics in PyTorch: `torch.nn.Module`, losses, optimizers, training loops.
3. Vision basics: image tensors, transforms, masks, convolutions, simple segmentation.
4. Generative editing: inpainting, diffusion pipelines, mask-conditioned generation.
5. Agent pipeline: interpret edit intent, choose/edit masks or regions, call model, evaluate/refine output.

Tutoring style:

- Keep examples small, runnable, and tied to image-editing tasks.
- Emphasize tensor shapes constantly.
- Prefer hands-on scripts under `pytorch-basics/` before larger prototypes.
- Use CPU-friendly examples unless the environment changes.
