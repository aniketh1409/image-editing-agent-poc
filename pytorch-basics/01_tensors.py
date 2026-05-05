"""PyTorch tensor basics for image-editing ML work.

Run:
    .venv\\Scripts\\python.exe pytorch-basics\\01_tensors.py

Why tensors matter for your internship goal:
    Images, masks, text embeddings, model weights, and diffusion noise are all
    represented as tensors. If tensor shapes feel natural, the rest of vision ML
    becomes much less mysterious.
"""

import torch


def show(name: str, tensor: torch.Tensor) -> None:
    """Print the most useful debugging facts for a tensor."""
    print(f"\n{name}")
    print(tensor)
    print(f"shape: {tuple(tensor.shape)} | dtype: {tensor.dtype} | device: {tensor.device}")


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ---------------------------------------------------------------------------
# 1. Scalars, vectors, matrices, and batches
# ---------------------------------------------------------------------------
# A tensor is a typed, rectangular block of numbers.
# Rank 0: scalar       -> one number
# Rank 1: vector       -> one list of numbers
# Rank 2: matrix       -> rows and columns
# Rank 3+: common in ML -> images, batches, videos, embeddings, etc.

scalar = torch.tensor(7)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

show("scalar", scalar)
show("vector", vector)
show("matrix", matrix)

# ---------------------------------------------------------------------------
# 2. Image tensors
# ---------------------------------------------------------------------------
# In PyTorch vision code, a single RGB image is usually shaped:
#     channels, height, width
# For example: 3 x 4 x 5 means 3 color channels, 4 rows, 5 columns.

image = torch.zeros((3, 4, 5))
image[0, :, :] = 1.0  # red channel
image[1, 1:3, 2:4] = 0.5  # green-ish patch in the middle

show("single image: C x H x W", image)

# A batch of images usually adds a leading batch dimension:
#     batch, channels, height, width

batch = torch.stack([image, torch.ones_like(image) * 0.25])
show("batch of images: B x C x H x W", batch)

# Shape reading drill:
# batch[0]        -> first image, shape C x H x W
# batch[0, 1]     -> first image's green channel, shape H x W
# batch[:, :, 0]  -> top row from every image/channel, shape B x C x W

show("first image", batch[0])
show("green channel of first image", batch[0, 1])
show("top row across batch", batch[:, :, 0])

# ---------------------------------------------------------------------------
# 3. Masks for object removal
# ---------------------------------------------------------------------------
# Object removal often starts with a mask: 1 where the object is, 0 elsewhere.
# A model then has to fill/inpaint the masked area using surrounding context.

mask = torch.zeros((1, 4, 5))
mask[:, 1:3, 2:4] = 1.0
masked_image = image * (1 - mask)

show("mask: 1 means edit this region", mask)
show("image with masked region removed", masked_image)

# ---------------------------------------------------------------------------
# 4. Broadcasting
# ---------------------------------------------------------------------------
# Broadcasting lets PyTorch combine tensors with compatible shapes.
# Here mean_per_channel has shape C x 1 x 1, so it can be subtracted from every
# pixel in each corresponding channel.

mean_per_channel = image.mean(dim=(1, 2), keepdim=True)
centered_image = image - mean_per_channel

show("mean per channel: C x 1 x 1", mean_per_channel)
show("centered image", centered_image)

# ---------------------------------------------------------------------------
# 5. Matrix multiplication: the heart of neural network layers
# ---------------------------------------------------------------------------
# A linear layer computes:
#     output = input @ weights + bias
# This is the same idea as the 3Blue1Brown neuron story, just batched.

inputs = torch.tensor(
    [
        [0.2, 0.7, 0.1],
        [0.9, 0.1, 0.3],
    ]
)
weights = torch.tensor(
    [
        [0.5, -0.3],
        [0.8, 0.2],
        [-0.1, 0.4],
    ]
)
bias = torch.tensor([0.1, -0.2])

activations = inputs @ weights + bias
show("linear activations: batch x output_features", activations)

# ---------------------------------------------------------------------------
# 6. Autograd: letting PyTorch compute gradients
# ---------------------------------------------------------------------------
# Gradients tell us how to change weights to reduce a loss.

x = torch.tensor([2.0], requires_grad=True)
y = (x - 5) ** 2
y.backward()

show("x", x)
show("loss y = (x - 5)^2", y)
print("gradient dy/dx at x=2:", x.grad.item())

# ---------------------------------------------------------------------------
# 7. Tiny exercises
# ---------------------------------------------------------------------------
# Try these by editing the code above:
# 1. Change the mask region. Can you remove a different rectangle?
# 2. Make the fake image shape 3 x 8 x 8. What breaks? What still works?
# 3. Create a batch of 4 images instead of 2.
# 4. Add torch.sigmoid(activations). What range are the outputs in?
# 5. Change x from 2 to 10. What happens to the gradient?

print("\nDone. Next: build one tiny neural net layer using torch.nn.")

