# PyTorch Tensors

### 1. Scalars, Vectors, Matrices and Images
- **Scalar**: Rank 0 - [7]
- **Vector**: Rank 1 = [1,2,5] 
- **Matrix**: Rank 2 = [[1,2], [3,4]]
- **Images, Batches, Videos**: Rank 3+



### 2. Images and Batches

#### Images:
- C x H x W (Channel, Height, Width)
- *Channel*: Stands for the RGB Channels, R - 0, G- 1, B - 2; usually only 3 channels present.
- For each channel, intensity of colors is set between `[0.0, 1.0]`
    
    Create zero intensity image using:

    ```py
    image = torch.zeros((c, h, w))
    ```

    In the above we will get `c` number of `h x w` matrices which is all zeroes

    More examples:

    ```py
    image[0, :, :] = 1.0  # red channel - since first channel and all rows and columns
    image[1, 1:3, 2:4] = 0.5  # green-ish patch in the middle

    show("single image: C x H x W", image) #prints out the three grids -> for r,g,b with the intensity/activation values, and the shape of the tensor
    ```

#### Batches
- Basically multiple images in one (the only added parameter is the no. of images - `B`)
- `B x C x H x W` (Batch, Channel, Height, Width)
- Similar slicing and accessing operation as images
- Added info:
    ```py
    batch = torch.stack([image, torch.ones_like(image) * 0.25]) #
    # batch of 2 images: one is the original image, the other retains the same "shape" but has all values one multiplied by 0.25 (Q: would this be a gray image with 25% intensity? Ans: yes, since all channels have the same value, it would be a shade of gray)

    show("batch of images: B x C x H x W", batch)
    show("first image", batch[0]) #batch[0] --> first image with all channels, rows and columns
    show("green channel of first image", batch[0, 1]) #first image, green channel, all rows and columns (no slicing used so we get the whole channel)
    show("top row across batch", batch[:, :, 0]) #all images, all channels, top row (row 0), all columns (no slicing used so we get the whole row) -> shape B x C x W
    ```

### 3. Masking
- Basically the stencil for what we want to keep / remove
- Has only one channel since we are choosing what pixel to keep/remove from entire image and not individual RGB Channel
- Note: `masked_image = image * (1 - mask)`
(Mask is essentially the same data structure as image, but usually has 1 channel instead of 3)

    ```py
    mask = torch.zeros((1, 4, 5))
    mask[:, 1:3, 2:4] = 1.0
    masked_image = image * (1 - mask)
    ```

- `mask.shape = (1, 4, 5)` means:
    - 1 channel
    - 4 rows
    - 5 columns
- Mask values:
    - `0` = keep this pixel
    - `1` = edit/remove this pixel
- Why `1 - mask`?
    - If mask is `0`: `1 - 0 = 1`, so multiplying keeps the pixel
    - If mask is `1`: `1 - 1 = 0`, so multiplying removes/zeros the pixel
- The mask is not masking only red/green/blue. It masks a spatial region and gets applied to all RGB channels through broadcasting.
- In real object removal:
    - image = original image tensor
    - mask = object location / edit region
    - inpainting model = fills the masked region with plausible background


### 4. Broadcasting
- Broadcasting is when PyTorch lets tensors with compatible shapes work together by reusing/stretching dimensions of size `1`.
- Example from file:

    ```py
    mean_per_channel = image.mean(dim=(1, 2), keepdim=True)
    centered_image = image - mean_per_channel
    ```

- `image.shape = (3, 4, 5)` means:
    - 3 channels
    - 4 rows
    - 5 columns
- `dim=(1, 2)` means average over height and width, but not over channels.
- So it calculates:
    - mean of red channel
    - mean of green channel
    - mean of blue channel
- `keepdim=True` keeps the averaged dimensions as size `1`.
    - With `keepdim=True`: shape is `(3, 1, 1)`
    - Without it: shape would be `(3,)`
- Why keep dimensions?
    - `(3, 1, 1)` can broadcast against `(3, 4, 5)`
    - The channel dimension matches: `3` with `3`
    - The height/width dimensions stretch: `1 -> 4`, `1 -> 5`
- Beginner broadcasting rule:
    - matching dimensions are okay
    - dimension `1` can stretch
    - different non-1 dimensions fail
- Broadcasting is used a lot for:
    - applying masks: `(3, H, W) * (1, H, W)`
    - normalizing images: `(image - mean) / std`
    - applying one scalar/value across a big tensor


### 5. Matrix Multiplication
- Matrix multiplication is the core operation behind a linear neural network layer.
- Main formula:

    ```py
    output = input @ weights + bias
    ```

- In the file:

    ```py
    inputs.shape  # (2, 3)
    weights.shape # (3, 2)
    bias.shape    # (2,)
    ```

- Read `inputs.shape = (2, 3)` as:
    - 2 examples in the batch
    - 3 input features per example
- Read `weights.shape = (3, 2)` as:
    - 3 input features
    - 2 output features / neurons
- Read `bias.shape = (2,)` as:
    - one bias for each output neuron
- Shape rule:

    ```txt
    (A, B) @ (B, C) = (A, C)
    ```

- In this example:

    ```txt
    (2, 3) @ (3, 2) = (2, 2)
    ```

- The middle dimensions must match.
- The result is `(2, 2)`:
    - 2 input examples
    - 2 output values for each example
- `bias` gets broadcast across the batch:

    ```py
    inputs @ weights      # shape (2, 2)
    bias                  # shape (2,)
    inputs @ weights+bias # bias added to each row
    ```

- This manual formula is basically what `torch.nn.Linear(3, 2)` does internally.
- Connection to neural networks:
    - input features go in
    - weights decide how strongly each feature matters
    - bias shifts the result
    - output values are activations


### 6. Autograd
- Autograd is PyTorch's automatic gradient engine.
- It tracks operations on tensors with `requires_grad=True`.
- It lets PyTorch calculate derivatives for us instead of doing all calculus manually.

    ```py
    x = torch.tensor([2.0], requires_grad=True)
    y = (x - 5) ** 2
    y.backward()
    ```

- `requires_grad=True` means:
    - watch this tensor
    - track math done with it
    - allow gradient calculation later
- `y = (x - 5) ** 2`
    - when `x = 2`, `y = 9`
    - smallest value is when `x = 5`
- `y.backward()` means:
    - calculate gradient of `y` with respect to tracked tensors
    - here, calculate `dy/dx`
- Derivative:

    ```txt
    y = (x - 5)^2
    dy/dx = 2(x - 5)
    ```

- At `x = 2`:

    ```txt
    dy/dx = 2(2 - 5) = -6
    ```

- PyTorch stores this in:

    ```py
    x.grad
    ```

- What the gradient means:
    - negative gradient means increasing `x` would reduce the loss
    - positive gradient means decreasing `x` would reduce the loss
- Gradient descent idea:

    ```py
    x = x - learning_rate * gradient
    ```

- In a real neural network:

    ```py
    prediction = model(input)
    loss = loss_fn(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ```

- Meaning:
    - forward pass: model computes prediction/loss
    - backward pass: autograd computes gradients
    - optimizer step: weights are updated

