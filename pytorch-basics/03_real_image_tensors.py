from pathlib import Path 

import torch 
from PIL import Image
from torchvision import transforms

image_path = Path(f"images/sample.jpg")
image = Image.open(image_path).convert("RGB")

print(f"PIL Image size: {image.size}")
print(f"PIL image mode: {image.mode}")

to_tensor = transforms.ToTensor()
image_tensor = to_tensor(image)

print(f"Tensor shape: {image_tensor.shape}")
print(f"Tensor dtype: {image_tensor.dtype}")
print(f"Tensor min value: {image_tensor.min()}")
print(f"Tensor max value: {image_tensor.max()}")

_, height, width = image_tensor.shape
mask = torch.zeros((1, height, width))

row_start = 50
row_end = 130 
col_start = 90
col_end = 190

mask[:, row_start:row_end, col_start:col_end] = 1.0

masked_tensor = image_tensor * (1 - mask) #image_without_masked_zone = image * (1 - mask)

print(f"Mask shape: {mask.shape}")
print(f"Masked tensor shape: {masked_tensor.shape}")
print(f"mask min/max", mask.min().item(), mask.max().item())

output_dir = Path("outputs")
output_dir.mkdir(exist_ok= True)

to_pil = transforms.ToPILImage()

masked_image = to_pil(masked_tensor)
masked_image.save(output_dir / "masked_sample.jpg")
mask_image = to_pil(mask)
mask_image.save(output_dir / "mask.jpg")

print(f"saved masked image to: {output_dir}/masked_sample.jpg")