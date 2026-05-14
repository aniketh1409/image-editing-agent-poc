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

keep_region = 1 - mask

sum_per_channel = (image_tensor * keep_region).sum(dim=(1,2), keepdim = True) #keep 3 diff sums for red, green and blue since we only selected 1,2 dim
count_per_channel = keep_region.sum(dim = (1,2), keepdim = True)

average_color = sum_per_channel / count_per_channel

filled_tensor = image_tensor * keep_region + average_color * mask
filled_image = to_pil(filled_tensor)
filled_image.save(output_dir / "average_filled_sample.jpg")

print("average color:", average_color.flatten())
print("saved average filled image to:", output_dir / "average_filled_sample.jpg")

blur = transforms.GaussianBlur(kernel_size = 31, sigma = 10) 
blurred_tensor = blur(image_tensor)
#Gaussian blur is a speciric type of blur -> replaces each pixel with a weighted average of nearby pixels
#i.e nearby pixels matter more, far away matter less (lower weight)
#kernel size --> 31 x 31 neighbourhood around each pixel (usually odd numbers assigned to kernel size)
#sigma --> controls how spread out the gaussian weighting is (smaller = weights concentrated near centre -> less blur; bigger = weights spread further out --> more blur)

blur_filled_tensor = image_tensor * keep_region + blurred_tensor * mask

blur_filled_image = to_pil(blur_filled_tensor)
blur_filled_image.save(output_dir/"blur_filled_sample.jpg")
print(f"saved blur filled image to: {output_dir}/blur_filled_sample.jpg")

#__CROP TENSOR__

crop_tensor = image_tensor[:, 40:140, 70:200]

print("crop shape:", crop_tensor.shape)

crop_image = to_pil(crop_tensor)
crop_image.save(output_dir / "crop_sample.jpg")

print("saved crop image to:", output_dir / "crop_sample.jpg")

#__RESIZING TENSOR__

resize = transforms.Resize((256, 256)) #resize using transforms.Resize(m, n)

resized_tensor = resize(image_tensor) # use the resize object with image_tensor to create resized tensor

print("resized shape:", resized_tensor.shape)

resized_image = to_pil(resized_tensor)
resized_image.save(output_dir / "resized_sample.jpg")

print("saved resized image to:", output_dir / "resized_sample.jpg")

#batching real images 

single_batch = image_tensor.unsqueeze(0)

print(f"single image tensor shape: {image_tensor.shape}")
print(f"single batch shape: {single_batch.shape}")


#first create 3 different types/versions of the same image --> they need to have the same shape (b c h w)
#hence use resized versions so shapes match
resized_original = resize(image_tensor)
resized_masked = resize(masked_tensor)
resized_blur_filled = resize(blur_filled_tensor)

image_batch = torch.stack([
    resized_original,
    resized_masked,
    resized_blur_filled
])

print(f"image batch shape {image_batch.shape}")

#Normalization

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) #mean shape is [3, 1, 1] -> it broadcasts across image_tensor.shape
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#where do we get these mean and std values from? --> they are commonly used values for normalizing images in computer vision tasks, 
# often derived from the ImageNet dataset

normalized_tensor = (image_tensor - mean) / std

print(f"normalized tensor shape: {normalized_tensor.shape}")
print(f"normalized tensor min value: {normalized_tensor.min()}")
print(f"normalized tensor max value: {normalized_tensor.max()}")
print(f"normalized tensor mean-ish value: {normalized_tensor.mean(dim =(1,2))}") #mean across height and width dimensions for each channel
#do we need the dim(1,2) here? --> yes because we want to calculate the mean for each channel separately across the height and width dimensions,
#what if we didn't specify dim(1,2)? --> it would calculate the mean across all dimensions, giving us a single mean value for the entire tensor instead of separate means for each channel.

#imporant: usually o not save normalized tensors directly as images as values may be outside the [0,1] range and may not display correctly.
#normalization is typically used as a preprocessing step before feeding images into a model, and the normalized tensors are not meant to be visualized directly (only for model input)

#__SOFT MASK / FEATHERED MASKS__

'''right now the mask has very sharp edges (rectangular right now for example)
- Soft masks have values between 0.0 and 1.0 
- The edge can gradually transition: {0.0 -> 0.2 -> 0.5 -> 0.8 -> 1.0} [makes the blending smoother]
- we can create this soft mask by blurring the hard mask
'''

mask_blur = transforms.GaussianBlur(kernel_size = 31, sigma = 8)

soft_mask = mask_blur(mask)
soft_mask = soft_mask.clamp(0, 1)

print(f"soft mask shape: {soft_mask.shape}")
print(f"soft mask min value: {soft_mask.min().item()}")
print(f"soft mask max value: {soft_mask.max().item()}")

soft_mask_image = to_pil(soft_mask)
soft_mask_image.save(output_dir / "soft_mask.jpg")

soft_blur_filled_tensor = image_tensor * (1 - soft_mask) + blurred_tensor * soft_mask #this is called alpha blending
soft_blur_filled_image = to_pil(soft_blur_filled_tensor)
soft_blur_filled_image.save(output_dir / "soft_blur_filled_sample.jpg")

print("saved soft mask image to:", output_dir / "soft_mask.jpg")
print(f"saved soft blur filled image to: {output_dir}/soft_blur_filled_sample.jpg")

# soft mask = smooth boundary / smooth selection
# blurred image = actual blurry image content

#ALPHA BLENDING

#technique for combining two images (or tensors) together using a mask that defines the blending ratio between the two images at each pixel location
#alpha blending formula: blended_pixel = (1 - alpha) * image_a + alpha * image_b

'''in our cases:
alpha = 0.0 -> fully image_a
alpha = 1.0 -> fully image_b
alpha = 0.5 -> half image_a, half image_b
where image_a = original image and image_b = blurred image, and alpha is our soft mask
'''

#Explicit demo:

alpha = 0.35
global_blend_tensor = image_tensor * (1 - alpha) + blurred_tensor * alpha
global_blend_image = to_pil(global_blend_tensor)
global_blend_image.save(output_dir / "global_blend_sample.jpg")

print(f"saved global blend image to: {output_dir}/global_blend_sample.jpg")

#Loading a mask from disk
loaded_mask_image = Image.open(output_dir / "mask.jpg").convert("L") # convert to grayscale (L mode) since it's a mask and we only need one channel for the mask (values between 0 and 255)
loaded_mask_tensor = to_tensor(loaded_mask_image)

print("loaded mask tensor shape:", loaded_mask_tensor.shape)
print("loaded mask min/max:", loaded_mask_tensor.min().item(), loaded_mask_tensor.max().item())

loaded_hard_mask = (loaded_mask_tensor > 0.5).float()

print("loaded hard mask min/max:", loaded_hard_mask.min().item(), loaded_hard_mask.max().item())

#what if we didn't convert to grayscale and kept it as RGB? --> we would have a 3 channel mask tensor instead of a single channel, which could lead to issues when applying the mask for blending or other operations, as the mask values would be duplicated across all three channels instead of being a single channel that defines the blending ratio. 

#IMAGE-MASK SIZE ALIGNMENT

print("image spatial size:", image_tensor.shape[-2:])
print("mask spatial size:", loaded_hard_mask.shape[-2:])

if image_tensor.shape[-2:] != loaded_hard_mask.shape[-2:]:
    resize_mask = transforms.Resize(image_tensor.shape[-2:])
    loaded_hard_mask = resize_mask(loaded_hard_mask)
    loaded_hard_mask = (loaded_hard_mask > 0.5).float()

print("aligned mask shape:", loaded_hard_mask.shape)
