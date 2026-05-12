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