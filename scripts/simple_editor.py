from pathlib import Path

from PIL import Image
from torchvision import transforms

import argparse

"""Simple image editing POC.

Usage:
    .venv\\Scripts\\python.exe scripts\\simple_editor.py --image images\\sample.jpg --mask outputs\\mask.jpg --mode blur --output outputs\\simple_blur.jpg

This is intentionally classical/non-generative. The goal is to make the
image + mask + edit-mode pipeline work before replacing the fill step with a
learned inpainting model.
"""

VALID_MODES = ["black", "average", "blur", "soft-blur"] 

def load_rgb_image(path):
    image = Image.open(path).convert("RGB")
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    return image_tensor


#we need to make sure that mask size matches image size (else resize)
def load_mask(path, spatial_size):
    mask_image = Image.open(path).convert("L")
    to_tensor = transforms.ToTensor()
    mask_tensor = to_tensor(mask_image)

    if mask_tensor.shape[-2:] != spatial_size:
        resize = transforms.Resize(spatial_size)
        mask_tensor = resize(mask_tensor)

    hard_mask = (mask_tensor> 0.5).float()
    return hard_mask

        
def fill_black(image_tensor, mask_tensor):
    #inside mask -> black; outside -> image
    edited_tensor = image_tensor * (1 - mask_tensor) 
    return edited_tensor


def fill_average(image_tensor, mask_tensor):
    keep_region = 1 - mask_tensor

    sum_per_channel = (image_tensor * keep_region).sum(dim=(1, 2), keepdim = True)
    count_per_channel = keep_region.sum(dim = (1, 2), keepdim = True)

    average_color = sum_per_channel / count_per_channel

    edited_tensor = image_tensor * keep_region + average_color * mask_tensor

    return edited_tensor


def fill_blur(image_tensor, mask_tensor):
    keep_region = 1 - mask_tensor

    blur = transforms.GaussianBlur(kernel_size=31, sigma=10)
    blurred_tensor = blur(image_tensor)

    edited_tensor = (keep_region * image_tensor) + (mask_tensor * blurred_tensor)

    return edited_tensor

def fill_soft_blur(image_tensor, mask_tensor):
    blur = transforms.GaussianBlur(kernel_size=31, sigma=10)
    mask_blur = transforms.GaussianBlur(kernel_size=31, sigma=8)

    blurred_tensor = blur(image_tensor)
    soft_mask = mask_blur(mask_tensor).clamp(0,1) 
    #why is there a risk of values outside [0, 1]? because the blurring can create intermediate values that are not strictly 0 or 1, especially around the edges of the mask.
   
    edited_tensor = (1 - soft_mask) * image_tensor + soft_mask * blurred_tensor

    return edited_tensor

def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.clamp(0, 1) #forces the values to be in [0, 1] (if < 0 -> 0 if > 1 -> 1)

    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    image.save(path)


def edit_image(image_tensor, mask_tensor, mode):
    if mode == "black":
        return fill_black(image_tensor, mask_tensor)

    if mode == "average":
        return fill_average(image_tensor, mask_tensor)

    if mode == "blur":
        return fill_blur(image_tensor, mask_tensor)

    if mode == "soft-blur":
        return fill_soft_blur(image_tensor, mask_tensor)

    raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple mask-based image editor")

    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--mode", choices=VALID_MODES, required = True)
    parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()

def validate_args(args):
    if not args.image.exists():
        raise FileNotFoundError(f"Image {args.image} not found")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask {args.mask} does not exist")


def main():

    args = parse_args()
    validate_args(args)
    
    image_tensor = load_rgb_image(args.image)
    mask_tensor = load_mask(args.mask, image_tensor.shape[-2:])

    edited_tensor = edit_image(image_tensor, mask_tensor, args.mode)

    save_tensor_image(edited_tensor, args.output)

    print("mode:", args.mode)
    print("saved:", args.output)

if __name__ == "__main__":
    main()



