"""Simple image editing POC.

Usage:
    .venv\\Scripts\\python.exe scripts\\simple_editor.py --image images\\sample.jpg --mask outputs\\mask.jpg --mode blur --output outputs\\simple_blur.jpg

This is intentionally classical/non-generative. The goal is to make the
image + mask + edit-mode pipeline work before replacing the fill step with a
learned inpainting model.
"""

from pathlib import Path

from PIL import Image
from torchvision import transforms

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



def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.clamp(0, 1) #forces the values to be in [0, 1] (if < 0 -> 0 if > 1 -> 1)

    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    image.save(path)

def main():
    #gonna hardcode the path for now
    image_path = Path("images/sample.jpg")
    mask_path = Path("images/mask.jpg")
    output_path = Path("outputs/practice_average.jpg")

    image_tensor = load_rgb_image(image_path)
    mask_tensor = load_mask(mask_path, image_tensor.shape[-2:])

    edited_tensor = fill_average(image_tensor, mask_tensor)

    save_tensor_image(edited_tensor, output_path)

if __name__ == "__main__":
    main()
