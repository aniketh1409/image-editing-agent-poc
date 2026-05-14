from pathlib import Path

from PIL import Image
from torchvision import transforms

image_path = Path("images/sample.jpg")
output_dir = Path("outputs/augmentations")
output_dir.mkdir(parents=True, exist_ok=True)

image = Image.open(image_path).convert("RGB")

augment = transforms.Compose( #chain multiple image transforms
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p = 1.0),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomRotation(degrees=15)
    ]
)

for i in range(5):
    augmented_image = augment(image)
    augmented_image.save(output_dir / f"augmented_{i}.jpg")
    print(f"Saved to {output_dir}/augmented_{i}.jpg")