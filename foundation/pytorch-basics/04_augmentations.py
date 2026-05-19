from pathlib import Path

from PIL import Image
from torchvision import transforms

FOUNDATION_DIR = Path(__file__).resolve().parents[1]

image_path = FOUNDATION_DIR / "images" / "sample.jpg"
output_dir = FOUNDATION_DIR / "outputs" / "augmentations"
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
