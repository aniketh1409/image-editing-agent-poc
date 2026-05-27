from pathlib import Path
import json

from PIL import Image, ImageDraw


EXAMPLES_DIR = Path("prototypes/contact-eval/examples")


def load_frames(path):
    with path.open() as file:
        data = json.load(file)

    return data["frames"]


def draw_box(draw, box, fill, outline):
    rect = [
        box["col_start"],
        box["row_start"],
        box["col_end"],
        box["row_end"],
    ]
    draw.rectangle(rect, fill=fill, outline=outline, width=2)


def generate_case(case_name):
    frames = load_frames(EXAMPLES_DIR / f"{case_name}_boxes.json")
    output_dir = EXAMPLES_DIR / f"{case_name}_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        frame_idx = frame["frame_idx"]
        image = Image.new("RGB", (160, 120), color=(236, 238, 241))
        draw = ImageDraw.Draw(image)

        draw.rectangle([0, 82, 160, 120], fill=(220, 224, 229))
        draw_box(
            draw,
            frame["object_box"],
            fill=(90, 135, 255),
            outline=(20, 65, 210),
        )
        draw_box(
            draw,
            frame["hand_box"],
            fill=(255, 120, 105),
            outline=(205, 35, 35),
        )
        draw.text((8, 8), f"{case_name} frame {frame_idx}", fill=(35, 35, 35))

        image.save(output_dir / f"frame_{frame_idx:02d}.jpg")

    print("saved frames:", output_dir)


def main():
    generate_case("good")
    generate_case("broken")


if __name__ == "__main__":
    main()
