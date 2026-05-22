from pathlib import Path
import csv
from math import sqrt
from PIL import Image, ImageDraw

OUTPUT_DIR = Path("prototypes/contact-eval/outputs")

def make_box(row_start, row_end, col_start, col_end):
    return {
        "row_start": row_start,
        "row_end": row_end,
        "col_start": col_start,
        "col_end": col_end
    }

def box_center(box):
    row = (box["row_start"] + box["row_end"]) / 2
    col = (box["col_start"] + box["col_end"]) / 2
    return row, col


def center_distance(box_a, box_b):
    row_a, col_a = box_center(box_a)
    row_b, col_b = box_center(box_b)
    distance = sqrt((row_a - row_b) ** 2 + (col_a - col_b) ** 2)
    return distance

def draw_frame(output_path, hand_box, object_box, contact_ok, image_size = (160, 120)): #explicit rn
    width, height = image_size

    image = Image.new("RGB", (width, height), color = (245, 245, 245))
    draw = ImageDraw.Draw(image)

    object_rect = [
        object_box["col_start"], 
        object_box["row_start"],
        object_box["col_end"],
        object_box["row_end"]
    ]

    hand_rect = [
        hand_box["col_start"], 
        hand_box["row_start"],
        hand_box["col_end"],
        hand_box["row_end"]
    ]

    draw.rectangle(object_rect, outline = (40, 90, 225), width = 3)
    draw.rectangle(hand_rect, outline = (255, 60, 60), width = 3)

    status_text = "Contact OK!" if contact_ok else "Contact BROKEN!"
    status_color = (20, 130, 60) if contact_ok else ((200, 40, 40))

    draw.text((10, 10), status_text, fill=status_color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    #TODO: pass/fail check
    pass

def contact_ok(distance, threshold):
    pass

def main():
    #TODO: generate a few fake boxes over time
    #TODO: compute contact distance for each frame
    #TODO: save the annotated frames
    #TODO: save metric data (CSV)
    pass


if __name__ == "__main__":
    main()