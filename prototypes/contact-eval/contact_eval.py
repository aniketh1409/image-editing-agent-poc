from pathlib import Path
import csv
import json
from math import sqrt
from PIL import Image, ImageDraw

OUTPUT_DIR = Path("prototypes/contact-eval/outputs")
EXAMPLES_DIR = Path("prototypes/contact-eval/examples")

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

def draw_frame(
    output_path,
    hand_box,
    object_box,
    contact_ok,
    input_frame_path=None,
    image_size=(160, 120),
):
    width, height = image_size

    if input_frame_path is not None and input_frame_path.exists():
        image = Image.open(input_frame_path).convert("RGB")
    else:
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


def contact_ok(distance, threshold):
    return distance < threshold


def box_edge_distance(box_a, box_b):
    a_top = box_a["row_start"]
    a_bottom = box_a["row_end"]
    a_left = box_a["col_start"]
    a_right = box_a["col_end"]

    b_top = box_b["row_start"]
    b_bottom = box_b["row_end"]
    b_left = box_b["col_start"]
    b_right = box_b["col_end"]

    if a_right < b_left:
        horizontal_gap = b_left - a_right
    elif b_right < a_left:
        horizontal_gap = a_left - b_right
    else:
        horizontal_gap = 0

    if a_bottom < b_top:
        vertical_gap = b_top - a_bottom
    elif b_bottom < a_top:
        vertical_gap = a_top - b_bottom
    else:
        vertical_gap = 0

    distance = sqrt(horizontal_gap ** 2 + vertical_gap ** 2)
    return distance


def load_boxes_json(path):
    with path.open() as file:
        data = json.load(file)

    return data["frames"]


def write_metrics_csv(csv_path, rows):
    with csv_path.open("w", newline="") as file:
        fieldnames = [
            "frame_idx",
            "hand_center_row",
            "hand_center_col",
            "object_center_row",
            "object_center_col",
            "center_distance",
            "edge_distance",
            "edge_threshold",
            "contact_ok",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(case_name, boxes_path):
    edge_threshold = 8
    output_dir = OUTPUT_DIR / case_name
    input_frames_dir = EXAMPLES_DIR / f"{case_name}_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_boxes_json(boxes_path)
    rows = []

    for frame in frames:
        frame_idx = frame["frame_idx"]
        hand_box = frame["hand_box"]
        object_box = frame["object_box"]

        center_dist = center_distance(hand_box, object_box)
        edge_dist = box_edge_distance(hand_box, object_box)
        ok = contact_ok(edge_dist, edge_threshold)

        output_path = output_dir / f"frame_{frame_idx:02d}.jpg"
        input_frame_path = input_frames_dir / f"frame_{frame_idx:02d}.jpg"
        draw_frame(output_path, hand_box, object_box, ok, input_frame_path)

        hand_row, hand_col = box_center(hand_box)
        object_row, object_col = box_center(object_box)

        rows.append({
            "frame_idx": frame_idx,
            "hand_center_row": hand_row,
            "hand_center_col": hand_col,
            "object_center_row": object_row,
            "object_center_col": object_col,
            "center_distance": center_dist,
            "edge_distance": edge_dist,
            "edge_threshold": edge_threshold,
            "contact_ok": ok,
        })

        print(case_name, frame_idx, "center:", center_dist, "edge:", edge_dist, ok)

    csv_path = output_dir / "metrics.csv"
    write_metrics_csv(csv_path, rows)
    print("saved metrics:", csv_path)


def main():
    run_case("good", EXAMPLES_DIR / "good_boxes.json")
    run_case("broken", EXAMPLES_DIR / "broken_boxes.json")

if __name__ == "__main__":
    main()


