from pathlib import Path
import csv
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
    row = (box["row_start"] + box["row_end"])
    col = (box["col_start"] + box["col_end"])



def main():
    pass
if __name__ == "__main__":
    main()