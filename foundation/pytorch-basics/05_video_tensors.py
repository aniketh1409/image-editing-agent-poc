from pathlib import Path

import torch
from torchvision import transforms


FOUNDATION_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = FOUNDATION_DIR / "outputs" / "video_tensors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_frame(tensor, path):
    """Save a C x H x W tensor as an image."""
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)


def draw_rectangle(video_tensor, frame_idx, box, color):
    """Draw a colored rectangle into one video frame.

    video_tensor shape: T x C x H x W
    box: (row_start, row_end, col_start, col_end)
    color shape after view: C x 1 x 1
    """
    row_start, row_end, col_start, col_end = box
    color_tensor = torch.tensor(color).view(3, 1, 1)
    video_tensor[frame_idx, :, row_start:row_end, col_start:col_end] = color_tensor


def rectangle_center(box):
    """Return the (row, col) center of a rectangle."""
    row_start, row_end, col_start, col_end = box
    row_center = (row_start + row_end) / 2
    col_center = (col_start + col_end) / 2
    return torch.tensor([row_center, col_center])


# A video tensor is one image tensor per time step.
#
# Single image:
#   C x H x W
#
# Single video:
#   T x C x H x W
#
# Batch of videos:
#   B x T x C x H x W
num_frames = 8
channels = 3
height = 96
width = 128


# Create a simple fake video with a horizontal color gradient.
# Shape: C x H x W
x_gradient = torch.linspace(0, 1, width).view(1, 1, width)
x_gradient = x_gradient.expand(channels, height, width)

# Stack shifted versions of the image to create motion over time.
# Shape after stack: T x C x H x W
video = torch.stack(
    [torch.roll(x_gradient, shifts=frame_idx * 6, dims=2) for frame_idx in range(num_frames)]
)

print("video shape:", video.shape)
print("video means per frame:", video.mean(dim=(1, 2, 3)))


# Create a mask for each frame.
# Shape: T x 1 x H x W
mask = torch.zeros((num_frames, 1, height, width))

box_height = 32
box_width = 32
row_start = 32

for frame_idx in range(num_frames):
    col_start = 8 + frame_idx * 10
    row_end = row_start + box_height
    col_end = col_start + box_width

    mask[frame_idx, :, row_start:row_end, col_start:col_end] = 1.0

print("mask shape:", mask.shape)
print("mask min/max:", mask.min().item(), mask.max().item())


# Toy human-object interaction:
# - blue rectangle = articulated object part, like a drawer front/handle area
# - red rectangle = hand
#
# The two rectangles stay close for the first few frames, then the hand drifts
# away. That drift is a toy version of a video editing failure: the object
# motion changes, but hand-object contact is not preserved.
interaction_video = video.clone()
contact_distances = []
contact_threshold_pixels = 30

for frame_idx in range(num_frames):
    object_col_start = 44 + frame_idx * 5
    object_box = (40, 64, object_col_start, object_col_start + 28)

    if frame_idx < 5:
        hand_col_start = object_col_start - 18
    else:
        hand_col_start = object_col_start - 18 - (frame_idx - 4) * 10

    hand_box = (44, 60, hand_col_start, hand_col_start + 16)

    draw_rectangle(interaction_video, frame_idx, object_box, color=(0.1, 0.3, 1.0))
    draw_rectangle(interaction_video, frame_idx, hand_box, color=(1.0, 0.1, 0.1))

    object_center = rectangle_center(object_box)
    hand_center = rectangle_center(hand_box)
    distance = torch.linalg.vector_norm(object_center - hand_center)
    contact_distances.append(distance)

contact_distances = torch.stack(contact_distances)
contact_ok = contact_distances < contact_threshold_pixels

print("interaction video shape:", interaction_video.shape)
print("contact distances:", contact_distances)
print("contact preserved per frame:", contact_ok)
print("contact failure frames:", torch.where(~contact_ok)[0])


# Broadcasting:
# video shape: T x 3 x H x W
# mask shape:  T x 1 x H x W
#
# The mask's 1 channel broadcasts across RGB channels.
edited_video = video * (1 - mask)

print("edited video shape:", edited_video.shape)


# Add a batch dimension.
# Shape: B x T x C x H x W
video_batch = edited_video.unsqueeze(0)

print("video batch shape:", video_batch.shape)


# Frame differences are a tiny way to measure motion/change over time.
# Shape:
#   video[1:]  -> (T - 1) x C x H x W
#   video[:-1] -> (T - 1) x C x H x W
frame_differences = (video[1:] - video[:-1]).abs()

print("frame differences shape:", frame_differences.shape)
print("average frame difference:", frame_differences.mean().item())


for frame_idx in range(num_frames):
    save_frame(video[frame_idx], OUTPUT_DIR / f"frame_{frame_idx:02d}_original.jpg")
    save_frame(mask[frame_idx], OUTPUT_DIR / f"frame_{frame_idx:02d}_mask.jpg")
    save_frame(edited_video[frame_idx], OUTPUT_DIR / f"frame_{frame_idx:02d}_edited.jpg")
    save_frame(
        interaction_video[frame_idx],
        OUTPUT_DIR / f"frame_{frame_idx:02d}_interaction.jpg",
    )

print("saved frames to:", OUTPUT_DIR)
