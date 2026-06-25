from __future__ import annotations

from pathlib import Path
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL_PATH = Path("/project/rrg-vislearn/kini1409/models/deepeyes/DeepEyesV2_7B_1031")
IMAGE_PATH = Path("/project/rrg-vislearn/kini1409/toilet_frame.png")


def log(message: str) -> None:
    print(f"[deepeyes] {message}", flush=True)


def main() -> None:
    log("start")
    log(f"model path: {MODEL_PATH}")
    log(f"image path: {IMAGE_PATH}")
    log(f"torch: {torch.__version__}")
    log(f"cuda available: {torch.cuda.is_available()}")
    log(f"cuda device count: {torch.cuda.device_count()}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model path: {MODEL_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing image path: {IMAGE_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this inside a GPU allocation.")

    t0 = time.time()
    log("loading processor")
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    log(f"processor loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    log("loading model")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    log(f"model loaded in {time.time() - t0:.1f}s")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(IMAGE_PATH)},
                {
                    "type": "text",
                    "text": (
                        "Identify the toilet lid, the hinge or pivot region, and the front tip of the lid. "
                        "Then explain whether this frame is useful for evaluating hinge-like opening motion. "
                        "If possible, return rough part locations in words, not exact masks."
                    ),
                },
            ],
        }
    ]

    log("processing image/prompt")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    log("generating")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    log(f"generated in {time.time() - t0:.1f}s")

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n========== DeepEyesV2 output ==========\n", flush=True)
    print(output_text, flush=True)
    print("\n=======================================\n", flush=True)


if __name__ == "__main__":
    main()
