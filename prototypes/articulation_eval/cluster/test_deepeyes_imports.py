from __future__ import annotations

import time

print("[smoke] start", flush=True)

t0 = time.time()
import torch
print(f"[smoke] torch imported in {time.time() - t0:.1f}s", flush=True)
print(f"[smoke] torch version: {torch.__version__}", flush=True)
print(f"[smoke] cuda available: {torch.cuda.is_available()}", flush=True)
print(f"[smoke] cuda devices: {torch.cuda.device_count()}", flush=True)

t0 = time.time()
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
print(f"[smoke] transformers qwen imports in {time.time() - t0:.1f}s", flush=True)

t0 = time.time()
from qwen_vl_utils import process_vision_info
print(f"[smoke] qwen_vl_utils imported in {time.time() - t0:.1f}s", flush=True)

print("[smoke] ok", flush=True)
