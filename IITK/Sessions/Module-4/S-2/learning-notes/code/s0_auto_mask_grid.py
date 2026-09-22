"""Data-engine Stage 3 in miniature: prompt SAM with a regular point grid, keep confident + stable masks."""
from pathlib import Path
import time
from PIL import Image
from transformers import pipeline

ASSETS = Path(__file__).resolve().parent.parent / "assets"
image = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")
gen = pipeline("mask-generation", model="facebook/sam-vit-base", device="cpu")
t = time.time()
out = gen(image, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.92)
print(f"{len(out['masks'])} masks kept after IoU/stability filtering + NMS  ({time.time()-t:.0f}s on CPU)")
areas = sorted((int(m.sum()) for m in out["masks"]), reverse=True)
print("largest mask areas (px):", areas[:8])
