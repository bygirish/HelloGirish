from pathlib import Path
ASSETS = Path(__file__).resolve().parent.parent / "assets"
HERE = Path(__file__).resolve().parent
import time, torch
from PIL import Image
from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained("facebook/sam-vit-base").eval()
proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
image = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")
print("image size (w,h):", image.size)

# 1) Heavy part: run the image encoder ONCE and cache the embedding
t = time.time()
inputs = proc(image, return_tensors="pt")
with torch.no_grad():
    emb = model.get_image_embeddings(inputs["pixel_values"])
print(f"image encoder: {time.time()-t:.2f}s  embedding shape {tuple(emb.shape)}")

# 2) Light part: each click reuses the cached embedding
clicks = {"dog body": [[118, 258]], "dog ear patch": [[92, 218]], "bird": [[200, 196]]}
saved = {}
for name, point in clicks.items():
    t = time.time()
    p = proc(image, input_points=[[point]], return_tensors="pt")
    with torch.no_grad():
        out = model(image_embeddings=emb, input_points=p["input_points"], multimask_output=True)
    masks = proc.image_processor.post_process_masks(out.pred_masks, p["original_sizes"], p["reshaped_input_sizes"])[0]
    ious = out.iou_scores[0, 0].tolist()
    areas = [int(m.sum()) for m in masks[0]]
    saved[name] = (masks[0], ious)
    print(f"{name:14s} click {point[0]}: decoder {time.time()-t:.3f}s | 3 masks, predicted IoU {[round(i,3) for i in ious]} | areas(px) {areas}")

# visualise the 3 candidate masks for the "dog body" click
import numpy as np
m, ious = saved["dog body"]
base = np.array(image).astype(float)
tiles = []
for k in range(3):
    over = base.copy(); mk = m[k].numpy()
    over[mk] = 0.45 * over[mk] + 0.55 * np.array([255, 40, 40])
    tiles.append(Image.fromarray(over.astype("uint8")))
W, H = image.size
sheet = Image.new("RGB", (W * 3 + 20, H + 30), "white")
from PIL import ImageDraw
d = ImageDraw.Draw(sheet)
for k, t in enumerate(tiles):
    sheet.paste(t, (k * (W + 10), 30)); d.text((k * (W + 10) + 4, 8), f"mask {k}  predicted IoU {ious[k]:.2f}", fill="black")
sheet.save(ASSETS / "sam-demo-dog-3-masks.png")
