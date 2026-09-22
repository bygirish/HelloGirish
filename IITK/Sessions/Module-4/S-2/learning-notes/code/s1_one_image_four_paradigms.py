"""Same photo, four paradigms: CLIP scores, a detector's boxes, SAM's mask, a VLM's sentence."""
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ASSETS = Path(__file__).resolve().parent.parent / "assets"
image = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
labels = ["a dog on a beach", "a bird on a beach", "a dog and a bird on a beach", "a cat on a sofa"]
with torch.no_grad():
    probs = clip(**cp(text=labels, images=image, return_tensors="pt", padding=True)).logits_per_image.softmax(-1)[0]
print("CLIP  (match)   :", {l: round(p, 3) for l, p in zip(labels, probs.tolist())})
print("OWLv2 (boxes)   : see s0_grounded_sam.py  -> bird box, dog box")
print("SAM   (outline) : see s0_grounded_sam.py  -> 672-px bird mask, 2,438-px dog mask")
print("VLM   (describe): see s2_s3_vlm_probes.py -> free-form sentence")
