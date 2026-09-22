from pathlib import Path
ASSETS = Path(__file__).resolve().parent.parent / "assets"
HERE = Path(__file__).resolve().parent
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(ASSETS / "caption-motorcycle-karnataka.png").convert("RGB")
labels = ["a motorcycle", "a car", "a bicycle", "a satellite dish", "a dog", "a laptop"]
inputs = proc(text=[f"a photo of {l}" for l in labels], images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    out = model(**inputs)
probs = out.logits_per_image.softmax(dim=-1)[0]          # logits = 100 * cosine similarity
for l, p in sorted(zip(labels, probs.tolist()), key=lambda x: -x[1]):
    print(f"{p:6.3f}  {l}")
print("logit scale:", round(model.logit_scale.exp().item(), 1))

# Multi-label tagging: look at raw cosine similarity instead of a softmax that forces one winner
img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
cos = (img_emb @ txt_emb.T)[0]
print("\nraw cosine similarity:")
for l, c in sorted(zip(labels, cos.tolist()), key=lambda x: -x[1]):
    print(f"{c:6.3f}  {l}")
