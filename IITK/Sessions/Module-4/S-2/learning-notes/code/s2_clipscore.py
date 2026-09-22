"""CLIPScore (Hessel et al. 2021): reference-free caption score = 2.5 * max(cos(image, caption), 0)."""
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
ASSETS = Path(__file__).resolve().parent.parent / "assets"
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
img = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")
caps = ["A black and white dog sits on a pebble beach near a black bird.",
        "The two dogs are sitting on the ground.",                # SmolVLM's hallucinated caption
        "A black and white dog is sitting on a beach looking at his owner.",   # LLaVA-style, invented owner
        "A photo of a scene."]                                    # over-generic
with torch.no_grad():
    o = m(**p(text=caps, images=img, return_tensors="pt", padding=True))
ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
for c, s in zip(caps, (2.5 * (ie @ te.T).clamp(min=0))[0].tolist()):
    print(f"CLIPScore {s:.3f}  {c}")
