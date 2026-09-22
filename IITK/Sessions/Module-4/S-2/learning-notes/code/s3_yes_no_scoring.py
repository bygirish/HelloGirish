"""Decoding-free VQA: read P(Yes) vs P(No) directly from the next-token logits of a VLM."""
from pathlib import Path
import torch
from PIL import Image
from vlm_small import proc, model
ASSETS = Path(__file__).resolve().parent.parent / "assets"
dog = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")

def p_yes(image, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + " Answer yes or no."}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[image], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]                        # next-token distribution
    tok = proc.tokenizer
    yes = max(logits[tok.encode(w, add_special_tokens=False)[0]] for w in (" Yes", "Yes", " yes", "yes"))
    no  = max(logits[tok.encode(w, add_special_tokens=False)[0]] for w in (" No", "No", " no", "no"))
    return torch.softmax(torch.stack([yes, no]), 0)[0].item()

for q in ["Is there a dog in the image?", "Is there a bird in the image?",
          "Is there a cat in the image?", "Is there a person in the image?"]:
    print(f"P(yes)={p_yes(dog, q):.3f}  {q}")
