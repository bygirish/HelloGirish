"""Slide 25's experiment: how much does each answer token depend on the image? Compare log-probs with a blurred image."""
from pathlib import Path
import torch
from PIL import Image, ImageFilter
from vlm_small import proc, model
ASSETS = Path(__file__).resolve().parent.parent / "assets"
img = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")
blur = img.filter(ImageFilter.GaussianBlur(25))
question = "Describe the image in one sentence."
answer = "A black and white dog is sitting on a pebble beach next to a black bird."

def token_logprobs(image):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    full = proc(text=prompt + " " + answer, images=[image], return_tensors="pt")
    n_prompt = proc(text=prompt, images=[image], return_tensors="pt")["input_ids"].shape[1]
    with torch.no_grad():
        logits = model(**full).logits[0]
    ids = full["input_ids"][0]
    lp = torch.log_softmax(logits[:-1], -1).gather(1, ids[1:, None])[:, 0]     # logp(token_t | tokens_<t)
    toks = [proc.tokenizer.decode([i]) for i in ids[n_prompt:]]
    return toks, lp[n_prompt - 1:]

toks, lp_img = token_logprobs(img)
_, lp_blur = token_logprobs(blur)
print(f"{'token':>10s} {'p(orig)':>8s} {'p(blur)':>8s}  change")
for t, a, b in zip(toks, lp_img.exp(), lp_blur.exp()):
    flag = "  <-- image-dependent" if abs(a - b) > 0.2 else ""
    print(f"{t!r:>10s} {a:8.3f} {b:8.3f}  {a-b:+.3f}{flag}")
print(f"mean |change| = {(lp_img.exp()-lp_blur.exp()).abs().mean():.3f} over {len(toks)} tokens")
