"""The four caption prompt styles from slide 10, on the same image, with a tiny VLM."""
from pathlib import Path
from PIL import Image
from vlm_small import ask
ASSETS = Path(__file__).resolve().parent.parent / "assets"
img = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")
styles = {
  "short":    "Generate a concise caption that highlights the main subject.",
  "detailed": "Generate a comprehensive caption including the main objects, their spatial relationships, and textural details.",
  "tags":     "List 5 comma-separated tags for the main subjects and scene.",
  "catalog":  "One sentence, under 125 characters, describing only what is visually present.",
}
for name, p in styles.items():
    print(f"[{name:8s}] {ask(img, p, max_new_tokens=70)}")
