from pathlib import Path
ASSETS = Path(__file__).resolve().parent.parent / "assets"
HERE = Path(__file__).resolve().parent
from vlm_small import ask
from PIL import Image, ImageFilter
dog = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")
print("CAPTION      :", ask(dog, "Write a one-sentence caption describing only what is visually present."))
print("CAT (naive)  :", ask(dog, "What color is the cat?"))
print("CAT (guarded):", ask(dog, "What color is the cat? If there is no cat in the image, answer exactly: 'There is no cat.'"))
blur = dog.filter(ImageFilter.GaussianBlur(25))
print("BLURRED      :", ask(blur, "Write a one-sentence caption describing only what is visually present."))
