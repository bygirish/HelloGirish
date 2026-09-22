"""Render a simple quarterly revenue bar chart (known ground truth) for chart-QA experiments."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ASSETS = Path(__file__).resolve().parent.parent / "assets"
data = {"Q1": 120, "Q2": 135, "Q3": 160, "Q4": 150}
W, H, left, bottom, top = 640, 420, 80, 360, 40
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
f, fs = ImageFont.load_default(size=18), ImageFont.load_default(size=14)
ymax = 200
def y(v): return bottom - (bottom - top) * v / ymax
for g in range(0, ymax + 1, 50):                                   # gridlines every 50 (so 135 falls BETWEEN lines)
    d.line([(left, y(g)), (W - 30, y(g))], fill=(220, 220, 220)); d.text((left - 40, y(g) - 8), str(g), fill="black", font=fs)
for i, (q, v) in enumerate(data.items()):
    x0 = left + 30 + i * 130
    d.rectangle([x0, y(v), x0 + 80, bottom], fill=(66, 110, 180))
    d.text((x0 + 25, bottom + 8), q, fill="black", font=f)
d.line([(left, top), (left, bottom), (W - 30, bottom)], fill="black", width=2)
d.text((left, 8), "Quarterly revenue (Rs lakh)", fill="black", font=f)
img.save(ASSETS / "chart-quarterly-revenue.png"); print("saved chart; truth:", data)
