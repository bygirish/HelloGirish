from pathlib import Path
ASSETS = Path(__file__).resolve().parent.parent / "assets"
HERE = Path(__file__).resolve().parent
from PIL import Image, ImageDraw, ImageFont

# Render a synthetic invoice (the slide's Acme example)
img = Image.new("RGB", (640, 360), "white"); d = ImageDraw.Draw(img)
font = ImageFont.load_default(size=26)
lines = ["INVOICE", "Vendor:     Acme Supplies Ltd.", "Invoice #:  INV-20394", "Date:       14 Mar 2026", "Total:      $481.50"]
for i, t in enumerate(lines):
    d.text((30, 30 + i * 60), t, fill="black", font=font)
img.save(HERE / "invoice.png")

