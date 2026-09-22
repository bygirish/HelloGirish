"""Scan clean-up before OCR/VLM: grayscale, autocontrast, deskew, crop-to-content, upscale small text."""
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
HERE = Path(__file__).resolve().parent

def preprocess(img, skew_deg=0.0, min_width=1600):
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)                        # stretch faded ink
    if skew_deg:
        g = g.rotate(-skew_deg, expand=True, fillcolor=255)       # deskew (angle from e.g. a Hough-line estimate)
    g = g.filter(ImageFilter.MedianFilter(3))                     # remove salt-and-pepper scanner noise
    bbox = ImageOps.invert(g).getbbox()                           # crop to where the ink is
    g = g.crop(bbox) if bbox else g
    if g.width < min_width:                                       # upscale so small print gets enough pixels
        g = g.resize((min_width, round(g.height * min_width / g.width)), Image.LANCZOS)
    return g

scan = Image.open(HERE / "invoice.png").rotate(4, expand=True, fillcolor="white")   # simulate a 4° skewed scan
clean = preprocess(scan, skew_deg=4)
print("scan", scan.size, "-> cleaned", clean.size)
