"""Make a figure: the same 10-pt line on an A4 page, seen at different page widths (what the vision encoder gets)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ASSETS = Path(__file__).resolve().parent.parent / "assets"

A4_W_IN = 8.27
text = "Invoice # INV-20394   Total: $481.50   Date: 14 Mar 2026"
# 1) render the line as it would look on a 300-DPI scan: 10 pt = 10/72 in = 41.7 px tall
hi = Image.new("L", (2481, 70), 255)
ImageDraw.Draw(hi).text((20, 10), text, fill=0, font=ImageFont.load_default(size=40))

rows = []
for width_px in (224, 448, 533, 2481):
    scale = width_px / 2481
    small = hi.resize((max(1, round(2481 * scale)), max(1, round(70 * scale))), Image.BILINEAR)
    shown = small.resize((1240, 35), Image.NEAREST)          # blow back up so you can SEE the pixels
    rows.append((width_px, round(width_px / A4_W_IN), shown))

fig = Image.new("L", (1020, 45 * len(rows) + 10), 255)
d = ImageDraw.Draw(fig)
for i, (w, dpi, im) in enumerate(rows):
    fig.paste(im, (250, 5 + i * 45))
    d.text((8, 12 + i * 45), f"{w:>4}px page = {dpi:>3} DPI", fill=0, font=ImageFont.load_default(size=18))
fig.save(ASSETS / "doc-resolution-legibility-strip.png")
print("saved", ASSETS / "doc-resolution-legibility-strip.png")
