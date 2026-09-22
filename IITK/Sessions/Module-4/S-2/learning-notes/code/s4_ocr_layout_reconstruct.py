"""Why flat OCR text loses tables, and how x/y coordinates rebuild them."""
# What an OCR engine really returns: words + bounding boxes (x = left edge, y = top edge, in px)
words = [("Income", 40, 10), ("Overheads", 160, 10), ("Interest", 300, 10),
         ("10", 60, 50), ("20", 190, 50), ("30", 320, 50),
         ("40", 60, 90), ("50", 190, 90), ("60", 320, 90)]

print("FLAT TEXT :", " ".join(w for w, _, _ in words))

# Rebuild: group by row (y), then assign each value to the header whose x is closest
rows = {}
for w, x, y in words:
    rows.setdefault(y, []).append((x, w))
header_y, *data_ys = sorted(rows)
headers = sorted(rows[header_y])
table = []
for y in data_ys:
    rec = {}
    for x, w in rows[y]:
        col = min(headers, key=lambda h: abs(h[0] - x))[1]        # nearest header by x-position
        rec[col] = int(w)
    table.append(rec)
print("TABLE     :", table)
print("Interest total:", sum(r["Interest"] for r in table))
