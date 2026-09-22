import math

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def anls(pred, truth, tau=0.5):
    nl = levenshtein(pred.lower(), truth.lower()) / max(len(pred), len(truth))
    return 1 - nl if nl < tau else 0.0

for truth, pred in [("INV-20394", "INV-2O394"), ("481.50", "486.50"), ("June 7, 1998", "June 7, 1988"), ("481.50", "481.50")]:
    print(f"ANLS={anls(pred, truth):.3f}  exact={pred == truth}   truth={truth!r} pred={pred!r}")

print("\n--- Effective resolution of an A4 page ---")
A4_W_IN, A4_H_IN, TEXT_10PT_IN = 8.27, 11.69, 10 / 72
for width_px in (224, 448, 533, 2481):
    dpi = width_px / A4_W_IN
    print(f"{width_px:5d}px wide -> {dpi:6.1f} DPI -> 10pt text is {TEXT_10PT_IN*dpi:5.1f}px tall")

print("\n--- Qwen2.5-VL visual tokens (1 token = 28x28 px) ---")
def qwen_tokens(w, h, max_pixels=512*28*28, min_pixels=256*28*28):
    scale = 1.0
    if w * h > max_pixels: scale = math.sqrt(max_pixels / (w * h))
    if w * h < min_pixels: scale = math.sqrt(min_pixels / (w * h))
    W, H = max(28, round(w * scale / 28) * 28), max(28, round(h * scale / 28) * 28)
    return W, H, (W // 28) * (H // 28)
for w, h, note in [(2481, 3508, "A4 @300DPI, notebook budget"), (309, 400, "after thumbnail((400,400))")]:
    print(note, "->", qwen_tokens(w, h))
print("A4 @300DPI, no cap ->", qwen_tokens(2481, 3508, max_pixels=10**9))
