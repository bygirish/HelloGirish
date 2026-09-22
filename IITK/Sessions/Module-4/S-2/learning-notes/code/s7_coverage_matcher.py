"""Design 2 from the live Q&A: match atomic design requirements to verification statements, report coverage.
Step 5 of the pipeline (retrieval) with a small embedding model. Steps 6+ (rerank, LLM judge) would follow."""
from sentence_transformers import SentenceTransformer, util

design = [
    "DR-1: The FIFO shall assert the full flag when 16 entries are stored.",
    "DR-2: The FIFO shall assert the empty flag after reset.",
    "DR-3: The write pointer shall wrap around to zero after entry 15.",
    "DR-4: The interface shall support a clock frequency of 200 MHz.",
    "DR-5: Parity errors on the read data bus shall raise an interrupt.",
]
verification = [
    "VR-10: Check that full goes high after writing sixteen words into the FIFO.",
    "VR-11: After power-on reset, verify the empty indicator is set.",
    "VR-12: Write 17 entries and confirm the write address rolls over to 0.",
    "VR-13: Run the regression at 200 MHz clock and confirm no timing violations.",
]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
D, V = model.encode(design, normalize_embeddings=True), model.encode(verification, normalize_embeddings=True)
sim = util.cos_sim(D, V)

THRESH = 0.45                                                  # tune on a labelled set, never by eye!
covered = 0
for i, req in enumerate(design):
    j = int(sim[i].argmax()); s = float(sim[i, j])
    status = "COVERED (candidate)" if s >= THRESH else "NOT COVERED"
    covered += s >= THRESH
    print(f"{req[:5]} -> {verification[j][:6]}  sim={s:.2f}  {status}")
print(f"coverage = {covered}/{len(design)} = {covered/len(design):.0%}  (before LLM-judge confirmation)")

# --- Step 8 of the pipeline: evaluate against a (tiny) labelled set BEFORE trusting any threshold ---
gold = {0: 0, 1: 1, 2: 2, 3: 3, 4: None}                    # DR index -> correct VR index (DR-5 is truly uncovered)
print(f"\ntrue coverage = {sum(v is not None for v in gold.values())}/{len(gold)}")
for t in (0.30, 0.35, 0.45):
    tp = fp = fn = 0
    for i, g in gold.items():
        j = int(sim[i].argmax()); hit = float(sim[i, j]) >= t
        if hit and j == g: tp += 1
        elif hit: fp += 1                                      # matched, but to the wrong item / nothing true
        elif g is not None: fn += 1
    print(f"threshold {t:.2f}: correct={tp} wrong-match={fp} missed={fn}  "
          f"precision={tp/max(tp+fp,1):.2f} recall={tp/4:.2f}")
