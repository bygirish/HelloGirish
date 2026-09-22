from pathlib import Path
ASSETS = Path(__file__).resolve().parent.parent / "assets"
HERE = Path(__file__).resolve().parent
import re, torch
from collections import Counter
from PIL import Image
from vlm_small import ask

crowd = Image.open(ASSETS / "halluc-umbrella-crowd-image.png").convert("RGB")
def vote(q, k=7):
    torch.manual_seed(0)
    answers = []
    for _ in range(k):
        a = ask(crowd, q, sample=True, max_new_tokens=10)
        n = re.search(r"\d+", a)
        answers.append(n.group() if n else a)
    votes = Counter(answers)
    best, cnt = votes.most_common(1)[0]
    print("samples:", answers)
    print(f"majority = {best}  (agreement {cnt}/{len(answers)} = {cnt/len(answers):.2f})")

for q in ["How many umbrellas are open in this image? Answer with a single number.",
          "How many people are in this image? Answer with a single number."]:
    print("Q:", q); print("greedy :", ask(crowd, q, max_new_tokens=10)); vote(q)
