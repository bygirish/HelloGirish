import re, math
from collections import Counter

def normalize(a):
    a = a.lower().strip().rstrip(".")
    a = re.sub(r"\b(a|an|the)\b", " ", a)
    return " ".join(a.split())

def vqa_accuracy(pred, human_answers):
    """Official-style VQA accuracy: min(#matching humans / 3, 1)."""
    n = sum(normalize(h) == normalize(pred) for h in human_answers)
    return min(n / 3, 1.0)

humans = ["walking"] * 4 + ["crossing street"] * 3 + ["walking on street"] * 3
for p in ["walking", "Crossing the street.", "The man appears to be walking across the street."]:
    print(f"{vqa_accuracy(p, humans):.2f}  <- {p!r}")

print("\n--- DPO loss for one preference pair ---")
def dpo_loss(logratio_chosen, logratio_rejected, beta=0.1):
    z = beta * (logratio_chosen - logratio_rejected)
    return -math.log(1 / (1 + math.exp(-z)))
for lc, lr in [(0.0, 0.0), (0.5, -0.3), (3.0, -3.0)]:
    print(f"chosen {lc:+.1f}, rejected {lr:+.1f} -> loss {dpo_loss(lc, lr):.3f}")

print("\n--- Building preference triples by perturbation ---")
import random
random.seed(0)
data = [
    {"image": "img_001.jpg", "q": "What is the man holding?", "a": "a bat"},
    {"image": "img_002.jpg", "q": "How many dogs are there?", "a": "2"},
    {"image": "img_003.jpg", "q": "What color is the bus?",   "a": "red"},
]
triples = []
for i, ex in enumerate(data):
    other = data[(i + 1) % len(data)]
    triples.append({"x": (ex["image"], ex["q"]), "chosen": ex["a"], "rejected": other["a"], "kind": "random answer"})
    triples.append({"x": (ex["image"], other["q"]), "chosen": "cannot be determined from the image",
                    "rejected": ex["a"], "kind": "random question"})   # old answer no longer fits the new question
    triples.append({"x": (ex["image"].replace(".jpg", "_objects_blacked.jpg"), ex["q"]),
                    "chosen": "not visible in the image", "rejected": ex["a"], "kind": "image perturbation"})
for t in triples[:3]:
    print(t)
