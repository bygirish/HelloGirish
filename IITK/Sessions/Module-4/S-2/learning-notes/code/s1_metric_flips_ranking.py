"""Skill != metric: the same two models swap places depending on the metric."""
import re
def norm(s): return " ".join(re.sub(r"\b(a|an|the)\b", " ", s.lower().strip().rstrip(".")).split())
def exact(p, t): return float(norm(p) == norm(t))
def contains(p, t): return float(norm(t) in norm(p))            # lenient "answer appears in output"

gold = {"What is the man doing?": "walking", "What color is the bus?": "red", "How many dogs?": "2"}
model_terse   = {"What is the man doing?": "walking", "What color is the bus?": "blue", "How many dogs?": "2"}
model_verbose = {"What is the man doing?": "The man is walking along the street.",
                 "What color is the bus?": "The bus is red with white stripes.",
                 "How many dogs?": "There are 2 dogs playing."}
for name, m in [("terse (1 wrong)", model_terse), ("verbose (all right)", model_verbose)]:
    em = sum(exact(m[q], a) for q, a in gold.items()) / len(gold)
    ct = sum(contains(m[q], a) for q, a in gold.items()) / len(gold)
    print(f"{name:20s} exact-match={em:.2f}   answer-contained={ct:.2f}")
