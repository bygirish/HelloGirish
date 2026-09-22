"""Direct vs decomposed prompting on (1) a chart question and (2) a safety-rule question."""
from pathlib import Path
from PIL import Image
from vlm_small import ask
ASSETS = Path(__file__).resolve().parent.parent / "assets"
chart = Image.open(ASSETS / "chart-quarterly-revenue.png").convert("RGB")
moto = Image.open(ASSETS / "caption-motorcycle-karnataka.png").convert("RGB")

print("=== CHART (truth: Q1 120, Q2 135, Q3 160, Q4 150; largest growth Q2->Q3 = +25) ===")
print("DIRECT :", ask(chart, "Which quarter had the largest growth over the previous quarter?", max_new_tokens=40))
print("EXTRACT:", ask(chart, "Read the chart. List each quarter and its approximate value, one per line, like 'Q1: 100'.", max_new_tokens=60))

print("\n=== SAFETY RULE (truth: rider wears a helmet, the passenger does NOT) ===")
print("DIRECT :", ask(moto, "Are all the people on the motorcycle wearing helmets? Answer yes or no.", max_new_tokens=10))
print("COT    :", ask(moto, "First, list each person on the motorcycle and say whether that person is wearing a helmet. "
                             "Then answer: are all of them wearing helmets?", max_new_tokens=80))
