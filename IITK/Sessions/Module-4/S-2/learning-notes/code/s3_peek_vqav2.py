"""Look at real VQAv2 rows (the mini split the course notebook uses): stores the single most-common of the ~10 human answers."""
from datasets import load_dataset
ds = load_dataset("dutta18/miniVQAv2DatasetForCollab", split="train")
print(ds)
for i in (0, 5, 81):
    row = ds[i]
    print(f"\n[{i}] Q: {row['question']}\n    answer (majority of ~10 humans): {row['multiple_choice_answer']!r}")
