print("chain accuracy p^n")
for p in (0.95, 0.90):
    print(p, [round(p ** n, 2) for n in (1, 3, 5, 10)])

print("\n--- extract-then-reason for a chart ---")
extracted = {"Q1": 120, "Q2": 135, "Q3": 160, "Q4": 150}     # step 1: the VLM outputs this table
qs = list(extracted)
growth = {qs[i]: extracted[qs[i]] - extracted[qs[i-1]] for i in range(1, len(qs))}
best = max(growth, key=growth.get)                           # step 2: plain code does the maths
print("growth:", growth, "-> largest growth:", best)
uncertainty = 5                                               # step 3: +/- reading error from the chart
# each growth = difference of two readings, so its error is up to +/-2*uncertainty
worst_best = growth[best] - 2 * uncertainty
rivals = {q: g + 2 * uncertainty for q, g in growth.items() if q != best}
print(f"{best} growth could be as low as {worst_best}; rivals' best case: {rivals}")
print("answer robust to +/-5 reading error?", all(worst_best > g for g in rivals.values()))
