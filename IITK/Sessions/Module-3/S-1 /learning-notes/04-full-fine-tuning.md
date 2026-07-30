# 04 — Full Fine-Tuning

> **Why this matters:** Full fine-tuning is the *baseline* every other technique is measured
> against. It gives the **most adaptation capacity** — and pays for it in **memory, money, and
> forgetting**. Understanding exactly *why* it's expensive (the memory math is in file 09) and
> *why* it forgets is what motivates PEFT, curriculum learning, and every efficiency trick in
> this course.

---

## 1. What It Is

**Full fine-tuning = updating *every* parameter of the model on the new data.**

- It is a **parameter-update strategy**, *not* an objective. Any recipe — unsupervised, SFT,
  alignment — can be *run as* full fine-tuning. Full FT describes **what is trained (all
  weights), not what it's trained on.** (This is the key point from file 03.)
- Contrast: PEFT freezes most weights and trains a small subset; full FT freezes **nothing**.

---

## 2. The Three Defining Properties (slide 33) ⭐

Every discussion of full FT comes back to this trade-off triangle:

### (+) Maximum adaptation capacity
All weights are free to move, so the model can absorb **large domain shifts** and **deep
behavior changes** — the most thorough specialization possible. When a task is truly far from
the base distribution, nothing adapts as completely as full FT.

### (–) High cost
- **Adam training needs ~16 bytes of GPU memory *per parameter*** (weights + gradients + two
  optimizer moments, in mixed precision). See file 09 for the full arithmetic.
- Plus **one full checkpoint per task** — every fine-tuned copy is as big as the original model.
- For a 175B-parameter model (InstructGPT), that's an enormous, expensive footprint.

### (–) Catastrophic Forgetting
**Aggressive updates erase general skills.** Because *every* weight moves, the model can drift
far from its pre-trained state and lose capabilities it used to have (general reasoning, other
domains). This is covered mechanically in **Part II (S-2), file 03** — here just anchor the term:
*full FT is the most forgetting-prone strategy precisely because it moves the most weights.*

> 💡 **Learning Thought:** The three properties are one story: **all weights move → maximum
> capacity → but maximum memory cost → and maximum forgetting risk.** PEFT exists to keep most
> of the capacity while cutting the last two. Every efficiency method is a way to *move fewer
> weights, less far.*

---

## 3. Why "Any Recipe Can Be Full FT"

This trips people up, so make it concrete:

| Objective (Axis A) | Run as Full FT means… |
|--------------------|------------------------|
| Unsupervised | Update *all* weights via next-token prediction on raw domain text. |
| Supervised (SFT/IFT) | Update *all* weights to imitate human input→output demonstrations. |
| Alignment (RLHF/DPO) | Update *all* weights to optimize a preference/reward signal. |

The objective decides *the loss*; "full" decides *how many parameters that loss is allowed to
change.* Independent choices (file 03, Axis A × Axis B).

---

## 4. When to Actually Choose Full FT

- You have a **large, high-quality dataset** and the domain is **far** from the base model.
- You have the **GPU budget** (or use the efficiency tricks in file 09 to fit it).
- You can **tolerate keeping a full model checkpoint** per task.
- You **guard against forgetting** (replay general data, smaller learning rate, fewer epochs —
  see Part II).

Otherwise, prefer **PEFT (LoRA/QLoRA)**: ~comparable quality on many tasks, a *fraction* of the
memory, and a tiny adapter checkpoint instead of a full model copy.

> 💡 **Learning Thought:** Full FT is the "sledgehammer" — unmatched when the task demands deep
> change, wasteful when it doesn't. In 2024+ practice most teams start with LoRA and only reach
> for full FT when they can prove they need it.

---

## 5. The Reality Check (slide 65)

The professor's blunt summary:

> **Full fine-tuning is very expensive — in *time* and *memory* both. But the adaptation to the
> new task is better.**

That single sentence is the tension driving the rest of the lecture (and PEFT as a field):
**better adaptation vs. worse cost.** File 09 quantifies the cost; files 05–08 show the recipes
you'd typically run; Part II shows how to keep the run stable.

---

## 🎯 Interview Questions

**Q1. Define full fine-tuning and classify it on the two-axis taxonomy.**
Updating *all* parameters of a pre-trained model on new data. It's a **parameter-update
strategy** (Axis B), not an objective (Axis A) — any objective (unsupervised/SFT/alignment) can
be run as full FT.

**Q2. What are the three defining properties of full fine-tuning?**
(1) **Maximum adaptation capacity** — all weights can move; (2) **High cost** — ~16 bytes/param
for Adam plus a full checkpoint per task; (3) **Catastrophic forgetting** — aggressive updates
erase general skills.

**Q3. Why is full fine-tuning the most forgetting-prone strategy?**
Because *every* weight is free to move, the model can drift far from its pre-trained
configuration, overwriting the general capabilities encoded there. PEFT forgets less because it
freezes most weights.

**Q4. Where does the "~16 bytes per parameter" figure come from?**
The Adam optimizer state in mixed-precision training: the parameter, its gradient, and two
optimizer moments (plus fp32 master copies) — summing to roughly 16 bytes per parameter. (Full
breakdown in file 09.)

**Q5. Your task is close to the base model's existing abilities and you have one GPU. Full FT or
PEFT?**
**PEFT (LoRA).** The adaptation needed is small, so full FT's cost and forgetting risk aren't
justified; LoRA gets comparable quality with a fraction of the memory and a tiny checkpoint.

**Q6. "The objective and 'full vs. PEFT' are the same choice." Correct this.**
They're **independent**. The objective (Axis A) sets the *loss* (what the model learns); "full
vs. PEFT" (Axis B) sets *how many parameters that loss may change*. E.g. SFT can be done full or
with LoRA.

**Q7. State the core trade-off of full fine-tuning in one sentence.**
It delivers the **best task adaptation** at the **highest cost in time and memory** and the
**greatest risk of forgetting** — better quality bought with worse economics.

---

**One-line takeaway:** *Full fine-tuning moves every weight — giving maximum adaptation capacity
at maximum cost and maximum forgetting risk — which is exactly the trade-off that motivates PEFT
and the efficiency tricks in file 09.*
